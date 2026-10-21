"""End-to-end smoke of the SIMPLE console flow over the live API.

Replicates exactly what the browser does per UI action with a real face
(Olivetti subject 0 frame 0, downsampled to the console's canonical 32x32):

  upload -> enroll -> verify (bypass + fail-closed variants)
         -> identify -> PAD-only -> N:N dedup (incl. duplicate detection)
         -> registry forget (GDPR erasure hook)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

BASE = "http://localhost:8000"
OLIVETTI = Path("artifacts/dataset/olivetti.npz")


def ensure_bootstrap() -> None:
    """Fresh-clone: fetch the Olivetti npz once; require the live docker API."""
    if not OLIVETTI.exists():
        print("(one-time: fetching Olivetti via scripts/prepare_dataset.py …)")
        subprocess.run(
            [sys.executable, "scripts/prepare_dataset.py", "--dataset",
             "olivetti", "--out", str(OLIVETTI)], check=True)
    try:
        requests.get(f"{BASE}/health", timeout=5)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"no live API at {BASE} ({exc}) — start it with:\n"
            "  docker compose up -d --build\n"
            "then re-run this script.") from None


def frames_of(img32: np.ndarray, n: int) -> list:
    """F x H x W x RGB nested triples, matching the API contract."""
    rgb = (img32[..., ::-1] / 255.0).round(4).tolist()  # BGR->RGB, H x W x 3
    return [rgb] * n  # replicate the still like the console does


def main() -> int:
    ensure_bootstrap()
    failures: list[str] = []

    def check(name: str, cond: bool, detail: object = "") -> None:
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
        if not cond:
            failures.append(name)

    # ---- 1. "upload": Olivetti subject 0 frame 0 -> 32x32 BGR image ---- #
    z = np.load("artifacts/dataset/olivetti.npz")
    frames_np = z["frames"]  # (400,64,64,3) [0,1]
    labels = z["labels"].astype(int)
    f0 = frames_np[np.flatnonzero(labels == 0)[0]]  # subject 0
    img32 = cv2.resize((f0 * 255).astype(np.uint8), (32, 32),
                       interpolation=cv2.INTER_AREA)
    probe = frames_of(img32, 8)

    # ---- 2. enroll ---- #
    r = requests.post(f"{BASE}/enroll", json={"identity_id": "smoke_face0",
                                             "frames": probe}, timeout=30)
    check("enroll(new identity)", r.ok and r.json()["identity_id"] == "smoke_face0",
          r.json().get("dim"))

    # ---- 3. verify bypass (as-live OFF -> pad=false) ---- #
    r = requests.post(f"{BASE}/verify?identity_id=smoke_face0&pad=false",
                      json={"frames": probe}, timeout=30).json()
    check("verify(1:1, photo mode) match", r["is_match"] and r["score"] > 0.9,
          r["score"])
    check("verify pad bypass honest None", r["pad_live"] is None, r["pad_live"])

    # ---- 4. verify as-live (pad=true, still probe) ---- #
    # New still-mode contract: replicated stills are honestly labeled
    # ("still mode: single-frame replica …"), PAD passes (weights
    # renormalized over flash+VLM) and the MATCHER decides on embeddings ---
    # a still of the enrolled face must MATCH.
    r = requests.post(f"{BASE}/verify?identity_id=smoke_face0&pad=true",
                      json={"frames": probe}, timeout=30).json()
    check("verify(1:1, live-mode) still -> honest PAD, matcher decides",
          r["is_match"] is True and r["pad_live"] is True
          and any("still mode" in x for x in r.get("pad_reasons", [])),
          f"match={r['is_match']} pad={r['pad_live']} "
          f"score={r.get('score')}")

    # ---- 5. identify (1:N) ---- #
    r = requests.post(f"{BASE}/identify?pad=false", json={"frames": probe},
                      timeout=30).json()
    cands = r.get("candidates", [])
    top = cands[0]["identity_id"] if cands else None
    check("identify(1:N) rank-1 = enroll subject", top == "smoke_face0", top)

    # ---- 6. second identity (different subject) ---- #
    f1 = frames_np[np.flatnonzero(labels == 17)[0]]  # different person
    img32b = cv2.resize((f1 * 255).astype(np.uint8), (32, 32),
                        interpolation=cv2.INTER_AREA)
    probe2 = frames_of(img32b, 8)
    r = requests.post(f"{BASE}/enroll", json={"identity_id": "smoke_face17",
                                             "frames": probe2}, timeout=30)
    check("enroll(2nd identity)", r.ok, r.json().get("dim"))
    r = requests.get(f"{BASE}/identities", timeout=15).json()
    check("registry lists both", {"smoke_face0", "smoke_face17"}
          <= set(r["identities"]), len(r["identities"]))

    # probe of face0 still ranks face0 first
    r = requests.post(f"{BASE}/identify?pad=false", json={"frames": probe},
                      timeout=30).json()
    check("identify with 2 ids ranks correct subj",
          r["candidates"][0]["identity_id"] == "smoke_face0",
          r["candidates"][0])

    # ---- 7. PAD-only action ---- #
    # Still probe: rPPG excluded, flash+VLM renormalized -> honest LIVE label
    # with an explicit "still mode" reason (not a silent not-live verdict).
    r = requests.post(f"{BASE}/pad", json={"frames": frames_of(img32, 16)},
                      timeout=30).json()
    check("pad(still) honest still-mode LIVE label",
          r["is_live"] is True and r["fused_score"] is not None
          and any("still mode" in x for x in r.get("reasons", [])),
          f"fused={r.get('fused_score')} reasons={r.get('reasons')}")

    # ---- 8. N:N clean, then duplicate detection ---- #
    r = requests.post(f"{BASE}/crossmatch?limit=16&threshold=0.9",
                      timeout=60).json()
    check("N:N clean registry -> 0 duplicates",
          r["n_identities"] == 2 and r["n_duplicates"] == 0, r["n_identities"])
    r = requests.post(f"{BASE}/enroll",
                      json={"identity_id": "smoke_face0_dup", "frames": probe},
                      timeout=30)
    r = requests.post(f"{BASE}/crossmatch?limit=16&threshold=0.9",
                      timeout=60).json()
    dups = r["duplicates"]
    pair = {(d["a"], d["b"]) for d in dups}
    check("N:N detects double enrollment",
          any('smoke_face0' in p and 'smoke_face0_dup' in p for p in pair),
          dups[:2])

    # ---- 9. GDPR erasure ---- #
    for gid in ("smoke_face0", "smoke_face17", "smoke_face0_dup"):
        r = requests.delete(f"{BASE}/identities/{gid}", timeout=15)
        assert r.status_code == 200 and r.json().get("removed"), gid
    r = requests.get(f"{BASE}/identities", timeout=15).json()
    check("registry clean after forget", "smoke_face0" not in r["identities"],
          len(r["identities"]))

    print("\n--", "ALL PASS" if not failures else f"FAILURES: {failures}")
    Path("artifacts/ui_smoke.json").write_text(
        json.dumps({"failures": failures}, indent=2), encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
