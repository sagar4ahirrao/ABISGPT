#!/usr/bin/env python
"""Name/series comparison on CASIA-FASD (real files, labels in filenames).

Every CASIA color image name parses into (subject, video, frame, kind).
Series structure (official): per subject, videos 1-2 = live captures;
higher-numbered videos = fake-attack replays of THE SAME face. The report
therefore compares:

 PAD   - per subject: mean fused PAD score across REAL-capture series vs
         FAKE-attack series, at the console path (32x32 replicated still)
         AND at 128x128 (where the tiers may un-saturate).
 MATCH - per subject (shipped ONNX encoder path): enroll video-1 real
         frames (multi-frame template), probe (a) video-2 real frames ->
         genuine cross-video cosine; (b) a fake-attack video's frames ->
         attack-probe cosine. Replay attacks biometrically MATCH the same
         face — that is exactly why the PAD layer must reject them.

Usage:
    python scripts/series_report_casia.py [--out artifacts/series_report.json]
                                          [--skip-pad] [--skip-match]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path("artifacts/dataset/raw/casiafasd")
ONNX = Path("artifacts/encoder/face_encoder.onnx")
PAT = re.compile(r"^(\d+)_(.+)\.avi_(\d+)_(real|fake)$")


def parse_name(stem: str) -> tuple[int, str, int, str] | None:
    m = PAT.match(stem)
    if m is None:
        return None
    return int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)


def collect():
    rows = []
    for split in ("train", "test"):
        for p in sorted((ROOT / f"{split}_img" / f"{split}_img" / "color")
                        .glob("*.jpg")):
            info = parse_name(p.stem)
            if info is not None:
                rows.append((*info, split, p))
    return rows


def _load(path: Path, size: int) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    return im


def pad_series(rows) -> dict:
    """Per-subject: fused PAD real-series vs fake-series mean (32px + 128px)."""
    from abisgpt.config import AbisGptConfig
    from abisgpt.pad.fusion import PassivePadEngine
    from abisgpt.pad.vlm import HeuristicVlmPad

    engine = PassivePadEngine(AbisGptConfig().pad, vlm_scorer=HeuristicVlmPad())
    acc = defaultdict(lambda: {"real_32": [], "fake_32": [],
                               "real_128": [], "fake_128": []})
    for k, (subj, _, _, kind, _, path) in enumerate(rows):
        for size, rk, fk in ((32, "real_32", "fake_32"),
                             (128, "real_128", "fake_128")):
            img = _load(path, size)
            v = engine([img] * 8)
            acc[subj][rk if kind == "real" else fk].append(float(v.fused_score))
        if k % 600 == 0:
            print(f"  pad {k}/{len(rows)}", flush=True)
    out = {}
    for subj, d in sorted(acc.items()):
        out[str(subj)] = {
            f"mean_{k}": round(float(np.mean(v)), 4) if v else None
            for k, v in d.items()
        }
    return out


def match_series(rows) -> dict:
    """Per-subject genuine cross-video + attack-probe cosine (ONNX path)."""
    if not ONNX.exists():
        return {"error": "onnx encoder missing"}
    from abisgpt.encoder import ViTBiometricEncoder

    enc = ViTBiometricEncoder(strategy="onnx")
    by = defaultdict(list)
    for subj, video, _, kind, split, path in rows:
        if split == "test":
            by[(subj, video, kind)].append(path)

    real_vids = defaultdict(list)
    fake_vids = defaultdict(list)
    for (subj, video, kind) in by:
        (real_vids if kind == "real" else fake_vids)[subj].append(video)
    real_vids = {s: sorted(v) for s, v in real_vids.items()}
    fake_vids = {s: sorted(v) for s, v in fake_vids.items()}

    def embed(paths):
        v = enc.encode([_load(p, 64).astype(np.float32) / 255.0
                        for p in paths])
        m = v.mean(0)
        return m / max(float(np.linalg.norm(m)), 1e-9)

    per_subject = {}
    for subj in sorted(set(real_vids) | set(fake_vids)):
        rv = real_vids.get(subj, [])
        fv = fake_vids.get(subj, [])
        entry: dict = {"real_videos": rv, "fake_videos": fv}
        if len(rv) >= 2:
            tmpl = embed(by[(subj, rv[0], "real")][:6])
            pr = enc.encode([_load(p, 64).astype(np.float32) / 255.0
                             for p in by[(subj, rv[1], "real")]])
            entry["genuine_cross_video_sim"] = round(
                float(np.mean(pr @ tmpl)), 4)
        if fv:
            if rv:
                tmpl = embed(by[(subj, rv[0], "real")][:6])
            else:
                tmpl = embed(by[(subj, fv[0], "fake")][:6])
            pf = enc.encode([_load(p, 64).astype(np.float32) / 255.0
                             for p in by[(subj, fv[0], "fake")]])
            entry["attack_probe_sim"] = round(float(np.mean(pf @ tmpl)), 4)
        per_subject[str(subj)] = entry
    return per_subject


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/series_report.json")
    ap.add_argument("--skip-pad", action="store_true")
    ap.add_argument("--skip-match", action="store_true")
    args = ap.parse_args()

    rows = collect()
    print(f"parsed {len(rows)} casia images (train+test)")

    subjects = defaultdict(lambda: {"videos": set(), "real": 0, "fake": 0})
    for subj, video, _, kind, _, _ in rows:
        d = subjects[subj]
        d["videos"].add(video)
        d["real" if kind == "real" else "fake"] += 1
    overview = {str(s): {"n_videos": len(d["videos"]), "real": d["real"],
                         "fake": d["fake"],
                         "videos": sorted(d["videos"])}
                for s, d in sorted(subjects.items())}

    report: dict = {
        "dataset": "CASIA-FASD color (train+test); filenames = labels",
        "n_images": len(rows),
        "n_subjects": len(overview),
        "series_semantics":
            "subject_V<1-2>=live captures; V>=3<=5 = fake-attack replays",
        "pad_series": None,
        "matcher_series": None,
        "subjects_overview": overview,
    }
    if not args.skip_pad:
        report["pad_series"] = pad_series(rows)
    if not args.skip_match:
        report["matcher_series"] = match_series(rows)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved ->", args.out)

    ps = report["pad_series"] or {}
    for size in ("32", "128"):
        mr = [v[f"mean_real_{size}"] for v in ps.values()
              if v[f"mean_real_{size}"] is not None]
        mf = [v[f"mean_fake_{size}"] for v in ps.values()
              if v[f"mean_fake_{size}"] is not None]
        if mr and mf:
            print(f"PAD@{size}px within-subject series: real {np.mean(mr):.3f}"
                  f" vs fake {np.mean(mf):.3f} (n_subjects={len(mr)})")
    ms = report["matcher_series"]
    if isinstance(ms, dict) and ms and "error" not in ms:
        g = [v["genuine_cross_video_sim"] for v in ms.values()
             if v.get("genuine_cross_video_sim") is not None]
        f = [v["attack_probe_sim"] for v in ms.values()
             if v.get("attack_probe_sim") is not None]
        if g:
            print(f"MATCH genuine cross-video sim: mean {np.mean(g):.3f}"
                  f" (n={len(g)})")
        if f:
            print(f"MATCH attack(fake-video) probe sim: mean {np.mean(f):.3f}"
                  f" (n={len(f)}) — same face ⇒ high; rejection must come"
                  " from PAD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
