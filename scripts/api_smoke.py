#!/usr/bin/env python
"""End-to-end HTTP smoke: enroll -> identify -> forget against a running service.

    python scripts/api_smoke.py http://localhost:8000

Generates pulse-carrying synthetic captures (slow green flicker, as in
demo_synthetic.py), enrolls two identities, probes each, and verifies the
decisions are 'match'; then deletes the templates (GDPR hook) and confirms 404.
"""

from __future__ import annotations

import json
import sys
import urllib.request

import numpy as np

FPS = 60
N_FRAMES = 180          # 3 s @ 60 fps — ≥2.5 pulse cycles keeps the linear
                        # detrend from absorbing the BVP (see rppg.bandpass)
# On-bin (integer-cycle) BVP so the rectangular FFT window has zero
# spectral leakage — 1.0 Hz = 60 bpm, inside the plausible pulse band.
HR_HZ = 1.0


def gen_frames(rng: np.random.Generator, base: float) -> list:
    """`N_FRAMES` 32x32 RGB captures whose green channel carries a 1 Hz BVP.

    `base` is in 0..255 gray levels (demo_synthetic.py convention: base 120
    + a ±2.5-level pulse). Frames are POSTED in the service's 0..1 float
    convention — pulse amplitude stays valid either way, and never drives
    values out of range (an out-of-range frame made the VLM tier mis-score
    real captures as spoof; see tests/test_vlm.py)."""
    frames = []
    for f in range(N_FRAMES):
        pulse = 2.5 * np.sin(2 * np.pi * HR_HZ * f / FPS)
        img = np.full((32, 32, 3), base, dtype=np.float32)
        img[:, :, 1] += pulse + rng.normal(0, 0.6, (32, 32))
        img[:, :, 0] += rng.normal(0, 0.3, (32, 32))
        img[:, :, 2] += rng.normal(0, 0.3, (32, 32))
        frames.append((img / 255.0).round(4).tolist())
    return frames


def call(method: str, url: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def main(base_url: str) -> int:
    ok = True

    status, health = call("GET", f"{base_url}/health")
    print(f"GET  /health        -> {status} {health}")
    ok &= status == 200 and health["status"] == "ok"
    enrolled_before = health["enrolled"]

    # Enroll two identities (different base intensity = distinct templates).
    # Bases are 0..255 gray levels (see gen_frames docstring).
    tom_frames = gen_frames(np.random.default_rng(11), base=120)
    jane_frames = gen_frames(np.random.default_rng(12), base=155)
    status, body = call("POST", f"{base_url}/enroll",
                        {"identity_id": "tom", "frames": tom_frames})
    print(f"POST /enroll tom    -> {status} {body}")
    ok &= status == 200
    status, body = call("POST", f"{base_url}/enroll",
                        {"identity_id": "jane", "frames": jane_frames})
    print(f"POST /enroll jane   -> {status} {body}")
    ok &= status == 200

    # Identify: re-use identical frames -> template cosine ~1.0 -> match.
    status, body = call("POST", f"{base_url}/identify", {"frames": tom_frames})
    top = body.get("candidates", [{}])[:1]
    print(f"POST /identify tom  -> {status} decision={body['decision']} "
          f"pad_live={body['pad_live']} top={top}")
    ok &= body["decision"] == "match"

    status, body = call("POST", f"{base_url}/identify", {"frames": jane_frames})
    top = body.get("candidates", [{}])[:1]
    print(f"POST /identify jane -> {status} decision={body['decision']} "
          f"pad_live={body['pad_live']} top={top}")
    ok &= body["decision"] == "match"

    # PAD-only endpoint on a flat (pulse-less) capture must be able to gate.
    flat = [np.full((32, 32, 3), 0.55, dtype=np.float32).tolist()] * N_FRAMES
    status, body = call("POST", f"{base_url}/pad", {"frames": flat})
    print(f"POST /pad flat      -> {status} is_live={body['is_live']} "
          f"fused={body['fused_score']} tiers={body['tier_scores']}")

    # GDPR erasure hook.
    for name, expect in (("tom", 200), ("ghost", 404)):
        status, body = call("DELETE", f"{base_url}/identities/{name}")
        print(f"DEL  /identities/{name:<5} -> {status} {body}")
        ok &= status == expect
    call("DELETE", f"{base_url}/identities/jane")   # leave the store empty

    status, health = call("GET", f"{base_url}/health")
    # Post-condition: the store has exactly what it started with (this smoke
    # creates tom + jane and deletes both). Works on clean AND dirty stores.
    ok &= status == 200 and health["enrolled"] == enrolled_before
    print(f"GET  /health        -> {health} (expected {enrolled_before})")

    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"))

