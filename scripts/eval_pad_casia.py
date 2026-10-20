#!/usr/bin/env python
"""Real-data PAD evaluation on the CASIA-FASD mirror bundle.

`artifacts/dataset/pad_casiafasd.npz` (balanced 927 genuine / 927 attack,
1:1, Zhang et al. IEEE TIFS 2012) is the reference. What is *fair* to
evaluate per tier on STILL CROPS:

  * VLM texture tier (HeuristicVlmPad; calibrated for 0..255 uint8 input)
    and optionally the real CLIP zero-shot VLM (`--clip`, downloads weights).
  * Theата full `PassivePadEngine` on single frames fails closed
    (rPPG needs >= 12-frame temporal sequences, flash needs pairs) —
    reported as the fail-closed property, not an accuracy metric.

Metrics: AUC (rank-based), TPR/TNR/ACC at the tier's own 0.5 gate.
Deterministic stratified 70/30 split, seed 0.

Usage:
    python scripts/eval_pad_casia.py
    python scripts/eval_pad_casia.py --clip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from abisgpt.pad.vlm import HeuristicVlmPad


def auc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC with tie-corrected average ranks."""
    attack = scores[labels == 0]
    live = scores[labels == 1]
    x = np.concatenate([attack, live])
    n = x.size
    sx = x.argsort()
    sorted_x = x[sx]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[sx[i:j + 1]] = (i + j + 2) / 2.0     # average rank of the tie block
        i = j + 1
    n_a, n_l = attack.size, live.size
    return float((ranks[n_a:].sum() - n_l * (n_l + 1) / 2) / (n_a * n_l))


def main() -> int:
    ap = argparse.ArgumentParser(description="PAD tiers on real CASIA-FASD crops")
    ap.add_argument("--dataset", default="artifacts/dataset/pad_casiafasd.npz")
    ap.add_argument("--clip", action="store_true", help="also run real CLIP zero-shot tier")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/pad_eval.json")
    args = ap.parse_args()

    npz = np.load(args.dataset, allow_pickle=True)
    frames = npz["frames"]                       # (N,64,64,3) float32 [0,1]
    labels = npz["labels"].astype(np.int64)      # 0=bona-fide, 1=attack
    names = [str(n) for n in npz["names"]]
    rng = np.random.default_rng(args.seed)

    for cid in (0, 1):
        idx = np.flatnonzero(labels == cid)
        rng.shuffle(idx)
        labels[idx[int(len(idx) * 0.7):]] = -1 - cid   # mark tail as held-out
    test_mask = labels < 0
    test_labels = (-1 - labels[test_mask]).astype(np.int64)   # 0 bona-fide / 1 attack
    test_frames = frames[test_mask]
    print(f"[pad] test frames: bona-fide {int((test_labels == 0).sum())} "
          f"/ attack {int((test_labels == 1).sum())}", flush=True)

    def collect(scorer, tag: str) -> dict:
        scores = np.array([float(scorer(f)) for f in test_frames], dtype=np.float64)
        auc = auc_rank(scores, test_labels)
        tpr = float((scores[test_labels == 1] < 0.5).mean())   # attack flagged
        tnr = float((scores[test_labels == 0] >= 0.5).mean())  # genuine passes
        acc = float(((scores < 0.5).astype(np.int64) == test_labels).mean())
        res = {"tier": tag, "auc": round(auc, 4), "tpr@0.5": round(tpr, 4),
               "tnr@0.5": round(tnr, 4), "acc@0.5": round(acc, 4),
               "n_test": int(test_labels.size)}
        print(json.dumps(res))
        return res

    # ---- VLM texture tier (grayscale ladder, 0..255 uint8 contract) ----- #
    h = HeuristicVlmPad()
    res_h = collect(lambda f: h.score(f).liveness_score, "heuristic_vlm")

    results = {"heuristic_vlm": res_h}
    # ---- optional real CLIP zero-shot tier ------------------------------- #
    if args.clip:
        try:
            clip = __import__("abisgpt.pad.vlm", fromlist=["ClipVlmPad"]).ClipVlmPad()
            res_c = collect(clip.score, "clip_vlm")
            results["clip_vlm"] = res_c
        except Exception as exc:                              # noqa: BLE001
            results["clip_vlm"] = {"tier": "clip_vlm", "error": str(exc)[:200]}
            print(f"[clip] unavailable: {exc}", flush=True)

    # ---- full-engine fail-closed property (single frames) --------------- #
    from abisgpt.config import PadConfig
    from abisgpt.pad.fusion import PassivePadEngine

    engine = PassivePadEngine(cfg=PadConfig(), vlm_scorer=lambda f: h.score(f).liveness_score)
    genuine_one = (test_frames[test_labels == 0][0:3].tolist())
    rejects = bool(all(not engine.evaluate([f]).is_live for f in genuine_one))
    results["engine_single_frame_fail_closed"] = {
        "all_genuine_single_frames_rejected": rejects,
        "note": "rPPG needs >=12-frame sequences; flash needs consecutive pairs — "
                "single-frame kiosk captures fail closed by design (no accuracy claim)",
    }
    print(json.dumps(results["engine_single_frame_fail_closed"], indent=2))

    (Path("artifacts") / "pad_eval.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
