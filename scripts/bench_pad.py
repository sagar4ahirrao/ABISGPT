#!/usr/bin/env python
"""Benchmark the shipped PassivePadEngine fusion on CASIA-FASD (real files).

Faithful to the console path: every test image is decoded, resized to the
console's canonical 32x32, replicated 8x (exactly what the UI does for an
uploaded still), and fused by `abisgpt.pad.PassivePadEngine`. Labels are
read from the official filenames (`*_real` / `*_fake`).

A/B: `--no-stillfix` disables the still-mode renormalization so the report
can show fusion before/after the fix (old path treated "no pulse" as attack
evidence and sank every single still).

Usage:
    python scripts/bench_pad.py                          # console path, 32x32
    python scripts/bench_pad.py --resize 128             # higher-res tiers
    python scripts/bench_pad.py --no-stillfix            # legacy A/B
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from abisgpt.config import AbisGptConfig  # noqa: E402
from abisgpt.pad.fusion import PassivePadEngine  # noqa: E402
from abisgpt.pad.vlm import HeuristicVlmPad  # noqa: E402

ROOT = Path("artifacts/dataset/raw/casiafasd")


def files_for(split: str) -> list[tuple[Path, int]]:
    color = ROOT / f"{split}_img" / f"{split}_img" / "color"
    out = []
    for j in sorted(color.glob("*.jpg")):
        if j.stem.lower().endswith("_real"):
            out.append((j, 1))
        elif j.stem.lower().endswith("_fake"):
            out.append((j, 0))
    return out


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def metrics(scores: np.ndarray, labels: np.ndarray, gate: float) -> dict:
    a = auc(scores, labels)
    thr = np.unique(scores)
    far = np.array([(scores[labels == 0] >= t).mean() for t in thr])
    frr = np.array([(scores[labels == 1] < t).mean() for t in thr])
    i = int(np.argmin(np.abs(far - frr)))
    gate_pred = (scores >= gate).astype(int)
    return {
        "n": int(len(labels)),
        "real": int(labels.sum()),
        "auc": round(a, 4),
        "eer": round(float((far[i] + frr[i]) / 2), 4),
        "best_threshold_acc": round(
            float(((scores >= thr[i]).astype(int) == labels).mean()), 4),
        "gate": gate,
        "gate_acc": round(float((gate_pred == labels).mean()), 4),
        "gate_tpr": round(float(gate_pred[labels == 1].mean()), 4),
        "gate_tnr": round(float((1 - gate_pred[labels == 0]).mean()), 4),
    }


def run_split(engine: PassivePadEngine, pairs: list[tuple[Path, int]],
              tag: str) -> tuple[np.ndarray, np.ndarray, dict]:
    scores, labels = np.zeros(len(pairs), np.float32), np.zeros(len(pairs), np.int64)
    t0 = time.perf_counter()
    for k, (path, lab) in enumerate(pairs):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        frames = [img] * 8                      # console replica contract
        scores[k] = engine(frames).fused_score
        labels[k] = lab
        if k % 400 == 0:
            print(f"  [{tag}] {k}/{len(pairs)}", flush=True)
    dt = time.perf_counter() - t0
    m = metrics(scores, labels, engine.cfg.gate_threshold)
    m["img_per_s"] = round(len(pairs) / dt, 1)
    return scores, labels, m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-stillfix", action="store_true",
                    help="disable still-mode renormalization (legacy A/B)")
    ap.add_argument("--out", default="artifacts/pad_bench_fusion.json")
    args = ap.parse_args()

    cfg = AbisGptConfig()
    if args.no_stillfix:
        # legacy behaviour: force pulse tier ON even for replicated stills
        orig = PassivePadEngine._is_still_replica
        PassivePadEngine._is_still_replica = staticmethod(
            lambda frames: False)
    engine = PassivePadEngine(cfg.pad, vlm_scorer=HeuristicVlmPad())
    report: dict = {"fix": not args.no_stillfix, "gate_threshold":
                    cfg.pad.gate_threshold, "splits": {}}
    for split in ("train", "test"):
        pairs = files_for(split)
        if not pairs:
            continue
        print(f"[{split}] n={len(pairs)}", flush=True)
        _, _, m = run_split(engine, pairs, split)
        report["splits"][split] = m
        print(json.dumps(m), flush=True)

    if args.no_stillfix:
        # re-wrap: class-attribute access strips the staticmethod descriptor
        PassivePadEngine._is_still_replica = staticmethod(orig)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
