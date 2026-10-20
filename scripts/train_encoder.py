#!/usr/bin/env python
"""Train the toy biometric encoder end-to-end and evaluate recall.

Usage:
    python scripts/train_encoder.py --identities 64 --frames 40 --epochs 4
    python scripts/train_encoder.py --identities 128 --frames 64 --epochs 6 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openabis.training.trainer import TrainConfig, train


def main() -> int:
    ap = argparse.ArgumentParser(description="Train toy ViT/CNN biometric encoder")
    ap.add_argument("--identities", type=int, default=64)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="artifacts")   # gitignored
    args = ap.parse_args()

    cfg = TrainConfig(
        train_identities=args.identities,
        frames_per_identity=args.frames,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print(f"[train] identities={cfg.train_identities} frames/identity={cfg.frames_per_identity} "
          f"epochs={cfg.epochs} device={args.device}", flush=True)
    metrics = train(cfg, device=args.device)
    print(f"[e2e] final metrics: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
