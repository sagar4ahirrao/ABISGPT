"""Diagnostic: HeuristicVlmPad at NATIVE resolution (256x256) vs the 64x64 npz.

Hypothesis under test: the proposed-cascade 64x64 downscale destroys the LCD
banding/moire energy the texture tier keys on, not the tier itself. If AUC
at native resolution jumps well above 0.5 the tier is salvageable and the
preprocessing budget is the defect (documented finding either way).

Usage:
    python scripts/diag_pad_native.py --dir artifacts/dataset/raw/casiafasd \
        --modes train test --size 256 128 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _iter_images(root: Path) -> list[tuple[Path, int]]:
    """(path, label) pairs; label 1 = bona-fide (_real), 0 = attack (_fake)."""
    pairs: list[tuple[Path, int]] = []
    for jpg in sorted(root.glob("*.jpg")):
        lower = jpg.stem.lower()
        if lower.endswith("_real"):
            pairs.append((jpg, 1))
        elif lower.endswith("_fake"):
            pairs.append((jpg, 0))
    return pairs


def _auc(liveness: np.ndarray, labels: np.ndarray) -> float:
    """Rank (Mann-Whitney U) AUC of liveness: live should be higher."""
    pos = liveness[labels == 1]
    neg = liveness[labels == 0]
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def _load_rgb(path: Path, size: int | None) -> np.ndarray:
    import cv2  # defer: behind --slow only path

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:  # corrupt/missing -> skip at caller level
        raise ValueError(path.name)
    if size is not None and (img.shape[0] != size or img.shape[1] != size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img  # BGR uint8; grayscale mean is channel-order invariant


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/dataset/raw/casiafasd")
    ap.add_argument("--modes", nargs="+", default=["train", "test"],
                    choices=["train", "test"])
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[256, 128, 64], help="resize targets; omit=keep native")
    ap.add_argument("--cap", type=int, default=0, help="per-class cap (0 = all)")
    args = ap.parse_args()

    from abisgpt.pad.vlm import HeuristicVlmPad

    pad = HeuristicVlmPad()
    raw_root = Path(args.dir) / "{m}_img" / "{m}_img" / "color"
    report: dict[str, object] = {"hypothesis": "64x64 downscale destroyed banding, "
                                                  "not the tier"}
    for mode in args.modes:
        root = Path(str(raw_root).format(m=mode))
        pairs = _iter_images(root)
        if args.cap:
            reals = [(p, lab) for p, lab in pairs if lab == 1][: args.cap]
            fakes = [(p, lab) for p, lab in pairs if lab == 0][: args.cap]
            pairs = reals + fakes
        if not pairs:
            report[mode] = "no labelled jpgs found"
            continue
        labels_all = np.array([lab for _, lab in pairs], dtype=np.int64)
        entry: dict[str, object] = {
            "files": len(pairs),
            "real": int(labels_all.sum()),
            "fake": int((labels_all == 0).sum()),
        }
        for size in args.sizes:
            liveness = np.empty(len(pairs), dtype=np.float64)
            t0 = time.perf_counter()
            n_ok = 0
            for i, (path, _label) in enumerate(pairs):
                try:
                    img = _load_rgb(path, None if size == 0 else size)
                except ValueError:
                    continue
                liveness[i] = pad(img)  # callable contract -> liveness in [0,1]
                n_ok += 1
            dt = time.perf_counter() - t0
            valid = np.arange(n_ok)
            auc = _auc(liveness[valid], labels_all[valid])
            entry[str(size if size else "native")] = {
                "auc:live>attack": round(auc, 4),
                "mean_live_real": round(float(liveness[valid][labels_all == 1].mean()), 4),
                "mean_live_fake": round(float(liveness[valid][labels_all == 0].mean()), 4),
                "images/sec": round(n_ok / dt, 1) if dt > 0 else None,
            }
        report[mode] = entry

    out = Path("artifacts/pad_native_diag.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
