"""Optimal-threshold calibration of the raw |lap|>4 energy feature.

Quantifies the ceiling of the shipped texture feature after removing the
clamp defect (monotone score instead of the saturating clip). If calibrated
accuracy is still far below production quality, the conclusion is final:
the heuristic tier is a synthetic-video stand-in; CASIA-FASD photo PAD
belongs to trainable discriminators / CLIP.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

TIER = "artifacts/dataset/raw/casiafasd/{m}_img/{m}_img/color"


def labeled(mode: str) -> list[tuple[Path, int]]:
    out = []
    for jpg in sorted(Path(str(TIER).format(m=mode)).glob("*.jpg")):
        s = jpg.stem.lower()
        if s.endswith("_real"):
            out.append((jpg, 1))
        elif s.endswith("_fake"):
            out.append((jpg, 0))
    return out


def energy(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float((np.abs(lap) > 4.0).mean())  # fraction of high-freq pixels


rows = {}
for mode in ("train", "test"):
    vals, labs = [], []
    for path, y in labeled(mode):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        vals.append(energy(cv2.resize(img, (256, 256),
                                      interpolation=cv2.INTER_AREA)))
        labs.append(y)
    e = np.asarray(vals, dtype=np.float32)
    y = np.asarray(labs)  # 1=live, 0=attack
    # calibrate: attack has higher energy -> live if e < tau
    grid = np.quantile(e, np.linspace(0.01, 0.99, 397))
    accs = np.array([(y[(e < t)].mean() if (e < t).any() else 0.0) *
                     (e < t).mean() + ((1 - y[(e >= t)]).mean() if (e >= t).any() else 0.0) *
                     (e >= t).mean() for t in grid])
    best = int(np.argmax(accs))
    tau = float(grid[best])
    rows[mode] = {
        "n": len(y),
        "best_accuracy": round(float(accs[best]), 4),
        "tau_energy": round(tau, 5),
        "predicted_live_accuracy": round(float(accs[best]), 4),
    }

out = Path("artifacts/pad_native_calib.json")
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
print("->", out)
