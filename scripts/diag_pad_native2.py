"""Probe: does a NORMALIZED high-freq statistic separate CASIA-FASD classes?

Follow-up to diag_pad_native (fixed Laplacian threshold saturates: AUC 0.5,
all scores 0.0 at native uint8). If scale-invariant statistics (relative
Laplacian energy vs flat-region? per-image std) carry no class signal at
either resolution, the heuristic tier is documented as unsuited to this
corpus and the recalibration path is marked and closed.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("artifacts/dataset/raw/casiafasd")
TIER = str(Path("artifacts/dataset/raw/casiafasd/{m}_img/{m}_img/color"))


def labeled(mode: str) -> list[tuple[Path, int]]:
    out = []
    for jpg in sorted((Path(str(TIER).format(m=mode))).glob("*.jpg")):
        s = jpg.stem.lower()
        if s.endswith("_real"):
            out.append((jpg, 1))
        elif s.endswith("_fake"):
            out.append((jpg, 0))
    return out


def auc(x: np.ndarray, y: np.ndarray) -> float:
    pos, neg = x[y == 1], x[y == 0]
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def stats(img: np.ndarray) -> tuple[float, float, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    flat = float(np.std(lap))                    # absolute texture energy
    rel = float(np.std(lap) / (np.std(gray) + 1e-6))  # contrast-normalized
    sat = float((np.abs(lap) > 4.0).mean())      # what the shipped tier keys on
    return rel, flat, sat


rows = []
for mode in ("test", "train"):
    feats, labs = [], []
    for path, y in labeled(mode):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img256 = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        feats.append(stats(img256))
        labs.append(y)
    f = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labs)
    # attack users of each stat: higher energy -> more attack. score=energy,
    # auc of energy over attack-vs-live (attack should be higher)
    rows.append({
        "mode": mode,
        "n": len(y),
        "auc_lap_std(attack_higher)": round(1 - auc(f[:, 1], y), 4),
        "auc_rel_energy(attack_higher)": round(1 - auc(f[:, 0], y), 4),
        "auc_shipped_tier_energy": round(1 - auc(f[:, 2], y), 4),
        "mean_rel_real_vs_fake": (round(float(f[y == 1, 0].mean()), 3),
                                   round(float(f[y == 0, 0].mean()), 3)),
    })

out = Path("artifacts/pad_native_diag2.json")
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
print("->", out)
