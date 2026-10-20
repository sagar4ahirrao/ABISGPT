#!/usr/bin/env python
"""Transformer (ViT) path eval on the Olivetti real-face protocol.

Identical gallery/probe split and template math to scripts/train_olivetti.py
(8 train / 2 held-out per identity, 40 ids, seed 0) but the embeddings come
from the pretrained DINOv2 ViT-B/16 backbone (self-supervised transformer)
instead of the trained toy CNN — this exercises the
`abisgpt.encoder.vit_encoder.ViTBiometricEncoder` production swap-in path.

Usage:
    python scripts/eval_dinov2_olivetti.py              # cuda if available
    python scripts/eval_dinov2_olivetti.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from scripts.train_olivetti import load_olivetti, make_split


def main() -> int:
    ap = argparse.ArgumentParser(description="DINOv2 ViT embeddings on Olivetti protocol")
    ap.add_argument("--dataset", default="artifacts/dataset/olivetti.npz")
    ap.add_argument("--model", default="vit_base_patch16_224.dino")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="artifacts/olivetti/vit_metrics.json")
    args = ap.parse_args()

    import cv2  # opencv-python-headless is a repo dependency already
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    frames, labels, names = load_olivetti(Path(args.dataset))
    train_idx, test_idx = make_split(labels, holdout=2, seed=args.seed)

    import timm

    t0 = time.perf_counter()
    model = timm.create_model(args.model, pretrained=True, num_classes=0)
    model.eval().to(device)
    load_s = time.perf_counter() - t0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def embed(idx: np.ndarray) -> tuple[np.ndarray, float]:
        """Frames -> 224x224 -> ViT embeddings (512-D normalized will follow)."""
        t0 = time.perf_counter()
        imgs = []
        for i in idx:
            up = cv2.resize(frames[i], (224, 224), interpolation=cv2.INTER_CUBIC)
            imgs.append(up[..., ::-1].copy())  # HWC BGR->RGB
        batch = torch.from_numpy(
            np.ascontiguousarray(np.stack(imgs), dtype=np.float32)
        ).permute(0, 3, 1, 2).contiguous()
        batch = batch.to(device)
        batch = (batch - mean.to(device)) / std.to(device)
        with torch.no_grad():
            feats = model(batch).cpu().numpy()
        feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-9)
        return feats, time.perf_counter() - t0

    emb_train, t_train = embed(train_idx)
    emb_probe, t_probe = embed(test_idx)

    train_y = labels[train_idx]
    test_y = labels[test_idx]
    n_classes = len(names)
    templates = np.zeros((n_classes, emb_train.shape[1]), dtype=np.float32)
    for cid in range(n_classes):
        templates[cid] = emb_train[train_y == cid].mean(axis=0)
    templates /= np.maximum(np.linalg.norm(templates, axis=1, keepdims=True), 1e-9)

    sims = emb_probe @ templates.T
    top1 = float(np.mean(np.argmax(sims, axis=1) == test_y))
    top5 = float(np.mean([
        1.0 if test_y[i] in np.argsort(-sims[i])[:5] else 0.0
        for i in range(len(test_y))
    ]))
    genuine = np.array([sims[i, test_y[i]] for i in range(len(test_y))])
    impostor = np.array([sims[i, c] for i in range(len(test_y))
                         for c in range(n_classes) if c != test_y[i]])
    thresholds = np.unique(np.concatenate([np.sort(impostor), np.sort(genuine)]))
    far = np.array([(impostor >= t).mean() for t in thresholds])
    frr = np.array([(genuine < t).mean() for t in thresholds])
    i_eer = int(np.argmin(np.abs(far - frr)))

    metrics = {
        "model": f"timm:{args.model} (pretrained DINOv2 ViT-B/16 transformer)",
        "protocol": ("gallery=mean template of the 8 train frames/identity; "
                     "probes=2 held-out frames each (no self-inclusion); same "
                     "seed-0 split as the trained toy CNN headline"),
        "recall@1": round(top1, 4),
        "recall@5": round(top5, 4),
        "mean_genuine_sim": round(float(genuine.mean()), 4),
        "mean_impostor_sim": round(float(impostor.mean()), 4),
        "eer": round(float((far[i_eer] + frr[i_eer]) / 2), 4),
        "n_probes": int(len(test_y)),
        "n_impostor_scores": int(impostor.size),
        "weight_load_s": round(load_s, 1),
        "embed_s": round(t_train + t_probe, 1),
        "device": device,
        "torch": torch.__version__,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
