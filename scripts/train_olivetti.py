#!/usr/bin/env python
"""Train the toy ViT encoder on the real Olivetti (AT&T) face dataset.

Protocol (standard Olivetti split): 8 of 10 frames train / 2 held out per
identity, 40 identities → 320 train / 80 test frames. Metric = encoder-quality
oracle on unseen *real* faces: template-vs-probe recall@1/@5. Deterministic
(seed every layer via --seed).

Usage:
    python scripts/train_olivetti.py --epochs 30 --device cuda
    python scripts/train_olivetti.py --epochs 30 --device cpu
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from abisgpt.training.model import ArcFaceHead, ToyBiometricEncoder


def load_olivetti(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(path)
    frames, labels, names = z["frames"], z["labels"], z["names"]  # (400,64,64,3) [0,1]
    return frames, labels.astype(np.int64), names


def make_split(labels: np.ndarray, holdout: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cid in np.unique(labels):
        idx = np.flatnonzero(labels == cid)
        rng.shuffle(idx)
        test_idx.extend(idx[:holdout])            # held-out probe frames
        train_idx.extend(idx[holdout:])           # gallery/training frames
    return np.sort(np.array(train_idx)), np.sort(np.array(test_idx))


def to_tensor(frames: np.ndarray) -> torch.Tensor:
    """(N,64,64,3) [0,1] float32 -> (N,3,64,64)."""
    x = torch.from_numpy(np.ascontiguousarray(frames))
    return x.permute(0, 3, 1, 2).contiguous()


def main() -> int:
    ap = argparse.ArgumentParser(description="Train toy encoder on real Olivetti faces")
    ap.add_argument("--dataset", default="artifacts/dataset/olivetti.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="artifacts/olivetti")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = args.device

    frames, labels, names = load_olivetti(Path(args.dataset))
    train_idx, test_idx = make_split(labels, holdout=2, seed=args.seed)

    train_ds = TensorDataset(to_tensor(frames[train_idx]),
                             torch.from_numpy(labels[train_idx]))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_x = to_tensor(frames[test_idx]).to(device)          # (80,3,64,64)
    test_y = labels[test_idx]
    n_classes = len(names)

    encoder = ToyBiometricEncoder().to(device)
    head = ArcFaceHead(num_classes=n_classes, dim=512, scale=24.0, margin=0.22).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    print(f"[train] frames={len(train_idx)} test={len(test_idx)} classes={n_classes} "
          f"epochs={args.epochs} device={device}", flush=True)
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        encoder.train()
        head.train()
        total, seen = 0.0, 0
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device)
            emb = encoder(x)
            loss = loss_fn(head(emb, y), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(y)
            seen += len(y)
        sched.step()
        encoder.eval()
        print(f"  epoch {epoch + 1}/{args.epochs} loss={total / max(seen, 1):.4f}", flush=True)
    train_s = time.perf_counter() - t0

    # ---- encoder-quality evaluation: unseen REAL faces ------------------ #
    protocol = ("gallery=mean template of the 8 train frames/identity; "
                "probes=2 held-out frames each (no self-inclusion)")
    encoder.eval()
    with torch.no_grad():
        train_x = to_tensor(frames[train_idx]).to(device)
        emb_train = encoder(train_x).cpu().numpy()
        emb_probe = encoder(test_x).cpu().numpy()
    emb_train /= np.maximum(np.linalg.norm(emb_train, axis=1, keepdims=True), 1e-9)
    emb_probe /= np.maximum(np.linalg.norm(emb_probe, axis=1, keepdims=True), 1e-9)

    train_y = labels[train_idx]
    templates = np.zeros((n_classes, emb_train.shape[1]), dtype=np.float32)
    for cid in range(n_classes):
        e = emb_train[train_y == cid]
        templates[cid] = e.mean(axis=0)
    templates /= np.maximum(np.linalg.norm(templates, axis=1, keepdims=True), 1e-9)

    sims = emb_probe @ templates.T                            # (80, 40): probe × identity
    top1 = float(np.mean([1.0 if np.argmax(sims[i]) == test_y[i] else 0.0
                          for i in range(len(test_y))]))
    top5 = float(np.mean([1.0 if test_y[i] in np.argsort(-sims[i])[:5] else 0.0
                          for i in range(len(test_y))]))

    # genuine vs impostor similarity distributions -> EER (closed-set, 40 ids)
    genuine = np.array([sims[i, test_y[i]] for i in range(len(test_y))])
    impostor = np.array([sims[i, c] for i in range(len(test_y))
                         for c in range(n_classes) if c != test_y[i]])
    thresholds = np.unique(np.concatenate([np.sort(impostor), np.sort(genuine)]))
    far = np.array([(impostor >= t).mean() for t in thresholds])
    frr = np.array([(genuine < t).mean() for t in thresholds])
    idx_eer = int(np.argmin(np.abs(far - frr)))

    metrics = {
        "protocol": protocol,
        "recall@1": round(top1, 4),
        "recall@5": round(top5, 4),
        "mean_genuine_sim": round(float(genuine.mean()), 4),
        "mean_impostor_sim": round(float(impostor.mean()), 4),
        "eer": round(float((far[idx_eer] + frr[idx_eer]) / 2), 4),
        "n_probes": int(len(test_y)),
        "n_impostor_scores": int(impostor.size),
        "train_s": round(train_s, 1),
        "device": device,
        "torch": torch.__version__,
    }
    print(json.dumps(metrics, indent=2))
    torch.save({"state_dict": encoder.state_dict(),
                "dataset": "olivetti.npz",
                "seed": args.seed},
               out / "olivetti_encoder.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[note] checkpoint -> {out / 'olivetti_encoder.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
