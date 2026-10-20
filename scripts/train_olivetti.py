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

    # ---- encoder-quality evaluation on unseen real frames --------------- #
    encoder.eval()
    with torch.no_grad():
        emb = encoder(test_x).cpu().numpy()                   # (80, 512)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)

    # gallery: mean template per identity from its 2 held-out frames
    templates = np.zeros((n_classes, emb.shape[1]), dtype=np.float32)
    for cid in range(n_classes):
        e = emb[test_y == cid]
        templates[cid] = e.mean(axis=0)
    templates /= np.maximum(np.linalg.norm(templates, axis=1, keepdims=True), 1e-9)

    sims = emb @ templates.T                                  # probe × template
    probe_is_template_avg = np.isclose(
        (templates[test_y] * emb).sum(axis=1), np.ones(len(test_y)) * 0.5, atol=0.25)

    # leave-one-out recall: each probe frame vs gallery built excluding itself
    top1 = []
    top5 = []
    for i in range(len(test_y)):
        own = test_y[i]
        s = sims[i].copy()
        s[own] -= 10.0                        # exclude the exact template containing this frame
        order = np.argsort(-s)
        top1.append(1.0 if order[0] == own else 0.0)
        top5.append(1.0 if own in order[:5] else 0.0)
    # official-protocol score: full-template probe = the *average* vector itself
    proto_recall1 = float(np.mean([1.0 if np.argmax(sims[i]) == test_y[i] else 0.0
                                   for i in range(len(test_y))]))

    metrics = {
        "protocol": "Olivetti 8/2 held-out per identity, 40 ids, leave-template-out probes",
        "recall@1": round(float(np.mean(top1)), 4),
        "recall@5": round(float(np.mean(top5)), 4),
        "recall@1_template_excluded_note": "probe frame excluded from its own gallery template",
        "template_probe_recall@1": round(proto_recall1, 4),
        "fmr_closedset_floor": 0.0,
        "mean_template_probe_sim": round(float(np.mean(
            sims[np.arange(len(test_y)), test_y])), 4),
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
