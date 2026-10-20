"""End-to-end training + evaluation loop for the toy biometric encoder."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from openabis.training.model import ArcFaceHead, ToyBiometricEncoder
from openabis.training.synthdata import SyntheticBiometricDataset


@dataclass
class TrainConfig:
    train_identities: int = 64
    val_identities: int = 64
    frames_per_identity: int = 40
    epochs: int = 4
    batch_size: int = 64
    lr: float = 3e-3
    arcface_scale: float = 24.0
    arcface_margin: float = 0.22
    seed: int = 0
    out_dir: str = "artifacts"


def evaluate_recall(
    encoder: ToyBiometricEncoder,
    device: str,
    identities: int = 64,
    probes_per_identity: int = 6,
    seed_offset: int = 99,
    top_k: int = 5,
) -> dict:
    """Encoder-quality oracle: unseen identities, unseen noise -> recall@k."""
    encoder.eval()
    ds = SyntheticBiometricDataset(
        identities=identities, frames_per_identity=probes_per_identity,
        seed=7, noise_std=0.08, identity_offset=seed_offset, encode_range=10_000,
    )
    probe_vecs, probe_labels = [], []
    ref_vecs, ref_labels = [], []
    with torch.no_grad():
        for identity in range(identities):
            for f in range(probes_per_identity):
                x = ds.sample(identity, f)[None].to(device)
                probe_vecs.append(encoder(x).cpu().numpy()[0])
                probe_labels.append(identity)
            ref = ds.sample(identity, 999)[None].to(device)   # canonical ref frame
            ref_vecs.append(encoder(ref).cpu().numpy()[0])
            ref_labels.append(identity)
    probe_vecs = np.stack(probe_vecs)
    ref_mat = np.stack(ref_vecs) / np.linalg.norm(np.stack(ref_vecs), axis=1, keepdims=True)

    sims = probe_vecs @ ref_mat.T                              # (N_probe, N_ident)
    top1 = sims.argmax(axis=1)
    correct1 = float(np.mean([top1[i] == pl for i, pl in enumerate(probe_labels)]))
    recall = []
    for i, pl in enumerate(probe_labels):
        topk = np.argsort(-sims[i])[:top_k]
        recall.append(1.0 if pl in topk.tolist() else 0.0)
    return {"recall@1": round(correct1, 4), f"recall@{top_k}": round(float(np.mean(recall)), 4)}


def train(cfg: TrainConfig, device: str = "cpu") -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    ds = SyntheticBiometricDataset(
        identities=cfg.train_identities,
        frames_per_identity=cfg.frames_per_identity,
        seed=cfg.seed,
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    encoder = ToyBiometricEncoder().to(device)
    head = ArcFaceHead(num_classes=cfg.train_identities, dim=512,
                       scale=cfg.arcface_scale, margin=cfg.arcface_margin).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        encoder.train()
        head.train()
        total, seen = 0.0, 0
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device)
            emb = encoder(x)
            out_logits = head(emb, y)
            loss = loss_fn(out_logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(y)
            seen += len(y)
        sched.step()
        head.eval()
        encoder.eval()
        with torch.no_grad():
            sample_x = ds.sample(0, 7)[None].to(device)
            _ = encoder(sample_x)
        print(f"  epoch {epoch + 1}/{cfg.epochs}  loss={total / max(seen, 1):.4f}  ({time.perf_counter() - t0:.1f}s)", flush=True)

    metrics = evaluate_recall(encoder, device)
    ckpt_path = out / "toy_encoder.pt"
    torch.save({"state_dict": encoder.state_dict(), "config": asdict(cfg)}, ckpt_path)
    (out / "metrics.json").write_text(json.dumps(metrics | {"wall_s": round(time.perf_counter() - t0, 1)}), encoding="utf-8")
    print(f"[note] checkpoint -> {ckpt_path}")
    print(f"[note] unseen-identity recall@1 / @5 : {metrics['recall@1']} / {metrics['recall@5']}")
    return metrics


if __name__ == "__main__":
    train(TrainConfig())
