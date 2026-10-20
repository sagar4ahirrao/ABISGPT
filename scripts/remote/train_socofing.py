#!/usr/bin/env python
"""SOCOFing fingerprint ArcFace fine-tune (remote GPU run).

Identity-level split (60 train / 40 eval ids, seed 0): train encoder+ArcFace
head on 60 unseen-people's 600 images; evaluate on the 40 eval ids with
5 template / 5 probe frames each (200 probes, 40 templates).
Output: /home/development/abisgpt/out/socofing_metrics.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA = Path.home() / "abisgpt" / "fp_sokocoventry.npz"
OUT = Path.home() / "abisgpt" / "out" / "socofing_metrics.json"
OUT.parent.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 150
BATCH = 128


def augment(x: torch.Tensor) -> torch.Tensor:
    """Per-sample small shift + rotation — the classic overfit shield for
    tiny fingerprint sets (60 ids x 10 frames)."""
    b = x.size(0)
    ang = (torch.rand(b) * 16 - 8)                          # ±8 degrees
    scale = 1.0 + (torch.rand(b) * 0.16 - 0.08)
    tx = (torch.rand(b) * 6 - 3) / (64 / 2)                 # ±3 px
    ty = (torch.rand(b) * 6 - 3) / (64 / 2)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0] = cos * scale
    theta[:, 0, 1] = -sin * scale
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin * scale
    theta[:, 1, 1] = cos * scale
    theta[:, 1, 2] = ty
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    return torch.nn.functional.grid_sample(
        x, grid, padding_mode="reflection", align_corners=False)


class FingerNet(nn.Module):
    """Minutiae-friendly stack: BN per conv, stride down-samples, maxpool."""
    def __init__(self, dim: int = 512):
        super().__init__()
        c = [16, 32, 64, 128]
        layers = []
        ch = 3
        for ci in c:
            layers += [
                nn.Conv2d(ch, ci, 3, padding=1, bias=False),
                nn.BatchNorm2d(ci), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 64->32->16->8->4
            ]
            ch = ci
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, dim, bias=False)

    def forward(self, x):
        x = self.body(x)
        x = self.pool(x).flatten(1)
        return torch.nn.functional.normalize(self.head(x), dim=-1)


class ArcFace(nn.Module):
    def __init__(self, dim, n_cls, s=16.0, m=0.10):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_cls, dim) * 0.01)
        self.s, self.m = s, m

    def forward(self, emb, y=None):
        w = self.w / self.w.norm(dim=1, keepdim=True)
        cos = emb @ w.T
        if y is None:
            return cos
        cos_y = cos.gather(1, y.view(-1, 1))
        cos_m = torch.cos(torch.acos(cos_y.clamp(-1 + 1e-7, 1 - 1e-7)) + self.m)
        oh = torch.nn.functional.one_hot(y, cos.size(1)).bool()
        # non-inplace (autograd-safe) class-margin substitution
        return self.s * torch.where(oh, cos_m, cos)


def main():
    z = np.load(DATA)
    frames = z["frames"]                       # (1000,64,64,3) [0,1]
    labels = z["labels"].astype(np.int64)
    ids = np.unique(labels)
    rng = np.random.default_rng(0)
    perm = rng.permutation(ids)
    train_ids, eval_ids = perm[:60], perm[60:]
    cls_of = {int(c): i for i, c in enumerate(train_ids)}   # label -> 0..59
    ti = np.concatenate([np.flatnonzero(labels == c) for c in train_ids])
    train_x = torch.from_numpy(frames[ti]).permute(0, 3, 1, 2).contiguous()
    train_y = torch.tensor([cls_of[int(v)] for v in labels[ti]], dtype=torch.long)
    loader = DataLoader(TensorDataset(train_x, train_y),
                        batch_size=BATCH, shuffle=True)

    enc, head = FingerNet().to(DEVICE), ArcFace(512, 60).to(DEVICE)
    opt = torch.optim.AdamW([*enc.parameters(), *head.parameters()], lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        enc.train()
        head.train()
        tot, seen = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE)
            loss = ce(head(enc(augment(x)), y), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(y)
            seen += len(y)
        sched.step()
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"epoch {ep:2d}/{EPOCHS} loss {tot/max(seen,1):.4f}", flush=True)

    # ---- eval on UNSEEN 40 fingerprints ---------------------------------
    enc.eval()
    with torch.no_grad():
        templates = {}
        for c in eval_ids:
            idx = np.flatnonzero(labels == c)
            tpl_idx, probe_idx = idx[:5], idx[5:]     # template 5 / probe 5
            e_t = torch.from_numpy(frames[tpl_idx]).permute(0, 3, 1, 2).to(DEVICE).float()
            e_p = torch.from_numpy(frames[probe_idx]).permute(0, 3, 1, 2).to(DEVICE).float()
            tp = enc(e_t).cpu().numpy().mean(0)
            tp /= max(np.linalg.norm(tp), 1e-9)
            templates[int(c)] = (tp, enc(e_p).cpu().numpy())
    keys = list(templates)
    T = np.stack([templates[k][0] for k in keys])
    genuine, impostor, rank1 = [], [], 0
    for j, k in enumerate(keys):
        P = templates[k][1]
        sims = P @ T.T
        for i in range(len(P)):
            if int(np.argmax(sims[i])) == j:
                rank1 += 1
        rank5 = sum(int(j in np.argsort(-sims[i])[:5]) for i in range(len(P)))
        genuine.extend(sims[:, j].tolist())
        for j2 in range(len(keys)):
            if j2 != j:
                impostor.extend(sims[:, j2].tolist())
    genuine, impostor = np.array(genuine), np.array(impostor)
    thr = np.unique(np.concatenate([np.sort(impostor), np.sort(genuine)]))
    far = np.array([(impostor >= t).mean() for t in thr])
    frr = np.array([(genuine < t).mean() for t in thr])
    i = int(np.argmin(np.abs(far - frr)))

    # ---- baseline: UNtrained raw-pixel cosine, identical protocol --------
    raw_t = {}
    for c in eval_ids:
        idx = np.flatnonzero(labels == c)
        mean5 = frames[idx[:5]].reshape(5, -1).mean(0)
        mean5 /= max(np.linalg.norm(mean5), 1e-9)
        probes = frames[idx[5:]].reshape(5, -1)
        raw_t[int(c)] = (mean5,
                         probes / (np.linalg.norm(probes, axis=1,
                                                  keepdims=True) + 1e-9))
    rk = list(raw_t)
    RT = np.stack([raw_t[k][0] for k in rk])
    rg, ri, br1 = [], [], 0
    for j, k in enumerate(rk):
        sims = raw_t[k][1] @ RT.T
        for r in range(len(sims)):
            if int(np.argmax(sims[r])) == j:
                br1 += 1
        rg.extend(sims[:, j].tolist())
        for j2 in range(len(rk)):
            if j2 != j:
                ri.extend(sims[:, j2].tolist())
    rg, ri = np.array(rg), np.array(ri)
    thr_r = np.unique(np.concatenate([np.sort(ri), np.sort(rg)]))
    far_r = np.array([(ri >= t).mean() for t in thr_r])
    frr_r = np.array([(rg < t).mean() for t in thr_r])
    i_r = int(np.argmin(np.abs(far_r - frr_r)))

    n_probe = sum(len(templates[k][1]) for k in keys)
    metrics = {
        "protocol": ("identity-split: 60 train ids unseen-by-eval; eval 40 ids, "
                     "5 template / 5 probe frames each"),
        "epochs": EPOCHS, "device": DEVICE,
        "final_loss": round(tot / max(seen, 1), 4),
        "train_s": round(time.perf_counter() - t0, 1),
        "recall@1": round(rank1 / n_probe, 4),
        "recall@5": round(rank5 / n_probe, 4),
        "mean_genuine_sim": round(float(genuine.mean()), 4),
        "mean_impostor_sim": round(float(impostor.mean()), 4),
        "eer": round(float((far[i] + frr[i]) / 2), 4),
        "baseline_raw_pixel": {
            "recall@1": round(br1 / n_probe, 4),
            "eer": round(float((far_r[i_r] + frr_r[i_r]) / 2), 4),
            "mean_genuine_sim": round(float(rg.mean()), 4),
            "mean_impostor_sim": round(float(ri.mean()), 4),
        },
        "n_probes": int(n_probe), "n_class_templates": len(keys),
        "n_impostor": int(impostor.size),
    }
    OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
