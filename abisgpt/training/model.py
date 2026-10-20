"""Compact ViT-flavoured CNN embedding net (ViT would be overkill for a CPU
toy run; this is a *mechanics* model mirroring `vit_encoder` proportions).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyBiometricEncoder(nn.Module):
    """3-conv CNN → 512-D L2-normalized embedding (same 512-D contract)."""

    def __init__(self, dim: int = 512, channels=(3, 32, 64, 96)):
        super().__init__()
        c1, c2, c3 = channels[1], channels[2], channels[3]
        self.features = nn.Sequential(
            nn.Conv2d(channels[0], c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(c3, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (N, C3, h', w')
        x = x.flatten(1)              # CAM pooling -> (N, C3)
        return F.normalize(self.head(x), dim=-1)


class ArcFaceHead(nn.Module):
    """Additive angular margin (white paper §3.2): s·cos(θ + m) classifier."""

    def __init__(self, num_classes: int, dim: int = 512, scale: float = 24.0, margin: float = 0.22):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, dim))
        self.scale = scale
        self.margin = margin

    def forward(self, emb: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        w = F.normalize(self.weight, dim=-1)
        e = emb / emb.norm(dim=-1, keepdim=True)
        logits = self.scale * (e @ w.T)                 # s·cosθ
        if labels is None or not self.training:
            return logits
        one_hot = F.one_hot(labels, num_classes=w.shape[0]).float()
        # θ target: cos(θ - m) ≈ cosθ·cosm − sinθ·sinm (standard additive margin)
        cos_m = float(torch.cos(torch.tensor(self.margin)))
        sin_m = float(torch.sin(torch.tensor(self.margin)))
        theta_cos = logits / self.scale
        theta_target = theta_cos * cos_m - torch.sqrt(torch.clamp(1.0 - theta_cos**2, 1e-8)) * sin_m
        out = torch.where(one_hot.bool(), self.scale * theta_target, logits)
        return out

