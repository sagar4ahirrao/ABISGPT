"""Procedural biometric identity dataset for encoder training (CPU-friendly).

Each identity is a stable set of low-frequency sinusoidal weights laid over
a shared face-like radial gradient — generic enough for a small CNN to
learn transferable embedding geometry across *unseen* identities, whose
weights are sampled at eval time.

NOT real biometrics — train/eval mechanics only, zero privacy implications.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

IMG = 48
N_BASIS = 12


class SyntheticBiometricDataset(Dataset):
    """Few-shot biometric images: `n_identities`, `frames_per_identity`.

    Image K = base skin gradient + Σ_j w_j^id * pattern_j + noise
    Per-identity weights are seeded deterministically — reproducible.
    """

    def __init__(
        self,
        identities: int,
        frames_per_identity: int = 48,
        seed: int = 0,
        noise_std: float = 0.08,
        identity_offset: int = 0,   # unseen identities for eval
        encode_range: int = 10_000,
    ):
        super().__init__()
        self.n = identities * frames_per_identity
        self.frames_per_identity = frames_per_identity
        self.noise_std = noise_std
        self.rng = None
        self._seed = seed
        self.identity_offset = identity_offset
        self.encode_range = encode_range

        yy, xx = np.meshgrid(np.linspace(-1, 1, IMG), np.linspace(-1, 1, IMG), indexing="ij")
        r = np.sqrt(xx**2 + yy**2)
        radial = np.exp(-((r - 0.45) ** 2) / 0.08)
        self.base = 0.35 + 0.25 * radial  # face-ish pale oval on dark bg
        # per-identity low-frequency weights
        self.weights = np.random.default_rng(seed + identity_offset).normal(
            0, 1, (identities, N_BASIS * 2)
        ).astype(np.float32)
        grid = np.stack(np.meshgrid(np.linspace(-1, 1, IMG), np.linspace(-1, 1, IMG), indexing="ij"), 0)
        self._patterns: list = []
        for j in range(N_BASIS):
            fx, fy = 1 + (j % 4), 1 + (j % 3)
            gx, gy = grid[0], grid[1]
            self._patterns += [
                np.sin(2 * np.pi * fx * gx + 0.1 * j) * np.cos(2 * np.pi * fy * gy - 0.05 * j),
                np.cos(2 * np.pi * fx * gy + 0.1 * j) * np.sin(2 * np.pi * fy * gx - 0.05 * j),
            ]
        self._patterns_t = torch.stack([torch.from_numpy(p.copy()).float() for p in self._patterns])

    def __len__(self) -> int:
        return self.n

    def sample(self, identity: int, idx: int, noise_std: float | None = None) -> torch.Tensor:
        """Deterministic `idx`-th frame for a given identity id (0-based)."""
        w = self.weights[identity % len(self.weights)]  # (2*N_BASIS,)
        field = sum(float(w[j]) * self._patterns_t[j] for j in range(self._patterns_t.shape[0]))
        scale = 0.18
        img = (
            self._base_tensor()
            + scale * field[None, :, :] * torch.ones(3, IMG, IMG)
            + torch.randn(3, IMG, IMG) * (noise_std if noise_std is not None else self.noise_std)
        )
        return torch.clamp(img, 0, 1)

    def _base_tensor(self) -> torch.Tensor:
        import numpy as np

        return torch.from_numpy(np.broadcast_to(self.base, (3, IMG, IMG)).copy()).float()

    def __getitem__(self, i: int) -> tuple:
        identity = i // self.frames_per_identity
        idx = i % self.frames_per_identity
        return self.sample(identity, idx), identity

