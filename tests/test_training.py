"""Training mechanics smoke test — tiny data, one forward/backward, no GPU."""

from __future__ import annotations

import torch

from abisgpt.training.model import ArcFaceHead, ToyBiometricEncoder
from abisgpt.training.synthdata import SyntheticBiometricDataset


def test_dataset_shapes_and_determinism():
    ds = SyntheticBiometricDataset(identities=4, frames_per_identity=8, seed=0)
    assert len(ds) == 32
    x, y = ds[0]
    assert x.shape == (3, 48, 48)
    assert y == 0
    x2 = ds.sample(0, 7)             # noise differs but identity weights are stable
    assert x2.shape == (3, 48, 48)
    w1 = SyntheticBiometricDataset(identities=4, frames_per_identity=8, seed=0).weights
    w2 = SyntheticBiometricDataset(identities=4, frames_per_identity=8, seed=0).weights
    assert torch.equal(torch.from_numpy(w1), torch.from_numpy(w2))


def test_arcface_forward_backward():
    torch.manual_seed(0)
    enc = ToyBiometricEncoder(dim=64)
    head = ArcFaceHead(num_classes=8, dim=64)
    x = torch.rand(4, 3, 48, 48)
    emb = enc(x)
    assert emb.shape == (4, 64)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(4), atol=1e-4)

    labels = torch.tensor([0, 3, 7, 2])
    logits = head(emb, labels)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert head.weight.grad is not None
    assert torch.isfinite(loss)


def test_arcface_eval_mode_returns_plain_logits():
    head = ArcFaceHead(num_classes=5, dim=16)
    head.eval()
    emb = torch.nn.functional.normalize(torch.rand(3, 16), dim=-1)
    out_train = head(emb, torch.zeros(3, dtype=torch.long))
    out_eval = head(emb)
    assert out_eval.shape == out_train.shape

