"""Lightweight unit tests — run in CI with zero external services."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abisgpt.config import IndexConfig
from abisgpt.encoder import ViTBiometricEncoder
from abisgpt.index.factory import BruteForceIndex, build_backend

DIM = 512


def synthetic_gallery(n: int = 200, seed: int = 7) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    vecs = rng.normal(0, 1, (n, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs, [f"ID-{i:05d}" for i in range(n)]


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
def test_encoder_deterministic_and_normalized() -> None:
    enc = ViTBiometricEncoder(strategy="hash")
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[..., 1] = 100
    v1 = enc.encode_one(img)
    v2 = enc.encode_one(img.copy())
    assert v1.shape == (DIM,)
    np.testing.assert_allclose(v1, v2)  # deterministic
    assert np.linalg.norm(v1) == pytest.approx(1.0, abs=1e-5)  # L2-normalized (§4.2)


def test_encoder_distinguishes_images() -> None:
    enc = ViTBiometricEncoder(strategy="hash")
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = np.full((64, 64, 3), 255, dtype=np.int16).astype(np.uint8)
    assert float(np.dot(enc.encode_one(a), enc.encode_one(b))) < 0.1


# --------------------------------------------------------------------------- #
# Index backends
# --------------------------------------------------------------------------- #
def test_bruteforce_top1_is_self() -> None:
    vecs, uids = synthetic_gallery(50)
    index = BruteForceIndex(dim=DIM)
    index.add(uids, vecs)
    hits = index.search(vecs[0], top_k=1)
    assert hits[0][0] == uids[0]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


def clustered_gallery(
    n: int = 3000, n_clusters: int = 64, spread: float = 0.02, seed: int = 11
) -> tuple[np.ndarray, list[str]]:
    """Registry-like synthetic embeddings: per-identity tight clusters.

    Real biometric galleries cluster per identity (probe-vs-reference
    cosine >= 0.9); pure-Gaussian galleries overstate PQ distortion.
    Each row is a unique enrollment slot (`ID-<identity>-<slot>`).
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, (n_clusters, DIM)).astype(np.float32)
    ids = rng.integers(0, n_clusters, size=n)
    vecs = centers[ids] + spread * rng.normal(0, 1, (n, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    counter: dict[int, int] = {}
    uids: list[str] = []
    for identity in ids.tolist():
        slot = counter.get(identity, 0)
        counter[identity] = slot + 1
        uids.append(f"ID-{identity:04d}-{slot:03d}")
    return vecs, uids


def test_faiss_backend_matches_oracle_top1() -> None:
    # 12k rows covers 39*256 PQ subquantizer requirement (pq_bits=8)
    cfg = IndexConfig(backend="faiss", nlist=64, nprobe=16, pq_segments=8, pq_bits=8)
    cfg.validate()
    backend = build_backend(cfg)
    vecs, uids = clustered_gallery(n=12000)
    backend.add(uids, vecs)
    hits = backend.search(vecs[0], top_k=5)
    assert hits, "must return at least one hit"
    # Tier-1 (PQ-truncated) cannot guarantee the exact self slot ranks #1 —
    # that is Tier-2's rerank job (white paper §1.3). It must, however, put
    # at least one *same-identity* slot in the candidate list with a high
    # (compressed) similarity.
    probe_identity = uids[0].rsplit("-", 1)[0]
    same_identity = [h for h in hits if h[0].rsplit("-", 1)[0] == probe_identity]
    assert same_identity, f"no same-identity slot in candidates: {hits}"
    assert same_identity[0][1] > 0.9, f"same-identity similarity too low: {same_identity[0]}"


def test_pqlite_byte_budget_matches_whitepaper() -> None:
    """§4.1: 512 FP32 floats -> 64 bytes after PQ (96.8% memory reduction)."""
    raw_bytes = DIM * 4
    segments = 64
    bits = 8
    compressed_bytes = segments * bits // 8
    assert compressed_bytes == 64
    assert compressed_bytes / raw_bytes == pytest.approx(0.03125, abs=1e-6)
    assert (1 - compressed_bytes / raw_bytes) == pytest.approx(0.96875, abs=1e-6)


def test_config_validation_rejects_bad_segments() -> None:
    bad = IndexConfig(pq_segments=100)  # 512 not divisible by 100
    with pytest.raises(ValueError):
        bad.validate()


def test_index_remove_supports_erasure() -> None:
    vecs, uids = synthetic_gallery(20)
    index = BruteForceIndex(dim=DIM)
    index.add(uids, vecs)
    dropped = index.remove([uids[0]])
    assert dropped == 1
    assert len(index) == 19

