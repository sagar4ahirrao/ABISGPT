"""Restart-restore regression: the exact Tier-2 gallery must survive.

The service previously lost its in-memory gallery on restart while a
persistent backend (Qdrant) still held the vectors — `/identities` read
empty while `/health` counted stale points, and old identities could
neither be verified nor erased properly. The pipeline now rebuilds `_gallery`
from an exact backend `items()` snapshot at construction time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abisgpt.config import AbisGptConfig
from abisgpt.encoder import ViTBiometricEncoder
from abisgpt.index.base import VectorIndex
from abisgpt.pipeline.triage import TwoTierPipeline


class _PersistentStub(VectorIndex):
    """VectorIndex whose `items()` acts like a persistent store snapshot
    (exactly what QdrantIndex scroll provides)."""

    def __init__(self, store: dict[str, np.ndarray]):
        self._store = store

    def add(self, ids, embeddings) -> None:
        for uid, vec in zip(ids, np.asarray(embeddings, np.float32),
                            strict=False):
            self._store[uid] = np.asarray(vec, np.float32)

    def search(self, query, top_k: int = 10):
        if not self._store:
            return []
        q = np.asarray(query, np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        hits = [(uid, float(q @ vec)) for uid, vec in self._store.items()]
        hits.sort(key=lambda h: -h[1])
        return hits[:top_k]

    def remove(self, ids) -> int:
        n = 0
        for uid in ids:
            if uid in self._store:
                del self._store[uid]
                n += 1
        return n

    def items(self):
        return [(uid, vec.copy()) for uid, vec in self._store.items()]

    def __len__(self) -> int:
        return len(self._store)


def test_gallery_restores_from_backend_snapshot():
    """A restarted pipeline (fresh in-memory state, persistent backend)
    regains the exact gallery: identities listing + 1:1 verification."""
    cfg = AbisGptConfig()
    enc = ViTBiometricEncoder(strategy="hash")
    frames = _frames()
    template_enc = enc.encode(frames).mean(axis=0)
    template_enc = template_enc / max(float(np.linalg.norm(template_enc)), 1e-9)

    # session 1: enroll, then the process "ends" (store persists)
    store: dict[str, np.ndarray] = {}
    p1 = TwoTierPipeline(encoder=enc, index_backend=_PersistentStub(store),
                         config=cfg)
    p1.enroll("alice", frames)
    assert p1.identities() == ["alice"]

    # session 2: new pipeline over the SAME persistent backend
    p2 = TwoTierPipeline(encoder=enc, index_backend=_PersistentStub(store),
                         config=cfg)
    assert p2.identities() == ["alice"]          # restored, not empty
    assert p2.template("alice").shape == template_enc.shape
    assert float(p2.verifier.score(template_enc[None, :], "alice")) >= 0.99

    # erasure keeps working across restarts
    assert p2.drop("alice") == 1
    p3 = TwoTierPipeline(encoder=enc, index_backend=_PersistentStub(store),
                         config=cfg)
    assert p3.identities() == []


def _frames() -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    return [rng.random((16, 16, 3)).astype(np.float32) for _ in range(4)]
