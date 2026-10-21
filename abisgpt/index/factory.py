"""Backend factory + in-memory NumPy fallback for tests and CI."""

from __future__ import annotations

from collections.abc import Iterable  # noqa: E402

import numpy as np

from abisgpt.config import IndexBackend, IndexConfig
from abisgpt.index.base import VectorIndex


class BruteForceIndex(VectorIndex):
    """Exact cosine search over NumPy矩阵 — Tier-2 precision verification.

    Also serves as the ground-truth oracle for backend benchmarking and
    for tiny galleries (< few thousand) where wn ANN graph is overhead.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None

    @property
    def _mat(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        return self._matrix

    def add(self, ids, embeddings) -> None:
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {x.shape[1]}.")
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norms, 1e-9)
        self._ids.extend(ids)
        self._matrix = np.vstack([self._mat, x]) if len(self._mat) else x

    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        sims = (self._mat @ q).astype(np.float32)
        order = np.argsort(-sims)[:top_k]
        return [(self._ids[int(i)], float(sims[i])) for i in order]

    def remove(self, ids) -> int:
        drop = {uid for uid in ids}
        keep = [(i, uid) for i, uid in enumerate(self._ids) if uid not in drop]
        dropped = len(self._ids) - len(keep)
        if dropped:
            keep_rows = [i for i, _ in keep]
            self._matrix = self._mat[keep_rows]
            self._ids = [uid for _, uid in keep]
        return dropped

    def items(self) -> Iterable[tuple[str, np.ndarray]]:
        for uid, row in zip(self._ids, self._mat, strict=False):
            yield uid, row.copy()

    def __len__(self) -> int:
        return len(self._ids)


def build_backend(config: IndexConfig | None = None) -> VectorIndex:
    """Instantiate a Tier-1 backend, or fall back to BruteForceIndex if the
    requested wheel is not installed (keeps CI green with zero extras).

    Passing `backend="brute"` explicitly pins the exact-search oracle.
    """
    config = config or IndexConfig()
    backend: IndexBackend | str = getattr(config, "backend", "faiss")
    if backend == "brute":
        return BruteForceIndex(dim=config.dim)
    if backend == "faiss":
        try:
            from abisgpt.index.faiss_backend import FaissIndex

            return FaissIndex(config)
        except ImportError:
            return BruteForceIndex(dim=config.dim)

    if backend == "faiss-gpu":
        try:
            from abisgpt.index.faiss_backend import FaissIndex

            config.use_gpu = True
            return FaissIndex(config)
        except ImportError:
            return BruteForceIndex(dim=config.dim)
    if backend == "qdrant":
        from abisgpt.index.qdrant_backend import QdrantIndex

        return QdrantIndex(config)
    if backend == "milvus":
        from abisgpt.index.milvus_backend import MilvusIndex

        return MilvusIndex(config)
    raise ValueError(f"Unknown index backend: {backend!r}")

