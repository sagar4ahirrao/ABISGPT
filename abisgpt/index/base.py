"""Backend-agnostic ANN contract (white paper §1.3 / §4)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import numpy as np


class VectorIndex(ABC):
    """Tier-1 vector triage interface.

    Implementations must:
      * accept L2-normalized FP32 embeddings (cosine == dot, white paper
        §4.2 — SIMD dot product on AVX-512).
      * persist/lookup by a user-supplied `id` (string UUID preferred).
      * approximate nearest neighbors in `O(log N)` through an
        IVF-HNSW-PQ graph (or equivalent).
    """

    @abstractmethod
    def add(self, ids: Sequence[str], embeddings: np.ndarray) -> None:
        """Bulk ingest; ids and rows must align."""

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """Return up to `top_k` (id, similarity) pairs in cosine space."""

    @abstractmethod
    def remove(self, ids: Iterable[str]) -> int:
        """Delete ids, return how many were actually dropped."""

    def items(self) -> Iterable[tuple[str, np.ndarray]]:
        """Iterate all stored (id, embedding) pairs. Backends that cannot
        iterate (or would be prohibitively slow) raise NotImplementedError —
        callers must degrade gracefully (empty restored gallery)."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int: ...

