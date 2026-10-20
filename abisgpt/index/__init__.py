"""Vector-index backends (Tier-1 ANN triage, white paper §4)."""

from abisgpt.index.base import VectorIndex
from abisgpt.index.factory import build_backend

__all__ = ["VectorIndex", "build_backend", "FaissIndex", "QdrantIndex", "MilvusIndex"]


def __getattr__(name):
    if name == "FaissIndex":
        from abisgpt.index.faiss_backend import FaissIndex

        return FaissIndex
    if name == "QdrantIndex":
        from abisgpt.index.qdrant_backend import QdrantIndex

        return QdrantIndex
    if name == "MilvusIndex":
        from abisgpt.index.milvus_backend import MilvusIndex

        return MilvusIndex
    raise AttributeError(f"module 'abisgpt.index' has no attribute {name!r}")

