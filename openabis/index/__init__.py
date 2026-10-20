"""Vector-index backends (Tier-1 ANN triage, white paper §4)."""

from openabis.index.base import VectorIndex
from openabis.index.factory import build_backend

__all__ = ["VectorIndex", "build_backend", "FaissIndex", "QdrantIndex", "MilvusIndex"]


def __getattr__(name):
    if name == "FaissIndex":
        from openabis.index.faiss_backend import FaissIndex

        return FaissIndex
    if name == "QdrantIndex":
        from openabis.index.qdrant_backend import QdrantIndex

        return QdrantIndex
    if name == "MilvusIndex":
        from openabis.index.milvus_backend import MilvusIndex

        return MilvusIndex
    raise AttributeError(f"module 'openabis.index' has no attribute {name!r}")
