"""Live Qdrant integration — run with:

    OPENABIS_RUN_INTEGRATION=1 OPENABIS_QDRANT_URI=http://localhost:6333 pytest tests/test_qdrant_integration.py

`docker compose up -d qdrant` provides the server.
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

_OPENABIS_INTEG = os.environ.get("OPENABIS_RUN_INTEGRATION", "") == "1"
_QDRANT_URI = os.environ.get("OPENABIS_QDRANT_URI", "http://localhost:6333")

pytestmark = pytest.mark.skipif(
    not _OPENABIS_INTEG, reason="set OPENABIS_RUN_INTEGRATION=1 (needs `docker compose up -d qdrant`)"
)


def _try_client():
    from qdrant_client import QdrantClient

    client = QdrantClient(url=_QDRANT_URI, timeout=10)
    client.get_collections()
    return client


def test_qdrant_hnsw_roundtrip():
    client = _try_client()

    from openabis.config import IndexConfig
    from openabis.index.qdrant_backend import QdrantIndex

    rng = np.random.default_rng(7)
    n_points, dim = 128, 512
    centers = rng.standard_normal((n_points, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    embeddings = centers
    ids = [f"tpl_{i}" for i in range(n_points)]

    cfg = IndexConfig(backend="qdrant", uri=_QDRANT_URI)
    collection = f"openabis_it_{uuid.uuid4().hex[:8]}"
    index = QdrantIndex(cfg, collection=collection)
    try:
        index.add(ids, embeddings)
        assert len(index) == len(ids)

        hit_id, hit_score = index.search(embeddings[0], top_k=3)[0]
        assert hit_id == "tpl_0", f"self-query returned {hit_id} (score={hit_score:.4f})"
        assert hit_score > 0.95

        assert index.remove(["tpl_0"]) >= 1
        assert len(index) == n_points - 1
    finally:
        client.delete_collection(collection)
