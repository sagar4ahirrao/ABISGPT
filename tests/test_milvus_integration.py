"""Live Milvus 2.3 integration — run with:

    ABISGPT_RUN_INTEGRATION=1 pytest tests/test_milvus_integration.py

`docker compose --profile milvus up -d milvus` provides the server.
"""

from __future__ import annotations

import os
import time
import uuid

import numpy as np
import pytest

_ABISGPT_INTEG = os.environ.get("ABISGPT_RUN_INTEGRATION", "") == "1"
_MILVUS_URI = os.environ.get("ABISGPT_MILVUS_URI", "http://localhost:19530")

pytestmark = pytest.mark.skipif(
    not _ABISGPT_INTEG, reason="set ABISGPT_RUN_INTEGRATION=1 (needs `docker compose --profile milvus up -d`)"
)


def _retry_len(index, want: int, tries: int = 12) -> int:
    """Milvus num_entities is eventually consistent — wait briefly."""
    got = len(index)
    for _ in range(tries):
        if got == want:
            break
        time.sleep(1.0)
        got = len(index)
    return got


def _try_connect():
    from pymilvus import connections

    connections.connect(alias="default", uri=_MILVUS_URI, timeout=15)
    return connections


def test_milvus_hnsw_roundtrip():
    _try_connect()

    from abisgpt.config import IndexConfig
    from abisgpt.index.milvus_backend import MilvusIndex

    rng = np.random.default_rng(7)
    n_points, dim = 128, 512
    centers = rng.standard_normal((n_points, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    ids = [f"tpl_{i}" for i in range(n_points)]

    cfg = IndexConfig(backend="milvus", uri=_MILVUS_URI)
    collection = f"abisgpt_it_{uuid.uuid4().hex[:8]}"
    index = MilvusIndex(cfg, collection=collection)
    try:
        index.add(ids, centers)
        got = _retry_len(index, len(ids))
        assert got == len(ids), f"num_entities settles at {got}, expected {len(ids)}"

        hit_id, hit_score = index.search(centers[0], top_k=3)[0]
        assert hit_id == "tpl_0", f"self-query returned {hit_id} (score={hit_score:.4f})"
        assert hit_score > 0.95

        assert index.remove(["tpl_0"]) >= 1
        # functional contract: the deleted template is no longer retrievable
        hits = dict(index.search(centers[0], top_k=5))
        assert "tpl_0" not in hits, f"deleted tpl_0 still returned: {list(hits)}"
        # its runner-up should be a (far less similar) neighbour, proving deletion worked
        assert hits[next(iter(hits))] < hit_score
    finally:
        from pymilvus import utility

        utility.drop_collection(collection)
