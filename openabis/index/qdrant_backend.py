"""Qdrant backend — HNSW with server-side Product Quantization compression."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from openabis.config import IndexConfig
from openabis.index.base import VectorIndex


def _ensure_qdrant():
    try:
        from qdrant_client import QdrantClient, models  # type: ignore

        return QdrantClient, models
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "qdrant-client is required: `pip install qdrant-client`."
        ) from exc


class QdrantIndex(VectorIndex):
    """Remote Qdrant collection with HNSW + scalar-quantized payload.

    Maps the white paper's `m` (16–32) and `ef_search` (64–256) knobs to
    Qdrant's `hnsw_config.m` / `search_params.hnsw_ef`, and PQ storage to
    `ProductQuantization(compression=x32/x64)`.
    """

    COLLECTION_PREFIX = "openabis"

    def __init__(self, config: IndexConfig | None = None, collection: str | None = None):
        config = config or IndexConfig()
        config.validate()
        self.config = config
        self.collection = collection or f"{self.COLLECTION_PREFIX}_v{config.dim}"
        QdrantClient, models = _ensure_qdrant()
        self._models = models
        self._client = QdrantClient(url=config.uri, timeout=60)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        models = self._models
        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.config.dim,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(
                        m=self.config.m,
                        ef_construct=self.config.ef_construction,
                    ),
                ),
                # Product quantization: 512 floats -> 64 bytes (white paper §4.1)
                quantization_config=models.ProductQuantization(
                    compression=(models.CompressionRatio.X16
                                 if self.config.pq_segments == 64 and self.config.pq_bits == 8
                                 else models.CompressionRatio.X4),
                    always_ram=False,
                ),
            )

    # ------------------------------------------------------------------ #
    def add(self, ids: Sequence[str], embeddings: np.ndarray) -> None:
        models = self._models
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        points = [
            models.PointStruct(id=uid, vector=vec.astype(float).tolist(), payload={})
            for uid, vec in zip(ids, x)
        ]
        # Upload in chunks to respect gRPC message limits
        for start in range(0, len(points), 256):
            self._client.upsert(
                collection_name=self.collection,
                points=points[start : start + 256],
                wait=True,
            )

    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        models = self._models
        vec = np.asarray(query, dtype=np.float32).reshape(-1).tolist()
        res = self._client.search(
            collection_name=self.collection,
            query_vector=vec,
            limit=top_k,
            search_params=models.SearchParams(hnsw_ef=self.config.ef_search),
            with_payload=False,
        )
        return [(str(hit.id), float(hit.score)) for hit in res]

    def remove(self, ids: Iterable[str]) -> int:
        wanted = list(set(ids))
        if not wanted:
            return 0
        res = self._client.delete(
            collection_name=self.collection,
            points_selector=self._models.PointIdsList(points=wanted),
            wait=True,
        )
        return int(getattr(res, "operation_count", len(wanted)) or 0)

    def __len__(self) -> int:
        info = self._client.get_collection(self.collection)
        return int(info.points_count or 0)
