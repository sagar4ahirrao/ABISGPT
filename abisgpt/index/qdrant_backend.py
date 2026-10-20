"""Qdrant backend — HNSW with server-side Product Quantization compression."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence

import numpy as np

from abisgpt.config import IndexConfig
from abisgpt.index.base import VectorIndex

# Deterministic namespace: string identity ids -> stable UUID point ids.
_NS = uuid.UUID("3af1c2ab-07b0-4d3e-916a-2fcb1b5a9d55")


def _point_uuid(identity_id: str) -> str:
    return str(uuid.uuid5(_NS, identity_id))


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

    COLLECTION_PREFIX = "abisgpt"

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
            try:
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
                        product=models.ProductQuantizationConfig(
                            compression=(models.CompressionRatio.X16
                                         if self.config.pq_segments == 64 and self.config.pq_bits == 8
                                         else models.CompressionRatio.X4),
                        ),
                    ),
                )
            except Exception as exc:                    # concurrent create race
                if "already exists" not in str(exc):
                    raise

    # ------------------------------------------------------------------ #
    def add(self, ids: Sequence[str], embeddings: np.ndarray) -> None:
        models = self._models
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        points = [
            models.PointStruct(
                id=_point_uuid(identity),
                vector=vec.astype(float).tolist(),
                payload={"identity_id": identity},
            )
            for identity, vec in zip(ids, x, strict=False)
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
        if hasattr(self._client, "query_points"):      # qdrant-client >= 1.10
            res = self._client.query_points(
                collection_name=self.collection,
                query=vec,
                limit=top_k,
                search_params=models.SearchParams(hnsw_ef=self.config.ef_search),
                with_payload=True,
            ).points
        else:                                          # qdrant-client 1.9
            res = self._client.search(
                collection_name=self.collection,
                query_vector=vec,
                limit=top_k,
                search_params=models.SearchParams(hnsw_ef=self.config.ef_search),
                with_payload=True,
            )
        return [(str(hit.payload.get("identity_id", hit.id)), float(hit.score))
                for hit in res]

    def _count_matching(self, identity_ids: list[str]) -> int:
        models = self._models
        return int(self._client.count(
            collection_name=self.collection,
            count_filter=models.Filter(
                must=[models.FieldCondition(
                    key="identity_id",
                    match=models.MatchAny(any=identity_ids),
                )],
            ),
        ).count)

    def remove(self, ids: Iterable[str]) -> int:
        wanted = sorted(set(str(i) for i in ids))
        if not wanted:
            return 0
        before = self._count_matching(wanted)
        self._client.delete(
            collection_name=self.collection,
            points_selector=self._models.PointIdsList(
                points=[_point_uuid(i) for i in wanted],
            ),
            wait=True,
        )
        return before - self._count_matching(wanted)

    def __len__(self) -> int:
        info = self._client.get_collection(self.collection)
        return int(info.points_count or 0)

