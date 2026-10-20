"""Milvus 2.x backend — HNSW / IVF hybrid mode."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from abisgpt.config import IndexConfig
from abisgpt.index.base import VectorIndex


def _ensure_milvus():
    try:
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections

        return connections, Collection, CollectionSchema, DataType, FieldSchema
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pymilvus is required: `pip install pymilvus`.") from exc


class MilvusIndex(VectorIndex):
    """Milvus collection using HNSW index with COSINE metric.

    Milvus does not expose raw `ef_search` per query the way FAISS does —
    instead both `search_params.nprobe` (IVF) and `search_params.ef` (HNSW)
    pass through to underlying graphs. Milvus 2.x honors `ef` for HNSW
    indexes and `nprobe` for IVF variants.
    """

    def __init__(self, config: IndexConfig | None = None, collection: str = "abisgpt"):
        config = config or IndexConfig()
        config.validate()
        self.config = config
        self.collection_name = collection
        connections, Collection, CollectionSchema, DataType, FieldSchema = _ensure_milvus()
        self._connections = connections
        self._Collection = Collection
        self._CollectionSchema = CollectionSchema
        self._DataType = DataType
        self._FieldSchema = FieldSchema

        alias = getattr(config, "uri", "") or "http://localhost:19530"
        try:
            connections.connect(alias="default", uri=alias)
        except Exception as exc:  # pragma: no cover — network layer
            raise RuntimeError(
                f"Could not connect to Milvus at {alias}. Start>: `milvus standalone`."
            ) from exc
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        DT = self._DataType
        FS = self._FieldSchema
        from pymilvus import utility  # type: ignore — lazy

        if utility.has_collection(self.collection_name):
            self.collection = self._Collection(self.collection_name)
            return
        schema = self._CollectionSchema(
            fields=[
                FS("id", DT.VARCHAR, max_length=64, is_primary=True),
                FS("embedding", DT.FLOAT_VECTOR, dim=self.config.dim),
            ],
            description="ABISGPT Tier-1 vector triage registry",
        )
        col = self._Collection(self.collection_name, schema=schema)
        col.create_index(
            field_name="embedding",
            index_params={
                "index_type": "HNSW",  # graph index: constant-factor O(log N) hops
                "metric_type": "COSINE",
                "params": {"M": self.config.m, "efConstruction": self.config.ef_construction},
            },
        )
        self.collection = col

    def add(self, ids: Sequence[str], embeddings: np.ndarray) -> None:
        ids_list = [str(u) for u in ids]
        vec_list = np.asarray(embeddings, dtype=np.float32).tolist()
        # pymilvus 2.x columnar schema order must match the schema fields
        self.collection.insert([ids_list, vec_list])
        self.collection.flush()

    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        self.collection.load()
        res = self.collection.search(
            data=[np.asarray(query, dtype=np.float32).reshape(-1).tolist()],
            anns_field="embedding",
            param={
                "metric_type": "COSINE",
                "params": {
                    "ef": self.config.ef_search,
                    "nprobe": self.config.nprobe,
                },
            },
            limit=top_k,
            output_fields=["id"],
        )
        hits = res[0]
        return [(str(h.id), float(h.score)) for h in hits]

    def remove(self, ids: Iterable[str]) -> int:
        expr = ",".join(f'"{u}"' for u in set(ids))
        res = self.collection.delete(f"id in [{expr}]")
        self.collection.flush()
        return int(getattr(res, "delete_count", 0) or 0)

    def __len__(self) -> int:
        self.collection.flush()
        return int(self.collection.num_entities)

