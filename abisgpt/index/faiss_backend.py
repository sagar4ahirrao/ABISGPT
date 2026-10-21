"""FAISS IVF-HNSW + Product Quantization backend (white paper §4.1)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from abisgpt.config import IndexConfig
from abisgpt.index.base import VectorIndex


def _ensure_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "faiss is required for the FAISS backend. Install with `pip install faiss-cpu`."
        ) from exc


class FaissIndex(VectorIndex):
    """IVF-HNSW-PQ composite index.

    Composition:
      * Coarse quantizer   = `IndexHNSWFlat` (graph-based IVF allocation),
      * Fine residual code = `IndexIVFPQ` with `pq_segments` × `pq_bits` code.

    GPU vector search (`config.use_gpu=True` / env `ABISGPT_INDEX_GPU=1`,
    `pip install abisgpt[gpu]`): the trained IVFPQ index is transparently
    moved to the CUDA device via `index_cpu_to_gpu` and every search runs
    on-GPU. With no CUDA device / missing wheel the constructor falls back
    to CPU execution and records `gpu="unavailable"` — behaviour is
    identical either way.

    NOTE: stored vectors are PQ *codes* — `items()` deliberately stays
    unimplemented (default `NotImplementedError`) because reconstruction
    would yield approximate templates; Tier-2 precision must not silently
    degrade to approximations on gallery restore.
    """

    def __init__(self, config: IndexConfig | None = None):
        config = config or IndexConfig()
        config.validate()
        self.config = config
        faiss = _ensure_faiss()
        d = config.dim
        m = int(min(config.m, d - 1))  # HNSW M must be < dim
        coarse = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
        coarse.hnsw.efConstruction = config.ef_construction
        self._cpu_index = faiss.IndexIVFPQ(
            coarse,
            d,
            config.nlist,
            config.pq_segments,
            config.pq_bits,
            faiss.METRIC_INNER_PRODUCT,
        )
        self._cpu_index.nprobe = config.nprobe
        # GPU transparently hosts the IVFPQ index; CPU fallback otherwise.
        self.gpu: str = "off"
        self._resources = None
        if config.use_gpu:
            if hasattr(faiss, "StandardGpuResources") and hasattr(
                    faiss, "index_cpu_to_gpu"):
                try:
                    self._resources = faiss.StandardGpuResources()
                    self._index = faiss.index_cpu_to_gpu(
                        self._resources, 0, self._cpu_index)
                    self.gpu = "cuda"
                except RuntimeError:    # no CUDA device on this host
                    self._index = self._cpu_index
                    self.gpu = "unavailable"
            else:
                self._index = self._cpu_index
                self.gpu = "unavailable"
        else:
            self._index = self._cpu_index
        self._index.nprobe = config.nprobe
        self._hnsw = coarse                     # live efSearch re-tune handle
        self._ids: list[str] = []
        self._uid_set: set[str] = set()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _sanitize(self, emb: np.ndarray) -> np.ndarray:
        """L2-normalize (white paper §4.2) so cosine == dot product."""
        x = np.ascontiguousarray(np.asarray(emb, dtype=np.float32))
        if x.ndim == 1:
            x = x[None, :]
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return (x / np.maximum(norms, 1e-9)).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Ingest
    # ------------------------------------------------------------------ #
    def add(self, ids: Sequence[str], embeddings: np.ndarray) -> None:
        if len(set(ids)) != len(ids):
            raise ValueError("FaissIndex.add received duplicate ids.")
        if len(ids) != len(embeddings):
            raise ValueError("ids and embeddings length mismatch.")
        x = self._sanitize(embeddings)
        if not self._index.is_trained:
            # IVFPQ clustering needs >= 39 training rows per centroid, both for
            # the IVF coarse cells and for each PQ subquantizer dataset.
            pq_clusters = 2 ** self.config.pq_bits
            required = max(39 * self.config.nlist, 39 * pq_clusters)
            if len(x) < required:
                raise RuntimeError(
                    f"FaissIndex needs >= {required} vectors in one `add()` batch to "
                    f"train IVF-HNSW-PQ (nlist={self.config.nlist}, "
                    f"pq_segments={self.config.pq_segments}). Options: "
                    "(a) bulk-enroll in batches, "
                    "(b) lower `nlist`/`pq_segments` in configs/model.yaml, or "
                    "(c) switch `index.backend: brute` for small galleries."
                )
            self._index.train(x)
        self._index.add(x)
        self._ids.extend(ids)
        self._uid_set.update(ids)

    # ------------------------------------------------------------------ #
    # Search — Tier-1 triage (top-k candidates on IVF-HNSW-PQ)
    # ------------------------------------------------------------------ #
    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        q = self._sanitize(query)
        try:
            self._hnsw.hnsw.efSearch = self.config.ef_search  # live re-tune
        except AttributeError:      # quantizer swapped under the GPU host
            pass
        self._index.nprobe = self.config.nprobe
        sims, idx = self._index.search(q, top_k)
        hits: list[tuple[str, float]] = []
        for pos, sim in zip(idx[0], sims[0], strict=False):
            pos = int(pos)
            if pos < 0 or pos >= len(self._ids):
                continue
            hits.append((self._ids[pos], float(sim)))
        hits.sort(key=lambda h: h[1], reverse=True)
        return hits

    # ------------------------------------------------------------------ #
    # Management
    # ------------------------------------------------------------------ #
    def remove(self, ids: Iterable[str]) -> int:
        wanted = [u for u in set(ids) if u in self._uid_set]
        if not wanted:
            return 0
        positions = {uid: i for i, uid in enumerate(self._ids) if uid in self._uid_set}
        pos_arr = np.asarray([positions[u] for u in wanted], dtype=np.int64)
        faiss = _ensure_faiss()
        selector = faiss.IDSelectorBatch(pos_arr.size, faiss.swig_ptr(pos_arr))
        before = int(self._index.ntotal)
        self._index.remove_ids(selector)
        removed = before - int(self._index.ntotal)
        drop = set(wanted)
        self._ids = [u for u in self._ids if u not in drop]
        self._uid_set -= drop
        return removed

    def __len__(self) -> int:
        return int(self._index.ntotal)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FaissIndex(ntotal={len(self)}, nlist={self.config.nlist}, "
            f"nprobe={self.config.nprobe}, pq_segments={self.config.pq_segments}, "
            f"pq_bits={self.config.pq_bits}, m={self.config.m}, "
            f"ef_search={self.config.ef_search})"
        )

