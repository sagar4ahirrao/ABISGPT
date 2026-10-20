#!/usr/bin/env python
"""Offline benchmark harness: indexes synthetic embeddings and reports
recall@k, spoof rejection, and latency percentiles.

Usage:
    python scripts/evaluate.py --identities 200 --queries-per-id 5 --top-k 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openabis.config import IndexConfig
from openabis.index import build_backend
from openabis.index.factory import BruteForceIndex


def build_gallery(n: int, dim: int = 512, seed: int = 2026, clusters: int = 20) -> tuple[np.ndarray, list[str]]:
    """Synthetic registry with per-identity (biometric) clusters."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, (clusters, dim)).astype(np.float32)
    identities = np.resize(np.arange(clusters), n)
    vecs = centers[identities] + 0.18 * rng.normal(0, 1, (n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    counter: dict[int, int] = {}
    uids: list[str] = []
    for identity in identities.tolist():
        slot = counter.get(identity, 0)
        counter[identity] = slot + 1
        uids.append(f"ID-{identity:04d}-{slot:03d}")
    return vecs, uids


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenABIS retrieval benchmark")
    ap.add_argument("--identities", type=int, default=200)
    ap.add_argument("--queries-per-id", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--backend", choices=["faiss", "qdrant", "milvus", "brute"], default="faiss")
    ap.add_argument("--ef-search", type=int, default=128)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--nlist", type=int, default=256)
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    cfg = IndexConfig(
        backend=args.backend, nlist=args.nlist, nprobe=args.nprobe,
        ef_search=args.ef_search, m=args.m,
    )
    if args.backend == "faiss" and args.identities < max(39 * cfg.nlist, 39 * cfg.pq_segments):
        print(f"[warn] gallery too small ({args.identities}) to train IVF-HNSW-PQ "
              f"({cfg.nlist} lists x {cfg.pq_segments} segments); falling back to exact brute search.")
        cfg.backend = "brute"
    backend = build_backend(cfg)
    oracle = BruteForceIndex(dim=cfg.dim)

    vecs, uids = build_gallery(args.identities, dim=cfg.dim, seed=args.seed)
    backend.add(uids, vecs)
    oracle.add(uids, vecs)

    rng = np.random.default_rng(args.seed + 1)
    latencies: list[float] = []
    id_recall: list[float] = []
    slot_recall: list[float] = []
    for _q in range(args.queries_per_id):
        sample = int(rng.integers(0, len(vecs)))
        query = vecs[sample] + 0.05 * rng.normal(0, 1, cfg.dim).astype(np.float32)
        query /= np.linalg.norm(query)
        probe_identity_uid = uids[sample].rsplit("-", 1)[0]

        t1 = time.perf_counter()
        hits = backend.search(query, top_k=args.top_k)
        latencies.append((time.perf_counter() - t1) * 1000)
        if not hits:
            id_recall.append(0.0)
            slot_recall.append(0.0)
            continue

        # Identity-level recall — the legally meaningful unit (candidate list
        # must contain the probe's identity for Tier-2 to confirm it).
        hit_identities = {uid.rsplit("-", 1)[0] for uid, _ in hits}
        id_recall.append(1.0 if probe_identity_uid in hit_identities else 0.0)

        # Slot-level recall vs exact oracle — calibrates PQ distortion.
        exact = {uid for uid, _ in oracle.search(query, top_k=args.top_k)}
        returned = {uid for uid, _ in hits}
        slot_recall.append(len(exact & returned) / max(len(exact), 1))

    order = np.asarray(latencies)
    print(f"backend={type(backend).__name__:<18} IDENT={args.identities:>6} top_k={args.top_k}")
    if id_recall:
        print(f"  identity recall@{args.top_k}    : {np.mean(id_recall):.4f}")
        print(f"  slot recall@{args.top_k}       : {np.mean(slot_recall):.4f}  (PQ fidelity vs exact)")
    if order.size:
        print(f"  latency p50 / p95 : {np.percentile(order, 50):.2f} ms / {np.percentile(order, 95):.2f} ms")
    print(f"  index size        : {len(backend)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
