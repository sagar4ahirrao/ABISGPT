#!/usr/bin/env python
"""National-ID scale harness: ingest → snapshot → Tier-1 search → N:N crossmatch.

Measures, on synthetic 512-D biometric embeddings (one vector per identity,
exact ground truth), everything that matters for a registry of 0..10M:

  1. IVF-HNSW-PQ training cost (first batch trains the graph)
  2. bulk ingest rate (vectors/s)
  3. snapshot save/load (the "registry of N in seconds" restore path)
  4. Tier-1 probe latency p50/p95 + identity recall@5
  5. N:N cross-match throughput (chunked GEMM, pairs/s) with 10M extrapolation

Usage:
    python scripts/scale_test.py --scale 100000
    python scripts/scale_test.py --scale 500000 --nlist 4096 --batch 100000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from abisgpt.config import IndexConfig
from abisgpt.index.faiss_backend import FaissIndex
from abisgpt.pipeline.matching import MatchingEngine
from abisgpt.pipeline.triage import TwoTierPipeline

DIM = 512


def ingest(index: FaissIndex, centers: np.ndarray, n: int, batch: int,
           train_pool: int) -> tuple[float, float]:
    """Stream `n` one-per-identity vectors into the index; returns (ingest_s, train_s)."""
    rng = np.random.default_rng(7)
    buffer: list = []
    buffered_ids = []
    buffered = 0
    t_train = 0.0
    t_add = 0.0
    done = 0
    while done < n:
        m = min(batch, n - done)
        idx = np.arange(done, done + m)
        block = centers[idx] + 0.02 * rng.standard_normal((m, DIM)).astype(np.float32)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        ids = [f"id_{i}" for i in idx]
        if not index._index.is_trained and buffered + m < train_pool:
            buffer.append(block)
            buffered_ids += ids
            buffered += m
            done += m
            continue
        if not index._index.is_trained:
            first = np.vstack([*buffer, block]) if buffer else block
            t0 = time.perf_counter()
            index.add([*buffered_ids, *ids], first)
            t_train += time.perf_counter() - t0
            buffer, buffered_ids, buffered = [], [], 0
        else:
            t0 = time.perf_counter()
            index.add(ids, block)
            t_add += time.perf_counter() - t0
        done += m
    if buffer:                       # n < training pool: single-batch add w/ small nlist
        t0 = time.perf_counter()
        index.add(buffered_ids, np.vstack(buffer))
        t_add += time.perf_counter() - t0
    return t_add, t_train


def probe_latency_recall(index: FaissIndex, centers: np.ndarray, queries: int) -> tuple[list, list]:
    rng = np.random.default_rng(11)
    lats, recall = [], []
    for _ in range(queries):
        target = int(rng.integers(0, len(centers)))
        qv = centers[target] + 0.02 * rng.standard_normal(DIM).astype(np.float32)
        t0 = time.perf_counter()
        top = index.search(qv, top_k=5)
        lats.append((time.perf_counter() - t0) * 1000)
        recall.append(1.0 if any(h[0] == f"id_{target}" for h in top) else 0.0)
    return lats, recall


def run(n: int, nlist: int, queries: int, batch: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n, DIM)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    cfg = IndexConfig(backend="faiss", nlist=nlist, nprobe=min(32, nlist),
                      pq_segments=64, pq_bits=8, dim=DIM)
    index = FaissIndex(cfg)
    train_pool = 39 * max(nlist, 2 ** cfg.pq_bits)

    ingest_s, train_s = ingest(index, centers, n, batch, train_pool)

    import faiss

    snapshot = Path("artifacts")
    snapshot.mkdir(exist_ok=True)
    snap = snapshot / f"faiss_snap_{n}.idx"
    t0 = time.perf_counter()
    faiss.write_index(index._index, str(snap))
    t_save = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = faiss.read_index(str(snap))
    t_load = time.perf_counter() - t0
    mb = snap.stat().st_size / 1e6
    snap.unlink()                     # keep artifacts lean

    lats, recall = probe_latency_recall(index, centers, queries)
    lats = np.sort(lats)
    total_ingest_s = ingest_s + train_s

    # N:N cross-match, chunked GEMM, disjoint halves -> threshold filters all
    m = min(n // 2, 8192)
    g = {f"g_{i}": centers[2 * i] for i in range(m // 2)}
    p = {f"p_{i}": centers[2 * i + 1] for i in range(m // 2)}
    engine = MatchingEngine(TwoTierPipeline(index_backend=index))
    cm = engine.crossmatch_nxn(p, g, threshold=0.9)

    return {
        "scale_n": n,
        "nlist": nlist,
        "ivf_pq_train_s": round(train_s, 2),
        "ingest_s": round(total_ingest_s, 2),
        "ingest_vec_per_s": round(n / max(total_ingest_s, 1e-6)),
        "snapshot_size_mb": round(mb, 1),
        "snapshot_save_s": round(t_save, 3),
        "snapshot_load_s": round(t_load, 3),
        "queries": queries,
        "probe_p50_ms": round(float(lats[len(lats) // 2]), 3),
        "probe_p95_ms": round(float(lats[int(len(lats) * 0.95)]), 3),
        "probe_recall@5": round(float(np.mean(recall)), 4),
        "nn_probes": m // 2,
        "nn_pairs_scored": (m // 2) * (m // 2),
        "nn_pairs_per_sec": round(cm.pairs_per_sec),
        "nn_1M_x_1M_extrap_s": round((1_000_000 ** 2) / cm.pairs_per_sec, 1),
        "nn_10M_x_10M_extrap_s": round((10_000_000 ** 2) / cm.pairs_per_sec, 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="ABISGPT national-ID scale harness")
    ap.add_argument("--scale", type=int, default=100_000)
    ap.add_argument("--nlist", type=int, default=2048)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--batch", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    # nlist must satisfy 39·nlist ≤ n (IVF-PQ training pool). Auto-shrink if not.
    nlist = args.nlist
    while 39 * nlist > args.scale and nlist > 256:
        nlist //= 2
    if 39 * nlist > args.scale:
        raise SystemExit(f"scale {args.scale:,} too small even for nlist=256 "
                         f"(needs ≥ {39 * 256:,} vectors); use --scale ≥ 10,000")
    print(f"[scale] n={args.scale:,} nlist={nlist} batch={args.batch}", flush=True)
    report = run(args.scale, nlist, args.queries, args.batch, args.seed)
    line = json.dumps(report, indent=2)
    print(line)
    out = Path("artifacts") / "scale_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(line, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
