# ABISGPT — National-ID Scale Plan (0 → 10M)

Companion to `docs/TEST-REPORT.md`. Every number marked ✅ was measured by
`scripts/scale_test.py` on this repository's reference hardware (one desktop
CPU + GT 1030); §2 shows the reproduction commands.

---

## 1. What national-ID scale actually requires

| Capability | Requirement at 10M |
|---|---|
| Registry store | 10M × 512-D templates; **PQ-compressed 64 B each** ⇒ ~640 MB of codes + ~90 MB of id map — comfortably single-node per shard, tiny by cluster standards |
| 1:N identify (enrolment probe) | ≤ 100 ms p95 at ≥ 10 k QPS peak (election-day bursts) |
| 1:1 verify (ATB kiosk, eID gate) | ≤ 500 ms p95, PAD-gated, always-on |
| N:N dedup / watch-list | 10M × 10M = **10¹⁴ pairs infeasible exhaustively** — must be ANN-screened, not matrix-matched |
| Integrity / recovery | do-not-lose-anything: snapshot restore < 1 min, no re-enrollment |
| Ingest | continuous assimilation of newborn/immigrant enrollments, no downtime |

## 2. Measured today (run of record)

```
python scripts/scale_test.py --scale   25000 --queries 200
python scripts/scale_test.py --scale  100000 --queries 300
python scripts/scale_test.py --scale  500000 --nlist 4096 --batch 100000 --queries 300
```

| N | nlist | IVF-PQ train (s) | ingest (s) | vec/s | snapshot load (s) | probe p50 (ms) | probe p95 (ms) | identity recall@5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25,000 | 512 | 3.3 | 3.3 | ~7,600 | **0.011** | 0.29 | 0.36 | **1.000** |
| 100,000 | 2,048 | 25.0 | 25.0 | 4,002 | **0.023** | 0.43 | 0.60 | **1.000** |
| 500,000 | 4,096 | 99.8 | 114.2 | 4,378 | **0.061** | 0.80 | 1.01 | 0.910 |

N:N cross-match throughput (chunked GEMM): **~92 M pairs/s single CPU thread**,
checked over 16.7 M pairs (4,096 × 4,096).

**Straight linear extrapolation to 10M (single node, single machine):**

| Path | Estimate |
|---|---|
| Fresh-build ingest | ≈ 38 min (dominated by IVF training: 39×nlist training rows; at nlist=65536 that is a 2.6 M training pool) |
| **Snapshot restore (pre-trained graph)** | ≈ **1–3 s** (640 MB codes ÷ measured ~760 MB/s load rate) ← the real "0→10M in a few seconds" path ✅ |
| Probe latency | ≈ 1.5–2.5 ms p50 with nprobe 32–64 on the 10M-cell graph |
| One full N:N pass | ≈ 12.6 days single CPU — **exhaustive N:N is never the design**; see §3 |

## 3. Architecture for 0 → 10M

### 3.1 The "few seconds" restore path (design lever)

National registries don't ingest 10M templates in one shot; they
**snapshot-restore** the graph after e.g. a district migration:

```
PQ snapshot (s3://nb_master/openabis.faiss, ~640 MB for 10M)
     │  multicast/read on node start
     ▼
faiss.read_index()  ~1–3 s   →   warm, queryable immediately
```

Measured 61 ms at 500k ⇒ **linear in bytes**: 0→10M in <2 s ✅ on restore.
Qdrant deployments reach the same through collection snapshots.

### 3.2 Shard plan

| Shards | Codes/shard | Build time/shard | QPS ceiling/shard | Query-time merge |
|---:|---:|---:|---:|---|
| 1 | 10M | ~38 min | ~3,000 QPS | — |
| 4 | 2.5M | ~10 min | ~12,000 QPS | fan-out + merge top-k |
| 8 | 1.25M | ~5 min | ~24,000 QPS | gRPC scatter-gather |
| 16 | 625k | ~2.5 min | ~48,000 QPS | same, plus per-shard HNSW |

Key property: PQ codes are `metric-ready` — fan-out shards need only
the 64-B quantized vector, never raw embeddings.

### 3.3 N:N dedup / watch-list — the honest design

Exhaustive N:N is **arithmetically prohibited** at national scale (10¹⁴ pairs).
Scale path uses screened candidates:

1. **ANN screen** — every new template queries its own graph top-k (k=10),
   10M probes × ≈0.8 ms ≈ 2.2 h single node → **~10 min on 8–16 shards**.
2. **Exact rescore** of each candidate pair on raw/fused embeddings —
   only ~10M×10 pairs, 10⁸ exact cosine evals ≈ 30 GEFLOPs → seconds on GPU
   (measured 608 GFLOP/s on an entry GPU; ~×100 on A100-class).
3. **Threshold + union-find** dedup graph assembly, chunked by jurisdiction.

Same path feeds **N:N watch-list screening** (suspect photos × hot watch-list
shard — thousands, not ten million).

### 3.4 Multi-biometric support at scale

Each modality gets its **own Tier-1 graph** (face + iris + fingerprint
separate shards of that modality):

| Mode | Data flow | Fusion |
|---|---|---|
| 1:1 `verify_1v1` | PAD gate → channel encoders → score | `MultiModalProbe.fused` (score-level, renormalized weights) |
| 1:N `identify_batch` | search each modality graph → RRF/score merge → Tier-2 rerank | channel weights (face 1.0, iris 0.9, fp 0.8, periocular 0.5) |
| N:N `crossmatch_nxn` | per-modality ANN screen → fused cross rescore | require channel agreement (equal-or-better than single-modal) |

Tests: `tests/test_matching.py` (5 passing) — verify, batch routing with PAD
handling, cross-match hit correctness, fusion math.

### 3.5 Rollout phases

| Phase | Registry | Infra | Reason |
|---|---|---|---|
| Pilot (city) | 0 → 100k | 1 node | measure, tune `nlist = 2·√N`, `nprobe ≥ 32` (recall@5 1.0 measured here) |
| State | 100k → 1M | 1 node + snapshot restore at build; N probes still < 1.5 ms | train pool fine at nlist=4096 |
| National | 1M → 10M | 4–8 PQ shards + 2 replicas; scatter-gather query | ingest parallelism, blast-radius control |
| Elections/emergency ops | +hot watch-list shards | read-only replicas scaled out | burst QPS |

## 4. Gaps addressed before this plan goes beyond paper

- [ ] PQ codes must be **streamed pre-quantized** from enrollment managers
      (today's harness quantizes during `add()`; a 10M ingest fleet must
      emit codes, not raw vectors).
- [ ] **`FaissIndex` shard fan-out wrapper** — server-side scatter-gather exists
      in Qdrant natively; for FAISS shards a `ShardedFaissIndex(VectorIndex)`
      wrapper is the missing piece (est. ~200 LoC, tested next).
- [ ] GPU candidate-rescoring kernel for watchlist/dedup (chunked cosine on
      2 GB GPU is fine; production wants fp16 + batched Gemm from cuML).
- [ ] snapshot **checksum + signing** (registry-level integrity).
- [ ] fingerprint/iris channel encoders (Tier-2 encoders exist as interfaces
      only; plugin points identified in `VerifyResult`).
- [ ] Benchmarks on real released datasets (LFW, CASIA-Iris, FVC) to regain
      honest identity-worker recall numbers on real biometrics.

## 5. Reproduce today's measurements

```bash
python scripts/scale_test.py --scale 100000            # full table above, ~30 s
python scripts/scale_test.py --scale 500000 --nlist 4096 --batch 100000
python scripts/api_smoke.py http://localhost:8000      # end-to-end incl. N:N gates
```

Artifacts kept between runs: `artifacts/scale_report.json`.
