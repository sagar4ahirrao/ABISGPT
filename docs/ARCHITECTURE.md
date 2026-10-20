# ABISGPT Architecture

## Design Goals

1. **Decoupled scaling** — edge capture, inference acceleration, and vector
   storage scale independently.
2. **Constant-time retrieval** — approximate nearest neighbor search over
   trajectories of `O(log N)` instead of `O(N)` linear scans.
3. **Passive liveness** — zero user friction; fraud rejection happens in
   parallel with identification.
4. **Privacy by architecture** — vectors without PII; PII never leaves an
   isolated relational vault.

## Data Flow

```
Camera (Kiosk / Mobile)
  |
  | gRPC (protobuf) -- N frames @ 60fps, NIR + ToF mesh
  v
Triton Inference Server (L4 GPU cluster)
  |
  |--- Passive PAD Engine
  |      |--- VLM semantic guardrails (spoof attributes)
  |      |--- Temporal rPPG (blood volume pulse @ ROI skin)
  |      +--- Flash diffusion analyzer (kiosk ring light)
  |
  |    "pass" --> continue; "fail" --> reject (never proceed to search)
  |
  |--- Unified Multi-Modal ViT Encoder
         |
         v  512-D L2-normalized embedding
Vector DB (Milvus / Qdrant / FAISS)
  |
  | Tier 1 : IVF-HNSW + PQ approximate search --> top-10 candidates  (ms)
  | Tier 2 : classical geometric / exact rerank on the 10 candidates
              --> match / no_match verdict
  |
  v
Decision Layer (audit log, SLA timer, threshold policy)
  |
  v
Identity resolution through the isolated Relational Vault (PII)
```

## Module Map (this repo)

| Layer | Module | Notes |
|---|---|---|
| Encoder | `abisgpt/encoder` | ViT wrapper; masks-aware attention weighting |
| PAD | `abisgpt/pad` | rPPG + VLM guardrails + flash check fused |
| Index | `abisgpt/index` | FAISS / Qdrant / Milvus backends behind one API |
| Pipeline | `abisgpt/pipeline` | two-tier triage + threshold policy |
| Vault | `abisgpt/vault` | UUID↔PII relational decoupling facade |

## Backend Abstraction

`abisgpt.index` exposes a single interface:

```python
backend = build_backend(config)      # faiss | qdrant | milvus
backend.add(ids, embeddings)         # bulk ingest
hits = backend.search(query, top_k)  # tier-1 ANN triage
```

`config.search.ef_search` and `config.index.m` are runtime-tunable
(see `docs/BENCHMARKS.md`), matching the "soft search knobs" described in the
white paper.

## Concurrency Model

- Encoder batched (dynamic batching controlled by `config.encode.batch`).
- rPPG runs on a sliding window (2–4 s, 60 fps) in a sidecar process — it can
  overlap with Tier-1 search rather than serialize before it.
- Returns are sent through a gRPC `BiometricVerdict` message: one
  decision + N ranked `IdentityMatch` scores.

## Failure Modes

| Failure | Behavior |
|---|---|
| PAD gate fails (VLM/rPPG/flash) | search aborted, fail-closed |
| Vector DB timeout | fail-closed + degrade to smaller `ef_search` retry |
| Encoder OOM | drop to `batch/2` and retry once |
| Vault unreachable | verdict computed, PII resolution withheld |

