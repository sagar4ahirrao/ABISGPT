# ABISGPT — Test Report (run of record)

Date: 2026-10-20 · Host: Windows 10, Python 3.10 · Repo: commit `2fec1e3` + rename commit

Everything below was executed end-to-end on this machine against v0.1.0.
All data is synthetic and deterministic (seeded); no real biometrics were used.

## 1. Environment

| Item | Value |
|---|---|
| CPU/GPU | Server-class CPU · NVIDIA GT 1030 (2 GB, Pascal, sm_61) |
| torch | 2.6.0 + cu118 build (CUDA available) |
| Index backends | FAISS 1.9.0 (IVF-HNSW-PQ) · Qdrant v1.9.7 container · brute |
| API | FastAPI 0.116 + uvicorn; Docker image = torch-free |
| CI | `.github/workflows/ci.yml`: ruff + pytest, py3.10/3.11 × ubuntu/windows |

## 2. Data preparation (3 synthetic datasets)

| Dataset | Generator | Composition |
|---|---|---|
| **Training** | `abisgpt/training/synthdata.py` | per-identity low-frequency sinusoidal bases on a face-like radial gradient + Gaussian noise; train identities seeded separately from eval identities |
| **PAD probes** | `scripts/demo_synthetic.py`, `scripts/api_smoke.py` | live = green-channel 1.0 Hz (60 bpm) BVP across ≥3 s @ 60 fps; spoof = flat static screen replay / high-freq moiré |
| **Retrieval galleries** | `scripts/evaluate.py`, `scripts/evaluate.py` registry bulk | per-identity 512-D clusters (center + 0.05 noise), a known oracle for recall |

Design property: identity weights for the eval split come from a seed never
used in training, so recall measures *generalization to unseen identities*
rather than memorization.

## 3. Results by test layer

### 3.1 Model training (ArcFace, procedural identities)

| Run | Device | Epochs | Final loss | Wall | recall@1 | recall@5 |
|---|---|---|---|---|---|---|
| 64 ids × 40 frames | CPU | 4 | 0.0097 | 23.6 s | 1.000 | 1.000 |
| 64 ids × 40 frames | CUDA | 4 | 0.0097 | 33.9 s | 1.000 | 1.000 |
| 256 ids × 96 frames, batch 256 | CUDA | 6 | 0.0128 | 439.5 s | 1.000 | 1.000 |
| 256 ids × 96 frames, batch 256 | CPU | 6 | 0.0128 | **400.8 s** | 1.000 | 1.000 |

Key observations
- Identical recall on both devices → deterministic math, GPU correctness verified.
- GT 1030 loses to the CPU even at batch 256: inputs are 48×48, numpy data
  generation is host-bound single Worker; GPU only pays off for larger models
  (timm/ViT path is opt-in, not shipped).

### 3.2 Retrieval (Tier-1 ANN)

| Test | Scale | Metric | Result |
|---|---|---|---|
| `scripts/evaluate.py` | 12,000 ids, FAISS IVF-HNSW-PQ | identity recall@10 | **1.000** |
| | | slot (exact-vector) recall@10 | 0.38 (expected under PQ distortion; 512-D→64-B) |
| | | Tier-1 latency p50 | 0.32 ms |
| `scripts/demo_synthetic.py` | 12 ids | probe recall@1 | 1.000 |
| | | static-replay spoofs rejected | **12/12** |
| | | latency p50/p95 | 50.4 / 62.5 ms |
| In-repo unit oracle | 200-id gallery | FAISS top-1 == exhaustive top-1 | pass (`tests/test_encoder_index.py`) |

### 3.3 PAD / liveness gate (fail-closed)

| Probe | rPPG verdict | Fused | Decision |
|---|---|---|---|
| Pulse-carrying live capture (180 f) | SNR 19.6 dB, live | ≥ 0.6 | PASS (proceed to search) |
| Flat static capture (replay) | 0 dB noise floor | 0.55 < 0.6 | REJECT ✓ |
| Bench scoreline (demo) | — | — | 12/12 spoofs rejected |

### 3.4 Unit + API test suite

```
pytest tests -q        → 28 passed, 1 skipped   (Windows + CI-matrix config)
ruff check abisgpt scripts tests   → clean
```

The suite covers: config validation, encoder contract, FAISS-vs-oracle ranking,
PQ training remediation, PAD fusion weights and fail-closed gate, VLM
determinism, ArcFace forward/backward + eval-mode behavior, training-dataset
shape/determinism, FastAPI endpoints (TestClient), dashboard HTML served,
`/api/status` (torch build info), `/api/bench/*` acceptance bounds.

### 3.5 Live Qdrant integration (container)

`tests/test_qdrant_integration.py` with `ABISGPT_RUN_INTEGRATION=1`:
- 128 clustered templates → HNSW self-query top-1 correct, cosine > 0.95
- re-enroll overwrite semantics (uuid5 identity mapping)
- GDPR delete: point count drops exactly; unknown identity deletes 0 → HTTP 404

### 3.6 End-to-end HTTP smoke (Dockerized stack: Qdrant + API)

`scripts/api_smoke.py` → **PASS**

```
GET  /health               → 200 ok, backend QdrantIndex
POST /enroll  tom/jane     → 200, 512-D, norm 1.0
POST /identify (tom, jane) → match, pad_live=True, tier2 cosine ≈ 1.0
POST /pad (flat replay)    → is_live=False, fused 0.55 < gate 0.60
DELETE /identities/{tom}   → 200 removed:1    (ghost → 404)
GET  /health               → enrolled 0 (fully erased)
```

### 3.7 GPU sanity

| Test | Result |
|---|---|
| `torch.cuda.is_available()` | True (cu118 build, sm_61 supported) |
| 2048³ fp32 matmul throughput | 608 GFLOP/s GPU vs 210 GFLOP/s CPU (2.9×) |
| End-to-end ArcFace training | correct + deterministic (identical recall) |
| CLIP VLM pad scorer | runs on CPU; CUDA path available via `ClipVlmPad(device='cuda')` |

## 4. Defects found & fixed during hardening

| Defect | Where | Fix |
|---|---|---|
| Off-bin BVP leaked power → rPPG SNR floored at ~6 dB | API smoke data gen | use on-bin 1.0 Hz BVP + ≥3 s window (detrend no longer absorbs the sine) |
| `ProductQuantization` schema missing `product` field (qdrant-client 1.9+) | `qdrant_backend._ensure_collection` | nested `ProductQuantizationConfig` |
| `client.search` removed in qdrant-client ≥1.10 | `QdrantBackend.search` | `query_points` compat path |
| String point IDs rejected by Qdrant | `add/remove/search` | deterministic uuid5 identity mapping + payload recovery |
| `operation_count` absent in newer delete responses | `remove` | before/after `count()` filter diff (never-truthy fallback removed) |
| linear detrend absorbed ~1-cycle signals | discovered while tuning smoke data (documented behavior of `rppg.bandpass`) | windowing guidance in report/docs |

## 5. Known limits / not validated here

- End-to-end accuracy on real face imagery (requires data release ⚠ out of scope).
- torch/timm encoder weights not shipped; `encoder.strategy="torch"` path is
  opt-in and untested against real weights in CI.
- Milvus backend implemented but not exercised live (lazy import, no container).
- GT 1030 VRAM (2 GB) bounds ViT batch sizes; nothing in this repo is
  validated beyond toy-scale training on it.
