# ABISGPT — Open Multi-Modal ABIS & Passive PAD Research Stack

[![CI](https://img.shields.io/badge/CI-github_actions-blue)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%7C3.11-blue)
![GPU](https://img.shields.io/badge/GPU-optional_(CUDA)_-9cf)

**ABISGPT** is an open, reproducible reference implementation for a
*next-generation Automated Biometric Identification System* (ABIS), built
around three connected ideas from its companion white paper
([docs/white_paper.md](docs/white_paper.md)):

1. **Multi-modal biometric embedding** — a ViT-class encoder producing
   L2-normalized 512-D templates from face/periocular video frames.
2. **Constant-time Tier-1 triage** — an *encrypted vector graph*
   (IVF-HNSW-PQ) that searches registries of millions in `O(log N)` per probe,
   with every template compressed to **64 bytes** via Product Quantization.
3. **Fail-closed passive PAD** — weighted fusion of rPPG (§2.2), flash
   surface-diffusion, and **VLM semantic guardrails** (§2.1, zero-shot CLIP)
   that blocks presentation attacks *before* retrieval happens.

Everything ships synthetic-data first: deterministic, seeded, no real
biometrics — run the whole stack on a laptop, then bring your own data.

> ⚠️ **Research code.** Not a certified biometric system, not a medical
> device, no production guarantees. Every numeric claim in this README is
> reproducible via the listed commands, and every limitation is stated.
> See [RESPONSIBLE-USE.md](docs/RESPONSIBLE-USE.md).

---

## What works today (all verified by tests)

| Layer | Component | Status |
|---|---|---|
| Encoder | Hash projection (CI-safe) · torch/ViT strategy (opt-in) | ✅ / ⚠️ |
| PAD | rPPG FFT/SNR tier · flash diffusion tier · fusion `PassivePadEngine` | ✅ |
| PAD | **VLM semantic tier** — CLIP zero-shot + heuristic fallback | ✅ |
| Index | FAISS IVF-HNSW-PQ (bulk-trained) · Qdrant (live-tested) · Milvus | ✅/⚠️/⚠️ |
| Pipeline | Two-tier triage `Tier1 ANN → Tier2 exact rerank`, fail-closed gate | ✅ |
| Privacy | GDPR Art. 17 erasure endpoint + `abisgpt.vault` SQLite facade | ✅ |
| API | FastAPI service + **interactive test dashboard** at `/` | ✅ |
| Docker | Qdrant + torch-free API image via `docker compose up` | ✅ |
| Training | Procedural biometrics + ArcFace head, CPU **and GPU** | ✅ |
| Eval | Registry-scale recall/latency harness (`scripts/evaluate.py`) | ✅ |

⚠️ = implemented but not exercised against a live service/weights in CI.
The qdrant/milvus backends are lazy-imported by design.

## Quickstart

```bash
git clone https://github.com/YOUR_ORG/ABISGPT.git
cd ABISGPT
pip install -e ".[dev]"

# Full kiosk-workflow demo on synthetic data (enroll -> PAD -> search)
python scripts/demo_synthetic.py --identities 12
```

Expected output ends with:

```
[note] spoof recall det on replay   : 12/12
[note] latency p50 / p95            : ~50 ms / ~60 ms
```

### Train the reference encoder

```bash
python scripts/train_encoder.py --identities 64 --epochs 4 --device cpu   # ~24 s
python scripts/train_encoder.py --identities 256 --frames 96 --epochs 6 --batch-size 256 --device cuda
```

Trains a compact ArcFace-style CNN (`abisgpt/training/`) on procedural
synthetic identities and evaluates **unseen-identity recall@1/5** — identities
never seen in training generalize. Metrics (latest runs, both seeds):

| Config (ids × frames × epochs) | Wall time | recall@1 | recall@5 |
|---|---|---|---|
| 64 × 40 × 4 (CPU) | 23.6 s | 1.000 | 1.000 |
| 64 × 40 × 4 (GT 1030 GPU) | 33.9 s | 1.000 | 1.000 |
| 256 × 96 × 6 (GT 1030 GPU, batch 256) | 439.5 s | 1.000 | 1.000 |
| 256 × 96 × 6 (CPU, batch 256) | **400.8 s** | 1.000 | 1.000 |

> Honest note: recall=1.0 reflects synthetic separability, not biometric
> quality. On the 2-GB Pascal-class GPU the toy workload is *not* faster
> than this CPU — 48-px inputs + single-worker numpy data pipelines are
> host-bound, so the GPU only pays off with heavier models (timm/ViT
> strategy) that this reference implementation does not ship. `artifacts/`
> stores checkpoints (`toy_encoder.pt`) + `metrics.json` per run.

### Scale test — 12,000 identities on the encrypted vector graph

```bash
python scripts/evaluate.py --identities 12000          # FAISS IVF-HNSW-PQ
```

| Metric | Result |
|---|---|
| **identity-level recall@10** | **1.000** |
| slot-level (exact vector) recall@10 | 0.38 — expected cost of 512-D → 64-byte PQ |
| Tier-1 latency p50 | **0.32 ms** |

### Docker: live Qdrant + HTTP API + test dashboard

```bash
docker compose up -d --build
browse http://localhost:8000/          # interactive test dashboard
python scripts/api_smoke.py http://localhost:8000   # SMOKE: PASS
```

The dashboard (vanilla JS, no build step) lets you fire one-off runs from the
browser: **service status** (incl. torch/cuda build), **PAD gate probes**
(live pulse-carrier vs flat screen-replay), **retrieval bench**
(recall@5 + latency at 300/3000 scale), and **quick-train** ArcFace runs
(CPU or CUDA) with live progress polling.

Equivalent scripted checks:

```bash
curl -s localhost:8000/api/status     # version, backend, cuda availability
curl -X POST localhost:8000/api/bench/pad
curl -X POST "localhost:8000/api/bench/retrieval?n_identities=300"
curl -X POST "localhost:8000/api/bench/train?identities=24&epochs=3"
```

### Live Qdrant integration tests

```bash
docker compose up -d qdrant
ABISGPT_RUN_INTEGRATION=1 ABISGPT_QDRANT_URI=http://localhost:6333 pytest tests/test_qdrant_integration.py
```

## Architecture at a glance

```
            frames (video)
                |
   ┌────────────▼────────────┐
   │  passive PAD (fail-closed)│──✗──▶  reject spoof
   │   rPPG · flash · VLM tier │
   └────────────┬────────────┘
                │ live
   ┌────────────▼────────────┐
   │  ViT encoder (512-D)    │       L2-normalized template
   └────────────┬────────────┘
                │
   ┌────────────▼────────────┐        ┌──────────────────────┐
   │ Tier-1: ANN triage      │───k───▶│ Tier-2: exact rerank │──▶ MATCH
   │ IVF-HNSW-PQ / Qdrant    │        │ on top-k only        │
   └─────────────────────────┘        └──────────────────────┘
     search O(log N), 64-B vec         cosine / minutiae-ready
```

Deep-dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
full protocol: [docs/BENCHMARKS.md](docs/BENCHMARKS.md) ·
per-module reference: [docs/api.md](docs/api.md).

## Repository Structure

```
ABISGPT/
|-- README.md · LICENSE · CITATION.cff
|-- pyproject.toml · requirements.txt
|-- docker-compose.yml          # Qdrant + torch-free API containers
|-- configs/model.yaml          # every tuning knob (m, ef_search, PQ, weights)
|-- .github/workflows/ci.yml    # ruff + pytest py3.10/3.11 × ubuntu/windows
|-- docs/
|   |-- white_paper.md          # canonical white paper
|   |-- ARCHITECTURE.md · BENCHMARKS.md · RESPONSIBLE-USE.md · api.md
|   +-- TEST-REPORT.md          # latest run-of-record, incl. GPU results
|-- abisgpt/                    # the Python package (pip install -e .)
|   |-- config.py               # dataclass config + YAML loader
|   |-- encoder/                # hash strategy (CI) + torch strategy
|   |-- pad/                    # rppg · flashdiff · vlm (CLIP/heuristic) · fusion
|   |-- index/                  # factory · brute · faiss IVF-PQ · qdrant · milvus
|   |-- pipeline/               # two-tier triage + PAD gating
|   |-- api/                    # FastAPI service + HTML dashboard (GET /)
|   |-- training/               # synthetic dataset · ArcFace · trainer/eval
|   +-- vault/                  # SQLite UUID<->PII decoupling facade
|-- scripts/                    # demo_synthetic · evaluate · train_encoder · api_smoke
+-- tests/                      # 28 tests (+1 live-container integration test)
```

## Feature Matrix

| Capability | Default | Key knobs (`configs/model.yaml`) |
|---|---|---|
| Tier-1 ANN graph | FAISS IVF-HNSW-PQ | `m`, `ef_search`, `nlist`, `nprobe`, `pq_segments`, `pq_bits` |
| Server index | Qdrant HNSW + server-side PQ | `ABISGPT_INDEX_BACKEND=qdrant`, `ABISGPT_INDEX_URI` |
| PAD fusion weights | 0.4 rPPG / 0.4 flash / 0.2 VLM | `pad.rppg_weight` etc., `gate_threshold` |
| VLM tier | none | `ABISGPT_VLM=heuristic|clip` |
| Encoder | hash (clean-room, no weights) | `encoder.strategy=torch` (opt-in) |
| Batch rerank | exact cosine | swap `PrecisionVerifier` for minutiae |
| Erasure | HTTP `DELETE /identities/{id}` | built in |

### Matching modes (white paper §1.3 — `abisgpt/pipeline/matching.py`)

| Mode | Entry point | Notes |
|---|---|---|
| **1:1 verify** | `engine.verify_1v1(id, frames)` · `POST /verify` | PAD gate → template → exact cosine vs claimed identity |
| 1:N identify | `pipeline.identify(frames)` · `POST /identify` | Tier-1 ANN → Tier-2 exact rerank on top-k |
| Q×1N batch | `engine.identify_batch([...])` | per-probe PAD gating + throughput (QPS) |
| N:N watchlist | `engine.crossmatch_nxn(probes, gallery)` · `engine.watchlist_screen([...])` | chunked GEMM (measured **~92 M pairs/s**/CPU-thread) or ANN screening for millions |
| Multi-modal | `MultiModalProbe({...}).fused(other)` | score-level fusion, channel weights (face/iris/fingerprint/periocular), renormalizes when a channel is missing |

`tests/test_matching.py` covers all five, including PAD-blocked replays and
channel-agreement fusion semantics (an unrelated iris legitimately drags a
face match down — multi-modal identity requires every channel to agree).

### National-ID scale (0 → 10M)

Measured + extrapolated in [docs/SCALE-PLAN.md](docs/SCALE-PLAN.md):

| N (identities) | Ingest | Snapshot restore | probe p50 | recall@5 |
|---:|---:|---:|---:|---:|
| 25k | 3.3 s | **11 ms** | 0.29 ms | 1.000 |
| 100k | 25 s | **23 ms** | 0.43 ms | 1.000 |
| 500k | 114 s | **61 ms** | 0.80 ms | 0.910 (nprobe-tunable) |

→ **10M snapshot restore lands in ~1–3 s** (640 MB PQ codes at the measured
~760 MB/s load rate). The real "0→10M in seconds" path is snapshot-first
restore, not fresh ingest. Exhaustive N:N at 10M is 10¹⁴ pairs (~12 days on
one CPU core): the plan uses ANN screening + exact rescoring, which reduces
it to minutes on 8–16 shards. Full reasoning and reproduction commands in
the doc.

## Case study nuances (genuine limits, stated)

- **Slot vs identity recall under PQ**: dense same-identity clusters distort
  hash-level slot ranking; identity-level recall is the contract metric
  (1.0 @ 12k identities). Dial down `pq_segments` for higher slot fidelity.
- **rPPG spectral guards**: probes need ≥2–3 heart cycles; short flat probes
  are correctly fail-closed (score 0.55 < gate 0.6).
- **CLIP VLM tier on non-face data**: works mechanically; margins are only
  meaningful with real face imagery.
- **Windows + torch + faiss in-process**: OpenMP DLL conflict solved by lazy
  torch imports and `KMP_DUPLICATE_LIB_OK=TRUE` in the test harness.
- **GT 1030-class GPUs**: fine for CUDA inference demos but *not* faster
  than a modern CPU for this toy training workload (see table above).

## Citation

```bibtex
@misc{abisgpt2026,
  title  = {ABISGPT: Open Multi-Modal ABIS \& Passive PAD Research Stack},
  author = {ABISGPT Contributors},
  year   = {2026},
  url    = {https://github.com/YOUR_ORG/ABISGPT},
}
```

Also machine-readable: [CITATION.cff](CITATION.cff).

## License & Conduct

MIT — see [LICENSE](LICENSE). Code of conduct:
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Security disclosures:
[SECURITY.md](SECURITY.md). All biometric touch-points are synthetic; real
biometrics must follow your jurisdiction's data laws *before* this repo goes
near production data.
