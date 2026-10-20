# OpenABIS — Multi-Modal ABIS & Passive PAD

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![Code style: black](https://img.shields.io/badge/Code%20Style-black-000000.svg)]()
[![DOI](https://img.shields.io/badge/DOI-pending-3F51B5.svg)](CITATION.cff)

**Next-generation, reference-architecture implementation for Automated Biometric
Identification Systems (ABIS) with fully passive Presentation Attack Detection
(PAD)** — built on Vision Transformers, vision-language models, temporal rPPG,
and two-tier ANN vector retrieval (IVF-HNSW + Product Quantization).

> Research/reference code intended for benchmarking, education, and prototyping.
> **Not certified for production deployment in regulated identity workflows.**
> See [SECURITY.md](SECURITY.md) and [docs/RESPONSIBLE-USE.md](docs/RESPONSIBLE-USE.md).

---

## Why modern ABIS

Legacy ABIS/PAD stacks suffer from:

- **Fragmented CNN pipelines** — one model per modality/condition; accuracy
  collapses under masks, glasses, or aged captures.
- **Hardware-locked linear matching** — `O(N)` brute-force search at tens of
  millions of identities.
- **Active-liveness friction** — blink/nod prompts are gameable by video
  injection attacks.
- **Store-everything privacy debt** — raw face/iris/fingerprint images kept for
  years.

OpenABIS replaces those with a three-tier, decoupled design:

```
[ EDGE CAPTURE TIER ]
   |-- RGB camera (1080p @ 60 FPS)
   |-- NIR sensor
   +-- ToF / structured light mesh
              |
              | gRPC (protocol buffers)
              v
[ INFERENCE ACCELERATION TIER ]
   |-- Triton Inference Server (GPU cluster)
   |     |-- Passive PAD Engine (VLM semantic guardrails + temporal rPPG)
   |     +-- Biometric Encoder (unified multi-modal ViT)
              |
              | 512-D normalized embeddings
              v
[ STORAGE & RETRIEVAL TIER ]
   Tier 1: IVF-HNSW + PQ ANN triage  --> top-10 candidates
   Tier 2: classical geometric matcher --> legal identity verdict
```

Full rationale in
[`docs/white_paper.md`](docs/white_paper.md) and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Feature Matrix

| Capability | Component | Status |
|---|---|---|
| ViT biometric encoder (face/fingerprint templates) | `openabis.encoder` | experimental |
| Template-aware masked-face attention reweighting | `openabis.encoder` | experimental |
| VLM zero-shot spoof-attribute scoring | `openabis.pad` | experimental |
| Temporal rPPG liveness (60 FPS BVP + FFT) | `openabis.pad.rppg` | experimental |
| Flash-induced surface diffusion check | `openabis.pad.flashdiff` | experimental |
| IVF-HNSW + PQ vector index (FAISS/Qdrant/Milvus) | `openabis.index` | experimental |
| Two-tier triage + geometric verifier | `openabis.pipeline` | experimental |
| Vector-vault privacy protocol (no PII, UUID decoupling) | `openabis.vault` | experimental |
| gRPC-style microservice contract | `openabis.proto/` | roadmap |
| Triton/TensorRT export | `scripts/export_triton.py` | roadmap |

## Quickstart

```bash
git clone https://github.com/YOUR_ORG/OpenABIS.git
cd OpenABIS
pip install -e ".[dev]"

# Smoke-test the full pipeline end-to-end (CPU, synthetic data)
python scripts/demo_synthetic.py --identities 20 --queries-per-id 2 --top-k 5
```

You should see a retrieval report with recall@k, spoof rejection, and
latency percentiles. No GPU, no camera, no real biometric data required — the
demo fabricates deterministic synthetic embeddings and frames.

### Docker: Qdrant + HTTP API

```bash
docker compose up -d --build      # qdrant + torch-free FastAPI image
python scripts/api_smoke.py http://localhost:8000
```

`scripts/api_smoke.py` enrolls pulse-carrying synthetic captures via
`POST /enroll`, verifies `POST /identify` returns `match` behind the
fail-closed rPPG/flash/VLM PAD gate, exercises `POST /pad` on a flat replay
attempt, and checks the GDPR erasure hook (`DELETE /identities/{id}` → 404
once removed). Run the live Qdrant integration tests against the container
with:

```bash
docker compose up -d qdrant
OPENABIS_RUN_INTEGRATION=1 OPENABIS_QDRANT_URI=http://localhost:6333 pytest tests/test_qdrant_integration.py
```

### Train the reference encoder

```bash
python scripts/train_encoder.py --identities 64 --epochs 4
```

Trains a compact ArcFace-head embedding network on procedural synthetic
identities (CPU, ~25 s), reports **unseen-identity recall@1** on held-out
identities, and writes `artifacts/toy_encoder.pt`. The synthetic-task metrics
validate training mechanics; they are not a biometric-quality benchmark.

### Optional backends

```bash
pip install -e ".[qdrant]"    # qdrant-client backend
pip install -e ".[milvus]"    # pymilvus backend
pip install -e ".[gpu]"       # faiss-gpu, torch CUDA build
```

### Torch / GPU

The pipeline runs on CPU by default and auto-detects CUDA with a modern torch
build.

## Repository Structure

```
OpenABIS/
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- pyproject.toml
|-- requirements.txt
|-- .github/workflows/ci.yml
|-- docs/
|   |-- white_paper.md          # full white paper (canonical)
|   |-- ARCHITECTURE.md         # system design deep-dive
|   |-- RESPONSIBLE-USE.md      # ethics, law, misuse policy
|   |-- BENCHMARKS.md           # evaluation protocol
|   +-- api.md                  # library API reference
|-- openabis/
|   |-- __init__.py
|   |-- config.py               # dataclass config + YAML loader
|   |-- encoder/                # ViT biometric encoders
|   |   |-- __init__.py
|   |   +-- vit_encoder.py
|   |-- pad/                    # passive presentation attack detection
|   |   |-- __init__.py
|   |   |-- rppg.py             # temporal rPPG (FFT/SNR classical tier)
|   |   |-- flashdiff.py        # flash surface-diffusion tier
|   |   +-- vlm.py              # CLIP zero-shot + heuristic guardrails
|   |-- index/                  # vector search backends
|   |   |-- __init__.py
|   |   |-- faiss_backend.py    # IVF-HNSW-PQ
|   |   |-- qdrant_backend.py   # remote HNSW + server-side PQ
|   |   +-- milvus_backend.py
|   |-- api/                    # FastAPI service (Docker reference)
|   |   +-- service.py
|   |-- training/               # toy biometric encoder (ArcFace, CPU)
|   |-- pipeline/               # two-tier triage orchestration
|   |   |-- __init__.py
|   |   +-- triage.py
|   +-- vault/                  # privacy preserving vector storage
|       |-- __init__.py
|       +-- vault.py
|-- scripts/
|   |-- demo_synthetic.py       # E2E demo on synthetic data
|   +-- evaluate.py             # benchmark harness CLI
|-- tests/
    |-- test_encoder.py
    |-- test_pad.py
    +-- test_index.py
```

## Benchmarks Summary

Latest synthetic-pipeline benchmark (see
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for protocol; **not** comparative to
commercial ABIS):

| Metric | Value | Hardware |
|---|---|---|
| recall@1 | 0.985 | faiss-cpu, 20 identities synthetic |
| recall@10 | 1.000 | — |
| top-1 report latency (p50) | 2.1 ms | faiss-cpu IVF-HNSW-PQ |
| VLM spoof scoring latency (p50) | 18.4 ms | CPU VLM head (tiny) |
| rPPG decision latency (p50) | 46 ms (2 s @ 60 fps) | CPU |

Reproduce with:

```bash
python scripts/evaluate.py --identities 200 --queries-per-id 5 --top-k 10
```

## Citation

```bibtex
@software{OpenABIS_2026,
  title  = {OpenABIS: Multi-Modal ABIS \& Passive PAD with
            Transformers, VLMs and Vector Search},
  author = {OpenABIS Contributors},
  year   = {2026},
  url    = {https://github.com/YOUR_ORG/OpenABIS},
  note   = {Reference implementation of the open white paper}
}
```

## Responsible Use

Biometrics are dual-use. Before deploying anything derived from this repo,
read [`docs/RESPONSIBLE-USE.md`](docs/RESPONSIBLE-USE.md). Do **not** use
OpenABIS for covert surveillance, for matching people without informed consent
and a lawful basis, or where prohibited by local law (GDPR Art. 9, CCPA, BIPA,
India DPDP, etc.).

## License

MIT — see [`LICENSE`](LICENSE).
