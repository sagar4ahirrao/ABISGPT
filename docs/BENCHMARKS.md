# Benchmarks & Evaluation Protocol

## What is measured

`scripts/evaluate.py` runs the full synthetic pipeline:

1. Generates `--identities` × `--queries-per-id` deterministic synthetic
   512-D embeddings and liveness frames.
2. Inserts all gallery embeddings into the selected index backend with
   IVF-HNSW + Product Quantization parameters from `configs/model.yaml`.
3. Issues identification queries and records:
   - `recall@1` / `recall@k`
   - spoof classification accuracy (rPPG + VLM fusion)
   - latency percentiles (p50 / p95)

## Protocol

| Setting | Value |
|---|---|
| Gallery size | configurable (default 200 identities) |
| Queries per identity | 5 (default) |
| Embedding dim | 512 (L2-normalized) |
| Index | IVF-HNSW-PQ |
| `nlist` | 256 |
| `nprobe` | 16 |
| PQ sub-vectors (`m_pq`) | 64 (8-dim each, 1 byte each) |
| PQ bits per sub-vector | 8 |
| rPPG window | 2.0 s @ 60 fps |
| VLM backbone | any CLIP-family vision-language head |

## How to run

```bash
python scripts/evaluate.py --identities 200 \
                           --queries-per-id 5 \
                           --top-k 10 \
                           --backend faiss \
                           --seed 2026
python scripts/evaluate.py --backend qdrant --ef-search 128
python scripts/evaluate.py --backend milvus --m 32
```

## Reproducing the white-paper claims on real data

The synthetic suite only validates **pipeline mechanics** (recall, latency,
spoof-gate wiring). To substantiate the >=99.9% recall figure and the PAD
AUC claims, use public biometric corpora on your own lawful, consented
instances:

| Corpus | Purpose | Access |
|---|---|---|
| CASIA-WebFace / MS-Celeb-1M (cleaned) | encoder pretraining | academic license |
| LFW / CFP-FP / AgeDB / IJB-C | face verification | public |
| FVC-ongoing | fingerprint verification | public |
| CASIA-FASD, OULU-NPU, CelebA-Spoof | PAD spoof corpora | request-based |
| UBFC / PURE / MMSE | rPPG signal quality | request-based |
| DFDC / FaceForensics++ | deepfake rejection | request-based |

Publish per-corpus metrics in a PR that updates the table below — issues
without reproducible scripts will be closed.

| Corpus Case | Metric | Reported | Reference |
|---|---|---|---|
| *(your PR here)* | | | |

