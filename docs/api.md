# API Reference

## `openabis.config`

```python
@dataclass
class IndexConfig:      # backend, nlist, nprobe, ef_search, m, pq_segments
@dataclass
class PadConfig:        # weights for VLM, rPPG, flash; thresholds
@dataclass
class PipelineConfig:   # encode batch, match threshold, top_k
```

Example YAML (see `configs/model.yaml`):

```yaml
index:
  backend: faiss
  nlist: 256
  nprobe: 16
  ef_search: 128
  m: 32
  pq_segments: 64
pad:
  vlm_weight: 0.4
  rppg_weight: 0.4
  flash_weight: 0.2
  gate_threshold: 0.6
pipeline:
  top_k: 10
  match_threshold: 0.72
```

## `openabis.encoder.ViTBiometricEncoder`

```python
enc = ViTBiometricEncoder.from_pretrained(cfg)
emb = enc.encode(frames)               # list[np.ndarray HxWx3] -> (n, 512), L2-normalized
```

## `openabis.pad.PassivePadEngine`

```python
pad = PassivePadEngine(cfg.pad)
verdict = pad.evaluate(frames, fps=60)      # -> PadVerdict(passed, scores, reasons)
```

Components are independently callable:

- `openabis.pad.rppg.RppgAnalyzer.from_frames(frames, fps=60)` → pulse stats + liveness
- `openabis.pad.flashdiff.FlashDiffusionAnalyzer.from_frames(frames, fps=60)` → diffuse map stats

## `openabis.index`

```python
backend = openabis.index.build_backend(cfg.index)
backend.add(ids=..., embeddings=...)      # bulk gallery enroll
hits = backend.search(query, top_k=10)    # Tier-1 ANN triage
```

Backends: `faiss` (IVF-HNSW + PQ), `qdrant` (client server-side `hnsw_config` +
scalar/product quantization), `milvus` (`IVF_SQ8` / `HNSW` hybrid mode).

## `openabis.pipeline.TwoTierPipeline`

```python
pipe = TwoTierPipeline(encoder, pad, index_backend, cfg.pipeline)
result = pipe.enroll(identity_id, images)             # returns (id, embedding)
verdict = pipe.identify(probe, top_k=10)              # Tier-1 triage + Tier-2 rerank
```

## `openabis.vault.VectorVault`

```python
vault = VectorVault(db_path="vault.db")   # reference impl; swap for prod store
vault.link("uuid-...", {"name_hash": sha256(name), "kyc_ref": "..."})
```

Never store raw PII in the vector DB — only `uuid`.
