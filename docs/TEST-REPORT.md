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

---

# Supplement (2026-10-20 session): real-data runs + transformer backbones

## S.1 Real datasets actually downloaded, trained and evaluated

| Dataset | n | Result | Verdict |
|---|---|---|---|
| **Olivetti/AT&T faces** (`artifacts/dataset/olivetti.npz`, sklearn figshare) | 400 imgs / 40 ids | trained toy CNN+ArcFace 30 ep @ 39.7 s CUDA → **recall@1 0.9875 · recall@5 1.0 · EER 0.0014** (80 probes / 3120 impostor scores) | PASS — real-data recognition works |
| **SOCOFing fingerprints** (HF `veerav96/sokoto_real`, 1000×64×64×3 / 100 ids) | 1000 | recall@1 0.01 · EER 0.506 = chance; pixel-corr intra 0.9035 ≈ inter 0.9092 | known limit — global-avg-pool toy encoder can't read local minutiae; production minutiae/ArcFace-finger encoder is the documented swap-in |
| **UniDataPro iris sample** (HF, 12 imgs / 6 ids) | 12 | n too small for any split | dataset shortfall, not a code claim |
| **CASIA-FASD PAD** (mirror npz 927/927 + raw 256×256 archive) | 4063 imgs | see S.2/S.3 | tier-level honest + fail-closed engine |

## S.2 Transformer backbones actually exercised (user question S-2026-10-20)

**What was *created and trained* here: one CNN+ArcFace model (toy scale).**
Transformer models in this repo are *pretrained* backbones (downloaded, not
regionally trained) — the industry-correct usage — and both were actually
downloaded and evaluated in this session:

| Backbone | Where | Run | Result |
|---|---|---|---|
| **DINOv2 ViT-B/16** (timm `vit_base_patch16_224.dino`, ~86 M params) | `scripts/eval_dinov2_olivetti.py`, same seed-0 protocol as the trained CNN | embed 400 imgs (287.6 s CUDA, weights auto-download) | **recall@1 0.95 · recall@5 0.9625 · EER 0.0875** |
| **CLIP ViT-B/32 zero-shot** (`openai/clip-vit-base-patch32`) | `scripts/eval_pad_casia.py --clip` PAD tier | 558 still crops (64×64) | **AUC 0.572 — catches 80% of attacks (TPR 0.796) but only 10% TNR at the 0.5 gate** → attack-biased at crop resolution |

Reading, honestly:
- Frozen DINOv2 is *behind* the from-scratch CNN+ArcFace head on Olivetti
  (0.95 vs 0.9875 recall@1; EER 8.75% vs 0.14%) — general-purpose SSL
  features lose to task-finetuning at this scale, which is exactly why the
  pipeline's ArcFace fine-tune path is the production line and DINOv2 is a
  seed/fallback backbone.
- CLIP zero-shot on 64×64 stills is above chance but not deployment-grade;
  the tier's usable-tier status stays "research", full kiosk liveness still
  belongs to the 12+ frame rPPG engine (fail-closed verified on single frames).

## S.3 Fixed-threshold texture tier diagnosed & ceiling measured

Hypothesis "64×64 downscale destroyed the banding signal, not the tier" was
**falsified** (`artifacts/pad_native_diag*.json`, native 256×256 re-run):
- Shipped scorer saturates: ALL cropped photos → liveness 0.0 at native res
  (fixed |lap|>4 threshold is contrast-brittle); AUC exactly 0.5.
- Raw energy feature alone is mildly informative: best-threshold calibration
  reaches **0.7505 train / 0.7442 test accuracy** (AUC ≈ 0.68) — still below
  production quality. Conclusion stands: `HeuristicVlmPad` is a synthetic-bench
  stand-in; photo-PAD = CLIP/post-prod tier + pulse engine.

## S.4 Simple console UI (user request "keep simple to upload biometric
and perform all features available as chosen")

Replaced the multi-card dashboard with a 3-card flow:
`upload face photo (click or drag) → choose action (Enroll / Verify 1:1 /
Identify 1:N / Liveness;PAD / N:N cross-match) → Run`, plus a registry card
with GDPR forget and a compact API reference. Stills default to research
PAD-bypass (`pad=false`, honest `pad_live: null`); a single "treat as live
frames" switch re-enables the fail-closed gate.

## S.5 Live e2e smoke of the new console (real faces, Qdrant container)

`scripts/ui_smoke.py` → **11/11 PASS** (real Olivetti face, browser-identical
32×32 conversion):
enroll → verify match (cosine 1.0) → bypass pad=null → fail-closed when gate
on → identify rank-1 → 2nd identity → registry list → rank-stable identify →
PAD-only verdict → N:N clean 0 dups → double-enroll detected (score 1.0) →
forget (HTTP 200, removed 1, then 404).

`pytest tests -q` → **39 passed / 2 skipped** · `ruff check` all-clean after fixes:
verify-bypass returns honest `pad_live: null` (`abisgpt/api/service.py`),
console-title test, endpoint contract (`decision` for identify vs `is_match`
for verify), `/pad` response shape fix for CLIP wrapper.

## S.6 Remaining GPU-backed next steps

- CLIP tier at native 224² on full CASIA-FASD train+test — **done, §S.7**.
- SoCOFing fingerprint ArcFace fine-tune — **done, §S.7 (negative result)**.

## S.7 Remote GPU runs + real-browser UI sweep (final session additions)

### S.7.1 Playwright real-browser e2e of the console (`scripts/e2e_ui.py`, chromium)

Drove the actual upload→action→Run flow in headless Chromium against the live
container, with per-step screenshots (`artifacts/ui_shots/*.png`). Final state
**12/12 PASS**, zero console/page errors. During the sweep it found and
verified **four real shipped-page bugs**, all fixed in
`artifacts/dashboard_new.html` + re-grafted:

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Clicking upload did nothing; **every** feature dead | `_DASHBOARD_HTML = """…"""` unescaped `\n`/`\'` in the inlined JS at Python-parse time → browser saw literal newlines → "Invalid or unexpected token" → whole script never ran | raw string `r"""` (graft emits it); node `--check` on the served page added (`artifacts/check_served_js.py`) |
| 2 | After one Verify, **no second person could ever be enrolled** (name field became a dropdown) | Verify mode `replaceWith`-swapped the free-text `<input id=identity>` for a `<select>` and never swapped back | both widgets kept in DOM, visibility toggles by mode (`#identity` input ↔ `#identity-sel` select) |
| 3 | N:N cross-match hung on "N:N cross-match…" forever | `await fetch(...).json()` calls `.json` on the *Promise*, not the Response → TypeError | `await (await fetch(...)).json()` |
| 4 | Identify verdict showed "MATCH — undefined" | top candidate read a field identify doesn't return | use `r.candidates[0].identity_id` |

Checks covered: page load, upload via file chooser (preview + Run enable),
enroll (real Olivetti jpg through the UI), identity dropdown population,
verify 1:1 (score 1.0000), identify 1:N rank-1, PAD verdict rendering,
N:N crossmatch (duplicate detection exercised), registry listing, forget.

### S.7.2 Remote RTX 5090 runs (`scripts/remote/*.py`, results in `artifacts/remote/`)

**CLIP ViT-B/32 zero-shot PAD at native resolution, full CASIA-FASD**
(`eval_pad_clip_native.py` — 4,063 images across both `train`+`test` color
splits, spoof-anchor vs live-anchor margin scoring, ~860 img/s):

| split | n (real) | AUC | best-τ acc | TPR @0.5 gate |
|---|---|---|---|---|
| train | 1,655 (404) | 0.7089 | 0.7656 | 0.2252 |
| test | 2,408 (591) | 0.7182 | 0.7550 | 0.1980 |

Native resolution roughly doubles the crop-tier AUC (0.572 → 0.718) but the
zero-shot margin stays attack-biased at a fixed gate (stops most photo
attacks while letting ~80% of real presentations through). Deployable
liveness verdict remains the engineered 12+ frame rPPG engine; CLIP stays a
"research"-tier fallback.

**SOCOFing ArcFace fingerprint fine-tune — honest negative result**
(`train_socofing.py`, identity-split 60 train / 40 eval ids, 5+5 template/
probe frames, 150 epochs with shift+rotation augmentation, ArcFace s=16/m=0.10):

- Trained model on **unseen** fingers: recall@1 0.01, EER 0.495
  (final train loss 3.77 — converged on its 60 training identities).
- Untrained raw-pixel cosine baseline on the identical split: recall@1 0.01,
  EER 0.500, and the tell — mean genuine sim 0.9462 ≈ impostor 0.9458:
  every 64×64 grayscale crop cosine-collapses onto the shared ridge/ink
  background structure, so the cohort carries essentially *no cross-frame
  discriminative signal* for unseen fingers.
- Conclusion: discriminative fingerprint matching from this 100-identity,
  10-frame/finger SOKOKO subset requires minutiae-classical features or
  training on orders more identities (SOCOFing-paper scale). Recorded as a
  documented limitation; the repo's fingerprint path stays at its
  verified pipeline score rather than a false fine-tune claim.

