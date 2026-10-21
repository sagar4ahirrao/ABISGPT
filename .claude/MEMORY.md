# ABISGPT — Project Memory (local; kept in D:\GenAI\ABISGPT\.claude, gitignored)

- PAD/MATCHER ALL-DATA HARDENING DONE (Oct 21 "liveness check on all data +
  fix all issues"): every fix pytest+live-gate verified; details in
  TEST-REPORT §S.8. Key defects fixed:
  (1) HeuristicVlmPad scale bug (float=1.0 live always / uint8=0 always) →
      canonical ×255 for true-0..1 frames (peak≤1.5 AND min≥0);
  (2) resolution bug (64-128px reals SC-score higher than moiré; ordering
      inverts) → tier resamples every input to console 32² INTER_AREA;
      recalibrated at 32²/0..255: LO=0.975 HI=0.995 (osl314 reals≤0.977,
      moiré 0.995, noise 0.998);
  (3) single-frame PAD blocked 100% of genuine verify probes (FNMR 1.0):
      still_mode = replica OR len<2; zero measurable tiers → honest
      "unproven" pass, fused=None, is_live=True — bench_liveness_all
      Part2 pad_1 == pad_0 EXACTLY on all 4 datasets (block_g=0,
      still_cov=1.0);
  (4) restart gallery loss (Qdrant points vs empty in-memory gallery):
      VectorIndex.items() snapshot hook, TwoTierPipeline restores exact
      gallery on init; Faiss PQ restore deliberately blocked; test_restore.py;
  (5) moiré test must be 32² native amp=1.6 (amp=2.5 near-Nyquist
      INTER_AREA smears → ae 0.64 → live!); synthetic generators must use
      demo_synthetic 0..255 convention (±2.5 pulse on 0..1 base = out of
      range → green −1.7..2.4 → VLM misjudged);
  (6) PrecisionVerifier dot deprecation + dim mismatch → reshape+guard.
- Faiss GPU DONE: IndexConfig.use_gpu + ABISGPT_INDEX_GPU=1 + backend
  "faiss-gpu" + pyproject [gpu]; cuda ok → index_cpu_to_gpu w/
  StandardGpuResources, else gpu="unavailable" honest CPU fallback;
  efSearch tune via self._hnsw (pre-wrap handle); tests incl skipif-CUDA.
- Android DONE (delivered, not device-verified): scripts/android/
  termux_bootstrap.sh + docs/ANDROID.md (Mermaid) — Termux proot Ubuntu,
  requirements-light venv, brute backend, ONNX trained on device
  (12ep ~minutes) or --no-train with copied encoder. NOT run on real phone.
- Final verdicts Oct 21: pytest 45 passed / 3 skipped; ruff clean; all 4
  live gates PASS (diag_live_onnx 0.632/0.204/1:N 0.753, api_smoke,
  ui_smoke 12/12, e2e_ui 12/12). CASIA still AUC ≈0.50 toy-tier ceiling
  (documented); cross-domain FMR 0.83 honest note. Bench artifacts:
  artifacts/liveness_all.json + pad_bench_fusion{,_noStillfix}.json.
- RESUME LIST: ① user pushes pending commits ② optional: iris larger real
  set (UniDataPro gated); ③ Android script on-device verification if user
  has a phone; ④ production PAD tier = CLIP native-res + 12+ frame rppg.
- UI UPLOAD BUG FIXED (Oct 20, user-reported "upload not working"):
  ROOT = `_DASHBOARD_HTML` was a plain """ string → Python unescaped `\n`/`\'`
  in the inlined JS → browser "Invalid or unexpected token" → ALL UI JS dead.
  FIX = raw string r""" (graft_dashboard.py now emits r"""). Also fixed:
  identity widget replaceWith(select) hack (2nd enroll impossible after
  1 verify — now input#identity + select#identity-sel visibility-toggled);
  doCross `await fetch(...).json()` promise-`-json` TypeError (double await);
  identify label candidates[0].identity_id (was undefined).
  scripts/e2e_ui.py = Playwright chromium REAL-browser e2e, 12/12 PASS,
  screenshots artifacts/ui_shots/. Gate: node --check served JS via
  artifacts/check_served_js.py. Graft pipeline unchanged (edit artifact →
  graft → docker compose up -d --build api).
- REMOTE GPU DONE (13.235.199.152:2222, pw in $env:ABISGPT_SSH_PW, venv
  ~/fine-tuning/.venv + transformers 5.5.0 + cv2 5.0.0 + torch cu130):
  (a) CLIP PAD native-res full CASIA-FASD → test AUC 0.7182 best-acc 0.7550
  TPR@0.5 0.198 (attack-biased; ~860 img/s) — artifacts/remote/pad_clip_native.json;
  (b) SOCOFing ArcFace fine-tune = NEGATIVE result: unseen-finger recall@1 0.01
  EER 0.495; raw-pixel baseline ALSO chance (genuine 0.9462 ≈ impostor 0.9458)
  → 64² crops 10 frames/finger carry no generalizable signal; documented in
  TEST-REPORT §S.7.2. Scripts scripts/remote/{train_socofing,eval_pad_clip_native}.py;
  remote side needs nothing else installed.
  Remote gotchas: pgrep/pkill -f kills own ssh wrapper (kill by exact
  venv path); ArcFace scatter_ inplace breaks autograd (use torch.where
  one-hot); transformers 5 get_text_features may return output object not
  tensor; remap identity labels to 0..K-1 before ArcFace head.
- RESUME LIST: ① user pushes remaining commits (git push -u origin main)
  ② optional iris larger set (gated) ③ dataset results table refresh in
  README if user wants (CLIP numbers in TEST-REPORT §S.7).


- Session Oct 20 2026 (OpCode/GLM): branch published to GitHub by user (manual push, no gh CLI). Commits: 7dc924d scaffold → 2fec1e3 VLM/training/API/Docker → 7cc8bcc match-layer+scale → fa41e59 wip Olivetti+Milvus.
- USER-FACING RULES: all diagrams in repo docs/README must be Mermaid (no ASCII art); user requested agents/skills usage; memory lives in .claude folder of the repo; user pushes to GitHub manually.
- REMOTE GPU (user offered): `ssh -p 2222 development@13.235.199.152` — ASK USER FOR PASSWORD when a heavy GPU run is needed (user enters it interactively); use for CLIP-224² CASIA full eval / fingerprint ArcFace fine-tune if user green-lights.
- TRANSFORMERS STATUS (question answered + executed): (1) trained-from-scratch model is CNN+ArcFace (ToyBiometricEncoder, abisgpt/training/model.py) — NOT transformer; (2) DINOv2 ViT-B/16 pretrained EVALED for real via scripts/eval_dinov2_olivetti.py = recall@1 0.95 EER 0.0875 (BEHIND toy CNN 0.9875/0.0014 — finetune wins, as expected); weights cached; timm 1.0.30 installed locally; (3) CLIP ViT-B/32 PAD tier real-eval via eval_pad_casia.py --clip = AUC 0.572 on 64×64 crops (TPR 0.80 TNR 0.10 — attack-biased at low res); (4) heuristic VLM tier ceiling on CASIA-FASD = 0.7505/0.7442 acc after clamp-defect recalibration (scripts/diag_pad_native3.py); saturates at native res → tier stays research-only.
- SIMPLE UI DONE: dashboard v2 (graft pipeline: artifacts/dashboard_new.html + artifacts/graft_dashboard.py → service.py `_DASHBOARD_HTML`, idempotent loop-consume closer). 3 cards: upload face → action select (Enroll/Verify/Identify/PAD/N:N) → Run + registry + API ref. Frames contract = F×H×W×RGB NESTED triples (NOT flat!) — embedding API FrameShape nested. cameras/webcam dropped per user "keep simple".
- UI e2e: scripts/ui_smoke.py 11/11 PASS vs live container (real Olivetti face): enroll→verify 1.0→pad-bypass pad_live=null→fail-closed when gate on→identify rank1→N:N clean 0 dups→double-enroll detected 1.0→forget 200/404-after. Verify schema = {identity_id,is_match,score,pad_live,elapsed_ms}; identify = {decision,candidates[tier1_score,tier2_score],pad_live,elapsed_ms}; verify bypass → pad_live None (service.py override; honest marker). GDPR delete returns removed as INT count (1), not bool.
- pytest 39 passed / 2 skipped after: test_verify_photo_mode_pad_bypass_honest_none + endpoints fixes (eval_pad_casia clip wrapper float(report), diag scripts lint, count filter OK, qdrant remove filter OK).
- Stale-state caveat: container rebuild keeps Qdrant points but wipes in-memory _gallery — rerun ui_smoke may hit ghost-404½? No: enroll upserts; only crashed-interrupt leaves orphans → clean via DELETE before smoke.
- RESUME LIST (superseded — see top block): ② optional CLIP 224² CASIA on remote GPU → DONE ③ fingerprint fine-tune → DONE (negative) ④ README UI section + dataset results → DONE ⑤ iris larger set (UniDataPro gated?) if user wants.
- Datasets: artifacts/dataset/olivetti.npz = sklearn fetch_olivetti_faces (figshare ndownloader.figshare.com/files/5976027, AT&T ORL, Samaria & Harter 1994; no auth). shapes (400,64,64,3) float32 [0,1]; labels 0-39 contiguous; names person_000-039; manifest JSON w/ sha256; sklearn cache at artifacts/data/sklearn for offline reuse. Prepare script scripts/prepare_dataset.py.
- pad/liveness + iris + fingerprint: known sets gated (CelebA-Spoof, REPLAY-ATTACK, OULU-NPU, CASIA-SURF/-IrisV4, FVC2000) — no anonymous download; continue per honest synthetic-only PAD tests unless subagent finds ungated.
- Encoder contract: vit_encoder.py _preprocess → np.uint8 H×W×3 or H×W (255, no resize); toy model ToyBiometricEncoder (trainer) any spatial, 3 channels REQUIRED, AdaptiveAvgPool2d(1); strategy="hash" bypasses preprocess — SHA256 of RAW array bytes (uint8 implied). ViTBiometricEncoder strategy "torch" = timm DINOv2 lazy load 350 MB (works, evaluated).
- Olivetti train run v1 (cuda, 30ep, loss .48, 31.5 s): template_probe recall@1 = 1.0 (self-included template); the shipped leave-outs 0.0 because templates built from the PROBE frames themselves (self-referential). FIX = standard protocol: gallery = mean template of 8 TRAIN frames per identity (encoded by trained encoder); probes = 2 held-out frames; no self-exclusion. After fix optionally add impostor-distribution EER/AUC.
- pymilvus 2.3.4 gotchas (verified in source): DataType members UPPER (DT.VARCHAR / DT.FLOAT_VECTOR — "VarChar" AttributeError); Collection.insert accepts columnar list-of-lists [[ids…],[vectors…]] matching schema order (row-of-tuples fails "expect 2 list, got 128") → fix add() to `insert([list(ids), np.asarray(embeddings, np.float32).tolist()])`.
- Milvus docker: `docker compose --profile milvus up -d` — etcd v3.5.5 + milvusdb/minio:RELEASE.2024-12-18T13-15-44Z + milvusdb/milvus:v2.3.4 (old minio/minio RELEASE tags REMOVED from Docker Hub — use milvusdb mirror; hub itself reachable: hello-world + quay.io etcd pulled OK). Health localhost:9091/healthz ("OK"), gRPC 19530.
- Windows/venv: $env:KMP_DUPLICATE_LIB_OK='TRUE' for torch+faiss co-runs; torch 2.6.0+cu118 GT 1030; NEVER PowerShell `-Encoding UTF8` (adds BOM → breaks TOML); long outputs `> $env:TEMP\x.log` then Get-Content; `docker compose up -d --build api` and `python scripts/api_smoke.py http://localhost:8000` (SMOKE PASS in containers).
- Bench baselines (harness scripts/scale_test.py): crossmatch chunked GEMM ~92 M pairs/s CPU; 100k p50 0.43 ms / 500k 0.80 ms, recall@5 1.0/0.91 (nprobe-tunable), snapshot restore ~760 MB/s → 10M ≈ 1-3 s. IVF-PQ train needs 39·max(nlist,2^8) rows per bulk add.
- /verify endpoint + MatchingEngine (1:1, batch identify, NxN crossmatch, watchlist_screen, MultiModalProbe fused) tested in tests/test_matching.py.
