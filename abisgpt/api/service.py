"""ABISGPT FastAPI service — enroll / identify / PAD gate over HTTP.

Run locally:
    uvicorn abisgpt.api.service:app --host 0.0.0.0 --port 8000
or via Docker Compose (`docker compose up` from the repo root).

Environment:
    ABISGPT_CONFIG          path to a YAML config (configs/model.yaml)
    ABISGPT_INDEX_BACKEND   brute | faiss | qdrant   (overrides config)
    ABISGPT_INDEX_URI       endpoint for qdrant/milvus backends
    ABISGPT_VLM             unset | heuristic | clip   (VLM PAD tier)

Frames arrive as JSON arrays of HxW RGB float images (0..1). Keep them small
in this reference service; production should stream binary frames.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    import abisgpt
    VERSION = abisgpt.__version__
except Exception:                                     # pragma: no cover
    VERSION = "0.1.0"

from abisgpt.config import AbisGptConfig
from abisgpt.encoder import ViTBiometricEncoder
from abisgpt.index import build_backend
from abisgpt.pad import PassivePadEngine
from abisgpt.pipeline.triage import TwoTierPipeline

app = FastAPI(title="ABISGPT", version=VERSION,
              description="Open, reproducible biometric identification research stack")


# --------------------------------------------------------------------- #
# Pipeline singleton                                                     #
# --------------------------------------------------------------------- #
_PIPE: TwoTierPipeline | None = None


def _make_vlm_scorer():
    """Build the optional VLM tier from the ABISGPT_VLM env var."""
    choice = os.environ.get("ABISGPT_VLM", "").strip().lower()
    if choice == "heuristic":
        from abisgpt.pad.vlm import HeuristicVlmPad

        return HeuristicVlmPad()
    if choice == "clip":
        from abisgpt.pad.vlm import ClipVlmPad

        return ClipVlmPad()
    return None


def get_pipeline() -> TwoTierPipeline:
    global _PIPE
    if _PIPE is None:
        cfg_path = os.environ.get("ABISGPT_CONFIG")
        cfg = AbisGptConfig.from_yaml(cfg_path) if cfg_path else AbisGptConfig()

        backend = os.environ.get("ABISGPT_INDEX_BACKEND", "").strip().lower()
        if backend:
            cfg.index.backend = backend            # type: ignore[assignment]
        uri = os.environ.get("ABISGPT_INDEX_URI", "").strip()
        if uri:
            cfg.index.uri = uri

        encoder = ViTBiometricEncoder(strategy="hash")
        pad = PassivePadEngine(cfg.pad, vlm_scorer=_make_vlm_scorer())
        backend_obj = build_backend(cfg.index)
        _PIPE = TwoTierPipeline(encoder=encoder, pad=pad, index_backend=backend_obj, config=cfg)
    return _PIPE


def reset_pipeline() -> None:          # used by tests
    global _PIPE
    _PIPE = None


# --------------------------------------------------------------------- #
# Schemas                                                                #
# --------------------------------------------------------------------- #
class EnrollRequest(BaseModel):
    identity_id: str
    frames: list[list[list[list[float]]]]     # F x H x W x RGB floats (0..1)


class IdentifyRequest(BaseModel):
    frames: list[list[list[list[float]]]]     # F x H x W x RGB floats (0..1)
    top_k: int | None = None


# --------------------------------------------------------------------- #
# Routes                                                                 #
# --------------------------------------------------------------------- #
@app.get("/health")
def health():
    pipe = get_pipeline()
    return {
        "status": "ok",
        "version": VERSION,
        "index_backend": type(pipe.index).__name__,
        "enrolled": len(pipe.index),
    }


@app.post("/enroll")
def enroll(req: EnrollRequest):
    try:
        frames = [np.asarray(f, dtype=np.float32) for f in req.frames]
        rec = get_pipeline().enroll(req.identity_id, frames)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"identity_id": rec.identity_id, "dim": int(rec.embedding.shape[0]),
            "norm": round(float(np.linalg.norm(rec.embedding)), 6)}


@app.post("/identify")
def identify(req: IdentifyRequest):
    if not req.frames:
        raise HTTPException(status_code=400, detail="frames must not be empty")
    frames = [np.asarray(f, dtype=np.float32) for f in req.frames]
    result = get_pipeline().identify(frames, top_k=req.top_k or None)
    return result.as_dict()


@app.post("/pad")
def pad_only(req: IdentifyRequest):
    frames = [np.asarray(f, dtype=np.float32) for f in req.frames]
    verdict = get_pipeline().pad.evaluate(frames)
    return verdict.as_dict()


@app.post("/verify")
def verify_two(identity_id: str, req: IdentifyRequest):
    """1:1 authentication: PAD gate → encode → exact cosine vs claimed identity."""
    from abisgpt.pipeline.matching import MatchingEngine

    if not req.frames:
        raise HTTPException(status_code=400, detail="frames must not be empty")
    frames = [np.asarray(f, dtype=np.float32) for f in req.frames]
    engine = MatchingEngine(get_pipeline())
    result = engine.verify_1v1(identity_id, frames)
    return result.as_dict()


@app.delete("/identities/{identity_id}")
def forget(identity_id: str):
    removed = get_pipeline().drop(identity_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"{identity_id!r} not enrolled")
    return {"identity_id": identity_id, "removed": removed}


# --------------------------------------------------------------------- #
# Interactive test dashboard                                             #
# --------------------------------------------------------------------- #
_BENCH = {"running": False, "status": "idle", "metrics": None, "error": None}


def _clean_probe(n_frames: int, base: float, rng: np.random.Generator) -> list:
    """Pulse-carrying synthetic live capture (on-bin 1.0 Hz BVP)."""
    frames = []
    for f in range(n_frames):
        img = np.full((32, 32, 3), base, dtype=np.float32)
        img[:, :, 1] += 2.5 * np.sin(2 * np.pi * 1.0 * f / 60.0) + rng.normal(0, 0.6, (32, 32))
        frames.append(img.round(3).tolist())
    return frames


@app.get("/api/status")
def status():
    gpu = None
    try:
        import torch
        gpu = {
            "torch_build": torch.version.cuda or "cpu-only",
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        gpu = {"torch_build": "not-installed", "cuda_available": False}
    pipe = get_pipeline()
    return {"version": VERSION, "index_backend": type(pipe.index).__name__,
            "enrolled": len(pipe.index), "torch": gpu,
            "bench_running": _BENCH["running"]}


@app.post("/api/bench/pad")
def bench_pad():
    """Live probe (pulse) vs flat skip-gen replay attempt -> PAD verdicts."""
    pipe = get_pipeline()
    rng = np.random.default_rng(int(time.time()) & 0xFFFF)
    live = pipe.pad.evaluate([np.asarray(f, dtype=np.float32) for f in _clean_probe(180, 0.50, rng)])
    replay = pipe.pad.evaluate(
        np.full((180, 32, 32, 3), 0.55, dtype=np.float32))  # static flat frames
    return {"live": live.as_dict(), "replay": replay.as_dict()}


@app.post("/api/bench/retrieval")
def bench_retrieval(n_identities: int = 300, top_k: int = 5):
    """Exact (brute-force) recall@k + latency on synthetic clustered vectors."""
    from abisgpt.index.factory import BruteForceIndex

    rng = np.random.default_rng(1234)
    dim, vecs_per_id = 512, 2
    centers = rng.standard_normal((n_identities, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    vecs = np.concatenate(
        [centers[i][None, :] + rng.normal(0, 0.05, (vecs_per_id, dim)).astype(np.float32)
         for i in range(n_identities)])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms
    labels = np.repeat(np.arange(n_identities), vecs_per_id)

    index = BruteForceIndex(dim=dim)
    index.add([f"id_{i}" for i in labels], vecs)
    hits_correct, lat = 0, []
    for i in range(n_identities):
        probe = vecs[i * vecs_per_id] + 0.01 * rng.standard_normal(dim).astype(np.float32)
        t0 = time.perf_counter()
        top = index.search(probe, top_k=top_k)
        lat.append((time.perf_counter() - t0) * 1000)
        if any(hit_id == f"id_{i}" for hit_id, _ in top):
            hits_correct += 1
    lats = np.sort(lat)
    return {
        "n_identities": n_identities, "queries": n_identities, "top_k": top_k,
        "recall@k": round(hits_correct / n_identities, 4),
        "latency_p50_ms": round(float(lats[len(lats) // 2]), 4),
        "latency_p95_ms": round(float(lats[int(len(lats) * 0.95)]), 4),
    }


@app.post("/api/bench/train")
def bench_train(identities: int = 24, frames: int = 16, epochs: int = 3):
    """Kick off a tiny ArcFace training run in a background thread."""
    if _BENCH["running"]:
        return {"started": False, "reason": "already running", "status": _BENCH["status"]}
    _BENCH.update(running=True, status="starting", metrics=None, error=None)

    def _run():
        try:
            import torch  # noqa
            from abisgpt.training.trainer import TrainConfig, train

            cfg = TrainConfig(train_identities=identities, frames_per_identity=frames,
                              epochs=epochs, out_dir="artifacts/ui_runs")
            _BENCH["status"] = f"training ({identities} ids x {frames} frames x {epochs} epochs)"
            _BENCH["metrics"] = train(cfg, device="cuda" if torch.cuda.is_available() else "cpu")
            _BENCH["status"] = "done"
        except Exception as exc:                       # noqa: BLE001
            _BENCH["status"], _BENCH["error"] = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            _BENCH["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return {"started": True, "status": _BENCH["status"]}


@app.get("/api/bench/train")
def bench_train_status():
    return {k: v for k, v in _BENCH.items()}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return _DASHBOARD_HTML


_DASHBOARD_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>ABISGPT Test Dashboard</title>
<style>
 :root{--bg:#0d1117;--card:#161b22;--fg:#e6edf3;--dim:#8b949e;--accent:#58a6ff;
       --ok:#3fb950;--warn:#d29922;--bad:#f85149;}
 body{background:var(--bg);color:var(--fg);font-family:'Segoe UI',system-ui,sans-serif;
      margin:0;padding:24px;max-width:1100px;}
 h1{font-size:1.5rem;margin:0 0 4px} p.sub{color:var(--dim);margin:0 0 20px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
 .card{background:var(--card);border:1px solid #30363d;border-radius:10px;padding:16px}
 .card h2{font-size:.95rem;margin:0 0 4px;color:var(--accent)}
 .card p.hint{font-size:.8rem;color:var(--dim);margin:0 0 10px}
 button{background:#21262d;border:1px solid #30363d;color:var(--fg);padding:7px 14px;
        border-radius:6px;cursor:pointer;margin-right:6px;margin-top:4px}
 button:hover{border-color:var(--accent)} button:disabled{opacity:.4;cursor:wait}
 .out{background:#0d1117;border:1px solid #30363d;border-radius:6px;margin-top:10px;
      padding:10px;max-height:300px;overflow:auto;font:12px/1.5 Consolas,monospace;
      white-space:pre-wrap;color:var(--dim)}
 .ok{color:var(--ok)} .bad{color:var(--bad)} .warn{color:var(--warn)}
 #status pill:after{content:'  '}
 .pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.75rem;
       border:1px solid #30363d;background:#21262d;margin-left:8px}
</style></head>
<body>
<h1>ABISGPT Interactive Test Dashboard <span class="pill" id="backend-pill">loading…</span>
    <span class="pill" id="gpu-pill">gpu: ?</span></h1>
<p class="sub">Reference research stack — synthetic data only, deterministic, no real biometrics.</p>
<div class="grid">
 <div class="card"><h2>Service status</h2>
   <p class="hint">Version, vector backend, enrolled templates, torch/GPU build.</p>
   <button onclick="loadStatus()">refresh</button>
   <div class="out" id="status-out">–</div></div>
 <div class="card"><h2>PAD gate probes</h2>
   <p class="hint">Fused rPPG+flash+VLM verdict on a pulse-carrying live probe vs a flat
    screen-replay attempt. Fail-closed under threshold = PASS for replay.</p>
   <button onclick="runPad()">run PAD probes</button>
   <div class="out" id="pad-out">–</div></div>
 <div class="card"><h2>Retrieval bench</h2>
   <p class="hint">Exact-match recall@5 + latency percentiles over clustered synthetic
    512-D templates (index-level, no PAD).</p>
   <button onclick="runRetrieval('small')">300 ids</button>
   <button onclick="runRetrieval('big')">3000 ids</button>
   <div class="out" id="ret-out">–</div></div>
 <div class="card"><h2>Quick train (ArcFace)</h2>
   <p class="hint">Tiny encoder-training run in a background thread; reports
    unseen-identity recall. Uses CUDA when torch+GPU are available.</p>
   <button onclick="launchTrain(24,16,3)">24 ids / 3 epochs</button>
   <button onclick="launchTrain(48,24,4)">48 ids / 4 epochs</button>
   <button onclick="pollTrain(true)">poll status</button>
   <div class="out" id="train-out">–</div></div>
 <div class="card"><h2>HTTP API reference</h2>
   <p class="hint">Same checks, scripted — for CI and reproducibility.</p>
   <div class="out">GET  /health
POST /enroll            {"identity_id": "...", "frames": [[...frame...]]}
POST /identify          {"frames": [...]}
POST /pad               {"frames": [...]}
DELETE /identities/{id}
POST /api/bench/pad
POST /api/bench/retrieval
POST /api/bench/train   then GET /api/bench/train
GET  /api/status</div></div>
</div>
<script>
const $=id=>document.getElementById(id);
function put(id,obj){const el=$(id);el.textContent=typeof obj==='string'?obj:JSON.stringify(obj,null,2);
  el.className='out';}
function mark(id,ok){const el=$(id);el.classList.remove('ok','bad','warn');
  el.classList.add(ok?'ok':'bad');}
async function loadStatus(){
  const j=await(await fetch('/api/status')).json();
  put('status-out',j);
}
async function statusPills(){
  try{const r=await(await fetch('/api/status')).json();
    $('backend-pill').textContent='backend: '+r.index_backend+' · enrolled '+r.enrolled;
    $('gpu-pill').textContent='torch '+(r.torch.torch_build)+
      ' · cuda '+(r.torch.cuda_available?'available':'unavailable');
    $('gpu-pill').className='pill '+(r.torch.cuda_available?'ok':'warn');
  }catch(e){$('backend-pill').textContent='service unreachable';}
}
async function runPad(){
  put('pad-out','running…');
  const j=await(await fetch('/api/bench/pad',{method:'POST'})).json();
  put('pad-out',j);mark('pad-out',!j.replay.is_live && j.live.fused_score>=0.6);
}
async function runRetrieval(scale){
  put('ret-out','running…');
  const n=scale==='big'?3000:300;
  const j=await(await fetch(`/api/bench/retrieval?n_identities=${n}`,{method:'POST'})).json();
  put('ret-out',j);mark('ret-out',j['recall@k']>=0.95);
}
let trainPoll=null;
async function launchTrain(i,f,e){
  put('train-out','launching…');
  const j=await(await fetch(`/api/bench/train?identities=${i}&frames=${f}&epochs=${e}`,{method:'POST'})).json();
  if(j.started&&!trainPoll){trainPoll=setInterval(()=>pollTrain(),2000);}
  put('train-out',JSON.stringify(j,null,2));
}
async function pollTrain(once){
  const j=await(await fetch('/api/bench/train')).json();
  put('train-out',JSON.stringify(j,null,2));
  if(j.status==='done'&&!j.error){mark('train-out',true);}
  else if(j.status==='failed'){mark('train-out',false);}
  if(!j.running||once){clearInterval(trainPoll);trainPoll=null;statusPills();}
}
loadStatus();statusPills();
setInterval(statusPills,10000);</script>
</body></html>"""

