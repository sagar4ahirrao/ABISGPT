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
def identify(req: IdentifyRequest, pad: bool = True):
    if not req.frames:
        raise HTTPException(status_code=400, detail="frames must not be empty")
    frames = [np.asarray(f, dtype=np.float32) for f in req.frames]
    result = get_pipeline().identify(frames, top_k=req.top_k or None, pad=pad)
    return result.as_dict()


@app.post("/pad")
def pad_only(req: IdentifyRequest):
    frames = [np.asarray(f, dtype=np.float32) for f in req.frames]
    verdict = get_pipeline().pad.evaluate(frames)
    return verdict.as_dict()


@app.post("/verify")
def verify_two(identity_id: str, req: IdentifyRequest, pad: bool = True):
    """1:1 authentication: PAD gate → encode → exact cosine vs claimed identity."""
    from abisgpt.pipeline.matching import MatchingEngine

    if not req.frames:
        raise HTTPException(status_code=400, detail="frames must not be empty")
    frames = [np.asarray(f, dtype=np.float32) for f in req.frames]
    engine = MatchingEngine(get_pipeline())
    result = engine.verify_1v1(identity_id, frames, pad=pad)
    return result.as_dict()


@app.get("/identities")
def identities():
    pipe = get_pipeline()
    ids = pipe.identities()
    return {"identities": ids, "count": len(ids)}


@app.post("/crossmatch")
def crossmatch(limit: int = 64, threshold: float = 0.9):
    """N:N registry dedup probe: first `limit` enrolled templates crossed against
    each other, self-pairs excluded — finds double enrollments (near-duplicates)."""
    from abisgpt.pipeline.matching import MatchingEngine

    pipe = get_pipeline()
    ids = pipe.identities()[: max(1, min(limit, 512))]
    if len(ids) < 2:
        return {"n_identities": len(ids), "pairs": [], "n_duplicates": 0,
                "note": "need >=2 enrolled templates"}
    gal = {i: pipe.template(i) for i in ids}
    engine = MatchingEngine(pipe)
    res = engine.crossmatch_nxn(gal, gal, threshold=threshold)
    real_dups = [(p, g, s) for (p, g, s) in res.pairs if p != g]
    return {
        "n_identities": len(ids),
        "pairs_scored": res.n_queries * res.n_candidates,
        "elapsed_ms": round(res.elapsed_ms, 3),
        "pairs_per_sec": round(res.pairs_per_sec),
        "duplicates": [{"a": p, "b": g, "score": round(s, 6)} for p, g, s in real_dups[:200]],
        "n_duplicates": len(real_dups),
    }


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
<html><head><meta charset="utf-8"><title>ABISGPT Biometric Test Console</title>
<style>
 :root{--bg:#0d1117;--card:#161b22;--fg:#e6edf3;--dim:#8b949e;--accent:#58a6ff;
       --ok:#3fb950;--warn:#d29922;--bad:#f85149;}
 body{background:var(--bg);color:var(--fg);font-family:'Segoe UI',system-ui,sans-serif;
      margin:0;padding:24px;max-width:1100px}
 h1{font-size:1.5rem;margin:0 0 4px} p.sub{color:var(--dim);margin:0 0 18px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
 .card{background:var(--card);border:1px solid #30363d;border-radius:10px;padding:16px}
 .card h2{font-size:.95rem;margin:0 0 6px;color:var(--accent)}
 .card p.hint{font-size:.78rem;color:var(--dim);margin:0 0 10px}
 button{background:#21262d;border:1px solid #30363d;color:var(--fg);padding:7px 14px;
        border-radius:6px;cursor:pointer;margin-right:6px;margin-top:4px;font-size:.85rem}
 button:hover{border-color:var(--accent)} button:disabled{opacity:.4;cursor:wait}
 button.primary{background:#1f6feb;border-color:#1f6feb;color:#fff}
 input,select{background:#0d1117;border:1px solid #30363d;color:var(--fg);
        border-radius:6px;padding:6px 10px;font-size:.85rem;margin-top:4px}
 .out{background:#0d1117;border:1px solid #30363d;border-radius:6px;margin-top:10px;
      padding:10px;max-height:280px;overflow:auto;font:12px/1.5 Consolas,monospace;
      white-space:pre-wrap;color:var(--dim)}
 .ok{color:var(--ok)} .bad{color:var(--bad)} .warn{color:var(--warn)}
 .pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.75rem;
       border:1px solid #30363d;background:#21262d;margin-left:8px}
 .verdict{font-size:1rem;font-weight:600;margin-top:8px}
 #preview{width:128px;height:128px;image-rendering:pixelated;border:1px solid #30363d;
       border-radius:8px;background:#0d1117;display:none}
 .rowline{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
 .bar{height:8px;background:#0d1117;border:1px solid #30363d;border-radius:99px;
      margin-top:4px;overflow:hidden}
 .bar>i{display:block;height:100%;background:var(--accent)}
 table{width:100%;border-collapse:collapse;font-size:.8rem;margin-top:8px}
 td{border-top:1px solid #21262d;padding:4px 6px;color:var(--dim)}
 td button{padding:2px 8px;margin:0}
</style></head>
<body>
<h1>ABISGPT Biometric Test Console <span class="pill" id="backend-pill">loading…</span>
    <span class="pill" id="gpu-pill">gpu: ?</span></h1>
<p class="sub">Enroll identities, then exercise every match mode — 1:1 verify, 1:N
 identify, N:N dedup, and the fail-closed PAD gate. Webcam bursts carry a real pulse;
 still photos don't — use photo mode (research PAD bypass) for them.</p>
<div class="grid">

 <div class="card"><h2>1 · Probe source</h2>
   <p class="hint">The live probe template everyone else acts on. Pick a source:
    synthetic pulse-video (deterministic demo), a still photo you upload or drag
    (single frame — PAD will fail-closed unless photo mode is on), or a short
    webcam burst (real rPPG pulse, PAD can pass).</p>
   <div class="rowline">
     <button onclick="synthProbe()">synthetic pulse</button>
     <button onclick="$('imgfile').click()">photo…</button>
     <button onclick="camBurst()">webcam burst</button>
     <input type="file" id="imgfile" accept="image/*" hidden
            onchange="imageFile(this.files[0])">
   </div>
   <div class="rowline" style="margin-top:8px">
     <img id="preview" alt="generated preview">
     <div id="probe-meta" style="color:var(--dim);font-size:.8rem">no probe yet</div>
   </div>
   <label style="font-size:.8rem;margin-top:8px">
     <input type="checkbox" id="pad-bypass" checked>
     photo mode — research PAD bypass (pad=false, honest pad_live: null)</label>
 </div>

 <div class="card"><h2>2 · Enroll new identity</h2>
   <p class="hint">Signature: live probe source → template → registry. Webcam/synthetic
    bursts stay PAD-gated on later probes; photos enroll fine too.</p>
   <input id="new-id" placeholder="identity, e.g. alice" size="14">
   <button class="primary" onclick="enroll()">Enroll</button>
   <div class="out" id="enroll-out">–</div></div>

 <div class="card"><h2>3 · Verify (1:1)</h2>
   <p class="hint">Probe → claimed identity: PAD gate → encode → exact cosine vs the
    claimed template. Yes = authentication pass.</p>
   <div class="rowline">
     <select id="verify-id" style="min-width:130px"></select>
     <button class="primary" onclick="doVerify()">verify</button></div>
   <div class="out" id="verify-out">–</div></div>

 <div class="card"><h2>4 · Identify (1:N)</h2>
   <p class="hint">Probe → whole registry: Tier-1 ANN → Tier-2 exact rerank returns
    the decision + ranked candidates.</p>
   <button class="primary" onclick="doIdentify()">identify</button>
   <div class="out" id="identify-out">–</div></div>

 <div class="card"><h2>5 · PAD gate only</h2>
   <p class="hint">Verdict of the fused passive liveness engine on the probe
    (rPPG + flash + VLM tiers). Single stills fail closed by design — the tiers
    need temporal pulse structure.</p>
   <button onclick="doPad()">run PAD</button>
   <div class="out" id="pad-out">–</div></div>

 <div class="card"><h2>6 · Registry</h2>
   <p class="hint">Everything enrolled in the vector graph right now. Deletion is the
    GDPR Art. 17 erasure hook.</p>
   <button onclick="refreshRegistry()">refresh</button>
   <table id="reg-table"><tbody></tbody></table></div>

 <div class="card"><h2>7 · N:N cross-match (dedup)</h2>
   <p class="hint">First 64 enrolled templates crossed against each other, self-pairs
    excluded — surfaces double enrollments of the same person (matrix GEMM path).
    Scale story: docs/SCALE-PLAN.md.</p>
   <button class="primary" onclick="doCrossmatch()">run N:N</button>
   <div class="out" id="cross-out">–</div></div>

 <div class="card"><h2>Performance & CI benches</h2>
   <p class="hint">Scripted stand-ins of the console checks for regression testing —
    clustered synthetic vectors, no PAD.</p>
   <div class="rowline">
     <button onclick="runPadBench()">PAD probes</button>
     <button onclick="runRetrieval(300)">300 ids</button>
     <button onclick="runRetrieval(3000)">3000 ids</button>
     <button onclick="launchTrain(24,16,3)">ArcFace mini</button></div>
   <div class="out" id="bench-out">–</div></div>

 <div class="card"><h2>HTTP API</h2>
   <p class="hint">Every console action maps 1:1 to a route — CI/widgets scripting.</p>
   <div class="out">GET  /health · GET /identities
POST /enroll        {identity_id, frames}
POST /verify        ?identity_id=…&pad=true|false   {frames}
POST /identify      ?pad=true|false                 {frames}
POST /pad           {frames}
POST /crossmatch    ?limit=64&threshold=0.9
DELETE /identities/{id}
POST /api/bench/pad · /api/bench/retrieval · /api/bench/train → GET /api/bench/train
GET  /api/status</div></div>
</div>
<script>
const $=id=>document.getElementById(id);
const fmt=o=>typeof o==='string'?o:JSON.stringify(o,null,2);
function put(id,obj){const el=$(id);el.textContent=fmt(obj);el.className='out';}
function say(id,html,cls){const el=$(id);el.innerHTML=html;el.className='out '+(cls||'');}
let PROBE=null;            // {kind, frames:[F][H*W*3] 0..1, pulse:boolean}

/* ---------- synthetic (deterministic) pulse capture ---------- */
function mulberry(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function synthProbe(){
  const name=prompt('identity name for the synthetic capture:','live_'+String(Date.now()%10000).padStart(4,'0'));
  if(!name)return;
  const rnd=mulberry([...name].reduce((a,c)=>(a*31+c.charCodeAt(0))>>>0,7)),F=72,H=32;
  const frames=[];
  for(let f=0;f<F;f++){
    const img=new Array(32*32*3);
    for(let p=0;p<img.length;p+=3){
      img[p]=.5;
      img[p+1]=Math.min(1,Math.max(0,.5+2.5*Math.sin(2*Math.PI*1.0*f/60)+((rnd()-.5)*1.2)));
      img[p+2]=.5;
    }
    frames.push(img);
  }
  PROBE={kind:'synthetic pulse video',frames,pulse:true};
  previewFrames(frames);
}
/* ---------- still photo ---------- */
async function imageFile(file){
  if(!file)return;
  const bmp=await createImageBitmap(file);
  const c=document.createElement('canvas');c.width=c.height=32;
  c.getContext('2d').drawImage(bmp,0,0,32,32);
  const d=c.getContext('2d').getImageData(0,0,32,32).data,a=[];
  for(let i=0;i<d.length;i+=4)a.push(d[i]/255,d[i+1]/255,d[i+2]/255);
  PROBE={kind:'still photo: '+(file.name||'image'),frames:[a],pulse:false};
  previewFrames(PROBE.frames);
}
/* ---------- webcam burst (32 frames @ ~18 fps) ---------- */
async function camBurst(){
  try{
    const stream=await navigator.mediaDevices.getUserMedia({video:{width:320,height:240}});
    const v=document.createElement('video');v.srcObject=stream;await v.play();
    const c=document.createElement('canvas');c.width=c.height=32;const ctx=c.getContext('2d');
    const frames=[];
    for(let f=0;f<32;f++){
      await new Promise(r=>setTimeout(r,55));
      ctx.drawImage(v,0,0,32,32);
      const d=ctx.getImageData(0,0,32,32).data,a=[];
      for(let i=0;i<d.length;i+=4)a.push(d[i]/255,d[i+1]/255,d[i+2]/255);
      frames.push(a);
    }
    stream.getTracks().forEach(t=>t.stop());
    PROBE={kind:'webcam burst',frames,pulse:true};
    previewFrames(frames);
  }catch(e){$('probe-meta').textContent='camera unavailable: '+e.message;}
}
function previewFrames(frames){
  const c=document.createElement('canvas');c.width=32;c.height=32;
  const ctx=c.getContext('2d'),img=ctx.createImageData(32,32);
  for(let i=0;i<frames[0].length;i++)img.data[i]=Math.round(Math.min(1,Math.max(0,frames[0][i]))*255);
  for(let i=3;i<img.data.length;i+=4)img.data[i]=255;
  ctx.putImageData(img,0,0);
  const p=$('preview');p.src=c.toDataURL();p.style.display='block';
  $('probe-meta').textContent=PROBE.kind+' · '+frames.length+' frames (32×32)';
}
function needProbe(){if(!PROBE){say('pad-out','make a probe first (step 1)','warn');return true;}return false;}
function padFlag(){return $('pad-bypass').checked;}

/* ---------- identity ops ---------- */
async function enroll(){
  if(needProbe())return;
  const id=$('new-id').value.trim();if(!id){say('enroll-out','enter an identity name','warn');return;}
  say('enroll-out','enrolling '+id+' …');
  const r=await fetch('/enroll',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({identity_id:id,frames:PROBE.frames})});
  put('enroll-out',await r.json());if(r.ok)refreshRegistry();
}
async function doVerify(){
  if(needProbe())return;
  const id=$('verify-id').value;if(!id){say('verify-out','refresh registry first','warn');return;}
  say('verify-out','verifying '+id+' …');
  const r=await(await fetch(`/verify?identity_id=${encodeURIComponent(id)}&pad=${!padFlag()}`,
    {method:'POST',headers:{'Content-Type':'application/json'},
     body:JSON.stringify({frames:PROBE.frames})})).json();
  showVerdict('verify-out',r);
}
async function doIdentify(){
  if(needProbe())return;
  say('identify-out','identifying …');
  const r=await(await fetch(`/identify?pad=${!padFlag()}`,{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({frames:PROBE.frames})})).json();
  showIdentify('identify-out',r);
}
function showVerdict(id,r){
  const cls=r.decision==='rejected'?'bad':(r.decision==='no_match'?'warn':'ok');
  const badge=r.decision==='match'?'<span class="ok">MATCH ✔</span>'
    :r.decision==='no_match'?'<span class="warn">NO MATCH</span>'
    :'<span class="bad">REJECTED (fail-closed PAD) ✘</span>';
  const pad=r.pad_live===null?'<span class="warn">pad bypassed (photo mode)</span>'
    :('pad: '+(r.pad_live?'live ✔':'spoof ✘'));
  const score=r.score!==undefined?r.score:(r.top_hit&&r.top_hit.tier2_score)||0;
  say(id,`<div class="verdict ${cls}">${badge}</div>score ${Number(score).toFixed(4)} · ${pad} · ${Math.round(r.elapsed_ms)} ms`,cls);
}
function showIdentify(id,r){
  if(r.decision==='rejected'){showVerdict(id,{decision:r.decision,pad_live:false,elapsed_ms:r.elapsed_ms});return;}
  const cls=r.decision==='match'?'ok':'warn';
  const badge=r.decision==='match'?'<span class="ok">MATCH ✔</span>':'<span class="warn">NO MATCH</span>';
  const pad=r.pad_live===null?'<span class="warn">pad bypassed (photo mode)</span>'
    :('pad: '+(r.pad_live?'live ✔':'spoof ✘'));
  const tops=(r.candidates||[]).slice(0,5).map(c=>
    `${c.identity_id}: tier1 ${Number(c.tier1_score).toFixed(4)} · tier2 ${Number(c.tier2_score??0).toFixed(4)}`).join('\n');
  say(id,`<div class="verdict ${cls}">${badge}</div>${pad} · ${Math.round(r.elapsed_ms)} ms\n${tops}`,cls);
}
async function doPad(){
  if(needProbe())return;
  say('pad-out','running PAD gate …');
  const r=await(await fetch('/pad',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({frames:PROBE.frames})})).json();
  const bars=Object.entries(r.tier_scores||{}).map(([k,v])=>`${k}: ${Number(v).toFixed(3)}<div class="bar"><i style="width:${Math.round(v*100)}%"></i></div>`).join('');
  say('pad-out',`${bars}\nfused ${Number(r.fused_score||0).toFixed(3)} — <b class="${r.is_live?'ok':'bad'}">${r.is_live?'LIVE ✔':'REJECTED ✘'}</b>\n${(r.reasons||[]).join(' · ')}`,r.is_live?'ok':'bad');
}
async function doCrossmatch(){
  say('cross-out','cross-matching registry …');
  const r=await(await fetch('/crossmatch?limit=64&threshold=0.9',{method:'POST'})).json();
  const dups=(r.duplicates||[]).map(d=>`${d.a} ≡ ${d.b} (${d.score.toFixed(3)})`).join('\n')||'no duplicate identities found';
  say('cross-out',`${r.n_identities} identities · ${fmt(r.pairs_scored)} pairs @ ${r.pairs_per_sec} pairs/s\n${dups}`);
}
async function refreshRegistry(){
  const j=await(await fetch('/identities')).json();
  const tb=$('reg-table').tBodies[0];tb.innerHTML='';
  (j.identities||[]).forEach(id=>{const tr=document.createElement('tr');
    tr.innerHTML=`<td>${id}</td><td><button onclick="forget('${encodeURIComponent(id)}')">forget</button></td>`;
    tb.appendChild(tr);});
  const sel=$('verify-id'),prev=sel.value;
  sel.innerHTML=(j.identities||[]).map(i=>`<option>${i}</option>`).join('')||'';
  if(prev&&[...sel.options].some(o=>o.value===prev))sel.value=prev;
}
async function forget(id){
  await fetch('/identities/'+id,{method:'DELETE'});refreshRegistry();
}
/* ---------- benches (same routes as the pytest suite) ---------- */
async function runPadBench(){
  say('bench-out','running PAD probes …');
  const r=await(await fetch('/api/bench/pad',{method:'POST'})).json();
  say('bench-out',fmt(r),(!r.replay.is_live&&r.live.fused_score>=0.6)?'ok':'bad');
}
async function runRetrieval(n){
  say('bench-out','retrieval bench n='+n+' …');
  const r=await(await fetch(`/api/bench/retrieval?n_identities=${n}`,{method:'POST'})).json();
  say('bench-out',fmt(r),r['recall@k']>=0.95?'ok':'bad');
}
let trainPoll=null;
async function launchTrain(i,f,e){
  say('bench-out','training launcher …');
  const j=await(await fetch(`/api/bench/train?identities=${i}&frames=${f}&epochs=${e}`,{method:'POST'})).json();
  if(j.started&&!trainPoll)trainPoll=setInterval(pollTrain,2000);
  put('bench-out',j);
}
async function pollTrain(){
  const j=await(await fetch('/api/bench/train')).json();
  say('bench-out',fmt(j),j.status==='done'?'ok':(j.status==='failed'?'bad':''));
  if(!j.running){clearInterval(trainPoll);trainPoll=null;}
}
/* ---------- status pills ---------- */
async function pills(){
  try{const r=await(await fetch('/api/status')).json();
    $('backend-pill').textContent='backend: '+r.index_backend+' · enrolled '+r.enrolled;
    $('gpu-pill').textContent='torch '+r.torch.torch_build+' · cuda '+(r.torch.cuda_available?'on':'off');
    $('gpu-pill').className='pill '+(r.torch.cuda_available?'ok':'warn');
  }catch(e){$('backend-pill').textContent='service unreachable';}
}
pills();refreshRegistry();
setInterval(pills,10000);
</script>
</body></html>"""

