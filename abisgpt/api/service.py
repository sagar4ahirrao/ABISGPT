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
    body = result.as_dict()
    if not pad:
        body["pad_live"] = None  # gate skipped — honest "not evaluated" marker
    return body


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
<html><head><meta charset="utf-8"><title>ABISGPT Biometric Console</title>
<style>
 :root{--bg:#0d1117;--card:#161b22;--fg:#e6edf3;--dim:#8b949e;--accent:#58a6ff;
       --ok:#3fb950;--bad:#f85149;--warn:#d29922;}
 *{box-sizing:border-box}
 body{background:var(--bg);color:var(--fg);font-family:'Segoe UI',system-ui,sans-serif;
      margin:0;padding:28px 18px;display:flex;flex-direction:column;align-items:center}
 body.drag{outline:3px dashed var(--accent);outline-offset:-10px}
 main{width:100%;max-width:640px}
 h1{font-size:1.4rem;margin:0}
 p.sub{color:var(--dim);font-size:.85rem;margin:4px 0 16px}
 .card{background:var(--card);border:1px solid #30363d;border-radius:10px;
       padding:18px;margin-bottom:14px}
 .card h2{font-size:.9rem;margin:0 0 10px;color:var(--accent);
       text-transform:uppercase;letter-spacing:.05em}
 .drop{border:2px dashed #30363d;border-radius:8px;padding:18px;text-align:center;
       color:var(--dim);cursor:pointer;font-size:.9rem;transition:border-color .2s}
 .drop:hover,.drop.over{border-color:var(--accent)}
 .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:12px}
 img#preview{width:96px;height:96px;object-fit:cover;border:1px solid #30363d;
       border-radius:8px;background:#090c10}
 .meta{color:var(--dim);font-size:.8rem}
 button{background:#21262d;border:1px solid #30363d;color:var(--fg);padding:9px 16px;
       border-radius:7px;cursor:pointer;font-size:.9rem}
 button:hover{border-color:var(--accent)} button:disabled{opacity:.35;cursor:not-allowed}
 button.go{background:#1f6feb;border-color:#1f6feb;color:#fff;font-weight:600;
       padding:10px 22px;font-size:.95rem}
 select,input[readonly]{background:#0d1117;border:1px solid #30363d;color:var(--fg);
       border-radius:7px;padding:8px 10px;font-size:.9rem}
 #identity{background:#0d1117;border:1px solid #30363d;color:var(--fg);
       border-radius:7px;padding:8px 10px;font-size:.9rem;width:180px}
 small.hint{color:var(--dim);font-size:.75rem;display:block;margin-top:6px}
 label.live{color:var(--dim);font-size:.8rem;display:flex;gap:6px;align-items:center}
 #result{margin-top:14px;min-height:52px}
 #result .verdict{font-size:1.15rem;font-weight:700}
 .v-match{color:var(--ok)} .v-reject{color:var(--bad)} .v-none{color:var(--warn)}
 .out{background:#090c10;border:1px solid #30363d;border-radius:7px;margin-top:10px;
       padding:10px;max-height:220px;overflow:auto;font:12px/1.5 Consolas,monospace;
       white-space:pre-wrap;color:var(--dim)}
 table{width:100%;border-collapse:collapse;font-size:.85rem;margin-top:6px}
 td{border-top:1px solid #21262d;padding:6px 4px;color:var(--dim)}
 td button{padding:2px 10px;margin:0}
 .pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.75rem;
       border:1px solid #30363d;background:#21262d;margin-left:10px;vertical-align:middle}
 .pill.ok{color:var(--ok)} .pill.warn{color:var(--warn)}
 .out.ref{font-size:11px;max-height:none}
</style></head>
<body>
<main>
<h1>ABISGPT Biometric Console <span class="pill" id="status">…</span></h1>
<p class="sub">Upload a biometric image (face photo). Pick an action. Run.
Everything maps 1:1 to the HTTP API listed at the bottom.</p>

<div class="card">
 <h2>1 · Upload biometric</h2>
 <div class="drop" id="drop">click or drag &amp; drop a face image here<br>
   <small>(JPG / PNG / BMP — used as the probe for every action below)</small>
 </div>
 <div class="row">
  <img id="preview" alt="probe preview" style="display:none">
  <div class="meta" id="meta">no image loaded</div>
 </div>
 <input type="file" id="file" accept="image/*" hidden>
</div>

<div class="card">
 <h2>2 · Choose action &amp; run</h2>
 <div class="row" style="gap:8px">
  <select id="mode">
   <option value="enroll">Enroll — create identity from this image</option>
   <option value="verify">Verify (1:1) — against one identity</option>
   <option value="identify">Identify (1:N) — search whole registry</option>
   <option value="pad">Liveness (PAD) — is this a live capture?</option>
   <option value="crossmatch">N:N cross-match — find duplicates</option>
  </select>
  <input id="identity" placeholder="identity name…">
  <button class="go" id="run" disabled>Run</button>
 </div>
 <label class="live"><input type="checkbox" id="as-live">
  treat upload as live frames (runs the fail-closed PAD gate)</label>
 <small class="hint" id="mode-hint"></small>
 <div id="result"></div>
</div>

<div class="card">
 <h2>3 · Registry <button onclick="refresh()" style="float:right;padding:3px 12px">refresh</button></h2>
 <table id="reg"><tbody><tr><td class="meta">empty</td></tr></tbody></table>
</div>

<div class="card">
 <h2>HTTP API</h2>
 <div class="out ref">GET    /health · /identities · /api/status
POST   /enroll          {identity_id, frames[[r,g,b]…]}
POST   /verify          ?identity_id=&amp;pad=   {frames}
POST   /identify        ?pad=                 {frames}
POST   /pad             {frames}
POST   /crossmatch      ?limit=64&amp;threshold=0.9
DELETE /identities/{id}
POST   /api/bench/pad · /api/bench/retrieval · /api/bench/train</div>
</div>
</main>

<script>
const $=id=>document.getElementById(id);
let PROBE=null;   // {bars:[r,g,b]…s flattened array 0..1}

/* ---------------- upload ---------------- */
$('drop').onclick=()=>$('file').click();
$('file').onchange=e=>loadImage(e.target.files[0]);
addEventListener('dragover',e=>{e.preventDefault();document.body.classList.add('drag')});
addEventListener('dragleave',()=>document.body.classList.remove('drag'));
addEventListener('drop',e=>{e.preventDefault();document.body.classList.remove('drag');
  if(e.dataTransfer.files[0])loadImage(e.dataTransfer.files[0])});

async function loadImage(file){
  if(!file)return;
  const bmp=await createImageBitmap(file);
  const c=document.createElement('canvas');c.width=c.height=32;
  c.getContext('2d').drawImage(bmp,0,0,32,32);
  const d=c.getContext('2d').getImageData(0,0,32,32).data,
        px=[];let ri=0;
  for(let y=0;y<32;y++){const row=[];
    for(let x=0;x<32;x++){row.push([d[ri]/255,d[ri+1]/255,d[ri+2]/255]);ri+=4;}
    px.push(row);}
  PROBE=px;                      // F x H x W x RGB contract
  $('preview').src=c.toDataURL();$('preview').style.display='block';
  $('meta').textContent=file.name+' · 32×32 px · 3072 px';
  $('run').disabled=false;
}

/* ---------------- mode machinery ---------------- */
const HINTS={enroll:'Enrolls the uploaded image as a NEW identity — then it appears in the registry and can be verified / found by 1:N identify.',
 verify:'Claims the identity chosen below; the uploaded image is compared 1:1 against that person’s stored template.',
 identify:'Searches the WHOLE registry for the closest match (Tier-1 ANN + Tier-2 exact rerank).',
 pad:'Runs the passive liveness engine (rPPG+flash+texture tiers). Uploaded stills are usually not live captures — keep “as live frames” OFF unless using the webcam pipeline.',
 crossmatch:'Crosses all enrolled identities against each other (self-pairs excluded) to surface double enrollments. No probe needed.'};
function mode(){return $('mode').value;}
function syncMode(){
  $('mode-hint').textContent=HINTS[mode()]||'';
  const need=mode()==='enroll'||mode()==='verify';
  $('identity').style.display=need?'':'none';
  if(need&&mode()==='verify')fillSelect();
  $('identity').placeholder=mode()==='enroll'
    ?'new identity, e.g. alice':'claimed identity';
}
$('mode').onchange=syncMode;syncMode();
const GO={enroll:enroll,verify:doVerify,identify:doIdentify,pad:doPad,crossmatch:doCross};
$('run').onclick=()=>GO[mode()]();

/* ---------------- API calls ---------------- */
const frames=()=>{ // single still replicated for template stability
  const n=mode()==='pad'||$('as-live').checked?16:8;
  return Array.from({length:n},()=>PROBE);};
const padFlag=()=>$('as-live').checked; // ON => gate active (fail-closed for stills)
async function jsonPost(url,body){return(await fetch(url,{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify(body||{})})).json();}
async function enroll(){
  const id=$('identity').value.trim();
  if(!id)return null;
  $('result').innerHTML='<div class="meta">enrolling…</div>';
  const r=await jsonPost('/enroll',{identity_id:id,frames:frames()});
  verdict(r,'ENROLLED ✔ — '+id,r);
  refresh();
}
async function doVerify(){
  const id=$('identity').value.trim();
  if(!id||!PROBE)return null;
  $('result').innerHTML='<div class="meta">verifying '+id+'…</div>';
  const r=await jsonPost('/verify?identity_id='+encodeURIComponent(id)
    +'&pad='+padFlag(),{frames:frames()});
  verdict(r,r.is_match?'MATCH ✔ — '+id
    :(r.pad_live===false?'REJECTED (PAD) ✘':'NO MATCH — '+id),r);
}
async function doIdentify(){
  if(!PROBE)return null;
  $('result').innerHTML='<div class="meta">searching registry…</div>';
  const r=await jsonPost('/identify?pad='+padFlag(),{frames:frames()});
  verdict(r,(r.decision==='match')?'MATCH ✔ — '+(r.identity_id||(r.top_hit&&r.identity_id))
    :(r.decision==='rejected'?'REJECTED (PAD) ✘':'NO MATCH in registry'),r);
}
async function doPad(){
  if(!PROBE)return null;
  $('result').innerHTML='<div class="meta">running PAD engine…</div>';
  const r=await jsonPost('/pad',{frames:frames()});
  verdict(r,r.is_live?'LIVE ✔ — passed liveness gate':'NOT LIVE ✘ — spoof / no pulse',
    r);
}
async function doCross(){
  $('result').innerHTML='<div class="meta">N:N cross-match…</div>';
  const r=await fetch('/crossmatch?limit=64&threshold=0.9',{method:'POST'}).json();
  const dups=(r.duplicates||r.matches||[]).map(d=>d.a+' ≡ '+d.b).join('\n')
            ||'no duplicates found';
  $('result').innerHTML='<div class="verdict '+(dups.startsWith('no')?'v-none':'v-match')
    +'">'+(dups.startsWith('no')?'clean registry':'duplicate identities detected')
    +'</div><div class="out">'+r.n_identities+' ids · '+dups+'</div>';
}
function verdict(r,label,full){
  if(r.detail){$('result').innerHTML='<div class="verdict v-reject">error</div>'
    +'<div class="out">'+JSON.stringify(r,null,2)+'</div>';return;}
  const famous=full&&full.pad_live===undefined;
  const cls=label.includes('✔')?'v-match':(label.includes('✘')?'v-reject':'v-none');
  const extras=[];
  if(full&&full.tier_scores)Object.entries(full.tier_scores).forEach(([k,v])=>
    extras.push(k+': '+Number(v).toFixed(3)));
  if(full&&full.pad_live!==undefined)extras.push('pad: '
    +(full.pad_live===null?'bypassed':(full.pad_live?'live':'spoof')));
  if(full&&full.score!==undefined)extras.push('score: '+Number(full.score).toFixed(4));
  if(full&&(full.identity_id))extras.push('identity: '+full.identity_id);
  if(full&&full.elapsed_ms!==undefined)extras.push(full.elapsed_ms.toFixed(0)+' ms');
  let top='';
  if(full&&full.candidates&&full.candidates.length)
    top='\n'+full.candidates.slice(0,5).map(c=>c.identity_id
      +(c.tier2_score!==undefined&&c.tier2_score!==null?
        '  tier2 '+Number(c.tier2_score).toFixed(4):'')).join('\n');
  $('result').innerHTML='<div class="verdict '+cls+'">'+label+'</div>'
   +'<div class="out">'+extras.join(' · ')+top+'</div>';
}

/* ---------------- registry ---------------- */
async function refresh(){
  const j=await(await fetch('/identities')).json();
  const ids=j.identities||[];REG=ids;
  const tb=$('reg').tBodies[0];tb.innerHTML='';
  if(!ids.length)tb.innerHTML='<tr><td class="meta empty">registry empty</td></tr>';
  ids.forEach(i=>{const tr=document.createElement('tr');
    tr.innerHTML='<td>'+i+'</td><td style="text-align:right">'
      +'<button onclick="forget(\''+encodeURIComponent(i)+'\')">forget</button>'
      +'</td>';tb.appendChild(tr);});
  fillSelect();
}
let REG=[];
async function forget(id){
  await fetch('/identities/'+id,{method:'DELETE'});refresh();
}
function fillSelect(){
  if(mode()!=='verify')return;
  const s=$('identity');
  if(s.tagName!=='SELECT'||s[0]){
    const sel=document.createElement('select');sel.id='identity';
    s.replaceWith(sel);}
  const sel=$('identity'),prev=sel.value;
  sel.innerHTML=REG.map(i=>'<option>'+i+'</option>').join('');
  if(prev)sel.value=prev;
}
/* status pill */
async function pill(){
  try{const r=await(await fetch('/api/status')).json();
    $('status').textContent=r.index_backend+' · '+ (REG||[]).length
      +' enrolled · cuda '+(r.torch&&r.torch.cuda_available?'on':'off');
    $('status').className='pill ok';
  }catch(e){$('status').textContent='service unreachable';
    $('status').className='pill warn';}
}
refresh();pill();setInterval(pill,10000);
</script>
</body></html>
"""
