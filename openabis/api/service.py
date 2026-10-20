"""OpenABIS FastAPI service — enroll / identify / PAD gate over HTTP.

Run locally:
    uvicorn openabis.api.service:app --host 0.0.0.0 --port 8000
or via Docker Compose (`docker compose up` from the repo root).

Environment:
    OPENABIS_CONFIG          path to a YAML config (configs/model.yaml)
    OPENABIS_INDEX_BACKEND   brute | faiss | qdrant   (overrides config)
    OPENABIS_INDEX_URI       endpoint for qdrant/milvus backends
    OPENABIS_VLM             unset | heuristic | clip   (VLM PAD tier)

Frames arrive as JSON arrays of HxW RGB float images (0..1). Keep them small
in this reference service; production should stream binary frames.
"""

from __future__ import annotations

import os

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    import openabis
    VERSION = openabis.__version__
except Exception:                                     # pragma: no cover
    VERSION = "0.1.0"

from openabis.config import OpenAbisConfig
from openabis.encoder import ViTBiometricEncoder
from openabis.index import build_backend
from openabis.pad import PassivePadEngine
from openabis.pipeline.triage import TwoTierPipeline

app = FastAPI(title="OpenABIS", version=VERSION,
              description="Open, reproducible biometric identification research stack")


# --------------------------------------------------------------------- #
# Pipeline singleton                                                     #
# --------------------------------------------------------------------- #
_PIPE: TwoTierPipeline | None = None


def _make_vlm_scorer():
    """Build the optional VLM tier from the OPENABIS_VLM env var."""
    choice = os.environ.get("OPENABIS_VLM", "").strip().lower()
    if choice == "heuristic":
        from openabis.pad.vlm import HeuristicVlmPad

        return HeuristicVlmPad()
    if choice == "clip":
        from openabis.pad.vlm import ClipVlmPad

        return ClipVlmPad()
    return None


def get_pipeline() -> TwoTierPipeline:
    global _PIPE
    if _PIPE is None:
        cfg_path = os.environ.get("OPENABIS_CONFIG")
        cfg = OpenAbisConfig.from_yaml(cfg_path) if cfg_path else OpenAbisConfig()

        backend = os.environ.get("OPENABIS_INDEX_BACKEND", "").strip().lower()
        if backend:
            cfg.index.backend = backend            # type: ignore[assignment]
        uri = os.environ.get("OPENABIS_INDEX_URI", "").strip()
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


@app.delete("/identities/{identity_id}")
def forget(identity_id: str):
    removed = get_pipeline().drop(identity_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"{identity_id!r} not enrolled")
    return {"identity_id": identity_id, "removed": removed}
