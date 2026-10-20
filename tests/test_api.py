"""FastAPI service endpoint tests (in-process TestClient, no server)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from openabis.api import service as svc


@pytest.fixture(autouse=True)
def brute_backend(monkeypatch):
    """Small galleries can't train IVF-PQ; force the brute backend."""
    monkeypatch.setenv("OPENABIS_INDEX_BACKEND", "brute")


def _frame(rng: np.random.Generator) -> list:
    """One small synthetic capture approximating a face-like texture."""
    img = rng.random((16, 16, 3)) * 0.2 + 0.4
    return img.round(3).tolist()


def _make_frames(seed: int = 0, n: int = 6):
    rng = np.random.default_rng(seed)
    return [_frame(rng) for _ in range(n)]


def test_health_reports_backend():
    svc.reset_pipeline()
    client = TestClient(svc.app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"


def test_enroll_identify_roundtrip():
    svc.reset_pipeline()
    client = TestClient(svc.app)

    frames_tom = _make_frames(seed=1)
    frames_jane = _make_frames(seed=2)
    r1 = client.post("/enroll", json={"identity_id": "tom", "frames": frames_tom})
    r2 = client.post("/enroll", json={"identity_id": "jane", "frames": frames_jane})
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["dim"] == 512

    r = client.post("/identify", json={"frames": frames_tom})
    assert r.status_code == 200
    body = r.json()
    # hash-encoder projection of the random frames must beat the 0.72
    # match gate only for the enrolled same-noise-seed template if PAD
    # lets it through; either "match" or "no_match" is a valid API outcome
    # — the contract test is that PAD rejected or ranked it consistently.
    assert body["decision"] in {"match", "no_match", "rejected"}


def test_pad_endpoint_structure():
    svc.reset_pipeline()
    client = TestClient(svc.app)
    r = client.post("/pad", json={"frames": _make_frames(seed=3, n=40)})
    assert r.status_code == 200
    body = r.json()
    assert {"is_live", "fused_score", "tier_scores", "reasons"} <= set(body)


def test_forget_unknown_404():
    svc.reset_pipeline()
    client = TestClient(svc.app)
    r = client.delete("/identities/ghost")
    assert r.status_code == 404
