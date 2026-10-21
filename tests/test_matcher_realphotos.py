"""Real-biometric matching tests: two DIFFERENT photos, same person.

Contract under test (the user's reported failure): the matcher must decide
MATCH / rank-1 for a photo different from the one enrolled — not only for
the exact same byte image. Needs the ONNX toy encoder
(`artifacts/encoder/face_encoder.onnx`); pytest.skip otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from abisgpt.api import service as svc

ONNX = Path("artifacts/encoder/face_encoder.onnx")
OLIVETTI = Path("artifacts/dataset/olivetti.npz")


@pytest.mark.skipif(not ONNX.exists(), reason="onnx encoder not exported")
def test_verify_1v1_different_photo_same_person(monkeypatch):
    monkeypatch.setenv("ABISGPT_ENCODER", "onnx")
    monkeypatch.setenv("ABISGPT_INDEX_BACKEND", "brute")
    svc.reset_pipeline()
    c = TestClient(svc.app)

    z = np.load(OLIVETTI)
    frames, labels = z["frames"], z["labels"].astype(int)
    id0 = np.flatnonzero(labels == 3)          # one person, two frames
    f_a, f_b = frames[id0[0]], frames[id0[1]]  # photo A vs photo B
    f_other = frames[np.flatnonzero(labels == 7)[0]]

    # --- enroll photo A ---------------------------------------------------
    r = c.post("/enroll", json={"identity_id": "t_alice",
                                "frames": [f_a.round(4).tolist()]})
    assert r.status_code == 200, r.text

    # --- verify with a DIFFERENT photo of the SAME person ------------------
    r = c.post("/verify", params={"identity_id": "t_alice", "pad": "false"},
               json={"frames": [f_b.round(4).tolist()]})
    body = r.json()
    assert r.status_code == 200
    assert body["is_match"] is True, body
    assert body["score"] > 0.5, body

    # --- different person must NOT match -----------------------------------
    r = c.post("/verify", params={"identity_id": "t_alice", "pad": "false"},
               json={"frames": [f_other.round(4).tolist()]})
    body = r.json()
    assert body["is_match"] is False, body

    # cleanup
    c.request("DELETE", "/identities/t_alice")


@pytest.mark.skipif(not ONNX.exists(), reason="onnx encoder not exported")
def test_identify_1n_ranks_same_person_from_different_photo(monkeypatch):
    monkeypatch.setenv("ABISGPT_ENCODER", "onnx")
    monkeypatch.setenv("ABISGPT_INDEX_BACKEND", "brute")
    svc.reset_pipeline()
    c = TestClient(svc.app)

    z = np.load(OLIVETTI)
    frames, labels = z["frames"], z["labels"].astype(int)
    for subj, name in ((3, "t_amy"), (9, "t_beth")):
        f = frames[np.flatnonzero(labels == subj)[0]]
        r = c.post("/enroll", json={"identity_id": name,
                                    "frames": [f.round(4).tolist()]})
        assert r.status_code == 200, r.text

    unseen = frames[np.flatnonzero(labels == 9)[1]]  # beth, 2nd photo
    r = c.post("/identify", params={"pad": "false"},
               json={"frames": [unseen.round(4).tolist()], "top_k": 2})
    body = r.json()
    assert body["decision"] == "match", body
    assert body["candidates"][0]["identity_id"] == "t_beth", body

    for nm in ("t_amy", "t_beth"):
        c.request("DELETE", f"/identities/{nm}")
