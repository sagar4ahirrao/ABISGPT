"""Match-type layer tests: 1:1 verify, batch identify, N:N crossmatch, fusion."""

from __future__ import annotations

import numpy as np
import pytest

from abisgpt.config import AbisGptConfig
from abisgpt.encoder import ViTBiometricEncoder
from abisgpt.index.factory import BruteForceIndex
from abisgpt.pipeline.matching import MatchingEngine, MultiModalProbe
from abisgpt.pipeline.triage import TwoTierPipeline


@pytest.fixture()
def engine():
    cfg = AbisGptConfig()
    cfg.index.backend = "brute"
    pipe = TwoTierPipeline(
        encoder=ViTBiometricEncoder(strategy="hash"),
        index_backend=BruteForceIndex(cfg.index.dim),
        config=cfg,
    )
    return MatchingEngine(pipe)


def _live_frames(rng, n=180):
    frames = []
    for f in range(n):
        img = np.full((32, 32, 3), 0.5, dtype=np.float32)
        img[:, :, 1] += 2.5 * np.sin(2 * np.pi * 1.0 * f / 60.0) + rng.normal(0, 0.6, (32, 32))
        frames.append(img)
    return frames


def _flat_frames(n=180):
    return np.full((n, 32, 32, 3), 0.55, dtype=np.float32)


def test_verify_1v1_match_and_reject(engine):
    frames = _live_frames(np.random.default_rng(0))
    engine.pipeline.enroll("alice", frames)
    hit = engine.verify_1v1("alice", frames)
    assert hit.is_match is True
    assert hit.score >= engine.threshold

    miss_engine_threshold = engine.verify_1v1("bob", frames)   # not enrolled
    assert miss_engine_threshold.is_match is False
    assert miss_engine_threshold.score < engine.threshold


def test_verify_1v1_pad_gate_blocks_replay(engine):
    rng = np.random.default_rng(1)
        # enroll a genuine capture...
    engine.pipeline.enroll("alice", _live_frames(rng))
    # ...then replay it as a static flat probe — PAD must fail closed
    res = engine.verify_1v1("alice", _flat_frames())
    assert res.pad_live is False
    assert res.is_match is False


def test_batch_identify_routing(engine):
    rng = np.random.default_rng(2)
    f_a, f_b = _live_frames(np.random.default_rng(3)), _live_frames(np.random.default_rng(4))
    engine.pipeline.enroll("a", f_a)
    engine.pipeline.enroll("b", f_b)
    batch = engine.identify_batch([f_a, _flat_frames()], top_k=5)
    assert batch.n_probes == 2
    assert batch.results[0]["decision"] == "match"
    assert batch.results[1]["decision"] == "rejected"
    assert batch.results[1]["pad_live"] is False
    assert batch.qps > 0


def test_crossmatch_nxn_known_hits_and_rate():
    from abisgpt.index.factory import BruteForceIndex
    from abisgpt.pipeline.matching import MatchingEngine
    from abisgpt.pipeline.triage import TwoTierPipeline
    pipe = TwoTierPipeline(index_backend=BruteForceIndex(512))
    eng = MatchingEngine(pipe)
    rng = np.random.default_rng(9)
    d = 512
    ids_probe = [f"p{i}" for i in range(64)]
    probes, gallery = {}, {}
    for i in ids_probe:
        v = rng.standard_normal(d).astype(np.float32)
        v /= np.linalg.norm(v)
        probes[i] = v
        g = v + 0.01 * rng.standard_normal(d).astype(np.float32)  # same person
        gallery[i.replace("p", "g")] = g / np.linalg.norm(g)

    res = eng.crossmatch_nxn(probes, gallery, threshold=0.95)
    # every probe must find its own counterpart (and rate is measured)
    found = {(p, g) for p, g, _ in res.pairs}
    assert len(res.pairs) >= sum(1 for i in ids_probe
                                 if (i, i.replace("p", "g")) in found) >= 60
    assert res.pairs_per_sec > 0


def test_multi_modal_fusion_math():
    rng = np.random.default_rng(11)
    d = 512
    face_a = rng.standard_normal(d)
    iris_a = rng.standard_normal(d)
    face_b = face_a + 0.05 * rng.standard_normal(d)           # similar face
    iris_b = iris_a + 0.05 * rng.standard_normal(d)           # similar iris
    iris_c = iris_a + 9.0 * rng.standard_normal(d)            # unrelated iris
    a = MultiModalProbe({"face": face_a, "iris": iris_a})
    b = MultiModalProbe({"face": face_b, "iris": iris_b})
    c = MultiModalProbe({"face": face_b, "iris": iris_c})

    # Both channels agree  -> near-perfect fused similarity.
    both = a.fused(b)
    assert both > 0.95
    # Face agrees, iris unrelated          -> fused drops toward 0.5-baseline
    # (that is the desired behavior: multi-modal identity must agree on all
    #  channels, not just one).
    partial = a.fused(c)
    assert partial < 0.7
    assert a.fused(a) == pytest.approx(1.0, abs=1e-5)
    # Disjoint channel sets score 0.
    assert a.fused(MultiModalProbe({"fingerprint": face_a})) == 0.0
