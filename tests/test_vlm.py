"""VLM semantic guardrail tests — mock + heuristic determinism (no weights)."""

from __future__ import annotations

import numpy as np

from openabis.pad.fusion import PassivePadEngine
from openabis.pad.vlm import HeuristicVlmPad, VlmReport


def test_heuristic_scores_clean_capture_high():
    rng = np.random.default_rng(0)
    img = (rng.random((64, 64, 3)) * 0.15 + 0.5).astype(np.float32)
    rep = HeuristicVlmPad().score(img)
    assert isinstance(rep, VlmReport)
    assert rep.liveness_score >= 0.9


def test_heuristic_scores_moire_low():
    yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    moire = (np.sin(xx * 0.9) * np.sin(yy * 0.9) > 0).astype(np.float32) * 255
    rep = HeuristicVlmPad().score(moire)
    assert rep.liveness_score < 0.5


def test_call_adapter_fuses_into_pad():
    """The fusion tier receives the *float* adapter, not the VlmReport."""
    rng = np.random.default_rng(1)
    frames = [(rng.random((32, 32, 3)) * 0.1 + 0.45).astype(np.float32)] * 50
    engine = PassivePadEngine(vlm_scorer=HeuristicVlmPad())
    verdict = engine.evaluate(frames)
    assert "vlm" in verdict.tier_scores
    assert 0.0 <= verdict.tier_scores["vlm"] <= 1.0


def test_vlm_rejection_reason_recorded():
    class AlwaysSpoof:
        def __call__(self, image):
            return 0.05

    rng = np.random.default_rng(2)
    frames = [(rng.random((32, 32, 3)) * 0.1 + 0.45).astype(np.float32)] * 50
    engine = PassivePadEngine(vlm_scorer=AlwaysSpoof())
    verdict = engine.evaluate(frames)
    assert any("VLM" in r for r in verdict.reasons)
