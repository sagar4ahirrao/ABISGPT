"""VLM semantic guardrail tests — mock + heuristic determinism (no weights)."""

from __future__ import annotations

import numpy as np

from abisgpt.pad.fusion import PassivePadEngine
from abisgpt.pad.vlm import HeuristicVlmPad, VlmReport


def test_heuristic_scores_clean_capture_high():
    # canonical-resample calibration holds at any input resolution: the
    # console feeds fs native camera stills, and the tier resamples to 32x32
    # before scoring (same verdict at 32, 64, 128 px).
    for size in (32, 64, 128):
        rng = np.random.default_rng(0)
        img = (rng.random((size, size, 3)) * 0.15 + 0.5).astype(np.float32)
        rep = HeuristicVlmPad().score(img)
        assert isinstance(rep, VlmReport)
        assert rep.liveness_score >= 0.9, size


def test_heuristic_scores_moire_low():
    # console-canonical 32x32 screen banding (period ~3.9 px, measured
    # attack_energy ~= 0.995 vs real captures <= 0.977 at 32 px).
    yy, xx = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    moire = (np.sin(xx * 1.6) * np.sin(yy * 1.6) > 0).astype(np.float32) * 255
    rep = HeuristicVlmPad().score(moire)
    assert rep.liveness_score < 0.5
    # convention invariance: the SAME capture in the 0..1 float convention
    # (what the console posts) must score identically to the uint8 form.
    rep_f = HeuristicVlmPad().score(moire / 255.0)
    assert abs(rep_f.liveness_score - rep.liveness_score) < 1e-6


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

