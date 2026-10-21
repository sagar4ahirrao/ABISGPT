"""Passive PAD unit tests (rPPG + flash tiers, §2.2 / §2.3)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abisgpt.pad.flashdiff import FlashDiffusionAnalyzer
from abisgpt.pad.rppg import LIVE_SNR_DB_MIN, RppgAnalyzer

FPS = 30.0


def _live_frames(seed: int = 1, count: int = 96, bpm: float = 72.0, base: int = 120) -> list[np.ndarray]:
    """Synthetic pulsating skin — green channel carries BVP at `bpm`."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(count):
        t = i / FPS
        pulse = (bpm / 60.0) * 2.4 * np.sin(2 * np.pi * (bpm / 60.0) * t)
        f = np.full((64, 64, 3), base, dtype=np.float32)
        f[..., 1] += pulse + rng.normal(0, 1.0)          # green carries BVP (§2.2)
        f[..., 0] += 0.3 * pulse                          # red follows faintly
        f[..., 2] += 0.1 * pulse
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return frames


def test_rppg_detects_live_pulse() -> None:
    analyzer = RppgAnalyzer(fps=FPS)
    result = analyzer.from_frames(_live_frames(bpm=72.0))
    assert result.is_live, f"expected LIVE, got {result!r}"
    assert result.bpm is not None and 60 <= result.bpm <= 90


def test_rppg_rejects_static_spoof() -> None:
    analyzer = RppgAnalyzer(fps=FPS)
    rng = np.random.default_rng(2)
    frames = [np.full((64, 64, 3), 120, dtype=np.uint8) +
              rng.normal(0, 2, (64, 64, 3)).astype(np.int16).astype(np.uint8)
              for _ in range(96)]
    result = analyzer.from_frames(frames)
    assert not result.is_live
    assert result.snr_db <= LIVE_SNR_DB_MIN


def test_rppg_rejects_moving_fake_pulse_outside_band() -> None:
    """BVP at 6 Hz is outside [0.7, 4.0] Hz plausible human band → rejected."""
    analyzer = RppgAnalyzer(fps=FPS)
    rng = np.random.default_rng(3)
    frames = []
    for i in range(96):
        t = i / FPS
        f = np.full((64, 64, 3), 120, dtype=np.float32)
        f[..., 1] += 4.0 * np.sin(2 * np.pi * 6.0 * t) + rng.normal(0, 1.0)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    assert not analyzer.from_frames(frames).is_live


def test_flashdiff_finds_high_laplacian_for_replays() -> None:
    """Flat replay frames yield near-zero differential → 2D replay signal."""
    frame = np.full((64, 64, 3), 120, dtype=np.uint8)
    pairs = [frame, frame.copy()]
    report = FlashDiffusionAnalyzer()(pairs)
    assert report.depth_flatness < 0.5   # flat: no gradient variation
    assert report.specular_ratio < 0.02


def test_pad_fusion_scores_live_and_spoof_frames() -> None:
    from abisgpt.config import PadConfig
    from abisgpt.pad import PassivePadEngine

    engine = PassivePadEngine(PadConfig(fps=int(FPS)))
    live = engine.evaluate(_live_frames(bpm=72.0))
    assert live.is_live

    rng = np.random.default_rng(4)
    dead_frames = [np.full((64, 64, 3), 120, dtype=np.uint8) +
                   rng.normal(0, 2, (64, 64, 3)).astype(np.int16).astype(np.uint8)
                   for _ in range(96)]
    dead = engine.evaluate(dead_frames)
    assert not dead.is_live


def test_pad_single_frame_passes_without_faking_evidence() -> None:
    """A truly single-frame still defeats BOTH temporal tiers (rPPG needs a
    sequence, flash needs >=2 frames). Liveness must be UNPROVEN — the gate
    passes (biometric matching decides), fused_score is honestly None, and
    the reason strings say so. Never assert 'spoof' from absent evidence."""
    from abisgpt.config import PadConfig
    from abisgpt.pad import PassivePadEngine

    engine = PassivePadEngine(PadConfig(fps=int(FPS)))
    one = np.full((32, 32, 3), 120, dtype=np.uint8)
    verdict = engine.evaluate([one])
    assert verdict.is_live is True
    assert verdict.fused_score is None
    assert verdict.tier_scores == {}
    assert any("still mode" in r for r in verdict.reasons)
    assert any("unproven" in r for r in verdict.reasons)

    # multi-frame replica stills keep the measurable flash tier -> real
    # fused score with the honest still-mode reason.
    frames = [one] * 16
    v2 = engine.evaluate(frames)
    assert v2.is_live is True
    assert v2.fused_score is not None
    assert "flash" in v2.tier_scores
    assert any("still mode" in r for r in v2.reasons)

