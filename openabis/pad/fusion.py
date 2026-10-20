"""Fusion of the three PAD tiers (VLM / rPPG / flash) into one decision."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from openabis.config import PadConfig


@dataclass
class PadVerdict:
    is_live: bool
    fused_score: float
    tier_scores: dict[str, float]
    reasons: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


# VLM ternary verifier callback contract:
#   callable(frame: np.ndarray) -> float in [0, 1]  (1 = clean human face)
VlmScorer = Callable[[np.ndarray], float]


class PassivePadEngine:
    """Weighted fusion of passive liveness tiers.

    Missing tiers are re-normalized so the fused score stays well-run
    on edge hardware (e.g., kiosks with no flash ring).
    """

    def __init__(self, cfg: PadConfig | None = None, vlm_scorer: VlmScorer | None = None):
        self.cfg = cfg or PadConfig()
        self.vlm_scorer = vlm_scorer

    # ------------------------------------------------------------------ #
    def evaluate(self, frames: Sequence[np.ndarray]) -> PadVerdict:
        """Fuse `rPPG + flash (+ optional VLM)` over incoming frames."""
        reasons: list[str] = []
        tier_scores: dict[str, float] = {}

        # ---- Tier 1: rPPG ------------------------------------------------ #
        from openabis.pad.rppg import RppgAnalyzer

        rppg = RppgAnalyzer(
            fps=float(self.cfg.fps),
            band_hz=tuple(self.cfg.deepfake_band),
            min_frames=max(8, int(self.cfg.fps * 0.4)),
        )
        result = rppg.from_frames(frames)
        # Margin-based mapping: noise floor (≈0 dB) → 0, strong pulse (~20 dB) → 1
        rppg_signal = float(np.clip((result.snr_db - 5.0) / 15.0, 0.0, 1.0))
        tier_scores["rppg"] = rppg_signal
        if not result.is_live:
            reasons.append(
                f"rPPG rejected: snr={result.snr_db:.2f} dB"
                + (f", bpm={result.bpm:.1f}" if result.bpm else "")
            )

        # ---- Tier 2: flash surface diffusion ------------------------------ #
        try:
            from openabis.pad.flashdiff import FlashDiffusionAnalyzer

            flash = FlashDiffusionAnalyzer(fps=float(self.cfg.fps))
            report = flash(frames)
            tier_scores["flash"] = report.to_signal()
            if report.specular_ratio > 0.35:
                reasons.append(f"flash: specular glare ratio={report.specular_ratio:.2f}")
        except ValueError:
            reasons.append("flash: skipped (insufficient frames)")

        # ---- Tier 3: VLM semantic guardrails (optional plug-in) ----------- #
        if self.vlm_scorer is not None and len(frames):
            reason_score = float(np.clip(self.vlm_scorer(frames[-1]), 0.0, 1.0))
            tier_scores["vlm"] = reason_score
            if reason_score < 0.5:
                reasons.append(f"VLM spoof attribute score={reason_score:.2f}")

        if "flash" not in tier_scores:
            reasons.append("PAD fusion degraded: flash tier absent")

        fused = self._weighted_fusion(tier_scores)
        is_live = fused >= self.cfg.gate_threshold
        if is_live:
            reasons.append("PASSIVE_PAD_PASS")
        return PadVerdict(
            is_live=is_live,
            fused_score=fused,
            tier_scores=tier_scores,
            reasons=reasons,
        )

    __call__ = evaluate

    def _weighted_fusion(self, scores: dict[str, float]) -> float:
        available = {k: v for k, v in scores.items() if v is not None}
        weight_map = {
            "vlm": self.cfg.vlm_weight,
            "rppg": self.cfg.rppg_weight,
            "flash": self.cfg.flash_weight,
        }
        total_w = sum(weight_map[k] for k in available)
        if total_w <= 0:
            return 0.0
        return float(
            sum(weight_map[k] * available[k] for k in available) / total_w
        )
