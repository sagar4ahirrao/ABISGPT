"""Flash-induced surface diffusion analyzer (white paper §2.3).

Key idea: human skin is a *subsurface scatterer* — light diffuses spherically
through tissue. Paper, latex, and LCD screens scatter either specularly or
linearly, producing distinct high-frequency residuals in the frame when a
controlled illumination flash fires.

Reference implementation computes three cheap, deterministic features:

1. `specular_ratio` — fraction of pixels with very high intensity after
   flash, typical of glass / screen glare.
2. `diffuse_energy` — spatial energy (Laplacian) of the flash differential
   map; uniform for skin, sharp for edges of printed/photo spoofs.
3. `depth_flatness` — variance across the differential-image gradient; real
   3D surfaces have mild variation, 2D replays are flat.

Production target in the white paper is a small transformer over the
differential map; this classical tier yields a stable baseline for a fused
`scores` dict consumed by `openabis.pad.fusion.PassivePadEngine`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class FlashReport:
    specular_ratio: float
    diffuse_energy: float
    depth_flatness: float

    def to_signal(self) -> float:
        """Map deterministic features to a [0,1] liveness score.

        Heuristic calibration:
        * high specular_ratio reduces score (screen/glass glare);
        * low diffuse_energy reduces score (flat 2D surface);
        * high depth_flatness reduces score (flat receipt).
        """
        s = 1.0
        s -= float(np.clip(self.specular_ratio * 2.2, 0.0, 0.45))
        s -= float(np.clip((0.45 - np.log1p(self.diffuse_energy)) / 2.0, 0.0, 0.4))
        s -= float(np.clip(self.depth_flatness * 1.2, 0.0, 0.35))
        return float(np.clip(s, 0.0, 1.0))

    def as_dict(self) -> dict:
        return asdict(self)


class FlashDiffusionAnalyzer:
    """Differential flash response analyzer.

    Expects frames captured with an off/on (or low/high) illumination ring.
    Approximates the differential map as consecutive-frame differences;
    a hardware ring fires frames 0..1 apart in production so the same
    estimator applies unchanged.
    """

    def __init__(self, fps: float = 60.0, window: int = 2):
        self.fps = fps
        self.window = max(2, window)

    @staticmethod
    def _differential_map(pair: Sequence[np.ndarray]) -> np.ndarray:
        a = cv2.cvtColor(np.asarray(pair[0]), cv2.COLOR_RGB2GRAY).astype(np.float32)
        b = cv2.cvtColor(np.asarray(pair[1]), cv2.COLOR_RGB2GRAY).astype(np.float32)
        return np.abs(b - a)

    @staticmethod
    def _laplacian_energy(map_local: np.ndarray) -> float:
        lap = cv2.Laplacian(map_local, cv2.CV_32F, ksize=3)
        return float(np.sqrt((lap ** 2).mean()))

    @staticmethod
    def _gradient_flatness(map_local: np.ndarray) -> float:
        gx = cv2.Sobel(map_local, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(map_local, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
        # flat surfaces give a small coefficient of variation
        return float(grad.std() / (grad.mean() + 1e-8))

    from_pair = _differential_map

    def evaluate(self, frames: Sequence[np.ndarray]) -> FlashReport:
        if len(frames) < self.window:
            raise ValueError(f"FlashDiffusionAnalyzer requires >= {self.window} frames.")
        diff = self._differential_map(frames[: self.window])
        bright = float((diff > 200).mean())  # almost-white pixels = specular blown highlights
        energy = self._laplacian_energy(diff)
        flatness = self._gradient_flatness(diff)
        report = FlashReport(
            specular_ratio=bright,
            diffuse_energy=energy,
            depth_flatness=flatness,
        )
        return report

    __call__ = evaluate
