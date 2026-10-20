"""Temporal rPPG (remote photoplethysmography) liveness analyzer.

Implements white paper §2.2 with a heavy-duty **signal-processing tier**:

    frames -> [skin ROI mask] -> [green-channel BVP trace]
            -> [bandpass 0.7–4.0 Hz] -> [FFT] -> pulse-BPM & liveness score

The production model in the white paper swaps the FFT topogram for a
temporal ViT; this module keeps a deterministic classical reference that
works with zero dependencies so the full pipeline stays reproducible
in CI. The beat band can also be re-tuned with the registration API
`configure_band()`.

Not a medical device — no regulatory claim, and the decision function is a
liveness signal *gate*, not a clinical pulse oximeter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RppgResult:
    """Single probe liveness report."""

    is_live: bool
    bpm: float | None          # estimated pulse rate (None if too weak)
    snr_db: float              # signal-to-noise ratio of cardiac-band FFT peak
    trace_length: int
    fps: float

    def __repr__(self) -> str:  # pragma: no cover — cosmetics only
        return (
            f"RppgResult(is_live={self.is_live}, bpm={self.bpm}, "
            f"snr_db={self.snr_db:.2f}, frames={self.trace_length}, fps={self.fps})"
        )


# cardiac-band peaks below this SNR (dB) are treated as noise floor
LIVE_SNR_DB_MIN = 10.0

# skin pixel mask in YCrCb space (classic CHROM-compatible mask)
_YCC_LOW = np.array([0, 133, 77], dtype=np.uint8)
_YCC_HIGH = np.array([255, 180, 127], dtype=np.uint8)

# cardiac-band bounds (Hz). 0.7–4.0 covers 42–240 beats / min.
DEFAULT_BAND_HZ = (0.7, 4.0)


class RppgAnalyzer:
    """Remote-PPG liveness analyzer on ROI-skin BVP traces."""

    def __init__(
        self,
        fps: float = 60.0,
        band_hz: tuple = DEFAULT_BAND_HZ,
        min_frames: int = 24,
    ):
        self.fps = fps
        self.band_hz = band_hz
        self.min_frames = max(min_frames, 8)

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    @staticmethod
    def skin_mask(frame_bgr_or_rgb: np.ndarray) -> np.ndarray:
        """Boolean mask of plausibly-skin pixels using YCrCb thresholds."""
        ycc = cv2.cvtColor(frame_bgr_or_rgb, cv2.COLOR_RGB2YCrCb)
        return cv2.inRange(ycc, _YCC_LOW, _YCC_HIGH).astype(bool)

    @staticmethod
    def green_bvp_trace(frames: Sequence[np.ndarray]) -> np.ndarray:
        """Mean green channel inside ROI skin across `frames` -> (T,) trace."""
        trace = np.empty(len(frames), dtype=np.float32)
        for i, f in enumerate(frames):
            mask = RppgAnalyzer.skin_mask(f)
            if mask.sum() < 32:  # fallback: whole-frame mean keeps pipeline alive
                mask = np.ones(f.shape[:2], dtype=bool)
            trace[i] = f[..., 1][mask].mean()
        return trace

    @staticmethod
    def bandpass(trace: np.ndarray, fps: float, band: tuple) -> np.ndarray:
        """Butterworth-free FFT-domain band filter ( Chop-less, deterministic).

        Keeps only bins within `band` (Hz) after removing DC & linear trend —
        mathematically equivalent to an ideal brick-wall (sturdier than
        IIR for tiny windows).
        """
        x = np.asarray(trace, dtype=np.float32)
        x = x - x.mean()
        # detrend (linear) to suppress slow shifts (background, cameras)
        if len(x) > 2:
            t = np.arange(len(x), dtype=np.float32) / fps
            poly = np.polyfit(t, x, 1)
            x = x - np.polyval(poly, t)
        spectrum = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), d=1.0 / fps)
        keep = (freqs >= band[0]) & (freqs <= band[1])
        spectrum[~keep] = 0
        return np.fft.irfft(spectrum, n=len(x))

    @staticmethod
    def pulse_power_snr(
        trace: np.ndarray, fps: float, band: tuple = DEFAULT_BAND_HZ, neighborhood_hz: float = 0.2
    ) -> tuple:
        """Peak-SNR in dB, plus peak frequency.

        SNR = peak power / average neighbouring-bin power (both in-band).
        The neighbour window is clamped to at least ±1 FFT bin so flat
        (white-noise) spectra cannot inflate SNR when `neighborhood_hz`
        is finer than the bin spacing.
        """
        spec = np.fft.rfft(trace - trace.mean())
        freqs = np.fft.rfftfreq(len(trace), d=1.0 / fps)
        df = fps / max(len(trace), 1)                      # bin spacing (Hz)
        span = max(neighborhood_hz, 2.0 * df)              # ≥ 1 neighbour bin/side
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        band_freqs = freqs[band_mask]
        band_power = np.abs(spec[band_mask]) ** 2
        if not band_power.size:
            return -np.inf, None
        peak_idx = int(np.argmax(band_power))
        peak_power = float(band_power[peak_idx])
        near = np.abs(band_freqs - band_freqs[peak_idx]) <= span
        neigh_power = float(band_power[near].sum() - peak_power)
        neigh_bins = max(int(near.sum()) - 1, 1)
        noise_density = neigh_power / neigh_bins
        snr_db = 10.0 * np.log10((peak_power + 1e-12) / (noise_density + 1e-12))
        return float(snr_db), float(band_freqs[peak_idx])

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def from_frames(self, frames: Sequence[np.ndarray]) -> RppgResult:
        """Analyze a raw RGB sequence; frames must be consecutive in time.

        Frames are `(T, H, W, 3)`-shaped ReadRGB (uint8 preferred).
        """
        if len(frames) < self.min_frames:
            return RppgResult(False, None, -np.inf, len(frames), self.fps)
        trace = self.green_bvp_trace(frames)
        if np.std(trace) < 1e-7:  # constant frames -> rejected outright
            return RppgResult(False, None, -np.inf, len(frames), self.fps)
        filtered = self.bandpass(trace, self.fps, self.band_hz)
        snr, peak_hz = self.pulse_power_snr(filtered, self.fps, self.band_hz)
        bpm = peak_hz * 60.0 if peak_hz else None
        in_band = peak_hz and self.band_hz[0] <= peak_hz <= self.band_hz[1]
        is_live = bool(in_band and snr >= LIVE_SNR_DB_MIN)
        return RppgResult(is_live, bpm, float(snr), len(frames), self.fps)

    __call__ = from_frames

    def from_video_path(self, path: str, max_frames: int = 512) -> RppgResult:
        """Read an MP4/AVI file then run `from_frames` (offline evaluation)."""
        frames: list[np.ndarray] = []
        cap = cv2.VideoCapture(path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
            self.fps = fps if fps and fps > 1 else self.fps
            while cap.isOpened() and len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(np.flipud(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        finally:
            cap.release()
        return self.from_frames(frames)

    def configure_band(self, low_hz: float, high_hz: float) -> None:
        if not 0 < low_hz < high_hz:
            raise ValueError("band_hz must satisfy 0 < low < high")
        self.band_hz = (low_hz, high_hz)

