"""VLM semantic attribute guardrails (white paper §2.1).

Two scorers share one contract:  `score(image) -> float in [0, 1]`
(1 = clean human capture; < 0.5 = spoof signal).

* `ClipVlmPad` — real zero-shot VLM via transformers/CLIP. Interrogates the
  frame against artifact anchor phrases (moiré glare, silicone edge, LCD
  pixel texture…) and live-skin anchors, exactly mirroring Eq. §2.1:
      spoof = sigmoid(k * ( mean(live anchors) - max(spoof anchors) ))
* `HeuristicVlmPad` — dependency-free fallback (texture-statistics proxy)
  for CI/CI-restricted environments; clearly NOT a VLM.

Pass either into `abisgpt.pad.PassivePadEngine(vlm_scorer=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass

SPOOF_ANCHORS = (
    "moire pattern glare",
    "silicone mask edge boundary",
    "screen pixel texture of a phone display",
    "printed paper photo of a face",
    "digital screen replay of a video",
    "flat hologram-like rainbow banding",
)
LIVE_ANCHORS = (
    "natural human skin texture",
    "genuine photograph of a real human face",
    "candid unedited photo of a person",
    "authentic selfie taken outdoors",
)


@dataclass
class VlmReport:
    liveness_score: float
    top_spoof_anchor: str | None
    margin: float


class ClipVlmPad:
    """Zero-shot CLIP-family VLM scorer (lazy weights)."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self._PILImage = Image
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        prompts = list(SPOOF_ANCHORS) + list(LIVE_ANCHORS)
        with torch.no_grad():
            text = self.processor(text=prompts, return_tensors="pt", padding=True)
            text = {k: v.to(self.device) for k, v in text.items()}
            emb = self.model.get_text_features(**text)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        self._spoof_txt = emb[: len(SPOOF_ANCHORS)]
        self._live_txt = emb[len(SPOOF_ANCHORS):]

    def score(self, image) -> VlmReport:
        torch = self._torch
        import numpy as np

        arr = np.asarray(image)
        if arr.dtype.kind == "f":                     # float inputs: gray-levels
            arr = np.clip(arr, 0.0, 255.0)
            # only TRUE 0..1 frames are upscaled (non-negative, peak <= 1.5);
            # out-of-range synthetic captures stay at native scale.
            if float(arr.min()) >= -1e-6 and float(arr.max()) <= 1.5:
                arr = arr * 255.0
            arr = arr.astype(np.uint8)
        img = self._PILImage.fromarray(arr).convert("RGB")
        with torch.no_grad():
            pixel = self.processor(images=img, return_tensors="pt")
            pixel = {k: v.to(self.device) for k, v in pixel.items()}
            vis = self.model.get_image_features(**pixel)
            vis = vis / vis.norm(dim=-1, keepdim=True)
            spoof_sim = (vis @ self._spoof_txt.T).max().item()
            live_sim = (vis @ self._live_txt.T).mean().item()
        margin = live_sim - spoof_sim
        return VlmReport(liveness_score=float(torch.sigmoid(torch.tensor(6.0 * margin))), top_spoof_anchor=None, margin=float(margin))

    def __call__(self, image) -> float:
        """Fusion-contract adapter: callable(frame) -> float in [0, 1]."""
        return float(self.score(image).liveness_score)


class HeuristicVlmPad:
    """Texture-statistics stand-in (no VLM weights needed).

    Mills high-frequency artifact energy (moire/banding) against a live
    margin. Actually a crude LCD/blob detector — production must use CLIP.

    Calibration (measured at the console's canonical 32x32, 0..255 scale):
      real captures        attack_energy 0.813 – 0.977
        (Olivetti + CASIA-FASD faces <= 0.960, fingerprints <= 0.958,
         iris <= 0.977)
      synthetic screen texture / noise / moire    >= 0.993
    Frames are FIRST normalized from the 0..1 float convention (console
    posts float, bench may post uint8) so the tier is scale-consistent;
    then scores outside the [LO, HI] band map linearly to [0, 1] with
    captures at/below LO fully live and saturated screen textures fully
    dead.
    """

    # measured bounds (see docstring); resolution-sensitive by design —
    # recalibrate if the console frame size changes.
    LO = 0.975   # <= this fraction of pixels with |lap|>4  -> fully live
    HI = 0.995   # >= this fraction (saturated screen texture) -> fully dead

    def __init__(self, scale: float = 6.0, tau: float = 3.0,
                 lo: float | None = None, hi: float | None = None):
        self.scale = scale
        self.tau = tau
        self.lo = self.LO if lo is None else lo
        self.hi = self.HI if hi is None else hi

    def score(self, image) -> VlmReport:
        import cv2
        import numpy as np

        gray = np.asarray(image, dtype=np.float32)
        if gray.ndim == 3:
            gray = gray.mean(axis=2)
        # Canonical 0..255 scale: the tau/LO/HI thresholds are calibrated
        # there. Only TRUE 0..1 frames (non-negative peak <= 1.5) are
        # upscaled — frames carrying out-of-range values (e.g. synthetic
        # captures with an absolute pulse amplitude) stay at native scale,
        # otherwise the ×255 would blow non-image noise into fake banding.
        peak = float(gray.max()) if gray.size else 0.0
        if peak <= 1.5 and float(gray.min()) >= -1e-6:
            gray = gray * 255.0
        # Canonical SIZE: Laplacian pixel-density grows with resolution, so
        # resample to the console's canonical 32x32 first — the tier must
        # give the same verdict for the same capture at any input size.
        if gray.shape != (32, 32):
            gray = cv2.resize(gray, (32, 32),
                              interpolation=cv2.INTER_AREA)
        # Laplacian > threshold highlights cd/moire banding energy
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        attack_energy = float((np.abs(lap) > 4.0).mean())
        if attack_energy <= self.lo:
            liveness = 1.0
        elif attack_energy >= self.hi:
            liveness = 0.0
        else:
            liveness = float(1.0 - (attack_energy - self.lo)
                             / max(self.hi - self.lo, 1e-9))
        return VlmReport(liveness_score=liveness, top_spoof_anchor="high-frequency artifact" if attack_energy > self.hi else None, margin=float(-attack_energy))

    def __call__(self, image) -> float:
        """Fusion-contract adapter: callable(frame) -> float in [0, 1]."""
        return float(self.score(image).liveness_score)

