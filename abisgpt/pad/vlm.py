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
            if arr.max() <= 1.5:                      # 0..1 float convention
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
    """

    def __init__(self, scale: float = 6.0, tau: float = 3.0):
        self.scale = scale
        self.tau = tau

    def score(self, image) -> VlmReport:
        import cv2
        import numpy as np

        gray = np.asarray(image, dtype=np.float32)
        if gray.ndim == 3:
            gray = gray.mean(axis=2)
        # Laplacian > threshold highlights cd/moire banding energy
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        attack_energy = float((np.abs(lap) > 4.0).mean())
        liveness = float(np.clip(1.0 - attack_energy * 6.0, 0.0, 1.0))
        return VlmReport(liveness_score=liveness, top_spoof_anchor="high-frequency artifact" if attack_energy > 0.05 else None, margin=float(-attack_energy))

    def __call__(self, image) -> float:
        """Fusion-contract adapter: callable(frame) -> float in [0, 1]."""
        return float(self.score(image).liveness_score)

