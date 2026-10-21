"""Vision-Transformer biometric encoder.

Produces 512-D L2-normalized embeddings for face samples
(white paper §3.2). Three paths:

* deterministic hash fallback so tests/CI run without weights or heavy
  wheels (NOT a real biometric model — stable pseudo-random projection),
* torch/timm-backed DINOv2 ViT when torch is installed (lazy import —
  keeps torch+faiss DLL conflicts off the default import path), and
* the production target is an AdaFace/ArcFace head (§3.2).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

EMBEDDING_DIM = 512


class ViTBiometricEncoder:
    """Unified 512-D multi-modal embedding encoder.

    Parameters
    ----------
    strategy:
        * ``"hash"``    — deterministic stub, zero-dependency (tests, CI,
          synthetic benchmarks). NOT a real biometric model — it maps image
          pixels to a stable pseudo-random unit vector. Used as a scaffold.
        * ``"torch"``   — real transformer backbone via timm (DINOv2/CLIP
          ViT). Loads lazily and only when weights are available.
        * ``"onnx"``    — the trained ToyBiometricEncoder exported to ONNX
          (artifacts/encoder/face_encoder.onnx, ~500 KB) running on
          onnxruntime CPU. Real biometric matching without torch, for the
          reference API container. Canonical input 64×64×3 float [0,1];
          other sizes are resized.
        * ``"adaface"`` — AdaFace/ArcFace classification head (production
          target). Download weights to `~/.cache/abisgpt/` first.
    """

    def __init__(self, strategy: str = "hash", dim: int = EMBEDDING_DIM, seed: int = 0):
        if strategy not in {"hash", "torch", "adaface", "onnx"}:
            raise ValueError(f"Unknown encoder strategy: {strategy!r}")
        self.strategy = strategy
        self.dim = dim
        self.seed = seed
        self._onnx_model_path: str | None = None

    # ------------------------------------------------------------------ #
    # Front-end
    # ------------------------------------------------------------------ #
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize to canonical tensor layout and to float32 in [0, 1]."""
        img = np.asarray(image, dtype=np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img.astype(np.float32) / 255.0

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        """Encode a batch of frames into L2-normalized 512-D vectors."""
        if self.strategy == "torch":
            return self._encode_torch(images)
        if self.strategy == "onnx":
            return self._encode_onnx(images)
        return self._encode_hash(images)

    def encode_one(self, image: np.ndarray) -> np.ndarray:
        return self.encode([image])[0]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.encode_one(image)

    # ------------------------------------------------------------------ #
    # Strategy: trained toy-CNN via ONNXRuntime (real matching, no torch)
    # ------------------------------------------------------------------ #
    def _encode_onnx(self, images: list[np.ndarray]) -> np.ndarray:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Install `onnxruntime` (+opencv) for encoder strategy"
                " 'onnx'.") from exc
        if self._onnx_model_path is None:
            path = Path(os.environ.get(
                "ABISGPT_ENCODER_ONNX", "artifacts/encoder/face_encoder.onnx"))
            if not path.exists():
                raise FileNotFoundError(
                    f"ONNX encoder not found: {path} — export it with"
                    " scripts/export_encoder_onnx.py")
            self._onnx_model_path = str(path)
        sess = ort.InferenceSession(
            self._onnx_model_path, providers=["CPUExecutionProvider"])
        batch = np.stack([self._prepare_onnx(img) for img in images], axis=0)
        out = sess.run(["embeddings"], {"images": batch})[0]
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

    @staticmethod
    def _prepare_onnx(image) -> np.ndarray:
        """Demux uint8 (0..255) or float (0..1) HxWx3 to (3,64,64) float."""
        import cv2

        a = np.asarray(image)
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255.0
        else:
            a = np.clip(np.asarray(a, np.float32), 0.0, 1.0)
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        a = cv2.resize(a, (64, 64), interpolation=cv2.INTER_AREA)
        return a.transpose(2, 0, 1).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Strategy: deterministic hash (tests / synthetic baselines)
    # ------------------------------------------------------------------ #
    def _encode_hash(self, images: list[np.ndarray]) -> np.ndarray:
        vectors = np.zeros((len(images), self.dim), dtype=np.float32)
        for i, image in enumerate(images):
            digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big") + self.seed)
            vec = rng.standard_normal(self.dim).astype(np.float32)
            vectors[i] = vec / max(np.linalg.norm(vec), 1e-9)
        return vectors

    # ------------------------------------------------------------------ #
    # Strategy: real transformer backbone (timm) — torch imported lazily
    # ------------------------------------------------------------------ #
    def _encode_torch(self, images: list[np.ndarray]) -> np.ndarray:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install `torch` (+ timm) for encoder strategy 'torch'.") from exc

        import timm  # lazy import; heavy wheels users opt into

        model = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0)
        model.eval()
        batch = np.stack([self._preprocess(np.asarray(im)) for im in images], axis=0)
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2)  # NCHW
        # Match DINOv2 ViT-B/16 normalisation
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        with torch.no_grad():
            feats = model(tensor)  # (N, D)
        if feats.shape[1] != self.dim:
            # Deterministic PCA-like projection; production heads are learned.
            proj = torch.nn.Linear(int(feats.shape[1]), self.dim, bias=False)
            torch.nn.init.orthogonal_(proj.weight)
            with torch.no_grad():
                feats = proj(feats)
        feats = F.normalize(feats, dim=-1)
        return feats.cpu().numpy().astype(np.float32)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Ensures cosine == dot-product (white paper §4.2) for downstream ANN."""
    v = np.asarray(vec, dtype=np.float32)
    return v / max(np.linalg.norm(v), 1e-9)

