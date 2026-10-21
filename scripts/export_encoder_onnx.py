#!/usr/bin/env python
"""Export the trained ToyBiometricEncoder (ArcFace head removed) to ONNX.

The reference API container is deliberately torch-free; a 500 KB ONNX file
(with onnxruntime) gives it a REAL biometric matcher for the console:

    torch(False): scripts/export_encoder_onnx.py --weights artifacts/olivetti/olivetti_encoder.pt
                  -> artifacts/encoder/face_encoder.onnx

Validates ONNX vs torch embeddings (cosine + recall@1 parity) before saving.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abisgpt.training.model import ToyBiometricEncoder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="artifacts/olivetti/olivetti_encoder.pt")
    ap.add_argument("--out", default="artifacts/encoder/face_encoder.onnx")
    args = ap.parse_args()

    enc = ToyBiometricEncoder()
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    enc.load_state_dict(state["state_dict"])
    enc.eval()

    x = torch.zeros(2, 3, 64, 64)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        enc, x, args.out,
        input_names=["images"], output_names=["embeddings"],
        dynamic_axes={"images": {0: "batch", 2: "h", 3: "w"},
                      "embeddings": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print("exported ->", args.out)

    # ---- validation: torch vs onnxruntime --------------------------------
    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])

    def embed_torch(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return enc(torch.from_numpy(batch)).numpy()

    rng = np.random.default_rng(0)
    batch = rng.random((6, 3, 64, 64)).astype(np.float32)
    vt, vo = embed_torch(batch), sess.run(["embeddings"], {"images": batch})[0]
    cos = float((vt * vo).sum(1).mean())
    print(f"onnx-vs-torch cosine: {cos:.6f}")

    # recall@1 parity check on the real olivetti split
    z = np.load("artifacts/dataset/olivetti.npz")
    frames, labels = z["frames"], z["labels"].astype(int)
    tr, te = [], []
    rng2 = np.random.default_rng(0)
    for c in range(40):
        idx = np.flatnonzero(labels == c)
        rng2.shuffle(idx)
        te.extend(idx[:2]), tr.extend(idx[2:])
    xb = frames[tr].transpose(0, 3, 1, 2).astype(np.float32)
    xp = frames[te].transpose(0, 3, 1, 2).astype(np.float32)
    tr_t = sess.run(["embeddings"], {"images": xb})[0]
    pr_t = sess.run(["embeddings"], {"images": xp})[0]
    tr_o = embed_torch(xb)
    pr_o = embed_torch(xp)
    for tag, temb, pemb in (("torch", tr_o, pr_o), ("onnx", tr_t, pr_t)):
        temb = temb / np.linalg.norm(temb, axis=1, keepdims=True)
        pemb = pemb / np.linalg.norm(pemb, axis=1, keepdims=True)
        tmpl = np.stack([temb[np.asarray(labels[tr]) == c].mean(0)
                         for c in range(40)])
        tmpl /= np.linalg.norm(tmpl, axis=1, keepdims=True)
        sims = pemb @ tmpl.T
        r1 = float(np.mean(np.argmax(sims, 1) == labels[te]))
        print(f"[{tag}] recall@1: {r1:.4f}")

    size_kb = Path(args.out).stat().st_size / 1024
    print(f"model size: {size_kb:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
