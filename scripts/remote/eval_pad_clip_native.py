#!/usr/bin/env python
"""CLIP-ViT PAD tier at NATIVE resolution on full CASIA-FASD (remote GPU run).

Input:  /home/development/abisgpt/casiafasd/{train,test}_img/{m}_img/color/*.jpg
Output: /home/development/abisgpt/out/pad_clip_native.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path.home() / "abisgpt" / "casiafasd"
OUT = Path.home() / "abisgpt" / "out" / "pad_clip_native.json"
OUT.parent.mkdir(exist_ok=True)

from transformers import CLIPModel, CLIPProcessor  # noqa: E402

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


def labeled(mode: str):
    root = ROOT / f"{mode}_img" / f"{mode}_img" / "color"
    for j in sorted(root.glob("*.jpg")):
        stem = j.stem.lower()
        if stem.endswith("_real"):
            yield j, 1
        elif stem.endswith("_fake"):
            yield j, 0


def auc(liveness, labels):
    pos, neg = liveness[labels == 1], liveness[labels == 0]
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    n = len(pos) * len(neg)
    return float((gt + 0.5 * eq) / n)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    prompts = list(SPOOF_ANCHORS) + list(LIVE_ANCHORS)
    with torch.no_grad():
        t = proc(text=prompts, return_tensors="pt", padding=True)
        t = {k: v.to(device) for k, v in t.items()}
        out = model.get_text_features(**t)
    txt = out if torch.is_tensor(out) else getattr(out, "text_embeds",
                                                   None) or out.pooler_output
    txt = txt / txt.norm(dim=-1, keepdim=True)
    spoof_txt, live_txt = txt[: len(SPOOF_ANCHORS)], txt[len(SPOOF_ANCHORS):]

    report = {"device": device, "splits": {}}
    for mode in ("train", "test"):
        pairs = list(labeled(mode))
        labels, scores = np.zeros(len(pairs), int), np.zeros(len(pairs), float)
        t0 = time.perf_counter()
        B = 128
        for s in range(0, len(pairs), B):
            batch = pairs[s:s + B]
            imgs = [Image.open(p).convert("RGB") for p, _ in batch]
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                out = model.get_image_features(pixel_values=px)
            vis = out if torch.is_tensor(out) else getattr(out, "image_embeds",
                                                           None) or out.pooler_output
            vis = vis / vis.norm(dim=-1, keepdim=True)
            spoof_sim = (vis @ spoof_txt.T).max(dim=1).values
            live_sim = (vis @ live_txt.T).mean(dim=1)
            margin = live_sim - spoof_sim
            scores[s:s + len(batch)] = torch.sigmoid(6.0 * margin).cpu().numpy()
            labels[s:s + len(batch)] = [y for _, y in batch]
        dt = time.perf_counter() - t0
        a = auc(scores, labels)
        # best-threshold accuracy (live = score >= tau)
        grid = np.quantile(scores, np.linspace(0.001, 0.999, 999))
        accs = [(scores >= tau).astype(int) == labels
                for tau in grid]
        accs = [float(a_.mean()) for a_ in accs]
        i = int(np.argmax(accs))
        report["splits"][mode] = {
            "n": len(pairs), "real": int(labels.sum()),
            "auc": round(a, 4),
            "best_tau": round(float(grid[i]), 4),
            "best_acc": round(accs[i], 4),
            "tpr_gate0.5": round(float((scores[labels == 1] >= 0.5).mean()), 4),
            "acc_gate0.5": round(float(((scores >= 0.5).astype(int) == labels).mean()), 4),
            "img_s": round(len(pairs) / dt, 1),
        }
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(mode, "->", json.dumps(report["splits"][mode]), flush=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
