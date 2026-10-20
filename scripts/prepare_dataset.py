#!/usr/bin/env python
"""Download a public face dataset and convert it to an .npz bundle for ABISGPT.

Datasets (both open, no auth直接 download via scikit-learn):
  * olivetti — AT&T Laboratories Cambridge / ORL faces
    (`sklearn.datasets.fetch_olivetti_faces`, figshare-hosted; 400 64x64
    grayscale images, 40 identities x 10 frames). Citation:
    F. Samaria, A. Harter — "Parameterisation of a stochastic model for
    human face identification", Proc. 2nd IEEE Workshop on Applications of
    Computer Vision, 1994.
  * lfw2 — Labeled Faces in the Wild, curated to identities with
    `--min-frames-per-identity` (>=8) faces.

Converter contract (must match `abisgpt.encoder.ViTBiometricEncoder`):
  the encoder front-end does
      img = np.asarray(image, dtype=np.uint8)
      if img.ndim == 2: img = np.stack([img] * 3, axis=-1)
      img.astype(np.float32) / 255.0
  i.e. HxW or HxWx3 uint8 RGB, values 0..255, no resize logic in the
  encoder itself. Frames are therefore stored blueprint-ready as
  (N, H, W, 3) float32 in [0, 1] (the post-`_preprocess` range); to feed
  the pre-normalization path (e.g. the torch/AdaFace strategy) use
  `(frames * 255).astype(np.uint8)`.

Usage:
    python scripts/prepare_dataset.py --dataset olivetti --out artifacts/dataset/olivetti.npz
    python scripts/prepare_dataset.py --dataset lfw2 --min-frames-per-identity 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_HOME = Path("artifacts/data/sklearn")   # explicit cache => reproducible offline


def load_olivetti(min_frames: int) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """Return (frames_u8_gray, labels, names, source_string)."""
    from sklearn.datasets import fetch_olivetti_faces

    ds = fetch_olivetti_faces(
        data_home=str(DATA_HOME), shuffle=False, download_if_missing=True
    )
    images = ds.images                      # (400, 64, 64) float32 in [0, 1]
    labels_all = np.asarray(ds.target, dtype=np.int64)
    counts = np.bincount(labels_all)
    keep_ids = np.where(counts >= min_frames)[0] if min_frames > 0 else np.arange(len(counts))
    mask = np.isin(labels_all, keep_ids)
    gray = (images[mask] * 255.0).astype(np.uint8)   # (n, 64, 64) uint8
    labels, names = _relabel(labels_all[mask], keep_ids)
    source = (
        "sklearn.datasets.fetch_olivetti_faces — AT&T Laboratories Cambridge "
        "(ORL); Samaria & Harter, 'Parameterisation of a stochastic model for "
        "human face identification', IEEE Workshop on Applications of Computer "
        "Vision, 1994"
    )
    return gray, labels, names, source


def load_lfw(min_frames: int) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """Return (frames_u8, labels, names, source_string); needs pillow."""
    from sklearn.datasets import fetch_lfw_people

    ds = fetch_lfw_people(
        data_home=str(DATA_HOME),
        min_faces_per_person=max(min_frames, 8),
        resize=0.5,                               # 0.5 -> 62x47, keeps cache small
        color=True,                               # (n, H, W, 3) RGB, needs pillow
        download_if_missing=True,
    )
    images = (ds.images * 255.0).astype(np.uint8) if ds.images.dtype != np.uint8 else ds.images
    if images.ndim == 3:                          # grayscale -> replicate to RGB
        images = np.stack([images] * 3, axis=-1)
    labels_all = np.asarray(ds.target, dtype=np.int64)
    counts = np.bincount(labels_all)
    keep_ids = np.where(counts >= min_frames)[0] if min_frames > 0 else np.arange(len(counts))
    mask = np.isin(labels_all, keep_ids)
    remap = {int(old): new for new, old in enumerate(sorted(int(i) for i in keep_ids))}
    labels = np.asarray([remap[int(lab)] for lab in labels_all[mask]], dtype=np.int64)
    names = [str(ds.target_names[int(old)]) for old in sorted(int(i) for i in keep_ids)]
    source = (
        "sklearn.datasets.fetch_lfw_people — Labeled Faces in the Wild, "
        "Gary B. Huang et al., UMass Amherst, 2007 (http://vis-www.cs.umass.edu/lfw)"
    )
    return images[mask], labels, names, source


def _relabel(labels: np.ndarray, keep_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Map kept raw identity ids -> contiguous 0..K-1 + identity names."""
    remap = {int(old): new for new, old in enumerate(sorted(int(i) for i in keep_ids))}
    new_labels = np.asarray([remap[int(lab)] for lab in labels], dtype=np.int64)
    names = [f"person_{i:03d}" for i in range(len(keep_ids))]
    return new_labels, names


def main() -> int:
    ap = argparse.ArgumentParser(description="Download + bundle a public face dataset")
    ap.add_argument("--dataset", choices=["olivetti", "lfw2"], default="olivetti")
    ap.add_argument("--out", default=None,
                    help="output npz path (default artifacts/dataset/<dataset>.npz)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-frames-per-identity", type=int, default=0,
                    help="keep only identities with at least this many frames "
                         "(Olivetti ships 10/identity; 0 keeps all)")
    args = ap.parse_args()

    out = Path(args.out or f"artifacts/dataset/{args.dataset}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    DATA_HOME.mkdir(parents=True, exist_ok=True)

    import sklearn

    loader = load_olivetti if args.dataset == "olivetti" else load_lfw
    gray, labels, names, source = loader(args.min_frames_per_identity)

    # Replicate gray -> 3 channels to match the encoder's exact contract.
    frames = gray if gray.ndim == 4 else np.stack([gray] * 3, axis=-1)  # (N, H, W, 3)
    frames = frames.astype(np.float32) / 255.0                          # [0, 1], post-_preprocess
    n, h, w, c = frames.shape

    # Deterministic seeded shuffle (images only; labels travel with frames).
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    frames, labels = frames[perm], labels[perm]

    np.savez_compressed(out, frames=frames, labels=labels, names=np.asarray(names))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()

    manifest = {
        "dataset": args.dataset,
        "n_images": int(n),
        "n_identities": int(len(names)),
        "shape": [int(n), int(h), int(w), int(c)],
        "dtype": "float32",
        "value_range": "[0, 1]",
        "shuffle_seed": int(args.seed),
        "source": source,
        "sha256": sha,
        "sklearn_version": sklearn.__version__,
    }
    manifest_path = out.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    counts = np.bincount(labels, minlength=len(names))
    print(f"[prepare] frames {frames.shape} {frames.dtype} range=[{frames.min():.3f}, {frames.max():.3f}]")
    print(f"[prepare] npz -> {out}  sha256={sha[:16]}...  manifest -> {manifest_path}")
    rows = [len(counts) - 1 - i for i in range(len(counts))]
    width = len(str(len(names) - 1))
    for chunk in range(0, len(rows), 8):
        print("id      : " + " ".join(f"{names[r][-3:]:>{width + 3}}" for r in rows[chunk:chunk + 8]))
        print("count   : " + " ".join(f"{counts[r]:>{width + 3}d}" for r in rows[chunk:chunk + 8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
