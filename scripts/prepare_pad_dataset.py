#!/usr/bin/env python
"""Build a presentation-attack-detection (PAD) bundle from an anonymously
downloadable face anti-spoofing subset.

Source (verified 2026-10): HuggingFace dataset mirror `akahana/anti-spoofing-casiafasd`
(https://huggingface.co/datasets/akahana/anti-spoofing-casiafasd), which ships
CASIA-FASD preprocessed face crops — no registration / login / gated access.
Original dataset: CASIA Face Anti-Spoofing Database (CASIA-FASD).
Citation: Z. Zhang, J. Yan, M. Liu, Z. Lei, D. Yi, S. Z. Li, "A Face Antispoofing
Database with Diverse Sources", IEEE Trans. Information Forensics and Security,
7(4):1021-1033, 2012.

Fan layout inside `casiafasd.tar.gz` (color selected; depth shipped alongside):
    train_img/train_img/color/<subj>[_HR]_<n>.avi_<frame>_real|.fake.jpg
    test_img/test_img/color/<subj>[_HR]_<n>.avi_<frame>_real|.fake.jpg
`real` => bona fide (label 0), `fake` => presentation attack (label 1).
Genuine frames come from the dataset itself, so no olivetti fill is needed.

Output contract (matches abisgpt.encoder.ViTBiometricEncoder preprocessing):
  frames  (N, 64, 64, 3) float32 in [0, 1]
  labels  (N,) int64     0 = bona fide, 1 = attack
  spoofs  (N,) int64     mask equal to `labels` (1 for attack frames)
  names   (N,) <U..      original crop filenames (provenance)

Usage:
    python scripts/prepare_pad_dataset.py
    python scripts/prepare_pad_dataset.py --max-frames 2000 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "artifacts" / "dataset" / "raw"

SOURCE_URL = (
    "https://huggingface.co/datasets/akahana/anti-spoofing-casiafasd/"
    "resolve/main/casiafasd.tar.gz"
)
SOURCE_CITATION = (
    "CASIA-FASD — Zhang, Yan, Liu, Lei, Yi & Li, 'A Face Antispoofing Database "
    "with Diverse Sources', IEEE Trans. Information Forensics and Security "
    "7(4):1021-1033, 2012; obtained via anonymous HuggingFace mirror "
    "akahana/anti-spoofing-casiafasd (third-party mirror, verified download "
    "2026-10-20)"
)


def _download_probe(url: str) -> str:
    """HEAD-police the source: anonymous GET of the first KB must succeed."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    req.add_header("Range", "bytes=0-1023")
    with urllib.request.urlopen(req, timeout=60) as r:
        magic = r.read(64)
        if not magic.startswith(b"\x1f\x8b"):
            raise RuntimeError(f"unexpected magic bytes {magic[:8]!r} from {url}")
    return "gzip-stream-verified"


def _fetch_tarball(dest: Path) -> str:
    """Anonymous download with progress; skips when already present."""
    if dest.exists() and dest.stat().st_size > 50_000_000:
        print(f"[prepare-pad] reusing cached {dest.name} "
              f"({dest.stat().st_size / 1e6:.1f} MB)")
        return hashlib.sha256(dest.read_bytes()).hexdigest()
    _download_probe(SOURCE_URL)
    print(f"[prepare-pad] downloading {SOURCE_URL} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    print(f"[prepare-pad] downloaded {total / 1e6:.1f} MB -> {dest}")
    return hashlib.sha256(dest.read_bytes()).hexdigest()


def load_casiafasd(frames_per_real_cap: int, frames_per_fake_cap: int,
                   max_frames: int, seed: int):
    """Return (names, labels) tuples; caller decodes with PIL lazily."""
    import tarfile
    import random

    gz_path = RAW_DIR / "casiafasd.tar.gz"
    _fetch_tarball(gz_path)
    tar = tarfile.open(gz_path)
    all_names = [n for n in tar.getnames()
                 if n.endswith(".jpg") and "/color/" in n]
    reals = [n for n in all_names if "_real" in n]
    fakes = [n for n in all_names if "_fake" in n]

    rng = random.Random(seed)
    rng.shuffle(reals)
    rng.shuffle(fakes)

    # cap frames-per-video so no single clip dominates, then honour max_frames
    def _cap_per_video(names: list[str], cap: int, want_bona_fide: bool
                       ) -> list[str]:
        per_clip: dict[str, int] = {}
        picked: list[str] = []
        for n in names:
            clip = n.rsplit("_", 2)[0]
            if per_clip.get(clip, 0) >= cap:
                continue
            per_clip[clip] = per_clip.get(clip, 0) + 1
            picked.append(n)
        return picked

    reals_kept = _cap_per_video(reals, frames_per_real_cap, True)
    n_real = len(reals_kept)
    fake_budget = min(len(fakes), max_frames - n_real) if max_frames else len(fakes)
    attack_ratio = 1.0 if not reals_kept else 1.0  # 1:1 bona fide:attack balance
    fake_budget = min(fake_budget, int(n_real * attack_ratio))
    fakes_kept = _cap_per_video(fakes[: fake_budget * 2], frames_per_fake_cap,
                                False)[:fake_budget]

    selected = [(n, 0) for n in reals_kept] + [(n, 1) for n in fakes_kept]
    rng.shuffle(selected)
    print(f"[prepare-pad] selected {len(reals_kept)} bona-fide + "
          f"{len(fakes_kept)} attack crops (of {len(reals)}/{len(fakes)} available)")
    # lazily decode via the still-open tarfile; caller gets (idx0 -> PIL) via
    # a member-extraction helper bound to `tar`.
    return tar, selected


import numpy as np

def _decode(tar, selected, resize: int):
    """Decode crops to (N, resize, resize, 3) float32 in [0, 1]."""
    from PIL import Image

    frames = []
    labels = []
    names = []
    for i, (name, label) in enumerate(selected):
        buf = tar.extractfile(name).read()
        img = Image.open(io.BytesIO(buf)).convert("RGB")
        if img.size != (resize, resize):
            img = img.resize((resize, resize), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        frames.append(arr)
        labels.append(label)
        names.append(name)
        if (i + 1) % 200 == 0:
            print(f"[prepare-pad] decoded {i + 1}/{len(selected)} frames")
    return np.stack(frames), np.asarray(labels, dtype=np.int64), names


def main() -> int:
    ap = argparse.ArgumentParser(description="Bundle CASIA-FASD (PAD) into an npz")
    ap.add_argument("--out", default="artifacts/dataset/pad_casiafasd.npz")
    ap.add_argument("--frames-per-real-cap", type=int, default=8,
                    help="max crops kept from any bona-fide video")
    ap.add_argument("--frames-per-fake-cap", type=int, default=4,
                    help="max crops kept from any attack video")
    ap.add_argument("--max-frames", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-archive", action="store_true",
                    help="keep artifacts/dataset/raw/casiafasd.tar.gz after build")
    args = ap.parse_args()

    out = Path(REPO_ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tar, selected = load_casiafasd(args.frames_per_real_cap,
                                   args.frames_per_fake_cap,
                                   args.max_frames, args.seed)
    frames, labels, names = _decode(tar, selected, resize=64)
    tar.close()
    print(f"[prepare-pad] raw archive kept at "
          f"{(REPO_ROOT / 'artifacts/dataset/raw/casiafasd.tar.gz')}")

    spoofs = (labels == 1).astype(np.int64)
    np.savez_compressed(out,
                        frames=frames,
                        labels=labels,
                        spoofs=spoofs,
                        names=np.asarray(names))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()

    manifest = {
        "dataset": "CASIA-FASD (anonymously-mirrored subset, color modality)",
        "pad_task": "2-class presentation attack detection",
        "label_map": {"0": "bona-fide", "1": "attack/spoof"},
        "n_frames": int(len(labels)),
        "n_bonafide": int((labels == 0).sum()),
        "n_attack": int((labels == 1).sum()),
        "shape": [int(frames.shape[0]), 64, 64, 3],
        "dtype": "float32",
        "value_range": "[0, 1]",
        "attack_types": "photo, cut, paint, print, video replay (CASIA-FASD P-classes)",
        "modalities_available": ["rgb(color)", "depth"],
        "source_url": SOURCE_URL,
        "source_sha256_archive": _fetch_tarball(RAW_DIR / "casiafasd.tar.gz"),
        "seed": int(args.seed),
        "frames_per_real_cap": int(args.frames_per_real_cap),
        "frames_per_fake_cap": int(args.frames_per_fake_cap),
        "citation": SOURCE_CITATION,
        "sha256": sha,
    }
    manifest_path = out.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[prepare-pad] frames {frames.shape} {frames.dtype} "
          f"range=[{frames.min():.3f}, {frames.max():.3f}]")
    print(f"[prepare-pad] bona-fide={int((labels == 0).sum())} "
          f"attack={int((labels == 1).sum())}")
    print(f"[prepare-pad] npz -> {out} sha256={sha[:16]}... "
          f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
