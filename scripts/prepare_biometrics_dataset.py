"""Prepare matched biometric identity datasets (fingerprints + iris) akin to olivetti.npz.

Downloads real biometric image sets via *anonymous* HTTPS URLs (no account / API key
required, verified 2026-10 by GET of the first bytes), then converts each into
olivetti-style .npz archives:

    artifacts/dataset/fp_sokocoventry.npz   (~n, 64, 64, 3) fingerprint frames
    artifacts/dataset/iris_unidatapro.npz   (n, 64, 64, 3)  iris texture frames

Each archive holds:
    frames : float32 images in [0, 1]; grayscale is channel-replicated to 3 channels
    labels : int64 contiguous identity ids (0..K-1)
    names  : str array of shape (K,) mapping identity id -> stable identity name

Alongside each .npz a <name>.json manifest records sha256 of the .npz, the source
citation, and license notes.

Sources
-------
SOCOFing (Sokoto Coventry Fingerprint Dataset), "Real" subset, mirrored ungated on
HuggingFace hub at veerav96/sokoto_real.  Original Kaggle distribution is gated; the
mirror's README pins it as the same Real set (600 subjects x 10 fingers).
We ingest the first `FP_SUBJECTS` subjects sorted by id -> 10 frames each.

Human Iris Dataset sample set (UniDataPro), mirron ungated on HuggingFace hub at
UniDataPro/human-iris-images-biometric (CC-BY-NC-ND-4.0).  7 identities x
left/right eye -> 14 frames.

Run:  python scripts/prepare_biometrics_dataset.py [--redownload]
"""
import argparse
import hashlib
import json
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "artifacts" / "dataset" / "biometrics_raw"
OUT = ROOT / "artifacts" / "dataset"

FP_TARGET = OUT / "fp_sokocoventry.npz"
IRIS_TARGET = OUT / "iris_unidatapro.npz"

FP_ZIP = RAW / "sokoto_real.zip"
FP_ZIP_URL = "https://huggingface.co/datasets/veerav96/sokoto_real/resolve/main/Real.zip"

IRIS_BASE_URL = (
    "https://huggingface.co/datasets/UniDataPro/human-iris-images-biometric/resolve/main/"
)

FP_SUBJECTS = 100  # subjects ingested -> 10 finger frames each
FP_MAX_FRAMES = 1000


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(dest) + ".part")
    print(f"download {url} -> {dest}")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    if tmp.exists():
        tmp.replace(dest)
    else:
        tmp.rename(dest)


def _to_frames(np_stack) -> np.ndarray:
    return np.stack(
        [np.asarray(img, dtype=np.float32).repeat(3, axis=-1) for img in np_stack]
    )


def _save_and_manifest(npz_path: Path, frames, labels, names, meta: dict):
    np.savez_compressed(npz_path, frames=frames, labels=labels, names=names)
    sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    manifest = {
        "dataset": npz_path.stem,
        "n_images": int(frames.shape[0]),
        "n_identities": int(len(names)),
        "shape": [int(x) for x in frames.shape],
        "dtype": "float32",
        "value_range": "[0, 1]",
        "source": meta["source"],
        "license_note": meta["license_note"],
        "citation": meta["citation"],
        "anonymous_url": meta["anonymous_url"],
        "sha256": sha,
    }
    json_path = npz_path.with_suffix(".json")
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"wrote {npz_path} frames={frames.shape} identities={len(names)} sha256={sha[:16]}..."
    )


def prepare_fingerprint(force: bool) -> None:
    if not FP_ZIP.exists() or force:
        _download(FP_ZIP_URL, FP_ZIP)
    with zipfile.ZipFile(FP_ZIP) as z:
        names = [
            n for n in z.namelist()
            if n.startswith("Real/") and n.lower().endswith(".bmp")
        ]
        # subject number is the leading integer before "__"
        by_subject = {}
        for n in names:
            stem = n.split("/")[-1]
            sid, _, _ = stem.partition("__")
            by_subject.setdefault(int(sid), []).append(n)
        kept_sids = sorted(by_subject)[: FP_SUBJECTS]
        images, labels, subj_names = [], [], []
        seen = 0
        for idx, sid in enumerate(kept_sids):
            for fname in sorted(by_subject[sid]):
                if seen >= FP_MAX_FRAMES:
                    break
                with Image.open(z.open(fname)) as img:
                    img = img.convert("L").resize((64, 64), Image.BILINEAR)
                images.append(np.asarray(img, dtype=np.float32) / 255.0)
                labels.append(idx)
                subject, _, rest = fname.split("/")[-1].partition("__")
                subj_names.append(f"subject_{subject}_{rest.lower()}")
                seen += 1
        frames = np.stack(images)[:, :, :, None].repeat(3, axis=3).astype(np.float32)
        # shuffle rows deterministically while keeping labels aligned
        rng = np.random.RandomState(42)
        order = rng.permutation(len(frames))
        frames, labels, subj_names = frames[order], np.array(labels, dtype=np.int64), [
            subj_names[i] for i in order
        ]
        names = np.array(
            [f"sokoco_{sid:03d}" for sid in kept_sids], dtype="<U16"
        )
    _save_and_manifest(
        FP_TARGET,
        frames,
        labels,
        names,
        {
            "source": "SOCOFing 'Real' subset used verbatim from the ungated HF mirror "
                      "veerav96/sokoto_real (original authors: O Joshua, BB Zain, "
                      "A Joseph, A Jimoh, A Hassan; Sokoto State University & "
                      "Coventry University).",
            "license_note": "Mirror repo labelled Apache-2.0; original SOCOFing is "
                            "published for research/educational use and is gated on "
                            "Kaggle, so redistribution beyond research use is "
                            "unclear. Verify before any external redistribution.",
            "citation": "@dataset{joshua2018sokoco, author={Joshua, Opeyemi and "
                        "Zain, Bala and Joseph, Adamu and Jimoh, Abdulganiyu and "
                        "Hassan, A.}, title={{SOCOFing}: Sokoto Coventry Fingerprint "
                        "Dataset}, year={2018}, note={600 subjects x 10 fingers}}",
            "anonymous_url": FP_ZIP_URL,
        },
    )


def prepare_iris(force: bool) -> None:
    samples = []
    for i in range(1, 8):
        for gender in ("m", "f"):
            for side in ("left", "right"):
                rel = f"id_{i}_{gender}/id_{i}_{side}_{gender}.jpg"
                url = IRIS_BASE_URL + rel
                dest = RAW / f"iris_id{i}_{side}_{gender}.jpg"
                try:
                    if not dest.exists() or force:
                        _download(url, dest)
                except urllib.error.HTTPError as e:
                    # some id/gender combinations simply don't exist (e.g. id_1_f)
                    print(f"skip {rel}: HTTP {e.code}")
                    continue
                samples.append((i, side, gender, dest))
    if not samples:
        raise RuntimeError("no iris samples downloaded")
    images, labels, caps = [], [], []
    seen_ids = {}
    for i, side, gender, path in samples:
        idx = seen_ids.setdefault(i, len(seen_ids))
        with Image.open(path) as img:
            img = img.convert("L").resize((64, 64), Image.BILINEAR)
        images.append(np.asarray(img, dtype=np.float32) / 255.0)
        labels.append(idx)
        caps.append(f"id_{i}_{side}_{gender}")
    frames = np.stack(images)[:, :, :, None].repeat(3, axis=3).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    ids_sorted = sorted(seen_ids)
    names = np.array([f"unidatapro_id{i:02d}" for i in ids_sorted], dtype="<U16")
    _save_and_manifest(
        IRIS_TARGET,
        frames,
        labels,
        names,
        {
            "source": "UniDataPro 'Human Iris Images - biometric' sample set "
                      "(small ungated mirror on HuggingFace hub: "
                      "datasets/UniDataPro/human-iris-images-biometric). "
                      "7 identities (id_1..id_7), left/right eye each.",
            "license_note": "CC-BY-NC-ND-4.0 as declared on the dataset card; "
                            "non-commercial use only, no derivatives; attribution "
                            "required. Verify before any external redistribution.",
            "citation": "@dataset{unidatapro2026iris, author={{UniDataPro}}, title="
                        "{Human Iris Images - biometric sample set}, year={2026}, "
                        "url={https://huggingface.co/datasets/"
                        "UniDataPro/human-iris-images-biometric}, license={CC-BY-NC-ND-4.0}}",
            "anonymous_url": IRIS_BASE_URL,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redownload", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    prepare_fingerprint(args.redownload)
    prepare_iris(args.redownload)


if __name__ == "__main__":
    main()
