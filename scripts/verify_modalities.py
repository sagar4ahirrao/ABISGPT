#!/usr/bin/env python
"""Verify ALL biometric modalities by name through the shipped matcher.

Protocol per modality (names = ground truth from the files/records):
 1. Enroll one template per NAME (multi-frame mean via the pipeline's own
    encode()).
 2. Held-out probes: same-name frames -> GENUINE verify; different-name
    frames -> IMPOSTOR verify. Run through MatchingEngine.verify_1v1 at the
    shipped threshold — the exact path the API/console uses.
 3. Reports FMR / FNMR / accuracy / score separation.

Modalities:
  face      CASIA-FASD color test — subject names parsed from FILENAMES
            (`NN_<vid>.avi_FF_<kind>`), ONNX toy encoder (the shipped
            container path). Attack mode: fake-video (replay) probes of the
            SAME subject — biometrically genuine, PAD's job to reject.
  face_ref  Olivetti — identity names from the npz; sanity parity.
  fingerprint  SOKOCOVENTRY — identity names from the npz (trained
            ArcFace toy encoder per modality, inside the real pipeline).
  iris      UniDataPro — 6 subjects (tiny; directional only).

Usage: python scripts/verify_modalities.py [--out artifacts/verify_modalities.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from os.path import basename
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bench_matcher import load_dataset, make_split, train_encoder  # noqa: E402

from abisgpt.config import AbisGptConfig  # noqa: E402
from abisgpt.encoder import ViTBiometricEncoder  # noqa: E402
from abisgpt.index.factory import BruteForceIndex  # noqa: E402
from abisgpt.pipeline.matching import MatchingEngine  # noqa: E402
from abisgpt.pipeline.triage import TwoTierPipeline  # noqa: E402

ROOT = Path("artifacts/dataset/raw/casiafasd")
ONNX = Path("artifacts/encoder/face_encoder.onnx")
OLIVETTI = Path("artifacts/dataset/olivetti.npz")
PAT = re.compile(r"^(\d+)_(.+)\.avi_(\d+)_(real|fake)$")

_THRESHOLD = AbisGptConfig().pipeline.match_threshold


def _engine(encoder_fn) -> tuple[MatchingEngine, TwoTierPipeline]:
    """Wrap any embedding function into the shipped TwoTierPipeline."""

    class _Wrapped(ViTBiometricEncoder):
        strategy = "torch"

        def __init__(self, fn):
            self.fn = fn

        def encode(self, frames_, **kw):
            v = np.asarray(self.fn(list(frames_)), np.float32)
            return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True),
                                  1e-9)

    pipe = TwoTierPipeline(
        encoder=_Wrapped(encoder_fn),
        index_backend=BruteForceIndex(dim=512),
        config=AbisGptConfig(),
    )
    return MatchingEngine(pipe), pipe


def _verify_report(engine: MatchingEngine, pipe: TwoTierPipeline,
                   names: list[str], gallery: dict[str, list],
                   probes_genuine, probes_impostor,
                   probes_attack=None, pad: bool = False) -> dict:
    """Enroll by name -> verify_1v1 each held-out probe (shipped path).

    pad=False bypasses the liveness gate; pad=True runs the fused PAD gate
    first (honest still-mode path) and additionally reports whether the gate
    ever blocked an honest capture (`pad_blocked_px`) and how often the
    "still mode" reason was carried (`still_reason_coverage`).
    """
    for n in names:
        pipe.enroll(n, gallery[n])
    gen, imp, atk = [], [], []
    still_hits, gate_calls, blocked = 0, 0, 0
    for n, frames in probes_genuine:
        r = engine.verify_1v1(n, frames, pad=pad)
        gen.append(r.score)
        if pad:
            gate_calls += 1
            if not r.pad_live:
                blocked += 1
            if r.pad_reasons and any("still mode" in x
                                     for x in r.pad_reasons):
                still_hits += 1
    for n, frames in probes_impostor:
        imp.append(engine.verify_1v1(n, frames, pad=pad).score)
    if probes_attack:
        for n, frames in probes_attack:
            r = engine.verify_1v1(n, frames, pad=pad)
            atk.append((r.score, r.is_match))
    gen, imp = np.array(gen), np.array(imp)

    fmr = float(np.mean(imp >= _THRESHOLD)) if len(imp) else None
    fnmr = float(np.mean(gen < _THRESHOLD)) if len(gen) else None
    correct = ((gen >= _THRESHOLD).sum() if len(gen) else 0) \
        + ((imp < _THRESHOLD).sum() if len(imp) else 0)
    acc = round(float(correct) / max(len(gen) + len(imp), 1), 4)
    if len(gen) and len(imp):
        # direction-aware: if impostor scores land HIGHER than genuine
        # (encoder out-of-domain), flip the decision direction so EER stays
        # meaningful instead of collapsing to ~0 at the crossing point.
        flip = float(imp.mean()) > float(gen.mean())
        g, i = (-gen if flip else gen), (-imp if flip else imp)
        thr = np.unique(np.concatenate([g, i]))
        far = np.array([(i >= t).mean() for t in thr])
        frr = np.array([(g < t).mean() for t in thr])
        eer = round(float(np.min(np.abs(far - frr))), 4)
        eer_note = "direction FLIPPED (impostor>genuine: encoder out-of- "\
            "domain for this corpus)" if flip else None
    else:
        eer, eer_note = None, None
    out = {
        "threshold": _THRESHOLD, "n_genuine": int(len(gen)),
        "n_impostor": int(len(imp)),
        "genuine_mean": round(float(gen.mean()), 4) if len(gen) else None,
        "impostor_mean": round(float(imp.mean()), 4) if len(imp) else None,
        "fmr": round(fmr, 4) if fmr is not None else None,
        "fnmr": round(fnmr, 4) if fnmr is not None else None,
        "accuracy": acc, "eer": eer, "eer_note": eer_note,
        "genuine_match_rate": round(float(np.mean(gen >= _THRESHOLD)), 4)
        if len(gen) else None,
    }
    if probes_attack:
        out["attack_mean_sim"] = round(float(np.mean(
            [s for s, _ in atk])), 4) if atk else None
        out["attack_match_rate"] = round(
            float(np.mean([m for _, m in atk])) if atk else 0.0, 4)
    if pad:
        out["pad_gate"] = True
        out["pad_blocked_genuine"] = blocked
        out["still_reason_coverage"] = round(still_hits / gate_calls, 4) \
            if gate_calls else None
    return out


def _load64(p: Path) -> np.ndarray:
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
    return im.astype(np.float32) / 255.0


def verify_face_casia() -> dict:
    """CASIA-FASD: names parsed from image filenames; shipped ONNX path.

    This is the CROSS-DOMAIN check (encoder trained on Olivetti, probes are
    CASIA stills). A second in-domain result is produced by
    verify_face_casia_indomain().
    """
    if not ONNX.exists():
        return {"error": "onnx encoder missing"}
    enc = ViTBiometricEncoder(strategy="onnx")
    engine, pipe = _engine(lambda imgs: enc.encode(imgs))

    by = defaultdict(lambda: defaultdict(list))
    for p in sorted((ROOT / "test_img" / "test_img" / "color").glob("*.jpg")):
        m = PAT.match(basename(p).rsplit(".", 1)[0])
        if m:
            by[m.group(1)][m.group(4)].append((m.group(2), p))

    gallery: dict[str, list] = {}
    gen_probes, imp_probes, atk_probes = [], [], []
    rng = np.random.default_rng(0)
    subjects = sorted(by)
    for s in subjects:
        reals = by[s].get("real", [])
        fakes = by[s].get("fake", [])
        if not reals:
            continue
        vids = sorted({v for v, _ in reals})
        first = [p for v, p in reals if v == vids[0]]
        other = [p for v, p in reals if v != vids[0]]
        gallery[f"face_{s}"] = [_load64(p) for p in first[:6]]
        for p in other[:6]:
            gen_probes.append((f"face_{s}", [_load64(p)]))
        for _v, p in fakes[:6]:
            # replay attack: claims to be THIS subject via its own spoofed face
            atk_probes.append((f"face_{s}", [_load64(p)]))
        subs = [x for x in subjects if x != s and by[x].get("real")]
        if subs:
            other_s = subs[int(rng.integers(len(subs)))]
            ov = by[other_s]["real"][0][1]
            imp_probes.append((f"face_{s}", [_load64(ov)]))
    rep = _verify_report(engine, pipe, list(gallery), gallery, gen_probes,
                         imp_probes, atk_probes)
    rep["note_attack"] = ("fake-video probes replay the SAME subject's face;"
                          " high match rate is expected biometrically —"
                          " rejection belongs to the PAD layer")
    rep["note_domain"] = ("CROSS-DOMAIN: ONNX encoder is Olivetti-trained —"
                          " impostor sims above genuine mean signal"
                          " style-dominance on unseen capture styles")
    return rep


def verify_face_casia_indomain(device: str,
                               pads: tuple[bool, ...] = (False,)) -> dict:
    """CASIA-FASD: train the toy encoder ON casia real frames, then verify
    every subject by name from the filenames through the shipped matcher."""
    import bench_matcher as bm

    frames, labels, names_all = bm.load_dataset(
        Path("artifacts/dataset/pad_casiafasd.npz"))
    # bm.load_dataset already converts to real-subject labels; here we need
    # name strings -> rebuild from names directly.
    z = np.load("artifacts/dataset/pad_casiafasd.npz")
    names_arr = z["names"].astype(str)
    stem = np.array([n.rsplit(".", 1)[0] for n in names_arr])
    is_real = np.array([n.endswith("real") for n in stem])
    labels_full = np.array([int(n.split("/")[-1].split("_")[0])
                            for n in stem])
    frames = z["frames"][is_real]
    labels = labels_full[is_real]
    u, counts = np.unique(labels, return_counts=True)
    if (counts >= 2).sum() < 2:
        return {"error": "not enough in-domain subjects"}
    train_idx, test_idx, id_map = make_split(labels, 2, 0)
    enc, _, _ = train_encoder(frames, labels, train_idx, id_map, 30, device,
                              32, 1e-3)
    _ = u, counts, names_all, bm
    enc.eval()

    def fn(imgs):
        x = torch.from_numpy(np.ascontiguousarray(
            np.asarray(imgs, np.float32)))
        x = x.permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            return enc(x).cpu().numpy()

    engine, pipe = _engine(fn)
    tr_by, te_by = defaultdict(list), defaultdict(list)
    for t in train_idx:
        tr_by[id_map[int(labels[t])]].append(frames[t])
    for t in test_idx:
        te_by[id_map[int(labels[t])]].append(frames[t])
    cids = list(tr_by)
    gallery = {f"subject_{c:03d}": [f.astype(np.float32) for f in tr_by[c]]
               for c in cids}
    gen_p, imp_p = [], []
    for i, cid in enumerate(cids):
        nm = f"subject_{cid:03d}"
        for f in te_by.get(cid, []):
            gen_p.append((nm, [f.astype(np.float32)]))
        other = cids[(i + 1) % len(cids)]
        if te_by[other]:
            imp_p.append((nm, [te_by[other][0].astype(np.float32)]))
    reps = {f"pad_{int(p)}": _verify_report(engine, pipe, list(gallery),
                                            gallery, gen_p, imp_p, pad=p)
            for p in pads}
    for r in reps.values():
        r["note_domain"] = "IN-DOMAIN: toy encoder trained on this corpus"
    return reps if len(pads) > 1 else reps["pad_0"]


def _verify_trained(name: str, path: Path, device: str,
                    probe_holdout: int = 2,
                    pads: tuple[bool, ...] = (False,)) -> dict:
    """Generic npz modality: train toy encoder, verify by identity name.

    `pads` trains ONCE and evaluates the identical engine with the PAD gate
    off/on so the two columns are directly comparable.

    Returns `{report}` when a single mode, or `{"pad_0": …, "pad_1": …}`.
    """
    frames, labels, names = load_dataset(path)
    u, counts = np.unique(labels, return_counts=True)
    if (counts >= 2).sum() < 2:
        return {"error": "not enough identities with >=2 frames"}
    train_idx, test_idx, id_map = make_split(labels, probe_holdout, 0)
    enc, _, _ = train_encoder(frames, labels, train_idx, id_map, 30, device,
                              32, 1e-3)
    enc.eval()
    inv = {v: k for k, v in id_map.items()}

    def fn(imgs):
        x = torch.from_numpy(np.ascontiguousarray(
            np.asarray(imgs, np.float32)))
        x = x.permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            return enc(x).cpu().numpy()

    engine, pipe = _engine(fn)
    tr_by, te_by = defaultdict(list), defaultdict(list)
    for t in train_idx:
        tr_by[id_map[int(labels[t])]].append(frames[t])
    for t in test_idx:
        te_by[id_map[int(labels[t])]].append(frames[t])
    cids = list(tr_by)
    gallery = {f"id_{inv[c]:04d}": [f.astype(np.float32) for f in tr_by[c]]
               for c in cids}
    gen_p, imp_p = [], []
    for i, cid in enumerate(cids):
        nm = f"id_{inv[cid]:04d}"
        for f in te_by.get(cid, []):
            gen_p.append((nm, [f.astype(np.float32)]))
        other = cids[(i + 1) % len(cids)]
        if te_by[other]:
            imp_p.append((nm, [te_by[other][0].astype(np.float32)]))
    reps = {f"pad_{int(p)}": _verify_report(engine, pipe, list(gallery),
                                            gallery, gen_p, imp_p, pad=p)
            for p in pads}
    return reps if len(pads) > 1 else reps["pad_0"]


def verify_olivetti(device: str) -> dict:
    return _verify_trained("olivetti", OLIVETTI, device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/verify_modalities.json")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    report = {
        "protocol": (
            "names = ground truth; one template per name; verify via the"
            " shipped MatchingEngine.verify_1v1"
            f" (threshold {_THRESHOLD}, pad=False still path)"),
        "modalities": {},
    }
    print("[face/casia] cross-domain (shipped ONNX, Olivetti-trained)...")
    report["modalities"]["face_casia_crossdomain"] = verify_face_casia()
    print("[face/casia] IN-DOMAIN trained encoder by filename subject...")
    report["modalities"]["face_casia_indomain"] = verify_face_casia_indomain(
        device)
    print("[face/olivetti] verifying by identity name (trained encoder)...")
    report["modalities"]["face_olivetti"] = verify_olivetti(device)
    print("[fingerprint/sokocoventry] verifying by identity name...")
    report["modalities"]["fingerprint_sokocoventry"] = _verify_trained(
        "fp_sokocoventry", Path("artifacts/dataset/fp_sokocoventry.npz"),
        device)
    print("[iris/unidatapro] (6 subjects, directional)...")
    report["modalities"]["iris_unidatapro"] = _verify_trained(
        "iris", Path("artifacts/dataset/iris_unidatapro.npz"), device,
        probe_holdout=1)

    Path(args.out).write_text(json.dumps(report, indent=2),
                              encoding="utf-8")
    print("saved ->", args.out)

    hdr = ("modality", "n_gen", "n_imp", "gen_mean", "imp_mean",
           "FMR", "FNMR", "acc", "EER", "atk")
    print("\n== NAME-VERIFY REPORT (shipped verify_1v1) ==")
    print("{:<26}{:>7}{:>7}{:>10}{:>10}{:>7}{:>7}{:>7}{:>7}{:>7}"
          .format(*hdr))
    for k, r in report["modalities"].items():
        if "error" in r:
            print("{:<26} ERROR {}".format(k, r["error"][:40]))
            continue
        print("{:<26}{:>7}{:>7}{:>10}{:>10}{:>7}{:>7}{:>7}{:>7}{:>7}"
              .format(k, r["n_genuine"], r["n_impostor"], r["genuine_mean"],
                      r["impostor_mean"], r["fmr"], r["fnmr"],
                      r["accuracy"], r["eer"],
                      r.get("attack_match_rate", "-")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
