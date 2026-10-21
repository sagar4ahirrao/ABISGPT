#!/usr/bin/env python
"""Benchmark the shipped matcher on every npz dataset under artifacts/dataset/.

Per dataset (frames+labels+names contract):
 1. Train the shipped ToyBiometricEncoder + ArcFaceHead on the gallery split
    (identity-holdout: `--probe-frames` held out per id, rest train).
 2. Evaluate PRODUCTION-CODE path: TwoTierPipeline(backend=brute, encoder =
    trained) enroll gallery -> identify each held-out probe frame.
    recall@1 / recall@5 come from what the API actually returns
    (`decision`/`tier2_score`), not from offline math.
 3. Both distributions: genuine vs impostor Tier-2 cosine -> EER.

Datasets with <2 frames/identity are skipped (highlighted as limited).
`pad_casiafasd.npz` labels are fake/real, not identities — its embedded
"Sxxxx" real-subject ids parsed from `names` are used instead.

Usage:
    python scripts/bench_matcher.py                       # all datasets
    python scripts/bench_matcher.py --dataset olivetti.npz --epochs 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from abisgpt.config import AbisGptConfig  # noqa: E402
from abisgpt.encoder import ViTBiometricEncoder  # noqa: E402
from abisgpt.index.factory import BruteForceIndex  # noqa: E402
from abisgpt.pipeline.triage import TwoTierPipeline  # noqa: E402
from abisgpt.training.model import ArcFaceHead, ToyBiometricEncoder  # noqa: E402

REG_DIR = Path("artifacts/dataset")


def casia_subject_names(labels: np.ndarray, names: np.ndarray):
    """CASIA-FASD npz names: 'test_img/.../NN_<vid>.avi_FF_<kind>.jpg'.

    Returns (subject_id, is_real) parsed from the basename.
    """
    import re
    from os.path import basename

    subj = np.empty(len(names), np.int64)
    is_real = np.empty(len(names), bool)
    pat = re.compile(r"^(\d+)_.+\.avi_(\d+)_(real|fake)$")
    for i, n in enumerate(names):
        stem = basename(str(n)).rsplit(".", 1)[0]
        m = pat.match(stem)
        if m is None:
            raise ValueError(f"unparseable casia name: {n!r}")
        subj[i] = int(m.group(1))
        is_real[i] = m.group(3) == "real"
    return subj, is_real


def load_dataset(path: Path):
    z = np.load(path, allow_pickle=False)
    frames = z["frames"]
    names = z["names"]
    if path.name == "pad_casiafasd.npz":
        labels, keep = casia_subject_names(z["labels"], names)
        frames, names = frames[keep], names[keep]
        labels = labels[keep]
    else:
        labels = z["labels"].astype(np.int64)
    return frames, labels, names


def make_split(labels: np.ndarray, holdout: int, seed: int):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    ids = list(np.unique(labels))
    for cid in ids:
        idx = np.flatnonzero(labels == cid)
        rng.shuffle(idx)
        n_hold = int(min(holdout, len(idx) - 1))
        if n_hold < 1:
            continue
        test_idx.extend(idx[:n_hold])
        train_idx.extend(idx[n_hold:])
    train_idx = np.array(sorted(train_idx), int)
    test_idx = np.array(sorted(test_idx), int)
    keep_ids = {int(labels[i]) for i in np.concatenate([train_idx, test_idx])}
    id_map = {c: k for k, c in enumerate(sorted(keep_ids))}
    return train_idx, test_idx, id_map


def to_tensor(frames: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(frames))
    return x.permute(0, 3, 1, 2).contiguous()


def train_encoder(frames, labels, train_idx, id_map, epochs: int,
                  device: str, batch_size: int, lr: float):
    y_train = np.array([id_map[int(c)] for c in labels[train_idx]], np.int64)
    loader = DataLoader(
        TensorDataset(to_tensor(frames[train_idx]),
                      torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True, num_workers=0)
    encoder = ToyBiometricEncoder().to(device)
    head = ArcFaceHead(num_classes=len(id_map), dim=512,
                       scale=24.0, margin=0.22).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters())
                            + list(head.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = torch.nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    for _ep in range(epochs):
        encoder.train()
        head.train()
        tot, seen = 0.0, 0
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device)
            loss = ce(head(encoder(x), y), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(y)
            seen += len(y)
        sched.step()
    train_s = time.perf_counter() - t0
    return encoder, float(tot / max(seen, 1)), round(train_s, 1)


def bench(dataset_path: Path, probe_frames: int, epochs: int, device: str,
          seed: int, batch_size: int, lr: float) -> dict:
    frames, labels, names = load_dataset(dataset_path)
    if frames.ndim == 4 and frames.shape[1] != 64:  # defensive
        frames = frames[:, :64, :64]
    u, counts = np.unique(labels, return_counts=True)
    if (counts >= 2).sum() < 2:
        return {"error": "not enough identities with >=2 frames"}

    train_idx, test_idx, id_map = make_split(labels, probe_frames, seed)
    encoder, loss, train_s = train_encoder(
        frames, labels, train_idx, id_map, epochs, device, batch_size, lr)
    encoder.eval()

    # ---- PRODUCTION PATH: the shipped pipeline identifies the probes ----
    cfg = AbisGptConfig()

    class _Wrapped(ViTBiometricEncoder):
        """Minimal encoder adapter exposing the trained toy model."""

        strategy = "torch"

        def __init__(self, model):
            self.m = model

        def encode(self, frames_: list[np.ndarray], **kw) -> np.ndarray:  # noqa:ARG002
            x = to_tensor(np.asarray(frames_, np.float32)).to(device)
            with torch.no_grad():
                v = self.m(x).cpu().numpy()
            return v / np.maximum(np.linalg.norm(
                v, axis=1, keepdims=True), 1e-9)

    pipe = TwoTierPipeline(
        encoder=_Wrapped(encoder),
        index_backend=BruteForceIndex(dim=512),
        config=cfg,
    )

    # gallery: enroll each id's RAW train frames — the pipeline builds the
    # mean template itself via the same encode() the probes go through.
    t0 = time.perf_counter()
    y_train = np.array([id_map[int(c)] for c in labels[train_idx]])
    for cid in id_map.values():
        t_pos = np.flatnonzero(y_train == cid)
        raw = [frames[train_idx[t]].astype(np.float32) for t in t_pos]
        pipe.enroll(f"p_{cid:04d}", raw)
    t_emb = time.perf_counter() - t0

    top1 = top5 = 0
    latencies, genuine, worst_impostor = [], [], []
    gallery_mat = pipe.index._mat.copy()
    gallery_ids = list(pipe.index._ids)
    import os as _os
    if _os.getenv("BENCH_DEBUG"):
        print("DL[dbg] index type:", type(pipe.index).__name__,
              "len:", len(pipe.index))
        p0 = test_idx[0]
        emb0 = pipe.encode([frames[p0].astype(np.float32)])
        s0 = gallery_mat @ emb0
        order = np.argsort(-s0)[:3]
        print("DL[dbg] probe label:", id_map[int(labels[p0])],
              "true_id", f"p_{id_map[int(labels[p0])]:04d}")
        print("DL[dbg] top3:", [(gallery_ids[j], round(float(s0[j]), 3))
                                for j in order])
        print("DL[dbg] emb norm:", round(float(np.linalg.norm(emb0)), 4),
              "gallery row norm:",
              round(float(np.linalg.norm(gallery_mat[0])), 4))
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(
                frames[p0:p0 + 1].astype(np.float32)))
            x = x.permute(0, 3, 1, 2).contiguous().to(device).float()
            v_direct = encoder(x).cpu().numpy()[0]
        v_direct = v_direct / max(float(np.linalg.norm(v_direct)), 1e-9)
        print("DL[dbg] pipe vs direct-embed cosine:",
              round(float(emb0 @ v_direct), 6))
        # ---- gallery template honesty: mean of THIS id's train frames ----
        cid_t = id_map[int(labels[p0])]
        t_idx = np.flatnonzero(y_train == cid_t)
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(
                frames[train_idx[t_idx]].astype(np.float32)))
            x = x.permute(0, 3, 1, 2).contiguous().to(device).float()
            batch = encoder(x).cpu().numpy()
            batch = batch / np.maximum(
                np.linalg.norm(batch, axis=1, keepdims=True), 1e-9)
            tmpl_direct = batch.mean(axis=0)
            tmpl_direct /= max(float(np.linalg.norm(tmpl_direct)), 1e-9)
        row = gallery_mat[gallery_ids.index(f"p_{cid_t:04d}")]
        print("DL[dbg] gallery-row vs straight-template cosine:",
              round(float(row @ tmpl_direct), 6))
        print("DL[dbg] gallery row L2:",
              round(float(np.linalg.norm(row)), 4))
        print("DL[dbg] direct vs gallery-true cosine:",
              round(float(v_direct @ gallery_mat[gallery_ids.index(
                  f"p_{id_map[int(labels[p0])]:04d}")]), 4),
              "| pipe emb vs gallery-true:",
              round(float(emb0 @ gallery_mat[gallery_ids.index(
                  f"p_{id_map[int(labels[p0])]:04d}")]), 4))
    rank_s = time.perf_counter()
    for _, pidx in enumerate(test_idx):
        y_true = id_map[int(labels[pidx])]
        true_id = f"p_{y_true:04d}"
        j_true = gallery_ids.index(true_id)
        t1 = time.perf_counter()
        probe = frames[pidx].astype(np.float32)
        emb_q = pipe.encode([probe])
        result = pipe.identify([probe], top_k=5, pad=False)
        latencies.append(time.perf_counter() - t1)
        top = result.candidates
        if top and top[0].identity_id == true_id:
            top1 += 1
        if any(c.identity_id == true_id for c in top[:5]):
            top5 += 1
        sims = gallery_mat @ emb_q          # probe vs EVERY gallery template
        genuine.append(float(sims[j_true]))
        others = np.delete(sims, j_true)
        worst_impostor.append(float(others.max()))
    rank_s = time.perf_counter() - rank_s

    genuine, impostor = np.array(genuine), np.array(worst_impostor)
    thr = np.unique(np.concatenate([np.sort(impostor), np.sort(genuine)]))
    far = np.array([(impostor >= t).mean() for t in thr])
    frr = np.array([(genuine < t).mean() for t in thr])
    i_eer = int(np.argmin(np.abs(far - frr)))

    probes_n = len(test_idx)
    return {
        "dataset": dataset_path.name,
        "n_frames": int(len(frames)),
        "n_ids": len(id_map),
        "frames_per_id_mean": round(float(counts[counts >= 2].mean()), 1),
        "protocol": (f"holdout {probe_frames} frame(s)/id; gallery=mean of "
                     "train frames; identify through TwoTierPipeline; "
                     "impostor=worst-case gallery cosine per probe"),
        "epochs": epochs,
        "final_loss": round(loss, 4),
        "train_s": train_s,
        "recall@1": round(top1 / probes_n, 4),
        "recall@5": round(top5 / probes_n, 4),
        "eer": round(float((far[i_eer] + frr[i_eer]) / 2), 4),
        "mean_genuine_sim": round(float(genuine.mean()), 4),
        "mean_impostor_sim": round(float(impostor.mean()), 4),
        "enroll_ms_per_frame": round(t_emb / len(train_idx) * 1000, 2),
        "identify_ms_p50": round(float(np.percentile(latencies, 50)) * 1000, 2),
        "identify_ms_p95": round(float(np.percentile(latencies, 95)) * 1000, 2),
        "identify_s": round(rank_s, 1),
        "n_probes": probes_n,
        "device": device,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", help="subset of npz names")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--probe-frames", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="artifacts/bench_matcher.json")
    args = ap.parse_args()

    npzs = sorted(REG_DIR.glob("*.npz"))
    npzs = [p for p in npzs
            if p.name != "pad_casiafasd.npz"] + [REG_DIR / "pad_casiafasd.npz"]
    if args.dataset:
        npzs = [p for p in npzs if p.name in args.dataset]
    report, failures = [], []
    for p in npzs:
        name = "casiafasd(real-subjects)" if p.name == "pad_casiafasd.npz" \
            else p.name
        print(f"== {name}", flush=True)
        try:
            m = bench(p, args.probe_frames, args.epochs, args.device,
                      args.seed, args.batch_size, args.lr)
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            m = {"dataset": name, "error": f"{exc}"}
        m["dataset"] = name
        report.append(m)
        print(json.dumps(m, indent=2), flush=True)
        Path(args.out).write_text(
            json.dumps({"results": report, "failures": failures},
                       indent=2), encoding="utf-8")

    print("\n== MATCHER BENCH (shipped TwoTierPipeline identify) ==")
    hdr = ("dataset", "ids", "r@1", "r@5", "EER", "genuine",
           "impostor", "id_p50(ms)")
    print("{:<28}{:>5}{:>8}{:>8}{:>8}{:>10}{:>10}{:>12}".format(*hdr))
    for r in report:
        if "error" in r:
            print("{:<28}  ERROR: {}".format(r["dataset"],
                                             r["error"][:40]))
            continue
        print("{:<28}{:>5}{:>8.3f}{:>8.3f}{:>8.3f}{:>10.3f}{:>10.3f}"
              "{:>12.2f}".format(r["dataset"], r["n_ids"], r["recall@1"],
                                 r["recall@5"], r["eer"],
                                 r["mean_genuine_sim"],
                                 r["mean_impostor_sim"],
                                 r["identify_ms_p50"]))
    if failures:
        print("FAILURES:", failures)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
