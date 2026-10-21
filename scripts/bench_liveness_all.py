#!/usr/bin/env python
"""Liveness (PAD) + matcher battery across ALL available datasets.

Part 1 — passive PAD sweep, per corpus, stillfix vs legacy A/B:
  every still is replicated 8x (console contract) and fused by
  `PassivePadEngine`. Reports live-rate, fused mean, per-tier means,
  and "still mode" reason coverage. Legacy run disables
  `_is_still_replica` so the report shows exactly what the fix changed.

  Corpora:
    casia_fasd   real captures (label 1) vs replay attacks (label 0)
                 -> also reports AUC / gate accuracy (attack realism)
    olivetti     all stills
    fp_sokocoventry   all stills (fingerprint crops)
    iris_unidatapro   all stills (iris crops)

Part 2 — joint matcher x PAD parity:
  per modality the SAME trained engine verifies held-out probes with the
  gate OFF (pad_0) and ON (pad_1); honest stills must never be blocked
  (pad_blocked_genuine == 0, still_reason_coverage == 1.0) and the
  FMR/FNMR columns must remain identical.

Usage:
    python scripts/bench_liveness_all.py [--out artifacts/liveness_all.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402
import verify_modalities as vm  # noqa: E402

from abisgpt.config import AbisGptConfig  # noqa: E402
from abisgpt.pad.fusion import PassivePadEngine  # noqa: E402
from abisgpt.pad.vlm import HeuristicVlmPad  # noqa: E402

CASIA = Path("artifacts/dataset/raw/casiafasd")


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / max(len(pos) * len(neg), 1))


def _pad_sweep(engine: PassivePadEngine, stills: list[np.ndarray],
               labels: np.ndarray | None, tag: str) -> dict:
    fused, flash, rppg, vlm, live, still_reason = [], [], [], [], 0, 0
    for k, img in enumerate(stills):
        frames = [img] * 8
        v = engine(frames)
        d = v.as_dict()
        fused.append(v.fused_score)
        live += int(bool(v.is_live))
        tiers = d.get("tier_scores") or {}
        flash.append(tiers.get("flash"))
        rppg.append(tiers.get("rppg"))
        vlm.append(tiers.get("vlm"))
        if any("still mode" in x for x in (d.get("reasons") or [])):
            still_reason += 1
        if k % 400 == 0:
            print(f"  [{tag}] {k}/{len(stills)}", flush=True)
    fused = np.asarray([f for f in fused if f is not None], np.float32)
    tiers_stats = {}
    for k_tier, arr in (("flash", flash), ("rppg", rppg), ("vlm", vlm)):
        vals = np.asarray([x for x in arr if x is not None], np.float32)
        tiers_stats[f"{k_tier}_mean"] = (
            round(float(vals.mean()), 4) if len(vals) else None)
        tiers_stats[f"{k_tier}_coverage"] = round(
            len(vals) / len(stills), 4)
    out = {
        "n": len(stills),
        "live_rate": round(live / len(stills), 4),
        "fused_mean": round(float(fused.mean()), 4) if len(fused) else None,
        "still_reason_coverage": round(still_reason / len(stills), 4),
        "gate": engine.cfg.gate_threshold,
        **tiers_stats,
    }
    if labels is not None and len(fused) == len(labels):
        labels = np.asarray(labels)
        pos, neg = fused[labels == 1], fused[labels == 0]
        out["auc"] = round(_auc(pos, neg), 4)
        g = engine.cfg.gate_threshold
        out["gate_acc"] = round(float(
            ((fused >= g).astype(int) == labels).mean()), 4)
        out["gate_tnr"] = round(float((neg < g).mean()), 4) if len(neg) else None
        out["gate_tpr"] = round(float((pos >= g).mean()), 4) if len(pos) else None
    return out


def _load_casia(split: str, cap: int = 1500) -> tuple[list, np.ndarray]:
    color = CASIA / f"{split}_img" / f"{split}_img" / "color"
    files = []
    for p in sorted(color.glob("*.jpg")):
        m = vm.PAT.match(p.name.rsplit(".", 1)[0])
        if m:
            files.append((m.group(1), m.group(2), m.group(4), p))
    if cap and len(files) > cap:
        idx = np.linspace(0, len(files) - 1, cap).astype(int)
        files = [files[i] for i in idx]
    stills, labels = [], []
    for _s, _v, kind, p in files:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA)
        stills.append(im)
        labels.append(1 if kind == "real" else 0)
    return stills, np.asarray(labels)


def _load_npz_stills(name: str, cap: int = 240) -> list:
    z = np.load(f"artifacts/dataset/{name}.npz")
    frames = z["frames"]
    if cap and len(frames) > cap:
        idx = np.linspace(0, len(frames) - 1, cap).astype(int)
        frames = frames[idx]
    return [cv2.resize((f * 255).astype(np.uint8), (32, 32),
                       interpolation=cv2.INTER_AREA) if f.shape[:2] != (32, 32)
            else (f * 255).astype(np.uint8) for f in frames]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/liveness_all.json")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    cfg = AbisGptConfig()
    report: dict = {"fix_mode": {}, "joint_matcher_pad": {}}

    def run_all(engine: PassivePadEngine, tag: str) -> dict:
        blob = {}
        print(f"== PAD sweep [{tag}] ==")
        for split in ("train", "test"):
            stills, labels = _load_casia(split)
            blob[f"casia_fasd_{split}"] = _pad_sweep(
                engine, stills, labels, f"casia_{split}")
            r = blob[f"casia_fasd_{split}"]
            print(f"  casia_{split}: live_rate={r['live_rate']} "
                  f"fused={r['fused_mean']} auc={r.get('auc')} "
                  f"gate_acc={r.get('gate_acc')}")
        for name in ("olivetti", "fp_sokocoventry", "iris_unidatapro"):
            stills = _load_npz_stills(name)
            blob[name] = _pad_sweep(engine, stills, None, name)
            r = blob[name]
            print(f"  {name}: live_rate={r['live_rate']} "
                  f"fused={r['fused_mean']} "
                  f"still_cov={r['still_reason_coverage']}")
        return blob

    print("== Part 1: PAD sweep, stillfix ON ==")
    engine = PassivePadEngine(cfg.pad, vlm_scorer=HeuristicVlmPad())
    report["fix_mode"]["stillfix_on"] = run_all(engine, "stillfix")

    orig = PassivePadEngine._is_still_replica
    PassivePadEngine._is_still_replica = staticmethod(lambda frames: False)
    print("== Part 1: PAD sweep, legacy (stillfix OFF) ==")
    report["fix_mode"]["legacy"] = run_all(engine, "legacy")
    PassivePadEngine._is_still_replica = staticmethod(orig)

    print("== Part 2: joint matcher x PAD (same engine, gate off/on) ==")
    pads = (False, True)
    report["joint_matcher_pad"]["face_casia_indomain"] = \
        vm.verify_face_casia_indomain(device, pads=pads)
    print("  face_casia_indomain done")
    report["joint_matcher_pad"]["face_olivetti"] = vm._verify_trained(
        "olivetti", vm.OLIVETTI, device, pads=pads)
    print("  face_olivetti done")
    report["joint_matcher_pad"]["fingerprint_sokocoventry"] = \
        vm._verify_trained("fp", Path("artifacts/dataset/fp_sokocoventry.npz"),
                           device, pads=pads)
    print("  fingerprint done")
    report["joint_matcher_pad"]["iris_unidatapro"] = vm._verify_trained(
        "iris", Path("artifacts/dataset/iris_unidatapro.npz"), device,
        probe_holdout=1, pads=pads)
    print("  iris done")

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved ->", args.out)

    print("\n== JOINT MATCHER x PAD (gate OFF -> gate ON) ==")
    hdr = ("modality", "mode", "FMR", "FNMR", "acc", "block_g", "still_cov")
    print("{:<26}{:>7}{:>7}{:>7}{:>7}{:>8}{:>9}".format(*hdr))
    for k, reps in report["joint_matcher_pad"].items():
        if "error" in reps:
            print("{:<26} ERROR {}".format(k, str(reps["error"])[:36]))
            continue
        for mode in ("pad_0", "pad_1"):
            r = reps.get(mode)
            if r is None or "error" in r:
                continue
            print("{:<26}{:>7}{:>7}{:>7}{:>7}{:>8}{:>9}".format(
                k, mode, r["fmr"], r["fnmr"], r["accuracy"],
                r.get("pad_blocked_genuine", "-"),
                r.get("still_reason_coverage", "-")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
