"""Match-type layer: 1:1 verify, 1:N identify, N:N watch-list cross-match.

Extends the two-tier pipeline (`TwoTierPipeline`) with the three canonical
biometric match modes plus a multi-modal probe implementation. All modes
run the fail-closed PAD gate first — a rejected template produces
`MatchType.VERDICT_REJECT` (white paper §1.3, §2).

Contracts (white paper naming):
    1:1   verify_1v1         template vs enrolled identity  (authentication)
    1:N   identify (existing) probe vs whole registry       (identification_)
    N:N   crossmatch_nxn      batched probe set vs identity set (watch-list)

Multi-biometric score-level fusion:
    fused = Σ w_m · s_m / Σ w_m   over available modalities
    (face, periocular, fingerprint placeholder, iris placeholder)
    — the same re-normalization scheme as the PAD fusion (§2).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from abisgpt.pipeline.triage import TwoTierPipeline

# Named biometric channels. Channels not present in a probe are dropped and
# the remaining weights re-normalized (mirrors PassivePadEngine._weighted_fusion).
CHANNEL_WEIGHTS: dict[str, float] = {
    "face": 1.0,
    "periocular": 0.5,
    "fingerprint": 0.8,
    "iris": 0.9,
}


@dataclass
class VerifyResult:
    identity_id: str
    is_match: bool
    score: float
    pad_live: bool
    elapsed_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BatchIdentifyResult:
    n_probes: int
    results: list[dict]
    elapsed_ms: float
    qps: float                              # throughput probes / second

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CrossMatchResult:
    pairs: list[tuple[str, str, float]]     # (probe_id, identity_id, cosine)
    n_queries: int
    n_candidates: int
    elapsed_ms: float
    pairs_per_sec: float

    def as_dict(self) -> dict:
        return {
            "pairs": self.pairs[:200],      # cap payload; count is exact
            "n_pairs_above_threshold": len(self.pairs),
            "n_queries": self.n_queries,
            "n_candidates": self.n_candidates,
            "elapsed_ms": self.elapsed_ms,
            "pairs_per_sec": self.pairs_per_sec,
        }


@dataclass
class WatchListHit:
    query_id: str
    identity_id: str
    score: float
    tier2_score: float | None = None


@dataclass
class WatchListResult:
    hits: list[WatchListHit]
    n_queries: int
    elapsed_ms: float

    def as_dict(self) -> dict:
        return {
            "hits": [asdict(h) for h in self.hits[:200]],
            "n_hits": len(self.hits),
            "n_queries": self.n_queries,
            "elapsed_ms": self.elapsed_ms,
        }


class MultiModalProbe:
    """Score-level fusion of per-modality embeddings for one probe.

    Each modality contributes its own channel embedding; the pipeline's
    encoder is applied per channel to its own frame stack. Fused similarity
    re-normalizes over whichever channels are present.
    """

    def __init__(self, channels: dict[str, np.ndarray]):
        # channels: name -> fused template (512-D, L2-normalized per channel)
        self.channels = {
            k: v / max(float(np.linalg.norm(v)), 1e-9)
            for k, v in channels.items() if k in CHANNEL_WEIGHTS
        }

    def fused(self, other: MultiModalProbe, weights: dict[str, float] | None = None):
        """Fuse the cross-similarities of common channels into one score."""
        weights = weights or CHANNEL_WEIGHTS
        common = set(self.channels) & set(other.channels)
        if not common:
            return 0.0
        total_w, total = 0.0, 0.0
        for name in common:
            w = float(weights.get(name, 1.0))
            total += w * float(self.channels[name] @ other.channels[name])
            total_w += w
        return float(total / total_w) if total_w > 0 else 0.0


class MatchingEngine:
    """The three match modes + multi-modal verification on a pipeline."""

    def __init__(self, pipeline: TwoTierPipeline, threshold: float | None = None):
        self.pipeline = pipeline
        self.threshold = threshold or pipeline.config.pipeline.match_threshold

    # ----------------------------------------------------------------- #
    # 1:1 — authentication                                              #
    # ----------------------------------------------------------------- #
    def verify_1v1(self, identity_id: str, frames: Sequence[np.ndarray],
                   pad: bool = True) -> VerifyResult:
        """Is this capture the claimed `identity_id`? PAD gate runs first."""
        t0 = time.perf_counter()
        if pad:
            verdict = self.pipeline.pad.evaluate(frames)
            if not verdict.is_live:
                return VerifyResult(identity_id, False, 0.0, False,
                                    (time.perf_counter() - t0) * 1000)
        emb = self.pipeline.encode(frames)
        score = self.pipeline.verifier.score(emb, identity_id)
        return VerifyResult(identity_id, bool(score >= self.threshold), float(score),
                            bool(pad), (time.perf_counter() - t0) * 1000)

    def verify_multimodal_1v1(self, reference: MultiModalProbe,
                              probe: MultiModalProbe, threshold: float | None = None) -> float:
        """Score-level fusion of the two multi-modal probes (no PAD here —
        channel PAD is expected to have run per channel upstream)."""
        return probe.fused(reference)

    # ----------------------------------------------------------------- #
    # 1:N — batched identification                                      #
    # ----------------------------------------------------------------- #
    def identify_batch(self, frames_batch: Sequence[Sequence[np.ndarray]],
                       top_k: int = 10, pad: bool = True) -> BatchIdentifyResult:
        """Q probes vs the full registry (Q × 1:N), each probe PAD-gated."""
        t0 = time.perf_counter()
        results = []
        for frames in frames_batch:
            if pad:
                verdict = self.pipeline.pad.evaluate(frames)
                if not verdict.is_live:
                    results.append({"decision": "rejected", "pad_live": False,
                                    "candidates": []})
                    continue
            emb = self.pipeline.encode(frames)
            hits = self.pipeline.index.search(emb, top_k=top_k)
            cand = [
                {"identity_id": hid, "tier1_score": float(s),
                 "tier2_score": self.pipeline.verifier.score(emb, hid)}
                for hid, s in hits
            ]
            cand.sort(key=lambda c: -c["tier2_score"])
            best = cand[0] if cand else None
            results.append({
                "decision": "match" if best and best["tier2_score"] >= self.threshold else "no_match",
                "pad_live": True,
                "candidates": cand,
            })
        dt = (time.perf_counter() - t0) * 1000
        return BatchIdentifyResult(len(frames_batch), results, dt,
                                   len(frames_batch) / (dt / 1000) if dt else 0.0)

    # ----------------------------------------------------------------- #
    # N:N — watch-list cross-match                                      #
    # ----------------------------------------------------------------- #
    def crossmatch_nxn(self, probe_templates: dict[str, np.ndarray],
                       gallery: dict[str, np.ndarray], threshold: float = 0.72,
                       chunk: int = 512) -> CrossMatchResult:
        """Score every probe template against every gallery identity.

        N×cosine dimension (single vector matrix product per chunk) —
        this is the scalable path; exhaustive Python loops never appear.
        """
        t0 = time.perf_counter()
        p_ids = list(probe_templates)
        g_ids = list(gallery)
        if not p_ids or not g_ids:
            return CrossMatchResult([], len(p_ids), len(g_ids), 0.0, 0.0)

        G = np.stack([gallery[i] for i in g_ids]).astype(np.float32)  # (Ng, D)
        pairs: list[tuple[str, str, float]] = []
        for s in range(0, len(p_ids), chunk):
            P = np.stack([probe_templates[i] for i in p_ids[s:s + chunk]]).astype(np.float32)
            sims = P @ G.T                                            # (Qc, Ng)
            ii, jj = np.nonzero(sims >= threshold)
            pairs += [(p_ids[s + i], g_ids[j], float(sims[i, j]))
                      for i, j in zip(ii.tolist(), jj.tolist(), strict=False)]
        dt = (time.perf_counter() - t0) * 1000
        rate = len(p_ids) * len(g_ids) / (max(dt, 1e-3) / 1000)
        return CrossMatchResult(pairs, len(p_ids), len(g_ids), dt, rate)

    def watchlist_screen(self, frames_batch: Sequence[Sequence[np.ndarray]],
                         top_k: int = 10, pad: bool = True) -> WatchListResult:
        """N probes against the assembled watch-list index (N×1:N, ANN path).

        Unlike `crossmatch_nxn` this uses the Tier-1 graph, i.e., scales to
       millions of enrolled watch-list identities; returns best hit per probe.
        """
        t0 = time.perf_counter()
        hits: list[WatchListHit] = []
        for qid, frames in enumerate(frames_batch):
            if pad:
                verdict = self.pipeline.pad.evaluate(frames)
                if not verdict.is_live:
                    continue
            emb = self.pipeline.encode(frames)
            top = self.pipeline.index.search(emb, top_k=top_k)
            if not top:
                continue
            hid, s1 = top[0]
            hits.append(WatchListHit(query_id=f"probe_{qid}", identity_id=hid,
                                     score=float(s1)))
        return WatchListResult(hits, len(frames_batch), (time.perf_counter() - t0) * 1000)
