"""Two-tier identification pipeline.

Tier 1  — constant-time ANN triage over the encrypted vector graph
          (IVF-HNSW-PQ; `O(log N)` traversal, white paper §4.1).
Tier 2  — deterministic verification loop restricted to the top-k
          candidates only (white paper §1.3), using a precision verifier
          such as fingerprint minutiae scoring or an exact cosine re-rank
          on the un-quantized reference embedding.

The PAD gate (white paper §2) runs *before* search — fail-closed.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

from abisgpt.config import AbisGptConfig
from abisgpt.encoder import ViTBiometricEncoder
from abisgpt.index import build_backend
from abisgpt.index.base import VectorIndex
from abisgpt.pad import PassivePadEngine


@dataclass
class Tier1Candidate:
    identity_id: str
    tier1_score: float
    tier2_score: float | None = None


@dataclass
class IdentificationResult:
    query_id: str
    decision: str                     # "match" | "no_match" | "rejected"
    pad_live: bool | None
    pad_scores: dict[str, float] = field(default_factory=dict)
    candidates: list[Tier1Candidate] = field(default_factory=list)
    elapsed_ms: float = 0.0
    tier1_hit_ms: float = 0.0
    tier2_rerank_ms: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def top_hit(self) -> Tier1Candidate | None:
        return self.candidates[0] if self.candidates else None


@dataclass
class EnrolledIdentity:
    identity_id: str
    embedding: np.ndarray             # 512-D, L2-normalized reference template


class PrecisionVerifier:
    """Tier-2 verifier.

    Default implementation is exact cosine on the un-quantized reference
    template (no PQ loss). Swap in a minutiae scorer for fingerprint
    roll-off in production deployments (white paper §1.3).
    """

    def __init__(self, gallery: dict[str, np.ndarray]):
        self.gallery = gallery

    def score(self, query: np.ndarray, identity_id: str) -> float:
        ref = self.gallery.get(identity_id)
        if ref is None:
            return 0.0
        q = query / max(float(np.linalg.norm(query)), 1e-9)
        r = ref / max(float(np.linalg.norm(ref)), 1e-9)
        return float(np.dot(q, r))


class TwoTierPipeline:
    """Full biometric identification path with PAD gating."""

    def __init__(
        self,
        encoder: ViTBiometricEncoder | None = None,
        pad: PassivePadEngine | None = None,
        index_backend: VectorIndex | None = None,
        config: AbisGptConfig | None = None,
    ):
        self.config = config or AbisGptConfig()
        self.encoder = encoder or ViTBiometricEncoder(strategy="hash")
        self.pad = pad or PassivePadEngine(self.config.pad)
        self.index = index_backend or build_backend(self.config.index)
        self._gallery: dict[str, np.ndarray] = {}
        self.verifier = PrecisionVerifier(self._gallery)

    # ------------------------------------------------------------------ #
    # Enrollment
    # ------------------------------------------------------------------ #
    def encode(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """Multi-frame template average, L2-normalized (pipeline contract)."""
        emb = self.encoder.encode(list(frames)).mean(axis=0)
        emb = emb / max(float(np.linalg.norm(emb)), 1e-9)
        return emb

    def enroll(self, identity_id: str, images: Sequence[np.ndarray]) -> EnrolledIdentity:
        emb = self.encode(images)
        self.index.add([identity_id], emb[None, :])
        self._gallery[identity_id] = emb
        return EnrolledIdentity(identity_id=identity_id, embedding=emb)

    # ------------------------------------------------------------------ #
    # Identification
    # ------------------------------------------------------------------ #
    def identify(self, frames: Sequence[np.ndarray], top_k: int | None = None,
                 pad: bool = True) -> IdentificationResult:
        """Run PAD gate -> 512-D encode -> Tier-1 triage -> Tier-2 rerank.

        `pad=False` is the research/photo path for still captures that
        cannot carry a pulse: the gate is skipped (documented in the
        result as `pad_live=None`) — never silently rejected.
        """
        top_k = top_k or self.config.pipeline.top_k
        t_start = time.perf_counter()

        # ---- Gate 0: passive PAD (fail-closed) --------------------------- #
        pad_scores: dict[str, float] = {}
        if pad:
            pad_verdict = self.pad.evaluate(frames)
            pad_scores = pad_verdict.tier_scores
            if not pad_verdict.is_live:
                return IdentificationResult(
                    query_id="probe",
                    decision="rejected",
                    pad_live=False,
                    pad_scores=pad_verdict.tier_scores,
                    elapsed_ms=(time.perf_counter() - t_start) * 1000,
                )

        # ---- Encode: multi-frame template averaging (same as enrollment) -- #
        if not frames:
            raise ValueError("TwoTierPipeline.identify requires at least one frame.")
        emb = self.encoder.encode(list(frames)).mean(axis=0)
        emb = emb / max(float(np.linalg.norm(emb)), 1e-9)

        # ---- Tier 1: ANN vector triage ------------------------------------ #
        t1 = time.perf_counter()
        hits = self.index.search(emb, top_k=top_k)
        tier1_ms = (time.perf_counter() - t1) * 1000

        # ---- Tier 2: precision rerank on top-k only ----------------------- #
        t2 = time.perf_counter()
        candidates = [
            Tier1Candidate(identity_id=hit_id, tier1_score=score)
            for hit_id, score in hits
        ]
        for cand in candidates:
            cand.tier2_score = self.verifier.score(emb, cand.identity_id)
        candidates.sort(key=lambda c: (-c.tier2_score, -c.tier1_score))
        tier2_ms = (time.perf_counter() - t2) * 1000

        best = candidates[0] if candidates else None
        decision = "no_match"
        if best and (best.tier2_score or 0.0) >= self.config.pipeline.match_threshold:
            decision = "match"

        return IdentificationResult(
            query_id="probe",
            decision=decision,
            pad_live=pad_verdict.is_live if pad else None,
            pad_scores=pad_scores,
            candidates=candidates,
            tier1_hit_ms=tier1_ms,
            tier2_rerank_ms=tier2_ms,
            elapsed_ms=(time.perf_counter() - t_start) * 1000,
        )

    # ------------------------------------------------------------------ #
    # Management
    # ------------------------------------------------------------------ #
    def drop(self, identity_id: str) -> int:
        """GDPR Art. 17 right-to-erasure hook: remove from both tiers."""
        self._gallery.pop(identity_id, None)
        return self.index.remove([identity_id])

    def identities(self) -> list[str]:
        """Enrolled identity ids (registry listing)."""
        return sorted(self._gallery)

    def template(self, identity_id: str) -> np.ndarray:
        """The stored L2-normalized template (`KeyError` when unknown)."""
        return self._gallery[identity_id]

    def __repr__(self) -> str:  # pragma: no cover
        return f"TwoTierPipeline(index={type(self.index).__name__}, enrolled={len(self)})"

