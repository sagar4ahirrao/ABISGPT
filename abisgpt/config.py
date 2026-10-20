"""Typed configuration shared by every ABISGPT stage."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

IndexBackend = Literal["faiss", "qdrant", "milvus"]


@dataclass
class IndexConfig:
    """Tier-1 ANN triage configuration (white paper §4).

    `m` and `ef_search` mirror the HNSW knobs described in §4.3; `pq_segments`
    controls Product Quantization (white paper §4.1: 64 sub-vectors of 8 vals).
    """

    backend: IndexBackend = "faiss"
    dim: int = 512
    nlist: int = 256
    nprobe: int = 16
    m: int = 32                 # HNSW edges per node (white paper range 16–32)
    ef_search: int = 128        # dynamic candidate list (white paper range 64–256)
    ef_construction: int = 200
    pq_segments: int = 64       # # of sub-vectors (64 -> 64 bytes per embedding)
    pq_bits: int = 8            # bits per segment code (256 centroids)
    metric: Literal["cosine", "l2", "inner_product"] = "cosine"
    uri: str = "http://localhost:19530"  # qdrant/milvus endpoint

    def validate(self) -> None:
        if self.dim % self.pq_segments != 0:
            raise ValueError(
                f"IndexConfig.dim({self.dim}) must be divisible by pq_segments({self.pq_segments})."
            )
        if not 16 <= self.m <= 64:
            raise ValueError(f"IndexConfig.m must be within 16..32..64, got {self.m}.")
        if not 32 <= self.ef_search <= 1024:
            raise ValueError(f"IndexConfig.ef_search must be within 32..256..1024, got {self.ef_search}.")


@dataclass
class PadConfig:
    """Passive PAD fusion weights (white paper §2)."""

    vlm_weight: float = 0.4
    rppg_weight: float = 0.4
    flash_weight: float = 0.2
    gate_threshold: float = 0.6  # fused score >= threshold => live
    fps: int = 60
    rppg_window_sec: float = 2.0
    deepfake_band: tuple = (0.7, 4.0)  # Hz, plausible human heart-rate band


@dataclass
class PipelineConfig:
    """Two-tier triage policy (white paper §1.3)."""

    top_k: int = 10
    match_threshold: float = 0.72
    batch_size: int = 32
    device: str = "auto"


@dataclass
class AbisGptConfig:
    index: IndexConfig = field(default_factory=IndexConfig)
    pad: PadConfig = field(default_factory=PadConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> AbisGptConfig:
        """Load a YAML tree where keys mirror the dataclass field names."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        root = raw.get("abisgpt", raw)
        cfg = cls()
        for section, cc_dataclass in (
            ("index", cfg.index),
            ("pad", cfg.pad),
            ("pipeline", cfg.pipeline),
        ):
            if section in root and root[section]:
                for key, val in dict(root[section]).items():
                    if hasattr(cc_dataclass, key):
                        setattr(cc_dataclass, key, val)
        cfg.index.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

