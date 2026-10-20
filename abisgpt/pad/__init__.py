"""Passive Presentation Attack Detection (white paper §2)."""

from abisgpt.pad.fusion import PadVerdict, PassivePadEngine

__all__ = ["PassivePadEngine", "PadVerdict", "RppgAnalyzer", "FlashDiffusionAnalyzer"]


def __getattr__(name):
    # Lazy sub-module re-exports keep `import abisgpt.pad` cheap.
    if name == "RppgAnalyzer":
        from abisgpt.pad.rppg import RppgAnalyzer

        return RppgAnalyzer
    if name == "FlashDiffusionAnalyzer":
        from abisgpt.pad.flashdiff import FlashDiffusionAnalyzer

        return FlashDiffusionAnalyzer
    raise AttributeError(f"module 'abisgpt.pad' has no attribute {name!r}")

