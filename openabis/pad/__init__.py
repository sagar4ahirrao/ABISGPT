"""Passive Presentation Attack Detection (white paper §2)."""

from openabis.pad.fusion import PassivePadEngine, PadVerdict

__all__ = ["PassivePadEngine", "PadVerdict", "RppgAnalyzer", "FlashDiffusionAnalyzer"]


def __getattr__(name):
    # Lazy sub-module re-exports keep `import openabis.pad` cheap.
    if name == "RppgAnalyzer":
        from openabis.pad.rppg import RppgAnalyzer

        return RppgAnalyzer
    if name == "FlashDiffusionAnalyzer":
        from openabis.pad.flashdiff import FlashDiffusionAnalyzer

        return FlashDiffusionAnalyzer
    raise AttributeError(f"module 'openabis.pad' has no attribute {name!r}")
