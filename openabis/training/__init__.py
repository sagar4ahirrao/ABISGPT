"""Training utilities (toy biometric encoder mechanics)."""

from openabis.training.trainer import TrainConfig, evaluate_recall, train

__all__ = ["TrainConfig", "train", "evaluate_recall"]


def __getattr__(name):
    if name in ("SyntheticBiometricDataset",):
        from openabis.training.synthdata import SyntheticBiometricDataset

        return SyntheticBiometricDataset
    if name in ("ToyBiometricEncoder", "ArcFaceHead"):
        import openabis.training.model as m

        return getattr(m, name)
    raise AttributeError(f"openabis.training has no attribute {name!r}")
