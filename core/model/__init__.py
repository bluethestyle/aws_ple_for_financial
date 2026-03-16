"""
Model package — PLE and LGBM model implementations.

LGBM is imported lazily to avoid pandas dependency at module load time.
"""
from .registry import ModelRegistry

# PLE is always available (torch-only)
from .ple import PLEModel

__all__ = ["ModelRegistry", "PLEModel"]


def get_lgbm_model():
    """Lazy import for LGBMModel (requires pandas/lightgbm)."""
    from .lgbm import LGBMModel
    return LGBMModel
