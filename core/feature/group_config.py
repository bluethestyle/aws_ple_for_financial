"""
Feature Group Configuration -- backward compatibility shim.

This module re-exports everything from :mod:`core.feature.group`, which is
the single source of truth.  All new code should import directly from
``core.feature.group``.
"""

from core.feature.group import (  # noqa: F401
    FeatureGroupConfig,
    FeatureGroupRegistry,
    FeatureInterpretationConfig,
    InterpretationConfig,
    ContainerConfig,
    load_feature_groups,
)

__all__ = [
    "FeatureGroupConfig",
    "FeatureGroupRegistry",
    "FeatureInterpretationConfig",
    "InterpretationConfig",
    "ContainerConfig",
    "load_feature_groups",
]
