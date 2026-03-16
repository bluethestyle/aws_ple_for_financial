"""
Feature engineering layer.

Provides the schema, transformer base class, registry, pipeline, builder,
and built-in transformers for config-driven feature pipelines.

The new feature group system (:class:`FeatureGroupConfig`,
:class:`FeatureGroupRegistry`, :class:`FeatureGroupPipeline`) is the
primary orchestrator for rich feature engineering -- supporting both
feature transformation and feature generation (TDA, HMM, graph
embeddings, multidisciplinary models).

Feature group configuration is the single source of truth for feature
definitions, expert routing, interpretation metadata, and distillation
settings -- propagated automatically to downstream systems.
"""

from .base import AbstractFeatureTransformer, FeatureSchema
from .generator import AbstractFeatureGenerator, FeatureGeneratorRegistry
from .group import (
    FeatureGroupConfig,
    FeatureGroupRegistry,
    FeatureInterpretationConfig,
)
from .group_config import (
    FeatureGroupConfig as LegacyFeatureGroupConfig,
    InterpretationConfig,
    load_feature_groups,
)
from .group_pipeline import FeatureGroupPipeline
from .pipeline import FeaturePipeline
from .pipeline_builder import FeaturePipelineBuilder
from .registry import FeatureRegistry

# Importing transformers triggers their @FeatureRegistry.register decorators
from . import transformers as _transformers  # noqa: F401

# Importing generators triggers their @FeatureGeneratorRegistry.register decorators
from . import generators as _generators  # noqa: F401

__all__ = [
    # Base
    "AbstractFeatureTransformer",
    "FeatureSchema",
    # Generator
    "AbstractFeatureGenerator",
    "FeatureGeneratorRegistry",
    # Feature Group (new)
    "FeatureGroupConfig",
    "FeatureGroupRegistry",
    "FeatureInterpretationConfig",
    "FeatureGroupPipeline",
    # Feature Group (legacy compat)
    "LegacyFeatureGroupConfig",
    "InterpretationConfig",
    "load_feature_groups",
    # Pipeline (legacy)
    "FeaturePipeline",
    "FeaturePipelineBuilder",
    "FeatureRegistry",
]
