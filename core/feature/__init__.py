"""
Feature engineering layer.

Provides the schema, transformer base class, registry, pipeline, builder,
and built-in transformers for config-driven feature pipelines.
"""

from .base import AbstractFeatureTransformer, FeatureSchema
from .pipeline import FeaturePipeline
from .pipeline_builder import FeaturePipelineBuilder
from .registry import FeatureRegistry

# Importing transformers triggers their @FeatureRegistry.register decorators
from . import transformers as _transformers  # noqa: F401

__all__ = [
    "AbstractFeatureTransformer",
    "FeatureSchema",
    "FeaturePipeline",
    "FeaturePipelineBuilder",
    "FeatureRegistry",
]
