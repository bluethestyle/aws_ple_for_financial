"""
Feature Group Configuration
=============================

Defines the :class:`FeatureGroupConfig` dataclass -- the **single source of
truth** for feature group definitions across the entire system.

Downstream consumers (reverse mapper, template engine, distillation pipeline,
recommendation pipeline) all derive their configuration from these objects,
eliminating manual duplication and ensuring consistency.

Usage::

    import yaml
    from core.feature.group_config import FeatureGroupConfig

    with open("configs/feature_groups.yaml") as f:
        raw = yaml.safe_load(f)

    groups = FeatureGroupConfig.from_yaml_list(raw["feature_groups"])

Or construct programmatically::

    group = FeatureGroupConfig(
        name="base_profile",
        type="transform",
        output_dim=4,
        interpretation=InterpretationConfig(
            category="demographics",
            template="{feature} indicates a {direction} profile",
            narrative_lens="lifecycle",
            primary_tasks=["churn", "ltv"],
        ),
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "FeatureGroupConfig",
    "InterpretationConfig",
    "load_feature_groups",
]


@dataclass
class InterpretationConfig:
    """Interpretation metadata for a single feature group.

    Controls how features in this group are translated into human-readable
    recommendation reasons by the reverse mapper and template engine.

    Attributes:
        category: Semantic category (e.g. ``"demographics"``,
                  ``"behavioral_state"``).  Used as the key in template
                  engine's ``feature_category_map``.
        template: Default interpretation template string with placeholders.
        narrative_lens: Narrative perspective for reason framing (e.g.
                        ``"lifecycle"``, ``"engagement"``, ``"consumption"``).
        primary_tasks: Task types for which this group is most relevant.
                       Used by the reverse mapper to weight feature
                       importance per task.
    """

    category: str = "general"
    template: str = "{feature} has a notable value."
    narrative_lens: str = "general"
    primary_tasks: List[str] = field(default_factory=list)


@dataclass
class FeatureGroupConfig:
    """Single source of truth for a feature group definition.

    Encapsulates everything needed to:

    * Generate features (type, generator/transformer config, output_dim).
    * Interpret features (interpretation templates, category mapping).
    * Distill features (distill flag, distill_weight).
    * Route features to PLE experts (target_experts).

    Attributes:
        name: Unique group identifier (e.g. ``"base_profile"``).
        type: ``"transform"`` (modifies existing columns) or ``"generate"``
              (creates new columns).
        output_dim: Number of output feature dimensions for this group.
        enabled: Whether this group is active in the pipeline.
        columns: Source columns (for transform-type groups).
        output_columns: Explicit output column names.  Auto-generated from
                        ``name`` and ``output_dim`` if not provided.
        transformers: Transformer names to apply (for transform-type groups).
        generator: Generator name (for generate-type groups).
        generator_params: Parameters passed to the generator constructor.
        target_experts: PLE expert towers that consume this group's features.
        interpretation: Interpretation config for the reason engine.
        distill: Whether this group is distillable to an LGBM student.
        distill_weight: Relative importance weight for feature-level
                        distillation loss.  Lower values mean the group
                        is harder to replicate in the student model.
    """

    name: str
    type: str = "transform"
    output_dim: int = 0
    enabled: bool = True

    # Source / output column info
    columns: List[str] = field(default_factory=list)
    output_columns: List[str] = field(default_factory=list)

    # Transform-type config
    transformers: List[str] = field(default_factory=list)

    # Generate-type config
    generator: str = ""
    generator_params: Dict[str, Any] = field(default_factory=dict)

    # PLE routing
    target_experts: List[str] = field(default_factory=list)

    # Interpretation
    interpretation: InterpretationConfig = field(
        default_factory=InterpretationConfig,
    )

    # Distillation
    distill: bool = True
    distill_weight: float = 1.0

    def __post_init__(self) -> None:
        """Auto-generate output_columns if not explicitly provided."""
        if not self.output_columns and self.output_dim > 0:
            self.output_columns = [
                f"{self.name}_{i}" for i in range(self.output_dim)
            ]

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureGroupConfig":
        """Create a FeatureGroupConfig from a plain dict (e.g. YAML-parsed).

        Handles nested ``interpretation`` dict conversion to
        :class:`InterpretationConfig`.

        Args:
            d: Dictionary with feature group fields.

        Returns:
            A fully constructed FeatureGroupConfig.
        """
        data = dict(d)

        # Parse nested interpretation config
        interp_raw = data.pop("interpretation", {})
        if isinstance(interp_raw, dict):
            interp = InterpretationConfig(**interp_raw)
        elif isinstance(interp_raw, InterpretationConfig):
            interp = interp_raw
        else:
            interp = InterpretationConfig()

        return cls(interpretation=interp, **data)

    @classmethod
    def from_yaml_list(
        cls, items: List[Dict[str, Any]]
    ) -> List["FeatureGroupConfig"]:
        """Parse a list of YAML-style dicts into FeatureGroupConfig objects.

        Args:
            items: List of dicts, each describing one feature group.

        Returns:
            List of FeatureGroupConfig instances (preserving order).
        """
        groups = [cls.from_dict(item) for item in items]
        logger.info(
            "Loaded %d feature group configs: %s",
            len(groups),
            [g.name for g in groups],
        )
        return groups

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def compute_range(self, offset: int = 0) -> Tuple[int, int]:
        """Compute the (start, end) index range for this group.

        Args:
            offset: Starting index for this group in the concatenated
                    feature vector.

        Returns:
            Tuple of (start_index, end_index).
        """
        return offset, offset + self.output_dim

    @staticmethod
    def compute_group_ranges(
        groups: List["FeatureGroupConfig"],
    ) -> Dict[str, Tuple[int, int]]:
        """Compute feature index ranges for all enabled groups.

        Args:
            groups: Ordered list of feature group configs.

        Returns:
            Dict mapping group name to (start, end) index tuple.
        """
        ranges: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for group in groups:
            if not group.enabled:
                continue
            start = offset
            end = offset + group.output_dim
            ranges[group.name] = (start, end)
            offset = end
        return ranges

    @staticmethod
    def total_dim(groups: List["FeatureGroupConfig"]) -> int:
        """Compute total feature dimension across all enabled groups.

        Args:
            groups: List of feature group configs.

        Returns:
            Sum of output_dim for all enabled groups.
        """
        return sum(g.output_dim for g in groups if g.enabled)


def load_feature_groups(path: str) -> List[FeatureGroupConfig]:
    """Load feature group configs from a YAML file.

    The YAML file must have a top-level ``feature_groups`` key containing
    a list of group definitions.

    Args:
        path: Path to the YAML file.

    Returns:
        List of FeatureGroupConfig instances.

    Raises:
        FileNotFoundError: If the path does not exist.
        KeyError: If the YAML file has no ``feature_groups`` key.
    """
    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feature groups config not found: {path}")

    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if "feature_groups" not in raw:
        raise KeyError(
            f"YAML file {path} must have a top-level 'feature_groups' key"
        )

    return FeatureGroupConfig.from_yaml_list(raw["feature_groups"])
