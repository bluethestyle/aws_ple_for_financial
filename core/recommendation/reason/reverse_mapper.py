"""
Feature Reverse Mapper
=======================

Interprets top-K feature importances (e.g. from Integrated Gradients)
into human-readable natural language descriptions.

All feature ranges, group definitions, and interpretation labels come
from **config** -- not hardcoded.

Config example (``reason.reverse_mapper``)::

    reason:
      reverse_mapper:
        feature_groups:
          profile:
            start: 0
            end: 238
            label: Profile
            description: Unified customer profile features
          multi_source:
            start: 238
            end: 329
            label: Multi-source
            description: Multi-source aggregated features
          domain:
            start: 329
            end: 488
            label: Domain
            description: Domain-specific features
          # ... more groups
        range_labels:
          - min: 0.0
            max: 0.2
            label: very_low
          - min: 0.2
            max: 0.4
            label: low
          - min: 0.4
            max: 0.6
            label: medium
          - min: 0.6
            max: 0.8
            label: high
          - min: 0.8
            max: 1.01
            label: very_high
        interpretation_templates:
          very_low: "{feature_label} is notably low for this customer."
          low: "{feature_label} is below average."
          medium: "{feature_label} is at a moderate level."
          high: "{feature_label} is above average."
          very_high: "{feature_label} is exceptionally high."
        task_interpretation_weights:
          churn:
            profile: 1.2
            domain: 1.5
          ltv:
            profile: 1.0
            multi_source: 1.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ReverseMapper", "FeatureInterpretation"]


@dataclass
class FeatureInterpretation:
    """Interpretation of a single feature.

    Attributes:
        feature_name: Original feature name or index.
        group: Config-defined group the feature belongs to.
        group_label: Human-readable group label.
        value: Raw feature value.
        range_label: Classified range label (e.g. ``"high"``).
        interpretation: Natural language interpretation string.
        ig_score: Importance score (Integrated Gradients or SHAP).
        rank: Importance rank (1 = most important).
    """

    feature_name: str
    group: str
    group_label: str
    value: float
    range_label: str
    interpretation: str
    ig_score: float
    rank: int


class ReverseMapper:
    """Config-driven feature interpretation engine.

    Given a feature vector and importance scores, produces ranked
    human-readable interpretations for the top-K features.

    Args:
        config: Full pipeline config.  Reads ``config["reason"]["reverse_mapper"]``.
        feature_names: Optional list of feature names aligned with vector
                       dimensions.  If ``None``, features are referenced
                       by index (``"feature_42"``).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        feature_names: Optional[List[str]] = None,
    ) -> None:
        rm_cfg = config.get("reason", {}).get("reverse_mapper", {})

        # Feature groups with (start, end) ranges
        self.feature_groups: Dict[str, Dict[str, Any]] = rm_cfg.get(
            "feature_groups", {},
        )

        # Range classification bins
        self.range_labels: List[Dict[str, Any]] = rm_cfg.get(
            "range_labels",
            [
                {"min": 0.0, "max": 0.2, "label": "very_low"},
                {"min": 0.2, "max": 0.4, "label": "low"},
                {"min": 0.4, "max": 0.6, "label": "medium"},
                {"min": 0.6, "max": 0.8, "label": "high"},
                {"min": 0.8, "max": 1.01, "label": "very_high"},
            ],
        )

        # Interpretation templates keyed by range_label
        self.interpretation_templates: Dict[str, str] = rm_cfg.get(
            "interpretation_templates",
            {
                "very_low": "{feature_label} is notably low for this customer.",
                "low": "{feature_label} is below average.",
                "medium": "{feature_label} is at a moderate level.",
                "high": "{feature_label} is above average.",
                "very_high": "{feature_label} is exceptionally high.",
            },
        )

        # Task-specific group weights for re-ranking
        self.task_weights: Dict[str, Dict[str, float]] = rm_cfg.get(
            "task_interpretation_weights", {},
        )

        self.feature_names: Optional[List[str]] = feature_names

        logger.info(
            "ReverseMapper initialised: %d groups, %d range bins",
            len(self.feature_groups),
            len(self.range_labels),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpret_top_k(
        self,
        feature_importances: np.ndarray,
        k: int = 5,
        feature_vector: Optional[np.ndarray] = None,
        task: Optional[str] = None,
    ) -> List[FeatureInterpretation]:
        """Interpret the top-K most important features.

        Args:
            feature_importances: 1-D array of importance scores (e.g. IG).
            k: Number of top features to interpret.
            feature_vector: Optional raw feature vector (same length).
                            Used for range classification.  If ``None``,
                            importance magnitude is used instead.
            task: Optional task type for task-aware re-weighting.

        Returns:
            Sorted list of :class:`FeatureInterpretation`, most important first.
        """
        n = len(feature_importances)
        importances = np.abs(np.asarray(feature_importances, dtype=float))

        # Apply task-specific group weights if available
        if task and task in self.task_weights:
            importances = self._apply_task_weights(importances, task)

        # Get top-K indices
        top_indices = np.argsort(importances)[::-1][:k]

        results: List[FeatureInterpretation] = []
        for rank, idx in enumerate(top_indices, start=1):
            idx = int(idx)
            feat_name = self._get_feature_name(idx)
            group, group_label = self._get_feature_group(idx)

            value = float(feature_vector[idx]) if feature_vector is not None and idx < len(feature_vector) else float(importances[idx])
            range_label = self._classify_range(value)
            interpretation = self._render_interpretation(feat_name, group_label, range_label)

            results.append(FeatureInterpretation(
                feature_name=feat_name,
                group=group,
                group_label=group_label,
                value=value,
                range_label=range_label,
                interpretation=interpretation,
                ig_score=float(importances[idx]),
                rank=rank,
            ))

        return results

    def get_group_summary(
        self,
        feature_vector: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-group summary statistics of a feature vector.

        Returns:
            ``{group_name: {"mean": ..., "std": ..., "max": ..., "min": ...}}``
        """
        summaries: Dict[str, Dict[str, float]] = {}
        for group_name, group_cfg in self.feature_groups.items():
            start = group_cfg.get("start", 0)
            end = group_cfg.get("end", 0)
            if end <= start or start >= len(feature_vector):
                continue
            segment = feature_vector[start: min(end, len(feature_vector))]
            summaries[group_name] = {
                "mean": float(np.mean(segment)),
                "std": float(np.std(segment)),
                "max": float(np.max(segment)),
                "min": float(np.min(segment)),
            }
        return summaries

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_feature_name(self, idx: int) -> str:
        """Return feature name for index, falling back to index-based name."""
        if self.feature_names and idx < len(self.feature_names):
            return self.feature_names[idx]
        return f"feature_{idx}"

    def _get_feature_group(self, idx: int) -> Tuple[str, str]:
        """Determine which config group a feature index belongs to."""
        for group_name, group_cfg in self.feature_groups.items():
            start = group_cfg.get("start", 0)
            end = group_cfg.get("end", 0)
            if start <= idx < end:
                label = group_cfg.get("label", group_name)
                return group_name, label
        return "unknown", "Unknown"

    def _classify_range(self, value: float) -> str:
        """Classify a feature value into a range label using config bins."""
        for bin_def in self.range_labels:
            if bin_def["min"] <= value < bin_def["max"]:
                return bin_def["label"]
        # Edge cases
        if value >= self.range_labels[-1]["max"]:
            return self.range_labels[-1]["label"]
        return self.range_labels[0]["label"]

    def _render_interpretation(
        self,
        feature_name: str,
        group_label: str,
        range_label: str,
    ) -> str:
        """Render a natural language interpretation."""
        template = self.interpretation_templates.get(
            range_label,
            "{feature_label} has a {range_label} value.",
        )
        feature_label = f"{group_label}/{feature_name}"
        try:
            return template.format(
                feature_label=feature_label,
                feature_name=feature_name,
                group_label=group_label,
                range_label=range_label,
            )
        except (KeyError, IndexError):
            return f"{feature_label}: {range_label}"

    def _apply_task_weights(
        self,
        importances: np.ndarray,
        task: str,
    ) -> np.ndarray:
        """Re-weight importances by task-specific group multipliers."""
        weights = self.task_weights.get(task, {})
        if not weights:
            return importances

        weighted = importances.copy()
        for group_name, multiplier in weights.items():
            group_cfg = self.feature_groups.get(group_name, {})
            start = group_cfg.get("start", 0)
            end = group_cfg.get("end", 0)
            if end > start:
                end_clamp = min(end, len(weighted))
                weighted[start:end_clamp] *= multiplier

        return weighted
