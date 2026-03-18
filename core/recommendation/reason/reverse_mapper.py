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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from core.feature.group_config import FeatureGroupConfig

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

        # Financial glossary lookup tables (populated by _load_glossary)
        self._glossary_features: Dict[str, Dict[str, Any]] = {}
        self._glossary_group_descriptions: Dict[str, str] = {}
        self._glossary_task_weights: Dict[str, Dict[str, Any]] = {}

        # Task-specific text interpretations: {feature_id: {task: text}}
        # Populated by _load_glossary from task_interpretations section
        self._task_interpretations: Dict[str, Dict[str, str]] = {}

        logger.info(
            "ReverseMapper initialised: %d groups, %d range bins",
            len(self.feature_groups),
            len(self.range_labels),
        )

    # ------------------------------------------------------------------
    # Auto-configuration from FeatureGroupConfig
    # ------------------------------------------------------------------

    @classmethod
    def from_feature_groups(
        cls,
        groups: List["FeatureGroupConfig"],
        feature_names: Optional[List[str]] = None,
        range_labels: Optional[List[Dict[str, Any]]] = None,
        interpretation_templates: Optional[Dict[str, str]] = None,
    ) -> "ReverseMapper":
        """Auto-build a ReverseMapper from feature group definitions.

        Instead of manually defining ranges and templates in config YAML,
        this reads FeatureGroupConfig objects (the single source of truth)
        and generates:

        * Feature group ranges (start_idx, end_idx per group).
        * Per-group interpretation templates and labels.
        * Task relevance mapping for task-aware re-weighting.

        This ensures the feature pipeline and interpretation layer are
        always in sync -- no manual config duplication needed.

        Args:
            groups: Ordered list of FeatureGroupConfig instances.
            feature_names: Optional feature name list aligned with the
                           concatenated feature vector.  If ``None``,
                           auto-generated from group output_columns.
            range_labels: Optional custom range classification bins.
                          Uses sensible defaults if not provided.
            interpretation_templates: Optional custom templates keyed by
                                      range label.  Uses defaults if not
                                      provided.

        Returns:
            A fully configured ReverseMapper instance.
        """
        from core.feature.group_config import FeatureGroupConfig

        # Build feature_groups config section
        feature_groups_cfg: Dict[str, Dict[str, Any]] = {}
        task_weights: Dict[str, Dict[str, float]] = {}
        offset = 0
        all_feature_names: List[str] = []

        for group in groups:
            if not group.enabled:
                continue

            start = offset
            end = offset + group.output_dim

            feature_groups_cfg[group.name] = {
                "start": start,
                "end": end,
                "label": group.interpretation.category.replace("_", " ").title(),
                "description": group.interpretation.template,
                "template": group.interpretation.template,
                "category": group.interpretation.category,
                "narrative_lens": group.interpretation.narrative_lens,
            }

            # Build task-specific weights: each group has primary_tasks where
            # it is most relevant (weight=1.5); other tasks get default (1.0).
            for task in group.interpretation.primary_tasks:
                if task not in task_weights:
                    task_weights[task] = {}
                task_weights[task][group.name] = 1.5

            # Collect feature names from output_columns
            all_feature_names.extend(group.output_columns)

            offset = end

        # Assemble the config dict in the format ReverseMapper expects
        config: Dict[str, Any] = {
            "reason": {
                "reverse_mapper": {
                    "feature_groups": feature_groups_cfg,
                    "task_interpretation_weights": task_weights,
                },
            },
        }

        if range_labels is not None:
            config["reason"]["reverse_mapper"]["range_labels"] = range_labels

        if interpretation_templates is not None:
            config["reason"]["reverse_mapper"][
                "interpretation_templates"
            ] = interpretation_templates

        # Use provided feature_names, or auto-generated ones
        resolved_names = feature_names if feature_names is not None else (
            all_feature_names if all_feature_names else None
        )

        instance = cls(config=config, feature_names=resolved_names)
        logger.info(
            "ReverseMapper.from_feature_groups: %d groups, total_dim=%d, "
            "%d task weight sets",
            len(feature_groups_cfg),
            offset,
            len(task_weights),
        )
        return instance

    # ------------------------------------------------------------------
    # Glossary-based construction and interpretation
    # ------------------------------------------------------------------

    @classmethod
    def from_glossary(
        cls,
        glossary_path: str,
        feature_groups_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "ReverseMapper":
        """Create a ReverseMapper with financial glossary loaded from YAML.

        The glossary provides Korean financial business language templates
        for each feature, enabling human-readable interpretations like
        ``"월 평균 47건 거래"`` instead of generic ``"rfm_002: high"``.

        Args:
            glossary_path: Path to ``feature_glossary.yaml``.
            feature_groups_path: Optional path to ``feature_groups.yaml``
                for auto-deriving group index ranges and feature names.
                If ``None``, group ranges must be set via the base config
                or ``from_feature_groups``.
            feature_names: Optional explicit feature name list.

        Returns:
            A fully configured ReverseMapper with glossary loaded.
        """
        if yaml is None:
            raise ImportError(
                "PyYAML is required for glossary loading. "
                "Install it with: pip install pyyaml"
            )

        # Build base config from feature_groups.yaml if provided
        config: Dict[str, Any] = {"reason": {"reverse_mapper": {}}}
        resolved_names: Optional[List[str]] = feature_names

        if feature_groups_path and os.path.exists(feature_groups_path):
            with open(feature_groups_path, "r", encoding="utf-8") as fh:
                fg_data = yaml.safe_load(fh)

            feature_groups_cfg: Dict[str, Dict[str, Any]] = {}
            all_names: List[str] = []
            offset = 0

            for group_def in fg_data.get("feature_groups", []):
                name = group_def["name"]
                dim = group_def.get("output_dim", 0)
                feature_groups_cfg[name] = {
                    "start": offset,
                    "end": offset + dim,
                    "label": name.replace("_", " ").title(),
                    "description": group_def.get("interpretation", {}).get(
                        "template", ""
                    ),
                }
                # Collect column names for feature_names
                for col in group_def.get("columns", []):
                    all_names.append(col)
                offset += dim

            config["reason"]["reverse_mapper"]["feature_groups"] = (
                feature_groups_cfg
            )
            if resolved_names is None and all_names:
                resolved_names = all_names

        instance = cls(config=config, feature_names=resolved_names)
        instance._load_glossary(glossary_path)

        logger.info(
            "ReverseMapper.from_glossary: loaded %d glossary features, "
            "%d group descriptions, %d task weight sets",
            len(instance._glossary_features),
            len(instance._glossary_group_descriptions),
            len(instance._glossary_task_weights),
        )
        return instance

    def _load_glossary(self, path: str) -> None:
        """Load feature glossary YAML and build lookup tables.

        Populates:
            - ``_glossary_features``: ``{feature_id: {name, template, unit, direction}}``
            - ``_glossary_group_descriptions``: ``{group_name: description}``
            - ``_glossary_task_weights``: ``{task: {priority_groups, weight_overrides}}``
        """
        if yaml is None:
            raise ImportError(
                "PyYAML is required for glossary loading. "
                "Install it with: pip install pyyaml"
            )

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        # Parse feature_groups section
        for group_name, group_def in data.get("feature_groups", {}).items():
            desc = group_def.get("description", "")
            self._glossary_group_descriptions[group_name] = desc

            for feat_id, feat_def in group_def.get("features", {}).items():
                self._glossary_features[feat_id] = {
                    "name": feat_def.get("name", feat_id),
                    "template": feat_def.get("template", "{value}"),
                    "unit": feat_def.get("unit", ""),
                    "direction": feat_def.get("direction", ""),
                    "group": group_name,
                    "group_description": desc,
                }

        # Parse task_feature_weights section
        for task_name, task_def in data.get("task_feature_weights", {}).items():
            self._glossary_task_weights[task_name] = {
                "priority_groups": task_def.get("priority_groups", []),
                "weight_overrides": task_def.get("weight_overrides", {}),
            }

        # Parse task_interpretations section:
        # {feature_id: {task: "Korean text interpretation"}}
        self._task_interpretations = {}
        for feat_id, task_map in data.get("task_interpretations", {}).items():
            if isinstance(task_map, dict):
                self._task_interpretations[feat_id] = task_map

        logger.debug(
            "Glossary loaded: %d features across %d groups, "
            "%d task-specific interpretations",
            len(self._glossary_features),
            len(self._glossary_group_descriptions),
            len(self._task_interpretations),
        )

    def interpret_financial(
        self,
        feature_name: str,
        value: float,
        task: Optional[str] = None,
    ) -> str:
        """Return a financial business language interpretation of a feature.

        When ``task`` is provided and the glossary contains a
        ``task_interpretations`` entry for that feature+task pair,
        the task-specific Korean text is returned instead of the
        generic template.  This is critical for LLM input material:
        the same TDA topology change means "이탈 위험" for churn
        but "성장 잠재력" for ltv.

        Lookup priority:
            1. Exact ``feature_name`` + ``task`` in task_interpretations
            2. Prefix match (``tda_short_`` etc.) + ``task``
            3. Generic glossary template with ``{value}`` substitution
            4. Range-label fallback

        Examples::

            interpret_financial("tda_short_001", 0.8, task="churn")
            # => "최근 소비 패턴에 급격한 구조 변화가 감지되어 이탈 위험 신호로 분석됩니다"

            interpret_financial("tda_short_001", 0.8, task="ltv")
            # => "소비 영역이 새롭게 확장되는 패턴이 감지되어 성장 잠재력이 높습니다"

            interpret_financial("rfm_002", 47.0)
            # => "월 평균 47건 거래"

        Args:
            feature_name: Feature identifier (e.g. ``"rfm_002"``).
            value: Raw or transformed feature value.
            task: Task name for task-specific interpretation.

        Returns:
            Korean financial language string.
        """
        # 1. Task-specific text interpretation (exact match or prefix)
        if task and self._task_interpretations:
            task_text = self._lookup_task_interpretation(feature_name, task)
            if task_text:
                return task_text

        # 2. Generic glossary template with value substitution
        glossary_entry = self._glossary_features.get(feature_name)
        if glossary_entry is not None:
            template = glossary_entry["template"]
            try:
                if isinstance(value, float):
                    if value == int(value):
                        display_value = str(int(value))
                    elif abs(value) >= 100:
                        display_value = f"{value:,.0f}"
                    elif abs(value) >= 1:
                        display_value = f"{value:.1f}"
                    else:
                        display_value = f"{value:.2f}"
                else:
                    display_value = str(value)

                return template.format(value=display_value)
            except (KeyError, IndexError, ValueError):
                return f"{glossary_entry['name']}: {value}"

        # 3. Fallback: generic range-label interpretation
        range_label = self._classify_range(value)
        _, group_label = self._get_feature_group_by_name(feature_name)
        return self._render_interpretation(feature_name, group_label, range_label)

    def _lookup_task_interpretation(
        self,
        feature_name: str,
        task: str,
    ) -> Optional[str]:
        """Look up task-specific interpretation text.

        Tries exact feature_name match first, then prefix match
        (e.g. ``graph_embed`` matches ``graph_embed_001``).
        """
        # Exact match
        entry = self._task_interpretations.get(feature_name, {})
        if task in entry:
            return entry[task]

        # Prefix match: "tda_short_001" → check "tda_short_" prefix entries
        for key, task_map in self._task_interpretations.items():
            if feature_name.startswith(key) and task in task_map:
                return task_map[task]

        return None

    def _get_feature_group_by_name(
        self,
        feature_name: str,
    ) -> Tuple[str, str]:
        """Determine feature group by feature name (glossary or feature_names list)."""
        # Check glossary first
        glossary_entry = self._glossary_features.get(feature_name)
        if glossary_entry is not None:
            group = glossary_entry["group"]
            label = self._glossary_group_descriptions.get(group, group)
            return group, label

        # Fall back to index-based lookup
        if self.feature_names:
            try:
                idx = self.feature_names.index(feature_name)
                return self._get_feature_group(idx)
            except ValueError:
                pass

        return "unknown", "Unknown"

    def get_glossary_task_weights(
        self,
        task: str,
    ) -> Dict[str, float]:
        """Return glossary-defined weight overrides for a given task.

        These can be used by callers to re-weight feature importances
        before calling ``interpret_top_k``.

        Returns:
            ``{group_name: weight_multiplier}`` dict, empty if task unknown.
        """
        task_def = self._glossary_task_weights.get(task, {})
        return task_def.get("weight_overrides", {})

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
            interpretation = self._render_interpretation(feat_name, group_label, range_label, value=value)

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
        value: Optional[float] = None,
    ) -> str:
        """Render a natural language interpretation.

        When a glossary is loaded and the feature has a Korean template,
        uses the financial business language template.  Otherwise falls
        back to generic range-label interpretation.
        """
        # Try glossary-based financial interpretation first
        if self._glossary_features and value is not None:
            glossary_entry = self._glossary_features.get(feature_name)
            if glossary_entry is not None:
                return self.interpret_financial(feature_name, value)

        # Fallback: generic range-label template
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
