"""Per-task feature selection via LGBM gain importance.

Feature selection during distillation uses the student LGBM model's gain
importance — the actual serving model's perspective on feature value.

Teacher IG attribution is reserved for:
  - Offline analysis (paper experiments, one-time runs)
  - Recommendation reason pipeline (Agent 1 FactExtractor, at serving time)
It is NOT run in the distillation pipeline to avoid OOM and redundant computation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger("distill-entry")


def run_feature_selection(
    pipeline_config: Any,
    students: Dict[str, Any],
    features: np.ndarray,
    feature_cols: List[str],
    trainer: Any,
    config_path: str,
    distillation_cfg: dict,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run per-task feature selection using LGBM gain importance.

    Args:
        pipeline_config:  Parsed pipeline config with ``.tasks``.
        students:         Dict of {task_name: lgbm_model}.
        features:         Feature matrix (N x D).
        feature_cols:     Ordered list of feature column names.
        trainer:          StudentTrainer (unused, kept for interface compat).
        config_path:      Path to pipeline.yaml (unused, kept for interface compat).
        distillation_cfg: ``distillation`` sub-dict from pipeline YAML.

    Returns:
        (feature_selections, ig_raw_scores)
        ``feature_selections`` maps task_name -> FeatureSelectionResult.
        ``ig_raw_scores`` is always empty (IG not run in distillation pipeline).
    """
    from core.training.feature_selector import (
        FeatureSelector,
        FeatureSelectionConfig,
        FeatureSelectionResult,
    )

    _fs_cfg_raw: dict = distillation_cfg.get("feature_selection", {})
    _cumulative_threshold: float = float(_fs_cfg_raw.get("cumulative_threshold", 0.95))

    logger.info("Running feature selection (LGBM gain, threshold=%.2f)...", _cumulative_threshold)

    fs_config = FeatureSelectionConfig(cumulative_threshold=_cumulative_threshold)
    feature_selector = FeatureSelector(config=fs_config)

    feature_selections: Dict[str, Any] = {}

    for task_spec in pipeline_config.tasks:
        task_name = task_spec.name
        if task_name not in students:
            continue

        student_model = students[task_name]

        pruned_indices = feature_selector.prune_by_lgbm(
            lgbm_model=student_model, feature_names=feature_cols,
        )
        pruned_names = [feature_cols[i] for i in pruned_indices]
        lgbm_gains = student_model.feature_importance(importance_type="gain")

        selection_result = FeatureSelectionResult(
            task_name=task_name,
            original_count=len(feature_cols),
            selected_count=len(pruned_indices),
            reduction_pct=round((1 - len(pruned_indices) / len(feature_cols)) * 100, 1),
            cumulative_threshold_used=_cumulative_threshold,
            selection_method="lgbm_gain",
            selected_indices=sorted(pruned_indices),
            selected_names=pruned_names,
            feature_importances={
                pruned_names[i]: float(lgbm_gains[pruned_indices[i]])
                for i in range(min(50, len(pruned_indices)))
            },
            mandatory_included=[],
        )

        feature_selections[task_name] = selection_result
        logger.info(
            "  %s: %d/%d features selected (%.1f%% reduction)",
            task_name,
            selection_result.selected_count,
            selection_result.original_count,
            selection_result.reduction_pct,
        )

    return feature_selections, {}
