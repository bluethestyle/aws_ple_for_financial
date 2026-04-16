"""Hard-label direct LGBM training for tasks that fail teacher threshold gating.

CLAUDE.md §1.8: tasks below 2x random baseline are trained with hard labels only.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("distill-entry")


def train_direct_lgbm(
    hardlabel_tasks: List[str],
    pipeline_config: Any,
    hard_labels: Dict[str, np.ndarray],
    features: np.ndarray,
    student_config: Any,
    model_output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Train LGBM with hard labels for DIRECT tasks (teacher quality insufficient).

    Args:
        hardlabel_tasks: Task names routed to direct hard-label training.
        pipeline_config: Parsed pipeline config with ``.tasks``.
        hard_labels:     Dict mapping task name -> ground-truth numpy array.
        features:        Feature matrix (N x D) as float32.
        student_config:  StudentConfig with ``lgbm_params`` and ``task_lgbm_overrides``.
        model_output_dir: If provided, save each model immediately after training.

    Returns:
        Dict of {task_name: trained_lgbm_model} for successfully trained tasks.
    """
    if not hardlabel_tasks:
        return {}

    import lightgbm as lgb

    logger.info(
        "  [DIRECT] %d tasks: %s -- retraining with hard labels only",
        len(hardlabel_tasks), hardlabel_tasks,
    )

    direct_models: Dict[str, Any] = {}
    for task_name in hardlabel_tasks:
        t_spec = next((t for t in pipeline_config.tasks if t.name == task_name), None)
        if t_spec is None or task_name not in hard_labels:
            continue

        y = hard_labels[task_name]
        lgbm_params = dict(student_config.lgbm_params)
        task_overrides = student_config.task_lgbm_overrides.get(task_name, {})
        lgbm_params.update(task_overrides)

        if t_spec.type == "binary":
            lgbm_params.setdefault("objective", "binary")
            lgbm_params.setdefault("metric", "auc")
        elif t_spec.type == "multiclass":
            n_classes = getattr(t_spec, "num_classes", None) or int(y.max()) + 1
            lgbm_params["objective"] = "multiclass"
            lgbm_params["num_class"] = n_classes
            lgbm_params.setdefault("metric", "multi_logloss")
        else:
            lgbm_params.setdefault("objective", "regression")
            lgbm_params.setdefault("metric", "mae")

        lgbm_params.setdefault("verbosity", -1)
        n_estimators = lgbm_params.pop("n_estimators", 300)

        ds = lgb.Dataset(features, label=y)
        model = lgb.train(lgbm_params, ds, num_boost_round=n_estimators)
        direct_models[task_name] = model
        logger.info("    %s: direct LGBM trained (%d rounds)", task_name, n_estimators)

        # Incremental checkpoint
        if model_output_dir:
            task_dir = Path(model_output_dir) / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(task_dir / "model.lgbm"))
            logger.info("    checkpoint saved: %s/model.lgbm", task_name)

    return direct_models
