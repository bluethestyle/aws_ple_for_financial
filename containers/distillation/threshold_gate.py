"""Teacher threshold gating for adaptive distillation.

CLAUDE.md §1.8: tasks below 2x random baseline use hard labels instead of soft.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger("distill-entry")


def evaluate_teacher_thresholds(
    pipeline_config: Any,
    soft_labels: Dict[str, Any],
    hard_labels: Dict[str, np.ndarray],
    distillation_cfg: dict,
) -> Tuple[List[str], List[str], List[str]]:
    """Classify tasks into distill / hard-label / skip buckets.

    Args:
        pipeline_config:  Parsed pipeline config with ``.tasks``.
        soft_labels:      Dict mapping task name -> teacher soft prediction tensor.
        hard_labels:      Dict mapping task name -> numpy ground-truth array.
        distillation_cfg: ``distillation`` sub-dict from pipeline YAML.

    Returns:
        (distill_tasks, hardlabel_tasks, skip_tasks)
    """
    from sklearn.metrics import roc_auc_score, f1_score, r2_score

    _thresh_cfg = distillation_cfg.get("teacher_threshold", {})
    _floor_binary_auc = _thresh_cfg.get("floor_binary_auc", 0.52)
    _floor_mc_f1_ratio = _thresh_cfg.get("floor_multiclass_f1_ratio", 1.0)
    _floor_r2 = _thresh_cfg.get("floor_regression_r2", 0.01)

    logger.info("=" * 60)
    logger.info("Teacher threshold check (2x random baseline + floor)")
    logger.info("=" * 60)

    distill_tasks: List[str] = []
    hardlabel_tasks: List[str] = []
    skip_tasks: List[str] = []

    for t in pipeline_config.tasks:
        task_name = t.name
        task_type = t.type
        teacher_soft = soft_labels.get(task_name)
        hard = hard_labels.get(task_name)

        if teacher_soft is None or hard is None:
            logger.warning("  %s: no soft/hard labels -- skipping", task_name)
            continue

        teacher_np = teacher_soft.numpy() if hasattr(teacher_soft, "numpy") else np.array(teacher_soft)
        hard_np = hard if isinstance(hard, np.ndarray) else np.array(hard)

        viable = False
        metric_str = ""

        try:
            if task_type == "binary":
                preds = teacher_np.flatten()
                labels_np = hard_np.flatten()
                valid = (labels_np == 0) | (labels_np == 1)
                if valid.sum() > 10 and len(set(labels_np[valid].tolist())) == 2:
                    auc = roc_auc_score(labels_np[valid], preds[valid])
                    viable = auc > 0.60
                    metric_str = f"AUC={auc:.4f} (threshold=0.60)"

            elif task_type == "multiclass":
                n_classes = getattr(t, "num_classes", None) or int(hard_np.max()) + 1
                threshold = 2.0 / n_classes
                if teacher_np.ndim > 1 and teacher_np.shape[-1] > 1:
                    pred_classes = teacher_np.argmax(axis=-1)
                else:
                    pred_classes = teacher_np.flatten().astype(int)
                true_classes = hard_np.flatten().astype(int)
                valid = true_classes >= 0
                if valid.sum() > 10:
                    f1 = f1_score(true_classes[valid], pred_classes[valid],
                                  average="macro", zero_division=0)
                    viable = f1 > threshold
                    metric_str = f"F1={f1:.4f} (threshold={threshold:.4f}, {n_classes}-class)"

            elif task_type == "regression":
                preds = teacher_np.flatten()
                labels_np = hard_np.flatten()
                if np.var(labels_np) > 1e-10:
                    r2 = r2_score(labels_np, preds)
                    viable = r2 > 0.05
                    metric_str = f"R2={r2:.4f} (threshold=0.05)"

        except Exception as exc:
            logger.debug("  %s: threshold check error: %s", task_name, exc)

        below_floor = False
        try:
            if task_type == "binary":
                _v = float(metric_str.split("=")[1].split(" ")[0]) if "AUC=" in metric_str else 0.0
                below_floor = _v <= _floor_binary_auc
            elif task_type == "multiclass":
                _v = float(metric_str.split("=")[1].split(" ")[0]) if "F1=" in metric_str else 0.0
                nc = getattr(t, "num_classes", None) or int(hard_np.max()) + 1
                below_floor = _v <= (1.0 / nc) * _floor_mc_f1_ratio
            elif task_type == "regression":
                _v = float(metric_str.split("=")[1].split(" ")[0]) if "R2=" in metric_str else 0.0
                below_floor = _v <= _floor_r2
        except Exception:
            pass

        if viable:
            distill_tasks.append(task_name)
            logger.info("  [DISTILL] %s: %s", task_name, metric_str)
        elif below_floor:
            skip_tasks.append(task_name)
            logger.info("  [SKIP]    %s: %s -- below floor, Layer 3 only", task_name, metric_str)
        else:
            hardlabel_tasks.append(task_name)
            logger.info("  [DIRECT]  %s: %s -- below threshold, using hard labels",
                        task_name, metric_str)

    logger.info(
        "Distillation: %d, Direct: %d, SKIP: %d",
        len(distill_tasks), len(hardlabel_tasks), len(skip_tasks),
    )
    logger.info("=" * 60)
    return distill_tasks, hardlabel_tasks, skip_tasks
