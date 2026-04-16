"""Fidelity validation: teacher vs. student agreement per task.

Each task is scored against 8 metrics (AUC gap, JSD, ranking corr, etc.).
The shared ``get_student_predictions()`` helper is also used by drift_step.py.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("distill-entry")


def get_student_predictions(
    task_spec: Any,
    student_model: Any,
    calibrated_models: Dict[str, Any],
    features: np.ndarray,
    student_config: Any,
) -> np.ndarray:
    """Return calibrated (or raw) student predictions for one task.

    Args:
        task_spec:        Task config object with ``.name``, ``.type``, ``.num_classes``.
        student_model:    Trained LGBM model.
        calibrated_models: Dict of {task_name: calibrated_model_or_dict}.
        features:         Feature matrix (N x D).
        student_config:   StudentConfig (provides ``use_custom_objective`` flag).

    Returns:
        1-D or 2-D numpy array of predictions.
    """
    task_name = task_spec.name

    if task_name in calibrated_models:
        calib = calibrated_models[task_name]
        if task_spec.type == "binary" and hasattr(calib, "predict_proba"):
            return calib.predict_proba(features)[:, 1]
        elif task_spec.type == "regression" and isinstance(calib, dict):
            raw = calib["lgbm"].predict(features)
            return calib["bias_corrector"].predict(raw.reshape(-1, 1))
        else:
            return student_model.predict(features)

    raw_preds = student_model.predict(features)
    if task_spec.type == "binary" and student_config.use_custom_objective:
        return 1.0 / (1.0 + np.exp(-raw_preds))
    elif task_spec.type == "multiclass" and student_config.use_custom_objective:
        n_classes = task_spec.num_classes
        raw_2d = raw_preds.reshape(-1, n_classes)
        exp_shifted = np.exp(raw_2d - raw_2d.max(axis=1, keepdims=True))
        return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
    return raw_preds


def validate_fidelity(
    pipeline_config: Any,
    students: Dict[str, Any],
    calibrated_models: Dict[str, Any],
    features: np.ndarray,
    hard_labels: Dict[str, np.ndarray],
    student_config: Any,
    distillation_cfg: dict,
    trainer: Any,
    out_dir: Path,
    skip_gate: bool,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Run per-task fidelity validation and write fidelity_report.json.

    Args:
        pipeline_config:  Parsed pipeline config with ``.tasks``.
        students:         Dict of {task_name: lgbm_model}.
        calibrated_models: Calibrated wrappers (may be empty).
        features:         Feature matrix (N x D).
        hard_labels:      Ground-truth arrays per task.
        student_config:   StudentConfig with ``use_custom_objective``.
        distillation_cfg: ``distillation`` sub-dict from pipeline YAML.
        trainer:          StudentTrainer (to retrieve soft labels).
        out_dir:          Directory where ``fidelity_report.json`` is written.
        skip_gate:        When True, continue despite failures.

    Returns:
        (fidelity_results, fidelity_report_dict)
    """
    from core.training.distillation_validator import (
        DistillationValidator,
        FidelityResult,
        ValidationCriteria,
    )

    _fidelity_cfg = distillation_cfg.get("fidelity", {})
    _bin_cfg = _fidelity_cfg.get("binary", {})
    _mc_cfg = _fidelity_cfg.get("multiclass", {})
    _reg_cfg = _fidelity_cfg.get("regression", {})

    criteria = ValidationCriteria(
        max_auc_gap=_bin_cfg.get("max_auc_gap", 0.05),
        min_binary_agreement=_bin_cfg.get("min_agreement", 0.85),
        max_jsd=_bin_cfg.get("max_jsd", 0.10),
        min_ranking_corr=_bin_cfg.get("min_ranking_corr", 0.90),
        max_calibration_gap=_bin_cfg.get("max_calibration_gap", 0.05),
        min_multiclass_agreement=_mc_cfg.get("min_agreement", 0.70),
        max_f1_macro_gap=_mc_cfg.get("max_f1_macro_gap", 0.10),
        regression_quartile_agreement_min=_reg_cfg.get("min_quartile_agreement", 0.70),
        max_mae_gap=_reg_cfg.get("max_mae_gap", 0.05),
        max_rmse_gap=_reg_cfg.get("max_rmse_gap", 0.10),
    )
    validator = DistillationValidator(criteria=criteria)
    soft_labels = trainer.get_soft_labels()
    fidelity_results: List[FidelityResult] = []

    logger.info("Running fidelity validation (8 metrics per task)...")

    for task_spec in pipeline_config.tasks:
        task_name = task_spec.name
        if task_name not in students:
            continue

        student_preds = get_student_predictions(
            task_spec, students[task_name], calibrated_models, features, student_config,
        )

        teacher_preds = soft_labels.get(task_name)
        if teacher_preds is None:
            logger.warning("No soft labels for task %s, skipping fidelity", task_name)
            continue

        labels_for_task = hard_labels.get(task_name)

        try:
            result = validator.validate_task(
                task_name=task_name,
                task_type=task_spec.type,
                teacher_preds=teacher_preds,
                student_preds=student_preds,
                labels=labels_for_task,
            )
        except Exception as exc:
            logger.warning("Fidelity validation failed for %s: %s", task_name, exc)
            result = FidelityResult(
                task_name=task_name, task_type=task_spec.type, passed=False,
                metrics={}, failures=[str(exc)],
            )
        fidelity_results.append(result)

        status = "PASS" if result.passed else "FAIL"
        logger.info(
            "  [%s] %s -- metrics: %s%s",
            status, task_name,
            {k: round(v, 4) for k, v in result.metrics.items()},
            f" failures: {result.failures}" if result.failures else "",
        )

    passed_count = sum(1 for r in fidelity_results if r.passed)
    failed_count = len(fidelity_results) - passed_count
    logger.info("Fidelity summary: %d/%d tasks passed", passed_count, passed_count + failed_count)

    failed_tasks: List[str] = [r.task_name for r in fidelity_results if not r.passed]

    fidelity_report: Dict[str, Any] = {
        "status": "FAILED" if failed_count > 0 else "PASSED",
        "passed": passed_count,
        "failed": failed_count,
        "details": {
            r.task_name: {"passed": r.passed, "metrics": r.metrics, "failures": r.failures}
            for r in fidelity_results
        },
    }
    if failed_count > 0:
        fidelity_report["failed_tasks"] = failed_tasks
        logger.error("Fidelity FAILED for %d task(s): %s", failed_count, failed_tasks)

    with open(out_dir / "fidelity_report.json", "w") as f:
        json.dump(fidelity_report, f, indent=2, default=str)

    if failed_count > 0 and not skip_gate:
        logger.error("Aborting distillation pipeline due to fidelity failure.")
        raise RuntimeError(f"Fidelity gate failed on {failed_count} task(s): {failed_tasks}")
    elif failed_count > 0:
        logger.warning("skip_fidelity_gate=True -- continuing despite failures.")

    return fidelity_results, fidelity_report
