"""Prediction-level and feature-level drift monitoring.

Step 4.5b — temporal prediction drift (SR 11-7 MRM safeguard).
Step 4.6  — feature PSI drift (complements prediction drift).

``get_student_predictions`` is imported from fidelity.py to avoid duplication.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("distill-entry")


def run_prediction_drift(
    distillation_cfg: dict,
    pipeline_config: Any,
    students: Dict[str, Any],
    calibrated_models: Dict[str, Any],
    features: np.ndarray,
    hard_labels: Dict[str, np.ndarray],
    student_config: Any,
    out_dir: Path,
) -> None:
    """Run temporal prediction drift monitoring and save ``drift_report.json``.

    Args:
        distillation_cfg: ``distillation`` sub-dict from pipeline YAML.
        pipeline_config:  Parsed pipeline config with ``.tasks``.
        students:         Dict of {task_name: lgbm_model}.
        calibrated_models: Calibrated wrappers (may be empty).
        features:         Feature matrix (N x D).
        hard_labels:      Ground-truth arrays per task.
        student_config:   StudentConfig with ``use_custom_objective``.
        out_dir:          Directory where ``drift_report.json`` is written.
    """
    from containers.distillation.fidelity import get_student_predictions

    _drift_cfg = distillation_cfg.get("drift_monitoring", {})
    if not _drift_cfg.get("enabled", False):
        logger.info("Drift monitoring disabled (distillation.drift_monitoring.enabled=false).")
        return

    from core.training.distillation_drift import DistillationDriftMonitor

    logger.info("=" * 60)
    logger.info("Step 4.5b: Temporal drift monitoring (SR 11-7)")
    logger.info("=" * 60)

    _drift_monitor = DistillationDriftMonitor(config=_drift_cfg)
    _default_baseline = (
        os.path.join(os.environ["SM_OUTPUT_DATA_DIR"], "distillation_baseline")
        if "SM_OUTPUT_DATA_DIR" in os.environ
        else "outputs/distillation_baseline/"
    )
    _baseline_path = _drift_cfg.get("baseline_path", _default_baseline)

    _current_student_preds: Dict[str, Any] = {}
    for _task_spec in pipeline_config.tasks:
        _tname = _task_spec.name
        if _tname not in students:
            continue
        _current_student_preds[_tname] = get_student_predictions(
            _task_spec, students[_tname], calibrated_models, features, student_config,
        )

    _prev_preds = _drift_monitor.load_baseline(_baseline_path)

    if _prev_preds is not None:
        _drift_report = _drift_monitor.compare_versions(
            current_preds=_current_student_preds,
            previous_preds=_prev_preds,
            labels=hard_labels,
        )

        _n_alerts = len(_drift_report["alert_tasks"])
        if _drift_report["any_alert"]:
            logger.warning(
                "Drift alerts on %d task(s): %s -- MRM review recommended.",
                _n_alerts, _drift_report["alert_tasks"],
            )
        else:
            logger.info(
                "Drift check PASSED: all %d tasks within thresholds.",
                len(_drift_report["per_task"]),
            )

        drift_out = {
            "any_alert": _drift_report["any_alert"],
            "alert_tasks": _drift_report["alert_tasks"],
            "thresholds": _drift_report["thresholds"],
            "per_task": {
                t: {k: round(v, 6) if isinstance(v, float) else v for k, v in m.items()}
                for t, m in _drift_report["per_task"].items()
            },
        }
        with open(out_dir / "drift_report.json", "w") as _drf:
            json.dump(drift_out, _drf, indent=2, default=str)
        logger.info("Drift report saved to %s/drift_report.json", out_dir)
    else:
        logger.info("No previous baseline -- current predictions become first baseline.")

    _drift_monitor.save_baseline(_current_student_preds, _baseline_path)
    logger.info("Drift baseline updated at %s", _baseline_path)


def run_feature_drift(
    distillation_cfg: dict,
    features: np.ndarray,
    feature_cols: list,
    out_dir: Path,
) -> None:
    """Run PSI-based feature-level drift monitoring and save ``feature_drift_report.json``.

    Args:
        distillation_cfg: ``distillation`` sub-dict from pipeline YAML.
        features:         Feature matrix (N x D).
        feature_cols:     Ordered list of feature column names.
        out_dir:          Directory where ``feature_drift_report.json`` is written.
    """
    _feat_drift_cfg = distillation_cfg.get("feature_drift", {})
    if not _feat_drift_cfg.get("enabled", False):
        logger.info("Feature drift monitoring disabled (distillation.feature_drift.enabled=false).")
        return

    try:
        from core.monitoring.drift_detector import DriftDetector

        _feat_baseline_path = Path(
            _feat_drift_cfg.get(
                "baseline_path", "outputs/distillation_baseline/feature_baseline.json"
            )
        )
        _psi_warn = float(_feat_drift_cfg.get("psi_threshold_warning", 0.1))
        _psi_crit = float(_feat_drift_cfg.get("psi_threshold_critical", 0.25))
        _feat_drift_detector = DriftDetector(
            psi_threshold_warning=_psi_warn,
            psi_threshold_critical=_psi_crit,
        )

        logger.info("=" * 60)
        logger.info("Step 4.6: Feature-level drift monitoring (PSI)")
        logger.info("  warning=%.2f  critical=%.2f", _psi_warn, _psi_crit)
        logger.info("=" * 60)

        if _feat_baseline_path.exists():
            with open(_feat_baseline_path, encoding="utf-8") as _fdf:
                _feat_baseline: Dict[str, Any] = json.load(_fdf)

            _current_feat_dict: Dict[str, np.ndarray] = {
                c: features[:, i]
                for i, c in enumerate(feature_cols)
                if i < features.shape[1]
            }
            _baseline_feat_dict = _feat_baseline.get("feature_samples", {})

            _feat_drift_result = _feat_drift_detector.detect_drift(
                baseline_data=_baseline_feat_dict,
                current_data=_current_feat_dict,
            )
            _fd_summary = _feat_drift_result["summary"]
            logger.info(
                "Feature PSI — critical: %d, warning: %d, max_psi: %.4f",
                _fd_summary["critical_count"],
                _fd_summary["warning_count"],
                _fd_summary["max_psi"],
            )
            with open(out_dir / "feature_drift_report.json", "w") as _fdf_out:
                json.dump(
                    {
                        "summary": _fd_summary,
                        "warning_features": _feat_drift_result["warning_features"],
                        "critical_features": _feat_drift_result["critical_features"],
                        "psi_scores": {
                            k: round(v, 6) if isinstance(v, float) else v
                            for k, v in _feat_drift_result["psi_scores"].items()
                        },
                    },
                    _fdf_out, indent=2, default=str,
                )
            logger.info(
                "Feature drift report saved to %s/feature_drift_report.json", out_dir
            )
        else:
            logger.info(
                "No feature baseline at %s — saving current as first baseline",
                _feat_baseline_path,
            )

        # Save current feature samples as baseline (sampled to save space)
        _n_sample = min(10000, features.shape[0])
        _rng_sample = np.random.RandomState(42)
        _sample_idx = _rng_sample.choice(features.shape[0], size=_n_sample, replace=False)
        _new_baseline = {
            "feature_samples": {
                c: features[_sample_idx, i].tolist()
                for i, c in enumerate(feature_cols)
                if i < features.shape[1]
            }
        }
        _feat_baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_feat_baseline_path, "w", encoding="utf-8") as _fdf_save:
            json.dump(_new_baseline, _fdf_save)
        logger.info("Feature baseline updated at %s", _feat_baseline_path)

    except Exception as _feat_drift_exc:
        logger.warning(
            "Feature drift monitoring failed (non-fatal): %s",
            _feat_drift_exc, exc_info=True,
        )
