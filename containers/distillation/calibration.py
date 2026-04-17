"""Platt scaling / linear bias correction for student LGBM models.

CLAUDE.md §1.8: calibration is applied only to tasks listed in
``distillation.calibration.tasks`` — ranking-based tasks skip calibration.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Set

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger("distill-entry")


class _LGBMProbWrapper(BaseEstimator, ClassifierMixin):
    """Wrap a trained LGBM model for sklearn calibration.

    Custom-objective LGBM outputs raw logits, so we apply sigmoid
    before returning probabilities. Defined at module level so that
    joblib/pickle can serialise the fitted CalibratedClassifierCV.
    """

    def __init__(self, lgbm_model: Any = None) -> None:
        self.lgbm_model = lgbm_model
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LGBMProbWrapper":
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.lgbm_model.predict(X)
        prob = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack([1 - prob, prob])


def calibrate_students(
    students: Dict[str, Any],
    hard_labels: Dict[str, np.ndarray],
    features: np.ndarray,
    pipeline_config: Any,
    distillation_cfg: dict,
) -> Dict[str, Any]:
    """Apply Platt scaling or linear bias correction to eligible student models.

    Args:
        students:         Dict of {task_name: lgbm_model}.
        hard_labels:      Dict of {task_name: ground-truth array}.
        features:         Feature matrix (N x D) as float32.
        pipeline_config:  Parsed pipeline config with ``.tasks``.
        distillation_cfg: ``distillation`` sub-dict from pipeline YAML.

    Returns:
        Dict of {task_name: calibrated_model_or_dict} for calibrated tasks.
        Tasks not in the return dict were skipped or failed calibration.
    """
    from sklearn.calibration import CalibratedClassifierCV

    _calib_cfg = distillation_cfg.get("calibration", {})
    _calib_tasks: Set[str] = set(_calib_cfg.get("tasks", []))
    _calib_method: str = _calib_cfg.get("method", "platt")

    calibrated_models: Dict[str, Any] = {}

    if not (_calib_cfg.get("enabled", False) and _calib_tasks):
        return calibrated_models

    n_total = len(features)
    calib_start = int(n_total * 0.8)
    X_calib = features[calib_start:]
    _sklearn_method = "sigmoid" if _calib_method == "platt" else "isotonic"

    logger.info("=" * 60)
    logger.info("Calibration: %s method on %d tasks", _calib_method, len(_calib_tasks))
    logger.info("  Calibration set: %d samples (last 20%%)", len(X_calib))
    logger.info("=" * 60)

    for task_name in _calib_tasks:
        if task_name not in students or task_name not in hard_labels:
            logger.warning("  %s: skipped (no model or labels)", task_name)
            continue

        t_spec = next((t for t in pipeline_config.tasks if t.name == task_name), None)
        if t_spec is None:
            continue

        y_calib = hard_labels[task_name][calib_start:]
        model = students[task_name]

        try:
            if t_spec.type == "binary":
                wrapper = _LGBMProbWrapper(model)
                calibrator = CalibratedClassifierCV(
                    wrapper, method=_sklearn_method, cv="prefit",
                )
                calibrator.fit(X_calib, y_calib)
                calibrated_models[task_name] = calibrator
                logger.info("  [CALIBRATED] %s: %s scaling applied", task_name, _calib_method)

            elif t_spec.type == "regression":
                from sklearn.linear_model import LinearRegression
                raw_preds = model.predict(X_calib)
                lr = LinearRegression()
                lr.fit(raw_preds.reshape(-1, 1), y_calib)
                calibrated_models[task_name] = {"lgbm": model, "bias_corrector": lr}
                logger.info(
                    "  [CALIBRATED] %s: linear bias correction (slope=%.4f, intercept=%.4f)",
                    task_name, lr.coef_[0], lr.intercept_,
                )

            else:
                logger.info("  %s: multiclass calibration not implemented, skipped", task_name)

        except Exception as exc:
            logger.warning("  %s: calibration failed: %s", task_name, exc)

    return calibrated_models
