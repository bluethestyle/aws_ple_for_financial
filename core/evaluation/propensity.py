"""
Propensity score estimation for offline policy evaluation.

Uses LightGBM for the base model with optional isotonic or Platt
calibration.  Extreme propensity weights are clipped to bound
variance in downstream IPS / SNIPS / DR estimators.

Usage::

    estimator = PropensityEstimator(calibration="isotonic", clip_bounds=(0.01, 0.99))
    estimator.fit(logged_data, treatment_col="action", feature_cols=["f1", "f2"])
    scores = estimator.predict(new_data)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PropensityConfig:
    """Configuration for propensity score estimation.

    Attributes
    ----------
    calibration : str
        Calibration method: ``"isotonic"``, ``"platt"`` (sigmoid), or
        ``"none"``.
    clip_bounds : tuple[float, float]
        Lower and upper bounds applied to predicted propensity scores
        to prevent extreme inverse-propensity weights.
    lgbm_params : dict
        LightGBM parameters forwarded to ``LGBMClassifier``.
    n_calibration_folds : int
        Number of cross-validation folds used for calibration fitting.
    """

    calibration: str = "isotonic"
    clip_bounds: Tuple[float, float] = (0.01, 0.99)
    lgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "verbose": -1,
    })
    n_calibration_folds: int = 5


class PropensityEstimator:
    """Estimate propensity scores for logged actions.

    Propensity scores ``p(action | context)`` are required by IPS-family
    estimators.  This class wraps a LightGBM classifier with optional
    calibration and weight clipping for safe off-policy evaluation.

    Parameters
    ----------
    calibration : str
        ``"isotonic"`` | ``"platt"`` | ``"none"``.
    clip_bounds : tuple[float, float]
        ``(min_score, max_score)`` applied after prediction.
    lgbm_params : dict, optional
        Override default LightGBM hyperparameters.
    n_calibration_folds : int
        Folds used when calibrating with ``CalibratedClassifierCV``.

    Examples
    --------
    >>> est = PropensityEstimator()
    >>> est.fit(df, treatment_col="action", feature_cols=["x1", "x2"])
    >>> scores = est.predict(df_new)
    """

    def __init__(
        self,
        calibration: str = "isotonic",
        clip_bounds: Tuple[float, float] = (0.01, 0.99),
        lgbm_params: Optional[Dict[str, Any]] = None,
        n_calibration_folds: int = 5,
    ) -> None:
        self.config = PropensityConfig(
            calibration=calibration,
            clip_bounds=clip_bounds,
            lgbm_params=lgbm_params or PropensityConfig().lgbm_params,
            n_calibration_folds=n_calibration_folds,
        )
        self._model: Any = None
        self._calibrated_model: Any = None
        self._feature_cols: List[str] = []
        self._classes: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        logged_data: Any,
        treatment_col: str,
        feature_cols: List[str],
    ) -> "PropensityEstimator":
        """Fit the propensity model on logged interaction data.

        Parameters
        ----------
        logged_data : pandas DataFrame or dict[str, array]
            Historical log data containing features and the treatment
            (action) column.
        treatment_col : str
            Column name of the treatment / action variable.
        feature_cols : list[str]
            Feature column names used as predictors.

        Returns
        -------
        PropensityEstimator
            ``self``, for method chaining.
        """
        X, y = self._extract_Xy(logged_data, treatment_col, feature_cols)
        self._feature_cols = list(feature_cols)
        self._classes = np.unique(y)

        logger.info(
            "Fitting propensity model: %d samples, %d features, %d classes.",
            X.shape[0], X.shape[1], len(self._classes),
        )

        # Base model -- LightGBM
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            logger.warning(
                "LightGBM not available; falling back to sklearn "
                "GradientBoostingClassifier."
            )
            from sklearn.ensemble import GradientBoostingClassifier

            base_params = {
                "n_estimators": self.config.lgbm_params.get("n_estimators", 200),
                "max_depth": self.config.lgbm_params.get("max_depth", 6),
                "learning_rate": self.config.lgbm_params.get("learning_rate", 0.05),
                "subsample": self.config.lgbm_params.get("subsample", 0.8),
            }
            self._model = GradientBoostingClassifier(**base_params)
            self._model.fit(X, y)
            self._apply_calibration(X, y)
            self._is_fitted = True
            return self

        self._model = LGBMClassifier(**self.config.lgbm_params)
        self._model.fit(X, y)

        self._apply_calibration(X, y)
        self._is_fitted = True

        logger.info("Propensity model fitted (calibration=%s).", self.config.calibration)
        return self

    def predict(
        self,
        data: Any,
        feature_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Predict propensity scores for each sample.

        For binary treatment, returns a 1-D array of ``p(treatment=1)``.
        For multi-valued treatments, returns a 2-D array of shape
        ``(n_samples, n_classes)``.

        Parameters
        ----------
        data : pandas DataFrame or dict[str, array]
            Input data containing the feature columns.
        feature_cols : list[str], optional
            Override stored feature columns.

        Returns
        -------
        np.ndarray
            Clipped propensity scores.
        """
        if not self._is_fitted:
            raise RuntimeError("PropensityEstimator has not been fitted. Call fit() first.")

        cols = feature_cols or self._feature_cols
        X = self._extract_features(data, cols)

        model = self._calibrated_model if self._calibrated_model is not None else self._model
        proba = model.predict_proba(X)

        # Binary: return p(class=1)
        if proba.shape[1] == 2:
            scores = proba[:, 1]
        else:
            scores = proba

        scores = self._clip(scores)

        logger.debug(
            "Propensity scores: min=%.4f, max=%.4f, mean=%.4f",
            float(np.min(scores)), float(np.max(scores)), float(np.mean(scores)),
        )
        return scores

    def predict_for_action(
        self,
        data: Any,
        actions: np.ndarray,
        feature_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Return the propensity score for each sample's *observed* action.

        Parameters
        ----------
        data : pandas DataFrame or dict[str, array]
            Feature data.
        actions : np.ndarray
            Observed action for each sample.
        feature_cols : list[str], optional
            Override stored feature columns.

        Returns
        -------
        np.ndarray
            1-D array of clipped propensity scores for the observed actions.
        """
        if not self._is_fitted:
            raise RuntimeError("PropensityEstimator has not been fitted. Call fit() first.")

        cols = feature_cols or self._feature_cols
        X = self._extract_features(data, cols)

        model = self._calibrated_model if self._calibrated_model is not None else self._model
        proba = model.predict_proba(X)

        # Map each sample to its observed action's probability
        if self._classes is not None:
            class_to_idx = {c: i for i, c in enumerate(self._classes)}
            indices = np.array([class_to_idx.get(a, 0) for a in actions])
        else:
            indices = actions.astype(int)

        scores = proba[np.arange(len(proba)), indices]
        scores = self._clip(scores)
        return scores

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _apply_calibration(self, X: np.ndarray, y: np.ndarray) -> None:
        """Wrap the base model in a calibrated classifier if requested."""
        if self.config.calibration == "none":
            self._calibrated_model = None
            return

        from sklearn.calibration import CalibratedClassifierCV

        method = "isotonic" if self.config.calibration == "isotonic" else "sigmoid"
        n_folds = min(self.config.n_calibration_folds, len(np.unique(y)))
        n_folds = max(n_folds, 2)

        logger.info(
            "Calibrating propensity model: method=%s, cv=%d.", method, n_folds,
        )

        try:
            self._calibrated_model = CalibratedClassifierCV(
                estimator=self._model,
                method=method,
                cv=n_folds,
            )
            self._calibrated_model.fit(X, y)
        except Exception as exc:
            logger.warning(
                "Calibration failed (%s); using uncalibrated model.", exc,
            )
            self._calibrated_model = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clip(self, scores: np.ndarray) -> np.ndarray:
        """Clip scores to configured bounds."""
        lo, hi = self.config.clip_bounds
        return np.clip(scores, lo, hi)

    @staticmethod
    def _extract_Xy(
        data: Any,
        treatment_col: str,
        feature_cols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and label vector from heterogeneous data."""
        if hasattr(data, "iloc"):  # pandas DataFrame
            X = data[feature_cols].values.astype(np.float64)
            y = data[treatment_col].values
        elif isinstance(data, dict):
            X = np.column_stack([np.asarray(data[c]) for c in feature_cols]).astype(np.float64)
            y = np.asarray(data[treatment_col])
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        return X, y

    @staticmethod
    def _extract_features(data: Any, feature_cols: List[str]) -> np.ndarray:
        """Extract feature matrix from heterogeneous data."""
        if hasattr(data, "iloc"):
            return data[feature_cols].values.astype(np.float64)
        if isinstance(data, dict):
            return np.column_stack([np.asarray(data[c]) for c in feature_cols]).astype(np.float64)
        raise TypeError(f"Unsupported data type: {type(data)}")


__all__ = ["PropensityEstimator", "PropensityConfig"]
