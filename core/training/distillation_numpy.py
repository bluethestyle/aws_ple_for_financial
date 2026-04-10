"""
Custom LightGBM objective functions for knowledge distillation (NumPy backend).

Ports the on-prem distillation loss into LightGBM's fobj interface so that
a LGBM student can be trained against soft labels produced by the PLE teacher.

Distillation formula (Hinton et al., 2015):

    L = alpha * L_hard + (1 - alpha) * T^2 * L_soft

    L_hard : standard loss vs ground-truth hard labels
    L_soft  : KL loss vs teacher's temperature-scaled soft probabilities
    T       : temperature  (higher -> softer teacher distribution)
    alpha   : weight for hard loss  (0 = pure KL, 1 = no distillation)

All hyperparameters are injected at construction time from the caller's YAML
config — no dataset-specific constants or task names are embedded here.

Supported task types
--------------------
- Binary classification  : DistillationLossNumpy.binary_loss
- Regression             : DistillationLossNumpy.regression_loss
- Multiclass             : make_multiclass_objective(alpha, temperature, num_classes)

Validation-set guard
--------------------
train_data.soft_labels is attached only to the training Dataset.  When the
validation Dataset is passed (e.g. inside LightGBM's eval loop), soft_labels
is absent, so all methods fall back to the plain hard-label loss automatically.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Minimum hessian value to avoid numerical instability in LightGBM's Newton step
_HESS_MIN: float = 1e-6


class DistillationLossNumpy:
    """Config-driven LightGBM custom objective for knowledge distillation.

    Parameters
    ----------
    alpha : float
        Weight for the hard (ground-truth) loss component.  ``1.0`` disables
        distillation entirely; ``0.0`` trains purely on teacher soft labels.
    temperature : float
        Softmax temperature applied to both teacher and student logits before
        computing the soft KL loss.  Must be > 0.
    """

    def __init__(self, alpha: float, temperature: float) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.alpha = alpha
        self.temperature = temperature
        logger.info(
            "DistillationLossNumpy initialised — alpha=%.3f, temperature=%.2f",
            alpha,
            temperature,
        )

    # ------------------------------------------------------------------
    # Binary classification
    # ------------------------------------------------------------------

    def binary_loss(
        self,
        preds: np.ndarray,
        train_data: object,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """LightGBM fobj for binary classification with soft-label distillation.

        Parameters
        ----------
        preds : np.ndarray
            Raw (logit) predictions from the current LGBM model, shape (n,).
        train_data : lgb.Dataset
            Active dataset.  Must carry a ``soft_labels`` attribute (shape (n,))
            on the training split; validation datasets may omit it.

        Returns
        -------
        grad, hess : Tuple[np.ndarray, np.ndarray]
        """
        hard_labels: np.ndarray = train_data.get_label()
        soft_labels: Optional[np.ndarray] = getattr(train_data, "soft_labels", None)

        sigmoid_preds = 1.0 / (1.0 + np.exp(-preds))

        # Validation-set guard: no soft labels → plain BCE
        if soft_labels is None or len(soft_labels) != len(preds):
            logger.debug("binary_loss: soft_labels absent — using hard-only BCE")
            grad = sigmoid_preds - hard_labels
            hess = np.clip(sigmoid_preds * (1.0 - sigmoid_preds), _HESS_MIN, None)
            return grad, hess

        # Hard gradient / hessian (BCE)
        grad_hard = sigmoid_preds - hard_labels
        hess_hard = sigmoid_preds * (1.0 - sigmoid_preds)

        # Soft gradient / hessian (temperature-scaled BCE ≈ KL)
        T = self.temperature
        sigmoid_soft = 1.0 / (1.0 + np.exp(-preds / T))
        grad_soft = (sigmoid_soft - soft_labels) / T
        hess_soft = sigmoid_soft * (1.0 - sigmoid_soft) / (T**2)

        # Combined: alpha * hard + (1-alpha) * T^2 * soft
        T_sq = T**2
        grad = self.alpha * grad_hard + (1.0 - self.alpha) * T_sq * grad_soft
        hess = np.clip(
            self.alpha * hess_hard + (1.0 - self.alpha) * T_sq * hess_soft,
            _HESS_MIN,
            None,
        )

        logger.debug(
            "binary_loss: grad_mean=%.4f, hess_mean=%.4f",
            float(grad.mean()),
            float(hess.mean()),
        )
        return grad, hess

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------

    def regression_loss(
        self,
        preds: np.ndarray,
        train_data: object,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """LightGBM fobj for regression with soft-label distillation (MSE base).

        No T^2 scaling for regression — teacher outputs are already in the
        same value space as the student predictions.

        Parameters
        ----------
        preds : np.ndarray
            Current model predictions, shape (n,).
        train_data : lgb.Dataset
            Active dataset, optionally carrying a ``soft_labels`` attribute.

        Returns
        -------
        grad, hess : Tuple[np.ndarray, np.ndarray]
        """
        hard_labels: np.ndarray = train_data.get_label()
        soft_labels: Optional[np.ndarray] = getattr(train_data, "soft_labels", None)

        # Validation-set guard
        if soft_labels is None or len(soft_labels) != len(preds):
            logger.debug("regression_loss: soft_labels absent — using hard-only MSE")
            grad = preds - hard_labels
            hess = np.clip(np.ones_like(preds), _HESS_MIN, None)
            return grad, hess

        grad_hard = preds - hard_labels
        hess_hard = np.ones_like(preds)

        grad_soft = preds - soft_labels
        hess_soft = np.ones_like(preds)

        # No T^2 for regression
        grad = self.alpha * grad_hard + (1.0 - self.alpha) * grad_soft
        hess = np.clip(
            self.alpha * hess_hard + (1.0 - self.alpha) * hess_soft,
            _HESS_MIN,
            None,
        )

        logger.debug(
            "regression_loss: grad_mean=%.4f, hess_mean=%.4f",
            float(grad.mean()),
            float(hess.mean()),
        )
        return grad, hess


# ------------------------------------------------------------------
# Multiclass factory
# ------------------------------------------------------------------


def make_multiclass_objective(
    alpha: float,
    temperature: float,
    num_classes: int,
) -> Callable:
    """Return a LightGBM fobj closure for multiclass distillation.

    Parameters
    ----------
    alpha : float
        Hard-loss weight, in [0, 1].
    temperature : float
        Softmax temperature for the soft KL term.
    num_classes : int
        Number of output classes.  Must match the LGBM ``num_class`` param.

    Returns
    -------
    Callable
        A function with signature ``(preds, train_data) -> (grad, hess)``
        suitable for passing as LightGBM's ``fobj``.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    logger.info(
        "make_multiclass_objective — alpha=%.3f, temperature=%.2f, num_classes=%d",
        alpha,
        temperature,
        num_classes,
    )

    def multiclass_distillation_obj(
        preds: np.ndarray,
        train_data: object,
    ) -> Tuple[np.ndarray, np.ndarray]:
        labels = train_data.get_label().astype(np.int64)
        n_samples = len(labels)
        preds_2d = preds.reshape((n_samples, num_classes))

        soft: Optional[np.ndarray] = getattr(train_data, "soft_labels", None)

        # Stable softmax for hard path
        shifted = preds_2d - np.max(preds_2d, axis=1, keepdims=True)
        exp_preds = np.exp(shifted)
        pred_proba = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

        # Hard: standard softmax CE
        y_onehot = np.zeros((n_samples, num_classes), dtype=np.float64)
        y_onehot[np.arange(n_samples), labels] = 1.0
        grad_hard = pred_proba - y_onehot
        hess_hard = pred_proba * (1.0 - pred_proba)

        # Validation-set guard
        if soft is None or len(soft) != n_samples:
            logger.debug("multiclass_obj: soft_labels absent — using hard-only CE")
            hess_hard = np.clip(hess_hard, _HESS_MIN, None)
            return grad_hard.reshape(-1), hess_hard.reshape(-1)

        soft_2d = (
            soft.reshape((n_samples, num_classes)) if soft.ndim == 1 else soft
        )

        # Soft: temperature-scaled student softmax vs teacher probs
        T = temperature
        T_sq = T * T
        preds_scaled = preds_2d / T
        shifted_scaled = preds_scaled - np.max(preds_scaled, axis=1, keepdims=True)
        exp_scaled = np.exp(shifted_scaled)
        pred_proba_soft = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)

        grad_soft = (pred_proba_soft - soft_2d) / T
        hess_soft = pred_proba_soft * (1.0 - pred_proba_soft) / T_sq

        grad = alpha * grad_hard + (1.0 - alpha) * T_sq * grad_soft
        hess = np.clip(
            alpha * hess_hard + (1.0 - alpha) * T_sq * hess_soft,
            _HESS_MIN,
            None,
        )

        logger.debug(
            "multiclass_obj: grad_mean=%.4f, hess_mean=%.4f",
            float(grad.mean()),
            float(hess.mean()),
        )
        return grad.reshape(-1), hess.reshape(-1)

    return multiclass_distillation_obj


# ------------------------------------------------------------------
# Version-aware LightGBM training wrapper
# ------------------------------------------------------------------


def train_with_custom_objective(
    params: dict,
    train_set: object,
    objective_fn: Callable,
    **kwargs,
) -> object:
    """Version-aware wrapper for LightGBM training with a custom objective.

    Handles the API difference between LightGBM 3.x (``fobj`` keyword inside
    ``lgb.train``) and 4.x (same, but ``objective`` key in ``params`` must be
    absent).  Also ensures ``metric`` is explicitly set so that LightGBM does
    not attempt to infer a built-in metric that conflicts with the custom loss.

    Parameters
    ----------
    params : dict
        LightGBM parameter dict.  The ``"objective"`` key is removed
        automatically before training.
    train_set : lgb.Dataset
        Training dataset (with ``soft_labels`` attribute if distillation is
        active).
    objective_fn : Callable
        Custom fobj with signature ``(preds, train_data) -> (grad, hess)``.
    **kwargs
        Forwarded to ``lgb.train`` (e.g. ``num_boost_round``, ``valid_sets``,
        ``callbacks``).

    Returns
    -------
    lgb.Booster
    """
    import lightgbm as lgb  # lazy import — optional dependency

    # Replace built-in objective string with the custom callable.
    # LightGBM 4.x: pass callable via params["objective"]
    # LightGBM 3.x: pass via fobj kwarg
    training_params = dict(params)
    training_params["objective"] = objective_fn

    # Guarantee explicit metric so LightGBM does not guess
    if "metric" not in training_params:
        training_params["metric"] = "custom"
        logger.debug("train_with_custom_objective: defaulting metric='custom'")

    logger.info(
        "train_with_custom_objective: num_boost_round=%s",
        kwargs.get("num_boost_round", "?"),
    )

    import inspect
    _sig = inspect.signature(lgb.train)
    if "fobj" in _sig.parameters:
        # LightGBM 3.x: use fobj kwarg
        training_params.pop("objective", None)
        booster = lgb.train(
            training_params, train_set, fobj=objective_fn, **kwargs,
        )
    else:
        # LightGBM 4.x: callable in params["objective"]
        booster = lgb.train(training_params, train_set, **kwargs)
    return booster
