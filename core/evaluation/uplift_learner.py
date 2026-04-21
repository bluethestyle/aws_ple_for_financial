"""
Uplift learners (C1) — Pearl Rung 2 treatment-effect estimation.

Paper 2 v2 claims coverage of Pearl's causal ladder across three rungs:
- Rung 1 (Observation): CEH attribution head (Finding 9)
- Rung 2 (Intervention): **THIS MODULE** — CATE / uplift estimation
- Rung 3 (Counterfactual): CCP counterfactual probe (Finding 12) +
  Counterfactual Champion-Challenger (S14, IPS/SNIPS)

Rung 2 estimates the **Conditional Average Treatment Effect (CATE)**:
``tau(x) = E[Y | T=1, X=x] - E[Y | T=0, X=x]``. When real offer/exposure
data becomes available the estimators here produce per-customer uplift
scores that the recommender can use to target treatment-responsive
segments without needing an experimental A/B.

Estimators
----------
- :class:`TLearner` — two independent outcome models (treated / control).
  Baseline; simple to audit; sensitive to treatment-group imbalance.
- :class:`XLearner` — T-Learner + cross-fit stage that re-weighs the
  rare class using propensity. More robust under imbalance (Künzel 2019).

Both accept any sklearn-compatible regressor (``fit`` + ``predict``).

Evaluation
----------
- :func:`qini_coefficient` — area-under-Qini-curve (uplift analog of AUC).
- :func:`uplift_at_k` — mean uplift over the top-K% predicted uplift
  recipients. Matches the business deployment pattern.

This is a dependency-light implementation (numpy-only) designed for
offline evaluation and integration tests. Production training can swap
in sklearn / lightgbm regressors behind the same interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "TLearner",
    "XLearner",
    "PropensityEstimator",
    "qini_coefficient",
    "uplift_at_k",
    "evaluate_uplift",
]


# ---------------------------------------------------------------------------
# Base regressor protocol (duck-typed)
# ---------------------------------------------------------------------------

RegressorFactory = Callable[[], Any]


def _default_regressor() -> Any:
    """Simple numpy-only regressor used when sklearn is unavailable."""
    return _RidgeLikeRegressor(alpha=1.0)


class _RidgeLikeRegressor:
    """Minimal ridge-regression fallback (no sklearn dependency)."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._coef: Optional[np.ndarray] = None
        self._intercept: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RidgeLikeRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # Augment with intercept column
        Xa = np.hstack([X, np.ones((X.shape[0], 1))])
        # Ridge: (X'X + αI)^-1 X'y, but don't penalise the intercept
        d = Xa.shape[1]
        penalty = np.eye(d) * self.alpha
        penalty[-1, -1] = 0.0
        A = Xa.T @ Xa + penalty
        b = Xa.T @ y
        w = np.linalg.solve(A, b)
        self._coef = w[:-1]
        self._intercept = float(w[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self._coef is None:
            return np.zeros(X.shape[0])
        return X @ self._coef + self._intercept


# ---------------------------------------------------------------------------
# Propensity estimator (for X-Learner and IPS)
# ---------------------------------------------------------------------------

class PropensityEstimator:
    """Simple logistic-regression propensity model.

    Fits ``P(T=1 | X)`` on the observed treatment assignment. Clipped
    to ``[eps, 1-eps]`` to prevent IPS blow-up.
    """

    def __init__(self, eps: float = 0.01, max_iter: int = 200) -> None:
        if not (0.0 < eps < 0.5):
            raise ValueError(f"eps={eps} must be in (0, 0.5)")
        self._eps = float(eps)
        self._max_iter = int(max_iter)
        self._coef: Optional[np.ndarray] = None
        self._intercept: float = 0.0

    def fit(self, X: np.ndarray, T: np.ndarray) -> "PropensityEstimator":
        X = np.asarray(X, dtype=float)
        T = np.asarray(T, dtype=float)
        n, d = X.shape
        self._coef = np.zeros(d)
        self._intercept = 0.0
        # Simple gradient ascent on log-likelihood
        lr = 0.1
        for _ in range(self._max_iter):
            logits = X @ self._coef + self._intercept
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            grad_w = X.T @ (T - p) / n
            grad_b = float(np.mean(T - p))
            self._coef += lr * grad_w
            self._intercept += lr * grad_b
            if np.max(np.abs(grad_w)) < 1e-6:
                break
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self._coef is None:
            return np.full(X.shape[0], 0.5)
        logits = X @ self._coef + self._intercept
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        return np.clip(p, self._eps, 1.0 - self._eps)


# ---------------------------------------------------------------------------
# T-Learner
# ---------------------------------------------------------------------------

@dataclass
class TLearner:
    """Two-model T-Learner for CATE estimation.

    Fits separate regressors on the treated (T=1) and control (T=0)
    populations, then estimates ``tau(x) = mu1(x) - mu0(x)``.

    Usage::

        learner = TLearner().fit(X, T, Y)
        tau = learner.predict(X_test)       # per-row CATE estimate
    """

    regressor_factory: RegressorFactory = _default_regressor

    def __post_init__(self) -> None:
        self._mu0: Optional[Any] = None
        self._mu1: Optional[Any] = None

    def fit(self, X, T, Y) -> "TLearner":
        X = np.asarray(X, dtype=float)
        T = np.asarray(T).astype(int)
        Y = np.asarray(Y, dtype=float)
        if not set(np.unique(T)).issubset({0, 1}):
            raise ValueError("T must be binary (0 / 1)")
        if not (X.shape[0] == len(T) == len(Y)):
            raise ValueError("X / T / Y length mismatch")

        X0, Y0 = X[T == 0], Y[T == 0]
        X1, Y1 = X[T == 1], Y[T == 1]
        if len(X0) == 0 or len(X1) == 0:
            raise ValueError(
                "Both treatment (T=1) and control (T=0) rows are required"
            )

        self._mu0 = self.regressor_factory().fit(X0, Y0)
        self._mu1 = self.regressor_factory().fit(X1, Y1)
        logger.info(
            "TLearner fit: n_control=%d n_treated=%d", len(X0), len(X1),
        )
        return self

    def predict(self, X) -> np.ndarray:
        if self._mu0 is None or self._mu1 is None:
            raise RuntimeError("TLearner must be fit before predict")
        X = np.asarray(X, dtype=float)
        return self._mu1.predict(X) - self._mu0.predict(X)


# ---------------------------------------------------------------------------
# X-Learner
# ---------------------------------------------------------------------------

@dataclass
class XLearner:
    """X-Learner for CATE (Künzel et al., 2019, PNAS).

    Two-stage:
    1. Fit outcome models mu0, mu1 (as in T-Learner).
    2. Impute the counterfactual residual for each subgroup and fit
       two CATE models tau0 (on control) and tau1 (on treated).
    3. Combine: ``tau(x) = g(x) * tau0(x) + (1 - g(x)) * tau1(x)`` where
       ``g(x) = e(x)`` (propensity). This down-weights the CATE learned
       on the rarer arm.

    More stable than T-Learner when treatment groups are imbalanced.
    """

    regressor_factory: RegressorFactory = _default_regressor

    def __post_init__(self) -> None:
        self._mu0: Optional[Any] = None
        self._mu1: Optional[Any] = None
        self._tau0: Optional[Any] = None
        self._tau1: Optional[Any] = None
        self._propensity: Optional[PropensityEstimator] = None

    def fit(
        self, X, T, Y,
        propensity: Optional[PropensityEstimator] = None,
    ) -> "XLearner":
        X = np.asarray(X, dtype=float)
        T = np.asarray(T).astype(int)
        Y = np.asarray(Y, dtype=float)
        if not set(np.unique(T)).issubset({0, 1}):
            raise ValueError("T must be binary (0 / 1)")
        if not (X.shape[0] == len(T) == len(Y)):
            raise ValueError("X / T / Y length mismatch")

        # Stage 1: outcome models
        X0, Y0 = X[T == 0], Y[T == 0]
        X1, Y1 = X[T == 1], Y[T == 1]
        if len(X0) == 0 or len(X1) == 0:
            raise ValueError("Both treatment groups are required")
        self._mu0 = self.regressor_factory().fit(X0, Y0)
        self._mu1 = self.regressor_factory().fit(X1, Y1)

        # Stage 2: imputed treatment effects
        # For treated rows:  D1 = Y1 - mu0(X1)
        # For control rows:  D0 = mu1(X0) - Y0
        D1 = Y1 - self._mu0.predict(X1)
        D0 = self._mu1.predict(X0) - Y0
        self._tau1 = self.regressor_factory().fit(X1, D1)
        self._tau0 = self.regressor_factory().fit(X0, D0)

        # Propensity
        self._propensity = propensity or PropensityEstimator().fit(X, T)
        logger.info(
            "XLearner fit: n_control=%d n_treated=%d", len(X0), len(X1),
        )
        return self

    def predict(self, X) -> np.ndarray:
        if self._tau0 is None or self._tau1 is None \
                or self._propensity is None:
            raise RuntimeError("XLearner must be fit before predict")
        X = np.asarray(X, dtype=float)
        t0 = self._tau0.predict(X)
        t1 = self._tau1.predict(X)
        g = self._propensity.predict_proba(X)
        # Down-weight the rarer-arm CATE: use propensity as mixing weight.
        return g * t0 + (1.0 - g) * t1


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def qini_coefficient(
    uplift_pred: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
) -> float:
    """Normalized Qini coefficient (area under Qini curve).

    Ranges in ``[-1, 1]`` where higher is better. Zero = random.
    ``Y`` is the observed outcome; ``T`` is treatment assignment (0/1).
    """
    u = np.asarray(uplift_pred, dtype=float)
    T = np.asarray(T).astype(int)
    Y = np.asarray(Y, dtype=float)
    if not (len(u) == len(T) == len(Y)):
        raise ValueError("uplift_pred / T / Y length mismatch")
    order = np.argsort(-u)           # descending
    u_sorted, T_sorted, Y_sorted = u[order], T[order], Y[order]
    n = len(u)
    qini_vals = np.zeros(n)
    cum_t, cum_t_y, cum_c, cum_c_y = 0, 0.0, 0, 0.0
    for i in range(n):
        if T_sorted[i] == 1:
            cum_t += 1
            cum_t_y += Y_sorted[i]
        else:
            cum_c += 1
            cum_c_y += Y_sorted[i]
        if cum_t == 0 or cum_c == 0:
            qini_vals[i] = 0.0
        else:
            qini_vals[i] = cum_t_y - cum_c_y * (cum_t / cum_c)
    # Normalize by area of ideal / perfect-target ranking
    total_t = int(np.sum(T == 1))
    total_c = int(np.sum(T == 0))
    if total_t == 0 or total_c == 0:
        return 0.0
    total_y_t = float(np.sum(Y[T == 1]))
    total_y_c = float(np.sum(Y[T == 0]))
    perfect = total_y_t - total_y_c * (total_t / total_c)
    if abs(perfect) < 1e-12:
        return 0.0
    return float(np.mean(qini_vals) / perfect)


def uplift_at_k(
    uplift_pred: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    k: float = 0.1,
) -> float:
    """Mean observed uplift over the top-K% predicted uplift subset.

    ``k`` is the fraction in ``(0, 1]``. Useful for business deployment
    planning (e.g. "top 10% targeted recipients"). Returns
    ``mean(Y|top-K, T=1) - mean(Y|top-K, T=0)`` within the selected
    subset; 0 when one arm is empty inside the top-K slice.
    """
    if not (0.0 < k <= 1.0):
        raise ValueError(f"k={k} must be in (0, 1]")
    u = np.asarray(uplift_pred, dtype=float)
    T = np.asarray(T).astype(int)
    Y = np.asarray(Y, dtype=float)
    n = len(u)
    cutoff = max(1, int(round(k * n)))
    order = np.argsort(-u)[:cutoff]
    T_top, Y_top = T[order], Y[order]
    if not np.any(T_top == 1) or not np.any(T_top == 0):
        return 0.0
    return float(np.mean(Y_top[T_top == 1]) - np.mean(Y_top[T_top == 0]))


def evaluate_uplift(
    uplift_pred: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    k_values: Tuple[float, ...] = (0.1, 0.2, 0.3),
) -> Dict[str, float]:
    """Shorthand: compute qini + uplift@k bundle for reporting."""
    out: Dict[str, float] = {
        "qini": qini_coefficient(uplift_pred, T, Y),
    }
    for k in k_values:
        out[f"uplift@{int(k * 100)}"] = uplift_at_k(
            uplift_pred, T, Y, k=k,
        )
    return out
