"""
Offline policy evaluation using IPS, SNIPS, and Doubly Robust estimators.

Given logged interaction data collected under a *behaviour policy*, these
estimators evaluate the expected reward of a *new (target) policy* without
deploying it online.

Supports:
- **IPS** (Inverse Propensity Scoring): unbiased but high variance.
- **SNIPS** (Self-Normalized IPS): biased but dramatically lower variance.
- **DR** (Doubly Robust): combines a reward model with IPS correction for
  double robustness -- consistent if *either* the propensity or the reward
  model is correct.

Multi-task aware: evaluations can be run per-task for the PLE platform.

Usage::

    evaluator = OfflinePolicyEvaluator(estimator_types=["ips", "snips", "dr"])
    result = evaluator.evaluate(
        rewards=rewards,
        actions=actions,
        new_policy_probs=new_probs,
        propensity_scores=prop_scores,
        reward_model_preds=reward_hat,
    )
    print(result.estimates)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class EstimatorType(str, Enum):
    """Supported off-policy estimator types."""

    IPS = "ips"
    SNIPS = "snips"
    DR = "dr"


@dataclass
class OPEResult:
    """Result container for offline policy evaluation.

    Attributes
    ----------
    estimates : dict[str, float]
        Point estimates keyed by estimator name.
    confidence_intervals : dict[str, tuple[float, float]]
        ``(lower, upper)`` 95 % bootstrap confidence intervals.
    effective_sample_size : float
        Kish's effective sample size based on importance weights.
    max_weight : float
        Largest importance weight observed (diagnostic for weight clipping).
    n_samples : int
        Number of logged samples used.
    task_name : str, optional
        Task identifier when running multi-task evaluation.
    """

    estimates: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    effective_sample_size: float = 0.0
    max_weight: float = 0.0
    n_samples: int = 0
    task_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Core estimators (stateless functions)
# ---------------------------------------------------------------------------

def _importance_weights(
    new_policy_probs: np.ndarray,
    propensity_scores: np.ndarray,
    clip: float = 100.0,
) -> np.ndarray:
    """Compute clipped importance weights ``w = pi_new / pi_old``.

    Parameters
    ----------
    new_policy_probs : np.ndarray
        Target policy probability for the *observed* action, shape ``(n,)``.
    propensity_scores : np.ndarray
        Behaviour policy probability for the *observed* action, shape ``(n,)``.
    clip : float
        Maximum allowed weight (reduces variance at the cost of bias).

    Returns
    -------
    np.ndarray
        Clipped importance weights, shape ``(n,)``.
    """
    safe_prop = np.maximum(propensity_scores, 1e-10)
    weights = new_policy_probs / safe_prop
    return np.clip(weights, 0.0, clip)


def estimate_ips(
    rewards: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Inverse Propensity Scoring estimator.

    .. math::
        \\hat{V}_{IPS} = \\frac{1}{n} \\sum_{i=1}^{n} w_i \\, r_i

    Parameters
    ----------
    rewards : np.ndarray
        Observed rewards, shape ``(n,)``.
    weights : np.ndarray
        Importance weights, shape ``(n,)``.

    Returns
    -------
    float
        IPS estimate of the target policy value.
    """
    n = len(rewards)
    if n == 0:
        return 0.0
    return float(np.mean(weights * rewards))


def estimate_snips(
    rewards: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Self-Normalized Inverse Propensity Scoring estimator.

    .. math::
        \\hat{V}_{SNIPS} = \\frac{\\sum w_i \\, r_i}{\\sum w_i}

    Parameters
    ----------
    rewards : np.ndarray
        Observed rewards, shape ``(n,)``.
    weights : np.ndarray
        Importance weights, shape ``(n,)``.

    Returns
    -------
    float
        SNIPS estimate of the target policy value.
    """
    w_sum = float(np.sum(weights))
    if w_sum < 1e-10:
        return 0.0
    return float(np.sum(weights * rewards) / w_sum)


def estimate_dr(
    rewards: np.ndarray,
    weights: np.ndarray,
    reward_model_preds: np.ndarray,
    new_policy_reward_preds: np.ndarray,
) -> float:
    """Doubly Robust estimator.

    .. math::
        \\hat{V}_{DR} = \\frac{1}{n} \\sum_{i=1}^{n}
            \\left[ \\hat{r}_{\\pi}(x_i)
            + w_i \\bigl(r_i - \\hat{r}(x_i, a_i)\\bigr) \\right]

    where ``reward_model_preds`` is :math:`\\hat{r}(x_i, a_i)` (the reward
    model prediction for the *observed* action) and
    ``new_policy_reward_preds`` is :math:`\\hat{r}_{\\pi}(x_i)` (the expected
    reward under the *target* policy, computed as the weighted average of the
    reward model over all actions using target policy probabilities).

    Parameters
    ----------
    rewards : np.ndarray
        Observed rewards, shape ``(n,)``.
    weights : np.ndarray
        Importance weights, shape ``(n,)``.
    reward_model_preds : np.ndarray
        Reward model prediction for the observed action, shape ``(n,)``.
    new_policy_reward_preds : np.ndarray
        Expected reward under the target policy, shape ``(n,)``.

    Returns
    -------
    float
        DR estimate of the target policy value.
    """
    n = len(rewards)
    if n == 0:
        return 0.0
    correction = weights * (rewards - reward_model_preds)
    return float(np.mean(new_policy_reward_preds + correction))


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------

def _effective_sample_size(weights: np.ndarray) -> float:
    """Kish's effective sample size: ``(sum w)^2 / sum(w^2)``."""
    w_sum = float(np.sum(weights))
    w_sq_sum = float(np.sum(weights ** 2))
    if w_sq_sum < 1e-10:
        return 0.0
    return w_sum ** 2 / w_sq_sum


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    rewards: np.ndarray,
    weights: np.ndarray,
    estimator_fn,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for an OPE estimator.

    Parameters
    ----------
    rewards, weights : np.ndarray
        Logged rewards and importance weights.
    estimator_fn : callable
        One of ``estimate_ips``, ``estimate_snips``, ``estimate_dr``.
    n_bootstrap : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (0.05 -> 95 % CI).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    **kwargs
        Extra arrays forwarded to ``estimator_fn`` (e.g.
        ``reward_model_preds``, ``new_policy_reward_preds``).

    Returns
    -------
    tuple[float, float]
        ``(lower, upper)`` confidence bounds.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(rewards)
    estimates = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        r_b = rewards[idx]
        w_b = weights[idx]
        kw = {k: v[idx] for k, v in kwargs.items()}
        estimates[b] = estimator_fn(r_b, w_b, **kw)

    lo = float(np.percentile(estimates, 100 * alpha / 2))
    hi = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------

class OfflinePolicyEvaluator:
    """Offline policy evaluation with IPS / SNIPS / DR estimators.

    Parameters
    ----------
    estimator_types : list[str | EstimatorType]
        Estimators to compute.  Defaults to all three.
    weight_clip : float
        Maximum importance weight (controls variance-bias trade-off).
    n_bootstrap : int
        Number of bootstrap resamples for confidence intervals.
    ci_alpha : float
        Significance level for confidence intervals.
    random_seed : int
        Seed for bootstrap reproducibility.

    Examples
    --------
    >>> evaluator = OfflinePolicyEvaluator()
    >>> result = evaluator.evaluate(
    ...     rewards=rewards,
    ...     actions=actions,
    ...     new_policy_probs=new_probs,
    ...     propensity_scores=prop_scores,
    ... )
    >>> print(result.estimates["snips"])
    """

    def __init__(
        self,
        estimator_types: Optional[Sequence[Union[str, EstimatorType]]] = None,
        weight_clip: float = 100.0,
        n_bootstrap: int = 1000,
        ci_alpha: float = 0.05,
        random_seed: int = 42,
    ) -> None:
        if estimator_types is None:
            self._estimator_types = [EstimatorType.IPS, EstimatorType.SNIPS, EstimatorType.DR]
        else:
            self._estimator_types = [
                EstimatorType(e) if isinstance(e, str) else e
                for e in estimator_types
            ]
        self.weight_clip = weight_clip
        self.n_bootstrap = n_bootstrap
        self.ci_alpha = ci_alpha
        self._rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Single-task evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        rewards: np.ndarray,
        actions: np.ndarray,
        new_policy_probs: np.ndarray,
        propensity_scores: np.ndarray,
        reward_model_preds: Optional[np.ndarray] = None,
        new_policy_reward_preds: Optional[np.ndarray] = None,
        task_name: Optional[str] = None,
    ) -> OPEResult:
        """Run offline policy evaluation on logged data.

        Parameters
        ----------
        rewards : np.ndarray
            Observed rewards for each logged interaction, shape ``(n,)``.
        actions : np.ndarray
            Actions taken by the behaviour policy, shape ``(n,)``.
        new_policy_probs : np.ndarray
            Target policy probability for the *observed* action, shape ``(n,)``.
            (i.e., ``pi_new(a_i | x_i)`` for each logged sample *i*.)
        propensity_scores : np.ndarray
            Behaviour policy probability for the *observed* action, shape ``(n,)``.
        reward_model_preds : np.ndarray, optional
            Reward model prediction for the observed ``(x, a)`` pair.
            Required when ``"dr"`` estimator is active.
        new_policy_reward_preds : np.ndarray, optional
            Expected reward under the target policy, shape ``(n,)``.
            Required when ``"dr"`` estimator is active.
        task_name : str, optional
            Task identifier (for multi-task reporting).

        Returns
        -------
        OPEResult
            Contains point estimates and bootstrap confidence intervals.
        """
        rewards = np.asarray(rewards, dtype=np.float64).ravel()
        new_policy_probs = np.asarray(new_policy_probs, dtype=np.float64).ravel()
        propensity_scores = np.asarray(propensity_scores, dtype=np.float64).ravel()

        n = len(rewards)
        if n == 0:
            logger.warning("Empty data passed to OfflinePolicyEvaluator.evaluate().")
            return OPEResult(task_name=task_name)

        # Importance weights
        weights = _importance_weights(new_policy_probs, propensity_scores, clip=self.weight_clip)
        ess = _effective_sample_size(weights)

        logger.info(
            "OPE evaluate (task=%s): n=%d, ESS=%.1f, max_weight=%.2f",
            task_name or "default", n, ess, float(np.max(weights)),
        )

        estimates: Dict[str, float] = {}
        cis: Dict[str, Tuple[float, float]] = {}

        # IPS
        if EstimatorType.IPS in self._estimator_types:
            estimates["ips"] = estimate_ips(rewards, weights)
            cis["ips"] = _bootstrap_ci(
                rewards, weights, estimate_ips,
                n_bootstrap=self.n_bootstrap, alpha=self.ci_alpha,
                rng=np.random.default_rng(self._rng.integers(0, 2**32)),
            )

        # SNIPS
        if EstimatorType.SNIPS in self._estimator_types:
            estimates["snips"] = estimate_snips(rewards, weights)
            cis["snips"] = _bootstrap_ci(
                rewards, weights, estimate_snips,
                n_bootstrap=self.n_bootstrap, alpha=self.ci_alpha,
                rng=np.random.default_rng(self._rng.integers(0, 2**32)),
            )

        # DR
        if EstimatorType.DR in self._estimator_types:
            if reward_model_preds is None or new_policy_reward_preds is None:
                logger.warning(
                    "DR estimator requested but reward model predictions are missing. "
                    "Skipping DR."
                )
            else:
                rm_preds = np.asarray(reward_model_preds, dtype=np.float64).ravel()
                np_rp = np.asarray(new_policy_reward_preds, dtype=np.float64).ravel()
                estimates["dr"] = estimate_dr(rewards, weights, rm_preds, np_rp)
                cis["dr"] = _bootstrap_ci(
                    rewards, weights, estimate_dr,
                    n_bootstrap=self.n_bootstrap, alpha=self.ci_alpha,
                    rng=np.random.default_rng(self._rng.integers(0, 2**32)),
                    reward_model_preds=rm_preds,
                    new_policy_reward_preds=np_rp,
                )

        return OPEResult(
            estimates=estimates,
            confidence_intervals=cis,
            effective_sample_size=ess,
            max_weight=float(np.max(weights)),
            n_samples=n,
            task_name=task_name,
        )

    # ------------------------------------------------------------------
    # Multi-task evaluation
    # ------------------------------------------------------------------

    def evaluate_multi_task(
        self,
        task_data: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, OPEResult]:
        """Evaluate multiple tasks independently.

        Parameters
        ----------
        task_data : dict[str, dict[str, np.ndarray]]
            Mapping ``{task_name: {"rewards": ..., "actions": ...,
            "new_policy_probs": ..., "propensity_scores": ...,
            "reward_model_preds": ... (optional),
            "new_policy_reward_preds": ... (optional)}}``.

        Returns
        -------
        dict[str, OPEResult]
            Per-task evaluation results.
        """
        results: Dict[str, OPEResult] = {}
        for task_name, data in task_data.items():
            logger.info("Evaluating task '%s'.", task_name)
            results[task_name] = self.evaluate(
                rewards=data["rewards"],
                actions=data["actions"],
                new_policy_probs=data["new_policy_probs"],
                propensity_scores=data["propensity_scores"],
                reward_model_preds=data.get("reward_model_preds"),
                new_policy_reward_preds=data.get("new_policy_reward_preds"),
                task_name=task_name,
            )
        return results


__all__ = [
    "OfflinePolicyEvaluator",
    "OPEResult",
    "EstimatorType",
    "estimate_ips",
    "estimate_snips",
    "estimate_dr",
]
