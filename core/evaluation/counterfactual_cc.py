"""
Counterfactual Champion-Challenger evaluator (Sprint 2 S14).

Off-policy evaluation via Inverse Propensity Scoring (IPS) and Self-Normalized
IPS (SNIPS). Produces a counterfactual expected reward for a challenger
policy using logged data collected under the champion policy, along with
bootstrap confidence intervals.

This is a lightweight numerical implementation — it does not compete with
libraries such as `obp` or `ab-test-sharp`, but it is dependency-light
(numpy only) and integrates with the existing `ModelCompetition`
offline-gate flow.

Legal / MRM basis:
- EU AI Act Art. 15 (accuracy) supports off-policy evaluation as a valid
  validation technique for recommender systems.
- 금감원 SR 11-7 (effective challenge): requires evidence that a challenger
  would have produced better outcomes than the champion before promotion.

API
---
>>> evaluator = CounterfactualEvaluator()
>>> result = evaluator.compare(
...     rewards=[0.0, 1.0, 0.0, 1.0],
...     logged_propensities=[0.5, 0.5, 0.5, 0.5],
...     challenger_propensities=[0.2, 0.8, 0.3, 0.7],
... )
>>> result.ips_estimate, result.snips_estimate, result.challenger_better
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "CounterfactualResult",
    "CounterfactualEvaluator",
    "ips_estimate",
    "snips_estimate",
    "paired_bootstrap_ci",
]


# ---------------------------------------------------------------------------
# Core estimators
# ---------------------------------------------------------------------------

def _as_arrays(
    rewards, logged_propensities, new_propensities,
):
    r = np.asarray(rewards, dtype=float)
    p_log = np.asarray(logged_propensities, dtype=float)
    p_new = np.asarray(new_propensities, dtype=float)
    if not (len(r) == len(p_log) == len(p_new)):
        raise ValueError(
            f"length mismatch: rewards={len(r)} "
            f"logged={len(p_log)} new={len(p_new)}"
        )
    if np.any(p_log <= 0):
        raise ValueError("logged_propensities must all be > 0 for IPS")
    return r, p_log, p_new


def ips_estimate(
    rewards: Sequence[float],
    logged_propensities: Sequence[float],
    new_propensities: Sequence[float],
    clip: Optional[float] = 10.0,
) -> float:
    """Vanilla Inverse Propensity Scoring.

    V_IPS = (1/n) Σ (p_new / p_log) * r

    `clip` caps the importance weight to reduce variance from tiny
    logged propensities. Set to None to disable.
    """
    r, p_log, p_new = _as_arrays(rewards, logged_propensities, new_propensities)
    weights = p_new / p_log
    if clip is not None:
        weights = np.clip(weights, 0.0, clip)
    return float((weights * r).mean())


def snips_estimate(
    rewards: Sequence[float],
    logged_propensities: Sequence[float],
    new_propensities: Sequence[float],
    clip: Optional[float] = 10.0,
) -> float:
    """Self-Normalized IPS.

    V_SNIPS = Σ (p_new / p_log) * r  /  Σ (p_new / p_log)

    Less biased than IPS when weights have heavy tails.
    """
    r, p_log, p_new = _as_arrays(rewards, logged_propensities, new_propensities)
    weights = p_new / p_log
    if clip is not None:
        weights = np.clip(weights, 0.0, clip)
    denom = weights.sum()
    if denom <= 0:
        logger.warning("SNIPS denominator <= 0; returning 0.0")
        return 0.0
    return float((weights * r).sum() / denom)


def paired_bootstrap_ci(
    rewards: Sequence[float],
    logged_propensities: Sequence[float],
    challenger_propensities: Sequence[float],
    champion_propensities: Optional[Sequence[float]] = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = 0,
    estimator: str = "snips",
    clip: Optional[float] = 10.0,
) -> Dict[str, float]:
    """Paired bootstrap CI for (challenger - champion) counterfactual reward.

    If ``champion_propensities`` is None we assume the logged distribution
    *is* the champion — reward under champion is simply the logged mean.
    """
    r, p_log, p_chal = _as_arrays(
        rewards, logged_propensities, challenger_propensities,
    )
    n = len(r)
    if champion_propensities is None:
        p_champ = p_log.copy()
    else:
        p_champ = np.asarray(champion_propensities, dtype=float)
        if len(p_champ) != n:
            raise ValueError("champion_propensities length mismatch")

    rng = np.random.default_rng(random_state)
    estimator_fn = snips_estimate if estimator == "snips" else ips_estimate
    diffs: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        c_est = estimator_fn(
            r[idx], p_log[idx], p_chal[idx], clip=clip,
        )
        ch_est = estimator_fn(
            r[idx], p_log[idx], p_champ[idx], clip=clip,
        )
        diffs.append(c_est - ch_est)
    diffs_arr = np.asarray(diffs)
    lower = float(np.quantile(diffs_arr, alpha / 2))
    upper = float(np.quantile(diffs_arr, 1 - alpha / 2))
    return {
        "mean_diff": float(diffs_arr.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
        "significant": (lower > 0) or (upper < 0),
    }


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualResult:
    ips_estimate: float
    snips_estimate: float
    champion_estimate: float
    challenger_estimate: float
    estimated_lift: float            # challenger - champion
    bootstrap: Dict[str, float] = field(default_factory=dict)
    challenger_better: bool = False
    reason: str = ""
    sample_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ips_estimate": self.ips_estimate,
            "snips_estimate": self.snips_estimate,
            "champion_estimate": self.champion_estimate,
            "challenger_estimate": self.challenger_estimate,
            "estimated_lift": self.estimated_lift,
            "bootstrap": dict(self.bootstrap),
            "challenger_better": self.challenger_better,
            "reason": self.reason,
            "sample_size": self.sample_size,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class CounterfactualEvaluator:
    """Config-driven counterfactual Champion-Challenger comparison."""

    def __init__(
        self,
        estimator: str = "snips",
        min_lift: float = 0.0,
        clip: Optional[float] = 10.0,
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
    ) -> None:
        if estimator not in ("ips", "snips"):
            raise ValueError(f"estimator={estimator!r} must be 'ips' or 'snips'")
        self.estimator = estimator
        self.min_lift = min_lift
        self.clip = clip
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha

    @classmethod
    def from_config(
        cls, config: Optional[Dict[str, Any]] = None,
    ) -> "CounterfactualEvaluator":
        cfg = ((config or {}).get("serving") or {}).get("counterfactual_cc") or {}
        return cls(
            estimator=str(cfg.get("estimator", "snips")),
            min_lift=float(cfg.get("min_lift", 0.0)),
            clip=(None if cfg.get("clip") is None else float(cfg["clip"])),
            n_bootstrap=int(cfg.get("n_bootstrap", 1000)),
            alpha=float(cfg.get("alpha", 0.05)),
        )

    def compare(
        self,
        rewards: Sequence[float],
        logged_propensities: Sequence[float],
        challenger_propensities: Sequence[float],
        champion_propensities: Optional[Sequence[float]] = None,
    ) -> CounterfactualResult:
        """Return a full counterfactual comparison result."""
        r, p_log, p_chal = _as_arrays(
            rewards, logged_propensities, challenger_propensities,
        )
        # Champion estimate: reward under champion; if champion not provided,
        # use logged propensities (collection policy == champion).
        if champion_propensities is None:
            p_champ = p_log.copy()
        else:
            p_champ = np.asarray(champion_propensities, dtype=float)

        ips_val = ips_estimate(r, p_log, p_chal, clip=self.clip)
        snips_val = snips_estimate(r, p_log, p_chal, clip=self.clip)
        chal_est = snips_val if self.estimator == "snips" else ips_val
        champ_est = (
            snips_estimate(r, p_log, p_champ, clip=self.clip)
            if self.estimator == "snips"
            else ips_estimate(r, p_log, p_champ, clip=self.clip)
        )
        lift = chal_est - champ_est

        boot = paired_bootstrap_ci(
            rewards=r, logged_propensities=p_log,
            challenger_propensities=p_chal,
            champion_propensities=p_champ,
            n_bootstrap=self.n_bootstrap, alpha=self.alpha,
            estimator=self.estimator, clip=self.clip,
        )
        challenger_better = (
            lift >= self.min_lift and boot.get("ci_lower", -math.inf) > 0
        )
        reason = (
            f"{self.estimator.upper()} challenger={chal_est:.4f}, "
            f"champion={champ_est:.4f}, lift={lift:.4f}, "
            f"CI[{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]"
        )
        if not challenger_better:
            if lift < self.min_lift:
                reason += f"; lift below min_lift={self.min_lift}"
            elif boot.get("ci_lower", -math.inf) <= 0:
                reason += "; CI includes zero — not significant"

        return CounterfactualResult(
            ips_estimate=ips_val,
            snips_estimate=snips_val,
            champion_estimate=champ_est,
            challenger_estimate=chal_est,
            estimated_lift=lift,
            bootstrap=boot,
            challenger_better=challenger_better,
            reason=reason,
            sample_size=len(r),
        )
