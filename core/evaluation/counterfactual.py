"""
Counterfactual Policy Evaluation (Stage C Interpretability)
============================================================

Offline policy evaluation using IPS, SNIPS, and Doubly Robust estimators.
Wraps the lower-level :class:`OfflinePolicyEvaluator` with a higher-level
API that mirrors the reference project's ``CounterfactualEvaluator`` while
staying dependency-light (no DuckDB requirement -- works on numpy arrays).

Key additions over the existing ``OfflinePolicyEvaluator``:
- Per-estimator :class:`PolicyEvaluationResult` dataclass with ``go_decision``
- Sensitivity analysis across clipping thresholds
- Go/No-Go deployment decision framework
- Markdown report generation

Reference: gotothemoon/workspace/code/src/evaluation/counterfactual.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .policy_evaluator import (
    OfflinePolicyEvaluator,
    OPEResult,
    EstimatorType,
    _importance_weights,
    _effective_sample_size,
    _bootstrap_ci,
    estimate_ips,
    estimate_snips,
    estimate_dr,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CounterfactualEvaluator",
    "PolicyEvaluationResult",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PolicyEvaluationResult:
    """Result of a single counterfactual policy estimator.

    Attributes
    ----------
    estimated_value : float
        Point estimate of the target policy value.
    ci_lower : float
        Lower bound of the 95 % bootstrap confidence interval.
    ci_upper : float
        Upper bound of the 95 % bootstrap confidence interval.
    standard_error : float
        Bootstrap standard error.
    effective_sample_size : float
        Kish's effective sample size.
    estimator : str
        Name of the estimator (``"ips"`` / ``"snips"`` / ``"dr"``).
    go_decision : bool
        ``True`` if ``estimated_value`` exceeds the baseline.
    n_samples : int
        Number of logged samples used.
    timestamp : str
        ISO-8601 generation timestamp.
    """

    estimated_value: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    standard_error: float = 0.0
    effective_sample_size: float = 0.0
    estimator: str = "ips"
    go_decision: bool = False
    n_samples: int = 0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "estimator": self.estimator,
            "estimated_value": float(self.estimated_value),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "standard_error": float(self.standard_error),
            "effective_sample_size": float(self.effective_sample_size),
            "go_decision": self.go_decision,
            "n_samples": self.n_samples,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# CounterfactualEvaluator
# ---------------------------------------------------------------------------

class CounterfactualEvaluator:
    """Offline A/B testing substitute using logged data.

    Provides IPS, SNIPS, and Doubly Robust estimators with bootstrap
    confidence intervals, sensitivity analysis over clipping thresholds,
    and a Go/No-Go deployment decision framework.

    Parameters
    ----------
    propensity_clip_range : tuple[float, float]
        ``(min_clip, max_clip)`` for importance weight clipping.
        The *min_clip* value clamps propensity scores from below to
        avoid division-by-near-zero; *max_clip* caps the resulting
        importance weights.
    n_bootstrap : int
        Number of bootstrap resamples for confidence intervals.
    ci_alpha : float
        Significance level for CIs (0.05 -> 95 % CI).
    random_seed : int
        Seed for reproducibility.
    output_dir : str | Path | None
        Optional directory for saving reports and artefacts.

    Examples
    --------
    >>> evaluator = CounterfactualEvaluator()
    >>> result = evaluator.evaluate_ips(rewards, logging_probs, new_probs)
    >>> print(result.estimated_value, result.go_decision)
    """

    def __init__(
        self,
        propensity_clip_range: Tuple[float, float] = (0.01, 100.0),
        n_bootstrap: int = 1000,
        ci_alpha: float = 0.05,
        random_seed: int = 42,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.prop_clip_min, self.prop_clip_max = propensity_clip_range
        self.n_bootstrap = n_bootstrap
        self.ci_alpha = ci_alpha
        self._rng = np.random.default_rng(random_seed)

        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Accumulate results for report generation
        self.results: List[PolicyEvaluationResult] = []

        logger.info(
            "CounterfactualEvaluator initialised: clip=(%.4f, %.1f), "
            "n_bootstrap=%d",
            self.prop_clip_min, self.prop_clip_max, self.n_bootstrap,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_weights(
        self,
        logging_probs: np.ndarray,
        new_probs: np.ndarray,
        clip_max: Optional[float] = None,
    ) -> np.ndarray:
        """Compute clipped importance weights."""
        safe_log = np.maximum(logging_probs, self.prop_clip_min)
        weights = new_probs / safe_log
        max_w = clip_max if clip_max is not None else self.prop_clip_max
        return np.clip(weights, 0.0, max_w)

    def _make_result(
        self,
        estimator_name: str,
        value: float,
        ci: Tuple[float, float],
        weights: np.ndarray,
        n_samples: int,
        baseline: float = 0.0,
    ) -> PolicyEvaluationResult:
        """Build a :class:`PolicyEvaluationResult`."""
        bootstrap_values = np.array([value])  # placeholder for SE
        se = abs(ci[1] - ci[0]) / (2 * 1.96) if ci[1] != ci[0] else 0.0
        ess = _effective_sample_size(weights)

        result = PolicyEvaluationResult(
            estimated_value=value,
            ci_lower=ci[0],
            ci_upper=ci[1],
            standard_error=se,
            effective_sample_size=ess,
            estimator=estimator_name,
            go_decision=value > baseline,
            n_samples=n_samples,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
        self.results.append(result)
        return result

    # ------------------------------------------------------------------
    # Public estimators
    # ------------------------------------------------------------------

    def evaluate_ips(
        self,
        rewards: np.ndarray,
        logging_probs: np.ndarray,
        new_probs: np.ndarray,
        baseline: float = 0.0,
    ) -> PolicyEvaluationResult:
        """Inverse Propensity Scoring: ``E[r * w]`` where ``w = new_prob / logging_prob``.

        Parameters
        ----------
        rewards : np.ndarray
            Observed rewards, shape ``(n,)``.
        logging_probs : np.ndarray
            Behaviour policy probabilities, shape ``(n,)``.
        new_probs : np.ndarray
            Target policy probabilities, shape ``(n,)``.
        baseline : float
            Baseline policy value for go/no-go decision.

        Returns
        -------
        PolicyEvaluationResult
        """
        rewards = np.asarray(rewards, dtype=np.float64).ravel()
        logging_probs = np.asarray(logging_probs, dtype=np.float64).ravel()
        new_probs = np.asarray(new_probs, dtype=np.float64).ravel()
        n = len(rewards)

        weights = self._compute_weights(logging_probs, new_probs)
        value = estimate_ips(rewards, weights)
        ci = _bootstrap_ci(
            rewards, weights, estimate_ips,
            n_bootstrap=self.n_bootstrap, alpha=self.ci_alpha,
            rng=np.random.default_rng(self._rng.integers(0, 2**32)),
        )

        logger.info(
            "[IPS] value=%.6f, CI=[%.6f, %.6f], ESS=%.1f",
            value, ci[0], ci[1], _effective_sample_size(weights),
        )
        return self._make_result("ips", value, ci, weights, n, baseline)

    def evaluate_snips(
        self,
        rewards: np.ndarray,
        logging_probs: np.ndarray,
        new_probs: np.ndarray,
        baseline: float = 0.0,
    ) -> PolicyEvaluationResult:
        """Self-Normalized IPS: ``sum(r * w) / sum(w)``.

        Lower variance than IPS at the cost of slight bias.

        Parameters
        ----------
        rewards, logging_probs, new_probs, baseline
            See :meth:`evaluate_ips`.

        Returns
        -------
        PolicyEvaluationResult
        """
        rewards = np.asarray(rewards, dtype=np.float64).ravel()
        logging_probs = np.asarray(logging_probs, dtype=np.float64).ravel()
        new_probs = np.asarray(new_probs, dtype=np.float64).ravel()
        n = len(rewards)

        weights = self._compute_weights(logging_probs, new_probs)
        value = estimate_snips(rewards, weights)
        ci = _bootstrap_ci(
            rewards, weights, estimate_snips,
            n_bootstrap=self.n_bootstrap, alpha=self.ci_alpha,
            rng=np.random.default_rng(self._rng.integers(0, 2**32)),
        )

        logger.info(
            "[SNIPS] value=%.6f, CI=[%.6f, %.6f], ESS=%.1f",
            value, ci[0], ci[1], _effective_sample_size(weights),
        )
        return self._make_result("snips", value, ci, weights, n, baseline)

    def evaluate_dr(
        self,
        rewards: np.ndarray,
        logging_probs: np.ndarray,
        new_probs: np.ndarray,
        reward_model_preds: np.ndarray,
        baseline: float = 0.0,
    ) -> PolicyEvaluationResult:
        """Doubly Robust estimator.

        Combines a direct reward model with IPS correction::

            value = mean(reward_model_pred + w * (reward - reward_model_pred))

        Parameters
        ----------
        rewards : np.ndarray
            Observed rewards, shape ``(n,)``.
        logging_probs : np.ndarray
            Behaviour policy probabilities, shape ``(n,)``.
        new_probs : np.ndarray
            Target policy probabilities, shape ``(n,)``.
        reward_model_preds : np.ndarray
            Reward model predictions for observed ``(x, a)`` pairs,
            shape ``(n,)``.
        baseline : float
            Baseline policy value for go/no-go decision.

        Returns
        -------
        PolicyEvaluationResult
        """
        rewards = np.asarray(rewards, dtype=np.float64).ravel()
        logging_probs = np.asarray(logging_probs, dtype=np.float64).ravel()
        new_probs = np.asarray(new_probs, dtype=np.float64).ravel()
        reward_model_preds = np.asarray(reward_model_preds, dtype=np.float64).ravel()
        n = len(rewards)

        weights = self._compute_weights(logging_probs, new_probs)

        # DR formula: reward_hat + w * (r - reward_hat)
        dr_values = reward_model_preds + weights * (rewards - reward_model_preds)
        value = float(np.mean(dr_values))

        # Bootstrap CI -- we need a custom estimator fn for DR
        def _dr_estimator(r: np.ndarray, w: np.ndarray, **kw) -> float:
            rmp = kw["reward_model_preds"]
            return float(np.mean(rmp + w * (r - rmp)))

        ci = _bootstrap_ci(
            rewards, weights, _dr_estimator,
            n_bootstrap=self.n_bootstrap, alpha=self.ci_alpha,
            rng=np.random.default_rng(self._rng.integers(0, 2**32)),
            reward_model_preds=reward_model_preds,
        )

        logger.info(
            "[DR] value=%.6f, CI=[%.6f, %.6f], ESS=%.1f",
            value, ci[0], ci[1], _effective_sample_size(weights),
        )
        return self._make_result("dr", value, ci, weights, n, baseline)

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        rewards: np.ndarray,
        logging_probs: np.ndarray,
        new_probs: np.ndarray,
        clip_values: Optional[List[float]] = None,
        baseline: float = 0.0,
    ) -> Dict[str, Any]:
        """Test sensitivity of SNIPS to the propensity clipping threshold.

        Runs SNIPS at each clip value and returns a mapping of
        ``clip_value -> PolicyEvaluationResult``.

        Parameters
        ----------
        rewards, logging_probs, new_probs
            Logged data arrays.
        clip_values : list[float], optional
            Clipping thresholds to test. Defaults to ``[5, 10, 20, 50, 100]``.
        baseline : float
            Baseline for go/no-go.

        Returns
        -------
        dict
            ``{"clip_results": {clip: result_dict}, "recommended_clip": float}``
        """
        if clip_values is None:
            clip_values = [5.0, 10.0, 20.0, 50.0, 100.0]

        rewards = np.asarray(rewards, dtype=np.float64).ravel()
        logging_probs = np.asarray(logging_probs, dtype=np.float64).ravel()
        new_probs = np.asarray(new_probs, dtype=np.float64).ravel()

        clip_results: Dict[float, Dict[str, Any]] = {}
        best_ess_ratio = 0.0
        recommended_clip = clip_values[-1]

        for clip_val in clip_values:
            weights = self._compute_weights(logging_probs, new_probs, clip_max=clip_val)
            value = estimate_snips(rewards, weights)
            ess = _effective_sample_size(weights)
            n = len(rewards)

            ci = _bootstrap_ci(
                rewards, weights, estimate_snips,
                n_bootstrap=min(self.n_bootstrap, 200),  # faster for sensitivity
                alpha=self.ci_alpha,
                rng=np.random.default_rng(self._rng.integers(0, 2**32)),
            )

            ess_ratio = ess / n if n > 0 else 0.0
            clip_results[clip_val] = {
                "estimated_value": float(value),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
                "effective_sample_size": float(ess),
                "ess_ratio": float(ess_ratio),
                "max_weight": float(np.max(weights)),
                "go_decision": value > baseline,
            }

            # Recommend clip with best ESS ratio that still shows positive lift
            if ess_ratio > best_ess_ratio and value > baseline:
                best_ess_ratio = ess_ratio
                recommended_clip = clip_val

        logger.info(
            "Sensitivity analysis: %d clip values tested, recommended=%.1f",
            len(clip_values), recommended_clip,
        )

        return {
            "clip_results": {str(k): v for k, v in clip_results.items()},
            "recommended_clip": recommended_clip,
            "baseline": baseline,
        }

    # ------------------------------------------------------------------
    # Go / No-Go decision framework
    # ------------------------------------------------------------------

    def go_no_go(
        self,
        current_policy_value: float,
        new_policy_result: PolicyEvaluationResult,
        min_improvement: float = 0.02,
        min_ess_ratio: float = 0.10,
    ) -> Dict[str, Any]:
        """Go/No-Go deployment decision framework.

        Criteria (aligned with reference project v3.8):
        - **Required**: CI lower bound > ``current_policy_value``
        - **Recommended**: Relative lift > ``min_improvement``
        - **Recommended**: ESS ratio > ``min_ess_ratio``

        Parameters
        ----------
        current_policy_value : float
            Current production policy value.
        new_policy_result : PolicyEvaluationResult
            Evaluation result for the candidate policy.
        min_improvement : float
            Minimum relative improvement (e.g. 0.02 = 2 %).
        min_ess_ratio : float
            Minimum ESS ratio (ESS / n_samples).

        Returns
        -------
        dict
            Decision report with ``"decision"`` key (``"Go"`` or ``"No-Go"``).
        """
        estimated = new_policy_result.estimated_value
        ci_lower = new_policy_result.ci_lower
        ess = new_policy_result.effective_sample_size
        n = new_policy_result.n_samples

        # Relative lift
        if abs(current_policy_value) > 1e-10:
            lift = (estimated - current_policy_value) / abs(current_policy_value)
        else:
            lift = estimated - current_policy_value

        ess_ratio = ess / n if n > 0 else 0.0

        # Criteria checks
        criteria = {}
        reasons = []

        # Required: CI lower > baseline
        criteria["ci_lower_above_baseline"] = ci_lower > current_policy_value
        if criteria["ci_lower_above_baseline"]:
            reasons.append(
                f"[Required PASS] CI lower ({ci_lower:.6f}) > "
                f"baseline ({current_policy_value:.6f})"
            )
        else:
            reasons.append(
                f"[Required FAIL] CI lower ({ci_lower:.6f}) <= "
                f"baseline ({current_policy_value:.6f})"
            )

        # Recommended: lift > min_improvement
        criteria["min_lift"] = lift > min_improvement
        if criteria["min_lift"]:
            reasons.append(f"[Recommended PASS] Lift ({lift:.2%}) > {min_improvement:.2%}")
        else:
            reasons.append(f"[Recommended MISS] Lift ({lift:.2%}) <= {min_improvement:.2%}")

        # Recommended: ESS ratio
        criteria["ess_ratio"] = ess_ratio > min_ess_ratio
        if criteria["ess_ratio"]:
            reasons.append(
                f"[Recommended PASS] ESS ratio ({ess_ratio:.2%}) > {min_ess_ratio:.2%}"
            )
        else:
            reasons.append(
                f"[Recommended MISS] ESS ratio ({ess_ratio:.2%}) <= {min_ess_ratio:.2%}"
            )

        # Decision: required criterion must pass
        decision = "Go" if criteria["ci_lower_above_baseline"] else "No-Go"

        result = {
            "decision": decision,
            "estimator": new_policy_result.estimator,
            "current_policy_value": current_policy_value,
            "new_policy_value": estimated,
            "lift": float(lift),
            "ci_lower": ci_lower,
            "ci_upper": new_policy_result.ci_upper,
            "ess_ratio": float(ess_ratio),
            "criteria": criteria,
            "reasons": reasons,
        }

        logger.info(
            "Go/No-Go decision: %s (lift=%.2f%%, estimator=%s)",
            decision, lift * 100, new_policy_result.estimator,
        )

        return result

    # ------------------------------------------------------------------
    # Convenience: evaluate all estimators
    # ------------------------------------------------------------------

    def evaluate_all(
        self,
        rewards: np.ndarray,
        logging_probs: np.ndarray,
        new_probs: np.ndarray,
        reward_model_preds: Optional[np.ndarray] = None,
        baseline: float = 0.0,
    ) -> Dict[str, PolicyEvaluationResult]:
        """Run IPS, SNIPS, and (optionally) DR estimators.

        Returns
        -------
        dict[str, PolicyEvaluationResult]
            Keyed by estimator name.
        """
        results: Dict[str, PolicyEvaluationResult] = {}
        results["ips"] = self.evaluate_ips(rewards, logging_probs, new_probs, baseline)
        results["snips"] = self.evaluate_snips(rewards, logging_probs, new_probs, baseline)

        if reward_model_preds is not None:
            results["dr"] = self.evaluate_dr(
                rewards, logging_probs, new_probs, reward_model_preds, baseline,
            )

        return results

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """Generate a Markdown report from accumulated results.

        Parameters
        ----------
        output_path : str | Path, optional
            File path to write. Falls back to ``output_dir/cpe_report.md``.

        Returns
        -------
        str
            The Markdown report text.
        """
        if not self.results:
            return "No counterfactual evaluation results available."

        lines = [
            "# Counterfactual Policy Evaluation Report",
            "",
            f"**Generated at**: {datetime.utcnow().isoformat()}Z",
            "",
            "---",
            "",
        ]

        for res in self.results:
            lines.extend([
                f"## {res.estimator.upper()} Estimator",
                "",
                f"- **Estimated Policy Value**: {res.estimated_value:.6f}",
                f"- **95% CI**: [{res.ci_lower:.6f}, {res.ci_upper:.6f}]",
                f"- **Standard Error**: {res.standard_error:.6f}",
                f"- **ESS**: {res.effective_sample_size:.1f}",
                f"- **Samples**: {res.n_samples:,}",
                f"- **Go Decision**: {res.go_decision}",
                "",
            ])

        # Comparison table
        if len(self.results) > 1:
            lines.extend([
                "---",
                "",
                "## Estimator Comparison",
                "",
                "| Estimator | Estimate | 95% CI | Std Error | ESS | Go |",
                "|-----------|----------|--------|-----------|-----|-----|",
            ])
            for res in self.results:
                lines.append(
                    f"| {res.estimator.upper()} "
                    f"| {res.estimated_value:.6f} "
                    f"| [{res.ci_lower:.6f}, {res.ci_upper:.6f}] "
                    f"| {res.standard_error:.6f} "
                    f"| {res.effective_sample_size:.1f} "
                    f"| {res.go_decision} |"
                )
            lines.append("")

        report = "\n".join(lines)

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(report, encoding="utf-8")
            logger.info("CPE report saved to %s", output_path)
        elif self.output_dir is not None:
            path = self.output_dir / "cpe_report.md"
            path.write_text(report, encoding="utf-8")
            logger.info("CPE report saved to %s", path)

        return report

    def to_json(self) -> str:
        """Serialise all results to JSON."""
        return json.dumps(
            [r.to_dict() for r in self.results],
            indent=2,
            ensure_ascii=False,
        )
