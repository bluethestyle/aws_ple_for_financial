"""
Automated model competition framework with statistical significance testing.

Orchestrates weekly (or on-demand) comparisons between a *champion* model
and one or more *challengers* using both standard metric evaluation and
offline policy evaluation.  Applies Go/No-Go gates with configurable
thresholds and statistical significance requirements.

Integrates with :class:`~core.training.evaluator.ModelEvaluator` for
standard metric comparison and :class:`OfflinePolicyEvaluator` for
counterfactual evaluation.

Usage::

    competition = ModelCompetition(
        config=CompetitionConfig(
            min_improvement=0.005,
            significance_level=0.05,
        ),
    )
    result = competition.run(
        champion_metrics=champ,
        challenger_metrics=chall,
    )
    if result["promote"]:
        deploy(challenger)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CompetitionConfig:
    """Configuration for automated model competitions.

    Attributes
    ----------
    min_improvement : float
        Minimum relative improvement required for promotion (e.g. 0.005
        means the challenger must be at least 0.5 % better).
    significance_level : float
        P-value threshold for statistical significance (default 0.05).
    n_bootstrap : int
        Number of bootstrap resamples for paired bootstrap test.
    ope_estimator : str
        Which OPE estimator to use for counterfactual comparison
        (``"snips"`` is recommended for stability).
    require_all_tasks : bool
        If True, the challenger must improve on *every* task.
        If False, aggregate improvement is sufficient.
    min_effective_sample_size : float
        Minimum ESS required for OPE results to be considered reliable.
    higher_is_better_metrics : set[str]
        Metric names where higher values indicate better performance.
    lower_is_better_metrics : set[str]
        Metric names where lower values indicate better performance.
    """

    min_improvement: float = 0.005
    significance_level: float = 0.05
    n_bootstrap: int = 2000
    ope_estimator: str = "snips"
    require_all_tasks: bool = True
    min_effective_sample_size: float = 50.0
    higher_is_better_metrics: frozenset = frozenset({
        "auc_roc", "auc_pr", "f1", "precision", "recall",
        "accuracy", "macro_f1", "weighted_f1", "ndcg", "map", "mrr",
        "r2", "snips", "ips", "dr",
    })
    lower_is_better_metrics: frozenset = frozenset({
        "mae", "rmse", "mape", "log_loss",
    })


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def paired_t_test(
    champion_scores: np.ndarray,
    challenger_scores: np.ndarray,
) -> Tuple[float, float]:
    """Two-sided paired t-test for matched samples.

    Parameters
    ----------
    champion_scores : np.ndarray
        Per-sample metric for the champion, shape ``(n,)``.
    challenger_scores : np.ndarray
        Per-sample metric for the challenger, shape ``(n,)``.

    Returns
    -------
    tuple[float, float]
        ``(t_statistic, p_value)``.
    """
    diffs = challenger_scores - champion_scores
    n = len(diffs)
    if n < 2:
        return (0.0, 1.0)

    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))
    if std_d < 1e-12:
        return (0.0, 1.0 if abs(mean_d) < 1e-12 else 0.0)

    t_stat = mean_d / (std_d / math.sqrt(n))

    # Two-sided p-value using normal approximation for large n,
    # otherwise use scipy if available
    try:
        from scipy import stats as sp_stats
        p_value = float(sp_stats.t.sf(abs(t_stat), df=n - 1) * 2)
    except ImportError:
        # Normal approximation fallback
        p_value = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))

    return (t_stat, p_value)


def paired_bootstrap_test(
    champion_scores: np.ndarray,
    challenger_scores: np.ndarray,
    n_bootstrap: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Paired bootstrap significance test.

    Tests the null hypothesis that the challenger is *not* better than
    the champion by checking whether the bootstrap distribution of the
    mean difference includes zero.

    Parameters
    ----------
    champion_scores : np.ndarray
        Per-sample scores for champion, shape ``(n,)``.
    challenger_scores : np.ndarray
        Per-sample scores for challenger, shape ``(n,)``.
    n_bootstrap : int
        Number of bootstrap iterations.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    tuple[float, float]
        ``(observed_diff, p_value)``.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    diffs = challenger_scores - champion_scores
    n = len(diffs)
    observed = float(np.mean(diffs))

    # Centre diffs under H0
    centred = diffs - observed

    count_extreme = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_mean = float(np.mean(centred[idx]))
        if abs(boot_mean) >= abs(observed):
            count_extreme += 1

    p_value = count_extreme / n_bootstrap
    return (observed, p_value)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# ModelCompetition
# ---------------------------------------------------------------------------

class ModelCompetition:
    """Automated champion-vs-challenger model competition.

    Combines standard metric comparison with optional offline policy
    evaluation and statistical significance testing to produce a
    deterministic Go / No-Go promotion decision.

    Parameters
    ----------
    config : CompetitionConfig, optional
        Competition settings.  Uses defaults if not provided.

    Examples
    --------
    >>> comp = ModelCompetition()
    >>> result = comp.run(
    ...     champion_metrics={"ctr": {"auc_roc": 0.810}},
    ...     challenger_metrics={"ctr": {"auc_roc": 0.818}},
    ...     primary_metrics={"ctr": "auc_roc"},
    ... )
    >>> print(result["promote"])
    True
    """

    def __init__(self, config: Optional[CompetitionConfig] = None) -> None:
        self.config = config or CompetitionConfig()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        champion_metrics: Dict[str, Dict[str, float]],
        challenger_metrics: Dict[str, Dict[str, float]],
        primary_metrics: Optional[Dict[str, str]] = None,
        champion_per_sample: Optional[Dict[str, np.ndarray]] = None,
        challenger_per_sample: Optional[Dict[str, np.ndarray]] = None,
        ope_results: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Execute a full model competition.

        Parameters
        ----------
        champion_metrics : dict[str, dict[str, float]]
            ``{task_name: {metric_name: value}}`` for the champion model.
        challenger_metrics : dict[str, dict[str, float]]
            Same structure for the challenger model.
        primary_metrics : dict[str, str], optional
            ``{task_name: metric_name}`` identifying the primary metric
            per task.  Defaults to the first metric in each task's dict.
        champion_per_sample : dict[str, np.ndarray], optional
            Per-sample metric arrays for paired significance tests.
        challenger_per_sample : dict[str, np.ndarray], optional
            Per-sample metric arrays for the challenger.
        ope_results : dict[str, dict[str, float]], optional
            Pre-computed OPE estimates ``{task_name: {"champion": v1,
            "challenger": v2}}``.

        Returns
        -------
        dict
            * ``promote`` -- bool, whether the challenger should be promoted.
            * ``tasks`` -- per-task comparison details.
            * ``significance_tests`` -- statistical test results.
            * ``ope_comparison`` -- OPE results (if provided).
            * ``summary`` -- human-readable summary.
            * ``reasons`` -- list of reasons for the decision.
        """
        tasks = sorted(set(champion_metrics.keys()) & set(challenger_metrics.keys()))
        if not tasks:
            logger.warning("No overlapping tasks between champion and challenger.")
            return {
                "promote": False,
                "tasks": {},
                "significance_tests": {},
                "ope_comparison": {},
                "summary": "No overlapping tasks.",
                "reasons": ["No overlapping tasks between champion and challenger."],
            }

        task_results: Dict[str, Dict[str, Any]] = {}
        sig_results: Dict[str, Dict[str, Any]] = {}
        ope_comparison: Dict[str, Dict[str, Any]] = {}
        reasons: List[str] = []
        all_pass = True

        for task in tasks:
            champ = champion_metrics[task]
            chall = challenger_metrics[task]

            # Determine primary metric
            if primary_metrics and task in primary_metrics:
                pm = primary_metrics[task]
            else:
                pm = next(iter(champ.keys()))

            champ_val = champ.get(pm, 0.0)
            chall_val = chall.get(pm, 0.0)
            higher_better = self._is_higher_better(pm)

            # Compute improvement
            if higher_better:
                improvement = (chall_val - champ_val) / max(abs(champ_val), 1e-10)
                passes_threshold = improvement >= self.config.min_improvement
            else:
                improvement = (champ_val - chall_val) / max(abs(champ_val), 1e-10)
                passes_threshold = improvement >= self.config.min_improvement

            task_results[task] = {
                "primary_metric": pm,
                "champion_value": champ_val,
                "challenger_value": chall_val,
                "improvement": improvement,
                "higher_is_better": higher_better,
                "passes_threshold": passes_threshold,
                "all_metrics": {
                    m: {"champion": champ.get(m, 0.0), "challenger": chall.get(m, 0.0)}
                    for m in set(champ.keys()) | set(chall.keys())
                },
            }

            # Statistical significance (if per-sample data provided)
            if (
                champion_per_sample is not None
                and challenger_per_sample is not None
                and task in champion_per_sample
                and task in challenger_per_sample
            ):
                champ_samples = np.asarray(champion_per_sample[task])
                chall_samples = np.asarray(challenger_per_sample[task])

                t_stat, t_pval = paired_t_test(champ_samples, chall_samples)
                boot_diff, boot_pval = paired_bootstrap_test(
                    champ_samples, chall_samples,
                    n_bootstrap=self.config.n_bootstrap,
                )

                is_significant = min(t_pval, boot_pval) < self.config.significance_level

                sig_results[task] = {
                    "t_statistic": t_stat,
                    "t_pvalue": t_pval,
                    "bootstrap_mean_diff": boot_diff,
                    "bootstrap_pvalue": boot_pval,
                    "is_significant": is_significant,
                }

                if not is_significant:
                    passes_threshold = False
                    task_results[task]["passes_threshold"] = False
                    reasons.append(
                        f"Task '{task}': improvement not statistically significant "
                        f"(t p={t_pval:.4f}, boot p={boot_pval:.4f})."
                    )

                task_results[task]["is_significant"] = is_significant

            # OPE comparison (if provided)
            if ope_results and task in ope_results:
                ope = ope_results[task]
                ope_champ = ope.get("champion", 0.0)
                ope_chall = ope.get("challenger", 0.0)
                ope_ess = ope.get("effective_sample_size", 0.0)

                ope_reliable = ope_ess >= self.config.min_effective_sample_size
                ope_better = ope_chall > ope_champ

                ope_comparison[task] = {
                    "champion_ope": ope_champ,
                    "challenger_ope": ope_chall,
                    "ess": ope_ess,
                    "reliable": ope_reliable,
                    "challenger_better": ope_better,
                }

                if ope_reliable and not ope_better:
                    reasons.append(
                        f"Task '{task}': challenger loses on OPE "
                        f"({ope_chall:.4f} vs {ope_champ:.4f})."
                    )
                    passes_threshold = False
                    task_results[task]["passes_threshold"] = False

            if not passes_threshold:
                all_pass = False
                if not any(task in r for r in reasons):
                    reasons.append(
                        f"Task '{task}': insufficient improvement on {pm} "
                        f"({improvement:.4f} < {self.config.min_improvement})."
                    )

        promote = all_pass if self.config.require_all_tasks else self._majority_vote(task_results)

        wins = sum(1 for t in task_results.values() if t["passes_threshold"])
        summary = (
            f"Competition result: {'PROMOTE' if promote else 'REJECT'} "
            f"({wins}/{len(tasks)} tasks passed, "
            f"min_improvement={self.config.min_improvement})."
        )

        logger.info(summary)

        return {
            "promote": promote,
            "tasks": task_results,
            "significance_tests": sig_results,
            "ope_comparison": ope_comparison,
            "summary": summary,
            "reasons": reasons,
        }

    # ------------------------------------------------------------------
    # Compare with existing evaluator reports
    # ------------------------------------------------------------------

    def run_from_evaluator_reports(
        self,
        champion_report: Dict[str, Any],
        challenger_report: Dict[str, Any],
        champion_per_sample: Optional[Dict[str, np.ndarray]] = None,
        challenger_per_sample: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Run competition using reports from :class:`ModelEvaluator`.

        Extracts ``{task: {metric: value}}`` from evaluator report format
        and delegates to :meth:`run`.

        Parameters
        ----------
        champion_report : dict
            Report from ``ModelEvaluator.evaluate()``.
        challenger_report : dict
            Report from ``ModelEvaluator.evaluate()``.
        champion_per_sample : dict, optional
            Per-sample arrays for significance testing.
        challenger_per_sample : dict, optional
            Per-sample arrays for the challenger.

        Returns
        -------
        dict
            Same as :meth:`run`.
        """
        champ_metrics: Dict[str, Dict[str, float]] = {}
        chall_metrics: Dict[str, Dict[str, float]] = {}
        primary_metrics: Dict[str, str] = {}

        for task_name, task_data in champion_report.get("tasks", {}).items():
            champ_metrics[task_name] = task_data.get("metrics", {})
            primary_metrics[task_name] = task_data.get("primary_metric", "")

        for task_name, task_data in challenger_report.get("tasks", {}).items():
            chall_metrics[task_name] = task_data.get("metrics", {})

        return self.run(
            champion_metrics=champ_metrics,
            challenger_metrics=chall_metrics,
            primary_metrics=primary_metrics,
            champion_per_sample=champion_per_sample,
            challenger_per_sample=challenger_per_sample,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_higher_better(self, metric_name: str) -> bool:
        """Determine if a metric is higher-is-better."""
        if metric_name in self.config.lower_is_better_metrics:
            return False
        return True

    @staticmethod
    def _majority_vote(task_results: Dict[str, Dict[str, Any]]) -> bool:
        """Promote if a majority of tasks pass."""
        passes = sum(1 for t in task_results.values() if t.get("passes_threshold", False))
        return passes > len(task_results) / 2


__all__ = [
    "ModelCompetition",
    "CompetitionConfig",
    "paired_t_test",
    "paired_bootstrap_test",
]
