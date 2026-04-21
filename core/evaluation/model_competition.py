"""
Champion-Challenger model evaluation and automatic promotion.

Compares a challenger model against the current champion using configurable
metrics and promotion criteria, with S3-based model registry and audit
logging.

Promotion rules:
  - Primary metric must improve by ``min_improvement`` (default 0.5%).
  - No secondary metric may degrade by more than ``max_degradation``
    (default 2%).
  - Optional statistical significance via paired bootstrap CI.

Integrates with S3-based model registry (JSON manifest) for tracking
which model is currently serving.

Usage::

    competition = ModelCompetition(config={
        "primary_metric": "avg_auc",
        "min_improvement": 0.005,
    })
    result = competition.evaluate(champion, challenger)
    if result.promotion_approved:
        competition.promote(result, registry_path="s3://bucket/registry.json")
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ModelCandidate",
    "CompetitionResult",
    "CompetitionConfig",
    "ModelCompetition",
    "paired_t_test",
    "paired_bootstrap_test",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModelCandidate:
    """A model participating in a champion-challenger competition.

    Attributes
    ----------
    model_id : str
        Unique identifier for this model (e.g. ``"ple-teacher-20260315"``).
    model_uri : str
        S3 path to model artifacts (e.g. ``"s3://bucket/models/v2.1.0/"``).
    model_type : str
        Model family: ``"ple_teacher"`` or ``"lgbm_student"``.
    version : str
        Semantic version string (e.g. ``"v2.1.0"``).
    trained_at : str
        ISO 8601 timestamp of when training completed.
    metrics : dict[str, float]
        Pre-computed evaluation metrics
        (e.g. ``{"avg_auc": 0.82, "avg_f1": 0.71}``).
    metadata : dict[str, Any]
        Arbitrary metadata (training config hash, dataset id, etc.).
    """

    model_id: str
    model_uri: str
    model_type: str  # "ple_teacher" | "lgbm_student"
    version: str
    trained_at: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompetitionResult:
    """Outcome of a champion-challenger evaluation.

    Attributes
    ----------
    champion : ModelCandidate
        The incumbent model.
    challenger : ModelCandidate
        The candidate model being evaluated.
    winner : str
        ``"champion"`` or ``"challenger"``.
    promotion_approved : bool
        True if all promotion criteria are satisfied.
    comparison : dict[str, dict[str, float]]
        Per-metric comparison detail, e.g.
        ``{"avg_auc": {"champion": 0.80, "challenger": 0.82, "delta": 0.02}}``.
    decision_reason : str
        Human-readable explanation of the promotion decision.
    evaluated_at : str
        ISO 8601 timestamp of when the evaluation was performed.
    significance : dict[str, Any]
        Statistical significance details (empty if not computed).
    """

    champion: ModelCandidate
    challenger: ModelCandidate
    winner: str  # "champion" | "challenger"
    promotion_approved: bool
    comparison: Dict[str, Dict[str, float]]
    decision_reason: str
    evaluated_at: str
    significance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompetitionConfig:
    """Configuration for model competitions.

    Attributes
    ----------
    primary_metric : str
        The metric that must improve for promotion.
    min_improvement : float
        Minimum relative improvement on the primary metric (0.005 = 0.5%).
    max_degradation : float
        Maximum allowable relative degradation on any secondary metric
        (0.02 = 2%).
    secondary_metrics : list[str]
        Metric names to check for degradation.
    significance_level : float
        P-value threshold for statistical significance (default 0.05).
    n_bootstrap : int
        Number of bootstrap resamples for paired bootstrap test.
    higher_is_better_metrics : frozenset[str]
        Metrics where higher values indicate better performance.
    lower_is_better_metrics : frozenset[str]
        Metrics where lower values indicate better performance.
    """

    primary_metric: str = "avg_auc"
    min_improvement: float = 0.005  # 0.5%
    max_degradation: float = 0.02  # 2%
    secondary_metrics: List[str] = field(
        default_factory=lambda: ["avg_f1", "avg_mae"],
    )
    significance_level: float = 0.05
    n_bootstrap: int = 2000
    higher_is_better_metrics: frozenset = frozenset({
        "avg_auc", "auc_roc", "auc_pr", "f1", "avg_f1",
        "precision", "recall", "accuracy", "macro_f1",
        "weighted_f1", "ndcg", "map", "mrr", "r2",
    })
    lower_is_better_metrics: frozenset = frozenset({
        "avg_mae", "mae", "rmse", "mape", "log_loss",
    })
    # Sprint 2 S15: human approval enforcement (EU AI Act Art. 14, 인적 감독).
    # When False, a challenger whose metrics meet the threshold is still
    # marked as NOT auto-promotable: the reason column records
    # "metrics_gate_passed_requires_manual_approval" so the operator must
    # re-run with --force-promote.
    #
    # Default stays True for backward compatibility with legacy tests. The
    # production posture is False, enforced via
    # ``serving.competition.auto_promote`` in pipeline.yaml (see
    # submit_pipeline.py which builds the config through
    # :meth:`from_dict`). Do not flip this default in code.
    auto_promote: bool = True

    @classmethod
    def from_dict(
        cls, data: Optional[Dict[str, Any]] = None,
    ) -> "CompetitionConfig":
        """Build a CompetitionConfig from ``serving.competition`` block."""
        if not data:
            return cls()
        kwargs: Dict[str, Any] = {}
        for field_name in (
            "primary_metric", "min_improvement", "max_degradation",
            "significance_level", "n_bootstrap", "auto_promote",
        ):
            if field_name in data:
                kwargs[field_name] = data[field_name]
        if "secondary_metrics" in data:
            kwargs["secondary_metrics"] = list(data["secondary_metrics"])
        return cls(**kwargs)


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

    # Two-sided p-value -- prefer scipy, fall back to normal approximation
    try:
        from scipy import stats as sp_stats
        p_value = float(sp_stats.t.sf(abs(t_stat), df=n - 1) * 2)
    except ImportError:
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
    """Champion-Challenger model evaluation framework.

    Compares a challenger model against the current champion using
    configurable metrics and promotion criteria.

    Promotion rules:
      - Primary metric must improve by ``min_improvement`` (default 0.5%).
      - No secondary metric may degrade by more than ``max_degradation``
        (default 2%).
      - Statistical significance test (optional, via bootstrap CI).

    Parameters
    ----------
    config : dict or CompetitionConfig, optional
        Competition settings.  Accepts a plain dict (keys mapped to
        :class:`CompetitionConfig` fields) or a ``CompetitionConfig``
        instance.  Uses defaults if not provided.
    audit_store : object, optional
        Any object with a ``.log(record: dict)`` method.  When provided,
        every evaluation and promotion is recorded.

    Examples
    --------
    >>> competition = ModelCompetition({"primary_metric": "avg_auc"})
    >>> result = competition.evaluate(champion, challenger)
    >>> if result.promotion_approved:
    ...     competition.promote(result)
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        audit_store: Optional[Any] = None,
    ) -> None:
        if config is None:
            self._config = CompetitionConfig()
        elif isinstance(config, CompetitionConfig):
            self._config = config
        elif isinstance(config, dict):
            # Filter to known fields only
            known = {
                k: v for k, v in config.items()
                if k in CompetitionConfig.__dataclass_fields__
            }
            self._config = CompetitionConfig(**known)
        else:
            raise TypeError(
                f"config must be dict or CompetitionConfig, got {type(config).__name__}"
            )

        self._audit_store = audit_store
        self._history: List[CompetitionResult] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> CompetitionConfig:
        """Return the active competition configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        champion: ModelCandidate,
        challenger: ModelCandidate,
        eval_features: Optional[Any] = None,
        eval_labels: Optional[Any] = None,
        champion_per_sample: Optional[np.ndarray] = None,
        challenger_per_sample: Optional[np.ndarray] = None,
    ) -> CompetitionResult:
        """Compare champion vs challenger on evaluation data.

        If ``eval_features`` / ``eval_labels`` are provided, subclasses can
        override :meth:`_run_inference` to compute metrics on-the-fly.
        Otherwise the pre-computed ``ModelCandidate.metrics`` are used.

        Parameters
        ----------
        champion : ModelCandidate
            The incumbent champion model.
        challenger : ModelCandidate
            The challenger model being evaluated.
        eval_features : Any, optional
            Evaluation feature data (numpy array, DataFrame, etc.).
            Not used when relying on pre-computed metrics.
        eval_labels : Any, optional
            Evaluation labels corresponding to ``eval_features``.
        champion_per_sample : np.ndarray, optional
            Per-sample metric array for champion (enables significance test).
        challenger_per_sample : np.ndarray, optional
            Per-sample metric array for challenger.

        Returns
        -------
        CompetitionResult
            Full comparison result including winner and promotion decision.
        """
        champ_metrics = dict(champion.metrics)
        chall_metrics = dict(challenger.metrics)

        # Core comparison
        approved, reason, comparison = self._compare_metrics(
            champ_metrics, chall_metrics,
        )

        # Optional statistical significance test
        significance: Dict[str, Any] = {}
        if champion_per_sample is not None and challenger_per_sample is not None:
            champ_arr = np.asarray(champion_per_sample)
            chall_arr = np.asarray(challenger_per_sample)

            t_stat, t_pval = paired_t_test(champ_arr, chall_arr)
            boot_diff, boot_pval = paired_bootstrap_test(
                champ_arr, chall_arr,
                n_bootstrap=self._config.n_bootstrap,
            )
            is_significant = min(t_pval, boot_pval) < self._config.significance_level

            significance = {
                "t_statistic": round(t_stat, 6),
                "t_pvalue": round(t_pval, 6),
                "bootstrap_mean_diff": round(boot_diff, 6),
                "bootstrap_pvalue": round(boot_pval, 6),
                "is_significant": is_significant,
            }

            if approved and not is_significant:
                approved = False
                reason = (
                    f"Challenger improves on primary metric but result is not "
                    f"statistically significant (t p={t_pval:.4f}, "
                    f"bootstrap p={boot_pval:.4f})."
                )

        # Sprint 2 S15: when auto_promote is disabled (the default under
        # EU AI Act Art. 14 human-oversight posture), a passing challenger is
        # NOT auto-approved. The operator must re-run with --force-promote.
        if approved and not self._config.auto_promote:
            approved = False
            reason = (
                f"{reason} | metrics_gate_passed but auto_promote=False "
                f"(EU AI Act Art. 14 human oversight); re-run with "
                f"--force-promote to promote manually."
            )

        winner = "challenger" if approved else "champion"
        now = datetime.now(timezone.utc).isoformat()

        result = CompetitionResult(
            champion=champion,
            challenger=challenger,
            winner=winner,
            promotion_approved=approved,
            comparison=comparison,
            decision_reason=reason,
            evaluated_at=now,
            significance=significance,
        )

        # Store in history
        self._history.append(result)

        # Audit log
        self._audit(
            event="competition_evaluated",
            champion_id=champion.model_id,
            challenger_id=challenger.model_id,
            winner=winner,
            promotion_approved=approved,
            reason=reason,
            comparison=comparison,
            significance=significance,
            evaluated_at=now,
        )

        logger.info(
            "ModelCompetition: %s vs %s -> winner=%s, promote=%s | %s",
            champion.model_id,
            challenger.model_id,
            winner,
            approved,
            reason,
        )

        return result

    # ------------------------------------------------------------------
    # Promote
    # ------------------------------------------------------------------

    def promote(
        self,
        result: CompetitionResult,
        registry_path: str = "",
    ) -> None:
        """Promote the winner to champion status.

        Updates an S3-based model registry (JSON manifest) if
        ``registry_path`` is provided.  The manifest records the current
        champion's model_id, model_uri, version, and promotion timestamp.

        Parameters
        ----------
        result : CompetitionResult
            A competition result (should have ``promotion_approved=True``).
        registry_path : str, optional
            S3 URI to the registry manifest JSON file
            (e.g. ``"s3://bucket/model-registry/champion.json"``).
            If empty, only logging and audit are performed.

        Raises
        ------
        ValueError
            If ``result.promotion_approved`` is False.
        """
        if not result.promotion_approved:
            raise ValueError(
                f"Cannot promote: promotion not approved. "
                f"Reason: {result.decision_reason}"
            )

        winner_candidate = (
            result.challenger if result.winner == "challenger" else result.champion
        )
        now = datetime.now(timezone.utc).isoformat()

        manifest = {
            "champion_model_id": winner_candidate.model_id,
            "champion_model_uri": winner_candidate.model_uri,
            "champion_model_type": winner_candidate.model_type,
            "champion_version": winner_candidate.version,
            "promoted_at": now,
            "previous_champion_id": result.champion.model_id,
            "competition_evaluated_at": result.evaluated_at,
            "decision_reason": result.decision_reason,
        }

        if registry_path:
            self._write_registry(registry_path, manifest)

        self._audit(
            event="model_promoted",
            promoted_model_id=winner_candidate.model_id,
            promoted_model_uri=winner_candidate.model_uri,
            previous_champion_id=result.champion.model_id,
            registry_path=registry_path,
            promoted_at=now,
        )

        logger.info(
            "ModelCompetition: promoted %s (version %s) to champion at %s",
            winner_candidate.model_id,
            winner_candidate.version,
            registry_path or "(no registry path)",
        )

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_history(self, limit: int = 10) -> List[CompetitionResult]:
        """Return recent competition history (most recent first).

        Parameters
        ----------
        limit : int
            Maximum number of results to return.

        Returns
        -------
        list[CompetitionResult]
            Up to ``limit`` most recent competition results.
        """
        return list(reversed(self._history[-limit:]))

    # ------------------------------------------------------------------
    # Legacy run() interface (backward compatibility)
    # ------------------------------------------------------------------

    def run(
        self,
        champion_metrics: Dict[str, Dict[str, float]],
        challenger_metrics: Dict[str, Dict[str, float]],
        primary_metrics: Optional[Dict[str, str]] = None,
        champion_per_sample: Optional[Dict[str, np.ndarray]] = None,
        challenger_per_sample: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Execute a task-level model competition (legacy interface).

        Compares champion and challenger across multiple tasks, each with
        its own set of metrics.  Useful when integrating with
        :class:`~core.training.evaluator.ModelEvaluator`.

        Parameters
        ----------
        champion_metrics : dict[str, dict[str, float]]
            ``{task_name: {metric_name: value}}`` for the champion.
        challenger_metrics : dict[str, dict[str, float]]
            Same structure for the challenger.
        primary_metrics : dict[str, str], optional
            ``{task_name: metric_name}`` identifying primary metric per task.
        champion_per_sample : dict[str, np.ndarray], optional
            Per-sample arrays for paired significance tests.
        challenger_per_sample : dict[str, np.ndarray], optional
            Per-sample arrays for the challenger.

        Returns
        -------
        dict
            ``promote`` (bool), ``tasks``, ``significance_tests``,
            ``summary``, ``reasons``.
        """
        tasks = sorted(
            set(champion_metrics.keys()) & set(challenger_metrics.keys())
        )
        if not tasks:
            logger.warning("No overlapping tasks between champion and challenger.")
            return {
                "promote": False,
                "tasks": {},
                "significance_tests": {},
                "summary": "No overlapping tasks.",
                "reasons": ["No overlapping tasks between champion and challenger."],
            }

        task_results: Dict[str, Dict[str, Any]] = {}
        sig_results: Dict[str, Dict[str, Any]] = {}
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

            if higher_better:
                improvement = (chall_val - champ_val) / max(abs(champ_val), 1e-10)
            else:
                improvement = (champ_val - chall_val) / max(abs(champ_val), 1e-10)

            passes_threshold = improvement >= self._config.min_improvement

            task_results[task] = {
                "primary_metric": pm,
                "champion_value": champ_val,
                "challenger_value": chall_val,
                "improvement": improvement,
                "higher_is_better": higher_better,
                "passes_threshold": passes_threshold,
                "all_metrics": {
                    m: {
                        "champion": champ.get(m, 0.0),
                        "challenger": chall.get(m, 0.0),
                    }
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
                champ_arr = np.asarray(champion_per_sample[task])
                chall_arr = np.asarray(challenger_per_sample[task])

                t_stat, t_pval = paired_t_test(champ_arr, chall_arr)
                boot_diff, boot_pval = paired_bootstrap_test(
                    champ_arr, chall_arr,
                    n_bootstrap=self._config.n_bootstrap,
                )
                is_significant = (
                    min(t_pval, boot_pval) < self._config.significance_level
                )

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
                        f"Task '{task}': improvement not statistically "
                        f"significant (t p={t_pval:.4f}, "
                        f"boot p={boot_pval:.4f})."
                    )

                task_results[task]["is_significant"] = is_significant

            if not passes_threshold:
                all_pass = False
                if not any(task in r for r in reasons):
                    reasons.append(
                        f"Task '{task}': insufficient improvement on {pm} "
                        f"({improvement:.4f} < {self._config.min_improvement})."
                    )

        promote = all_pass
        wins = sum(1 for t in task_results.values() if t["passes_threshold"])
        summary = (
            f"Competition result: {'PROMOTE' if promote else 'REJECT'} "
            f"({wins}/{len(tasks)} tasks passed, "
            f"min_improvement={self._config.min_improvement})."
        )

        logger.info(summary)

        return {
            "promote": promote,
            "tasks": task_results,
            "significance_tests": sig_results,
            "summary": summary,
            "reasons": reasons,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compare_metrics(
        self,
        champion_metrics: Dict[str, float],
        challenger_metrics: Dict[str, float],
    ) -> Tuple[bool, str, Dict[str, Dict[str, float]]]:
        """Core comparison logic.

        Checks that:
          1. The primary metric improves by at least ``min_improvement``.
          2. No secondary metric degrades by more than ``max_degradation``.

        Returns
        -------
        tuple[bool, str, dict]
            ``(approved, reason, comparison_detail)``.
        """
        primary = self._config.primary_metric
        comparison: Dict[str, Dict[str, float]] = {}

        # -- Primary metric check ------------------------------------------
        champ_primary = champion_metrics.get(primary, 0.0)
        chall_primary = challenger_metrics.get(primary, 0.0)
        higher_better = self._is_higher_better(primary)

        if higher_better:
            delta = chall_primary - champ_primary
            relative = delta / max(abs(champ_primary), 1e-10)
        else:
            delta = champ_primary - chall_primary  # positive means challenger is better
            relative = delta / max(abs(champ_primary), 1e-10)

        comparison[primary] = {
            "champion": round(champ_primary, 6),
            "challenger": round(chall_primary, 6),
            "delta": round(delta, 6),
            "relative_change": round(relative, 6),
        }

        if relative < self._config.min_improvement:
            reason = (
                f"Primary metric '{primary}' did not improve sufficiently: "
                f"relative change {relative:.4f} < "
                f"min_improvement {self._config.min_improvement}."
            )
            return (False, reason, comparison)

        # -- Secondary metric checks ---------------------------------------
        for metric in self._config.secondary_metrics:
            champ_val = champion_metrics.get(metric)
            chall_val = challenger_metrics.get(metric)

            if champ_val is None or chall_val is None:
                continue

            sec_higher_better = self._is_higher_better(metric)

            if sec_higher_better:
                sec_delta = chall_val - champ_val
                sec_relative = sec_delta / max(abs(champ_val), 1e-10)
                degraded = sec_relative < -self._config.max_degradation
            else:
                sec_delta = champ_val - chall_val
                sec_relative = sec_delta / max(abs(champ_val), 1e-10)
                degraded = sec_relative < -self._config.max_degradation

            comparison[metric] = {
                "champion": round(champ_val, 6),
                "challenger": round(chall_val, 6),
                "delta": round(sec_delta, 6),
                "relative_change": round(sec_relative, 6),
            }

            if degraded:
                reason = (
                    f"Secondary metric '{metric}' degraded beyond tolerance: "
                    f"relative change {sec_relative:.4f} exceeds "
                    f"max_degradation -{self._config.max_degradation}."
                )
                return (False, reason, comparison)

        # -- All checks passed ---------------------------------------------
        reason = (
            f"Challenger improves primary metric '{primary}' by "
            f"{relative:.4f} (>= {self._config.min_improvement}) with no "
            f"secondary metric degradation beyond {self._config.max_degradation}."
        )
        return (True, reason, comparison)

    def _is_higher_better(self, metric_name: str) -> bool:
        """Determine if higher values are better for a given metric."""
        if metric_name in self._config.lower_is_better_metrics:
            return False
        return True

    def _write_registry(self, registry_path: str, manifest: Dict[str, Any]) -> None:
        """Write the champion manifest to S3 or local filesystem.

        Parameters
        ----------
        registry_path : str
            S3 URI (``s3://...``) or local file path.
        manifest : dict
            Champion manifest to write.
        """
        payload = json.dumps(manifest, indent=2, default=str)

        if registry_path.startswith("s3://"):
            try:
                import boto3

                parts = registry_path.replace("s3://", "").split("/", 1)
                bucket = parts[0]
                key = parts[1] if len(parts) > 1 else "champion.json"

                s3 = boto3.client("s3")
                s3.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=payload.encode("utf-8"),
                    ContentType="application/json",
                )
                logger.info(
                    "ModelCompetition: wrote registry manifest to s3://%s/%s",
                    bucket, key,
                )
            except Exception:
                logger.error(
                    "ModelCompetition: failed to write registry to %s",
                    registry_path,
                    exc_info=True,
                )
                raise
        else:
            # Local file fallback
            from pathlib import Path

            path = Path(registry_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")
            logger.info(
                "ModelCompetition: wrote registry manifest to %s", registry_path,
            )

    def _audit(self, **kwargs: Any) -> None:
        """Log an event to the audit store if configured."""
        if self._audit_store is None:
            return

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": "ModelCompetition",
            **kwargs,
        }

        try:
            self._audit_store.log(record)
        except Exception:
            logger.debug(
                "ModelCompetition: failed to write audit record",
                exc_info=True,
            )
