"""
Explainable AI (XAI) quality evaluator.

Assesses the quality of model explanations across three key dimensions:

- **Faithfulness**       How accurately explanations reflect model behaviour
- **Stability**          Consistency of explanations for similar inputs
- **Comprehensibility**  Understandability of explanations for stakeholders

Also provides:
- SHAP/LIME consistency checks
- Explanation coverage metrics per task
- Quality score aggregation

Quality scores are in [0.0, 1.0] (higher = better quality).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------

DEFAULT_QUALITY_THRESHOLDS: Dict[str, float] = {
    "faithfulness_min": 0.7,
    "stability_min": 0.7,
    "comprehensibility_min": 0.6,
    "coverage_min": 0.8,
    "consistency_min": 0.7,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExplanationQualityMetrics:
    """Quality metrics for explanations on a single task."""

    task_name: str
    faithfulness_score: float
    stability_score: float
    comprehensibility_score: float
    coverage: float
    overall_quality: float
    meets_threshold: bool
    violations: List[str]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsistencyResult:
    """Result of SHAP/LIME consistency check."""

    task_name: str
    method_a: str  # e.g. "shap"
    method_b: str  # e.g. "lime"
    rank_correlation: float
    top_k_overlap: float
    is_consistent: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class XAIQualityReport:
    """Container for a full XAI quality evaluation report."""

    report_id: str
    evaluated_at: str  # ISO datetime
    system_name: str
    task_metrics: List[ExplanationQualityMetrics]
    consistency_results: List[ConsistencyResult]
    aggregate_quality: float
    tasks_passing: int
    tasks_failing: int
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# XAIQualityEvaluator
# ---------------------------------------------------------------------------

class XAIQualityEvaluator:
    """Evaluate the quality of ML model explanations.

    Assesses explanation faithfulness, stability, and comprehensibility
    across tasks and provides SHAP/LIME consistency checks.

    Parameters
    ----------
    system_name : str
        Name of the ML system.
    thresholds : dict, optional
        Override default quality thresholds.
    """

    def __init__(
        self,
        system_name: str = "PLE-Cluster-adaTT",
        thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.system_name = system_name
        self.thresholds: Dict[str, float] = {
            **DEFAULT_QUALITY_THRESHOLDS,
            **(thresholds or {}),
        }

    # ------------------------------------------------------------------
    # Faithfulness evaluation
    # ------------------------------------------------------------------

    def evaluate_faithfulness(
        self,
        original_predictions: np.ndarray,
        perturbed_predictions: np.ndarray,
        feature_attributions: np.ndarray,
        perturbation_mask: np.ndarray,
    ) -> float:
        """Evaluate explanation faithfulness via perturbation fidelity.

        Measures whether removing important features (as identified by
        attributions) causes proportional prediction changes.

        Parameters
        ----------
        original_predictions : np.ndarray
            Shape ``(n_samples,)``.  Original model predictions.
        perturbed_predictions : np.ndarray
            Shape ``(n_samples,)``.  Predictions after perturbing top features.
        feature_attributions : np.ndarray
            Shape ``(n_samples, n_features)``.  Feature importance scores.
        perturbation_mask : np.ndarray
            Shape ``(n_samples, n_features)``.  Boolean mask of perturbed features.

        Returns
        -------
        float
            Faithfulness score in [0.0, 1.0].
        """
        if len(original_predictions) == 0:
            return 0.0

        try:
            # Prediction change magnitude
            pred_change = np.abs(original_predictions - perturbed_predictions)

            # Expected change from attributions (sum of perturbed feature importances)
            masked_attributions = np.abs(feature_attributions) * perturbation_mask
            expected_change = masked_attributions.sum(axis=1)

            # Normalize
            pred_change_norm = pred_change / (np.max(pred_change) + 1e-10)
            expected_change_norm = expected_change / (np.max(expected_change) + 1e-10)

            # Correlation between expected and actual change
            if np.std(pred_change_norm) < 1e-10 or np.std(expected_change_norm) < 1e-10:
                return 0.5

            correlation = np.corrcoef(pred_change_norm, expected_change_norm)[0, 1]
            # Map correlation from [-1, 1] to [0, 1]
            score = float(max(0.0, (correlation + 1.0) / 2.0))
            return round(score, 4)
        except Exception as exc:
            logger.warning("Faithfulness evaluation failed: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Stability evaluation
    # ------------------------------------------------------------------

    def evaluate_stability(
        self,
        attributions_run1: np.ndarray,
        attributions_run2: np.ndarray,
    ) -> float:
        """Evaluate explanation stability across repeated evaluations.

        Measures consistency of feature attributions when the same inputs
        are explained multiple times (or with slight perturbations).

        Parameters
        ----------
        attributions_run1 : np.ndarray
            Shape ``(n_samples, n_features)``.  First explanation run.
        attributions_run2 : np.ndarray
            Shape ``(n_samples, n_features)``.  Second explanation run.

        Returns
        -------
        float
            Stability score in [0.0, 1.0].
        """
        if attributions_run1.shape != attributions_run2.shape:
            logger.warning("Attribution shape mismatch: %s vs %s",
                           attributions_run1.shape, attributions_run2.shape)
            return 0.0

        if attributions_run1.size == 0:
            return 0.0

        try:
            # Per-sample cosine similarity
            similarities = []
            for a1, a2 in zip(attributions_run1, attributions_run2):
                norm1 = np.linalg.norm(a1)
                norm2 = np.linalg.norm(a2)
                if norm1 < 1e-10 or norm2 < 1e-10:
                    similarities.append(1.0 if norm1 < 1e-10 and norm2 < 1e-10 else 0.0)
                else:
                    cos_sim = float(np.dot(a1, a2) / (norm1 * norm2))
                    similarities.append(max(0.0, cos_sim))

            return round(float(np.mean(similarities)), 4)
        except Exception as exc:
            logger.warning("Stability evaluation failed: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Comprehensibility evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_comprehensibility(
        attributions: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_features_shown: int = 10,
    ) -> float:
        """Evaluate explanation comprehensibility.

        Measures how concentrated and interpretable the explanations are:
        - Sparsity: fewer important features = more comprehensible
        - Dominance: clear feature ranking = more comprehensible

        Parameters
        ----------
        attributions : np.ndarray
            Shape ``(n_samples, n_features)``.
        feature_names : list of str, optional
            Human-readable feature names (affects scoring if present).
        max_features_shown : int
            Maximum features to display in explanations.

        Returns
        -------
        float
            Comprehensibility score in [0.0, 1.0].
        """
        if attributions.size == 0:
            return 0.0

        try:
            abs_attr = np.abs(attributions)
            n_features = abs_attr.shape[1] if abs_attr.ndim > 1 else abs_attr.shape[0]

            if abs_attr.ndim == 1:
                abs_attr = abs_attr.reshape(1, -1)

            # Sparsity: what fraction of total importance is in top-k features
            sparsity_scores = []
            for row in abs_attr:
                total = row.sum()
                if total < 1e-10:
                    sparsity_scores.append(1.0)
                    continue
                sorted_vals = np.sort(row)[::-1]
                top_k = sorted_vals[:max_features_shown].sum()
                sparsity_scores.append(float(top_k / total))

            sparsity = float(np.mean(sparsity_scores))

            # Dominance: ratio of top feature to second feature
            dominance_scores = []
            for row in abs_attr:
                sorted_vals = np.sort(row)[::-1]
                if len(sorted_vals) >= 2 and sorted_vals[1] > 1e-10:
                    dominance_scores.append(
                        min(float(sorted_vals[0] / sorted_vals[1]) / 5.0, 1.0)
                    )
                else:
                    dominance_scores.append(1.0)
            dominance = float(np.mean(dominance_scores))

            # Feature naming bonus
            naming_bonus = 0.05 if feature_names and len(feature_names) >= n_features else 0.0

            score = 0.5 * sparsity + 0.4 * dominance + naming_bonus + 0.05
            return round(min(score, 1.0), 4)
        except Exception as exc:
            logger.warning("Comprehensibility evaluation failed: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # SHAP/LIME consistency
    # ------------------------------------------------------------------

    def check_explanation_consistency(
        self,
        task_name: str,
        attributions_a: np.ndarray,
        attributions_b: np.ndarray,
        method_a: str = "shap",
        method_b: str = "lime",
        top_k: int = 5,
    ) -> ConsistencyResult:
        """Check consistency between two explanation methods.

        Parameters
        ----------
        task_name : str
            Task identifier.
        attributions_a : np.ndarray
            Shape ``(n_samples, n_features)``.  Attributions from method A.
        attributions_b : np.ndarray
            Shape ``(n_samples, n_features)``.  Attributions from method B.
        method_a : str
            Name of explanation method A.
        method_b : str
            Name of explanation method B.
        top_k : int
            Number of top features to compare for overlap.

        Returns
        -------
        ConsistencyResult
        """
        if attributions_a.shape != attributions_b.shape:
            logger.warning("Consistency check: shape mismatch for task %s", task_name)
            return ConsistencyResult(
                task_name=task_name,
                method_a=method_a,
                method_b=method_b,
                rank_correlation=0.0,
                top_k_overlap=0.0,
                is_consistent=False,
                details={"error": "shape_mismatch"},
            )

        if attributions_a.size == 0:
            return ConsistencyResult(
                task_name=task_name,
                method_a=method_a,
                method_b=method_b,
                rank_correlation=0.0,
                top_k_overlap=0.0,
                is_consistent=False,
                details={"error": "empty_attributions"},
            )

        try:
            # Per-sample rank correlation (Spearman)
            rank_corrs = []
            top_k_overlaps = []

            for a, b in zip(attributions_a, attributions_b):
                # Rank correlation
                rank_a = np.argsort(np.argsort(-np.abs(a))).astype(float)
                rank_b = np.argsort(np.argsort(-np.abs(b))).astype(float)
                n = len(rank_a)
                if n > 1:
                    d_sq = np.sum((rank_a - rank_b) ** 2)
                    rho = 1.0 - (6.0 * d_sq) / (n * (n ** 2 - 1))
                    rank_corrs.append(float(rho))
                else:
                    rank_corrs.append(1.0)

                # Top-k overlap
                top_a = set(np.argsort(-np.abs(a))[:top_k])
                top_b = set(np.argsort(-np.abs(b))[:top_k])
                overlap = len(top_a & top_b) / top_k
                top_k_overlaps.append(overlap)

            avg_rank_corr = round(float(np.mean(rank_corrs)), 4)
            avg_overlap = round(float(np.mean(top_k_overlaps)), 4)
            threshold = self.thresholds.get("consistency_min", 0.7)
            is_consistent = avg_rank_corr >= threshold and avg_overlap >= threshold

            return ConsistencyResult(
                task_name=task_name,
                method_a=method_a,
                method_b=method_b,
                rank_correlation=avg_rank_corr,
                top_k_overlap=avg_overlap,
                is_consistent=is_consistent,
                details={
                    "n_samples": len(attributions_a),
                    "top_k": top_k,
                    "threshold": threshold,
                },
            )
        except Exception as exc:
            logger.warning("Consistency check failed for task %s: %s", task_name, exc)
            return ConsistencyResult(
                task_name=task_name,
                method_a=method_a,
                method_b=method_b,
                rank_correlation=0.0,
                top_k_overlap=0.0,
                is_consistent=False,
                details={"error": str(exc)},
            )

    # ------------------------------------------------------------------
    # Per-task evaluation
    # ------------------------------------------------------------------

    def evaluate_task(
        self,
        task_name: str,
        faithfulness_score: Optional[float] = None,
        stability_score: Optional[float] = None,
        comprehensibility_score: Optional[float] = None,
        coverage: Optional[float] = None,
    ) -> ExplanationQualityMetrics:
        """Evaluate explanation quality for a single task.

        Accepts pre-computed scores.  Use the individual ``evaluate_*``
        methods to compute scores from raw data.

        Parameters
        ----------
        task_name : str
            Task identifier.
        faithfulness_score : float, optional
            Pre-computed faithfulness score.
        stability_score : float, optional
            Pre-computed stability score.
        comprehensibility_score : float, optional
            Pre-computed comprehensibility score.
        coverage : float, optional
            Fraction of predictions with explanations available.

        Returns
        -------
        ExplanationQualityMetrics
        """
        faith = faithfulness_score if faithfulness_score is not None else 0.0
        stab = stability_score if stability_score is not None else 0.0
        comp = comprehensibility_score if comprehensibility_score is not None else 0.0
        cov = coverage if coverage is not None else 0.0

        # Weighted overall quality
        overall = round(0.35 * faith + 0.30 * stab + 0.20 * comp + 0.15 * cov, 4)

        violations: List[str] = []
        if faith < self.thresholds["faithfulness_min"]:
            violations.append(
                f"Faithfulness {faith:.4f} below threshold {self.thresholds['faithfulness_min']}"
            )
        if stab < self.thresholds["stability_min"]:
            violations.append(
                f"Stability {stab:.4f} below threshold {self.thresholds['stability_min']}"
            )
        if comp < self.thresholds["comprehensibility_min"]:
            violations.append(
                f"Comprehensibility {comp:.4f} below threshold {self.thresholds['comprehensibility_min']}"
            )
        if cov < self.thresholds["coverage_min"]:
            violations.append(
                f"Coverage {cov:.4f} below threshold {self.thresholds['coverage_min']}"
            )

        metrics = ExplanationQualityMetrics(
            task_name=task_name,
            faithfulness_score=round(faith, 4),
            stability_score=round(stab, 4),
            comprehensibility_score=round(comp, 4),
            coverage=round(cov, 4),
            overall_quality=overall,
            meets_threshold=len(violations) == 0,
            violations=violations,
        )

        if violations:
            logger.warning(
                "XAI quality violations for task %s: %s",
                task_name, "; ".join(violations),
            )
        else:
            logger.info(
                "XAI quality PASSED for task %s (overall=%.4f)", task_name, overall,
            )

        return metrics

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        task_metrics: Optional[List[ExplanationQualityMetrics]] = None,
        consistency_results: Optional[List[ConsistencyResult]] = None,
    ) -> XAIQualityReport:
        """Generate a comprehensive XAI quality evaluation report.

        Parameters
        ----------
        task_metrics : list of ExplanationQualityMetrics, optional
            Pre-computed per-task quality metrics.
        consistency_results : list of ConsistencyResult, optional
            Pre-computed SHAP/LIME consistency results.

        Returns
        -------
        XAIQualityReport
        """
        task_metrics = task_metrics or []
        consistency_results = consistency_results or []

        tasks_passing = sum(1 for m in task_metrics if m.meets_threshold)
        tasks_failing = len(task_metrics) - tasks_passing

        if task_metrics:
            aggregate_quality = round(
                float(np.mean([m.overall_quality for m in task_metrics])), 4,
            )
        else:
            aggregate_quality = 0.0

        summary = self._build_summary(
            task_metrics, consistency_results, aggregate_quality,
            tasks_passing, tasks_failing,
        )

        report = XAIQualityReport(
            report_id=f"XAI-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}",
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            system_name=self.system_name,
            task_metrics=task_metrics,
            consistency_results=consistency_results,
            aggregate_quality=aggregate_quality,
            tasks_passing=tasks_passing,
            tasks_failing=tasks_failing,
            summary=summary,
        )

        logger.info(
            "XAI quality report generated: %s (aggregate=%.4f, pass=%d, fail=%d)",
            report.report_id, aggregate_quality, tasks_passing, tasks_failing,
        )
        return report

    def to_dict(self, report: XAIQualityReport) -> Dict[str, Any]:
        """Serialize an ``XAIQualityReport`` to a JSON-compatible dict."""
        return {
            "report_id": report.report_id,
            "evaluated_at": report.evaluated_at,
            "system_name": report.system_name,
            "aggregate_quality": report.aggregate_quality,
            "tasks_passing": report.tasks_passing,
            "tasks_failing": report.tasks_failing,
            "summary": report.summary,
            "task_metrics": [
                {
                    "task_name": m.task_name,
                    "faithfulness_score": m.faithfulness_score,
                    "stability_score": m.stability_score,
                    "comprehensibility_score": m.comprehensibility_score,
                    "coverage": m.coverage,
                    "overall_quality": m.overall_quality,
                    "meets_threshold": m.meets_threshold,
                    "violations": m.violations,
                    "details": m.details,
                }
                for m in report.task_metrics
            ],
            "consistency_results": [
                {
                    "task_name": c.task_name,
                    "method_a": c.method_a,
                    "method_b": c.method_b,
                    "rank_correlation": c.rank_correlation,
                    "top_k_overlap": c.top_k_overlap,
                    "is_consistent": c.is_consistent,
                    "details": c.details,
                }
                for c in report.consistency_results
            ],
            "metadata": report.metadata,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        task_metrics: List[ExplanationQualityMetrics],
        consistency_results: List[ConsistencyResult],
        aggregate_quality: float,
        tasks_passing: int,
        tasks_failing: int,
    ) -> str:
        """Build an executive summary for the XAI quality report."""
        if not task_metrics:
            return "No tasks evaluated. XAI quality assessment requires task metrics."

        parts = [
            f"XAI quality evaluation completed for {len(task_metrics)} task(s).",
            f"Aggregate quality score: {aggregate_quality:.4f}.",
            f"{tasks_passing} task(s) passing, {tasks_failing} task(s) failing thresholds.",
        ]

        if consistency_results:
            consistent = sum(1 for c in consistency_results if c.is_consistent)
            parts.append(
                f"SHAP/LIME consistency: {consistent}/{len(consistency_results)} task(s) consistent."
            )

        if tasks_failing > 0:
            failing_tasks = [m.task_name for m in task_metrics if not m.meets_threshold]
            parts.append(f"Tasks requiring attention: {', '.join(failing_tasks)}.")

        return " ".join(parts)


__all__ = [
    "XAIQualityEvaluator",
    "XAIQualityReport",
    "ExplanationQualityMetrics",
    "ConsistencyResult",
]
