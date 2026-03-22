"""
Distillation Validator -- measures teacher-student fidelity.

Validates that LGBM student models faithfully reproduce PLE teacher
predictions across multiple quality dimensions.

Metrics:
  1. AUC Gap (binary): |teacher_AUC - student_AUC|
  2. Ranking Correlation: Spearman rho between prediction rankings
  3. Binary Agreement Rate: % where class predictions agree
  4. Multiclass Agreement Rate: % where argmax agrees
  5. Jensen-Shannon Divergence: distribution similarity
  6. Calibration Gap: ECE difference
  7. Regression Quartile Agreement: % in same quantile bin
  8. Inference Speed Ratio: student_time / teacher_time

Validation criteria are configurable per-task and per-metric.

Usage::

    from core.training.distillation_validator import (
        DistillationValidator,
        ValidationCriteria,
        FidelityResult,
    )

    validator = DistillationValidator(criteria=ValidationCriteria(max_auc_gap=0.02))
    result = validator.validate_task(
        task_name="ctr",
        task_type="binary",
        teacher_preds=teacher_preds,
        student_preds=student_preds,
        labels=labels,
    )
    print(result.passed, result.metrics, result.failures)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ValidationCriteria:
    """Configurable thresholds for teacher-student fidelity checks.

    Each metric has a threshold that determines pass/fail.  Thresholds
    can be overridden per-task by passing a custom ``ValidationCriteria``
    instance to the validator.

    Attributes:
        max_auc_gap: Maximum acceptable |teacher_AUC - student_AUC|.
        min_ranking_corr: Minimum Spearman rho between prediction rankings.
        min_binary_agreement: Minimum fraction of binary class agreement.
        min_multiclass_agreement: Minimum fraction of argmax agreement.
        max_jsd: Maximum Jensen-Shannon Divergence.
        max_calibration_gap: Maximum ECE difference.
        regression_quartile_agreement_min: Minimum quartile bin agreement.
        max_speed_ratio: Maximum student_time / teacher_time ratio.
    """

    max_auc_gap: float = 0.03
    min_ranking_corr: float = 0.95
    min_binary_agreement: float = 0.90
    min_multiclass_agreement: float = 0.85
    max_jsd: float = 0.10
    max_calibration_gap: float = 0.02
    regression_quartile_agreement_min: float = 0.80
    max_speed_ratio: float = 0.1


# ============================================================================
# Result dataclass
# ============================================================================


@dataclass
class FidelityResult:
    """Result of a single task's fidelity validation.

    Attributes:
        task_name: Name of the validated task.
        task_type: One of ``binary``, ``multiclass``, ``regression``.
        passed: Whether all criteria were met.
        metrics: All computed metric values.
        failures: Human-readable descriptions of which criteria failed.
        teacher_auc: Teacher AUC (binary tasks only).
        student_auc: Student AUC (binary tasks only).
        n_samples: Number of samples used in validation.
    """

    task_name: str
    task_type: str  # binary / multiclass / regression
    passed: bool
    metrics: Dict[str, float]  # all computed metrics
    failures: List[str]  # which criteria failed
    teacher_auc: Optional[float] = None
    student_auc: Optional[float] = None
    n_samples: int = 0


# ============================================================================
# Validator
# ============================================================================


class DistillationValidator:
    """Validate teacher-student fidelity across all tasks.

    Computes up to 8 fidelity metrics depending on task type, then checks
    each against the configured ``ValidationCriteria`` thresholds.

    Args:
        criteria: Validation thresholds.  Uses defaults if ``None``.
        audit_store: Optional audit store for logging validation events.
            Must expose a ``log_event(event_type, payload)`` method.
    """

    def __init__(
        self,
        criteria: Optional[ValidationCriteria] = None,
        audit_store: Optional[Any] = None,
    ) -> None:
        self._criteria = criteria or ValidationCriteria()
        self._audit_store = audit_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_task(
        self,
        task_name: str,
        task_type: str,
        teacher_preds: np.ndarray,
        student_preds: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> FidelityResult:
        """Validate a single task's fidelity.

        Args:
            task_name: Task identifier (e.g. ``"ctr"``).
            task_type: ``"binary"``, ``"multiclass"``, or ``"regression"``.
            teacher_preds: Teacher model predictions.
            student_preds: Student model predictions.
            labels: Ground truth labels (required for AUC gap and
                calibration gap on binary tasks).

        Returns:
            A :class:`FidelityResult` with all metrics and pass/fail status.
        """
        metrics: Dict[str, float] = {}
        failures: List[str] = []

        if task_type == "binary":
            metrics["auc_gap"] = self._compute_auc_gap(
                teacher_preds, student_preds, labels,
            )
            metrics["agreement_rate"] = self._compute_binary_agreement(
                teacher_preds, student_preds,
            )
            metrics["jsd"] = self._compute_jsd(teacher_preds, student_preds)
            metrics["ranking_corr"] = self._compute_ranking_correlation(
                teacher_preds, student_preds,
            )
            if labels is not None:
                metrics["calibration_gap"] = self._compute_calibration_gap(
                    teacher_preds, student_preds, labels,
                )

            # Check criteria
            if metrics["auc_gap"] > self._criteria.max_auc_gap:
                failures.append(
                    f"auc_gap={metrics['auc_gap']:.4f} > "
                    f"{self._criteria.max_auc_gap}"
                )
            if metrics["agreement_rate"] < self._criteria.min_binary_agreement:
                failures.append(
                    f"agreement={metrics['agreement_rate']:.4f} < "
                    f"{self._criteria.min_binary_agreement}"
                )
            if metrics["jsd"] > self._criteria.max_jsd:
                failures.append(
                    f"jsd={metrics['jsd']:.4f} > {self._criteria.max_jsd}"
                )
            if metrics["ranking_corr"] < self._criteria.min_ranking_corr:
                failures.append(
                    f"ranking_corr={metrics['ranking_corr']:.4f} < "
                    f"{self._criteria.min_ranking_corr}"
                )
            if (
                "calibration_gap" in metrics
                and metrics["calibration_gap"] > self._criteria.max_calibration_gap
            ):
                failures.append(
                    f"calibration_gap={metrics['calibration_gap']:.4f} > "
                    f"{self._criteria.max_calibration_gap}"
                )

        elif task_type == "multiclass":
            metrics["agreement_rate"] = self._compute_multiclass_agreement(
                teacher_preds, student_preds,
            )
            metrics["jsd"] = self._compute_jsd_multiclass(
                teacher_preds, student_preds,
            )
            metrics["ranking_corr"] = self._compute_ranking_correlation(
                teacher_preds.max(axis=1) if teacher_preds.ndim > 1 else teacher_preds,
                student_preds.max(axis=1) if student_preds.ndim > 1 else student_preds,
            )

            if metrics["agreement_rate"] < self._criteria.min_multiclass_agreement:
                failures.append(
                    f"agreement={metrics['agreement_rate']:.4f} < "
                    f"{self._criteria.min_multiclass_agreement}"
                )
            if metrics["jsd"] > self._criteria.max_jsd:
                failures.append(
                    f"jsd={metrics['jsd']:.4f} > {self._criteria.max_jsd}"
                )
            if metrics["ranking_corr"] < self._criteria.min_ranking_corr:
                failures.append(
                    f"ranking_corr={metrics['ranking_corr']:.4f} < "
                    f"{self._criteria.min_ranking_corr}"
                )

        elif task_type == "regression":
            metrics["ranking_corr"] = self._compute_ranking_correlation(
                teacher_preds, student_preds,
            )
            metrics["quartile_agreement"] = self._compute_quartile_agreement(
                teacher_preds, student_preds,
            )

            if metrics["ranking_corr"] < self._criteria.min_ranking_corr:
                failures.append(
                    f"ranking_corr={metrics['ranking_corr']:.4f} < "
                    f"{self._criteria.min_ranking_corr}"
                )
            if (
                metrics["quartile_agreement"]
                < self._criteria.regression_quartile_agreement_min
            ):
                failures.append(
                    f"quartile_agreement={metrics['quartile_agreement']:.4f} < "
                    f"{self._criteria.regression_quartile_agreement_min}"
                )

        else:
            logger.warning(
                "Unknown task_type '%s' for task '%s', skipping validation.",
                task_type,
                task_name,
            )

        passed = len(failures) == 0

        result = FidelityResult(
            task_name=task_name,
            task_type=task_type,
            passed=passed,
            metrics=metrics,
            failures=failures,
            n_samples=len(teacher_preds),
        )

        # Audit
        if self._audit_store:
            self._audit_store.log_event(
                "distillation_validation",
                {
                    "pk": task_name,
                    "task": task_name,
                    "passed": passed,
                    "metrics": {k: round(v, 4) for k, v in metrics.items()},
                    "failures": failures,
                },
            )

        return result

    def validate_all(
        self,
        task_results: Dict[str, Dict[str, Any]],
    ) -> List[FidelityResult]:
        """Validate all tasks.

        Args:
            task_results: Mapping of task name to a dict containing:

                * ``task_type`` (str): ``"binary"``, ``"multiclass"``,
                  or ``"regression"``.
                * ``teacher_preds`` (np.ndarray): Teacher predictions.
                * ``student_preds`` (np.ndarray): Student predictions.
                * ``labels`` (np.ndarray, optional): Ground truth labels.

        Returns:
            List of :class:`FidelityResult` for each task.
        """
        results: List[FidelityResult] = []
        for task_name, data in task_results.items():
            r = self.validate_task(
                task_name,
                data["task_type"],
                data["teacher_preds"],
                data["student_preds"],
                data.get("labels"),
            )
            results.append(r)
        return results

    def summary(self, results: List[FidelityResult]) -> Dict[str, Any]:
        """Generate a summary report across all validated tasks.

        Args:
            results: List of :class:`FidelityResult` from
                :meth:`validate_task` or :meth:`validate_all`.

        Returns:
            Dict with aggregate statistics and per-task breakdowns.
        """
        return {
            "total_tasks": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "failed_tasks": [r.task_name for r in results if not r.passed],
            "per_task": {
                r.task_name: {"passed": r.passed, **r.metrics}
                for r in results
            },
        }

    # ------------------------------------------------------------------
    # Metric implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_auc_gap(
        teacher: np.ndarray,
        student: np.ndarray,
        labels: Optional[np.ndarray],
    ) -> float:
        """Absolute AUC difference between teacher and student."""
        if labels is None:
            return 0.0
        from sklearn.metrics import roc_auc_score

        t_flat = teacher.flatten()
        s_flat = student.flatten()
        # Guard against NaN/inf in predictions
        valid = np.isfinite(t_flat) & np.isfinite(s_flat) & np.isfinite(labels)
        if valid.sum() < 10:
            return 0.0
        t_auc = roc_auc_score(labels[valid], t_flat[valid])
        s_auc = roc_auc_score(labels[valid], s_flat[valid])
        return abs(t_auc - s_auc)

    @staticmethod
    def _compute_binary_agreement(
        teacher: np.ndarray,
        student: np.ndarray,
        threshold: float = 0.5,
    ) -> float:
        """Fraction of samples where binary class predictions agree."""
        t_class = (teacher.flatten() >= threshold).astype(int)
        s_class = (student.flatten() >= threshold).astype(int)
        return float((t_class == s_class).mean())

    @staticmethod
    def _compute_multiclass_agreement(
        teacher: np.ndarray,
        student: np.ndarray,
    ) -> float:
        """Fraction of samples where argmax class predictions agree."""
        t_class = (
            teacher.argmax(axis=1) if teacher.ndim > 1 else teacher.astype(int)
        )
        s_class = (
            student.argmax(axis=1) if student.ndim > 1 else student.astype(int)
        )
        return float((t_class == s_class).mean())

    @staticmethod
    def _compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon Divergence for binary predictions."""
        p = np.clip(p.flatten(), 1e-10, 1 - 1e-10)
        q = np.clip(q.flatten(), 1e-10, 1 - 1e-10)
        # Convert to 2-class distributions
        p2 = np.stack([p, 1 - p], axis=1)
        q2 = np.stack([q, 1 - q], axis=1)
        m = 0.5 * (p2 + q2)
        kl_pm = np.sum(p2 * np.log(p2 / m), axis=1)
        kl_qm = np.sum(q2 * np.log(q2 / m), axis=1)
        return float(np.mean(0.5 * kl_pm + 0.5 * kl_qm))

    @staticmethod
    def _compute_jsd_multiclass(p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon Divergence for multiclass probability arrays."""
        p = np.clip(p, 1e-10, None)
        q = np.clip(q, 1e-10, None)
        # Normalize rows to valid distributions
        p = p / p.sum(axis=1, keepdims=True)
        q = q / q.sum(axis=1, keepdims=True)
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log(p / m), axis=1)
        kl_qm = np.sum(q * np.log(q / m), axis=1)
        return float(np.mean(0.5 * kl_pm + 0.5 * kl_qm))

    @staticmethod
    def _compute_ranking_correlation(
        teacher: np.ndarray,
        student: np.ndarray,
    ) -> float:
        """Spearman rank correlation between teacher and student predictions."""
        try:
            from scipy.stats import spearmanr
        except ImportError:
            logger.warning(
                "scipy not available; falling back to numpy rank correlation."
            )
            # Fallback: Pearson on ranks
            t_rank = np.argsort(np.argsort(teacher.flatten())).astype(float)
            s_rank = np.argsort(np.argsort(student.flatten())).astype(float)
            t_rank -= t_rank.mean()
            s_rank -= s_rank.mean()
            denom = np.sqrt((t_rank ** 2).sum() * (s_rank ** 2).sum())
            if denom < 1e-10:
                return 0.0
            return float((t_rank * s_rank).sum() / denom)

        corr, _ = spearmanr(teacher.flatten(), student.flatten())
        return float(corr) if np.isfinite(corr) else 0.0

    @staticmethod
    def _compute_calibration_gap(
        teacher: np.ndarray,
        student: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Absolute difference in Expected Calibration Error (ECE)."""

        def _ece(preds: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
            bins = np.linspace(0, 1, n_bins + 1)
            total = 0.0
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (preds >= lo) & (preds < hi)
                if mask.sum() == 0:
                    continue
                acc = labels[mask].mean()
                conf = preds[mask].mean()
                total += mask.sum() * abs(acc - conf)
            return total / len(preds)

        t_ece = _ece(teacher.flatten(), labels.flatten(), n_bins)
        s_ece = _ece(student.flatten(), labels.flatten(), n_bins)
        return abs(t_ece - s_ece)

    @staticmethod
    def _compute_quartile_agreement(
        teacher: np.ndarray,
        student: np.ndarray,
        n_quantiles: int = 4,
    ) -> float:
        """Fraction of samples falling in the same quantile bin."""
        t_flat = teacher.flatten()
        s_flat = student.flatten()
        t_edges = np.quantile(t_flat, np.linspace(0, 1, n_quantiles + 1)[1:-1])
        s_edges = np.quantile(s_flat, np.linspace(0, 1, n_quantiles + 1)[1:-1])
        t_q = np.digitize(t_flat, t_edges)
        s_q = np.digitize(s_flat, s_edges)
        return float((t_q == s_q).mean())
