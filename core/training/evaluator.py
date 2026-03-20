"""
Model evaluation with per-task metrics and Champion vs Challenger comparison.

Supports the four task paradigms used by the PLE platform:

* **binary** — AUC-ROC, AUC-PR, F1, Precision, Recall, Log-Loss
* **multiclass** — Macro-F1, Weighted-F1, Accuracy, Top-K Accuracy
* **regression** — MAE, RMSE, R-squared, MAPE
* **ranking** — NDCG@K, MAP@K, MRR

Usage::

    evaluator = ModelEvaluator(
        task_specs=[
            {"name": "ctr", "type": "binary", "primary_metric": "auc_roc"},
            {"name": "revenue", "type": "regression", "primary_metric": "mae"},
        ],
    )
    report = evaluator.evaluate(predictions, targets)
    passed = evaluator.champion_vs_challenger(
        champion_report=old_report,
        challenger_report=report,
    )
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task evaluation specification
# ---------------------------------------------------------------------------

@dataclass
class TaskEvalSpec:
    """Per-task evaluation specification.

    Parameters
    ----------
    name : str
        Task name (must match the key in predictions/targets dicts).
    type : str
        One of ``binary``, ``multiclass``, ``regression``, ``ranking``.
    primary_metric : str
        The single metric used for Champion vs Challenger gating.
    thresholds : dict[str, float], optional
        Minimum acceptable values for named metrics.  If any threshold is
        not met, the evaluation is marked as failed for this task.
    k : int
        Top-K for ranking metrics (NDCG@K, MAP@K).
    higher_is_better : bool, optional
        Override auto-detection of metric direction.  By default this is
        inferred from ``primary_metric`` (e.g. ``auc_roc`` -> True,
        ``mae`` -> False).
    """

    name: str
    type: str
    primary_metric: str = ""
    thresholds: Dict[str, float] = field(default_factory=dict)
    k: int = 10
    higher_is_better: Optional[bool] = None

    def __post_init__(self) -> None:
        if not self.primary_metric:
            self.primary_metric = _DEFAULT_PRIMARY[self.type]
        if self.higher_is_better is None:
            self.higher_is_better = self.primary_metric not in _LOWER_IS_BETTER


_DEFAULT_PRIMARY: Dict[str, str] = {
    "binary": "auc_roc",
    "multiclass": "macro_f1",
    "regression": "mae",
    "ranking": "ndcg",
}

_LOWER_IS_BETTER = frozenset({"mae", "rmse", "mape", "log_loss"})


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _safe(fn, *args, default: float = 0.0, **kwargs) -> float:
    """Call *fn* and return *default* on any exception."""
    try:
        val = float(fn(*args, **kwargs))
        return val if math.isfinite(val) else default
    except Exception:
        return default


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute metrics for a binary classification task.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels (0/1).
    y_pred : np.ndarray
        Predicted probabilities in [0, 1].
    threshold : float
        Decision threshold for hard predictions.

    Returns
    -------
    dict[str, float]
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        log_loss,
    )

    y_hard = (y_pred >= threshold).astype(int)
    return {
        "auc_roc": _safe(roc_auc_score, y_true, y_pred),
        "auc_pr": _safe(average_precision_score, y_true, y_pred),
        "f1": _safe(f1_score, y_true, y_hard, zero_division=0),
        "precision": _safe(precision_score, y_true, y_hard, zero_division=0),
        "recall": _safe(recall_score, y_true, y_hard, zero_division=0),
        "log_loss": _safe(log_loss, y_true, y_pred),
    }


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 5,
) -> Dict[str, float]:
    """Compute metrics for a multiclass classification task.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth class indices.
    y_pred : np.ndarray
        Predicted class probabilities ``(n, num_classes)``.
    k : int
        Top-K for accuracy computation.

    Returns
    -------
    dict[str, float]
    """
    from sklearn.metrics import (
        f1_score,
        accuracy_score,
    )

    y_hard = y_pred.argmax(axis=1)
    n_classes = y_pred.shape[1] if y_pred.ndim == 2 else len(np.unique(y_true))

    metrics: Dict[str, float] = {
        "macro_f1": _safe(f1_score, y_true, y_hard, average="macro", zero_division=0),
        "weighted_f1": _safe(f1_score, y_true, y_hard, average="weighted", zero_division=0),
        "accuracy": _safe(accuracy_score, y_true, y_hard),
    }

    # Top-K accuracy
    if y_pred.ndim == 2 and k <= n_classes:
        top_k_classes = np.argsort(y_pred, axis=1)[:, -k:]
        top_k_hits = np.array([
            y_true[i] in top_k_classes[i] for i in range(len(y_true))
        ])
        metrics[f"top{k}_accuracy"] = float(top_k_hits.mean())

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute metrics for a regression task.

    Returns
    -------
    dict[str, float]
    """
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    mape_denom = np.abs(y_true) + 1e-8
    mape = float(np.mean(np.abs(y_true - y_pred) / mape_denom))

    return {
        "mae": _safe(mean_absolute_error, y_true, y_pred),
        "rmse": _safe(lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))), y_true, y_pred),
        "r2": _safe(r2_score, y_true, y_pred),
        "mape": mape,
    }


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
) -> Dict[str, float]:
    """Compute ranking metrics (NDCG, MAP, MRR).

    Expects ``y_true`` to be relevance scores (non-negative) and
    ``y_pred`` to be predicted scores used for ordering.

    Parameters
    ----------
    y_true : np.ndarray
        Relevance labels per item.
    y_pred : np.ndarray
        Predicted relevance scores.
    k : int
        Cutoff for @K metrics.

    Returns
    -------
    dict[str, float]
    """
    from sklearn.metrics import ndcg_score

    # ndcg_score expects 2-D arrays
    if y_true.ndim == 1:
        y_true_2d = y_true.reshape(1, -1)
        y_pred_2d = y_pred.reshape(1, -1)
    else:
        y_true_2d = y_true
        y_pred_2d = y_pred

    ndcg = _safe(ndcg_score, y_true_2d, y_pred_2d, k=k)

    # MAP@K — mean average precision over ranked lists
    order = np.argsort(-y_pred)[:k] if y_pred.ndim == 1 else np.argsort(-y_pred, axis=1)[:, :k]
    if y_pred.ndim == 1:
        relevant = y_true[order] > 0
        precisions = np.cumsum(relevant) / np.arange(1, len(relevant) + 1)
        ap = float(precisions[relevant].mean()) if relevant.any() else 0.0
    else:
        # Batch MAP
        aps = []
        for i in range(len(y_true_2d)):
            rel = y_true_2d[i][order[i]] > 0
            prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
            aps.append(float(prec[rel].mean()) if rel.any() else 0.0)
        ap = float(np.mean(aps))

    # MRR — reciprocal rank of first relevant item
    if y_pred.ndim == 1:
        rel_positions = np.where(y_true[np.argsort(-y_pred)] > 0)[0]
        mrr = float(1.0 / (rel_positions[0] + 1)) if len(rel_positions) > 0 else 0.0
    else:
        mrrs = []
        for i in range(len(y_true_2d)):
            sorted_labels = y_true_2d[i][np.argsort(-y_pred_2d[i])]
            pos = np.where(sorted_labels > 0)[0]
            mrrs.append(float(1.0 / (pos[0] + 1)) if len(pos) > 0 else 0.0)
        mrr = float(np.mean(mrrs))

    return {
        "ndcg": ndcg,
        f"ndcg@{k}": ndcg,
        "map": ap,
        f"map@{k}": ap,
        "mrr": mrr,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_METRIC_FN = {
    "binary": compute_binary_metrics,
    "multiclass": compute_multiclass_metrics,
    "regression": compute_regression_metrics,
    "ranking": compute_ranking_metrics,
}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """Evaluate model predictions and compare champion vs challenger.

    Parameters
    ----------
    task_specs : list[dict | TaskEvalSpec]
        Per-task evaluation specifications.  Dicts are converted to
        :class:`TaskEvalSpec` automatically.

    Examples
    --------
    >>> evaluator = ModelEvaluator(task_specs=[
    ...     {"name": "ctr", "type": "binary", "primary_metric": "auc_roc"},
    ...     {"name": "revenue", "type": "regression", "primary_metric": "mae"},
    ... ])
    >>> report = evaluator.evaluate(predictions, targets)
    >>> print(report["overall_pass"])
    True
    """

    def __init__(self, task_specs: Sequence[Union[dict, TaskEvalSpec]]):
        self.task_specs: List[TaskEvalSpec] = []
        for spec in task_specs:
            if isinstance(spec, TaskEvalSpec):
                self.task_specs.append(spec)
            else:
                self.task_specs.append(TaskEvalSpec(**spec))

    def evaluate(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Compute per-task metrics and determine pass/fail.

        Parameters
        ----------
        predictions : dict[str, np.ndarray]
            ``{task_name: predicted_values}``.
        targets : dict[str, np.ndarray]
            ``{task_name: ground_truth_values}``.

        Returns
        -------
        dict
            Evaluation report with keys:

            * ``tasks`` — per-task metric dicts
            * ``overall_pass`` — True if all tasks pass their thresholds
            * ``failed_tasks`` — list of task names that failed
        """
        report: Dict[str, Any] = {"tasks": {}, "failed_tasks": []}
        all_pass = True

        for spec in self.task_specs:
            if spec.name not in predictions or spec.name not in targets:
                logger.warning(
                    f"Task '{spec.name}' missing from predictions or targets, "
                    f"skipping evaluation."
                )
                continue

            y_pred = np.asarray(predictions[spec.name])
            y_true = np.asarray(targets[spec.name])

            # Compute metrics
            metric_fn = _METRIC_FN.get(spec.type)
            if metric_fn is None:
                logger.warning(f"Unknown task type '{spec.type}' for '{spec.name}'")
                continue

            if spec.type == "ranking":
                metrics = metric_fn(y_true, y_pred, k=spec.k)
            elif spec.type == "multiclass":
                metrics = metric_fn(y_true, y_pred, k=spec.k)
            else:
                metrics = metric_fn(y_true, y_pred)

            # Check thresholds
            task_pass = True
            threshold_results: Dict[str, Dict[str, Any]] = {}
            for metric_name, min_val in spec.thresholds.items():
                actual = metrics.get(metric_name, 0.0)
                is_lower_better = metric_name in _LOWER_IS_BETTER
                if is_lower_better:
                    ok = actual <= min_val
                else:
                    ok = actual >= min_val
                threshold_results[metric_name] = {
                    "value": actual,
                    "threshold": min_val,
                    "pass": ok,
                }
                if not ok:
                    task_pass = False

            if not task_pass:
                all_pass = False
                report["failed_tasks"].append(spec.name)

            report["tasks"][spec.name] = {
                "type": spec.type,
                "metrics": metrics,
                "primary_metric": spec.primary_metric,
                "primary_value": metrics.get(spec.primary_metric, 0.0),
                "thresholds": threshold_results,
                "pass": task_pass,
            }

        report["overall_pass"] = all_pass
        return report

    def champion_vs_challenger(
        self,
        champion_report: Dict[str, Any],
        challenger_report: Dict[str, Any],
        tolerance: float = 0.0,
    ) -> Dict[str, Any]:
        """Compare a challenger model against the reigning champion.

        The challenger must beat (or match within tolerance) the champion
        on the **primary metric** of **every** task to be promoted.

        Parameters
        ----------
        champion_report : dict
            Evaluation report from :meth:`evaluate` for the current champion.
        challenger_report : dict
            Evaluation report from :meth:`evaluate` for the candidate model.
        tolerance : float
            Allowed degradation ratio (e.g. 0.01 = 1% worse is OK).

        Returns
        -------
        dict
            * ``promote`` — True if challenger beats champion on all tasks.
            * ``comparison`` — per-task comparison details.
            * ``summary`` — human-readable summary string.
        """
        comparison: Dict[str, Dict[str, Any]] = {}
        promote = True

        for spec in self.task_specs:
            champ_data = champion_report.get("tasks", {}).get(spec.name)
            chall_data = challenger_report.get("tasks", {}).get(spec.name)

            if champ_data is None or chall_data is None:
                logger.warning(
                    f"Missing data for task '{spec.name}' in one of the reports."
                )
                comparison[spec.name] = {"status": "skipped"}
                continue

            champ_val = champ_data.get("primary_value", 0.0)
            chall_val = chall_data.get("primary_value", 0.0)

            higher_is_better = spec.higher_is_better
            if higher_is_better:
                threshold = champ_val * (1.0 - tolerance)
                is_better = chall_val >= threshold
                delta = chall_val - champ_val
            else:
                threshold = champ_val * (1.0 + tolerance)
                is_better = chall_val <= threshold
                delta = champ_val - chall_val  # positive = improvement

            comparison[spec.name] = {
                "metric": spec.primary_metric,
                "champion": champ_val,
                "challenger": chall_val,
                "delta": delta,
                "higher_is_better": higher_is_better,
                "threshold": threshold,
                "pass": is_better,
            }

            if not is_better:
                promote = False
                logger.info(
                    f"Task '{spec.name}': challenger LOST "
                    f"({spec.primary_metric}: {chall_val:.6f} vs "
                    f"champion {champ_val:.6f}, threshold={threshold:.6f})"
                )

        # Build summary
        wins = sum(1 for c in comparison.values() if c.get("pass"))
        total = sum(1 for c in comparison.values() if c.get("status") != "skipped")
        summary = (
            f"Challenger {'PROMOTED' if promote else 'REJECTED'}: "
            f"{wins}/{total} tasks passed."
        )

        return {
            "promote": promote,
            "comparison": comparison,
            "summary": summary,
        }

    @staticmethod
    def report_to_json(report: Dict[str, Any], path: Union[str, Path]) -> None:
        """Serialize an evaluation report to JSON.

        Parameters
        ----------
        report : dict
            Report from :meth:`evaluate` or :meth:`champion_vs_challenger`.
        path : str or Path
            Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _default(o: Any) -> Any:
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=_default)

        logger.info(f"Evaluation report saved to {path}")

    @staticmethod
    def report_to_dict(report: Dict[str, Any]) -> Dict[str, Any]:
        """Return the report as a plain dict (already is, but normalizes numpy types)."""

        def _normalize(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _normalize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_normalize(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        return _normalize(report)
