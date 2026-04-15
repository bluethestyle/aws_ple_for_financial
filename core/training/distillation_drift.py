"""
Distillation Drift Monitor — temporal stability check for daily re-distillation.

Detects when a newly distilled LGBM student deviates significantly from the
previous version, triggering alerts for MRM review (SR 11-7 compliance).

Metrics per task:
  - PSI  (Population Stability Index): distribution shift across 10 equal-width bins
  - JSD  (Jensen-Shannon Divergence): symmetrised KL between prediction distributions
  - AUC shift: |current_AUC - previous_AUC| (binary tasks, if labels provided)
  - Mean prediction shift: |mean(current) - mean(previous)|
  - Rank correlation: Spearman rho between current and previous rankings

Usage::

    monitor = DistillationDriftMonitor(config=drift_cfg)

    prev_preds = monitor.load_baseline("/outputs/distillation_baseline")
    if prev_preds is not None:
        report = monitor.compare_versions(
            current_preds=student_preds,
            previous_preds=prev_preds,
            labels=hard_labels,
        )
        for task, metrics in report["per_task"].items():
            if metrics["alert"]:
                logger.warning("Drift alert for %s: %s", task, metrics["alerts"])

    monitor.save_baseline(student_preds, "/outputs/distillation_baseline")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default thresholds (overridden by config)
# ---------------------------------------------------------------------------
_DEFAULT_MAX_PSI: float = 0.20
_DEFAULT_MAX_JSD: float = 0.10
_DEFAULT_MAX_AUC_SHIFT: float = 0.05
_DEFAULT_MAX_PREDICTION_SHIFT: float = 0.10
_DEFAULT_MIN_RANK_CORRELATION: float = 0.95
_PSI_BINS: int = 10


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class DistillationDriftMonitor:
    """Compare consecutive distillation versions for prediction drift.

    Detects when a newly distilled student deviates significantly from the
    previous version, triggering alerts for MRM review (SR 11-7).

    Args:
        config: Optional dict with keys:
            - ``max_psi`` (float): PSI threshold.  Default 0.20.
            - ``max_jsd`` (float): JSD threshold.  Default 0.10.
            - ``max_auc_shift`` (float): |ΔAUC| threshold.  Default 0.05.
            - ``max_prediction_shift`` (float): |Δmean| threshold.  Default 0.10.
            - ``min_rank_correlation`` (float): Spearman rho minimum.  Default 0.95.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._max_psi: float = float(cfg.get("max_psi", _DEFAULT_MAX_PSI))
        self._max_jsd: float = float(cfg.get("max_jsd", _DEFAULT_MAX_JSD))
        self._max_auc_shift: float = float(cfg.get("max_auc_shift", _DEFAULT_MAX_AUC_SHIFT))
        self._max_prediction_shift: float = float(
            cfg.get("max_prediction_shift", _DEFAULT_MAX_PREDICTION_SHIFT)
        )
        self._min_rank_correlation: float = float(
            cfg.get("min_rank_correlation", _DEFAULT_MIN_RANK_CORRELATION)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare_versions(
        self,
        current_preds: Dict[str, np.ndarray],
        previous_preds: Dict[str, np.ndarray],
        labels: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Compare two distillation versions across all shared tasks.

        Args:
            current_preds: Mapping of task_name → current student predictions.
            previous_preds: Mapping of task_name → previous baseline predictions.
            labels: Optional mapping of task_name → hard labels (enables AUC shift).

        Returns:
            Dict with keys:
                - ``per_task``: per-task drift metrics dict
                - ``any_alert``: True if any task exceeded a threshold
                - ``alert_tasks``: list of task names that triggered alerts
                - ``thresholds``: the configured thresholds used
        """
        common_tasks = set(current_preds.keys()) & set(previous_preds.keys())
        if not common_tasks:
            logger.warning(
                "No shared tasks between current and previous predictions — skipping drift check."
            )
            return {"per_task": {}, "any_alert": False, "alert_tasks": [], "thresholds": self._thresholds()}

        per_task: Dict[str, Any] = {}
        alert_tasks: List[str] = []

        for task in sorted(common_tasks):
            curr = np.array(current_preds[task], dtype=np.float32).flatten()
            prev = np.array(previous_preds[task], dtype=np.float32).flatten()
            hard = (
                np.array(labels[task], dtype=np.float64).flatten()
                if (labels and task in labels)
                else None
            )

            task_metrics = self._compute_task_metrics(curr, prev, hard)
            task_alerts = self._evaluate_alerts(task_metrics)
            task_metrics["alert"] = len(task_alerts) > 0
            task_metrics["alerts"] = task_alerts
            per_task[task] = task_metrics

            if task_metrics["alert"]:
                alert_tasks.append(task)
                logger.warning(
                    "DRIFT ALERT [%s]: %s",
                    task,
                    "; ".join(task_alerts),
                )
            else:
                logger.info(
                    "Drift OK [%s]: psi=%.4f jsd=%.4f mean_shift=%.4f rank_corr=%.4f",
                    task,
                    task_metrics.get("psi", float("nan")),
                    task_metrics.get("jsd", float("nan")),
                    task_metrics.get("mean_prediction_shift", float("nan")),
                    task_metrics.get("rank_correlation", float("nan")),
                )

        return {
            "per_task": per_task,
            "any_alert": len(alert_tasks) > 0,
            "alert_tasks": alert_tasks,
            "thresholds": self._thresholds(),
        }

    def save_baseline(self, preds: Dict[str, np.ndarray], path: str) -> None:
        """Save current predictions as baseline for the next comparison.

        Args:
            preds: Mapping of task_name → predictions array.
            path: Directory path (or .npz file path) to save the baseline.
                  If a directory is given, saves as ``{path}/baseline.npz``.
        """
        save_path = Path(path)
        if save_path.suffix != ".npz":
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / "baseline.npz"
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {task: arr.astype(np.float32) for task, arr in preds.items()}
        np.savez_compressed(str(save_path), **arrays)
        logger.info(
            "Drift baseline saved: %d tasks → %s",
            len(arrays),
            save_path,
        )

    def load_baseline(self, path: str) -> Optional[Dict[str, np.ndarray]]:
        """Load previous baseline predictions.

        Args:
            path: Directory or .npz file path saved by :meth:`save_baseline`.

        Returns:
            Mapping of task_name → predictions array, or ``None`` if not found.
        """
        load_path = Path(path)
        if load_path.suffix != ".npz":
            load_path = load_path / "baseline.npz"

        if not load_path.exists():
            logger.info(
                "No drift baseline found at %s — first run, skipping drift check.",
                load_path,
            )
            return None

        try:
            data = np.load(str(load_path), allow_pickle=False)
            result = {task: data[task] for task in data.files}
            logger.info(
                "Drift baseline loaded: %d tasks from %s",
                len(result),
                load_path,
            )
            return result
        except Exception:
            logger.warning(
                "Failed to load drift baseline from %s — treating as first run.",
                load_path,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_task_metrics(
        self,
        curr: np.ndarray,
        prev: np.ndarray,
        labels: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """Compute all drift metrics for a single task."""
        metrics: Dict[str, float] = {}

        metrics["psi"] = _compute_psi(prev, curr, n_bins=_PSI_BINS)
        metrics["jsd"] = _compute_jsd(prev, curr)
        metrics["mean_prediction_shift"] = float(abs(curr.mean() - prev.mean()))
        metrics["rank_correlation"] = _compute_rank_correlation(curr, prev)

        if labels is not None:
            metrics["auc_shift"] = _compute_auc_shift(curr, prev, labels)

        return metrics

    def _evaluate_alerts(self, metrics: Dict[str, float]) -> List[str]:
        """Return list of human-readable alert messages for exceeded thresholds."""
        alerts: List[str] = []

        psi = metrics.get("psi", 0.0)
        if psi > self._max_psi:
            alerts.append(f"PSI={psi:.4f} > max_psi={self._max_psi}")

        jsd = metrics.get("jsd", 0.0)
        if jsd > self._max_jsd:
            alerts.append(f"JSD={jsd:.4f} > max_jsd={self._max_jsd}")

        mean_shift = metrics.get("mean_prediction_shift", 0.0)
        if mean_shift > self._max_prediction_shift:
            alerts.append(
                f"mean_shift={mean_shift:.4f} > max_prediction_shift={self._max_prediction_shift}"
            )

        rank_corr = metrics.get("rank_correlation", 1.0)
        if rank_corr < self._min_rank_correlation:
            alerts.append(
                f"rank_corr={rank_corr:.4f} < min_rank_correlation={self._min_rank_correlation}"
            )

        if "auc_shift" in metrics:
            auc_shift = metrics["auc_shift"]
            if auc_shift > self._max_auc_shift:
                alerts.append(f"auc_shift={auc_shift:.4f} > max_auc_shift={self._max_auc_shift}")

        return alerts

    def _thresholds(self) -> Dict[str, float]:
        return {
            "max_psi": self._max_psi,
            "max_jsd": self._max_jsd,
            "max_auc_shift": self._max_auc_shift,
            "max_prediction_shift": self._max_prediction_shift,
            "min_rank_correlation": self._min_rank_correlation,
        }


# ---------------------------------------------------------------------------
# Standalone metric functions (stateless, reusable)
# ---------------------------------------------------------------------------


def _compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index between two prediction arrays.

    Uses equal-width bins over [0, 1] (probability scale).  Zero-count bins
    are replaced with a small epsilon to avoid log(0).

    PSI interpretation (industry convention):
      < 0.10 → stable
      0.10–0.20 → slight change, monitor
      > 0.20 → significant shift, alert

    Args:
        baseline: Previous version predictions (1-D, [0, 1]).
        current: Current version predictions (1-D, [0, 1]).
        n_bins: Number of equal-width bins.

    Returns:
        PSI as a non-negative float.
    """
    eps = 1e-7
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Clip to [0, 1] to handle slight floating-point overflow
    b = np.clip(baseline, 0.0, 1.0)
    c = np.clip(current, 0.0, 1.0)

    b_counts, _ = np.histogram(b, bins=bin_edges)
    c_counts, _ = np.histogram(c, bins=bin_edges)

    # If predictions fall entirely outside [0,1] (e.g. regression), use
    # quantile-based bins on the *baseline* instead.
    if b_counts.sum() == 0 or c_counts.sum() == 0:
        # Fallback: equal-width bins over combined range
        combined_min = min(float(baseline.min()), float(current.min()))
        combined_max = max(float(baseline.max()), float(current.max()))
        if combined_max <= combined_min:
            return 0.0
        bin_edges = np.linspace(combined_min, combined_max, n_bins + 1)
        b_counts, _ = np.histogram(baseline, bins=bin_edges)
        c_counts, _ = np.histogram(current, bins=bin_edges)

    b_pct = (b_counts / max(b_counts.sum(), 1)).astype(np.float64)
    c_pct = (c_counts / max(c_counts.sum(), 1)).astype(np.float64)

    # Replace zeros to avoid log(0) / division by zero
    b_pct = np.where(b_pct < eps, eps, b_pct)
    c_pct = np.where(c_pct < eps, eps, c_pct)

    psi = float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))
    return max(psi, 0.0)  # numerical guard — PSI is non-negative


def _compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon Divergence between two scalar prediction arrays.

    Converts each 1-D array into a 2-class probability distribution
    ``[x, 1-x]`` and computes sample-wise JSD, then takes the mean.
    This mirrors the pattern in ``distillation_validator.py``.

    Args:
        p: First predictions array (1-D, values in [0, 1]).
        q: Second predictions array (1-D, values in [0, 1]).

    Returns:
        Mean JSD as a float in [0, ln(2)].
    """
    p = np.clip(p.flatten(), 1e-10, 1 - 1e-10)
    q = np.clip(q.flatten(), 1e-10, 1 - 1e-10)

    # Align lengths (take intersection)
    n = min(len(p), len(q))
    p, q = p[:n], q[:n]

    p2 = np.stack([p, 1.0 - p], axis=1)
    q2 = np.stack([q, 1.0 - q], axis=1)
    m = 0.5 * (p2 + q2)

    kl_pm = np.sum(p2 * np.log(p2 / m), axis=1)
    kl_qm = np.sum(q2 * np.log(q2 / m), axis=1)
    return float(np.mean(0.5 * kl_pm + 0.5 * kl_qm))


def _compute_rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two prediction arrays.

    Falls back to numpy-based rank Pearson correlation when scipy is absent.

    Args:
        a: First predictions array.
        b: Second predictions array.

    Returns:
        Spearman rho in [-1, 1].
    """
    a = a.flatten()
    b = b.flatten()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    try:
        from scipy.stats import spearmanr

        corr, _ = spearmanr(a, b)
        return float(corr) if np.isfinite(corr) else 0.0
    except ImportError:
        logger.debug("scipy unavailable — using numpy rank Pearson for Spearman approximation.")
        a_rank = np.argsort(np.argsort(a)).astype(float)
        b_rank = np.argsort(np.argsort(b)).astype(float)
        a_rank -= a_rank.mean()
        b_rank -= b_rank.mean()
        denom = np.sqrt((a_rank ** 2).sum() * (b_rank ** 2).sum())
        if denom < 1e-10:
            return 0.0
        return float((a_rank * b_rank).sum() / denom)


def _compute_auc_shift(
    current: np.ndarray,
    previous: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Absolute AUC shift between current and previous predictions.

    Args:
        current: Current student predictions (binary probabilities).
        previous: Previous student predictions (binary probabilities).
        labels: Hard binary labels (0/1).

    Returns:
        |AUC(current) - AUC(previous)| or 0.0 if computation fails.
    """
    try:
        from sklearn.metrics import roc_auc_score

        valid = np.isfinite(current) & np.isfinite(previous) & np.isfinite(labels)
        valid &= (labels == 0) | (labels == 1)
        if valid.sum() < 10 or len(set(labels[valid].tolist())) < 2:
            return 0.0

        n = min(len(current[valid]), len(previous[valid]), len(labels[valid]))
        curr_auc = roc_auc_score(labels[valid][:n], current[valid][:n])
        prev_auc = roc_auc_score(labels[valid][:n], previous[valid][:n])
        return float(abs(curr_auc - prev_auc))
    except Exception:
        logger.debug("AUC shift computation failed.", exc_info=True)
        return 0.0
