"""
Data drift detection using Population Stability Index (PSI).

Provides:

- :class:`PSICalculator` -- per-feature PSI computation
- :class:`DriftDetector` -- batch drift detection with warning / critical classification
- :class:`ConsecutiveDriftTracker` -- tracks consecutive critical days and
  triggers auto-retrain when the threshold is reached (default: 3 days)

Thresholds (configurable):
- Warning:  PSI >= 0.1
- Critical: PSI >= 0.25
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PSICalculator
# ---------------------------------------------------------------------------

class PSICalculator:
    """Compute Population Stability Index for feature distributions.

    Parameters
    ----------
    n_bins : int
        Number of histogram bins (default 10).
    min_pct : float
        Floor applied to bin proportions to avoid division by zero.
    """

    def __init__(self, n_bins: int = 10, min_pct: float = 0.0001) -> None:
        self.n_bins = n_bins
        self.min_pct = min_pct

    def calculate_psi(self, baseline: np.ndarray, current: np.ndarray) -> float:
        """Compute PSI between *baseline* and *current* distributions.

        PSI = sum( (current_pct - baseline_pct) * ln(current_pct / baseline_pct) )

        Returns ``numpy.nan`` when either array is empty after NaN removal.
        """
        baseline = baseline[~np.isnan(baseline)]
        current = current[~np.isnan(current)]

        if len(baseline) == 0 or len(current) == 0:
            logger.warning("Empty array; PSI cannot be computed.")
            return float(np.nan)

        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)

        baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
        current_counts, _ = np.histogram(current, bins=bin_edges)

        baseline_pct = np.maximum(baseline_counts / len(baseline), self.min_pct)
        current_pct = np.maximum(current_counts / len(current), self.min_pct)

        psi = float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))
        return psi

    def calculate_psi_batch(
        self,
        baseline_data: Any,
        current_data: Any,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute PSI for every numeric feature column.

        Parameters
        ----------
        baseline_data : dict, pandas DataFrame, or PyArrow Table
            Baseline (training) data.
        current_data : same type as *baseline_data*
            Current (inference) data.
        feature_columns : list of str, optional
            Columns to evaluate.  ``None`` evaluates all shared numeric columns.

        Returns
        -------
        dict
            ``{feature_name: psi_value}``.
        """
        baseline_dict, baseline_cols = self._to_column_dict(baseline_data)
        current_dict, current_cols = self._to_column_dict(current_data)

        if feature_columns is None:
            feature_columns = [c for c in baseline_cols if c in current_cols]

        results: Dict[str, float] = {}
        for col in feature_columns:
            if col not in baseline_dict or col not in current_dict:
                logger.warning("Column '%s' missing from data; skipping.", col)
                continue
            try:
                b_arr = np.asarray(baseline_dict[col], dtype=np.float64)
                c_arr = np.asarray(current_dict[col], dtype=np.float64)
                results[col] = self.calculate_psi(b_arr, c_arr)
            except Exception as exc:
                logger.error("PSI calculation failed for '%s': %s", col, exc)
                results[col] = float(np.nan)
        return results

    @staticmethod
    def _to_column_dict(data: Any) -> tuple:
        """Convert heterogeneous data to ``{column: array}``."""
        try:
            import pyarrow as pa

            if isinstance(data, pa.Table):
                numeric_types = (
                    pa.int8(), pa.int16(), pa.int32(), pa.int64(),
                    pa.float16(), pa.float32(), pa.float64(),
                )
                cols = [f.name for f in data.schema if f.type in numeric_types]
                return {c: data.column(c).to_numpy() for c in cols}, cols
        except ImportError:
            pass

        if isinstance(data, dict):
            return data, list(data.keys())

        # pandas DataFrame fallback
        if hasattr(data, "select_dtypes"):
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
            return {c: data[c].values for c in cols}, cols

        raise TypeError(f"Unsupported data type: {type(data)}")


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class DriftDetector:
    """Detect feature drift using PSI with configurable thresholds.

    Parameters
    ----------
    psi_threshold_warning : float
        PSI value at which a feature is flagged as *warning* (default 0.1).
    psi_threshold_critical : float
        PSI value at which a feature is flagged as *critical* (default 0.25).
    n_bins : int
        Histogram bins for PSI (default 10).
    config : dict, optional
        Full pipeline config dict (from ``load_merged_config()``).  When
        provided, ``monitoring.drift.psi_warning``, ``monitoring.drift.psi_critical``,
        and ``monitoring.drift.n_bins`` are used as defaults before explicit
        kwargs take effect.
    """

    def __init__(
        self,
        psi_threshold_warning: float = 0.1,
        psi_threshold_critical: float = 0.25,
        n_bins: int = 10,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Resolve defaults from pipeline config when provided
        if config is not None:
            _cfg_drift = config.get("monitoring", {}).get("drift", {})
            psi_threshold_warning = float(_cfg_drift.get("psi_warning", psi_threshold_warning))
            psi_threshold_critical = float(_cfg_drift.get("psi_critical", psi_threshold_critical))
            n_bins = int(_cfg_drift.get("n_bins", n_bins))

        self.psi_threshold_warning = psi_threshold_warning
        self.psi_threshold_critical = psi_threshold_critical
        self.calculator = PSICalculator(n_bins=n_bins)

    def detect_drift(
        self,
        baseline_data: Any,
        current_data: Any,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run drift detection across all features.

        Returns
        -------
        dict
            Keys: ``psi_scores``, ``warning_features``, ``critical_features``,
            ``summary``.
        """
        logger.info("Starting drift detection.")
        psi_scores = self.calculator.calculate_psi_batch(
            baseline_data, current_data, feature_columns,
        )

        warning_features: List[str] = []
        critical_features: List[str] = []

        for feature, psi in psi_scores.items():
            if np.isnan(psi):
                continue
            if psi >= self.psi_threshold_critical:
                critical_features.append(feature)
                logger.warning("Critical drift: %s (PSI=%.4f)", feature, psi)
            elif psi >= self.psi_threshold_warning:
                warning_features.append(feature)
                logger.info("Warning drift: %s (PSI=%.4f)", feature, psi)

        valid_psi = [v for v in psi_scores.values() if not np.isnan(v)]
        summary = {
            "total_features": len(psi_scores),
            "warning_count": len(warning_features),
            "critical_count": len(critical_features),
            "max_psi": max(valid_psi) if valid_psi else 0.0,
            "avg_psi": float(np.mean(valid_psi)) if valid_psi else 0.0,
            "drift_detected": len(critical_features) > 0 or len(warning_features) > 0,
        }

        logger.info("Drift detection complete: %s", summary)

        return {
            "psi_scores": psi_scores,
            "warning_features": warning_features,
            "critical_features": critical_features,
            "summary": summary,
        }


# ---------------------------------------------------------------------------
# ConsecutiveDriftTracker
# ---------------------------------------------------------------------------

class ConsecutiveDriftTracker:
    """Track consecutive days of critical drift for auto-retrain decisions.

    Reads historical PSI results (stored as JSON) from a monitoring
    directory and counts how many recent consecutive days had critical
    feature counts above the threshold.

    Parameters
    ----------
    monitoring_dir : str or Path
        Directory containing ``drift_YYYY-MM-DD.json`` result files.
    consecutive_threshold : int
        Number of consecutive critical days required to trigger retrain
        (default 3).
    critical_feature_threshold : int
        Minimum number of critical-PSI features in a day for that day to
        be counted as "critical" (default 5).
    """

    def __init__(
        self,
        monitoring_dir: Union[str, Path],
        consecutive_threshold: int = 3,
        critical_feature_threshold: int = 5,
    ) -> None:
        self.monitoring_dir = Path(monitoring_dir)
        self.consecutive_threshold = consecutive_threshold
        self.critical_feature_threshold = critical_feature_threshold

    def get_consecutive_critical_days(self) -> Dict[str, Any]:
        """Count recent consecutive critical-drift days.

        Returns
        -------
        dict
            ``{"consecutive_days", "should_trigger_retrain", "history",
            "latest_critical_count"}``.
        """
        result_files = sorted(self.monitoring_dir.glob("drift_*.json"))
        if not result_files:
            logger.info("No drift history files found.")
            return {
                "consecutive_days": 0,
                "should_trigger_retrain": False,
                "history": [],
                "latest_critical_count": 0,
            }

        # Examine the most recent files (up to threshold + 2)
        recent_files = result_files[-(self.consecutive_threshold + 2):]
        recent_files.reverse()  # newest first

        history: List[Dict[str, Any]] = []
        consecutive_days = 0
        counting = True

        for f in recent_files:
            try:
                date_str = f.stem.replace("drift_", "")
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                critical_count = data.get("summary", {}).get("critical_count", 0)
                is_critical = critical_count >= self.critical_feature_threshold

                history.append({
                    "date": date_str,
                    "critical_count": critical_count,
                    "is_critical": is_critical,
                })

                if counting and is_critical:
                    consecutive_days += 1
                else:
                    counting = False
            except Exception as exc:
                logger.warning("Failed to process drift file %s: %s", f.name, exc)
                counting = False

        should_trigger = consecutive_days >= self.consecutive_threshold
        latest_critical_count = history[0]["critical_count"] if history else 0

        if should_trigger:
            logger.warning(
                "Consecutive critical drift for %d days (threshold: %d) -- retrain recommended.",
                consecutive_days,
                self.consecutive_threshold,
            )
        else:
            logger.info(
                "Consecutive critical drift: %d days (threshold: %d) -- within normal range.",
                consecutive_days,
                self.consecutive_threshold,
            )

        return {
            "consecutive_days": consecutive_days,
            "should_trigger_retrain": should_trigger,
            "history": history,
            "latest_critical_count": latest_critical_count,
        }


__all__ = ["PSICalculator", "DriftDetector", "ConsecutiveDriftTracker"]
