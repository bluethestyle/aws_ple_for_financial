"""
5-axis Schema Classifier
========================

Automatically classifies DataFrame columns into five semantic axes:

* **state** -- scalar numeric or categorical features describing current state
  (e.g. age, balance, risk_score).
* **snapshot** -- list/array-valued columns representing a point-in-time
  snapshot of a variable-length structure.
* **timeseries** -- ordered temporal sequences (reserved for explicit config
  or future auto-detection based on temporal ordering cues).
* **hierarchy** -- columns encoding hierarchical relationships (e.g.
  category codes, org trees).
* **item** -- binary indicator columns (0/1 flags).

The classifier supports two modes:

1. **Config-driven** -- axes are pre-specified in a config dict; only
   unclassified columns are auto-detected.
2. **Full auto-detection** -- every column is classified by heuristic.

Usage::

    from core.pipeline.schema_classifier import SchemaClassifier

    clf = SchemaClassifier()
    axes = clf.classify(df)
    # => {"state": ["age", "balance"], "item": ["is_vip"], ...}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["SchemaClassifier"]


class SchemaClassifier:
    """Classify DataFrame columns into 5 axes: state, snapshot, timeseries, hierarchy, item."""

    AXES = ["state", "snapshot", "timeseries", "hierarchy", "item"]

    def classify(
        self,
        df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        """Classify columns into axes.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame whose columns are to be classified.
        config : dict, optional
            If provided and contains an ``"axes"`` key, those explicit
            assignments take priority.  Remaining unclassified columns
            are auto-detected.

        Returns
        -------
        dict[str, list[str]]
            Mapping from axis name to list of column names.
        """
        result: Dict[str, List[str]] = {ax: [] for ax in self.AXES}

        # Config override takes priority
        if config and "axes" in config:
            for ax, cols in config["axes"].items():
                if ax not in result:
                    logger.warning(
                        "SchemaClassifier: unknown axis '%s' in config, skipping", ax
                    )
                    continue
                result[ax] = list(cols)

            # Remaining unclassified columns
            classified = set(sum(result.values(), []))
            unclassified = [c for c in df.columns if c not in classified]
            for col in unclassified:
                result[self._auto_detect_axis(df, col)].append(col)

            logger.info(
                "SchemaClassifier (config+auto): %s",
                {ax: len(cols) for ax, cols in result.items()},
            )
            return result

        # Full auto-detection
        for col in df.columns:
            result[self._auto_detect_axis(df, col)].append(col)

        logger.info(
            "SchemaClassifier (auto): %s",
            {ax: len(cols) for ax, cols in result.items()},
        )
        return result

    def _auto_detect_axis(self, df: pd.DataFrame, col: str) -> str:
        """Detect which axis a column belongs to using heuristics.

        Parameters
        ----------
        df : pd.DataFrame
            The source DataFrame.
        col : str
            Column name to classify.

        Returns
        -------
        str
            One of :attr:`AXES`.
        """
        dtype = df[col].dtype

        # List/array type -> snapshot (e.g. PyArrow list columns)
        if hasattr(dtype, "pyarrow_dtype") or str(dtype).startswith("list"):
            return "snapshot"

        # Check first non-null value for list-like content
        first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        if isinstance(first_valid, (list,)):
            return "snapshot"

        # Binary (0/1 only, nunique <= 2) -> item
        nunique = df[col].nunique()
        if nunique <= 2:
            unique_vals = set(df[col].dropna().unique())
            if unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
                return "item"

        # Categorical string -> state
        if dtype == "object" or str(dtype) == "string":
            return "state"

        # Numeric -> state (default)
        return "state"
