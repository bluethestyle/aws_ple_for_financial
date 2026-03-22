"""
Config-driven Label Deriver
============================

Derives label columns from raw data based on declarative configuration
objects.  Supports:

* **column** -- use an existing column directly as a label.
* **derive** -- compute a label from another column using one of several
  derivation methods (``first_from_list``, ``contains_in_list``,
  ``len_of_list``, ``threshold``, ``binned``).
* **transforms** -- optional post-processing for regression labels
  (percentile clipping, log1p).

Usage::

    from core.pipeline.label_deriver import LabelDeriver, LabelConfig

    configs = [
        LabelConfig(name="next_item", source="derive",
                    method="first_from_list", input_col="future_items"),
        LabelConfig(name="churn", source="column", col="is_churned",
                    type="classification"),
    ]
    deriver = LabelDeriver()
    labels_df = deriver.derive(raw_df, configs)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["LabelDeriver", "LabelConfig"]


@dataclass
class LabelConfig:
    """Declarative configuration for a single derived label.

    Parameters
    ----------
    name : str
        Output column name for this label.
    source : str
        ``"column"`` to copy an existing column, ``"derive"`` to compute
        from another column using *method*.
    type : str
        ``"classification"`` or ``"regression"``.  Controls whether
        post-processing transforms are applied.
    col : str
        Source column name (when ``source="column"``).
    method : str
        Derivation method (when ``source="derive"``).  One of:
        ``first_from_list``, ``contains_in_list``, ``len_of_list``,
        ``threshold``, ``binned``.
    input_col : str
        Input column for derivation methods.
    item_index : int
        Item index for ``contains_in_list`` method.
    threshold : float
        Threshold value for ``threshold`` method.
    num_classes : int
        Number of bins for ``binned`` method.
    transform : dict
        Post-processing transform config for regression labels.
        Supported keys: ``clip_percentile`` (float), ``log1p`` (bool).
    """

    name: str = ""
    source: str = "column"  # "column" | "derive"
    type: str = "classification"  # "classification" | "regression"
    col: str = ""
    method: str = ""
    input_col: str = ""
    item_index: int = 0
    threshold: float = 0.0
    num_classes: int = 5
    transform: Dict[str, Any] = field(default_factory=dict)
    formula: str = ""
    boundaries: Optional[List[float]] = None
    product_prefix: str = "prod_"
    sequence_cols: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LabelConfig":
        """Create a LabelConfig from a plain dict."""
        return cls(
            name=d.get("name", ""),
            source=d.get("source", "column"),
            type=d.get("type", "classification"),
            col=d.get("col", ""),
            method=d.get("method", ""),
            input_col=d.get("input_col", ""),
            item_index=d.get("item_index", 0),
            threshold=d.get("threshold", 0.0),
            num_classes=d.get("num_classes", 5),
            transform=d.get("transform", {}),
            formula=d.get("formula", ""),
            boundaries=d.get("boundaries"),
            product_prefix=d.get("product_prefix", "prod_"),
            sequence_cols=d.get("sequence_cols", []),
        )


class LabelDeriver:
    """Derive labels from raw data based on config."""

    def derive(
        self,
        df: pd.DataFrame,
        label_configs: List[LabelConfig],
    ) -> pd.DataFrame:
        """Derive all configured label columns.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data.
        label_configs : list[LabelConfig]
            Declarative label definitions.

        Returns
        -------
        pd.DataFrame
            DataFrame with one column per label config, indexed like *df*.
        """
        labels = pd.DataFrame(index=df.index)

        for lc in label_configs:
            name = lc.name
            if lc.source == "column":
                if lc.col not in df.columns:
                    logger.warning(
                        "LabelDeriver: column '%s' not found for label '%s', "
                        "filling with NaN",
                        lc.col,
                        name,
                    )
                    labels[name] = np.nan
                else:
                    labels[name] = df[lc.col]
            elif lc.source == "derive":
                labels[name] = self._derive_label(df, lc)
            else:
                raise ValueError(
                    f"Unknown label source '{lc.source}' for label '{name}'"
                )

            # Apply transforms (clip + log1p for regression)
            if lc.type == "regression" and lc.transform:
                labels[name] = self._apply_transform(labels[name], lc.transform)

            logger.debug(
                "LabelDeriver: derived '%s' (source=%s, type=%s)",
                name,
                lc.source,
                lc.type,
            )

        logger.info(
            "LabelDeriver: derived %d labels: %s",
            len(label_configs),
            [lc.name for lc in label_configs],
        )
        return labels

    def _derive_label(self, df: pd.DataFrame, config: LabelConfig) -> pd.Series:
        """Apply a derivation method to produce a label series.

        Parameters
        ----------
        df : pd.DataFrame
            Source data.
        config : LabelConfig
            Label configuration with method and parameters.

        Returns
        -------
        pd.Series
        """
        method = config.method
        input_col = config.input_col

        if input_col not in df.columns:
            raise KeyError(
                f"LabelDeriver: input_col '{input_col}' not found "
                f"for label '{config.name}' (method={method})"
            )

        if method == "first_from_list":
            return df[input_col].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else -1
            )
        elif method == "contains_in_list":
            item_idx = config.item_index
            return df[input_col].apply(
                lambda x: int(item_idx in x) if isinstance(x, list) else 0
            )
        elif method == "len_of_list":
            return df[input_col].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        elif method == "threshold":
            return (df[input_col] > config.threshold).astype(int)
        elif method == "binned":
            return pd.qcut(
                df[input_col],
                q=config.num_classes,
                labels=False,
                duplicates="drop",
            )
        elif method == "product_not_held":
            # Check if product column == 0 (customer does not hold product)
            return (df[input_col] == 0).astype(int)
        elif method == "sequence_decline":
            # Check if last N elements of sequence are declining
            def _is_declining(x):
                if isinstance(x, (list, np.ndarray)) and len(x) >= 2:
                    arr = np.array(x[-3:]) if len(x) >= 3 else np.array(x)
                    return int(all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1)) and arr[-1] < arr[0])
                return 0
            return df[input_col].apply(_is_declining)
        elif method == "sequence_last":
            # Take last element of a list
            return df[input_col].apply(
                lambda x: x[-1] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0
            )
        elif method == "sequence_stable":
            # Check stability (std < threshold)
            def _is_stable(x, threshold=0.1):
                if isinstance(x, (list, np.ndarray)) and len(x) >= 2:
                    return int(np.std(x) < threshold)
                return 1
            return df[input_col].apply(_is_stable)
        elif method == "computed":
            # Generic expression evaluation using DataFrame columns
            formula = getattr(config, "formula", "") or ""
            if not formula:
                raise ValueError(
                    f"LabelDeriver: 'computed' method requires a formula for label '{config.name}'"
                )
            return df.eval(formula)
        elif method == "age_group_map":
            # Bin age into groups via label encoding of existing groups
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            return pd.Series(le.fit_transform(df[input_col].astype(str).fillna("unknown")), index=df.index)
        elif method == "bucket":
            # Bin continuous into N buckets using boundaries
            boundaries = getattr(config, "boundaries", None)
            if boundaries:
                return pd.cut(
                    df[input_col],
                    bins=[-np.inf] + list(boundaries) + [np.inf],
                    labels=False,
                ).astype(int)
            return pd.qcut(
                df[input_col],
                q=config.num_classes,
                labels=False,
                duplicates="drop",
            )
        elif method == "categorical_encode":
            # Label encode a string column
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            return pd.Series(le.fit_transform(df[input_col].astype(str).fillna("unknown")), index=df.index)
        elif method == "count_active":
            # Count non-zero values across specified columns (by prefix)
            product_prefix = getattr(config, "product_prefix", "prod_")
            prod_cols = [c for c in df.columns if c.startswith(product_prefix)]
            if prod_cols:
                return (df[prod_cols] != 0).sum(axis=1)
            return pd.Series(0, index=df.index)
        elif method == "nba_from_sequences":
            # Derive NBA from sequence changes (argmax of positive deltas)
            sequence_cols = getattr(config, "sequence_cols", [])
            def _nba(row):
                best_idx, best_delta = 0, -np.inf
                for i, col in enumerate(sequence_cols):
                    seq = row.get(col, [])
                    if isinstance(seq, (list, np.ndarray)) and len(seq) >= 2:
                        delta = float(seq[-1]) - float(seq[-2])
                        if delta > best_delta:
                            best_delta = delta
                            best_idx = i
                return best_idx
            return df.apply(_nba, axis=1)
        elif method == "sequence_trend":
            # Compute linear trend slope and classify as up(0)/flat(1)/down(2)
            def _trend(x):
                if isinstance(x, (list, np.ndarray)) and len(x) >= 2:
                    arr = np.array(x, dtype=float)
                    slope = np.polyfit(range(len(arr)), arr, 1)[0]
                    if slope > 0.01:
                        return 0  # up
                    elif slope < -0.01:
                        return 2  # down
                    return 1  # flat
                return 1
            return df[input_col].apply(_trend)
        elif method == "sequence_mean":
            # Compute mean of sequence
            return df[input_col].apply(
                lambda x: float(np.mean(x)) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0.0
            )
        elif method == "sequence_entropy":
            # Compute entropy of sequence
            from scipy.stats import entropy as sp_entropy
            def _entropy(x):
                if isinstance(x, (list, np.ndarray)) and len(x) > 0:
                    arr = np.array(x, dtype=float)
                    counts = np.bincount(arr.astype(int).clip(min=0))
                    probs = counts / counts.sum()
                    return float(sp_entropy(probs + 1e-10))
                return 0.0
            return df[input_col].apply(_entropy)
        else:
            raise ValueError(f"Unknown label derivation method: {method}")

    @staticmethod
    def _apply_transform(
        series: pd.Series,
        transform_config: Dict[str, Any],
    ) -> pd.Series:
        """Apply post-processing transforms to a label series.

        Parameters
        ----------
        series : pd.Series
            Raw label values.
        transform_config : dict
            Keys: ``clip_percentile`` (float, 0-100), ``log1p`` (bool).

        Returns
        -------
        pd.Series
            Transformed label values.
        """
        if transform_config.get("clip_percentile"):
            clip_val = series.quantile(transform_config["clip_percentile"] / 100)
            series = series.clip(upper=clip_val)
        if transform_config.get("log1p"):
            series = np.log1p(series.clip(lower=0))
        return series
