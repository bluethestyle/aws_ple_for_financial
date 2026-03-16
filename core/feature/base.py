"""
Feature base classes -- FeatureSchema and AbstractFeatureTransformer.

FeatureSchema declares the column layout of a dataset.  ``input_dim`` is
always computed dynamically from the schema so there is never a hardcoded
dimension anywhere in the system.

AbstractFeatureTransformer defines the sklearn-style ``fit`` / ``transform``
contract that every transformer must follow.

Note: Transformers continue to accept/return pandas DataFrames internally.
The :class:`~core.data.dataframe.DataFrameBackend` converts to pandas
before passing data to transformers, so all transformer logic can assume
``pandas.DataFrame`` without change.
"""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# FeatureSchema
# ──────────────────────────────────────────────────────────────────────


@dataclass
class FeatureSchema:
    """Declares the column layout of a dataset.

    Every column belongs to exactly one category (numeric, categorical,
    sequence, timestamp, or label).  Helper properties compute derived
    quantities such as ``input_dim`` so dimensions are never hardcoded.

    Parameters
    ----------
    numeric : list[str]
        Continuous numeric feature columns.
    categorical : list[str]
        Categorical feature columns (will be encoded).
    sequence : list[str]
        Variable-length sequence feature columns.
    timestamp : list[str]
        Timestamp / date columns.
    label_cols : list[str]
        Target / label columns (excluded from features).
    id_cols : list[str]
        Row-identifier columns (excluded from features and dimensions).
    power_law_skewness_threshold : float
        Skewness above this value flags a column as power-law distributed.
        Power-law columns get an additional ``log1p`` copy in the
        pipeline.
    """

    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    sequence: List[str] = field(default_factory=list)
    timestamp: List[str] = field(default_factory=list)
    label_cols: List[str] = field(default_factory=list)
    id_cols: List[str] = field(default_factory=list)
    power_law_skewness_threshold: float = 2.0

    # ── derived properties ─────────────────────────────────────────────

    @property
    def feature_cols(self) -> List[str]:
        """All feature columns (excludes labels and IDs)."""
        return self.numeric + self.categorical + self.sequence + self.timestamp

    @property
    def input_dim(self) -> int:
        """Dynamically computed feature dimension (numeric + categorical count).

        For categorical columns the dimension equals the number of
        columns *before* one-hot encoding.  Post-encoding dimension
        should be obtained via :meth:`encoded_dim` once the pipeline
        has been fitted.
        """
        return len(self.numeric) + len(self.categorical)

    @property
    def all_columns(self) -> List[str]:
        """All columns including IDs and labels."""
        return self.id_cols + self.feature_cols + self.label_cols

    # ── power-law helpers ──────────────────────────────────────────────

    def detect_power_law_columns(
        self,
        df: pd.DataFrame,
        *,
        threshold: Optional[float] = None,
    ) -> List[str]:
        """Identify numeric columns with skewness above *threshold*.

        Parameters
        ----------
        df : pd.DataFrame
            A sample DataFrame (train split recommended).
        threshold : float, optional
            Skewness cutoff.  Falls back to
            ``self.power_law_skewness_threshold``.

        Returns
        -------
        list[str]
            Column names considered power-law / heavy-tailed.
        """
        threshold = threshold or self.power_law_skewness_threshold
        power_law: List[str] = []
        for col in self.numeric:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < 10:
                continue
            skew = float(series.skew())
            if abs(skew) > threshold:
                power_law.append(col)
        if power_law:
            logger.info(
                "Detected %d power-law columns (|skew| > %.1f): %s",
                len(power_law), threshold, power_law[:5],
            )
        return power_law

    # ── serialisation ──────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary (JSON-safe)."""
        return {
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "sequence": list(self.sequence),
            "timestamp": list(self.timestamp),
            "label_cols": list(self.label_cols),
            "id_cols": list(self.id_cols),
            "power_law_skewness_threshold": self.power_law_skewness_threshold,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureSchema":
        """Deserialise from a dictionary."""
        return cls(
            numeric=d.get("numeric", []),
            categorical=d.get("categorical", []),
            sequence=d.get("sequence", []),
            timestamp=d.get("timestamp", []),
            label_cols=d.get("label_cols", []),
            id_cols=d.get("id_cols", []),
            power_law_skewness_threshold=d.get(
                "power_law_skewness_threshold", 2.0
            ),
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        id_cols: Optional[List[str]] = None,
        label_cols: Optional[List[str]] = None,
        categorical_max_cardinality: int = 50,
    ) -> "FeatureSchema":
        """Auto-detect schema from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
        id_cols : list[str], optional
            Columns to treat as identifiers.
        label_cols : list[str], optional
            Target columns.
        categorical_max_cardinality : int
            Object / string columns with cardinality below this are
            treated as categorical; above this they are dropped with a
            warning.

        Returns
        -------
        FeatureSchema
        """
        id_cols = id_cols or []
        label_cols = label_cols or []
        skip: Set[str] = set(id_cols) | set(label_cols)

        numeric: List[str] = []
        categorical: List[str] = []
        timestamp: List[str] = []

        for col in df.columns:
            if col in skip:
                continue
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                numeric.append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                timestamp.append(col)
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                cardinality = df[col].nunique()
                if cardinality <= categorical_max_cardinality:
                    categorical.append(col)
                else:
                    logger.warning(
                        "Dropping high-cardinality string column '%s' "
                        "(cardinality=%d > %d)",
                        col, cardinality, categorical_max_cardinality,
                    )

        return cls(
            numeric=numeric,
            categorical=categorical,
            timestamp=timestamp,
            label_cols=label_cols,
            id_cols=id_cols,
        )

    # ── repr ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FeatureSchema("
            f"numeric={len(self.numeric)}, "
            f"categorical={len(self.categorical)}, "
            f"sequence={len(self.sequence)}, "
            f"timestamp={len(self.timestamp)}, "
            f"input_dim={self.input_dim})"
        )


# ──────────────────────────────────────────────────────────────────────
# AbstractFeatureTransformer
# ──────────────────────────────────────────────────────────────────────


class AbstractFeatureTransformer(ABC):
    """
    Base class for all feature transformers.

    Follows the sklearn ``fit`` / ``transform`` pattern.  Subclasses
    implement the actual transformation logic and are registered via
    :class:`~core.feature.registry.FeatureRegistry`.

    Attributes
    ----------
    name : str
        Human-readable name used in logging and serialisation.
    columns : list[str] or None
        Columns this transformer operates on.  ``None`` means all
        numeric columns.
    fitted : bool
        Whether ``fit()`` has been called.
    """

    name: str = "base_transformer"

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.columns = columns
        self._fitted = False
        self._fit_columns: Optional[List[str]] = None

    @property
    def fitted(self) -> bool:
        return self._fitted

    # ── core API ───────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "AbstractFeatureTransformer":
        """Learn parameters from *df* (e.g. mean, std, vocabulary)."""
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned transformation to *df*."""
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: ``fit(df).transform(df)``."""
        return self.fit(df).transform(df)

    # ── column resolution ──────────────────────────────────────────────

    def _resolve_columns(self, df: pd.DataFrame) -> List[str]:
        """Return the columns this transformer should operate on.

        If ``self.columns`` is ``None``, defaults to all numeric columns
        in *df*.  Stores the result so that ``transform()`` uses the
        same columns as ``fit()``.
        """
        if self.columns is not None:
            cols = [c for c in self.columns if c in df.columns]
        else:
            cols = df.select_dtypes(include=["number"]).columns.tolist()
        return cols

    # ── serialisation ──────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist the transformer to *path* via pickle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("Saved %s to %s", type(self).__name__, path)

    @classmethod
    def load(cls, path: str) -> "AbstractFeatureTransformer":
        """Load a transformer from *path*."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, AbstractFeatureTransformer):
            raise TypeError(
                f"Expected AbstractFeatureTransformer, got {type(obj).__name__}"
            )
        return obj

    def get_params(self) -> Dict[str, Any]:
        """Return transformer parameters as a dictionary."""
        return {
            "name": self.name,
            "columns": self.columns,
            "fitted": self._fitted,
        }

    # ── repr ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        col_info = f"columns={self.columns}" if self.columns else "columns=all"
        return f"{type(self).__name__}({col_info}, fitted={self._fitted})"
