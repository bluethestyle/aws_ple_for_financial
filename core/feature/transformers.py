"""
Built-in feature transformers.

Every class is auto-registered with the :class:`FeatureRegistry` via the
``@FeatureRegistry.register(...)`` decorator so they can be referenced by
name in YAML pipeline configs.

Transformers operate on pandas DataFrames internally.  The
:class:`~core.data.dataframe.DataFrameBackend` handles conversion from
the active backend (DuckDB, cuDF) to pandas before transformer execution.

Registered transformers
-----------------------
* ``standard_scaler`` -- z-score normalisation
* ``quantile_transformer`` -- quantile (rank) mapping to a normal/uniform distribution
* ``log_transformer`` -- ``log1p(max(x, 0))``
* ``minmax_scaler`` -- scale to [0, 1]
* ``label_encoder`` -- categorical string -> integer
* ``hash_encoder`` -- categorical string -> deterministic hash
* ``null_filler`` -- configurable null imputation (mean / median / zero / mode)
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import AbstractFeatureTransformer
from .registry import FeatureRegistry

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Numeric scalers
# ──────────────────────────────────────────────────────────────────────


@FeatureRegistry.register(
    "standard_scaler",
    description="Z-score normalisation (mean=0, std=1).",
    tags=["numeric", "scaler"],
)
class StandardScaler(AbstractFeatureTransformer):
    """Z-score normalisation: ``(x - mean) / std``.

    Parameters
    ----------
    columns : list[str], optional
        Columns to scale.  ``None`` = all numeric columns.
    clip_std : float, optional
        If set, clip the scaled values to [-clip_std, clip_std].
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        clip_std: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(columns=columns, **kwargs)
        self.clip_std = clip_std
        self._mean: Optional[pd.Series] = None
        self._std: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "StandardScaler":
        cols = self._resolve_columns(df)
        self._fit_columns = cols
        self._mean = df[cols].mean()
        self._std = df[cols].std().replace(0, 1.0)
        self._fitted = True
        logger.debug("StandardScaler fitted on %d columns", len(cols))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("StandardScaler must be fitted before transform().")
        df = df.copy()
        cols = self._fit_columns
        scaled = (df[cols] - self._mean) / self._std
        if self.clip_std is not None:
            scaled = scaled.clip(-self.clip_std, self.clip_std)
        df[cols] = scaled
        return df

    def get_params(self) -> Dict[str, Any]:
        base = super().get_params()
        if self._mean is not None:
            base["mean"] = self._mean.to_dict()
            base["std"] = self._std.to_dict()
        return base


@FeatureRegistry.register(
    "quantile_transformer",
    description="Quantile-based mapping to a normal or uniform distribution.",
    tags=["numeric", "scaler"],
)
class QuantileTransformer(AbstractFeatureTransformer):
    """Rank-based mapping to a target distribution.

    Wraps ``sklearn.preprocessing.QuantileTransformer`` internally.

    Parameters
    ----------
    columns : list[str], optional
    n_quantiles : int
        Number of quantiles.
    output_distribution : str
        ``"normal"`` or ``"uniform"``.
    random_state : int
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        n_quantiles: int = 1000,
        output_distribution: str = "normal",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(columns=columns, **kwargs)
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.random_state = random_state
        self._qt = None  # sklearn QuantileTransformer

    def fit(self, df: pd.DataFrame) -> "QuantileTransformer":
        from sklearn.preprocessing import (
            QuantileTransformer as SklearnQT,
        )

        cols = self._resolve_columns(df)
        self._fit_columns = cols
        self._qt = SklearnQT(
            n_quantiles=self.n_quantiles,
            output_distribution=self.output_distribution,
            random_state=self.random_state,
        )
        self._qt.fit(df[cols].values.astype(np.float64))
        self._fitted = True
        logger.debug("QuantileTransformer fitted on %d columns", len(cols))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("QuantileTransformer must be fitted before transform().")
        df = df.copy()
        cols = self._fit_columns
        df[cols] = self._qt.transform(df[cols].values.astype(np.float64))
        return df


@FeatureRegistry.register(
    "log_transformer",
    description="log1p(max(x, 0)) for heavy-tailed distributions.",
    tags=["numeric", "scaler"],
)
class LogTransformer(AbstractFeatureTransformer):
    """Apply ``log1p(max(x, 0))`` to reduce heavy-tail skew.

    Parameters
    ----------
    columns : list[str], optional
    add_raw_copy : bool
        If ``True``, keep the original column as ``{col}_raw`` before
        applying the log transform.
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        add_raw_copy: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(columns=columns, **kwargs)
        self.add_raw_copy = add_raw_copy

    def fit(self, df: pd.DataFrame) -> "LogTransformer":
        cols = self._resolve_columns(df)
        self._fit_columns = cols
        self._fitted = True
        logger.debug("LogTransformer fitted on %d columns", len(cols))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("LogTransformer must be fitted before transform().")
        df = df.copy()
        cols = self._fit_columns
        for col in cols:
            if self.add_raw_copy:
                df[f"{col}_raw"] = df[col]
            df[col] = np.log1p(np.maximum(df[col].values.astype(np.float64), 0))
        return df


@FeatureRegistry.register(
    "minmax_scaler",
    description="Scale features to [0, 1] range.",
    tags=["numeric", "scaler"],
)
class MinMaxScaler(AbstractFeatureTransformer):
    """Scale each column to the [0, 1] range.

    Parameters
    ----------
    columns : list[str], optional
    feature_range : tuple[float, float]
        Desired range (default ``(0.0, 1.0)``).
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        feature_range: tuple = (0.0, 1.0),
        **kwargs: Any,
    ) -> None:
        super().__init__(columns=columns, **kwargs)
        self.feature_range = feature_range
        self._min: Optional[pd.Series] = None
        self._max: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "MinMaxScaler":
        cols = self._resolve_columns(df)
        self._fit_columns = cols
        self._min = df[cols].min()
        self._max = df[cols].max()
        self._fitted = True
        logger.debug("MinMaxScaler fitted on %d columns", len(cols))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("MinMaxScaler must be fitted before transform().")
        df = df.copy()
        cols = self._fit_columns
        lo, hi = self.feature_range
        denom = (self._max - self._min).replace(0, 1.0)
        scaled = (df[cols] - self._min) / denom
        df[cols] = scaled * (hi - lo) + lo
        return df


# ──────────────────────────────────────────────────────────────────────
# Categorical encoders
# ──────────────────────────────────────────────────────────────────────


@FeatureRegistry.register(
    "label_encoder",
    description="Map categorical strings to integer codes.",
    tags=["categorical", "encoder"],
)
class LabelEncoder(AbstractFeatureTransformer):
    """Map each unique string value to a sequential integer.

    Unknown values seen at ``transform()`` time are mapped to a
    reserved ``unknown_value`` index.

    Parameters
    ----------
    columns : list[str], optional
        Categorical columns to encode.
    unknown_value : int
        Integer code assigned to unseen categories (default ``-1``).
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        unknown_value: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(columns=columns, **kwargs)
        self.unknown_value = unknown_value
        self._mappings: Dict[str, Dict[Any, int]] = {}

    def fit(self, df: pd.DataFrame) -> "LabelEncoder":
        cols = self.columns or df.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()
        self._fit_columns = cols
        self._mappings = {}
        for col in cols:
            uniques = sorted(df[col].dropna().unique(), key=str)
            self._mappings[col] = {v: i for i, v in enumerate(uniques)}
        self._fitted = True
        logger.debug(
            "LabelEncoder fitted on %d columns, vocab sizes: %s",
            len(cols),
            {c: len(m) for c, m in self._mappings.items()},
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("LabelEncoder must be fitted before transform().")
        df = df.copy()
        for col in self._fit_columns:
            mapping = self._mappings[col]
            df[col] = df[col].map(mapping).fillna(self.unknown_value).astype(int)
        return df

    def get_params(self) -> Dict[str, Any]:
        base = super().get_params()
        base["vocab_sizes"] = {c: len(m) for c, m in self._mappings.items()}
        return base


@FeatureRegistry.register(
    "hash_encoder",
    description="Deterministic hash encoding for high-cardinality categoricals.",
    tags=["categorical", "encoder"],
)
class HashEncoder(AbstractFeatureTransformer):
    """Map categorical values to a hash-based integer in ``[0, n_bins)``.

    Deterministic (same value always maps to the same bin) and handles
    unseen values gracefully.

    Parameters
    ----------
    columns : list[str], optional
        Categorical columns.
    n_bins : int
        Number of hash buckets (default 1024).
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        n_bins: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(columns=columns, **kwargs)
        self.n_bins = n_bins

    def fit(self, df: pd.DataFrame) -> "HashEncoder":
        cols = self.columns or df.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()
        self._fit_columns = cols
        self._fitted = True
        logger.debug("HashEncoder fitted on %d columns, n_bins=%d", len(cols), self.n_bins)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("HashEncoder must be fitted before transform().")
        df = df.copy()
        for col in self._fit_columns:
            df[col] = df[col].apply(self._hash_value).astype(int)
        return df

    def _hash_value(self, val: Any) -> int:
        if pd.isna(val):
            return 0
        digest = hashlib.md5(str(val).encode("utf-8")).hexdigest()
        return int(digest, 16) % self.n_bins


# ──────────────────────────────────────────────────────────────────────
# Null handling
# ──────────────────────────────────────────────────────────────────────


@FeatureRegistry.register(
    "null_filler",
    description="Fill NULLs with a configurable strategy (mean/median/zero/mode).",
    tags=["numeric", "categorical", "imputer"],
)
class NullFiller(AbstractFeatureTransformer):
    """Impute missing values.

    Parameters
    ----------
    columns : list[str], optional
        Columns to fill.  ``None`` = all columns.
    strategy : str
        ``"mean"`` (numeric), ``"median"`` (numeric), ``"zero"``
        (numeric), or ``"mode"`` (categorical).
    fill_value : Any, optional
        Explicit constant to use when ``strategy="constant"``.
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        strategy: str = "zero",
        fill_value: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(columns=columns, **kwargs)
        if strategy not in ("mean", "median", "zero", "mode", "constant"):
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Choose from: mean, median, zero, mode, constant."
            )
        self.strategy = strategy
        self.fill_value = fill_value
        self._fill_values: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> "NullFiller":
        cols = self.columns or df.columns.tolist()
        self._fit_columns = [c for c in cols if c in df.columns]
        self._fill_values = {}

        for col in self._fit_columns:
            if self.strategy == "mean":
                self._fill_values[col] = float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else 0
            elif self.strategy == "median":
                self._fill_values[col] = float(df[col].median()) if pd.api.types.is_numeric_dtype(df[col]) else 0
            elif self.strategy == "zero":
                self._fill_values[col] = 0
            elif self.strategy == "mode":
                mode_vals = df[col].mode()
                self._fill_values[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else 0
            elif self.strategy == "constant":
                self._fill_values[col] = self.fill_value

        self._fitted = True
        logger.debug("NullFiller (strategy=%s) fitted on %d columns", self.strategy, len(self._fit_columns))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("NullFiller must be fitted before transform().")
        df = df.copy()
        for col in self._fit_columns:
            if col in df.columns and col in self._fill_values:
                df[col] = df[col].fillna(self._fill_values[col])
        return df
