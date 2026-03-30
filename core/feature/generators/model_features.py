"""
Model-Derived Feature Generator (27D output).

Combines three families of model-inspired features into a single generator:

1. **GMM Cluster Probabilities (5D)** -- soft cluster membership probabilities
   from a Gaussian Mixture Model.  Each column is the probability that the
   sample belongs to the k-th component (continuous 0-1, sums to 1).
2. **Bandit / MAB Features (4D)** -- multi-armed bandit style exploration and
   exploitation metrics derived from transaction / engagement patterns.
3. **LNN (Liquid Neural Network) Features (18D)** -- temporal dynamics features
   inspired by liquid time-constant networks, computed purely with numpy
   (multi-scale derivatives, exponential moving averages, temporal attention
   weights, adaptive time constants).

Total output: 5 + 4 + 18 = 27 dimensions.

Hardware acceleration
---------------------
GPU acceleration is **not** used by this generator.  All computations are
numpy / sklearn (GaussianMixture) based.  cuDF may be used for fast
numeric extraction when available.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.data.dataframe import df_backend
from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry
from .gpu_utils import has_cudf

# ---------------------------------------------------------------------------
# Lazy import cuDF (optional GPU acceleration)
# ---------------------------------------------------------------------------
try:
    import cudf as _cudf
except ImportError:
    _cudf = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelFeaturesConfig:
    """Hyper-parameters for the model-derived feature generator."""

    gmm_dim: int = 5
    bandit_dim: int = 4
    lnn_dim: int = 18

    # GMM clustering
    gmm_n_components: int = 5
    gmm_random_state: int = 42

    # LNN temporal scales
    lnn_velocity_windows: List[int] = field(default_factory=lambda: [1, 7, 30])
    lnn_ema_decays: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5])
    lnn_attention_periods: List[int] = field(default_factory=lambda: [7, 14, 30])
    lnn_autocorr_lags: List[int] = field(default_factory=lambda: [1, 7, 30])


# ---------------------------------------------------------------------------
# Model Features Generator
# ---------------------------------------------------------------------------

@FeatureGeneratorRegistry.register(
    "model_features",
    description=(
        "Model-derived features: GMM soft probabilities (5D) + Bandit/MAB (4D) "
        "+ LNN temporal dynamics (18D)."
    ),
    tags=["gmm", "bandit", "lnn", "temporal", "model"],
)
class ModelFeaturesGenerator(AbstractFeatureGenerator):
    """Model-derived feature generator (27D).

    Produces three families of features without requiring heavy model
    training at inference time:

    * **GMM probabilities (5D)** -- soft cluster membership probabilities
      from a Gaussian Mixture Model (one column per component, sums to 1).
    * **Bandit/MAB (4D)** -- exploration / exploitation metrics from
      transaction and engagement columns.
    * **LNN (18D)** -- multi-scale temporal derivatives, exponential
      moving averages, attention weights, and adaptive time constants.

    Parameters
    ----------
    config : ModelFeaturesConfig, optional
        Generator hyper-parameters.
    feature_columns : list[str], optional
        Columns to use for GMM / general numeric features.
        Defaults to all numeric columns.
    engagement_columns : list[str], optional
        Columns to use for bandit metrics (e.g. product/channel usage).
        Defaults to all numeric columns.
    temporal_columns : list[str], optional
        Columns to use for LNN temporal features.
        Defaults to first two numeric columns.
    prefix : str
        Column-name prefix (default ``""`` -- each sub-family has its own).
    """

    supports_gpu: bool = False
    required_libraries: List[str] = ["sklearn"]
    optional_libraries: List[str] = ["cudf"]

    def __init__(
        self,
        config: Optional[ModelFeaturesConfig] = None,
        feature_columns: Optional[List[str]] = None,
        engagement_columns: Optional[List[str]] = None,
        temporal_columns: Optional[List[str]] = None,
        prefix: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config or ModelFeaturesConfig()
        self.feature_columns = feature_columns or []
        self.engagement_columns = engagement_columns or []
        self.temporal_columns = temporal_columns or []
        self.prefix = prefix

        # Fitted state
        self._gmm_model: Any = None
        self._col_means: Optional[np.ndarray] = None
        self._col_stds: Optional[np.ndarray] = None
        self._temporal_means: Optional[np.ndarray] = None
        self._temporal_stds: Optional[np.ndarray] = None

    # -- Output description ------------------------------------------------

    @property
    def output_dim(self) -> int:
        return self.config.gmm_dim + self.config.bandit_dim + self.config.lnn_dim

    @property
    def output_columns(self) -> List[str]:
        p = f"{self.prefix}_" if self.prefix else ""
        cols: List[str] = []
        # GMM soft probabilities (K columns)
        for k in range(self.config.gmm_n_components):
            cols.append(f"{p}gmm_prob_{k}")
        # Bandit/MAB (4D)
        cols.append(f"{p}bandit_exploration_rate")
        cols.append(f"{p}bandit_exploitation_score")
        cols.append(f"{p}bandit_ucb_score")
        cols.append(f"{p}bandit_regret_proxy")
        # LNN (18D)
        for metric_idx in range(2):
            for w in self.config.lnn_velocity_windows:
                cols.append(f"{p}lnn_velocity_m{metric_idx}_{w}d")
        for metric_idx in range(2):
            for decay in self.config.lnn_ema_decays:
                cols.append(f"{p}lnn_ema_m{metric_idx}_{decay}")
        for i, period in enumerate(self.config.lnn_attention_periods):
            cols.append(f"{p}lnn_attention_w{i}_{period}d")
        for i, lag in enumerate(self.config.lnn_autocorr_lags):
            cols.append(f"{p}lnn_time_const_lag{lag}")
        return cols

    @classmethod
    def estimated_output_dim(cls, config: Dict[str, Any]) -> int:
        gmm = config.get("gmm_dim", 5)
        bandit = config.get("bandit_dim", 4)
        lnn = config.get("lnn_dim", 18)
        return gmm + bandit + lnn

    # -- Core API ----------------------------------------------------------

    def fit(self, df: Any, **context: Any) -> "ModelFeaturesGenerator":
        """Fit GMM for soft cluster probabilities and cache normalisation stats."""
        num_cols = self._resolve_numeric_columns(df, self.feature_columns)

        if len(num_cols) > 0:
            X = self._extract_numeric(df, num_cols)
            col_means = np.nanmean(X, axis=0)
            nan_mask = np.isnan(X)
            if nan_mask.any():
                for j in range(X.shape[1]):
                    X[nan_mask[:, j], j] = (
                        col_means[j] if np.isfinite(col_means[j]) else 0.0
                    )
            self._col_means = X.mean(axis=0)
            self._col_stds = X.std(axis=0)
            self._col_stds[self._col_stds < 1e-10] = 1.0

            # Fit GMM for soft cluster probabilities
            X_norm = (X - self._col_means) / self._col_stds
            self._fit_gmm(X_norm)

        # Cache temporal normalisation stats
        temp_cols = self._resolve_temporal_columns(df)
        if len(temp_cols) >= 2:
            T = self._extract_numeric(df, temp_cols[:2])
            self._temporal_means = np.nanmean(T, axis=0)
            self._temporal_stds = np.nanstd(T, axis=0)
            self._temporal_stds[self._temporal_stds < 1e-10] = 1.0

        self._fitted = True
        return self

    def _fit_gmm(self, X_norm: np.ndarray) -> None:
        """Fit sklearn GaussianMixture for soft cluster probabilities.

        cuML does not provide GMM, so we always use sklearn.  Input data
        may have been extracted via CuPy -> numpy for speed.
        """
        try:
            from sklearn.mixture import GaussianMixture

            self._gmm_model = GaussianMixture(
                n_components=self.config.gmm_n_components,
                random_state=self.config.gmm_random_state,
                covariance_type="full",
                max_iter=100,
                n_init=3,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._gmm_model.fit(X_norm)
            logger.info(
                "ModelFeaturesGenerator GMM fitted (sklearn): K=%d, "
                "n_samples=%d, n_features=%d",
                self.config.gmm_n_components,
                X_norm.shape[0],
                X_norm.shape[1],
            )
        except ImportError:
            logger.warning(
                "sklearn not available; GMM features will use fallback."
            )
            self._gmm_model = None

    def generate(self, df: Any, **context: Any) -> Any:
        """Generate model-derived features (27D)."""
        if not self._fitted:
            raise RuntimeError(
                "ModelFeaturesGenerator must be fitted before generate()."
            )

        # Convert to pandas for sub-generators that do row-level iteration;
        # the heavy numeric extraction already uses the GPU path.
        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        n_rows = len(pdf)
        results: Dict[str, np.ndarray] = {}

        p = f"{self.prefix}_" if self.prefix else ""

        # ---- GMM Soft Probabilities (K-D) --------------------------------
        self._generate_gmm_features(pdf, n_rows, p, results)

        # ---- Bandit / MAB Features (4D) ---------------------------------
        self._generate_bandit_features(pdf, n_rows, p, results)

        # ---- LNN Features (18D) -----------------------------------------
        self._generate_lnn_features(pdf, n_rows, p, results)

        # Build output DataFrame via cuDF when available
        if has_cudf() and _cudf is not None:
            return _cudf.DataFrame(results)
        return pd.DataFrame(results)

    # -- GMM Soft Probabilities (K-D) -------------------------------------

    def _generate_gmm_features(
        self,
        pdf: pd.DataFrame,
        n_rows: int,
        p: str,
        results: Dict[str, np.ndarray],
    ) -> None:
        """Generate soft cluster membership probabilities via GMM predict_proba."""
        K = self.config.gmm_n_components
        num_cols = self._resolve_numeric_columns(pdf, self.feature_columns)

        if len(num_cols) == 0 or self._gmm_model is None:
            # Fallback: uniform probabilities
            uniform = np.full(n_rows, 1.0 / K, dtype=np.float32)
            for k in range(K):
                results[f"{p}gmm_prob_{k}"] = uniform.copy()
            return

        X = self._extract_numeric(pdf, num_cols)
        # Impute NaN
        col_means = np.nanmean(X, axis=0)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            for j in range(X.shape[1]):
                X[nan_mask[:, j], j] = (
                    col_means[j] if np.isfinite(col_means[j]) else 0.0
                )

        # Normalise using fitted stats
        if self._col_means is not None and self._col_stds is not None:
            X = (X - self._col_means) / self._col_stds

        # Predict soft probabilities -- shape (n_rows, K)
        proba = self._gmm_model.predict_proba(X)
        proba = np.asarray(proba, dtype=np.float32)

        for k in range(K):
            results[f"{p}gmm_prob_{k}"] = proba[:, k]

    # -- Bandit / MAB (4D) -------------------------------------------------

    def _generate_bandit_features(
        self,
        pdf: pd.DataFrame,
        n_rows: int,
        p: str,
        results: Dict[str, np.ndarray],
    ) -> None:
        """Compute multi-armed bandit style exploration / exploitation metrics."""
        eng_cols = self._resolve_numeric_columns(pdf, self.engagement_columns)

        if len(eng_cols) == 0:
            results[f"{p}bandit_exploration_rate"] = np.zeros(n_rows, dtype=np.float32)
            results[f"{p}bandit_exploitation_score"] = np.zeros(n_rows, dtype=np.float32)
            results[f"{p}bandit_ucb_score"] = np.zeros(n_rows, dtype=np.float32)
            results[f"{p}bandit_regret_proxy"] = np.zeros(n_rows, dtype=np.float32)
            return

        X = self._extract_numeric(pdf, eng_cols)
        # Impute NaN with 0
        X = np.nan_to_num(X, nan=0.0)

        # Exploration rate: diversity of engagement across columns
        # unique non-zero columns / total columns per row
        non_zero_counts = np.sum(X != 0, axis=1).astype(np.float64)
        total_cols = float(X.shape[1])
        exploration_rate = (non_zero_counts / total_cols).astype(np.float32)

        # Exploitation score: concentration on top-1 column
        row_sums = np.sum(np.abs(X), axis=1)
        row_max = np.max(np.abs(X), axis=1)
        exploitation_score = np.where(
            row_sums > 1e-10,
            row_max / row_sums,
            0.0,
        ).astype(np.float32)

        # UCB score: mean + std across engagement columns (upper confidence proxy)
        row_means = np.mean(X, axis=1)
        row_stds = np.std(X, axis=1)
        ucb_score = (row_means + row_stds).astype(np.float32)

        # Regret proxy: difference between best column value and mean
        regret_proxy = (row_max - row_means).astype(np.float32)

        results[f"{p}bandit_exploration_rate"] = exploration_rate
        results[f"{p}bandit_exploitation_score"] = exploitation_score
        results[f"{p}bandit_ucb_score"] = ucb_score
        results[f"{p}bandit_regret_proxy"] = regret_proxy

    # -- LNN Features (18D) ------------------------------------------------

    def _generate_lnn_features(
        self,
        pdf: pd.DataFrame,
        n_rows: int,
        p: str,
        results: Dict[str, np.ndarray],
    ) -> None:
        """Compute Liquid Neural Network inspired temporal dynamics features."""
        temp_cols = self._resolve_temporal_columns(pdf)
        cfg = self.config

        # We need at least 2 temporal columns for the 2-metric scheme
        if len(temp_cols) < 2:
            # Fill all 18 LNN columns with zeros
            for metric_idx in range(2):
                for w in cfg.lnn_velocity_windows:
                    results[f"{p}lnn_velocity_m{metric_idx}_{w}d"] = np.zeros(
                        n_rows, dtype=np.float32
                    )
            for metric_idx in range(2):
                for decay in cfg.lnn_ema_decays:
                    results[f"{p}lnn_ema_m{metric_idx}_{decay}"] = np.zeros(
                        n_rows, dtype=np.float32
                    )
            for i, period in enumerate(cfg.lnn_attention_periods):
                results[f"{p}lnn_attention_w{i}_{period}d"] = np.zeros(
                    n_rows, dtype=np.float32
                )
            for i, lag in enumerate(cfg.lnn_autocorr_lags):
                results[f"{p}lnn_time_const_lag{lag}"] = np.zeros(
                    n_rows, dtype=np.float32
                )
            return

        # Extract two key metrics
        metrics = self._extract_numeric(pdf, temp_cols[:2])
        metrics = np.nan_to_num(metrics, nan=0.0)

        # --- 6 features: multi-scale temporal derivatives (velocity) ---
        # For each of 2 metrics, compute diff at 1d, 7d, 30d windows
        for metric_idx in range(2):
            series = metrics[:, metric_idx]
            for w in cfg.lnn_velocity_windows:
                velocity = np.zeros(n_rows, dtype=np.float64)
                if n_rows > w:
                    velocity[w:] = series[w:] - series[:-w]
                    # Normalise by window size
                    velocity[w:] /= w
                results[f"{p}lnn_velocity_m{metric_idx}_{w}d"] = velocity.astype(
                    np.float32
                )

        # --- 6 features: exponential moving averages at different decays ---
        for metric_idx in range(2):
            series = metrics[:, metric_idx]
            for decay in cfg.lnn_ema_decays:
                ema = np.zeros(n_rows, dtype=np.float64)
                if n_rows > 0:
                    ema[0] = series[0]
                    for i in range(1, n_rows):
                        ema[i] = decay * series[i] + (1.0 - decay) * ema[i - 1]
                results[f"{p}lnn_ema_m{metric_idx}_{decay}"] = ema.astype(
                    np.float32
                )

        # --- 3 features: temporal attention weights ---
        # Softmax of average absolute value in recent periods for first metric
        series0 = metrics[:, 0]
        attention = np.zeros((n_rows, len(cfg.lnn_attention_periods)), dtype=np.float64)
        for pi, period in enumerate(cfg.lnn_attention_periods):
            for i in range(n_rows):
                start = max(0, i - period + 1)
                window_vals = np.abs(series0[start : i + 1])
                attention[i, pi] = np.mean(window_vals) if len(window_vals) > 0 else 0.0

        # Softmax across the 3 period importances per row
        # Shift for numerical stability
        attention_max = attention.max(axis=1, keepdims=True)
        exp_att = np.exp(attention - attention_max)
        att_sum = exp_att.sum(axis=1, keepdims=True)
        att_sum[att_sum < 1e-10] = 1.0
        attention_weights = exp_att / att_sum

        for i, period in enumerate(cfg.lnn_attention_periods):
            results[f"{p}lnn_attention_w{i}_{period}d"] = attention_weights[
                :, i
            ].astype(np.float32)

        # --- 3 features: adaptive time constants (from autocorrelation) ---
        series0 = metrics[:, 0]
        s0_mean = np.mean(series0) if n_rows > 0 else 0.0
        s0_var = np.var(series0) if n_rows > 0 else 1.0
        if s0_var < 1e-10:
            s0_var = 1.0
        centered = series0 - s0_mean

        for i, lag in enumerate(cfg.lnn_autocorr_lags):
            autocorr = np.zeros(n_rows, dtype=np.float64)
            if n_rows > lag:
                # Rolling autocorrelation with given lag
                window = max(lag * 2, 10)
                for row in range(n_rows):
                    start = max(0, row - window + 1)
                    seg = centered[start : row + 1]
                    seg_len = len(seg)
                    if seg_len > lag:
                        c1 = seg[lag:]
                        c0 = seg[: seg_len - lag]
                        var_local = np.mean(seg ** 2)
                        if var_local > 1e-10:
                            autocorr[row] = np.mean(c1 * c0) / var_local
            # Convert autocorrelation to time constant:
            # Higher autocorrelation -> slower dynamics -> larger time constant
            time_const = np.clip(1.0 / (1.0 - np.abs(autocorr) + 1e-6), 0.0, 100.0)
            results[f"{p}lnn_time_const_lag{lag}"] = time_const.astype(np.float32)

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _extract_numeric(df: Any, cols: List[str]) -> np.ndarray:
        """Extract columns as a numpy float64 array from any DataFrame type."""
        from .gpu_utils import _to_numpy_safe
        return _to_numpy_safe(df, cols, fill=0.0)

    def _resolve_numeric_columns(
        self, df: Any, preferred: List[str]
    ) -> List[str]:
        """Resolve columns, falling back to all numeric."""
        if preferred:
            cols = list(df.columns) if hasattr(df, 'columns') else []
            return [c for c in preferred if c in cols]
        if hasattr(df, 'select_dtypes'):
            cols = df.select_dtypes(include=["number"]).columns.tolist()
            return cols if cols else []
        pdf = df_backend.to_pandas(df)
        cols = pdf.select_dtypes(include=["number"]).columns.tolist()
        return cols if cols else []

    def _resolve_temporal_columns(self, df: Any) -> List[str]:
        """Resolve temporal columns, falling back to first 2 numeric."""
        if self.temporal_columns:
            cols = list(df.columns) if hasattr(df, 'columns') else []
            valid = [c for c in self.temporal_columns if c in cols]
            if len(valid) >= 2:
                return valid
        # Fallback: first two numeric columns
        num_cols = self._resolve_numeric_columns(df, [])
        return num_cols[:2] if len(num_cols) >= 2 else num_cols
