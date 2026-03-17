"""
Merchant Hierarchy Feature Generator (21D output).

Extracts multi-level merchant category code (MCC) features, SVD-based brand
embeddings, aggregate merchant statistics, and hierarchy radius features
from transaction data.

Output per row (21 dimensions total):
  - MCC Level 1 features (4D): softmax encoding of dominant top-level category
  - MCC Level 2 features (4D): diversity, concentration, trend, novelty
  - Brand embedding (8D): truncated SVD of user-merchant interaction matrix
  - Aggregate stats (4D): merchant count, loyalty, recency, spend Gini
  - Hierarchy radius (1D): categorical spread of merchant visits

Hardware acceleration
---------------------
CPU-only generator.  Uses numpy and sklearn (TruncatedSVD) for all
computation.  No GPU backend is available.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.data.dataframe import df_backend
from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry
from .gpu_utils import has_cupy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MerchantHierarchyConfig:
    """Merchant hierarchy feature hyper-parameters."""

    mcc_level1_dim: int = 4
    mcc_level2_dim: int = 4
    brand_embed_dim: int = 8
    agg_stats_dim: int = 4
    hierarchy_radius_dim: int = 1


# ---------------------------------------------------------------------------
# Merchant Hierarchy Feature Generator
# ---------------------------------------------------------------------------

@FeatureGeneratorRegistry.register(
    "merchant_hierarchy",
    description="Merchant hierarchy features: MCC levels, brand SVD, aggregate stats, radius.",
    tags=["merchant", "hierarchy", "mcc", "svd"],
)
class MerchantHierarchyGenerator(AbstractFeatureGenerator):
    """Merchant hierarchy feature generator.

    Produces merchant-level features by analysing MCC (Merchant Category Code)
    distributions, computing SVD-based brand embeddings from user-merchant
    interaction matrices, and deriving aggregate merchant statistics.

    Output columns (for prefix ``mh`` and default config, 21D total):
      - ``mh_mcc_l1_dominant_0`` ... ``mh_mcc_l1_dominant_3``: softmax L1 encoding
      - ``mh_mcc_l2_diversity``: entropy of L2 category distribution
      - ``mh_mcc_l2_concentration``: HHI of L2 categories
      - ``mh_mcc_l2_trend``: slope of L2 usage over time
      - ``mh_mcc_l2_novelty``: ratio of new L2 categories in recent period
      - ``mh_brand_embed_0`` ... ``mh_brand_embed_7``: SVD brand embedding
      - ``mh_merchant_count``: log-scaled unique merchant count
      - ``mh_merchant_loyalty``: repeat visit ratio
      - ``mh_merchant_recency``: normalised days since last new merchant
      - ``mh_merchant_spend_gini``: Gini coefficient of merchant spend
      - ``mh_hierarchy_radius``: categorical spread of merchant visits

    Parameters
    ----------
    config : MerchantHierarchyConfig, optional
        Feature dimension hyper-parameters.
    prefix : str
        Column-name prefix.
    """

    supports_gpu: bool = False
    required_libraries: List[str] = ["sklearn"]

    def __init__(
        self,
        config: Optional[MerchantHierarchyConfig] = None,
        prefix: str = "mh",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config or MerchantHierarchyConfig()
        self.prefix = prefix

        # Fitted state
        self._svd_model: Any = None
        self._merchant_cols: List[str] = []
        self._mcc_cols: List[str] = []
        self._numeric_cols: List[str] = []

    # -- Output description ------------------------------------------------

    @property
    def output_dim(self) -> int:
        cfg = self.config
        return (
            cfg.mcc_level1_dim
            + cfg.mcc_level2_dim
            + cfg.brand_embed_dim
            + cfg.agg_stats_dim
            + cfg.hierarchy_radius_dim
        )

    @property
    def output_columns(self) -> List[str]:
        cfg = self.config
        cols: List[str] = []

        # MCC Level 1 (4D)
        for i in range(cfg.mcc_level1_dim):
            cols.append(f"{self.prefix}_mcc_l1_dominant_{i}")

        # MCC Level 2 (4D)
        cols.append(f"{self.prefix}_mcc_l2_diversity")
        cols.append(f"{self.prefix}_mcc_l2_concentration")
        cols.append(f"{self.prefix}_mcc_l2_trend")
        cols.append(f"{self.prefix}_mcc_l2_novelty")

        # Brand embedding (8D)
        for i in range(cfg.brand_embed_dim):
            cols.append(f"{self.prefix}_brand_embed_{i}")

        # Aggregate stats (4D)
        cols.append(f"{self.prefix}_merchant_count")
        cols.append(f"{self.prefix}_merchant_loyalty")
        cols.append(f"{self.prefix}_merchant_recency")
        cols.append(f"{self.prefix}_merchant_spend_gini")

        # Hierarchy radius (1D)
        cols.append(f"{self.prefix}_hierarchy_radius")

        return cols

    # -- Core API ----------------------------------------------------------

    def fit(self, df: Any, **context: Any) -> "MerchantHierarchyGenerator":
        """Fit the merchant hierarchy generator.

        Identifies relevant merchant/MCC columns and fits the TruncatedSVD
        model for brand embedding extraction.
        """
        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df

        self._mcc_cols = self._find_mcc_columns(pdf)
        self._merchant_cols = self._find_merchant_columns(pdf)
        self._numeric_cols = pdf.select_dtypes(include=["number"]).columns.tolist()

        # Fit SVD on user-merchant interaction matrix
        self._fit_svd(pdf)

        self._fitted = True
        logger.info(
            "MerchantHierarchyGenerator fitted: mcc_cols=%d, merchant_cols=%d, "
            "n_samples=%d, svd_fitted=%s",
            len(self._mcc_cols),
            len(self._merchant_cols),
            len(pdf),
            self._svd_model is not None,
        )
        return self

    def generate(self, df: Any, **context: Any) -> Any:
        """Generate merchant hierarchy features."""
        if not self._fitted:
            raise RuntimeError(
                "MerchantHierarchyGenerator must be fitted before generate()."
            )

        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        n_rows = len(pdf)
        cfg = self.config
        results: Dict[str, np.ndarray] = {}

        # --- MCC Level 1 Features (4D) ---
        l1_features = self._compute_mcc_level1(pdf, n_rows, cfg.mcc_level1_dim)
        for i in range(cfg.mcc_level1_dim):
            results[f"{self.prefix}_mcc_l1_dominant_{i}"] = l1_features[:, i]

        # --- MCC Level 2 Features (4D) ---
        l2_features = self._compute_mcc_level2(pdf, n_rows)
        results[f"{self.prefix}_mcc_l2_diversity"] = l2_features[:, 0]
        results[f"{self.prefix}_mcc_l2_concentration"] = l2_features[:, 1]
        results[f"{self.prefix}_mcc_l2_trend"] = l2_features[:, 2]
        results[f"{self.prefix}_mcc_l2_novelty"] = l2_features[:, 3]

        # --- Brand Embedding (8D) ---
        brand_features = self._compute_brand_embedding(pdf, n_rows, cfg.brand_embed_dim)
        for i in range(cfg.brand_embed_dim):
            results[f"{self.prefix}_brand_embed_{i}"] = brand_features[:, i]

        # --- Aggregate Stats (4D) ---
        agg_features = self._compute_aggregate_stats(pdf, n_rows)
        results[f"{self.prefix}_merchant_count"] = agg_features[:, 0]
        results[f"{self.prefix}_merchant_loyalty"] = agg_features[:, 1]
        results[f"{self.prefix}_merchant_recency"] = agg_features[:, 2]
        results[f"{self.prefix}_merchant_spend_gini"] = agg_features[:, 3]

        # --- Hierarchy Radius (1D) ---
        radius = self._compute_hierarchy_radius(pdf, n_rows)
        results[f"{self.prefix}_hierarchy_radius"] = radius

        return df_backend.from_dict(results, index=pdf.index)

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _find_mcc_columns(df: pd.DataFrame) -> List[str]:
        """Find columns related to merchant category codes."""
        keywords = ["mcc", "merchant", "category"]
        cols = []
        for c in df.columns:
            c_lower = c.lower()
            if any(kw in c_lower for kw in keywords):
                cols.append(c)
        return cols

    @staticmethod
    def _find_merchant_columns(df: pd.DataFrame) -> List[str]:
        """Find columns related to merchants (broader than MCC)."""
        keywords = ["merchant", "brand", "store", "vendor", "seller"]
        cols = []
        for c in df.columns:
            c_lower = c.lower()
            if any(kw in c_lower for kw in keywords):
                cols.append(c)
        return cols

    def _fit_svd(self, pdf: pd.DataFrame) -> None:
        """Fit TruncatedSVD on user-merchant interaction matrix."""
        merchant_cols = self._merchant_cols
        if not merchant_cols:
            # Fall back to numeric columns if no merchant columns found
            merchant_cols = self._numeric_cols

        if len(merchant_cols) < 2:
            self._svd_model = None
            return

        try:
            from sklearn.decomposition import TruncatedSVD

            X = pdf[merchant_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
            n_components = min(self.config.brand_embed_dim, X.shape[1] - 1, X.shape[0] - 1)
            if n_components < 1:
                self._svd_model = None
                return

            self._svd_model = TruncatedSVD(
                n_components=n_components, random_state=42
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._svd_model.fit(X)
        except Exception as exc:
            logger.warning("SVD fitting failed: %s. Brand embedding will use fallback.", exc)
            self._svd_model = None

    def _compute_mcc_level1(
        self, pdf: pd.DataFrame, n_rows: int, dim: int
    ) -> np.ndarray:
        """Compute MCC Level 1 features: softmax encoding of dominant L1 category.

        Groups by top-level category, computes spend proportions, and returns
        a softmax-like encoding over the top-``dim`` categories.
        """
        out = np.full((n_rows, dim), 1.0 / dim, dtype=np.float32)

        mcc_cols = self._mcc_cols
        if not mcc_cols:
            return out

        try:
            # Pick the first MCC-related column as the L1 category proxy
            col = mcc_cols[0]
            series = pdf[col]

            if series.dtype == object or hasattr(series, "cat"):
                # Categorical: compute value counts as proportions
                counts = series.fillna("unknown").astype(str).value_counts(normalize=True)
            else:
                # Numeric: bin into top-dim buckets
                numeric_vals = pd.to_numeric(series, errors="coerce").fillna(0)
                # Use integer division to create L1 buckets (e.g., MCC 5411 -> bucket 54)
                buckets = (numeric_vals // 100).astype(int)
                counts = buckets.value_counts(normalize=True)

            # Take top-dim categories
            top_cats = counts.head(dim)
            proportions = top_cats.values[:dim]

            # Softmax normalization
            if len(proportions) > 0:
                exp_p = np.exp(proportions - np.max(proportions))
                softmax_p = exp_p / (exp_p.sum() + 1e-10)
                # Broadcast per-dataset proportions to all rows
                for i in range(min(dim, len(softmax_p))):
                    out[:, i] = softmax_p[i]

        except Exception as exc:
            logger.debug("MCC L1 computation fell back to uniform: %s", exc)

        return out

    def _compute_mcc_level2(self, pdf: pd.DataFrame, n_rows: int) -> np.ndarray:
        """Compute MCC Level 2 features: diversity, concentration, trend, novelty."""
        out = np.zeros((n_rows, 4), dtype=np.float32)

        mcc_cols = self._mcc_cols
        if not mcc_cols:
            # Uniform synthetic fallback
            out[:, 0] = 0.5  # diversity
            out[:, 1] = 0.5  # concentration
            out[:, 2] = 0.0  # trend
            out[:, 3] = 0.5  # novelty
            return out

        try:
            # Use the best available MCC column
            col = mcc_cols[0]
            series = pd.to_numeric(pdf[col], errors="coerce").fillna(0)

            # Create L2 subcategories (e.g., MCC 5411 -> subcategory 54)
            l2_codes = (series // 10).astype(int)

            # --- Diversity: Shannon entropy of L2 distribution ---
            value_counts = l2_codes.value_counts(normalize=True).values
            value_counts = value_counts[value_counts > 0]
            if len(value_counts) > 1:
                entropy = -np.sum(value_counts * np.log(value_counts + 1e-10))
                max_entropy = np.log(len(value_counts))
                normalised_entropy = entropy / (max_entropy + 1e-10)
            else:
                normalised_entropy = 0.0
            out[:, 0] = np.float32(normalised_entropy)

            # --- Concentration: HHI (Herfindahl-Hirschman Index) ---
            hhi = np.sum(value_counts ** 2)
            out[:, 1] = np.float32(hhi)

            # --- Trend: slope of L2 usage over row index (time proxy) ---
            if n_rows > 1:
                x = np.arange(n_rows, dtype=np.float64)
                y = l2_codes.values.astype(np.float64)
                x_mean = x.mean()
                y_mean = y.mean()
                denom = np.sum((x - x_mean) ** 2)
                if denom > 1e-10:
                    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
                    # Normalise slope to [-1, 1] via tanh
                    out[:, 2] = np.float32(np.tanh(slope))

            # --- Novelty: ratio of new L2 categories in recent half vs first half ---
            if n_rows > 1:
                mid = n_rows // 2
                historical_cats = set(l2_codes.iloc[:mid].unique())
                recent_cats = set(l2_codes.iloc[mid:].unique())
                new_cats = recent_cats - historical_cats
                total_recent = len(recent_cats) if len(recent_cats) > 0 else 1
                novelty = len(new_cats) / total_recent
                out[:, 3] = np.float32(novelty)

        except Exception as exc:
            logger.debug("MCC L2 computation fell back to defaults: %s", exc)
            out[:, 0] = 0.5
            out[:, 1] = 0.5
            out[:, 3] = 0.5

        return out

    def _compute_brand_embedding(
        self, pdf: pd.DataFrame, n_rows: int, dim: int
    ) -> np.ndarray:
        """Compute SVD-based brand embedding from user-merchant interaction matrix.

        Applies the fitted TruncatedSVD model to extract latent merchant
        preference features that capture co-occurrence patterns.
        """
        out = np.full((n_rows, dim), 1.0 / dim, dtype=np.float32)

        if self._svd_model is None:
            return out

        merchant_cols = self._merchant_cols if self._merchant_cols else self._numeric_cols
        if len(merchant_cols) < 2:
            return out

        try:
            X = pdf[merchant_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
            n_components = self._svd_model.n_components

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                transformed = self._svd_model.transform(X)  # (n_rows, n_components)

            # Normalise each row to unit norm for stable embeddings
            # CuPy GPU path for norm computation on large matrices
            if has_cupy() and transformed.shape[0] > 1000:
                try:
                    import cupy as cp
                    t_gpu = cp.asarray(transformed)
                    norms_gpu = cp.linalg.norm(t_gpu, axis=1, keepdims=True)
                    norms_gpu = cp.where(norms_gpu < 1e-10, 1.0, norms_gpu)
                    transformed = cp.asnumpy(t_gpu / norms_gpu)
                except Exception:
                    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-10, 1.0, norms)
                    transformed = transformed / norms
            else:
                norms = np.linalg.norm(transformed, axis=1, keepdims=True)
                norms = np.where(norms < 1e-10, 1.0, norms)
                transformed = transformed / norms

            # Fill available components (pad with zeros if n_components < dim)
            fill_dim = min(n_components, dim)
            out[:, :fill_dim] = transformed[:, :fill_dim].astype(np.float32)

        except Exception as exc:
            logger.debug("Brand embedding computation fell back to uniform: %s", exc)

        return out

    def _compute_aggregate_stats(
        self, pdf: pd.DataFrame, n_rows: int
    ) -> np.ndarray:
        """Compute aggregate merchant statistics.

        Returns (n_rows, 4) array with:
          [0] merchant_count: log-scaled unique merchant count
          [1] merchant_loyalty: repeat visit ratio
          [2] merchant_recency: normalised days since last new merchant
          [3] merchant_spend_gini: Gini coefficient of spending across merchants
        """
        out = np.zeros((n_rows, 4), dtype=np.float32)

        merchant_cols = self._merchant_cols
        if not merchant_cols:
            # Synthetic uniform fallback
            out[:, 0] = 0.5
            out[:, 1] = 0.5
            out[:, 2] = 0.5
            out[:, 3] = 0.5
            return out

        try:
            col = merchant_cols[0]
            series = pdf[col]

            # --- Merchant count (log-scaled) ---
            n_unique = series.nunique()
            log_count = np.log1p(n_unique)
            # Normalise by log of total rows
            normalised_count = log_count / (np.log1p(n_rows) + 1e-10)
            out[:, 0] = np.float32(np.clip(normalised_count, 0, 1))

            # --- Merchant loyalty: top merchant visits / total ---
            value_counts = series.value_counts()
            if len(value_counts) > 0:
                top_count = value_counts.iloc[0]
                loyalty = top_count / (n_rows + 1e-10)
                out[:, 1] = np.float32(np.clip(loyalty, 0, 1))

            # --- Merchant recency: position of last new merchant / total ---
            if n_rows > 1:
                seen = set()
                last_new_pos = 0
                for idx_pos, val in enumerate(series.values):
                    if val not in seen:
                        seen.add(val)
                        last_new_pos = idx_pos
                recency = 1.0 - (last_new_pos / (n_rows - 1 + 1e-10))
                out[:, 2] = np.float32(np.clip(recency, 0, 1))

            # --- Merchant spend Gini coefficient ---
            if len(value_counts) > 1:
                counts_arr = np.sort(value_counts.values.astype(np.float64))
                n = len(counts_arr)
                index = np.arange(1, n + 1)
                gini = (2.0 * np.sum(index * counts_arr)) / (n * np.sum(counts_arr)) - (n + 1) / n
                out[:, 3] = np.float32(np.clip(gini, 0, 1))

        except Exception as exc:
            logger.debug("Aggregate stats computation fell back to defaults: %s", exc)
            out[:, 0] = 0.5
            out[:, 1] = 0.5
            out[:, 2] = 0.5
            out[:, 3] = 0.5

        return out

    def _compute_hierarchy_radius(
        self, pdf: pd.DataFrame, n_rows: int
    ) -> np.ndarray:
        """Compute hierarchy radius: normalised std of merchant category codes.

        Measures the geographic/categorical spread of merchant visits.
        """
        out = np.full(n_rows, 0.5, dtype=np.float32)

        mcc_cols = self._mcc_cols
        if not mcc_cols:
            return out

        try:
            col = mcc_cols[0]
            values = pd.to_numeric(pdf[col], errors="coerce").fillna(0).values.astype(np.float64)

            if len(values) > 1:
                std = np.std(values)
                mean = np.abs(np.mean(values)) + 1e-10
                # Coefficient of variation, clipped and normalised
                cv = std / mean
                normalised = np.float32(np.clip(np.tanh(cv), 0, 1))
                out[:] = normalised

        except Exception as exc:
            logger.debug("Hierarchy radius computation fell back to default: %s", exc)

        return out
