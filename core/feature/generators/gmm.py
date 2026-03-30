"""
GMM (Gaussian Mixture Model) Clustering Feature Generator.

Fits a Gaussian Mixture Model to produce soft cluster assignments and
associated uncertainty features.

Output per row:
  - cluster_id (1D): most-likely cluster assignment (argmax)
  - cluster_probs (K D): posterior probability for each of K clusters
  - entropy (1D): Shannon entropy of the cluster probability vector

Total output: K + 2 dimensions (default K=20 -> 22D).

BIC-based validation logs a warning if the configured K is far from the
BIC-optimal K.  A cold-start fallback returns uniform distributions when
the training data is too small for reliable GMM fitting.

Hardware acceleration
---------------------
When cuML is available (RAPIDS), the GPU-accelerated GaussianMixture is
used for ~10x speedup on datasets with >100k rows.  Falls back to
scikit-learn's CPU implementation otherwise.
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
from .gpu_utils import has_cuml, has_cudf

# ---------------------------------------------------------------------------
# Lazy import cuDF (optional GPU acceleration)
# ---------------------------------------------------------------------------
try:
    import cudf as _cudf
except ImportError:
    _cudf = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import: cuML GPU GMM -> sklearn CPU GMM fallback
# ---------------------------------------------------------------------------
_GMM_CLASS = None
_GMM_BACKEND: Optional[str] = None


def _resolve_gmm_backend():
    """Resolve the best available GMM implementation.

    Fallback chain:
      1. cuML GaussianMixture — GPU-accelerated, ~10x faster for n > 100k
      2. sklearn GaussianMixture — CPU, standard implementation
    """
    global _GMM_CLASS, _GMM_BACKEND

    if _GMM_BACKEND is not None:
        return _GMM_CLASS, _GMM_BACKEND

    # 1. cuML GPU GMM: ~10x faster for large datasets (>100k rows)
    # The GPU parallelisation of EM iterations provides significant speedup
    # when the data is large enough to amortise the CPU-GPU transfer cost.
    try:
        from cuml.cluster import GaussianMixture as cuGMM  # type: ignore[import-untyped]
        _GMM_CLASS = cuGMM
        _GMM_BACKEND = "cuml"
        logger.info("GMM backend: cuML (GPU-accelerated GaussianMixture)")
        return _GMM_CLASS, _GMM_BACKEND
    except (ImportError, Exception):
        pass

    # 2. sklearn CPU GMM
    try:
        from sklearn.mixture import GaussianMixture  # type: ignore[import-untyped]
        _GMM_CLASS = GaussianMixture
        _GMM_BACKEND = "sklearn"
        logger.info("GMM backend: scikit-learn (CPU GaussianMixture)")
        return _GMM_CLASS, _GMM_BACKEND
    except ImportError:
        _GMM_CLASS = None
        _GMM_BACKEND = "none"
        logger.debug("No GMM backend available (install scikit-learn or cuml).")
        return _GMM_CLASS, _GMM_BACKEND


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GMMConfig:
    """GMM hyper-parameters."""

    n_clusters: int = 20
    covariance_type: str = "full"
    max_iter: int = 200
    n_init: int = 3
    random_state: int = 42
    bic_check: bool = True
    bic_k_range: int = 5  # check K +/- this range for BIC
    min_samples_per_cluster: int = 5  # cold-start threshold


# ---------------------------------------------------------------------------
# GMM Feature Generator
# ---------------------------------------------------------------------------

@FeatureGeneratorRegistry.register(
    "gmm",
    description="Gaussian Mixture Model soft clustering with BIC validation.",
    tags=["gmm", "clustering", "probabilistic"],
)
class GMMClusteringGenerator(AbstractFeatureGenerator):
    """Gaussian Mixture Model clustering feature generator.

    Produces soft cluster assignments by fitting a GaussianMixture model
    on the numeric columns of the input DataFrame.

    When cuML (RAPIDS) is available, uses GPU-accelerated GMM for ~10x
    speedup on large datasets.  Falls back to scikit-learn otherwise.

    Output columns (for prefix ``gmm`` and K clusters):
      - ``gmm_cluster_id``: argmax cluster (int)
      - ``gmm_cluster_prob_0`` ... ``gmm_cluster_prob_{K-1}``: posterior probs
      - ``gmm_entropy``: Shannon entropy of the probability vector

    Parameters
    ----------
    config : GMMConfig, optional
        GMM hyper-parameters.
    feature_columns : list[str], optional
        Which columns to cluster on.  Defaults to all numeric.
    prefix : str
        Column-name prefix.
    """

    supports_gpu: bool = True
    required_libraries: List[str] = ["sklearn"]
    optional_libraries: List[str] = ["cuml"]

    def __init__(
        self,
        config: Optional[GMMConfig] = None,
        feature_columns: Optional[List[str]] = None,
        prefix: str = "gmm",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config or GMMConfig()
        self.feature_columns = feature_columns or []
        self.prefix = prefix

        # Fitted state
        self._model: Any = None
        self._col_means: Optional[np.ndarray] = None
        self._col_stds: Optional[np.ndarray] = None
        self._is_coldstart: bool = False

        # Eagerly resolve the GMM backend so it is cached
        _resolve_gmm_backend()

    # -- Output description ------------------------------------------------

    @property
    def output_dim(self) -> int:
        # cluster_id + K probs + entropy
        return 1 + self.config.n_clusters + 1

    @property
    def output_columns(self) -> List[str]:
        cols = [f"{self.prefix}_cluster_id"]
        for k in range(self.config.n_clusters):
            cols.append(f"{self.prefix}_cluster_prob_{k}")
        cols.append(f"{self.prefix}_entropy")
        return cols

    # -- Core API ----------------------------------------------------------

    def fit(self, df: Any, **context: Any) -> "GMMClusteringGenerator":
        """Fit the GMM on numeric features.

        If the data has fewer samples than ``min_samples_per_cluster * K``,
        the generator enters cold-start mode and will produce uniform
        distributions at inference time.
        """
        gmm_cls, backend = _resolve_gmm_backend()

        if gmm_cls is None:
            raise ImportError(
                "A GMM backend is required for GMMClusteringGenerator. "
                "Install with: pip install scikit-learn  (or pip install cuml for GPU)"
            )

        X, _ = self._prepare_features(df, fit=True)

        K = self.config.n_clusters
        min_needed = self.config.min_samples_per_cluster * K

        if X.shape[0] < min_needed:
            logger.warning(
                "Insufficient data for GMM (%d rows < %d needed). "
                "Entering cold-start mode with uniform assignments.",
                X.shape[0],
                min_needed,
            )
            self._is_coldstart = True
            self._model = None
            self._fitted = True
            return self

        self._is_coldstart = False

        # Fit main model using the resolved backend (cuML or sklearn)
        if backend == "cuml":
            # cuML GaussianMixture has a slightly different API
            self._model = gmm_cls(
                n_components=K,
                covariance_type=self.config.covariance_type,
                max_iter=self.config.max_iter,
                n_init=self.config.n_init,
                random_state=self.config.random_state,
            )
        else:
            self._model = gmm_cls(
                n_components=K,
                covariance_type=self.config.covariance_type,
                max_iter=self.config.max_iter,
                n_init=self.config.n_init,
                random_state=self.config.random_state,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X)

        # Check convergence (sklearn exposes converged_; cuML may not)
        converged = getattr(self._model, "converged_", "N/A")
        logger.info(
            "GMMClusteringGenerator fitted: K=%d, n_samples=%d, n_features=%d, "
            "converged=%s, backend=%s",
            K, X.shape[0], X.shape[1], converged, backend,
        )

        # BIC validation (sklearn only; cuML may not support bic())
        if self.config.bic_check and backend == "sklearn":
            self._run_bic_check(X, K)

        self._fitted = True
        return self

    def generate(self, df: Any, **context: Any) -> Any:
        """Generate GMM cluster features."""
        if not self._fitted:
            raise RuntimeError(
                "GMMClusteringGenerator must be fitted before generate()."
            )

        n_rows = len(df)
        K = self.config.n_clusters
        results: Dict[str, np.ndarray] = {}

        if self._is_coldstart or self._model is None:
            # Uniform fallback
            results[f"{self.prefix}_cluster_id"] = np.zeros(n_rows, dtype=np.int32)
            uniform = 1.0 / K
            for k in range(K):
                results[f"{self.prefix}_cluster_prob_{k}"] = np.full(
                    n_rows, uniform, dtype=np.float32
                )
            results[f"{self.prefix}_entropy"] = np.full(
                n_rows, float(np.log(K)), dtype=np.float32
            )
        else:
            X, _ = self._prepare_features(df, fit=False)

            # Predict
            probs = self._model.predict_proba(X)  # (N, K)

            # cuML may return cupy arrays; ensure numpy
            if hasattr(probs, 'get'):
                probs = probs.get()  # cupy -> numpy
            probs = np.asarray(probs)

            cluster_ids = probs.argmax(axis=1)

            results[f"{self.prefix}_cluster_id"] = cluster_ids.astype(np.int32)
            for k in range(K):
                results[f"{self.prefix}_cluster_prob_{k}"] = probs[:, k].astype(np.float32)

            # Shannon entropy
            entropy = -np.sum(
                probs * np.log(probs + 1e-300), axis=1
            ).astype(np.float32)
            results[f"{self.prefix}_entropy"] = entropy

        # Build output DataFrame via cuDF when available
        if has_cudf() and _cudf is not None:
            return _cudf.DataFrame(results)
        return pd.DataFrame(results)

    # -- Helpers -----------------------------------------------------------

    def _prepare_features(
        self, df: Any, fit: bool = False
    ) -> tuple:
        """Extract and normalise feature matrix.

        Uses cuDF/CuPy GPU path when available, falls back to pandas/numpy.
        Returns (X, col_names) where X is always a numpy float64 array.
        """
        cols = self._resolve_columns(df)
        X = self._extract_numeric(df, cols)

        # Replace NaN with column means
        col_means = np.nanmean(X, axis=0)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            for j in range(X.shape[1]):
                X[nan_mask[:, j], j] = col_means[j] if np.isfinite(col_means[j]) else 0.0

        if fit:
            self._col_means = X.mean(axis=0)
            self._col_stds = X.std(axis=0)
            self._col_stds[self._col_stds < 1e-10] = 1.0

        # Standardise
        if self._col_means is not None and self._col_stds is not None:
            X = (X - self._col_means) / self._col_stds

        return X, cols

    @staticmethod
    def _extract_numeric(df: Any, cols: List[str]) -> np.ndarray:
        """Extract columns as a numpy float64 array from any DataFrame type."""
        from .gpu_utils import _to_numpy_safe
        return _to_numpy_safe(df, cols, fill=0.0)

    def _resolve_columns(self, df: Any) -> List[str]:
        """Resolve feature columns, falling back to all numeric."""
        if self.feature_columns:
            cols = list(df.columns) if hasattr(df, 'columns') else []
            return [c for c in self.feature_columns if c in cols]
        # cuDF and pandas both support select_dtypes
        if hasattr(df, 'select_dtypes'):
            cols = df.select_dtypes(include=["number"]).columns.tolist()
            return cols if cols else []
        pdf = df_backend.to_pandas(df)
        cols = pdf.select_dtypes(include=["number"]).columns.tolist()
        return cols if cols else []

    def _run_bic_check(self, X: np.ndarray, configured_k: int) -> None:
        """Fit GMMs at neighbouring K values and warn if BIC suggests a better K."""
        gmm_cls, _ = _resolve_gmm_backend()
        if gmm_cls is None:
            return

        lo = max(2, configured_k - self.config.bic_k_range)
        hi = configured_k + self.config.bic_k_range + 1
        best_bic = self._model.bic(X)
        best_k = configured_k

        for k in range(lo, hi):
            if k == configured_k:
                continue
            try:
                m = gmm_cls(
                    n_components=k,
                    covariance_type=self.config.covariance_type,
                    max_iter=self.config.max_iter,
                    n_init=1,
                    random_state=self.config.random_state,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(X)
                bic_val = m.bic(X)
                if bic_val < best_bic:
                    best_bic = bic_val
                    best_k = k
            except Exception:
                continue

        if best_k != configured_k:
            logger.warning(
                "BIC suggests K=%d (BIC=%.1f) is better than configured K=%d "
                "(BIC=%.1f). Consider updating GMMConfig.n_clusters.",
                best_k,
                best_bic,
                configured_k,
                self._model.bic(X),
            )
        else:
            logger.info(
                "BIC check passed: configured K=%d has best BIC among "
                "[%d, %d).",
                configured_k,
                lo,
                hi,
            )
