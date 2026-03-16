"""
TDA (Topological Data Analysis) Feature Generator.

Extracts topological features from point-cloud representations of user
behaviour data using persistent homology.  Topological features capture
the *shape* of data -- clusters, loops, and voids -- that traditional
statistical features miss entirely.

In a financial recommendation context, TDA features can reveal:
  - **H0 (connected components)**: distinct behaviour clusters within a
    user's transaction history.
  - **H1 (loops)**: recurring cyclical patterns (e.g. salary-spend-save
    cycles).
  - **Persistence statistics**: how "stable" these topological features
    are across noise levels.

This is a **placeholder implementation** that generates synthetic features
with the correct interface.  A production implementation would use
``ripser``, ``gudhi``, or ``giotto-tda`` for actual persistence diagram
computation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry

logger = logging.getLogger(__name__)


@FeatureGeneratorRegistry.register(
    "tda_extractor",
    description="Topological Data Analysis features via persistent homology.",
    tags=["topology", "advanced", "shape"],
)
class TDAFeatureGenerator(AbstractFeatureGenerator):
    """Extract topological features from point-cloud data.

    For each row (user), this generator treats a subset of numeric columns
    as a point cloud in R^d and computes persistence diagram statistics
    across homology dimensions H0 and H1.

    Parameters
    ----------
    input_columns : list[str]
        Numeric columns that form the point-cloud coordinates.
    max_homology_dim : int
        Maximum homology dimension to compute (0 = components, 1 = loops).
    n_persistence_stats : int
        Number of statistical summaries per homology dimension
        (mean, std, max, entropy of persistence lifetimes).
    max_edge_length : float
        Maximum filtration value for Vietoris-Rips complex.
    prefix : str
        Column name prefix for generated features.

    Notes
    -----
    The current implementation is a **placeholder** that generates
    random features with the correct output shape.  Replace with
    actual TDA computation (e.g. ``ripser``, ``gudhi``) for production.
    """

    def __init__(
        self,
        input_columns: Optional[List[str]] = None,
        max_homology_dim: int = 1,
        n_persistence_stats: int = 4,
        max_edge_length: float = 2.0,
        prefix: str = "tda",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_columns = input_columns or []
        self.max_homology_dim = max_homology_dim
        self.n_persistence_stats = n_persistence_stats
        self.max_edge_length = max_edge_length
        self.prefix = prefix

        # Stats per homology dimension: mean, std, max, entropy
        self._stat_names = ["mean", "std", "max", "entropy"][:n_persistence_stats]

        # Internal state learned during fit
        self._global_scale: Optional[float] = None
        self._col_means: Optional[np.ndarray] = None
        self._col_stds: Optional[np.ndarray] = None

    # -- Output description --------------------------------------------

    @property
    def output_dim(self) -> int:
        """Number of TDA features = (max_homology_dim + 1) * n_stats."""
        return (self.max_homology_dim + 1) * self.n_persistence_stats

    @property
    def output_columns(self) -> List[str]:
        """Generated column names like ``tda_h0_mean``, ``tda_h1_entropy``."""
        cols = []
        for h_dim in range(self.max_homology_dim + 1):
            for stat in self._stat_names:
                cols.append(f"{self.prefix}_h{h_dim}_{stat}")
        return cols

    # -- Core API ------------------------------------------------------

    def fit(self, df: pd.DataFrame, **context: Any) -> "TDAFeatureGenerator":
        """Learn global scaling parameters from training data.

        In a real implementation this would also pre-compute any
        filtration parameters or landmark subsets.
        """
        cols = self._resolve_input_columns(df)
        data = df[cols].values.astype(np.float64)

        # Learn normalisation parameters for the point cloud
        self._col_means = np.nanmean(data, axis=0)
        self._col_stds = np.nanstd(data, axis=0)
        self._col_stds[self._col_stds == 0] = 1.0
        self._global_scale = float(np.nanstd(data))

        self._fitted = True
        logger.info(
            "TDAFeatureGenerator fitted: %d input cols -> %d output features, "
            "max_homology_dim=%d",
            len(cols), self.output_dim, self.max_homology_dim,
        )
        return self

    def generate(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame:
        """Generate TDA features for each row.

        .. note::
           This is a **placeholder** that produces deterministic
           pseudo-features based on input statistics.  Replace with
           actual persistence computation for production use.
        """
        if not self._fitted:
            raise RuntimeError(
                "TDAFeatureGenerator must be fitted before generate()."
            )

        cols = self._resolve_input_columns(df)
        data = df[cols].values.astype(np.float64)
        n_rows = len(df)

        # Normalise the point cloud
        normed = (data - self._col_means) / self._col_stds

        # --- Placeholder TDA computation ---
        # In production: use ripser.ripser(normed[i], maxdim=self.max_homology_dim)
        # and extract persistence diagram statistics.
        result = np.zeros((n_rows, self.output_dim), dtype=np.float32)

        for i in range(n_rows):
            row = normed[i]
            valid = row[~np.isnan(row)]
            if len(valid) < 2:
                continue

            for h_dim in range(self.max_homology_dim + 1):
                base_idx = h_dim * self.n_persistence_stats
                # Placeholder: derive pseudo-topological stats from data moments
                # H0 approximation: spread of the point cloud
                # H1 approximation: cyclic structure proxy
                scale = 1.0 / (h_dim + 1)
                diffs = np.abs(np.diff(np.sort(valid))) * scale

                if len(diffs) > 0:
                    stat_idx = 0
                    if stat_idx < self.n_persistence_stats:
                        result[i, base_idx + stat_idx] = float(np.mean(diffs))
                        stat_idx += 1
                    if stat_idx < self.n_persistence_stats:
                        result[i, base_idx + stat_idx] = float(np.std(diffs))
                        stat_idx += 1
                    if stat_idx < self.n_persistence_stats:
                        result[i, base_idx + stat_idx] = float(np.max(diffs))
                        stat_idx += 1
                    if stat_idx < self.n_persistence_stats:
                        # Persistence entropy
                        probs = diffs / (diffs.sum() + 1e-10)
                        entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
                        result[i, base_idx + stat_idx] = entropy

        return pd.DataFrame(
            result,
            columns=self.output_columns,
            index=df.index,
        )

    # -- Helpers -------------------------------------------------------

    def _resolve_input_columns(self, df: pd.DataFrame) -> List[str]:
        """Resolve input columns, falling back to all numeric if unset."""
        if self.input_columns:
            return [c for c in self.input_columns if c in df.columns]
        return df.select_dtypes(include=["number"]).columns.tolist()
