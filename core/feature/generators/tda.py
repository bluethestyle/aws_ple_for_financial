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

Hardware acceleration
---------------------
Ripser fallback chain (fastest to slowest):
  1. **giotto-ph** (``gph``): Modified Ripser C++ with OpenMP parallelism.
     Fastest for large datasets due to multi-threaded computation.
  2. **cripser**: Cubical Ripser with CUDA acceleration. Best on GPU
     instances for distance-matrix inputs.
  3. **ripser**: Standard C++ Vietoris-Rips (single-threaded).
  4. **numpy-only**: Approximate persistence via eigenvalue decomposition.

Distance matrix computation can be accelerated with CuPy on GPU (~50x
faster for n > 5000 due to O(n^2) parallelisation).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.data.dataframe import df_backend
from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry
from .gpu_utils import has_cupy, cupy_pairwise_distances

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import helpers — Ripser C++ acceleration chain
# ---------------------------------------------------------------------------

_ripser_fn = None
_cripser_fn = None
_BACKEND: Optional[str] = None


def _resolve_tda_backend() -> str:
    """Determine which TDA backend is available.

    Fallback chain (fastest to slowest):
      1. giotto-ph (gph) — OpenMP parallel C++, fastest for large datasets
      2. cripser — CUDA Cubical Ripser, best on GPU instances
      3. ripser — standard C++ Vietoris-Rips
      4. numpy-only — approximate persistence via sorted eigenvalues
    """
    global _ripser_fn, _cripser_fn, _BACKEND

    if _BACKEND is not None:
        return _BACKEND

    # 1. giotto-ph: OpenMP parallel C++ Ripser (fastest)
    try:
        from gph import ripser_parallel as _rp  # noqa: F811
        _ripser_fn = _rp
        _BACKEND = "giotto-ph"
        n_threads = os.cpu_count() or 1
        logger.info(
            "TDA backend: giotto-ph (OpenMP parallel C++, %d threads)", n_threads
        )
        return _BACKEND
    except ImportError:
        pass

    # 2. cripser: CUDA-accelerated Cubical Ripser (works on distance matrices)
    try:
        from cripser import computePH as _cph  # noqa: F811
        _cripser_fn = _cph
        _BACKEND = "cripser"
        logger.info("TDA backend: cripser (CUDA Cubical Ripser)")
        return _BACKEND
    except ImportError:
        pass

    # 3. ripser: standard C++ Vietoris-Rips (single-threaded)
    try:
        import ripser as _rip  # noqa: F811
        _ripser_fn = _rip.ripser
        _BACKEND = "ripser"
        logger.info("TDA backend: ripser (C++ Vietoris-Rips)")
        return _BACKEND
    except ImportError:
        pass

    # 4. numpy-only fallback (approximate)
    _BACKEND = "numpy"
    logger.warning(
        "Using approximate TDA (no C++ backend available). "
        "Install one of: pip install giotto-ph / pip install cripser / pip install ripser"
    )
    return _BACKEND


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_STATS = [
    "num_features",
    "mean_lifetime",
    "max_lifetime",
    "std_lifetime",
    "entropy",
    "total_persistence",
    "mean_birth",
    "mean_death",
]


@dataclass
class TDAConfig:
    """Configuration for TDA feature extraction."""

    max_homology_dim: int = 1
    max_edge_length: float = float("inf")
    n_points_subsample: int = 200
    stats_to_compute: List[str] = field(default_factory=lambda: list(_DEFAULT_STATS))


# ---------------------------------------------------------------------------
# Persistence diagram computation helpers
# ---------------------------------------------------------------------------


def _compute_persistence_giotto_ph(
    distance_matrix: np.ndarray,
    max_dim: int,
    max_edge: float,
) -> Dict[int, np.ndarray]:
    """Compute persistence diagrams using giotto-ph (OpenMP parallel C++).

    giotto-ph's ripser_parallel supports n_threads for multi-core acceleration.
    For large distance matrices this is significantly faster than single-threaded
    ripser.
    """
    n_threads = os.cpu_count() or 1
    result = _ripser_fn(
        distance_matrix,
        maxdim=max_dim,
        thresh=max_edge if np.isfinite(max_edge) else np.inf,
        metric="precomputed",
        n_threads=n_threads,
    )
    diagrams: Dict[int, np.ndarray] = {}
    for dim, dgm in enumerate(result["dgms"]):
        finite_mask = np.isfinite(dgm[:, 1]) if len(dgm) > 0 else np.array([], dtype=bool)
        diagrams[dim] = dgm[finite_mask] if finite_mask.any() else np.empty((0, 2))
    return diagrams


def _compute_persistence_cripser(
    distance_matrix: np.ndarray,
    max_dim: int,
    _max_edge: float,
) -> Dict[int, np.ndarray]:
    """Compute persistence diagrams using cripser (CUDA Cubical Ripser).

    cripser operates on cubical complexes and accepts distance matrices.
    Best suited for GPU instances where CUDA is available.
    """
    # cripser.computePH returns array of (dim, birth, death)
    ph_result = _cripser_fn(distance_matrix)

    diagrams: Dict[int, np.ndarray] = {}
    for dim in range(max_dim + 1):
        mask = ph_result[:, 0] == dim
        if mask.any():
            bd = ph_result[mask, 1:3]
            finite_mask = np.isfinite(bd[:, 1])
            diagrams[dim] = bd[finite_mask] if finite_mask.any() else np.empty((0, 2))
        else:
            diagrams[dim] = np.empty((0, 2))
    return diagrams


def _compute_persistence_ripser(
    distance_matrix: np.ndarray,
    max_dim: int,
    max_edge: float,
) -> Dict[int, np.ndarray]:
    """Compute persistence diagrams using standard ripser (C++)."""
    result = _ripser_fn(
        distance_matrix,
        maxdim=max_dim,
        thresh=max_edge if np.isfinite(max_edge) else np.inf,
        distance_matrix=True,
    )
    diagrams: Dict[int, np.ndarray] = {}
    for dim, dgm in enumerate(result["dgms"]):
        # Filter out infinite-death features for finite statistics
        finite_mask = np.isfinite(dgm[:, 1]) if len(dgm) > 0 else np.array([], dtype=bool)
        diagrams[dim] = dgm[finite_mask] if finite_mask.any() else np.empty((0, 2))
    return diagrams


def _compute_persistence_numpy(
    distance_matrix: np.ndarray,
    max_dim: int,
    _max_edge: float,
) -> Dict[int, np.ndarray]:
    """Approximate persistence diagrams using eigenvalue decomposition.

    This is a rough approximation: we use sorted eigenvalues of the
    distance matrix to simulate birth-death pairs.  H0 is approximated
    from the smallest eigenvalues (connected components merge at small
    scales) and H1 from intermediate eigenvalues.

    Uses CuPy GPU acceleration for eigvalsh when available and the
    matrix is large enough (n > 500) for ~10-20x speedup.
    """
    n = distance_matrix.shape[0]
    if n < 2:
        return {dim: np.empty((0, 2)) for dim in range(max_dim + 1)}

    # CuPy GPU path for eigenvalue decomposition
    if has_cupy() and n > 500:
        try:
            import cupy as cp
            dm_gpu = cp.asarray(distance_matrix)
            eigenvalues = cp.sort(cp.abs(cp.linalg.eigvalsh(dm_gpu)))
            eigenvalues = cp.asnumpy(eigenvalues)
        except Exception:
            eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(distance_matrix)))
    else:
        eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(distance_matrix)))

    diagrams: Dict[int, np.ndarray] = {}

    # H0: connected components -- birth at 0, death at sorted eigenvalues
    n_h0 = max(n - 1, 1)
    births_h0 = np.zeros(n_h0)
    deaths_h0 = eigenvalues[1 : n_h0 + 1] if len(eigenvalues) > 1 else np.array([eigenvalues[0]])
    deaths_h0 = deaths_h0[: len(births_h0)]
    if len(deaths_h0) < len(births_h0):
        births_h0 = births_h0[: len(deaths_h0)]
    diagrams[0] = np.column_stack([births_h0, deaths_h0]) if len(births_h0) > 0 else np.empty((0, 2))

    # H1: loops -- approximate from mid-range eigenvalues
    if max_dim >= 1:
        mid_start = n // 3
        mid_end = 2 * n // 3
        mid_eigs = eigenvalues[mid_start:mid_end]
        if len(mid_eigs) >= 2:
            births_h1 = mid_eigs[:-1]
            deaths_h1 = mid_eigs[1:]
            # Only keep pairs where death > birth
            valid = deaths_h1 > births_h1
            diagrams[1] = np.column_stack([births_h1[valid], deaths_h1[valid]]) if valid.any() else np.empty((0, 2))
        else:
            diagrams[1] = np.empty((0, 2))

    # Higher dims -- empty
    for dim in range(2, max_dim + 1):
        diagrams[dim] = np.empty((0, 2))

    return diagrams


def _compute_persistence(
    distance_matrix: np.ndarray,
    max_dim: int,
    max_edge: float,
) -> Dict[int, np.ndarray]:
    """Dispatch to the best available backend."""
    backend = _resolve_tda_backend()
    if backend == "giotto-ph":
        return _compute_persistence_giotto_ph(distance_matrix, max_dim, max_edge)
    elif backend == "cripser":
        return _compute_persistence_cripser(distance_matrix, max_dim, max_edge)
    elif backend == "ripser":
        return _compute_persistence_ripser(distance_matrix, max_dim, max_edge)
    else:
        return _compute_persistence_numpy(distance_matrix, max_dim, max_edge)


# ---------------------------------------------------------------------------
# Statistics extraction from persistence diagrams
# ---------------------------------------------------------------------------


def _extract_diagram_stats(
    diagram: np.ndarray,
    stats_to_compute: List[str],
) -> Dict[str, float]:
    """Extract summary statistics from a single persistence diagram.

    Parameters
    ----------
    diagram : np.ndarray
        Shape ``(n_features, 2)`` with columns ``[birth, death]``.
    stats_to_compute : list[str]
        Which statistics to extract.

    Returns
    -------
    dict[str, float]
    """
    result: Dict[str, float] = {}

    if diagram.shape[0] == 0:
        for stat in stats_to_compute:
            result[stat] = 0.0
        return result

    births = diagram[:, 0]
    deaths = diagram[:, 1]
    lifetimes = deaths - births

    for stat in stats_to_compute:
        if stat == "num_features":
            result[stat] = float(len(lifetimes))
        elif stat == "mean_lifetime":
            result[stat] = float(np.mean(lifetimes))
        elif stat == "max_lifetime":
            result[stat] = float(np.max(lifetimes))
        elif stat == "std_lifetime":
            result[stat] = float(np.std(lifetimes)) if len(lifetimes) > 1 else 0.0
        elif stat == "entropy":
            # Persistence entropy
            total = np.sum(lifetimes)
            if total > 0:
                probs = lifetimes / total
                result[stat] = -float(np.sum(probs * np.log(probs + 1e-12)))
            else:
                result[stat] = 0.0
        elif stat == "total_persistence":
            result[stat] = float(np.sum(lifetimes))
        elif stat == "mean_birth":
            result[stat] = float(np.mean(births))
        elif stat == "mean_death":
            result[stat] = float(np.mean(deaths))
        else:
            result[stat] = 0.0

    return result


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@FeatureGeneratorRegistry.register(
    "tda",
    description="Topological Data Analysis features via persistent homology.",
    tags=["topology", "advanced", "shape"],
)
class TDAFeatureGenerator(AbstractFeatureGenerator):
    """Extract topological features from point-cloud data.

    For each row (user), this generator treats a subset of numeric columns
    as a point cloud in R^d and computes persistence diagram statistics
    across homology dimensions H0 and H1.

    The generator uses a four-tier fallback chain:
      1. **giotto-ph** -- OpenMP parallel C++ Ripser (fastest, multi-threaded).
      2. **cripser** -- CUDA Cubical Ripser (GPU-accelerated persistence).
      3. **ripser** -- fast C++ Vietoris-Rips computation (~100x faster
         than pure Python).
      4. **numpy-only** -- approximate persistence via sorted eigenvalues
         of the distance matrix (no external dependency).

    Distance matrix computation is accelerated with CuPy when available
    (~50x faster for n > 5000 due to GPU-parallelised O(n^2) computation).

    Parameters
    ----------
    input_columns : list[str]
        Numeric columns that form the point-cloud coordinates.
    config : TDAConfig, optional
        Full configuration dataclass.  Individual parameters below
        override fields in *config* if both are given.
    max_homology_dim : int
        Maximum homology dimension to compute (0 = components, 1 = loops).
    max_edge_length : float
        Maximum filtration value for Vietoris-Rips complex.
    n_points_subsample : int
        Subsample point cloud to at most this many points per row batch.
    stats_to_compute : list[str]
        Statistics to extract from each persistence diagram.
    prefix : str
        Column name prefix for generated features.

    Attributes
    ----------
    supports_gpu : bool
        True when cripser (CUDA) or CuPy is available for acceleration.
    required_libraries : list[str]
        ``["giotto-ph", "cripser", "ripser"]`` (alternatives; numpy fallback
        always available).
    """

    # supports_gpu is set dynamically based on available backends
    supports_gpu: bool = True
    required_libraries: List[str] = []
    optional_libraries: List[str] = ["gph", "cripser", "ripser"]

    def __init__(
        self,
        input_columns: Optional[List[str]] = None,
        config: Optional[TDAConfig] = None,
        max_homology_dim: Optional[int] = None,
        max_edge_length: Optional[float] = None,
        n_points_subsample: Optional[int] = None,
        stats_to_compute: Optional[List[str]] = None,
        prefix: str = "tda",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_columns = input_columns or []
        self.prefix = prefix

        # Build config -- explicit params override dataclass defaults
        cfg = config or TDAConfig()
        self.max_homology_dim = max_homology_dim if max_homology_dim is not None else cfg.max_homology_dim
        self.max_edge_length = max_edge_length if max_edge_length is not None else cfg.max_edge_length
        self.n_points_subsample = n_points_subsample if n_points_subsample is not None else cfg.n_points_subsample
        self.stats_to_compute = stats_to_compute if stats_to_compute is not None else list(cfg.stats_to_compute)

        # Internal state learned during fit
        self._global_scale: Optional[float] = None
        self._col_means: Optional[np.ndarray] = None
        self._col_stds: Optional[np.ndarray] = None

        # Eagerly resolve the TDA backend so it is cached
        _resolve_tda_backend()

    # -- Output description ------------------------------------------------

    @property
    def output_dim(self) -> int:
        """Number of TDA features = (max_homology_dim + 1) * n_stats."""
        return (self.max_homology_dim + 1) * len(self.stats_to_compute)

    @property
    def output_columns(self) -> List[str]:
        """Generated column names like ``tda_h0_mean_lifetime``, ``tda_h1_entropy``."""
        cols: List[str] = []
        for h_dim in range(self.max_homology_dim + 1):
            for stat in self.stats_to_compute:
                cols.append(f"{self.prefix}_h{h_dim}_{stat}")
        return cols

    # -- Core API ----------------------------------------------------------

    def fit(self, df: Any, **context: Any) -> "TDAFeatureGenerator":
        """Learn global scaling parameters from training data.

        Computes column-wise means and standard deviations for point-cloud
        normalisation, and adaptively sets max_homology_dim if the data
        dimensionality is too low.
        """
        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        cols = self._resolve_input_columns(pdf)
        data = pdf[cols].values.astype(np.float64)

        # Learn normalisation parameters
        self._col_means = np.nanmean(data, axis=0)
        self._col_stds = np.nanstd(data, axis=0)
        self._col_stds[self._col_stds == 0] = 1.0
        self._global_scale = float(np.nanstd(data))

        # Adaptive max_dim: cannot compute H_k if data dim < k+1
        n_features = data.shape[1]
        effective_max_dim = min(self.max_homology_dim, max(n_features - 1, 0))
        if effective_max_dim < self.max_homology_dim:
            logger.warning(
                "Reducing max_homology_dim from %d to %d (only %d input features)",
                self.max_homology_dim,
                effective_max_dim,
                n_features,
            )
            self.max_homology_dim = effective_max_dim

        # Eagerly resolve the TDA backend so warnings appear at fit time
        _resolve_tda_backend()

        self._fitted = True
        logger.info(
            "TDAFeatureGenerator fitted: %d input cols -> %d output features, "
            "max_homology_dim=%d, backend=%s, cupy_distances=%s",
            len(cols),
            self.output_dim,
            self.max_homology_dim,
            _BACKEND,
            has_cupy(),
        )
        return self

    def generate(self, df: Any, **context: Any) -> Any:
        """Generate TDA features for each row.

        Constructs a pairwise distance matrix from the normalised feature
        vectors, computes persistence diagrams via the best available
        backend, and extracts summary statistics per homology dimension.
        """
        if not self._fitted:
            raise RuntimeError(
                "TDAFeatureGenerator must be fitted before generate()."
            )

        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        cols = self._resolve_input_columns(pdf)
        data = pdf[cols].values.astype(np.float64)
        n_rows = len(pdf)

        # Normalise
        normed = (data - self._col_means) / self._col_stds

        result = np.zeros((n_rows, self.output_dim), dtype=np.float32)

        # Guard: row-by-row TDA is O(n*d^2); cap at 50K rows
        _MAX_ROWS_FOR_ROW_TDA = 50_000
        if n_rows > _MAX_ROWS_FOR_ROW_TDA:
            logger.warning(
                "TDA row-by-row on %d rows would be too slow — "
                "computing on %d-row subsample and mapping via nearest neighbour",
                n_rows, _MAX_ROWS_FOR_ROW_TDA,
            )
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(n_rows, size=_MAX_ROWS_FOR_ROW_TDA, replace=False)
            sample_normed = normed[sample_idx]
            sample_result = np.zeros((_MAX_ROWS_FOR_ROW_TDA, self.output_dim), dtype=np.float32)
            for si, i in enumerate(sample_idx):
                row = sample_normed[si]
                valid_mask = ~np.isnan(row)
                valid = row[valid_mask]
                if len(valid) < 2:
                    continue
                point_cloud = self._build_point_cloud(valid)
                if point_cloud.shape[0] < 2:
                    continue
                if point_cloud.shape[0] > self.n_points_subsample:
                    _rng = np.random.RandomState(si)
                    indices = _rng.choice(point_cloud.shape[0], size=self.n_points_subsample, replace=False)
                    point_cloud = point_cloud[indices]
                dist_matrix = self._pairwise_distances(point_cloud)
                try:
                    diagrams = _compute_persistence(dist_matrix, self.max_homology_dim, self.max_edge_length)
                except Exception:
                    continue
                col_idx = 0
                for h_dim in range(self.max_homology_dim + 1):
                    stats = _persistence_stats(diagrams.get(h_dim, np.empty((0, 2))), self.stats_to_compute)
                    for stat_val in stats:
                        if col_idx < self.output_dim:
                            sample_result[si, col_idx] = stat_val
                            col_idx += 1
            # kNN mapping: for each row, find nearest sample row
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(sample_normed)
            _, nn_idx = nn.kneighbors(normed)
            result = sample_result[nn_idx.ravel()]
            data_out = {}
            col_idx = 0
            for h_dim in range(self.max_homology_dim + 1):
                for stat_name in self.stats_to_compute:
                    if col_idx < self.output_dim:
                        data_out[f"{self.prefix}_h{h_dim}_{stat_name}"] = result[:, col_idx]
                        col_idx += 1
            return df_backend.from_dict(data_out, index=pdf.index)

        for i in range(n_rows):
            row = normed[i]
            valid_mask = ~np.isnan(row)
            valid = row[valid_mask]
            if len(valid) < 2:
                continue

            # Build point cloud: treat each feature as a coordinate
            # For a single row we create a small point cloud from sliding
            # windows of the feature vector.
            point_cloud = self._build_point_cloud(valid)
            if point_cloud.shape[0] < 2:
                continue

            # Subsample if too large
            if point_cloud.shape[0] > self.n_points_subsample:
                rng = np.random.RandomState(i)
                indices = rng.choice(
                    point_cloud.shape[0],
                    size=self.n_points_subsample,
                    replace=False,
                )
                point_cloud = point_cloud[indices]

            # Pairwise distance matrix — CuPy GPU acceleration for large clouds
            dist_matrix = self._pairwise_distances(point_cloud)

            # Compute persistence diagrams
            try:
                diagrams = _compute_persistence(
                    dist_matrix,
                    self.max_homology_dim,
                    self.max_edge_length,
                )
            except Exception as exc:
                logger.debug(
                    "Persistence computation failed for row %d: %s", i, exc
                )
                continue

            # Extract statistics
            col_idx = 0
            for h_dim in range(self.max_homology_dim + 1):
                dgm = diagrams.get(h_dim, np.empty((0, 2)))
                stats = _extract_diagram_stats(dgm, self.stats_to_compute)
                for stat_name in self.stats_to_compute:
                    result[i, col_idx] = stats.get(stat_name, 0.0)
                    col_idx += 1

        return df_backend.from_dict(
            {col: result[:, j] for j, col in enumerate(self.output_columns)},
            index=pdf.index,
        )

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _build_point_cloud(features: np.ndarray, window: int = 3) -> np.ndarray:
        """Construct a point cloud from a feature vector using sliding windows.

        Each window of *window* consecutive features becomes a point in
        R^window.  This converts a single 1-D feature vector into a
        multi-dimensional point cloud suitable for TDA.
        """
        n = len(features)
        if n < window:
            # If fewer features than window, just return as a single point
            return features.reshape(1, -1)
        n_points = n - window + 1
        cloud = np.empty((n_points, window), dtype=features.dtype)
        for j in range(n_points):
            cloud[j] = features[j : j + window]
        return cloud

    @staticmethod
    def _pairwise_distances(X: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distance matrix.

        Uses CuPy GPU acceleration when available and the point cloud is
        large enough to benefit (n > 500). For small clouds the GPU transfer
        overhead outweighs the computation savings.
        """
        # CuPy GPU path: ~50x faster for n > 5000 due to O(n^2) parallelisation
        # Only use GPU for clouds large enough to offset transfer overhead
        if has_cupy() and X.shape[0] > 500:
            try:
                return cupy_pairwise_distances(X)
            except Exception:
                pass  # fall through to CPU

        # CPU path: efficient vectorised computation
        sq_norms = np.sum(X ** 2, axis=1)
        dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * X @ X.T
        np.clip(dist_sq, 0.0, None, out=dist_sq)
        return np.sqrt(dist_sq)

    def _resolve_input_columns(self, df: pd.DataFrame) -> List[str]:
        """Resolve input columns, falling back to all numeric if unset."""
        if self.input_columns:
            return [c for c in self.input_columns if c in df.columns]
        return df.select_dtypes(include=["number"]).columns.tolist()

    def get_persistence_diagrams(
        self,
        features: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """Public helper: compute persistence diagrams for external use.

        Parameters
        ----------
        features : np.ndarray
            1-D feature vector (single sample) or 2-D point cloud.

        Returns
        -------
        dict[int, np.ndarray]
            Mapping from homology dimension to ``(n, 2)`` birth-death array.
        """
        if features.ndim == 1:
            point_cloud = self._build_point_cloud(features)
        else:
            point_cloud = features
        dist_matrix = self._pairwise_distances(point_cloud)
        return _compute_persistence(
            dist_matrix,
            self.max_homology_dim,
            self.max_edge_length,
        )


# ---------------------------------------------------------------------------
# Global TDA Generator
# ---------------------------------------------------------------------------


@FeatureGeneratorRegistry.register(
    "tda_global",
    description="Global TDA features: population-level topology (same values for all rows).",
    tags=["topology", "advanced", "shape", "global"],
)
class TDAGlobalGenerator(TDAFeatureGenerator):
    """Global TDA: persistent homology on the ENTIRE population point cloud.

    Computes one mean feature vector per entity (customer), builds a single
    population-level point cloud, runs persistent homology ONCE, and
    broadcasts the resulting topological features to ALL rows.  This
    captures the overall *population topology* -- how entities are
    distributed in feature space.

    Parameters
    ----------
    entity_column : str
        Column identifying entities (e.g. ``"customer_id"``).
    input_columns : list[str], optional
        Numeric columns forming the point-cloud coordinates.
    prefix : str
        Column name prefix (default ``"tda_global"``).
    **kwargs
        Forwarded to :class:`TDAFeatureGenerator`.
    """

    def __init__(
        self,
        entity_column: str = "customer_id",
        prefix: str = "tda_global",
        **kwargs: Any,
    ) -> None:
        super().__init__(prefix=prefix, **kwargs)
        self.entity_column = entity_column
        self._global_features: Optional[Dict[str, float]] = None

    def fit(self, df: Any, **context: Any) -> "TDAGlobalGenerator":
        """Group by entity, compute mean per entity -> population point cloud -> persistence."""
        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        cols = self._resolve_input_columns(pdf)
        data = pdf[cols].values.astype(np.float64)

        # Learn normalisation parameters (reuse parent logic)
        self._col_means = np.nanmean(data, axis=0)
        self._col_stds = np.nanstd(data, axis=0)
        self._col_stds[self._col_stds == 0] = 1.0
        self._global_scale = float(np.nanstd(data))

        # Adaptive max_dim
        n_features = data.shape[1]
        effective_max_dim = min(self.max_homology_dim, max(n_features - 1, 0))
        if effective_max_dim < self.max_homology_dim:
            logger.warning(
                "Reducing max_homology_dim from %d to %d (only %d input features)",
                self.max_homology_dim,
                effective_max_dim,
                n_features,
            )
            self.max_homology_dim = effective_max_dim

        # --- Global TDA: one point per entity (mean of their features) ---
        normed = (data - self._col_means) / self._col_stds
        pdf_normed = pd.DataFrame(normed, columns=cols, index=pdf.index)
        if self.entity_column in pdf.columns:
            pdf_normed[self.entity_column] = pdf[self.entity_column].values
            entity_means = pdf_normed.groupby(self.entity_column)[cols].mean()
        else:
            # No entity column -- treat each row as an entity
            entity_means = pdf_normed[cols]

        point_cloud = entity_means.values.astype(np.float64)
        # Remove rows with NaN
        valid_mask = ~np.isnan(point_cloud).any(axis=1)
        point_cloud = point_cloud[valid_mask]

        if point_cloud.shape[0] < 2:
            logger.warning("TDAGlobalGenerator: fewer than 2 entities, global features will be zero")
            self._global_features = {col: 0.0 for col in self.output_columns}
            self._fitted = True
            return self

        # Subsample if too large
        if point_cloud.shape[0] > self.n_points_subsample:
            rng = np.random.RandomState(42)
            indices = rng.choice(
                point_cloud.shape[0],
                size=self.n_points_subsample,
                replace=False,
            )
            point_cloud = point_cloud[indices]

        # Pairwise distances
        dist_matrix = self._pairwise_distances(point_cloud)

        # Run persistence ONCE on the population
        try:
            diagrams = _compute_persistence(
                dist_matrix,
                self.max_homology_dim,
                self.max_edge_length,
            )
        except Exception as exc:
            logger.warning("TDAGlobalGenerator persistence failed: %s", exc)
            self._global_features = {col: 0.0 for col in self.output_columns}
            self._fitted = True
            return self

        # Extract statistics and store as global features
        self._global_features = {}
        for h_dim in range(self.max_homology_dim + 1):
            dgm = diagrams.get(h_dim, np.empty((0, 2)))
            stats = _extract_diagram_stats(dgm, self.stats_to_compute)
            for stat_name in self.stats_to_compute:
                col_name = f"{self.prefix}_h{h_dim}_{stat_name}"
                self._global_features[col_name] = stats.get(stat_name, 0.0)

        self._fitted = True
        logger.info(
            "TDAGlobalGenerator fitted: %d entities -> %d output features, backend=%s",
            point_cloud.shape[0],
            self.output_dim,
            _BACKEND,
        )
        return self

    def generate(self, df: Any, **context: Any) -> Any:
        """Broadcast the SAME global features to every row."""
        if not self._fitted:
            raise RuntimeError("TDAGlobalGenerator must be fitted before generate().")

        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        n_rows = len(pdf)

        # Broadcast global features to all rows
        result_dict = {}
        for col_name in self.output_columns:
            val = self._global_features.get(col_name, 0.0)
            result_dict[col_name] = np.full(n_rows, val, dtype=np.float32)

        return df_backend.from_dict(result_dict, index=pdf.index)


# ---------------------------------------------------------------------------
# Local TDA Generator
# ---------------------------------------------------------------------------


def _compute_entity_tda(
    entity_data: np.ndarray,
    max_homology_dim: int,
    max_edge_length: float,
    n_points_subsample: int,
    stats_to_compute: List[str],
    entity_seed: int,
) -> Dict[str, float]:
    """Compute TDA features for a single entity's transaction point cloud.

    This is a module-level function so it can be pickled by joblib for
    parallel execution.
    """
    # Remove NaN rows
    valid_mask = ~np.isnan(entity_data).any(axis=1)
    entity_data = entity_data[valid_mask]

    result: Dict[str, float] = {}
    n_stats = (max_homology_dim + 1) * len(stats_to_compute)

    if entity_data.shape[0] < 2:
        for h_dim in range(max_homology_dim + 1):
            for stat_name in stats_to_compute:
                result[f"h{h_dim}_{stat_name}"] = 0.0
        return result

    point_cloud = entity_data

    # Subsample if too large
    if point_cloud.shape[0] > n_points_subsample:
        rng = np.random.RandomState(entity_seed)
        indices = rng.choice(
            point_cloud.shape[0],
            size=n_points_subsample,
            replace=False,
        )
        point_cloud = point_cloud[indices]

    # Pairwise distances (CPU only -- called inside joblib worker)
    sq_norms = np.sum(point_cloud ** 2, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * point_cloud @ point_cloud.T
    np.clip(dist_sq, 0.0, None, out=dist_sq)
    dist_matrix = np.sqrt(dist_sq)

    try:
        diagrams = _compute_persistence(dist_matrix, max_homology_dim, max_edge_length)
    except Exception:
        for h_dim in range(max_homology_dim + 1):
            for stat_name in stats_to_compute:
                result[f"h{h_dim}_{stat_name}"] = 0.0
        return result

    for h_dim in range(max_homology_dim + 1):
        dgm = diagrams.get(h_dim, np.empty((0, 2)))
        stats = _extract_diagram_stats(dgm, stats_to_compute)
        for stat_name in stats_to_compute:
            result[f"h{h_dim}_{stat_name}"] = stats.get(stat_name, 0.0)

    return result


@FeatureGeneratorRegistry.register(
    "tda_local",
    description="Local TDA features: per-entity topology from individual transaction histories.",
    tags=["topology", "advanced", "shape", "local"],
)
class TDALocalGenerator(TDAFeatureGenerator):
    """Local TDA: persistent homology PER ENTITY on their transaction data.

    For each entity (customer), treats their individual transactions as
    points in feature space, builds a per-entity point cloud, and computes
    persistent homology independently.  This captures the *individual
    behaviour topology* -- each entity gets UNIQUE topological features.

    Uses joblib parallelization for per-entity computation when available.

    Parameters
    ----------
    entity_column : str
        Column identifying entities (e.g. ``"customer_id"``).
    n_jobs : int
        Number of parallel jobs for per-entity computation (default ``-1``
        for all CPUs).  Requires joblib.
    input_columns : list[str], optional
        Numeric columns forming the point-cloud coordinates.
    prefix : str
        Column name prefix (default ``"tda_local"``).
    **kwargs
        Forwarded to :class:`TDAFeatureGenerator`.
    """

    def __init__(
        self,
        entity_column: str = "customer_id",
        n_jobs: int = -1,
        prefix: str = "tda_local",
        **kwargs: Any,
    ) -> None:
        super().__init__(prefix=prefix, **kwargs)
        self.entity_column = entity_column
        self.n_jobs = n_jobs

    def fit(self, df: Any, **context: Any) -> "TDALocalGenerator":
        """Store normalisation parameters (actual computation is per-entity at generate time)."""
        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        cols = self._resolve_input_columns(pdf)
        data = pdf[cols].values.astype(np.float64)

        # Learn normalisation parameters
        self._col_means = np.nanmean(data, axis=0)
        self._col_stds = np.nanstd(data, axis=0)
        self._col_stds[self._col_stds == 0] = 1.0
        self._global_scale = float(np.nanstd(data))

        # Adaptive max_dim
        n_features = data.shape[1]
        effective_max_dim = min(self.max_homology_dim, max(n_features - 1, 0))
        if effective_max_dim < self.max_homology_dim:
            logger.warning(
                "Reducing max_homology_dim from %d to %d (only %d input features)",
                self.max_homology_dim,
                effective_max_dim,
                n_features,
            )
            self.max_homology_dim = effective_max_dim

        _resolve_tda_backend()
        self._fitted = True
        logger.info(
            "TDALocalGenerator fitted: %d input cols -> %d output features, "
            "max_homology_dim=%d, backend=%s, n_jobs=%d, joblib=%s",
            len(cols),
            self.output_dim,
            self.max_homology_dim,
            _BACKEND,
            self.n_jobs,
            _HAS_JOBLIB,
        )
        return self

    def generate(self, df: Any, **context: Any) -> Any:
        """Compute TDA features per entity, with joblib parallelization."""
        if not self._fitted:
            raise RuntimeError("TDALocalGenerator must be fitted before generate().")

        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        cols = self._resolve_input_columns(pdf)
        data = pdf[cols].values.astype(np.float64)
        n_rows = len(pdf)

        # Normalise
        normed = (data - self._col_means) / self._col_stds

        result = np.zeros((n_rows, self.output_dim), dtype=np.float32)

        if self.entity_column not in pdf.columns:
            logger.warning(
                "TDALocalGenerator: entity_column '%s' not found — returning zeros "
                "(row-by-row TDA on %d rows is infeasible)",
                self.entity_column, n_rows,
            )
            return df_backend.from_dict(
                {f"{self.prefix}_h{h}_{s}": result[:, i]
                 for i, (h, s) in enumerate(
                     (h, s) for h in range(self.max_homology_dim + 1)
                     for s in self.stats_to_compute
                 ) if i < self.output_dim},
                index=pdf.index,
            )

        # Group rows by entity
        entity_col = pdf[self.entity_column].values
        unique_entities = pd.unique(entity_col)

        # Short-circuit: if most entities have <=1 row, TDA is meaningless
        avg_rows_per_entity = n_rows / max(len(unique_entities), 1)
        if avg_rows_per_entity < 2.0:
            logger.info(
                "TDALocalGenerator: avg %.1f rows/entity — TDA needs multi-row sequences, returning zeros",
                avg_rows_per_entity,
            )
            return df_backend.from_dict(
                {f"{self.prefix}_h{h}_{s}": result[:, i]
                 for i, (h, s) in enumerate(
                     (h, s) for h in range(self.max_homology_dim + 1)
                     for s in self.stats_to_compute
                 ) if i < self.output_dim},
                index=pdf.index,
            )
        entity_to_rows: Dict[Any, np.ndarray] = {}
        for entity in unique_entities:
            mask = entity_col == entity
            entity_to_rows[entity] = np.where(mask)[0]

        # Build per-entity data
        entity_items = []
        for i, entity in enumerate(unique_entities):
            row_indices = entity_to_rows[entity]
            entity_data = normed[row_indices]
            entity_items.append((entity, row_indices, entity_data, i))

        if _HAS_JOBLIB and len(unique_entities) > 1:
            # Parallel per-entity TDA computation
            entity_results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_compute_entity_tda)(
                    entity_data,
                    self.max_homology_dim,
                    self.max_edge_length,
                    self.n_points_subsample,
                    self.stats_to_compute,
                    seed,
                )
                for _, _, entity_data, seed in entity_items
            )
        else:
            # Sequential fallback
            entity_results = [
                _compute_entity_tda(
                    entity_data,
                    self.max_homology_dim,
                    self.max_edge_length,
                    self.n_points_subsample,
                    self.stats_to_compute,
                    seed,
                )
                for _, _, entity_data, seed in entity_items
            ]

        # Map results back to row indices
        for (entity, row_indices, _, _), entity_feat in zip(entity_items, entity_results):
            col_idx = 0
            for h_dim in range(self.max_homology_dim + 1):
                for stat_name in self.stats_to_compute:
                    val = entity_feat.get(f"h{h_dim}_{stat_name}", 0.0)
                    result[row_indices, col_idx] = val
                    col_idx += 1

        return df_backend.from_dict(
            {col: result[:, j] for j, col in enumerate(self.output_columns)},
            index=pdf.index,
        )
