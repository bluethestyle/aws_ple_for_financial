"""
Phase Transition Feature Generator.

Detects topological phase transitions by comparing persistence diagrams
between consecutive time windows.  A "phase transition" occurs when the
topological structure of the data changes significantly -- e.g. a cluster
splits, a loop appears, or a void collapses.

In a financial context, phase transitions can signal:
  - Regime changes in user spending behaviour.
  - Emergence of new transaction patterns (new loop in H1).
  - Market-driven structural shifts in user cohorts.

Hardware acceleration
---------------------
This generator depends on TDA persistence diagrams (see
:class:`TDAFeatureGenerator`).  Wasserstein distance computation uses
``persim`` when available, falling back to ``scipy.optimize`` or a
simplified sorted-matching approach with numpy only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.data.dataframe import df_backend
from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

_PERSIM = None
_SCIPY_OPT = None
_WASSERSTEIN_BACKEND: Optional[str] = None


def _resolve_wasserstein_backend() -> str:
    """Determine which Wasserstein distance backend is available.

    Fallback chain:
      1. persim (dedicated persistence diagram distances)
      2. scipy.optimize.linear_sum_assignment (optimal matching)
      3. numpy-only (sorted matching with L1 distance)
    """
    global _PERSIM, _SCIPY_OPT, _WASSERSTEIN_BACKEND

    if _WASSERSTEIN_BACKEND is not None:
        return _WASSERSTEIN_BACKEND

    # Try persim
    try:
        import persim as _ps
        _PERSIM = _ps
        _WASSERSTEIN_BACKEND = "persim"
        logger.info("Wasserstein backend: persim")
        return _WASSERSTEIN_BACKEND
    except ImportError:
        pass

    # Try scipy
    try:
        from scipy.optimize import linear_sum_assignment as _lsa
        _SCIPY_OPT = _lsa
        _WASSERSTEIN_BACKEND = "scipy"
        logger.info("Wasserstein backend: scipy.optimize.linear_sum_assignment")
        return _WASSERSTEIN_BACKEND
    except ImportError:
        pass

    # Numpy fallback
    _WASSERSTEIN_BACKEND = "numpy"
    logger.warning(
        "Using approximate Wasserstein distance (persim/scipy not available). "
        "Install persim for exact computation: pip install persim"
    )
    return _WASSERSTEIN_BACKEND


# ---------------------------------------------------------------------------
# Wasserstein distance implementations
# ---------------------------------------------------------------------------


def _wasserstein_persim(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Wasserstein-1 distance using persim."""
    return float(_PERSIM.wasserstein(dgm1, dgm2, order=1))


def _wasserstein_scipy(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Wasserstein-1 distance using scipy optimal matching.

    Augments both diagrams with projections onto the diagonal (cost of
    matching to the diagonal = lifetime / 2) then solves the optimal
    assignment on the L1 cost matrix.
    """
    n1, n2 = len(dgm1), len(dgm2)
    if n1 == 0 and n2 == 0:
        return 0.0

    # Diagonal projections: midpoint of birth-death
    def _diag_proj(dgm: np.ndarray) -> np.ndarray:
        mid = (dgm[:, 0] + dgm[:, 1]) / 2.0
        return np.column_stack([mid, mid])

    # Build augmented cost matrix
    total = n1 + n2
    cost = np.full((total, total), np.inf, dtype=np.float64)

    # Cost between real points
    for i in range(n1):
        for j in range(n2):
            cost[i, j] = np.abs(dgm1[i, 0] - dgm2[j, 0]) + np.abs(dgm1[i, 1] - dgm2[j, 1])

    # Cost of dgm1 points matched to diagonal
    for i in range(n1):
        lifetime = dgm1[i, 1] - dgm1[i, 0]
        for j in range(n2, total):
            cost[i, j] = lifetime  # L1 cost to diagonal

    # Cost of dgm2 points matched to diagonal
    for j in range(n2):
        lifetime = dgm2[j, 1] - dgm2[j, 0]
        for i in range(n1, total):
            cost[i, j] = lifetime

    # Diagonal-to-diagonal: zero cost
    for i in range(n1, total):
        for j in range(n2, total):
            cost[i, j] = 0.0

    row_ind, col_ind = _SCIPY_OPT(cost)
    return float(cost[row_ind, col_ind].sum())


def _wasserstein_numpy(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Approximate Wasserstein-1 distance using sorted matching.

    Sorts both diagrams by lifetime, pads the shorter one with zero-
    lifetime entries, and computes the L1 distance between matched pairs.
    This is a lower bound on the true Wasserstein distance.
    """
    def _lifetimes(dgm: np.ndarray) -> np.ndarray:
        if len(dgm) == 0:
            return np.array([], dtype=np.float64)
        return dgm[:, 1] - dgm[:, 0]

    l1 = np.sort(_lifetimes(dgm1))[::-1]
    l2 = np.sort(_lifetimes(dgm2))[::-1]

    # Pad to equal length
    max_len = max(len(l1), len(l2))
    if max_len == 0:
        return 0.0
    l1_pad = np.zeros(max_len, dtype=np.float64)
    l2_pad = np.zeros(max_len, dtype=np.float64)
    l1_pad[: len(l1)] = l1
    l2_pad[: len(l2)] = l2

    return float(np.sum(np.abs(l1_pad - l2_pad)))


def _wasserstein_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Compute Wasserstein-1 distance between two persistence diagrams."""
    backend = _resolve_wasserstein_backend()
    if backend == "persim":
        return _wasserstein_persim(dgm1, dgm2)
    elif backend == "scipy":
        return _wasserstein_scipy(dgm1, dgm2)
    else:
        return _wasserstein_numpy(dgm1, dgm2)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PhaseTransitionConfig:
    """Configuration for phase transition detection."""

    window_size: int = 10
    step_size: int = 5
    distance_metric: str = "wasserstein"
    threshold: float = 1.0


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@FeatureGeneratorRegistry.register(
    "phase_transition",
    description="Detect topological phase transitions via persistence diagram distances.",
    tags=["topology", "phase_transition", "advanced"],
)
class PhaseTransitionGenerator(AbstractFeatureGenerator):
    """Detect topological phase transitions between consecutive time windows.

    Compares persistence diagrams computed from sliding windows over the
    feature space and measures how the topological structure changes over
    time.  Large Wasserstein distances between consecutive windows
    indicate a phase transition.

    Depends on :class:`TDAFeatureGenerator` for persistence diagram
    computation.

    Parameters
    ----------
    input_columns : list[str]
        Numeric columns used for point-cloud construction.
    config : PhaseTransitionConfig, optional
        Full configuration dataclass.
    window_size : int
        Number of rows per sliding window.
    step_size : int
        Step between consecutive windows.
    max_homology_dim : int
        Maximum homology dimension for persistence (inherited from TDA).
    threshold : float
        Distance threshold above which a transition is "significant".
    prefix : str
        Column name prefix.

    Attributes
    ----------
    supports_gpu : bool
        False -- CPU-only computation.
    required_libraries : list[str]
        ``["ripser"]`` (with numpy fallback for both TDA and Wasserstein).

    Notes
    -----
    Wasserstein distance computation uses persim when available, falling
    back to scipy.optimize.linear_sum_assignment for optimal matching,
    or a simplified sorted-matching approach with numpy only.
    """

    supports_gpu: bool = False
    required_libraries: List[str] = []
    optional_libraries: List[str] = ["ripser", "persim"]

    def __init__(
        self,
        input_columns: Optional[List[str]] = None,
        config: Optional[PhaseTransitionConfig] = None,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
        max_homology_dim: int = 1,
        threshold: Optional[float] = None,
        prefix: str = "pt",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_columns = input_columns or []
        self.prefix = prefix
        self.max_homology_dim = max_homology_dim

        cfg = config or PhaseTransitionConfig()
        self.window_size = window_size if window_size is not None else cfg.window_size
        self.step_size = step_size if step_size is not None else cfg.step_size
        self.threshold = threshold if threshold is not None else cfg.threshold

        # Fitted state
        self._col_means: Optional[np.ndarray] = None
        self._col_stds: Optional[np.ndarray] = None
        self._history_distances: List[float] = []

    # -- Output description ------------------------------------------------

    _OUTPUT_STATS = [
        "pd_distance_h0",
        "pd_distance_h1",
        "topological_change_magnitude",
        "max_persistence_shift",
        "transition_probability",
        "transition_direction",
        "transition_frequency",
        "transition_confidence",
    ]

    @property
    def output_dim(self) -> int:
        """Number of phase transition features (~10D)."""
        # 2 distance dims (h0, h1) + 6 derived = 8, but we cap at
        # max_homology_dim + 1 distance columns + 6 derived
        return min(self.max_homology_dim + 1, 2) + 6

    @property
    def output_columns(self) -> List[str]:
        """Generated column names."""
        cols: List[str] = []
        # Per-dimension distances
        for h_dim in range(min(self.max_homology_dim + 1, 2)):
            cols.append(f"{self.prefix}_pd_distance_h{h_dim}")
        # Derived features
        cols.extend([
            f"{self.prefix}_topological_change_magnitude",
            f"{self.prefix}_max_persistence_shift",
            f"{self.prefix}_transition_probability",
            f"{self.prefix}_transition_direction",
            f"{self.prefix}_transition_frequency",
            f"{self.prefix}_transition_confidence",
        ])
        return cols

    # -- Input column declaration -----------------------------------------

    @property
    def input_cols(self) -> List[str]:
        """Source columns consumed by fit() and generate().

        Returns the explicitly configured ``input_columns`` when set.
        When ``input_columns`` is empty the columns are resolved at runtime
        from all numeric columns in the DataFrame (``_resolve_input_columns``
        fallback), so only the declared columns are returned here.
        """
        return list(self.input_columns)

    # -- Core API ----------------------------------------------------------

    def fit(self, df: Any, **context: Any) -> "PhaseTransitionGenerator":
        """Learn normalisation parameters and baseline transition statistics."""
        # Extract declared input columns up-front (slim frame boundary).
        if self.input_cols:
            col_arrays = self._input_to_numpy(df, columns=self.input_cols)
        else:
            col_arrays = self._input_to_numpy(df)

        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        cols = self._resolve_input_columns(pdf)

        if self.input_cols and col_arrays:
            # Build data matrix directly from col_arrays (avoids re-reading df).
            data = np.stack(
                [col_arrays[c].astype(np.float64) for c in cols if c in col_arrays],
                axis=1,
            ) if cols else np.empty((len(next(iter(col_arrays.values()))), 0))
        else:
            data = pdf[cols].values.astype(np.float64)

        self._col_means = np.nanmean(data, axis=0)
        self._col_stds = np.nanstd(data, axis=0)
        self._col_stds[self._col_stds == 0] = 1.0

        # Compute baseline distances on training windows
        self._history_distances = self._compute_window_distances(data)

        # Eagerly resolve backends
        _resolve_wasserstein_backend()

        self._fitted = True
        logger.info(
            "PhaseTransitionGenerator fitted: %d input cols -> %d output features, "
            "window_size=%d, step_size=%d",
            len(cols),
            self.output_dim,
            self.window_size,
            self.step_size,
        )
        return self

    def generate(self, df: Any, **context: Any) -> Any:
        """Generate phase transition features for each row.

        For each row, the generator considers the surrounding window of
        data, computes persistence diagrams for the current and previous
        windows, and measures topological distance.
        """
        if not self._fitted:
            raise RuntimeError(
                "PhaseTransitionGenerator must be fitted before generate()."
            )

        # Extract declared input columns up-front (slim frame boundary).
        if self.input_cols:
            col_arrays = self._input_to_numpy(df, columns=self.input_cols)
        else:
            col_arrays = self._input_to_numpy(df)
        n_rows = len(next(iter(col_arrays.values()))) if col_arrays else 0

        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        cols = self._resolve_input_columns(pdf)

        if self.input_cols and col_arrays:
            # Build data matrix directly from col_arrays (avoids re-reading df).
            data = np.stack(
                [col_arrays[c].astype(np.float64) for c in cols if c in col_arrays],
                axis=1,
            ) if cols else np.empty((n_rows, 0))
        else:
            data = pdf[cols].values.astype(np.float64)

        # Normalise
        normed = (data - self._col_means) / self._col_stds

        result = np.zeros((n_rows, self.output_dim), dtype=np.float32)

        # Pre-compute persistence diagrams for all windows
        window_diagrams = self._compute_all_window_diagrams(normed)

        # Baseline statistics for normalisation
        baseline_dist = np.mean(self._history_distances) if self._history_distances else 1.0
        baseline_dist = max(baseline_dist, 1e-8)

        # Track significant transitions for frequency computation
        significant_count = 0
        total_comparisons = 0

        for i in range(n_rows):
            # Determine which window this row falls in
            win_idx = min(i // max(self.step_size, 1), len(window_diagrams) - 1)
            prev_win_idx = max(win_idx - 1, 0)

            if win_idx == prev_win_idx or win_idx >= len(window_diagrams):
                # No previous window to compare against
                continue

            curr_dgms = window_diagrams[win_idx]
            prev_dgms = window_diagrams[prev_win_idx]

            col_idx = 0

            # Per-dimension Wasserstein distances
            distances: List[float] = []
            for h_dim in range(min(self.max_homology_dim + 1, 2)):
                curr_d = curr_dgms.get(h_dim, np.empty((0, 2)))
                prev_d = prev_dgms.get(h_dim, np.empty((0, 2)))
                try:
                    dist = _wasserstein_distance(curr_d, prev_d)
                except Exception:
                    dist = 0.0
                distances.append(dist)
                result[i, col_idx] = dist
                col_idx += 1

            # Topological change magnitude: L2 norm of all distances
            magnitude = float(np.sqrt(sum(d ** 2 for d in distances)))
            result[i, col_idx] = magnitude
            col_idx += 1

            # Max persistence shift: largest change in max-lifetime
            max_shift = 0.0
            for h_dim in range(min(self.max_homology_dim + 1, 2)):
                curr_d = curr_dgms.get(h_dim, np.empty((0, 2)))
                prev_d = prev_dgms.get(h_dim, np.empty((0, 2)))
                curr_max = float(np.max(curr_d[:, 1] - curr_d[:, 0])) if len(curr_d) > 0 else 0.0
                prev_max = float(np.max(prev_d[:, 1] - prev_d[:, 0])) if len(prev_d) > 0 else 0.0
                max_shift = max(max_shift, abs(curr_max - prev_max))
            result[i, col_idx] = max_shift
            col_idx += 1

            # Transition probability: sigmoid of normalised distance
            norm_dist = magnitude / baseline_dist
            prob = 1.0 / (1.0 + np.exp(-norm_dist + 2.0))  # shifted sigmoid
            result[i, col_idx] = prob
            col_idx += 1

            # Transition direction: sign of change (positive = growing complexity)
            curr_total = sum(
                float(np.sum(curr_dgms.get(h, np.empty((0, 2)))[:, 1] - curr_dgms.get(h, np.empty((0, 2)))[:, 0]))
                if len(curr_dgms.get(h, np.empty((0, 2)))) > 0 else 0.0
                for h in range(min(self.max_homology_dim + 1, 2))
            )
            prev_total = sum(
                float(np.sum(prev_dgms.get(h, np.empty((0, 2)))[:, 1] - prev_dgms.get(h, np.empty((0, 2)))[:, 0]))
                if len(prev_dgms.get(h, np.empty((0, 2)))) > 0 else 0.0
                for h in range(min(self.max_homology_dim + 1, 2))
            )
            direction = 1.0 if curr_total > prev_total else (-1.0 if curr_total < prev_total else 0.0)
            result[i, col_idx] = direction
            col_idx += 1

            # Track significant transitions
            total_comparisons += 1
            if magnitude > self.threshold:
                significant_count += 1

            # Transition frequency: fraction of significant transitions so far
            result[i, col_idx] = significant_count / max(total_comparisons, 1)
            col_idx += 1

            # Transition confidence: based on window sample size
            n_points_in_window = min(self.window_size, n_rows - win_idx * self.step_size)
            confidence = 1.0 - np.exp(-n_points_in_window / max(self.window_size, 1))
            result[i, col_idx] = confidence
            col_idx += 1

        return df_backend.from_dict(
            {col: result[:, j] for j, col in enumerate(self.output_columns)},
            index=pdf.index,
        )

    # -- Helpers -----------------------------------------------------------

    def _compute_all_window_diagrams(
        self,
        normed: np.ndarray,
    ) -> List[Dict[int, np.ndarray]]:
        """Compute persistence diagrams for all sliding windows."""
        from .tda import _compute_persistence, TDAFeatureGenerator

        n_rows = normed.shape[0]
        diagrams: List[Dict[int, np.ndarray]] = []

        for start in range(0, n_rows, max(self.step_size, 1)):
            end = min(start + self.window_size, n_rows)
            window_data = normed[start:end]

            if window_data.shape[0] < 2:
                diagrams.append({h: np.empty((0, 2)) for h in range(self.max_homology_dim + 1)})
                continue

            # Build distance matrix from the window
            dist_matrix = TDAFeatureGenerator._pairwise_distances(window_data)

            try:
                dgms = _compute_persistence(
                    dist_matrix,
                    self.max_homology_dim,
                    float("inf"),
                )
            except Exception as exc:
                logger.debug("Window persistence failed at row %d: %s", start, exc)
                dgms = {h: np.empty((0, 2)) for h in range(self.max_homology_dim + 1)}

            diagrams.append(dgms)

        return diagrams

    def _compute_window_distances(self, data: np.ndarray) -> List[float]:
        """Compute Wasserstein distances between consecutive windows on training data."""
        normed = (data - self._col_means) / self._col_stds
        window_diagrams = self._compute_all_window_diagrams(normed)

        distances: List[float] = []
        for i in range(1, len(window_diagrams)):
            total_dist = 0.0
            for h_dim in range(min(self.max_homology_dim + 1, 2)):
                d1 = window_diagrams[i - 1].get(h_dim, np.empty((0, 2)))
                d2 = window_diagrams[i].get(h_dim, np.empty((0, 2)))
                try:
                    total_dist += _wasserstein_distance(d1, d2)
                except Exception:
                    pass
            distances.append(total_dist)

        return distances

    def _resolve_input_columns(self, df: pd.DataFrame) -> List[str]:
        """Resolve input columns, falling back to all numeric if unset."""
        if self.input_columns:
            return [c for c in self.input_columns if c in df.columns]
        return df.select_dtypes(include=["number"]).columns.tolist()
