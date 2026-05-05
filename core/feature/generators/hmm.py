"""
HMM (Hidden Markov Model) Feature Generator -- triple-mode state estimation.

Estimates latent states from sequential user data using three complementary
perspectives:

1. **Journey mode** (5 states): Awareness -> Interest -> Consideration ->
   Retention -> Advocacy.
2. **Lifecycle mode** (5 states): New -> Growing -> Mature -> AtRisk -> Churned.
3. **Behavior mode** (6 states): Dormant -> Conservative -> Routine ->
   Exploratory -> Splurge -> Investor.

Each mode trains a separate Gaussian HMM via Baum-Welch (hmmlearn) or a
simplified numpy-only EM fallback, then produces:
  - state_id (1D): most likely state from Viterbi decoding
  - state_probs (n_states D): posterior state probabilities
  - dwell_time (1D): expected dwell time in assigned state
  - transition_entropy (1D): Shannon entropy of the outgoing transition row

Total output: ~24D (depends on mode configuration).
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
# Lazy import hmmlearn
# ---------------------------------------------------------------------------
try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore[import-untyped]
except ImportError:
    GaussianHMM = None  # type: ignore[misc,assignment]
    logger.debug("hmmlearn not available -- will use numpy EM fallback.")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HMMConfig:
    """Per-mode HMM configuration."""

    n_states: int = 5
    n_iter: int = 100
    covariance_type: str = "diag"
    random_state: int = 42


# Default mode definitions
_DEFAULT_MODES: Dict[str, Dict[str, Any]] = {
    "journey": {
        "n_states": 5,
        "state_names": [
            "awareness", "interest", "consideration", "retention", "advocacy",
        ],
        "description": "Product adoption journey stages",
    },
    "lifecycle": {
        "n_states": 5,
        "state_names": ["new", "growing", "mature", "at_risk", "churned"],
        "description": "Customer lifecycle phases",
    },
    "behavior": {
        "n_states": 6,
        "state_names": [
            "dormant", "conservative", "routine",
            "exploratory", "splurge", "investor",
        ],
        "description": "Short-term behavioural patterns",
    },
}


# ---------------------------------------------------------------------------
# Simplified numpy-only EM (CPU fallback)
# ---------------------------------------------------------------------------

class _NumpyGaussianHMM:
    """Minimal Gaussian HMM with diagonal covariance (numpy only).

    Implements Baum-Welch EM for a single observation sequence.  This is a
    lightweight fallback when ``hmmlearn`` is not installed.
    """

    def __init__(
        self,
        n_components: int = 5,
        n_iter: int = 50,
        covariance_type: str = "diag",
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self._rng = np.random.RandomState(random_state)

        # Parameters (set after fit)
        self.startprob_: np.ndarray = np.array([])
        self.transmat_: np.ndarray = np.array([])
        self.means_: np.ndarray = np.array([])
        self.covars_: np.ndarray = np.array([])

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _log_normalize(log_a: np.ndarray) -> np.ndarray:
        """Log-sum-exp normalise along last axis, return log-probabilities."""
        max_val = log_a.max(axis=-1, keepdims=True)
        log_a_shifted = log_a - max_val
        log_sum = max_val + np.log(np.exp(log_a_shifted).sum(axis=-1, keepdims=True) + 1e-300)
        return log_a - log_sum

    def _log_gauss(self, X: np.ndarray) -> np.ndarray:
        """Log probability of X under each Gaussian component.

        Returns shape (T, K).
        """
        T, D = X.shape
        K = self.n_components
        log_prob = np.zeros((T, K))
        for k in range(K):
            diff = X - self.means_[k]  # (T, D)
            var = self.covars_[k] + 1e-6  # (D,)
            log_prob[:, k] = -0.5 * (
                D * np.log(2 * np.pi)
                + np.sum(np.log(var))
                + np.sum(diff ** 2 / var, axis=1)
            )
        return log_prob

    # -- forward / backward ------------------------------------------------

    def _forward(self, log_emiss: np.ndarray) -> np.ndarray:
        """Log-space forward pass. Returns log-alpha (T, K)."""
        T, K = log_emiss.shape
        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0] = np.log(self.startprob_ + 1e-300) + log_emiss[0]
        log_trans = np.log(self.transmat_ + 1e-300)
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = (
                    np.logaddexp.reduce(log_alpha[t - 1] + log_trans[:, k])
                    + log_emiss[t, k]
                )
        return log_alpha

    def _backward(self, log_emiss: np.ndarray) -> np.ndarray:
        """Log-space backward pass. Returns log-beta (T, K)."""
        T, K = log_emiss.shape
        log_beta = np.full((T, K), -np.inf)
        log_beta[T - 1] = 0.0
        log_trans = np.log(self.transmat_ + 1e-300)
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = np.logaddexp.reduce(
                    log_trans[k] + log_emiss[t + 1] + log_beta[t + 1]
                )
        return log_beta

    # -- fit ---------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "_NumpyGaussianHMM":
        """Fit HMM via Baum-Welch EM on a single observation matrix (T, D)."""
        T, D = X.shape
        K = self.n_components

        # Initialise parameters
        self.startprob_ = np.ones(K) / K
        self.transmat_ = np.ones((K, K)) / K
        # K-means-ish init: assign rows uniformly then compute stats
        assignments = np.arange(T) % K
        self._rng.shuffle(assignments)
        self.means_ = np.zeros((K, D))
        self.covars_ = np.ones((K, D))
        for k in range(K):
            mask = assignments == k
            if mask.any():
                self.means_[k] = X[mask].mean(axis=0)
                self.covars_[k] = X[mask].var(axis=0) + 1e-3
            else:
                self.means_[k] = X.mean(axis=0) + self._rng.randn(D) * 0.1
                self.covars_[k] = X.var(axis=0) + 1e-3

        for iteration in range(self.n_iter):
            log_emiss = self._log_gauss(X)
            log_alpha = self._forward(log_emiss)
            log_beta = self._backward(log_emiss)

            # Gamma (posterior state probs)
            log_gamma = log_alpha + log_beta
            log_gamma = self._log_normalize(log_gamma)
            gamma = np.exp(log_gamma)

            # Xi (pairwise transition posteriors)
            log_trans = np.log(self.transmat_ + 1e-300)
            xi_sum = np.zeros((K, K))
            for t in range(T - 1):
                log_xi_t = (
                    log_alpha[t, :, None]
                    + log_trans
                    + log_emiss[t + 1, None, :]
                    + log_beta[t + 1, None, :]
                )
                log_xi_t -= np.logaddexp.reduce(log_xi_t.ravel())
                xi_sum += np.exp(log_xi_t)

            # M-step
            self.startprob_ = gamma[0] + 1e-10
            self.startprob_ /= self.startprob_.sum()

            row_sums = xi_sum.sum(axis=1, keepdims=True) + 1e-10
            self.transmat_ = xi_sum / row_sums

            for k in range(K):
                g_k = gamma[:, k]
                g_sum = g_k.sum() + 1e-10
                self.means_[k] = (g_k[:, None] * X).sum(axis=0) / g_sum
                diff = X - self.means_[k]
                self.covars_[k] = (
                    (g_k[:, None] * diff ** 2).sum(axis=0) / g_sum + 1e-3
                )

        return self

    # -- predict / score ---------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding -- returns most-likely state sequence (T,)."""
        log_emiss = self._log_gauss(X)
        T, K = log_emiss.shape
        log_trans = np.log(self.transmat_ + 1e-300)

        viterbi = np.full((T, K), -np.inf)
        backptr = np.zeros((T, K), dtype=np.int32)
        viterbi[0] = np.log(self.startprob_ + 1e-300) + log_emiss[0]

        for t in range(1, T):
            for k in range(K):
                scores = viterbi[t - 1] + log_trans[:, k]
                backptr[t, k] = int(np.argmax(scores))
                viterbi[t, k] = scores[backptr[t, k]] + log_emiss[t, k]

        states = np.zeros(T, dtype=np.int32)
        states[T - 1] = int(np.argmax(viterbi[T - 1]))
        for t in range(T - 2, -1, -1):
            states[t] = backptr[t + 1, states[t + 1]]
        return states

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Forward-backward posterior probabilities (T, K)."""
        log_emiss = self._log_gauss(X)
        log_alpha = self._forward(log_emiss)
        log_beta = self._backward(log_emiss)
        log_gamma = log_alpha + log_beta
        log_gamma = self._log_normalize(log_gamma)
        return np.exp(log_gamma)


# ---------------------------------------------------------------------------
# HMM Feature Generator
# ---------------------------------------------------------------------------

@FeatureGeneratorRegistry.register(
    "hmm",
    description="Triple-mode HMM state estimation (journey / lifecycle / behavior).",
    tags=["hmm", "temporal", "sequential", "state"],
)
class HMMFeatureGenerator(AbstractFeatureGenerator):
    """Triple-mode Hidden Markov Model feature generator.

    Trains a Gaussian HMM per mode using Baum-Welch (via hmmlearn when
    available, otherwise a numpy EM fallback).  For each mode the generator
    outputs:

      - ``{prefix}_{mode}_state_id``: Viterbi most-likely state (int)
      - ``{prefix}_{mode}_prob_{name}``: posterior probability per state
      - ``{prefix}_{mode}_dwell_time``: expected dwell time in current state
      - ``{prefix}_{mode}_transition_entropy``: Shannon entropy of the
        outgoing transition row from the current state

    Parameters
    ----------
    modes : list[str]
        Which modes to enable.  Subset of ``["journey", "lifecycle",
        "behavior"]``.  Default: all three.
    hmm_config : HMMConfig, optional
        Shared HMM hyper-parameters (overridable per mode via
        ``mode_hmm_configs``).
    mode_hmm_configs : dict[str, HMMConfig], optional
        Per-mode HMM configs.
    sequence_columns : list[str], optional
        Columns used as observation features.  Defaults to all numeric.
    prefix : str
        Column-name prefix.
    """

    supports_gpu: bool = True
    required_libraries: List[str] = []
    optional_libraries: List[str] = ["hmmlearn", "cudf"]

    def __init__(
        self,
        modes: Optional[List[str]] = None,
        hmm_config: Optional[HMMConfig] = None,
        mode_hmm_configs: Optional[Dict[str, HMMConfig]] = None,
        sequence_columns: Optional[List[str]] = None,
        mode_observation_cols: Optional[Dict[str, List[str]]] = None,
        prefix: str = "hmm",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.modes = modes or list(_DEFAULT_MODES.keys())
        self.hmm_config = hmm_config or HMMConfig()
        self.mode_hmm_configs = mode_hmm_configs or {}
        self.sequence_columns = sequence_columns or []
        self.mode_observation_cols = mode_observation_cols or {}
        self.prefix = prefix

        for mode in self.modes:
            if mode not in _DEFAULT_MODES:
                raise ValueError(
                    f"Unknown HMM mode '{mode}'. "
                    f"Available: {list(_DEFAULT_MODES.keys())}"
                )

        self._models: Dict[str, Any] = {}
        self._mode_configs = {m: dict(_DEFAULT_MODES[m]) for m in self.modes}
        self._mode_resolved_cols: Dict[str, List[str]] = {}

    # -- helpers -----------------------------------------------------------

    def _get_hmm_config(self, mode: str) -> HMMConfig:
        """Return the HMMConfig for a given mode.

        If the user did not provide a per-mode override, derive a unique
        random_state from the mode name so that different modes do not
        converge to identical local optima when given the same observation
        matrix.  This breaks the journey/lifecycle symmetry that previously
        produced numerically identical features (n_states=5 vs 5 with the
        same shared random_state=42).
        """
        if mode in self.mode_hmm_configs:
            return self.mode_hmm_configs[mode]
        base = self.hmm_config
        # Stable per-mode seed derived from the mode name (range fits int32).
        mode_seed = (base.random_state + (hash(mode) & 0xFFFF)) & 0x7FFFFFFF
        return HMMConfig(
            n_states=base.n_states,
            n_iter=base.n_iter,
            covariance_type=base.covariance_type,
            random_state=mode_seed,
        )

    def _build_model(self, mode: str) -> Any:
        """Construct the HMM model object for *mode*."""
        cfg = self._get_hmm_config(mode)
        n_states = self._mode_configs[mode]["n_states"]

        if GaussianHMM is not None:
            return GaussianHMM(
                n_components=n_states,
                covariance_type=cfg.covariance_type,
                n_iter=cfg.n_iter,
                random_state=cfg.random_state,
            )
        else:
            logger.info(
                "Using numpy EM fallback for mode '%s' (hmmlearn unavailable).",
                mode,
            )
            return _NumpyGaussianHMM(
                n_components=n_states,
                n_iter=min(cfg.n_iter, 30),  # cap iterations for perf
                covariance_type=cfg.covariance_type,
                random_state=cfg.random_state,
            )

    @staticmethod
    def _transition_entropy(transmat: np.ndarray, state: int) -> float:
        """Shannon entropy of outgoing transition probabilities from *state*."""
        row = transmat[state]
        row = row[row > 0]
        return float(-np.sum(row * np.log(row + 1e-300)))

    @staticmethod
    def _dwell_time(transmat: np.ndarray, state: int) -> float:
        """Expected dwell time = 1 / (1 - self-transition prob)."""
        p_self = transmat[state, state]
        return float(1.0 / (1.0 - p_self + 1e-10))

    # -- Input column declaration -----------------------------------------

    @property
    def input_cols(self) -> List[str]:
        """Source columns consumed by fit() and generate().

        Returns the explicitly configured ``sequence_columns`` when set.
        When ``sequence_columns`` is empty the observation columns are
        resolved at runtime from all numeric columns in the DataFrame
        (``_resolve_observation_columns`` fallback), so only the declared
        columns are returned here; the runner is expected to include them
        in the slim frame.
        """
        return list(self.sequence_columns)

    # -- Output description ------------------------------------------------

    @property
    def output_dim(self) -> int:
        total = 0
        for mode in self.modes:
            n_states = self._mode_configs[mode]["n_states"]
            # state_id + n_states probs + dwell_time + transition_entropy
            total += 1 + n_states + 1 + 1
        return total

    @property
    def output_columns(self) -> List[str]:
        cols: List[str] = []
        for mode in self.modes:
            cfg = self._mode_configs[mode]
            cols.append(f"{self.prefix}_{mode}_state_id")
            for sn in cfg["state_names"]:
                cols.append(f"{self.prefix}_{mode}_prob_{sn}")
            cols.append(f"{self.prefix}_{mode}_dwell_time")
            cols.append(f"{self.prefix}_{mode}_transition_entropy")
        return cols

    # -- Core API ----------------------------------------------------------

    def _resolve_mode_columns(self, mode: str, df: Any) -> List[str]:
        """Resolve observation columns specific to *mode*.

        Resolution order:
          1. Explicit ``mode_observation_cols[mode]`` from config (filtered
             to columns actually present in df).
          2. Fallback to the global ``sequence_columns`` / numeric columns
             via :meth:`_resolve_observation_columns`.
        """
        wanted = self.mode_observation_cols.get(mode)
        if wanted:
            cols = list(df.columns) if hasattr(df, "columns") else []
            present = [c for c in wanted if c in cols]
            if present:
                return present
            logger.warning(
                "HMM mode '%s' requested cols %s but none present in df; "
                "falling back to shared observation columns.",
                mode,
                wanted,
            )
        return self._resolve_observation_columns(df)

    @staticmethod
    def _fill_nan(X: np.ndarray) -> np.ndarray:
        """Replace NaN with per-column mean in-place; return X."""
        col_means = np.nanmean(X, axis=0)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        return X

    def fit(self, df: Any, **context: Any) -> "HMMFeatureGenerator":
        """Fit a Gaussian HMM per mode on the observation matrix.

        When ``mode_observation_cols`` is configured each mode fits on its
        own feature subset (so journey, lifecycle, behavior receive
        semantically distinct observations).  Otherwise all modes share a
        single matrix derived from ``sequence_columns``.
        """
        if self.input_cols:
            col_arrays = self._input_to_numpy(df, columns=self.input_cols)
        else:
            col_arrays = self._input_to_numpy(df)
        n_rows = len(next(iter(col_arrays.values()))) if col_arrays else 0

        for mode in self.modes:
            obs_cols = self._resolve_mode_columns(mode, df)
            self._mode_resolved_cols[mode] = obs_cols
            if not obs_cols:
                logger.warning(
                    "HMM mode '%s' found no observation columns; "
                    "using uniform fallback.",
                    mode,
                )
                self._models[mode] = None
                continue

            X_mode = self._fill_nan(self._extract_numeric(df, obs_cols))
            if X_mode.shape[0] == 0:
                X_mode = np.zeros((n_rows, max(1, len(obs_cols))))

            model = self._build_model(mode)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model.fit(X_mode)
                    self._models[mode] = model
                    logger.info(
                        "HMM mode '%s' fitted: n_states=%d, n_obs=%d, n_features=%d, "
                        "cols=%s",
                        mode,
                        self._mode_configs[mode]["n_states"],
                        X_mode.shape[0],
                        X_mode.shape[1],
                        obs_cols[:6] + (["..."] if len(obs_cols) > 6 else []),
                    )
                except Exception as exc:
                    logger.warning(
                        "HMM fit failed for mode '%s': %s. Using random init.",
                        mode,
                        exc,
                    )
                    self._models[mode] = None

        self._fitted = True
        logger.info(
            "HMMFeatureGenerator fitted: modes=%s, total output_dim=%d",
            self.modes,
            self.output_dim,
        )
        return self

    def generate(self, df: Any, **context: Any) -> Any:
        """Generate HMM state features via Viterbi decoding and forward-backward."""
        if not self._fitted:
            raise RuntimeError(
                "HMMFeatureGenerator must be fitted before generate()."
            )

        if self.input_cols:
            col_arrays = self._input_to_numpy(df, columns=self.input_cols)
        else:
            col_arrays = self._input_to_numpy(df)
        n_rows = len(next(iter(col_arrays.values()))) if col_arrays else 0

        results: Dict[str, np.ndarray] = {}

        for mode in self.modes:
            cfg = self._mode_configs[mode]
            n_states = cfg["n_states"]
            state_names = cfg["state_names"]
            model = self._models.get(mode)

            obs_cols = self._mode_resolved_cols.get(mode) or self._resolve_mode_columns(mode, df)
            X_mode = self._fill_nan(self._extract_numeric(df, obs_cols)) if obs_cols else np.zeros((n_rows, 1))

            if model is not None:
                try:
                    states = model.predict(X_mode)
                except Exception:
                    states = np.zeros(n_rows, dtype=np.int32)

                try:
                    probs = model.predict_proba(X_mode)
                except Exception:
                    probs = np.full((n_rows, n_states), 1.0 / n_states, dtype=np.float32)

                transmat = model.transmat_
            else:
                # Fallback: uniform
                states = np.zeros(n_rows, dtype=np.int32)
                probs = np.full((n_rows, n_states), 1.0 / n_states, dtype=np.float32)
                transmat = np.ones((n_states, n_states)) / n_states

            # state_id
            results[f"{self.prefix}_{mode}_state_id"] = states.astype(np.int32)

            # per-state probabilities
            for j, sn in enumerate(state_names):
                results[f"{self.prefix}_{mode}_prob_{sn}"] = probs[:, j].astype(np.float32)

            # dwell_time and transition_entropy per row (based on assigned state)
            dwell = np.array(
                [self._dwell_time(transmat, int(s)) for s in states],
                dtype=np.float32,
            )
            t_entropy = np.array(
                [self._transition_entropy(transmat, int(s)) for s in states],
                dtype=np.float32,
            )
            results[f"{self.prefix}_{mode}_dwell_time"] = dwell
            results[f"{self.prefix}_{mode}_transition_entropy"] = t_entropy

        # Build output DataFrame via cuDF when available
        if has_cudf() and _cudf is not None:
            return _cudf.DataFrame(results)
        return pd.DataFrame(results)

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _extract_numeric(df: Any, cols: List[str]) -> np.ndarray:
        """Extract columns as a numpy float64 array from any DataFrame type."""
        from .gpu_utils import _to_numpy_safe
        return _to_numpy_safe(df, cols, fill=0.0)

    def _resolve_observation_columns(self, df: Any) -> List[str]:
        """Resolve observation columns, falling back to all numeric."""
        if self.sequence_columns:
            cols = list(df.columns) if hasattr(df, 'columns') else []
            return [c for c in self.sequence_columns if c in cols]
        # cuDF and pandas both support select_dtypes
        if hasattr(df, 'select_dtypes'):
            return df.select_dtypes(include=["number"]).columns.tolist()
        pdf = df_backend.to_pandas(df)
        return pdf.select_dtypes(include=["number"]).columns.tolist()
