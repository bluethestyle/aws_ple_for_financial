"""
HMM (Hidden Markov Model) Feature Generator -- triple-mode state estimation.

Estimates latent states from sequential user data using three complementary
perspectives:

1. **Journey mode**: models the user's progression through product adoption
   stages (awareness -> consideration -> purchase -> loyalty).
2. **Lifecycle mode**: models the user's lifecycle phase (new -> growing ->
   mature -> declining -> churned).
3. **Behavior mode**: models short-term behavioural patterns (browsing,
   comparing, transacting, dormant).

Each mode produces a state probability vector and a most-likely-state
indicator, enabling the PLE model to capture different temporal dynamics
simultaneously.

This is a **placeholder implementation** that generates synthetic state
features.  A production implementation would use ``hmmlearn`` or a custom
Baum-Welch implementation fitted on actual sequential data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.data.dataframe import df_backend
from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry

logger = logging.getLogger(__name__)


# Default mode definitions
_DEFAULT_MODES = {
    "journey": {
        "n_states": 4,
        "state_names": ["awareness", "consideration", "purchase", "loyalty"],
        "description": "Product adoption journey stages",
    },
    "lifecycle": {
        "n_states": 5,
        "state_names": ["new", "growing", "mature", "declining", "churned"],
        "description": "Customer lifecycle phases",
    },
    "behavior": {
        "n_states": 4,
        "state_names": ["browsing", "comparing", "transacting", "dormant"],
        "description": "Short-term behavioural patterns",
    },
}


@FeatureGeneratorRegistry.register(
    "hmm_triple_mode",
    description="Triple-mode HMM state estimation (journey / lifecycle / behavior).",
    tags=["hmm", "temporal", "sequential", "state"],
)
class HMMFeatureGenerator(AbstractFeatureGenerator):
    """Triple-mode Hidden Markov Model feature generator.

    For each enabled mode, the generator produces:
      - ``{prefix}_{mode}_state``: most-likely state index (int).
      - ``{prefix}_{mode}_prob_{state_name}``: probability of each state
        (float, sums to 1.0).

    Parameters
    ----------
    modes : list[str]
        Which modes to enable.  Subset of ``["journey", "lifecycle",
        "behavior"]``.  Default: all three.
    mode_configs : dict, optional
        Override the default mode definitions.  Maps mode name to
        ``{"n_states": int, "state_names": list[str]}``.
    sequence_columns : list[str]
        Columns that contain sequential / temporal signals used as HMM
        observations.
    prefix : str
        Column name prefix for generated features.
    n_iter : int
        Number of Baum-Welch iterations (for production use).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        modes: Optional[List[str]] = None,
        mode_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        sequence_columns: Optional[List[str]] = None,
        prefix: str = "hmm",
        n_iter: int = 100,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.modes = modes or list(_DEFAULT_MODES.keys())
        self.sequence_columns = sequence_columns or []
        self.prefix = prefix
        self.n_iter = n_iter
        self.random_state = random_state

        # Build mode configs
        self._mode_configs: Dict[str, Dict[str, Any]] = {}
        for mode in self.modes:
            if mode_configs and mode in mode_configs:
                self._mode_configs[mode] = mode_configs[mode]
            elif mode in _DEFAULT_MODES:
                self._mode_configs[mode] = dict(_DEFAULT_MODES[mode])
            else:
                raise ValueError(
                    f"Unknown HMM mode '{mode}'. "
                    f"Available: {list(_DEFAULT_MODES.keys())} "
                    f"or provide custom mode_configs."
                )

        # Internal fitted state (transition matrices, emission params, etc.)
        self._transition_matrices: Dict[str, np.ndarray] = {}
        self._initial_probs: Dict[str, np.ndarray] = {}

    # -- Output description --------------------------------------------

    @property
    def output_dim(self) -> int:
        """Total output dimension across all modes.

        For each mode: 1 (state index) + n_states (probabilities).
        """
        total = 0
        for mode in self.modes:
            n_states = self._mode_configs[mode]["n_states"]
            total += 1 + n_states  # state_idx + probs
        return total

    @property
    def output_columns(self) -> List[str]:
        """Generated column names."""
        cols = []
        for mode in self.modes:
            cfg = self._mode_configs[mode]
            cols.append(f"{self.prefix}_{mode}_state")
            for state_name in cfg["state_names"]:
                cols.append(f"{self.prefix}_{mode}_prob_{state_name}")
        return cols

    # -- Core API ------------------------------------------------------

    def fit(self, df: Any, **context: Any) -> "HMMFeatureGenerator":
        """Fit HMM parameters for each mode.

        In production, this would run Baum-Welch on sequential data.
        The placeholder learns simple statistics to generate plausible
        state distributions.
        """
        rng = np.random.RandomState(self.random_state)

        for mode in self.modes:
            n_states = self._mode_configs[mode]["n_states"]

            # Placeholder: create random but valid transition matrix
            raw = rng.dirichlet(np.ones(n_states), size=n_states)
            # Add self-transition bias (states tend to persist)
            self._transition_matrices[mode] = 0.6 * np.eye(n_states) + 0.4 * raw
            # Normalise rows
            row_sums = self._transition_matrices[mode].sum(axis=1, keepdims=True)
            self._transition_matrices[mode] /= row_sums

            # Initial state distribution (uniform with slight bias toward early states)
            init = np.ones(n_states) / n_states
            init[0] += 0.1
            init /= init.sum()
            self._initial_probs[mode] = init

        self._fitted = True
        logger.info(
            "HMMFeatureGenerator fitted: modes=%s, total output_dim=%d",
            self.modes, self.output_dim,
        )
        return self

    def generate(self, df: Any, **context: Any) -> Any:
        """Generate HMM state features for each row.

        .. note::
           Placeholder: derives state probabilities from input feature
           statistics.  Replace with actual Viterbi / forward-backward
           decoding for production use.
        """
        if not self._fitted:
            raise RuntimeError(
                "HMMFeatureGenerator must be fitted before generate()."
            )

        pdf = df_backend.to_pandas(df) if not isinstance(df, pd.DataFrame) else df
        n_rows = len(pdf)
        result_arrays: Dict[str, np.ndarray] = {}

        # Resolve observation columns
        obs_cols = self._resolve_observation_columns(pdf)
        obs_data = pdf[obs_cols].values.astype(np.float64) if obs_cols else None

        for mode in self.modes:
            cfg = self._mode_configs[mode]
            n_states = cfg["n_states"]
            state_names = cfg["state_names"]

            # Placeholder: generate plausible state distributions
            # In production: run forward-backward or Viterbi
            probs = np.zeros((n_rows, n_states), dtype=np.float32)
            states = np.zeros(n_rows, dtype=np.int32)

            for i in range(n_rows):
                # Create pseudo-observations from available data
                if obs_data is not None and obs_data.shape[1] > 0:
                    row = obs_data[i]
                    valid = row[~np.isnan(row)]
                    if len(valid) > 0:
                        # Use row statistics to bias toward certain states
                        z_score = (valid.mean() - np.nanmean(obs_data)) / (
                            np.nanstd(obs_data) + 1e-10
                        )
                        # Map z-score to state probabilities via softmax
                        logits = np.array([
                            -abs(z_score - (2 * s / (n_states - 1) - 1))
                            for s in range(n_states)
                        ])
                    else:
                        logits = np.zeros(n_states)
                else:
                    logits = np.zeros(n_states)

                # Softmax
                exp_logits = np.exp(logits - logits.max())
                probs[i] = exp_logits / (exp_logits.sum() + 1e-10)
                states[i] = int(np.argmax(probs[i]))

            # Store results
            result_arrays[f"{self.prefix}_{mode}_state"] = states
            for j, state_name in enumerate(state_names):
                result_arrays[f"{self.prefix}_{mode}_prob_{state_name}"] = probs[:, j]

        return df_backend.from_dict(result_arrays, index=pdf.index)

    # -- Helpers -------------------------------------------------------

    def _resolve_observation_columns(self, df: pd.DataFrame) -> List[str]:
        """Resolve observation columns, falling back to all numeric."""
        if self.sequence_columns:
            return [c for c in self.sequence_columns if c in df.columns]
        return df.select_dtypes(include=["number"]).columns.tolist()
