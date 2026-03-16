"""
Multidisciplinary Feature Generator.

Applies models from non-ML scientific disciplines to financial user data,
generating features that capture dynamics invisible to standard statistical
or deep-learning approaches.

Supported sub-models
--------------------
1. **Chemical kinetics** (Michaelis-Menten): models product adoption
   velocity as an enzyme-substrate reaction -- users "saturate" as they
   adopt more products, following a hyperbolic curve.

2. **Epidemic diffusion** (SIR-inspired): models information/trend
   propagation through a user network.  Users transition between
   susceptible (unaware), infected (active adopter), and recovered
   (past adopter) states.

3. **Wave interference**: models how multiple marketing channels
   interact (constructive/destructive interference) at the user level,
   captured as amplitude and phase features.

4. **Crime pattern** (repeat-victimisation): models the temporal
   clustering of financial events (transactions, logins, complaints)
   using the "flag" and "boost" hypothesis from criminology.

This is a **placeholder implementation** with proper interfaces.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry

logger = logging.getLogger(__name__)


# Sub-model definitions
_SUB_MODELS = {
    "chemical_kinetics": {
        "output_cols": ["vmax", "km", "saturation_ratio", "adoption_velocity"],
        "description": "Michaelis-Menten product adoption kinetics",
    },
    "epidemic_diffusion": {
        "output_cols": ["sir_susceptible", "sir_infected", "sir_recovered", "r0_estimate"],
        "description": "SIR-inspired trend propagation state",
    },
    "interference": {
        "output_cols": ["wave_amplitude", "wave_phase", "constructive_score", "destructive_score"],
        "description": "Multi-channel marketing wave interference",
    },
    "crime_pattern": {
        "output_cols": ["flag_score", "boost_score", "event_clustering", "inter_event_decay"],
        "description": "Repeat-victimisation temporal clustering",
    },
}


@FeatureGeneratorRegistry.register(
    "multidisciplinary",
    description="Chemical kinetics, epidemic diffusion, wave interference, crime pattern features.",
    tags=["multidisciplinary", "kinetics", "diffusion", "interference", "advanced"],
)
class MultidisciplinaryGenerator(AbstractFeatureGenerator):
    """Multidisciplinary feature generator.

    Computes features derived from scientific models outside the ML domain,
    applied to financial user data.

    Parameters
    ----------
    sub_models : list[str]
        Which sub-models to enable.  Subset of ``["chemical_kinetics",
        "epidemic_diffusion", "interference", "crime_pattern"]``.
        Default: all four.
    observation_columns : list[str]
        Numeric columns used as inputs to the sub-models.
    time_column : str, optional
        Column containing timestamps or ordinal time indices.
        Required for ``crime_pattern`` and ``epidemic_diffusion``.
    prefix : str
        Column name prefix for generated features.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        sub_models: Optional[List[str]] = None,
        observation_columns: Optional[List[str]] = None,
        time_column: Optional[str] = None,
        prefix: str = "multi",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sub_models = sub_models or list(_SUB_MODELS.keys())
        self.observation_columns = observation_columns or []
        self.time_column = time_column
        self.prefix = prefix
        self.random_state = random_state

        # Validate sub-model names
        for sm in self.sub_models:
            if sm not in _SUB_MODELS:
                raise ValueError(
                    f"Unknown sub-model '{sm}'. "
                    f"Available: {list(_SUB_MODELS.keys())}"
                )

        # Fitted parameters for each sub-model
        self._fitted_params: Dict[str, Dict[str, Any]] = {}

    # -- Output description --------------------------------------------

    @property
    def output_dim(self) -> int:
        """Total features across all enabled sub-models (4 per sub-model)."""
        return sum(
            len(_SUB_MODELS[sm]["output_cols"]) for sm in self.sub_models
        )

    @property
    def output_columns(self) -> List[str]:
        """Generated column names with prefix."""
        cols = []
        for sm in self.sub_models:
            for col_name in _SUB_MODELS[sm]["output_cols"]:
                cols.append(f"{self.prefix}_{col_name}")
        return cols

    # -- Core API ------------------------------------------------------

    def fit(self, df: pd.DataFrame, **context: Any) -> "MultidisciplinaryGenerator":
        """Fit sub-model parameters from training data.

        Each sub-model learns its own set of parameters:
        - Chemical kinetics: Vmax and Km from observed adoption curves.
        - Epidemic diffusion: beta (infection rate) and gamma (recovery rate).
        - Interference: base frequencies and phases per channel.
        - Crime pattern: flag/boost decay parameters.
        """
        rng = np.random.RandomState(self.random_state)
        obs_cols = self._resolve_observation_columns(df)

        if obs_cols:
            data = df[obs_cols].values.astype(np.float64)
            global_mean = float(np.nanmean(data))
            global_std = float(np.nanstd(data)) + 1e-10
        else:
            global_mean = 0.0
            global_std = 1.0

        for sm in self.sub_models:
            if sm == "chemical_kinetics":
                self._fitted_params[sm] = {
                    "vmax": global_mean + 2 * global_std,
                    "km": max(global_mean, 0.1),
                    "global_mean": global_mean,
                    "global_std": global_std,
                }
            elif sm == "epidemic_diffusion":
                self._fitted_params[sm] = {
                    "beta": 0.3,  # infection rate
                    "gamma": 0.1,  # recovery rate
                    "population_mean": global_mean,
                    "population_std": global_std,
                }
            elif sm == "interference":
                n_channels = max(len(obs_cols), 2)
                self._fitted_params[sm] = {
                    "frequencies": rng.uniform(0.1, 2.0, size=n_channels).tolist(),
                    "phases": rng.uniform(0, 2 * np.pi, size=n_channels).tolist(),
                    "n_channels": n_channels,
                }
            elif sm == "crime_pattern":
                self._fitted_params[sm] = {
                    "flag_decay": 0.85,
                    "boost_factor": 1.5,
                    "temporal_window": 7,
                    "global_mean": global_mean,
                    "global_std": global_std,
                }

        self._fitted = True
        logger.info(
            "MultidisciplinaryGenerator fitted: sub_models=%s, output_dim=%d",
            self.sub_models, self.output_dim,
        )
        return self

    def generate(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame:
        """Generate multidisciplinary features for each row.

        .. note::
           Placeholder implementations derive features from input
           statistics.  Replace with actual model computations for
           production use.
        """
        if not self._fitted:
            raise RuntimeError(
                "MultidisciplinaryGenerator must be fitted before generate()."
            )

        n_rows = len(df)
        results: Dict[str, np.ndarray] = {}
        obs_cols = self._resolve_observation_columns(df)
        obs_data = df[obs_cols].values.astype(np.float64) if obs_cols else None

        for sm in self.sub_models:
            params = self._fitted_params[sm]

            if sm == "chemical_kinetics":
                results.update(
                    self._generate_kinetics(obs_data, n_rows, params)
                )
            elif sm == "epidemic_diffusion":
                results.update(
                    self._generate_sir(obs_data, n_rows, params)
                )
            elif sm == "interference":
                results.update(
                    self._generate_interference(obs_data, n_rows, params)
                )
            elif sm == "crime_pattern":
                results.update(
                    self._generate_crime_pattern(obs_data, n_rows, params)
                )

        return pd.DataFrame(results, index=df.index)

    # -- Sub-model implementations (placeholders) ----------------------

    def _generate_kinetics(
        self,
        obs: Optional[np.ndarray],
        n_rows: int,
        params: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Michaelis-Menten kinetics: V = Vmax * [S] / (Km + [S])."""
        vmax = params["vmax"]
        km = params["km"]

        substrate = np.full(n_rows, params["global_mean"], dtype=np.float32)
        if obs is not None:
            substrate = np.nanmean(np.abs(obs), axis=1).astype(np.float32)

        velocity = vmax * substrate / (km + substrate + 1e-10)
        saturation = substrate / (km + substrate + 1e-10)

        return {
            f"{self.prefix}_vmax": np.full(n_rows, vmax, dtype=np.float32),
            f"{self.prefix}_km": np.full(n_rows, km, dtype=np.float32),
            f"{self.prefix}_saturation_ratio": saturation,
            f"{self.prefix}_adoption_velocity": velocity,
        }

    def _generate_sir(
        self,
        obs: Optional[np.ndarray],
        n_rows: int,
        params: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """SIR epidemic model: susceptible / infected / recovered states."""
        beta = params["beta"]
        gamma = params["gamma"]
        pop_mean = params["population_mean"]
        pop_std = params["population_std"]

        # Derive state probabilities from observation intensity
        if obs is not None:
            intensity = np.nanmean(obs, axis=1).astype(np.float32)
            z = (intensity - pop_mean) / (pop_std + 1e-10)
        else:
            z = np.zeros(n_rows, dtype=np.float32)

        # Map z-score to SIR compartments via sigmoid
        infected = 1.0 / (1.0 + np.exp(-z))
        recovered = np.clip(1.0 / (1.0 + np.exp(-(z - 1.0))), 0, 1)
        susceptible = np.clip(1.0 - infected - recovered, 0, 1)
        r0 = np.full(n_rows, beta / (gamma + 1e-10), dtype=np.float32)

        return {
            f"{self.prefix}_sir_susceptible": susceptible.astype(np.float32),
            f"{self.prefix}_sir_infected": infected.astype(np.float32),
            f"{self.prefix}_sir_recovered": recovered.astype(np.float32),
            f"{self.prefix}_r0_estimate": r0,
        }

    def _generate_interference(
        self,
        obs: Optional[np.ndarray],
        n_rows: int,
        params: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Wave interference: superposition of multiple channel signals."""
        frequencies = np.array(params["frequencies"])
        phases = np.array(params["phases"])

        amplitude = np.zeros(n_rows, dtype=np.float32)
        phase_out = np.zeros(n_rows, dtype=np.float32)
        constructive = np.zeros(n_rows, dtype=np.float32)
        destructive = np.zeros(n_rows, dtype=np.float32)

        for i in range(n_rows):
            if obs is not None and obs.shape[1] > 0:
                row = obs[i]
                valid = row[~np.isnan(row)]
                if len(valid) == 0:
                    continue

                # Treat each column as a "channel" signal
                n_ch = min(len(valid), len(frequencies))
                # Superposition
                total_real = 0.0
                total_imag = 0.0
                for c in range(n_ch):
                    a = abs(float(valid[c]))
                    phi = float(frequencies[c]) * float(valid[c]) + float(phases[c])
                    total_real += a * np.cos(phi)
                    total_imag += a * np.sin(phi)

                amplitude[i] = float(np.sqrt(total_real**2 + total_imag**2))
                phase_out[i] = float(np.arctan2(total_imag, total_real))

                # Constructive: channels reinforce each other
                # Destructive: channels cancel each other
                channel_amps = np.abs(valid[:n_ch])
                max_possible = float(np.sum(channel_amps))
                if max_possible > 0:
                    constructive[i] = amplitude[i] / max_possible
                    destructive[i] = 1.0 - constructive[i]

        return {
            f"{self.prefix}_wave_amplitude": amplitude,
            f"{self.prefix}_wave_phase": phase_out,
            f"{self.prefix}_constructive_score": constructive,
            f"{self.prefix}_destructive_score": destructive,
        }

    def _generate_crime_pattern(
        self,
        obs: Optional[np.ndarray],
        n_rows: int,
        params: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Repeat-victimisation: flag (static risk) + boost (temporal risk)."""
        flag_decay = params["flag_decay"]
        boost_factor = params["boost_factor"]
        global_mean = params["global_mean"]
        global_std = params["global_std"]

        flag_score = np.zeros(n_rows, dtype=np.float32)
        boost_score = np.zeros(n_rows, dtype=np.float32)
        clustering = np.zeros(n_rows, dtype=np.float32)
        decay = np.zeros(n_rows, dtype=np.float32)

        if obs is not None:
            for i in range(n_rows):
                row = obs[i]
                valid = row[~np.isnan(row)]
                if len(valid) < 2:
                    continue

                # Flag: static risk level (based on overall magnitude)
                flag_score[i] = float(np.mean(np.abs(valid))) / (global_std + 1e-10)

                # Boost: temporal acceleration
                diffs = np.diff(valid)
                if len(diffs) > 0:
                    boost_score[i] = float(np.mean(np.abs(diffs))) * boost_factor

                # Event clustering (coefficient of variation of inter-event gaps)
                abs_diffs = np.abs(diffs) + 1e-10
                if len(abs_diffs) > 1:
                    clustering[i] = float(np.std(abs_diffs) / np.mean(abs_diffs))

                # Decay: exponential decay from most recent event
                decay[i] = float(flag_decay ** len(valid))

        return {
            f"{self.prefix}_flag_score": flag_score,
            f"{self.prefix}_boost_score": boost_score,
            f"{self.prefix}_event_clustering": clustering,
            f"{self.prefix}_inter_event_decay": decay,
        }

    # -- Helpers -------------------------------------------------------

    def _resolve_observation_columns(self, df: pd.DataFrame) -> List[str]:
        """Resolve observation columns, falling back to all numeric."""
        if self.observation_columns:
            return [c for c in self.observation_columns if c in df.columns]
        return df.select_dtypes(include=["number"]).columns.tolist()
