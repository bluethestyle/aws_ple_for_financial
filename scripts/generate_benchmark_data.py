#!/usr/bin/env python3
"""
Benchmark Data Generator — 4-Layer Generative Model
====================================================
Produces realistic synthetic financial data with controllable AUC ceilings.

Layers:
  1. Latent personas  (z_i categorical, l_i 5D continuous)
  2. Observable profiles (demographics + products via copula)
  3. Transaction sequences (Poisson + AR(1) + seasonality + MCC)
  4. Labels with variance budget (obs / latent / noise split)

Output schema matches configs/santander/pipeline.yaml exactly so the
generated parquet can be fed directly into the PLE pipeline.

Usage:
    python scripts/generate_benchmark_data.py \\
        --n-customers 1000000 \\
        --seed 42 \\
        --calibration configs/santander/calibration_params.yaml \\
        --output data/benchmark_v1.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy.stats import norm, truncnorm, lognorm, gamma

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (non-dataset-specific)
# ---------------------------------------------------------------------------
LATENT_DIM = 5
LATENT_NAMES = [
    "wealth_propensity",
    "activity_level",
    "risk_tolerance",
    "digital_affinity",
    "loyalty",
]
PRODUCT_NAMES = [
    "saving",
    "guarantee",
    "checking",
    "derivados",
    "payroll_acct",
    "junior_acct",
    "particular_acct",
    "particular_plus",
    "short_deposit",
    "medium_deposit",
    "long_deposit",
    "e_account",
    "funds",
    "mortgage",
    "pension_plan",
    "loans",
    "taxes",
    "credit_card",
    "securities",
    "home_acct",
    "payroll",
    "pension_deposit",
    "direct_debit",
    "auto_debit",
]
N_PRODUCTS = len(PRODUCT_NAMES)  # 24
SEGMENTS = ["01-TOP", "02-PARTICULARES", "03-UNIVERSITARIO", "UNKNOWN"]
GENDERS = ["F", "M"]
N_SEQ_MONTHS = 16  # product sequence length (minus truncated label month)
TXN_SEQ_MAX_LEN = 200


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _normalize(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1], handling constant arrays."""
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _quantile_rank(x: np.ndarray) -> np.ndarray:
    """Rank-based quantile in [0, 1]."""
    from scipy.stats import rankdata

    r = rankdata(x, method="average", nan_policy="omit")
    return (r - 1) / max(len(r) - 1, 1)


# Product-index → NBA group mapping (7 classes total, 0-6)
# class 0: no NBA (handled separately — empty list or missing)
# class 1: savings_guarantee    (prod_saving=0, prod_guarantee=1)
# class 2: checking_accounts    (prod_checking=2, prod_derivados=3,
#                                prod_payroll_acct=4, prod_junior_acct=5,
#                                prod_particular_acct=6, prod_particular_plus=7,
#                                prod_home_acct=19, prod_payroll=20)
# class 3: deposits             (prod_short_deposit=8, prod_medium_deposit=9,
#                                prod_long_deposit=10, prod_e_account=11)
# class 4: investments          (prod_funds=12, prod_mortgage=13, prod_pension_plan=14)
# class 5: credit_loans         (prod_loans=15, prod_taxes=16,
#                                prod_credit_card=17, prod_securities=18)
# class 6: debits               (prod_pension_deposit=21, prod_direct_debit=22,
#                                prod_auto_debit=23)
_NBA_GROUP_MAP: dict[int, int] = {
    0: 1, 1: 1,                          # savings_guarantee
    2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 19: 2, 20: 2,  # checking_accounts
    8: 3, 9: 3, 10: 3, 11: 3,           # deposits
    12: 4, 13: 4, 14: 4,                 # investments
    15: 5, 16: 5, 17: 5, 18: 5,         # credit_loans
    21: 6, 22: 6, 23: 6,                 # debits
}


def _product_idx_to_nba_group(idx: int) -> int:
    """Map a product index (0-23) to an NBA group class (1-6).

    Returns 6 (debits/other) for any index not explicitly listed.
    Callers must handle the "no NBA" case (class 0) before calling this.
    """
    return _NBA_GROUP_MAP.get(idx, 6)


# ============================================================================
# BenchmarkDataGenerator
# ============================================================================
class BenchmarkDataGenerator:
    """4-layer generative model for benchmark financial data."""

    def __init__(
        self,
        calibration_path: str,
        n_customers: int = 1_000_000,
        seed: int = 42,
        output_path: str = "data/benchmark_v1.parquet",
        ground_truth_path: Optional[str] = None,
        validate: bool = True,
    ):
        self.n = n_customers
        self.seed = seed
        self.output_path = output_path
        self.ground_truth_path = ground_truth_path or str(
            Path(output_path).parent / "benchmark_ground_truth.parquet"
        )
        self.validate = validate

        logger.info("Loading calibration params from %s", calibration_path)
        with open(calibration_path, "r", encoding="utf-8") as f:
            self.cal = yaml.safe_load(f)

        self.persona_names: List[str] = self.cal["personas"]["names"]
        self.n_personas = len(self.persona_names)
        self.persona_weights = np.array(self.cal["personas"]["weights"])
        self.persona_weights /= self.persona_weights.sum()  # ensure sums to 1

        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def generate(self) -> None:
        """Full pipeline: generate all layers and save to parquet."""
        t0 = time.time()
        logger.info(
            "Generating benchmark data: n=%d, seed=%d", self.n, self.seed
        )

        # Layer 1: Latent personas + situation assignment
        z, l, situations = self._generate_latent_personas()
        logger.info("Layer 1 done: personas shape=%s, latent shape=%s", z.shape, l.shape)

        # Layer 2: Observable profiles
        profiles = self._generate_profiles(z, l)
        logger.info("Layer 2 done: %d profile columns", len(profiles))

        # Layer 3: Transaction sequences (situations modulate patterns, not aggregates)
        txn_data = self._generate_transactions(z, l, profiles, situations)
        logger.info("Layer 3 done: %d txn columns", len(txn_data))

        # Merge all features
        features = {**profiles, **txn_data}

        # Layer 4: Labels
        labels = self._generate_labels(z, l, features)
        logger.info("Layer 4 done: %d label columns", len(labels))

        # Save via DuckDB
        self._save_parquet(features, labels, z, l, situations)

        elapsed = time.time() - t0
        logger.info("Generation complete in %.1fs", elapsed)

        # Validation
        if self.validate:
            self._validate_auc(features, labels)

    # ------------------------------------------------------------------
    # Variance-budget logit calibrator
    # ------------------------------------------------------------------
    def _calibrate_logit(
        self,
        obs_signal: np.ndarray,
        latent_signal: np.ndarray,
        obs_frac: float,
        lat_frac: float,
        noise_frac: float,
        intercept: float,
        target_pos_rate: float = 0.5,
    ) -> np.ndarray:
        """
        Build a logit with exact variance budget: Var(obs)/Var(total) = obs_frac.

        The obs_frac controls XGBoost AUC ceiling:
        - obs_frac=0.45 -> XGB AUC ~0.68-0.75
        - obs_frac=0.35 -> XGB AUC ~0.60-0.68

        The latent+noise components are invisible to XGBoost (not in features),
        ensuring the AUC ceiling is bounded.

        Steps:
        1. Standardize obs and latent signals to unit variance
        2. Scale each by sqrt(frac) -> Var contribution = frac
        3. Add Gaussian noise with Var = noise_frac
        4. Binary-search intercept for target positive rate
        """
        n = len(obs_signal)

        # Standardize to zero mean, unit variance
        def _std_normalize(x: np.ndarray) -> np.ndarray:
            std = max(x.std(), 1e-8)
            return (x - x.mean()) / std

        obs_n = _std_normalize(obs_signal)
        lat_n = _std_normalize(latent_signal)
        noise_n = self.rng.standard_normal(n)

        # Scale by sqrt of target variance fraction
        # Total Var = obs_frac + lat_frac + noise_frac = 1
        logit = (
            np.sqrt(obs_frac) * obs_n
            + np.sqrt(lat_frac) * lat_n
            + np.sqrt(noise_frac) * noise_n
            + intercept
        )

        # Binary search intercept to hit target positive rate
        current_rate = (_sigmoid(logit) > 0.5).mean()
        if abs(current_rate - target_pos_rate) > 0.005:
            lo, hi = -10.0, 10.0
            for _ in range(50):
                mid = (lo + hi) / 2
                trial = logit - intercept + mid
                rate = (_sigmoid(trial) > 0.5).mean()
                if rate > target_pos_rate:
                    hi = mid
                else:
                    lo = mid
            logit = logit - intercept + (lo + hi) / 2

        return logit

    def _apply_label_noise(self, labels: np.ndarray, noise_rate: float) -> np.ndarray:
        """Randomly flip binary labels to cap AUC ceiling."""
        flip_mask = self.rng.random(len(labels)) < noise_rate
        return np.where(flip_mask, 1 - labels, labels)

    # ==================================================================
    # Layer 1: Latent Personas
    # ==================================================================
    def _generate_latent_personas(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        z_i ~ Categorical(K=6, pi)   -- discrete persona index
        l_i ~ N(mu_z, Sigma_z)       -- 5D continuous latent vector
        """
        z = self.rng.choice(self.n_personas, size=self.n, p=self.persona_weights)

        # Per-persona latent mean/cov (designed, not from calibration GMM means)
        # These encode the 5 latent dimensions per persona archetype.
        persona_latent_mu = {
            "conservative_saver": [0.3, -0.5, -0.8, -0.3, 0.7],
            "active_spender": [0.5, 0.8, 0.3, 0.4, 0.2],
            "young_digital": [-0.2, 0.6, 0.5, 0.9, -0.3],
            "high_value": [0.9, 0.3, 0.4, 0.2, 0.6],
            "occasional_user": [-0.5, -0.8, -0.2, -0.5, -0.4],
            "diversified": [0.4, 0.5, 0.6, 0.5, 0.4],
        }
        # Covariance: moderate within-persona spread
        persona_latent_cov = np.eye(LATENT_DIM) * 0.3

        l = np.empty((self.n, LATENT_DIM), dtype=np.float64)
        for k in range(self.n_personas):
            mask = z == k
            n_k = mask.sum()
            if n_k == 0:
                continue
            name = self.persona_names[k]
            mu = np.array(persona_latent_mu[name])
            l[mask] = self.rng.multivariate_normal(mu, persona_latent_cov, size=n_k)

        # Decouple latent from persona: mix 50/50 with independent noise
        # so that latent factors are partially independent of observables,
        # genuinely capping model AUC.
        independent_noise = self.rng.standard_normal(l.shape)
        l = 0.7 * l + 0.3 * independent_noise

        # ----------------------------------------------------------------
        # Financial DNA Situation Assignment
        # Situations are INDEPENDENT of persona (pure rng.choice).
        # They modulate BEHAVIORAL PATTERNS (sequences) not static features.
        # Latent modulation here creates a path from situation → label,
        # but the situation itself is invisible in aggregate features.
        # ----------------------------------------------------------------
        n = self.n

        # Engagement axis: steady(0) / surging(1) / declining(2) / volatile(3)
        engagement_sit = self.rng.choice(4, size=n, p=[0.50, 0.20, 0.20, 0.10])

        # Lifecycle axis: stable(0) / growing(1) / consolidating(2) / transitioning(3)
        lifecycle_sit = self.rng.choice(4, size=n, p=[0.50, 0.20, 0.15, 0.15])

        # Value axis: stable_value(0) / ascending(1) / descending(2) / shock(3)
        value_sit = self.rng.choice(4, size=n, p=[0.50, 0.20, 0.15, 0.15])

        # Consumption axis: consistent(0) / exploring(1) / focusing(2) / switching(3)
        consumption_sit = self.rng.choice(4, size=n, p=[0.50, 0.20, 0.15, 0.15])

        # --- Latent modulation (additive, applied after base generation) ---
        # Engagement → activity_level (l[:,1])
        l[engagement_sit == 1, 1] += 0.5   # surging → more active
        l[engagement_sit == 2, 1] -= 0.5   # declining → less active
        volatile_mask = engagement_sit == 3
        n_volatile = volatile_mask.sum()
        if n_volatile > 0:
            l[volatile_mask, 1] += self.rng.choice(
                [-0.3, 0.3], size=n_volatile
            )

        # Lifecycle → loyalty (l[:,4]) and wealth (l[:,0])
        l[lifecycle_sit == 1, 4] += 0.4    # growing → more loyal
        l[lifecycle_sit == 2, 4] -= 0.4    # consolidating → less loyal
        l[lifecycle_sit == 2, 0] -= 0.3    # consolidating → wealth concern

        # Value → wealth_propensity (l[:,0])
        l[value_sit == 1, 0] += 0.5        # ascending → wealth up
        l[value_sit == 2, 0] -= 0.5        # descending → wealth down
        shock_mask = value_sit == 3
        n_shock = shock_mask.sum()
        if n_shock > 0:
            l[shock_mask, 0] += self.rng.choice(
                [-0.8, 0.8], size=n_shock
            )

        # Consumption → risk_tolerance (l[:,2])
        l[consumption_sit == 1, 2] += 0.4  # exploring → higher risk tolerance
        l[consumption_sit == 2, 2] -= 0.4  # focusing → lower risk tolerance

        logger.info(
            "  Situations assigned: engagement dist=%s, lifecycle dist=%s, "
            "value dist=%s, consumption dist=%s",
            np.unique(engagement_sit, return_counts=True)[1].tolist(),
            np.unique(lifecycle_sit, return_counts=True)[1].tolist(),
            np.unique(value_sit, return_counts=True)[1].tolist(),
            np.unique(consumption_sit, return_counts=True)[1].tolist(),
        )

        situations = {
            "sit_engagement": engagement_sit,
            "sit_lifecycle": lifecycle_sit,
            "sit_value": value_sit,
            "sit_consumption": consumption_sit,
        }

        return z, l, situations

    # ==================================================================
    # Layer 2: Observable Profiles (Gaussian Copula)
    # ==================================================================
    def _get_persona_correlation(self, persona_id: int) -> np.ndarray:
        """Get per-persona correlation matrix. Default if not calibrated."""
        calib_corr = self.cal.get('personas', {}).get('correlations', {})
        if str(persona_id) in calib_corr:
            return np.array(calib_corr[str(persona_id)])
        # Default: moderate positive correlations
        # Dimensions: age, income, tenure, num_products, activity
        return np.array([
            [1.0, 0.3, 0.5, 0.2, -0.1],   # age
            [0.3, 1.0, 0.4, 0.3, 0.2],     # income
            [0.5, 0.4, 1.0, 0.3, 0.1],     # tenure
            [0.2, 0.3, 0.3, 1.0, 0.4],     # num_products
            [-0.1, 0.2, 0.1, 0.4, 1.0],    # activity
        ])

    def _generate_profiles(self, z: np.ndarray, l: np.ndarray) -> dict:
        """Generate demographics + product holdings conditioned on persona."""
        demo = self.cal["demographics"]
        n = self.n
        profiles: Dict[str, np.ndarray] = {}

        # --- customer_id ---
        profiles["customer_id"] = np.arange(1, n + 1, dtype=np.int64)

        # --- snapshot_date ---
        # Generate monthly snapshots from a 17-month window
        base_dates = [
            f"2015-{m:02d}-28" for m in range(1, 13)
        ] + [
            f"2016-{m:02d}-28" for m in range(1, 6)
        ]
        date_indices = self.rng.integers(0, len(base_dates), size=n)
        profiles["snapshot_date"] = np.array(
            [base_dates[i] for i in date_indices], dtype="U10"
        )

        # --- Demographics per persona (Gaussian Copula) ---
        # Copula preserves within-persona correlations between
        # age, income, tenure, num_products proxy, and activity proxy.
        age = np.empty(n, dtype=np.float64)
        income = np.empty(n, dtype=np.float64)
        tenure_months = np.empty(n, dtype=np.float64)
        # Copula also yields correlated uniforms for num_products
        # and activity; stored temporarily for downstream use.
        copula_u_num_products = np.empty(n, dtype=np.float64)
        copula_u_activity = np.empty(n, dtype=np.float64)

        for k in range(self.n_personas):
            mask = z == k
            n_k = mask.sum()
            if n_k == 0:
                continue
            name = self.persona_names[k]
            p = demo[name]

            # Per-persona correlation matrix (5D: age, income, tenure,
            # num_products, activity)
            corr = self._get_persona_correlation(k)

            # Generate correlated Gaussians, then map to uniform [0,1]
            Z_corr = self.rng.multivariate_normal(
                np.zeros(corr.shape[0]), corr, size=n_k
            )
            U = norm.cdf(Z_corr)  # correlated uniforms

            # --- Age: inverse CDF via truncated normal ---
            ap = p["age"]
            a_lo = (ap["lo"] - ap["mu"]) / ap["sigma"]
            a_hi = (ap["hi"] - ap["mu"]) / ap["sigma"]
            age[mask] = truncnorm.ppf(
                U[:, 0],
                a=a_lo, b=a_hi,
                loc=ap["mu"], scale=ap["sigma"],
            )

            # --- Income: inverse CDF via lognormal ---
            ip = p["income"]
            raw_income = lognorm.ppf(
                U[:, 1],
                s=ip["sigma"],
                scale=np.exp(ip["mu"]),
            )
            zero_mask_inc = self.rng.random(n_k) < ip.get("zero_rate", 0.0)
            raw_income[zero_mask_inc] = 0.0
            income[mask] = raw_income

            # --- Tenure: inverse CDF via gamma ---
            tp = p["tenure_months"]
            tenure_raw = gamma.ppf(
                U[:, 2],
                a=tp["shape"],
                scale=tp["scale"],
            )
            tenure_months[mask] = np.clip(tenure_raw, 0, 256).astype(int)

            # Store correlated uniforms for downstream modulation
            copula_u_num_products[mask] = U[:, 3]
            copula_u_activity[mask] = U[:, 4]

        # Apply income interaction with latent wealth_propensity
        income_nonzero = income > 0
        income[income_nonzero] *= np.exp(0.3 * l[income_nonzero, 0])

        # ~5% tenure unknown -> sentinel
        tenure_unknown = self.rng.random(n) < 0.05
        tenure_months[tenure_unknown] = -999999

        profiles["age"] = np.round(age).astype(np.int32)
        profiles["income"] = np.round(income, 2)
        profiles["tenure_months"] = tenure_months.astype(np.int32)

        # --- Categorical demographics ---
        # gender
        profiles["gender"] = self.rng.choice(GENDERS, size=n, p=[0.52, 0.48])

        # segment: conditioned on persona
        seg_probs = {
            "conservative_saver": [0.05, 0.60, 0.15, 0.20],
            "active_spender": [0.15, 0.65, 0.10, 0.10],
            "young_digital": [0.02, 0.30, 0.60, 0.08],
            "high_value": [0.40, 0.45, 0.05, 0.10],
            "occasional_user": [0.02, 0.40, 0.35, 0.23],
            "diversified": [0.20, 0.55, 0.10, 0.15],
        }
        segment = np.empty(n, dtype="U20")
        for k in range(self.n_personas):
            mask = z == k
            n_k = mask.sum()
            if n_k == 0:
                continue
            name = self.persona_names[k]
            segment[mask] = self.rng.choice(
                SEGMENTS, size=n_k, p=seg_probs[name]
            )
        profiles["segment"] = segment

        # country: 96% ES
        countries = ["ES"] + [f"C{i:03d}" for i in range(1, 118)]
        country_probs = [0.96] + [0.04 / 117] * 117
        profiles["country"] = self.rng.choice(countries, size=n, p=country_probs)

        # channel: simplified to 163 codes
        channels = [f"CH{i:03d}" for i in range(163)]
        ch_probs = np.ones(163) / 163
        profiles["channel"] = self.rng.choice(channels, size=n, p=ch_probs)

        # is_active: conditioned on persona
        active_rates = {
            "conservative_saver": 0.20,
            "active_spender": 0.65,
            "young_digital": 0.55,
            "high_value": 0.50,
            "occasional_user": 0.15,
            "diversified": 0.45,
        }
        is_active = np.zeros(n, dtype=np.int32)
        for k in range(self.n_personas):
            mask = z == k
            n_k = mask.sum()
            if n_k == 0:
                continue
            name = self.persona_names[k]
            rate = active_rates[name]
            # Modulate by latent activity_level
            adjusted = _sigmoid(
                np.log(rate / (1 - rate)) + 0.5 * l[mask, 1]
            )
            is_active[mask] = (self.rng.random(n_k) < adjusted).astype(np.int32)
        profiles["is_active"] = is_active

        # --- age_group, income_group (derived categoricals) ---
        age_arr = profiles["age"]
        age_group = np.where(
            age_arr < 25, "young",
            np.where(age_arr < 40, "adult",
                     np.where(age_arr < 55, "middle",
                              np.where(age_arr < 70, "senior", "elderly")))
        )
        profiles["age_group"] = age_group

        inc = profiles["income"]
        income_group = np.where(
            inc <= 0, "unknown",
            np.where(inc < 30000, "low",
                     np.where(inc < 80000, "mid",
                              np.where(inc < 200000, "high", "very_high")))
        )
        profiles["income_group"] = income_group

        # --- Product holdings: 24 binary ---
        # Base rates per product (approximate from schema comments)
        base_product_rates = np.array([
            0.003, 0.001, 0.60, 0.002, 0.08,
            0.004, 0.01, 0.11, 0.04, 0.001,
            0.001, 0.03, 0.08, 0.02, 0.003,
            0.01, 0.002, 0.05, 0.04, 0.02,
            0.001, 0.05, 0.06, 0.12,
        ])
        # Persona multipliers: some personas more likely to hold certain products
        persona_product_mult = {
            "conservative_saver": np.array([
                2.0, 0.5, 0.8, 0.3, 0.5, 0.3, 0.5, 0.8, 3.0, 2.0,
                2.0, 0.5, 0.3, 0.2, 1.5, 0.3, 0.5, 0.3, 0.5, 0.5,
                0.5, 2.0, 0.5, 0.3,
            ]),
            "active_spender": np.array([
                0.8, 0.8, 1.2, 1.0, 1.5, 0.5, 1.0, 1.2, 0.8, 0.8,
                0.8, 1.2, 1.0, 0.8, 0.5, 1.2, 0.8, 2.0, 1.0, 1.0,
                1.0, 0.8, 1.5, 2.0,
            ]),
            "young_digital": np.array([
                0.3, 0.2, 1.0, 0.3, 0.3, 2.0, 0.5, 0.5, 0.3, 0.2,
                0.2, 2.0, 0.5, 0.1, 0.1, 0.5, 0.2, 1.5, 0.3, 1.5,
                0.5, 0.2, 1.0, 1.0,
            ]),
            "high_value": np.array([
                1.5, 1.5, 1.0, 1.5, 1.2, 0.3, 1.5, 1.5, 1.5, 1.5,
                1.5, 1.0, 2.5, 2.0, 2.0, 1.0, 1.5, 1.5, 2.5, 1.0,
                1.0, 1.5, 1.0, 1.0,
            ]),
            "occasional_user": np.array([
                0.5, 0.3, 0.7, 0.2, 0.3, 0.3, 0.3, 0.5, 0.5, 0.3,
                0.3, 0.5, 0.3, 0.2, 0.2, 0.3, 0.3, 0.5, 0.3, 0.5,
                0.3, 0.3, 0.3, 0.3,
            ]),
            "diversified": np.array([
                1.2, 1.0, 1.1, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.0,
                1.0, 1.0, 1.0, 1.0,
            ]),
        }

        prod_cols = {}
        for k in range(self.n_personas):
            mask = z == k
            n_k = mask.sum()
            if n_k == 0:
                continue
            name = self.persona_names[k]
            mult = persona_product_mult[name]
            rates = np.clip(base_product_rates * mult, 0.0001, 0.95)
            # Modulate by latent wealth_propensity for investment products
            # and activity_level for transaction products
            wealth_mod = _sigmoid(0.3 * l[mask, 0])  # n_k
            activity_mod = _sigmoid(0.3 * l[mask, 1])  # n_k
            for p_idx in range(N_PRODUCTS):
                if f"prod_{PRODUCT_NAMES[p_idx]}" not in prod_cols:
                    prod_cols[f"prod_{PRODUCT_NAMES[p_idx]}"] = np.zeros(n, dtype=np.int32)
                # Investment products modulated by wealth
                if p_idx in [8, 9, 10, 12, 14, 18]:  # deposits, funds, pension, securities
                    adj_rate = rates[p_idx] * (0.5 + wealth_mod)
                # Transaction products modulated by activity
                elif p_idx in [2, 17, 22, 23]:  # checking, credit_card, direct/auto debit
                    adj_rate = rates[p_idx] * (0.5 + activity_mod)
                else:
                    adj_rate = np.full(n_k, rates[p_idx])
                adj_rate = np.clip(adj_rate, 0.0001, 0.95)
                prod_cols[f"prod_{PRODUCT_NAMES[p_idx]}"][mask] = (
                    self.rng.random(n_k) < adj_rate
                ).astype(np.int32)

        profiles.update(prod_cols)

        # num_products
        num_products = np.zeros(n, dtype=np.int32)
        for pn in PRODUCT_NAMES:
            num_products += profiles[f"prod_{pn}"]
        profiles["num_products"] = num_products

        return profiles

    # ==================================================================
    # Layer 3: Transaction Sequences (DuckDB vectorized)
    # ==================================================================
    def _generate_transactions(
        self, z: np.ndarray, l: np.ndarray, profiles: dict,
        situations: dict,
    ) -> dict:
        """Generate transaction sequences and synth_* aggregates via DuckDB.

        Replaces the original Python-loop implementation with pure SQL to
        reduce RAM from 20 GB+ to ~8 GB (disk-spill capable) and runtime
        from ~20 min to ~30 s for 1M customers.
        """
        import duckdb
        import pyarrow as pa

        txn_cal = self.cal["transactions"]
        n = self.n

        # Collect all MCC codes (top-50 for sequence encoding)
        all_mccs = sorted(set(int(m) for m in self.cal["personas"]["mcc_codes"]))
        top_mccs = all_mccs[:50] if len(all_mccs) >= 50 else all_mccs
        n_top_mccs = len(top_mccs)

        # Per-persona monthly txn lambda
        persona_lambda = {
            "conservative_saver": 12.0,
            "active_spender": 30.0,
            "young_digital": 22.0,
            "high_value": 18.0,
            "occasional_user": 8.0,
            "diversified": 25.0,
        }

        # Build base_lambda per customer (persona lambda * latent activity mod)
        base_lambda = np.empty(n, dtype=np.float64)
        for k in range(self.n_personas):
            mask = z == k
            name = self.persona_names[k]
            lam = persona_lambda[name]
            activity_mod = np.exp(0.3 * l[mask, 1])
            base_lambda[mask] = lam * activity_mod

        # ------------------------------------------------------------------
        # Vectorized LIST generation (no row explosion)
        # Each customer = 1 row, sequences stored as Python lists → Arrow LIST
        # ------------------------------------------------------------------
        MAX_TXN = min(int(np.max(base_lambda) * 12 * 1.5), TXN_SEQ_MAX_LEN)
        logger.info("  Generating txn sequences: n=%d, max_txn_per_customer=%d", n, MAX_TXN)

        # Pre-generate fixed-size matrices, then truncate to variable-length lists
        # Shape: (n, MAX_TXN) — ~1M × 400 × 4B = ~1.6GB
        _total_txns = np.clip((base_lambda * 12).astype(int), 1, MAX_TXN)

        # MCC indices: persona-based weighted distribution
        _cust_ids = np.arange(n, dtype=np.int64)
        _mcc_matrix = np.zeros((n, MAX_TXN), dtype=np.int32)
        _amt_matrix = np.zeros((n, MAX_TXN), dtype=np.float32)
        _hour_matrix = np.zeros((n, MAX_TXN), dtype=np.int32)
        _offset_matrix = np.zeros((n, MAX_TXN), dtype=np.int32)

        # Persona → MCC preference weights (index over top_mccs slots 0..n_top_mccs-1)
        # Groups map to index ranges in the sorted top_mccs list:
        #   0-5:  grocery/food basics  (MCC ~5411,5499,5300,5310)
        #   6-10: utilities/fuel       (MCC ~4900,5541,5533,4814)
        #   11-15: dining/restaurants  (MCC ~5812,5813,5814,5815)
        #   16-25: retail/clothing     (MCC ~5311,5651,5621,5661,5712,5719,5722)
        #   26-30: entertainment       (MCC ~5816,7832,7922,7995,7996)
        #   31-35: online/digital      (MCC ~5045,5094,5192,5193,4899)
        #   36-40: subscriptions/svcs  (MCC ~7210,7230,7276,7349,7393)
        #   41-45: travel/airline      (MCC ~3000-3132 airlines, 4511,4722)
        #   46-48: luxury/finance      (MCC ~6300,8111,8931)
        #   49:    miscellaneous       (last slot)
        def _make_mcc_weights(n_mccs: int, persona_name: str) -> np.ndarray:
            """Build a normalized MCC weight vector for a given persona."""
            w = np.ones(n_mccs, dtype=np.float64)
            # Clamp index ranges to available MCC slots
            def _boost(lo: int, hi: int, factor: float) -> None:
                lo_c, hi_c = min(lo, n_mccs), min(hi, n_mccs)
                if lo_c < hi_c:
                    w[lo_c:hi_c] *= factor

            if persona_name == "conservative_saver":
                _boost(0, 6, 10.0)  # grocery (dominant)
                _boost(6, 11, 8.0)  # utilities/fuel
                _boost(11, 16, 2.0) # some dining
                _boost(26, 49, 0.1) # minimal entertainment/digital/travel
            elif persona_name == "active_spender":
                _boost(11, 16, 10.0) # dining (dominant)
                _boost(16, 26, 8.0)  # retail
                _boost(26, 31, 5.0)  # entertainment
                _boost(0, 6, 1.5)    # some grocery
            elif persona_name == "young_digital":
                _boost(31, 36, 12.0) # online/digital (dominant)
                _boost(36, 41, 9.0)  # subscriptions
                _boost(26, 31, 5.0)  # entertainment/gaming
                _boost(11, 16, 2.0)  # dining
                _boost(0, 11, 0.1)   # minimal grocery/utilities
            elif persona_name == "high_value":
                _boost(41, 46, 12.0) # travel/airline (dominant)
                _boost(46, 49, 10.0) # luxury/finance
                _boost(16, 26, 2.0)  # retail
                _boost(11, 16, 1.5)  # dining
            elif persona_name == "occasional_user":
                _boost(0, 6, 12.0)   # concentrated grocery (dominant)
                _boost(6, 9, 8.0)    # fuel
                _boost(9, 49, 0.1)   # minimal everything else
            elif persona_name == "diversified":
                # Roughly uniform with moderate retail tilt
                _boost(16, 26, 2.5) # retail
            # Normalize to probability distribution
            return w / w.sum()

        # Sticky MCC probability: with this prob a transaction repeats the previous MCC
        STICKY_PROB = 0.60

        # Vectorized generation per persona (bulk random)
        for pid in range(self.n_personas):
            mask = (z == pid)
            n_p = mask.sum()
            if n_p == 0:
                continue
            pname = self.persona_names[pid]
            mcc_weights = _make_mcc_weights(n_top_mccs, pname)

            # Draw base MCC choices using persona weights
            base_mccs = self.rng.choice(
                n_top_mccs, size=(n_p, MAX_TXN), p=mcc_weights
            ).astype(np.int32)

            # Apply fixed stickiness per persona (no latent coupling to avoid leakage
            # through aggregate MCC features)
            sticky_mask = self.rng.random(size=(n_p, MAX_TXN)) < STICKY_PROB
            sticky_mask[:, 0] = False  # first txn always uses base choice
            for t in range(1, MAX_TXN):
                base_mccs[sticky_mask[:, t], t] = base_mccs[sticky_mask[:, t], t - 1]

            _mcc_matrix[mask] = base_mccs

            # Persona + latent-dependent amount distribution (log-normal)
            # Latent wealth (l[:,0]) shifts mean, risk_tolerance (l[:,2]) shifts variance.
            # This creates individual-level spending patterns that Phase 0 generators
            # (Mamba, TDA, GMM) can encode but raw aggregate features cannot.
            amt_base_params = {
                "conservative_saver": (2.5, 0.8),
                "active_spender":     (3.5, 1.0),
                "young_digital":      (2.8, 0.9),
                "high_value":         (4.0, 1.2),
                "occasional_user":    (2.3, 0.7),
                "diversified":        (3.2, 1.0),
            }
            base_mu, base_sigma = amt_base_params.get(pname, (3.0, 1.0))
            # Persona-only amount distribution (no latent coupling to avoid aggregate leakage)
            customer_sigma = np.clip(base_sigma, 0.3, 2.0)  # safety bound
            # Generate amounts with persona-level distribution (uniform per persona)
            _amt_matrix[mask] = np.exp(
                self.rng.normal(
                    base_mu,
                    customer_sigma,
                    size=(mask.sum(), MAX_TXN),
                )
            ).astype(np.float32)

            # Persona-dependent hour distribution (weighted, not uniform)
            hour_weights = {
                "conservative_saver": [0.01]*6 + [0.05]*3 + [0.08]*4 + [0.06]*5 + [0.02]*6,
                "active_spender":     [0.01]*6 + [0.03]*3 + [0.06]*4 + [0.08]*5 + [0.05]*6,
                "young_digital":      [0.03]*6 + [0.02]*3 + [0.04]*4 + [0.06]*5 + [0.08]*6,
                "high_value":         [0.01]*6 + [0.04]*3 + [0.07]*4 + [0.07]*5 + [0.03]*6,
                "occasional_user":    [0.01]*6 + [0.06]*3 + [0.09]*4 + [0.05]*5 + [0.01]*6,
                "diversified":        [0.02]*6 + [0.04]*3 + [0.06]*4 + [0.06]*5 + [0.04]*6,
            }
            hw = np.array(hour_weights.get(pname, [1/24]*24), dtype=np.float64)
            hw = hw / hw.sum()
            _hour_matrix[mask] = self.rng.choice(
                24, size=(n_p, MAX_TXN), p=hw
            ).astype(np.int32)
            # Day offsets: spread across 360 days
            _offset_matrix[mask] = np.sort(
                self.rng.integers(0, 360, size=(n_p, MAX_TXN)), axis=1
            )[:, ::-1]  # descending (most recent first)

        # ------------------------------------------------------------------
        # DNA Situation modulations (applied to matrices BEFORE list conversion)
        # These change temporal PATTERNS visible to Mamba/TDA/GMM but not
        # to XGBoost aggregate features (avg_amount, synth_unique_mcc, etc.)
        # ------------------------------------------------------------------
        engagement_sit = situations["sit_engagement"]
        lifecycle_sit = situations["sit_lifecycle"]  # noqa: F841 — used in product seq below
        value_sit = situations["sit_value"]
        consumption_sit = situations["sit_consumption"]

        logger.info("  Applying DNA situation modulations to sequence matrices...")

        # --- Engagement: modulate temporal spacing of transactions ---
        # "surging" (1): cluster more txns in last 90 days (days 270-360)
        # "declining" (2): cluster more txns in first 90 days (days 0-90)
        # "volatile" (3): alternate high/low 30-day blocks

        # Surging: resample recent portion with 2× density
        surging_idx = np.where(engagement_sit == 1)[0]
        for i in surging_idx:
            t = _total_txns[i]
            if t < 4:
                continue
            # Split: ~1/3 recent (days 270-360), ~2/3 older (days 0-270)
            n_recent = max(1, int(t * 0.50))
            n_older = t - n_recent
            recent_days = np.sort(self.rng.integers(270, 360, size=n_recent))
            older_days = np.sort(self.rng.integers(0, 270, size=n_older))
            merged = np.concatenate([older_days, recent_days])
            _offset_matrix[i, :t] = merged[::-1]  # descending (most recent first)

        # Declining: cluster more txns in first 90 days
        declining_idx = np.where(engagement_sit == 2)[0]
        for i in declining_idx:
            t = _total_txns[i]
            if t < 4:
                continue
            n_early = max(1, int(t * 0.50))
            n_later = t - n_early
            early_days = np.sort(self.rng.integers(0, 90, size=n_early))
            later_days = np.sort(self.rng.integers(90, 360, size=n_later))
            merged = np.concatenate([early_days, later_days])
            _offset_matrix[i, :t] = merged[::-1]  # descending

        # Volatile: alternating 30-day blocks of high/low density
        volatile_idx = np.where(engagement_sit == 3)[0]
        for i in volatile_idx:
            t = _total_txns[i]
            if t < 6:
                continue
            # 12 blocks of 30 days; even blocks = high density, odd = low
            # Assign each txn to a block based on day range
            days_flat = np.zeros(t, dtype=np.int32)
            high_budget = max(1, int(t * 0.65))
            low_budget = t - high_budget
            # High-density blocks: 0-30, 60-90, 120-150, 180-210, 240-270, 300-330
            high_day_ranges = [(0, 30), (60, 90), (120, 150), (180, 210), (240, 270), (300, 330)]
            low_day_ranges  = [(30, 60), (90, 120), (150, 180), (210, 240), (270, 300), (330, 360)]
            high_days_pool = np.concatenate([
                self.rng.integers(lo, hi, size=max(1, high_budget // 6))
                for lo, hi in high_day_ranges
            ])
            low_days_pool = np.concatenate([
                self.rng.integers(lo, hi, size=max(1, low_budget // 6))
                for lo, hi in low_day_ranges
            ])
            all_days = np.concatenate([high_days_pool, low_days_pool])
            if len(all_days) >= t:
                chosen = np.sort(self.rng.choice(all_days, size=t, replace=False))
            else:
                # pad with random days if pool too small
                extra = self.rng.integers(0, 360, size=t - len(all_days))
                chosen = np.sort(np.concatenate([all_days, extra]))
            _offset_matrix[i, :t] = chosen[::-1]  # descending

        # --- Value: apply amount TREND multiplier per transaction position ---
        # "ascending" (1): multiplier = 0.7 + 0.6*(pos/total)  [0.7 → 1.3]
        # "descending" (2): multiplier = 1.3 - 0.6*(pos/total)  [1.3 → 0.7]
        # "shock" (3): multiplier = 1.0 until random midpoint, then 1.5x or 0.5x
        # Key: average amount stays similar → synth_avg_amount unchanged
        ascending_idx = np.where(value_sit == 1)[0]
        for i in ascending_idx:
            t = _total_txns[i]
            if t < 2:
                continue
            pos = np.arange(t, dtype=np.float32) / max(t - 1, 1)
            multiplier = (0.7 + 0.6 * pos).astype(np.float32)
            _amt_matrix[i, :t] *= multiplier

        descending_idx = np.where(value_sit == 2)[0]
        for i in descending_idx:
            t = _total_txns[i]
            if t < 2:
                continue
            pos = np.arange(t, dtype=np.float32) / max(t - 1, 1)
            multiplier = (1.3 - 0.6 * pos).astype(np.float32)
            _amt_matrix[i, :t] *= multiplier

        shock_idx = np.where(value_sit == 3)[0]
        for i in shock_idx:
            t = _total_txns[i]
            if t < 4:
                continue
            # Random midpoint in [25%, 75%] of sequence
            midpoint = self.rng.integers(t // 4, max(t // 4 + 1, 3 * t // 4))
            # 50% chance of upward shock (1.5x), 50% downward (0.5x)
            shock_mult = 1.5 if self.rng.random() < 0.5 else 0.5
            _amt_matrix[i, midpoint:t] *= shock_mult
            # Re-normalize post-shock portion to keep average similar:
            # pre-shock mean + post-shock mean should stay ~= original mean
            # We skip renormalization here to preserve the pattern signal;
            # synth_avg_amount is computed AFTER this and WILL reflect it slightly,
            # but the TREND is what matters for Mamba.

        # --- Consumption: MCC distribution shift in second half ---
        # "exploring" (1): reduce top-category weight by 50%, spread to others
        # "focusing"  (2): double top-category weight
        # "switching" (3): swap weights of top-2 categories
        # We modify the second half of _mcc_matrix in-place.
        exploring_idx = np.where(consumption_sit == 1)[0]
        focusing_idx  = np.where(consumption_sit == 2)[0]
        switching_idx = np.where(consumption_sit == 3)[0]

        for i in exploring_idx:
            t = _total_txns[i]
            if t < 4:
                continue
            half = t // 2
            second_half = _mcc_matrix[i, half:t]
            if len(second_half) == 0:
                continue
            # Find the dominant MCC in second half
            counts = np.bincount(second_half, minlength=n_top_mccs)
            top_mcc = int(counts.argmax())
            # Replace 50% of top-mcc occurrences with random other MCCs
            top_positions = np.where(second_half == top_mcc)[0]
            n_replace = max(1, len(top_positions) // 2)
            replace_pos = self.rng.choice(top_positions, size=n_replace, replace=False)
            # Draw replacement MCCs excluding the top_mcc
            other_mccs = [m for m in range(n_top_mccs) if m != top_mcc]
            new_mccs = self.rng.choice(other_mccs, size=n_replace)
            _mcc_matrix[i, half + replace_pos] = new_mccs

        for i in focusing_idx:
            t = _total_txns[i]
            if t < 4:
                continue
            half = t // 2
            second_half = _mcc_matrix[i, half:t]
            if len(second_half) == 0:
                continue
            counts = np.bincount(second_half, minlength=n_top_mccs)
            top_mcc = int(counts.argmax())
            # Replace non-top MCCs with top_mcc at 50% rate
            non_top_pos = np.where(second_half != top_mcc)[0]
            if len(non_top_pos) == 0:
                continue
            n_replace = max(1, len(non_top_pos) // 2)
            replace_pos = self.rng.choice(non_top_pos, size=n_replace, replace=False)
            _mcc_matrix[i, half + replace_pos] = top_mcc

        for i in switching_idx:
            t = _total_txns[i]
            if t < 4:
                continue
            half = t // 2
            # Find top-2 MCCs in FIRST half to determine what to swap
            first_half = _mcc_matrix[i, :half]
            if len(first_half) < 2:
                continue
            counts_first = np.bincount(first_half, minlength=n_top_mccs)
            top2 = np.argsort(counts_first)[::-1][:2]
            if len(top2) < 2:
                continue
            mcc_a, mcc_b = int(top2[0]), int(top2[1])
            # In second half: replace mcc_a with mcc_b and mcc_b with mcc_a
            second_half = _mcc_matrix[i, half:t].copy()
            second_half[second_half == mcc_a] = n_top_mccs  # temp sentinel
            second_half[second_half == mcc_b] = mcc_a
            second_half[second_half == n_top_mccs] = mcc_b
            _mcc_matrix[i, half:t] = second_half

        logger.info("  DNA situation modulations applied.")

        # Convert to variable-length Python lists (ragged → LIST column)
        logger.info("  Converting to ragged LIST columns...")
        txn_amount_seq = [_amt_matrix[i, :_total_txns[i]].round(2).tolist() for i in range(n)]
        txn_mcc_seq = [_mcc_matrix[i, :_total_txns[i]].tolist() for i in range(n)]
        txn_hour_seq = [_hour_matrix[i, :_total_txns[i]].tolist() for i in range(n)]
        txn_day_offset_seq = [_offset_matrix[i, :_total_txns[i]].tolist() for i in range(n)]

        # Scalar aggregates (vectorized, no loop)
        total_txns_arr = _total_txns
        avg_amount_arr = np.array([_amt_matrix[i, :_total_txns[i]].mean() if _total_txns[i] > 0 else 0 for i in range(n)], dtype=np.float32)
        total_spend_arr = np.array([_amt_matrix[i, :_total_txns[i]].sum() for i in range(n)], dtype=np.float32)
        unique_mcc_arr = np.array([len(set(_mcc_matrix[i, :_total_txns[i]])) for i in range(n)], dtype=np.int32)

        # Compute time-of-day ratios from actual hour sequences (before deleting matrices)
        _morning = np.zeros(n, dtype=np.float64)   # 6-11
        _afternoon = np.zeros(n, dtype=np.float64)  # 12-17
        _evening = np.zeros(n, dtype=np.float64)    # 18-23
        _night = np.zeros(n, dtype=np.float64)      # 0-5
        for i in range(n):
            t = _total_txns[i]
            if t > 0:
                hours = _hour_matrix[i, :t]
                _morning[i] = np.mean((hours >= 6) & (hours < 12))
                _afternoon[i] = np.mean((hours >= 12) & (hours < 18))
                _evening[i] = np.mean((hours >= 18))
                _night[i] = np.mean(hours < 6)

        del _mcc_matrix, _amt_matrix, _hour_matrix, _offset_matrix
        logger.info("  Transaction sequences generated (no row explosion)")

        # ------------------------------------------------------------------
        # Compute synth aggregates from vectorized arrays (no DuckDB needed)
        # ------------------------------------------------------------------
        synth_monthly_txns = np.maximum(total_txns_arr // 12, 1).astype(np.int32)
        synth_avg_amount = avg_amount_arr.astype(np.float64)
        synth_monthly_spend = (total_spend_arr / 12.0).astype(np.float64)
        synth_unique_mcc = unique_mcc_arr.astype(np.int32)
        synth_unique_merchants = np.minimum(unique_mcc_arr + self.rng.integers(3, 15, size=n), 33).astype(np.int32)
        synth_frequency = total_txns_arr.astype(np.int32)
        synth_monetary = total_spend_arr.astype(np.float64)
        # Time-of-day ratios: computed from actual hour sequences (above)
        synth_morning_ratio = _morning
        synth_afternoon_ratio = _afternoon
        synth_evening_ratio = _evening
        synth_night_ratio = _night
        synth_recency_days = self.rng.uniform(1, 30, size=n).astype(np.float64)
        cv = np.maximum(np.abs(total_txns_arr / 12.0 - base_lambda) / np.maximum(base_lambda, 1), 0.01)
        synth_stability = (1.0 / cv).astype(np.float64)
        synth_fraud_ratio = self.rng.uniform(0, 0.005, size=n).astype(np.float64)
        txn_date_seq = txn_day_offset_seq  # alias

        logger.info("  Vectorized txn generation done for %d customers", n)

        # ------------------------------------------------------------------
        # Product sequences (16 months of holdings) — kept in numpy
        # ------------------------------------------------------------------
        total_acquisitions = np.zeros(n, dtype=np.int64)
        total_churns = np.zeros(n, dtype=np.int64)
        months_observed = np.zeros(n, dtype=np.int64)
        product_diversity = np.zeros(n, dtype=np.int64)

        prod_sequences = {
            f"seq_{pn}": [None] * n for pn in PRODUCT_NAMES
        }
        seq_num_products_all = [None] * n
        seq_acquisitions_all = [None] * n
        seq_churns_all = [None] * n

        for k in range(self.n_personas):
            mask_idx = np.where(z == k)[0]
            n_k = len(mask_idx)
            if n_k == 0:
                continue
            name = self.persona_names[k]

            chunk_size = min(50000, n_k)
            for chunk_start in range(0, n_k, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_k)
                idx = mask_idx[chunk_start:chunk_end]
                nc = len(idx)

                for ci in range(nc):
                    cust_idx = idx[ci]
                    num_prods_now = profiles["num_products"][cust_idx]

                    n_obs = self.rng.integers(8, N_SEQ_MONTHS + 1)
                    months_observed[cust_idx] = n_obs

                    current_holdings = np.array(
                        [profiles[f"prod_{pn}"][cust_idx] for pn in PRODUCT_NAMES],
                        dtype=np.int32,
                    )
                    tot_acq = 0
                    tot_churn = 0
                    prod_div_set = set()

                    # Lifecycle situation modulates acquisition/churn probabilities.
                    # Base churn_prob=0.05; base acquisition is implicit via held=True path.
                    lc_sit = lifecycle_sit[cust_idx]
                    # acquisition_prob: prob that a held product was acquired mid-sequence
                    # (vs held from start).  Base is implicit (acq_month ~ Uniform[0, n_obs]).
                    # We modulate by biasing acq_month to be smaller (later acquisition)
                    # for growing and larger (earlier) for consolidating.
                    # churn_prob: prob that a not-held product had a churn event
                    if lc_sit == 1:      # growing: acquisitions > churns
                        churn_prob = 0.02     # less churn
                        # Bias acquisition to recent half: acq_month in [n_obs//2, n_obs)
                        acq_recent_bias = True
                        acq_early_bias = False
                    elif lc_sit == 2:    # consolidating: churns > acquisitions
                        churn_prob = 0.12     # more churn
                        acq_recent_bias = False
                        acq_early_bias = True  # acquisitions happened earlier
                    elif lc_sit == 3:    # transitioning: both high
                        churn_prob = 0.10
                        acq_recent_bias = True
                        acq_early_bias = False
                    else:                # stable: base rates
                        churn_prob = 0.05
                        acq_recent_bias = False
                        acq_early_bias = False

                    for pn_idx, pn in enumerate(PRODUCT_NAMES):
                        seq = np.full(N_SEQ_MONTHS, -1, dtype=np.int32)
                        start = N_SEQ_MONTHS - n_obs
                        held = current_holdings[pn_idx]
                        if held:
                            # Modulate acquisition month based on lifecycle
                            if acq_recent_bias and n_obs >= 4:
                                # Growing: acquired in recent half
                                lo_m = max(0, n_obs // 2)
                                acq_month = self.rng.integers(lo_m, max(lo_m + 1, n_obs))
                            elif acq_early_bias and n_obs >= 4:
                                # Consolidating: acquired early
                                hi_m = max(1, n_obs // 2)
                                acq_month = self.rng.integers(0, hi_m)
                            else:
                                acq_month = self.rng.integers(0, max(n_obs, 1))
                            for m in range(start, N_SEQ_MONTHS):
                                rel_m = m - start
                                seq[m] = 1 if rel_m >= acq_month else 0
                            if acq_month > 0:
                                tot_acq += 1
                                prod_div_set.add(pn_idx)
                        else:
                            if self.rng.random() < churn_prob:
                                churn_m = self.rng.integers(0, max(n_obs - 1, 1))
                                for m in range(start, N_SEQ_MONTHS):
                                    rel_m = m - start
                                    seq[m] = 1 if rel_m < churn_m else 0
                                tot_churn += 1
                            else:
                                for m in range(start, N_SEQ_MONTHS):
                                    seq[m] = 0
                        prod_sequences[f"seq_{pn}"][cust_idx] = seq.tolist()

                    total_acquisitions[cust_idx] = tot_acq
                    total_churns[cust_idx] = tot_churn
                    product_diversity[cust_idx] = len(prod_div_set) + num_prods_now

                    np_seq = np.zeros(N_SEQ_MONTHS, dtype=np.int32)
                    acq_seq = np.zeros(N_SEQ_MONTHS, dtype=np.int32)
                    ch_seq = np.zeros(N_SEQ_MONTHS, dtype=np.int32)
                    for m in range(N_SEQ_MONTHS):
                        cnt = 0
                        for pn in PRODUCT_NAMES:
                            s = prod_sequences[f"seq_{pn}"][cust_idx]
                            if s[m] == 1:
                                cnt += 1
                                if m > 0 and s[m - 1] == 0:
                                    acq_seq[m] += 1
                            elif s[m] == 0 and m > 0:
                                prev = prod_sequences[f"seq_{pn}"][cust_idx]
                                if prev[m - 1] == 1:
                                    ch_seq[m] += 1
                        np_seq[m] = cnt
                    seq_num_products_all[cust_idx] = np_seq.tolist()
                    seq_acquisitions_all[cust_idx] = acq_seq.tolist()
                    seq_churns_all[cust_idx] = ch_seq.tolist()

            logger.info(
                "  Persona %s (%d customers) product-seq done", name, n_k
            )

        # --- Package results ---
        result = {
            "synth_monthly_txns": synth_monthly_txns,
            "synth_avg_amount": np.round(synth_avg_amount, 2),
            "synth_monthly_spend": np.round(synth_monthly_spend, 2),
            "synth_unique_mcc": synth_unique_mcc,
            "synth_unique_merchants": synth_unique_merchants,
            "synth_morning_ratio": synth_morning_ratio,
            "synth_afternoon_ratio": synth_afternoon_ratio,
            "synth_evening_ratio": synth_evening_ratio,
            "synth_night_ratio": synth_night_ratio,
            "synth_recency_days": synth_recency_days,
            "synth_frequency": synth_frequency,
            "synth_monetary": synth_monetary,
            "synth_stability": synth_stability,
            "synth_fraud_ratio": synth_fraud_ratio,
            "total_acquisitions": total_acquisitions,
            "total_churns": total_churns,
            "months_observed": months_observed,
            "product_diversity": product_diversity,
            "txn_amount_seq": txn_amount_seq,
            "txn_mcc_seq": txn_mcc_seq,
            "txn_hour_seq": txn_hour_seq,
            "txn_day_offset_seq": txn_day_offset_seq,
            "txn_date_seq": txn_date_seq,
        }
        # Product sequences
        result.update(prod_sequences)
        result["seq_num_products"] = seq_num_products_all
        result["seq_acquisitions"] = seq_acquisitions_all
        result["seq_churns"] = seq_churns_all

        return result

    # ==================================================================
    # Layer 4: Labels with Variance Budget
    # ==================================================================
    def _generate_labels(
        self, z: np.ndarray, l: np.ndarray, features: dict
    ) -> dict:
        """Generate 18 task labels with controllable AUC ceilings."""
        n = self.n
        labels: Dict[str, np.ndarray] = {}

        # Pre-compute normalized features for reuse
        nf = {}
        for key in [
            "num_products", "tenure_months", "synth_monthly_spend",
            "synth_monthly_txns", "synth_avg_amount", "synth_unique_mcc",
            "synth_frequency", "synth_monetary", "synth_stability",
            "income", "age", "is_active", "synth_recency_days",
            "total_acquisitions", "total_churns", "product_diversity",
        ]:
            arr = features[key].astype(np.float64)
            nf[key] = _normalize(arr)

        # Income quantile (handling zeros)
        inc = features["income"].astype(np.float64)
        inc_valid = inc.copy()
        inc_valid[inc_valid <= 0] = np.nan
        nf["income_q"] = _quantile_rank(np.nan_to_num(inc_valid, nan=0.0))

        # Persona bias helper
        def _persona_bias(task_name: str, bias_dict: dict) -> np.ndarray:
            """Per-persona logit bias."""
            out = np.zeros(n, dtype=np.float64)
            for k_idx in range(self.n_personas):
                mask = z == k_idx
                pname = self.persona_names[k_idx]
                out[mask] = bias_dict.get(pname, 0.0)
            return out

        noise = lambda std: std * self.rng.standard_normal(n)

        # ================================================================
        # Tier 1 — Easy: segment, income_tier, tenure_stage
        # obs=60%, latent=20%, noise=20%
        # ================================================================

        # --- segment_prediction (multiclass 4) ---
        # Derived directly from segment string in profiles
        seg_map = {"01-TOP": 0, "02-PARTICULARES": 1, "03-UNIVERSITARIO": 2, "UNKNOWN": 3}
        labels["label_segment"] = np.array(
            [seg_map.get(s, 3) for s in features["segment"]], dtype=np.int64
        )

        # --- income_tier (multiclass 4) ---
        inc_arr = features["income"].astype(np.float64)
        labels["label_income_tier"] = np.where(
            inc_arr <= 0, 0,  # low (missing treated as low)
            np.where(inc_arr < 30000, 0,
                     np.where(inc_arr < 80000, 1,
                              np.where(inc_arr < 200000, 2, 3)))
        ).astype(np.int64)

        # --- tenure_stage (multiclass 5) ---
        ten = features["tenure_months"].astype(np.float64)
        labels["label_tenure_stage"] = np.where(
            ten < 0, 0,  # unknown_or_new (sentinel)
            np.where(ten < 6, 0,
                     np.where(ten < 24, 1,
                              np.where(ten < 60, 2,
                                       np.where(ten < 120, 3, 4))))
        ).astype(np.int64)

        # ================================================================
        # Label Generation — High-Order Continuous Multiplicative Interactions
        # ================================================================
        # Each task's obs signal contains:
        #   L0: Linear combination of features (base terms)
        #   High-order continuous multiplicative interactions (3rd–5th order)
        #
        # Interaction orders per task:
        #   has_nba            : 3rd order  (income × num_products_inv × activity)
        #   churn_signal       : 4th order  (inactivity × recency × low-freq × short-tenure)
        #   product_stability  : 4th order  (stability × tenure × products × activity)
        #   will_acquire_deposits    : 3rd order  (income × tenure × stability)
        #   will_acquire_investments : 5th order  (income × spend × tenure × products × stability)
        #   will_acquire_accounts    : 3rd order  (product_gap × activity × tenure)
        #   will_acquire_lending     : 4th order  (spend × low_income × frequency × instability)
        #   will_acquire_payments    : 4th order  (frequency × products × activity × spending)
        #
        # Key principle: raw continuous values (no threshold .astype(float))
        # XGBoost needs O(2^k) splits for k-th order; neural nets capture in one layer.
        # ================================================================

        # ================================================================
        # Tier 1 — Core: has_nba, churn_signal, product_stability
        # Medium: obs=45%, latent=30%, noise=25%
        # ================================================================

        # --- _has_nba (internal gate, NOT a prediction task) ---
        # Determines which customers have product recommendations.
        # Previously a binary task; now folded into nba_primary (class 0 = no NBA).
        # target_pos_rate=0.15 → ~15% of customers will have NBA recommendations.
        obs_nba = (
            0.3 * nf["num_products"]
            + 0.2 * nf["synth_monthly_spend"]
            + 0.15 * nf["is_active"]
            + 0.15 * nf["tenure_months"]
            # 3rd order: income × num_products_inv × activity (continuous)
            + 0.20 * nf["income_q"] * (1 - nf["num_products"]) * nf["is_active"]
        )
        lat_nba = (
            0.6 * l[:, 1]  # activity_level
            + 0.4 * _persona_bias("has_nba", {
                "conservative_saver": -0.5,
                "active_spender": 0.5,
                "young_digital": 0.3,
                "high_value": 0.4,
                "occasional_user": -0.8,
                "diversified": 0.2,
            })
        )
        logit_nba = self._calibrate_logit(
            obs_nba, lat_nba, obs_frac=0.15, lat_frac=0.35,
            noise_frac=0.50, intercept=-3.5, target_pos_rate=0.15,
        )
        _has_nba = self._apply_label_noise(
            (_sigmoid(logit_nba) > 0.5).astype(np.int64), noise_rate=0.06
        )
        # NOT added to labels — nba_primary class 0 encodes "no NBA"

        # --- churn_signal (binary, ~5% positive) ---
        # Variance budget: obs=15%, latent=35%, noise=50%
        # Interaction profile: L0 + 4th-order continuous multiplicative
        obs_churn = (
            - 0.3 * nf["num_products"]
            - 0.25 * nf["synth_frequency"]
            - 0.15 * nf["synth_recency_days"]
            + 0.2 * (1 - nf["is_active"])
            + 0.1 * nf["total_churns"]
            # 4th order: inactivity × recency × low-frequency × short-tenure (continuous, no threshold)
            + 0.20 * (1 - nf["is_active"]) * nf["synth_recency_days"] * (1 - nf["synth_frequency"]) * (1 - nf["tenure_months"])
        )
        lat_churn = (
            - 0.5 * l[:, 1]  # low activity -> churn
            - 0.3 * l[:, 4]  # low loyalty -> churn
            + 0.2 * _persona_bias("churn", {
                "conservative_saver": 0.3,
                "active_spender": -0.5,
                "young_digital": 0.1,
                "high_value": -0.3,
                "occasional_user": 0.8,
                "diversified": -0.2,
            })
        )
        logit_churn = self._calibrate_logit(
            obs_churn, lat_churn, obs_frac=0.15, lat_frac=0.35,
            noise_frac=0.50, intercept=-2.9, target_pos_rate=0.05,
        )
        labels["churn_signal"] = self._apply_label_noise(
            (_sigmoid(logit_churn) > 0.5).astype(np.int64), noise_rate=0.06
        )

        # --- product_stability (regression, 0-1, avg ~0.92) ---
        # Variance budget: obs=10%, latent=18%, noise=15% (see mix below)
        # Interaction profile: L0 + 4th-order continuous multiplicative
        obs_stab = (
            0.4 * nf["synth_stability"]
            + 0.25 * nf["tenure_months"]
            + 0.2 * nf["num_products"]
            + 0.15 * nf["is_active"]
            # 4th order: stability × tenure × products × activity compound (continuous, no threshold)
            + 0.15 * nf["synth_stability"] * nf["tenure_months"] * nf["num_products"] * nf["is_active"]
        )
        lat_stab = (
            0.5 * _sigmoid(l[:, 4])  # loyalty
            + 0.3 * _sigmoid(l[:, 0])  # wealth_propensity
            + 0.2 * _persona_bias("stability", {
                "conservative_saver": 0.15,
                "active_spender": 0.05,
                "young_digital": -0.10,
                "high_value": 0.12,
                "occasional_user": -0.15,
                "diversified": 0.05,
            })
        )
        # Scale obs and latent to unit variance, then mix
        obs_s = _normalize(obs_stab)
        lat_s = _normalize(lat_stab)
        noise_s = self.rng.standard_normal(n)
        stability_raw = (
            0.55  # base
            + 0.08 * obs_s   # obs: ~12% (weak raw signal, needs Phase 0 encoding)
            + 0.18 * lat_s   # lat: ~40% (recoverable via TDA/Mamba patterns)
            + 0.15 * noise_s
        )
        labels["product_stability"] = np.clip(stability_raw, 0.0, 1.0).astype(
            np.float64
        )
        # ~5% nullable
        null_mask = self.rng.random(n) < 0.05
        labels["product_stability"] = np.where(
            null_mask, np.nan, labels["product_stability"]
        )

        # ================================================================
        # Tier 2 — Derived: spend_level, engagement_score, cross_sell_count
        # ================================================================

        # --- spend_level (multiclass 4) ---
        # Use RAW (pre-normalization) synth_monthly_spend from features dict.
        # Boundaries are quantile-based (33rd/66th/90th pct) so class
        # distribution is always meaningful regardless of the generated spend
        # scale. This avoids hardcoded currency thresholds that only hold for
        # specific amount distributions.
        spend = features["synth_monthly_spend"].astype(np.float64)
        spend_q33 = np.nanpercentile(spend, 33)
        spend_q66 = np.nanpercentile(spend, 66)
        spend_q90 = np.nanpercentile(spend, 90)
        labels["label_spend_level"] = np.where(
            spend < spend_q33, 0,
            np.where(spend < spend_q66, 1,
                     np.where(spend < spend_q90, 2, 3))
        ).astype(np.int64)
        logger.info(
            "  spend_level boundaries (quantile-based): q33=%.1f, q66=%.1f, q90=%.1f",
            spend_q33, spend_q66, spend_q90,
        )

        # --- engagement_score (regression, 0-1) ---
        # weighted sum: is_active*0.3 + frequency_norm*0.4 + num_products_norm*0.3
        eng = (
            0.3 * features["is_active"].astype(np.float64)
            + 0.4 * nf["synth_frequency"]
            + 0.3 * nf["num_products"]
        )
        labels["label_engagement_score"] = np.clip(eng, 0, 1).astype(np.float64)

        # --- cross_sell_count (regression) ---
        # How many products to recommend (0-24)
        # Correlated with has_nba and product portfolio
        cs_base = (
            0.3 * nf["num_products"]
            + 0.2 * nf["synth_monthly_spend"]
            + 0.15 * l[:, 1]  # activity
            + 0.15 * l[:, 0]  # wealth
            + noise(0.3)
        )
        cs_count = np.clip(np.round(cs_base * 4), 0, 24).astype(np.float64)
        # Only customers with has_nba=1 have nonzero cross-sell
        cs_count = np.where(_has_nba == 1, np.maximum(cs_count, 1), 0)
        labels["label_cross_sell_count"] = cs_count

        # ================================================================
        # nba_label (list) — needed to derive nba_primary and product group tasks
        # ================================================================
        nba_label_list = [[] for _ in range(n)]
        for i in range(n):
            if _has_nba[i] == 1:
                n_products_to_rec = max(int(labels["label_cross_sell_count"][i]), 1)
                # Weight by persona and latent for product selection
                persona_idx = z[i]
                pname = self.persona_names[persona_idx]
                # Products not currently held
                not_held = [
                    p_idx for p_idx in range(N_PRODUCTS)
                    if features[f"prod_{PRODUCT_NAMES[p_idx]}"][i] == 0
                ]
                if not not_held:
                    not_held = list(range(N_PRODUCTS))
                n_rec = min(n_products_to_rec, len(not_held))
                selected = self.rng.choice(not_held, size=n_rec, replace=False)
                nba_label_list[i] = sorted(selected.tolist())

        # --- nba_primary (multiclass 7: 0=no_nba, 1-6=product_group) ---
        nba_primary = np.zeros(n, dtype=np.int64)  # class 0 = no NBA
        for i in range(n):
            if nba_label_list[i]:
                nba_primary[i] = _product_idx_to_nba_group(nba_label_list[i][0])
        labels["label_nba_primary"] = nba_primary

        # ================================================================
        # Tier 3 — Hard: will_acquire_* (binary)
        # obs=35%, latent=35%, noise=30%
        # Each product group has its own logit model
        # ================================================================
        product_group_config = {
            "deposits": {
                "indices": [8, 9, 10],
                # Interaction profile: L0 + 3rd-order continuous multiplicative
                "obs_fn": lambda: (
                    0.3 * nf["income_q"]
                    + 0.25 * nf["tenure_months"]
                    + 0.25 * nf["synth_monetary"]
                    + 0.2 * nf["synth_stability"]
                    # 3rd order: income × tenure × stability (continuous, no threshold)
                    + 0.20 * nf["income_q"] * nf["tenure_months"] * nf["synth_stability"]
                ),
                "lat_fn": lambda: (
                    0.5 * l[:, 0]  # wealth_propensity
                    + 0.3 * l[:, 4]  # loyalty
                    + 0.2 * _persona_bias("acq_dep", {
                        "conservative_saver": 0.6, "active_spender": 0.0,
                        "young_digital": -0.3, "high_value": 0.4,
                        "occasional_user": -0.2, "diversified": 0.1,
                    })
                ),
                "target_rate": 0.10,
            },
            "investments": {
                "indices": [12, 18],
                # Interaction profile: L0 + 5th-order continuous multiplicative (hardest task)
                "obs_fn": lambda: (
                    0.35 * nf["income_q"]
                    + 0.25 * nf["num_products"]
                    + 0.2 * nf["synth_monetary"]
                    + 0.2 * nf["tenure_months"]
                    # 5th order: income × spend × tenure × products × stability (continuous, no threshold)
                    + 0.20 * nf["income_q"] * nf["synth_monthly_spend"] * nf["tenure_months"] * nf["num_products"] * nf["synth_stability"]
                ),
                "lat_fn": lambda: (
                    0.4 * l[:, 0]  # wealth
                    + 0.3 * l[:, 2]  # risk_tolerance
                    + 0.3 * _persona_bias("acq_inv", {
                        "conservative_saver": -0.3, "active_spender": 0.2,
                        "young_digital": 0.1, "high_value": 0.6,
                        "occasional_user": -0.4, "diversified": 0.3,
                    })
                ),
                "target_rate": 0.08,
            },
            "accounts": {
                "indices": [2, 5, 6, 7, 11, 19],
                # Interaction profile: L0 + 3rd-order continuous multiplicative
                "obs_fn": lambda: (
                    0.3 * (1 - nf["num_products"])
                    + 0.25 * nf["is_active"]
                    + 0.25 * nf["tenure_months"]
                    + 0.2 * nf["synth_frequency"]
                    # 3rd order: product_gap × activity × tenure (continuous, no threshold)
                    + 0.20 * (1 - nf["num_products"]) * nf["is_active"] * nf["tenure_months"]
                ),
                "lat_fn": lambda: (
                    0.4 * l[:, 1]  # activity
                    + 0.3 * l[:, 3]  # digital_affinity
                    + 0.3 * _persona_bias("acq_acct", {
                        "conservative_saver": -0.2, "active_spender": 0.3,
                        "young_digital": 0.5, "high_value": 0.1,
                        "occasional_user": -0.3, "diversified": 0.2,
                    })
                ),
                "target_rate": 0.12,
            },
            "lending": {
                "indices": [13, 15],
                # Interaction profile: L0 + 4th-order continuous multiplicative
                "obs_fn": lambda: (
                    0.3 * nf["income_q"]
                    + 0.25 * nf["tenure_months"]
                    + 0.25 * nf["num_products"]
                    + 0.2 * nf["is_active"]
                    # 4th order: high_spend × low_income × frequency × instability (continuous, no threshold)
                    + 0.20 * nf["synth_monthly_spend"] * (1 - nf["income_q"]) * nf["synth_frequency"] * (1 - nf["synth_stability"])
                ),
                "lat_fn": lambda: (
                    0.4 * l[:, 0]  # wealth
                    + 0.35 * l[:, 2]  # risk_tolerance
                    + 0.25 * _persona_bias("acq_lend", {
                        "conservative_saver": -0.4, "active_spender": 0.3,
                        "young_digital": 0.1, "high_value": 0.5,
                        "occasional_user": -0.3, "diversified": 0.2,
                    })
                ),
                "target_rate": 0.08,
            },
            "payments": {
                "indices": [4, 17, 20, 22, 23],
                # Interaction profile: L0 + 4th-order continuous multiplicative
                "obs_fn": lambda: (
                    0.3 * nf["synth_frequency"]
                    + 0.25 * nf["is_active"]
                    + 0.25 * nf["synth_monthly_spend"]
                    + 0.2 * nf["num_products"]
                    # 4th order: frequency × products × activity × spending_regularity (continuous, no threshold)
                    + 0.18 * nf["synth_frequency"] * nf["num_products"] * nf["is_active"] * nf["synth_monthly_spend"]
                ),
                "lat_fn": lambda: (
                    0.4 * l[:, 1]  # activity
                    + 0.3 * l[:, 3]  # digital_affinity
                    + 0.3 * _persona_bias("acq_pay", {
                        "conservative_saver": -0.3, "active_spender": 0.4,
                        "young_digital": 0.3, "high_value": 0.0,
                        "occasional_user": -0.4, "diversified": 0.2,
                    })
                ),
                "target_rate": 0.10,
            },
        }
        for group_name, cfg in product_group_config.items():
            col_name = f"label_acquire_{group_name}"
            obs = cfg["obs_fn"]()
            lat = cfg["lat_fn"]()
            logit = self._calibrate_logit(
                obs, lat, obs_frac=0.15, lat_frac=0.35,
                noise_frac=0.50, intercept=-3.5,
                target_pos_rate=cfg["target_rate"],
            )
            labels[col_name] = self._apply_label_noise(
                (_sigmoid(logit) > 0.5).astype(np.int64), noise_rate=0.08
            )

        # Rebuild nba_label list from the independent acquire labels
        for i in range(n):
            if _has_nba[i] == 1:
                rec = []
                for group_name, cfg in product_group_config.items():
                    col = f"label_acquire_{group_name}"
                    if labels[col][i] == 1:
                        rec.extend(cfg["indices"])
                if not rec:
                    # If has_nba but no group triggered, pick random product
                    not_held = [
                        p_idx for p_idx in range(N_PRODUCTS)
                        if features[f"prod_{PRODUCT_NAMES[p_idx]}"][i] == 0
                    ]
                    if not_held:
                        rec = [self.rng.choice(not_held)]
                    else:
                        rec = [0]
                nba_label_list[i] = sorted(set(rec))
            else:
                nba_label_list[i] = []

        # Recompute nba_primary and cross_sell_count from updated nba_label
        for i in range(n):
            if nba_label_list[i]:
                nba_primary[i] = _product_idx_to_nba_group(nba_label_list[i][0])
                cs_count[i] = len(nba_label_list[i])
            else:
                nba_primary[i] = 0  # class 0 = no NBA
                cs_count[i] = 0
        labels["label_nba_primary"] = nba_primary
        labels["label_cross_sell_count"] = cs_count

        # ================================================================
        # Tier 5 — Very hard: next_mcc, mcc_diversity_trend, top_mcc_shift
        # obs=30%, latent=25%, noise=45%
        # ================================================================

        # --- next_mcc (multiclass 50) ---
        mcc_seqs = features["txn_mcc_seq"]
        next_mcc = np.zeros(n, dtype=np.int64)
        for i in range(n):
            seq = mcc_seqs[i]
            if seq and len(seq) > 0:
                next_mcc[i] = seq[-1]  # last MCC in sequence
            else:
                next_mcc[i] = 0
        labels["label_next_mcc"] = next_mcc

        # --- mcc_diversity_trend (regression) ---
        mcc_div_trend = np.zeros(n, dtype=np.float64)
        for i in range(n):
            seq = mcc_seqs[i]
            if seq and len(seq) >= 30:
                recent = set(seq[-10:])
                older = set(seq[-30:-10])
                mcc_div_trend[i] = len(recent) - len(older)
            elif seq and len(seq) >= 10:
                mcc_div_trend[i] = len(set(seq[-10:])) - len(set(seq[: len(seq) // 2]))
        labels["label_mcc_diversity_trend"] = mcc_div_trend

        # --- top_mcc_shift (binary) ---
        # Use 30-txn windows (60 total) for stability: with stickiness=0.60,
        # the mode over 30 txns is much more stable → shift rate ~40-60%
        top_mcc_shift = np.zeros(n, dtype=np.int64)
        for i in range(n):
            seq = mcc_seqs[i]
            if seq and len(seq) >= 60:
                recent_mode = Counter(seq[-30:]).most_common(1)[0][0]
                older_mode = Counter(seq[-60:-30]).most_common(1)[0][0]
                if recent_mode != older_mode:
                    top_mcc_shift[i] = 1
        labels["label_top_mcc_shift"] = self._apply_label_noise(
            top_mcc_shift, noise_rate=0.05
        )

        # --- nba_label as list column ---
        labels["nba_label"] = nba_label_list

        # Log label statistics
        for lname, larr in labels.items():
            if lname == "nba_label":
                continue
            if isinstance(larr, np.ndarray):
                valid = larr[~np.isnan(larr)] if larr.dtype == np.float64 else larr
                if larr.dtype in (np.int64, np.int32):
                    unique, counts = np.unique(larr, return_counts=True)
                    if len(unique) <= 10:
                        dist_str = ", ".join(
                            f"{u}:{c/n:.3f}" for u, c in zip(unique, counts)
                        )
                        logger.info("  Label %s: %s", lname, dist_str)
                    else:
                        logger.info(
                            "  Label %s: %d classes, range [%d, %d]",
                            lname, len(unique), unique.min(), unique.max(),
                        )
                else:
                    valid_arr = valid if isinstance(valid, np.ndarray) else np.array(valid)
                    logger.info(
                        "  Label %s: mean=%.4f, std=%.4f, nan=%.1f%%",
                        lname,
                        np.nanmean(valid_arr),
                        np.nanstd(valid_arr),
                        100.0 * np.isnan(larr).sum() / n,
                    )

        return labels

    # ==================================================================
    # Save to Parquet via DuckDB
    # ==================================================================
    def _save_parquet(
        self,
        features: dict,
        labels: dict,
        z: np.ndarray,
        l: np.ndarray,
        situations: Optional[dict] = None,
    ) -> None:
        """Save main and ground truth parquets using DuckDB."""
        import duckdb

        logger.info("Saving to %s via DuckDB...", self.output_path)
        n = self.n

        # Ensure output directory exists
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.ground_truth_path).parent.mkdir(parents=True, exist_ok=True)

        con = duckdb.connect()

        # Build column dict for main table (non-list columns)
        scalar_cols: Dict[str, np.ndarray] = {}
        list_cols: Dict[str, list] = {}

        # Features
        for key, val in features.items():
            if isinstance(val, np.ndarray):
                scalar_cols[key] = val
            elif isinstance(val, list) and len(val) == n:
                if isinstance(val[0], (list, type(None))):
                    list_cols[key] = val
                else:
                    scalar_cols[key] = np.array(val)
            elif isinstance(val, np.ndarray) and val.dtype.kind == "U":
                scalar_cols[key] = val

        # Labels
        for key, val in labels.items():
            if isinstance(val, np.ndarray):
                scalar_cols[key] = val
            elif isinstance(val, list):
                list_cols[key] = val

        # Create main table from scalar columns
        # Use pyarrow for list columns
        import pyarrow as pa
        import pyarrow.parquet as pq

        fields = []
        arrays = []

        for col_name, arr in scalar_cols.items():
            if arr.dtype.kind == "U":
                pa_arr = pa.array(arr.tolist(), type=pa.string())
                fields.append(pa.field(col_name, pa.string()))
            elif arr.dtype == np.int32:
                pa_arr = pa.array(arr, type=pa.int32())
                fields.append(pa.field(col_name, pa.int32()))
            elif arr.dtype == np.int64:
                pa_arr = pa.array(arr, type=pa.int64())
                fields.append(pa.field(col_name, pa.int64()))
            elif arr.dtype == np.float64:
                # Handle NaN: pyarrow handles NaN natively
                pa_arr = pa.array(arr, type=pa.float64())
                fields.append(pa.field(col_name, pa.float64()))
            else:
                pa_arr = pa.array(arr.tolist())
                fields.append(pa.field(col_name, pa_arr.type))
            arrays.append(pa_arr)

        # List columns
        for col_name, lst in list_cols.items():
            # Determine inner type from first non-None element
            sample = None
            for item in lst:
                if item is not None and len(item) > 0:
                    sample = item[0]
                    break
            if sample is not None and isinstance(sample, float):
                inner_type = pa.float64()
            else:
                inner_type = pa.int32()

            pa_list = []
            for item in lst:
                if item is None:
                    pa_list.append(None)
                else:
                    pa_list.append(item)

            pa_arr = pa.array(pa_list, type=pa.list_(inner_type))
            fields.append(pa.field(col_name, pa.list_(inner_type)))
            arrays.append(pa_arr)

        schema = pa.schema(fields)
        table = pa.table(dict(zip([f.name for f in fields], arrays)), schema=schema)

        pq.write_table(table, self.output_path, compression="snappy")
        logger.info(
            "Main data saved: %s (%.1f MB, %d cols)",
            self.output_path,
            Path(self.output_path).stat().st_size / 1e6,
            len(fields),
        )

        # Ground truth — includes latent vectors AND situation assignments.
        # Situation variables are stored HERE only (not in main feature table)
        # so they are never used as model inputs.
        gt_arrays = {
            "customer_id": pa.array(features["customer_id"], type=pa.int64()),
            "persona_idx": pa.array(z, type=pa.int32()),
            "persona_name": pa.array(
                [self.persona_names[i] for i in z], type=pa.string()
            ),
        }
        for d in range(LATENT_DIM):
            gt_arrays[f"latent_{LATENT_NAMES[d]}"] = pa.array(
                l[:, d], type=pa.float64()
            )
        if situations is not None:
            sit_names = {
                "sit_engagement":  ["steady", "surging", "declining", "volatile"],
                "sit_lifecycle":   ["stable", "growing", "consolidating", "transitioning"],
                "sit_value":       ["stable_value", "ascending", "descending", "shock"],
                "sit_consumption": ["consistent", "exploring", "focusing", "switching"],
            }
            for sit_col, sit_labels in sit_names.items():
                sit_idx = situations[sit_col]
                # Store as both integer index and string label for interpretability
                gt_arrays[sit_col] = pa.array(sit_idx.astype(np.int32), type=pa.int32())
                gt_arrays[f"{sit_col}_name"] = pa.array(
                    [sit_labels[i] for i in sit_idx], type=pa.string()
                )
        gt_table = pa.table(gt_arrays)
        pq.write_table(gt_table, self.ground_truth_path, compression="snappy")
        logger.info("Ground truth saved: %s (includes sit_* columns)", self.ground_truth_path)

    # ==================================================================
    # Validation: XGBoost AUC check
    # ==================================================================
    def _validate_auc(self, features: dict, labels: dict) -> None:
        """Train XGBoost on observable features and check AUC per task."""
        try:
            from xgboost import XGBClassifier, XGBRegressor
            from sklearn.metrics import roc_auc_score, r2_score
            from sklearn.model_selection import train_test_split
        except ImportError:
            logger.warning(
                "xgboost or sklearn not available; skipping AUC validation"
            )
            return

        logger.info("=" * 60)
        logger.info("AUC Validation (XGBoost on observable features only)")
        logger.info("=" * 60)

        # Build feature matrix from observable columns only
        obs_cols = [
            "age", "income", "tenure_months", "is_active", "num_products",
            "synth_monthly_txns", "synth_avg_amount", "synth_monthly_spend",
            "synth_unique_mcc", "synth_unique_merchants",
            "synth_morning_ratio", "synth_afternoon_ratio",
            "synth_evening_ratio", "synth_night_ratio",
            "synth_recency_days", "synth_frequency", "synth_monetary",
            "synth_stability", "total_acquisitions", "total_churns",
            "months_observed", "product_diversity",
        ]
        # Add product holdings
        for pn in PRODUCT_NAMES:
            obs_cols.append(f"prod_{pn}")

        X = np.column_stack([
            features[c].astype(np.float64) for c in obs_cols
        ])

        # Replace NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Subsample for speed
        max_val = min(50000, self.n)
        idx = self.rng.choice(self.n, size=max_val, replace=False)
        X_sub = X[idx]

        # Expected metric ranges per task tier
        # Note: deterministic labels (income_tier, tenure_stage, spend_level)
        # are direct functions of observable columns -> near-perfect accuracy.
        # Segment has randomness per persona -> lower accuracy.
        # Stochastic labels use variance budget to control XGB AUC ceiling.
        expected = {
            # Easy deterministic (obs=100% -- direct from columns)
            "label_segment": {"type": "multiclass", "range": (0.45, 0.75)},
            "label_income_tier": {"type": "multiclass", "range": (0.95, 1.00)},
            "label_tenure_stage": {"type": "multiclass", "range": (0.95, 1.00)},
            # Medium stochastic (obs_frac=0.15, high-order interactions)
            # has_nba removed — folded into nba_primary (class 0 = no NBA)
            "churn_signal": {"type": "binary", "range": (0.65, 0.88)},
            "product_stability": {"type": "regression", "range": (-0.1, 0.3)},
            # Deterministic derived
            "label_spend_level": {"type": "multiclass", "range": (0.95, 1.00)},
            "label_engagement_score": {"type": "regression", "range": (0.90, 1.00)},
            "label_cross_sell_count": {"type": "regression", "range": (-0.5, 0.2)},
            # Hard stochastic (obs_frac=0.25, L0-L4 interactions)
            "label_nba_primary": {"type": "multiclass", "range": (0.02, 0.30)},
            "label_acquire_deposits": {"type": "binary", "range": (0.55, 0.78)},
            "label_acquire_investments": {"type": "binary", "range": (0.55, 0.78)},
            "label_acquire_accounts": {"type": "binary", "range": (0.55, 0.78)},
            "label_acquire_lending": {"type": "binary", "range": (0.55, 0.78)},
            "label_acquire_payments": {"type": "binary", "range": (0.55, 0.78)},
            # Very hard (sequence-derived)
            "label_next_mcc": {"type": "multiclass", "range": (0.05, 0.85)},
            "label_top_mcc_shift": {"type": "binary", "range": (0.45, 0.90)},
            "label_mcc_diversity_trend": {"type": "regression", "range": (-0.1, 0.3)},
        }

        results = {}
        for label_name, spec in expected.items():
            if label_name not in labels:
                continue
            y_full = labels[label_name]
            if isinstance(y_full, list):
                continue

            y = y_full[idx].copy()

            # Skip if too few valid samples
            valid_mask = ~np.isnan(y) if y.dtype == np.float64 else np.ones(len(y), dtype=bool)
            if valid_mask.sum() < 100:
                logger.info("  %s: SKIP (too few samples)", label_name)
                continue

            X_valid = X_sub[valid_mask]
            y_valid = y[valid_mask]

            # Skip labels with only one class
            if spec["type"] == "binary":
                unique_vals = np.unique(y_valid)
                if len(unique_vals) < 2:
                    logger.info("  %s: SKIP (single class)", label_name)
                    continue

            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_valid, y_valid, test_size=0.3,
                    random_state=self.seed,
                )

                if spec["type"] == "binary":
                    model = XGBClassifier(
                        n_estimators=100, max_depth=6, learning_rate=0.1,
                        use_label_encoder=False, eval_metric="logloss",
                        verbosity=0,
                    )
                    model.fit(X_tr, y_tr)
                    y_prob = model.predict_proba(X_te)[:, 1]
                    score = roc_auc_score(y_te, y_prob)
                    metric = "AUC"
                elif spec["type"] == "regression":
                    model = XGBRegressor(
                        n_estimators=100, max_depth=6, learning_rate=0.1,
                        verbosity=0,
                    )
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_te)
                    score = r2_score(y_te, y_pred)
                    metric = "R2"
                else:  # multiclass
                    # All valid classes are >= 0; keep all samples
                    mc_mask = y_valid >= 0
                    if mc_mask.sum() < 100:
                        logger.info("  %s: SKIP (too few valid)", label_name)
                        continue
                    X_mc = X_valid[mc_mask]
                    y_mc = y_valid[mc_mask]
                    unique_classes = np.unique(y_mc)
                    if len(unique_classes) < 2:
                        logger.info("  %s: SKIP (single class)", label_name)
                        continue
                    # Remap to contiguous 0..K-1
                    class_map = {int(c): i for i, c in enumerate(sorted(unique_classes))}
                    y_mapped = np.array([class_map[int(c)] for c in y_mc])
                    n_classes = len(unique_classes)
                    # Filter rare classes that can't be stratified
                    class_counts = Counter(y_mapped.tolist())
                    keep_classes = {c for c, cnt in class_counts.items() if cnt >= 2}
                    keep_mask = np.array([c in keep_classes for c in y_mapped])
                    X_mc_f = X_mc[keep_mask]
                    y_mc_f = y_mapped[keep_mask]
                    if len(y_mc_f) < 100:
                        logger.info("  %s: SKIP (too few after filtering)", label_name)
                        continue
                    # Re-remap to contiguous 0..K-1
                    final_classes = sorted(set(y_mc_f.tolist()))
                    remap2 = {c: i for i, c in enumerate(final_classes)}
                    y_final = np.array([remap2[c] for c in y_mc_f])
                    n_classes_final = len(final_classes)
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X_mc_f, y_final, test_size=0.3,
                        random_state=self.seed, stratify=y_final,
                    )
                    model = XGBClassifier(
                        n_estimators=100, max_depth=6, learning_rate=0.1,
                        use_label_encoder=False, eval_metric="mlogloss",
                        num_class=n_classes_final, verbosity=0,
                    )
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_te)
                    from sklearn.metrics import accuracy_score

                    score = accuracy_score(y_te, y_pred)
                    metric = "Acc"

                lo, hi = spec["range"]
                in_range = lo <= score <= hi
                marker = "OK" if in_range else "WARN"
                results[label_name] = {
                    "metric": metric, "score": score,
                    "expected": spec["range"], "ok": in_range,
                }
                logger.info(
                    "  %s: target_%s=%.2f-%.2f, achieved=%.4f %s",
                    label_name, metric.lower(), lo, hi, score, marker,
                )
            except Exception as e:
                logger.warning("  %s: validation failed: %s", label_name, e)

        # Summary
        n_ok = sum(1 for r in results.values() if r["ok"])
        n_total = len(results)
        logger.info(
            "Validation: %d/%d tasks within expected range", n_ok, n_total
        )


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark financial data with controllable AUC ceilings"
    )
    parser.add_argument(
        "--n-customers", type=int, default=1_000_000,
        help="Number of customers to generate (default: 1M)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--calibration", type=str,
        default="configs/santander/calibration_params.yaml",
        help="Path to calibration YAML",
    )
    parser.add_argument(
        "--output", type=str, default="data/benchmark_v1.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--ground-truth", type=str, default=None,
        help="Ground truth parquet path (default: <output_dir>/benchmark_ground_truth.parquet)",
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip XGBoost AUC validation",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    gen = BenchmarkDataGenerator(
        calibration_path=args.calibration,
        n_customers=args.n_customers,
        seed=args.seed,
        output_path=args.output,
        ground_truth_path=args.ground_truth,
        validate=not args.no_validate,
    )
    gen.generate()


if __name__ == "__main__":
    main()
