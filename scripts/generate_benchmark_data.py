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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy import stats as sp_stats

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

        # Layer 1: Latent personas
        z, l = self._generate_latent_personas()
        logger.info("Layer 1 done: personas shape=%s, latent shape=%s", z.shape, l.shape)

        # Layer 2: Observable profiles
        profiles = self._generate_profiles(z, l)
        logger.info("Layer 2 done: %d profile columns", len(profiles))

        # Layer 3: Transaction sequences
        txn_data = self._generate_transactions(z, l, profiles)
        logger.info("Layer 3 done: %d txn columns", len(txn_data))

        # Merge all features
        features = {**profiles, **txn_data}

        # Layer 4: Labels
        labels = self._generate_labels(z, l, features)
        logger.info("Layer 4 done: %d label columns", len(labels))

        # Save via DuckDB
        self._save_parquet(features, labels, z, l)

        elapsed = time.time() - t0
        logger.info("Generation complete in %.1fs", elapsed)

        # Validation
        if self.validate:
            self._validate_auc(features, labels)

    # ==================================================================
    # Layer 1: Latent Personas
    # ==================================================================
    def _generate_latent_personas(self) -> Tuple[np.ndarray, np.ndarray]:
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

        return z, l

    # ==================================================================
    # Layer 2: Observable Profiles
    # ==================================================================
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

        # --- Demographics per persona ---
        age = np.empty(n, dtype=np.float64)
        income = np.empty(n, dtype=np.float64)
        tenure_months = np.empty(n, dtype=np.float64)

        for k in range(self.n_personas):
            mask = z == k
            n_k = mask.sum()
            if n_k == 0:
                continue
            name = self.persona_names[k]
            p = demo[name]

            # Age: truncated normal
            ap = p["age"]
            a_lo = (ap["lo"] - ap["mu"]) / ap["sigma"]
            a_hi = (ap["hi"] - ap["mu"]) / ap["sigma"]
            age[mask] = sp_stats.truncnorm.rvs(
                a_lo, a_hi, loc=ap["mu"], scale=ap["sigma"],
                size=n_k, random_state=self.rng.integers(2**31),
            )

            # Income: lognormal with zero_rate (missing)
            ip = p["income"]
            raw_income = np.exp(
                self.rng.normal(ip["mu"], ip["sigma"], size=n_k)
            )
            zero_mask = self.rng.random(n_k) < ip.get("zero_rate", 0.0)
            raw_income[zero_mask] = 0.0
            income[mask] = raw_income

            # Tenure: gamma
            tp = p["tenure_months"]
            tenure_raw = self.rng.gamma(
                tp["shape"], tp["scale"], size=n_k
            )
            tenure_months[mask] = np.clip(tenure_raw, 0, 256).astype(int)

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
    # Layer 3: Transaction Sequences
    # ==================================================================
    def _generate_transactions(
        self, z: np.ndarray, l: np.ndarray, profiles: dict
    ) -> dict:
        """Generate transaction sequences and synth_* aggregates."""
        txn_cal = self.cal["transactions"]
        n = self.n

        # Collect all MCC codes (top-50 for sequence encoding)
        all_mccs = sorted(set(int(m) for m in self.cal["personas"]["mcc_codes"]))
        top_mccs = all_mccs[:50] if len(all_mccs) >= 50 else all_mccs
        mcc_to_idx = {m: i for i, m in enumerate(top_mccs)}

        # Per-persona monthly txn rates (per 3-month window)
        # Original data range: synth_monthly_txns 20-157, avg 69
        # synth_monthly_txns = sum(12 months) / 3, so target sum ~ 207
        # Monthly lambda = target_monthly_txns (since /3 is 3-month avg)
        persona_lambda = {
            "conservative_saver": 12.0,
            "active_spender": 30.0,
            "young_digital": 22.0,
            "high_value": 18.0,
            "occasional_user": 8.0,
            "diversified": 25.0,
        }

        # --- Generate per-customer transaction data ---
        # We generate monthly txn counts for ~12 months, then build sequences

        # Pre-allocate synth aggregates
        synth_monthly_txns = np.zeros(n, dtype=np.int32)
        synth_avg_amount = np.zeros(n, dtype=np.float64)
        synth_monthly_spend = np.zeros(n, dtype=np.float64)
        synth_unique_mcc = np.zeros(n, dtype=np.int32)
        synth_unique_merchants = np.zeros(n, dtype=np.int32)
        synth_morning_ratio = np.zeros(n, dtype=np.float64)
        synth_afternoon_ratio = np.zeros(n, dtype=np.float64)
        synth_evening_ratio = np.zeros(n, dtype=np.float64)
        synth_night_ratio = np.zeros(n, dtype=np.float64)
        synth_recency_days = np.zeros(n, dtype=np.float64)
        synth_frequency = np.zeros(n, dtype=np.int32)
        synth_monetary = np.zeros(n, dtype=np.float64)
        synth_stability = np.zeros(n, dtype=np.float64)
        synth_fraud_ratio = np.zeros(n, dtype=np.float64)  # always zero

        # Sequence lists
        txn_amount_seq = [None] * n
        txn_mcc_seq = [None] * n
        txn_hour_seq = [None] * n
        txn_day_offset_seq = [None] * n
        txn_date_seq = [None] * n  # YYYYMMDD int for pipeline time-based windowing

        # Derived temporal aggregates
        total_acquisitions = np.zeros(n, dtype=np.int64)
        total_churns = np.zeros(n, dtype=np.int64)
        months_observed = np.zeros(n, dtype=np.int64)
        product_diversity = np.zeros(n, dtype=np.int64)

        # Product sequences (16 months)
        prod_sequences = {
            f"seq_{pn}": [None] * n for pn in PRODUCT_NAMES
        }
        seq_num_products_all = [None] * n
        seq_acquisitions_all = [None] * n
        seq_churns_all = [None] * n

        # Process per persona for efficiency
        for k in range(self.n_personas):
            mask_idx = np.where(z == k)[0]
            n_k = len(mask_idx)
            if n_k == 0:
                continue
            name = self.persona_names[k]
            td = txn_cal[name]
            lam = persona_lambda[name]
            ar1 = td.get("ar1_median", 0.0)
            harmonics = td.get("seasonality_harmonics", []) or []

            # MCC distribution for this persona
            mcc_codes_p = list(td["mcc_probs"].keys())
            mcc_probs_p = np.array(list(td["mcc_probs"].values()), dtype=np.float64)
            mcc_probs_p /= mcc_probs_p.sum()
            mcc_codes_int = [int(c) for c in mcc_codes_p]

            # Amount params per MCC
            amount_params = td.get("amount_params", {})

            # Hour distribution
            hour_dist = td.get("hour_dist", {})
            hours = sorted(hour_dist.keys())
            hour_probs = np.array([hour_dist[h] for h in hours], dtype=np.float64)
            if hour_probs.sum() > 0:
                hour_probs /= hour_probs.sum()
            else:
                hour_probs = np.ones(24) / 24
            hour_vals = np.array([int(h) for h in hours], dtype=np.int32)

            # Process in chunks for memory efficiency
            chunk_size = min(50000, n_k)
            for chunk_start in range(0, n_k, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_k)
                idx = mask_idx[chunk_start:chunk_end]
                nc = len(idx)

                # Latent modulation of transaction rate
                activity_mod = np.exp(0.3 * l[idx, 1])  # activity_level
                customer_lambda = lam * activity_mod

                # Generate monthly txn counts with AR(1)
                n_months = 12
                monthly_counts = np.zeros((nc, n_months), dtype=np.int32)

                # Base Poisson rate per month with seasonality
                for m in range(n_months):
                    season_mod = 0.0
                    for h in harmonics:
                        freq = h.get("frequency_idx", 1)
                        mag = h.get("magnitude", 0.0)
                        phase = h.get("phase", 0.0)
                        # Normalize magnitude to a small effect
                        season_mod += 0.1 * np.sin(
                            2 * np.pi * freq * m / 12 + phase
                        )

                    if m == 0:
                        lam_m = customer_lambda * (1.0 + season_mod)
                    else:
                        # AR(1): mix previous realization with baseline
                        prev_dev = (
                            monthly_counts[:, m - 1] / np.maximum(customer_lambda, 1)
                            - 1.0
                        )
                        lam_m = customer_lambda * (
                            1.0 + ar1 * prev_dev + season_mod
                        )

                    lam_m = np.maximum(lam_m, 0.5)
                    monthly_counts[:, m] = self.rng.poisson(lam_m)

                # Life events: 5% per year -> ~0.4% per month regime shift
                life_event = self.rng.random((nc, n_months)) < 0.004
                shift_factor = self.rng.choice(
                    [0.3, 0.5, 1.5, 2.0], size=(nc, n_months)
                )
                monthly_counts = np.where(
                    life_event,
                    (monthly_counts * shift_factor).astype(np.int32),
                    monthly_counts,
                )

                total_txns = monthly_counts.sum(axis=1)

                # Generate individual transactions for sequence building
                for ci in range(nc):
                    cust_idx = idx[ci]
                    n_txn = min(int(total_txns[ci]), TXN_SEQ_MAX_LEN)
                    if n_txn == 0:
                        n_txn = 1  # at least 1

                    # MCC selection
                    mcc_sample = self.rng.choice(
                        len(mcc_codes_int), size=n_txn, p=mcc_probs_p
                    )
                    mcc_vals = np.array(
                        [mcc_codes_int[j] for j in mcc_sample], dtype=np.int32
                    )

                    # Amount per txn: LogNorm conditioned on MCC
                    amounts = np.zeros(n_txn, dtype=np.float64)
                    for j in range(n_txn):
                        mcc_str = str(mcc_codes_int[mcc_sample[j]])
                        if mcc_str in amount_params:
                            ap = amount_params[mcc_str]
                            amounts[j] = np.exp(
                                self.rng.normal(ap["mu"], ap["sigma"])
                            )
                        else:
                            amounts[j] = np.exp(self.rng.normal(3.0, 1.0))
                    amounts = np.clip(amounts, 0.01, 50000)

                    # Hour per txn
                    txn_hours = self.rng.choice(
                        hour_vals, size=n_txn, p=hour_probs
                    )

                    # Day offset within 90-day window
                    day_offsets = np.sort(
                        self.rng.integers(0, 90, size=n_txn)
                    )

                    # Convert day offsets to YYYYMMDD dates
                    # Base date: 2016-03-28 minus 90 days = ~2015-12-29
                    base_epoch = 20151229
                    txn_dates_yyyymmdd = []
                    for d in day_offsets:
                        # Simple date arithmetic: add d days to base
                        month = 12 + (29 + int(d)) // 31
                        year = 2015 + (month - 1) // 12
                        month = ((month - 1) % 12) + 1
                        day = max(1, min(28, (29 + int(d)) % 31))
                        txn_dates_yyyymmdd.append(
                            year * 10000 + month * 100 + day
                        )

                    # Map MCC to top-50 index (for model input)
                    mcc_indexed = np.array(
                        [mcc_to_idx.get(int(m), 0) for m in mcc_vals],
                        dtype=np.int32,
                    )

                    txn_amount_seq[cust_idx] = amounts.tolist()
                    txn_mcc_seq[cust_idx] = mcc_indexed.tolist()
                    txn_hour_seq[cust_idx] = txn_hours.tolist()
                    txn_day_offset_seq[cust_idx] = day_offsets.tolist()
                    txn_date_seq[cust_idx] = txn_dates_yyyymmdd

                    # Synth aggregates
                    synth_monthly_txns[cust_idx] = max(
                        int(total_txns[ci] / 3), 1
                    )  # 3-month avg
                    synth_avg_amount[cust_idx] = round(float(amounts.mean()), 2)
                    synth_monthly_spend[cust_idx] = round(
                        float(amounts.sum() / 3), 2
                    )
                    unique_mccs = len(set(mcc_vals.tolist()))
                    synth_unique_mcc[cust_idx] = unique_mccs
                    synth_unique_merchants[cust_idx] = min(
                        unique_mccs + self.rng.integers(3, 15), 33
                    )

                    # Time-of-day ratios
                    morning = np.sum((txn_hours >= 6) & (txn_hours < 12))
                    afternoon = np.sum((txn_hours >= 12) & (txn_hours < 18))
                    evening = np.sum((txn_hours >= 18) & (txn_hours < 22))
                    night = n_txn - morning - afternoon - evening
                    denom = max(n_txn, 1)
                    synth_morning_ratio[cust_idx] = round(morning / denom, 4)
                    synth_afternoon_ratio[cust_idx] = round(afternoon / denom, 4)
                    synth_evening_ratio[cust_idx] = round(evening / denom, 4)
                    synth_night_ratio[cust_idx] = round(night / denom, 4)

                    # RFM
                    synth_recency_days[cust_idx] = round(
                        1.0 - day_offsets[-1] / 90.0, 4
                    )
                    synth_frequency[cust_idx] = n_txn
                    synth_monetary[cust_idx] = round(float(amounts.sum()), 2)

                    # Stability: 1/CV of monthly counts
                    mc = monthly_counts[ci]
                    mc_mean = mc.mean()
                    mc_std = mc.std()
                    if mc_mean > 0:
                        synth_stability[cust_idx] = round(
                            1.0 / max(mc_std / mc_mean, 0.01), 4
                        )
                    else:
                        synth_stability[cust_idx] = 1.0

                # Product sequences (16 months of holdings)
                for ci in range(nc):
                    cust_idx = idx[ci]
                    num_prods_now = profiles["num_products"][cust_idx]

                    # Simple model: products gradually acquired over time
                    n_obs = self.rng.integers(8, N_SEQ_MONTHS + 1)
                    months_observed[cust_idx] = n_obs

                    current_holdings = np.array(
                        [profiles[f"prod_{pn}"][cust_idx] for pn in PRODUCT_NAMES],
                        dtype=np.int32,
                    )
                    tot_acq = 0
                    tot_churn = 0
                    prod_div_set = set()

                    for pn_idx, pn in enumerate(PRODUCT_NAMES):
                        seq = np.full(N_SEQ_MONTHS, -1, dtype=np.int32)
                        # Last n_obs months have data
                        start = N_SEQ_MONTHS - n_obs
                        held = current_holdings[pn_idx]
                        # Probability the product was held from the beginning
                        if held:
                            # Product likely acquired at some random month
                            acq_month = self.rng.integers(0, max(n_obs, 1))
                            for m in range(start, N_SEQ_MONTHS):
                                rel_m = m - start
                                seq[m] = 1 if rel_m >= acq_month else 0
                            if acq_month > 0:
                                tot_acq += 1
                                prod_div_set.add(pn_idx)
                        else:
                            # Maybe briefly held then churned
                            if self.rng.random() < 0.05:
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

                    # num_products and acq/churn per month
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
                "  Persona %s (%d customers) txn generation done", name, n_k
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
        # Tier 1 — Core: has_nba, churn_signal, product_stability
        # Medium: obs=45%, latent=30%, noise=25%
        # ================================================================

        # --- has_nba (binary, ~3% positive) ---
        # Target logit intercept: log(0.03/0.97) ~ -3.47
        logit_nba = (
            # Observable 45%
            1.2 * nf["num_products"]
            + 0.8 * nf["tenure_months"]
            + 0.8 * nf["synth_monthly_spend"]
            + 0.6 * nf["is_active"]
            # XOR interaction
            + 1.0 * (nf["income_q"] > 0.7).astype(float) * (nf["num_products"] < 0.3).astype(float)
            # Latent 30%
            + 0.8 * l[:, 1]  # activity_level
            + 0.5 * _persona_bias("has_nba", {
                "conservative_saver": -0.5,
                "active_spender": 0.5,
                "young_digital": 0.3,
                "high_value": 0.4,
                "occasional_user": -0.8,
                "diversified": 0.2,
            })
            # Noise 25%
            + noise(1.0)
            - 4.5  # shift to get ~3% positive rate
        )
        labels["has_nba"] = (_sigmoid(logit_nba) > 0.5).astype(np.int64)

        # --- churn_signal (binary, ~5% positive) ---
        # Target intercept: log(0.05/0.95) ~ -2.94
        logit_churn = (
            # Observable 45%
            - 1.0 * nf["num_products"]
            - 0.8 * nf["synth_frequency"]
            - 0.6 * nf["synth_recency_days"]
            + 0.8 * (1 - nf["is_active"])
            + 0.6 * nf["total_churns"]
            # Latent 30%
            - 0.7 * l[:, 1]  # low activity -> churn
            - 0.5 * l[:, 4]  # low loyalty -> churn
            + 0.3 * _persona_bias("churn", {
                "conservative_saver": 0.3,
                "active_spender": -0.5,
                "young_digital": 0.1,
                "high_value": -0.3,
                "occasional_user": 0.8,
                "diversified": -0.2,
            })
            # Noise 25%
            + noise(1.0)
            - 2.5  # ~5% positive
        )
        labels["churn_signal"] = (_sigmoid(logit_churn) > 0.5).astype(np.int64)

        # --- product_stability (regression, 0-1, avg ~0.92) ---
        stability_raw = (
            # Base: high average (0.92)
            0.60
            # Observable 45%
            + 0.15 * nf["synth_stability"]
            + 0.08 * nf["tenure_months"]
            + 0.05 * nf["num_products"]
            + 0.04 * nf["is_active"]
            # Latent 30%
            + 0.10 * _sigmoid(l[:, 4])  # loyalty
            + 0.05 * _sigmoid(l[:, 0])  # wealth_propensity
            + 0.05 * _persona_bias("stability", {
                "conservative_saver": 0.15,
                "active_spender": 0.05,
                "young_digital": -0.10,
                "high_value": 0.12,
                "occasional_user": -0.15,
                "diversified": 0.05,
            })
            # Noise 25%
            + noise(0.06)
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
        spend = features["synth_monthly_spend"].astype(np.float64)
        labels["label_spend_level"] = np.where(
            spend < 1500, 0,
            np.where(spend < 3000, 1,
                     np.where(spend < 5000, 2, 3))
        ).astype(np.int64)

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
        cs_count = np.where(labels["has_nba"] == 1, np.maximum(cs_count, 1), 0)
        labels["label_cross_sell_count"] = cs_count

        # ================================================================
        # nba_label (list) — needed to derive nba_primary and product group tasks
        # ================================================================
        nba_label_list = [[] for _ in range(n)]
        for i in range(n):
            if labels["has_nba"][i] == 1:
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

        # --- nba_primary (multiclass 24) ---
        nba_primary = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            if nba_label_list[i]:
                nba_primary[i] = nba_label_list[i][0]
        labels["label_nba_primary"] = nba_primary

        # ================================================================
        # Tier 3 — Hard: will_acquire_* (binary)
        # obs=35%, latent=35%, noise=30%
        # ================================================================
        product_group_indices = {
            "deposits": [8, 9, 10],
            "investments": [12, 18],
            "accounts": [2, 5, 6, 7, 11, 19],
            "lending": [13, 15],
            "payments": [4, 17, 20, 22, 23],
        }
        for group_name, indices in product_group_indices.items():
            col_name = f"label_acquire_{group_name}"
            arr = np.zeros(n, dtype=np.int64)
            for i in range(n):
                if any(idx in nba_label_list[i] for idx in indices):
                    arr[i] = 1
            labels[col_name] = arr

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
        from collections import Counter

        top_mcc_shift = np.zeros(n, dtype=np.int64)
        for i in range(n):
            seq = mcc_seqs[i]
            if seq and len(seq) >= 30:
                recent_mode = Counter(seq[-15:]).most_common(1)[0][0]
                older_mode = Counter(seq[-30:-15]).most_common(1)[0][0]
                if recent_mode != older_mode:
                    top_mcc_shift[i] = 1
        labels["label_top_mcc_shift"] = top_mcc_shift

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

        # Ground truth
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
        gt_table = pa.table(gt_arrays)
        pq.write_table(gt_table, self.ground_truth_path, compression="snappy")
        logger.info("Ground truth saved: %s", self.ground_truth_path)

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

        # Expected AUC ranges per task tier
        expected = {
            # Easy (obs=60%)
            "label_segment": {"type": "multiclass", "range": (0.85, 0.98)},
            "label_income_tier": {"type": "multiclass", "range": (0.85, 0.98)},
            "label_tenure_stage": {"type": "multiclass", "range": (0.80, 0.95)},
            # Medium (obs=45%)
            "has_nba": {"type": "binary", "range": (0.62, 0.78)},
            "churn_signal": {"type": "binary", "range": (0.65, 0.80)},
            "product_stability": {"type": "regression", "range": (0.3, 0.7)},
            # Medium derived
            "label_spend_level": {"type": "multiclass", "range": (0.85, 0.98)},
            "label_engagement_score": {"type": "regression", "range": (0.5, 0.9)},
            "label_cross_sell_count": {"type": "regression", "range": (0.2, 0.6)},
            # Hard (obs=35%)
            "label_nba_primary": {"type": "multiclass", "range": (0.10, 0.50)},
            "label_acquire_deposits": {"type": "binary", "range": (0.55, 0.75)},
            "label_acquire_investments": {"type": "binary", "range": (0.55, 0.75)},
            "label_acquire_accounts": {"type": "binary", "range": (0.55, 0.75)},
            "label_acquire_lending": {"type": "binary", "range": (0.55, 0.75)},
            "label_acquire_payments": {"type": "binary", "range": (0.55, 0.75)},
            # Very hard (obs=30%)
            "label_next_mcc": {"type": "multiclass", "range": (0.05, 0.40)},
            "label_top_mcc_shift": {"type": "binary", "range": (0.50, 0.70)},
            "label_mcc_diversity_trend": {"type": "regression", "range": (0.0, 0.4)},
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
                    unique_classes = np.unique(y_valid)
                    if len(unique_classes) < 2:
                        logger.info("  %s: SKIP (single class)", label_name)
                        continue
                    model = XGBClassifier(
                        n_estimators=100, max_depth=6, learning_rate=0.1,
                        use_label_encoder=False, eval_metric="mlogloss",
                        num_class=int(unique_classes.max()) + 1,
                        verbosity=0,
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
