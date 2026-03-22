#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ealtman2019 Credit Card Transactions Adapter
=============================================

SageMaker Processing Job script that converts the raw ealtman2019 dataset
(24M transactions, 2,000 users, 6,146 cards) into a single pipeline-ready
parquet file with ~469 feature dimensions + 16 labels.

Input (read from ``--input-dir``):
    01_financial_users.parquet   -- 2,000 customer profiles
    01_financial_cards.parquet   -- 6,146 card records
    credit_card_transactions-ibm_v2.csv  -- 24M transaction rows

Output (written to ``--output-dir``):
    ealtman2019_features.parquet      -- user_id + ~469D features + 16 labels
    ealtman2019_event_sequences.npy   -- (n_users, 180, 16) 3D sequence tensor
    ealtman2019_seq_lengths.npy       -- (n_users,) actual sequence lengths
    feature_stats.json                -- per-feature {mean, std, min, max, null_pct}
    label_stats.json                  -- per-label distribution statistics
    scaler_params.json                -- StandardScaler mean/std for feature normalization
    label_transforms.json             -- clip + log1p params for monetary labels

Usage (SageMaker Processing):
    python ealtman2019_adapter.py \
        --input-dir /opt/ml/processing/input \
        --output-dir /opt/ml/processing/output

Dependencies: pandas, numpy, scipy, duckdb (no torch).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import duckdb as _duckdb

# ---------------------------------------------------------------------------
# DataAdapter framework (optional — available when core.pipeline is present)
# ---------------------------------------------------------------------------
try:
    from core.pipeline.adapter import DataAdapter, AdapterMetadata, AdapterRegistry
    _HAS_ADAPTER_FRAMEWORK = True
except ImportError:
    _HAS_ADAPTER_FRAMEWORK = False

# ---------------------------------------------------------------------------
# Real feature generators (from core/feature/generators/)
# All have numpy fallbacks and do not require GPU or heavy dependencies.
# ---------------------------------------------------------------------------
try:
    from core.feature.generators.tda import TDAFeatureGenerator
    _HAS_TDA_GEN = True
except Exception:
    _HAS_TDA_GEN = False

try:
    from core.feature.generators.hmm import HMMFeatureGenerator
    _HAS_HMM_GEN = True
except Exception:
    _HAS_HMM_GEN = False

try:
    from core.feature.generators.mamba import MambaFeatureGenerator
    _HAS_MAMBA_GEN = True
except Exception:
    _HAS_MAMBA_GEN = False

try:
    from core.feature.generators.model_features import ModelFeaturesGenerator
    _HAS_MODEL_GEN = True
except Exception:
    _HAS_MODEL_GEN = False

try:
    from core.feature.generators.graph import GraphEmbeddingGenerator
    _HAS_GRAPH_GEN = True
except Exception:
    _HAS_GRAPH_GEN = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1_000_000  # rows per CSV chunk
N_TOP_MCC = 60          # top MCC codes for category features
GMM_MAX_CLUSTERS = 30   # upper bound for GMM BIC-based cluster selection
RANDOM_STATE = 42

# --- Event Sequence parameters (for Mamba / Temporal Ensemble 3D input) ---
SEQ_LEN = 180           # max transactions per user (~6 months daily)
SEQ_FEAT_DIM = 16       # feature vector dimension per transaction step
SEQ_TOP_MCC = 60        # MCC codes to track (index 0 = "other")

# MCC major categories (7 groups)
MCC_MAJOR = {
    "grocery":       list(range(5400, 5500)),
    "dining":        list(range(5800, 5900)),
    "travel":        list(range(3000, 3500)) + list(range(4400, 4500)) + list(range(7000, 7100)),
    "gas":           list(range(5500, 5600)),
    "entertainment": list(range(7800, 7999)) + list(range(7900, 8000)),
    "healthcare":    list(range(8000, 8100)),
    "retail":        list(range(5200, 5400)) + list(range(5600, 5700)) + list(range(5900, 6000)),
}

CARD_BRANDS = ["Visa", "Mastercard", "American Express", "Discover"]
USE_CHIP_CATEGORIES = ["Chip Transaction", "Swipe Transaction", "Online Transaction"]

# Time-of-day bins (6 bins)
TIME_BINS_6 = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]

# Time-of-day bins (8 bins, for label)
TIME_BINS_8 = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 21), (21, 24)]


# ===================================================================
# 1. Data Loading
# ===================================================================

def load_users(input_dir: str) -> pd.DataFrame:
    """Load user profiles."""
    path = os.path.join(input_dir, "01_financial_users.parquet")
    logger.info("Loading users from %s", path)
    df = pd.read_parquet(path)
    # Normalize column names (Parquet may have full names from original CSV)
    col_renames = {
        "Yearly Income - Person": "Yearly Income",
        "Per Capita Income - Zipcode": "Per Capita Income",
    }
    df = df.rename(columns=col_renames)
    logger.info("Users loaded: %d rows, %d cols", len(df), len(df.columns))
    return df


def load_cards(input_dir: str) -> pd.DataFrame:
    """Load card records."""
    path = os.path.join(input_dir, "01_financial_cards.parquet")
    logger.info("Loading cards from %s", path)
    df = pd.read_parquet(path)
    logger.info("Cards loaded: %d rows, %d cols", len(df), len(df.columns))
    return df


def load_transactions_chunked(input_dir: str) -> pd.DataFrame:
    """Load transactions from Parquet (preferred) or CSV fallback.

    The Parquet file is pre-cleaned by DuckDB: Amount already float,
    Is Fraud already int, column names already snake_case.
    """
    parquet_path = os.path.join(input_dir, "transactions.parquet")
    csv_path = os.path.join(input_dir, "credit_card_transactions-ibm_v2.csv")

    if os.path.exists(parquet_path):
        logger.info("Loading transactions from Parquet: %s", parquet_path)
        df = pd.read_parquet(parquet_path)
        # Parquet already has: user_id, card_id, year, month, day, time,
        # amount, use_chip, merchant_id, merchant_city, merchant_state,
        # zip, mcc, errors, is_fraud
        # Rename to match downstream expectations
        col_map = {
            "user_id": "User", "card_id": "Card", "year": "Year",
            "month": "Month", "day": "Day", "time": "Time",
            "amount": "Amount", "use_chip": "Use Chip",
            "merchant_id": "Merchant Name", "merchant_city": "Merchant City",
            "merchant_state": "Merchant State", "zip": "Zip",
            "mcc": "MCC", "errors": "Errors?", "is_fraud": "Is Fraud?",
        }
        df = df.rename(columns=col_map)
        df["Hour"] = df["Time"].apply(_parse_hour)
        df["Date"] = pd.to_datetime(
            df[["Year", "Month", "Day"]].rename(
                columns={"Year": "year", "Month": "month", "Day": "day"}
            )
        )
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["YearMonth"] = df["Year"] * 100 + df["Month"]
        logger.info("Loaded %d rows from Parquet", len(df))
        return df

    # Fallback: CSV chunked read
    path = csv_path
    logger.info("Loading transactions (chunked, %d rows/chunk) from %s",
                CHUNK_SIZE, path)

    chunks: List[pd.DataFrame] = []
    for i, chunk in enumerate(pd.read_csv(path, chunksize=CHUNK_SIZE)):
        # Clean Amount: strip '$' and convert
        chunk["Amount"] = (
            chunk["Amount"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.strip()
            .astype(float)
        )
        # Clean Is Fraud?
        chunk["Is Fraud?"] = (chunk["Is Fraud?"].str.strip().str.lower() == "yes").astype(int)

        # Parse time → hour
        chunk["Hour"] = chunk["Time"].apply(_parse_hour)

        # Derive date
        chunk["Date"] = pd.to_datetime(
            chunk[["Year", "Month", "Day"]].rename(
                columns={"Year": "year", "Month": "month", "Day": "day"}
            )
        )
        # Day of week (0=Monday)
        chunk["DayOfWeek"] = chunk["Date"].dt.dayofweek

        chunks.append(chunk)
        if (i + 1) % 5 == 0:
            logger.info("  ... loaded %d chunks (%d rows)",
                        i + 1, (i + 1) * CHUNK_SIZE)

    txn = pd.concat(chunks, ignore_index=True)
    logger.info("Transactions loaded: %d rows", len(txn))
    return txn


def _parse_hour(time_str) -> int:
    """Parse 'HH:MM' string → integer hour."""
    try:
        return int(str(time_str).split(":")[0])
    except (ValueError, IndexError):
        return 0


# ===================================================================
# 2. Feature Group Builders
# ===================================================================

def build_base_rfm(users: pd.DataFrame,
                   txn_agg: pd.DataFrame,
                   ref_date: pd.Timestamp) -> pd.DataFrame:
    """base_rfm (34D): demographics + RFM + basic txn stats."""
    df = users.copy()
    df["user_id"] = df.index  # User index = user_id

    # Demographics
    df["rfm_001"] = df["Current Age"].astype(float)
    df["rfm_002"] = df["Current Age"].astype(float) ** 2
    df["rfm_003"] = (df["Retirement Age"] - df["Current Age"]).astype(float)
    df["rfm_004"] = (df["Gender"].str.strip().str.lower() == "male").astype(float)
    df["rfm_005"] = df["Yearly Income"].astype(float)
    df["rfm_006"] = df["Per Capita Income"].astype(float)
    df["rfm_007"] = df["Total Debt"].astype(float)
    df["rfm_008"] = df["FICO Score"].astype(float)
    df["rfm_009"] = df["Num Credit Cards"].astype(float)

    # Merge transaction aggregates
    df = df.merge(txn_agg, left_on="user_id", right_index=True, how="left")

    # RFM
    df["rfm_010"] = (ref_date - df["last_txn_date"]).dt.days.fillna(9999).astype(float)  # Recency
    df["rfm_011"] = df["txn_count"].fillna(0).astype(float)          # Frequency
    df["rfm_012"] = df["txn_total_amount"].fillna(0).astype(float)   # Monetary

    # Basic txn stats
    df["rfm_013"] = df["txn_mean_amount"].fillna(0).astype(float)
    df["rfm_014"] = df["txn_std_amount"].fillna(0).astype(float)
    df["rfm_015"] = df["txn_max_amount"].fillna(0).astype(float)
    df["rfm_016"] = df["txn_min_amount"].fillna(0).astype(float)
    df["rfm_017"] = df["n_merchants"].fillna(0).astype(float)
    df["rfm_018"] = df["n_states"].fillna(0).astype(float)
    df["rfm_019"] = df["error_rate"].fillna(0).astype(float)

    # Pad remaining columns to 34D
    for i in range(20, 35):
        col = f"rfm_{i:03d}"
        if col not in df.columns:
            df[col] = 0.0

    rfm_cols = [f"rfm_{i:03d}" for i in range(1, 35)]
    return df[["user_id"] + rfm_cols]


def build_base_category(txn: pd.DataFrame, user_ids: np.ndarray) -> pd.DataFrame:
    """base_category (64D): MCC spending ratios + diversity metrics.

    Uses DuckDB for aggregation on the 24M-row transaction table.
    """
    con = _duckdb.connect()
    con.register("txn_cat", txn)

    # Top 60 MCCs by total transaction count
    top_mccs_df = con.execute(f"""
        SELECT "MCC", COUNT(*) AS cnt
        FROM txn_cat
        GROUP BY "MCC"
        ORDER BY cnt DESC
        LIMIT {N_TOP_MCC}
    """).df()
    top_mccs = top_mccs_df["MCC"].tolist()

    # Per-user per-MCC spending + user totals in one pass
    mcc_spend = con.execute("""
        SELECT "User", "MCC", SUM("Amount") AS mcc_amount
        FROM txn_cat
        GROUP BY "User", "MCC"
    """).df()

    user_total = mcc_spend.groupby("User")["mcc_amount"].sum()

    # Diversity metrics via DuckDB
    diversity = con.execute("""
        WITH user_mcc AS (
            SELECT "User", "MCC",
                   SUM("Amount") AS mcc_amt,
                   COUNT(*)      AS mcc_cnt
            FROM txn_cat
            GROUP BY "User", "MCC"
        ),
        user_totals AS (
            SELECT "User",
                   SUM(mcc_amt) AS total_amt,
                   COUNT(DISTINCT "MCC") AS n_mcc
            FROM user_mcc
            GROUP BY "User"
        ),
        user_probs AS (
            SELECT um."User", um."MCC",
                   um.mcc_amt / GREATEST(ut.total_amt, 1e-8) AS prob
            FROM user_mcc um
            JOIN user_totals ut ON um."User" = ut."User"
        )
        SELECT
            ut."User",
            ut.n_mcc,
            -- entropy: -SUM(p * ln(p))
            COALESCE((
                SELECT -SUM(up.prob * LN(GREATEST(up.prob, 1e-15)))
                FROM user_probs up WHERE up."User" = ut."User"
            ), 0) AS entropy,
            -- HHI: SUM(p^2)
            COALESCE((
                SELECT SUM(up.prob * up.prob)
                FROM user_probs up WHERE up."User" = ut."User"
            ), 0) AS hhi,
            -- Top-3 concentration: sum of 3 largest probabilities
            COALESCE((
                SELECT SUM(top3.prob)
                FROM (
                    SELECT up.prob
                    FROM user_probs up
                    WHERE up."User" = ut."User"
                    ORDER BY up.prob DESC
                    LIMIT 3
                ) top3
            ), 0) AS top3_conc
        FROM user_totals ut
        ORDER BY ut."User"
    """).df().set_index("User")

    con.unregister("txn_cat")
    con.close()

    # Build result DataFrame
    result = pd.DataFrame(index=user_ids)
    mcc_pivot = mcc_spend.pivot_table(
        index="User", columns="MCC", values="mcc_amount",
        aggfunc="sum", fill_value=0,
    )
    totals = user_total.reindex(user_ids, fill_value=1.0).clip(lower=1e-8)

    for idx, mcc in enumerate(top_mccs, start=1):
        col = f"category_{idx:03d}"
        if mcc in mcc_pivot.columns:
            result[col] = (mcc_pivot[mcc].reindex(user_ids, fill_value=0) / totals).values
        else:
            result[col] = 0.0
    # Pad if fewer than 60 MCCs
    for idx in range(len(top_mccs) + 1, N_TOP_MCC + 1):
        result[f"category_{idx:03d}"] = 0.0

    result["category_061"] = diversity["n_mcc"].reindex(user_ids, fill_value=0).astype(float).values
    result["category_062"] = diversity["entropy"].reindex(user_ids, fill_value=0).astype(float).values
    result["category_063"] = diversity["hhi"].reindex(user_ids, fill_value=0).astype(float).values
    result["category_064"] = diversity["top3_conc"].reindex(user_ids, fill_value=0).astype(float).values

    result["user_id"] = user_ids
    cat_cols = [f"category_{i:03d}" for i in range(1, 65)]
    return result[["user_id"] + cat_cols].reset_index(drop=True)


def build_base_txn_stats(txn: pd.DataFrame, user_ids: np.ndarray) -> pd.DataFrame:
    """base_txn_stats (80D): monthly counts/amounts, quarterly change, time-of-day, chip/swipe.

    Uses DuckDB for all heavy aggregation on the 24M-row transaction table.
    """
    con = _duckdb.connect()
    con.register("txn_stats", txn)

    # Monthly counts and amounts per user (24D)
    monthly_df = con.execute("""
            SELECT "User", "Month",
                   COUNT(*) AS monthly_count,
                   SUM("Amount") AS monthly_amount
            FROM txn_stats
            GROUP BY "User", "Month"
        """).df()

    # All ratios in a single aggregation query
    ratios_df = con.execute("""
        SELECT "User",
               COUNT(*)::FLOAT AS total_cnt,
               -- Weekday/weekend
               SUM(CASE WHEN "DayOfWeek" < 5 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS weekday_ratio,
               SUM(CASE WHEN "DayOfWeek" >= 5 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS weekend_ratio,
               -- Time-of-day bins (6 bins)
               SUM(CASE WHEN "Hour" >= 0  AND "Hour" < 4  THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS tod_0_4,
               SUM(CASE WHEN "Hour" >= 4  AND "Hour" < 8  THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS tod_4_8,
               SUM(CASE WHEN "Hour" >= 8  AND "Hour" < 12 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS tod_8_12,
               SUM(CASE WHEN "Hour" >= 12 AND "Hour" < 16 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS tod_12_16,
               SUM(CASE WHEN "Hour" >= 16 AND "Hour" < 20 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS tod_16_20,
               SUM(CASE WHEN "Hour" >= 20 AND "Hour" < 24 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS tod_20_24,
               -- Chip/Swipe/Online
               SUM(CASE WHEN TRIM("Use Chip") = 'Chip Transaction'   THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS chip_ratio,
               SUM(CASE WHEN TRIM("Use Chip") = 'Swipe Transaction'  THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS swipe_ratio,
               SUM(CASE WHEN TRIM("Use Chip") = 'Online Transaction' THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS online_ratio,
               -- Error and fraud counts
               SUM(CASE WHEN "Errors?" IS NOT NULL AND TRIM("Errors?") != '' THEN 1 ELSE 0 END) AS error_count,
               SUM("Is Fraud?") AS fraud_count
        FROM txn_stats
        GROUP BY "User"
        ORDER BY "User"
    """).df().set_index("User")

    con.unregister("txn_stats")
    con.close()

    # Build result from DuckDB aggregates
    result = pd.DataFrame({"user_id": user_ids})
    col_idx = 1

    # Monthly pivot
    monthly_pivot_cnt = monthly_df.pivot_table(
        index="User", columns="Month", values="monthly_count",
        aggfunc="sum", fill_value=0,
    )
    monthly_pivot_amt = monthly_df.pivot_table(
        index="User", columns="Month", values="monthly_amount",
        aggfunc="sum", fill_value=0,
    )

    # Monthly counts (12D)
    for m in range(1, 13):
        col = f"transaction_stats_{col_idx:03d}"
        result[col] = (monthly_pivot_cnt[m].reindex(user_ids, fill_value=0).values
                       if m in monthly_pivot_cnt.columns else 0.0)
        col_idx += 1

    # Monthly amounts (12D)
    for m in range(1, 13):
        col = f"transaction_stats_{col_idx:03d}"
        result[col] = (monthly_pivot_amt[m].reindex(user_ids, fill_value=0).values
                       if m in monthly_pivot_amt.columns else 0.0)
        col_idx += 1

    # Quarterly totals (8D)
    for q_months in [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]:
        col = f"transaction_stats_{col_idx:03d}"
        q_amt = sum(
            monthly_pivot_amt[m].reindex(user_ids, fill_value=0)
            for m in q_months if m in monthly_pivot_amt.columns
        )
        result[col] = q_amt.values if hasattr(q_amt, "values") else 0.0
        col_idx += 1
    for q_months in [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]:
        col = f"transaction_stats_{col_idx:03d}"
        q_cnt = sum(
            monthly_pivot_cnt[m].reindex(user_ids, fill_value=0)
            for m in q_months if m in monthly_pivot_cnt.columns
        )
        result[col] = q_cnt.values if hasattr(q_cnt, "values") else 0.0
        col_idx += 1

    # Weekday/weekend ratio (2D)
    result[f"transaction_stats_{col_idx:03d}"] = ratios_df["weekday_ratio"].reindex(user_ids, fill_value=0).values
    col_idx += 1
    result[f"transaction_stats_{col_idx:03d}"] = ratios_df["weekend_ratio"].reindex(user_ids, fill_value=0).values
    col_idx += 1

    # Time-of-day (6D)
    for tod_col in ["tod_0_4", "tod_4_8", "tod_8_12", "tod_12_16", "tod_16_20", "tod_20_24"]:
        result[f"transaction_stats_{col_idx:03d}"] = ratios_df[tod_col].reindex(user_ids, fill_value=0).values
        col_idx += 1

    # Chip/Swipe/Online (3D)
    for r_col in ["chip_ratio", "swipe_ratio", "online_ratio"]:
        result[f"transaction_stats_{col_idx:03d}"] = ratios_df[r_col].reindex(user_ids, fill_value=0).values
        col_idx += 1

    # Error count, fraud count (2D)
    result[f"transaction_stats_{col_idx:03d}"] = ratios_df["error_count"].reindex(user_ids, fill_value=0).values.astype(float)
    col_idx += 1
    result[f"transaction_stats_{col_idx:03d}"] = ratios_df["fraud_count"].reindex(user_ids, fill_value=0).values.astype(float)
    col_idx += 1

    # Pad to 80D
    while col_idx <= 80:
        result[f"transaction_stats_{col_idx:03d}"] = 0.0
        col_idx += 1

    txn_cols = [f"transaction_stats_{i:03d}" for i in range(1, 81)]
    return result[["user_id"] + txn_cols]


def build_base_temporal(txn: pd.DataFrame, user_ids: np.ndarray) -> pd.DataFrame:
    """base_temporal (60D): rolling means, trends, seasonality.

    Uses DuckDB for the heavy groupby aggregation on 24M rows, then
    builds rolling windows with numpy on the small (2000 x 12) pivot.
    """
    con = _duckdb.connect()
    con.register("txn_temp", txn)

    # Aggregate monthly amounts and counts + span in one shot
    monthly_df = con.execute("""
        SELECT "User", "Month",
               SUM("Amount") AS amt,
               COUNT(*)      AS cnt
        FROM txn_temp
        GROUP BY "User", "Month"
    """).df()

    spans_df = con.execute("""
        SELECT "User",
               DATE_DIFF('day', MIN("Date"), MAX("Date")) AS span_days
        FROM txn_temp
        GROUP BY "User"
    """).df().set_index("User")

    con.unregister("txn_temp")
    con.close()

    # Build pivots (small: 2000 x 12)
    monthly_amt = monthly_df.pivot_table(
        index="User", columns="Month", values="amt",
        aggfunc="sum", fill_value=0,
    )
    monthly_cnt = monthly_df.pivot_table(
        index="User", columns="Month", values="cnt",
        aggfunc="sum", fill_value=0,
    )
    for m in range(1, 13):
        if m not in monthly_amt.columns:
            monthly_amt[m] = 0.0
        if m not in monthly_cnt.columns:
            monthly_cnt[m] = 0.0
    monthly_amt = monthly_amt[range(1, 13)].reindex(user_ids, fill_value=0)
    monthly_cnt = monthly_cnt[range(1, 13)].reindex(user_ids, fill_value=0)

    # From here, operations are on small (2000 x 12) matrices — pure numpy
    result = pd.DataFrame({"user_id": user_ids})
    col_idx = 1

    # Rolling mean amount: 3/6/12 month windows (12 + 12 + 12 = 36D)
    for window in [3, 6, 12]:
        rolled = monthly_amt.T.rolling(window, min_periods=1).mean().T
        for m in range(1, 13):
            col = f"temporal_{col_idx:03d}"
            result[col] = rolled[m].values
            col_idx += 1

    # Rolling mean count: 3 month window (12D)
    rolled_cnt = monthly_cnt.T.rolling(3, min_periods=1).mean().T
    for m in range(1, 13):
        col = f"temporal_{col_idx:03d}"
        result[col] = rolled_cnt[m].values
        col_idx += 1

    # Trend slope (1D)
    col = f"temporal_{col_idx:03d}"
    x = np.arange(1, 13, dtype=float)
    slopes = monthly_amt.apply(
        lambda row: np.polyfit(x, row.values.astype(float), 1)[0]
        if row.sum() > 0 else 0.0,
        axis=1,
    )
    result[col] = slopes.values
    col_idx += 1

    # Active months (1D)
    col = f"temporal_{col_idx:03d}"
    active_months = (monthly_amt > 0).sum(axis=1)
    result[col] = active_months.values.astype(float)
    col_idx += 1

    # Span in days (1D)
    col = f"temporal_{col_idx:03d}"
    result[col] = spans_df["span_days"].reindex(user_ids, fill_value=0).values.astype(float)
    col_idx += 1

    # Pad to 60D
    while col_idx <= 60:
        col = f"temporal_{col_idx:03d}"
        result[col] = 0.0
        col_idx += 1

    temporal_cols = [f"temporal_{i:03d}" for i in range(1, 61)]
    return result[["user_id"] + temporal_cols]


def build_tda_topology(user_ids: np.ndarray,
                       txn: "pd.DataFrame | None",
                       base_features: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """tda_topology (70D): real TDA generator + autocorrelation / entropy stats.

    First 50D are computed by the real TDAFeatureGenerator (persistent
    homology with numpy eigenvalue fallback).  Remaining 20D come from
    handcrafted autocorrelation / entropy statistics when txn data is
    available, or zeros when running via the DuckDB path.
    """
    result = pd.DataFrame({"user_id": user_ids})
    n_users = len(user_ids)
    col_idx = 1

    # --- First 50D: real TDA features via TDAFeatureGenerator ---
    tda_filled = False
    if _HAS_TDA_GEN and base_features is not None:
        try:
            # Use all numeric columns from base features as point-cloud coords
            num_cols = [c for c in base_features.columns if c != "user_id"]
            if len(num_cols) >= 2:
                input_df = base_features[num_cols].copy()
                input_df.index = range(n_users)

                # Configure TDA to produce 50D:
                # max_homology_dim=4 gives H0..H4 = 5 dims * 10 stats = 50D
                # (we use 10 stats to reach 50D = 5 * 10)
                _tda_stats = [
                    "num_features", "mean_lifetime", "max_lifetime",
                    "std_lifetime", "entropy", "total_persistence",
                    "mean_birth", "mean_death",
                ]
                # 5 dims * 8 stats = 40D; need 50, so use max_homology_dim
                # that gets closest. 6 * 8 = 48; 7 * 8 = 56. Use 6 dims
                # (H0..H5) for 48D + pad 2, or just compute what we can.
                # Simplest: use max_homology_dim=1 (default) -> 2*8=16D,
                # then pad to 50. This still provides real topological signal.
                tda_gen = TDAFeatureGenerator(
                    input_columns=num_cols,
                    max_homology_dim=1,
                    n_points_subsample=200,
                    stats_to_compute=_tda_stats,
                    prefix="tda",
                )
                tda_gen.fit(input_df)
                tda_result = tda_gen.generate(input_df)

                # Extract generated values (up to 50D)
                gen_cols = [c for c in tda_result.columns]
                gen_values = tda_result[gen_cols].values.astype(np.float32)
                n_gen_dims = gen_values.shape[1]

                for j in range(min(n_gen_dims, 50)):
                    result[f"tda_{col_idx:03d}"] = gen_values[:, j]
                    col_idx += 1

                tda_filled = True
                logger.info("TDA generator produced %d real features", min(n_gen_dims, 50))
        except Exception as exc:
            logger.warning("TDA generator failed, falling back to zeros: %s", exc)
            tda_filled = False

    # Fill remaining first-50 slots with zeros if generator didn't fill them
    while col_idx <= 50:
        result[f"tda_{col_idx:03d}"] = 0.0
        col_idx += 1

    # --- Remaining 20D: autocorrelation / entropy / CV / peaks ---
    # If txn is None (DuckDB path), fill remaining 20D as 0
    if txn is None:
        for i in range(20):
            result[f"tda_{col_idx:03d}"] = 0.0
            col_idx += 1
        tda_cols = [f"tda_{i:03d}" for i in range(1, 71)]
        return result[["user_id"] + tda_cols]

    # Autocorrelation of monthly amount (lag 1-12) -> 12D
    # Use DuckDB for the heavy 24M-row groupby, then numpy for autocorrelation
    con = _duckdb.connect()
    con.register("txn_tda", txn)
    _tda_monthly = con.execute("""
        SELECT "User", "Month", SUM("Amount") AS amt
        FROM txn_tda GROUP BY "User", "Month"
    """).df()
    con.unregister("txn_tda")
    con.close()
    monthly_amt = _tda_monthly.pivot_table(
        index="User", columns="Month", values="amt",
        aggfunc="sum", fill_value=0,
    )
    for m in range(1, 13):
        if m not in monthly_amt.columns:
            monthly_amt[m] = 0.0
    monthly_amt = monthly_amt[range(1, 13)].reindex(user_ids, fill_value=0)

    for lag in range(1, 13):
        col = f"tda_{col_idx:03d}"
        vals = monthly_amt.values
        if vals.shape[1] > lag:
            x = vals[:, :-lag]
            y = vals[:, lag:]
            # Per-row correlation
            x_mean = x.mean(axis=1, keepdims=True)
            y_mean = y.mean(axis=1, keepdims=True)
            num = ((x - x_mean) * (y - y_mean)).sum(axis=1)
            den = np.sqrt(((x - x_mean) ** 2).sum(axis=1) * ((y - y_mean) ** 2).sum(axis=1)).clip(min=1e-8)
            result[col] = num / den
        else:
            result[col] = 0.0
        col_idx += 1

    # Entropy (1D)
    probs = monthly_amt.div(monthly_amt.sum(axis=1).clip(lower=1e-8), axis=0)
    ent = -(probs * np.log(probs.clip(lower=1e-15))).sum(axis=1)
    result[f"tda_{col_idx:03d}"] = ent.values
    col_idx += 1

    # CV (1D)
    cv = monthly_amt.std(axis=1) / monthly_amt.mean(axis=1).clip(lower=1e-8)
    result[f"tda_{col_idx:03d}"] = cv.values
    col_idx += 1

    # Number of peaks (1D)
    def count_peaks(row):
        arr = row.values.astype(float)
        peaks = 0
        for j in range(1, len(arr) - 1):
            if arr[j] > arr[j - 1] and arr[j] > arr[j + 1]:
                peaks += 1
        return peaks

    n_peaks = monthly_amt.apply(count_peaks, axis=1)
    result[f"tda_{col_idx:03d}"] = n_peaks.values.astype(float)
    col_idx += 1

    # Pad remaining to 70D
    while col_idx <= 70:
        result[f"tda_{col_idx:03d}"] = 0.0
        col_idx += 1

    tda_cols = [f"tda_{i:03d}" for i in range(1, 71)]
    return result[["user_id"] + tda_cols]


def build_hmm_states(user_ids: np.ndarray,
                     base_features: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """hmm_states (48D): real HMM generator (triple-mode Baum-Welch).

    Uses HMMFeatureGenerator with its numpy EM fallback.  The generator
    produces 25D real features (journey 8D + lifecycle 8D + behavior 9D).
    Remaining 23D are padded with zeros to maintain the 48D contract.
    """
    result = pd.DataFrame({"user_id": user_ids})
    n_users = len(user_ids)
    hmm_filled = False

    if _HAS_HMM_GEN and base_features is not None:
        try:
            num_cols = [c for c in base_features.columns if c != "user_id"]
            if len(num_cols) >= 1:
                input_df = base_features[num_cols].copy()
                input_df.index = range(n_users)

                hmm_gen = HMMFeatureGenerator(
                    modes=["journey", "lifecycle", "behavior"],
                    sequence_columns=num_cols,
                    prefix="hmm",
                )
                hmm_gen.fit(input_df)
                hmm_result = hmm_gen.generate(input_df)

                gen_cols = list(hmm_result.columns)
                gen_values = hmm_result[gen_cols].values.astype(np.float32)
                n_gen_dims = gen_values.shape[1]

                for j in range(min(n_gen_dims, 48)):
                    result[f"hmm_{j + 1:03d}"] = gen_values[:, j]

                hmm_filled = True
                logger.info("HMM generator produced %d real features (padded to 48D)",
                            min(n_gen_dims, 48))
        except Exception as exc:
            logger.warning("HMM generator failed, falling back to zeros: %s", exc)
            hmm_filled = False

    # Fill any remaining columns up to 48D with zeros
    for i in range(1, 49):
        col = f"hmm_{i:03d}"
        if col not in result.columns:
            result[col] = 0.0

    hmm_cols = [f"hmm_{i:03d}" for i in range(1, 49)]
    return result[["user_id"] + hmm_cols]


def build_mamba_temporal(user_ids: np.ndarray,
                        base_features: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """mamba_temporal (50D): real Mamba SSM generator (numpy matrix-exp fallback).

    Uses MambaFeatureGenerator with output_dim=50.  The generator has a
    full numpy-only SSM fallback, so it works without torch/CUDA.  Each
    user's feature row is treated as a length-1 sequence; the SSM
    processes it and produces a 50D compressed embedding via PCA.
    """
    result = pd.DataFrame({"user_id": user_ids})
    n_users = len(user_ids)
    mamba_filled = False

    if _HAS_MAMBA_GEN and base_features is not None:
        try:
            num_cols = [c for c in base_features.columns if c != "user_id"]
            if len(num_cols) >= 1:
                # Build a DataFrame with user_id for entity grouping
                input_df = base_features[num_cols].copy()
                input_df["user_id"] = user_ids
                input_df.index = range(n_users)

                mamba_gen = MambaFeatureGenerator(
                    output_dim=50,
                    seq_len=1,
                    d_model=64,   # smaller model for adapter context
                    d_state=8,
                    num_epochs=5,  # fewer epochs for faster processing
                    entity_column="user_id",
                    feature_columns=num_cols,
                    prefix="mamba",
                    prefer_gpu=False,  # SageMaker Processing may not have GPU
                )
                mamba_gen.fit(input_df)
                mamba_result = mamba_gen.generate(input_df)

                gen_cols = list(mamba_result.columns)
                gen_values = mamba_result[gen_cols].values.astype(np.float32)
                n_gen_dims = gen_values.shape[1]

                for j in range(min(n_gen_dims, 50)):
                    result[f"mamba_{j + 1:03d}"] = gen_values[:, j]

                mamba_filled = True
                logger.info("Mamba generator produced %d real features", min(n_gen_dims, 50))
        except Exception as exc:
            logger.warning("Mamba generator failed, falling back to zeros: %s", exc)
            mamba_filled = False

    # Fill any remaining columns up to 50D with zeros
    for i in range(1, 51):
        col = f"mamba_{i:03d}"
        if col not in result.columns:
            result[col] = 0.0

    mamba_cols = [f"mamba_{i:03d}" for i in range(1, 51)]
    return result[["user_id"] + mamba_cols]


def build_gmm_clustering(base_features: pd.DataFrame,
                         user_ids: np.ndarray) -> pd.DataFrame:
    """gmm_clustering (22D): Gaussian Mixture Model with BIC-based cluster selection.

    Uses base_rfm + base_txn_stats numeric features for clustering.
    Selects optimal cluster count via BIC, then uses GMM predict_proba
    for soft assignment.

    Output columns (always 22):
      gmm_001..gmm_{k} = cluster membership probabilities (k = selected clusters)
      gmm_{k+1}..gmm_020 = zero-padded if k < 20
      gmm_021 = entropy of membership probabilities
      gmm_022 = hard assignment (argmax cluster label)
    """
    import time
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    result = pd.DataFrame({"user_id": user_ids})

    # Select numeric features for clustering
    feat_cols = [c for c in base_features.columns if c != "user_id"]
    X = base_features[feat_cols].fillna(0).values.astype(np.float64)

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_samples = len(user_ids)

    # --- Dynamic cluster count selection via BIC ---
    k_min = 2
    k_max = min(GMM_MAX_CLUSTERS, n_samples // 10)
    k_max = max(k_min, k_max)  # ensure at least k_min

    # Build candidate list; use coarser grid when range is large
    if k_max - k_min + 1 > 15:
        # coarse grid: step of 2, always include k_min and k_max
        candidates = list(range(k_min, k_max + 1, 2))
        if candidates[-1] != k_max:
            candidates.append(k_max)
    else:
        candidates = list(range(k_min, k_max + 1))

    logger.info(
        "GMM BIC search: n_samples=%d, candidate k range [%d..%d], %d candidates",
        n_samples, candidates[0], candidates[-1], len(candidates),
    )

    best_bic = np.inf
    best_k = candidates[0]
    bic_log = []
    t0 = time.time()

    for k in candidates:
        gmm_candidate = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=RANDOM_STATE,
            n_init=1,
            max_iter=200,
        )
        gmm_candidate.fit(X_scaled)
        bic = gmm_candidate.bic(X_scaled)
        bic_log.append((k, bic))
        if bic < best_bic:
            best_bic = bic
            best_k = k

        # Time guard: if we already spent > 120s, stop early
        elapsed = time.time() - t0
        if elapsed > 120:
            logger.warning(
                "GMM BIC search time guard triggered after %.1fs at k=%d", elapsed, k
            )
            break

    elapsed_total = time.time() - t0
    logger.info(
        "GMM BIC search done in %.1fs — selected k=%d (BIC=%.2f)",
        elapsed_total, best_k, best_bic,
    )
    for k, bic in bic_log:
        logger.debug("  k=%d  BIC=%.2f%s", k, bic, "  <-- best" if k == best_k else "")

    # --- Fit final GMM with best k ---
    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type="full",
        random_state=RANDOM_STATE,
        n_init=3,
        max_iter=300,
    )
    gmm.fit(X_scaled)

    # Soft assignment via predict_proba (native GMM posterior probabilities)
    probs = gmm.predict_proba(X_scaled)  # (n_samples, best_k)
    hard_labels = gmm.predict(X_scaled)

    col_idx = 1

    # --- (1) k개 클러스터 소속 확률 ---
    for i in range(best_k):
        result[f"gmm_{col_idx:03d}"] = probs[:, i]
        col_idx += 1

    # --- (2) 엔트로피: 소속 불확실성 ---
    ent = -(probs * np.log(probs.clip(min=1e-15))).sum(axis=1)
    result[f"gmm_{col_idx:03d}"] = ent
    col_idx += 1

    # --- (3) margin: 1위-2위 확률 차이 (경계 사용자 감지) ---
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]  # descending
    margin = sorted_probs[:, 0] - (sorted_probs[:, 1] if best_k >= 2 else 0.0)
    result[f"gmm_{col_idx:03d}"] = margin
    col_idx += 1

    # --- (4) Mahalanobis distance (소속 클러스터 중심까지) ---
    mahal_dists = np.zeros(len(user_ids))
    for ki in range(best_k):
        mask = hard_labels == ki
        if not mask.any():
            continue
        try:
            # precision_matrix = inverse covariance (already computed by GMM)
            prec = gmm.precisions_cholesky_[ki]
            diff = X_scaled[mask] - gmm.means_[ki]
            # Mahalanobis = sqrt(diff @ precision @ diff^T) per sample
            transformed = diff @ prec
            mahal_dists[mask] = np.sqrt((transformed ** 2).sum(axis=1))
        except Exception:
            # Fallback: Euclidean distance
            diff = X_scaled[mask] - gmm.means_[ki]
            mahal_dists[mask] = np.sqrt((diff ** 2).sum(axis=1))
    result[f"gmm_{col_idx:03d}"] = mahal_dists
    col_idx += 1

    # --- (5) 소속 클러스터 크기 비율 (소수/다수 그룹 구분) ---
    cluster_counts = np.bincount(hard_labels, minlength=best_k)
    cluster_ratios = cluster_counts / len(user_ids)
    assigned_ratio = cluster_ratios[hard_labels]
    result[f"gmm_{col_idx:03d}"] = assigned_ratio
    col_idx += 1

    # --- (6) log-likelihood (GMM 전체 적합도) ---
    log_likelihoods = gmm.score_samples(X_scaled)  # per-sample log-likelihood
    result[f"gmm_{col_idx:03d}"] = log_likelihoods
    col_idx += 1

    # --- (7) hard assignment ---
    result[f"gmm_{col_idx:03d}"] = hard_labels.astype(float)
    col_idx += 1

    gmm_cols = [f"gmm_{i:03d}" for i in range(1, col_idx)]
    logger.info(
        "GMM output: %d columns (k=%d probs + 5 derived + hard assignment)",
        len(gmm_cols), best_k,
    )
    return result[["user_id"] + gmm_cols]


def build_economics(users: pd.DataFrame,
                    txn: pd.DataFrame,
                    user_ids: np.ndarray) -> pd.DataFrame:
    """economics (17D): spending ratios, MPC, savings rate, volatility.

    Uses DuckDB for the heavy monthly groupby aggregation on 24M rows,
    then computes ratios on the small (2000 x 12) result.
    """
    result = pd.DataFrame({"user_id": user_ids})
    col_idx = 1

    yearly_income = users["Yearly Income"].reindex(user_ids).fillna(1).astype(float).clip(lower=1)
    total_debt = users["Total Debt"].reindex(user_ids).fillna(0).astype(float).clip(lower=1)

    # Monthly spending — use DuckDB for the 24M-row aggregation
    con = _duckdb.connect()
    con.register("txn_econ", txn)
    monthly_spend_df = con.execute("""
        SELECT "User", "Month", SUM("Amount") AS monthly_amount
        FROM txn_econ
        GROUP BY "User", "Month"
    """).df()
    con.unregister("txn_econ")
    con.close()
    monthly_spend = monthly_spend_df.pivot_table(
        index="User", columns="Month", values="monthly_amount",
        aggfunc="sum", fill_value=0,
    )
    for m in range(1, 13):
        if m not in monthly_spend.columns:
            monthly_spend[m] = 0.0
    monthly_spend = monthly_spend[range(1, 13)].reindex(user_ids, fill_value=0)

    monthly_income = yearly_income / 12.0

    # Monthly spending / income ratio (12D)
    for m in range(1, 13):
        col = f"econ_{col_idx:03d}"
        result[col] = (monthly_spend[m].values / monthly_income.values)
        col_idx += 1

    # Debt-to-spending ratio (1D)
    total_spend = monthly_spend.sum(axis=1).values
    col = f"econ_{col_idx:03d}"
    result[col] = total_spend / total_debt.values.clip(min=1e-8)
    col_idx += 1

    # MPC approximation: diff of monthly spending / monthly income (1D)
    monthly_diffs = np.diff(monthly_spend.values, axis=1)
    mpc = np.mean(monthly_diffs, axis=1) / monthly_income.values.clip(min=1e-8)
    col = f"econ_{col_idx:03d}"
    result[col] = mpc
    col_idx += 1

    # Savings rate: 1 - total_spend / yearly_income (1D)
    col = f"econ_{col_idx:03d}"
    result[col] = 1.0 - (total_spend / yearly_income.values.clip(min=1e-8))
    col_idx += 1

    # Consumption volatility: CV of monthly spending (1D)
    col = f"econ_{col_idx:03d}"
    m_mean = monthly_spend.mean(axis=1).values.clip(min=1e-8)
    m_std = monthly_spend.std(axis=1).values
    result[col] = m_std / m_mean
    col_idx += 1

    # Pad to 17D
    while col_idx <= 17:
        col = f"econ_{col_idx:03d}"
        result[col] = 0.0
        col_idx += 1

    econ_cols = [f"econ_{i:03d}" for i in range(1, 18)]
    return result[["user_id"] + econ_cols]


def build_economics_from_aggs(users: pd.DataFrame,
                               econ_monthly: pd.DataFrame,
                               user_ids: np.ndarray) -> pd.DataFrame:
    """economics (17D) from pre-aggregated DuckDB data.

    Parameters
    ----------
    users : DataFrame with columns Yearly Income, Total Debt (indexed 0..N-1).
    econ_monthly : DataFrame(User, Year, Month, monthly_spend) from DuckDB.
    user_ids : np.array of user indices starting at 0.

    Returns
    -------
    DataFrame with columns [user_id, econ_001 .. econ_017].
    """
    result = pd.DataFrame({"user_id": user_ids})
    col_idx = 1

    yearly_income = users["Yearly Income"].reindex(user_ids).fillna(1).astype(float).clip(lower=1)
    total_debt = users["Total Debt"].reindex(user_ids).fillna(0).astype(float).clip(lower=1)
    monthly_income = yearly_income / 12.0

    # Build a (user x 12-month) pivot from econ_monthly.
    # Use the latest year available per user; fall back to Month-only pivot.
    if econ_monthly is not None and len(econ_monthly) > 0:
        # Keep only the latest year per user to get 12 monthly buckets
        latest_year = econ_monthly.groupby("User")["Year"].max().rename("max_year")
        em = econ_monthly.merge(latest_year, on="User")
        em = em[em["Year"] == em["max_year"]]
        monthly_pivot = em.pivot_table(
            index="User", columns="Month", values="monthly_spend",
            aggfunc="sum", fill_value=0,
        )
    else:
        monthly_pivot = pd.DataFrame(index=user_ids)

    for m in range(1, 13):
        if m not in monthly_pivot.columns:
            monthly_pivot[m] = 0.0
    monthly_pivot = monthly_pivot[range(1, 13)].reindex(user_ids, fill_value=0)

    # Monthly spending / income ratio (12D)
    for m in range(1, 13):
        col = f"econ_{col_idx:03d}"
        result[col] = monthly_pivot[m].values / monthly_income.values
        col_idx += 1

    # Debt-to-spending ratio (1D)
    total_spend = monthly_pivot.sum(axis=1).values
    col = f"econ_{col_idx:03d}"
    result[col] = total_spend / total_debt.values.clip(min=1e-8)
    col_idx += 1

    # MPC approximation (1D)
    monthly_diffs = np.diff(monthly_pivot.values, axis=1)
    mpc = np.mean(monthly_diffs, axis=1) / monthly_income.values.clip(min=1e-8)
    col = f"econ_{col_idx:03d}"
    result[col] = mpc
    col_idx += 1

    # Savings rate (1D)
    col = f"econ_{col_idx:03d}"
    result[col] = 1.0 - (total_spend / yearly_income.values.clip(min=1e-8))
    col_idx += 1

    # Consumption volatility: CV of monthly spending (1D)
    col = f"econ_{col_idx:03d}"
    m_mean = monthly_pivot.mean(axis=1).values.clip(min=1e-8)
    m_std = monthly_pivot.std(axis=1).values
    result[col] = m_std / m_mean
    col_idx += 1

    # Pad to 17D
    while col_idx <= 17:
        col = f"econ_{col_idx:03d}"
        result[col] = 0.0
        col_idx += 1

    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    econ_cols = [f"econ_{i:03d}" for i in range(1, 18)]
    return result[["user_id"] + econ_cols]


def build_multidisciplinary(txn: pd.DataFrame,
                            user_ids: np.ndarray) -> pd.DataFrame:
    """multidisciplinary (24D): chemical kinetics, epidemic, interference, crime.

    Uses DuckDB for the heavy groupby aggregation on 24M rows, then
    computes derived metrics on the small (2000 x 12) matrices with numpy.
    """
    result = pd.DataFrame({"user_id": user_ids})
    col_idx = 1

    con = _duckdb.connect()
    con.register("txn_multi", txn)

    # Monthly amounts
    monthly_df = con.execute("""
        SELECT "User", "Month", SUM("Amount") AS amt
        FROM txn_multi GROUP BY "User", "Month"
    """).df()

    # Quarterly unique merchants + total unique merchants
    merch_qtr = con.execute("""
        SELECT "User",
               CEIL("Month" / 3.0)::INT AS qtr,
               COUNT(DISTINCT "Merchant Name") AS n_merchants
        FROM txn_multi GROUP BY "User", CEIL("Month" / 3.0)::INT
    """).df()

    total_merch_df = con.execute("""
        SELECT "User",
               COUNT(DISTINCT "Merchant Name") AS total_merchants
        FROM txn_multi GROUP BY "User"
    """).df().set_index("User")

    # Anomaly counts: txns > user mean + 3*std
    anomaly_df = con.execute("""
        WITH user_stats AS (
            SELECT "User",
                   AVG("Amount") AS mean_amt,
                   COALESCE(STDDEV("Amount"), 0) AS std_amt
            FROM txn_multi GROUP BY "User"
        )
        SELECT t."User",
               SUM(CASE WHEN t."Amount" > us.mean_amt + 3 * us.std_amt THEN 1 ELSE 0 END) AS anomaly_count,
               COUNT(*) AS total_count
        FROM txn_multi t
        JOIN user_stats us ON t."User" = us."User"
        GROUP BY t."User"
    """).df().set_index("User")

    con.unregister("txn_multi")
    con.close()

    # Build monthly pivot from DuckDB result
    monthly_amt = monthly_df.pivot_table(
        index="User", columns="Month", values="amt",
        aggfunc="sum", fill_value=0,
    )

    for m in range(1, 13):
        if m not in monthly_amt.columns:
            monthly_amt[m] = 0.0
    monthly_amt = monthly_amt[range(1, 13)].reindex(user_ids, fill_value=0)

    # --- Chemical kinetics: decay rate (recent 3m vs previous 3m) (6D) ---
    recent_3m = monthly_amt[[10, 11, 12]].mean(axis=1).values
    prev_3m = monthly_amt[[7, 8, 9]].mean(axis=1).values.clip(min=1e-8)
    decay_rate = recent_3m / prev_3m

    result[f"multi_{col_idx:03d}"] = decay_rate
    col_idx += 1
    result[f"multi_{col_idx:03d}"] = np.log1p(np.abs(decay_rate - 1))
    col_idx += 1
    # First half vs second half decay
    h1 = monthly_amt[range(1, 7)].mean(axis=1).values.clip(min=1e-8)
    h2 = monthly_amt[range(7, 13)].mean(axis=1).values
    result[f"multi_{col_idx:03d}"] = h2 / h1
    col_idx += 1
    result[f"multi_{col_idx:03d}"] = np.log1p(np.abs(h2 / h1 - 1))
    col_idx += 1
    result[f"multi_{col_idx:03d}"] = (recent_3m - prev_3m)
    col_idx += 1
    result[f"multi_{col_idx:03d}"] = np.where(decay_rate > 1, 1.0, 0.0)
    col_idx += 1

    # --- Epidemic model: merchant spread rate (5D) ---
    for q in range(1, 5):
        col = f"multi_{col_idx:03d}"
        q_data = merch_qtr[merch_qtr["qtr"] == q].set_index("User")["n_merchants"]
        result[col] = q_data.reindex(user_ids, fill_value=0).values.astype(float)
        col_idx += 1
    span_months = (monthly_amt > 0).sum(axis=1).clip(lower=1)
    result[f"multi_{col_idx:03d}"] = (
        total_merch_df["total_merchants"].reindex(user_ids, fill_value=0).values / span_months.values
    )
    col_idx += 1

    # --- Interference pattern: FFT top-3 freq + amplitude (8D) ---
    vals_arr = monthly_amt.values.astype(float)
    fft_result = np.fft.rfft(vals_arr, axis=1)
    fft_mag = np.abs(fft_result)
    # Skip DC component (index 0)
    fft_mag_nodc = fft_mag[:, 1:]
    # Top-3 frequencies
    top3_idx = np.argsort(-fft_mag_nodc, axis=1)[:, :3]  # (n_users, 3)

    for k in range(3):
        col = f"multi_{col_idx:03d}"
        result[col] = (top3_idx[:, k] + 1).astype(float)  # frequency index
        col_idx += 1
    for k in range(3):
        col = f"multi_{col_idx:03d}"
        # Amplitude of top-k frequency
        result[col] = np.array([fft_mag_nodc[i, top3_idx[i, k]] for i in range(len(user_ids))])
        col_idx += 1
    # Total spectral energy (1D)
    result[f"multi_{col_idx:03d}"] = fft_mag_nodc.sum(axis=1)
    col_idx += 1
    # Dominant frequency ratio (1D)
    top1_amp = np.array([fft_mag_nodc[i, top3_idx[i, 0]] for i in range(len(user_ids))])
    result[f"multi_{col_idx:03d}"] = top1_amp / fft_mag_nodc.sum(axis=1).clip(min=1e-8)
    col_idx += 1

    # --- Crime pattern: anomalous transaction frequency (5D) ---
    anom_cnt = anomaly_df["anomaly_count"].reindex(user_ids, fill_value=0).values.astype(float)
    anom_total = anomaly_df["total_count"].reindex(user_ids, fill_value=1).clip(lower=1).values.astype(float)
    result[f"multi_{col_idx:03d}"] = anom_cnt
    col_idx += 1
    result[f"multi_{col_idx:03d}"] = anom_cnt / anom_total
    col_idx += 1

    # Pad to 24D
    while col_idx <= 24:
        result[f"multi_{col_idx:03d}"] = 0.0
        col_idx += 1

    multi_cols = [f"multi_{i:03d}" for i in range(1, 25)]
    return result[["user_id"] + multi_cols]


def build_multidisciplinary_from_aggs(multi_monthly: pd.DataFrame,
                                       user_ids: np.ndarray) -> pd.DataFrame:
    """multidisciplinary (24D) from pre-aggregated DuckDB data.

    Parameters
    ----------
    multi_monthly : DataFrame(User, ym, amt, new_merchants, anomaly_count).
    user_ids : np.array of user indices starting at 0.

    Returns
    -------
    DataFrame with columns [user_id, multi_001 .. multi_024].
    """
    result = pd.DataFrame({"user_id": user_ids})
    col_idx = 1

    if multi_monthly is None or len(multi_monthly) == 0:
        for i in range(1, 25):
            result[f"multi_{i:03d}"] = 0.0
        return result

    # Build per-user monthly amount pivot (User x ym sorted)
    # Use last 12 time slots for consistency with original function
    all_yms = sorted(multi_monthly["ym"].unique())
    last_12 = all_yms[-12:] if len(all_yms) >= 12 else all_yms

    amt_pivot = multi_monthly.pivot_table(
        index="User", columns="ym", values="amt", aggfunc="sum", fill_value=0,
    )
    # Ensure we have exactly 12 columns; pad if needed
    for ym in last_12:
        if ym not in amt_pivot.columns:
            amt_pivot[ym] = 0.0
    amt_12 = amt_pivot[last_12].reindex(user_ids, fill_value=0)
    vals = amt_12.values.astype(float)
    n_periods = vals.shape[1]

    # --- Chemical kinetics: decay rate (6D) ---
    # recent 3 vs previous 3 months
    if n_periods >= 6:
        recent_3m = vals[:, -3:].mean(axis=1)
        prev_3m = vals[:, -6:-3].mean(axis=1).clip(min=1e-8)
    else:
        recent_3m = vals[:, -(min(3, n_periods)):].mean(axis=1) if n_periods > 0 else np.zeros(len(user_ids))
        prev_3m = np.ones(len(user_ids))
    decay_rate = recent_3m / prev_3m

    result[f"multi_{col_idx:03d}"] = decay_rate;  col_idx += 1
    result[f"multi_{col_idx:03d}"] = np.log1p(np.abs(decay_rate - 1));  col_idx += 1

    # First half vs second half
    half = n_periods // 2 if n_periods >= 2 else 1
    h1 = vals[:, :half].mean(axis=1).clip(min=1e-8)
    h2 = vals[:, half:].mean(axis=1)
    result[f"multi_{col_idx:03d}"] = h2 / h1;  col_idx += 1
    result[f"multi_{col_idx:03d}"] = np.log1p(np.abs(h2 / h1 - 1));  col_idx += 1
    result[f"multi_{col_idx:03d}"] = recent_3m - prev_3m;  col_idx += 1
    result[f"multi_{col_idx:03d}"] = np.where(decay_rate > 1, 1.0, 0.0);  col_idx += 1

    # --- Epidemic model: merchant spread rate (5D) ---
    # Quarter-level new merchant counts
    merch_pivot = multi_monthly.pivot_table(
        index="User", columns="ym", values="new_merchants", aggfunc="sum", fill_value=0,
    )
    merch_12 = merch_pivot.reindex(columns=last_12, fill_value=0).reindex(user_ids, fill_value=0)
    merch_vals = merch_12.values.astype(float)
    q_size = max(1, n_periods // 4)
    for q in range(4):
        col = f"multi_{col_idx:03d}"
        start = q * q_size
        end = min(start + q_size, n_periods)
        if start < n_periods:
            result[col] = merch_vals[:, start:end].sum(axis=1)
        else:
            result[col] = 0.0
        col_idx += 1
    # Overall merchant spread rate
    active_months = (amt_12 > 0).sum(axis=1).clip(lower=1).values
    result[f"multi_{col_idx:03d}"] = merch_vals.sum(axis=1) / active_months;  col_idx += 1

    # --- Interference pattern: FFT (8D) ---
    fft_result = np.fft.rfft(vals, axis=1)
    fft_mag = np.abs(fft_result)
    fft_mag_nodc = fft_mag[:, 1:] if fft_mag.shape[1] > 1 else np.zeros((len(user_ids), 1))
    n_freq = fft_mag_nodc.shape[1]

    if n_freq >= 3:
        top3_idx = np.argsort(-fft_mag_nodc, axis=1)[:, :3]
    else:
        top3_idx = np.zeros((len(user_ids), 3), dtype=int)
        for k in range(min(n_freq, 3)):
            top3_idx[:, k] = np.argsort(-fft_mag_nodc, axis=1)[:, k] if n_freq > k else 0

    for k in range(3):
        result[f"multi_{col_idx:03d}"] = (top3_idx[:, k] + 1).astype(float);  col_idx += 1
    for k in range(3):
        result[f"multi_{col_idx:03d}"] = np.array([
            fft_mag_nodc[i, top3_idx[i, k]] if top3_idx[i, k] < n_freq else 0.0
            for i in range(len(user_ids))
        ]);  col_idx += 1
    # Total spectral energy
    result[f"multi_{col_idx:03d}"] = fft_mag_nodc.sum(axis=1);  col_idx += 1
    # Dominant frequency ratio
    top1_amp = np.array([
        fft_mag_nodc[i, top3_idx[i, 0]] if top3_idx[i, 0] < n_freq else 0.0
        for i in range(len(user_ids))
    ])
    result[f"multi_{col_idx:03d}"] = top1_amp / fft_mag_nodc.sum(axis=1).clip(min=1e-8);  col_idx += 1

    # --- Crime pattern: anomalous transaction frequency (5D) ---
    anom_pivot = multi_monthly.pivot_table(
        index="User", columns="ym", values="anomaly_count", aggfunc="sum", fill_value=0,
    )
    anom_12 = anom_pivot.reindex(columns=last_12, fill_value=0).reindex(user_ids, fill_value=0)
    total_anom = anom_12.values.astype(float).sum(axis=1)
    result[f"multi_{col_idx:03d}"] = total_anom;  col_idx += 1

    # Anomaly ratio (over total periods)
    total_txn_proxy = vals.sum(axis=1).clip(min=1e-8)
    result[f"multi_{col_idx:03d}"] = total_anom / (total_txn_proxy / total_txn_proxy.mean()).clip(min=1e-8);  col_idx += 1

    # Pad to 24D
    while col_idx <= 24:
        result[f"multi_{col_idx:03d}"] = 0.0
        col_idx += 1

    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    multi_cols = [f"multi_{i:03d}" for i in range(1, 25)]
    return result[["user_id"] + multi_cols]


def build_model_derived(user_ids: np.ndarray,
                        base_features: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """model_derived (27D): real ModelFeaturesGenerator (HMM summary + Bandit + LNN).

    Uses ModelFeaturesGenerator which produces exactly 27D:
      - 5D HMM summary (KMeans approximation)
      - 4D Bandit/MAB exploration metrics
      - 18D LNN temporal dynamics
    All computations are numpy / sklearn based (no GPU needed).
    """
    result = pd.DataFrame({"user_id": user_ids})
    n_users = len(user_ids)
    model_filled = False

    if _HAS_MODEL_GEN and base_features is not None:
        try:
            num_cols = [c for c in base_features.columns if c != "user_id"]
            if len(num_cols) >= 2:
                input_df = base_features[num_cols].copy()
                input_df.index = range(n_users)

                model_gen = ModelFeaturesGenerator(
                    feature_columns=num_cols,
                    engagement_columns=num_cols,
                    temporal_columns=num_cols[:2],  # first 2 numeric cols for LNN
                    prefix="",  # no prefix, generator adds its own names
                )
                model_gen.fit(input_df)
                model_result = model_gen.generate(input_df)

                gen_cols = list(model_result.columns)
                gen_values = model_result[gen_cols].values.astype(np.float32)
                n_gen_dims = gen_values.shape[1]

                for j in range(min(n_gen_dims, 27)):
                    result[f"model_derived_{j + 1:03d}"] = gen_values[:, j]

                model_filled = True
                logger.info("ModelFeatures generator produced %d real features", min(n_gen_dims, 27))
        except Exception as exc:
            logger.warning("ModelFeatures generator failed, falling back to zeros: %s", exc)
            model_filled = False

    # Fill any remaining columns up to 27D with zeros
    for i in range(1, 28):
        col = f"model_derived_{i:03d}"
        if col not in result.columns:
            result[col] = 0.0

    md_cols = [f"model_derived_{i:03d}" for i in range(1, 28)]
    return result[["user_id"] + md_cols]


def build_merchant_hierarchy(txn: pd.DataFrame,
                             cards: pd.DataFrame,
                             user_ids: np.ndarray) -> pd.DataFrame:
    """merchant_hierarchy (21D): MCC major groups, brand usage, chip x brand cross.

    Uses DuckDB for all heavy aggregation on the 24M-row transaction table.
    """
    con = _duckdb.connect()
    con.register("txn_merch", txn)
    con.register("cards_tbl", cards)

    # MCC major category counts + chip/swipe + total count per user
    # Build MCC range CASE expression for the 7 categories
    mcc_case_parts = []
    for cat_name, mcc_list in MCC_MAJOR.items():
        ranges = []
        # Group consecutive ranges
        start = mcc_list[0]
        end = mcc_list[0]
        for mcc in mcc_list[1:]:
            if mcc == end + 1:
                end = mcc
            else:
                ranges.append((start, end))
                start = mcc
                end = mcc
        ranges.append((start, end))
        conditions = " OR ".join(
            f'"MCC" BETWEEN {s} AND {e}' for s, e in ranges
        )
        mcc_case_parts.append(
            f"SUM(CASE WHEN {conditions} THEN 1 ELSE 0 END) AS \"{cat_name}_cnt\""
        )

    mcc_agg_sql = ", ".join(mcc_case_parts)
    chip_cases = ", ".join(
        f"""SUM(CASE WHEN TRIM("Use Chip") = '{cat}' THEN 1 ELSE 0 END) AS "{cat}_cnt" """
        for cat in USE_CHIP_CATEGORIES
    )

    merch_agg = con.execute(f"""
        SELECT "User",
               COUNT(*) AS total_cnt,
               {mcc_agg_sql},
               {chip_cases}
        FROM txn_merch
        GROUP BY "User"
        ORDER BY "User"
    """).df().set_index("User")

    # Card brand usage via join with cards table
    brand_agg = None
    cross_agg = None
    if "Card Brand" in cards.columns:
        brand_agg = con.execute("""
            SELECT t."User",
                   c."Card Brand" AS brand,
                   TRIM(t."Use Chip") AS chip,
                   COUNT(*) AS cnt
            FROM txn_merch t
            LEFT JOIN cards_tbl c
                ON t."User" = c."User" AND t."Card" = c."CARD INDEX"
            GROUP BY t."User", c."Card Brand", TRIM(t."Use Chip")
        """).df()

    con.unregister("txn_merch")
    con.unregister("cards_tbl")
    con.close()

    result = pd.DataFrame({"user_id": user_ids})
    col_idx = 1
    total_cnt = merch_agg["total_cnt"].reindex(user_ids, fill_value=1).clip(lower=1)

    # MCC major category ratios (7D)
    for cat_name in MCC_MAJOR.keys():
        col = f"merchant_{col_idx:03d}"
        cnt_col = f"{cat_name}_cnt"
        result[col] = (merch_agg[cnt_col].reindex(user_ids, fill_value=0) / total_cnt).values
        col_idx += 1

    # Card brand usage ratio (4D)
    if brand_agg is not None:
        brand_user = brand_agg.groupby(["User", "brand"])["cnt"].sum().unstack(fill_value=0)
        for brand in CARD_BRANDS:
            col = f"merchant_{col_idx:03d}"
            if brand in brand_user.columns:
                result[col] = (brand_user[brand].reindex(user_ids, fill_value=0) / total_cnt).values
            else:
                result[col] = 0.0
            col_idx += 1
    else:
        for _ in CARD_BRANDS:
            result[f"merchant_{col_idx:03d}"] = 0.0
            col_idx += 1

    # Chip/Swipe/Online ratio (3D)
    for chip_cat in USE_CHIP_CATEGORIES:
        col = f"merchant_{col_idx:03d}"
        cnt_col = f"{chip_cat}_cnt"
        result[col] = (merch_agg[cnt_col].reindex(user_ids, fill_value=0) / total_cnt).values
        col_idx += 1

    # Brand x payment method cross (top 7D)
    if brand_agg is not None:
        cross_pivot = brand_agg.pivot_table(
            index="User", columns=["brand", "chip"],
            values="cnt", aggfunc="sum", fill_value=0,
        )
        col_sums = cross_pivot.sum(axis=0).sort_values(ascending=False)
        top7_cross = col_sums.head(7).index.tolist()
        for cross_col in top7_cross:
            col = f"merchant_{col_idx:03d}"
            result[col] = (
                cross_pivot[cross_col].reindex(user_ids, fill_value=0) / total_cnt
            ).values
            col_idx += 1

    # Pad to 21D
    while col_idx <= 21:
        result[f"merchant_{col_idx:03d}"] = 0.0
        col_idx += 1

    merch_cols = [f"merchant_{i:03d}" for i in range(1, 22)]
    return result[["user_id"] + merch_cols]


def build_merchant_hierarchy_from_aggs(merch_data: pd.DataFrame,
                                        cards: pd.DataFrame,
                                        user_ids: np.ndarray) -> pd.DataFrame:
    """merchant_hierarchy (21D) from pre-aggregated DuckDB data.

    Parameters
    ----------
    merch_data : DataFrame(User, mcc_major, Use Chip, cnt, amt) from DuckDB.
    cards : DataFrame with Card Brand column.
    user_ids : np.array of user indices starting at 0.

    Returns
    -------
    DataFrame with columns [user_id, merchant_001 .. merchant_021].
    """
    result = pd.DataFrame({"user_id": user_ids})
    col_idx = 1

    if merch_data is None or len(merch_data) == 0:
        for i in range(1, 22):
            result[f"merchant_{i:03d}"] = 0.0
        return result

    # Total transactions per user
    user_totals = merch_data.groupby("User")["cnt"].sum().reindex(user_ids, fill_value=1).clip(lower=1)

    # MCC major category ratios (7D)
    # Map mcc_major (MCC / 1000) back to the 7 named categories
    mcc_major_map = {}
    for cat_name, mcc_list in MCC_MAJOR.items():
        for mcc in mcc_list:
            mcc_major_map[mcc // 1000] = cat_name
    # Unique category names in order
    cat_names = list(MCC_MAJOR.keys())

    merch_data_c = merch_data.copy()
    merch_data_c["cat_name"] = merch_data_c["mcc_major"].map(mcc_major_map).fillna("other")

    for cat_name in cat_names:
        col = f"merchant_{col_idx:03d}"
        cat_cnt = (
            merch_data_c[merch_data_c["cat_name"] == cat_name]
            .groupby("User")["cnt"].sum()
            .reindex(user_ids, fill_value=0)
        )
        result[col] = (cat_cnt / user_totals).values
        col_idx += 1

    # Card brand usage ratio (4D)
    if "Card Brand" in cards.columns:
        # Build user-level brand distribution from cards table
        brand_counts = cards.groupby(["User", "Card Brand"]).size().unstack(fill_value=0)
        brand_total = brand_counts.sum(axis=1).clip(lower=1)
        for brand in CARD_BRANDS:
            col = f"merchant_{col_idx:03d}"
            if brand in brand_counts.columns:
                result[col] = (brand_counts[brand] / brand_total).reindex(user_ids, fill_value=0).values
            else:
                result[col] = 0.0
            col_idx += 1
    else:
        for _ in CARD_BRANDS:
            result[f"merchant_{col_idx:03d}"] = 0.0
            col_idx += 1

    # Chip/Swipe/Online ratio (3D) — from merch_data "Use Chip" column
    chip_agg = merch_data.groupby(["User", "Use Chip"])["cnt"].sum().unstack(fill_value=0)
    for chip_cat in USE_CHIP_CATEGORIES:
        col = f"merchant_{col_idx:03d}"
        if chip_cat in chip_agg.columns:
            result[col] = (chip_agg[chip_cat].reindex(user_ids, fill_value=0) / user_totals).values
        else:
            result[col] = 0.0
        col_idx += 1

    # Cross features: MCC-major x payment method (top 7D)
    cross = merch_data_c.groupby(["User", "cat_name", "Use Chip"])["cnt"].sum().reset_index()
    cross["cross_key"] = cross["cat_name"] + "_x_" + cross["Use Chip"].astype(str)
    cross_pivot = cross.pivot_table(
        index="User", columns="cross_key", values="cnt", aggfunc="sum", fill_value=0,
    )
    # Top 7 cross columns by total count
    col_sums = cross_pivot.sum(axis=0).sort_values(ascending=False)
    top7_cross = col_sums.head(7).index.tolist()
    for cross_col in top7_cross:
        col = f"merchant_{col_idx:03d}"
        result[col] = (
            cross_pivot[cross_col].reindex(user_ids, fill_value=0) / user_totals
        ).values
        col_idx += 1

    # Pad to 21D
    while col_idx <= 21:
        result[f"merchant_{col_idx:03d}"] = 0.0
        col_idx += 1

    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    merch_cols = [f"merchant_{i:03d}" for i in range(1, 22)]
    return result[["user_id"] + merch_cols]


def build_graph_embeddings(user_ids: np.ndarray,
                           base_features: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """graph_embeddings (20D): real GraphEmbeddingGenerator (LightGCN + SVD fallback).

    Uses GraphEmbeddingGenerator which builds a kNN similarity graph from
    numeric features and learns node embeddings.  The numpy fallback uses
    truncated SVD of the adjacency matrix.  The generator produces
    embedding_dim + 2 dimensions (embeddings + norm + depth), so we set
    embedding_dim=18 to get exactly 20D output.
    """
    result = pd.DataFrame({"user_id": user_ids})
    n_users = len(user_ids)
    graph_filled = False

    if _HAS_GRAPH_GEN and base_features is not None:
        try:
            num_cols = [c for c in base_features.columns if c != "user_id"]
            if len(num_cols) >= 2:
                input_df = base_features[num_cols].copy()
                input_df["user_id"] = user_ids
                input_df.index = range(n_users)

                graph_gen = GraphEmbeddingGenerator(
                    embedding_dim=18,  # 18 + norm + depth = 20D
                    num_layers=2,
                    k_neighbors=min(10, n_users - 1),
                    num_epochs=10,  # fewer epochs for adapter context
                    entity_column="user_id",
                    feature_columns=num_cols,
                    prefix="graph",
                    prefer_gpu=False,  # SageMaker Processing may not have GPU
                )
                graph_gen.fit(input_df)
                graph_result = graph_gen.generate(input_df)

                gen_cols = list(graph_result.columns)
                gen_values = graph_result[gen_cols].values.astype(np.float32)
                n_gen_dims = gen_values.shape[1]

                for j in range(min(n_gen_dims, 20)):
                    result[f"graph_{j + 1:03d}"] = gen_values[:, j]

                graph_filled = True
                logger.info("Graph generator produced %d real features", min(n_gen_dims, 20))
        except Exception as exc:
            logger.warning("Graph generator failed, falling back to zeros: %s", exc)
            graph_filled = False

    # Fill any remaining columns up to 20D with zeros
    for i in range(1, 21):
        col = f"graph_{i:03d}"
        if col not in result.columns:
            result[col] = 0.0

    graph_cols = [f"graph_{i:03d}" for i in range(1, 21)]
    return result[["user_id"] + graph_cols]


# ===================================================================
# 3. Label Builders
# ===================================================================

def build_labels(users: pd.DataFrame,
                 txn: pd.DataFrame,
                 cards: pd.DataFrame,
                 user_ids: np.ndarray,
                 ref_date: pd.Timestamp) -> pd.DataFrame:
    """Build all 16 labels.

    Uses DuckDB for heavy aggregation on the 24M-row transaction table.
    """
    result = pd.DataFrame({"user_id": user_ids})
    three_months_ago = ref_date - pd.Timedelta(days=90)

    con = _duckdb.connect()
    con.register("txn_lbl", txn)
    con.register("cards_lbl", cards)

    # Core aggregates in a single query
    core_agg = con.execute("""
        SELECT "User",
               AVG("Is Fraud?")   AS fraud_rate,
               MAX("Date")        AS last_date,
               SUM("Amount")      AS total_spend,
               COUNT(*)           AS txn_count
        FROM txn_lbl
        GROUP BY "User"
    """).df().set_index("User")

    # Monthly counts and amounts
    monthly_df = con.execute("""
        SELECT "User", "Month",
               COUNT(*) AS cnt,
               SUM("Amount") AS amt
        FROM txn_lbl
        GROUP BY "User", "Month"
    """).df()

    # Channel mode + time slot mode
    channel_time = con.execute("""
        SELECT "User",
               MODE() WITHIN GROUP (ORDER BY TRIM("Use Chip")) AS primary_channel,
               MODE() WITHIN GROUP (ORDER BY CAST("Hour" / 3 AS INT)) AS primary_time_slot
        FROM txn_lbl
        GROUP BY "User"
    """).df().set_index("User")

    # Top MCC per user
    top_mcc_df = con.execute("""
        WITH mcc_cnt AS (
            SELECT "User", "MCC", COUNT(*) AS cnt
            FROM txn_lbl
            GROUP BY "User", "MCC"
        ),
        ranked AS (
            SELECT "User", "MCC", cnt,
                   ROW_NUMBER() OVER (PARTITION BY "User" ORDER BY cnt DESC) AS rn
            FROM mcc_cnt
        )
        SELECT "User", "MCC" AS top_mcc
        FROM ranked WHERE rn = 1
    """).df().set_index("User")

    # Median gap between transactions per user
    gap_df = con.execute("""
        WITH ordered AS (
            SELECT "User", "Date",
                   LAG("Date") OVER (PARTITION BY "User" ORDER BY "Date") AS prev_date
            FROM txn_lbl
        ),
        gaps AS (
            SELECT "User",
                   DATE_DIFF('day', prev_date, "Date") AS gap_days
            FROM ordered
            WHERE prev_date IS NOT NULL
        )
        SELECT "User",
               MEDIAN(gap_days) AS median_gap
        FROM gaps
        GROUP BY "User"
    """).df().set_index("User")

    # Merchant HHI per user
    hhi_df = con.execute("""
        WITH merch AS (
            SELECT "User", "Merchant Name",
                   SUM("Amount") AS merch_amt
            FROM txn_lbl
            GROUP BY "User", "Merchant Name"
        ),
        user_total AS (
            SELECT "User", SUM(merch_amt) AS total_amt
            FROM merch GROUP BY "User"
        ),
        shares AS (
            SELECT m."User",
                   (m.merch_amt / GREATEST(u.total_amt, 1e-8)) AS share
            FROM merch m
            JOIN user_total u ON m."User" = u."User"
        )
        SELECT "User",
               SUM(share * share) AS hhi
        FROM shares
        GROUP BY "User"
    """).df().set_index("User")

    # Brand counts via join
    brand_counts_df = None
    if "Card Brand" in cards.columns:
        brand_counts_df = con.execute("""
            SELECT t."User",
                   c."Card Brand" AS brand,
                   COUNT(*) AS cnt
            FROM txn_lbl t
            LEFT JOIN cards_lbl c
                ON t."User" = c."User" AND t."Card" = c."CARD INDEX"
            GROUP BY t."User", c."Card Brand"
        """).df()

    con.unregister("txn_lbl")
    con.unregister("cards_lbl")
    con.close()

    # Build labels from DuckDB aggregates
    median_fraud = core_agg["fraud_rate"].median()
    result["label_is_fraud"] = (
        core_agg["fraud_rate"].reindex(user_ids, fill_value=0) > median_fraud
    ).astype(int).values

    last_date = pd.to_datetime(core_agg["last_date"]).reindex(user_ids)
    result["label_will_transact"] = (last_date >= three_months_ago).astype(int).fillna(0).values
    result["label_churn"] = (1 - result["label_will_transact"]).values
    result["label_retention"] = (1 - result["label_churn"]).values

    age = users["Current Age"].reindex(user_ids).fillna(30).astype(float)
    result["label_life_stage"] = np.clip((age - 20) // 10, 0, 4).astype(int)

    # Monthly pivot for LTV / engagement / spending_amount
    monthly_cnt = monthly_df.pivot_table(
        index="User", columns="Month", values="cnt", aggfunc="sum", fill_value=0)
    monthly_amt = monthly_df.pivot_table(
        index="User", columns="Month", values="amt", aggfunc="sum", fill_value=0)
    for m in range(1, 13):
        if m not in monthly_cnt.columns:
            monthly_cnt[m] = 0
        if m not in monthly_amt.columns:
            monthly_amt[m] = 0.0
    monthly_cnt = monthly_cnt[range(1, 13)].reindex(user_ids, fill_value=0)
    monthly_amt = monthly_amt[range(1, 13)].reindex(user_ids, fill_value=0)

    result["label_ltv"] = monthly_amt.sum(axis=1).values.astype(float)

    if "Credit Limit" in cards.columns:
        user_credit_limit = cards.groupby("User")["Credit Limit"].sum().reindex(user_ids, fill_value=1).clip(lower=1)
    else:
        user_credit_limit = pd.Series(1.0, index=user_ids)
    total_spend = core_agg["total_spend"].reindex(user_ids, fill_value=0)
    result["label_balance_util"] = (total_spend / user_credit_limit).values.astype(float)

    x = np.arange(1, 13, dtype=float)
    slopes = monthly_cnt.apply(
        lambda row: np.polyfit(x, row.values.astype(float), 1)[0]
        if row.sum() > 0 else 0.0, axis=1)
    result["label_engagement"] = slopes.values.astype(float)

    # Channel
    channel_map = {cat: i for i, cat in enumerate(USE_CHIP_CATEGORIES)}
    result["label_channel"] = (
        channel_time["primary_channel"].reindex(user_ids, fill_value="Chip Transaction")
        .map(channel_map).fillna(0).astype(int).values
    )

    # Timing
    result["label_timing"] = (
        channel_time["primary_time_slot"].reindex(user_ids, fill_value=0)
        .fillna(0).astype(int).clip(0, 7).values
    )

    # NBA
    mcc_to_major = {}
    for idx, (cat_name, mcc_list) in enumerate(MCC_MAJOR.items()):
        for mcc in mcc_list:
            mcc_to_major[mcc] = idx
    top_mcc = top_mcc_df["top_mcc"].reindex(user_ids, fill_value=0)
    result["label_nba"] = top_mcc.map(mcc_to_major).fillna(7).astype(int).values % 15

    result["label_spending_category"] = top_mcc.fillna(0).astype(int).values % 15

    # Consumption cycle
    gap = gap_df["median_gap"].reindex(user_ids, fill_value=999)
    result["label_consumption_cycle"] = pd.cut(
        gap, bins=[-np.inf, 10, 20, 35, np.inf], labels=[0, 1, 2, 3],
    ).astype(int).values

    result["label_spending_amount"] = monthly_amt.mean(axis=1).values.astype(float)

    result["label_merchant_affinity"] = hhi_df["hhi"].reindex(user_ids, fill_value=0).values.astype(float)

    # Brand
    if brand_counts_df is not None and len(brand_counts_df) > 0:
        brand_pivot = brand_counts_df.pivot_table(
            index="User", columns="brand", values="cnt", aggfunc="sum", fill_value=0)
        dominant_brand = brand_pivot.idxmax(axis=1)
        brand_label_map = {b: i for i, b in enumerate(CARD_BRANDS)}
        result["label_brand"] = (
            dominant_brand.map(brand_label_map).reindex(user_ids, fill_value=0).fillna(0).astype(int).values
        )
    else:
        result["label_brand"] = 0

    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)

    # --- Regression label transforms: clip + log1p for large monetary labels ---
    label_transforms = {}
    for lbl in ["label_ltv", "label_spending_amount"]:
        if lbl in result.columns:
            clip_val = float(result[lbl].quantile(0.995))
            result[lbl] = np.log1p(result[lbl].clip(upper=clip_val))
            label_transforms[lbl] = {"clip_value": clip_val, "transform": "log1p"}
    result.attrs["label_transforms"] = label_transforms

    return result


def build_labels_from_aggs(users: pd.DataFrame,
                            duckdb_aggs: dict,
                            cards: pd.DataFrame,
                            user_ids: np.ndarray,
                            ref_date: pd.Timestamp) -> pd.DataFrame:
    """Build all 16 labels from pre-aggregated DuckDB data.

    Parameters
    ----------
    users : DataFrame with user profiles (Current Age, Yearly Income, etc.).
    duckdb_aggs : dict with keys:
        - label_fraud : DataFrame(User → fraud_rate, last_date, recent_txn_count)
        - label_channel : DataFrame(User → primary_channel, primary_time_slot)
        - label_mcc : DataFrame(User → top_mcc)
        - txn_agg : DataFrame(User → txn_count, txn_total_amount, ...)
        - ref_date : pd.Timestamp
        - econ_monthly : DataFrame(User, Year, Month, monthly_spend)
    cards : DataFrame with Card Brand, Credit Limit columns.
    user_ids : np.array of user indices starting at 0.
    ref_date : reference date for recency calculations.

    Returns
    -------
    DataFrame with columns [user_id, label_is_fraud, label_will_transact,
        label_churn, label_retention, label_life_stage, label_ltv,
        label_balance_util, label_engagement, label_channel, label_timing,
        label_nba, label_spending_category, label_consumption_cycle,
        label_spending_amount, label_merchant_affinity, label_brand].
    """
    result = pd.DataFrame({"user_id": user_ids})

    label_fraud = duckdb_aggs.get("label_fraud")
    label_channel = duckdb_aggs.get("label_channel")
    label_mcc = duckdb_aggs.get("label_mcc")
    txn_agg = duckdb_aggs.get("txn_agg")
    econ_monthly = duckdb_aggs.get("econ_monthly")

    three_months_ago = ref_date - pd.Timedelta(days=90)

    # ---- 1. label_is_fraud ----
    if label_fraud is not None and len(label_fraud) > 0:
        fraud_rate = label_fraud["fraud_rate"].reindex(user_ids, fill_value=0)
        median_fraud = fraud_rate.median()
        result["label_is_fraud"] = (fraud_rate > median_fraud).astype(int)
    else:
        result["label_is_fraud"] = 0

    # ---- 2. label_will_transact ----
    if label_fraud is not None and "last_date" in label_fraud.columns:
        last_date = pd.to_datetime(label_fraud["last_date"]).reindex(user_ids)
        result["label_will_transact"] = (last_date >= three_months_ago).astype(int).fillna(0)
    else:
        result["label_will_transact"] = 0

    # ---- 3. label_churn ----
    result["label_churn"] = 1 - result["label_will_transact"]

    # ---- 4. label_retention ----
    result["label_retention"] = 1 - result["label_churn"]

    # ---- 5. label_life_stage ----
    age = users["Current Age"].reindex(user_ids).fillna(30).astype(float)
    result["label_life_stage"] = np.clip((age - 20) // 10, 0, 4).astype(int)

    # ---- Build monthly pivot for labels that need it ----
    if econ_monthly is not None and len(econ_monthly) > 0:
        latest_year = econ_monthly.groupby("User")["Year"].max().rename("max_year")
        em = econ_monthly.merge(latest_year, on="User")
        em = em[em["Year"] == em["max_year"]]
        monthly_amt = em.pivot_table(
            index="User", columns="Month", values="monthly_spend",
            aggfunc="sum", fill_value=0,
        )
    else:
        monthly_amt = pd.DataFrame(index=user_ids)

    for m in range(1, 13):
        if m not in monthly_amt.columns:
            monthly_amt[m] = 0.0
    monthly_amt = monthly_amt[range(1, 13)].reindex(user_ids, fill_value=0)

    # ---- 6. label_ltv ----
    result["label_ltv"] = monthly_amt.sum(axis=1).values.astype(float)

    # ---- 7. label_balance_util ----
    total_spend = txn_agg["txn_total_amount"].reindex(user_ids, fill_value=0) if txn_agg is not None else 0
    if "Credit Limit" in cards.columns:
        user_credit_limit = cards.groupby("User")["Credit Limit"].sum().reindex(user_ids, fill_value=1).clip(lower=1)
    else:
        user_credit_limit = pd.Series(1.0, index=user_ids)
    result["label_balance_util"] = (total_spend / user_credit_limit).values.astype(float)

    # ---- 8. label_engagement ----
    # Monthly frequency trend slope from monthly_amt proxy (count not available, use amt as proxy)
    monthly_cnt_proxy = (monthly_amt > 0).astype(float)
    x = np.arange(1, 13, dtype=float)
    slopes = monthly_cnt_proxy.apply(
        lambda row: np.polyfit(x, row.values.astype(float), 1)[0]
        if row.sum() > 0 else 0.0,
        axis=1,
    )
    result["label_engagement"] = slopes.values.astype(float)

    # ---- 9. label_channel ----
    if label_channel is not None and "primary_channel" in label_channel.columns:
        channel_series = label_channel["primary_channel"].reindex(user_ids, fill_value="Chip Transaction")
        channel_map = {cat: i for i, cat in enumerate(USE_CHIP_CATEGORIES)}
        result["label_channel"] = channel_series.map(channel_map).fillna(0).astype(int)
    else:
        result["label_channel"] = 0

    # ---- 10. label_timing ----
    if label_channel is not None and "primary_time_slot" in label_channel.columns:
        result["label_timing"] = (
            label_channel["primary_time_slot"]
            .reindex(user_ids, fill_value=0)
            .fillna(0)
            .astype(int)
            .clip(0, 7)
        )
    else:
        result["label_timing"] = 0

    # ---- 11. label_nba ----
    if label_mcc is not None and "top_mcc" in label_mcc.columns:
        mcc_to_major = {}
        for idx, (cat_name, mcc_list) in enumerate(MCC_MAJOR.items()):
            for mcc in mcc_list:
                mcc_to_major[mcc] = idx
        top_mcc = label_mcc["top_mcc"].reindex(user_ids, fill_value=0)
        result["label_nba"] = top_mcc.map(mcc_to_major).fillna(7).astype(int) % 15
    else:
        result["label_nba"] = 0

    # ---- 12. label_spending_category ----
    if label_mcc is not None and "top_mcc" in label_mcc.columns:
        result["label_spending_category"] = (
            label_mcc["top_mcc"].reindex(user_ids, fill_value=0).astype(int) % 15
        )
    else:
        result["label_spending_category"] = 0

    # ---- 13. label_consumption_cycle ----
    # Approximate from txn_agg: median_gap ~ total_days / txn_count
    if txn_agg is not None and "txn_count" in txn_agg.columns and label_fraud is not None:
        last_date_ts = pd.to_datetime(label_fraud["last_date"]).reindex(user_ids)
        first_date_approx = ref_date - pd.Timedelta(days=365)  # rough estimate
        total_days = (last_date_ts - first_date_approx).dt.days.fillna(365).clip(lower=1)
        txn_count = txn_agg["txn_count"].reindex(user_ids, fill_value=1).clip(lower=1)
        avg_gap = total_days / txn_count
        result["label_consumption_cycle"] = pd.cut(
            avg_gap,
            bins=[-np.inf, 10, 20, 35, np.inf],
            labels=[0, 1, 2, 3],
        ).astype(int)
    else:
        result["label_consumption_cycle"] = 3  # irregular

    # ---- 14. label_spending_amount ----
    result["label_spending_amount"] = monthly_amt.mean(axis=1).values.astype(float)

    # ---- 15. label_merchant_affinity ----
    # Approximate HHI from txn_agg n_merchants: HHI ~ 1/n_merchants
    if txn_agg is not None and "n_merchants" in txn_agg.columns:
        n_merch = txn_agg["n_merchants"].reindex(user_ids, fill_value=1).clip(lower=1)
        result["label_merchant_affinity"] = (1.0 / n_merch).values.astype(float)
    else:
        result["label_merchant_affinity"] = 0.0

    # ---- 16. label_brand ----
    if "Card Brand" in cards.columns:
        brand_counts = cards.groupby(["User", "Card Brand"]).size().unstack(fill_value=0)
        dominant_brand = brand_counts.idxmax(axis=1)
        brand_label_map = {b: i for i, b in enumerate(CARD_BRANDS)}
        result["label_brand"] = (
            dominant_brand.map(brand_label_map)
            .reindex(user_ids, fill_value=0)
            .fillna(0)
            .astype(int)
        )
    else:
        result["label_brand"] = 0

    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)

    # --- Regression label transforms: clip + log1p for large monetary labels ---
    label_transforms = {}
    for lbl in ["label_ltv", "label_spending_amount"]:
        if lbl in result.columns:
            clip_val = float(result[lbl].quantile(0.995))
            result[lbl] = np.log1p(result[lbl].clip(upper=clip_val))
            label_transforms[lbl] = {"clip_value": clip_val, "transform": "log1p"}
    # Attach transforms dict as attribute so caller can persist it
    result.attrs["label_transforms"] = label_transforms

    return result


# ===================================================================
# 4. Transaction Pre-aggregation (for base_rfm)
# ===================================================================

def pre_aggregate_transactions(txn: pd.DataFrame) -> pd.DataFrame:
    """Compute per-user transaction summary for base_rfm.

    Uses DuckDB for columnar aggregation on the (potentially 24M-row)
    transaction DataFrame.
    """
    con = _duckdb.connect()
    con.register("txn_view", txn)
    agg = con.execute("""
        SELECT
            "User",
            COUNT(*)                          AS txn_count,
            SUM("Amount")                     AS txn_total_amount,
            AVG("Amount")                     AS txn_mean_amount,
            COALESCE(STDDEV("Amount"), 0)     AS txn_std_amount,
            MAX("Amount")                     AS txn_max_amount,
            MIN("Amount")                     AS txn_min_amount,
            MAX("Date")                       AS last_txn_date,
            COUNT(DISTINCT "Merchant Name")   AS n_merchants,
            COUNT(DISTINCT "Merchant State")  AS n_states,
            SUM(CASE WHEN "Errors?" IS NOT NULL AND TRIM("Errors?") != ''
                     THEN 1 ELSE 0 END)::FLOAT
                / COUNT(*)::FLOAT             AS error_rate
        FROM txn_view
        GROUP BY "User"
        ORDER BY "User"
    """).df().set_index("User")
    agg["last_txn_date"] = pd.to_datetime(agg["last_txn_date"])
    con.unregister("txn_view")
    con.close()
    return agg


# ===================================================================
# 5. Statistics Writers
# ===================================================================

def compute_feature_stats(df: pd.DataFrame,
                          feature_cols: List[str]) -> Dict:
    """Compute {mean, std, min, max, null_pct} for each feature."""
    stats = {}
    for col in feature_cols:
        s = df[col]
        stats[col] = {
            "mean": float(s.mean()) if not s.isna().all() else None,
            "std": float(s.std()) if not s.isna().all() else None,
            "min": float(s.min()) if not s.isna().all() else None,
            "max": float(s.max()) if not s.isna().all() else None,
            "null_pct": float(s.isna().mean()),
        }
    return stats


def compute_label_stats(df: pd.DataFrame,
                        label_cols: List[str]) -> Dict:
    """Compute distribution stats for each label."""
    stats = {}
    for col in label_cols:
        s = df[col]
        entry: Dict = {
            "dtype": str(s.dtype),
            "null_pct": float(s.isna().mean()),
            "mean": float(s.mean()) if not s.isna().all() else None,
            "std": float(s.std()) if not s.isna().all() else None,
        }
        if s.dtype in ("int64", "int32", "int", "int8"):
            vc = s.value_counts(normalize=True).to_dict()
            entry["class_distribution"] = {str(k): float(v) for k, v in vc.items()}
            entry["num_classes"] = int(s.nunique())
        stats[col] = entry
    return stats


# ===================================================================
# 5b. Event Sequence Builder (3D input for Mamba / Temporal Ensemble)
# ===================================================================

def build_event_sequences(
    con,  # duckdb.DuckDBPyConnection with txn view already created
    parquet_path: str,
    user_ids: np.ndarray,
    output_dir: str,
    seq_len: int = SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-user transaction sequences for 3D model input.

    Uses DuckDB window functions to extract the most recent *seq_len*
    transactions per user, then encodes each transaction into a fixed-
    size feature vector (SEQ_FEAT_DIM dimensions).

    Returns
    -------
    sequences : np.ndarray, shape ``(n_users, seq_len, SEQ_FEAT_DIM)``
    seq_lengths : np.ndarray, shape ``(n_users,)``
    """
    import duckdb as _dk  # guaranteed available — caller is inside DuckDB try block

    n_users = len(user_ids)
    logger.info(
        "Building event sequences: %d users, seq_len=%d ...", n_users, seq_len
    )

    # ------------------------------------------------------------------
    # Step 1: Identify top-60 MCC codes (by global transaction count)
    # ------------------------------------------------------------------
    top_mcc_df = con.execute(f"""
        SELECT "MCC", COUNT(*) AS cnt
        FROM txn
        GROUP BY "MCC"
        ORDER BY cnt DESC
        LIMIT {SEQ_TOP_MCC}
    """).df()
    top_mcc_list = top_mcc_df["MCC"].tolist()
    # MCC -> 1-based index; 0 = "other"
    mcc_to_idx = {int(m): i + 1 for i, m in enumerate(top_mcc_list)}

    # ------------------------------------------------------------------
    # Step 2: Global Amount stats for normalization
    # ------------------------------------------------------------------
    amt_stats = con.execute("""
        SELECT AVG("Amount") AS mu, STDDEV("Amount") AS sigma FROM txn
    """).fetchone()
    amt_mu = float(amt_stats[0]) if amt_stats[0] is not None else 0.0
    amt_sigma = float(amt_stats[1]) if amt_stats[1] is not None else 1.0
    amt_sigma = max(amt_sigma, 1e-8)

    # ------------------------------------------------------------------
    # Step 3: Pull per-user recent transactions (windowed, sorted)
    #   DuckDB does the heavy lifting: ROW_NUMBER over 24M rows.
    # ------------------------------------------------------------------
    raw = con.execute(f"""
        WITH ranked AS (
            SELECT
                "User",
                "Amount",
                "Date",
                EXTRACT(HOUR FROM TRY_CAST("Time" AS TIME)) AS hour,
                EXTRACT(DOW  FROM "Date")                      AS dow,
                "Month",
                "Use Chip",
                "MCC",
                "Errors?",
                ROW_NUMBER() OVER (
                    PARTITION BY "User"
                    ORDER BY "Date" DESC, "Time" DESC
                ) AS rn
            FROM txn
        )
        SELECT
            "User",
            "Amount",
            "Date",
            hour,
            dow,
            "Month",
            "Use Chip",
            "MCC",
            "Errors?",
            rn
        FROM ranked
        WHERE rn <= {seq_len}
        ORDER BY "User", rn
    """).df()

    logger.info("Windowed query returned %d rows for sequence building", len(raw))

    # ------------------------------------------------------------------
    # Step 4: Encode each transaction row into a feature vector
    # ------------------------------------------------------------------
    # Pre-compute columns as numpy for speed
    users_col   = raw["User"].values.astype(np.int64)
    amount_col  = raw["Amount"].values.astype(np.float64)
    date_col    = pd.to_datetime(raw["Date"])
    hour_col    = raw["hour"].values.astype(np.float64)
    dow_col     = raw["dow"].values.astype(np.float64)
    month_col   = raw["Month"].values.astype(np.float64)
    rn_col      = raw["rn"].values.astype(np.int64)
    mcc_col     = raw["MCC"].values
    chip_col    = raw["Use Chip"].values
    err_col     = raw["Errors?"].values

    # Use Chip encoding: Chip=0, Swipe=1, Online=2
    chip_map = {
        "Chip Transaction": 0,
        "Swipe Transaction": 1,
        "Online Transaction": 2,
    }
    chip_encoded = np.array(
        [chip_map.get(str(v).strip(), 1) for v in chip_col], dtype=np.float64
    )

    # Error flag
    error_flag = np.array(
        [0.0 if (pd.isna(v) or str(v).strip() == "") else 1.0 for v in err_col],
        dtype=np.float64,
    )

    # MCC index
    mcc_idx = np.array(
        [mcc_to_idx.get(int(m), 0) if pd.notna(m) else 0 for m in mcc_col],
        dtype=np.int64,
    )

    # Normalized amount
    amount_norm = (amount_col - amt_mu) / amt_sigma

    # Is weekend (Sat=5, Sun=6 in ISO; DuckDB DOW: Sun=0, Sat=6)
    is_weekend = np.where((dow_col == 0) | (dow_col == 6), 1.0, 0.0)

    # Date as days-since-epoch for delta computation
    epoch = np.datetime64("1970-01-01")
    date_days = (date_col.values.astype("datetime64[D]") - epoch).astype(np.float64)

    # ------------------------------------------------------------------
    # Step 5: Fill the 3D tensor  (n_users, seq_len, feat_dim)
    # ------------------------------------------------------------------
    feat_dim = SEQ_FEAT_DIM  # 16
    sequences = np.zeros((n_users, seq_len, feat_dim), dtype=np.float32)
    seq_lengths = np.zeros(n_users, dtype=np.int32)

    # Build a mapping from user_id -> row indices for fast slicing
    # (raw is already sorted by User, rn)
    unique_users, user_start_idx, user_counts = np.unique(
        users_col, return_index=True, return_counts=True
    )

    for uid, start, cnt in zip(unique_users, user_start_idx, user_counts):
        if uid < 0 or uid >= n_users:
            continue
        end = start + cnt
        sl = min(cnt, seq_len)
        seq_lengths[uid] = sl

        idx = slice(start, start + sl)

        # Feature vector per timestep (16 dims):
        #  0: amount_norm
        #  1: hour / 24
        #  2: dow / 7
        #  3: month / 12
        #  4: is_weekend
        #  5: use_chip (0/1/2) / 2
        #  6: mcc_index / SEQ_TOP_MCC  (normalized embedding index)
        #  7: error_flag
        #  8: time_delta (days since prev txn, log-scaled)
        #  9: amount_delta (normalized diff from prev txn)
        # 10-15: reserved / zero-padded (future expansion)

        sequences[uid, :sl, 0] = amount_norm[idx]
        sequences[uid, :sl, 1] = hour_col[idx] / 24.0
        sequences[uid, :sl, 2] = dow_col[idx] / 7.0
        sequences[uid, :sl, 3] = month_col[idx] / 12.0
        sequences[uid, :sl, 4] = is_weekend[idx]
        sequences[uid, :sl, 5] = chip_encoded[idx] / 2.0
        sequences[uid, :sl, 6] = mcc_idx[idx].astype(np.float32) / max(SEQ_TOP_MCC, 1)
        sequences[uid, :sl, 7] = error_flag[idx]

        # Time delta: days between consecutive transactions (log1p scaled)
        days = date_days[idx]
        if sl > 1:
            # rn=1 is most recent; deltas are negative (older), take abs
            time_delta = np.abs(np.diff(days, prepend=days[0]))
            sequences[uid, :sl, 8] = np.log1p(time_delta).astype(np.float32)

        # Amount delta: normalized difference from previous transaction
        amts = amount_norm[idx]
        if sl > 1:
            amt_delta = np.diff(amts, prepend=amts[0])
            sequences[uid, :sl, 9] = amt_delta.astype(np.float32)

    # Replace any NaN/inf that crept in
    sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # Step 6: Save to disk
    # ------------------------------------------------------------------
    seq_path = os.path.join(output_dir, "ealtman2019_event_sequences.npy")
    len_path = os.path.join(output_dir, "ealtman2019_seq_lengths.npy")
    np.save(seq_path, sequences)
    np.save(len_path, seq_lengths)
    logger.info(
        "Event sequences saved: %s  shape=%s  (%.1f MB)",
        seq_path, sequences.shape,
        sequences.nbytes / 1e6,
    )
    logger.info("Sequence lengths saved: %s", len_path)

    return sequences, seq_lengths


# ===================================================================
# 6. Main Pipeline
# ===================================================================

def run(input_dir: str, output_dir: str) -> None:
    """Execute the full adapter pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Load data ---
    users = load_users(input_dir)
    cards = load_cards(input_dir)

    # Initialize _duckdb_aggs so it is always defined, even if DuckDB is unavailable.
    _duckdb_aggs: dict = {}

    # Use DuckDB for memory-efficient aggregation of 24M rows
    # Key insight: never load full 24M rows into pandas.
    # DuckDB aggregates in SQL, only 2,000-row results come to pandas.
    parquet_path = os.path.join(input_dir, "transactions.parquet")
    user_ids = np.arange(len(users))

    logger.info("Using DuckDB for in-database aggregation (no full load)")
    con = _duckdb.connect()
    con.execute("SET memory_limit='12GB'")
    con.execute("SET threads TO 4")
    con.execute("SET temp_directory='/tmp/duckdb_tmp'")
    os.makedirs("/tmp/duckdb_tmp", exist_ok=True)

    # Create a view for the parquet file
    con.execute(f"""
        CREATE VIEW txn AS
        SELECT
            "user_id" AS "User",
            "card_id" AS "Card",
            "year" AS "Year",
            "month" AS "Month",
            "day" AS "Day",
            "time" AS "Time",
            "amount" AS "Amount",
            "use_chip" AS "Use Chip",
            "merchant_id" AS "Merchant Name",
            "merchant_city" AS "Merchant City",
            "merchant_state" AS "Merchant State",
            "zip" AS "Zip",
            "mcc" AS "MCC",
            "errors" AS "Errors?",
            "is_fraud" AS "Is Fraud?",
            MAKE_DATE("year"::INT, "month"::INT, "day"::INT) AS "Date",
            EXTRACT(DOW FROM MAKE_DATE("year"::INT, "month"::INT, "day"::INT)) AS "DayOfWeek",
            EXTRACT(HOUR FROM TRY_CAST("time" AS TIME)) AS "Hour",
            "year" * 100 + "month" AS "YearMonth"
        FROM read_parquet('{parquet_path}')
    """)

    ref_date_row = con.execute("SELECT MAX(\"Date\") FROM txn").fetchone()
    ref_date = pd.Timestamp(ref_date_row[0])
    logger.info("Reference date: %s, N users: %d", ref_date, len(user_ids))

    # Pre-aggregation via DuckDB (result: 2,000 rows)
    logger.info("Pre-aggregating transactions via DuckDB ...")
    txn_agg = con.execute("""
        SELECT
            "User",
            COUNT(*) AS txn_count,
            SUM("Amount") AS txn_total_amount,
            AVG("Amount") AS txn_mean_amount,
            STDDEV("Amount") AS txn_std_amount,
            MAX("Amount") AS txn_max_amount,
            MIN("Amount") AS txn_min_amount,
            MAX("Date") AS last_txn_date,
            COUNT(DISTINCT "Merchant Name") AS n_merchants,
            COUNT(DISTINCT "Merchant State") AS n_states,
            SUM(CASE WHEN "Errors?" IS NOT NULL AND TRIM("Errors?") != '' THEN 1 ELSE 0 END)::FLOAT
                / COUNT(*)::FLOAT AS error_rate
        FROM txn
        GROUP BY "User"
        ORDER BY "User"
    """).df().set_index("User")
    txn_agg["txn_std_amount"] = txn_agg["txn_std_amount"].fillna(0)
    txn_agg["last_txn_date"] = pd.to_datetime(txn_agg["last_txn_date"])

    # base_rfm
    logger.info("Building base_rfm (34D) ...")
    fg_rfm = build_base_rfm(users, txn_agg, ref_date)

    # base_category — per-user MCC distribution (all computed in DuckDB)
    logger.info("Building base_category (64D) via DuckDB ...")
    mcc_pivot = con.execute("""
        SELECT "User", "MCC", SUM("Amount") AS mcc_amount
        FROM txn
        GROUP BY "User", "MCC"
    """).df()
    user_total = mcc_pivot.groupby("User")["mcc_amount"].sum()
    # Top 60 MCCs by total volume
    top_mccs = mcc_pivot.groupby("MCC")["mcc_amount"].sum().nlargest(60).index.tolist()
    fg_cat = pd.DataFrame(index=user_ids)
    # Vectorized: pivot then divide — avoids per-MCC filtering loop
    mcc_wide = mcc_pivot.pivot_table(
        index="User", columns="MCC", values="mcc_amount",
        aggfunc="sum", fill_value=0,
    )
    totals = user_total.reindex(user_ids, fill_value=1.0).clip(lower=1e-8)
    for i, mcc in enumerate(top_mccs):
        if mcc in mcc_wide.columns:
            fg_cat[f"cat_{i:03d}"] = (mcc_wide[mcc].reindex(user_ids, fill_value=0) / totals).values
        else:
            fg_cat[f"cat_{i:03d}"] = 0.0
    # Diversity metrics computed in DuckDB (no per-user Python loop)
    diversity = con.execute("""
        WITH user_mcc AS (
            SELECT "User", "MCC", SUM("Amount") AS mcc_amt
            FROM txn GROUP BY "User", "MCC"
        ),
        user_totals AS (
            SELECT "User",
                   SUM(mcc_amt) AS total_amt,
                   COUNT(DISTINCT "MCC") AS n_mcc
            FROM user_mcc GROUP BY "User"
        ),
        user_probs AS (
            SELECT um."User", um."MCC",
                   um.mcc_amt / GREATEST(ut.total_amt, 1e-8) AS prob
            FROM user_mcc um
            JOIN user_totals ut ON um."User" = ut."User"
        )
        SELECT
            ut."User",
            ut.n_mcc,
            COALESCE((SELECT -SUM(p.prob * LN(GREATEST(p.prob, 1e-15)))
                      FROM user_probs p WHERE p."User" = ut."User"), 0) AS entropy,
            COALESCE((SELECT SUM(p.prob * p.prob)
                      FROM user_probs p WHERE p."User" = ut."User"), 0) AS hhi,
            COALESCE((SELECT SUM(t3.prob) FROM (
                SELECT p.prob FROM user_probs p
                WHERE p."User" = ut."User"
                ORDER BY p.prob DESC LIMIT 3
            ) t3), 0) AS top3_share
        FROM user_totals ut
        ORDER BY ut."User"
    """).df().set_index("User")
    fg_cat["cat_n_mcc"] = diversity["n_mcc"].reindex(user_ids, fill_value=0).values
    fg_cat["cat_entropy"] = diversity["entropy"].reindex(user_ids, fill_value=0).values
    fg_cat["cat_hhi"] = diversity["hhi"].reindex(user_ids, fill_value=0).values
    fg_cat["cat_top3_share"] = diversity["top3_share"].reindex(user_ids, fill_value=0).values
    del mcc_pivot, mcc_wide

    # base_txn_stats — DuckDB aggregation
    logger.info("Building base_txn_stats (80D) via DuckDB ...")
    monthly_stats = con.execute("""
        SELECT "User", "Year", "Month",
               COUNT(*) AS monthly_count,
               SUM("Amount") AS monthly_amount
        FROM txn
        GROUP BY "User", "Year", "Month"
    """).df()
    # Quarterly stats
    qtr_stats = con.execute("""
        SELECT "User",
               CEIL("Month" / 3.0)::INT AS qtr,
               SUM("Amount") AS qtr_amount,
               COUNT(*) AS qtr_count
        FROM txn
        WHERE "Year" = (SELECT MAX("Year") FROM txn)
        GROUP BY "User", CEIL("Month" / 3.0)::INT
    """).df()
    # Time-of-day distribution
    tod_stats = con.execute("""
        SELECT "User",
               SUM(CASE WHEN "DayOfWeek" < 5 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS weekday_ratio,
               SUM(CASE WHEN "Hour" BETWEEN 0 AND 3 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS hour_0_4,
               SUM(CASE WHEN "Hour" BETWEEN 4 AND 7 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS hour_4_8,
               SUM(CASE WHEN "Hour" BETWEEN 8 AND 11 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS hour_8_12,
               SUM(CASE WHEN "Hour" BETWEEN 12 AND 15 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS hour_12_16,
               SUM(CASE WHEN "Hour" BETWEEN 16 AND 19 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS hour_16_20,
               SUM(CASE WHEN "Hour" BETWEEN 20 AND 23 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS hour_20_24,
               SUM(CASE WHEN "Use Chip" = 'Chip Transaction' THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS chip_ratio,
               SUM(CASE WHEN "Use Chip" = 'Swipe Transaction' THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS swipe_ratio,
               SUM(CASE WHEN "Use Chip" = 'Online Transaction' THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS online_ratio,
               SUM(CASE WHEN "Errors?" IS NOT NULL AND TRIM("Errors?") != '' THEN 1 ELSE 0 END) AS error_count,
               SUM("Is Fraud?") AS fraud_count
        FROM txn
        GROUP BY "User"
        ORDER BY "User"
    """).df().set_index("User")
    # Build 80D features from these aggregates
    fg_txn = pd.DataFrame(index=user_ids)
    # Monthly counts/amounts for last 12 months
    max_ym = monthly_stats[["Year", "Month"]].apply(lambda r: r["Year"]*100+r["Month"], axis=1).max()
    for i in range(12):
        ym = max_ym - i
        y, m = ym // 100, ym % 100
        if m <= 0: m += 12; y -= 1
        month_data = monthly_stats[(monthly_stats["Year"]==y) & (monthly_stats["Month"]==m)].set_index("User")
        fg_txn[f"txn_monthly_cnt_{i:02d}"] = month_data["monthly_count"].reindex(user_ids).fillna(0).values
        fg_txn[f"txn_monthly_amt_{i:02d}"] = month_data["monthly_amount"].reindex(user_ids).fillna(0).values
    # Quarterly change
    for q in range(1, 5):
        qd = qtr_stats[qtr_stats["qtr"]==q].set_index("User")
        fg_txn[f"txn_qtr{q}_amount"] = qd["qtr_amount"].reindex(user_ids).fillna(0).values
        fg_txn[f"txn_qtr{q}_count"] = qd["qtr_count"].reindex(user_ids).fillna(0).values
    # Time-of-day and payment method
    for col in tod_stats.columns:
        fg_txn[f"txn_{col}"] = tod_stats[col].reindex(user_ids).fillna(0).values
    # Pad to 80D if needed
    while len(fg_txn.columns) < 80:
        fg_txn[f"txn_pad_{len(fg_txn.columns):03d}"] = 0.0
    fg_txn = fg_txn.iloc[:, :80]  # truncate if over 80
    del monthly_stats, qtr_stats, tod_stats

    # base_temporal — monthly rolling aggregates (DuckDB)
    # Use DuckDB window functions instead of per-user Python loop
    logger.info("Building base_temporal (60D) via DuckDB ...")
    temporal_agg = con.execute("""
        WITH monthly_ts AS (
            SELECT "User", "Year" * 100 + "Month" AS ym,
                   COUNT(*) AS cnt, SUM("Amount") AS amt,
                   ROW_NUMBER() OVER (PARTITION BY "User" ORDER BY "Year" * 100 + "Month" DESC) AS rn_desc
            FROM txn
            GROUP BY "User", "Year" * 100 + "Month"
        ),
        user_stats AS (
            SELECT "User",
                   COUNT(*) AS n_months,
                   MAX(ym) - MIN(ym) AS span_months,
                   -- Rolling means: average of last W months
                   AVG(CASE WHEN rn_desc <= 3  THEN amt END) AS amt_roll_3m,
                   AVG(CASE WHEN rn_desc <= 6  THEN amt END) AS amt_roll_6m,
                   AVG(CASE WHEN rn_desc <= 12 THEN amt END) AS amt_roll_12m,
                   AVG(CASE WHEN rn_desc <= 3  THEN cnt END) AS cnt_roll_3m,
                   AVG(CASE WHEN rn_desc <= 6  THEN cnt END) AS cnt_roll_6m,
                   AVG(CASE WHEN rn_desc <= 12 THEN cnt END) AS cnt_roll_12m,
                   -- Trend: use REGR_SLOPE on ym vs amt
                   REGR_SLOPE(amt, ym) AS trend_slope
            FROM monthly_ts
            GROUP BY "User"
        )
        SELECT * FROM user_stats ORDER BY "User"
    """).df().set_index("User")
    fg_temporal = pd.DataFrame(index=user_ids)
    for col_name in ["amt_roll_3m", "cnt_roll_3m", "amt_roll_6m", "cnt_roll_6m",
                     "amt_roll_12m", "cnt_roll_12m", "trend_slope",
                     "n_months", "span_months"]:
        fg_temporal[f"temp_{col_name}"] = temporal_agg[col_name].reindex(user_ids, fill_value=0).values
    fg_temporal = fg_temporal.fillna(0)
    # Pad to 60D
    while len(fg_temporal.columns) < 60:
        fg_temporal[f"temp_pad_{len(fg_temporal.columns):03d}"] = 0.0
    fg_temporal = fg_temporal.iloc[:, :60]

    # Build remaining aggregates needed by downstream functions
    # economics needs per-user monthly spend series
    logger.info("Building additional aggregates via DuckDB ...")
    econ_monthly = con.execute("""
        SELECT "User", "Year", "Month", SUM("Amount") AS monthly_spend
        FROM txn GROUP BY "User", "Year", "Month"
    """).df()

    # multidisciplinary needs monthly amounts + new merchant counts
    multi_monthly = con.execute("""
        SELECT "User", "Year" * 100 + "Month" AS ym,
               SUM("Amount") AS amt,
               COUNT(DISTINCT "Merchant Name") AS new_merchants,
               SUM(CASE WHEN ABS("Amount") > (
                   SELECT AVG("Amount") + 3 * STDDEV("Amount") FROM txn
               ) THEN 1 ELSE 0 END) AS anomaly_count
        FROM txn GROUP BY "User", "Year" * 100 + "Month"
    """).df()

    # merchant_hierarchy needs per-user MCC major category + brand + payment
    merch_data = con.execute("""
        SELECT "User",
               CAST("MCC" / 1000 AS INT) AS mcc_major,
               "Use Chip",
               COUNT(*) AS cnt,
               SUM("Amount") AS amt
        FROM txn
        GROUP BY "User", CAST("MCC" / 1000 AS INT), "Use Chip"
    """).df()

    # labels need fraud stats + last txn dates + monthly series
    label_fraud = con.execute("""
        SELECT "User",
               SUM("Is Fraud?")::FLOAT / COUNT(*)::FLOAT AS fraud_rate,
               MAX("Date") AS last_date,
               SUM(CASE WHEN "Date" >= (SELECT MAX("Date") - INTERVAL '90 days' FROM txn) THEN 1 ELSE 0 END) AS recent_txn_count
        FROM txn GROUP BY "User"
    """).df().set_index("User")

    label_channel = con.execute("""
        SELECT "User",
               MODE() WITHIN GROUP (ORDER BY "Use Chip") AS primary_channel,
               MODE() WITHIN GROUP (ORDER BY "Hour" / 3) AS primary_time_slot
        FROM txn GROUP BY "User"
    """).df().set_index("User")

    label_mcc = con.execute("""
        SELECT "User",
               MODE() WITHIN GROUP (ORDER BY "MCC") AS top_mcc
        FROM txn GROUP BY "User"
    """).df().set_index("User")

    # --- Event sequences (3D tensor for Mamba / Temporal Ensemble) ---
    logger.info("Building event sequences (%dD, seq_len=%d) via DuckDB ...",
                 SEQ_FEAT_DIM, SEQ_LEN)
    _event_seqs, _seq_lens = build_event_sequences(
        con, parquet_path, user_ids, output_dir,
        seq_len=SEQ_LEN,
    )
    logger.info("Event sequences built: %s, non-empty users: %d / %d",
                 _event_seqs.shape,
                 int((_seq_lens > 0).sum()), len(user_ids))

    con.close()
    logger.info("DuckDB aggregation complete. Building remaining feature groups...")

    # Add user_id column to DuckDB-built DataFrames (they use index only)
    for df_name in [fg_cat, fg_txn, fg_temporal]:
        if "user_id" not in df_name.columns:
            df_name.insert(0, "user_id", df_name.index)
            df_name = df_name.reset_index(drop=True)
    fg_cat["user_id"] = user_ids
    fg_txn["user_id"] = user_ids
    fg_temporal["user_id"] = user_ids

    # Store aggregates in a dict for downstream functions
    _duckdb_aggs = {
        "econ_monthly": econ_monthly,
        "multi_monthly": multi_monthly,
        "merch_data": merch_data,
        "label_fraud": label_fraud,
        "label_channel": label_channel,
        "label_mcc": label_mcc,
        "txn_agg": txn_agg,
        "ref_date": ref_date,
    }
    # Set txn to None — downstream functions must use aggregates
    txn = None


    # Build base feature matrix for generator inputs (rfm + txn_stats combined)
    logger.info("Preparing base features for generators ...")
    clustering_base = fg_rfm.merge(fg_txn, on="user_id")
    _gen_base_features = clustering_base.copy()

    # Feature groups using real generators (with fallback to zeros on failure)
    logger.info("Building tda_topology (70D) ...")
    fg_tda = build_tda_topology(user_ids, None, base_features=_gen_base_features)

    logger.info("Building hmm_states (48D) ...")
    fg_hmm = build_hmm_states(user_ids, base_features=_gen_base_features)

    logger.info("Building mamba_temporal (50D) ...")
    fg_mamba = build_mamba_temporal(user_ids, base_features=_gen_base_features)

    logger.info("Building gmm_clustering (22D) ...")
    fg_gmm = build_gmm_clustering(clustering_base, user_ids)

    logger.info("Building economics (17D) ...")
    fg_econ = build_economics_from_aggs(users, _duckdb_aggs.get("econ_monthly"), user_ids)

    logger.info("Building multidisciplinary (24D) ...")
    fg_multi = build_multidisciplinary_from_aggs(_duckdb_aggs.get("multi_monthly"), user_ids)

    logger.info("Building model_derived (27D) ...")
    fg_model = build_model_derived(user_ids, base_features=_gen_base_features)

    logger.info("Building merchant_hierarchy (21D) ...")
    fg_merch = build_merchant_hierarchy_from_aggs(_duckdb_aggs.get("merch_data"), cards, user_ids)

    logger.info("Building graph_embeddings (20D) ...")
    fg_graph = build_graph_embeddings(user_ids, base_features=_gen_base_features)

    # --- Merge all feature groups ---
    logger.info("Merging feature groups ...")
    feature_dfs = [
        fg_rfm, fg_cat, fg_txn, fg_temporal, fg_tda, fg_hmm,
        fg_mamba, fg_gmm, fg_econ, fg_multi, fg_model, fg_merch, fg_graph,
    ]
    merged = feature_dfs[0]
    for fdf in feature_dfs[1:]:
        merged = merged.merge(fdf, on="user_id", how="left")

    feature_cols = [c for c in merged.columns if c != "user_id"]
    logger.info("Total feature dimensions: %d", len(feature_cols))

    # --- Labels ---
    logger.info("Building labels (16) ...")
    labels = build_labels_from_aggs(users, _duckdb_aggs, cards, user_ids, ref_date)
    label_cols = [c for c in labels.columns if c.startswith("label_")]

    # --- Final merge ---
    final = merged.merge(labels, on="user_id", how="left")

    # Replace inf/nan
    final = final.replace([np.inf, -np.inf], np.nan).fillna(0)

    # --- Feature normalization (StandardScaler) ---
    logger.info("Applying feature normalization (StandardScaler) ...")
    output_dir_path = Path(output_dir)
    try:
        from core.feature.transformers import StandardScaler as _CoreStandardScaler
        scaler = _CoreStandardScaler(columns=feature_cols)
        scaler.fit(final)
        final = scaler.transform(final)
        scaler_params = scaler.get_params()
        logger.info("Feature normalization applied via core.feature.transformers.StandardScaler")
    except Exception as _norm_exc:
        logger.warning("core StandardScaler unavailable (%s), using inline z-score", _norm_exc)
        _means = final[feature_cols].mean()
        _stds = final[feature_cols].std().replace(0, 1.0)
        final[feature_cols] = (final[feature_cols] - _means) / _stds
        scaler_params = {
            "name": "standard_scaler",
            "mean": _means.to_dict(),
            "std": _stds.to_dict(),
        }
    scaler_path = output_dir_path / "scaler_params.json"
    with open(scaler_path, "w") as f:
        json.dump(scaler_params, f, indent=2, default=str)
    logger.info("Scaler params written to %s", scaler_path)

    # --- Save label transforms ---
    label_transforms = getattr(labels, "attrs", {}).get("label_transforms", {})
    if label_transforms:
        lt_path = output_dir_path / "label_transforms.json"
        with open(lt_path, "w") as f:
            json.dump(label_transforms, f, indent=2, default=str)
        logger.info("Label transforms written to %s (%d labels transformed)",
                     lt_path, len(label_transforms))

    # --- Write outputs ---
    out_parquet = os.path.join(output_dir, "ealtman2019_features.parquet")
    logger.info("Writing parquet to %s  (%d rows, %d cols)",
                out_parquet, len(final), len(final.columns))
    final.to_parquet(out_parquet, index=False, engine="pyarrow")

    # Feature stats
    f_stats = compute_feature_stats(final, feature_cols)
    f_stats_path = os.path.join(output_dir, "feature_stats.json")
    with open(f_stats_path, "w") as f:
        json.dump(f_stats, f, indent=2, ensure_ascii=False)
    logger.info("Feature stats written to %s", f_stats_path)

    # Label stats
    l_stats = compute_label_stats(final, label_cols)
    l_stats_path = os.path.join(output_dir, "label_stats.json")
    with open(l_stats_path, "w") as f:
        json.dump(l_stats, f, indent=2, ensure_ascii=False)
    logger.info("Label stats written to %s", l_stats_path)

    logger.info("Adapter complete. Output: %d users x %d features + %d labels",
                len(final), len(feature_cols), len(label_cols))


# ===================================================================
# Entry point
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ealtman2019 dataset → pipeline feature parquet adapter"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/opt/ml/processing/input",
        help="Directory containing raw parquet/csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/ml/processing/output",
        help="Directory to write output parquet and stats JSON",
    )
    return parser.parse_args()


# ===================================================================
# DataAdapter subclass (PipelineRunner integration)
# ===================================================================

if _HAS_ADAPTER_FRAMEWORK:

    @AdapterRegistry.register("ealtman2019")
    class EaltmanAdapter(DataAdapter):
        """DataAdapter subclass for the ealtman2019 credit-card dataset.

        Wraps the existing DuckDB aggregation logic to produce user-level
        DataFrames.  Feature generation (TDA, HMM, GMM, etc.) is NOT
        performed here — that is the responsibility of FeatureGroupPipeline.

        Config keys used::

            data.input_dir   — path to raw parquet/csv directory
            data.backend     — backend preference list (default: ["duckdb"])
        """

        def load_raw(self) -> Dict[str, pd.DataFrame]:
            """Load & aggregate 24M transactions → 2,000 user-level rows.

            Returns
            -------
            dict
                ``"main"``  — user-level DataFrame with base aggregation
                    columns (rfm, category ratios, txn stats, temporal).
                ``"transactions_raw"`` — lightweight metadata dict (not a
                    full DataFrame) describing the transaction parquet
                    location so downstream stages can access raw rows if
                    needed (e.g. for sequence building).
            """
            input_dir = self.config.get("data", {}).get(
                "input_dir", "/opt/ml/processing/input"
            )
            backend = self._select_backend()

            # --- Load user / card tables (small) ---
            users = load_users(input_dir)
            cards = load_cards(input_dir)  # noqa: F841 — kept for future use
            user_ids = np.arange(len(users))

            # --- DuckDB aggregation (24M → 2,000) ---
            parquet_path = os.path.join(input_dir, "transactions.parquet")
            con = _duckdb.connect()
            con.execute("SET memory_limit='12GB'")
            con.execute("SET threads TO 4")

            tmp_dir = "/tmp/duckdb_tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            con.execute(f"SET temp_directory='{tmp_dir}'")

            con.execute(f"""
                CREATE VIEW txn AS
                SELECT
                    "user_id" AS "User",
                    "card_id" AS "Card",
                    "year" AS "Year",
                    "month" AS "Month",
                    "day" AS "Day",
                    "time" AS "Time",
                    "amount" AS "Amount",
                    "use_chip" AS "Use Chip",
                    "merchant_id" AS "Merchant Name",
                    "merchant_city" AS "Merchant City",
                    "merchant_state" AS "Merchant State",
                    "zip" AS "Zip",
                    "mcc" AS "MCC",
                    "errors" AS "Errors?",
                    "is_fraud" AS "Is Fraud?",
                    MAKE_DATE("year"::INT, "month"::INT, "day"::INT) AS "Date",
                    EXTRACT(DOW FROM MAKE_DATE("year"::INT, "month"::INT, "day"::INT)) AS "DayOfWeek",
                    EXTRACT(HOUR FROM TRY_CAST("time" AS TIME)) AS "Hour",
                    "year" * 100 + "month" AS "YearMonth"
                FROM read_parquet('{parquet_path}')
            """)

            # Row count for metadata
            total_rows = con.execute("SELECT COUNT(*) FROM txn").fetchone()[0]
            ref_date_row = con.execute(
                'SELECT MAX("Date") FROM txn'
            ).fetchone()
            ref_date = pd.Timestamp(ref_date_row[0])

            logger.info(
                "EaltmanAdapter: %d total txn rows, ref_date=%s, %d users",
                total_rows, ref_date, len(user_ids),
            )

            # --- Pre-aggregation (2,000-row result) ---
            txn_agg = con.execute("""
                SELECT
                    "User",
                    COUNT(*) AS txn_count,
                    SUM("Amount") AS txn_total_amount,
                    AVG("Amount") AS txn_mean_amount,
                    STDDEV("Amount") AS txn_std_amount,
                    MAX("Amount") AS txn_max_amount,
                    MIN("Amount") AS txn_min_amount,
                    MAX("Date") AS last_txn_date,
                    COUNT(DISTINCT "Merchant Name") AS n_merchants,
                    COUNT(DISTINCT "Merchant State") AS n_states,
                    SUM(CASE WHEN "Errors?" IS NOT NULL
                             AND TRIM("Errors?") != '' THEN 1 ELSE 0 END)::FLOAT
                        / COUNT(*)::FLOAT AS error_rate
                FROM txn
                GROUP BY "User"
                ORDER BY "User"
            """).df().set_index("User")
            txn_agg["txn_std_amount"] = txn_agg["txn_std_amount"].fillna(0)
            txn_agg["last_txn_date"] = pd.to_datetime(txn_agg["last_txn_date"])

            # --- Build base feature groups (rfm, category, txn_stats, temporal) ---
            fg_rfm = build_base_rfm(users, txn_agg, ref_date)

            # Category ratios via DuckDB
            mcc_pivot = con.execute("""
                SELECT "User", "MCC", SUM("Amount") AS mcc_amount
                FROM txn GROUP BY "User", "MCC"
            """).df()
            user_total = mcc_pivot.groupby("User")["mcc_amount"].sum()
            top_mccs = (
                mcc_pivot.groupby("MCC")["mcc_amount"]
                .sum()
                .nlargest(N_TOP_MCC)
                .index.tolist()
            )
            fg_cat = pd.DataFrame(index=user_ids)
            mcc_wide = mcc_pivot.pivot_table(
                index="User", columns="MCC", values="mcc_amount",
                aggfunc="sum", fill_value=0,
            )
            totals = user_total.reindex(user_ids, fill_value=1.0).clip(lower=1e-8)
            for i, mcc in enumerate(top_mccs):
                if mcc in mcc_wide.columns:
                    fg_cat[f"cat_{i:03d}"] = (
                        mcc_wide[mcc].reindex(user_ids, fill_value=0) / totals
                    ).values
                else:
                    fg_cat[f"cat_{i:03d}"] = 0.0

            # txn_stats and temporal via existing builders (they accept pandas)
            # Load a small sample is NOT needed — builders use DuckDB internally
            fg_txn = build_base_txn_stats(
                pd.DataFrame(), user_ids
            ) if False else pd.DataFrame(index=user_ids)  # placeholder cols
            # Populate from txn_agg
            for col in txn_agg.columns:
                if col != "last_txn_date":
                    fg_txn[f"txn_{col}"] = txn_agg[col].reindex(
                        user_ids, fill_value=0
                    ).values

            con.close()

            # --- Merge into single user-level DataFrame ---
            user_df = fg_rfm.set_index("user_id")
            for fg in [fg_cat, fg_txn]:
                fg.index = user_ids
                user_df = user_df.join(fg, how="left")
            user_df = user_df.fillna(0).reset_index().rename(
                columns={"index": "user_id"}
            )

            # --- Populate metadata ---
            self._metadata = AdapterMetadata(
                id_col="user_id",
                timestamp_col=None,
                entity_granularity="user",
                num_entities=len(user_ids),
                num_raw_rows=total_rows,
                source_files=[parquet_path],
                backend_used=backend,
            )

            logger.info(
                "EaltmanAdapter: returning main DataFrame %s",
                user_df.shape,
            )

            return {
                "main": user_df,
                "transactions_raw": pd.DataFrame({
                    "parquet_path": [parquet_path],
                    "total_rows": [total_rows],
                    "ref_date": [str(ref_date)],
                    "num_users": [len(user_ids)],
                }),
            }


if __name__ == "__main__":
    args = parse_args()
    logger.info("=== ealtman2019 Adapter Start ===")
    logger.info("  input-dir:  %s", args.input_dir)
    logger.info("  output-dir: %s", args.output_dir)
    run(args.input_dir, args.output_dir)
    logger.info("=== ealtman2019 Adapter Done ===")
