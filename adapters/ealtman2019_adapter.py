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
    ealtman2019_features.parquet -- user_id + ~469D features + 16 labels
    feature_stats.json           -- per-feature {mean, std, min, max, null_pct}
    label_stats.json             -- per-label distribution statistics

Usage (SageMaker Processing):
    python ealtman2019_adapter.py \
        --input-dir /opt/ml/processing/input \
        --output-dir /opt/ml/processing/output

Dependencies: pandas, numpy, scipy (no torch).
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
N_CLUSTERS = 20         # K-means clusters for gmm_clustering placeholder
RANDOM_STATE = 42

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
    """base_category (64D): MCC spending ratios + diversity metrics."""
    # Top MCC codes by overall frequency
    top_mccs = txn["MCC"].value_counts().head(N_TOP_MCC).index.tolist()

    # Per-user total spending
    user_total = txn.groupby("User")["Amount"].sum().rename("total_amount")

    # Per-user per-MCC spending
    mcc_spend = (
        txn.groupby(["User", "MCC"])["Amount"]
        .sum()
        .unstack(fill_value=0)
    )

    # Build ratio for top 60 MCCs
    result = pd.DataFrame(index=user_ids)
    for idx, mcc in enumerate(top_mccs, start=1):
        col = f"category_{idx:03d}"
        if mcc in mcc_spend.columns:
            result[col] = mcc_spend[mcc].reindex(user_ids, fill_value=0)
        else:
            result[col] = 0.0

    # Normalise by total spending
    totals = user_total.reindex(user_ids, fill_value=1.0)
    for c in result.columns:
        result[c] = result[c] / totals.clip(lower=1e-8)

    # Diversity metrics (cols 61-64)
    n_unique_mcc = txn.groupby("User")["MCC"].nunique().reindex(user_ids, fill_value=0)
    result["category_061"] = n_unique_mcc.astype(float)

    # MCC entropy
    mcc_probs = mcc_spend.div(mcc_spend.sum(axis=1).clip(lower=1e-8), axis=0)
    entropy = -(mcc_probs * np.log(mcc_probs.clip(lower=1e-15))).sum(axis=1)
    result["category_062"] = entropy.reindex(user_ids, fill_value=0).astype(float)

    # HHI
    hhi = (mcc_probs ** 2).sum(axis=1)
    result["category_063"] = hhi.reindex(user_ids, fill_value=0).astype(float)

    # Top-3 MCC concentration
    top3_conc = mcc_probs.apply(lambda row: row.nlargest(3).sum(), axis=1)
    result["category_064"] = top3_conc.reindex(user_ids, fill_value=0).astype(float)

    result["user_id"] = user_ids
    cat_cols = [f"category_{i:03d}" for i in range(1, 65)]
    return result[["user_id"] + cat_cols].reset_index(drop=True)


def build_base_txn_stats(txn: pd.DataFrame, user_ids: np.ndarray) -> pd.DataFrame:
    """base_txn_stats (80D): monthly counts/amounts, quarterly change, time-of-day, chip/swipe."""
    result = pd.DataFrame({"user_id": user_ids})

    # Monthly aggregates
    txn_monthly = txn.groupby(["User", "Month"]).agg(
        monthly_count=("Amount", "count"),
        monthly_amount=("Amount", "sum"),
    ).unstack(fill_value=0)

    col_idx = 1

    # Monthly counts (12D)
    for m in range(1, 13):
        col = f"transaction_stats_{col_idx:03d}"
        if ("monthly_count", m) in txn_monthly.columns:
            result[col] = txn_monthly[("monthly_count", m)].reindex(user_ids, fill_value=0).values
        else:
            result[col] = 0.0
        col_idx += 1

    # Monthly amounts (12D)
    for m in range(1, 13):
        col = f"transaction_stats_{col_idx:03d}"
        if ("monthly_amount", m) in txn_monthly.columns:
            result[col] = txn_monthly[("monthly_amount", m)].reindex(user_ids, fill_value=0).values
        else:
            result[col] = 0.0
        col_idx += 1

    # Quarterly change rates (Q1-Q4 amount, Q1-Q4 count = 8D)
    q_amounts = []
    q_counts = []
    for q_months in [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]:
        q_amt = sum(
            txn_monthly[("monthly_amount", m)].reindex(user_ids, fill_value=0)
            for m in q_months
            if ("monthly_amount", m) in txn_monthly.columns
        )
        q_cnt = sum(
            txn_monthly[("monthly_count", m)].reindex(user_ids, fill_value=0)
            for m in q_months
            if ("monthly_count", m) in txn_monthly.columns
        )
        q_amounts.append(q_amt)
        q_counts.append(q_cnt)

    for i in range(4):
        col = f"transaction_stats_{col_idx:03d}"
        result[col] = q_amounts[i].values if hasattr(q_amounts[i], "values") else 0.0
        col_idx += 1
    for i in range(4):
        col = f"transaction_stats_{col_idx:03d}"
        result[col] = q_counts[i].values if hasattr(q_counts[i], "values") else 0.0
        col_idx += 1

    # Weekday/weekend ratio (2D)
    wd_counts = txn.groupby(["User", txn["DayOfWeek"] < 5])["Amount"].count().unstack(fill_value=0)
    total_txn = txn.groupby("User")["Amount"].count().reindex(user_ids, fill_value=1).clip(lower=1)
    col = f"transaction_stats_{col_idx:03d}"
    weekday_cnt = wd_counts[True].reindex(user_ids, fill_value=0) if True in wd_counts.columns else 0
    result[col] = (weekday_cnt / total_txn).values if not isinstance(weekday_cnt, int) else 0.0
    col_idx += 1
    col = f"transaction_stats_{col_idx:03d}"
    weekend_cnt = wd_counts[False].reindex(user_ids, fill_value=0) if False in wd_counts.columns else 0
    result[col] = (weekend_cnt / total_txn).values if not isinstance(weekend_cnt, int) else 0.0
    col_idx += 1

    # Time-of-day ratios (6D)
    for lo, hi in TIME_BINS_6:
        col = f"transaction_stats_{col_idx:03d}"
        mask = (txn["Hour"] >= lo) & (txn["Hour"] < hi)
        bin_cnt = txn[mask].groupby("User")["Amount"].count().reindex(user_ids, fill_value=0)
        result[col] = (bin_cnt / total_txn).values
        col_idx += 1

    # Chip / Swipe / Online ratios (3D)
    for chip_cat in USE_CHIP_CATEGORIES:
        col = f"transaction_stats_{col_idx:03d}"
        mask = txn["Use Chip"].str.strip() == chip_cat
        chip_cnt = txn[mask].groupby("User")["Amount"].count().reindex(user_ids, fill_value=0)
        result[col] = (chip_cnt / total_txn).values
        col_idx += 1

    # Error count, fraud count (2D)
    col = f"transaction_stats_{col_idx:03d}"
    err_cnt = txn[txn["Errors?"].notna() & (txn["Errors?"].str.strip() != "")].groupby("User")["Amount"].count()
    result[col] = err_cnt.reindex(user_ids, fill_value=0).values.astype(float)
    col_idx += 1

    col = f"transaction_stats_{col_idx:03d}"
    fraud_cnt = txn[txn["Is Fraud?"] == 1].groupby("User")["Amount"].count()
    result[col] = fraud_cnt.reindex(user_ids, fill_value=0).values.astype(float)
    col_idx += 1

    # Pad to 80D
    while col_idx <= 80:
        col = f"transaction_stats_{col_idx:03d}"
        result[col] = 0.0
        col_idx += 1

    txn_cols = [f"transaction_stats_{i:03d}" for i in range(1, 81)]
    return result[["user_id"] + txn_cols]


def build_base_temporal(txn: pd.DataFrame, user_ids: np.ndarray) -> pd.DataFrame:
    """base_temporal (60D): rolling means, trends, seasonality."""
    result = pd.DataFrame({"user_id": user_ids})
    col_idx = 1

    # Monthly amount pivot (User x Month)
    monthly_amt = txn.groupby(["User", "Month"])["Amount"].sum().unstack(fill_value=0)
    monthly_cnt = txn.groupby(["User", "Month"])["Amount"].count().unstack(fill_value=0)

    # Ensure months 1-12 exist
    for m in range(1, 13):
        if m not in monthly_amt.columns:
            monthly_amt[m] = 0.0
        if m not in monthly_cnt.columns:
            monthly_cnt[m] = 0.0
    monthly_amt = monthly_amt[range(1, 13)].reindex(user_ids, fill_value=0)
    monthly_cnt = monthly_cnt[range(1, 13)].reindex(user_ids, fill_value=0)

    # Rolling mean amount: 3/6/12 month windows (12 + 12 + 12 = 36D)
    for window in [3, 6, 12]:
        rolled = monthly_amt.T.rolling(window, min_periods=1).mean().T
        for m in range(1, 13):
            col = f"temporal_{col_idx:03d}"
            result[col] = rolled[m].values
            col_idx += 1

    # Rolling mean count: 3 month window (12D) — skip 6/12 to keep dim
    rolled_cnt = monthly_cnt.T.rolling(3, min_periods=1).mean().T
    for m in range(1, 13):
        col = f"temporal_{col_idx:03d}"
        result[col] = rolled_cnt[m].values
        col_idx += 1

    # Trend slope (linear regression on monthly amounts) (1D)
    col = f"temporal_{col_idx:03d}"
    x = np.arange(1, 13, dtype=float)
    slopes = monthly_amt.apply(
        lambda row: np.polyfit(x, row.values.astype(float), 1)[0]
        if row.sum() > 0 else 0.0,
        axis=1,
    )
    result[col] = slopes.values
    col_idx += 1

    # Seasonality: deviation from monthly mean (already captured in rolling)
    # First/last txn span, active months — pad to 60D
    # Active months
    col = f"temporal_{col_idx:03d}"
    active_months = (monthly_amt > 0).sum(axis=1)
    result[col] = active_months.values.astype(float)
    col_idx += 1

    # Span (first-to-last txn in days)
    spans = txn.groupby("User")["Date"].agg(lambda s: (s.max() - s.min()).days)
    col = f"temporal_{col_idx:03d}"
    result[col] = spans.reindex(user_ids, fill_value=0).values.astype(float)
    col_idx += 1

    # Pad to 60D
    while col_idx <= 60:
        col = f"temporal_{col_idx:03d}"
        result[col] = 0.0
        col_idx += 1

    temporal_cols = [f"temporal_{i:03d}" for i in range(1, 61)]
    return result[["user_id"] + temporal_cols]


def build_tda_topology(user_ids: np.ndarray,
                       txn: pd.DataFrame) -> pd.DataFrame:
    """tda_topology (70D): placeholder + autocorrelation / entropy stats."""
    result = pd.DataFrame({"user_id": user_ids})
    col_idx = 1

    # Fill first 50D as 0 (real TDA runs in separate generator)
    for i in range(1, 51):
        result[f"tda_{col_idx:03d}"] = 0.0
        col_idx += 1

    # Autocorrelation of monthly amount (lag 1-12) → 12D
    monthly_amt = txn.groupby(["User", "Month"])["Amount"].sum().unstack(fill_value=0)
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


def build_hmm_states(user_ids: np.ndarray) -> pd.DataFrame:
    """hmm_states (48D): placeholder zeros."""
    result = pd.DataFrame({"user_id": user_ids})
    for i in range(1, 49):
        result[f"hmm_{i:03d}"] = 0.0
    return result


def build_mamba_temporal(user_ids: np.ndarray) -> pd.DataFrame:
    """mamba_temporal (50D): placeholder zeros."""
    result = pd.DataFrame({"user_id": user_ids})
    for i in range(1, 51):
        result[f"mamba_{i:03d}"] = 0.0
    return result


def build_gmm_clustering(base_features: pd.DataFrame,
                         user_ids: np.ndarray) -> pd.DataFrame:
    """gmm_clustering (22D): simple K-means soft assignment placeholder.

    Uses base_rfm + base_txn_stats numeric features for clustering.
    Soft assignment via distance-based probabilities.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    result = pd.DataFrame({"user_id": user_ids})

    # Select numeric features for clustering
    feat_cols = [c for c in base_features.columns if c != "user_id"]
    X = base_features[feat_cols].fillna(0).values.astype(np.float32)

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-means
    n_clusters = min(N_CLUSTERS, len(user_ids))
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10, max_iter=100)
    km.fit(X_scaled)

    # Distance to each cluster centre → softmax → soft assignment probs
    distances = km.transform(X_scaled)  # (n_users, n_clusters)
    # Convert to probabilities (inverse distance softmax)
    neg_dist = -distances
    exp_dist = np.exp(neg_dist - neg_dist.max(axis=1, keepdims=True))
    probs = exp_dist / exp_dist.sum(axis=1, keepdims=True)

    for i in range(n_clusters):
        result[f"gmm_{i + 1:03d}"] = probs[:, i]

    # Entropy
    ent = -(probs * np.log(probs.clip(min=1e-15))).sum(axis=1)
    result["gmm_021"] = ent

    # Dominant cluster
    result["gmm_022"] = km.labels_.astype(float)

    gmm_cols = [f"gmm_{i:03d}" for i in range(1, 23)]
    return result[["user_id"] + gmm_cols]


def build_economics(users: pd.DataFrame,
                    txn: pd.DataFrame,
                    user_ids: np.ndarray) -> pd.DataFrame:
    """economics (17D): spending ratios, MPC, savings rate, volatility."""
    result = pd.DataFrame({"user_id": user_ids})
    col_idx = 1

    yearly_income = users["Yearly Income"].reindex(user_ids).fillna(1).astype(float).clip(lower=1)
    total_debt = users["Total Debt"].reindex(user_ids).fillna(0).astype(float).clip(lower=1)

    # Monthly spending
    monthly_spend = txn.groupby(["User", "Month"])["Amount"].sum().unstack(fill_value=0)
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


def build_multidisciplinary(txn: pd.DataFrame,
                            user_ids: np.ndarray) -> pd.DataFrame:
    """multidisciplinary (24D): chemical kinetics, epidemic, interference, crime."""
    result = pd.DataFrame({"user_id": user_ids})
    col_idx = 1

    monthly_amt = txn.groupby(["User", "Month"])["Amount"].sum().unstack(fill_value=0)
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
    # New merchant visit speed per quarter
    for q_months in [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]:
        col = f"multi_{col_idx:03d}"
        mask = txn["Month"].isin(q_months)
        q_merch = txn[mask].groupby("User")["Merchant Name"].nunique()
        result[col] = q_merch.reindex(user_ids, fill_value=0).values.astype(float)
        col_idx += 1
    # Overall new merchant rate
    total_merch = txn.groupby("User")["Merchant Name"].nunique()
    span_months = (monthly_amt > 0).sum(axis=1).clip(lower=1)
    result[f"multi_{col_idx:03d}"] = (
        total_merch.reindex(user_ids, fill_value=0).values / span_months.values
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
    user_mean = txn.groupby("User")["Amount"].mean()
    user_std = txn.groupby("User")["Amount"].std().fillna(0)
    threshold = user_mean + 3 * user_std

    # Count of txns > 3sigma
    anomaly_cnt = txn.groupby("User").apply(
        lambda g: (g["Amount"] > threshold.get(g.name, np.inf)).sum()
    )
    result[f"multi_{col_idx:03d}"] = anomaly_cnt.reindex(user_ids, fill_value=0).values.astype(float)
    col_idx += 1

    # Anomaly ratio
    total_cnt = txn.groupby("User")["Amount"].count().reindex(user_ids, fill_value=1).clip(lower=1)
    result[f"multi_{col_idx:03d}"] = (
        anomaly_cnt.reindex(user_ids, fill_value=0).values / total_cnt.values
    )
    col_idx += 1

    # Pad to 24D
    while col_idx <= 24:
        result[f"multi_{col_idx:03d}"] = 0.0
        col_idx += 1

    multi_cols = [f"multi_{i:03d}" for i in range(1, 25)]
    return result[["user_id"] + multi_cols]


def build_model_derived(user_ids: np.ndarray) -> pd.DataFrame:
    """model_derived (27D): placeholder zeros."""
    result = pd.DataFrame({"user_id": user_ids})
    for i in range(1, 28):
        result[f"model_derived_{i:03d}"] = 0.0
    return result


def build_merchant_hierarchy(txn: pd.DataFrame,
                             cards: pd.DataFrame,
                             user_ids: np.ndarray) -> pd.DataFrame:
    """merchant_hierarchy (21D): MCC major groups, brand usage, chip×brand cross."""
    result = pd.DataFrame({"user_id": user_ids})
    col_idx = 1

    total_cnt = txn.groupby("User")["Amount"].count().reindex(user_ids, fill_value=1).clip(lower=1)

    # MCC major category ratios (7D)
    for cat_name, mcc_list in MCC_MAJOR.items():
        col = f"merchant_{col_idx:03d}"
        mask = txn["MCC"].isin(mcc_list)
        cnt = txn[mask].groupby("User")["Amount"].count().reindex(user_ids, fill_value=0)
        result[col] = (cnt / total_cnt).values
        col_idx += 1

    # Card brand usage ratio (4D)
    # Need to map User+Card → Card Brand via cards table
    if "Card Brand" in cards.columns:
        card_brand = cards.set_index(["User", "CARD INDEX"])["Card Brand"]
        txn_brand = txn.set_index(["User", "Card"]).join(
            card_brand.rename("CardBrand"), how="left"
        ).reset_index()
    else:
        txn_brand = txn.copy()
        txn_brand["CardBrand"] = "Unknown"

    for brand in CARD_BRANDS:
        col = f"merchant_{col_idx:03d}"
        mask = txn_brand["CardBrand"].str.strip() == brand
        cnt = txn_brand[mask].groupby("User")["Amount"].count().reindex(user_ids, fill_value=0)
        result[col] = (cnt / total_cnt).values
        col_idx += 1

    # Chip/Swipe/Online ratio (3D)
    for chip_cat in USE_CHIP_CATEGORIES:
        col = f"merchant_{col_idx:03d}"
        mask = txn["Use Chip"].str.strip() == chip_cat
        cnt = txn[mask].groupby("User")["Amount"].count().reindex(user_ids, fill_value=0)
        result[col] = (cnt / total_cnt).values
        col_idx += 1

    # Brand x payment method cross (top 7D)
    cross = txn_brand.groupby(["User", "CardBrand", "Use Chip"])["Amount"].count().reset_index()
    cross_pivot = cross.pivot_table(
        index="User",
        columns=["CardBrand", "Use Chip"],
        values="Amount",
        aggfunc="sum",
        fill_value=0,
    )
    # Take top-7 cross columns by overall frequency
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


def build_graph_embeddings(user_ids: np.ndarray) -> pd.DataFrame:
    """graph_embeddings (20D): placeholder zeros."""
    result = pd.DataFrame({"user_id": user_ids})
    for i in range(1, 21):
        result[f"graph_{i:03d}"] = 0.0
    return result


# ===================================================================
# 3. Label Builders
# ===================================================================

def build_labels(users: pd.DataFrame,
                 txn: pd.DataFrame,
                 cards: pd.DataFrame,
                 user_ids: np.ndarray,
                 ref_date: pd.Timestamp) -> pd.DataFrame:
    """Build all 16 labels."""
    result = pd.DataFrame({"user_id": user_ids})

    # Pre-compute aggregates
    user_fraud_rate = txn.groupby("User")["Is Fraud?"].mean()
    median_fraud = user_fraud_rate.median()

    last_txn = txn.groupby("User")["Date"].max()
    three_months_ago = ref_date - pd.Timedelta(days=90)

    monthly_cnt = txn.groupby(["User", "Month"])["Amount"].count().unstack(fill_value=0)
    monthly_amt = txn.groupby(["User", "Month"])["Amount"].sum().unstack(fill_value=0)
    for m in range(1, 13):
        if m not in monthly_cnt.columns:
            monthly_cnt[m] = 0
        if m not in monthly_amt.columns:
            monthly_amt[m] = 0.0
    monthly_cnt = monthly_cnt[range(1, 13)].reindex(user_ids, fill_value=0)
    monthly_amt = monthly_amt[range(1, 13)].reindex(user_ids, fill_value=0)

    total_spend = txn.groupby("User")["Amount"].sum().reindex(user_ids, fill_value=0)

    # Card credit limits per user
    if "Credit Limit" in cards.columns:
        user_credit_limit = cards.groupby("User")["Credit Limit"].sum().reindex(user_ids, fill_value=1).clip(lower=1)
    else:
        user_credit_limit = pd.Series(1.0, index=user_ids)

    # 1. label_is_fraud
    result["label_is_fraud"] = (
        user_fraud_rate.reindex(user_ids, fill_value=0) > median_fraud
    ).astype(int)

    # 2. label_will_transact
    last_txn_ui = last_txn.reindex(user_ids)
    result["label_will_transact"] = (last_txn_ui >= three_months_ago).astype(int).fillna(0)

    # 3. label_churn
    result["label_churn"] = (last_txn_ui < three_months_ago).astype(int).fillna(1)

    # 4. label_retention
    result["label_retention"] = 1 - result["label_churn"]

    # 5. label_life_stage (0:20s, 1:30s, 2:40s, 3:50s, 4:60+)
    age = users["Current Age"].reindex(user_ids).fillna(30).astype(float)
    result["label_life_stage"] = np.clip((age - 20) // 10, 0, 4).astype(int)

    # 6. label_ltv (recent 12 months total)
    result["label_ltv"] = monthly_amt.sum(axis=1).values.astype(float)

    # 7. label_balance_util
    result["label_balance_util"] = (total_spend / user_credit_limit).values.astype(float)

    # 8. label_engagement (monthly frequency trend slope)
    x = np.arange(1, 13, dtype=float)
    slopes = monthly_cnt.apply(
        lambda row: np.polyfit(x, row.values.astype(float), 1)[0]
        if row.sum() > 0 else 0.0,
        axis=1,
    )
    result["label_engagement"] = slopes.values.astype(float)

    # 9. label_channel (0:Chip, 1:Swipe, 2:Online)
    chip_counts = txn.groupby(["User", "Use Chip"])["Amount"].count().unstack(fill_value=0)
    channel_map = {}
    for i, cat in enumerate(USE_CHIP_CATEGORIES):
        if cat in chip_counts.columns:
            channel_map[cat] = i
    if channel_map:
        chip_subset = chip_counts[[c for c in USE_CHIP_CATEGORIES if c in chip_counts.columns]]
        dominant = chip_subset.idxmax(axis=1)
        result["label_channel"] = dominant.map(
            {cat: i for i, cat in enumerate(USE_CHIP_CATEGORIES)}
        ).reindex(user_ids, fill_value=0).astype(int)
    else:
        result["label_channel"] = 0

    # 10. label_timing (8 time bins)
    hour_bins = txn.copy()
    hour_bins["time_bin"] = pd.cut(
        hour_bins["Hour"],
        bins=[b[0] for b in TIME_BINS_8] + [24],
        labels=range(8),
        right=False,
        include_lowest=True,
    )
    dominant_time = hour_bins.groupby("User")["time_bin"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else 0
    )
    result["label_timing"] = dominant_time.reindex(user_ids, fill_value=0).astype(int)

    # 11. label_nba (most frequent MCC major category, 15 classes)
    mcc_counts = txn.groupby(["User", "MCC"])["Amount"].count().reset_index()
    mcc_counts.columns = ["User", "MCC", "cnt"]
    # Map MCC → major category index
    mcc_to_major = {}
    for idx, (cat_name, mcc_list) in enumerate(MCC_MAJOR.items()):
        for mcc in mcc_list:
            mcc_to_major[mcc] = idx
    mcc_counts["major"] = mcc_counts["MCC"].map(mcc_to_major).fillna(7).astype(int)  # 7=other
    user_major = mcc_counts.groupby(["User", "major"])["cnt"].sum().reset_index()
    dominant_major = user_major.loc[user_major.groupby("User")["cnt"].idxmax()]
    dominant_major = dominant_major.set_index("User")["major"]
    result["label_nba"] = dominant_major.reindex(user_ids, fill_value=0).astype(int) % 15

    # 12. label_spending_category (same as nba but mid-level — use top MCC directly)
    top_mcc_per_user = (
        txn.groupby(["User", "MCC"])["Amount"]
        .count()
        .reset_index()
        .sort_values("Amount", ascending=False)
        .drop_duplicates("User")
        .set_index("User")["MCC"]
    )
    # Map to 0-14 via modulo
    result["label_spending_category"] = (
        top_mcc_per_user.reindex(user_ids, fill_value=0).astype(int) % 15
    )

    # 13. label_consumption_cycle (0:weekly, 1:bi-weekly, 2:monthly, 3:irregular)
    avg_gap = txn.sort_values("Date").groupby("User")["Date"].apply(
        lambda s: s.diff().dt.days.median() if len(s) > 1 else 999
    )
    gap = avg_gap.reindex(user_ids, fill_value=999)
    result["label_consumption_cycle"] = pd.cut(
        gap,
        bins=[-np.inf, 10, 20, 35, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int)

    # 14. label_spending_amount (monthly average)
    result["label_spending_amount"] = (monthly_amt.mean(axis=1)).values.astype(float)

    # 15. label_merchant_affinity (HHI of merchant spending)
    merch_spend = txn.groupby(["User", "Merchant Name"])["Amount"].sum()
    user_total_spend = txn.groupby("User")["Amount"].sum()
    merch_share = merch_spend / user_total_spend
    hhi = (merch_share ** 2).groupby("User").sum()
    result["label_merchant_affinity"] = hhi.reindex(user_ids, fill_value=0).values.astype(float)

    # 16. label_brand
    if "Card Brand" in cards.columns:
        card_brand_map = cards.set_index(["User", "CARD INDEX"])["Card Brand"]
        txn_with_brand = txn.set_index(["User", "Card"]).join(
            card_brand_map.rename("CardBrand"), how="left"
        ).reset_index()
        brand_counts = txn_with_brand.groupby(["User", "CardBrand"])["Amount"].count().unstack(fill_value=0)
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

    return result


# ===================================================================
# 4. Transaction Pre-aggregation (for base_rfm)
# ===================================================================

def pre_aggregate_transactions(txn: pd.DataFrame) -> pd.DataFrame:
    """Compute per-user transaction summary for base_rfm."""
    agg = txn.groupby("User").agg(
        txn_count=("Amount", "count"),
        txn_total_amount=("Amount", "sum"),
        txn_mean_amount=("Amount", "mean"),
        txn_std_amount=("Amount", "std"),
        txn_max_amount=("Amount", "max"),
        txn_min_amount=("Amount", "min"),
        last_txn_date=("Date", "max"),
        n_merchants=("Merchant Name", "nunique"),
        n_states=("Merchant State", "nunique"),
    )
    agg["txn_std_amount"] = agg["txn_std_amount"].fillna(0)

    # Error rate
    has_error = txn["Errors?"].notna() & (txn["Errors?"].str.strip() != "")
    error_cnt = txn[has_error].groupby("User")["Amount"].count().rename("err_cnt")
    total_cnt = txn.groupby("User")["Amount"].count().rename("total_cnt")
    error_rate = (error_cnt / total_cnt).fillna(0).rename("error_rate")
    agg = agg.join(error_rate, how="left")
    agg["error_rate"] = agg["error_rate"].fillna(0)

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
# 6. Main Pipeline
# ===================================================================

def run(input_dir: str, output_dir: str) -> None:
    """Execute the full adapter pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Load data ---
    users = load_users(input_dir)
    cards = load_cards(input_dir)

    # Use DuckDB for memory-efficient aggregation of 24M rows
    parquet_path = os.path.join(input_dir, "transactions.parquet")
    try:
        import duckdb
        logger.info("Using DuckDB for memory-efficient transaction loading")
        con = duckdb.connect()
        con.execute("SET memory_limit='8GB'")
        con.execute("SET threads TO 4")
        # Load only needed columns with proper types
        txn = con.execute(f"""
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
                EXTRACT(HOUR FROM "time"::TIME) AS "Hour"
            FROM read_parquet('{parquet_path}')
        """).df()
        con.close()
        logger.info("Loaded %d rows via DuckDB", len(txn))
        # Convert Date to pandas Timestamp
        txn["Date"] = pd.to_datetime(txn["Date"])
        txn["Hour"] = txn["Hour"].astype(int)
        txn["DayOfWeek"] = txn["DayOfWeek"].astype(int)
    except ImportError:
        logger.warning("DuckDB not available, falling back to pandas")
        txn = load_transactions_chunked(input_dir)

    # User IDs (0-indexed matching Person column)
    user_ids = np.arange(len(users))
    ref_date = txn["Date"].max()
    logger.info("Reference date: %s, N users: %d", ref_date, len(user_ids))

    # --- Pre-aggregation ---
    logger.info("Pre-aggregating transactions ...")
    txn_agg = pre_aggregate_transactions(txn)

    # --- Feature Groups ---
    logger.info("Building base_rfm (34D) ...")
    fg_rfm = build_base_rfm(users, txn_agg, ref_date)

    logger.info("Building base_category (64D) ...")
    fg_cat = build_base_category(txn, user_ids)

    logger.info("Building base_txn_stats (80D) ...")
    fg_txn = build_base_txn_stats(txn, user_ids)

    logger.info("Building base_temporal (60D) ...")
    fg_temporal = build_base_temporal(txn, user_ids)

    logger.info("Building tda_topology (70D) ...")
    fg_tda = build_tda_topology(user_ids, txn)

    logger.info("Building hmm_states (48D) ...")
    fg_hmm = build_hmm_states(user_ids)

    logger.info("Building mamba_temporal (50D) ...")
    fg_mamba = build_mamba_temporal(user_ids)

    logger.info("Building gmm_clustering (22D) ...")
    # Use rfm + txn_stats as base features for clustering
    clustering_base = fg_rfm.merge(fg_txn, on="user_id")
    fg_gmm = build_gmm_clustering(clustering_base, user_ids)

    logger.info("Building economics (17D) ...")
    fg_econ = build_economics(users, txn, user_ids)

    logger.info("Building multidisciplinary (24D) ...")
    fg_multi = build_multidisciplinary(txn, user_ids)

    logger.info("Building model_derived (27D) ...")
    fg_model = build_model_derived(user_ids)

    logger.info("Building merchant_hierarchy (21D) ...")
    fg_merch = build_merchant_hierarchy(txn, cards, user_ids)

    logger.info("Building graph_embeddings (20D) ...")
    fg_graph = build_graph_embeddings(user_ids)

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
    labels = build_labels(users, txn, cards, user_ids, ref_date)
    label_cols = [c for c in labels.columns if c.startswith("label_")]

    # --- Final merge ---
    final = merged.merge(labels, on="user_id", how="left")

    # Replace inf/nan
    final = final.replace([np.inf, -np.inf], np.nan).fillna(0)

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


if __name__ == "__main__":
    args = parse_args()
    logger.info("=== ealtman2019 Adapter Start ===")
    logger.info("  input-dir:  %s", args.input_dir)
    logger.info("  output-dir: %s", args.output_dir)
    run(args.input_dir, args.output_dir)
    logger.info("=== ealtman2019 Adapter Done ===")
