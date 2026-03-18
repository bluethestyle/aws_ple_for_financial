#!/usr/bin/env python3
"""
Adapter: Bank Churners → PLE Pipeline Input Format.

Converts the Kaggle BankChurners dataset (10,127 customers × 21 columns)
into the format expected by run_distillation.py:
  - Feature columns (numeric, encoded categoricals)
  - Label columns (label_ctr, label_churn, label_ltv, etc.)

Source: kaggle.com/datasets/sakshigoyal7/credit-card-customers
Input:  data/converted/02_financial_bank_churners.parquet
Output: data/adapted/bank_churners_train.parquet

Usage:
    python adapters/bank_churners.py
    python adapters/bank_churners.py --output-s3 s3://aiops-ple-financial/data/adapted/

Then test the pipeline:
    python scripts/run_distillation.py \
        --data-path data/adapted/bank_churners_train.parquet \
        --config configs/test/bank_churners_pipeline.yaml \
        --output-dir /opt/ml/model \
        --skip-soft-label-gen
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("adapter.bank_churners")


def adapt(input_path: str, output_path: str) -> pd.DataFrame:
    """Convert BankChurners to pipeline format."""
    logger.info("Reading %s", input_path)
    df = pd.read_parquet(input_path)
    logger.info("Input: %d rows × %d cols", len(df), len(df.columns))

    # ==================================================================
    # 1. Feature engineering (raw → numeric features)
    # ==================================================================

    features = pd.DataFrame()

    # Customer ID (for tracking, not a feature)
    features["customer_id"] = df["CLIENTNUM"]

    # --- Demographic features ---
    features["age"] = df["Customer_Age"].astype("float32")
    features["age_squared"] = (df["Customer_Age"] ** 2).astype("float32")
    features["dependent_count"] = df["Dependent_count"].astype("float32")
    features["gender_male"] = (df["Gender"] == "M").astype("float32")
    features["months_on_book"] = df["Months_on_book"].astype("float32")

    # Education level encoding (ordinal)
    edu_map = {
        "Uneducated": 0, "High School": 1, "College": 2,
        "Graduate": 3, "Post-Graduate": 4, "Doctorate": 5, "Unknown": 2,
    }
    features["education_level"] = df["Education_Level"].map(edu_map).fillna(2).astype("float32")

    # Marital status one-hot
    for status in ["Married", "Single", "Divorced"]:
        features[f"marital_{status.lower()}"] = (df["Marital_Status"] == status).astype("float32")

    # Income category encoding (ordinal)
    income_map = {
        "Less than $40K": 1, "$40K - $60K": 2, "$60K - $80K": 3,
        "$80K - $120K": 4, "$120K +": 5, "Unknown": 3,
    }
    features["income_level"] = df["Income_Category"].map(income_map).fillna(3).astype("float32")

    # Card category one-hot
    for cat in ["Blue", "Silver", "Gold", "Platinum"]:
        features[f"card_{cat.lower()}"] = (df["Card_Category"] == cat).astype("float32")

    # --- Financial behavior features ---
    features["total_relationship_count"] = df["Total_Relationship_Count"].astype("float32")
    features["months_inactive_12mon"] = df["Months_Inactive_12_mon"].astype("float32")
    features["contacts_count_12mon"] = df["Contacts_Count_12_mon"].astype("float32")
    features["credit_limit"] = df["Credit_Limit"].astype("float32")
    features["total_revolving_bal"] = df["Total_Revolving_Bal"].astype("float32")
    features["avg_open_to_buy"] = df["Avg_Open_To_Buy"].astype("float32")
    features["total_amt_chng_q4q1"] = df["Total_Amt_Chng_Q4_Q1"].astype("float32")
    features["total_trans_amt"] = df["Total_Trans_Amt"].astype("float32")
    features["total_trans_ct"] = df["Total_Trans_Ct"].astype("float32")
    features["total_ct_chng_q4q1"] = df["Total_Ct_Chng_Q4_Q1"].astype("float32")
    features["avg_utilization_ratio"] = df["Avg_Utilization_Ratio"].astype("float32")

    # --- Derived features ---
    features["avg_trans_amount"] = (
        df["Total_Trans_Amt"] / df["Total_Trans_Ct"].clip(lower=1)
    ).astype("float32")

    features["credit_usage_pct"] = (
        df["Total_Revolving_Bal"] / df["Credit_Limit"].clip(lower=1)
    ).astype("float32")

    features["trans_frequency"] = (
        df["Total_Trans_Ct"] / df["Months_on_book"].clip(lower=1)
    ).astype("float32")

    features["inactive_ratio"] = (
        df["Months_Inactive_12_mon"] / 12.0
    ).astype("float32")

    features["contact_per_trans"] = (
        df["Contacts_Count_12_mon"] / df["Total_Trans_Ct"].clip(lower=1)
    ).astype("float32")

    features["amt_per_relationship"] = (
        df["Total_Trans_Amt"] / df["Total_Relationship_Count"].clip(lower=1)
    ).astype("float32")

    features["revolving_per_limit"] = (
        df["Total_Revolving_Bal"] / df["Credit_Limit"].clip(lower=1)
    ).astype("float32")

    features["q4q1_amt_x_ct"] = (
        df["Total_Amt_Chng_Q4_Q1"] * df["Total_Ct_Chng_Q4_Q1"]
    ).astype("float32")

    features["engagement_score"] = (
        (df["Total_Trans_Ct"] / df["Total_Trans_Ct"].max()) * 0.5
        + (1 - df["Months_Inactive_12_mon"] / 12.0) * 0.3
        + (df["Total_Relationship_Count"] / df["Total_Relationship_Count"].max()) * 0.2
    ).astype("float32")

    logger.info("Features: %d columns", len(features.columns) - 1)  # exclude customer_id

    # ==================================================================
    # 2. Label generation (16 tasks)
    # ==================================================================

    labels = pd.DataFrame()

    # --- Binary tasks ---
    labels["label_churn"] = df["Churn"].astype("int8")

    labels["label_retention"] = (1 - df["Churn"]).astype("int8")

    # CTR proxy: high transaction count = engaged = would click
    trans_median = df["Total_Trans_Ct"].median()
    labels["label_ctr"] = (df["Total_Trans_Ct"] > trans_median).astype("int8")

    # CVR proxy: high utilization = actively using = converted
    util_median = df["Avg_Utilization_Ratio"].median()
    labels["label_cvr"] = (df["Avg_Utilization_Ratio"] > util_median).astype("int8")

    # --- Regression tasks ---
    labels["label_ltv"] = df["Total_Trans_Amt"].astype("float32")

    labels["label_balance_util"] = df["Avg_Utilization_Ratio"].astype("float32")

    labels["label_engagement"] = features["engagement_score"].astype("float32")

    labels["label_spending_amount"] = df["Total_Trans_Amt"].astype("float32")

    labels["label_merchant_affinity"] = (
        df["Total_Relationship_Count"] / df["Total_Relationship_Count"].max()
    ).astype("float32")

    # --- Multiclass tasks ---
    # Life stage: age-based bucketing (6 classes)
    labels["label_life_stage"] = pd.cut(
        df["Customer_Age"],
        bins=[0, 30, 40, 50, 60, 70, 100],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype("int8")

    # Channel: card category as proxy (3 classes: basic/mid/premium)
    channel_map = {"Blue": 0, "Silver": 1, "Gold": 2, "Platinum": 2}
    labels["label_channel"] = df["Card_Category"].map(channel_map).fillna(0).astype("int8")

    # Timing: months_on_book bucketed into 28 bins
    labels["label_timing"] = pd.cut(
        df["Months_on_book"],
        bins=28, labels=range(28),
    ).astype("int8")

    # NBA: next best action proxy (12 classes based on behavior segments)
    nba_score = (
        df["Total_Amt_Chng_Q4_Q1"].rank(pct=True) * 4
        + df["Total_Ct_Chng_Q4_Q1"].rank(pct=True) * 4
        + df["Avg_Utilization_Ratio"].rank(pct=True) * 4
    ).astype(int).clip(0, 11)
    labels["label_nba"] = nba_score.astype("int8")

    # Spending category: amount-based bucketing (12 classes)
    labels["label_spending_category"] = pd.qcut(
        df["Total_Trans_Amt"], q=12, labels=range(12), duplicates="drop",
    ).astype("int8")

    # Consumption cycle: transaction frequency bucketing (7 classes)
    labels["label_consumption_cycle"] = pd.qcut(
        df["Total_Trans_Ct"], q=7, labels=range(7), duplicates="drop",
    ).astype("int8")

    # Brand prediction: income × card combination (proxy, up to 24 classes → cap at 128)
    labels["label_next_brand"] = (
        df["Income_Category"].cat.codes * 4 + df["Card_Category"].cat.codes
    ).clip(0, 23).astype("int8")

    logger.info("Labels: %d tasks", len(labels.columns))

    # ==================================================================
    # 3. Combine and save
    # ==================================================================

    result = pd.concat([features, labels], axis=1)

    # Replace any remaining NaN
    result = result.fillna(0)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, engine="pyarrow")

    size_mb = output.stat().st_size / (1024 * 1024)
    feature_cols = [c for c in result.columns if not c.startswith("label_") and c != "customer_id"]
    label_cols = [c for c in result.columns if c.startswith("label_")]

    logger.info("=" * 60)
    logger.info("Output: %s (%.1fMB)", output, size_mb)
    logger.info("  Rows: %d", len(result))
    logger.info("  Features: %d", len(feature_cols))
    logger.info("  Labels: %d", len(label_cols))
    logger.info("  Label distribution:")
    for lc in label_cols:
        if result[lc].nunique() <= 2:
            pos_rate = result[lc].mean()
            logger.info("    %s: %.1f%% positive", lc, pos_rate * 100)
        else:
            logger.info("    %s: %d classes", lc, result[lc].nunique())
    logger.info("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(description="Adapt BankChurners for PLE pipeline")
    parser.add_argument(
        "--input", type=str,
        default="data/converted/02_financial_bank_churners.parquet",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/adapted/bank_churners_train.parquet",
    )
    parser.add_argument("--output-s3", type=str, default="")
    args = parser.parse_args()

    df = adapt(args.input, args.output)

    if args.output_s3:
        import boto3
        s3_key = f"{args.output_s3.rstrip('/')}/bank_churners_train.parquet"
        bucket = s3_key.replace("s3://", "").split("/")[0]
        key = "/".join(s3_key.replace("s3://", "").split("/")[1:])
        boto3.client("s3").upload_file(args.output, bucket, key)
        logger.info("Uploaded to %s", s3_key)


if __name__ == "__main__":
    main()
