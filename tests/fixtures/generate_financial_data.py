"""
Synthetic financial customer data generator for end-to-end testing.

Generates realistic, deterministic financial data with correlated multi-task
labels across 16 tasks.  Output is saved as Parquet files (train / validation /
test split).  Feature dimensions approximate the production 644-D space using
a simplified but representative feature set.

Usage::

    # As a module
    from tests.fixtures.generate_financial_data import FinancialDataGenerator
    gen = FinancialDataGenerator(n_samples=10_000, seed=42)
    train_df, val_df, test_df = gen.generate_split()

    # As a CLI
    python -m tests.fixtures.generate_financial_data --n 10000 --output outputs/synthetic_financial/
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 16 task definitions (name, type, label_col)
# ---------------------------------------------------------------------------
TASK_DEFINITIONS: List[Dict[str, str]] = [
    {"name": "ctr",              "type": "binary",      "label_col": "label_ctr"},
    {"name": "cvr",              "type": "binary",      "label_col": "label_cvr"},
    {"name": "churn",            "type": "binary",      "label_col": "label_churn"},
    {"name": "retention",        "type": "binary",      "label_col": "label_retention"},
    {"name": "nba",              "type": "multiclass",  "label_col": "label_nba"},
    {"name": "life_stage",       "type": "multiclass",  "label_col": "label_life_stage"},
    {"name": "ltv",              "type": "regression",  "label_col": "label_ltv"},
    {"name": "engagement",       "type": "regression",  "label_col": "label_engagement"},
    {"name": "credit_risk",      "type": "binary",      "label_col": "label_credit_risk"},
    {"name": "fraud",            "type": "binary",      "label_col": "label_fraud"},
    {"name": "upsell",           "type": "binary",      "label_col": "label_upsell"},
    {"name": "cross_sell",       "type": "binary",      "label_col": "label_cross_sell"},
    {"name": "satisfaction",     "type": "regression",  "label_col": "label_satisfaction"},
    {"name": "default_prob",     "type": "regression",  "label_col": "label_default_prob"},
    {"name": "channel_pref",     "type": "multiclass",  "label_col": "label_channel_pref"},
    {"name": "product_affinity", "type": "ranking",     "label_col": "label_product_affinity"},
]

# ---------------------------------------------------------------------------
# Feature group definitions (name -> number of numeric features to generate)
# ---------------------------------------------------------------------------
FEATURE_GROUPS: Dict[str, int] = {
    "demographics":         12,   # age, income, tenure, etc.
    "account_balances":     20,   # checking, savings, investment, loan amounts
    "transaction_stats":    40,   # count/amount/avg per category, rolling windows
    "card_usage":           25,   # card count, utilisation, foreign txns, rewards
    "digital_engagement":   20,   # app logins, web visits, feature usage
    "temporal":             30,   # day-of-week, month, recency, frequency, monetary
    "product_holdings":     15,   # binary/count of held products
    "credit_profile":       20,   # credit score, limits, utilisation, inquiries
    "interaction_history":  25,   # call centre, branch visits, complaints
    "derived_ratios":       20,   # balance-to-income, utilisation ratios
}

# Categorical features
CATEGORICAL_FEATURES: Dict[str, List[str]] = {
    "gender":            ["M", "F", "O"],
    "marital_status":    ["single", "married", "divorced", "widowed"],
    "education":         ["high_school", "bachelors", "masters", "doctorate", "other"],
    "employment":        ["employed", "self_employed", "retired", "student", "unemployed"],
    "region":            ["seoul", "gyeonggi", "busan", "daegu", "incheon", "gwangju",
                          "daejeon", "ulsan", "sejong", "other"],
    "customer_segment":  ["mass", "affluent", "premier", "private_banking"],
    "primary_channel":   ["mobile", "web", "branch", "call_centre", "atm"],
    "risk_grade":        ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C"],
}

# Total: sum(FEATURE_GROUPS.values()) numeric + len(CATEGORICAL_FEATURES) categorical
# = 227 numeric + 8 categorical = 235 raw features
# After one-hot encoding of categoricals, this approximates ~280 features.
# With interaction/polynomial features in the pipeline, reaches ~644-D.


@dataclass
class FinancialDataGenerator:
    """Generate deterministic synthetic financial customer data.

    Parameters
    ----------
    n_samples : int
        Number of customers to generate.
    seed : int
        Random seed for reproducibility.
    train_ratio : float
        Fraction of data for training.
    val_ratio : float
        Fraction of data for validation.  Remainder goes to test.
    """

    n_samples: int = 10_000
    seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    def generate(self) -> pd.DataFrame:
        """Generate the full DataFrame with features and labels."""
        rng = np.random.default_rng(self.seed)
        n = self.n_samples

        # --- Latent factors driving correlated features and labels ---
        # 5 latent factors: wealth, activity, risk, digital_savvy, loyalty
        latents = rng.standard_normal((n, 5))
        wealth    = latents[:, 0]
        activity  = latents[:, 1]
        risk      = latents[:, 2]
        digital   = latents[:, 3]
        loyalty   = latents[:, 4]

        # Collect all columns in a dict, then build DataFrame at once
        all_columns: Dict[str, np.ndarray] = {"customer_id": np.arange(n)}

        # --- Numeric feature groups ---
        for group_name, n_features in FEATURE_GROUPS.items():
            cols = self._generate_feature_group(
                rng, n, group_name, n_features, latents,
            )
            all_columns.update(cols)

        # --- Categorical features ---
        for cat_name, categories in CATEGORICAL_FEATURES.items():
            # Use latent factors to create non-uniform distributions
            logits = rng.standard_normal((n, len(categories)))
            # Bias certain categories based on latents
            if cat_name == "customer_segment":
                logits[:, -1] += wealth * 0.8  # wealthier -> private banking
                logits[:, -2] += wealth * 0.4
            elif cat_name == "primary_channel":
                logits[:, 0] += digital * 0.6  # digital-savvy -> mobile
                logits[:, 1] += digital * 0.3
            elif cat_name == "risk_grade":
                logits[:, 0] += -risk * 0.5  # low risk -> AAA
                logits[:, -1] += risk * 0.5   # high risk -> C

            probs = _softmax(logits)
            choices = np.array([
                rng.choice(categories, p=probs[i])
                for i in range(n)
            ])
            all_columns[cat_name] = choices

        # Build DataFrame in one shot (avoids fragmentation warnings)
        df = pd.DataFrame(all_columns)

        # --- Labels (16 tasks, correlated via latent factors) ---
        df = self._generate_labels(rng, df, wealth, activity, risk, digital, loyalty)

        n_numeric = sum(FEATURE_GROUPS.values())
        n_cat = len(CATEGORICAL_FEATURES)
        logger.info(
            "Generated %d samples: %d numeric + %d categorical features, 16 task labels",
            n, n_numeric, n_cat,
        )
        return df

    def generate_split(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate and split into train / validation / test DataFrames."""
        df = self.generate()
        n = len(df)
        rng = np.random.default_rng(self.seed + 1)
        indices = rng.permutation(n)

        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        logger.info(
            "Split: train=%d, val=%d, test=%d",
            len(train_df), len(val_df), len(test_df),
        )
        return train_df, val_df, test_df

    def save_splits(self, output_dir: str) -> Dict[str, str]:
        """Generate, split, and save as Parquet files.

        Returns
        -------
        dict
            Mapping of split name to file path.
        """
        train_df, val_df, test_df = self.generate_split()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}
        for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            path = str(out / f"{name}.parquet")
            split_df.to_parquet(path, index=False)
            paths[name] = path
            logger.info("Saved %s: %s (%d rows)", name, path, len(split_df))

        return paths

    # ------------------------------------------------------------------
    # Feature generation helpers
    # ------------------------------------------------------------------

    def _generate_feature_group(
        self,
        rng: np.random.Generator,
        n: int,
        group_name: str,
        n_features: int,
        latents: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Generate a group of correlated numeric features."""
        cols: Dict[str, np.ndarray] = {}

        # Each feature is a noisy linear combination of latent factors
        for i in range(n_features):
            col_name = f"{group_name}_{i:03d}"
            weights = rng.standard_normal(latents.shape[1]) * 0.5
            noise = rng.standard_normal(n) * 0.3
            raw = latents @ weights + noise

            # Apply group-specific transformations for realism
            if group_name == "demographics":
                if i == 0:  # age
                    raw = np.clip(raw * 10 + 45, 18, 85).astype(np.float64)
                elif i == 1:  # annual_income
                    raw = np.clip(np.exp(raw * 0.5 + 10), 1e4, 1e7)
                elif i == 2:  # tenure_months
                    raw = np.clip(raw * 24 + 60, 0, 360).astype(np.float64)
                else:
                    raw = _standardize(raw)
            elif group_name == "account_balances":
                # Log-normal distribution for balances
                raw = np.clip(np.exp(raw * 0.8 + 8), 0, 1e8)
            elif group_name == "transaction_stats":
                if i % 3 == 0:  # counts
                    raw = np.clip(np.abs(raw) * 20, 0, 500).astype(np.float64)
                elif i % 3 == 1:  # amounts
                    raw = np.clip(np.exp(raw * 0.5 + 6), 0, 1e6)
                else:  # averages
                    raw = np.clip(np.abs(raw) * 50 + 30, 0, 5000)
            elif group_name == "card_usage":
                if "utilisation" in col_name or i % 5 == 0:
                    raw = np.clip(_sigmoid(raw), 0, 1)
                else:
                    raw = np.clip(np.abs(raw) * 10, 0, 100)
            elif group_name == "temporal":
                if i < 7:  # day-of-week proportions
                    raw = np.clip(_sigmoid(raw), 0, 1)
                elif i < 19:  # month proportions
                    raw = np.clip(_sigmoid(raw), 0, 1)
                else:  # recency/frequency
                    raw = np.clip(np.abs(raw) * 30, 0, 365)
            elif group_name == "product_holdings":
                raw = (raw > 0).astype(np.float64)
            elif group_name == "credit_profile":
                if i == 0:  # credit score
                    raw = np.clip(raw * 80 + 700, 300, 850).astype(np.float64)
                else:
                    raw = _standardize(raw)
            else:
                raw = _standardize(raw)

            cols[col_name] = raw

        return cols

    def _generate_labels(
        self,
        rng: np.random.Generator,
        df: pd.DataFrame,
        wealth: np.ndarray,
        activity: np.ndarray,
        risk: np.ndarray,
        digital: np.ndarray,
        loyalty: np.ndarray,
    ) -> pd.DataFrame:
        """Generate correlated multi-task labels from latent factors."""
        n = len(df)

        # --- Binary labels ---
        # CTR: activity + digital
        score_ctr = 0.4 * activity + 0.3 * digital + rng.normal(0, 0.3, n)
        df["label_ctr"] = (score_ctr > 0.2).astype(int)

        # CVR: activity + wealth (conditioned on click)
        score_cvr = 0.3 * activity + 0.3 * wealth + rng.normal(0, 0.3, n)
        df["label_cvr"] = ((score_cvr > 0.5) & (df["label_ctr"] == 1)).astype(int)

        # Churn: -loyalty - activity + risk
        score_churn = -0.4 * loyalty - 0.2 * activity + 0.2 * risk + rng.normal(0, 0.3, n)
        df["label_churn"] = (score_churn > 0.3).astype(int)

        # Retention: loyalty + activity (inverse of churn, but not exact)
        score_ret = 0.4 * loyalty + 0.2 * activity - 0.1 * risk + rng.normal(0, 0.3, n)
        df["label_retention"] = (score_ret > -0.1).astype(int)

        # Credit risk: risk factor
        score_cr = 0.5 * risk - 0.2 * wealth + rng.normal(0, 0.3, n)
        df["label_credit_risk"] = (score_cr > 0.3).astype(int)

        # Fraud: risk + unusual activity patterns
        score_fraud = 0.3 * risk + 0.1 * activity + rng.normal(0, 0.5, n)
        df["label_fraud"] = (score_fraud > 1.5).astype(int)  # rare event

        # Upsell: wealth + loyalty
        score_upsell = 0.3 * wealth + 0.3 * loyalty + rng.normal(0, 0.3, n)
        df["label_upsell"] = (score_upsell > 0.4).astype(int)

        # Cross-sell: activity + digital
        score_xs = 0.3 * activity + 0.2 * digital + 0.1 * wealth + rng.normal(0, 0.3, n)
        df["label_cross_sell"] = (score_xs > 0.3).astype(int)

        # --- Multiclass labels ---
        # Next Best Action (5 classes: save, invest, borrow, insure, transact)
        nba_logits = np.column_stack([
            0.3 * wealth + rng.normal(0, 0.5, n),         # save
            0.4 * wealth + 0.2 * digital + rng.normal(0, 0.5, n),  # invest
            -0.2 * wealth + 0.3 * risk + rng.normal(0, 0.5, n),    # borrow
            0.1 * risk + rng.normal(0, 0.5, n),           # insure
            0.3 * activity + 0.2 * digital + rng.normal(0, 0.5, n),  # transact
        ])
        df["label_nba"] = nba_logits.argmax(axis=1)

        # Life stage (6 classes: student, early_career, family, established, pre_retire, retired)
        ls_logits = np.column_stack([
            -0.5 * wealth + rng.normal(0, 0.5, n),
            -0.2 * wealth + 0.2 * activity + rng.normal(0, 0.5, n),
            0.1 * wealth + 0.1 * loyalty + rng.normal(0, 0.5, n),
            0.3 * wealth + 0.2 * loyalty + rng.normal(0, 0.5, n),
            0.4 * wealth + 0.3 * loyalty + rng.normal(0, 0.5, n),
            0.5 * wealth + 0.4 * loyalty - 0.2 * activity + rng.normal(0, 0.5, n),
        ])
        df["label_life_stage"] = ls_logits.argmax(axis=1)

        # Channel preference (5 classes: mobile, web, branch, call, atm)
        cp_logits = np.column_stack([
            0.5 * digital + rng.normal(0, 0.5, n),
            0.3 * digital + rng.normal(0, 0.5, n),
            -0.3 * digital + 0.2 * loyalty + rng.normal(0, 0.5, n),
            -0.2 * digital + rng.normal(0, 0.5, n),
            -0.4 * digital + rng.normal(0, 0.5, n),
        ])
        df["label_channel_pref"] = cp_logits.argmax(axis=1)

        # --- Regression labels ---
        # LTV (lifetime value): wealth + loyalty + activity
        ltv_raw = 0.4 * wealth + 0.3 * loyalty + 0.2 * activity + rng.normal(0, 0.2, n)
        df["label_ltv"] = np.clip(np.exp(ltv_raw * 0.5 + 8), 100, 1e6).astype(np.float64)

        # Engagement score: activity + digital (0-100 scale)
        eng_raw = 0.4 * activity + 0.3 * digital + 0.1 * loyalty + rng.normal(0, 0.3, n)
        df["label_engagement"] = np.clip(eng_raw * 15 + 50, 0, 100).astype(np.float64)

        # Satisfaction score (1-10)
        sat_raw = 0.3 * loyalty + 0.2 * digital - 0.1 * risk + rng.normal(0, 0.3, n)
        df["label_satisfaction"] = np.clip(sat_raw * 1.5 + 6, 1, 10).astype(np.float64)

        # Default probability (0-1)
        dp_raw = 0.4 * risk - 0.3 * wealth - 0.1 * loyalty + rng.normal(0, 0.2, n)
        df["label_default_prob"] = np.clip(_sigmoid(dp_raw), 0.001, 0.999)

        # Product affinity score (ranking, 0-1)
        pa_raw = 0.3 * activity + 0.2 * wealth + 0.2 * digital + rng.normal(0, 0.3, n)
        df["label_product_affinity"] = np.clip(_sigmoid(pa_raw), 0, 1)

        return df

    # ------------------------------------------------------------------
    # Class-level accessors
    # ------------------------------------------------------------------

    @staticmethod
    def get_task_definitions() -> List[Dict[str, str]]:
        """Return the 16 task definitions."""
        return TASK_DEFINITIONS

    @staticmethod
    def get_numeric_feature_names() -> List[str]:
        """Return all numeric feature column names."""
        names: List[str] = []
        for group_name, n_features in FEATURE_GROUPS.items():
            for i in range(n_features):
                names.append(f"{group_name}_{i:03d}")
        return names

    @staticmethod
    def get_categorical_feature_names() -> List[str]:
        """Return all categorical feature column names."""
        return list(CATEGORICAL_FEATURES.keys())

    @staticmethod
    def get_label_columns() -> List[str]:
        """Return all label column names."""
        return [t["label_col"] for t in TASK_DEFINITIONS]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _standardize(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std < 1e-8:
        return x - x.mean()
    return (x - x.mean()) / std


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate synthetic financial data")
    parser.add_argument("--n", type=int, default=10_000, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="outputs/synthetic_financial/", help="Output directory")
    args = parser.parse_args()

    gen = FinancialDataGenerator(n_samples=args.n, seed=args.seed)
    paths = gen.save_splits(args.output)

    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
