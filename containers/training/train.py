"""
SageMaker training entry point.

This script is executed inside the SageMaker training container.  It:

1. Parses SageMaker environment variables and hyperparameters.
2. Loads training / validation data from ``/opt/ml/input/data/``.
3. Builds the PLE model from the config.
4. Optionally resumes from checkpoint (Spot auto-resume).
5. Runs the training loop.
6. Saves the final model to ``/opt/ml/model/`` for SageMaker packaging.
7. Prints metrics to stdout for CloudWatch capture.

SageMaker environment variables
-------------------------------
SM_MODEL_DIR        /opt/ml/model
SM_CHANNEL_TRAIN    /opt/ml/input/data/train
SM_CHANNEL_VALIDATION  /opt/ml/input/data/validation
SM_OUTPUT_DATA_DIR  /opt/ml/output/data
SM_NUM_GPUS         number of GPUs
SM_HPS              JSON dict of hyperparameters
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Enable synchronous CUDA error reporting for debugging
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.training.trainer import PLETrainer
from core.training.config import TrainingConfig

# ---------------------------------------------------------------------------
# Logging — SageMaker captures stdout/stderr for CloudWatch
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sagemaker-training")


# ---------------------------------------------------------------------------
# SageMaker environment helpers
# ---------------------------------------------------------------------------

def get_sm_env() -> Dict[str, str]:
    """Collect all SM_* environment variables into a dict."""
    return {k: v for k, v in os.environ.items() if k.startswith("SM_")}


def get_hyperparameters() -> Dict[str, Any]:
    """Parse hyperparameters from SM_HPS or /opt/ml/input/config/."""
    # Try SM_HPS first (set by SageMaker SDK)
    sm_hps = os.environ.get("SM_HPS")
    if sm_hps:
        return json.loads(sm_hps)

    # Fallback: read hyperparameters.json
    hp_path = Path("/opt/ml/input/config/hyperparameters.json")
    if hp_path.exists():
        with open(hp_path) as f:
            raw = json.load(f)
        # SageMaker stringifies all values
        return {k: _parse_hp_value(v) for k, v in raw.items()}

    return {}


def _parse_hp_value(v: str) -> Any:
    """Best-effort parse a stringified hyperparameter value."""
    if isinstance(v, str):
        lower = v.lower()
        if lower in ("true", "false"):
            return lower == "true"
        # Try JSON first (for lists/dicts passed as strings)
        if lower.startswith("[") or lower.startswith("{"):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                pass
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
    return v


# ---------------------------------------------------------------------------
# Feature group → column prefix mapping
# ---------------------------------------------------------------------------

# Maps feature group names (as used in the ablation config's "remove" lists)
# to the column name prefixes they produce in the feature DataFrame.
# Generated features use "{group_name}_{i}" naming; transform features use
# their source column prefixes.
# Module-level state for scaler persistence (populated by load_data, read by main)
_module_scaler = None
_module_continuous_cols = []

FEATURE_GROUP_COLUMN_PREFIXES: Dict[str, list] = {
    # Transform groups — covers both inline DuckDB path (cat_, txn_, temp_)
    # and standalone function path (category_, transaction_stats_, temporal_)
    "base_rfm": ["rfm_"],
    "base_category": ["category_", "cat_"],
    "base_txn_stats": ["transaction_stats_", "txn_"],
    "base_temporal": ["temporal_", "temp_"],
    "multi_source": [
        "deposit_behavior_", "membership_utilization_",
        "investment_propensity_", "credit_risk_",
        "digital_engagement_", "product_diversity_",
    ],
    "extended_source": [
        "insurance_holding_", "insurance_claim_",
        "refund_loan_", "consultation_history_",
        "stt_analysis_", "product_interest_",
        "campaign_participation_", "marketing_response_",
        "overseas_payment_", "open_banking_",
        "e_finance_", "merchant_preference_",
    ],
    # Generated groups — covers both long prefix and short DuckDB prefix
    "tda_topology": ["tda_topology_", "tda_"],
    "gmm_clustering": ["gmm_clustering_", "gmm_"],
    "mamba_temporal": ["mamba_temporal_", "mamba_"],
    "economics": ["economics_", "econ_"],
    "multidisciplinary": ["multidisciplinary_", "multi_"],
    "model_derived": ["model_derived_", "hmm_summary_", "bandit_", "lnn_"],
    "merchant_hierarchy": ["merchant_hierarchy_", "mcc_", "brand_embed_", "merchant_"],
    "hmm_states": ["hmm_states_", "hmm_journey_", "hmm_lifecycle_", "hmm_behavior_", "hmm_"],
    "graph_embeddings": ["graph_embeddings_", "hyperbolic_", "hgcn_", "graph_"],
    # Santander-specific groups
    "demographics": ["age", "income", "tenure_months", "is_active", "num_products", "gender_", "segment_", "country_", "channel_", "age_group_", "income_group_"],
    "product_holdings": ["prod_"],
    "txn_behavior": ["synth_"],
    "derived_temporal": ["total_acquisitions", "total_churns", "months_observed", "product_diversity"],
    "tda_global": ["tda_global_"],
    "tda_local": ["tda_local_"],
    "product_hierarchy": ["product_hierarchy_", "ph_"],
    "graph_collaborative": ["graph_collaborative_", "gc_"],
}


def _columns_to_drop(
    all_columns: list,
    removed_groups: list,
) -> list:
    """Return column names that belong to the removed feature groups.

    Parameters
    ----------
    all_columns : list
        All columns in the DataFrame.
    removed_groups : list
        Feature group names to remove (e.g. ``["tda_topology", "economics"]``).

    Returns
    -------
    list
        Column names to drop.
    """
    drop_cols = []
    for group_name in removed_groups:
        prefixes = FEATURE_GROUP_COLUMN_PREFIXES.get(group_name, [group_name + "_"])
        for col in all_columns:
            if any(col.startswith(pfx) for pfx in prefixes):
                drop_cols.append(col)
    return drop_cols


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_feature_column_spec(
    config: Dict[str, Any],
    available_columns: list,
) -> Optional["FeatureColumnSpec"]:
    """Build a :class:`FeatureColumnSpec` from hyperparameter config.

    Returns ``None`` when the config does not contain feature-spec
    definitions, in which case the caller should raise an error.
    """
    feature_cfg = config.get("feature_columns")
    if not feature_cfg:
        return None

    from core.data.dataloader import FeatureColumnSpec, SequenceConfig

    return FeatureColumnSpec(
        static_features=feature_cfg.get("static_features", []),
        hyperbolic_columns=feature_cfg.get("hyperbolic_columns", []),
        tda_columns=feature_cfg.get("tda_columns", []),
        collaborative_columns=feature_cfg.get("collaborative_columns", []),
        hmm_journey_columns=feature_cfg.get("hmm_journey_columns", []),
        hmm_lifecycle_columns=feature_cfg.get("hmm_lifecycle_columns", []),
        hmm_behavior_columns=feature_cfg.get("hmm_behavior_columns", []),
        multidisciplinary_columns=feature_cfg.get("multidisciplinary_columns", []),
        coldstart_columns=feature_cfg.get("coldstart_columns", []),
        anonymous_columns=feature_cfg.get("anonymous_columns", []),
        event_seq_pattern=feature_cfg.get(
            "event_seq_pattern", "txn_card_{feature}_{step:03d}"
        ),
        session_seq_pattern=feature_cfg.get(
            "session_seq_pattern", "sess_{feature}_{step:03d}"
        ),
        event_seq_features=feature_cfg.get("event_seq_features", []),
        session_seq_features=feature_cfg.get("session_seq_features", []),
        event_time_delta_prefix=feature_cfg.get(
            "event_time_delta_prefix", "txn_card_time_delta"
        ),
        session_time_delta_prefix=feature_cfg.get(
            "session_time_delta_prefix", "sess_time_delta"
        ),
    )



def _derive_santander_labels(df, tasks):
    """Derive missing Santander labels from raw columns.

    Handles all 15 derived labels:
    - label_nba_primary: first element of nba_label list
    - label_acquire_*: product group acquisition from nba_label indices
    - label_tenure_stage, label_segment, label_income_tier: from profile columns
    - label_spend_level: from synth_monthly_spend
    - label_cross_sell_count: len(nba_label)
    - label_engagement_score: composite from synth columns
    - label_next_mcc, label_mcc_diversity_trend, label_top_mcc_shift: from txn sequences
    """
    from collections import Counter

    needed = {t["label_col"] for t in tasks if t["label_col"] not in df.columns}
    if not needed:
        return df

    logger.info("_derive_santander_labels: need to derive %d labels: %s",
                len(needed), sorted(needed))

    # Product index mapping (0-23 -> product names)
    PRODUCT_GROUP_INDICES = {
        "deposits": [8, 9, 10],          # short_deposit, medium_deposit, long_deposit
        "investments": [12, 18],          # funds, securities
        "accounts": [2, 6, 7, 11, 19, 5],  # checking, particular_acct, particular_plus, e_account, home_acct, junior_acct
        "lending": [13, 15],              # mortgage, loans
        "payments": [17, 22, 23, 20, 4],  # credit_card, direct_debit, auto_debit, payroll, payroll_acct
    }

    # Helper: safely get list column
    nba = df.get("nba_label") if "nba_label" in df.columns else None

    # --- label_nba_primary ---
    if "label_nba_primary" in needed and nba is not None:
        df["label_nba_primary"] = nba.apply(
            lambda x: int(x[0]) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else -1
        )
        logger.info("Derived label_nba_primary")

    # --- label_acquire_* (product group acquisition) ---
    group_label_map = {
        "label_acquire_deposits": "deposits",
        "label_acquire_investments": "investments",
        "label_acquire_accounts": "accounts",
        "label_acquire_lending": "lending",
        "label_acquire_payments": "payments",
    }
    for label_col, group_name in group_label_map.items():
        if label_col in needed and nba is not None:
            indices = set(PRODUCT_GROUP_INDICES[group_name])
            df[label_col] = nba.apply(
                lambda x, idx=indices: int(bool(set(int(i) for i in x) & idx)) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0
            )
            logger.info("Derived %s (indices=%s)", label_col, sorted(indices))

    # --- label_tenure_stage ---
    if "label_tenure_stage" in needed and "tenure_months" in df.columns:
        def _tenure_bin(v):
            if v == -999999 or (np.isnan(v) if isinstance(v, float) else False):
                return -1  # unknown → ignore_index
            v = int(v)
            if v <= 12:
                return 0  # new
            elif v <= 36:
                return 1  # growing
            elif v <= 72:
                return 2  # mature
            elif v <= 120:
                return 3
            else:
                return 4
        df["label_tenure_stage"] = df["tenure_months"].apply(_tenure_bin)
        logger.info("Derived label_tenure_stage (5 classes)")

    # --- label_segment ---
    if "label_segment" in needed and "segment" in df.columns:
        seg_map = {"01-TOP": 0, "02-PARTICULARES": 1, "03-UNIVERSITARIO": 2}
        # If segment is already numeric (label-encoded), copy directly
        if np.issubdtype(df["segment"].dtype, np.number):
            # Already encoded by categorical encoder — use as-is but cap at num_classes
            df["label_segment"] = df["segment"].clip(upper=3).astype(int)
        else:
            df["label_segment"] = df["segment"].apply(
                lambda x: seg_map.get(str(x), 3)
            )
        logger.info("Derived label_segment (4 classes)")

    # --- label_income_tier ---
    if "label_income_tier" in needed and "income" in df.columns:
        def _income_bin(v):
            if v <= 0 or (isinstance(v, float) and np.isnan(v)):
                return 0  # low / unknown
            elif v < 30000:
                return 0
            elif v < 80000:
                return 1
            elif v < 200000:
                return 2
            else:
                return 3
        df["label_income_tier"] = df["income"].apply(_income_bin)
        logger.info("Derived label_income_tier (4 classes)")

    # --- label_spend_level ---
    if "label_spend_level" in needed and "synth_monthly_spend" in df.columns:
        def _spend_bin(v):
            if isinstance(v, float) and np.isnan(v):
                return 0
            if v < 1500:
                return 0
            elif v < 3000:
                return 1
            elif v < 5000:
                return 2
            else:
                return 3
        df["label_spend_level"] = df["synth_monthly_spend"].apply(_spend_bin)
        logger.info("Derived label_spend_level (4 classes)")

    # --- label_cross_sell_count ---
    if "label_cross_sell_count" in needed:
        if nba is not None:
            df["label_cross_sell_count"] = nba.apply(
                lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
            )
        elif "total_acquisitions" in df.columns:
            df["label_cross_sell_count"] = df["total_acquisitions"]
        else:
            df["label_cross_sell_count"] = 0
        logger.info("Derived label_cross_sell_count")

    # --- label_engagement_score ---
    if "label_engagement_score" in needed:
        # Composite: is_active * 0.3 + synth_frequency * 0.4 + num_products * 0.3
        # Normalize each component to [0, 1] range
        score = np.zeros(len(df))
        for col_name, weight in [("is_active", 0.3), ("synth_frequency", 0.4), ("num_products", 0.3)]:
            if col_name in df.columns:
                vals = df[col_name].fillna(0).astype(float)
                max_val = vals.max()
                if max_val > 0:
                    vals = vals / max_val
                score = score + weight * vals.values
        df["label_engagement_score"] = score
        logger.info("Derived label_engagement_score")

    # --- label_next_mcc ---
    if "label_next_mcc" in needed and "txn_mcc_seq" in df.columns:
        df["label_next_mcc"] = df["txn_mcc_seq"].apply(
            lambda x: min(int(x[-1]), 49) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else -1
        )
        logger.info("Derived label_next_mcc (capped at 50)")

    # --- label_mcc_diversity_trend ---
    if "label_mcc_diversity_trend" in needed and "txn_mcc_seq" in df.columns:
        def _diversity_trend(x):
            if not isinstance(x, (list, np.ndarray)) or len(x) < 4:
                return 0.0
            mid = len(x) // 2
            first_half_unique = len(set(x[:mid]))
            second_half_unique = len(set(x[mid:]))
            if first_half_unique == 0:
                return 0.0
            return (second_half_unique / first_half_unique) - 1.0
        df["label_mcc_diversity_trend"] = df["txn_mcc_seq"].apply(_diversity_trend)
        logger.info("Derived label_mcc_diversity_trend")

    # --- label_top_mcc_shift ---
    if "label_top_mcc_shift" in needed and "txn_mcc_seq" in df.columns:
        def _mode_shift(x):
            if not isinstance(x, (list, np.ndarray)) or len(x) < 4:
                return -1  # too short to judge shift → ignore
            mid = len(x) // 2
            mode_first = Counter(x[:mid]).most_common(1)[0][0]
            mode_second = Counter(x[mid:]).most_common(1)[0][0]
            return int(mode_first != mode_second)
        df["label_top_mcc_shift"] = df["txn_mcc_seq"].apply(_mode_shift)
        logger.info("Derived label_top_mcc_shift")

    # Report final status
    still_missing = [t["label_col"] for t in tasks if t["label_col"] not in df.columns]
    if still_missing:
        logger.warning("_derive_santander_labels: still missing %d labels: %s",
                        len(still_missing), still_missing)
    else:
        logger.info("_derive_santander_labels: all %d labels available", len(needed))

    return df


def _inject_sequences_into_ple_dataset(
    dataset: Any,
    event_sequences: torch.Tensor,
    seq_lengths: Optional[torch.Tensor] = None,
) -> None:
    """Inject externally-loaded event sequences into an existing PLEDataset.

    This patches the dataset's internal tensor dict so that ``__getitem__``
    returns ``event_sequences`` (and optionally ``seq_lengths``) alongside
    the features already extracted from the DataFrame.
    """
    if hasattr(dataset, "_tensors"):
        # CPU path: inject directly into the pre-converted tensor dict
        dataset._tensors["event_sequences"] = event_sequences
        if seq_lengths is not None:
            dataset._tensors["seq_lengths"] = seq_lengths
        logger.info(
            "Injected event_sequences %s into PLEDataset tensors",
            list(event_sequences.shape),
        )
    else:
        logger.warning(
            "Cannot inject event sequences: PLEDataset does not have _tensors "
            "(GPU/cuDF path not supported for npy injection yet)"
        )


def load_data(
    channel_dir: str,
    config: Dict[str, Any],
    *,
    use_gpu_loading: bool = False,
    batch_size: int = 2048,
    shuffle: bool = True,
    removed_feature_groups: Optional[list] = None,
) -> Any:
    """Load Parquet files from a SageMaker input channel.

    Returns a PyTorch ``DataLoader`` backed by :class:`PLEDataset` with
    optional cuDF GPU zero-copy support.

    Parameters
    ----------
    channel_dir : str
        Directory containing Parquet files (e.g. /opt/ml/input/data/train).
    config : dict
        Pipeline config dict with task/feature definitions.
    use_gpu_loading : bool
        Enable cuDF DLPack zero-copy loading when available.
    batch_size : int
        Batch size for the DataLoader.
    shuffle : bool
        Whether to shuffle the DataLoader.

    Returns
    -------
    DataLoader
        A DataLoader yielding PLE-compatible dicts.
    """
    import pandas as pd

    channel_path = Path(channel_dir)
    parquet_files = sorted(channel_path.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in {channel_dir}"
        )

    logger.info(f"Loading {len(parquet_files)} Parquet file(s) from {channel_dir}")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    # Subsample if max_rows HP is set (for fast testing)
    max_rows = config.get("max_rows", 0)
    if max_rows and max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        logger.info(f"Subsampled to {max_rows} rows for fast testing")
    # -- Handle sentinel/missing values BEFORE fillna --
    import numpy as _np
    # tenure_months: -999999 is sentinel for unknown
    if "tenure_months" in df.columns:
        df.loc[df["tenure_months"] <= -999, "tenure_months"] = _np.nan
    # income: 0 means missing (25% of data)
    if "income" in df.columns:
        df.loc[df["income"] == 0, "income"] = _np.nan
        # Impute with median of non-missing values
        _inc_median = df["income"].median()
        df["income"] = df["income"].fillna(_inc_median)
        logger.info("Income: imputed %d missing values with median=%.0f",
                     df["income"].isna().sum(), _inc_median if _inc_median == _inc_median else 0)
    # fillna(0) for remaining numeric columns
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # -- Load event sequence .npy files if present --
    event_sequences_tensor = None
    seq_lengths_tensor = None
    seq_config = config.get("features", {}).get("event_sequences", {})
    if seq_config.get("enabled", False):
        seq_pattern = seq_config.get("file_pattern", "*_event_sequences.npy")
        lengths_pattern = seq_config.get("lengths_file_pattern", "*_seq_lengths.npy")
        seq_files = sorted(channel_path.glob(f"**/{seq_pattern}"))
        lengths_files = sorted(channel_path.glob(f"**/{lengths_pattern}"))

        if seq_files:
            seq_arr = np.load(str(seq_files[0]))
            logger.info(
                "Loaded event sequences: %s from %s",
                seq_arr.shape, seq_files[0].name,
            )
            event_sequences_tensor = torch.tensor(seq_arr, dtype=torch.float32)
        else:
            logger.warning(
                "Event sequences enabled but no .npy file matching '%s' found in %s",
                seq_pattern, channel_dir,
            )

        if lengths_files:
            lengths_arr = np.load(str(lengths_files[0]))
            seq_lengths_tensor = torch.tensor(lengths_arr, dtype=torch.long)
            logger.info(
                "Loaded sequence lengths: %s from %s",
                lengths_arr.shape, lengths_files[0].name,
            )

    # -- Ablation: drop columns belonging to removed feature groups --
    if removed_feature_groups:
        drop_cols = _columns_to_drop(list(df.columns), removed_feature_groups)
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
            logger.info(
                "Ablation: removed %d columns from groups %s. "
                "Remaining: %d columns",
                len(drop_cols), removed_feature_groups, len(df.columns),
            )
        else:
            logger.warning(
                "Ablation: no columns matched for groups %s",
                removed_feature_groups,
            )

    tasks = config.get("tasks", [])
    label_cols = [t["label_col"] for t in tasks]

    # -- Encode categorical string columns to integers --
    # This makes gender, segment, country, channel, age_group, income_group
    # available as numeric features for auto-discovery.
    from sklearn.preprocessing import LabelEncoder as _LabelEncoder
    _cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    _id_and_date_cols = {"customer_id", "snapshot_date"}
    for _col in _cat_cols:
        if _col not in _id_and_date_cols:
            # Skip list-type columns stored as object dtype
            _sample = df[_col].dropna().iloc[0] if len(df[_col].dropna()) > 0 else None
            if isinstance(_sample, (list, np.ndarray)):
                continue
            _le = _LabelEncoder()
            df[_col] = _le.fit_transform(df[_col].astype(str))
            logger.info("Encoded categorical '%s': %d classes", _col, len(_le.classes_))

    # -- Run feature generators if Phase 0 didn't produce them --
    _has_generated = any(c.startswith("tda_") or c.startswith("graph_") or
                         c.startswith("hmm_") or c.startswith("mamba_")
                         for c in df.columns)
    if not _has_generated:
        # Skip inline generators if data is large (>10K rows) to avoid OOM
        # Generators should run in Phase 0 for large datasets
        if len(df) <= 10000:
            logger.info("Small dataset — running inline feature generation")
            try:
                from adapters.santander_adapter import _run_santander_generators
                df = _run_santander_generators(df)
                logger.info("Inline feature generation complete. Shape: %s", df.shape)
            except Exception as _gen_err:
                logger.warning("Inline feature generation failed: %s", _gen_err)
        else:
            logger.info("Large dataset (%d rows) — skipping inline generators (run Phase 0 instead)", len(df))
    else:
        _gen_cols = [c for c in df.columns
                     if any(c.startswith(p) for p in
                            ("tda_", "graph_", "hmm_", "mamba_", "gmm_", "model_derived_"))]
        logger.info("Found %d pre-generated feature columns from Phase 0", len(_gen_cols))

    # -- Derive missing labels if LabelDeriver config is present --
    missing_labels = [lc for lc in label_cols if lc not in df.columns]
    if missing_labels:
        logger.info("Missing %d label columns, attempting derivation: %s",
                     len(missing_labels), missing_labels)
        try:
            from core.pipeline.label_deriver import LabelDeriver, LabelConfig
            labels_cfg = config.get("labels", {})
            if labels_cfg:
                deriver = LabelDeriver()
                label_configs = []
                for name, cfg in labels_cfg.items():
                    if isinstance(cfg, dict):
                        label_configs.append(LabelConfig(name=name, **{
                            k: v for k, v in cfg.items()
                            if k in LabelConfig.__dataclass_fields__
                        }))
                derived = deriver.derive(df, label_configs)
                for col in derived.columns:
                    df[col] = derived[col]
                logger.info("Derived %d label columns", len(derived.columns))
            else:
                logger.warning("No labels config found for derivation")
        except Exception as e:
            logger.warning("Label derivation failed: %s", e)

    # -- Santander-specific label derivation (handles all 15 derived labels) --
    missing_after_deriver = [lc for lc in label_cols if lc not in df.columns]
    if missing_after_deriver:
        logger.info("Attempting Santander-specific derivation for %d labels: %s",
                     len(missing_after_deriver), missing_after_deriver)
        df = _derive_santander_labels(df, tasks)

    # Re-check after derivation — skip tasks whose labels still don't exist
    # Also skip list-type columns (can't be converted to tensors)
    available_labels = set(df.columns)
    valid_tasks = []
    for t in tasks:
        lc = t["label_col"]
        if lc not in available_labels:
            logger.warning("Skipping task %s: label_col '%s' not in data", t["name"], lc)
            continue
        # Check if label is a scalar type (not list/object)
        dtype = df[lc].dtype
        if dtype == object or str(dtype).startswith("list"):
            logger.warning("Skipping task %s: label_col '%s' is non-scalar (dtype=%s)", t["name"], lc, dtype)
            continue
        valid_tasks.append(t)
    tasks = valid_tasks
    label_cols = [t["label_col"] for t in tasks]
    logger.info("Valid tasks after label check: %d/%d", len(tasks), len(config.get("tasks", [])))
    if not tasks:
        raise ValueError("No tasks have valid label columns in the data.")

    # ---- PLEDataset path ----
    feature_spec = _parse_feature_column_spec(config, list(df.columns))
    if feature_spec is None:
        # Auto-discover: all non-label, non-id columns become static features
        from core.data.dataloader import FeatureColumnSpec
        id_cols = config.get("id_cols", ["user_id", "customer_id"])
        exclude = set(label_cols) | set(id_cols)
        # Only include numeric scalar columns (exclude strings, dates, list types)
        static_features = []
        for c in df.columns:
            if c in exclude:
                continue
            dtype = df[c].dtype
            # Skip string/object/categorical columns
            if dtype == object or str(dtype) in ("string", "category"):
                continue
            # Skip list/array columns (pyarrow list types show as object or 'list')
            if hasattr(dtype, "pyarrow_dtype"):
                continue
            # Only keep numeric types
            try:
                import numpy as _np
                if not _np.issubdtype(dtype, _np.number):
                    continue
            except (TypeError, AttributeError):
                continue
            static_features.append(c)
        feature_spec = FeatureColumnSpec(static_features=static_features)
        logger.info(
            "Auto-discovered %d static features (no feature_columns config)",
            len(static_features),
        )

    # -- Normalize features + regression labels to prevent NaN loss --
    import numpy as _np
    from sklearn.preprocessing import StandardScaler as _StdScaler

    # 1) Feature normalization (StandardScaler on continuous features, skip binary)
    _feat_cols = [c for c in (feature_spec.static_features if feature_spec else [])
                  if c in df.columns and _np.issubdtype(df[c].dtype, _np.number)]
    # Separate binary (0/1 only) from continuous for appropriate normalization
    _binary_cols = [c for c in _feat_cols if set(df[c].dropna().unique()).issubset({0, 0.0, 1, 1.0})]
    _continuous_cols = [c for c in _feat_cols if c not in _binary_cols]
    global _module_scaler, _module_continuous_cols
    _scaler = None
    if _continuous_cols:
        _scaler = _StdScaler()
        df[_continuous_cols] = _scaler.fit_transform(df[_continuous_cols].fillna(0).values)
        logger.info("StandardScaler applied to %d continuous features (skipped %d binary)",
                     len(_continuous_cols), len(_binary_cols))
        # Store in module-level state for eval report
        _module_scaler = _scaler
        _module_continuous_cols = _continuous_cols

        # HIGH-3: Persist scaler parameters for reproducibility
        scaler_params = {
            "columns": _continuous_cols,
            "mean": _scaler.mean_.tolist(),
            "scale": _scaler.scale_.tolist(),
        }
        _scaler_path = os.path.join(
            os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
            "scaler_params.json",
        )
        os.makedirs(os.path.dirname(_scaler_path), exist_ok=True)
        with open(_scaler_path, "w") as f:
            json.dump(scaler_params, f)
        logger.info("Scaler params saved to %s", _scaler_path)

    # 2) Regression label normalization (log1p + clip outliers)
    for t in tasks:
        if t.get("type") == "regression":
            lc = t["label_col"]
            if lc in df.columns:
                vals = df[lc].fillna(0).astype(float)
                # Clip outliers at 99.5th percentile
                clip_val = vals.quantile(0.995)
                if clip_val > 0:
                    vals = vals.clip(upper=clip_val)
                # log1p for positive-valued targets
                if vals.min() >= 0:
                    vals = _np.log1p(vals)
                df[lc] = vals
                logger.info("Regression label '%s' normalized: clip=%.2f, log1p=%s",
                            lc, clip_val, vals.min() >= 0)

    from core.data.dataloader import build_ple_dataloader, SequenceConfig

    label_map = {t["name"]: t["label_col"] for t in tasks}

    seq_cfg_raw = config.get("sequence_config", {})
    seq_cfg = SequenceConfig(**seq_cfg_raw) if seq_cfg_raw else None

    loader = build_ple_dataloader(
        df=df,
        feature_spec=feature_spec,
        label_columns=label_map,
        batch_size=batch_size,
        shuffle=shuffle,
        use_gpu_loading=use_gpu_loading,
        sequence_config=seq_cfg,
    )

    # Inject externally-loaded event sequences into the PLEDataset
    if event_sequences_tensor is not None:
        _inject_sequences_into_ple_dataset(
            loader.dataset, event_sequences_tensor, seq_lengths_tensor,
        )

    # Pre-compute label statistics before df goes out of scope
    _label_stats = {}
    for t in tasks:
        lc = t["label_col"]
        if lc in df.columns:
            col = df[lc]
            if t.get("type") == "binary":
                n_pos = int((col > 0.5).sum())
                _label_stats[t["name"]] = {
                    "positive_count": n_pos,
                    "positive_rate": round(n_pos / len(col), 4),
                    "total": len(col),
                }
            elif t.get("type") == "regression":
                _label_stats[t["name"]] = {
                    "mean": round(float(col.mean()), 4),
                    "std": round(float(col.std()), 4),
                    "total": len(col),
                }
            elif t.get("type") == "multiclass":
                _label_stats[t["name"]] = {
                    "num_classes": int(col.nunique()),
                    "total": len(col),
                }

    logger.info(
        "Using PLEDataset dataloader: %d samples, batch_size=%d",
        len(df), batch_size,
    )
    return loader, tasks, _label_stats


# ---------------------------------------------------------------------------
# Metric printing (captured by SageMaker → CloudWatch)
# ---------------------------------------------------------------------------

def report_metrics(prefix: str, metrics: Dict[str, float], epoch: int) -> None:
    """Print metrics in the format SageMaker expects for CloudWatch.

    Each metric is printed as ``<name>=<value>`` on the same line.
    The METRIC_DEFINITIONS regex in the trainer picks these up.
    """
    parts = [f"epoch={epoch}"]
    for name, value in sorted(metrics.items()):
        if isinstance(value, (int, float)):
            parts.append(f"{prefix}_{name}={value:.6f}")
        else:
            parts.append(f"{prefix}_{name}={value}")
    line = " ".join(parts)
    logger.info(line)
    # Also print to stdout directly for SageMaker metric capture
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _batch_to_ple_input(
    batch: Any,
    task_names: list,
    device: torch.device,
) -> "PLEInput":
    """Convert a batch (dict or PLEInput) into a device-resident PLEInput.

    Supports two batch formats:
      - ``dict`` from :class:`PLEDataset` / :func:`build_ple_dataloader`
      - ``PLEInput`` directly (e.g. when ``return_ple_input=True``)
    """
    from core.model.ple.model import PLEInput

    if isinstance(batch, PLEInput):
        return batch.to(device)

    if not isinstance(batch, dict):
        raise TypeError(
            f"Expected batch to be a dict or PLEInput, got {type(batch).__name__}"
        )

    kwargs: Dict[str, Any] = {
        "features": batch["features"],
    }
    if "targets" in batch:
        kwargs["targets"] = batch["targets"]

    # Map short collate keys to PLEInput field names
    _KEY_MAP = {
        "hyperbolic": "hyperbolic_features",
        "tda": "tda_features",
        "collaborative": "collaborative_features",
        "hmm_journey": "hmm_journey",
        "hmm_lifecycle": "hmm_lifecycle",
        "hmm_behavior": "hmm_behavior",
        "event_sequences": "event_sequences",
        "session_sequences": "session_sequences",
        "event_time_delta": "event_time_delta",
        "session_time_delta": "session_time_delta",
        "multidisciplinary": "multidisciplinary_features",
        "coldstart": "coldstart_features",
        "anonymous": "anonymous_features",
    }
    for src, dst in _KEY_MAP.items():
        val = batch.get(src)
        if val is not None:
            kwargs[dst] = val

    return PLEInput(**kwargs).to(device)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_names: list[str],
    epoch: int,
) -> Dict[str, float]:
    """Run one training epoch.

    Returns
    -------
    dict[str, float]
        Training metrics (loss, per-task losses).
    """
    from core.model.ple.model import PLEInput

    model.train()
    total_loss = 0.0
    task_losses_sum: Dict[str, float] = {name: 0.0 for name in task_names}
    n_batches = 0

    for batch in dataloader:
        inputs = _batch_to_ple_input(batch, task_names, device)
        output = model(inputs, compute_loss=True)

        if output.total_loss is None:
            continue

        optimizer.zero_grad()
        output.total_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += output.total_loss.item()
        if output.task_losses:
            for name, loss_val in output.task_losses.items():
                task_losses_sum[name] += loss_val.item()
        n_batches += 1

        # Update global step
        model.set_global_step(model.global_step + 1)

    if n_batches == 0:
        return {"loss": 0.0}

    metrics = {"loss": total_loss / n_batches}
    for name in task_names:
        metrics[f"loss_{name}"] = task_losses_sum[name] / n_batches
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    task_names: list[str],
    task_type_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Run validation and compute metrics.

    Parameters
    ----------
    task_type_map : dict, optional
        Maps task name to task type (``"binary"``, ``"multiclass"``,
        ``"regression"``).  When provided, uses the declared type instead
        of heuristic label counting.

    Returns
    -------
    dict[str, float]
        Validation metrics: loss, per-task AUC/accuracy/f1/MAE.
    """
    from core.model.ple.model import PLEInput
    from sklearn.metrics import (
        roc_auc_score,
        accuracy_score,
        f1_score,
        confusion_matrix as _confusion_matrix,
    )

    if task_type_map is None:
        task_type_map = {}

    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds: Dict[str, list] = {name: [] for name in task_names}
    all_targets: Dict[str, list] = {name: [] for name in task_names}

    for batch in dataloader:
        inputs = _batch_to_ple_input(batch, task_names, device)
        targets = inputs.targets or {}
        output = model(inputs, compute_loss=True)

        if output.total_loss is not None:
            total_loss += output.total_loss.item()
            n_batches += 1

        for name in task_names:
            if name in output.predictions and name in targets:
                pred = output.predictions[name].cpu().numpy()
                tgt = targets[name].cpu().numpy()
                all_preds[name].append(pred)
                all_targets[name].append(tgt)

    if n_batches == 0:
        return {"loss": 0.0}

    metrics: Dict[str, float] = {"loss": total_loss / n_batches}

    # Per-task metrics
    for name in task_names:
        if not all_preds[name]:
            continue

        preds = np.concatenate(all_preds[name])
        tgts = np.concatenate(all_targets[name])

        # Determine task type: use declared type if available, else heuristic
        task_type = task_type_map.get(name)

        if task_type == "binary":
            preds_sq = preds.squeeze()
            tgts_sq = tgts.squeeze()
            unique_labels = np.unique(tgts_sq)
            if len(unique_labels) >= 2:
                try:
                    metrics[f"auc_{name}"] = roc_auc_score(tgts_sq, preds_sq)
                except ValueError:
                    pass
            try:
                pred_labels = (preds_sq > 0.5).astype(int)
                metrics[f"accuracy_{name}"] = accuracy_score(tgts_sq, pred_labels)
                metrics[f"f1_{name}"] = f1_score(
                    tgts_sq, pred_labels, zero_division=0,
                )
                # HIGH-5: Confusion matrix for binary tasks
                cm = _confusion_matrix(tgts_sq, pred_labels, labels=[0, 1]).tolist()
                metrics[f"confusion_matrix_{name}"] = cm
            except Exception:
                pass

        elif task_type == "multiclass":
            # preds shape: (N, num_classes) -- take argmax for class prediction
            if preds.ndim == 2:
                pred_classes = np.argmax(preds, axis=1)
            else:
                pred_classes = np.round(preds).astype(int)
            tgts_int = tgts.astype(int).squeeze()

            try:
                metrics[f"accuracy_{name}"] = accuracy_score(tgts_int, pred_classes)
            except Exception:
                pass
            try:
                metrics[f"f1_macro_{name}"] = f1_score(
                    tgts_int, pred_classes, average="macro", zero_division=0,
                )
            except Exception:
                pass
            try:
                metrics[f"f1_weighted_{name}"] = f1_score(
                    tgts_int, pred_classes, average="weighted", zero_division=0,
                )
            except Exception:
                pass
            # HIGH-5: Confusion matrix for multiclass tasks
            try:
                valid_mask = tgts_int >= 0
                labels = sorted(set(tgts_int[valid_mask]))
                cm = _confusion_matrix(
                    tgts_int[valid_mask], pred_classes[valid_mask], labels=labels,
                ).tolist()
                metrics[f"confusion_matrix_{name}"] = cm
            except Exception:
                pass

        elif task_type == "regression":
            preds_sq = preds.squeeze()
            tgts_sq = tgts.squeeze()
            try:
                metrics[f"mae_{name}"] = float(np.mean(np.abs(preds_sq - tgts_sq)))
            except Exception:
                pass
            try:
                metrics[f"rmse_{name}"] = float(
                    np.sqrt(np.mean((preds_sq - tgts_sq) ** 2))
                )
            except Exception:
                pass

        else:
            # Fallback: heuristic-based metric selection (legacy path)
            preds_sq = preds.squeeze()
            tgts_sq = tgts.squeeze()
            unique_labels = np.unique(tgts_sq)
            if len(unique_labels) == 2:
                try:
                    metrics[f"auc_{name}"] = roc_auc_score(tgts_sq, preds_sq)
                except ValueError:
                    pass
            try:
                metrics[f"mae_{name}"] = float(np.mean(np.abs(preds_sq - tgts_sq)))
            except Exception:
                pass

    # Aggregate AUC (across binary tasks)
    auc_values = [v for k, v in metrics.items() if k.startswith("auc_")]
    if auc_values:
        metrics["auc"] = float(np.mean(auc_values))

    # Aggregate multiclass metrics
    f1_macro_values = [v for k, v in metrics.items() if k.startswith("f1_macro_")]
    if f1_macro_values:
        metrics["f1_macro_avg"] = float(np.mean(f1_macro_values))

    return metrics


# ---------------------------------------------------------------------------
# Phase control
# ---------------------------------------------------------------------------

def apply_phase_config(
    model: nn.Module,
    phase: str,
    pretrained_uri: Optional[str] = None,
) -> None:
    """Configure model for the specified training phase.

    Parameters
    ----------
    model : nn.Module
        PLE model.
    phase : str
        ``"1"`` to freeze towers, ``"2"`` to unfreeze all.
    pretrained_uri : str, optional
        S3 URI or local path for pre-trained weights (phase 2).
    """
    if phase == "1":
        # Phase 1: freeze task towers, train shared experts + gating
        logger.info("Phase 1: Freezing task towers")
        for name, param in model.named_parameters():
            if "task_towers" in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"  Trainable: {trainable:,} / {total:,} parameters")

    elif phase == "2":
        # Phase 2: unfreeze everything
        logger.info("Phase 2: Unfreezing all parameters")
        for param in model.parameters():
            param.requires_grad = True

        # Load phase 1 weights if available
        if pretrained_uri:
            logger.info(f"Loading pre-trained weights from {pretrained_uri}")
            if pretrained_uri.startswith("s3://"):
                _download_pretrained_from_s3(pretrained_uri)
                local_path = "/tmp/pretrained/model.pth"
            else:
                local_path = pretrained_uri

            if os.path.exists(local_path):
                state = torch.load(local_path, map_location="cpu", weights_only=False)
                if "model_state_dict" in state:
                    model.load_state_dict(state["model_state_dict"], strict=False)
                else:
                    model.load_state_dict(state, strict=False)
                logger.info("Pre-trained weights loaded successfully")
    else:
        logger.info(f"Single-phase training (phase='{phase}')")


def _download_pretrained_from_s3(s3_uri: str) -> None:
    """Download pre-trained model from S3."""
    import boto3
    import tarfile

    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]

    local_tar = "/tmp/pretrained_model.tar.gz"
    local_dir = "/tmp/pretrained/"

    os.makedirs(local_dir, exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_tar)

    with tarfile.open(local_tar, "r:gz") as tar:
        try:
            tar.extractall(local_dir, filter="data")
        except TypeError:
            tar.extractall(local_dir)

    logger.info(f"Extracted pre-trained model to {local_dir}")


# ---------------------------------------------------------------------------
# PipelineRunner entry point (Stage 8 integration)
# ---------------------------------------------------------------------------

def main_pipeline(config_path: str, **overrides) -> dict:
    """Entry point when called via PipelineRunner.

    Runs the full 9-stage pipeline (data loading through training) using
    :class:`PipelineRunner`.  This path is activated by passing
    ``--pipeline <config.yaml>`` on the command line.

    Parameters
    ----------
    config_path : str
        Path to pipeline YAML config.
    **overrides
        Hyperparameter overrides from SageMaker (merged into config).

    Returns
    -------
    dict
        Training results from PipelineRunner.run().
    """
    from core.pipeline.config import load_config
    from core.pipeline.runner import PipelineRunner

    config = load_config(config_path)

    # Apply HP overrides from SageMaker environment as attributes
    sm_hps = get_hyperparameters()
    if sm_hps:
        for k, v in sm_hps.items():
            if hasattr(config.training, k):
                setattr(config.training, k, type(getattr(config.training, k))(v))
    if overrides:
        for k, v in overrides.items():
            if hasattr(config.training, k):
                setattr(config.training, k, type(getattr(config.training, k))(v))

    output_dir = os.environ.get("SM_MODEL_DIR", "outputs/")

    runner = PipelineRunner(config)
    results = runner.run(output_dir=output_dir)

    logger.info("PipelineRunner completed. Results: %s", list(results.keys()))
    return results


# ---------------------------------------------------------------------------
# Main (legacy SageMaker Training Job entry point)
# ---------------------------------------------------------------------------

def main() -> None:
    """SageMaker training entry point."""
    start_time = time.time()

    # -- Environment --
    sm_env = get_sm_env()
    hp = get_hyperparameters()

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    # SageMaker channel name can be "train" or "training"
    train_dir = os.environ.get(
        "SM_CHANNEL_TRAIN",
        os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
    )
    val_dir = os.environ.get(
        "SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation",
    )
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    checkpoint_dir = os.environ.get(
        "SM_CHECKPOINT_DIR", "/opt/ml/checkpoints",
    )
    num_gpus = int(os.environ.get("SM_NUM_GPUS", "0"))

    logger.info("=" * 60)
    logger.info("SageMaker PLE Training Entry Point")
    logger.info("=" * 60)
    logger.info(f"GPUs available: {num_gpus}")
    logger.info(f"Model dir: {model_dir}")
    logger.info(f"Train dir: {train_dir}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")

    # -- Parse config from hyperparameters --
    # Supports three formats:
    #   1. JSON string (inline config via hyperparameters)
    #   2. YAML file path (relative to source_dir, e.g. "configs/test/xxx.yaml")
    #   3. Dict (already parsed)
    config_str = hp.get("config", "{}")
    if isinstance(config_str, dict):
        config = config_str
    elif isinstance(config_str, str) and (
        config_str.endswith(".yaml") or config_str.endswith(".yml")
    ):
        import yaml
        config_path = Path(config_str)
        if not config_path.exists():
            # SageMaker copies source_dir to /opt/ml/code/
            config_path = Path("/opt/ml/code") / config_str
        if not config_path.exists():
            code_dir = os.environ.get("SM_MODULE_DIR", "/opt/ml/code")
            config_path = Path(code_dir).parent / config_str
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            logger.info("Config loaded from YAML: %s", config_path)
        else:
            logger.error("Config YAML not found: %s", config_str)
            config = {}
    else:
        config = json.loads(config_str) if config_str else {}

    task_name = hp.get("task_name", config.get("task_name", "default"))
    batch_size = int(hp.get("batch_size", 2048))
    epochs = int(hp.get("epochs", 20))
    lr = float(hp.get("learning_rate", 1e-3))
    patience = int(hp.get("early_stopping_patience", 5))
    seed = int(hp.get("seed", 42))
    phase = str(hp.get("phase", "single"))
    use_amp = str(hp.get("amp", "false")).lower() in ("true", "1", "yes")
    grad_accum_steps = int(hp.get("gradient_accumulation_steps", 4))
    freeze_towers = hp.get("freeze_towers", False)
    pretrained_uri = hp.get("pretrained_model_uri")

    # -- Ablation hyperparameters --
    ablation_type = hp.get("ablation_type", "")
    ablation_scenario = hp.get("ablation_scenario", "")

    # Feature group ablation: JSON list of group names to remove
    removed_feature_groups_raw = hp.get("removed_feature_groups", "[]")
    if isinstance(removed_feature_groups_raw, str):
        try:
            removed_feature_groups = json.loads(removed_feature_groups_raw)
        except (json.JSONDecodeError, ValueError):
            removed_feature_groups = []
    elif isinstance(removed_feature_groups_raw, list):
        removed_feature_groups = removed_feature_groups_raw
    else:
        removed_feature_groups = []

    # Expert ablation: JSON list of shared expert names
    shared_experts_raw = hp.get("shared_experts", "")
    if isinstance(shared_experts_raw, str) and shared_experts_raw:
        try:
            shared_experts_override = json.loads(shared_experts_raw)
        except (json.JSONDecodeError, ValueError):
            shared_experts_override = None
    elif isinstance(shared_experts_raw, list):
        shared_experts_override = shared_experts_raw
    else:
        shared_experts_override = None

    # Parse active_tasks HP for task scaling ablation
    active_tasks_raw = hp.get("active_tasks")
    active_task_names = None
    if active_tasks_raw:
        active_task_names = json.loads(active_tasks_raw) if isinstance(active_tasks_raw, str) else active_tasks_raw
        # Filter config tasks to only active ones
        all_tasks = config.get("tasks", [])
        config["tasks"] = [t for t in all_tasks if t["name"] in active_task_names]
        logger.info("Task scaling ablation: %d/%d tasks active: %s",
                    len(config["tasks"]), len(all_tasks), active_task_names)

        # Filter task_groups to only reference active tasks
        for tg in config.get("task_groups", []):
            tg["tasks"] = [t for t in tg["tasks"] if t in active_task_names]
        config["task_groups"] = [tg for tg in config.get("task_groups", []) if tg.get("tasks")]
        logger.info("Task groups after filtering: %d", len(config.get("task_groups", [])))

        # Filter task_relationships to only reference active tasks
        config["task_relationships"] = [
            tr for tr in config.get("task_relationships", [])
            if tr["source"] in active_task_names and tr["target"] in active_task_names
        ]
        logger.info("Task relationships after filtering: %d", len(config.get("task_relationships", [])))

    # Hyperparameter ablation: num_layers and temperature
    hp_num_layers = hp.get("num_layers")
    hp_temperature = hp.get("temperature")

    # Inject max_rows HP into config for load_data()
    max_rows_hp = hp.get("max_rows", 0)
    if max_rows_hp:
        config["max_rows"] = int(max_rows_hp)

    logger.info(f"Task: {task_name}")
    logger.info(f"Phase: {phase}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    if max_rows_hp:
        logger.info(f"Max rows (subsample): {max_rows_hp}")
    if ablation_type:
        logger.info(f"Ablation type: {ablation_type}, scenario: {ablation_scenario}")
    if removed_feature_groups:
        logger.info(f"Removed feature groups: {removed_feature_groups}")
    if shared_experts_override is not None:
        logger.info(f"Shared experts override: {shared_experts_override}")
    if hp_num_layers is not None:
        logger.info(f"PLE num_layers override: {hp_num_layers}")
    if hp_temperature is not None:
        logger.info(f"Distillation temperature override: {hp_temperature}")

    # -- Seed --
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -- Device --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # -- Load data --
    use_gpu_loading = hp.get("use_gpu_loading", False) and num_gpus > 0
    train_data, tasks, _label_stats = load_data(
        train_dir, config,
        use_gpu_loading=use_gpu_loading,
        batch_size=batch_size,
        shuffle=True,
        removed_feature_groups=removed_feature_groups or None,
    )

    # --- Label sanity check ---
    try:
        _peek = next(iter(train_data))
        if _peek is not None:
            _targets = _peek.get("targets", {}) if isinstance(_peek, dict) else {}
            for _tn, _tv in _targets.items():
                if isinstance(_tv, torch.Tensor):
                    logger.info(
                        "Label check [%s]: shape=%s, dtype=%s, min=%.4f, max=%.4f, nan=%d",
                        _tn, _tv.shape, _tv.dtype, _tv.min().item(), _tv.max().item(),
                        _tv.isnan().sum().item(),
                    )
    except Exception as _e:
        logger.warning("Label check skipped: %s", _e)

    # load_data always returns a DataLoader (PLEDataset path)
    train_loader = train_data
    logger.info(f"Train dataloader: {len(train_loader.dataset)} samples")

    val_loader = None
    if os.path.isdir(val_dir):
        try:
            val_loader, _, _ = load_data(
                val_dir, config,
                use_gpu_loading=use_gpu_loading,
                batch_size=batch_size,
                shuffle=False,
                removed_feature_groups=removed_feature_groups or None,
            )
        except FileNotFoundError:
            logger.warning("No validation data found, using train data split")

    if val_loader is None:
        # Fall back to splitting the training dataset
        n = len(train_loader.dataset)
        val_size = max(1, int(n * 0.1))
        train_size = n - val_size
        gen = torch.Generator().manual_seed(config.get("training", {}).get("seed", 42))
        train_subset, val_subset = torch.utils.data.random_split(
            train_loader.dataset, [train_size, val_size], generator=gen,
        )
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            collate_fn=train_loader.collate_fn,
            num_workers=0,
            pin_memory=not use_gpu_loading,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            collate_fn=train_loader.collate_fn,
            num_workers=0,
            pin_memory=not use_gpu_loading,
        )

    # -- Build model --
    from core.model.ple.model import PLEModel
    from core.model.ple.config import PLEConfig

    model_config = config.get("model", {})

    # Structure ablation: PLE toggle
    use_ple_raw = hp.get("use_ple")
    if use_ple_raw is not None:
        use_ple = json.loads(use_ple_raw) if isinstance(use_ple_raw, str) else use_ple_raw
        if not use_ple:
            # Disable CGC gating — use single shared expert (shared-bottom baseline)
            model_config["ple"] = model_config.get("ple", {})
            model_config["ple"]["num_layers"] = 1
            model_config["ple"]["num_shared_experts"] = 1
            model_config.pop("expert_basket", None)  # Remove expert basket → MLP only
            logger.info("Structure ablation: PLE disabled (shared-bottom mode)")

    # Structure ablation: adaTT toggle
    use_adatt_raw = hp.get("use_adatt")
    if use_adatt_raw is not None:
        use_adatt = json.loads(use_adatt_raw) if isinstance(use_adatt_raw, str) else use_adatt_raw
        if not use_adatt:
            model_config["adatt"] = {"enabled": False}
            logger.info("Structure ablation: adaTT disabled")

    tasks = config.get("tasks", [])
    task_names = [t["name"] for t in tasks]

    # Build task_type_map for proper metric computation
    task_type_map: Dict[str, str] = {t["name"]: t.get("type", "binary") for t in tasks}

    # Determine input dimension dynamically from actual data (after ablation
    # column removal), falling back to config only when data introspection
    # is unavailable.  This ensures that when feature groups are removed,
    # the model input_dim matches the actual feature count.
    features_config = config.get("features", {})

    # Try to infer input_dim from the actual training data (DataLoader peek)
    _auto_input_dim = None
    try:
        _sample_batch = next(iter(train_data))
        if isinstance(_sample_batch, dict) and "features" in _sample_batch:
            _auto_input_dim = _sample_batch["features"].shape[-1]
    except StopIteration:
        pass

    if _auto_input_dim is not None:
        input_dim = _auto_input_dim
        logger.info("Model input_dim (auto-detected from data): %d", input_dim)
    elif features_config.get("input_dim") and not removed_feature_groups:
        # Only trust the YAML input_dim if no feature groups were removed
        input_dim = int(features_config["input_dim"])
        logger.info("Model input_dim (from config): %d", input_dim)
    else:
        input_dim = int(features_config.get("input_dim", model_config.get("expert_hidden_dim", 128)))
        logger.warning(
            "Model input_dim (fallback): %d — may be inaccurate if feature "
            "groups were removed", input_dim,
        )

    # Build PLEConfig with proper expert dimensions
    ple_cfg = model_config.get("ple", {})
    expert_cfg = model_config.get("expert_config", {})
    tower_cfg = model_config.get("task_tower", {})

    # Expert hidden dims — must be compatible with input_dim
    mlp_cfg = expert_cfg.get("mlp", {})
    expert_hidden = mlp_cfg.get("hidden_dims", [input_dim * 2, input_dim])
    expert_output = ple_cfg.get("extraction_dim", 32)

    from core.model.ple.config import ExpertConfig, ExpertBasketConfig, LossWeightingConfig

    shared_expert = ExpertConfig(
        hidden_dims=expert_hidden,
        output_dim=expert_output,
        dropout=model_config.get("dropout", 0.1),
    )
    task_expert = ExpertConfig(
        hidden_dims=expert_hidden,
        output_dim=expert_output,
        dropout=model_config.get("dropout", 0.1),
    )

    # Apply ablation overrides for num_layers
    num_extraction_layers = ple_cfg.get("num_layers", 2)
    if hp_num_layers is not None:
        num_extraction_layers = int(hp_num_layers)
        logger.info(
            "Ablation: overriding num_extraction_layers %d -> %d",
            ple_cfg.get("num_layers", 2), num_extraction_layers,
        )

    # Apply ablation overrides for shared experts
    num_shared_experts = ple_cfg.get("num_shared_experts", 2)
    expert_basket = None
    if shared_experts_override is not None:
        num_shared_experts = len(shared_experts_override)
        expert_basket_cfg = model_config.get("expert_basket", {})
        expert_basket = ExpertBasketConfig(
            shared_experts=shared_experts_override,
            task_experts=expert_basket_cfg.get("task", ["mlp"]),
            expert_configs={
                name: expert_cfg.get(name, {})
                for name in shared_experts_override
                if name in expert_cfg
            },
        )
        logger.info(
            "Ablation: overriding shared experts -> %s (%d experts)",
            shared_experts_override, num_shared_experts,
        )
    else:
        # Build expert basket from config if present
        eb_cfg = model_config.get("expert_basket", {})
        if eb_cfg.get("shared"):
            expert_basket = ExpertBasketConfig(
                shared_experts=eb_cfg["shared"],
                task_experts=eb_cfg.get("task", ["mlp"]),
                expert_configs={
                    name: expert_cfg.get(name, {})
                    for name in eb_cfg["shared"]
                    if name in expert_cfg
                },
            )

    # -- Loss weighting from YAML config ---
    lw_cfg = model_config.get("loss_weighting", {})
    loss_weighting = LossWeightingConfig(
        strategy=lw_cfg.get("strategy", "fixed"),
        gradnorm_alpha=lw_cfg.get("gradnorm_alpha", 1.5),
        gradnorm_interval=lw_cfg.get("gradnorm_interval", 1),
        dwa_temperature=lw_cfg.get("dwa_temperature", 2.0),
        dwa_window_size=lw_cfg.get("dwa_window_size", 5),
    )
    logger.info("Loss weighting strategy: %s", loss_weighting.strategy)

    ple_config = PLEConfig(
        input_dim=input_dim,
        task_names=task_names,
        num_shared_experts=num_shared_experts,
        num_extraction_layers=num_extraction_layers,
        num_task_experts_per_task=ple_cfg.get("num_task_experts", 1),
        shared_expert=shared_expert,
        task_expert=task_expert,
        dropout=model_config.get("dropout", 0.1),
        expert_basket=expert_basket,
        loss_weighting=loss_weighting,
    )

    # Set task overrides (type + output_dim + loss)
    for t in tasks:
        task_override = {
            "task_type": t.get("type", "binary"),
            "output_dim": t.get("num_classes", 1),
        }
        # Pass per-task loss type from YAML tasks config
        if "loss" in t:
            task_override["loss"] = t["loss"]
        if "loss_params" in t:
            task_override["loss_params"] = t["loss_params"]
        ple_config.task_overrides[t["name"]] = task_override

    # -- Logit transfers from task_relationships config ---
    logit_transfers_raw = config.get("task_relationships", [])
    if logit_transfers_raw:
        from core.model.ple.config import LogitTransferDef
        ple_config.logit_transfers = [
            LogitTransferDef(
                source=lt["source"],
                target=lt["target"],
                enabled=lt.get("enabled", True),
                transfer_method=lt.get("transfer_method", "residual"),
            )
            for lt in logit_transfers_raw
        ]
        ple_config.logit_transfer_strength = float(
            config.get("logit_transfer_strength", 0.5)
        )
        logger.info(
            "Logit transfers: %d relationships, strength=%.2f",
            len(ple_config.logit_transfers),
            ple_config.logit_transfer_strength,
        )

    # -- Multidisciplinary feature routing ---
    md_routing = model_config.get("multidisciplinary_routing", {})
    if md_routing:
        ple_config.multidisciplinary_routing = {
            str(k): list(v) for k, v in md_routing.items()
        }
        logger.info(
            "Multidisciplinary routing: %d groups configured",
            len(ple_config.multidisciplinary_routing),
        )

    # Task tower dims
    default_tower_dims = tower_cfg.get("default_dims", [expert_output, expert_output // 2])
    ple_config.task_tower.default_dims = default_tower_dims

    logger.info(
        "PLEConfig: input_dim=%d, expert_hidden=%s, expert_output=%d, "
        "shared=%d, task_experts=%d, layers=%d, tower=%s",
        input_dim, expert_hidden, expert_output,
        ple_config.num_shared_experts,
        ple_config.num_task_experts_per_task,
        ple_config.num_extraction_layers,
        default_tower_dims,
    )

    model = PLEModel(ple_config).to(device)
    logger.info(model.summary())

    # -- Apply phase configuration --
    # Load pretrained weights for phase 2. Freeze/unfreeze is handled by
    # PLETrainer internally, but apply_phase_config loads pretrained weights.
    if phase == "2" and pretrained_uri:
        apply_phase_config(model, phase, pretrained_uri)

    # -- Checkpoint manager --
    from core.training.checkpoint import CheckpointManager

    ckpt_mgr = CheckpointManager(
        local_dir=checkpoint_dir,
        max_keep=3,
    )

    # ================================================================
    # PLETrainer — production-quality training loop with AMP,
    # GradNorm, warmup scheduler, gradient accumulation, adaTT, etc.
    # ================================================================
    logger.info("Using PLETrainer (production training loop)")

    # -- Map SageMaker HP "phase" to PLETrainer phase values --
    # HP values: "single" | "1" | "2" | "full"
    # PLETrainer expects: "full" | "phase1" | "phase2"
    _phase_map = {"single": "full", "1": "phase1", "2": "phase2", "full": "full"}
    trainer_phase = _phase_map.get(phase, "full")

    # -- Build TrainingConfig from SageMaker hyperparameters --
    training_cfg_dict: Dict[str, Any] = {
        "batch_size": batch_size,
        "optimizer": {
            "name": "adamw",
            "learning_rate": lr,
            "weight_decay": 0.01,
        },
        "scheduler": {
            "name": "cosine",
            "warmup_epochs": 3,
            "cosine_t0": max(10, epochs // 3),
            "cosine_t_mult": 2,
            "phase2_warmup_epochs": 2,
            "phase2_cosine_t0": max(6, epochs // 5),
        },
        "amp": {"enabled": use_amp},
        "gradient": {
            "clip_norm": 5.0,
            "accumulation_steps": grad_accum_steps,
        },
        "early_stopping": {
            "enabled": True,
            "patience": patience,
        },
        "checkpoint": {
            "dir": checkpoint_dir,
            "save_every_n_epochs": 1,
            "max_to_keep": 3,
        },
        "phase1": {
            "epochs": epochs if trainer_phase in ("phase1", "full") else 0,
        },
        "phase2": {
            "epochs": epochs if trainer_phase in ("phase2", "full") else 0,
        },
        "experiment_name": task_name,
    }

    # If YAML config has a "training" block, merge it (YAML wins for
    # keys not overridden by SageMaker HPs)
    yaml_training = config.get("training", {})
    if yaml_training:
        import copy
        merged = copy.deepcopy(yaml_training)
        # SageMaker HPs override YAML for these critical keys
        merged.setdefault("optimizer", {}).update(training_cfg_dict["optimizer"])
        merged.setdefault("gradient", {}).update(training_cfg_dict["gradient"])
        merged.setdefault("amp", {}).update(training_cfg_dict["amp"])
        merged.setdefault("early_stopping", {}).update(training_cfg_dict["early_stopping"])
        merged.setdefault("checkpoint", {}).update(training_cfg_dict["checkpoint"])
        merged["batch_size"] = batch_size
        merged["experiment_name"] = task_name
        # Phase epoch counts from SageMaker HP override
        merged.setdefault("phase1", {})["epochs"] = training_cfg_dict["phase1"]["epochs"]
        merged.setdefault("phase2", {})["epochs"] = training_cfg_dict["phase2"]["epochs"]
        training_config = TrainingConfig.from_dict(merged)
    else:
        training_config = TrainingConfig.from_dict(training_cfg_dict)

    logger.info(
        "TrainingConfig: lr=%.5f, weight_decay=%.4f, clip_norm=%.1f, "
        "AMP=%s, warmup=%d, phase1_epochs=%d, phase2_epochs=%d",
        training_config.optimizer.learning_rate,
        training_config.optimizer.weight_decay,
        training_config.gradient.clip_norm,
        training_config.amp.enabled,
        training_config.scheduler.warmup_epochs,
        training_config.phase1.epochs,
        training_config.phase2.epochs,
    )

    # -- Create PLETrainer --
    # PLETrainer handles optimizer, scheduler, AMP, freeze/unfreeze internally.
    # Ensure all params are unfrozen so PLETrainer manages freeze/unfreeze.
    for param in model.parameters():
        param.requires_grad = True

    trainer = PLETrainer(
        model=model,
        config=training_config,
        device=device,
    )

    # Resume from checkpoint if available (Spot restart)
    resume_state = ckpt_mgr.load_latest(
        model, trainer.optimizer, trainer.scheduler, map_location=device,
    )
    if resume_state:
        trainer.current_epoch = resume_state.get("epoch", 0)
        trainer.global_step = resume_state.get("global_step", 0)
        logger.info(f"Resuming from epoch {trainer.current_epoch}")

    # -- Run training --
    trainer_results = trainer.train(
        train_loader, val_loader, phase=trainer_phase,
    )

    logger.info("PLETrainer finished: %s", trainer_results)

    # -- Extract final metrics --
    # PLETrainer returns {"best_val_loss": ..., "avg_auc": ..., ...}
    # We need to run a final validation pass to get the full metric dict
    # compatible with the eval report format.
    final_metrics = validate(model, val_loader, device, task_names, task_type_map)
    report_metrics("final_val", final_metrics, epochs)

    # Track last epoch for the eval report
    epoch = trainer.current_epoch - 1

    # -- Save model for SageMaker packaging --
    os.makedirs(model_dir, exist_ok=True)

    # Save as both state_dict and full checkpoint
    model_path = os.path.join(model_dir, "model.pth")
    save_dict = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "ple_config": {
            "input_dim": ple_config.input_dim,
            "task_names": ple_config.task_names,
            "num_shared_experts": ple_config.num_shared_experts,
            "num_extraction_layers": ple_config.num_extraction_layers,
        },
        "metrics": final_metrics,
        "task_name": task_name,
        "phase": phase,
    }
    # Persist ablation metadata so downstream stages (e.g. Phase 4
    # distillation) can retrieve the settings that produced this model.
    if ablation_type:
        save_dict["ablation"] = {
            "ablation_type": ablation_type,
            "ablation_scenario": ablation_scenario,
            "removed_feature_groups": removed_feature_groups,
            "shared_experts": shared_experts_override,
            "num_layers": num_extraction_layers,
            "temperature": float(hp_temperature) if hp_temperature is not None else None,
        }
    torch.save(save_dict, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save config separately for easy inspection
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Save evaluation report — filename must match what orchestrator/report
    # generator expect: "eval_metrics.json" (NOT "evaluation_report.json")
    report_path = os.path.join(output_dir, "eval_metrics.json")
    os.makedirs(output_dir, exist_ok=True)

    # Compute aggregate_score (avg AUC across binary tasks) for best-config selection
    auc_keys = [k for k in final_metrics if k.startswith("auc_")]
    aggregate_score = (
        sum(final_metrics[k] for k in auc_keys) / len(auc_keys)
        if auc_keys else final_metrics.get("auc", 0.0)
    )

    eval_report: Dict[str, Any] = {
        "task_name": task_name,
        "phase": phase,
        "final_metrics": final_metrics,
        "aggregate_score": aggregate_score,
        "epochs_trained": epoch + 1,
        "total_time_seconds": time.time() - start_time,
    }

    # Detailed training configuration for full reproducibility
    eval_report["training_config"] = {
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs_configured": epochs,
        "seed": config.get("training", {}).get("seed", 42),
        "max_rows": config.get("max_rows", 0),
        "weight_decay": config.get("training", {}).get("weight_decay", 0.01),
        "gradient_clip_norm": config.get("training", {}).get("gradient_clip_norm", 5.0),
        "amp_enabled": config.get("training", {}).get("amp", {}).get("enabled", False),
        "warmup_epochs": config.get("training", {}).get("scheduler", {}).get("warmup_epochs", 3),
    }

    # Model architecture details
    eval_report["architecture"] = {
        "input_dim": model.config.input_dim if hasattr(model, "config") else None,
        "num_tasks": len(tasks),
        "active_tasks": [t["name"] for t in tasks],
        "num_shared_experts": model.config.num_shared_experts if hasattr(model, "config") else None,
        "shared_expert_names": list(model.config.expert_basket.shared_experts) if hasattr(model, "config") and hasattr(model.config, "expert_basket") and model.config.expert_basket else None,
        "num_layers": model.config.num_extraction_layers if hasattr(model, "config") else None,
        "extraction_dim": getattr(model.config, "extraction_dim", getattr(model.config, "expert_output_dim", None)) if hasattr(model, "config") else None,
        "loss_weighting": model.config.loss_weighting.strategy if hasattr(model, "config") and hasattr(model.config, "loss_weighting") else None,
        "adatt_enabled": model.config.adatt.enabled if hasattr(model, "config") and hasattr(model.config, "adatt") else None,
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    # Normalization details
    eval_report["normalization"] = {
        "feature_scaler": "StandardScaler (continuous only, binary skipped)",
        "regression_label_transform": "clip(99.5%) + log1p",
        "n_continuous_features": len(_module_continuous_cols) if _module_continuous_cols else None,
        "n_binary_features": None,  # tracked per load_data call
    }
    # HIGH-3: Scaler parameter samples in eval report
    if _module_continuous_cols and _module_scaler is not None:
        eval_report["normalization"]["scaler_mean_sample"] = dict(
            zip(_module_continuous_cols[:10], _module_scaler.mean_[:10].tolist())
        )
        eval_report["normalization"]["scaler_std_sample"] = dict(
            zip(_module_continuous_cols[:10], _module_scaler.scale_[:10].tolist())
        )

    # HIGH-4: Record train/val split details
    eval_report["data_split"] = {
        "train_size": train_size if 'train_size' in dir() else len(train_loader.dataset),
        "val_size": val_size if 'val_size' in dir() else len(val_loader.dataset),
        "split_ratio": round(
            (train_size if 'train_size' in dir() else len(train_loader.dataset))
            / ((train_size if 'train_size' in dir() else len(train_loader.dataset))
               + (val_size if 'val_size' in dir() else len(val_loader.dataset))),
            3,
        ),
        "seed": config.get("training", {}).get("seed", 42),
    }

    # Label statistics (pre-computed in load_data before df goes out of scope)
    eval_report["label_stats"] = _label_stats if _label_stats else {}

    # Pos weights (if computed)
    if hasattr(model, '_pos_weights') and model._pos_weights:
        eval_report["pos_weights"] = {
            k: round(v.item(), 2) for k, v in model._pos_weights.items()
        }

    # Epoch-level training history (from PLETrainer)
    if hasattr(trainer, 'epoch_history') and trainer.epoch_history:
        eval_report["epoch_history"] = trainer.epoch_history

    # MEDIUM-9: Early stopping reason (persisted from trainer)
    if hasattr(trainer, 'early_stop_info') and trainer.early_stop_info:
        eval_report["early_stopping"] = trainer.early_stop_info

    # MEDIUM-10: Logit transfer edges in architecture
    if hasattr(model, 'config') and hasattr(model.config, 'logit_transfers') and model.config.logit_transfers:
        eval_report["architecture"]["logit_transfers"] = [
            {"source": lt.source, "target": lt.target, "method": lt.transfer_method}
            for lt in model.config.logit_transfers
        ]
    else:
        eval_report["architecture"]["logit_transfers"] = []

    # MEDIUM-11: Per-task loss function config
    eval_report["task_configs"] = {
        t["name"]: {
            "type": t.get("type", "binary"),
            "loss": t.get("loss", "default"),
            "loss_weight": t.get("loss_weight", 1.0),
            "num_classes": t.get("num_classes", 1),
        }
        for t in tasks
    }

    # MEDIUM-12: Label derivation method registry
    eval_report["label_derivation"] = {
        "has_nba": "direct_column",
        "churn_signal": "direct_column",
        "product_stability": "direct_column",
        "label_nba_primary": "nba_label[0] (list_first, default=-1)",
        "label_tenure_stage": "bin(tenure_months, [unknown->-1, 0-12->0, 13-36->1, 37-72->2, 73-120->3, 121+->4])",
        "label_segment": "map(segment, {01-TOP->0, 02-PARTICULARES->1, 03-UNIVERSITARIO->2, other->3})",
        "label_income_tier": "bin(income, [<30k->0, 30-80k->1, 80-200k->2, 200k+->3])",
        "label_spend_level": "bin(synth_monthly_spend, [<1500->0, 1500-3000->1, 3000-5000->2, 5000+->3])",
        "label_cross_sell_count": "len(nba_label) or total_acquisitions",
        "label_engagement_score": "composite(is_active*0.3 + synth_frequency*0.4 + num_products*0.3)",
        "label_next_mcc": "txn_mcc_seq[-1] (capped at 49, default=-1)",
        "label_mcc_diversity_trend": "(unique_second_half / unique_first_half) - 1.0",
        "label_top_mcc_shift": "int(mode(first_half) != mode(second_half))",
        "label_acquire_deposits": "any(nba_label & deposit_indices)",
        "label_acquire_investments": "any(nba_label & investment_indices)",
        "label_acquire_accounts": "any(nba_label & account_indices)",
        "label_acquire_lending": "any(nba_label & lending_indices)",
        "label_acquire_payments": "any(nba_label & payment_indices)",
    }

    # MEDIUM-13: Feature column names saved
    if '_continuous_cols' in dir() and _continuous_cols:
        eval_report["normalization"]["continuous_columns"] = _continuous_cols[:50]
    if '_binary_cols' in dir() and _binary_cols:
        eval_report["normalization"]["binary_columns"] = _binary_cols[:50]

    # MEDIUM-14: Generator summary persisted
    # Note: df is out of scope here (it was local to load_data).
    # Use feature_names from the model config or _label_stats keys instead.
    _gen_cols_for_report = [c for c in (getattr(model.config, '_feature_names', None) or [])
                            if any(c.startswith(p) for p in
                                   ("tda_", "graph_", "hmm_", "mamba_", "gmm_", "model_derived_"))]
    if _gen_cols_for_report:
        _gen_summary: Dict[str, int] = {}
        for c in _gen_cols_for_report:
            prefix = c
            for p in ("tda_global", "tda_local", "graph", "hmm", "mamba", "gmm", "model_derived"):
                if c.startswith(p):
                    prefix = p
                    break
            _gen_summary[prefix] = _gen_summary.get(prefix, 0) + 1
        eval_report["generators"] = _gen_summary

    # Final adaTT state
    if hasattr(model, 'adatt') and model.adatt is not None:
        try:
            affinity = model.adatt.get_transfer_matrix()
            if affinity is not None:
                task_names_list = list(model.task_names)
                n = len(task_names_list)
                eval_report["adatt_final"] = {
                    "affinity_matrix": [[round(affinity[i, j].item(), 4) for j in range(n)] for i in range(n)],
                    "task_names": task_names_list,
                    "config": {
                        "transfer_lambda": model.config.adatt.transfer_lambda,
                        "temperature": model.config.adatt.temperature,
                        "warmup_epochs": model.config.adatt.warmup_epochs,
                        "grad_interval": model.config.adatt.grad_interval,
                    },
                }
        except Exception:
            pass

    # Final uncertainty weights (log_vars)
    if hasattr(model, 'loss_weighting') and model.loss_weighting is not None:
        try:
            weights = model.get_loss_weights()
            if weights:
                eval_report["final_loss_weights"] = {k: round(v, 4) for k, v in weights.items()}
        except Exception:
            pass

    if ablation_type:
        eval_report["ablation"] = {
            "ablation_type": ablation_type,
            "ablation_scenario": ablation_scenario,
            "removed_feature_groups": removed_feature_groups,
            "shared_experts": shared_experts_override,
            "num_layers": num_extraction_layers,
            "temperature": float(hp_temperature) if hp_temperature is not None else None,
            "active_tasks": hp.get("active_tasks", None),
            "use_ple": hp.get("use_ple", None),
            "use_adatt": hp.get("use_adatt", None),
        }
    # Per-task metric breakdown by task type
    # Use "per_task" key (matches what report generator expects)
    eval_report["per_task"] = {}
    for t in tasks:
        tname = t["name"]
        ttype = t.get("type", "binary")
        task_metrics = {
            k: v for k, v in final_metrics.items()
            if k.endswith(f"_{tname}")
        }
        eval_report["per_task"][tname] = {
            "type": ttype,
            "metrics": task_metrics,
        }
    # Backward compat alias
    eval_report["per_task_metrics"] = eval_report["per_task"]

    # HIGH-6: Stage 8.5 Model Analysis (optional, config-gated)
    analysis_cfg = config.get("analysis", {})
    if analysis_cfg.get("enabled", False):
        analysis_results = {}

        # IG attribution (quick, limited batches)
        try:
            from core.evaluation.integrated_gradients import IntegratedGradients
            ig = IntegratedGradients(model, n_steps=analysis_cfg.get("ig_steps", 20))
            for t in tasks[:3]:  # Top 3 tasks only for speed
                importance = ig.feature_importance(val_loader, t["name"], max_batches=5)
                analysis_results[f"ig_{t['name']}"] = dict(list(importance.items())[:20])
        except Exception as e:
            logger.warning("IG analysis failed: %s", e)

        # Expert redundancy CCA
        try:
            from core.evaluation.expert_redundancy import ExpertRedundancyAnalyzer
            cca = ExpertRedundancyAnalyzer(model)
            redundancy = cca.analyze(val_loader, max_batches=5)
            analysis_results["expert_redundancy"] = redundancy.to_dict()
        except Exception as e:
            logger.warning("CCA analysis failed: %s", e)

        # Gate weights
        try:
            from core.evaluation.gate_analyzer import GateAnalyzer
            gate = GateAnalyzer(model)
            gate_result = gate.analyze(val_loader, max_batches=5)
            analysis_results["gate_weights"] = {
                "task_expert_weights": gate_result.task_expert_weights,
                "expert_utilization": gate_result.expert_utilization,
                "dominant_expert_per_task": gate_result.dominant_expert_per_task,
            }
        except Exception as e:
            logger.warning("Gate analysis failed: %s", e)

        if analysis_results:
            eval_report["analysis"] = analysis_results
            logger.info("Stage 8.5 analysis complete: %d results", len(analysis_results))

    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)

    # Also upload eval_metrics.json directly to S3 output path so orchestrator
    # and report generator can access it as a bare file (SageMaker packs
    # SM_OUTPUT_DATA_DIR into output.tar.gz which is NOT directly readable).
    _s3_output = hp.get("_s3_output", "")
    if _s3_output:
        try:
            import boto3
            s3_client = boto3.client("s3")
            parts = _s3_output.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            s3_key = (parts[1] if len(parts) > 1 else "").rstrip("/") + "/output/eval_metrics.json"
            s3_client.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=json.dumps(eval_report, indent=2),
                ContentType="application/json",
            )
            logger.info(f"Uploaded eval_metrics.json to s3://{bucket}/{s3_key}")
        except Exception as e:
            logger.warning(f"Failed to upload eval_metrics.json to S3: {e}")

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Training complete in {total_time:.1f}s")
    logger.info(f"Final metrics: {final_metrics}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse as _argparse

    _parser = _argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--pipeline", type=str, default=None,
                         help="Path to pipeline YAML config. "
                              "When set, runs via PipelineRunner "
                              "instead of legacy SageMaker path.")
    _known, _remaining = _parser.parse_known_args()

    if _known.pipeline:
        # Route to PipelineRunner-based entry point
        logger.info("Running via PipelineRunner (config=%s)", _known.pipeline)
        main_pipeline(_known.pipeline)
    else:
        # Legacy SageMaker Training Job path
        main()
