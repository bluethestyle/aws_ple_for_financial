"""
SageMaker training entry point — MINIMAL training-only script.

Receives "training-ready" artifacts from Phase 0 (PipelineRunner):
  features.parquet      — normalized numeric features
  labels.parquet        — derived labels
  sequences.npy         — padded 3D tensor (optional)
  seq_lengths.npy       — sequence lengths (optional)
  scaler_params.json    — for reference only
  feature_schema.json   — column names, group ranges, expert routing
  label_schema.json     — task definitions
  split_indices.json    — train/val/test row indices

ZERO preprocessing: no fillna, no encoding, no scaler, no label derivation.
ALL config from schema JSON files (produced by PipelineRunner).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Enable synchronous CUDA error reporting for debugging
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
    sm_hps = os.environ.get("SM_HPS")
    if sm_hps:
        return json.loads(sm_hps)

    hp_path = Path("/opt/ml/input/config/hyperparameters.json")
    if hp_path.exists():
        with open(hp_path) as f:
            raw = json.load(f)
        return {k: _parse_hp_value(v) for k, v in raw.items()}

    return {}


def _parse_hp_value(v: str) -> Any:
    """Best-effort parse a stringified hyperparameter value."""
    if isinstance(v, str):
        lower = v.lower()
        if lower in ("true", "false"):
            return lower == "true"
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
# Data loading — training-ready artifacts from Phase 0
# ---------------------------------------------------------------------------

def load_ready_data(channel_dir: str) -> dict:
    """Load training-ready artifacts produced by Phase 0.

    Returns
    -------
    dict with keys:
        features : pd.DataFrame — normalized numeric features
        labels : pd.DataFrame — derived labels
        sequences : np.ndarray or None — padded 3D tensor
        seq_lengths : np.ndarray or None — sequence lengths
        feature_schema : dict — column names, group ranges, expert routing
        label_schema : dict — task definitions
        split_indices : dict or None — train/val/test row indices
    """
    import pandas as pd

    channel_path = Path(channel_dir)

    # -- Features --
    features_path = channel_path / "features.parquet"
    if features_path.exists():
        features = pd.read_parquet(features_path)
        logger.info("Loaded features: %d rows, %d columns", len(features), len(features.columns))
    else:
        # Fallback: load from generic parquet files (backward compat)
        parquet_files = sorted(channel_path.glob("**/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No features.parquet or .parquet files in {channel_dir}")
        features = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        logger.info("Loaded %d parquet files (fallback): %d rows, %d columns",
                     len(parquet_files), len(features), len(features.columns))

    # Data loading diagnostics
    logger.info("Features: %d rows x %d cols, memory=%.1f MB",
                 len(features), len(features.columns), features.memory_usage(deep=True).sum() / 1e6)
    logger.info("Dtypes: %s", dict(features.dtypes.value_counts()))
    nan_count = features.isna().sum().sum()
    if nan_count > 0:
        logger.warning("Features contain %d NaN values across %d columns",
                        nan_count, features.isna().any().sum())

    # -- Labels --
    labels_path = channel_path / "labels.parquet"
    labels = pd.read_parquet(labels_path) if labels_path.exists() else None
    if labels is not None:
        logger.info("Loaded labels: %d rows, %d columns", len(labels), len(labels.columns))

    # -- Sequences (optional) --
    sequences = None
    seq_path = channel_path / "sequences.npy"
    if seq_path.exists():
        sequences = np.load(str(seq_path))
        logger.info("Loaded sequences: %s", sequences.shape)

    seq_lengths = None
    lengths_path = channel_path / "seq_lengths.npy"
    if lengths_path.exists():
        seq_lengths = np.load(str(lengths_path))
        logger.info("Loaded seq_lengths: %s", seq_lengths.shape)

    # -- Feature schema --
    feature_schema_path = channel_path / "feature_schema.json"
    if feature_schema_path.exists():
        with open(feature_schema_path) as f:
            feature_schema = json.load(f)
        logger.info("Loaded feature_schema: %d keys", len(feature_schema))
        group_ranges = feature_schema.get("group_ranges", {})
        if group_ranges:
            logger.info("=== Feature Groups ===")
            for name, (start, end) in sorted(group_ranges.items(), key=lambda x: x[1][0]):
                logger.info("  %s: cols %d-%d (%d dims)", name, start, end, end - start)
    else:
        # Auto-generate minimal schema from DataFrame columns
        feature_schema = {
            "columns": list(features.columns),
            "group_ranges": {},
            "expert_routing": {},
        }
        logger.warning("No feature_schema.json found — auto-generated from DataFrame columns")

    # -- Label schema --
    label_schema_path = channel_path / "label_schema.json"
    if label_schema_path.exists():
        with open(label_schema_path) as f:
            label_schema = json.load(f)
        logger.info("Loaded label_schema: %d tasks", len(label_schema.get("tasks", [])))
    else:
        label_schema = {}
        logger.warning("No label_schema.json found")

    # -- Split indices --
    split_path = channel_path / "split_indices.json"
    split_indices = None
    if split_path.exists():
        with open(split_path) as f:
            split_indices = json.load(f)
        logger.info("Loaded split_indices: %s",
                     {k: len(v) for k, v in split_indices.items()})

    return {
        "features": features,
        "labels": labels,
        "sequences": sequences,
        "seq_lengths": seq_lengths,
        "feature_schema": feature_schema,
        "label_schema": label_schema,
        "split_indices": split_indices,
    }


# ---------------------------------------------------------------------------
# Ablation filters
# ---------------------------------------------------------------------------

def apply_ablation(features, labels, feature_schema, label_schema, hp):
    """Apply ablation filters using schema-driven column/task selection.

    Modifies features and schemas in-place based on hyperparameters:
      - removed_feature_groups: drop columns by group using schema["group_ranges"]
      - active_tasks: filter tasks to only active ones
      - use_ple / use_adatt: stored as config overrides (handled at model build)

    Returns
    -------
    features, labels, feature_schema, label_schema (filtered)
    """
    # -- Feature ablation: drop columns by group using schema["group_ranges"] --
    removed_raw = hp.get("removed_feature_groups", "[]")
    if isinstance(removed_raw, str):
        try:
            removed = json.loads(removed_raw)
        except (json.JSONDecodeError, ValueError):
            removed = []
    elif isinstance(removed_raw, list):
        removed = removed_raw
    else:
        removed = []

    group_ranges = feature_schema.get("group_ranges", {})
    columns = feature_schema.get("columns", list(features.columns))

    if removed:
        cols_to_drop = []
        for group_name in removed:
            if group_name in group_ranges:
                start, end = group_ranges[group_name]
                cols_to_drop.extend(columns[start:end])
            else:
                logger.warning("Ablation: group '%s' not in schema group_ranges", group_name)
        if cols_to_drop:
            features = features.drop(columns=cols_to_drop, errors="ignore")
            logger.info("Ablation: removed %d columns from groups %s. Remaining: %d",
                         len(cols_to_drop), removed, len(features.columns))
        else:
            logger.warning("Ablation: no columns matched for groups %s", removed)

    # -- Task ablation: filter active tasks --
    active_raw = hp.get("active_tasks")
    tasks = label_schema.get("tasks", [])
    if active_raw:
        active = json.loads(active_raw) if isinstance(active_raw, str) else active_raw
        original_count = len(tasks)
        tasks = [t for t in tasks if t["name"] in active]
        label_schema["tasks"] = tasks
        logger.info("Task ablation: %d/%d tasks active: %s",
                     len(tasks), original_count, [t["name"] for t in tasks])

        # Filter task_groups to only reference active tasks
        active_set = set(t["name"] for t in tasks)
        for tg in label_schema.get("task_groups", []):
            tg["tasks"] = [t for t in tg["tasks"] if t in active_set]
        label_schema["task_groups"] = [
            tg for tg in label_schema.get("task_groups", []) if tg.get("tasks")
        ]

        # Filter task_relationships
        label_schema["task_relationships"] = [
            tr for tr in label_schema.get("task_relationships", [])
            if tr["source"] in active_set and tr["target"] in active_set
        ]

    # -- Filter labels DataFrame to only active task label columns --
    if labels is not None and tasks:
        label_cols = [t["label_col"] for t in tasks if t["label_col"] in labels.columns]
        labels = labels[label_cols]

    return features, labels, feature_schema, label_schema


# ---------------------------------------------------------------------------
# DataLoader construction
# ---------------------------------------------------------------------------

def build_dataloaders(features, labels, sequences, seq_lengths, feature_schema,
                      label_schema, split_indices, hp):
    """Build train/val DataLoaders from training-ready data.

    Returns
    -------
    train_loader, val_loader, tasks, task_type_map, label_stats
    """
    import pandas as pd
    from core.data.dataloader import build_ple_dataloader, FeatureColumnSpec

    tasks = label_schema.get("tasks", [])
    batch_size = int(hp.get("batch_size", 2048))

    # Subsample if max_rows HP is set (for fast testing)
    max_rows = int(hp.get("max_rows", 0))
    if max_rows and max_rows > 0 and len(features) > max_rows:
        idx = features.sample(n=max_rows, random_state=42).index
        features = features.loc[idx].reset_index(drop=True)
        if labels is not None:
            labels = labels.loc[idx].reset_index(drop=True)
        if sequences is not None:
            sequences = sequences[idx.values]
        if seq_lengths is not None:
            seq_lengths = seq_lengths[idx.values]
        logger.info("Subsampled to %d rows for fast testing", max_rows)

    # Validate tasks against available label columns
    if labels is not None:
        available_labels = set(labels.columns)
        valid_tasks = []
        for t in tasks:
            lc = t["label_col"]
            if lc not in available_labels:
                logger.warning("Skipping task %s: label_col '%s' not in labels", t["name"], lc)
                continue
            valid_tasks.append(t)
        tasks = valid_tasks
        label_schema["tasks"] = tasks
    if not tasks:
        raise ValueError("No tasks have valid label columns in the data.")

    task_names = [t["name"] for t in tasks]
    task_type_map = {t["name"]: t.get("type", "binary") for t in tasks}
    label_map = {t["name"]: t["label_col"] for t in tasks}

    # -- Merge features + labels into a single DataFrame for PLEDataset --
    df = features.copy()
    if labels is not None:
        for col in labels.columns:
            df[col] = labels[col].values

    # -- Build FeatureColumnSpec from schema --
    feature_columns = feature_schema.get("columns", list(features.columns))
    # Only include columns actually present in the DataFrame (post-ablation)
    static_features = [c for c in feature_columns if c in df.columns]

    feature_spec = FeatureColumnSpec(static_features=static_features)
    logger.info("FeatureColumnSpec: %d static features", len(static_features))

    # -- Pre-compute label statistics --
    label_stats = {}
    for t in tasks:
        lc = t["label_col"]
        if lc in df.columns:
            col = df[lc]
            if t.get("type") == "binary":
                n_pos = int((col > 0.5).sum())
                label_stats[t["name"]] = {
                    "positive_count": n_pos,
                    "positive_rate": round(n_pos / len(col), 4),
                    "total": len(col),
                }
            elif t.get("type") == "regression":
                label_stats[t["name"]] = {
                    "mean": round(float(col.mean()), 4),
                    "std": round(float(col.std()), 4),
                    "total": len(col),
                }
            elif t.get("type") == "multiclass":
                label_stats[t["name"]] = {
                    "num_classes": int(col.nunique()),
                    "total": len(col),
                }

    # -- Split into train/val --
    use_gpu_loading = hp.get("use_gpu_loading", False) and int(os.environ.get("SM_NUM_GPUS", "0")) > 0

    if split_indices:
        train_idx = split_indices.get("train", [])
        val_idx = split_indices.get("val", split_indices.get("validation", []))

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True) if val_idx else None

        train_loader = build_ple_dataloader(
            df=df_train, feature_spec=feature_spec, label_columns=label_map,
            batch_size=batch_size, shuffle=True, use_gpu_loading=use_gpu_loading,
        )

        val_loader = None
        if df_val is not None and len(df_val) > 0:
            val_loader = build_ple_dataloader(
                df=df_val, feature_spec=feature_spec, label_columns=label_map,
                batch_size=batch_size, shuffle=False, use_gpu_loading=use_gpu_loading,
            )

        # Inject sequences if present
        if sequences is not None:
            train_seqs = torch.tensor(sequences[train_idx], dtype=torch.float32)
            train_lens = torch.tensor(seq_lengths[train_idx], dtype=torch.long) if seq_lengths is not None else None
            _inject_sequences_into_ple_dataset(train_loader.dataset, train_seqs, train_lens)
            if val_loader is not None and val_idx:
                val_seqs = torch.tensor(sequences[val_idx], dtype=torch.float32)
                val_lens = torch.tensor(seq_lengths[val_idx], dtype=torch.long) if seq_lengths is not None else None
                _inject_sequences_into_ple_dataset(val_loader.dataset, val_seqs, val_lens)

        logger.info("Train: %d samples, Val: %d samples (from split_indices)",
                     len(df_train), len(df_val) if df_val is not None else 0)
    else:
        # No split indices — build single loader then split
        full_loader = build_ple_dataloader(
            df=df, feature_spec=feature_spec, label_columns=label_map,
            batch_size=batch_size, shuffle=True, use_gpu_loading=use_gpu_loading,
        )

        # Inject sequences if present
        if sequences is not None:
            event_seqs_tensor = torch.tensor(sequences, dtype=torch.float32)
            lens_tensor = torch.tensor(seq_lengths, dtype=torch.long) if seq_lengths is not None else None
            _inject_sequences_into_ple_dataset(full_loader.dataset, event_seqs_tensor, lens_tensor)

        # Split dataset
        n = len(full_loader.dataset)
        val_ratio = float(label_schema.get("val_split", hp.get("val_ratio", 0.1)))
        val_size = max(1, int(n * val_ratio))
        train_size = n - val_size
        seed = int(hp.get("seed", 42))
        gen = torch.Generator().manual_seed(seed)
        train_subset, val_subset = torch.utils.data.random_split(
            full_loader.dataset, [train_size, val_size], generator=gen,
        )
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            collate_fn=full_loader.collate_fn, num_workers=0,
            pin_memory=not use_gpu_loading, drop_last=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            collate_fn=full_loader.collate_fn, num_workers=0,
            pin_memory=not use_gpu_loading,
        )
        logger.info("Train: %d samples, Val: %d samples (random split)",
                     train_size, val_size)

    return train_loader, val_loader, tasks, task_type_map, label_stats


def _inject_sequences_into_ple_dataset(dataset, event_sequences, seq_lengths=None):
    """Inject externally-loaded event sequences into a PLEDataset."""
    if hasattr(dataset, "_tensors"):
        dataset._tensors["event_sequences"] = event_sequences
        if seq_lengths is not None:
            dataset._tensors["seq_lengths"] = seq_lengths
        logger.info("Injected event_sequences %s into PLEDataset", list(event_sequences.shape))
    else:
        logger.warning("Cannot inject event sequences: PLEDataset has no _tensors attribute")


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(feature_schema, label_schema, hp, input_dim, device):
    """Build PLEModel from schema + hyperparameters.

    ALL config comes from schema (produced by PipelineRunner).
    No hardcoded column names or domain logic.

    Returns
    -------
    model, ple_config
    """
    from core.model.ple.model import PLEModel
    from core.model.ple.config import (
        PLEConfig, ExpertConfig, ExpertBasketConfig,
        LossWeightingConfig, LogitTransferDef,
    )

    tasks = label_schema.get("tasks", [])
    task_names = [t["name"] for t in tasks]

    # -- Expert config from schema or defaults --
    model_config = label_schema.get("model", {})
    ple_cfg = model_config.get("ple", {})
    expert_cfg = model_config.get("expert_config", {})
    tower_cfg = model_config.get("task_tower", {})

    mlp_cfg = expert_cfg.get("mlp", {})
    expert_hidden = mlp_cfg.get("hidden_dims", [input_dim * 2, input_dim])
    expert_output = ple_cfg.get("extraction_dim", 32)

    shared_expert = ExpertConfig(
        hidden_dims=expert_hidden, output_dim=expert_output,
        dropout=model_config.get("dropout", 0.1),
    )
    task_expert = ExpertConfig(
        hidden_dims=expert_hidden, output_dim=expert_output,
        dropout=model_config.get("dropout", 0.1),
    )

    # -- Ablation overrides: num_layers --
    num_extraction_layers = ple_cfg.get("num_layers", 2)
    hp_num_layers = hp.get("num_layers")
    if hp_num_layers is not None:
        num_extraction_layers = int(hp_num_layers)
        logger.info("Ablation: num_extraction_layers -> %d", num_extraction_layers)

    # -- Ablation overrides: shared experts --
    num_shared_experts = ple_cfg.get("num_shared_experts", 2)
    expert_basket = None

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

    if shared_experts_override is not None:
        num_shared_experts = len(shared_experts_override)
        expert_basket_cfg = model_config.get("expert_basket", {})
        expert_basket = ExpertBasketConfig(
            shared_experts=shared_experts_override,
            task_experts=expert_basket_cfg.get("task", ["mlp"]),
            expert_configs={
                name: expert_cfg.get(name, {})
                for name in shared_experts_override if name in expert_cfg
            },
        )
        logger.info("Ablation: shared experts -> %s (%d)", shared_experts_override, num_shared_experts)
    else:
        eb_cfg = model_config.get("expert_basket", {})
        if eb_cfg.get("shared"):
            expert_basket = ExpertBasketConfig(
                shared_experts=eb_cfg["shared"],
                task_experts=eb_cfg.get("task", ["mlp"]),
                expert_configs={
                    name: expert_cfg.get(name, {}) for name in eb_cfg["shared"] if name in expert_cfg
                },
            )

    # -- Structure ablation: PLE toggle --
    use_ple_raw = hp.get("use_ple")
    if use_ple_raw is not None:
        use_ple = json.loads(use_ple_raw) if isinstance(use_ple_raw, str) else use_ple_raw
        if not use_ple:
            num_extraction_layers = 1
            num_shared_experts = 1
            expert_basket = None
            logger.info("Structure ablation: PLE disabled (shared-bottom mode)")

    # -- Structure ablation: adaTT toggle --
    use_adatt_raw = hp.get("use_adatt")
    if use_adatt_raw is not None:
        use_adatt = json.loads(use_adatt_raw) if isinstance(use_adatt_raw, str) else use_adatt_raw
        if not use_adatt:
            model_config["adatt"] = {"enabled": False}
            logger.info("Structure ablation: adaTT disabled")

    # -- Loss weighting --
    lw_cfg = model_config.get("loss_weighting", {})
    loss_weighting = LossWeightingConfig(
        strategy=lw_cfg.get("strategy", "fixed"),
        gradnorm_alpha=lw_cfg.get("gradnorm_alpha", 1.5),
        gradnorm_interval=lw_cfg.get("gradnorm_interval", 1),
        dwa_temperature=lw_cfg.get("dwa_temperature", 2.0),
        dwa_window_size=lw_cfg.get("dwa_window_size", 5),
    )
    logger.info("Loss weighting strategy: %s", loss_weighting.strategy)

    # -- Build PLEConfig --
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

    # -- Task loss weights from label schema --
    ple_config.task_loss_weights = {t["name"]: t.get("loss_weight", 1.0) for t in tasks}

    # -- Task overrides (type + output_dim + loss) --
    for t in tasks:
        task_override = {
            "task_type": t.get("type", "binary"),
            "output_dim": t.get("num_classes", 1),
        }
        if "loss" in t:
            task_override["loss"] = t["loss"]
        if "loss_params" in t:
            task_override["loss_params"] = t["loss_params"]
        ple_config.task_overrides[t["name"]] = task_override

    # -- Logit transfers from schema --
    logit_transfers_raw = label_schema.get("task_relationships", [])
    if logit_transfers_raw:
        ple_config.logit_transfers = [
            LogitTransferDef(
                source=lt["source"], target=lt["target"],
                enabled=lt.get("enabled", True),
                transfer_method=lt.get("transfer_method", "residual"),
            )
            for lt in logit_transfers_raw
        ]
        ple_config.logit_transfer_strength = float(
            label_schema.get("logit_transfer_strength", 0.5)
        )
        logger.info("Logit transfers: %d relationships", len(ple_config.logit_transfers))

    # -- Feature group ranges from schema --
    group_ranges = feature_schema.get("group_ranges", {})
    if group_ranges:
        ple_config.feature_group_ranges = {k: tuple(v) for k, v in group_ranges.items()}

    # -- Expert routing from schema --
    expert_routing = feature_schema.get("expert_routing", {})
    if expert_routing:
        from core.model.ple.config import ExpertInputConfig
        ple_config.expert_input_routing = [
            ExpertInputConfig(**r) if isinstance(r, dict) else r
            for r in expert_routing
        ]

    # -- Task group map from schema --
    task_group_map = label_schema.get("task_group_map", {})
    if task_group_map:
        ple_config.task_group_map = task_group_map

    # -- Multidisciplinary routing from schema --
    md_routing = model_config.get("multidisciplinary_routing", {})
    if md_routing:
        ple_config.multidisciplinary_routing = {str(k): list(v) for k, v in md_routing.items()}

    # -- Task tower dims --
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

    return model, ple_config


# ---------------------------------------------------------------------------
# Metric printing (captured by SageMaker -> CloudWatch)
# ---------------------------------------------------------------------------

def report_metrics(prefix: str, metrics: Dict[str, float], epoch: int) -> None:
    """Print metrics in SageMaker CloudWatch format."""
    parts = [f"epoch={epoch}"]
    for name, value in sorted(metrics.items()):
        if isinstance(value, (int, float)):
            parts.append(f"{prefix}_{name}={value:.6f}")
        else:
            parts.append(f"{prefix}_{name}={value}")
    line = " ".join(parts)
    logger.info(line)
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

def _batch_to_ple_input(batch, task_names, device):
    """Convert a batch dict into a device-resident PLEInput."""
    from core.model.ple.model import PLEInput

    if isinstance(batch, PLEInput):
        return batch.to(device)

    if not isinstance(batch, dict):
        raise TypeError(f"Expected dict or PLEInput, got {type(batch).__name__}")

    kwargs: Dict[str, Any] = {"features": batch["features"]}
    if "targets" in batch:
        kwargs["targets"] = batch["targets"]

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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, dataloader, device, task_names, task_type_map=None):
    """Run validation and compute per-task metrics.

    Returns dict of metrics: loss, per-task AUC/accuracy/F1/MAE.
    """
    from core.model.ple.model import PLEInput
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score,
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

    for name in task_names:
        if not all_preds[name]:
            continue

        preds = np.concatenate(all_preds[name])
        tgts = np.concatenate(all_targets[name])
        task_type = task_type_map.get(name)

        if task_type == "binary":
            preds_sq = preds.squeeze()
            tgts_sq = tgts.squeeze()
            unique_labels = np.unique(tgts_sq)
            if len(unique_labels) >= 2:
                try:
                    metrics[f"auc_{name}"] = roc_auc_score(tgts_sq, preds_sq)
                except ValueError as e:
                    logger.debug("Metric computation failed for task '%s': %s", name, e)
            try:
                _thresholds = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
                best_f1, best_t = 0.0, 0.5
                for _t in _thresholds:
                    _preds_t = (preds_sq > _t).astype(int)
                    _f1_t = f1_score(tgts_sq, _preds_t, zero_division=0)
                    if _f1_t > best_f1:
                        best_f1, best_t = _f1_t, _t
                pred_labels = (preds_sq > best_t).astype(int)
                metrics[f"accuracy_{name}"] = accuracy_score(tgts_sq, pred_labels)
                metrics[f"f1_{name}"] = best_f1
                metrics[f"f1_threshold_{name}"] = best_t
                pred_labels_05 = (preds_sq > 0.5).astype(int)
                metrics[f"f1_at_0.5_{name}"] = f1_score(tgts_sq, pred_labels_05, zero_division=0)
                cm = _confusion_matrix(tgts_sq, pred_labels, labels=[0, 1]).tolist()
                metrics[f"confusion_matrix_{name}"] = cm
            except Exception as e:
                logger.debug("Metric computation failed for task '%s': %s", name, e)

        elif task_type == "multiclass":
            if preds.ndim == 2:
                pred_classes = np.argmax(preds, axis=1)
            else:
                pred_classes = np.round(preds).astype(int)
            tgts_int = tgts.astype(int).squeeze()
            try:
                metrics[f"accuracy_{name}"] = accuracy_score(tgts_int, pred_classes)
            except Exception as e:
                logger.debug("Metric computation failed for task '%s': %s", name, e)
            try:
                metrics[f"f1_macro_{name}"] = f1_score(tgts_int, pred_classes, average="macro", zero_division=0)
            except Exception as e:
                logger.debug("Metric computation failed for task '%s': %s", name, e)
            try:
                metrics[f"f1_weighted_{name}"] = f1_score(tgts_int, pred_classes, average="weighted", zero_division=0)
            except Exception as e:
                logger.debug("Metric computation failed for task '%s': %s", name, e)
            try:
                valid_mask = tgts_int >= 0
                labels = sorted(set(tgts_int[valid_mask]))
                cm = _confusion_matrix(tgts_int[valid_mask], pred_classes[valid_mask], labels=labels).tolist()
                metrics[f"confusion_matrix_{name}"] = cm
            except Exception as e:
                logger.debug("Metric computation failed for task '%s': %s", name, e)

        elif task_type == "regression":
            preds_sq = preds.squeeze()
            tgts_sq = tgts.squeeze()
            try:
                metrics[f"mae_{name}"] = float(np.mean(np.abs(preds_sq - tgts_sq)))
            except Exception as e:
                logger.debug("Metric computation failed for task '%s': %s", name, e)
            try:
                metrics[f"rmse_{name}"] = float(np.sqrt(np.mean((preds_sq - tgts_sq) ** 2)))
            except Exception as e:
                logger.debug("Metric computation failed for task '%s': %s", name, e)

        else:
            # Fallback heuristic
            preds_sq = preds.squeeze()
            tgts_sq = tgts.squeeze()
            unique_labels = np.unique(tgts_sq)
            if len(unique_labels) == 2:
                try:
                    metrics[f"auc_{name}"] = roc_auc_score(tgts_sq, preds_sq)
                except ValueError as e:
                    logger.debug("Metric computation failed for task '%s': %s", name, e)
            try:
                metrics[f"mae_{name}"] = float(np.mean(np.abs(preds_sq - tgts_sq)))
            except Exception as e:
                logger.debug("Metric computation failed for task '%s': %s", name, e)

    # Aggregate AUC
    auc_values = [v for k, v in metrics.items() if k.startswith("auc_")]
    if auc_values:
        metrics["auc"] = float(np.mean(auc_values))

    f1_macro_values = [v for k, v in metrics.items() if k.startswith("f1_macro_")]
    if f1_macro_values:
        metrics["f1_macro_avg"] = float(np.mean(f1_macro_values))

    return metrics


# ---------------------------------------------------------------------------
# Phase control
# ---------------------------------------------------------------------------

def apply_phase_config(model, phase, pretrained_uri=None):
    """Configure model for the specified training phase."""
    if phase == "1":
        logger.info("Phase 1: Freezing task towers")
        for name, param in model.named_parameters():
            if "task_towers" in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("  Trainable: %d / %d parameters", trainable, total)

    elif phase == "2":
        logger.info("Phase 2: Unfreezing all parameters")
        for param in model.parameters():
            param.requires_grad = True
        if pretrained_uri:
            logger.info("Loading pre-trained weights from %s", pretrained_uri)
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
        logger.info("Single-phase training (phase='%s')", phase)


def _download_pretrained_from_s3(s3_uri):
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
    logger.info("Extracted pre-trained model to %s", local_dir)


# ---------------------------------------------------------------------------
# Eval report saving
# ---------------------------------------------------------------------------

def save_eval_report(
    output_dir, model, trainer, tasks, task_type_map,
    final_metrics, label_stats, hp, label_schema, feature_schema,
    start_time, epochs, ple_config,
):
    """Save eval_metrics.json with full reproducibility info."""
    report_path = os.path.join(output_dir, "eval_metrics.json")
    os.makedirs(output_dir, exist_ok=True)

    task_names = [t["name"] for t in tasks]
    task_name = hp.get("task_name", "default")

    # Aggregate score
    auc_keys = [k for k in final_metrics if k.startswith("auc_")]
    aggregate_score = (
        sum(final_metrics[k] for k in auc_keys) / len(auc_keys)
        if auc_keys else final_metrics.get("auc", 0.0)
    )

    eval_report: Dict[str, Any] = {
        "task_name": task_name,
        "phase": hp.get("phase", "single"),
        "final_metrics": final_metrics,
        "aggregate_score": aggregate_score,
        "epochs_trained": trainer.current_epoch,
        "total_time_seconds": time.time() - start_time,
    }

    # Training config
    eval_report["training_config"] = {
        "batch_size": int(hp.get("batch_size", 2048)),
        "learning_rate": float(hp.get("learning_rate", 1e-3)),
        "epochs_configured": epochs,
        "seed": int(hp.get("seed", 42)),
        "max_rows": int(hp.get("max_rows", 0)),
    }

    # Architecture
    eval_report["architecture"] = {
        "input_dim": ple_config.input_dim,
        "num_tasks": len(tasks),
        "active_tasks": task_names,
        "num_shared_experts": ple_config.num_shared_experts,
        "shared_expert_names": (
            list(ple_config.expert_basket.shared_experts)
            if ple_config.expert_basket else None
        ),
        "num_layers": ple_config.num_extraction_layers,
        "extraction_dim": getattr(ple_config, "extraction_dim",
                                  getattr(ple_config, "task_expert_output_dim", None)),
        "loss_weighting": ple_config.loss_weighting.strategy,
        "adatt_enabled": ple_config.adatt.enabled if hasattr(ple_config, "adatt") else None,
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    # Logit transfers
    if ple_config.logit_transfers:
        eval_report["architecture"]["logit_transfers"] = [
            {"source": lt.source, "target": lt.target, "method": lt.transfer_method}
            for lt in ple_config.logit_transfers
        ]
    else:
        eval_report["architecture"]["logit_transfers"] = []

    # Data split
    eval_report["data_split"] = {
        "seed": int(hp.get("seed", 42)),
    }

    # Label stats
    eval_report["label_stats"] = label_stats

    # Epoch history
    if hasattr(trainer, "epoch_history") and trainer.epoch_history:
        eval_report["epoch_history"] = trainer.epoch_history

    # Early stopping
    if hasattr(trainer, "early_stop_info") and trainer.early_stop_info:
        eval_report["early_stopping"] = trainer.early_stop_info

    # Task configs
    eval_report["task_configs"] = {
        t["name"]: {
            "type": t.get("type", "binary"),
            "loss": t.get("loss", "default"),
            "loss_weight": t.get("loss_weight", 1.0),
            "num_classes": t.get("num_classes", 1),
        }
        for t in tasks
    }

    # Per-task metrics
    eval_report["per_task"] = {}
    for t in tasks:
        tname = t["name"]
        task_metrics = {k: v for k, v in final_metrics.items() if k.endswith(f"_{tname}")}
        eval_report["per_task"][tname] = {"type": t.get("type", "binary"), "metrics": task_metrics}
    eval_report["per_task_metrics"] = eval_report["per_task"]

    # Ablation metadata
    ablation_type = hp.get("ablation_type", "")
    if ablation_type:
        eval_report["ablation"] = {
            "ablation_type": ablation_type,
            "ablation_scenario": hp.get("ablation_scenario", ""),
            "removed_feature_groups": json.loads(hp.get("removed_feature_groups", "[]"))
                if isinstance(hp.get("removed_feature_groups", "[]"), str) else hp.get("removed_feature_groups", []),
            "shared_experts": hp.get("shared_experts"),
            "num_layers": ple_config.num_extraction_layers,
            "temperature": float(hp["temperature"]) if hp.get("temperature") is not None else None,
            "active_tasks": hp.get("active_tasks"),
            "use_ple": hp.get("use_ple"),
            "use_adatt": hp.get("use_adatt"),
        }

    # Final adaTT state
    if hasattr(model, "adatt") and model.adatt is not None:
        try:
            affinity = model.adatt.get_transfer_matrix()
            if affinity is not None:
                n = len(task_names)
                eval_report["adatt_final"] = {
                    "affinity_matrix": [[round(affinity[i, j].item(), 4) for j in range(n)] for i in range(n)],
                    "task_names": task_names,
                }
        except Exception:
            pass

    # Final loss weights
    if hasattr(model, "loss_weighting") and model.loss_weighting is not None:
        try:
            weights = model.get_loss_weights()
            if weights:
                eval_report["final_loss_weights"] = {k: round(v, 4) for k, v in weights.items()}
        except Exception:
            pass

    # Pos weights
    if hasattr(model, "_pos_weights") and model._pos_weights:
        eval_report["pos_weights"] = {k: round(v.item(), 2) for k, v in model._pos_weights.items()}

    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    logger.info("Eval report saved to %s", report_path)

    # Upload to S3 if configured
    _s3_output = hp.get("_s3_output", "")
    if _s3_output:
        try:
            import boto3
            s3_client = boto3.client("s3")
            parts = _s3_output.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            s3_key = (parts[1] if len(parts) > 1 else "").rstrip("/") + "/output/eval_metrics.json"
            s3_client.put_object(
                Bucket=bucket, Key=s3_key,
                Body=json.dumps(eval_report, indent=2),
                ContentType="application/json",
            )
            logger.info("Uploaded eval_metrics.json to s3://%s/%s", bucket, s3_key)
        except Exception as e:
            logger.warning("Failed to upload eval_metrics.json to S3: %s", e)

    return eval_report


# ---------------------------------------------------------------------------
# PipelineRunner entry point
# ---------------------------------------------------------------------------

def main_pipeline(config_path: str, **overrides) -> dict:
    """Entry point when called via PipelineRunner."""
    from core.pipeline.config import load_config
    from core.pipeline.runner import PipelineRunner

    config = load_config(config_path)

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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """SageMaker training entry point.

    Receives training-ready data from Phase 0. No preprocessing.
    """
    start_time = time.time()

    # ---- 1. Parse hyperparameters ----
    hp = get_hyperparameters()

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    train_dir = os.environ.get(
        "SM_CHANNEL_TRAIN",
        os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
    )
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    checkpoint_dir = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
    num_gpus = int(os.environ.get("SM_NUM_GPUS", "0"))

    batch_size = int(hp.get("batch_size", 2048))
    epochs = int(hp.get("epochs", 20))
    lr = float(hp.get("learning_rate", 1e-3))
    patience = int(hp.get("early_stopping_patience", 5))
    seed = int(hp.get("seed", 42))
    phase = str(hp.get("phase", "single"))
    use_amp = str(hp.get("amp", "false")).lower() in ("true", "1", "yes")
    grad_accum_steps = int(hp.get("gradient_accumulation_steps", 4))
    task_name = hp.get("task_name", "default")
    pretrained_uri = hp.get("pretrained_model_uri")

    logger.info("=" * 60)
    logger.info("SageMaker PLE Training (minimal, schema-driven)")
    logger.info("=" * 60)
    logger.info("GPUs: %d, Device: %s", num_gpus,
                "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Train dir: %s", train_dir)
    logger.info("Epochs: %d, Batch: %d, LR: %s, Phase: %s",
                epochs, batch_size, lr, phase)

    # ---- Seed ----
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        logger.info("GPU: %s, VRAM: %.1f GB, Compute: %d.%d",
                     props.name, props.total_mem / 1e9, props.major, props.minor)
        logger.info("GPU memory: %.1f MB allocated, %.1f MB reserved",
                     torch.cuda.memory_allocated() / 1e6, torch.cuda.memory_reserved() / 1e6)

    # ---- 2. Load training-ready data ----
    data = load_ready_data(train_dir)
    features = data["features"]
    labels = data["labels"]
    sequences = data["sequences"]
    seq_lengths = data["seq_lengths"]
    feature_schema = data["feature_schema"]
    label_schema = data["label_schema"]
    split_indices = data["split_indices"]

    # Merge YAML config into label_schema if present (backward compat)
    config_str = hp.get("config", "{}")
    if isinstance(config_str, dict):
        config = config_str
    elif isinstance(config_str, str) and (config_str.endswith(".yaml") or config_str.endswith(".yml")):
        import yaml
        config_path = Path(config_str)
        if not config_path.exists():
            config_path = Path("/opt/ml/code") / config_str
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            logger.info("Config loaded from YAML: %s", config_path)
        else:
            config = {}
    else:
        config = json.loads(config_str) if config_str else {}

    # If label_schema has no tasks, fall back to YAML config tasks
    if not label_schema.get("tasks") and config.get("tasks"):
        label_schema["tasks"] = config["tasks"]
        label_schema["task_relationships"] = config.get("task_relationships", [])
        label_schema["task_groups"] = config.get("task_groups", [])
        label_schema["task_group_map"] = config.get("task_group_map", {})
        label_schema["model"] = config.get("model", {})
        label_schema["logit_transfer_strength"] = config.get("logit_transfer_strength", 0.5)
        logger.info("Using tasks from YAML config: %d tasks", len(label_schema["tasks"]))

    # If label_schema has no model config, merge from YAML
    if not label_schema.get("model") and config.get("model"):
        label_schema["model"] = config["model"]

    # ---- 3. Apply ablation filters ----
    features, labels, feature_schema, label_schema = apply_ablation(
        features, labels, feature_schema, label_schema, hp,
    )

    # If labels is None, try to extract label columns from features DataFrame
    if labels is None:
        import pandas as pd
        tasks = label_schema.get("tasks", [])
        label_cols_present = [t["label_col"] for t in tasks if t["label_col"] in features.columns]
        if label_cols_present:
            labels = features[label_cols_present].copy()
            features = features.drop(columns=label_cols_present, errors="ignore")
            logger.info("Extracted %d label columns from features DataFrame", len(label_cols_present))

    # ---- 4. Build DataLoaders ----
    train_loader, val_loader, tasks, task_type_map, label_stats = build_dataloaders(
        features, labels, sequences, seq_lengths,
        feature_schema, label_schema, split_indices, hp,
    )
    task_names = [t["name"] for t in tasks]

    # ---- Label distribution ----
    logger.info("=== Label Distribution ===")
    for task in tasks:
        col = task["label_col"]
        if labels is not None and col in labels.columns:
            vals = labels[col]
            if task["type"] == "binary":
                pos_rate = (vals > 0).mean()
                logger.info("  %s [binary]: positive=%.2f%% (%d/%d)",
                            task["name"], pos_rate * 100, (vals > 0).sum(), len(vals))
            elif task["type"] == "multiclass":
                logger.info("  %s [multi-%s]: top classes=%s",
                            task["name"], task.get("num_classes", "?"),
                            dict(vals.value_counts().head(5)))
            else:
                logger.info("  %s [regression]: mean=%.4f, std=%.4f, range=[%.4f, %.4f]",
                            task["name"], vals.mean(), vals.std(), vals.min(), vals.max())

    # ---- Label sanity check ----
    try:
        _peek = next(iter(train_loader))
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

    # ---- 5. Build model ----
    # Auto-detect input_dim from actual data
    _auto_input_dim = None
    try:
        _sample_batch = next(iter(train_loader))
        if isinstance(_sample_batch, dict) and "features" in _sample_batch:
            _auto_input_dim = _sample_batch["features"].shape[-1]
    except StopIteration:
        pass

    if _auto_input_dim is not None:
        input_dim = _auto_input_dim
        logger.info("Model input_dim (auto-detected): %d", input_dim)
    else:
        input_dim = len(feature_schema.get("columns", []))
        logger.info("Model input_dim (from schema): %d", input_dim)

    model, ple_config = build_model(feature_schema, label_schema, hp, input_dim, device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d parameters (%.1f MB), estimated GPU memory: %.1f MB",
                 n_params, n_params * 4 / 1e6, n_params * 4 * 3 / 1e6)  # params + grads + optim

    # ---- Apply phase config (load pretrained for phase 2) ----
    if phase == "2" and pretrained_uri:
        apply_phase_config(model, phase, pretrained_uri)

    # ---- 6. Train with PLETrainer ----
    from core.training.checkpoint import CheckpointManager

    ckpt_mgr = CheckpointManager(local_dir=checkpoint_dir, max_keep=3)

    _phase_map = {"single": "full", "1": "phase1", "2": "phase2", "full": "full"}
    trainer_phase = _phase_map.get(phase, "full")

    # Build TrainingConfig
    training_cfg_dict: Dict[str, Any] = {
        "batch_size": batch_size,
        "optimizer": {"name": "adamw", "learning_rate": lr, "weight_decay": 0.01},
        "scheduler": {
            "name": "cosine",
            "warmup_epochs": 3,
            "cosine_t0": max(10, epochs // 3),
            "cosine_t_mult": 2,
            "phase2_warmup_epochs": 2,
            "phase2_cosine_t0": max(6, epochs // 5),
        },
        "amp": {"enabled": use_amp},
        "gradient": {"clip_norm": 5.0, "accumulation_steps": grad_accum_steps},
        "early_stopping": {"enabled": True, "patience": patience},
        "checkpoint": {"dir": checkpoint_dir, "save_every_n_epochs": 1, "max_to_keep": 3},
        "phase1": {"epochs": epochs if trainer_phase in ("phase1", "full") else 0},
        "phase2": {"epochs": epochs if trainer_phase in ("phase2", "full") else 0},
        "experiment_name": task_name,
    }

    # Merge YAML training block if present
    yaml_training = config.get("training", {})
    if yaml_training:
        import copy
        merged = copy.deepcopy(yaml_training)
        merged.setdefault("optimizer", {}).update(training_cfg_dict["optimizer"])
        merged.setdefault("gradient", {}).update(training_cfg_dict["gradient"])
        merged.setdefault("amp", {}).update(training_cfg_dict["amp"])
        merged.setdefault("early_stopping", {}).update(training_cfg_dict["early_stopping"])
        merged.setdefault("checkpoint", {}).update(training_cfg_dict["checkpoint"])
        merged["batch_size"] = batch_size
        merged["experiment_name"] = task_name
        merged.setdefault("phase1", {})["epochs"] = training_cfg_dict["phase1"]["epochs"]
        merged.setdefault("phase2", {})["epochs"] = training_cfg_dict["phase2"]["epochs"]
        training_config = TrainingConfig.from_dict(merged)
    else:
        training_config = TrainingConfig.from_dict(training_cfg_dict)

    logger.info(
        "TrainingConfig: lr=%.5f, AMP=%s, phase1_epochs=%d, phase2_epochs=%d",
        training_config.optimizer.learning_rate,
        training_config.amp.enabled,
        training_config.phase1.epochs,
        training_config.phase2.epochs,
    )

    # Ensure all params unfrozen so PLETrainer manages freeze/unfreeze
    for param in model.parameters():
        param.requires_grad = True

    trainer = PLETrainer(model=model, config=training_config, device=device)

    # Resume from checkpoint (Spot restart)
    resume_state = ckpt_mgr.load_latest(
        model, trainer.optimizer, trainer.scheduler, map_location=device,
    )
    if resume_state:
        trainer.current_epoch = resume_state.get("epoch", 0)
        trainer.global_step = resume_state.get("global_step", 0)
        logger.info(
            "CHECKPOINT RESTORED: epoch %d, global_step %d from %s",
            trainer.current_epoch, trainer.global_step, ckpt_mgr.local_dir,
        )
    else:
        logger.info("No checkpoint found, starting from scratch")

    trainer_results = trainer.train(train_loader, val_loader, phase=trainer_phase)
    logger.info("PLETrainer finished: %s", trainer_results)

    # ---- 7. Final validation + save ----
    final_metrics = validate(model, val_loader, device, task_names, task_type_map)
    report_metrics("final_val", final_metrics, epochs)

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pth")
    save_dict = {
        "model_state_dict": model.state_dict(),
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
    ablation_type = hp.get("ablation_type", "")
    if ablation_type:
        save_dict["ablation"] = {
            "ablation_type": ablation_type,
            "ablation_scenario": hp.get("ablation_scenario", ""),
            "removed_feature_groups": json.loads(hp.get("removed_feature_groups", "[]"))
                if isinstance(hp.get("removed_feature_groups", "[]"), str) else hp.get("removed_feature_groups", []),
            "shared_experts": hp.get("shared_experts"),
            "num_layers": ple_config.num_extraction_layers,
            "temperature": float(hp["temperature"]) if hp.get("temperature") is not None else None,
        }
    torch.save(save_dict, model_path)
    logger.info("Model saved to %s", model_path)

    # Save config
    config_path_out = os.path.join(model_dir, "config.json")
    with open(config_path_out, "w") as f:
        json.dump({"feature_schema": feature_schema, "label_schema": label_schema}, f, indent=2)

    # Save eval report
    save_eval_report(
        output_dir, model, trainer, tasks, task_type_map,
        final_metrics, label_stats, hp, label_schema, feature_schema,
        start_time, epochs, ple_config,
    )

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Training complete in %.1fs", total_time)
    logger.info("Final metrics: %s", final_metrics)
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse as _argparse
    import traceback

    _parser = _argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--pipeline", type=str, default=None,
                         help="Path to pipeline YAML config.")
    _known, _remaining = _parser.parse_known_args()

    try:
        if _known.pipeline:
            logger.info("Running via PipelineRunner (config=%s)", _known.pipeline)
            main_pipeline(_known.pipeline)
        else:
            main()
    except Exception:
        logger.exception("FATAL: training failed")
        traceback.print_exc()
        sys.exit(1)
