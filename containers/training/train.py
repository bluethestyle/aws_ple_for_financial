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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure duckdb is installed in SageMaker container
try:
    import duckdb  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "duckdb>=1.0.0"])
    import duckdb  # noqa: F401

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

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

def _detect_scalar_cols_duckdb(con, parquet_uri: str) -> list:
    """Use DuckDB DESCRIBE to find scalar (non-list/struct/string) columns."""
    schema_df = con.execute(
        f"DESCRIBE SELECT * FROM '{parquet_uri}'"
    ).fetchall()
    return [
        row[0] for row in schema_df
        if not row[1].endswith("[]")
        and "STRUCT" not in row[1].upper()
        and row[1].upper() not in ("VARCHAR", "DATE", "TIMESTAMP")
    ]


def _try_cudf_available() -> bool:
    """Check if cuDF is importable and a CUDA GPU is accessible."""
    try:
        import cudf  # noqa: F401
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def load_ready_data(channel_dir: str) -> dict:
    """Load training-ready artifacts produced by Phase 0.

    Returns PyArrow Tables (zero-copy from parquet). No pandas in the hot path.

    Returns
    -------
    dict with keys:
        features : pyarrow.Table — normalized numeric features
        labels : pyarrow.Table or None — derived labels
        sequences : np.ndarray or None — padded 3D tensor
        seq_lengths : np.ndarray or None — sequence lengths
        feature_schema : dict — column names, group ranges, expert routing
        label_schema : dict — task definitions
        split_indices : dict or None — train/val/test row indices
    """
    import duckdb

    channel_path = Path(channel_dir)
    logger.info("Using PyArrow for training data loading (zero-copy parquet)")

    con = duckdb.connect()

    # -- Features --
    features_path = channel_path / "features.parquet"
    if features_path.exists():
        parquet_uri = str(features_path).replace("\\", "/")
        # Identify scalar columns via DuckDB DESCRIBE (lightweight, no data load)
        scalar_cols = _detect_scalar_cols_duckdb(con, parquet_uri)

        # PyArrow native parquet read (no pandas, no DuckDB data copy)
        features = pq.read_table(str(features_path), columns=scalar_cols)
        logger.info("Loaded features via PyArrow: %d rows, %d columns",
                    features.num_rows, features.num_columns)
    else:
        # Fallback: load from generic parquet files (backward compat)
        parquet_files = sorted(channel_path.glob("**/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No features.parquet or .parquet files in {channel_dir}")
        first_uri = str(parquet_files[0]).replace("\\", "/")
        scalar_cols = _detect_scalar_cols_duckdb(con, first_uri)
        logger.info("Selecting %d scalar columns (skipping list/struct)", len(scalar_cols))

        tables = [pq.read_table(str(f), columns=scalar_cols) for f in parquet_files]
        features = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        logger.info("Loaded %d parquet files via PyArrow (fallback): %d rows, %d columns",
                    len(parquet_files), features.num_rows, features.num_columns)

    # Data loading diagnostics (Arrow Table)
    n_rows, n_cols = features.num_rows, features.num_columns
    mem_bytes = sum(
        features.column(i).nbytes for i in range(features.num_columns)
    )
    logger.info("Features: %d rows x %d cols, memory=%.1f MB", n_rows, n_cols, mem_bytes / 1e6)
    # Dtype summary
    dtype_counts: Dict[str, int] = {}
    for i in range(features.num_columns):
        dt = str(features.schema.field(i).type)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
    logger.info("Dtypes: %s", dtype_counts)
    # NaN diagnostics
    nan_count = 0
    nan_cols_count = 0
    for i in range(features.num_columns):
        col_nulls = features.column(i).null_count
        if col_nulls > 0:
            nan_count += col_nulls
            nan_cols_count += 1
    if nan_count > 0:
        logger.warning("Features contain %d NaN values across %d columns",
                        nan_count, nan_cols_count)

    # -- Labels --
    labels_path = channel_path / "labels.parquet"
    if labels_path.exists():
        labels = pq.read_table(str(labels_path))
    else:
        labels = None
    if labels is not None:
        logger.info("Loaded labels via PyArrow: %d rows, %d columns",
                     labels.num_rows, labels.num_columns)

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
        # Auto-generate minimal schema from Arrow Table column names
        feature_schema = {
            "columns": features.column_names,
            "group_ranges": {},
            "expert_routing": {},
        }
        logger.warning("No feature_schema.json found -- auto-generated from Arrow Table columns")

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

    # Close DuckDB connection — all data is now in PyArrow Tables
    con.close()

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

    Accepts PyArrow Tables for features and labels.

    Modifies features and schemas in-place based on hyperparameters:
      - removed_feature_groups: drop columns by group using schema["group_ranges"]
      - active_tasks: filter tasks to only active ones
      - use_ple / use_adatt: stored as config overrides (handled at model build)

    Returns
    -------
    features, labels, feature_schema, label_schema (filtered)
    """
    # -- Feature ablation: drop columns by group using schema["feature_group_ranges"] --
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

    group_ranges = feature_schema.get("feature_group_ranges", feature_schema.get("group_ranges", {}))
    columns = feature_schema.get("columns", features.column_names)

    if removed:
        cols_to_drop = set()
        for group_name in removed:
            if group_name in group_ranges:
                start, end = group_ranges[group_name]
                cols_to_drop.update(columns[start:end])
            else:
                logger.warning("Ablation: group '%s' not in schema group_ranges", group_name)
        if cols_to_drop:
            # Arrow Table: drop columns
            remaining_cols = [c for c in features.column_names if c not in cols_to_drop]
            features = features.select(remaining_cols)
            # Update schema to reflect ablation (critical for model input_dim)
            feature_schema["columns"] = [c for c in columns if c not in cols_to_drop]
            feature_schema["num_features"] = len(feature_schema["columns"])
            # Update group_ranges: remove dropped groups, recompute offsets
            new_fgr = {}
            offset = 0
            for gname, (s, e) in sorted(group_ranges.items(), key=lambda x: x[1][0]):
                kept = [c for c in columns[s:e] if c not in cols_to_drop]
                if kept:
                    new_fgr[gname] = [offset, offset + len(kept)]
                    offset += len(kept)
            feature_schema["feature_group_ranges"] = new_fgr
            logger.info("Ablation: removed %d columns from groups %s. Remaining: %d features, %d groups",
                         len(cols_to_drop), removed, features.num_columns, len(new_fgr))
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

    # -- Filter labels Table to only active task label columns --
    if labels is not None and tasks:
        label_cols = [t["label_col"] for t in tasks if t["label_col"] in labels.column_names]
        labels = labels.select(label_cols)

    return features, labels, feature_schema, label_schema


# ---------------------------------------------------------------------------
# DataLoader construction
# ---------------------------------------------------------------------------

def build_dataloaders(features, labels, sequences, seq_lengths, feature_schema,
                      label_schema, split_indices, hp, config=None):
    """Build train/val DataLoaders from training-ready data.

    Returns
    -------
    train_loader, val_loader, tasks, task_type_map, label_stats
    """
    from core.data.dataloader import build_ple_dataloader, FeatureColumnSpec

    tasks = label_schema.get("tasks", [])
    batch_size = int(hp.get("batch_size", 2048))

    # Subsample if max_rows HP is set (for fast testing)
    max_rows = int(hp.get("max_rows", 0))
    if max_rows and max_rows > 0 and features.num_rows > max_rows:
        # Use numpy random choice for index-based subsampling
        rng = np.random.RandomState(42)
        idx = rng.choice(features.num_rows, size=max_rows, replace=False)
        idx.sort()
        features = features.take(idx)
        if labels is not None:
            labels = labels.take(idx)
        if sequences is not None:
            sequences = sequences[idx]
        if seq_lengths is not None:
            seq_lengths = seq_lengths[idx]
        logger.info("Subsampled to %d rows for fast testing", max_rows)

    # Validate tasks against available label columns
    if labels is not None:
        available_labels = set(labels.column_names)
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

    # -- Merge features + labels into a single Arrow Table for PLEDataset --
    # Capture feature column names before merge (needed for FeatureColumnSpec)
    _feature_col_names = features.column_names

    # Append label columns to features Arrow Table (zero-copy column concat)
    merged = features
    if labels is not None and labels.num_columns > 0:
        for col_name in labels.column_names:
            merged = merged.append_column(col_name, labels.column(col_name))
        logger.info("Merged features+labels via PyArrow: %d rows, %d columns",
                     merged.num_rows, merged.num_columns)

    # Release original tables to free memory
    del features

    # -- Build FeatureColumnSpec from schema --
    feature_columns = feature_schema.get("columns", _feature_col_names)
    # Only include columns actually present in the Table (post-ablation)
    merged_cols_set = set(merged.column_names)
    static_features = [c for c in feature_columns if c in merged_cols_set]

    feature_spec = FeatureColumnSpec(static_features=static_features)
    logger.info("FeatureColumnSpec: %d static features", len(static_features))

    # -- Pre-compute label statistics (Arrow compute, no pandas) --
    label_stats = {}
    for t in tasks:
        lc = t["label_col"]
        if lc in merged_cols_set:
            col_arr = merged.column(lc)
            n_total = merged.num_rows
            if t.get("type") == "binary":
                # Count values > 0.5
                n_pos = int(pc.sum(pc.greater(col_arr, 0.5)).as_py())
                label_stats[t["name"]] = {
                    "positive_count": n_pos,
                    "positive_rate": round(n_pos / n_total, 4),
                    "total": n_total,
                }
            elif t.get("type") == "regression":
                label_stats[t["name"]] = {
                    "mean": round(float(pc.mean(col_arr).as_py()), 4),
                    "std": round(float(pc.stddev(col_arr).as_py()), 4),
                    "total": n_total,
                }
            elif t.get("type") == "multiclass":
                label_stats[t["name"]] = {
                    "num_classes": int(pc.count_distinct(col_arr).as_py()),
                    "total": n_total,
                }

    # -- Split into train/val --
    use_gpu_loading = (
        hp.get("use_gpu_loading", False)
        and int(os.environ.get("SM_NUM_GPUS", "0")) > 0
    )

    if split_indices:
        train_idx = split_indices.get("train", [])
        val_idx = split_indices.get("val", split_indices.get("validation", []))

        tbl_train = merged.take(train_idx)
        tbl_val = merged.take(val_idx) if val_idx else None

        train_loader = build_ple_dataloader(
            df=tbl_train, feature_spec=feature_spec, label_columns=label_map,
            batch_size=batch_size, shuffle=True, use_gpu_loading=use_gpu_loading,
        )

        val_loader = None
        if tbl_val is not None and tbl_val.num_rows > 0:
            val_loader = build_ple_dataloader(
                df=tbl_val, feature_spec=feature_spec, label_columns=label_map,
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
                     tbl_train.num_rows, tbl_val.num_rows if tbl_val is not None else 0)
    else:
        # No split indices — build single loader then split
        full_loader = build_ple_dataloader(
            df=merged, feature_spec=feature_spec, label_columns=label_map,
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
        _dl_cfg = (config or {}).get("ablation", {}).get("training_defaults", {})
        _nw = int(_dl_cfg.get("num_workers", 2))
        _pm = bool(_dl_cfg.get("pin_memory", True))
        _dl = bool(_dl_cfg.get("drop_last", True))
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            collate_fn=full_loader.collate_fn, num_workers=_nw,
            pin_memory=_pm, drop_last=_dl,
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            collate_fn=full_loader.collate_fn, num_workers=_nw,
            pin_memory=_pm,
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
        GroupTaskExpertConfig, AdaTTConfig, TaskGroupDef,
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

    # -- Expert ablation: active_experts / removed_experts --
    # These hyperparameters filter expert_basket.expert_configs before model build.
    # active_experts: JSON list of expert names to KEEP (remove all others)
    # removed_experts: JSON list of expert names to REMOVE
    if expert_basket is not None:
        _active_raw = hp.get("active_experts")
        _removed_raw = hp.get("removed_experts")

        _active_experts = None
        if _active_raw:
            _active_experts = json.loads(_active_raw) if isinstance(_active_raw, str) else _active_raw

        _removed_experts = None
        if _removed_raw:
            _removed_experts = json.loads(_removed_raw) if isinstance(_removed_raw, str) else _removed_raw

        if _active_experts is not None:
            # Keep only experts in the active list
            filtered_shared = [e for e in expert_basket.shared_experts if e in _active_experts]
            filtered_task = [e for e in expert_basket.task_experts if e in _active_experts]
            filtered_configs = {k: v for k, v in expert_basket.expert_configs.items() if k in _active_experts}
            expert_basket = ExpertBasketConfig(
                shared_experts=filtered_shared,
                task_experts=filtered_task,
                expert_configs=filtered_configs,
            )
            num_shared_experts = len(filtered_shared)
            logger.info("Expert ablation (active_experts): kept %s, shared=%d, task=%s",
                        filtered_shared, num_shared_experts, filtered_task)

        elif _removed_experts is not None:
            # Remove specified experts
            filtered_shared = [e for e in expert_basket.shared_experts if e not in _removed_experts]
            filtered_task = [e for e in expert_basket.task_experts if e not in _removed_experts]
            filtered_configs = {k: v for k, v in expert_basket.expert_configs.items() if k not in _removed_experts}
            expert_basket = ExpertBasketConfig(
                shared_experts=filtered_shared,
                task_experts=filtered_task,
                expert_configs=filtered_configs,
            )
            num_shared_experts = len(filtered_shared)
            logger.info("Expert ablation (removed_experts): removed %s, remaining shared=%s, task=%s",
                        _removed_experts, filtered_shared, filtered_task)

    # -- Structure ablation: PLE toggle --
    # use_ple=false: disable PLE layering/CGC gating but KEEP all heterogeneous experts.
    use_ple_raw = hp.get("use_ple")
    if use_ple_raw is not None:
        use_ple = json.loads(use_ple_raw) if isinstance(use_ple_raw, str) else use_ple_raw
        if not use_ple:
            num_extraction_layers = 1
            # Keep num_shared_experts and expert_basket intact
            logger.info("Structure ablation: PLE disabled (single layer, all experts preserved)")

    # -- Structure ablation: adaTT toggle --
    use_adatt_raw = hp.get("use_adatt")
    if use_adatt_raw is not None:
        use_adatt = json.loads(use_adatt_raw) if isinstance(use_adatt_raw, str) else use_adatt_raw
        if not use_adatt:
            model_config["adatt"] = {"enabled": False}
            logger.info("Structure ablation: adaTT disabled")

    # -- Structure ablation: gate type (softmax vs sigmoid) --
    gate_type_raw = hp.get("gate_type", "softmax")
    if hasattr(ple_config, 'gate_type'):
        ple_config.gate_type = gate_type_raw
    else:
        # Dynamically add attribute if config class doesn't have it
        ple_config.gate_type = gate_type_raw
    if gate_type_raw != "softmax":
        logger.info("Structure ablation: gate_type=%s", gate_type_raw)

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

    # -- Per-expert input dimensions from model config --
    expert_input_dims_raw = model_config.get("expert_input_dims", {})
    if expert_input_dims_raw:
        ple_config.expert_input_dims = {
            k: int(v) for k, v in expert_input_dims_raw.items()
        }
        logger.info("Expert input_dim overrides: %s", ple_config.expert_input_dims)

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

    # -- Inject feature_group_ranges into DeepFM expert config ----------------
    # When field_dims="auto", the DeepFM expert reads feature_group_ranges
    # from its config dict to derive per-field boundaries for FM interaction.
    if group_ranges and ple_config.expert_basket is not None:
        deepfm_cfg = ple_config.expert_basket.expert_configs.get("deepfm")
        if deepfm_cfg is not None and deepfm_cfg.get("field_dims") == "auto":
            deepfm_cfg["feature_group_ranges"] = {
                k: tuple(v) for k, v in group_ranges.items()
            }
            logger.info(
                "Injected %d feature_group_ranges into DeepFM expert config "
                "for auto field splitting",
                len(group_ranges),
            )

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

    # -- adaTT task_groups from schema (enables build_task_group_map_from_groups) --
    raw_task_groups = label_schema.get("task_groups", [])
    if raw_task_groups:
        # Try model-level first, then root-level label_schema (where
        # runner.py stores the pipeline.yaml root-level adatt section).
        adatt_cfg_raw = model_config.get("adatt", {})
        if not adatt_cfg_raw:
            adatt_cfg_raw = label_schema.get("adatt", {})
        adatt_task_groups: Dict[str, TaskGroupDef] = {}
        for tg in raw_task_groups:
            tg_name = tg["name"] if isinstance(tg, dict) else tg.name
            tg_tasks = tg["tasks"] if isinstance(tg, dict) else tg.tasks
            tg_intra = (
                tg.get("adatt_intra_strength", 0.7) if isinstance(tg, dict)
                else getattr(tg, "adatt_intra_strength", 0.7)
            )
            adatt_task_groups[tg_name] = TaskGroupDef(
                members=list(tg_tasks),
                intra_strength=tg_intra,
            )

        ple_config.adatt = AdaTTConfig(
            enabled=adatt_cfg_raw.get("enabled", True),
            task_groups=adatt_task_groups,
            inter_group_strength=adatt_cfg_raw.get("inter_group_strength", 0.3),
            transfer_lambda=adatt_cfg_raw.get("transfer_lambda", 0.1),
            temperature=adatt_cfg_raw.get("temperature", 1.0),
            warmup_epochs=adatt_cfg_raw.get("warmup_epochs", 10),
            grad_interval=adatt_cfg_raw.get("grad_interval", 10),
        )
        logger.info(
            "AdaTT task_groups: %d groups (%s)",
            len(adatt_task_groups), list(adatt_task_groups.keys()),
        )

    # -- GroupTaskExpert config from model section --
    gte_cfg_raw = model_config.get("group_task_expert", {})
    if gte_cfg_raw:
        ple_config.group_task_expert = GroupTaskExpertConfig(
            enabled=gte_cfg_raw.get("enabled", True),
            group_hidden_dim=gte_cfg_raw.get("group_hidden_dim",
                                              gte_cfg_raw.get("group_hidden", 128)),
            group_output_dim=gte_cfg_raw.get("group_output_dim",
                                              gte_cfg_raw.get("group_output", 64)),
            cluster_embed_dim=gte_cfg_raw.get("cluster_embed_dim", 32),
            dropout=gte_cfg_raw.get("dropout", 0.2),
        )
        logger.info(
            "GroupTaskExpert: hidden=%d, output=%d, cluster_embed=%d",
            ple_config.group_task_expert.group_hidden_dim,
            ple_config.group_task_expert.group_output_dim,
            ple_config.group_task_expert.cluster_embed_dim,
        )

    # -- HMM group-to-mode mapping from model section --
    hmm_gm_map = model_config.get("hmm_group_mode_map", {})
    if hmm_gm_map:
        ple_config.hmm_group_mode_map = {str(k): str(v) for k, v in hmm_gm_map.items()}

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
# Per-task validation mask builder
# ---------------------------------------------------------------------------

def _build_task_val_masks(
    config: dict,
    features,
    split_indices: Optional[dict],
    label_schema: dict,
) -> Optional[Dict[str, np.ndarray]]:
    """Build per-task boolean masks for the validation set.

    Reads ``data.split_strategy`` from *config*.  For each task group whose
    ``val_method`` is ``temporal_latest``, the mask is ``True`` only for
    rows in the val set whose date equals the latest snapshot_date in that
    val set.  Tasks with ``val_method == random`` (or not listed) use the
    full val set (no mask entry needed).

    Returns
    -------
    dict or None
        Mapping ``task_name -> np.ndarray[bool]`` of length ``len(val_idx)``,
        or ``None`` if no split_strategy is configured.
    """
    split_strategy = config.get("data", {}).get("split_strategy")
    if not split_strategy:
        return None

    # Identify which tasks need temporal_latest masking
    temporal_task_names: set = set()
    for _group_cfg in split_strategy.values():
        if not isinstance(_group_cfg, dict):
            continue
        if _group_cfg.get("val_method") == "temporal_latest":
            for t in _group_cfg.get("tasks", []):
                temporal_task_names.add(t)

    if not temporal_task_names:
        logger.info("split_strategy: all tasks use random val — no masks needed")
        return None

    # We need val indices and a date column to build masks
    if not split_indices:
        logger.info("split_strategy: no split_indices yet — masks deferred to DataLoader")
        return None

    val_idx = split_indices.get("val", split_indices.get("validation", []))
    if not val_idx:
        logger.info("split_strategy: empty val set — no masks needed")
        return None

    date_col = config.get("data", {}).get("temporal_split", {}).get(
        "date_col",
        config.get("data", {}).get("date_col", "snapshot_date"),
    )

    # Try to get the date values for val rows
    _val_dates = None
    if date_col in features.column_names:
        _val_dates = features.column(date_col).take(val_idx).to_numpy(zero_copy_only=False)
    else:
        # Try loading from parquet via DuckDB (date may have been separated)
        try:
            import duckdb as _ddb_vm
            from pathlib import Path as _PathVM
            _parquet_paths = list(_PathVM(os.environ.get("SM_CHANNEL_TRAIN", ".")).glob("**/*.parquet"))
            if _parquet_paths:
                _uri = str(_parquet_paths[0]).replace("\\", "/")
                _con_vm = _ddb_vm.connect()
                try:
                    _all_dates_arrow = _con_vm.execute(
                        f'SELECT "{date_col}" FROM \'{_uri}\''
                    ).arrow()
                    _all_dates = _all_dates_arrow.column(0).to_numpy(zero_copy_only=False)
                    _val_dates = _all_dates[val_idx]
                except Exception:
                    pass
                finally:
                    _con_vm.close()
        except Exception:
            pass

    if _val_dates is None:
        logger.warning(
            "split_strategy: date column '%s' not available — cannot build temporal masks",
            date_col,
        )
        return None

    # Find latest date in val set (numpy datetime ops, no pandas)
    _val_dates_np = np.asarray(_val_dates, dtype="datetime64[ns]")
    _latest_date = np.max(_val_dates_np)
    _latest_mask = (_val_dates_np == _latest_date)  # bool array, len = len(val_idx)

    n_latest = int(_latest_mask.sum())
    n_val = len(val_idx)
    logger.info(
        "split_strategy: temporal_latest mask = %d/%d val rows (date=%s) for %d tasks",
        n_latest, n_val, _latest_date, len(temporal_task_names),
    )

    # Build masks dict — only for temporal_latest tasks
    task_val_masks: Dict[str, np.ndarray] = {}
    for t_name in temporal_task_names:
        task_val_masks[t_name] = _latest_mask

    # Log summary of which tasks use which strategy
    all_task_names = {t["name"] for t in label_schema.get("tasks", [])}
    random_tasks = all_task_names - temporal_task_names
    logger.info(
        "split_strategy: random val (%d tasks): %s",
        len(random_tasks), sorted(random_tasks),
    )
    logger.info(
        "split_strategy: temporal_latest val (%d tasks): %s",
        len(temporal_task_names), sorted(temporal_task_names),
    )

    return task_val_masks


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, dataloader, device, task_names, task_type_map=None,
             task_val_masks=None):
    """Run validation and compute per-task metrics.

    Parameters
    ----------
    task_val_masks : dict or None
        Optional mapping of task_name -> np.ndarray (bool) for per-task
        validation subset filtering.

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

        # Apply per-task validation mask if configured
        if task_val_masks is not None and name in task_val_masks:
            _mask = task_val_masks[name]
            _n = min(len(_mask), len(preds))
            _mask_t = _mask[:_n]
            preds = preds[_mask_t]
            tgts = tgts[_mask_t]
            if len(preds) == 0:
                logger.debug("task_val_mask[%s]: 0 samples after masking, skipping", name)
                continue

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
    grad_accum_steps = int(hp.get("gradient_accumulation_steps", 1))
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
        vram_bytes = getattr(props, "total_memory", 0) or getattr(props, "total_mem", 0)
        logger.info("GPU: %s, VRAM: %.1f GB, Compute: %d.%d",
                     props.name, vram_bytes / 1e9, props.major, props.minor)
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
            with open(config_path, encoding="utf-8") as f:
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

    # If label_schema has no training config, merge from YAML
    if not label_schema.get("training") and config.get("training"):
        label_schema["training"] = config["training"]

    # ---- Re-apply training defaults from YAML config ----
    # HP defaults (batch_size, epochs, lr, etc.) were set with generic
    # fallbacks before label_schema was loaded.  Now that we have the
    # full config, override with YAML values when HP was not explicitly
    # passed by the SageMaker hyperparameter channel.
    training_yaml = label_schema.get("training", config.get("training", {}))
    if training_yaml:
        if "batch_size" not in hp:
            batch_size = int(training_yaml.get("batch_size", batch_size))
        if "epochs" not in hp:
            epochs = int(training_yaml.get("epochs", epochs))
        if "learning_rate" not in hp:
            lr = float(training_yaml.get("learning_rate", lr))
        if "early_stopping_patience" not in hp:
            es_cfg = training_yaml.get("early_stopping", {})
            patience = int(es_cfg.get("patience", patience))
        if "seed" not in hp:
            seed = int(training_yaml.get("seed", seed))
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        logger.info(
            "Training defaults from YAML: epochs=%d, batch=%d, lr=%s, patience=%d",
            epochs, batch_size, lr, patience,
        )

    # ---- 3. Apply ablation filters ----
    features, labels, feature_schema, label_schema = apply_ablation(
        features, labels, feature_schema, label_schema, hp,
    )

    # If labels is None, try to extract label columns from features Arrow Table
    if labels is None:
        tasks = label_schema.get("tasks", [])
        _feat_cols_set = set(features.column_names)
        label_cols_present = [t["label_col"] for t in tasks if t["label_col"] in _feat_cols_set]
        if label_cols_present:
            labels = features.select(label_cols_present)
            remaining_cols = [c for c in features.column_names if c not in set(label_cols_present)]
            features = features.select(remaining_cols)
            logger.info("Extracted %d label columns from features Arrow Table", len(label_cols_present))


    # ---- 3b. Split strategy: temporal (if multi-date) or random (cross-sectional) ----
    if split_indices is None or not split_indices:
        date_col = config.get("data", {}).get("date_col", "snapshot_date")
        _date_arr = None  # numpy array of dates
        _use_temporal = False

        # Try to load date column from parquet
        if date_col not in features.column_names:
            import duckdb as _ddb_date
            _parquet_path = list(Path(train_dir).glob("**/*.parquet"))
            if _parquet_path:
                _uri = str(_parquet_path[0]).replace("\\", "/")
                _con_d = _ddb_date.connect()
                try:
                    _date_arrow = _con_d.execute(f'SELECT "{date_col}" FROM \'{_uri}\'').arrow()
                    _date_arr = _date_arrow.column(0).to_numpy(zero_copy_only=False)
                except Exception:
                    pass
                finally:
                    _con_d.close()

        # Detect if temporal split is appropriate:
        # If >80% of rows share the same date, it's cross-sectional → random split
        _has_dates = date_col in features.column_names or _date_arr is not None
        if _has_dates:
            if date_col in features.column_names:
                _dates_col = features.column(date_col)
            else:
                _dates_col = pa.array(_date_arr)
            # Compute value_counts via Arrow
            _vc = pc.value_counts(_dates_col)
            _max_count = max(entry["count"].as_py() for entry in _vc) if len(_vc) > 0 else len(_dates_col)
            _top_date_ratio = _max_count / len(_dates_col) if len(_dates_col) > 0 else 1.0
            if _top_date_ratio < 0.8:
                _use_temporal = True
                logger.info("Multi-date data (top date=%.1f%%) → temporal split", _top_date_ratio * 100)
            else:
                logger.info("Cross-sectional data (top date=%.1f%%) → random split", _top_date_ratio * 100)

        if _use_temporal:
            import duckdb as _ddb_split

            _ts_cfg = config.get("data", {}).get("temporal_split", {})
            gap_days = int(_ts_cfg.get("gap_days", 1))
            train_ratio = 0.7
            val_ratio = 0.15

            # Build a date array aligned to feature rows
            if date_col in features.column_names:
                _dates_np = features.column(date_col).to_numpy(zero_copy_only=False)
            else:
                _dates_np = _date_arr

            # Create a minimal Arrow Table with original row index and date
            _idx_table = pa.table({
                "_orig_idx": pa.array(range(features.num_rows), type=pa.int64()),
                "_split_date": pa.array(np.asarray(_dates_np, dtype="datetime64[ns]")),
            })

            _con_s = _ddb_split.connect()
            try:
                _con_s.register("idx_df", _idx_table)

                # Compute min/max date and total span via DuckDB
                _bounds = _con_s.execute("""
                    SELECT
                        MIN(_split_date) AS min_dt,
                        MAX(_split_date) AS max_dt,
                        DATE_DIFF('day', MIN(_split_date), MAX(_split_date)) AS total_span
                    FROM idx_df
                """).fetchone()
                _min_dt, _max_dt, _total_span = _bounds

                if _total_span is not None and _total_span > 0:
                    # Compute cutoff dates matching TemporalSplitter logic
                    _train_end_days = int(_total_span * train_ratio)
                    _val_span_days = int(_total_span * val_ratio)

                    # Query DuckDB for train and val original indices,
                    # sorted by date so the output is temporally ordered.
                    _split_arrow = _con_s.execute(f"""
                        WITH cutoffs AS (
                            SELECT
                                CAST('{_min_dt}' AS TIMESTAMP) + INTERVAL '{_train_end_days} days'
                                    AS train_end,
                                CAST('{_min_dt}' AS TIMESTAMP) + INTERVAL '{_train_end_days} days'
                                    + INTERVAL '{gap_days} days'
                                    AS val_start,
                                CAST('{_min_dt}' AS TIMESTAMP) + INTERVAL '{_train_end_days} days'
                                    + INTERVAL '{gap_days} days'
                                    + INTERVAL '{_val_span_days} days'
                                    AS val_end
                        )
                        SELECT
                            _orig_idx,
                            CASE
                                WHEN _split_date <= (SELECT train_end FROM cutoffs)
                                    THEN 'train'
                                WHEN _split_date >= (SELECT val_start FROM cutoffs)
                                     AND _split_date <= (SELECT val_end FROM cutoffs)
                                    THEN 'val'
                                ELSE NULL
                            END AS _split_label
                        FROM idx_df
                        WHERE _split_date <= (SELECT train_end FROM cutoffs)
                           OR (_split_date >= (SELECT val_start FROM cutoffs)
                               AND _split_date <= (SELECT val_end FROM cutoffs))
                        ORDER BY _split_date, _orig_idx
                    """).arrow()

                    # Extract train/val indices from Arrow result (no pandas)
                    _orig_idx_arr = _split_arrow.column("_orig_idx").to_numpy(zero_copy_only=False)
                    _label_arr = _split_arrow.column("_split_label")
                    _train_mask = pc.equal(_label_arr, "train").to_numpy(zero_copy_only=False)
                    _val_mask = pc.equal(_label_arr, "val").to_numpy(zero_copy_only=False)
                    _train_orig = _orig_idx_arr[_train_mask].tolist()
                    _val_orig = _orig_idx_arr[_val_mask].tolist()

                    # Reorder features (and labels) by the temporal sort order
                    _ordered_idx = _train_orig + _val_orig
                    features = features.take(_ordered_idx)
                    if labels is not None:
                        labels = labels.take(_ordered_idx)

                    train_idx = list(range(len(_train_orig)))
                    val_idx = list(range(len(_train_orig), len(_train_orig) + len(_val_orig)))
                    split_indices = {"train": train_idx, "val": val_idx}

                    logger.info(
                        "Temporal split (DuckDB): train=%d, val=%d (gap_days=%d, "
                        "discarded=%d rows in gaps)",
                        len(train_idx), len(val_idx), gap_days,
                        _idx_table.num_rows - len(_train_orig) - len(_val_orig),
                    )
                else:
                    logger.warning(
                        "All dates identical -- falling back to random split"
                    )
            finally:
                _con_s.close()

            # Clean up temporaries
            del _idx_table
        else:
            logger.warning("No date column '%s' found -- falling back to random split", date_col)

    # ---- 3c. Leakage validation ----
    # LeakageValidator accepts Arrow Tables directly (no pandas conversion)
    if labels is not None:
        try:
            from core.pipeline.leakage_validator import LeakageValidator
            validator = LeakageValidator(correlation_threshold=0.95)
            # Subsample for speed (1M correlation is too slow)
            _n = features.num_rows if hasattr(features, 'num_rows') else len(features)
            _sample_n = min(_n, 50_000)
            _idx = np.random.default_rng(42).choice(_n, _sample_n, replace=False)
            _feat_sample = features.take(_idx)
            _lbl_sample = labels.take(_idx)
            result = validator.validate(_feat_sample, _lbl_sample, config)
            del _feat_sample, _lbl_sample
            if not result.passed:
                for w in result.warnings[:5]:
                    logger.warning("LEAKAGE: %s", w)
                # Auto-drop features with >0.95 correlation to any label
                import re as _re
                drop_cols = []
                for w in result.warnings:
                    _m = _re.search(r"Feature '([^']+)'", str(w))
                    if _m:
                        drop_cols.append(_m.group(1))
                if drop_cols and features is not None:
                    _before = features.num_columns
                    _existing_drop = [c for c in drop_cols if c in features.column_names]
                    if _existing_drop:
                        remaining = [c for c in features.column_names if c not in set(_existing_drop)]
                        features = features.select(remaining)
                    logger.warning(
                        "Auto-dropped %d leaking features: %s (cols %d->%d)",
                        len(drop_cols), drop_cols, _before, features.num_columns,
                    )
            else:
                logger.info("LeakageValidator: PASSED (no leakage detected)")
        except Exception as e:
            logger.warning("LeakageValidator skipped: %s", e)

    # ---- 3d. Per-task validation masks (split_strategy from config) ----
    task_val_masks = _build_task_val_masks(
        config, features, split_indices, label_schema,
    )

    # ---- 4. Build DataLoaders ----
    train_loader, val_loader, tasks, task_type_map, label_stats = build_dataloaders(
        features, labels, sequences, seq_lengths,
        feature_schema, label_schema, split_indices, hp, config=config,
    )
    task_names = [t["name"] for t in tasks]

    # ---- Label distribution (Arrow compute, no pandas) ----
    logger.info("=== Label Distribution ===")
    for task in tasks:
        col = task["label_col"]
        if labels is not None and col in labels.column_names:
            col_arr = labels.column(col)
            n_total = labels.num_rows
            if task["type"] == "binary":
                n_pos = int(pc.sum(pc.greater(col_arr, 0)).as_py())
                pos_rate = n_pos / n_total if n_total > 0 else 0.0
                logger.info("  %s [binary]: positive=%.2f%% (%d/%d)",
                            task["name"], pos_rate * 100, n_pos, n_total)
            elif task["type"] == "multiclass":
                _vc = pc.value_counts(col_arr)
                # Sort by count descending, take top 5
                _vc_list = sorted(
                    [(entry["values"].as_py(), entry["counts"].as_py()) for entry in _vc],
                    key=lambda x: -x[1],
                )[:5]
                _vc_dict = {str(k): v for k, v in _vc_list}
                logger.info("  %s [multi-%s]: top classes=%s",
                            task["name"], task.get("num_classes", "?"), _vc_dict)
            else:
                _np_arr = col_arr.to_numpy(zero_copy_only=False).astype(np.float64)
                logger.info("  %s [regression]: mean=%.4f, std=%.4f, range=[%.4f, %.4f]",
                            task["name"], float(np.nanmean(_np_arr)), float(np.nanstd(_np_arr)),
                            float(np.nanmin(_np_arr)), float(np.nanmax(_np_arr)))

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

    # Inject per-task validation masks into trainer (for epoch-level validation)
    if task_val_masks:
        trainer.set_task_val_masks(task_val_masks)

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
    final_metrics = validate(
        model, val_loader, device, task_names, task_type_map,
        task_val_masks=task_val_masks,
    )
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
