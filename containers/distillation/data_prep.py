"""Data preparation helpers for the distillation pipeline.

Extracted from distill_entry.py to keep run_distillation() concise.
Covers: PyArrow data loading, quality gating, feature/label extraction,
and StudentTrainer construction.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("distill-entry")


def load_data_pyarrow(channel_path: Path) -> Tuple[Any, Dict, Dict, Dict]:
    """Load Phase 0 data via PyArrow (CLAUDE.md §3.3 — no pandas in hot path).

    Args:
        channel_path: Directory containing features.parquet and schema JSONs.

    Returns:
        (table, feature_schema, label_schema, split_indices)
        where ``table`` is a ``pyarrow.Table``.
    """
    import pyarrow.parquet as pq

    features_parquet = channel_path / "features.parquet"
    if features_parquet.exists():
        table = pq.read_table(str(features_parquet))
        logger.info(
            "Loaded features.parquet: %d rows, %d columns",
            table.num_rows, table.num_columns,
        )
    else:
        parquet_files = sorted(
            channel_path.glob("*.parquet"), key=lambda p: p.stat().st_size, reverse=True
        )
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {channel_path}")
        table = pq.read_table(str(parquet_files[0]))
        logger.info(
            "Loaded %s: %d rows, %d columns",
            parquet_files[0].name, table.num_rows, table.num_columns,
        )

    feature_schema_path = channel_path / "feature_schema.json"
    if not feature_schema_path.exists():
        raise FileNotFoundError(f"feature_schema.json not found in {channel_path}")
    with open(feature_schema_path) as f:
        feature_schema: Dict[str, Any] = json.load(f)

    label_schema: Dict[str, Any] = {}
    label_schema_path = channel_path / "label_schema.json"
    if label_schema_path.exists():
        with open(label_schema_path) as f:
            label_schema = json.load(f)

    split_indices: Dict[str, Any] = {}
    split_path = channel_path / "split_indices.json"
    if split_path.exists():
        with open(split_path) as f:
            split_indices = json.load(f)
        logger.info(
            "Split indices loaded: train=%d val=%d test=%d",
            len(split_indices.get("train", [])),
            len(split_indices.get("val", [])),
            len(split_indices.get("test", [])),
        )

    return table, feature_schema, label_schema, split_indices


def prepare_features_and_labels(
    table: Any,
    pipeline_config: Any,
    feature_schema: Dict[str, Any],
    checkpoint_path: str,
    skip_fidelity_gate: bool,
) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray]]:
    """Quality-gate the table, resolve feature columns and extract numpy arrays.

    Args:
        table:              PyArrow Table loaded from features.parquet.
        pipeline_config:    Parsed pipeline config with ``.tasks`` and ``.features``.
        feature_schema:     Dict loaded from feature_schema.json.
        checkpoint_path:    Path to teacher .pt file (used to find config.json sibling).
        skip_fidelity_gate: When True, continue despite quality gate failure.

    Returns:
        (feature_cols, features, hard_labels)
        ``features`` is (N x D) float32 numpy array.
        ``hard_labels`` maps task_name -> ground-truth numpy array.
    """
    from core.data.quality_gate import QualityGate, QualityGateError

    logger.info("Running quality gate on distillation data...")
    _quality_gate = QualityGate()
    try:
        _gate_result = _quality_gate.evaluate_and_block(
            table, source_name="distillation_train"
        )
        logger.info(
            "Quality gate PASSED (verdict=%s, checks=%d)",
            _gate_result.verdict.value, len(_gate_result.checks),
        )
    except QualityGateError as _exc:
        logger.error("Quality gate FAILED: %s", _exc)
        if not skip_fidelity_gate:
            raise RuntimeError(
                f"Quality gate blocked distillation pipeline: {_exc}"
            ) from _exc
        logger.warning("skip_fidelity_gate=True — continuing despite quality gate failure")

    label_cols = {t.label_col for t in pipeline_config.tasks}
    id_cols = set(pipeline_config.features.id_cols)
    exclude_cols = label_cols | id_cols
    table_cols_set = set(table.column_names)

    # Resolve feature column order: teacher schema → data schema → numeric fallback
    _teacher_schema_cols: List[str] = []
    _schema_path = Path(checkpoint_path).parent / "config.json"
    if _schema_path.exists():
        with open(_schema_path) as _sf:
            _schema = json.load(_sf)
        _teacher_schema_cols = _schema.get("feature_schema", {}).get("columns", [])
        logger.info(
            "Teacher schema: %d features from %s", len(_teacher_schema_cols), _schema_path
        )

    if not _teacher_schema_cols:
        _teacher_schema_cols = feature_schema.get(
            "columns", feature_schema.get("feature_columns", [])
        )

    if _teacher_schema_cols:
        feature_cols = [c for c in _teacher_schema_cols if c in table_cols_set]
        missing = [c for c in _teacher_schema_cols if c not in table_cols_set]
        if missing:
            logger.warning(
                "Missing %d columns from teacher schema (will fill 0)", len(missing)
            )
    else:
        import pyarrow.types as pat
        feature_cols = [
            c for c in table.column_names
            if c not in exclude_cols
            and (pat.is_floating(table.schema.field(c).type)
                 or pat.is_integer(table.schema.field(c).type))
        ]

    logger.info(
        "Features: %d columns, Labels: %d tasks, IDs: %d columns",
        len(feature_cols), len(label_cols), len(id_cols),
    )

    # Extract features as numpy via PyArrow (zero-copy where possible)
    feature_arrays = []
    for c in feature_cols:
        col = table.column(c)
        arr = col.to_numpy(zero_copy_only=False).astype(np.float32)
        np.nan_to_num(arr, copy=False)
        feature_arrays.append(arr)
    features = (
        np.column_stack(feature_arrays)
        if feature_arrays
        else np.empty((table.num_rows, 0), dtype=np.float32)
    )
    del feature_arrays
    logger.info("Features array: %s, %.1f MB", features.shape, features.nbytes / 1024**2)

    hard_labels: Dict[str, np.ndarray] = {}
    for t in pipeline_config.tasks:
        if t.label_col in table_cols_set:
            hard_labels[t.name] = table.column(t.label_col).to_numpy(zero_copy_only=False)

    del table
    return feature_cols, features, hard_labels


def build_trainer(
    distillation_cfg: dict,
    checkpoint_path: str,
    model_out_dir: Path,
    pipeline_config: Any,
    feature_cols: List[str],
    temperature: Optional[float],
    alpha: Optional[float],
) -> Tuple[Any, Any]:
    """Construct StudentConfig + StudentTrainer from config + HP overrides.

    Args:
        distillation_cfg: ``distillation`` sub-dict from pipeline YAML.
        checkpoint_path:  Path to teacher .pt checkpoint.
        model_out_dir:    Directory where student models will be saved.
        pipeline_config:  Parsed pipeline config with ``.tasks``.
        feature_cols:     Ordered list of feature column names.
        temperature:      HP override for distillation temperature (or None).
        alpha:            HP override for hard-label weight (or None).

    Returns:
        (student_config, trainer)
    """
    from core.training.student_trainer import StudentConfig, StudentTrainer

    _student_dict = dict(distillation_cfg)
    if "lgbm" in _student_dict and "lgbm_params" not in _student_dict:
        _student_dict["lgbm_params"] = _student_dict.pop("lgbm")

    _student_dict["teacher_checkpoint"] = checkpoint_path
    _student_dict["student_output_dir"] = str(model_out_dir)
    _student_dict["soft_label_path"] = distillation_cfg.get("soft_label_path", "")

    if temperature is not None:
        _student_dict["temperature"] = temperature
    if alpha is not None:
        _student_dict["alpha"] = alpha

    student_config = StudentConfig.from_dict(_student_dict)
    logger.info(
        "StudentConfig: temperature=%.2f alpha=%.2f lgbm_params=%s task_lgbm_overrides=%s",
        student_config.temperature,
        student_config.alpha,
        student_config.lgbm_params,
        list(student_config.task_lgbm_overrides.keys()) or "(none)",
    )

    trainer = StudentTrainer(
        config=student_config,
        task_specs=pipeline_config.tasks,
        feature_columns=feature_cols,
    )
    return student_config, trainer
