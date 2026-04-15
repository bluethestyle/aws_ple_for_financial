#!/usr/bin/env python3
"""SageMaker entry point for PLE -> LGBM knowledge distillation.

Replicates the full pipeline from ``scripts/run_distillation.py`` but reads
paths from SageMaker environment variables instead of CLI arguments.

SageMaker channel layout
------------------------
  SM_CHANNEL_TRAIN   -- data directory (features.parquet, feature_schema.json,
                         label_schema.json, split_indices.json)
  SM_CHANNEL_MODEL   -- teacher checkpoint directory (best.pt or *.pt + config.json)
  SM_HPS             -- JSON string with hyperparameters
  SM_OUTPUT_DATA_DIR -- write fidelity_report.json, drift_report.json here
  SM_MODEL_DIR       -- write LGBM models + distillation_summary.json here

Hyperparameters (SM_HPS)
------------------------
  config             -- container-internal path to common pipeline.yaml
                        (default: configs/pipeline.yaml)
  dataset_config     -- optional dataset-specific YAML deep-merged on top of config
                        (e.g. configs/datasets/santander.yaml)
  batch_size         -- teacher inference batch size for soft label generation
                        (default: 4096)
  temperature        -- distillation temperature (default: from YAML)
  alpha              -- hard label weight (default: from YAML)
  skip_fidelity_gate -- "true" to continue despite fidelity failures (default: "false")

Local execution
---------------
When SM_CHANNEL_TRAIN is absent the script falls back to CLI argument parsing
so it can be run locally for debugging without a SageMaker job.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure duckdb is installed (mirrors train.py / eval_entry.py pattern)
# ---------------------------------------------------------------------------
try:
    import duckdb  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "duckdb>=1.0.0"])

try:
    import lightgbm  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm>=4.0.0"])

try:
    import joblib  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "joblib>=1.3.0"])

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# When running inside SageMaker the working directory is /opt/ml/code
# where the source_dir was extracted. Ensure it is importable.
_code_dir = Path("/opt/ml/code")
if _code_dir.exists() and str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))

# Also support running from project root locally
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Force UTF-8 stdout (avoids cp949 errors on Windows dev boxes)
_utf8_stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(_utf8_stdout)],
)
logger = logging.getLogger("distill-entry")


# ---------------------------------------------------------------------------
# SageMaker environment helpers (mirrors eval_entry.py)
# ---------------------------------------------------------------------------

def _parse_hp_value(v: Any) -> Any:
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


def get_hyperparameters() -> Dict[str, Any]:
    """Read hyperparameters from SM_HPS or /opt/ml/input/config/."""
    sm_hps = os.environ.get("SM_HPS")
    if sm_hps:
        return json.loads(sm_hps)

    hp_path = Path("/opt/ml/input/config/hyperparameters.json")
    if hp_path.exists():
        with open(hp_path) as f:
            raw = json.load(f)
        return {k: _parse_hp_value(v) for k, v in raw.items()}

    return {}


def _is_sagemaker() -> bool:
    """Return True when running inside a SageMaker container."""
    return "SM_CHANNEL_TRAIN" in os.environ


# ---------------------------------------------------------------------------
# Checkpoint discovery (mirrors eval_entry.py)
# ---------------------------------------------------------------------------

def _find_checkpoint(model_dir: str) -> str:
    """Locate the best checkpoint in *model_dir*.

    Preference order:
      1. ``best.pt``
      2. Any ``*.pt`` file (most recently modified wins)

    Args:
        model_dir: Directory containing the .pt file(s).

    Returns:
        Absolute path to the checkpoint.

    Raises:
        FileNotFoundError: When no .pt file is found.
    """
    model_path = Path(model_dir)

    best = model_path / "best.pt"
    if best.exists():
        logger.info("Using checkpoint: %s", best)
        return str(best)

    candidates = sorted(
        list(model_path.glob("*.pt")) + list(model_path.glob("*.pth")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No .pt/.pth checkpoint found in {model_dir}")

    logger.info("Using checkpoint (most recent): %s", candidates[0])
    return str(candidates[0])


# ---------------------------------------------------------------------------
# Data loading — PyArrow (CLAUDE.md 3.3: no pandas in hot path)
# ---------------------------------------------------------------------------

def _load_data_pyarrow(channel_path: Path):
    """Load Phase 0 data via PyArrow.

    Returns:
        Tuple of (table, feature_schema, label_schema, split_indices)
        where table is a pyarrow.Table.
    """
    import pyarrow.parquet as pq

    # Load features parquet
    features_parquet = channel_path / "features.parquet"
    if features_parquet.exists():
        table = pq.read_table(str(features_parquet))
        logger.info(
            "Loaded features.parquet: %d rows, %d columns",
            table.num_rows, table.num_columns,
        )
    else:
        parquet_files = sorted(channel_path.glob("*.parquet"), key=lambda p: p.stat().st_size, reverse=True)
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {channel_path}")
        table = pq.read_table(str(parquet_files[0]))
        logger.info(
            "Loaded %s: %d rows, %d columns",
            parquet_files[0].name, table.num_rows, table.num_columns,
        )

    # Feature schema
    feature_schema_path = channel_path / "feature_schema.json"
    if not feature_schema_path.exists():
        raise FileNotFoundError(f"feature_schema.json not found in {channel_path}")
    with open(feature_schema_path) as f:
        feature_schema = json.load(f)

    # Label schema (optional)
    label_schema: Dict[str, Any] = {}
    label_schema_path = channel_path / "label_schema.json"
    if label_schema_path.exists():
        with open(label_schema_path) as f:
            label_schema = json.load(f)

    # Split indices (optional)
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


# ---------------------------------------------------------------------------
# Core distillation pipeline
# ---------------------------------------------------------------------------

def run_distillation(
    data_dir: str,
    model_dir: str,
    config_path: str,
    output_dir: str,
    model_output_dir: str,
    batch_size: int = 4096,
    temperature: Optional[float] = None,
    alpha: Optional[float] = None,
    skip_fidelity_gate: bool = False,
    dataset_config_path: Optional[str] = None,
) -> str:
    """Core distillation routine shared by SageMaker and local paths.

    Replicates ``scripts/run_distillation.py:main()`` logic with SageMaker
    paths substituted for CLI args.

    Args:
        data_dir:            SM_CHANNEL_TRAIN — Phase 0 data directory.
        model_dir:           SM_CHANNEL_MODEL — teacher checkpoint directory.
        config_path:         Path to pipeline.yaml.
        output_dir:          SM_OUTPUT_DATA_DIR — for fidelity/drift reports.
        model_output_dir:    SM_MODEL_DIR — for LGBM models + summary.
        batch_size:          Teacher inference batch size.
        temperature:         Override distillation.temperature from YAML.
        alpha:               Override distillation.alpha from YAML.
        skip_fidelity_gate:  Continue despite fidelity failures.
        dataset_config_path: Optional dataset YAML deep-merged on top of config.

    Returns:
        Path to the written distillation_summary.json.
    """
    import numpy as np
    import yaml as _yaml
    import joblib

    from core.pipeline.config import load_config, load_merged_config

    channel_path = Path(data_dir)
    out_dir = Path(output_dir)
    model_out_dir = Path(model_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_out_dir.mkdir(parents=True, exist_ok=True)

    # --- Config ---
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if dataset_config_path and Path(dataset_config_path).exists():
        pipeline_config = load_config(config_path, dataset_path=dataset_config_path)
        _raw_yaml: dict = load_merged_config(config_path, dataset_config_path)
        logger.info("Config loaded (merged): %s + %s", config_path, dataset_config_path)
    else:
        pipeline_config = load_config(config_path)
        with open(config_path, encoding="utf-8") as _f:
            _raw_yaml = _yaml.safe_load(_f) or {}
        logger.info("Config loaded: %s", config_path)

    _distillation_cfg: dict = _raw_yaml.get("distillation", {})

    # --- Load data via PyArrow (CLAUDE.md 3.3) ---
    logger.info("Loading Phase 0 data from %s", data_dir)
    table, feature_schema, label_schema, split_indices = _load_data_pyarrow(channel_path)

    # Quality gate on PyArrow Table — no to_pandas() conversion (CLAUDE.md 3.3)
    logger.info("Running quality gate on distillation data...")
    from core.data.quality_gate import QualityGate, QualityGateError

    _quality_gate = QualityGate()
    try:
        _gate_result = _quality_gate.evaluate_and_block(
            table, source_name="distillation_train"
        )
        logger.info(
            "Quality gate PASSED (verdict=%s, checks=%d)",
            _gate_result.verdict.value,
            len(_gate_result.checks),
        )
    except QualityGateError as _exc:
        logger.error("Quality gate FAILED: %s", _exc)
        if not skip_fidelity_gate:
            raise RuntimeError(
                f"Quality gate blocked distillation pipeline: {_exc}"
            ) from _exc
        logger.warning("skip_fidelity_gate=True — continuing despite quality gate failure")

    # --- Feature / label column routing (config-driven, no hardcoding) ---
    label_cols = {t.label_col for t in pipeline_config.tasks}
    id_cols = set(pipeline_config.features.id_cols)
    exclude_cols = label_cols | id_cols
    table_cols_set = set(table.column_names)

    # Find teacher checkpoint
    checkpoint_path = _find_checkpoint(model_dir)

    # Use teacher schema for column order if available
    _teacher_schema_cols: List[str] = []
    _schema_path = Path(checkpoint_path).parent / "config.json"
    if _schema_path.exists():
        with open(_schema_path) as _sf:
            _schema = json.load(_sf)
        _teacher_schema_cols = _schema.get("feature_schema", {}).get("columns", [])
        logger.info(
            "Teacher schema: %d features from %s",
            len(_teacher_schema_cols), _schema_path,
        )

    # Fall back to feature_schema.json from the data channel
    if not _teacher_schema_cols:
        _teacher_schema_cols = feature_schema.get("columns", feature_schema.get("feature_columns", []))

    if _teacher_schema_cols:
        feature_cols = [c for c in _teacher_schema_cols if c in table_cols_set]
        missing = [c for c in _teacher_schema_cols if c not in table_cols_set]
        if missing:
            logger.warning("Missing %d columns from teacher schema (will fill 0)", len(missing))
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

    # Extract hard labels via PyArrow (no pandas)
    hard_labels: Dict[str, np.ndarray] = {}
    for t in pipeline_config.tasks:
        if t.label_col in table_cols_set:
            hard_labels[t.name] = table.column(t.label_col).to_numpy(zero_copy_only=False)

    del table

    # --- Build StudentConfig from YAML + HP overrides ---
    from core.training.student_trainer import StudentConfig, StudentTrainer

    _student_dict = dict(_distillation_cfg)
    if "lgbm" in _student_dict and "lgbm_params" not in _student_dict:
        _student_dict["lgbm_params"] = _student_dict.pop("lgbm")

    _student_dict["teacher_checkpoint"] = checkpoint_path
    _student_dict["student_output_dir"] = str(model_out_dir)
    _student_dict["soft_label_path"] = _distillation_cfg.get("soft_label_path", "")

    # HP overrides: temperature and alpha (from SM_HPS take precedence over YAML)
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

    # --- Step 2: Generate soft labels from teacher ---
    logger.info("Generating soft labels from teacher checkpoint: %s", checkpoint_path)
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    trainer.load_teacher(
        checkpoint_path=checkpoint_path,
        pipeline_config=_raw_yaml,
    )

    features_tensor = torch.tensor(features, dtype=torch.float32)
    dataset = TensorDataset(features_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    trainer.generate_soft_labels(data_loader=loader, save_path=None)

    # --- Step 2.5: Teacher threshold gating (adaptive distillation) ---
    from sklearn.metrics import roc_auc_score, f1_score, r2_score

    soft_labels = trainer.get_soft_labels()
    distill_tasks: List[str] = []
    hardlabel_tasks: List[str] = []

    logger.info("=" * 60)
    logger.info("Teacher threshold check (2x random baseline)")
    logger.info("=" * 60)

    for t in pipeline_config.tasks:
        task_name = t.name
        task_type = t.type
        teacher_soft = soft_labels.get(task_name)
        hard = hard_labels.get(task_name)

        if teacher_soft is None or hard is None:
            logger.warning("  %s: no soft/hard labels -- skipping", task_name)
            continue

        teacher_np = teacher_soft.numpy() if hasattr(teacher_soft, "numpy") else np.array(teacher_soft)
        hard_np = hard if isinstance(hard, np.ndarray) else np.array(hard)

        viable = False
        metric_str = ""

        try:
            if task_type == "binary":
                preds = teacher_np.flatten()
                labels_np = hard_np.flatten()
                valid = (labels_np == 0) | (labels_np == 1)
                if valid.sum() > 10 and len(set(labels_np[valid].tolist())) == 2:
                    auc = roc_auc_score(labels_np[valid], preds[valid])
                    viable = auc > 0.60
                    metric_str = f"AUC={auc:.4f} (threshold=0.60)"

            elif task_type == "multiclass":
                n_classes = getattr(t, "num_classes", None) or int(hard_np.max()) + 1
                threshold = 2.0 / n_classes
                if teacher_np.ndim > 1 and teacher_np.shape[-1] > 1:
                    pred_classes = teacher_np.argmax(axis=-1)
                else:
                    pred_classes = teacher_np.flatten().astype(int)
                true_classes = hard_np.flatten().astype(int)
                valid = true_classes >= 0
                if valid.sum() > 10:
                    f1 = f1_score(true_classes[valid], pred_classes[valid],
                                  average="macro", zero_division=0)
                    viable = f1 > threshold
                    metric_str = f"F1={f1:.4f} (threshold={threshold:.4f}, {n_classes}-class)"

            elif task_type == "regression":
                preds = teacher_np.flatten()
                labels_np = hard_np.flatten()
                if np.var(labels_np) > 1e-10:
                    r2 = r2_score(labels_np, preds)
                    viable = r2 > 0.05
                    metric_str = f"R2={r2:.4f} (threshold=0.05)"

        except Exception as exc:
            logger.debug("  %s: threshold check error: %s", task_name, exc)

        if viable:
            distill_tasks.append(task_name)
            logger.info("  [DISTILL] %s: %s", task_name, metric_str)
        else:
            hardlabel_tasks.append(task_name)
            logger.info("  [DIRECT]  %s: %s -- below threshold, using hard labels",
                        task_name, metric_str)

    logger.info("Distillation: %d tasks, Direct LGBM: %d tasks",
                len(distill_tasks), len(hardlabel_tasks))
    logger.info("=" * 60)

    # --- Step 3: Train LGBM students ---
    logger.info("Training LGBM student models...")
    students = trainer.train_students(features, hard_labels)

    # Hard-label-only tasks
    if hardlabel_tasks:
        logger.info("  [DIRECT] %d tasks: %s -- retraining with hard labels only",
                    len(hardlabel_tasks), hardlabel_tasks)
        import lightgbm as lgb

        for task_name in hardlabel_tasks:
            t_spec = next((t for t in pipeline_config.tasks if t.name == task_name), None)
            if t_spec is None or task_name not in hard_labels:
                continue

            y = hard_labels[task_name]
            lgbm_params = dict(student_config.lgbm_params)
            task_overrides = student_config.task_lgbm_overrides.get(task_name, {})
            lgbm_params.update(task_overrides)

            if t_spec.type == "binary":
                lgbm_params.setdefault("objective", "binary")
                lgbm_params.setdefault("metric", "auc")
            elif t_spec.type == "multiclass":
                n_classes = getattr(t_spec, "num_classes", None) or int(y.max()) + 1
                lgbm_params["objective"] = "multiclass"
                lgbm_params["num_class"] = n_classes
                lgbm_params.setdefault("metric", "multi_logloss")
            else:
                lgbm_params.setdefault("objective", "regression")
                lgbm_params.setdefault("metric", "mae")

            lgbm_params.setdefault("verbosity", -1)
            n_estimators = lgbm_params.pop("n_estimators", 300)

            ds = lgb.Dataset(features, label=y)
            model = lgb.train(lgbm_params, ds, num_boost_round=n_estimators)
            students[task_name] = model
            logger.info("    %s: direct LGBM trained (%d rounds)", task_name, n_estimators)

    # --- Step 3.5: Calibration (Platt scaling) ---
    _calib_cfg = _distillation_cfg.get("calibration", {})
    _calib_tasks = set(_calib_cfg.get("tasks", []))
    _calib_method = _calib_cfg.get("method", "platt")
    calibrated_models: Dict[str, Any] = {}

    if _calib_cfg.get("enabled", False) and _calib_tasks:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.base import BaseEstimator, ClassifierMixin

        class _LGBMProbWrapper(BaseEstimator, ClassifierMixin):
            """Wrap a trained LGBM model for sklearn calibration."""
            def __init__(self, lgbm_model):
                self.lgbm_model = lgbm_model
                self.classes_ = np.array([0, 1])
            def fit(self, X, y):
                return self
            def predict_proba(self, X):
                raw = self.lgbm_model.predict(X)
                return np.column_stack([1 - raw, raw])

        n_total = len(features)
        calib_start = int(n_total * 0.8)
        X_calib = features[calib_start:]
        _sklearn_method = "sigmoid" if _calib_method == "platt" else "isotonic"

        logger.info("=" * 60)
        logger.info("Calibration: %s method on %d tasks", _calib_method, len(_calib_tasks))
        logger.info("  Calibration set: %d samples (last 20%%)", len(X_calib))
        logger.info("=" * 60)

        for task_name in _calib_tasks:
            if task_name not in students or task_name not in hard_labels:
                logger.warning("  %s: skipped (no model or labels)", task_name)
                continue

            t_spec = next((t for t in pipeline_config.tasks if t.name == task_name), None)
            if t_spec is None:
                continue

            y_calib = hard_labels[task_name][calib_start:]
            model = students[task_name]

            try:
                if t_spec.type == "binary":
                    wrapper = _LGBMProbWrapper(model)
                    calibrator = CalibratedClassifierCV(
                        wrapper, method=_sklearn_method, cv="prefit",
                    )
                    calibrator.fit(X_calib, y_calib)
                    calibrated_models[task_name] = calibrator
                    logger.info("  [CALIBRATED] %s: %s scaling applied", task_name, _calib_method)

                elif t_spec.type == "regression":
                    raw_preds = model.predict(X_calib)
                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression()
                    lr.fit(raw_preds.reshape(-1, 1), y_calib)
                    calibrated_models[task_name] = {"lgbm": model, "bias_corrector": lr}
                    logger.info(
                        "  [CALIBRATED] %s: linear bias correction (slope=%.4f, intercept=%.4f)",
                        task_name, lr.coef_[0], lr.intercept_,
                    )

                else:
                    logger.info(
                        "  %s: multiclass calibration not implemented, skipped", task_name
                    )

            except Exception as exc:
                logger.warning("  %s: calibration failed: %s", task_name, exc)

    # --- Step 4: Fidelity validation ---
    logger.info("Running fidelity validation (8 metrics per task)...")
    from core.training.distillation_validator import DistillationValidator, ValidationCriteria

    _fidelity_cfg = _distillation_cfg.get("fidelity", {})
    _bin_cfg = _fidelity_cfg.get("binary", {})
    _mc_cfg = _fidelity_cfg.get("multiclass", {})
    _reg_cfg = _fidelity_cfg.get("regression", {})

    criteria = ValidationCriteria(
        max_auc_gap=_bin_cfg.get("max_auc_gap", 0.05),
        min_binary_agreement=_bin_cfg.get("min_agreement", 0.85),
        max_jsd=_bin_cfg.get("max_jsd", 0.10),
        min_ranking_corr=_bin_cfg.get("min_ranking_corr", 0.90),
        max_calibration_gap=_bin_cfg.get("max_calibration_gap", 0.05),
        min_multiclass_agreement=_mc_cfg.get("min_agreement", 0.70),
        max_f1_macro_gap=_mc_cfg.get("max_f1_macro_gap", 0.10),
        regression_quartile_agreement_min=_reg_cfg.get("min_quartile_agreement", 0.70),
        max_mae_gap=_reg_cfg.get("max_mae_gap", 0.05),
        max_rmse_gap=_reg_cfg.get("max_rmse_gap", 0.10),
    )
    validator = DistillationValidator(criteria=criteria)
    fidelity_results = []
    soft_labels = trainer.get_soft_labels()

    for task_spec in pipeline_config.tasks:
        task_name = task_spec.name
        if task_name not in students:
            continue

        student_model = students[task_name]

        if task_name in calibrated_models:
            calib = calibrated_models[task_name]
            if task_spec.type == "binary" and hasattr(calib, "predict_proba"):
                student_preds = calib.predict_proba(features)[:, 1]
            elif task_spec.type == "regression" and isinstance(calib, dict):
                raw = calib["lgbm"].predict(features)
                student_preds = calib["bias_corrector"].predict(raw.reshape(-1, 1))
            else:
                student_preds = student_model.predict(features)
        else:
            raw_preds = student_model.predict(features)
            if task_spec.type == "binary" and student_config.use_custom_objective:
                student_preds = 1.0 / (1.0 + np.exp(-raw_preds))
            elif task_spec.type == "multiclass" and student_config.use_custom_objective:
                n_classes = task_spec.num_classes
                raw_2d = raw_preds.reshape(-1, n_classes)
                exp_shifted = np.exp(raw_2d - raw_2d.max(axis=1, keepdims=True))
                student_preds = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
            else:
                student_preds = raw_preds

        teacher_preds = soft_labels.get(task_name)
        if teacher_preds is None:
            logger.warning("No soft labels for task %s, skipping fidelity", task_name)
            continue

        labels_for_task = hard_labels.get(task_name)

        try:
            result = validator.validate_task(
                task_name=task_name,
                task_type=task_spec.type,
                teacher_preds=teacher_preds,
                student_preds=student_preds,
                labels=labels_for_task,
            )
        except Exception as exc:
            logger.warning("Fidelity validation failed for %s: %s", task_name, exc)
            from core.training.distillation_validator import FidelityResult
            result = FidelityResult(
                task_name=task_name, task_type=task_spec.type, passed=False,
                metrics={}, failures=[str(exc)],
            )
        fidelity_results.append(result)

        status = "PASS" if result.passed else "FAIL"
        logger.info(
            "  [%s] %s -- metrics: %s%s",
            status, task_name,
            {k: round(v, 4) for k, v in result.metrics.items()},
            f" failures: {result.failures}" if result.failures else "",
        )

    passed_count = sum(1 for r in fidelity_results if r.passed)
    failed_count = len(fidelity_results) - passed_count
    logger.info("Fidelity summary: %d/%d tasks passed", passed_count, passed_count + failed_count)

    if failed_count > 0:
        failed_tasks = [r.task_name for r in fidelity_results if not r.passed]
        logger.error(
            "Fidelity FAILED for %d task(s): %s",
            failed_count, failed_tasks,
        )
        fidelity_report = {
            "status": "FAILED",
            "passed": passed_count,
            "failed": failed_count,
            "failed_tasks": failed_tasks,
            "details": {
                r.task_name: {
                    "passed": r.passed,
                    "metrics": r.metrics,
                    "failures": r.failures,
                }
                for r in fidelity_results
            },
        }
        with open(out_dir / "fidelity_report.json", "w") as f:
            json.dump(fidelity_report, f, indent=2, default=str)

        if not skip_fidelity_gate:
            logger.error("Aborting distillation pipeline due to fidelity failure.")
            raise RuntimeError(
                f"Fidelity gate failed on {failed_count} task(s): {failed_tasks}"
            )
        else:
            logger.warning("skip_fidelity_gate=True -- continuing despite failures.")
    else:
        # Write passing fidelity report
        fidelity_report = {
            "status": "PASSED",
            "passed": passed_count,
            "failed": 0,
            "details": {
                r.task_name: {
                    "passed": r.passed,
                    "metrics": r.metrics,
                    "failures": r.failures,
                }
                for r in fidelity_results
            },
        }
        with open(out_dir / "fidelity_report.json", "w") as f:
            json.dump(fidelity_report, f, indent=2, default=str)

    # --- Step 4.5b: Drift monitoring (SR 11-7 MRM safeguard) ---
    _drift_cfg = _distillation_cfg.get("drift_monitoring", {})
    if _drift_cfg.get("enabled", False):
        from core.training.distillation_drift import DistillationDriftMonitor

        logger.info("=" * 60)
        logger.info("Step 4.5b: Temporal drift monitoring (SR 11-7)")
        logger.info("=" * 60)

        _drift_monitor = DistillationDriftMonitor(config=_drift_cfg)
        _baseline_path = _drift_cfg.get("baseline_path", "outputs/distillation_baseline/")

        _current_student_preds: Dict[str, Any] = {}
        for _task_spec in pipeline_config.tasks:
            _tname = _task_spec.name
            if _tname not in students:
                continue
            _sm = students[_tname]
            if _tname in calibrated_models:
                _cal = calibrated_models[_tname]
                if _task_spec.type == "binary" and hasattr(_cal, "predict_proba"):
                    _current_student_preds[_tname] = _cal.predict_proba(features)[:, 1]
                elif _task_spec.type == "regression" and isinstance(_cal, dict):
                    _raw = _cal["lgbm"].predict(features)
                    _current_student_preds[_tname] = _cal["bias_corrector"].predict(
                        _raw.reshape(-1, 1)
                    )
                else:
                    _current_student_preds[_tname] = _sm.predict(features)
            else:
                _raw_preds = _sm.predict(features)
                if _task_spec.type == "binary" and student_config.use_custom_objective:
                    _current_student_preds[_tname] = 1.0 / (1.0 + np.exp(-_raw_preds))
                elif (
                    _task_spec.type == "multiclass"
                    and student_config.use_custom_objective
                ):
                    _nc = _task_spec.num_classes
                    _r2d = _raw_preds.reshape(-1, _nc)
                    _ex = np.exp(_r2d - _r2d.max(axis=1, keepdims=True))
                    _current_student_preds[_tname] = (
                        _ex / _ex.sum(axis=1, keepdims=True)
                    ).max(axis=1)
                else:
                    _current_student_preds[_tname] = _raw_preds

        _prev_preds = _drift_monitor.load_baseline(_baseline_path)

        if _prev_preds is not None:
            _drift_report = _drift_monitor.compare_versions(
                current_preds=_current_student_preds,
                previous_preds=_prev_preds,
                labels=hard_labels,
            )

            _n_alerts = len(_drift_report["alert_tasks"])
            if _drift_report["any_alert"]:
                logger.warning(
                    "Drift alerts on %d task(s): %s -- MRM review recommended.",
                    _n_alerts, _drift_report["alert_tasks"],
                )
            else:
                logger.info(
                    "Drift check PASSED: all %d tasks within thresholds.",
                    len(_drift_report["per_task"]),
                )

            drift_out = {
                "any_alert": _drift_report["any_alert"],
                "alert_tasks": _drift_report["alert_tasks"],
                "thresholds": _drift_report["thresholds"],
                "per_task": {
                    t: {
                        k: round(v, 6) if isinstance(v, float) else v
                        for k, v in m.items()
                    }
                    for t, m in _drift_report["per_task"].items()
                },
            }
            with open(out_dir / "drift_report.json", "w") as _drf:
                json.dump(drift_out, _drf, indent=2, default=str)
            logger.info("Drift report saved to %s/drift_report.json", out_dir)
        else:
            logger.info("No previous baseline -- current predictions become first baseline.")

        _drift_monitor.save_baseline(_current_student_preds, _baseline_path)
        logger.info("Drift baseline updated at %s", _baseline_path)
    else:
        logger.info("Drift monitoring disabled (distillation.drift_monitoring.enabled=false).")

    # --- Step 4.5: Feature selection (IG dual-objective + LGBM gain pruning) ---
    logger.info("Running adaptive feature selection per task...")
    from core.training.feature_selector import (
        FeatureSelector,
        FeatureSelectionConfig,
        FeatureSelectionResult,
    )

    _fs_cfg_raw: dict = _distillation_cfg.get("feature_selection", {})
    _fs_method: str = _fs_cfg_raw.get("method", "lgbm_gain")
    _ig_alpha: float = float(_fs_cfg_raw.get("ig_alpha", 0.7))
    _cumulative_threshold: float = float(_fs_cfg_raw.get("cumulative_threshold", 0.95))
    _ig_sample_size: int = int(_fs_cfg_raw.get("ig_sample_size", 10000))

    logger.info(
        "Feature selection method=%s ig_alpha=%.2f cumulative_threshold=%.2f ig_sample_size=%d",
        _fs_method, _ig_alpha, _cumulative_threshold, _ig_sample_size,
    )

    _explain_scores: Dict[str, float] = {}
    if _fs_method in ("ig_dual", "both"):
        import yaml as _yaml_inner

        _fg_cfg_path = Path(config_path).parent / "feature_groups.yaml"
        _category_scores_from_cfg: dict = _fs_cfg_raw.get("category_explain_scores", {})
        _default_category_scores: dict = {
            "demographics": 1.0, "spending_pattern": 1.0,
            "transaction_behavior": 1.0, "temporal": 1.0,
            "economic_behavior": 1.0, "cross_domain": 0.9,
            "extended_services": 0.9, "domain_topology": 0.8,
            "customer_segment": 0.8, "temporal_state": 0.8,
            "merchant_structure": 0.9, "behavioral_state": 0.8,
            "graph_structure": 0.7, "model_insight": 0.3,
        }
        _category_scores: dict = {**_default_category_scores, **_category_scores_from_cfg}

        if _fg_cfg_path.exists():
            with open(_fg_cfg_path, encoding="utf-8") as _fgf:
                _fg_data: dict = _yaml_inner.safe_load(_fgf)
            for _grp in _fg_data.get("feature_groups", []):
                _interp = _grp.get("interpretation", {})
                _cat = _interp.get("category", "")
                _cat_score = _category_scores.get(_cat, 0.5)
                for _col in _grp.get("columns", []):
                    _explain_scores[_col] = _cat_score
            logger.info(
                "Explain-value map built: %d features from %s",
                len(_explain_scores), _fg_cfg_path,
            )
        else:
            logger.warning(
                "feature_groups.yaml not found at %s -- explain scores will be uniform",
                _fg_cfg_path,
            )

    _teacher_model = getattr(trainer, "_teacher", None)
    fs_config = FeatureSelectionConfig(cumulative_threshold=_cumulative_threshold)
    feature_selector = FeatureSelector(config=fs_config)

    feature_selections: Dict[str, Any] = {}
    _ig_raw_scores: Dict[str, Any] = {}

    for task_spec in pipeline_config.tasks:
        task_name = task_spec.name
        if task_name not in students:
            continue

        student_model = students[task_name]

        if _fs_method in ("ig_dual", "both") and _teacher_model is not None:
            try:
                ig_result = feature_selector.select_by_ig_dual(
                    model=_teacher_model,
                    features=features,
                    task_name=task_name,
                    feature_names=feature_cols,
                    n_samples=_ig_sample_size,
                    ig_alpha=_ig_alpha,
                    explain_scores=_explain_scores if _explain_scores else None,
                )
                _ig_raw_scores[task_name] = ig_result.feature_importances

                if _fs_method == "both":
                    pruned_indices = feature_selector.prune_by_lgbm(
                        lgbm_model=student_model,
                        feature_names=feature_cols,
                        selected_indices=ig_result.selected_indices,
                    )
                    pruned_names = [feature_cols[i] for i in pruned_indices]
                    lgbm_gains = student_model.feature_importance(importance_type="gain")
                    selection_result = FeatureSelectionResult(
                        task_name=task_name,
                        original_count=len(feature_cols),
                        selected_count=len(pruned_indices),
                        reduction_pct=round(
                            (1 - len(pruned_indices) / len(feature_cols)) * 100, 1,
                        ),
                        cumulative_threshold_used=_cumulative_threshold,
                        selection_method="ig_dual+lgbm",
                        selected_indices=sorted(pruned_indices),
                        selected_names=pruned_names,
                        feature_importances={
                            pruned_names[i]: float(lgbm_gains[pruned_indices[i]])
                            for i in range(min(50, len(pruned_indices)))
                        },
                        mandatory_included=ig_result.mandatory_included,
                    )
                else:
                    selection_result = ig_result

            except Exception as _ig_exc:
                logger.warning(
                    "IG dual selection failed for task %s (%s) -- falling back to lgbm_gain",
                    task_name, _ig_exc, exc_info=True,
                )
                pruned_indices = feature_selector.prune_by_lgbm(
                    lgbm_model=student_model, feature_names=feature_cols,
                )
                pruned_names = [feature_cols[i] for i in pruned_indices]
                lgbm_gains = student_model.feature_importance(importance_type="gain")
                selection_result = FeatureSelectionResult(
                    task_name=task_name,
                    original_count=len(feature_cols),
                    selected_count=len(pruned_indices),
                    reduction_pct=round(
                        (1 - len(pruned_indices) / len(feature_cols)) * 100, 1,
                    ),
                    cumulative_threshold_used=0.0,
                    selection_method="lgbm_fallback",
                    selected_indices=sorted(pruned_indices),
                    selected_names=pruned_names,
                    feature_importances={
                        pruned_names[i]: float(lgbm_gains[pruned_indices[i]])
                        for i in range(min(50, len(pruned_indices)))
                    },
                    mandatory_included=[],
                )

        elif _fs_method in ("ig_dual", "both") and _teacher_model is None:
            logger.warning(
                "method=%s but no teacher model loaded -- falling back to lgbm_gain for %s",
                _fs_method, task_name,
            )
            pruned_indices = feature_selector.prune_by_lgbm(
                lgbm_model=student_model, feature_names=feature_cols,
            )
            pruned_names = [feature_cols[i] for i in pruned_indices]
            lgbm_gains = student_model.feature_importance(importance_type="gain")
            selection_result = FeatureSelectionResult(
                task_name=task_name,
                original_count=len(feature_cols),
                selected_count=len(pruned_indices),
                reduction_pct=round(
                    (1 - len(pruned_indices) / len(feature_cols)) * 100, 1,
                ),
                cumulative_threshold_used=0.0,
                selection_method="lgbm_fallback",
                selected_indices=sorted(pruned_indices),
                selected_names=pruned_names,
                feature_importances={
                    pruned_names[i]: float(lgbm_gains[pruned_indices[i]])
                    for i in range(min(50, len(pruned_indices)))
                },
                mandatory_included=[],
            )

        else:
            # lgbm_gain only (default)
            pruned_indices = feature_selector.prune_by_lgbm(
                lgbm_model=student_model, feature_names=feature_cols,
            )
            pruned_names = [feature_cols[i] for i in pruned_indices]
            lgbm_gains = student_model.feature_importance(importance_type="gain")
            selection_result = FeatureSelectionResult(
                task_name=task_name,
                original_count=len(feature_cols),
                selected_count=len(pruned_indices),
                reduction_pct=round(
                    (1 - len(pruned_indices) / len(feature_cols)) * 100, 1,
                ),
                cumulative_threshold_used=0.0,
                selection_method="lgbm",
                selected_indices=sorted(pruned_indices),
                selected_names=pruned_names,
                feature_importances={
                    pruned_names[i]: float(lgbm_gains[pruned_indices[i]])
                    for i in range(min(50, len(pruned_indices)))
                },
                mandatory_included=[],
            )

        feature_selections[task_name] = selection_result
        logger.info(
            "  %s [%s]: %d/%d features selected (%.1f%% reduction)",
            task_name, selection_result.selection_method,
            selection_result.selected_count,
            selection_result.original_count,
            selection_result.reduction_pct,
        )

    # --- Step 4.6: Feature-level drift monitoring (DriftDetector / PSI) ---
    # This complements Step 4.5b (prediction drift via DistillationDriftMonitor).
    # Here we measure PSI on input features — avoids duplicating prediction drift.
    _feat_drift_cfg = _distillation_cfg.get("feature_drift", {})
    if _feat_drift_cfg.get("enabled", False):
        try:
            from core.monitoring.drift_detector import DriftDetector

            _feat_baseline_path = Path(
                _feat_drift_cfg.get("baseline_path", "outputs/distillation_baseline/feature_baseline.json")
            )
            _psi_warn = float(_feat_drift_cfg.get("psi_threshold_warning", 0.1))
            _psi_crit = float(_feat_drift_cfg.get("psi_threshold_critical", 0.25))
            _feat_drift_detector = DriftDetector(
                psi_threshold_warning=_psi_warn,
                psi_threshold_critical=_psi_crit,
            )

            logger.info("=" * 60)
            logger.info("Step 4.6: Feature-level drift monitoring (PSI)")
            logger.info("  warning=%.2f  critical=%.2f", _psi_warn, _psi_crit)
            logger.info("=" * 60)

            if _feat_baseline_path.exists():
                with open(_feat_baseline_path, encoding="utf-8") as _fdf:
                    _feat_baseline: Dict[str, Any] = json.load(_fdf)

                # Convert {col: list} baseline + current dict of numpy arrays
                _current_feat_dict: Dict[str, np.ndarray] = {
                    c: features[:, i]
                    for i, c in enumerate(feature_cols)
                    if i < features.shape[1]
                }
                _baseline_feat_dict: Dict[str, Any] = _feat_baseline.get("feature_samples", {})

                _feat_drift_result = _feat_drift_detector.detect_drift(
                    baseline_data=_baseline_feat_dict,
                    current_data=_current_feat_dict,
                )
                _fd_summary = _feat_drift_result["summary"]
                logger.info(
                    "Feature PSI — critical: %d, warning: %d, max_psi: %.4f",
                    _fd_summary["critical_count"],
                    _fd_summary["warning_count"],
                    _fd_summary["max_psi"],
                )
                with open(out_dir / "feature_drift_report.json", "w") as _fdf_out:
                    json.dump(
                        {
                            "summary": _fd_summary,
                            "warning_features": _feat_drift_result["warning_features"],
                            "critical_features": _feat_drift_result["critical_features"],
                            "psi_scores": {
                                k: round(v, 6) if isinstance(v, float) else v
                                for k, v in _feat_drift_result["psi_scores"].items()
                            },
                        },
                        _fdf_out, indent=2, default=str,
                    )
                logger.info("Feature drift report saved to %s/feature_drift_report.json", out_dir)
            else:
                logger.info(
                    "No feature baseline at %s — saving current as first baseline",
                    _feat_baseline_path,
                )

            # Save current feature samples as baseline (sampled to save space)
            _n_sample = min(10000, features.shape[0])
            _rng_sample = np.random.RandomState(42)
            _sample_idx = _rng_sample.choice(features.shape[0], size=_n_sample, replace=False)
            _new_baseline = {
                "feature_samples": {
                    c: features[_sample_idx, i].tolist()
                    for i, c in enumerate(feature_cols)
                    if i < features.shape[1]
                }
            }
            _feat_baseline_path.parent.mkdir(parents=True, exist_ok=True)
            with open(_feat_baseline_path, "w", encoding="utf-8") as _fdf_save:
                json.dump(_new_baseline, _fdf_save)
            logger.info("Feature baseline updated at %s", _feat_baseline_path)

        except Exception as _feat_drift_exc:
            logger.warning(
                "Feature drift monitoring failed (non-fatal): %s",
                _feat_drift_exc, exc_info=True,
            )
    else:
        logger.info("Feature drift monitoring disabled (distillation.feature_drift.enabled=false).")

    # --- Step 4.7: Audit log — distillation event ---
    _audit_cfg = _raw_yaml.get("monitoring", {}).get("audit", {})
    if _audit_cfg.get("enabled", False):
        try:
            from core.monitoring.audit_logger import AuditLogger

            _audit_logger = AuditLogger(
                s3_bucket=_audit_cfg.get("s3_bucket", ""),
                s3_prefix=_audit_cfg.get("s3_prefix", "audit_logs"),
            )
            _audit_logger.log_operation(
                operation="distillation:completed",
                user="system",
                status="SUCCESS",
                metadata={
                    "num_tasks": len(pipeline_config.tasks),
                    "distill_tasks": distill_tasks,
                    "direct_tasks": hardlabel_tasks,
                    "fidelity_passed": failed_count == 0,
                    "feature_count": len(feature_cols),
                    "sample_count": int(len(features)),
                    "config_path": config_path,
                },
            )
            logger.info("Audit log entry recorded for distillation:completed")
        except Exception as _audit_exc:
            logger.warning("Audit logging failed (non-fatal): %s", _audit_exc)

    # --- Step 5: Save LGBM models + fidelity + feature selections ---
    saved = trainer.save_students(
        str(model_out_dir),
        feature_selections=feature_selections,
        fidelity_results=fidelity_results,
    )

    # Step 5.5: Save calibration models
    saved_calibrators: Dict[str, str] = {}
    if calibrated_models:
        logger.info("Saving calibration models for %d task(s)...", len(calibrated_models))
        for task_name, calib in calibrated_models.items():
            task_dir = model_out_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            calib_path = task_dir / "calibrator.joblib"
            joblib.dump(calib, calib_path)
            saved_calibrators[task_name] = str(calib_path)
            logger.info("  [CALIBRATOR SAVED] %s -> %s", task_name, calib_path)

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("Distillation complete! (%d tasks saved)", len(saved))
    for task_name, path in saved.items():
        logger.info("  %s -> %s", task_name, path)
    logger.info("=" * 60)

    summary = {
        "tasks_distilled": distill_tasks,
        "tasks_direct_hardlabel": hardlabel_tasks,
        "num_students": len(saved),
        "adaptive_strategy": {
            "distill_count": len(distill_tasks),
            "direct_count": len(hardlabel_tasks),
            "threshold_rule": "binary: AUC>0.60, multiclass: F1>2/K, regression: R2>0.05",
        },
        "temperature": student_config.temperature,
        "alpha": student_config.alpha,
        "lgbm_params": student_config.lgbm_params,
        "task_lgbm_overrides": student_config.task_lgbm_overrides,
        "output_dir": str(model_out_dir),
        "feature_count": len(feature_cols),
        "sample_count": int(len(features)),
        "fidelity": {
            "all_passed": failed_count == 0,
            "passed": passed_count,
            "failed": failed_count,
            "per_task": {
                r.task_name: {
                    "passed": r.passed,
                    "metrics": {k: round(v, 4) for k, v in r.metrics.items()},
                }
                for r in fidelity_results
            },
        },
        "feature_selection": {
            task_name: {
                "selected": sel.selected_count,
                "original": sel.original_count,
                "reduction_pct": sel.reduction_pct,
                "method": sel.selection_method,
            }
            for task_name, sel in feature_selections.items()
        },
        "ig_dual_scores": _ig_raw_scores,
        "feature_selection_config": {
            "method": _fs_method,
            "ig_alpha": _ig_alpha,
            "cumulative_threshold": _cumulative_threshold,
            "ig_sample_size": _ig_sample_size,
        },
        "calibration": {
            "enabled": bool(calibrated_models),
            "method": _calib_method if calibrated_models else None,
            "calibrated_tasks": list(saved_calibrators.keys()),
            "calibrator_paths": saved_calibrators,
        },
    }

    summary_path = model_out_dir / "distillation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved to %s", summary_path)

    # --- Step 6: Governance report (optional) ---
    _gov_cfg = _raw_yaml.get("monitoring", {}).get("governance_report", {})
    if _gov_cfg.get("enabled", False):
        try:
            from core.monitoring.governance_report import GovernanceReportGenerator

            _gov_gen = GovernanceReportGenerator(
                s3_bucket=_gov_cfg.get("s3_bucket", ""),
                s3_prefix=_gov_cfg.get("s3_prefix", "governance_reports"),
                system_name=_gov_cfg.get("system_name", "PLE-Cluster-adaTT"),
            )

            # Build fidelity dict for drift_data shape expected by generator
            _drift_for_gov = {
                "summary": {
                    "drift_detected": bool(
                        summary.get("fidelity", {}).get("failed", 0) > 0
                    ),
                    "total_features": summary.get("feature_count", 0),
                    "warning_count": 0,
                    "critical_count": summary.get("fidelity", {}).get("failed", 0),
                    "max_psi": 0.0,
                    "avg_psi": 0.0,
                }
            }

            _gov_period = _gov_cfg.get("period", "monthly")
            _gov_report = _gov_gen.generate_report(
                period=_gov_period,
                drift_data=_drift_for_gov,
                model_changes=[
                    {
                        "event": "distillation_cycle",
                        "tasks_distilled": distill_tasks,
                        "tasks_direct": hardlabel_tasks,
                        "fidelity_all_passed": failed_count == 0,
                    }
                ],
            )
            _gov_report_dict = _gov_gen.to_dict(_gov_report)

            # Save locally
            _gov_path = out_dir / "governance_report.json"
            with open(_gov_path, "w", encoding="utf-8") as _gf:
                json.dump(_gov_report_dict, _gf, indent=2, default=str)
            logger.info("Governance report saved to %s", _gov_path)

            # Archive to S3 if configured
            _gov_s3_uri = _gov_gen.archive_report(_gov_report)
            if _gov_s3_uri:
                logger.info("Governance report archived: %s", _gov_s3_uri)

        except Exception as _gov_exc:
            logger.warning("Governance report generation failed (non-fatal): %s", _gov_exc, exc_info=True)

    # ---- Agent: distillation stage completion event (optional, non-blocking) ----
    # Emits a ChangeDetector event so OpsAgent CP4 can correlate distillation runs.
    try:
        from core.agent.change_detector import ChangeDetector as _ChangeDetector
        _cd = _ChangeDetector()
        _cd.on_pipeline_stage_complete(
            stage="stage_distill",
            artifacts={
                "summary_path": str(summary_path),
                "model_output_dir": str(model_out_dir),
                "output_dir": str(out_dir),
                "tasks_distilled": list(students.keys()) if "students" in dir() else [],
            },
        )
        logger.info("ChangeDetector: stage_distill event emitted")
    except Exception as _e:
        logger.debug("ChangeDetector stage_distill event failed (non-fatal): %s", _e)

    return str(summary_path)


# ---------------------------------------------------------------------------
# Entry points: SageMaker vs local
# ---------------------------------------------------------------------------

def _sagemaker_main() -> None:
    """Entry point when running inside SageMaker."""
    hp = get_hyperparameters()
    logger.info("SageMaker HPs: %s", hp)

    data_dir = os.environ["SM_CHANNEL_TRAIN"]
    model_dir = os.environ["SM_CHANNEL_MODEL"]
    output_dir = os.environ["SM_OUTPUT_DATA_DIR"]
    model_output_dir = os.environ.get("SM_MODEL_DIR", output_dir)

    # Config paths relative to /opt/ml/code (source_dir extraction root)
    default_config = "configs/pipeline.yaml"
    config_path_raw: str = hp.get("config", default_config)
    config_path = Path(config_path_raw)
    if not config_path.is_absolute():
        config_path = Path("/opt/ml/code") / config_path_raw
    config_path_str = str(config_path)

    dataset_config_path_str: Optional[str] = None
    dataset_config_raw: str = hp.get("dataset_config", "")
    if dataset_config_raw:
        dataset_config_path = Path(dataset_config_raw)
        if not dataset_config_path.is_absolute():
            dataset_config_path = Path("/opt/ml/code") / dataset_config_raw
        dataset_config_path_str = str(dataset_config_path)

    batch_size: int = int(hp.get("batch_size", 4096))
    temperature: Optional[float] = None
    if "temperature" in hp:
        temperature = float(hp["temperature"])
    alpha: Optional[float] = None
    if "alpha" in hp:
        alpha = float(hp["alpha"])
    skip_fidelity_gate: bool = bool(hp.get("skip_fidelity_gate", False))

    logger.info("SageMaker distillation configuration:")
    logger.info("  SM_CHANNEL_TRAIN: %s", data_dir)
    logger.info("  SM_CHANNEL_MODEL: %s", model_dir)
    logger.info("  SM_OUTPUT_DATA_DIR: %s", output_dir)
    logger.info("  SM_MODEL_DIR: %s", model_output_dir)
    logger.info("  config: %s", config_path_str)
    logger.info("  dataset_config: %s", dataset_config_path_str)
    logger.info("  batch_size: %d", batch_size)
    logger.info("  temperature: %s", temperature)
    logger.info("  alpha: %s", alpha)
    logger.info("  skip_fidelity_gate: %s", skip_fidelity_gate)

    run_distillation(
        data_dir=data_dir,
        model_dir=model_dir,
        config_path=config_path_str,
        output_dir=output_dir,
        model_output_dir=model_output_dir,
        batch_size=batch_size,
        temperature=temperature,
        alpha=alpha,
        skip_fidelity_gate=skip_fidelity_gate,
        dataset_config_path=dataset_config_path_str,
    )


def _local_main() -> None:
    """Entry point for local CLI execution."""
    parser = argparse.ArgumentParser(
        description="Run PLE->LGBM distillation locally (SageMaker entry point).",
    )
    parser.add_argument("--data-dir", required=True,
                        help="Phase 0 data directory (features.parquet + schemas)")
    parser.add_argument("--teacher-checkpoint", required=True,
                        help="Path to teacher .pt checkpoint (or its parent directory)")
    parser.add_argument("--config", default="configs/pipeline.yaml",
                        help="Common pipeline YAML")
    parser.add_argument("--dataset", default="",
                        help="Dataset-specific YAML deep-merged on top of --config")
    parser.add_argument("--output-dir", default="outputs/distillation",
                        help="Directory for fidelity/drift reports")
    parser.add_argument("--model-output-dir", default=None,
                        help="Directory for LGBM models + summary (default: --output-dir)")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Teacher inference batch size")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Distillation temperature override")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Hard label weight override")
    parser.add_argument("--skip-fidelity-gate", action="store_true",
                        help="Continue despite fidelity failures")
    args = parser.parse_args()

    # If teacher_checkpoint points to a .pt file, use its parent as model_dir
    teacher_path = Path(args.teacher_checkpoint)
    if teacher_path.is_file():
        model_dir = str(teacher_path.parent)
    else:
        model_dir = str(teacher_path)

    model_output_dir = args.model_output_dir or args.output_dir

    run_distillation(
        data_dir=args.data_dir,
        model_dir=model_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        model_output_dir=model_output_dir,
        batch_size=args.batch_size,
        temperature=args.temperature,
        alpha=args.alpha,
        skip_fidelity_gate=args.skip_fidelity_gate,
        dataset_config_path=args.dataset or None,
    )


def main() -> None:
    try:
        if _is_sagemaker():
            logger.info("Detected SageMaker environment")
            _sagemaker_main()
        else:
            logger.info("Running in local mode")
            _local_main()
    except Exception:
        logger.exception("distill_entry.py failed with unhandled exception")
        sys.exit(1)


if __name__ == "__main__":
    main()
