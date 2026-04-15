#!/usr/bin/env python3
"""SageMaker entry point for PLE -> LGBM knowledge distillation.

Reads channel paths from SageMaker env vars (SM_CHANNEL_TRAIN, SM_CHANNEL_MODEL,
SM_OUTPUT_DATA_DIR, SM_MODEL_DIR) and hyperparameters from SM_HPS.
Falls back to CLI argument parsing when running locally (SM_CHANNEL_TRAIN absent).

Heavy logic is extracted into sibling modules:
  data_prep.py, threshold_gate.py, train_students.py, calibration.py,
  fidelity.py, drift_step.py, feature_selection_step.py, post_steps.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
# Fall back to default stdout if fileno() is unavailable (e.g. SageMaker CloudWatch)
try:
    _utf8_stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
except Exception:
    _utf8_stdout = sys.stdout
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
    return "SM_CHANNEL_TRAIN" in os.environ or "SM_MODEL_DIR" in os.environ


# ---------------------------------------------------------------------------
# Checkpoint discovery (mirrors eval_entry.py)
# ---------------------------------------------------------------------------

def _find_checkpoint(model_dir: str) -> str:
    """Return path to best.pt, or the most recently modified *.pt/.pth file."""
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
    """Orchestrate the full PLE->LGBM distillation pipeline.

    Returns path to the written distillation_summary.json.
    """
    import numpy as np
    import yaml as _yaml
    import joblib

    from core.pipeline.config import load_config, load_merged_config
    from containers.distillation.data_prep import (
        load_data_pyarrow,
        prepare_features_and_labels,
        build_trainer,
    )
    from containers.distillation.threshold_gate import evaluate_teacher_thresholds
    from containers.distillation.train_students import train_direct_lgbm
    from containers.distillation.calibration import calibrate_students
    from containers.distillation.fidelity import validate_fidelity
    from containers.distillation.drift_step import run_prediction_drift, run_feature_drift
    from containers.distillation.feature_selection_step import run_feature_selection
    from containers.distillation.post_steps import build_summary, run_post_distillation

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

    # --- Load data + quality gate + extract numpy arrays ---
    logger.info("Loading Phase 0 data from %s", data_dir)
    table, feature_schema, label_schema, split_indices = load_data_pyarrow(channel_path)
    checkpoint_path = _find_checkpoint(model_dir)

    feature_cols, features, hard_labels = prepare_features_and_labels(
        table, pipeline_config, feature_schema, checkpoint_path, skip_fidelity_gate,
    )

    # --- Build StudentTrainer ---
    student_config, trainer = build_trainer(
        _distillation_cfg, checkpoint_path, model_out_dir,
        pipeline_config, feature_cols, temperature, alpha,
    )

    # --- Step 2: Generate soft labels from teacher ---
    logger.info("Generating soft labels from teacher checkpoint: %s", checkpoint_path)
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    trainer.load_teacher(checkpoint_path=checkpoint_path, pipeline_config=_raw_yaml)

    features_tensor = torch.tensor(features, dtype=torch.float32)
    dataset = TensorDataset(features_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    trainer.generate_soft_labels(data_loader=loader, save_path=None)

    # --- Step 2.5: Teacher threshold gating (adaptive distillation) ---
    soft_labels = trainer.get_soft_labels()
    distill_tasks, hardlabel_tasks, skip_tasks = evaluate_teacher_thresholds(
        pipeline_config, soft_labels, hard_labels, _distillation_cfg,
    )

    # --- Step 3: Train LGBM students ---
    trainer.config.enabled_tasks = distill_tasks
    logger.info("Training LGBM student models (distill: %d tasks)...", len(distill_tasks))
    students = trainer.train_students(features, hard_labels)

    direct_models = train_direct_lgbm(
        hardlabel_tasks, pipeline_config, hard_labels, features, student_config,
        model_output_dir=str(model_out_dir),
    )
    students.update(direct_models)

    # --- Step 3.5: Calibration ---
    calibrated_models = calibrate_students(
        students, hard_labels, features, pipeline_config, _distillation_cfg,
    )

    # --- Step 4: Fidelity validation ---
    fidelity_results, fidelity_report = validate_fidelity(
        pipeline_config=pipeline_config,
        students=students,
        calibrated_models=calibrated_models,
        features=features,
        hard_labels=hard_labels,
        student_config=student_config,
        distillation_cfg=_distillation_cfg,
        trainer=trainer,
        out_dir=out_dir,
        skip_gate=skip_fidelity_gate,
    )
    failed_count = fidelity_report["failed"]
    passed_count = fidelity_report["passed"]

    # --- Step 4.5b: Prediction drift monitoring ---
    run_prediction_drift(
        distillation_cfg=_distillation_cfg,
        pipeline_config=pipeline_config,
        students=students,
        calibrated_models=calibrated_models,
        features=features,
        hard_labels=hard_labels,
        student_config=student_config,
        out_dir=out_dir,
    )

    # --- Step 4.5: Feature selection ---
    feature_selections, _ig_raw_scores = run_feature_selection(
        pipeline_config=pipeline_config,
        students=students,
        features=features,
        feature_cols=feature_cols,
        trainer=trainer,
        config_path=config_path,
        distillation_cfg=_distillation_cfg,
    )

    # --- Step 4.6: Feature drift monitoring ---
    run_feature_drift(
        distillation_cfg=_distillation_cfg,
        features=features,
        feature_cols=feature_cols,
        out_dir=out_dir,
    )

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

    summary = build_summary(
        distillation_cfg=_distillation_cfg,
        distill_tasks=distill_tasks,
        hardlabel_tasks=hardlabel_tasks,
        student_config=student_config,
        model_out_dir=model_out_dir,
        feature_cols=feature_cols,
        features_len=len(features),
        fidelity_results=fidelity_results,
        failed_count=failed_count,
        passed_count=passed_count,
        feature_selections=feature_selections,
        ig_raw_scores=_ig_raw_scores,
        calibrated_models=calibrated_models,
        saved_calibrators=saved_calibrators,
    )
    summary["num_students"] = len(saved)

    summary_path = model_out_dir / "distillation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved to %s", summary_path)

    # --- Steps 4.7, 6, ChangeDetector ---
    run_post_distillation(
        raw_yaml=_raw_yaml,
        pipeline_config=pipeline_config,
        distill_tasks=distill_tasks,
        hardlabel_tasks=hardlabel_tasks,
        failed_count=failed_count,
        feature_cols=feature_cols,
        features_len=len(features),
        config_path=config_path,
        summary_path=summary_path,
        model_out_dir=model_out_dir,
        out_dir=out_dir,
        students=students,
        summary=summary,
    )

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

    # Config paths: centralized resolver (SageMaker + local)
    from containers.path_resolver import resolve_config_path
    config_path_str = resolve_config_path(hp.get("config", "configs/pipeline.yaml"))

    dataset_config_path_str: Optional[str] = None
    dataset_config_raw: str = hp.get("dataset_config", "")
    if dataset_config_raw:
        dataset_config_path_str = resolve_config_path(dataset_config_raw)

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

    from containers.path_resolver import resolve_config_path
    resolved_config = resolve_config_path(args.config)

    run_distillation(
        data_dir=args.data_dir,
        model_dir=model_dir,
        config_path=resolved_config,
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
