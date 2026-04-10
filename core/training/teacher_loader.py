"""
Teacher model loading utilities for knowledge distillation.

Extracted from StudentTrainer to keep that class focused on orchestration.
Provides:
  - load_teacher_model(): reconstruct a PLEModel from a checkpoint
  - build_student_target(): derive per-task LGBM blended targets
  - save_task_artifacts(): write per-task metadata/feature/fidelity files
  - _WrappedDataLoader: adapt TensorDataset loaders for SoftLabelGenerator
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from core.model.ple.config import PLEConfig
from core.model.ple.model import PLEModel, PLEInput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Teacher loading
# ---------------------------------------------------------------------------


def load_teacher_model(
    checkpoint_path: str,
    device: torch.device,
    pipeline_config: Optional[Dict[str, Any]] = None,
) -> Tuple[PLEModel, PLEConfig]:
    """Load a trained PLE teacher from checkpoint + config.json.

    Reuses ``containers.training.train.build_model()`` to guarantee the
    reconstructed architecture is byte-identical to the one that was
    trained.  The ``config.json`` file next to the checkpoint stores
    ``feature_schema`` and ``label_schema`` — exactly what
    ``build_model()`` needs.

    Args:
        checkpoint_path: Path to the ``.pth`` / ``.pt`` checkpoint.
        device: Torch device.
        pipeline_config: Raw YAML dict (optional fallback if config.json
            is missing).

    Returns:
        ``(model, ple_config)`` with model in eval mode.
    """
    import json as _json
    from containers.training.train import build_model

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # --- Load schemas from config.json alongside the checkpoint ---------
    config_json_path = Path(checkpoint_path).parent / "config.json"
    if config_json_path.exists():
        with open(config_json_path, encoding="utf-8") as f:
            saved_config = _json.load(f)
        feature_schema = saved_config.get("feature_schema", {})
        label_schema = saved_config.get("label_schema", {})
        logger.info(
            "Loaded schemas from %s (features=%d, tasks=%d)",
            config_json_path,
            len(feature_schema.get("columns", [])),
            len(label_schema.get("tasks", [])),
        )
    else:
        logger.warning("config.json not found next to checkpoint — using fallback")
        feature_schema = {}
        label_schema = {}

    # Merge pipeline_config into label_schema for any missing sections
    if pipeline_config and not label_schema.get("model"):
        label_schema["model"] = pipeline_config.get("model", {})
    if pipeline_config and not label_schema.get("tasks"):
        label_schema["tasks"] = pipeline_config.get("tasks", [])

    # input_dim from feature schema
    input_dim = len(feature_schema.get("columns", []))
    if input_dim == 0:
        input_dim = checkpoint.get("ple_config", {}).get("input_dim", 128)
        logger.warning("Could not determine input_dim from schema — using %d", input_dim)

    # Hyperparameter overrides from checkpoint (ablation scenario etc.)
    hp = checkpoint.get("ablation", {})

    # Build the exact same model architecture that train.py would build
    teacher, ple_config = build_model(
        feature_schema=feature_schema,
        label_schema=label_schema,
        hp=hp,
        input_dim=input_dim,
        device=device,
        config=pipeline_config or {},
    )

    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.to(device)
    teacher.eval()

    logger.info(
        "Teacher loaded from %s (%d params)",
        checkpoint_path,
        sum(p.numel() for p in teacher.parameters()),
    )
    return teacher, ple_config


# ---------------------------------------------------------------------------
# Per-task blended target builder
# ---------------------------------------------------------------------------


def build_student_target(
    task_type: str,
    hard: np.ndarray,
    soft: np.ndarray,
    alpha: float,
    num_classes: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Derive blended LGBM target and objective params for one task.

    Args:
        task_type: Task type string (``"binary"``, ``"multiclass"``,
            ``"regression"``, etc.).
        hard: Hard (ground truth) label array.
        soft: Soft (teacher) label array.  May be 1-D or 2-D for multiclass.
        alpha: Hard label weight (``1 - alpha`` = soft label weight).
        num_classes: Number of classes for multiclass tasks.

    Returns:
        ``(target, sample_weight, param_overrides)`` where ``param_overrides``
        contains the LightGBM ``objective`` / ``metric`` / ``num_class`` keys
        that should be applied on top of the base params dict.
    """
    sample_weight: Optional[np.ndarray] = None
    param_overrides: Dict[str, Any] = {}

    if task_type == "binary":
        target = alpha * hard.astype(np.float64) + (1 - alpha) * soft.astype(np.float64)
        param_overrides = {"objective": "binary", "metric": "auc"}

    elif task_type == "multiclass":
        target = hard.astype(int)
        if soft.ndim > 1:
            sample_weight = soft.max(axis=1)
        else:
            agreement = (soft.astype(int) == hard.astype(int)).astype(np.float64)
            sample_weight = 0.5 + 0.5 * agreement
        param_overrides = {
            "objective": "multiclass",
            "num_class": num_classes,
            "metric": "multi_logloss",
        }

    elif task_type == "regression":
        target = alpha * hard.astype(np.float64) + (1 - alpha) * soft.astype(np.float64)
        param_overrides = {"objective": "regression", "metric": "rmse"}

    else:
        target = hard
        param_overrides = {"objective": "regression", "metric": "rmse"}

    return target, sample_weight, param_overrides


# ---------------------------------------------------------------------------
# Per-task artifact saver
# ---------------------------------------------------------------------------


def save_task_artifacts(
    task_dir: Path,
    task_name: str,
    model: Any,
    task_type: str,
    temperature: float,
    alpha: float,
    feature_columns: List[str],
    feature_selection: Optional[Any],
    fidelity: Optional[Dict[str, Any]],
) -> str:
    """Save a single student model plus optional traceability artifacts.

    Writes:
      - ``model.lgbm``
      - ``metadata.json``
      - ``selected_features.json``  (if feature_selection is provided)
      - ``fidelity.json``           (if fidelity is provided)

    Args:
        task_dir: Output directory for this task.
        task_name: Task name (used in metadata).
        model: Trained LightGBM Booster.
        task_type: Task type string (stored in metadata).
        temperature: Distillation temperature (stored in metadata).
        alpha: Hard label weight (stored in metadata).
        feature_columns: Feature column names.
        feature_selection: FeatureSelectionResult dataclass or dict, or None.
        fidelity: Fidelity validation result dict, or None.

    Returns:
        Absolute path to the saved ``model.lgbm`` file.
    """
    from dataclasses import asdict

    task_dir.mkdir(parents=True, exist_ok=True)

    model_path = str(task_dir / "model.lgbm")
    model.save_model(model_path)

    meta = {
        "task_name": task_name,
        "task_type": task_type,
        "num_trees": model.num_trees(),
        "num_features": model.num_feature(),
        "temperature": temperature,
        "alpha": alpha,
        "feature_columns": feature_columns,
    }
    with open(task_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    if feature_selection is not None:
        fs_dict = (
            asdict(feature_selection)
            if hasattr(feature_selection, "__dataclass_fields__")
            else dict(feature_selection)
        )
        selected = {
            "indices": fs_dict.get("selected_indices", []),
            "names": fs_dict.get("selected_names", []),
            "count": fs_dict.get("selected_count", 0),
            "original_count": fs_dict.get("original_count", 0),
            "reduction_pct": fs_dict.get("reduction_pct", 0.0),
            "selection_method": fs_dict.get("selection_method", ""),
        }
        with open(task_dir / "selected_features.json", "w") as f:
            json.dump(selected, f, indent=2)

    if fidelity is not None:
        with open(task_dir / "fidelity.json", "w") as f:
            json.dump(fidelity, f, indent=2, default=str)

    return model_path


# ---------------------------------------------------------------------------
# DataLoader wrapper for TensorDataset compatibility
# ---------------------------------------------------------------------------


class _WrappedDataLoader:
    """Wraps a DataLoader so raw tensor tuples become PLEInput-compatible dicts.

    The SoftLabelGenerator._prepare_inputs expects PLEInput, dicts, or objects
    with a to_ple_input() method.  When the DataLoader yields TensorDataset
    tuples (e.g. ``(features_tensor,)``), this wrapper converts them to dicts.
    """

    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def __iter__(self):
        for batch in self._loader:
            if isinstance(batch, (list, tuple)):
                yield {"features": batch[0]}
            elif isinstance(batch, dict) or isinstance(batch, PLEInput):
                yield batch
            else:
                yield {"features": batch}

    def __len__(self):
        return len(self._loader)
