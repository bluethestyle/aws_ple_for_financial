"""
Student Model Trainer -- distillation from PLE teacher to LGBM students.

SageMaker Step Functions flow:
  Step 1: PLE Training Job -> model.pt (S3)
  Step 2: Soft Label Generation -> soft_labels.parquet (S3)
  Step 3: LGBM Student Training x N tasks (parallel) -> student_models/ (S3)

This module handles Steps 2 and 3. Step 1 is handled by PLETrainer.

Usage:
    trainer = StudentTrainer.from_config(config)
    trainer.generate_soft_labels(teacher_checkpoint, data_loader)
    trainer.train_students()
    trainer.save_students(output_dir)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.model.ple.config import PLEConfig, ExpertConfig
from core.model.ple.model import PLEModel, PLEInput, PLEOutput
from core.pipeline.config import TaskSpec
from core.training.distillation import DistillationConfig, SoftLabelGenerator

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class StudentConfig:
    """Configuration for LGBM student training via knowledge distillation.

    Controls teacher loading, soft label generation parameters, and
    per-task LGBM hyperparameters.

    Args:
        teacher_checkpoint: S3 or local path to the teacher ``.pt`` checkpoint.
        soft_label_path: S3 or local path to save/load soft labels parquet.
        student_output_dir: Directory to save trained student models.
        temperature: Soft label temperature (higher = softer distributions).
        alpha: Hard label weight in blended target (1-alpha = soft label weight).
        lgbm_params: Default LightGBM parameters applied to all tasks.
        task_lgbm_overrides: Per-task LGBM parameter overrides.
        enabled_tasks: Restrict distillation to these tasks (``None`` = all).
    """

    teacher_checkpoint: str = ""
    soft_label_path: str = ""
    student_output_dir: str = "student_models"
    temperature: float = 5.0
    alpha: float = 0.3  # hard label weight

    # LGBM defaults
    lgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary",  # overridden per task
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
    })

    # Per-task LGBM overrides
    task_lgbm_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Which tasks to distill (None = all)
    enabled_tasks: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StudentConfig":
        """Build StudentConfig from a dict (e.g. YAML output).

        Supports both flat and nested formats.  Unknown keys are ignored.
        """
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


# ============================================================================
# Student Trainer
# ============================================================================


class StudentTrainer:
    """Orchestrates teacher -> student knowledge distillation.

    Workflow:
        1. ``load_teacher()`` -- Load a trained PLE checkpoint.
        2. ``generate_soft_labels()`` -- Run teacher inference to produce
           temperature-softened predictions per task.
        3. ``train_students()`` -- Train LGBM models on blended
           (hard + soft) targets.
        4. ``save_students()`` -- Persist models and metadata to disk/S3.

    Args:
        config: Student training configuration.
        task_specs: Task specifications from PipelineConfig.
        feature_columns: Names of the feature columns used by the students.
        device: Torch device for teacher inference.
    """

    def __init__(
        self,
        config: StudentConfig,
        task_specs: List[TaskSpec],
        feature_columns: List[str],
        device: Optional[torch.device] = None,
        audit_store: Optional[Any] = None,
    ) -> None:
        self._audit_store = audit_store
        self.config = config
        self.task_specs = {t.name: t for t in task_specs}
        self.feature_columns = feature_columns
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._teacher: Optional[PLEModel] = None
        self._soft_labels: Optional[Dict[str, np.ndarray]] = None
        self._students: Dict[str, Any] = {}  # task_name -> trained LGBM model

    # ------------------------------------------------------------------
    # Teacher loading
    # ------------------------------------------------------------------

    def load_teacher(self, checkpoint_path: Optional[str] = None) -> PLEModel:
        """Load a trained PLE model as teacher.

        Args:
            checkpoint_path: Override ``config.teacher_checkpoint``.

        Returns:
            The teacher PLEModel in eval mode.
        """
        path = checkpoint_path or self.config.teacher_checkpoint
        if not path:
            raise ValueError(
                "No teacher checkpoint path provided. Set "
                "config.teacher_checkpoint or pass checkpoint_path."
            )

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Reconstruct PLEConfig from checkpoint
        # train.py saves "ple_config" as a dict with key fields,
        # and "config" as the full raw YAML dict.
        ple_dict = checkpoint.get("ple_config", {})
        raw_config = checkpoint.get("config", {})

        model_cfg = raw_config.get("model", {})
        expert_cfg = model_cfg.get("expert_config", {})
        mlp_cfg = expert_cfg.get("mlp", {})
        ple_yaml = model_cfg.get("ple", {})

        task_names = ple_dict.get("task_names", [])

        # Detect actual input_dim from saved model weights
        # The first shared expert's first linear layer has shape (hidden, input_dim)
        state_dict = checkpoint["model_state_dict"]
        input_dim = ple_dict.get("input_dim", 128)
        for key, tensor in state_dict.items():
            if "extraction_layers.0.shared_experts.0" in key and "weight" in key:
                input_dim = tensor.shape[1]  # Linear weight is (out, in)
                logger.info("Detected input_dim=%d from state_dict key %s", input_dim, key)
                break
        expert_hidden = mlp_cfg.get("hidden_dims", [input_dim * 2, input_dim])
        expert_output = ple_yaml.get("extraction_dim", 32)

        shared_expert = ExpertConfig(
            hidden_dims=expert_hidden,
            output_dim=expert_output,
        )
        task_expert = ExpertConfig(
            hidden_dims=expert_hidden,
            output_dim=expert_output,
        )

        ple_config = PLEConfig(
            input_dim=input_dim,
            task_names=task_names,
            num_shared_experts=ple_dict.get("num_shared_experts", 2),
            num_extraction_layers=ple_dict.get("num_extraction_layers", 2),
            shared_expert=shared_expert,
            task_expert=task_expert,
        )

        # Apply task overrides from raw config
        for t in raw_config.get("tasks", []):
            ple_config.task_overrides[t["name"]] = {
                "task_type": t.get("type", "binary"),
                "output_dim": t.get("num_classes", 1),
            }

        teacher = PLEModel(ple_config)
        teacher.load_state_dict(checkpoint["model_state_dict"])
        teacher.to(self.device)
        teacher.eval()

        self._teacher = teacher
        logger.info(
            "Teacher loaded from %s (%d params)",
            path,
            sum(p.numel() for p in teacher.parameters()),
        )
        return teacher

    # ------------------------------------------------------------------
    # Soft label generation
    # ------------------------------------------------------------------

    def generate_soft_labels(
        self,
        data_loader: DataLoader,
        save_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate soft labels from teacher for all tasks.

        Runs teacher inference on the full dataset and stores
        temperature-softened predictions per task.

        Args:
            data_loader: DataLoader yielding PLEInput-compatible batches.
                Accepts PLEInput, dicts with a ``"features"`` key, or
                raw TensorDataset tuples (feature tensors only).
            save_path: If provided, save soft labels as parquet to this path.

        Returns:
            ``{task_name: soft_label_array}`` dict.
        """
        if self._teacher is None:
            self.load_teacher()

        # Build a DistillationConfig to match our temperature and task list
        distill_config = DistillationConfig(
            temperature=self.config.temperature,
            enabled_tasks=self.config.enabled_tasks,
        )

        # Populate task type info so SoftLabelGenerator can apply correct softening
        for task_name, task_spec in self.task_specs.items():
            distill_config.task_configs[task_name] = {
                "task_type": task_spec.type,
                "weight": task_spec.loss_weight,
            }

        generator = SoftLabelGenerator(
            teacher=self._teacher,
            config=distill_config,
            device=self.device,
        )

        # Wrap the data_loader if it yields raw tensor tuples (from TensorDataset)
        # so the SoftLabelGenerator._prepare_inputs can handle them
        wrapped_loader = _WrappedDataLoader(data_loader)

        soft_labels_tensors = generator.generate(wrapped_loader)

        # Convert to numpy
        soft_labels: Dict[str, np.ndarray] = {}
        for task_name, tensor in soft_labels_tensors.items():
            soft_labels[task_name] = tensor.numpy()

        self._soft_labels = soft_labels

        # Save to disk/S3
        target_path = save_path or self.config.soft_label_path
        if target_path:
            self._save_soft_labels(target_path)

        logger.info(
            "Soft labels generated: %d tasks, %d samples each",
            len(soft_labels),
            len(next(iter(soft_labels.values()))) if soft_labels else 0,
        )
        return soft_labels

    def load_soft_labels(self, path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Load previously saved soft labels from parquet.

        Args:
            path: Override ``config.soft_label_path``.

        Returns:
            ``{task_name: soft_label_array}`` dict.
        """
        import pandas as pd

        path = path or self.config.soft_label_path
        if not path:
            raise ValueError("No soft label path provided.")

        df = pd.read_parquet(path)

        soft_labels: Dict[str, np.ndarray] = {}
        for col in df.columns:
            if col.startswith("soft_"):
                task_name = col[5:]  # strip "soft_" prefix
                soft_labels[task_name] = df[col].values

        self._soft_labels = soft_labels
        logger.info("Soft labels loaded from %s: %d tasks", path, len(soft_labels))
        return soft_labels

    def get_soft_labels(self) -> Dict[str, np.ndarray]:
        """Return the cached soft labels (teacher predictions).

        Must be called after :meth:`generate_soft_labels` or
        :meth:`load_soft_labels`.

        Returns:
            ``{task_name: soft_label_array}`` dict.

        Raises:
            RuntimeError: If soft labels have not been generated/loaded.
        """
        if self._soft_labels is None:
            raise RuntimeError(
                "Soft labels not available. "
                "Call generate_soft_labels() or load_soft_labels() first."
            )
        return dict(self._soft_labels)

    # ------------------------------------------------------------------
    # Student training
    # ------------------------------------------------------------------

    def train_students(
        self,
        features: np.ndarray,
        hard_labels: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Train LGBM student models for each task.

        For each task:
          - Constructs blended target: ``alpha * hard_label + (1-alpha) * soft_label``
          - Configures LGBM objective based on task type
          - Trains and stores the model

        Contrastive tasks are automatically skipped (not suitable for LGBM).

        Args:
            features: Feature matrix ``(n_samples, n_features)``.  Should
                contain only tabular features (no sequences/graphs).
            hard_labels: Original ground truth labels per task.

        Returns:
            ``{task_name: trained_lgbm_model}`` dict.

        Raises:
            RuntimeError: If soft labels have not been generated or loaded.
            ImportError: If lightgbm is not installed.
        """
        if self._soft_labels is None:
            raise RuntimeError(
                "Call generate_soft_labels() or load_soft_labels() first."
            )

        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "lightgbm is required for student training. "
                "Install with: pip install lightgbm"
            )

        alpha = self.config.alpha
        enabled = self.config.enabled_tasks or list(self.task_specs.keys())

        for task_name in enabled:
            if task_name not in self._soft_labels:
                logger.warning(
                    "No soft labels for task '%s', skipping.", task_name
                )
                continue
            if task_name not in hard_labels:
                logger.warning(
                    "No hard labels for task '%s', skipping.", task_name
                )
                continue

            task_spec = self.task_specs.get(task_name)
            if task_spec is None:
                logger.warning(
                    "No task spec for '%s', skipping.", task_name
                )
                continue

            # Skip contrastive tasks -- not suitable for LGBM
            if task_spec.type == "contrastive":
                logger.info(
                    "Skipping contrastive task '%s' (not suitable for LGBM).",
                    task_name,
                )
                continue

            soft = np.asarray(self._soft_labels[task_name]).squeeze()
            hard = np.asarray(hard_labels[task_name]).squeeze()

            # Build LGBM params
            params = dict(self.config.lgbm_params)
            task_overrides = self.config.task_lgbm_overrides.get(task_name, {})
            params.update(task_overrides)

            # Blended target and objective configuration
            sample_weight = None

            if task_spec.type == "binary":
                # Blend hard and soft labels for binary classification
                target = alpha * hard.astype(np.float64) + (1 - alpha) * soft.astype(np.float64)
                params["objective"] = "binary"
                params["metric"] = "auc"

            elif task_spec.type == "multiclass":
                # For multiclass: use hard labels as target, weight samples
                # by teacher confidence (max softmax probability).
                # Soft labels from teacher are class probabilities or argmax;
                # LGBM multiclass requires integer targets.
                target = hard.astype(int)
                if soft.ndim > 1:
                    # soft is (n_samples, n_classes) probability matrix
                    sample_weight = soft.max(axis=1)  # confidence weighting
                else:
                    # soft is teacher argmax predictions
                    # Weight higher when teacher agrees with hard label
                    agreement = (soft.astype(int) == hard.astype(int)).astype(np.float64)
                    sample_weight = 0.5 + 0.5 * agreement

                params["objective"] = "multiclass"
                params["num_class"] = task_spec.num_classes
                params["metric"] = "multi_logloss"

            elif task_spec.type == "regression":
                # Blend hard and soft targets for regression
                target = alpha * hard.astype(np.float64) + (1 - alpha) * soft.astype(np.float64)
                params["objective"] = "regression"
                params["metric"] = "rmse"

            else:
                # Fallback: use hard labels
                target = hard
                params["objective"] = "regression"
                params["metric"] = "rmse"

            # Extract n_estimators for train API (not a native lgb.train param)
            n_estimators = params.pop("n_estimators", 500)

            # Build LightGBM Dataset
            train_data = lgb.Dataset(
                features,
                label=target,
                weight=sample_weight,
                feature_name=self.feature_columns if len(self.feature_columns) == features.shape[1] else "auto",
                free_raw_data=False,
            )

            # Train
            model = lgb.train(
                params,
                train_data,
                num_boost_round=n_estimators,
            )

            self._students[task_name] = model
            logger.info(
                "Student trained: %s (type=%s, trees=%d, features=%d)",
                task_name,
                task_spec.type,
                model.num_trees(),
                features.shape[1],
            )

        logger.info(
            "All students trained: %d/%d tasks",
            len(self._students),
            len(enabled),
        )

        if self._audit_store:
            for task_name, model in self._students.items():
                self._audit_store.log_event("distillation", {
                    "pk": task_name,
                    "task": task_name,
                    "teacher_checkpoint": self.config.teacher_checkpoint,
                    "temperature": self.config.temperature,
                    "alpha": self.config.alpha,
                    "num_trees": model.num_trees() if hasattr(model, 'num_trees') else 0,
                })

        return dict(self._students)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_students(
        self,
        output_dir: Optional[str] = None,
        feature_selections: Optional[Dict[str, Any]] = None,
        fidelity_results: Optional[List[Any]] = None,
    ) -> Dict[str, str]:
        """Save trained student models with optional traceability artifacts.

        Each model is saved as::

            {output_dir}/{task_name}/model.lgbm
            {output_dir}/{task_name}/metadata.json
            {output_dir}/{task_name}/selected_features.json   (if provided)
            {output_dir}/{task_name}/fidelity.json            (if provided)

        Args:
            output_dir: Override ``config.student_output_dir``.
            feature_selections: ``{task_name: FeatureSelectionResult or dict}``
                with per-task feature selection outcomes.  When provided,
                each task directory receives a ``selected_features.json``
                containing the selected feature indices, names, and count
                so that the serving layer knows which features to extract.
            fidelity_results: List of ``FidelityResult`` or dicts from
                distillation validation.  When provided, each matching
                task directory receives a ``fidelity.json`` with the
                validation metrics and pass/fail status.

        Returns:
            ``{task_name: saved_model_path}`` dict.
        """
        from dataclasses import asdict

        output_dir = output_dir or self.config.student_output_dir
        feature_selections = feature_selections or {}
        fidelity_results = fidelity_results or []

        # Index fidelity results by task name for easy lookup
        fidelity_by_task: Dict[str, Dict[str, Any]] = {}
        for fr in fidelity_results:
            fr_dict = asdict(fr) if hasattr(fr, "__dataclass_fields__") else dict(fr)
            task = fr_dict.get("task_name", "unknown")
            fidelity_by_task[task] = fr_dict

        saved_paths: Dict[str, str] = {}
        for task_name, model in self._students.items():
            task_dir = Path(output_dir) / task_name
            task_dir.mkdir(parents=True, exist_ok=True)

            model_path = str(task_dir / "model.lgbm")
            model.save_model(model_path)

            # Save metadata
            meta = {
                "task_name": task_name,
                "task_type": (
                    self.task_specs[task_name].type
                    if task_name in self.task_specs
                    else "unknown"
                ),
                "num_trees": model.num_trees(),
                "num_features": model.num_feature(),
                "temperature": self.config.temperature,
                "alpha": self.config.alpha,
                "feature_columns": self.feature_columns,
            }
            with open(task_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            # Save feature selection results
            if task_name in feature_selections:
                fs = feature_selections[task_name]
                fs_dict = (
                    asdict(fs) if hasattr(fs, "__dataclass_fields__") else dict(fs)
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

            # Save fidelity validation results
            if task_name in fidelity_by_task:
                with open(task_dir / "fidelity.json", "w") as f:
                    json.dump(fidelity_by_task[task_name], f, indent=2, default=str)

            saved_paths[task_name] = model_path

        logger.info(
            "Students saved to %s: %d models", output_dir, len(saved_paths)
        )
        return saved_paths

    def load_student(self, task_name: str, model_path: str) -> None:
        """Load a previously saved student model.

        Args:
            task_name: Task name to register the model under.
            model_path: Path to the ``.lgbm`` model file.
        """
        import lightgbm as lgb

        self._students[task_name] = lgb.Booster(model_file=model_path)
        logger.info("Student loaded: %s from %s", task_name, model_path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, task_name: str, features: np.ndarray) -> np.ndarray:
        """Run inference on a trained student model.

        Args:
            task_name: Which student to use.
            features: Feature matrix ``(n_samples, n_features)``.

        Returns:
            Prediction array from the student model.

        Raises:
            KeyError: If no student model exists for the given task.
        """
        if task_name not in self._students:
            raise KeyError(
                f"No student model for '{task_name}'. "
                f"Available: {list(self._students.keys())}"
            )
        return self._students[task_name].predict(features)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_soft_labels(self, path: str) -> None:
        """Save soft labels as parquet."""
        import pandas as pd

        if self._soft_labels is None:
            return

        data: Dict[str, np.ndarray] = {}
        for name, arr in self._soft_labels.items():
            if arr.ndim == 1:
                data[f"soft_{name}"] = arr
            else:
                # For multiclass: save each class probability as a separate column
                for cls_idx in range(arr.shape[1]):
                    data[f"soft_{name}_cls{cls_idx}"] = arr[:, cls_idx]
                # Also save argmax for convenience
                data[f"soft_{name}"] = arr.argmax(axis=1)

        df = pd.DataFrame(data)

        # Ensure parent directory exists
        parent = Path(path).parent
        if parent != Path(path) and str(parent) != ".":
            parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(path, index=False)
        logger.info("Soft labels saved to %s", path)


# ============================================================================
# DataLoader wrapper for TensorDataset compatibility
# ============================================================================


class _WrappedDataLoader:
    """Wraps a DataLoader so raw tensor tuples become PLEInput-compatible dicts.

    The SoftLabelGenerator._prepare_inputs expects PLEInput, dicts, or objects
    with a to_ple_input() method.  When the DataLoader yields TensorDataset
    tuples (e.g. ``(features_tensor,)``), this wrapper converts them to dicts.
    """

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def __iter__(self):
        for batch in self._loader:
            if isinstance(batch, (list, tuple)):
                # TensorDataset yields tuples of tensors
                yield {"features": batch[0]}
            elif isinstance(batch, dict) or isinstance(batch, PLEInput):
                yield batch
            else:
                # Assume it is a single tensor
                yield {"features": batch}

    def __len__(self):
        return len(self._loader)
