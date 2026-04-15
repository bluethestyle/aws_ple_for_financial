"""
Student Model Trainer -- distillation from PLE teacher to LGBM students.

SageMaker Step Functions flow:
  Step 1: PLE Training Job -> model.pt (S3)
  Step 2: Soft Label Generation -> soft_labels.parquet (S3)
  Step 3: LGBM Student Training x N tasks (parallel) -> student_models/ (S3)

This module handles Steps 2 and 3. Step 1 is handled by PLETrainer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.model.ple.model import PLEModel
from core.pipeline.config import TaskSpec
from core.training.config import StudentConfig
from core.training.distillation import DistillationConfig, SoftLabelGenerator
from core.training.distillation_numpy import (
    DistillationLossNumpy,
    make_multiclass_objective,
    train_with_custom_objective,
)
from core.training.teacher_loader import (
    _WrappedDataLoader,
    build_student_target,
    load_teacher_model,
    save_task_artifacts,
)

# Re-export StudentConfig so existing imports from this module keep working.
__all__ = ["StudentConfig", "StudentTrainer"]

logger = logging.getLogger(__name__)


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

    def load_teacher(
        self,
        checkpoint_path: Optional[str] = None,
        pipeline_config: Optional[Dict[str, Any]] = None,
    ) -> PLEModel:
        """Load a trained PLE model as teacher.

        Args:
            checkpoint_path: Override ``config.teacher_checkpoint``.
            pipeline_config: Raw YAML dict for expert basket reconstruction
                when the checkpoint lacks a ``config`` key.

        Returns:
            The teacher PLEModel in eval mode.
        """
        path = checkpoint_path or self.config.teacher_checkpoint
        if not path:
            raise ValueError(
                "No teacher checkpoint path provided. Set "
                "config.teacher_checkpoint or pass checkpoint_path."
            )

        teacher, _ = load_teacher_model(path, self.device, pipeline_config)
        self._teacher = teacher
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
            save_path: If provided, save soft labels as parquet to this path.

        Returns:
            ``{task_name: soft_label_array}`` dict.
        """
        if self._teacher is None:
            self.load_teacher()

        distill_config = DistillationConfig(
            temperature=self.config.temperature,
            enabled_tasks=self.config.enabled_tasks,
        )

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

        soft_labels_tensors = generator.generate(_WrappedDataLoader(data_loader))

        soft_labels: Dict[str, np.ndarray] = {
            task_name: tensor.numpy()
            for task_name, tensor in soft_labels_tensors.items()
        }
        self._soft_labels = soft_labels

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

        Handles multiclass tasks correctly: per-class columns
        ``soft_{name}_cls{i}`` are stacked into a 2-D array keyed by
        ``{name}``, and the redundant argmax column is skipped.

        Args:
            path: Override ``config.soft_label_path``.

        Returns:
            ``{task_name: soft_label_array}`` dict.
        """
        import re
        import duckdb

        path = path or self.config.soft_label_path
        if not path:
            raise ValueError("No soft label path provided.")

        df = duckdb.execute(f"SELECT * FROM read_parquet('{path}')").df()

        # --- Pass 1: detect multiclass per-class columns ---
        cls_pattern = re.compile(r"^soft_(.+)_cls(\d+)$")
        multiclass_tasks: Dict[str, Dict[int, str]] = {}  # name -> {idx: col}
        for col in df.columns:
            m = cls_pattern.match(col)
            if m:
                task_name, cls_idx = m.group(1), int(m.group(2))
                multiclass_tasks.setdefault(task_name, {})[cls_idx] = col

        # Columns to skip: per-class cols + argmax convenience cols
        skip_cols: set = set()
        for task_name, idx_map in multiclass_tasks.items():
            skip_cols.update(idx_map.values())
            skip_cols.add(f"soft_{task_name}")  # argmax column

        # --- Build result ---
        soft_labels: Dict[str, np.ndarray] = {}

        # Multiclass: stack per-class columns in order -> (n_samples, n_classes)
        for task_name, idx_map in multiclass_tasks.items():
            ordered = [idx_map[i] for i in sorted(idx_map.keys())]
            soft_labels[task_name] = np.stack(
                [df[col].values for col in ordered], axis=1
            )

        # Binary / regression: single column
        for col in df.columns:
            if col.startswith("soft_") and col not in skip_cols:
                soft_labels[col[5:]] = df[col].values

        self._soft_labels = soft_labels
        logger.info("Soft labels loaded from %s: %d tasks", path, len(soft_labels))
        return soft_labels

    def get_soft_labels(self) -> Dict[str, np.ndarray]:
        """Return the cached soft labels (teacher predictions).

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

        Args:
            features: Feature matrix ``(n_samples, n_features)``.
            hard_labels: Original ground truth labels per task.

        Returns:
            ``{task_name: trained_lgbm_model}`` dict.
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
            task_spec = self.task_specs.get(task_name)
            if task_spec is None:
                logger.warning("No task spec for '%s', skipping.", task_name)
                continue
            if task_name not in self._soft_labels:
                logger.warning("No soft labels for task '%s', skipping.", task_name)
                continue
            if task_name not in hard_labels:
                logger.warning("No hard labels for task '%s', skipping.", task_name)
                continue
            if task_spec.type == "contrastive":
                logger.info(
                    "Skipping contrastive task '%s' (not suitable for LGBM).",
                    task_name,
                )
                continue

            soft = np.asarray(self._soft_labels[task_name]).squeeze()
            hard = np.asarray(hard_labels[task_name]).squeeze()

            target, sample_weight, param_overrides = build_student_target(
                task_type=task_spec.type,
                hard=hard,
                soft=soft,
                alpha=alpha,
                num_classes=task_spec.num_classes,
            )

            params = dict(self.config.lgbm_params)
            params.update(self.config.task_lgbm_overrides.get(task_name, {}))
            params.update(param_overrides)

            n_estimators = params.pop("n_estimators", 500)
            use_cols = (
                self.feature_columns
                if len(self.feature_columns) == features.shape[1]
                else "auto"
            )

            # Determine custom objective (if enabled via config)
            fobj = None
            if self.config.use_custom_objective:
                loss_fn = DistillationLossNumpy(
                    alpha=self.config.alpha,
                    temperature=self.config.temperature,
                )
                if task_spec.type == "binary":
                    fobj = loss_fn.binary_loss
                elif task_spec.type == "multiclass":
                    fobj = make_multiclass_objective(
                        alpha=self.config.alpha,
                        temperature=self.config.temperature,
                        num_classes=task_spec.num_classes,
                    )
                elif task_spec.type == "regression":
                    fobj = loss_fn.regression_loss
                # else: fobj stays None -> fallback to standard objective

            # fobj handles blending internally -> hard labels + no weight.
            # Fallback (fobj=None) -> blended target + sample_weight from build_student_target.
            dataset_label = hard if fobj is not None else target
            dataset_weight = None if fobj is not None else sample_weight

            train_data = lgb.Dataset(
                features,
                label=dataset_label,
                weight=dataset_weight,
                feature_name=use_cols,
                free_raw_data=False,
            )
            if fobj is not None:
                train_data.soft_labels = soft
                model = train_with_custom_objective(
                    params, train_data, fobj, num_boost_round=n_estimators
                )
            else:
                model = lgb.train(params, train_data, num_boost_round=n_estimators)
            self._students[task_name] = model
            logger.info(
                "Student trained: %s (type=%s, trees=%d, features=%d)",
                task_name,
                task_spec.type,
                model.num_trees(),
                features.shape[1],
            )

            # Incremental checkpoint: save each model immediately after training
            # so partial results survive OOM or crash in later steps.
            _out_dir = self.config.student_output_dir
            if _out_dir:
                save_task_artifacts(
                    task_dir=Path(_out_dir) / task_name,
                    task_name=task_name,
                    model=model,
                    task_type=task_spec.type,
                    temperature=self.config.temperature,
                    alpha=self.config.alpha,
                    feature_columns=self.feature_columns,
                    feature_selection=None,
                    fidelity=None,
                )
                logger.info("  checkpoint saved: %s/model.lgbm", task_name)

        logger.info(
            "All students trained: %d/%d tasks", len(self._students), len(enabled)
        )

        if self._audit_store:
            for task_name, model in self._students.items():
                self._audit_store.log_event("distillation", {
                    "pk": task_name,
                    "task": task_name,
                    "teacher_checkpoint": self.config.teacher_checkpoint,
                    "temperature": self.config.temperature,
                    "alpha": self.config.alpha,
                    "num_trees": model.num_trees() if hasattr(model, "num_trees") else 0,
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

        Args:
            output_dir: Override ``config.student_output_dir``.
            feature_selections: ``{task_name: FeatureSelectionResult or dict}``.
            fidelity_results: List of ``FidelityResult`` or dicts from
                distillation validation.

        Returns:
            ``{task_name: saved_model_path}`` dict.
        """
        from dataclasses import asdict

        output_dir = output_dir or self.config.student_output_dir
        feature_selections = feature_selections or {}

        fidelity_by_task: Dict[str, Dict[str, Any]] = {}
        for fr in (fidelity_results or []):
            fr_dict = asdict(fr) if hasattr(fr, "__dataclass_fields__") else dict(fr)
            fidelity_by_task[fr_dict.get("task_name", "unknown")] = fr_dict

        saved_paths: Dict[str, str] = {}
        for task_name, model in self._students.items():
            task_type = (
                self.task_specs[task_name].type
                if task_name in self.task_specs
                else "unknown"
            )
            model_path = save_task_artifacts(
                task_dir=Path(output_dir) / task_name,
                task_name=task_name,
                model=model,
                task_type=task_type,
                temperature=self.config.temperature,
                alpha=self.config.alpha,
                feature_columns=self.feature_columns,
                feature_selection=feature_selections.get(task_name),
                fidelity=fidelity_by_task.get(task_name),
            )
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
                for cls_idx in range(arr.shape[1]):
                    data[f"soft_{name}_cls{cls_idx}"] = arr[:, cls_idx]
                data[f"soft_{name}"] = arr.argmax(axis=1)

        df = pd.DataFrame(data)
        parent = Path(path).parent
        if parent != Path(path) and str(parent) != ".":
            parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("Soft labels saved to %s", path)
