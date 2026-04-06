"""
Training configuration for PLE multi-task models.

All training hyperparameters are defined here as a single dataclass.
YAML config maps directly to these fields via ``TrainingConfig(**yaml_block)``.

Design principles:
  - No hardcoded task names, dimensions, or domain references.
  - 2-Phase training (shared experts -> task heads) is first-class.
  - Optimizer, scheduler, AMP, gradient accumulation are all configurable.
  - Works identically on local machines and SageMaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    """Optimizer configuration.

    Supports AdamW with per-expert learning rate overrides.
    """

    name: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    # Per-expert LR overrides: {"expert_name": {"lr": 0.0005, "weight_decay": 0.001}}
    expert_lr_overrides: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration.

    Supports CosineAnnealingWarmRestarts with linear warmup prefix.
    """

    name: str = "cosine"  # "cosine" | "linear" | "none"

    # Warmup
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1

    # CosineAnnealingWarmRestarts
    cosine_t0: int = 10
    cosine_t_mult: int = 2
    cosine_eta_min: float = 0.0

    # Phase 2 overrides (shorter warmup, different period)
    phase2_warmup_epochs: int = 2
    phase2_cosine_t0: int = 6


@dataclass
class AMPConfig:
    """Automatic Mixed Precision configuration."""

    enabled: bool = True
    dtype: str = "float16"  # "float16" | "bfloat16"

    # GradScaler parameters (read from config to avoid train.py hardcoding)
    grad_scaler_init_scale: float = 1024.0
    grad_scaler_growth_factor: float = 2.0
    grad_scaler_backoff_factor: float = 0.5
    grad_scaler_growth_interval: int = 2000
    grad_scaler_max_scale: float = 4096.0


@dataclass
class GradientConfig:
    """Gradient handling configuration."""

    clip_norm: float = 5.0
    accumulation_steps: int = 1


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration.

    Primary criterion: validation loss plateau.
    Secondary criterion: average AUC decline over consecutive epochs.
    """

    enabled: bool = True
    patience: int = 5
    min_delta: float = 0.0

    # AUC-based secondary stopping
    auc_decline_patience: int = 3


@dataclass
class CheckpointConfig:
    """Checkpoint save/load configuration.

    Supports local directory and S3 prefix.  S3 upload is handled
    externally (e.g. by SageMaker or a callback).
    """

    dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    max_to_keep: int = 5
    save_optimizer: bool = True
    save_scheduler: bool = True

    # S3 integration (optional, handled by ExperimentTracker)
    s3_prefix: Optional[str] = None


@dataclass
class Phase1Config:
    """Phase 1: Shared expert training."""

    epochs: int = 30
    freeze_task_experts: bool = False  # optionally freeze task-specific experts


@dataclass
class Phase2Config:
    """Phase 2: Task head fine-tuning."""

    epochs: int = 20
    freeze_shared_experts: bool = True
    freeze_cgc: bool = True  # freeze CGC gating when shared experts are frozen
    disable_adatt: bool = True  # adaTT is meaningless when shared is frozen


@dataclass
class LoggingConfig:
    """Metric logging configuration."""

    log_every_n_steps: int = 100
    log_lr: bool = True
    log_grad_norm: bool = False
    log_loss_weights: bool = True


@dataclass
class DistillationPhaseConfig:
    """Distillation phase configuration (runs after main PLE training).

    When ``enabled`` is True, the training pipeline will generate soft
    labels from the trained PLE teacher and distill into LGBM student
    models.

    Args:
        enabled: Whether to run distillation after PLE training.
        temperature: Soft label temperature (higher = softer distributions).
        alpha: Hard label weight in blended target.
        soft_label_path: S3 path for caching soft labels.
        student_output_dir: Directory to save student models.
        lgbm_n_estimators: Number of boosting rounds for LGBM students.
        lgbm_num_leaves: Max leaves per tree for LGBM students.
        lgbm_learning_rate: LGBM learning rate.
        enabled_tasks: Restrict to these tasks (None = all non-contrastive).
    """

    enabled: bool = False
    temperature: float = 5.0
    alpha: float = 0.3
    soft_label_path: str = ""
    student_output_dir: str = "student_models"
    lgbm_n_estimators: int = 500
    lgbm_num_leaves: int = 63
    lgbm_learning_rate: float = 0.05
    enabled_tasks: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Complete training configuration for PLE models.

    Covers 2-phase training, optimization, scheduling, AMP, gradient
    handling, early stopping, checkpointing, and experiment tracking.

    Usage::

        import yaml
        with open("training_config.yaml") as f:
            raw = yaml.safe_load(f)
        config = TrainingConfig.from_dict(raw["training"])
    """

    # -- Batch & data --------------------------------------------------------
    batch_size: int = 4096
    num_workers: int = 4

    # -- Optimizer -----------------------------------------------------------
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # -- Scheduler -----------------------------------------------------------
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # -- AMP -----------------------------------------------------------------
    amp: AMPConfig = field(default_factory=AMPConfig)

    # -- Gradients -----------------------------------------------------------
    gradient: GradientConfig = field(default_factory=GradientConfig)

    # -- Early stopping ------------------------------------------------------
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    # -- Checkpointing -------------------------------------------------------
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # -- 2-Phase training ----------------------------------------------------
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)

    # -- Logging -------------------------------------------------------------
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # -- Distillation --------------------------------------------------------
    distillation: DistillationPhaseConfig = field(
        default_factory=DistillationPhaseConfig
    )

    # -- GPU optimizations ---------------------------------------------------
    enable_tf32: bool = True
    cudnn_benchmark: bool = True

    # -- Experiment tracking -------------------------------------------------
    experiment_name: str = "ple_training"
    run_name: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        """Build TrainingConfig from a flat or nested dict (e.g. YAML output).

        Supports both flat keys (``learning_rate``) and nested sub-dicts
        (``optimizer: {learning_rate: ...}``).
        """
        # Nested sub-configs
        optimizer_d = d.get("optimizer", {})
        scheduler_d = d.get("scheduler", {})
        amp_d = d.get("amp", {})
        gradient_d = d.get("gradient", {})
        early_stopping_d = d.get("early_stopping", {})
        checkpoint_d = d.get("checkpoint", {})
        phase1_d = d.get("phase1", {})
        phase2_d = d.get("phase2", {})
        logging_d = d.get("logging", {})
        distillation_d = d.get("distillation", {})

        # Support flat keys as fallback
        if not optimizer_d and "learning_rate" in d:
            optimizer_d = {
                "learning_rate": d.get("learning_rate", 1e-3),
                "weight_decay": d.get("weight_decay", 0.01),
                "name": d.get("optimizer_name", "adamw"),
                "expert_lr_overrides": d.get("expert_learning_rates"),
            }

        return cls(
            batch_size=d.get("batch_size", 4096),
            num_workers=d.get("num_workers", 4),
            optimizer=OptimizerConfig(**{
                k: v for k, v in optimizer_d.items()
                if k in OptimizerConfig.__dataclass_fields__
            }) if optimizer_d else OptimizerConfig(),
            scheduler=SchedulerConfig(**{
                k: v for k, v in scheduler_d.items()
                if k in SchedulerConfig.__dataclass_fields__
            }) if scheduler_d else SchedulerConfig(),
            amp=AMPConfig(**{
                k: v for k, v in amp_d.items()
                if k in AMPConfig.__dataclass_fields__
            }) if amp_d else AMPConfig(),
            gradient=GradientConfig(**{
                k: v for k, v in gradient_d.items()
                if k in GradientConfig.__dataclass_fields__
            }) if gradient_d else GradientConfig(),
            early_stopping=EarlyStoppingConfig(**{
                k: v for k, v in early_stopping_d.items()
                if k in EarlyStoppingConfig.__dataclass_fields__
            }) if early_stopping_d else EarlyStoppingConfig(),
            checkpoint=CheckpointConfig(**{
                k: v for k, v in checkpoint_d.items()
                if k in CheckpointConfig.__dataclass_fields__
            }) if checkpoint_d else CheckpointConfig(),
            phase1=Phase1Config(**{
                k: v for k, v in phase1_d.items()
                if k in Phase1Config.__dataclass_fields__
            }) if phase1_d else Phase1Config(),
            phase2=Phase2Config(**{
                k: v for k, v in phase2_d.items()
                if k in Phase2Config.__dataclass_fields__
            }) if phase2_d else Phase2Config(),
            logging=LoggingConfig(**{
                k: v for k, v in logging_d.items()
                if k in LoggingConfig.__dataclass_fields__
            }) if logging_d else LoggingConfig(),
            distillation=DistillationPhaseConfig(**{
                k: v for k, v in distillation_d.items()
                if k in DistillationPhaseConfig.__dataclass_fields__
            }) if distillation_d else DistillationPhaseConfig(),
            enable_tf32=d.get("enable_tf32", True),
            cudnn_benchmark=d.get("cudnn_benchmark", True),
            experiment_name=d.get("experiment_name", "ple_training"),
            run_name=d.get("run_name"),
            tags=d.get("tags", {}),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainingConfig":
        """Load from a YAML file.

        Expects the training config under a ``training:`` top-level key,
        or at the root level.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        training_block = raw.get("training", raw)
        return cls.from_dict(training_block)

    @property
    def total_epochs(self) -> int:
        """Total epochs across both phases."""
        return self.phase1.epochs + self.phase2.epochs
