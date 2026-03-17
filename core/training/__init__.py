"""
Training pipeline for PLE multi-task models.

Core components:
  - ``TrainingConfig``: Complete training configuration (YAML-mappable).
  - ``PLETrainer``: Two-phase trainer with AMP, gradient accumulation, callbacks.
  - ``ExperimentTracker``: Abstract experiment tracking interface.
  - ``auto_tracker``: Auto-detect local vs SageMaker tracker.

Callbacks:
  - ``EarlyStoppingCallback``: Stop on val_loss plateau or AUC decline.
  - ``CheckpointCallback``: Periodic and best-model checkpointing.
  - ``MetricLoggerCallback``: Log metrics to ExperimentTracker.
  - ``LRSchedulerCallback``: Step the LR scheduler per epoch.

Distillation:
  - ``DistillationConfig``: Distillation hyperparameters.
  - ``DistillationLoss``: Single-task distillation loss.
  - ``MultiTaskDistillationLoss``: Multi-task weighted distillation.
  - ``SoftLabelGenerator``: Generate teacher soft labels for LGBM student.

Checkpoint & Evaluation (pre-existing):
  - ``CheckpointManager``: S3-integrated checkpoint management.
  - ``ModelEvaluator``: Per-task metrics and champion-vs-challenger comparison.

Usage::

    from core.training import PLETrainer, TrainingConfig

    config = TrainingConfig.from_yaml("training.yaml")
    trainer = PLETrainer(model, config)
    results = trainer.train(train_loader, val_loader)
"""

# Config
from .config import (
    TrainingConfig,
    OptimizerConfig,
    SchedulerConfig,
    AMPConfig,
    GradientConfig,
    EarlyStoppingConfig,
    CheckpointConfig,
    Phase1Config,
    Phase2Config,
    LoggingConfig,
    DistillationPhaseConfig,
)

# Trainer
from .trainer import PLETrainer

# Experiment tracking
from .experiment import (
    ExperimentTracker,
    LocalTracker,
    SageMakerTracker,
    auto_tracker,
)

# Callbacks
from .callbacks import (
    TrainingCallback,
    CallbackList,
    EarlyStoppingCallback,
    CheckpointCallback,
    MetricLoggerCallback,
    LRSchedulerCallback,
)

# Distillation
from .distillation import (
    DistillationConfig,
    DistillationLoss,
    MultiTaskDistillationLoss,
    FeatureDistillationLoss,
    SoftLabelGenerator,
)

# Student training (distillation orchestration)
from .student_trainer import StudentTrainer, StudentConfig

# Pre-existing modules
from .checkpoint import CheckpointManager
from .evaluator import ModelEvaluator

__all__ = [
    # Config
    "TrainingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "AMPConfig",
    "GradientConfig",
    "EarlyStoppingConfig",
    "CheckpointConfig",
    "Phase1Config",
    "Phase2Config",
    "LoggingConfig",
    "DistillationPhaseConfig",
    # Trainer
    "PLETrainer",
    # Experiment tracking
    "ExperimentTracker",
    "LocalTracker",
    "SageMakerTracker",
    "auto_tracker",
    # Callbacks
    "TrainingCallback",
    "CallbackList",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "MetricLoggerCallback",
    "LRSchedulerCallback",
    # Distillation
    "DistillationConfig",
    "DistillationLoss",
    "MultiTaskDistillationLoss",
    "FeatureDistillationLoss",
    "SoftLabelGenerator",
    # Student training
    "StudentTrainer",
    "StudentConfig",
    # Pre-existing
    "CheckpointManager",
    "ModelEvaluator",
]
