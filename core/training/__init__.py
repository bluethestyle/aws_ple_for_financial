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

Distillation Validation:
  - ``DistillationValidator``: Teacher-student fidelity measurement (8 metrics).
  - ``ValidationCriteria``: Configurable per-metric thresholds.
  - ``FidelityResult``: Per-task validation result.

Feature Selection:
  - ``FeatureSelector``: Adaptive per-task feature selection (IG + LGBM pruning).
  - ``FeatureSelectionConfig``: Cumulative threshold and selection parameters.
  - ``FeatureSelectionResult``: Per-task selection result.

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

# Distillation validation
from .distillation_validator import (
    DistillationValidator,
    ValidationCriteria,
    FidelityResult,
)

# Feature selection
from .feature_selector import (
    FeatureSelector,
    FeatureSelectionConfig,
    FeatureSelectionResult,
)

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
    # Distillation validation
    "DistillationValidator",
    "ValidationCriteria",
    "FidelityResult",
    # Feature selection
    "FeatureSelector",
    "FeatureSelectionConfig",
    "FeatureSelectionResult",
    # Pre-existing
    "CheckpointManager",
    "ModelEvaluator",
]
