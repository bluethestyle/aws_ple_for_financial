"""
PLE (Progressive Layered Extraction) model package.

Core components:
  - ``PLEModel``: Full multi-task model with CGC layers, adaTT, and loss weighting.
  - ``PLEConfig``: Dataclass holding all hyperparameters.
  - ``PLEInput`` / ``PLEOutput``: Typed input/output containers.

Expert components:
  - ``BaseExpert``, ``MLPExpert``: Expert network implementations.
  - ``ExpertRegistry``: Plugin registry for custom expert types.
  - ``CGCLayer``: Customized Gate Control layer (PLE building block).
  - ``CGCAttention``: Per-task attention over shared expert outputs.

Gating:
  - ``SoftmaxGate``, ``AttentionGate``, ``MLPGate``: Gating network variants.

Adaptive Task Transfer:
  - ``AdaptiveTaskTransfer``: Gradient-based inter-task knowledge transfer.
  - ``TaskAffinityComputer``: Pairwise gradient cosine similarity tracker.

Loss weighting:
  - ``GradNormWeighting``, ``DWAWeighting``, ``UncertaintyWeighting``.
  - ``create_loss_weighting``: Factory function.
"""

from .config import (
    PLEConfig,
    ExpertConfig,
    CGCConfig,
    AdaTTConfig,
    TaskGroupDef,
    LossWeightingConfig,
    TaskTowerConfig,
    ClusterConfig,
    LogitTransferDef,
)
from .model import PLEModel, PLEInput, PLEOutput, TaskTower
from .experts import (
    BaseExpert,
    MLPExpert,
    ExpertRegistry,
    CGCLayer,
    CGCAttention,
)
from .gating import SoftmaxGate, AttentionGate, MLPGate, build_gate
from .adatt import AdaptiveTaskTransfer, TaskAffinityComputer
from .loss_weighting import (
    BaseLossWeighting,
    GradNormWeighting,
    DWAWeighting,
    UncertaintyWeighting,
    create_loss_weighting,
)

__all__ = [
    # Config
    "PLEConfig",
    "ExpertConfig",
    "CGCConfig",
    "AdaTTConfig",
    "TaskGroupDef",
    "LossWeightingConfig",
    "TaskTowerConfig",
    "ClusterConfig",
    "LogitTransferDef",
    # Model
    "PLEModel",
    "PLEInput",
    "PLEOutput",
    "TaskTower",
    # Experts
    "BaseExpert",
    "MLPExpert",
    "ExpertRegistry",
    "CGCLayer",
    "CGCAttention",
    # Gating
    "SoftmaxGate",
    "AttentionGate",
    "MLPGate",
    "build_gate",
    # adaTT
    "AdaptiveTaskTransfer",
    "TaskAffinityComputer",
    # Loss weighting
    "BaseLossWeighting",
    "GradNormWeighting",
    "DWAWeighting",
    "UncertaintyWeighting",
    "create_loss_weighting",
]
