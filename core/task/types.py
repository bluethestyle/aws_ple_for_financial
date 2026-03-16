"""
Task and loss type enumerations for the PLE multi-task learning platform.

These enums are the single source of truth for task categories and loss
function selection.  Tasks are defined by *config* (not by class name), so
adding a new task never requires editing this file -- only adding a new
``TaskType`` entry when a genuinely new learning paradigm is needed.
"""

from enum import Enum


class TaskType(str, Enum):
    """Supported task paradigms.

    Each value maps to a concrete ``AbstractTask`` subclass registered in
    :class:`~core.task.registry.TaskRegistry`.
    """

    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"
    RANKING = "ranking"
    CONTRASTIVE = "contrastive"


class LossType(str, Enum):
    """Loss function selector.

    The concrete loss modules live in :mod:`core.task.losses`.  A task's
    ``TaskConfig.loss_type`` picks which one to instantiate.
    """

    # ── Classification ──────────────────────────────────────────
    BCE = "bce"
    FOCAL = "focal"
    CROSS_ENTROPY = "ce"

    # ── Regression ──────────────────────────────────────────────
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    QUANTILE = "quantile"

    # ── Ranking / Retrieval ─────────────────────────────────────
    LISTNET = "listnet"
    INFONCE = "infonce"

    # ── Convenience alias for default per-task-type selection ───
    AUTO = "auto"


class ActivationType(str, Enum):
    """Output activation applied after the tower head."""

    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    SOFTPLUS = "softplus"
    LINEAR = "linear"
    RELU = "relu"
    TANH = "tanh"
