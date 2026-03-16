"""
Abstract base class for all PLE task heads.

A *task head* receives the gated expert output from the PLE backbone and
produces:

1. **logits** -- raw, un-activated output of the tower MLP,
2. **predictions** -- human-interpretable values (probabilities, scores, ...),
3. **loss** -- scalar training objective (only when labels are supplied).

Subclasses must implement :meth:`compute_loss` and :meth:`predict`.  The
tower MLP is built automatically from :class:`TaskConfig`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import build_loss
from .types import ActivationType, LossType, TaskType

logger = logging.getLogger(__name__)

__all__ = [
    "TaskConfig",
    "TaskOutput",
    "AbstractTask",
]


# ════════════════════════════════════════════════════════════════
# Data classes
# ════════════════════════════════════════════════════════════════


@dataclass
class TaskConfig:
    """Declarative specification of a single task head.

    Every task in the system is fully described by a ``TaskConfig`` -- the
    class of the task head is selected by :attr:`task_type`, not by the
    Python class name.  This makes it easy to add new tasks purely via
    configuration (YAML / dict).

    Attributes:
        name: Unique task identifier (e.g. ``"click_rate"``).
        task_type: Learning paradigm -- selects the registered task class.
        loss_type: Which loss function to use.  ``LossType.AUTO`` picks the
            canonical default for *task_type*.
        output_dim: Dimensionality of the final linear projection.  For
            binary tasks this is typically ``1``; for multiclass it equals
            the number of classes.
        loss_weight: Static multiplier applied to this task's loss when
            summing the multi-task objective.
        label_smoothing: Smoothing epsilon for CE-family losses.
        pos_weight: Positive-class weight for BCE loss.
        focal_alpha: Alpha parameter for focal loss.
        focal_gamma: Gamma (focusing) parameter for focal loss.
        huber_delta: Delta for Huber loss.
        tower_hidden_dims: Sequence of hidden-layer widths for the
            task-specific tower MLP.
        tower_dropout: Dropout probability inside the tower.
        tower_use_layer_norm: Whether to apply LayerNorm after each hidden
            layer (recommended for multi-task setups).
        tower_use_batch_norm: Whether to apply BatchNorm instead of
            LayerNorm (mutually exclusive with *tower_use_layer_norm*).
        tower_activation: Non-linearity used inside the tower.
        output_activation: Activation applied *after* the output projection
            to produce :attr:`TaskOutput.predictions`.
        primary_metric: Name of the metric used for model selection.
        secondary_metrics: Additional metrics to log during evaluation.
        num_classes: Number of target classes (multiclass / ranking).
        label_col: Column name of the label in the dataset.
        normalize_target: Whether to z-score targets during training
            (regression tasks).
        clip_gradient: Optional per-task gradient clipping norm.
        temperature: Temperature for contrastive / InfoNCE losses.
        description: Human-readable description of the task.
        extra: Catch-all dict for task-specific knobs that do not warrant a
            first-class field (e.g. ``{"uplift_method": "t_learner"}``).
    """

    name: str
    task_type: TaskType
    loss_type: LossType = LossType.AUTO
    output_dim: int = 1
    loss_weight: float = 1.0

    # Loss hyper-parameters
    label_smoothing: float = 0.0
    pos_weight: Optional[float] = None
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    huber_delta: float = 1.0
    quantiles: Optional[List[float]] = None

    # Tower architecture
    tower_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    tower_dropout: float = 0.2
    tower_use_layer_norm: bool = True
    tower_use_batch_norm: bool = False
    tower_activation: str = "silu"
    output_activation: ActivationType = ActivationType.LINEAR

    # Metrics
    primary_metric: str = "loss"
    secondary_metrics: List[str] = field(default_factory=list)

    # Classification specifics
    num_classes: Optional[int] = None
    class_names: Optional[Dict[int, str]] = None

    # Dataset
    label_col: str = "label"

    # Regression specifics
    normalize_target: bool = False
    clip_gradient: Optional[float] = None

    # Contrastive specifics
    temperature: float = 0.07

    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    # ── derived ─────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if self.loss_type == LossType.AUTO:
            self.loss_type = _default_loss_for(self.task_type)
        if self.output_activation == ActivationType.LINEAR:
            self.output_activation = _default_activation_for(self.task_type)
        if self.num_classes is None and self.task_type == TaskType.MULTICLASS:
            self.num_classes = self.output_dim


# ════════════════════════════════════════════════════════════════
# Task output
# ════════════════════════════════════════════════════════════════


@dataclass
class TaskOutput:
    """Container returned by every task head's ``forward()``.

    Attributes:
        logits: Raw tower output before activation.
        predictions: Human-readable output (probabilities, scores, ...).
        loss: Scalar training loss (``None`` at inference time).
        probabilities: Class probabilities for classification tasks.
        metrics: Optional dict of evaluation metrics.
        auxiliary_outputs: Free-form dict for task-specific extras (e.g.
            query embeddings, treatment/control splits, ...).
    """

    logits: torch.Tensor
    predictions: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    probabilities: Optional[torch.Tensor] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    auxiliary_outputs: Optional[Dict[str, Any]] = None


# ════════════════════════════════════════════════════════════════
# Abstract task head
# ════════════════════════════════════════════════════════════════


class AbstractTask(ABC, nn.Module):
    """Base class for all task heads.

    Subclasses **must** implement:

    * :meth:`compute_loss` -- given raw logits and ground-truth labels,
      return a scalar loss tensor.
    * :meth:`predict` -- given raw logits, return predictions suitable for
      evaluation (probabilities, class ids, scores, ...).

    The tower MLP and loss module are built automatically from
    :class:`TaskConfig` during ``__init__``.  Override
    :meth:`_build_tower` or :meth:`_build_loss` for custom behaviour.

    Args:
        config: Fully resolved task configuration.
        tower_input_dim: Dimensionality of the gated expert output that
            feeds into this task's tower.
    """

    def __init__(self, config: TaskConfig, tower_input_dim: int) -> None:
        super().__init__()
        self.config = config
        self._tower_input_dim = tower_input_dim

        # Build components
        self.tower: nn.Module = self._build_tower(tower_input_dim)
        self.loss_fn: nn.Module = self._build_loss()
        self.output_activation_fn: Callable = _get_activation(config.output_activation)

        # Learnable uncertainty weight (Kendall et al., 2018)
        self.log_var = nn.Parameter(torch.zeros(1))

        logger.debug("Task '%s' initialised (type=%s, tower_in=%d, out=%d)",
                      config.name, config.task_type.value, tower_input_dim, config.output_dim)

    # ── Tower construction ──────────────────────────────────────

    def _build_tower(self, input_dim: int) -> nn.Module:
        """Construct the task-specific MLP tower.

        The tower is a stack of ``[Linear -> Norm -> Activation -> Dropout]``
        blocks followed by a final ``Linear`` projection to
        ``config.output_dim``.

        Override this method to supply a completely custom architecture.
        """
        hidden_dims = self.config.tower_hidden_dims
        if not hidden_dims:
            return nn.Linear(input_dim, self.config.output_dim)

        activation_cls = _get_tower_activation_cls(self.config.tower_activation)
        layers: list[nn.Module] = []
        prev = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if self.config.tower_use_layer_norm:
                layers.append(nn.LayerNorm(h))
            elif self.config.tower_use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation_cls())
            if self.config.tower_dropout > 0:
                layers.append(nn.Dropout(self.config.tower_dropout))
            prev = h

        layers.append(nn.Linear(prev, self.config.output_dim))
        return nn.Sequential(*layers)

    # ── Loss construction ───────────────────────────────────────

    def _build_loss(self) -> nn.Module:
        """Instantiate the loss module from config."""
        return build_loss(
            self.config.loss_type,
            focal_alpha=self.config.focal_alpha,
            focal_gamma=self.config.focal_gamma,
            huber_delta=self.config.huber_delta,
            quantiles=self.config.quantiles,
            temperature=self.config.temperature,
            pos_weight=self.config.pos_weight,
            label_smoothing=self.config.label_smoothing,
        )

    # ── Forward ─────────────────────────────────────────────────

    def forward(
        self,
        expert_output: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TaskOutput:
        """Run the tower, compute predictions, and optionally compute loss.

        Args:
            expert_output: Gated expert output ``(B, D)``.
            labels: Ground-truth targets.  When ``None`` the loss is skipped
                (inference mode).
            sample_weights: Per-sample importance weights ``(B,)``.
            **kwargs: Forwarded to :meth:`compute_loss` (e.g.
                ``treatment_flags`` for uplift tasks).

        Returns:
            A :class:`TaskOutput` instance.
        """
        logits = self.tower(expert_output)
        predictions = self.predict(logits)

        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels, sample_weights=sample_weights, **kwargs)

        probabilities = None
        if self.config.task_type in (TaskType.BINARY, TaskType.MULTICLASS):
            probabilities = predictions

        return TaskOutput(
            logits=logits,
            predictions=predictions,
            loss=loss,
            probabilities=probabilities,
        )

    # ── Abstract interface ──────────────────────────────────────

    @abstractmethod
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the scalar training loss.

        Implementations should:

        1. Call ``self.loss_fn(logits, labels)`` to get unreduced per-sample
           losses.
        2. Apply *sample_weights* if provided.
        3. Reduce to a scalar.
        4. Optionally call :meth:`uncertainty_weighted_loss` to apply
           learned task weighting.
        5. Multiply by ``self.config.loss_weight``.
        """

    @abstractmethod
    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Transform raw logits into human-readable predictions.

        For binary classification this is typically ``sigmoid(logits)``; for
        multiclass ``softmax(logits)``; for regression just ``logits``.
        """

    # ── Utilities ───────────────────────────────────────────────

    def uncertainty_weighted_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply learned uncertainty weighting (Kendall et al., 2018).

        Formula::

            L_weighted = 0.5 * exp(-log_var) * loss + 0.5 * log_var

        This allows the model to down-weight noisy tasks automatically.
        """
        precision = torch.exp(-self.log_var)
        return 0.5 * precision * loss + 0.5 * self.log_var

    @property
    def name(self) -> str:
        return self.config.name


# ════════════════════════════════════════════════════════════════
# Private helpers
# ════════════════════════════════════════════════════════════════


def _default_loss_for(task_type: TaskType) -> LossType:
    """Select the canonical loss for a given task type."""
    return {
        TaskType.BINARY: LossType.BCE,
        TaskType.MULTICLASS: LossType.CROSS_ENTROPY,
        TaskType.REGRESSION: LossType.HUBER,
        TaskType.RANKING: LossType.LISTNET,
        TaskType.CONTRASTIVE: LossType.INFONCE,
    }[task_type]


def _default_activation_for(task_type: TaskType) -> ActivationType:
    """Select the canonical output activation for a given task type."""
    return {
        TaskType.BINARY: ActivationType.SIGMOID,
        TaskType.MULTICLASS: ActivationType.SOFTMAX,
        TaskType.REGRESSION: ActivationType.LINEAR,
        TaskType.RANKING: ActivationType.LINEAR,
        TaskType.CONTRASTIVE: ActivationType.LINEAR,
    }[task_type]


def _get_activation(act: ActivationType) -> Callable:
    """Return a callable that applies the given activation."""
    return {
        ActivationType.SIGMOID: torch.sigmoid,
        ActivationType.SOFTMAX: lambda x: F.softmax(x, dim=-1),
        ActivationType.SOFTPLUS: F.softplus,
        ActivationType.LINEAR: lambda x: x,
        ActivationType.RELU: F.relu,
        ActivationType.TANH: torch.tanh,
    }[act]


def _get_tower_activation_cls(name: str) -> type:
    """Map an activation name to an ``nn.Module`` class for tower layers."""
    mapping = {
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }
    cls = mapping.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown tower activation '{name}'. Choose from {list(mapping)}")
    return cls
