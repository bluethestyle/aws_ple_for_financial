"""
Task head registry and built-in implementations.

The :class:`TaskRegistry` maps :class:`~core.task.types.TaskType` values to
concrete :class:`~core.task.base.AbstractTask` subclasses.  Five task types
ship out of the box:

* **binary** -- Binary classification (BCE / Focal loss, sigmoid output)
* **multiclass** -- Multi-class classification (CE / Focal, softmax output)
* **regression** -- Scalar regression (Huber / MSE / MAE, linear output)
* **ranking** -- Listwise ranking (ListNet loss)
* **contrastive** -- Embedding retrieval (InfoNCE loss)

Custom tasks are added with the ``@TaskRegistry.register("my_type")``
decorator.

Usage::

    from core.task import TaskRegistry, TaskConfig, TaskType, LossType

    cfg = TaskConfig(
        name="click",
        task_type=TaskType.BINARY,
        loss_type=LossType.FOCAL,
        output_dim=1,
        tower_hidden_dims=[128, 64],
    )
    task = TaskRegistry.build(cfg, tower_input_dim=256)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AbstractTask, TaskConfig, TaskOutput
from .losses import InfoNCELoss
from .types import ActivationType, LossType, TaskType

logger = logging.getLogger(__name__)

__all__ = [
    "TaskRegistry",
    "BinaryTask",
    "MulticlassTask",
    "RegressionTask",
    "RankingTask",
    "ContrastiveTask",
]


# ════════════════════════════════════════════════════════════════
# Registry
# ════════════════════════════════════════════════════════════════


class TaskRegistry:
    """Plugin registry for task head implementations.

    Task classes are looked up by the string value of
    :class:`~core.task.types.TaskType`.  The five built-in types are
    registered at import time (bottom of this module).
    """

    _registry: Dict[str, Type[AbstractTask]] = {}

    # ── Registration ────────────────────────────────────────────

    @classmethod
    def register(cls, task_type: str):
        """Class decorator that registers a task head.

        Example::

            @TaskRegistry.register("my_custom_type")
            class MyTask(AbstractTask):
                ...
        """
        def decorator(task_cls: Type[AbstractTask]):
            cls._registry[task_type] = task_cls
            logger.debug("Registered task type '%s' -> %s", task_type, task_cls.__name__)
            return task_cls
        return decorator

    @classmethod
    def register_class(cls, task_type: str, task_cls: Type[AbstractTask]) -> None:
        """Imperatively register a task class (non-decorator form)."""
        cls._registry[task_type] = task_cls
        logger.debug("Registered task type '%s' -> %s", task_type, task_cls.__name__)

    # ── Lookup & construction ───────────────────────────────────

    @classmethod
    def build(cls, config: TaskConfig, tower_input_dim: int, **kwargs) -> AbstractTask:
        """Instantiate a task head from its config.

        Args:
            config: Fully resolved task configuration.
            tower_input_dim: Dimensionality of the expert output feeding
                this task.
            **kwargs: Extra keyword arguments forwarded to the task
                constructor.

        Returns:
            A ready-to-use task module.

        Raises:
            KeyError: If *config.task_type* has no registered implementation.
        """
        key = config.task_type.value if isinstance(config.task_type, TaskType) else config.task_type
        if key not in cls._registry:
            raise KeyError(
                f"Unknown task type '{key}'. "
                f"Registered types: {list(cls._registry.keys())}"
            )
        return cls._registry[key](config, tower_input_dim, **kwargs)

    @classmethod
    def get(cls, task_type: str) -> Type[AbstractTask]:
        """Return the class registered for *task_type*.

        Raises:
            KeyError: If *task_type* is not registered.
        """
        if task_type not in cls._registry:
            raise KeyError(
                f"Unknown task type '{task_type}'. "
                f"Registered types: {list(cls._registry.keys())}"
            )
        return cls._registry[task_type]

    @classmethod
    def list_registered(cls) -> List[str]:
        """Return all registered task type keys."""
        return list(cls._registry.keys())


# ════════════════════════════════════════════════════════════════
# Built-in: Binary Classification
# ════════════════════════════════════════════════════════════════


@TaskRegistry.register(TaskType.BINARY.value)
class BinaryTask(AbstractTask):
    """Binary classification head.

    * Default loss: BCE with logits (or Focal when configured).
    * Output activation: sigmoid.
    * Label smoothing is applied when ``config.label_smoothing > 0``.
    """

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        targets = labels.float().view_as(logits)

        # Optional label smoothing
        if self.config.label_smoothing > 0:
            targets = targets * (1.0 - self.config.label_smoothing) + 0.5 * self.config.label_smoothing

        loss = self.loss_fn(logits, targets)

        if sample_weights is not None:
            loss = loss * sample_weights.view_as(loss)

        loss = loss.mean()
        loss = self.uncertainty_weighted_loss(loss)
        return loss * self.config.loss_weight

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)


# ════════════════════════════════════════════════════════════════
# Built-in: Multi-class Classification
# ════════════════════════════════════════════════════════════════


@TaskRegistry.register(TaskType.MULTICLASS.value)
class MulticlassTask(AbstractTask):
    """Multi-class classification head.

    * Default loss: Cross-entropy (or Focal when configured).
    * Output activation: softmax.
    * Targets can be integer class indices or one-hot vectors.
    """

    def _build_tower(self, input_dim: int) -> nn.Module:
        """Override output_dim to num_classes when set."""
        out_dim = self.config.num_classes or self.config.output_dim
        # Build with corrected output dim
        original = self.config.output_dim
        self.config.output_dim = out_dim
        tower = super()._build_tower(input_dim)
        self.config.output_dim = original
        return tower

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Handle one-hot targets
        if labels.dim() > 1 and labels.size(-1) == logits.size(-1):
            labels = labels.argmax(dim=-1)
        targets = labels.long()

        loss = self.loss_fn(logits, targets)

        if sample_weights is not None:
            loss = loss * sample_weights.view_as(loss)

        loss = loss.mean()
        loss = self.uncertainty_weighted_loss(loss)
        return loss * self.config.loss_weight

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)


# ════════════════════════════════════════════════════════════════
# Built-in: Regression
# ════════════════════════════════════════════════════════════════


@TaskRegistry.register(TaskType.REGRESSION.value)
class RegressionTask(AbstractTask):
    """Scalar regression head.

    * Default loss: Huber.
    * Output activation: linear (identity).
    * Supports optional target normalisation (z-score) via config.

    When ``config.normalize_target`` is ``True``, call
    :meth:`set_target_stats` before training so that loss is computed on
    normalised targets and predictions are de-normalised at inference time.
    """

    def __init__(self, config: TaskConfig, tower_input_dim: int, **kwargs) -> None:
        super().__init__(config, tower_input_dim, **kwargs)
        self.target_mean: Optional[float] = None
        self.target_std: Optional[float] = None

    def set_target_stats(self, mean: float, std: float) -> None:
        """Set running statistics for target normalisation."""
        self.target_mean = mean
        self.target_std = std

    def _normalize(self, targets: torch.Tensor) -> torch.Tensor:
        if self.config.normalize_target and self.target_mean is not None:
            return (targets - self.target_mean) / (self.target_std + 1e-8)
        return targets

    def _denormalize(self, predictions: torch.Tensor) -> torch.Tensor:
        if self.config.normalize_target and self.target_mean is not None:
            return predictions * self.target_std + self.target_mean
        return predictions

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        targets = self._normalize(labels.float().view_as(logits))
        loss = self.loss_fn(logits, targets)

        if sample_weights is not None:
            loss = loss * sample_weights.view_as(loss)

        loss = loss.mean()
        loss = self.uncertainty_weighted_loss(loss)
        return loss * self.config.loss_weight

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return self._denormalize(logits)


# ════════════════════════════════════════════════════════════════
# Built-in: Ranking
# ════════════════════════════════════════════════════════════════


@TaskRegistry.register(TaskType.RANKING.value)
class RankingTask(AbstractTask):
    """Listwise ranking head.

    * Default loss: ListNet (softmax cross-entropy over lists).
    * Output activation: identity (scores are compared, not bounded).
    """

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # ListNet loss operates on full lists -- no per-element reduction
        loss = self.loss_fn(logits.squeeze(-1), labels.float())
        loss = self.uncertainty_weighted_loss(loss)
        return loss * self.config.loss_weight

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return logits


# ════════════════════════════════════════════════════════════════
# Built-in: Contrastive (Embedding Retrieval)
# ════════════════════════════════════════════════════════════════


@TaskRegistry.register(TaskType.CONTRASTIVE.value)
class ContrastiveTask(AbstractTask):
    """Embedding-based contrastive retrieval head.

    The tower projects gated expert output into a query embedding space.
    During training, an :class:`InfoNCELoss` is minimised against positive /
    negative key embeddings.  During inference the query embedding is
    returned directly for ANN lookup.

    * Default loss: InfoNCE.
    * Output: L2-normalised query embedding.
    * Requires external key embeddings (passed via ``kwargs``).

    Config extras (passed through ``config.extra``)::

        n_keys: int          -- number of items in the embedding table
        embedding_dim: int   -- dimensionality of key embeddings
        temperature: float   -- InfoNCE temperature (also on TaskConfig)
    """

    def __init__(self, config: TaskConfig, tower_input_dim: int, **kwargs) -> None:
        super().__init__(config, tower_input_dim, **kwargs)

        n_keys = config.extra.get("n_keys", 10000)
        embed_dim = config.extra.get("embedding_dim", config.output_dim)
        temperature = config.temperature

        # Key embedding table
        self.key_embedding = nn.Embedding(n_keys, embed_dim)
        nn.init.xavier_uniform_(self.key_embedding.weight)

        # Projection from tower output to embedding space
        tower_out = config.tower_hidden_dims[-1] if config.tower_hidden_dims else tower_input_dim
        self.query_projection = nn.Sequential(
            nn.Linear(tower_out, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.contrastive_loss = InfoNCELoss(temperature=temperature)

        logger.debug("ContrastiveTask '%s': n_keys=%d, embed_dim=%d, temp=%.3f",
                      config.name, n_keys, embed_dim, temperature)

    def _build_tower(self, input_dim: int) -> nn.Module:
        """Build tower *without* a final projection -- the projection to
        embedding space is handled by ``self.query_projection``.
        """
        hidden_dims = self.config.tower_hidden_dims
        if not hidden_dims:
            return nn.Identity()

        from .base import _get_tower_activation_cls
        act_cls = _get_tower_activation_cls(self.config.tower_activation)
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if self.config.tower_use_layer_norm:
                layers.append(nn.LayerNorm(h))
            elif self.config.tower_use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_cls())
            if self.config.tower_dropout > 0:
                layers.append(nn.Dropout(self.config.tower_dropout))
            prev = h
        return nn.Sequential(*layers)

    def forward(
        self,
        expert_output: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
        *,
        negative_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TaskOutput:
        """Contrastive forward pass.

        Args:
            expert_output: ``(B, D)`` gated expert features.
            labels: ``(B,)`` positive key IDs (integer indices).
            sample_weights: Per-sample weights (currently unused).
            negative_ids: ``(B, N)`` explicit hard-negative key IDs.
                When ``None``, in-batch negatives are used.

        Returns:
            :class:`TaskOutput` with ``logits`` set to the L2-normalised
            query embedding and ``loss`` computed via InfoNCE.
        """
        hidden = self.tower(expert_output)
        query = self.query_projection(hidden)  # (B, embed_dim)
        query_norm = F.normalize(query, p=2, dim=-1)

        loss = None
        if labels is not None:
            positive = self.key_embedding(labels.long())  # (B, embed_dim)

            negatives = None
            if negative_ids is not None:
                negatives = self.key_embedding(negative_ids.long())  # (B, N, embed_dim)

            loss = self.contrastive_loss(query, positive, negatives)
            loss = self.uncertainty_weighted_loss(loss)
            loss = loss * self.config.loss_weight

        return TaskOutput(
            logits=query_norm,
            predictions=query_norm,
            loss=loss,
            auxiliary_outputs={"query_embedding": query_norm},
        )

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute InfoNCE loss from query embeddings and positive key IDs.

        This method is provided for interface compliance but the main loss
        path runs through :meth:`forward`.
        """
        positive = self.key_embedding(labels.long())
        loss = self.contrastive_loss(logits, positive, negatives=None)
        loss = self.uncertainty_weighted_loss(loss)
        return loss * self.config.loss_weight

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised query embedding."""
        return F.normalize(logits, p=2, dim=-1)

    def predict_top_k(
        self,
        query_embedding: torch.Tensor,
        k: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve the top-*k* nearest keys for each query.

        Args:
            query_embedding: ``(B, D)`` L2-normalised query vectors.
            k: Number of results per query.

        Returns:
            ``(top_k_ids, top_k_scores)`` -- both of shape ``(B, k)``.
        """
        all_keys = F.normalize(self.key_embedding.weight, p=2, dim=-1)
        scores = torch.mm(F.normalize(query_embedding, p=2, dim=-1), all_keys.t())
        top_scores, top_ids = scores.topk(k, dim=-1)
        return top_ids, top_scores
