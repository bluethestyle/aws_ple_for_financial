"""
Knowledge distillation: PLE Teacher -> LGBM Student.

Implements temperature-based soft label generation and distillation loss
for transferring knowledge from a large PLE model to lightweight LGBM
student models.

Distillation formula::

    L_distill = alpha * L_hard + (1 - alpha) * T^2 * L_soft

    L_hard: Student's ground truth loss
    L_soft: KL divergence between teacher and student soft labels
    T:      Temperature (higher = softer labels)
    alpha:  Hard loss weight (0 = pure distillation, 1 = no distillation)

Usage::

    from core.training.distillation import (
        DistillationConfig,
        SoftLabelGenerator,
        DistillationLoss,
    )

    # Generate soft labels from teacher
    generator = SoftLabelGenerator(teacher_model, config)
    soft_labels = generator.generate(dataloader)

    # Train student with distillation loss
    loss_fn = DistillationLoss(temperature=5.0, alpha=0.3)
    loss = loss_fn(student_logits, teacher_logits, targets, task_type="binary")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.model.ple.model import PLEModel, PLEInput, PLEOutput

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class DistillationConfig:
    """Knowledge distillation configuration.

    Controls which tasks are distilled, the temperature and alpha balance,
    and per-task weighting for the distillation loss.

    All parameters are config-driven -- no hardcoded task names.
    """

    # Core distillation parameters
    temperature: float = 5.0
    alpha: float = 0.3  # hard loss weight (1-alpha = soft loss weight)

    # Per-task configuration
    # Each entry: {"task_name": {"weight": 1.0, "task_type": "binary"}}
    task_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Feature-level distillation
    use_feature_distillation: bool = False
    teacher_feature_dim: int = 64
    student_feature_dim: int = 32

    # Tasks to distill (if empty, distill all tasks from teacher)
    enabled_tasks: Optional[List[str]] = None

    def get_task_weight(self, task_name: str) -> float:
        """Return the distillation weight for a task."""
        return self.task_configs.get(task_name, {}).get("weight", 1.0)

    def get_task_type(self, task_name: str) -> str:
        """Return the task type for distillation loss selection."""
        return self.task_configs.get(task_name, {}).get("task_type", "binary")

    def should_distill(self, task_name: str) -> bool:
        """Whether to include this task in distillation."""
        if self.enabled_tasks is None:
            return True
        return task_name in self.enabled_tasks


# ============================================================================
# Distillation Loss
# ============================================================================


class DistillationLoss(nn.Module):
    """Knowledge distillation loss for a single task.

    Computes the weighted combination of hard loss (student vs ground truth)
    and soft loss (student vs temperature-softened teacher predictions).

    The soft loss uses task-type-appropriate formulations:

    * Binary: KL divergence on sigmoid-softened logits.
    * Multiclass: KL divergence on softmax-softened logits.
    * Regression: MSE between teacher and student outputs.

    Args:
        temperature: Soft label temperature (higher = softer).
        alpha: Weight of the hard loss (1-alpha goes to soft loss).
    """

    def __init__(self, temperature: float = 5.0, alpha: float = 0.3) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        task_type: str = "binary",
    ) -> Dict[str, torch.Tensor]:
        """Compute distillation loss.

        Args:
            student_logits: Student model raw outputs.
            teacher_logits: Teacher model raw outputs (will be detached).
            targets: Ground truth labels.
            task_type: ``"binary"``, ``"multiclass"``, or ``"regression"``.

        Returns:
            Dict with ``total``, ``hard``, and ``soft`` loss tensors.
        """
        hard_loss = self._hard_loss(student_logits, targets, task_type)
        soft_loss = self._soft_loss(student_logits, teacher_logits, task_type)

        total = (
            self.alpha * hard_loss
            + (1 - self.alpha) * (self.temperature ** 2) * soft_loss
        )

        return {"total": total, "hard": hard_loss, "soft": soft_loss}

    def _hard_loss(
        self,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        task_type: str,
    ) -> torch.Tensor:
        """Student vs ground truth loss."""
        if task_type == "binary":
            return F.binary_cross_entropy_with_logits(
                student_logits.squeeze(-1), targets.float(),
            )
        elif task_type == "multiclass":
            return F.cross_entropy(student_logits, targets.long())
        else:  # regression
            return F.mse_loss(student_logits.squeeze(-1), targets.float())

    def _soft_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        task_type: str,
    ) -> torch.Tensor:
        """Student vs teacher soft label KL divergence."""
        T = self.temperature
        teacher_logits = teacher_logits.detach()

        if task_type == "binary":
            student_soft = torch.sigmoid(student_logits / T).squeeze(-1)
            teacher_soft = torch.sigmoid(teacher_logits / T).squeeze(-1)

            eps = 1e-7
            teacher_soft = teacher_soft.clamp(eps, 1 - eps)
            student_soft = student_soft.clamp(eps, 1 - eps)

            # Binary KL divergence
            kl = -(
                teacher_soft * torch.log(student_soft / teacher_soft)
                + (1 - teacher_soft) * torch.log(
                    (1 - student_soft) / (1 - teacher_soft)
                )
            )
            return kl.mean()

        elif task_type == "multiclass":
            student_log_soft = F.log_softmax(student_logits / T, dim=-1)
            teacher_soft = F.softmax(teacher_logits / T, dim=-1)
            return F.kl_div(student_log_soft, teacher_soft, reduction="batchmean")

        else:  # regression
            return F.mse_loss(
                student_logits.squeeze(-1),
                teacher_logits.squeeze(-1),
            )


class MultiTaskDistillationLoss(nn.Module):
    """Multi-task distillation loss.

    Applies :class:`DistillationLoss` to each task with configurable
    per-task weights and task type selection.

    Args:
        config: Distillation configuration.
    """

    def __init__(self, config: Optional[DistillationConfig] = None) -> None:
        super().__init__()
        self.config = config or DistillationConfig()
        self.distill_loss = DistillationLoss(
            temperature=self.config.temperature,
            alpha=self.config.alpha,
        )

    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task distillation loss.

        Args:
            student_outputs: ``{task_name: logits}`` from student.
            teacher_outputs: ``{task_name: logits}`` from teacher.
            targets: ``{task_name: labels}`` ground truth.

        Returns:
            Dict with ``total`` loss and per-task losses.
        """
        device = next(iter(student_outputs.values())).device
        total_loss = torch.tensor(0.0, device=device)
        task_losses: Dict[str, torch.Tensor] = {}

        for task_name in student_outputs:
            if task_name not in teacher_outputs or task_name not in targets:
                continue
            if not self.config.should_distill(task_name):
                continue

            task_type = self.config.get_task_type(task_name)
            weight = self.config.get_task_weight(task_name)

            losses = self.distill_loss(
                student_logits=student_outputs[task_name],
                teacher_logits=teacher_outputs[task_name],
                targets=targets[task_name],
                task_type=task_type,
            )

            task_losses[task_name] = losses["total"]
            total_loss = total_loss + weight * losses["total"]

        task_losses["total"] = total_loss
        return task_losses


# ============================================================================
# Feature-level distillation
# ============================================================================


class FeatureDistillationLoss(nn.Module):
    """Intermediate representation distillation.

    Projects student features to match teacher feature dimensions, then
    minimises MSE between them.  Transfers richer information than
    logit-level distillation alone.

    Args:
        teacher_dim: Teacher feature dimensionality.
        student_dim: Student feature dimensionality.
        projection: Whether to project student features to teacher dim.
    """

    def __init__(
        self,
        teacher_dim: int = 64,
        student_dim: int = 32,
        projection: bool = True,
    ) -> None:
        super().__init__()
        if projection and teacher_dim != student_dim:
            self.projector: nn.Module = nn.Linear(student_dim, teacher_dim)
        else:
            self.projector = nn.Identity()

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute feature-level MSE loss.

        Args:
            student_features: ``(batch, student_dim)``
            teacher_features: ``(batch, teacher_dim)``

        Returns:
            Scalar MSE loss.
        """
        projected = self.projector(student_features)
        return F.mse_loss(projected, teacher_features.detach())


# ============================================================================
# Soft Label Generator
# ============================================================================


class SoftLabelGenerator:
    """Generate soft labels from a PLE teacher model.

    Runs inference on the teacher model and produces temperature-softened
    predictions for each task.  These soft labels are then used to train
    a student model (e.g., LGBM).

    Args:
        teacher: The PLE teacher model.
        config: Distillation configuration.
        device: Torch device for inference.
    """

    def __init__(
        self,
        teacher: PLEModel,
        config: Optional[DistillationConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.teacher = teacher
        self.config = config or DistillationConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.teacher.to(self.device)
        self.teacher.eval()

    @torch.no_grad()
    def generate(
        self,
        dataloader: DataLoader,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate soft labels for the entire dataset.

        Args:
            dataloader: Data loader yielding batches compatible with
                :class:`PLEInput`.
            return_features: Whether to also return intermediate features.

        Returns:
            Dict mapping task names to concatenated soft label tensors.
            If ``return_features`` is True, also includes a
            ``"__features__"`` key with intermediate representations.
        """
        all_soft_labels: Dict[str, List[torch.Tensor]] = {}
        T = self.config.temperature

        for batch in dataloader:
            inputs = self._prepare_inputs(batch)
            outputs: PLEOutput = self.teacher(inputs, compute_loss=False)

            for task_name, pred in outputs.predictions.items():
                if not self.config.should_distill(task_name):
                    continue

                task_type = self.config.get_task_type(task_name)

                # Apply temperature softening
                if task_type == "binary":
                    # Store raw logits (pre-sigmoid) for temperature scaling
                    # If model outputs probabilities, convert back to logits
                    if pred.min() >= 0 and pred.max() <= 1:
                        # Predictions are probabilities, convert to logits
                        eps = 1e-7
                        pred_clamped = pred.clamp(eps, 1 - eps)
                        logits = torch.log(pred_clamped / (1 - pred_clamped))
                    else:
                        logits = pred
                    soft = torch.sigmoid(logits / T)

                elif task_type == "multiclass":
                    if pred.dim() > 1 and pred.size(-1) > 1:
                        # Convert probabilities back to logits if needed
                        if (pred.sum(dim=-1) - 1.0).abs().max() < 0.01:
                            logits = torch.log(pred.clamp(1e-7))
                        else:
                            logits = pred
                        soft = F.softmax(logits / T, dim=-1)
                    else:
                        soft = pred

                else:  # regression
                    soft = pred

                all_soft_labels.setdefault(task_name, []).append(soft.cpu())

        # Concatenate
        result: Dict[str, torch.Tensor] = {}
        for task_name, tensors in all_soft_labels.items():
            result[task_name] = torch.cat(tensors, dim=0)

        logger.info(
            "Generated soft labels for %d tasks, %d samples each.",
            len(result),
            next(iter(result.values())).size(0) if result else 0,
        )
        return result

    def _prepare_inputs(self, batch: Any) -> PLEInput:
        """Convert batch to PLEInput and move to device."""
        if isinstance(batch, PLEInput):
            return batch.to(self.device)
        if hasattr(batch, "to_ple_input"):
            return batch.to_ple_input().to(self.device)
        if isinstance(batch, dict):
            return PLEInput(
                features=batch["features"],
                cluster_ids=batch.get("cluster_ids"),
                cluster_probs=batch.get("cluster_probs"),
                targets=batch.get("targets"),
            ).to(self.device)
        raise TypeError(f"Unsupported batch type: {type(batch)}")
