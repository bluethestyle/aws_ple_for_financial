"""
Adaptive Task Transfer (adaTT).

Implements gradient-based inter-task knowledge transfer for multi-task
learning.  Tasks that have similar gradient directions (positive affinity)
share knowledge, while tasks with opposing gradients (negative transfer)
are isolated.

Core equations:
  1. Task affinity:  A(i,j) = cosine(grad_theta L_i, grad_theta L_j)
  2. Transfer weight: w(i->j) = softmax(A(i,j) / tau)
  3. Enhanced loss:   L_i^adaTT = L_i + lambda * sum_{j!=i} w(i->j) * L_j

Three training phases:
  Phase 1 (epoch < warmup):   Measure affinity only, no transfer.
  Phase 2 (warmup..freeze):   Active transfer with annealed group prior.
  Phase 3 (epoch >= freeze):  Freeze transfer weights, fine-tune.

References:
  - Fifty et al., "Efficiently Identifying Task Groupings for Multi-Task Learning" (NeurIPS 2021)
  - Chen et al., "GradNorm" (ICML 2018) for gradient-based task balancing
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TaskAffinityComputer(nn.Module):
    """Computes pairwise task affinity via gradient cosine similarity.

    Maintains an EMA-smoothed affinity matrix for training stability.

    Args:
        task_names: Ordered list of task name strings.
        ema_decay: Exponential moving average decay for the affinity matrix.
    """

    def __init__(self, task_names: List[str], ema_decay: float = 0.9):
        super().__init__()
        self.task_names = task_names
        self.n_tasks = len(task_names)
        self.ema_decay = ema_decay

        self.register_buffer("affinity_matrix", torch.eye(self.n_tasks))
        self.register_buffer("update_count", torch.tensor(0))

    def compute_affinity(
        self,
        task_gradients: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute gradient-based affinity and update EMA.

        Args:
            task_gradients: ``{task_name: flattened_gradient_tensor}``.

        Returns:
            ``(n_tasks, n_tasks)`` cosine similarity matrix.
        """
        # Find a reference gradient for zero-padding missing tasks
        reference_grad = None
        for name in self.task_names:
            if name in task_gradients:
                reference_grad = task_gradients[name]
                break

        if reference_grad is None:
            return torch.eye(self.n_tasks, device=self.affinity_matrix.device)

        # Collect gradients (zero-pad missing tasks)
        grad_list = []
        for name in self.task_names:
            if name in task_gradients:
                grad_list.append(task_gradients[name])
            else:
                grad_list.append(torch.zeros_like(reference_grad))

        grad_matrix = torch.stack(grad_list, dim=0)  # (n_tasks, grad_dim)

        # Cosine similarity matrix
        norms = grad_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = grad_matrix / norms
        affinity = torch.mm(normalized, normalized.t())

        # EMA update
        with torch.no_grad():
            if self.update_count > 0:
                new = self.ema_decay * self.affinity_matrix + (1 - self.ema_decay) * affinity
                self.affinity_matrix.copy_(new.clamp(-1.0, 1.0))
            else:
                self.affinity_matrix.copy_(affinity.clamp(-1.0, 1.0))
            self.update_count.add_(1)

        return self.affinity_matrix

    def get_affinity_matrix(self) -> torch.Tensor:
        """Return the current (EMA-smoothed) affinity matrix."""
        return self.affinity_matrix


class AdaptiveTaskTransfer(nn.Module):
    """Adaptive Task Transfer (adaTT) module.

    Computes transfer-enhanced losses by weighting auxiliary task losses
    based on gradient affinity, modulated by a configurable group prior.

    Args:
        task_names: Ordered list of task name strings.
        transfer_lambda: Scaling factor for transfer losses.
        temperature: Softmax temperature for transfer weights.
        use_group_prior: Whether to incorporate task-group prior.
        warmup_epochs: Number of epochs for affinity-only warmup.
        freeze_epoch: Epoch at which to freeze transfer weights (or ``None``).
        negative_transfer_threshold: Affinity below this blocks transfer.
        task_groups: ``{group_name: {"members": [...], "intra_strength": float}}``.
        inter_group_strength: Default transfer strength between groups.
        ema_decay: EMA decay for affinity computation.
        prior_blend_start: Group prior weight at start of Phase 2.
        prior_blend_end: Group prior weight at end of Phase 2.
        max_transfer_ratio: Cap transfer loss at this fraction of original loss.
    """

    def __init__(
        self,
        task_names: List[str],
        transfer_lambda: float = 0.1,
        temperature: float = 1.0,
        use_group_prior: bool = True,
        warmup_epochs: int = 10,
        freeze_epoch: Optional[int] = None,
        negative_transfer_threshold: float = -0.1,
        task_groups: Optional[Dict[str, dict]] = None,
        inter_group_strength: float = 0.3,
        ema_decay: float = 0.9,
        prior_blend_start: float = 0.5,
        prior_blend_end: float = 0.1,
        max_transfer_ratio: float = 0.5,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.task_names = task_names
        self.n_tasks = len(task_names)
        self.transfer_lambda = transfer_lambda
        self.max_transfer_ratio = max_transfer_ratio
        self.temperature = temperature
        self.use_group_prior = use_group_prior
        self.warmup_epochs = warmup_epochs
        self.freeze_epoch = freeze_epoch
        self.neg_threshold = negative_transfer_threshold
        self.prior_blend_start = prior_blend_start
        self.prior_blend_end = prior_blend_end

        # Validate phase ordering
        if freeze_epoch is not None and freeze_epoch <= warmup_epochs:
            raise ValueError(
                f"freeze_epoch ({freeze_epoch}) must be > warmup_epochs ({warmup_epochs}). "
                f"Otherwise Phase 2 (active transfer) is skipped entirely."
            )

        # Parse task groups from config
        if task_groups is not None:
            self._task_groups = {
                gname: gcfg.get("members", []) if isinstance(gcfg, dict) else gcfg
                for gname, gcfg in task_groups.items()
            }
            self._intra_strength = {
                gname: gcfg.get("intra_strength", 0.5) if isinstance(gcfg, dict) else 0.5
                for gname, gcfg in task_groups.items()
            }
        else:
            self._task_groups = {}
            self._intra_strength = {}

        self._inter_group_strength = inter_group_strength

        # Affinity computer
        self.affinity_computer = TaskAffinityComputer(task_names, ema_decay=ema_decay)

        # Learnable transfer weight offsets
        self.transfer_weights = nn.Parameter(
            torch.zeros(self.n_tasks, self.n_tasks)
        )

        # Group prior matrix
        if use_group_prior and self._task_groups:
            group_prior = self._build_group_prior()
            self.register_buffer("group_prior", group_prior)
        else:
            self.use_group_prior = False

        # Diagonal mask (exclude self-transfer)
        self.register_buffer("diag_mask", torch.eye(self.n_tasks, dtype=torch.bool))

        # State tracking
        self.register_buffer("current_epoch", torch.tensor(0))
        self.register_buffer("is_frozen", torch.tensor(False))

        # Enable/disable toggle
        self._enabled = True

        logger.info(
            f"AdaptiveTaskTransfer init: tasks={len(task_names)}, "
            f"lambda={transfer_lambda}, tau={temperature}, "
            f"warmup={warmup_epochs}, groups={len(self._task_groups)}, "
            f"prior_blend={prior_blend_start}->{prior_blend_end}"
        )

    # ------------------------------------------------------------------
    # Enable / Disable
    # ------------------------------------------------------------------

    def disable(self) -> None:
        """Disable transfer computation (weights are preserved in state_dict)."""
        self._enabled = False

    def enable(self) -> None:
        """Re-enable transfer computation."""
        self._enabled = True

    @property
    def is_enabled(self) -> bool:
        return getattr(self, "_enabled", True)

    # ------------------------------------------------------------------
    # Group prior
    # ------------------------------------------------------------------

    def _build_group_prior(self) -> torch.Tensor:
        """Build the task-group prior matrix from config."""
        prior = torch.ones(self.n_tasks, self.n_tasks) * self._inter_group_strength

        for group_name, members in self._task_groups.items():
            strength = self._intra_strength.get(group_name, 0.5)
            indices = [
                self.task_names.index(m)
                for m in members
                if m in self.task_names
            ]
            for i in indices:
                for j in indices:
                    if i != j:
                        prior[i, j] = strength

        prior.fill_diagonal_(0.0)

        # Row normalize
        row_sums = prior.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return prior / row_sums

    # ------------------------------------------------------------------
    # Transfer weight computation
    # ------------------------------------------------------------------

    def _compute_transfer_weights(self, affinity: torch.Tensor) -> torch.Tensor:
        """Compute transfer weights with prior blend annealing."""
        raw_weights = self.transfer_weights + affinity

        # Blend with group prior (annealed over training)
        if self.use_group_prior:
            epoch = self.current_epoch.item()
            if epoch < self.warmup_epochs:
                r = self.prior_blend_start
            elif self.is_frozen.item():
                r = self.prior_blend_end
            else:
                freeze = self.freeze_epoch or (self.warmup_epochs + 30)
                progress = min(
                    (epoch - self.warmup_epochs) / max(freeze - self.warmup_epochs, 1),
                    1.0,
                )
                r = self.prior_blend_start - (self.prior_blend_start - self.prior_blend_end) * progress
            raw_weights = raw_weights * (1 - r) + self.group_prior * r

        # Block negative transfer
        raw_weights = torch.where(
            affinity > self.neg_threshold,
            raw_weights,
            torch.zeros_like(raw_weights),
        )

        # Zero diagonal (no self-transfer)
        raw_weights = raw_weights.masked_fill(self.diag_mask, 0.0)

        # Softmax normalize
        return F.softmax(raw_weights / max(self.temperature, 1e-6), dim=-1)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def compute_transfer_loss(
        self,
        task_losses: Dict[str, torch.Tensor],
        task_gradients: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute transfer-enhanced losses for all tasks.

        Args:
            task_losses: ``{task_name: scalar_loss}``.
            task_gradients: ``{task_name: flattened_gradient}`` for affinity update.

        Returns:
            ``{task_name: enhanced_loss}`` dict.
        """
        if not self.is_enabled:
            return task_losses

        epoch = self.current_epoch.item()

        # Phase 1: warmup (measure affinity only)
        if epoch < self.warmup_epochs:
            if task_gradients is not None:
                self.affinity_computer.compute_affinity(task_gradients)
            return task_losses

        # Phase 3: frozen weights
        if self.is_frozen.item():
            return self._apply_transfer(task_losses, detach_weights=True)

        # Phase 2: dynamic transfer
        if task_gradients is not None:
            self.affinity_computer.compute_affinity(task_gradients)

        return self._apply_transfer(task_losses, detach_weights=False)

    def _apply_transfer(
        self,
        task_losses: Dict[str, torch.Tensor],
        detach_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Apply transfer-enhanced loss computation."""
        affinity = self.affinity_computer.get_affinity_matrix()
        transfer_w = self._compute_transfer_weights(affinity)

        # Build loss tensor and mask (for tasks missing from this batch)
        loss_list: List[torch.Tensor] = []
        loss_mask: List[float] = []
        for name in self.task_names:
            if name in task_losses:
                loss_list.append(task_losses[name])
                loss_mask.append(1.0)
            else:
                loss_list.append(torch.tensor(0.0, device=affinity.device, requires_grad=False))
                loss_mask.append(0.0)

        loss_tensor = torch.stack(loss_list)
        mask_tensor = torch.tensor(loss_mask, device=affinity.device)

        enhanced: Dict[str, torch.Tensor] = {}
        for i, task_name in enumerate(self.task_names):
            if task_name not in task_losses:
                continue

            original_loss = task_losses[task_name]

            # Transfer loss: sum_j w(i->j) * L_j (masked)
            w = transfer_w[i].detach() if detach_weights else transfer_w[i]
            masked_w = w * mask_tensor
            transfer_loss = (masked_w * loss_tensor).sum()

            # Clamp transfer contribution
            raw_transfer = self.transfer_lambda * transfer_loss
            if self.max_transfer_ratio > 0:
                max_val = original_loss.detach() * self.max_transfer_ratio
                raw_transfer = torch.clamp(raw_transfer, max=max_val)

            enhanced[task_name] = original_loss + raw_transfer

        return enhanced

    # ------------------------------------------------------------------
    # Epoch lifecycle
    # ------------------------------------------------------------------

    def on_epoch_end(self, epoch: int) -> None:
        """Update internal state at the end of each epoch.

        Must be called by the training loop.
        """
        self.current_epoch.fill_(epoch)

        if self.freeze_epoch is not None and epoch >= self.freeze_epoch:
            if not self.is_frozen.item():
                self.is_frozen.fill_(True)
                logger.info(f"adaTT: transfer weights frozen at epoch {epoch}")

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def get_transfer_matrix(self) -> torch.Tensor:
        """Return the current transfer weight matrix (detached)."""
        with torch.no_grad():
            affinity = self.affinity_computer.get_affinity_matrix()
            return self._compute_transfer_weights(affinity)

    def detect_negative_transfer(self) -> Dict[str, List[str]]:
        """Identify task pairs with negative affinity.

        Returns:
            ``{task_i: [task_j, ...]}`` where A(i,j) < threshold.
        """
        affinity = self.affinity_computer.get_affinity_matrix()
        negative_pairs: Dict[str, List[str]] = {}

        for i, task_i in enumerate(self.task_names):
            neg_list = [
                self.task_names[j]
                for j in range(self.n_tasks)
                if i != j and affinity[i, j].item() < self.neg_threshold
            ]
            if neg_list:
                negative_pairs[task_i] = neg_list

        if negative_pairs:
            logger.warning(f"Negative transfer detected: {negative_pairs}")

        return negative_pairs
