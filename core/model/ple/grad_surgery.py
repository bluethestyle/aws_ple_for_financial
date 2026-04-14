"""
Task-Type Gradient Surgery for Heterogeneous Multi-Task Learning.

Addresses gradient conflict between task types (binary / multiclass /
regression) in large-scale MTL by projecting out conflicting gradient
components at the shared-parameter level.

Key differences from adaTT (loss-level transfer):
  - Operates on **gradients**, not losses -- prevents loss-scale mismatch.
  - Projects at **task-type group** level (3 groups) rather than all task
    pairs (156 pairs for 13 tasks) -- stable estimation.
  - Applied between backward() and optimizer.step() -- zero architecture
    change to PLE/expert basket.

Core equation (PCGrad-style projection):
  For two gradient vectors g_i, g_j with cos(g_i, g_j) < 0:
    g_i' = g_i - (g_i . g_j / ||g_j||^2) * g_j
  This removes the component of g_i that conflicts with g_j.

When applied at task-type group level:
  g_binary  = mean gradient of 7 binary tasks
  g_multi   = mean gradient of 3 multiclass tasks
  g_reg     = mean gradient of 3 regression tasks
  -> Project pairwise among {g_binary, g_multi, g_reg}
  -> Replace shared param gradient with projected sum

References:
  - Yu et al., "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
  - Liu et al., "Conflict-Averse Gradient Descent" (NeurIPS 2021)
  - Li et al., "AdaTT" (2023) for task affinity measurement
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class GradSurgeryConfig:
    """Configuration for GradSurgery module."""

    enabled: bool = False
    # Task type grouping: {group_name: [task_names]}
    task_type_groups: Dict[str, List[str]] = field(default_factory=dict)
    # Conflict threshold: project when cosine similarity < this value
    conflict_threshold: float = 0.0
    # Warmup: no projection for first N epochs (let model stabilize)
    warmup_epochs: int = 2
    # Apply projection every N steps (matches adaTT grad_interval for fair comparison)
    grad_interval: int = 10
    # EMA decay for group affinity tracking
    ema_decay: float = 0.9
    # Whether to log group affinity every N epochs
    log_interval: int = 1


class GradSurgery(nn.Module):
    """Task-type gradient surgery for heterogeneous MTL.

    Instead of projecting all k*(k-1) task pairs (O(k^2), unstable for
    k=13), we aggregate gradients into task-type groups and project only
    between groups (3 groups = 6 directed pairs).

    Usage in trainer::

        # After loss.backward(), before optimizer.step():
        if grad_surgery is not None and grad_surgery.is_active(epoch):
            grad_surgery.project(
                shared_params=list(model.shared_expert_parameters()),
                task_losses=per_task_losses,
                task_types=task_type_map,
            )

    Args:
        config: GradSurgeryConfig instance.
        task_names: Ordered list of all task names.
    """

    def __init__(self, config: GradSurgeryConfig, task_names: List[str]):
        super().__init__()
        self.config = config
        self.task_names = task_names
        self.n_tasks = len(task_names)

        # Build task -> group mapping
        self._task_to_group: Dict[str, str] = {}
        self._group_names: List[str] = []
        for group_name, members in config.task_type_groups.items():
            self._group_names.append(group_name)
            for task_name in members:
                self._task_to_group[task_name] = group_name

        self.n_groups = len(self._group_names)

        # EMA-smoothed group affinity matrix (for logging/analysis)
        self.register_buffer(
            "group_affinity",
            torch.eye(self.n_groups),
        )
        self.register_buffer("update_count", torch.tensor(0))
        self._step_count = 0

        logger.info(
            "GradSurgery init: %d tasks -> %d groups %s, "
            "conflict_threshold=%.2f, warmup=%d, grad_interval=%d",
            self.n_tasks, self.n_groups, self._group_names,
            config.conflict_threshold, config.warmup_epochs,
            config.grad_interval,
        )

    def is_active(self, epoch: int) -> bool:
        """Whether surgery should run at this epoch (warmup check only).

        The per-step grad_interval check is inside project().
        Trainer uses this to decide retain_graph on the right steps.
        """
        return self.config.enabled and epoch >= self.config.warmup_epochs

    def should_run_this_step(self, epoch: int) -> bool:
        """Whether surgery should run on this specific step.

        Combines epoch warmup + grad_interval. Trainer should use this
        to set retain_graph=True only when needed.
        """
        if not self.is_active(epoch):
            return False
        return self._step_count % self.config.grad_interval == 0

    def project(
        self,
        shared_params: List[nn.Parameter],
        task_losses: Dict[str, torch.Tensor],
        epoch: int = 0,
    ) -> Dict[str, float]:
        """Apply gradient surgery on shared parameters.

        Must be called AFTER loss.backward() and BEFORE optimizer.step().
        Modifies .grad of shared_params in-place.
        Only runs every grad_interval steps to reduce VRAM overhead.

        Args:
            shared_params: List of shared expert parameters.
            task_losses: Per-task scalar losses (already backpropagated).
            epoch: Current epoch number.

        Returns:
            Dict of group-pair cosine similarities (for logging).
        """
        self._step_count += 1
        if not shared_params or not task_losses:
            return {}

        # Only run every grad_interval steps (matches adaTT design)
        if self._step_count % self.config.grad_interval != 0:
            return {}

        # Filter to params that have gradients and require grad
        params_with_grad = [p for p in shared_params if p.grad is not None]
        if not params_with_grad:
            return {}

        # Step 1+2: Compute group-level gradients directly.
        # Instead of 13 individual autograd.grad() calls, sum losses per
        # task-type group first, then call autograd.grad() only 3 times.
        # Mathematically identical (gradient linearity) but ~4x less VRAM.
        group_grads = self._compute_group_gradients(
            params_with_grad, task_losses,
        )

        if len(group_grads) < 2:
            return {}

        # Step 3: Pairwise projection between groups
        projected_grads, affinities = self._project_groups(group_grads)

        # Step 4: Update EMA affinity
        self._update_affinity(affinities)

        # Step 5: Reconstruct final gradient and write back to .grad
        self._write_projected_gradient(params_with_grad, projected_grads)

        # Log
        if epoch % self.config.log_interval == 0:
            aff_str = "  ".join(
                f"{g1}->{g2}={v:.3f}"
                for (g1, g2), v in affinities.items()
            )
            logger.info(
                "GradSurgery (epoch %d): %s", epoch, aff_str,
            )

        return {k: v for k, v in affinities.items()}

    def _compute_group_gradients(
        self,
        params: List[nn.Parameter],
        task_losses: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute group-level gradients directly (VRAM-efficient).

        Sums task losses within each type group first, then calls
        autograd.grad() once per group (3 calls instead of 13).
        Mathematically identical to per-task grad + averaging,
        because grad(sum(L_i)) = sum(grad(L_i)).
        """
        # Sum losses per group
        group_losses: Dict[str, Optional[torch.Tensor]] = {}
        group_counts: Dict[str, int] = {}
        for group_name in self._group_names:
            members = self.config.task_type_groups[group_name]
            losses = [
                task_losses[t] for t in members
                if t in task_losses and task_losses[t].requires_grad
            ]
            if losses:
                group_losses[group_name] = torch.stack(losses).mean()
                group_counts[group_name] = len(losses)
            else:
                group_losses[group_name] = None

        # Compute gradient per group (only 3 autograd.grad calls)
        group_grads: Dict[str, torch.Tensor] = {}
        for group_name, gloss in group_losses.items():
            if gloss is None:
                continue
            try:
                grads = torch.autograd.grad(
                    gloss,
                    params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
                flat = torch.cat([
                    g.flatten() if g is not None else torch.zeros(
                        p.numel(), device=p.device,
                    )
                    for g, p in zip(grads, params)
                ])
                group_grads[group_name] = flat
            except RuntimeError:
                pass

        return group_grads

    def _compute_per_task_gradients(
        self,
        params: List[nn.Parameter],
        task_losses: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute per-task flattened gradients on shared params.

        Uses torch.autograd.grad with retain_graph=True to get
        individual task gradients without modifying .grad.
        """
        task_grads = {}
        for task_name, loss in task_losses.items():
            if task_name not in self._task_to_group:
                continue
            if not loss.requires_grad:
                continue

            try:
                grads = torch.autograd.grad(
                    loss,
                    params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
                # Flatten and concatenate
                flat = torch.cat([
                    g.flatten() if g is not None else torch.zeros(
                        p.numel(), device=p.device,
                    )
                    for g, p in zip(grads, params)
                ])
                task_grads[task_name] = flat
            except RuntimeError:
                # Graph already freed or other issue -- skip this task
                pass

        return task_grads

    def _aggregate_group_gradients(
        self,
        task_grads: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Average task gradients within each type group."""
        group_grads: Dict[str, torch.Tensor] = {}

        for group_name in self._group_names:
            members = self.config.task_type_groups[group_name]
            member_grads = [
                task_grads[t] for t in members if t in task_grads
            ]
            if member_grads:
                group_grads[group_name] = torch.stack(member_grads).mean(dim=0)

        return group_grads

    def _project_groups(
        self,
        group_grads: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str], float]]:
        """PCGrad-style pairwise projection between group gradients.

        For each pair (i, j) with cos(g_i, g_j) < threshold:
          g_i' = g_i - (g_i . g_j / ||g_j||^2) * g_j

        Returns:
            projected_grads: {group_name: projected_gradient}
            affinities: {(group_i, group_j): cosine_similarity}
        """
        names = list(group_grads.keys())
        projected = {name: group_grads[name].clone() for name in names}
        affinities: Dict[Tuple[str, str], float] = {}

        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i == j:
                    continue

                g_i = projected[name_i]
                g_j = group_grads[name_j]  # Use original, not projected

                # Cosine similarity
                cos_sim = torch.dot(g_i, g_j) / (
                    g_i.norm().clamp(min=1e-8) * g_j.norm().clamp(min=1e-8)
                )
                affinities[(name_i, name_j)] = cos_sim.item()

                # Project out conflicting component
                if cos_sim.item() < self.config.conflict_threshold:
                    dot = torch.dot(g_i, g_j)
                    norm_sq = g_j.norm().square().clamp(min=1e-8)
                    projected[name_i] = g_i - (dot / norm_sq) * g_j

        return projected, affinities

    def _update_affinity(
        self,
        affinities: Dict[Tuple[str, str], float],
    ) -> None:
        """Update EMA-smoothed group affinity matrix."""
        with torch.no_grad():
            new_aff = torch.eye(
                self.n_groups, device=self.group_affinity.device,
            )
            for (g_i, g_j), cos_val in affinities.items():
                i = self._group_names.index(g_i)
                j = self._group_names.index(g_j)
                new_aff[i, j] = cos_val

            if self.update_count > 0:
                blended = (
                    self.config.ema_decay * self.group_affinity
                    + (1 - self.config.ema_decay) * new_aff
                )
                self.group_affinity.copy_(blended.clamp(-1, 1))
            else:
                self.group_affinity.copy_(new_aff.clamp(-1, 1))
            self.update_count.add_(1)

    def _write_projected_gradient(
        self,
        params: List[nn.Parameter],
        projected_grads: Dict[str, torch.Tensor],
    ) -> None:
        """Sum projected group gradients and write back to param.grad."""
        # Sum all projected group gradients
        all_grads = list(projected_grads.values())
        if not all_grads:
            return

        final_grad = torch.stack(all_grads).sum(dim=0)

        # Unflatten and write to .grad
        offset = 0
        for p in params:
            numel = p.numel()
            p.grad = final_grad[offset:offset + numel].view_as(p).clone()
            offset += numel

    def get_group_affinity(self) -> Dict[str, Dict[str, float]]:
        """Return current group affinity as a nested dict (for eval report)."""
        result = {}
        for i, g_i in enumerate(self._group_names):
            result[g_i] = {}
            for j, g_j in enumerate(self._group_names):
                result[g_i][g_j] = self.group_affinity[i, j].item()
        return result
