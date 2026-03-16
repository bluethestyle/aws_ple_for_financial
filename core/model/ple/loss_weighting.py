"""
Loss Weighting Strategies for Multi-Task Learning.

Provides pluggable strategies to balance task losses during training:
  - **GradNorm**: Gradient-norm based dynamic weighting (Chen et al., ICML 2018).
  - **DWA**: Dynamic Weight Averaging based on loss rate of change (Liu et al., CVPR 2019).
  - **Uncertainty**: Homoscedastic uncertainty weighting (Kendall et al., CVPR 2018).

All strategies implement ``BaseLossWeighting`` and are created via the
``create_loss_weighting()`` factory.

References:
  - Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss
    Balancing in Deep Multitask Networks" (ICML 2018)
  - Liu et al., "End-to-End Multi-Task Learning with Attention" (CVPR 2019)
  - Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses
    for Scene Geometry and Semantics" (CVPR 2018)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# Abstract base
# ============================================================================

class BaseLossWeighting(nn.Module, ABC):
    """Abstract base for all loss weighting strategies.

    Args:
        num_tasks: Number of tasks.
        task_names: Ordered list of task name strings.
    """

    def __init__(self, num_tasks: int, task_names: List[str]):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_names = task_names

    @abstractmethod
    def compute_weights(self) -> Dict[str, float]:
        """Return current per-task loss weights."""

    @abstractmethod
    def update(
        self,
        losses: Dict[str, torch.Tensor],
        shared_params: Optional[List[nn.Parameter]] = None,
        epoch: int = 0,
    ) -> None:
        """Update weights given current task losses.

        Args:
            losses: ``{task_name: scalar_loss}``.
            shared_params: Shared-layer parameters (needed for GradNorm).
            epoch: Current epoch number.
        """


# ============================================================================
# GradNorm
# ============================================================================

class GradNormWeighting(BaseLossWeighting):
    """GradNorm loss weighting (Chen et al., 2018).

    Learns per-task weights in log-space and adjusts them so that tasks
    with slower relative training progress receive higher weight.

    Args:
        num_tasks: Number of tasks.
        task_names: Ordered list of task name strings.
        alpha: Asymmetry parameter controlling how aggressively to
            rebalance (higher = more aggressive).
        grad_interval: Only run GradNorm every N epochs (saves memory).
    """

    def __init__(
        self,
        num_tasks: int,
        task_names: List[str],
        alpha: float = 1.5,
        grad_interval: int = 1,
    ):
        super().__init__(num_tasks, task_names)
        self.alpha = alpha
        self.grad_interval = grad_interval

        # Learnable weights in log-space
        self.log_weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.0))
            for name in task_names
        })

        # Initial losses (set on first update)
        self.initial_losses: Optional[Dict[str, float]] = None

        logger.info(f"GradNorm init: {num_tasks} tasks, alpha={alpha}")

    def compute_weights(self) -> Dict[str, float]:
        return {
            name: torch.exp(self.log_weights[name]).item()
            for name in self.task_names
        }

    def update(
        self,
        losses: Dict[str, torch.Tensor],
        shared_params: Optional[List[nn.Parameter]] = None,
        epoch: int = 0,
    ) -> None:
        if shared_params is None or len(shared_params) == 0:
            logger.warning("GradNorm requires shared_params; skipping update")
            return

        # Run only every grad_interval epochs
        if self.grad_interval > 1 and epoch % self.grad_interval != 0:
            return

        # Store initial losses on first call
        if self.initial_losses is None:
            self.initial_losses = {
                name: (loss.item() if isinstance(loss, torch.Tensor) else loss)
                for name, loss in losses.items()
            }
            logger.info(f"GradNorm initial losses: {self.initial_losses}")
            return

        device = shared_params[0].device

        # Weighted losses
        weighted_losses = {
            name: torch.exp(self.log_weights[name]) * losses[name]
            for name in self.task_names
            if name in losses
        }

        # Per-task gradient norms on shared params
        grad_norms: Dict[str, torch.Tensor] = {}
        for name in self.task_names:
            if name not in weighted_losses:
                continue
            grads = torch.autograd.grad(
                weighted_losses[name],
                shared_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            grad_norm = torch.norm(
                torch.stack([torch.norm(g) for g in grads if g is not None])
            )
            grad_norms[name] = grad_norm

        if not grad_norms:
            return

        avg_grad_norm = sum(grad_norms.values()) / len(grad_norms)

        # Relative inverse training rates
        relative_rates: Dict[str, torch.Tensor] = {}
        for name in self.task_names:
            if name in losses and name in self.initial_losses:
                current = losses[name].item() if isinstance(losses[name], torch.Tensor) else losses[name]
                initial = self.initial_losses[name]
                if initial > 0:
                    relative_rates[name] = torch.tensor(
                        (current / initial) ** self.alpha,
                        device=device, dtype=torch.float32,
                    )
                else:
                    relative_rates[name] = torch.tensor(1.0, device=device)
            else:
                relative_rates[name] = torch.tensor(1.0, device=device)

        # Target gradient norms
        target_norms = {
            name: avg_grad_norm * relative_rates[name]
            for name in self.task_names
            if name in grad_norms
        }

        # GradNorm loss
        gradnorm_loss = torch.tensor(0.0, device=device)
        for name in self.task_names:
            if name in grad_norms and name in target_norms:
                gradnorm_loss = gradnorm_loss + torch.abs(
                    grad_norms[name] - target_norms[name]
                )

        # Backward to update log_weights
        gradnorm_loss.backward(retain_graph=True)

        # Renormalize weights (sum = num_tasks)
        with torch.no_grad():
            weight_sum = sum(torch.exp(self.log_weights[n]) for n in self.task_names)
            if weight_sum > 0:
                factor = self.num_tasks / weight_sum
                for name in self.task_names:
                    self.log_weights[name].data += torch.log(torch.tensor(factor))


# ============================================================================
# DWA (Dynamic Weight Averaging)
# ============================================================================

class DWAWeighting(BaseLossWeighting):
    """Dynamic Weight Averaging (Liu et al., 2019).

    Weights are derived from the rate of change of each task's loss
    over recent epochs.  Simple, efficient, and gradient-free.

    Args:
        num_tasks: Number of tasks.
        task_names: Ordered list of task name strings.
        temperature: Softmax temperature (higher = more uniform).
        window_size: Number of recent losses to track.
    """

    def __init__(
        self,
        num_tasks: int,
        task_names: List[str],
        temperature: float = 2.0,
        window_size: int = 5,
    ):
        super().__init__(num_tasks, task_names)
        self.temperature = temperature
        self.window_size = window_size

        self.loss_history: Dict[str, List[float]] = {name: [] for name in task_names}
        self.current_weights: Dict[str, float] = {
            name: 1.0 / num_tasks for name in task_names
        }

        logger.info(f"DWA init: {num_tasks} tasks, T={temperature}, window={window_size}")

    def compute_weights(self) -> Dict[str, float]:
        return self.current_weights.copy()

    def update(
        self,
        losses: Dict[str, torch.Tensor],
        shared_params: Optional[List[nn.Parameter]] = None,
        epoch: int = 0,
    ) -> None:
        # Record current losses
        for name in self.task_names:
            if name in losses:
                val = losses[name].item() if isinstance(losses[name], torch.Tensor) else losses[name]
                self.loss_history[name].append(val)
                if len(self.loss_history[name]) > self.window_size:
                    self.loss_history[name].pop(0)

        # Need at least 2 entries to compute rate of change
        if any(len(self.loss_history[n]) < 2 for n in self.task_names):
            return

        # Rate of change: r_k = L_k(t-1) / L_k(t-2)
        rates: Dict[str, float] = {}
        for name in self.task_names:
            recent = self.loss_history[name][-2:]
            rates[name] = recent[1] / recent[0] if recent[0] > 0 else 1.0

        # Softmax weights
        exp_rates = {
            name: torch.exp(torch.tensor(rate / self.temperature))
            for name, rate in rates.items()
        }
        total = sum(exp_rates.values())

        if total > 0:
            self.current_weights = {
                name: (exp_rates[name] / total).item()
                for name in self.task_names
            }
        else:
            self.current_weights = {
                name: 1.0 / self.num_tasks for name in self.task_names
            }


# ============================================================================
# Uncertainty Weighting
# ============================================================================

class UncertaintyWeighting(BaseLossWeighting):
    """Homoscedastic uncertainty weighting (Kendall et al., 2018).

    Learns per-task log-variance parameters.  The effective weight for
    task *k* is ``1 / (2 * sigma_k^2)`` plus a regularization term
    ``log(sigma_k)``.

    The log-variance parameters should be included in the optimizer's
    parameter list alongside the model parameters.

    Args:
        num_tasks: Number of tasks.
        task_names: Ordered list of task name strings.
    """

    def __init__(self, num_tasks: int, task_names: List[str]):
        super().__init__(num_tasks, task_names)

        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1))
            for name in task_names
        })

        logger.info(f"UncertaintyWeighting init: {num_tasks} tasks")

    def compute_weights(self) -> Dict[str, float]:
        """Return effective weights: 1 / (2 * exp(log_var))."""
        return {
            name: (1.0 / (2.0 * torch.exp(self.log_vars[name]))).item()
            for name in self.task_names
        }

    def update(
        self,
        losses: Dict[str, torch.Tensor],
        shared_params: Optional[List[nn.Parameter]] = None,
        epoch: int = 0,
    ) -> None:
        # Uncertainty weights are updated via gradient descent (no manual update)
        pass

    def weighted_loss(
        self,
        task_losses: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the uncertainty-weighted total loss.

        Loss_k_weighted = (1 / (2 * sigma_k^2)) * L_k + log(sigma_k)
                        = exp(-log_var_k) * L_k + log_var_k / 2

        Args:
            task_losses: ``{task_name: scalar_loss}``.

        Returns:
            Scalar total weighted loss.
        """
        total = torch.tensor(0.0, device=next(iter(task_losses.values())).device)
        for name in self.task_names:
            if name in task_losses:
                precision = torch.exp(-self.log_vars[name])
                total = total + precision * task_losses[name] + self.log_vars[name] / 2.0
        return total


# ============================================================================
# Factory
# ============================================================================

def create_loss_weighting(
    strategy: str,
    num_tasks: int,
    task_names: List[str],
    config: Optional[dict] = None,
) -> Optional[BaseLossWeighting]:
    """Create a loss weighting strategy by name.

    Args:
        strategy: One of ``"gradnorm"``, ``"dwa"``, ``"uncertainty"``,
            ``"fixed"``, ``"manual"``.
        num_tasks: Number of tasks.
        task_names: Ordered list of task name strings.
        config: Strategy-specific keyword arguments.

    Returns:
        A ``BaseLossWeighting`` instance, or ``None`` for ``"fixed"``/``"manual"``.
    """
    config = config or {}

    if strategy == "gradnorm":
        return GradNormWeighting(
            num_tasks=num_tasks,
            task_names=task_names,
            alpha=config.get("alpha", 1.5),
            grad_interval=config.get("grad_interval", 1),
        )
    elif strategy == "dwa":
        return DWAWeighting(
            num_tasks=num_tasks,
            task_names=task_names,
            temperature=config.get("temperature", 2.0),
            window_size=config.get("window_size", 5),
        )
    elif strategy == "uncertainty":
        return UncertaintyWeighting(
            num_tasks=num_tasks,
            task_names=task_names,
        )
    elif strategy in ("fixed", "manual"):
        return None
    else:
        raise ValueError(
            f"Unknown loss weighting strategy '{strategy}'. "
            f"Options: gradnorm, dwa, uncertainty, fixed, manual"
        )
