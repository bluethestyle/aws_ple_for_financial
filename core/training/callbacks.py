"""
Training callbacks for PLE training pipeline.

Callbacks hook into the training loop at well-defined points (epoch start/end,
step start/end, phase start/end) to implement cross-cutting concerns like
early stopping, checkpointing, metric logging, and LR scheduling.

Usage::

    callbacks = [
        EarlyStoppingCallback(patience=5),
        CheckpointCallback(checkpoint_dir="checkpoints"),
        MetricLoggerCallback(tracker=my_tracker),
        LRSchedulerCallback(scheduler=my_scheduler),
    ]
    trainer = PLETrainer(model, config, callbacks=callbacks)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from .experiment import ExperimentTracker

logger = logging.getLogger(__name__)


# ============================================================================
# Base callback
# ============================================================================


class TrainingCallback:
    """Base class for training callbacks.

    Subclasses override the hooks they need.  All hooks receive a
    ``state`` dict with contextual information (epoch, step, loss, etc.).
    """

    def on_train_begin(self, state: Dict[str, Any]) -> None:
        """Called at the very start of training."""

    def on_train_end(self, state: Dict[str, Any]) -> None:
        """Called at the very end of training."""

    def on_phase_begin(self, state: Dict[str, Any]) -> None:
        """Called at the start of a training phase (phase1 / phase2)."""

    def on_phase_end(self, state: Dict[str, Any]) -> None:
        """Called at the end of a training phase."""

    def on_epoch_begin(self, state: Dict[str, Any]) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(self, state: Dict[str, Any]) -> bool:
        """Called at the end of each epoch.

        Returns:
            ``True`` to stop training (early stopping), ``False`` to continue.
        """
        return False

    def on_step_end(self, state: Dict[str, Any]) -> None:
        """Called after each optimiser step (not every batch if accumulating)."""


class CallbackList:
    """Convenience container that dispatches to multiple callbacks."""

    def __init__(self, callbacks: Optional[List[TrainingCallback]] = None) -> None:
        self.callbacks = callbacks or []

    def on_train_begin(self, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(state)

    def on_train_end(self, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_train_end(state)

    def on_phase_begin(self, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_phase_begin(state)

    def on_phase_end(self, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_phase_end(state)

    def on_epoch_begin(self, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(state)

    def on_epoch_end(self, state: Dict[str, Any]) -> bool:
        """Returns ``True`` if **any** callback requests a stop."""
        return any(cb.on_epoch_end(state) for cb in self.callbacks)

    def on_step_end(self, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_step_end(state)


# ============================================================================
# Early stopping
# ============================================================================


class EarlyStoppingCallback(TrainingCallback):
    """Stop training when validation loss stops improving.

    Supports two complementary criteria:

    1. **Primary**: ``val_loss`` has not improved for ``patience`` epochs.
    2. **Secondary**: Average AUC has declined for ``auc_decline_patience``
       consecutive epochs, even if ``val_loss`` is still improving.

    Args:
        patience: Number of epochs to wait for ``val_loss`` improvement.
        min_delta: Minimum improvement to qualify as an improvement.
        auc_decline_patience: Consecutive AUC decline epochs before stopping.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        auc_decline_patience: int = 3,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.auc_decline_patience = auc_decline_patience
        self.reset()

    def reset(self) -> None:
        """Reset internal state (called between phases)."""
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_avg_auc = 0.0
        self.auc_decline_counter = 0

    def on_phase_begin(self, state: Dict[str, Any]) -> None:
        self.reset()

    def on_epoch_end(self, state: Dict[str, Any]) -> bool:
        val_loss = state.get("val_loss")
        if val_loss is None:
            return False

        # Primary: val_loss improvement
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Secondary: AUC decline
        avg_auc = state.get("val_metrics", {}).get("avg_auc", 0.0)
        if avg_auc > 0:
            if avg_auc >= self.best_avg_auc:
                self.best_avg_auc = avg_auc
                self.auc_decline_counter = 0
            else:
                self.auc_decline_counter += 1

        loss_stop = self.patience_counter >= self.patience
        auc_stop = (
            self.auc_decline_counter >= self.auc_decline_patience
            and self.best_avg_auc > 0
        )

        if loss_stop or auc_stop:
            reason = "val_loss patience" if loss_stop else "AUC decline"
            logger.info(
                "EarlyStopping: stopping at epoch %d (%s). "
                "best_val_loss=%.6f, best_avg_auc=%.4f",
                state.get("epoch", -1), reason,
                self.best_val_loss, self.best_avg_auc,
            )
            return True

        return False


# ============================================================================
# Checkpointing
# ============================================================================


class CheckpointCallback(TrainingCallback):
    """Save model checkpoints periodically and on best validation loss.

    Maintains a fixed pool of epoch checkpoints (oldest deleted when
    ``max_to_keep`` is exceeded).  The ``best.pt`` checkpoint is always
    kept separately.

    Args:
        checkpoint_dir: Directory to write checkpoint files.
        save_every_n_epochs: Save an epoch checkpoint every N epochs.
        max_to_keep: Maximum epoch checkpoints to retain (excludes ``best``).
        save_optimizer: Whether to include optimizer state.
        save_scheduler: Whether to include scheduler state.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        save_every_n_epochs: int = 5,
        max_to_keep: int = 5,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every_n_epochs = save_every_n_epochs
        self.max_to_keep = max_to_keep
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler

        self.best_val_loss = float("inf")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_phase_begin(self, state: Dict[str, Any]) -> None:
        self.best_val_loss = float("inf")

    def on_epoch_end(self, state: Dict[str, Any]) -> bool:
        epoch = state.get("epoch", 0)
        val_loss = state.get("val_loss")

        # Best checkpoint
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save(state, "best")

        # Periodic checkpoint
        if epoch > 0 and epoch % self.save_every_n_epochs == 0:
            self._save(state, f"epoch_{epoch}")
            self._cleanup_old()

        return False  # never stops training

    def _save(self, state: Dict[str, Any], name: str) -> None:
        """Save a checkpoint to disk."""
        checkpoint = {
            "model_state_dict": state["model"].state_dict(),
            "epoch": state.get("epoch", 0),
            "global_step": state.get("global_step", 0),
            "best_val_loss": self.best_val_loss,
            "phase": state.get("phase_name", ""),
        }

        if self.save_optimizer and "optimizer" in state:
            checkpoint["optimizer_state_dict"] = state["optimizer"].state_dict()
        if self.save_scheduler and "scheduler" in state:
            checkpoint["scheduler_state_dict"] = state["scheduler"].state_dict()

        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s", path)

        # Log artifact to tracker if available
        tracker = state.get("tracker")
        if tracker is not None:
            tracker.log_artifact(path)

    def _cleanup_old(self) -> None:
        """Remove oldest epoch checkpoints beyond ``max_to_keep``."""
        epoch_ckpts = sorted(
            self.checkpoint_dir.glob("epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if len(epoch_ckpts) > self.max_to_keep:
            for old in epoch_ckpts[: -self.max_to_keep]:
                old.unlink()
                logger.info("Deleted old checkpoint: %s", old)

    @staticmethod
    def load_checkpoint(
        path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Load a checkpoint and restore model/optimizer/scheduler state.

        Args:
            path: Path to the checkpoint file.
            model: Model to load state into.
            optimizer: Optional optimizer to restore.
            scheduler: Optional scheduler to restore.
            device: Device to map tensors to.

        Returns:
            The full checkpoint dict (for inspecting epoch, step, etc.).
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(
            "Checkpoint loaded from %s (epoch=%d, step=%d)",
            path, checkpoint.get("epoch", -1), checkpoint.get("global_step", -1),
        )
        return checkpoint


# ============================================================================
# Metric logger
# ============================================================================


class MetricLoggerCallback(TrainingCallback):
    """Log training metrics to an :class:`ExperimentTracker`.

    Args:
        tracker: The experiment tracker instance.
        log_every_n_steps: Log step-level metrics every N optimiser steps.
    """

    def __init__(
        self,
        tracker: "ExperimentTracker",
        log_every_n_steps: int = 100,
    ) -> None:
        self.tracker = tracker
        self.log_every_n_steps = log_every_n_steps

    def on_train_begin(self, state: Dict[str, Any]) -> None:
        # Log hyperparameters
        config = state.get("config")
        if config is not None:
            from dataclasses import asdict

            try:
                params = asdict(config)
                # Flatten nested dicts for logging
                flat = _flatten_dict(params)
                self.tracker.log_params(flat)
            except Exception:
                pass

    def on_step_end(self, state: Dict[str, Any]) -> None:
        step = state.get("global_step", 0)
        if step % self.log_every_n_steps != 0:
            return

        phase = state.get("phase_name", "train")
        metrics: Dict[str, float] = {}

        # Train loss
        train_loss = state.get("train_loss")
        if train_loss is not None:
            metrics[f"{phase}/train_loss"] = train_loss

        # Per-task losses
        task_losses = state.get("task_losses")
        if task_losses:
            for name, loss_val in task_losses.items():
                if hasattr(loss_val, "item"):
                    loss_val = loss_val.item()
                if isinstance(loss_val, (int, float)) and math.isfinite(loss_val):
                    metrics[f"{phase}/{name}_loss"] = loss_val

        # Learning rate
        optimizer = state.get("optimizer")
        if optimizer is not None:
            for i, pg in enumerate(optimizer.param_groups):
                metrics[f"{phase}/lr/group_{i}"] = pg["lr"]

        # Aux losses
        aux_losses = state.get("aux_losses")
        if aux_losses:
            for name, val in aux_losses.items():
                if hasattr(val, "item"):
                    val = val.item()
                if isinstance(val, (int, float)) and math.isfinite(val):
                    metrics[f"{phase}/{name}"] = val

        if metrics:
            self.tracker.log_metrics(metrics, step=step)

    def on_epoch_end(self, state: Dict[str, Any]) -> bool:
        epoch = state.get("epoch", 0)
        phase = state.get("phase_name", "train")
        metrics: Dict[str, float] = {}

        train_loss = state.get("train_loss")
        if train_loss is not None:
            metrics[f"{phase}/epoch_train_loss"] = train_loss

        val_loss = state.get("val_loss")
        if val_loss is not None:
            metrics[f"{phase}/epoch_val_loss"] = val_loss

        # Validation metrics (AUC, accuracy, etc.)
        val_metrics = state.get("val_metrics", {})
        for name, val in val_metrics.items():
            if name == "val_loss":
                continue
            if isinstance(val, (int, float)) and math.isfinite(val):
                metrics[f"{phase}/{name}"] = val

        if metrics:
            self.tracker.log_metrics(metrics, step=epoch)

        return False

    def on_train_end(self, state: Dict[str, Any]) -> None:
        self.tracker.end()


# ============================================================================
# LR Scheduler
# ============================================================================


class LRSchedulerCallback(TrainingCallback):
    """Step the learning rate scheduler at the end of each epoch.

    Handles scheduler recreation between phases (Phase 1 -> Phase 2) with
    different warmup and cosine period settings.

    Args:
        scheduler: The PyTorch LR scheduler to step.
    """

    def __init__(self, scheduler: Optional[Any] = None) -> None:
        self.scheduler = scheduler

    def set_scheduler(self, scheduler: Any) -> None:
        """Replace the scheduler (used between phases)."""
        self.scheduler = scheduler

    def on_epoch_end(self, state: Dict[str, Any]) -> bool:
        if self.scheduler is not None:
            self.scheduler.step()
        return False

    def get_last_lr(self) -> Optional[List[float]]:
        """Return the last computed learning rates."""
        if self.scheduler is not None and hasattr(self.scheduler, "get_last_lr"):
            try:
                return self.scheduler.get_last_lr()
            except RuntimeError:
                return None
        return None


# ============================================================================
# Helpers
# ============================================================================


def _flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = "/",
) -> Dict[str, Any]:
    """Flatten a nested dict into dot-separated keys."""
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
