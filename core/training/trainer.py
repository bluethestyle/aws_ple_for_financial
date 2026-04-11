"""
PLE Trainer with 2-phase training, AMP, gradient accumulation, and callbacks.

Implements a production-quality training loop for multi-task PLE models:

* **Phase 1** -- Train shared experts (task experts optionally frozen).
* **Phase 2** -- Fine-tune task heads (shared experts optionally frozen).

The trainer is environment-agnostic: it works identically on a local GPU
and on SageMaker, with experiment tracking abstracted behind
:class:`~core.training.experiment.ExperimentTracker`.

Usage::

    from core.model.ple import PLEModel, PLEConfig, PLEInput
    from core.training import PLETrainer, TrainingConfig

    config = TrainingConfig.from_yaml("training.yaml")
    model = PLEModel(ple_config)
    trainer = PLETrainer(model, config, device=torch.device("cuda"))
    results = trainer.train(train_loader, val_loader)
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# AMP imports -- PyTorch 2.0+ uses torch.amp, older uses torch.cuda.amp
_AMP_NEW_API = False
try:
    from torch.amp import GradScaler, autocast
    # Verify the new API actually accepts device_type kwarg
    import inspect as _insp
    if "device_type" in _insp.signature(autocast.__init__).parameters:
        _AMP_NEW_API = True
    else:
        from torch.cuda.amp import GradScaler, autocast  # type: ignore[no-redef]
except (ImportError, TypeError):
    from torch.cuda.amp import GradScaler, autocast  # type: ignore[no-redef]

from core.model.ple.model import PLEModel, PLEInput, PLEOutput

from .callbacks import (
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    LRSchedulerCallback,
    MetricLoggerCallback,
    TrainingCallback,
)
from .config import TrainingConfig
from .experiment import ExperimentTracker, auto_tracker

logger = logging.getLogger(__name__)


class PLETrainer:
    """Two-phase trainer for PLE multi-task models.

    Handles the full training lifecycle:

    1. GPU configuration (TF32, cuDNN benchmark).
    2. Optimizer and scheduler creation with per-expert LR overrides.
    3. Two-phase training with proper freeze/unfreeze semantics.
    4. AMP with dynamic loss scaling.
    5. Gradient accumulation and clipping.
    6. Validation with per-task metric computation.
    7. Checkpoint save/load.
    8. Experiment tracking via pluggable :class:`ExperimentTracker`.

    Args:
        model: The :class:`PLEModel` to train.
        config: Training configuration.
        device: Torch device (auto-detected if ``None``).
        tracker: Experiment tracker (auto-detected if ``None``).
        callbacks: Additional training callbacks.
    """

    def __init__(
        self,
        model: PLEModel,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
        tracker: Optional[ExperimentTracker] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        audit_store: Optional[Any] = None,
    ) -> None:
        self._audit_store = audit_store
        self.model = model
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        # GPU optimizations
        if self.device.type == "cuda":
            if config.enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            if config.cudnn_benchmark:
                torch.backends.cudnn.benchmark = True

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # AMP scaler
        self.scaler = self._create_grad_scaler() if config.amp.enabled else None

        # State
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_metrics: Dict[str, float] = {}

        # Per-task validation masks: {task_name: np.ndarray (bool)} over val set
        # Set via set_task_val_masks() before training begins
        self.task_val_masks: Optional[Dict[str, "np.ndarray"]] = None

        # Epoch-level history for detailed monitoring
        self.epoch_history: List[Dict[str, Any]] = []

        # Experiment tracker
        if tracker is None:
            tracker = auto_tracker(
                experiment_name=config.experiment_name,
                run_name=config.run_name,
                tags=config.tags,
            )
        self.tracker = tracker

        # Callbacks
        default_callbacks: List[TrainingCallback] = [
            EarlyStoppingCallback(
                patience=config.early_stopping.patience,
                min_delta=config.early_stopping.min_delta,
                auc_decline_patience=config.early_stopping.auc_decline_patience,
            ),
            CheckpointCallback(
                checkpoint_dir=config.checkpoint.dir,
                save_every_n_epochs=config.checkpoint.save_every_n_epochs,
                max_to_keep=config.checkpoint.max_to_keep,
                save_optimizer=config.checkpoint.save_optimizer,
                save_scheduler=config.checkpoint.save_scheduler,
            ),
            MetricLoggerCallback(
                tracker=self.tracker,
                log_every_n_steps=config.logging.log_every_n_steps,
            ),
            LRSchedulerCallback(scheduler=self.scheduler),
        ]
        all_callbacks = default_callbacks + (callbacks or [])
        self.callbacks = CallbackList(all_callbacks)

        # Quick reference to specific callbacks for phase transitions
        self._lr_callback = next(
            (cb for cb in all_callbacks if isinstance(cb, LRSchedulerCallback)), None
        )
        self._es_callback = next(
            (cb for cb in all_callbacks if isinstance(cb, EarlyStoppingCallback)), None
        )

        logger.info(
            "PLETrainer initialised: device=%s, AMP=%s, phases=%d+%d epochs",
            self.device, config.amp.enabled,
            config.phase1.epochs, config.phase2.epochs,
        )

    # ------------------------------------------------------------------
    # Per-task validation masks
    # ------------------------------------------------------------------

    def set_task_val_masks(self, masks: Optional[Dict[str, "np.ndarray"]]) -> None:
        """Set per-task boolean masks for validation subset filtering.

        Parameters
        ----------
        masks : dict mapping task_name -> np.ndarray of bool
            Each array has length == val_set_size.  ``True`` means the
            sample is included in that task's validation metric computation.
            Tasks not present in *masks* use the full val set (default).
        """
        self.task_val_masks = masks
        if masks:
            for tn, m in masks.items():
                logger.info(
                    "task_val_mask[%s]: %d/%d samples (%.1f%%)",
                    tn, int(m.sum()), len(m), m.sum() / max(len(m), 1) * 100,
                )

    # ------------------------------------------------------------------
    # Optimizer creation
    # ------------------------------------------------------------------

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with optional per-expert LR overrides.

        When ``config.optimizer.expert_lr_overrides`` is provided, creates
        separate param groups for each named expert module, allowing
        different learning rates and weight decay values per expert.
        """
        opt_cfg = self.config.optimizer
        overrides = opt_cfg.expert_lr_overrides

        # Attempt to find named expert submodules for per-expert LR
        has_experts = (
            overrides
            and hasattr(self.model, "extraction_layers")
        )

        if not has_experts:
            # Simple: single param group
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            if not trainable:
                logger.warning("No trainable parameters found in model.")
                trainable = list(self.model.parameters())
            return torch.optim.AdamW(
                trainable,
                lr=opt_cfg.learning_rate,
                weight_decay=opt_cfg.weight_decay,
                betas=opt_cfg.betas,
                eps=opt_cfg.eps,
            )

        # Per-expert param groups
        param_groups: List[Dict[str, Any]] = []
        assigned_ids: set = set()

        # Match override keys against model's named children
        for name, module in self.model.named_modules():
            short_name = name.split(".")[-1]
            if short_name in overrides:
                expert_params = [p for p in module.parameters() if p.requires_grad]
                if not expert_params:
                    continue
                assigned_ids.update(id(p) for p in expert_params)
                cfg = overrides[short_name]
                param_groups.append({
                    "params": expert_params,
                    "lr": cfg.get("lr", opt_cfg.learning_rate),
                    "weight_decay": cfg.get("weight_decay", opt_cfg.weight_decay),
                })
                logger.info(
                    "Expert param group '%s': lr=%s, wd=%s, params=%d",
                    short_name, cfg.get("lr"), cfg.get("weight_decay"),
                    len(expert_params),
                )

        # Remaining parameters
        rest_params = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in assigned_ids
        ]
        if rest_params:
            param_groups.append({
                "params": rest_params,
                "lr": opt_cfg.learning_rate,
                "weight_decay": opt_cfg.weight_decay,
            })

        logger.info("Optimizer: %d param groups", len(param_groups))
        return torch.optim.AdamW(
            param_groups,
            betas=opt_cfg.betas,
            eps=opt_cfg.eps,
        )

    # ------------------------------------------------------------------
    # Scheduler creation
    # ------------------------------------------------------------------

    def _create_scheduler(
        self,
        warmup_epochs: Optional[int] = None,
        cosine_t0: Optional[int] = None,
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Create learning rate scheduler.

        Supports:
        - ``"cosine"``: CosineAnnealingWarmRestarts with optional linear warmup.
        - ``"linear"``: Linear decay from start_factor to 1.0.
        - ``"none"``: No scheduler.

        Args:
            warmup_epochs: Override warmup epochs (used for Phase 2).
            cosine_t0: Override cosine T_0 (used for Phase 2).
        """
        sched_cfg = self.config.scheduler
        if sched_cfg.name == "none":
            return None

        _warmup = warmup_epochs if warmup_epochs is not None else sched_cfg.warmup_epochs
        _t0 = cosine_t0 if cosine_t0 is not None else sched_cfg.cosine_t0

        if sched_cfg.name == "cosine":
            if _warmup > 0:
                warmup_sched = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=sched_cfg.warmup_start_factor,
                    total_iters=_warmup,
                )
                cosine_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=_t0,
                    T_mult=sched_cfg.cosine_t_mult,
                    eta_min=sched_cfg.cosine_eta_min,
                )
                logger.info(
                    "Scheduler: LinearWarmup(%d epochs) -> "
                    "CosineAnnealingWarmRestarts(T_0=%d, T_mult=%d)",
                    _warmup, _t0, sched_cfg.cosine_t_mult,
                )
                return torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_sched, cosine_sched],
                    milestones=[_warmup],
                )
            else:
                logger.info(
                    "Scheduler: CosineAnnealingWarmRestarts(T_0=%d)", _t0,
                )
                return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=_t0,
                    T_mult=sched_cfg.cosine_t_mult,
                    eta_min=sched_cfg.cosine_eta_min,
                )

        elif sched_cfg.name == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=sched_cfg.warmup_start_factor,
                total_iters=_warmup,
            )

        logger.warning("Unknown scheduler '%s', disabling.", sched_cfg.name)
        return None

    # ------------------------------------------------------------------
    # GradScaler
    # ------------------------------------------------------------------

    def _create_grad_scaler(self) -> GradScaler:
        """Create AMP GradScaler with parameters from config."""
        amp_cfg = self.config.amp
        device_type = getattr(self.device, "type", "cuda")
        scaler_kwargs = dict(
            init_scale=amp_cfg.grad_scaler_init_scale,
            growth_factor=amp_cfg.grad_scaler_growth_factor,
            backoff_factor=amp_cfg.grad_scaler_backoff_factor,
            growth_interval=amp_cfg.grad_scaler_growth_interval,
        )
        try:
            scaler = GradScaler(device_type, **scaler_kwargs)
        except TypeError:
            # PyTorch < 2.0 fallback
            scaler = GradScaler(**scaler_kwargs)
        self._grad_scaler_max_scale = amp_cfg.grad_scaler_max_scale
        return scaler

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        phase: str = "full",
    ) -> Dict[str, float]:
        """Run training.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            phase: Training mode:
                - ``"full"``: Run both Phase 1 and Phase 2.
                - ``"phase1"``: Only shared expert training.
                - ``"phase2"``: Only task head fine-tuning.

        Returns:
            Dict with ``best_val_loss`` and any validation metrics from
            the best epoch.
        """
        state = self._make_state()
        self.callbacks.on_train_begin(state)

        # Log task type breakdown so users can verify metric grouping
        _type_groups: Dict[str, List[str]] = {"binary": [], "multiclass": [], "regression": [], "other": []}
        for _tn in self.model.task_names:
            _tt = self.model.config.get_task_type(_tn)
            _type_groups.get(_tt, _type_groups["other"]).append(_tn)
        logger.info(
            "Task type breakdown — binary: %d %s | multiclass: %d %s | regression: %d %s",
            len(_type_groups["binary"]), _type_groups["binary"],
            len(_type_groups["multiclass"]), _type_groups["multiclass"],
            len(_type_groups["regression"]), _type_groups["regression"],
        )
        logger.info(
            "Metric aggregation: avg_auc <- binary only | avg_f1_macro <- multiclass only | "
            "avg_ndcg@3 <- multiclass tasks with topk_k configured | avg_mae <- regression only"
        )

        # Auto-compute class weights for multiclass tasks before training
        self._auto_compute_class_weights(train_loader)
        # Auto-compute pos_weights for binary tasks before training
        self._auto_compute_pos_weights(train_loader)

        try:
            if phase == "full":
                # Phase 1: Shared experts
                logger.info("=== Phase 1: Shared Expert Training ===")
                self._train_phase(
                    train_loader, val_loader,
                    max_epochs=self.config.phase1.epochs,
                    phase_name="phase1",
                )

                # Phase 2: Task heads
                logger.info("=== Phase 2: Task Head Fine-tuning ===")
                self._setup_phase2()
                try:
                    self._train_phase(
                        train_loader, val_loader,
                        max_epochs=self.config.phase2.epochs,
                        phase_name="phase2",
                    )
                finally:
                    self._teardown_phase2()

            elif phase == "phase1":
                self._train_phase(
                    train_loader, val_loader,
                    max_epochs=self.config.phase1.epochs,
                    phase_name="phase1",
                )

            elif phase == "phase2":
                self._setup_phase2()
                try:
                    self._train_phase(
                        train_loader, val_loader,
                        max_epochs=self.config.phase2.epochs,
                        phase_name="phase2",
                    )
                finally:
                    self._teardown_phase2()

            else:
                raise ValueError(f"Unknown phase '{phase}'. Use 'full', 'phase1', or 'phase2'.")

        finally:
            self.callbacks.on_train_end(self._make_state())

        # Persist early stopping reason if triggered
        self.early_stop_info: Optional[Dict[str, Any]] = None
        for cb in self.callbacks.callbacks:
            if hasattr(cb, "stop_reason") and cb.stop_reason:
                self.early_stop_info = {
                    "reason": cb.stop_reason,
                    "epoch": cb.stop_epoch,
                    "best_val_loss": getattr(cb, "best_val_loss", None),
                    "best_avg_auc": getattr(cb, "best_avg_auc", None),
                }
                break

        if hasattr(self, '_audit_store') and self._audit_store:
            self._audit_store.log_event("training", {
                "pk": phase,
                "phase": phase,
                "total_epochs": self.current_epoch,
                "best_val_loss": self.best_val_loss,
                "best_val_metrics": self.best_val_metrics,
            })

        return {"best_val_loss": self.best_val_loss, **self.best_val_metrics}

    # ------------------------------------------------------------------
    # Class weight auto-computation
    # ------------------------------------------------------------------

    def _auto_compute_class_weights(self, train_loader: DataLoader) -> None:
        """Auto-compute balanced class weights for multiclass tasks.

        Iterates through the first ``max_batches`` batches of the training
        DataLoader, collects label counts per multiclass task, and computes
        inverse-frequency class weights using the sklearn balanced formula:

            w_c = n_samples / (n_classes * count_c)

        The computed weights are stored as ``Dict[str, torch.Tensor]`` and
        passed to the model via ``model.set_class_weights()``.

        Weights are clamped to [0.1, 10.0] to prevent extreme values from
        destabilizing training.
        """
        # Identify multiclass tasks
        multiclass_tasks: Dict[str, int] = {}
        for task_name in self.model.task_names:
            task_type = self.model.config.get_task_type(task_name)
            if task_type == "multiclass":
                n_classes = self.model.config.get_task_output_dim(task_name)
                if n_classes > 1:
                    multiclass_tasks[task_name] = n_classes

        if not multiclass_tasks:
            return

        logger.info(
            "Auto-computing class weights for: %s", list(multiclass_tasks.keys()),
        )

        # Collect label counts (sample up to 50 batches)
        max_batches = 50
        counters: Dict[str, Counter] = {name: Counter() for name in multiclass_tasks}

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= max_batches:
                break

            # Support PLEInput, dict, and object with .targets
            if hasattr(batch, "targets"):
                targets = batch.targets or {}
            elif isinstance(batch, dict):
                targets = batch.get("targets", {})
            else:
                continue

            for task_name in multiclass_tasks:
                if task_name in targets:
                    labels = targets[task_name].cpu().float().numpy().flatten().astype(int)
                    counters[task_name].update(labels.tolist())

        # Compute inverse-frequency weights
        class_weights: Dict[str, torch.Tensor] = {}
        cw_min, cw_max = 0.1, 10.0

        for task_name, counter in counters.items():
            if not counter:
                logger.warning(
                    "%s: no label samples found, skipping class weights", task_name,
                )
                continue

            n_classes = multiclass_tasks[task_name]
            n_samples = sum(counter.values())

            weights = []
            for c in range(n_classes):
                count_c = counter.get(c, 1)  # avoid division by zero
                w = n_samples / (n_classes * count_c)
                w = max(cw_min, min(cw_max, round(w, 4)))
                weights.append(w)

            weight_tensor = torch.tensor(weights, dtype=torch.float32)
            class_weights[task_name] = weight_tensor.to(self.device)

            logger.info(
                "%s: class_weights computed (n_samples=%d, n_classes=%d, "
                "range=[%.4f, %.4f])",
                task_name, n_samples, n_classes,
                min(weights), max(weights),
            )

        if class_weights:
            self.model.set_class_weights(class_weights)

    # ------------------------------------------------------------------
    # Pos-weight auto-computation for binary tasks
    # ------------------------------------------------------------------

    def _auto_compute_pos_weights(self, train_loader: DataLoader) -> None:
        """Auto-compute pos_weight for binary tasks to handle class imbalance.

        For each binary task, computes ``pos_weight = n_negative / n_positive``
        and clamps to [1.0, 50.0] to prevent extreme values.  The result is
        passed to the model via ``model.set_pos_weights()``.
        """
        binary_tasks: List[str] = []
        for task_name in self.model.task_names:
            task_type = self.model.config.get_task_type(task_name)
            if task_type == "binary":
                binary_tasks.append(task_name)

        if not binary_tasks:
            return

        logger.info(
            "Auto-computing pos_weights for binary tasks: %s", binary_tasks,
        )

        # Collect positive / negative counts (sample up to 50 batches)
        max_batches = 50
        pos_counts: Dict[str, int] = {name: 0 for name in binary_tasks}
        total_counts: Dict[str, int] = {name: 0 for name in binary_tasks}

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= max_batches:
                break

            if hasattr(batch, "targets"):
                targets = batch.targets or {}
            elif isinstance(batch, dict):
                targets = batch.get("targets", {})
            else:
                continue

            for task_name in binary_tasks:
                if task_name in targets:
                    labels = targets[task_name].cpu().float().flatten()
                    n = labels.numel()
                    n_pos = (labels > 0.5).sum().item()
                    pos_counts[task_name] += int(n_pos)
                    total_counts[task_name] += n

        # Compute pos_weight = n_negative / n_positive, clamped
        pw_min, pw_max = 1.0, 50.0
        pos_weights: Dict[str, torch.Tensor] = {}

        for task_name in binary_tasks:
            n_total = total_counts[task_name]
            n_pos = pos_counts[task_name]
            if n_pos == 0 or n_total == 0:
                logger.warning(
                    "%s: no positive samples found, using pos_weight=%.1f",
                    task_name, pw_max,
                )
                pw = pw_max
            else:
                n_neg = n_total - n_pos
                pw = n_neg / n_pos
                pw = max(pw_min, min(pw_max, pw))

            pos_weights[task_name] = torch.tensor(pw, dtype=torch.float32).to(self.device)
            logger.info(
                "%s: pos_weight=%.4f (n_pos=%d, n_total=%d, ratio=%.4f%%)",
                task_name, pw, n_pos, n_total,
                100.0 * n_pos / max(n_total, 1),
            )

        if pos_weights:
            self.model.set_pos_weights(pos_weights)

    # ------------------------------------------------------------------
    # Phase setup / teardown
    # ------------------------------------------------------------------

    def _setup_phase2(self) -> None:
        """Prepare for Phase 2: freeze shared, reset optimizer/scheduler."""
        p2 = self.config.phase2

        # Reset best tracking so Phase 2 is evaluated independently
        self.best_val_loss = float("inf")
        self.best_val_metrics = {}
        logger.info("Reset best_val_loss for Phase 2 (independent evaluation).")

        # Reset early stopping state
        for cb in self.callbacks.callbacks:
            if hasattr(cb, "counter"):
                cb.counter = 0
            if hasattr(cb, "best_score"):
                cb.best_score = None

        # Freeze shared experts
        if p2.freeze_shared_experts:
            self._freeze_module_group("extraction_layers")
            logger.info("Shared extraction layers frozen for Phase 2.")

        # Freeze CGC attention
        if p2.freeze_cgc and hasattr(self.model, "cgc_attention"):
            if self.model.cgc_attention is not None:
                for param in self.model.cgc_attention.parameters():
                    param.requires_grad = False
                logger.info("CGC attention frozen for Phase 2.")

        # Disable adaTT (meaningless when shared experts are frozen)
        if p2.disable_adatt and hasattr(self.model, "adatt") and self.model.adatt is not None:
            if hasattr(self.model.adatt, "disable"):
                self.model.adatt.disable()
            else:
                for param in self.model.adatt.parameters():
                    param.requires_grad = False
            logger.info("adaTT disabled for Phase 2.")

        # Recreate optimizer (frozen params are excluded)
        self.optimizer = self._create_optimizer()
        logger.info("Optimizer recreated for Phase 2.")

        # Recreate scheduler with Phase 2 settings
        sched_cfg = self.config.scheduler
        self.scheduler = self._create_scheduler(
            warmup_epochs=sched_cfg.phase2_warmup_epochs,
            cosine_t0=sched_cfg.phase2_cosine_t0,
        )
        if self._lr_callback is not None:
            self._lr_callback.set_scheduler(self.scheduler)
        logger.info(
            "Scheduler recreated for Phase 2 (warmup=%d, T_0=%d).",
            sched_cfg.phase2_warmup_epochs, sched_cfg.phase2_cosine_t0,
        )

        # Recreate GradScaler
        if self.config.amp.enabled:
            self.scaler = self._create_grad_scaler()

        # Reset early stopping
        if self._es_callback is not None:
            self._es_callback.reset()

    def _teardown_phase2(self) -> None:
        """Restore model to full-trainable state after Phase 2."""
        # Unfreeze shared experts
        if self.config.phase2.freeze_shared_experts:
            self._unfreeze_module_group("extraction_layers")

        # Unfreeze CGC
        if self.config.phase2.freeze_cgc and hasattr(self.model, "cgc_attention"):
            if self.model.cgc_attention is not None:
                for param in self.model.cgc_attention.parameters():
                    param.requires_grad = True

        # Re-enable adaTT
        if (self.config.phase2.disable_adatt
                and hasattr(self.model, "adatt")
                and self.model.adatt is not None):
            if hasattr(self.model.adatt, "enable"):
                self.model.adatt.enable()
            else:
                for param in self.model.adatt.parameters():
                    param.requires_grad = True

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train_phase(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        max_epochs: int,
        phase_name: str,
    ) -> None:
        """Run one training phase (Phase 1 or Phase 2)."""
        phase_state = self._make_state(phase_name=phase_name)
        self.callbacks.on_phase_begin(phase_state)

        for epoch_idx in range(max_epochs):
            self.current_epoch += 1

            epoch_state = self._make_state(phase_name=phase_name)
            self.callbacks.on_epoch_begin(epoch_state)

            # Train one epoch
            train_loss = self._train_epoch(train_loader, phase_name)

            # Validate
            val_loss = None
            val_metrics: Dict[str, float] = {}
            if val_loader is not None:
                val_loss, val_metrics = self._validate(val_loader)

            # Update model epoch state (adaTT, loss weighting)
            self.model.set_epoch(self.current_epoch)
            if hasattr(self.model, "update_loss_weights") and val_loss is not None:
                # Pass latest task losses if available
                pass  # Loss weight update happens in _train_epoch per-batch

            # Track best
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_metrics = val_metrics

            # Epoch end callbacks (includes scheduler step, early stopping, checkpointing)
            epoch_end_state = self._make_state(
                phase_name=phase_name,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
            )
            should_stop = self.callbacks.on_epoch_end(epoch_end_state)

            # Record per-task metrics for monitoring (val_metrics already has them)
            if val_metrics:
                per_task_epoch = {}
                for key, val in val_metrics.items():
                    if isinstance(val, (int, float)) and math.isfinite(val):
                        per_task_epoch[key] = round(val, 4)
                if per_task_epoch:
                    epoch_end_state["per_task_metrics"] = per_task_epoch

            # Logging
            self._log_epoch(epoch_idx, train_loss, val_loss, phase_name, val_metrics)

            if should_stop:
                logger.info(
                    "Training stopped at epoch %d (phase: %s).",
                    self.current_epoch, phase_name,
                )
                break

        self.callbacks.on_phase_end(self._make_state(phase_name=phase_name))

    def _train_epoch(self, train_loader: DataLoader, phase_name: str) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_start = time.time()
        accum_steps = self.config.gradient.accumulation_steps
        device_type = getattr(self.device, "type", "cuda")

        self.optimizer.zero_grad()
        _scaler_backward_count = 0  # Track backward passes for GradScaler guard
        logger.info("[%s] Starting epoch loop, loader has %d batches", phase_name, len(train_loader))
        import sys; sys.stdout.flush(); sys.stderr.flush()

        _total_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            # Flag last batch of epoch for adaTT gradient extraction
            self.model._is_epoch_end_step = (batch_idx == _total_batches - 1)

            if batch_idx == 0:
                logger.info("[%s] First batch received, type=%s", phase_name, type(batch).__name__)
                sys.stdout.flush(); sys.stderr.flush()
            inputs = self._prepare_inputs(batch)
            if batch_idx == 0:
                logger.info("[%s] First batch prepared as PLEInput, features=%s", phase_name, inputs.features.shape if inputs.features is not None else None)
                sys.stdout.flush(); sys.stderr.flush()

            # Forward pass
            try:
                if self.config.amp.enabled:
                    with (autocast(device_type=device_type) if _AMP_NEW_API else autocast()):
                        outputs: PLEOutput = self.model(inputs, compute_loss=True)
                    # Loss scaling outside autocast to avoid FP16 overflow
                    loss = outputs.total_loss.float() / accum_steps
                else:
                    outputs = self.model(inputs, compute_loss=True)
                    loss = outputs.total_loss / accum_steps
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(
                        "CUDA OOM at batch %d. Skipping batch.", batch_idx,
                    )
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    continue
                raise

            if batch_idx == 0:
                logger.info("[%s] First forward pass done, total_loss=%s", phase_name, outputs.total_loss)
                sys.stdout.flush(); sys.stderr.flush()

            # NaN/Inf check before backward
            loss_val = outputs.total_loss.item()
            if not math.isfinite(loss_val):
                logger.warning(
                    "[%s] Batch %d: NaN/Inf loss (%.4f), skipping.",
                    phase_name, batch_idx, loss_val,
                )
                self.optimizer.zero_grad()
                continue

            # Backward pass
            if self.config.amp.enabled and self.scaler is not None:
                self.scaler.scale(loss).backward()
                _scaler_backward_count += 1

                if (batch_idx + 1) % accum_steps == 0:
                    if _scaler_backward_count > 0:
                        try:
                            self.scaler.unscale_(self.optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.gradient.clip_norm,
                            )
                            if grad_norm > self.config.gradient.clip_norm * 10:
                                logger.warning("Gradient explosion: norm=%.2f (clip=%.2f) at batch %d",
                                                grad_norm, self.config.gradient.clip_norm, batch_idx)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            # Cap scale to prevent FP16 overflow
                            _max_scale = getattr(self, "_grad_scaler_max_scale", 4096.0)
                            if self.scaler.get_scale() > _max_scale:
                                self.scaler._scale = torch.tensor(_max_scale, device=self.device)
                        except (AssertionError, RuntimeError) as e:
                            logger.warning("GradScaler step failed: %s. Skipping.", e)
                    self.optimizer.zero_grad()
                    _scaler_backward_count = 0
            else:
                loss.backward()

                if (batch_idx + 1) % accum_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient.clip_norm,
                    )
                    if grad_norm > self.config.gradient.clip_norm * 10:
                        logger.warning("Gradient explosion: norm=%.2f (clip=%.2f) at batch %d",
                                        grad_norm, self.config.gradient.clip_norm, batch_idx)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss_val
            num_batches += 1
            self.global_step += 1

            # Sync global step with model (for adaTT gradient interval)
            self.model.set_global_step(self.global_step)

            # Update loss weights
            if outputs.task_losses is not None:
                self.model.update_loss_weights(
                    outputs.task_losses,
                    epoch=self.current_epoch,
                )

            # Step-level callback
            step_state = self._make_state(
                phase_name=phase_name,
                train_loss=loss_val,
                task_losses=outputs.task_losses,
                aux_losses=outputs.aux_losses,
            )
            self.callbacks.on_step_end(step_state)

        # Flush remaining accumulated gradients (tail batches not aligned to accum_steps)
        if _scaler_backward_count > 0 and self.config.amp.enabled and self.scaler is not None:
            try:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient.clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                _max_scale = getattr(self, "_grad_scaler_max_scale", 4096.0)
                if self.scaler.get_scale() > _max_scale:
                    self.scaler._scale = torch.tensor(_max_scale, device=self.device)
            except (AssertionError, RuntimeError) as e:
                logger.warning("GradScaler tail flush failed: %s. Skipping.", e)
            self.optimizer.zero_grad()
        elif _scaler_backward_count == 0 and num_batches == 0 and self.config.amp.enabled and self.scaler is not None:
            # All batches were NaN-skipped: no backward pass happened.
            # Do NOT call scaler.step() — it would assert "No inf checks recorded".
            logger.warning(
                "[%s] Epoch %d: ALL batches produced NaN/Inf loss. "
                "Skipping scaler.step() to avoid GradScaler assertion.",
                phase_name, self.current_epoch,
            )

        elapsed = time.time() - epoch_start
        avg_loss = total_loss / max(num_batches, 1)

        # VRAM diagnostics
        _vram_info = ""
        if self.device.type == "cuda":
            _alloc = torch.cuda.memory_allocated() / 1e6
            _reserved = torch.cuda.memory_reserved() / 1e6
            _peak = torch.cuda.max_memory_allocated() / 1e6
            _vram_info = f", VRAM: alloc={_alloc:.0f}MB reserved={_reserved:.0f}MB peak={_peak:.0f}MB"
            torch.cuda.reset_peak_memory_stats()

        logger.info(
            "[%s] Epoch %d: avg_loss=%.6f, batches=%d, time=%.1fs%s",
            phase_name, self.current_epoch, avg_loss, num_batches, elapsed, _vram_info,
        )
        return avg_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Run validation and compute per-task metrics.

        Returns:
            ``(avg_loss, metrics_dict)`` where ``metrics_dict`` contains
            per-task and aggregate metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_predictions: Dict[str, List[torch.Tensor]] = {}
        all_labels: Dict[str, List[torch.Tensor]] = {}
        device_type = getattr(self.device, "type", "cuda")

        for batch in val_loader:
            inputs = self._prepare_inputs(batch)

            if self.config.amp.enabled:
                with (autocast(device_type=device_type) if _AMP_NEW_API else autocast()):
                    outputs = self.model(inputs, compute_loss=True)
            else:
                outputs = self.model(inputs, compute_loss=True)

            if outputs.total_loss is not None:
                total_loss += outputs.total_loss.item()
                num_batches += 1

            # Collect predictions and labels for metric computation
            if outputs.predictions:
                for task_name, pred in outputs.predictions.items():
                    if pred is not None:
                        all_predictions.setdefault(task_name, []).append(pred.cpu())

            if inputs.targets:
                for task_name, label in inputs.targets.items():
                    if label is not None:
                        all_labels.setdefault(task_name, []).append(label.cpu())

        self.model.train()

        avg_loss = total_loss / max(num_batches, 1)
        metrics = self._compute_val_metrics(all_predictions, all_labels)
        metrics["val_loss"] = avg_loss

        return avg_loss, metrics

    @staticmethod
    def _compute_topk_metrics(
        logits: "np.ndarray",
        labels: "np.ndarray",
        k_values: List[int],
    ) -> Dict[str, float]:
        """Compute top-K ranking metrics for single-label multiclass.

        Assumes single-label ground truth (one relevant item per sample),
        which is the case for nba_primary and next_mcc tasks.

        Args:
            logits: Raw class scores, shape ``(n_samples, n_classes)``.
            labels: Integer class indices, shape ``(n_samples,)``.
            k_values: List of K values to evaluate (e.g. ``[1, 3, 5]``).

        Returns:
            Dict with the following keys for each K:

            * ``accuracy@K``  — fraction of samples whose true label is in top-K
            * ``hit@K``       — same as accuracy@K; exposed as a distinct key for
                                recsys reporting conventions
            * ``recall@K``    — same as hit@K for single-label tasks (|relevant|=1)
            * ``precision@K`` — hit@K / K  (|top-K ∩ relevant| / K)
            * ``ndcg@K``      — binary-relevance NDCG (IDCG=1 per sample)
        """
        import numpy as np  # noqa: PLC0415

        metrics: Dict[str, float] = {}
        if len(logits) == 0 or logits.ndim < 2:
            return metrics

        n_samples, n_classes = logits.shape
        max_k = min(max(k_values), n_classes)

        # Descending argsort; materialise only the columns we need.
        topk_preds = np.argsort(-logits, axis=1)[:, :max_k]  # (n, max_k)
        labels_col = labels.reshape(-1, 1)  # (n, 1)

        # Binary match matrix: matches[i, j] = 1 iff topk_preds[i, j] == labels[i]
        matches = (topk_preds == labels_col).astype(np.float32)  # (n, max_k)

        for k in k_values:
            k_eff = min(k, n_classes)
            # hit_mean = fraction of samples that have the true label in their top-K
            hit = matches[:, :k_eff].sum(axis=1)  # (n,) — 0 or 1 (single-label)
            hit_mean = float(hit.mean())

            metrics[f"accuracy@{k}"] = hit_mean          # backward-compat key
            metrics[f"hit@{k}"] = hit_mean               # recsys alias
            metrics[f"recall@{k}"] = hit_mean            # single-label: recall == hit
            metrics[f"precision@{k}"] = hit_mean / k     # single-label: precision == hit/K

            # NDCG@K: binary relevance; IDCG = 1 (single relevant item per sample)
            # DCG@K = sum_{i=0}^{k_eff-1} match[i] / log2(i + 2)
            discounts = 1.0 / np.log2(
                np.arange(2, k_eff + 2, dtype=np.float32)
            )  # (k_eff,)
            dcg = (matches[:, :k_eff] * discounts).sum(axis=1)  # (n,)
            metrics[f"ndcg@{k}"] = float(dcg.mean())

        return metrics

    def _compute_val_metrics(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        labels: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, float]:
        """Compute per-task validation metrics.

        Detects task types from ``PLEConfig.task_overrides`` and computes
        appropriate metrics:

        * Binary: AUC (ROC)
        * Multiclass: Accuracy, Macro F1, and optionally top-K accuracy /
          NDCG@K for tasks that declare ``topk_k`` in their task definition.
        * Regression: MAE, RMSE, R-squared
        """
        metrics: Dict[str, float] = {}

        try:
            import numpy as np
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                mean_absolute_error,
                r2_score,
                roc_auc_score,
            )
        except ImportError:
            logger.debug("sklearn not available, skipping validation metrics.")
            return metrics

        # Per-type accumulators: keyed by task type
        binary_aucs: List[float] = []
        binary_task_names_seen: List[str] = []
        multiclass_accuracies: List[float] = []
        multiclass_f1s: List[float] = []
        multiclass_task_names_seen: List[str] = []
        # top-K accumulators: only tasks with topk_k configured contribute
        topk_ndcg3_values: List[float] = []  # for avg_ndcg@3 across rec tasks
        regression_maes: List[float] = []
        regression_task_names_seen: List[str] = []

        for task_name, pred_list in predictions.items():
            label_list = labels.get(task_name)
            if not pred_list or not label_list:
                continue

            preds_cat = torch.cat(pred_list)
            labs_cat = torch.cat(label_list)

            if torch.isnan(preds_cat).any() or torch.isnan(labs_cat).any():
                logger.warning("NaN in %s predictions/labels, skipping.", task_name)
                continue

            preds_np = preds_cat.float().numpy()
            labs_np = labs_cat.float().numpy()

            # Apply per-task validation mask if configured
            if self.task_val_masks is not None and task_name in self.task_val_masks:
                _mask = self.task_val_masks[task_name]
                # Mask length may differ from concat length if val set size
                # doesn't match (e.g. drop_last in dataloader).  Truncate mask.
                _n = min(len(_mask), len(preds_np))
                _mask_t = _mask[:_n]
                preds_np = preds_np[_mask_t]
                labs_np = labs_np[_mask_t]
                if len(preds_np) == 0:
                    logger.debug("task_val_mask[%s]: 0 samples after masking, skipping", task_name)
                    continue

            task_type = self.model.config.get_task_type(task_name)

            try:
                if task_type == "regression":
                    preds_flat = preds_np.flatten()
                    labs_flat = labs_np.flatten()
                    mae = float(mean_absolute_error(labs_flat, preds_flat))
                    rmse = float(np.sqrt(np.mean((preds_flat - labs_flat) ** 2)))
                    r2 = float(r2_score(labs_flat, preds_flat)) if np.var(labs_flat) > 1e-10 else 0.0
                    metrics[f"{task_name}_mae"] = mae
                    metrics[f"{task_name}_rmse"] = rmse
                    metrics[f"{task_name}_r2"] = r2
                    regression_maes.append(mae)
                    regression_task_names_seen.append(task_name)

                elif task_type == "multiclass":
                    pred_classes = np.argmax(preds_np, axis=-1)
                    true_classes = labs_np.flatten().astype(int)
                    valid = true_classes >= 0
                    if valid.sum() < 2:
                        continue
                    acc = float(accuracy_score(true_classes[valid], pred_classes[valid]))
                    f1 = float(f1_score(
                        true_classes[valid], pred_classes[valid],
                        average="macro", zero_division=0,
                    ))
                    metrics[f"{task_name}_accuracy"] = acc
                    metrics[f"{task_name}_f1_macro"] = f1
                    multiclass_accuracies.append(acc)
                    multiclass_f1s.append(f1)
                    multiclass_task_names_seen.append(task_name)

                    # --- Top-K metrics (opt-in via topk_k in task config) ---
                    k_values = self.model.config.get_task_topk_k(task_name)
                    if k_values is not None:
                        topk_m = self._compute_topk_metrics(
                            preds_np[valid], true_classes[valid], k_values
                        )
                        for metric_name, metric_val in topk_m.items():
                            metrics[f"{task_name}_{metric_name}"] = metric_val
                        # Accumulate ndcg@3 for avg across recommendation tasks
                        ndcg3_key = "ndcg@3"
                        if ndcg3_key in topk_m:
                            topk_ndcg3_values.append(topk_m[ndcg3_key])

                else:  # binary
                    unique = set(labs_np.flatten().tolist())
                    if unique <= {0.0, 1.0} and len(unique) == 2:
                        auc = float(roc_auc_score(labs_np, preds_np))
                        metrics[f"{task_name}_auc"] = auc
                        binary_aucs.append(auc)
                        binary_task_names_seen.append(task_name)

            except Exception as e:
                logger.debug("Metric computation failed for %s: %s", task_name, e)

        # --- Task-type-grouped aggregation ---
        # Binary tasks: primary metric = AUC
        if binary_aucs:
            metrics["avg_auc"] = sum(binary_aucs) / len(binary_aucs)
        # Multiclass tasks: primary metrics = accuracy, macro-F1
        if multiclass_accuracies:
            metrics["avg_accuracy"] = sum(multiclass_accuracies) / len(multiclass_accuracies)
        if multiclass_f1s:
            metrics["avg_f1_macro"] = sum(multiclass_f1s) / len(multiclass_f1s)
        # Recommendation tasks: avg NDCG@3 across tasks that declared topk_k
        if topk_ndcg3_values:
            metrics["avg_ndcg@3"] = sum(topk_ndcg3_values) / len(topk_ndcg3_values)
        # Regression tasks: primary metric = MAE
        if regression_maes:
            metrics["avg_mae"] = sum(regression_maes) / len(regression_maes)

        # Task-type count and name summary keys
        metrics["n_binary_tasks"] = float(len(binary_task_names_seen))
        metrics["n_multiclass_tasks"] = float(len(multiclass_task_names_seen))
        metrics["n_regression_tasks"] = float(len(regression_task_names_seen))
        # Store names as a pipe-delimited string so they fit in the flat metrics dict
        metrics["binary_task_names"] = "|".join(binary_task_names_seen)
        metrics["multiclass_task_names"] = "|".join(multiclass_task_names_seen)
        metrics["regression_task_names"] = "|".join(regression_task_names_seen)

        return metrics

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _prepare_inputs(self, batch: Any) -> PLEInput:
        """Convert a batch to :class:`PLEInput` and move to device.

        Supports three input formats:

        1. ``PLEInput`` -- moved to device directly.
        2. Object with ``.to_ple_input()`` method -- converted and moved.
        3. ``Dict[str, Tensor]`` -- mapped to ``PLEInput`` fields.
        """
        if isinstance(batch, PLEInput):
            return batch.to(self.device)

        if hasattr(batch, "to_ple_input"):
            return batch.to_ple_input().to(self.device)

        # Dict fallback
        if isinstance(batch, dict):
            return PLEInput(
                features=batch["features"],
                cluster_ids=batch.get("cluster_ids"),
                cluster_probs=batch.get("cluster_probs"),
                targets=batch.get("targets"),
            ).to(self.device)

        raise TypeError(
            f"Unsupported batch type: {type(batch)}. "
            f"Expected PLEInput, dict, or object with to_ple_input()."
        )

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def _freeze_module_group(self, attr_name: str) -> None:
        """Freeze all parameters in a named module attribute."""
        module = getattr(self.model, attr_name, None)
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze_module_group(self, attr_name: str) -> None:
        """Unfreeze all parameters in a named module attribute."""
        module = getattr(self.model, attr_name, None)
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_epoch(
        self,
        epoch_idx: int,
        train_loss: float,
        val_loss: Optional[float],
        phase_name: str,
        val_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log epoch summary."""
        parts = [f"Epoch {self.current_epoch}: train_loss={train_loss:.6f}"]
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.6f}")

        if val_metrics:
            for key in ("avg_auc", "avg_accuracy", "avg_f1_macro", "avg_ndcg@3", "avg_mae"):
                val = val_metrics.get(key)
                if val is not None:
                    parts.append(f"{key}={val:.4f}")

        logger.info("  ".join(parts))

        # Log adaTT affinity matrix every 5 epochs
        if self.current_epoch % 5 == 0 and hasattr(self.model, "adatt") and self.model.adatt is not None:
            try:
                affinity = self.model.adatt.get_transfer_matrix()
                if affinity is not None:
                    task_names_adatt = self.model.task_names
                    n = len(task_names_adatt)
                    pairs = []
                    for i in range(n):
                        for j in range(n):
                            if i != j:
                                pairs.append((task_names_adatt[i], task_names_adatt[j], affinity[i, j].item()))
                    pairs.sort(key=lambda x: -x[2])
                    top3 = pairs[:3]
                    bot3 = pairs[-3:]
                    logger.info("=== adaTT Affinity (epoch %d) ===", self.current_epoch)
                    for src, tgt, strength in top3:
                        logger.info("  Top transfer: %s -> %s = %.4f", src, tgt, strength)
                    for src, tgt, strength in bot3:
                        logger.info("  Bottom transfer: %s -> %s = %.4f", src, tgt, strength)
            except Exception as e:
                logger.debug("adaTT affinity logging failed: %s", e)

        # Record epoch history for eval_metrics.json
        epoch_record = {
            "epoch": self.current_epoch,
            "phase": phase_name,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6) if val_loss is not None else None,
            "global_step": self.global_step,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        if val_metrics:
            for key in ("avg_auc", "avg_accuracy", "avg_f1_macro", "avg_ndcg@3", "avg_mae"):
                if key in val_metrics:
                    epoch_record[key] = round(val_metrics[key], 4)

            # Per-task metrics (e.g. "churn_signal_auc", "has_nba_auc", etc.)
            per_task_epoch = {}
            for key, val in val_metrics.items():
                if isinstance(val, (int, float)) and math.isfinite(val):
                    per_task_epoch[key] = round(val, 4)
            if per_task_epoch:
                epoch_record["per_task_metrics"] = per_task_epoch

        # Loss weights (uncertainty weighting log_vars)
        weights = self.model.get_loss_weights()
        if weights:
            epoch_record["loss_weights"] = {k: round(v, 4) for k, v in weights.items()}

        # adaTT affinity matrix (if available)
        if hasattr(self.model, "adatt") and self.model.adatt is not None:
            try:
                affinity = self.model.adatt.get_transfer_matrix()
                if affinity is not None:
                    # Store as task-pair summary (top 5 strongest positive transfers)
                    task_names = self.model.task_names
                    n = len(task_names)
                    pairs = []
                    for i in range(n):
                        for j in range(n):
                            if i != j:
                                pairs.append((task_names[i], task_names[j], affinity[i, j].item()))
                    pairs.sort(key=lambda x: -x[2])
                    epoch_record["adatt_top_transfers"] = [
                        {"from": p[0], "to": p[1], "strength": round(p[2], 4)}
                        for p in pairs[:5]
                    ]
                    epoch_record["adatt_mean_affinity"] = round(affinity.mean().item(), 4)
            except Exception:
                pass

        # GPU memory tracking
        if torch.cuda.is_available():
            epoch_record["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
            epoch_record["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 1)
            epoch_record["gpu_memory_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

        # Gradient norm (sampled from last batch)
        try:
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            epoch_record["gradient_norm"] = round(total_norm ** 0.5, 4)
        except Exception:
            pass

        self.epoch_history.append(epoch_record)

        # Log loss weights for monitoring
        if self.config.logging.log_loss_weights and weights:
            w_str = ", ".join(f"{k}={v:.4f}" for k, v in weights.items())
            logger.debug("  loss_weights: %s", w_str)

    # ------------------------------------------------------------------
    # Checkpoint load
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load a checkpoint and restore trainer state.

        Args:
            path: Path to the checkpoint ``.pt`` file.

        Returns:
            The full checkpoint dict.
        """
        checkpoint = CheckpointCallback.load_checkpoint(
            str(path),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        return checkpoint

    # ------------------------------------------------------------------
    # State dict for callbacks
    # ------------------------------------------------------------------

    def _make_state(self, **extras: Any) -> Dict[str, Any]:
        """Build a state dict for callback dispatch."""
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "config": self.config,
            "tracker": self.tracker,
            "device": self.device,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        state.update(extras)
        return state
