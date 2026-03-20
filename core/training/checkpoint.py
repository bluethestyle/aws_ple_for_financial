"""
Checkpoint management with S3 integration and best-model tracking.

Saves self-contained checkpoint files that include the model state dict,
optimizer state, scheduler state, training config, and metadata (epoch,
global step, best metric, etc.).  Integrates with S3 for upload/download
so that SageMaker Spot instance interruptions can resume seamlessly.

Usage::

    ckpt_mgr = CheckpointManager(
        local_dir="/opt/ml/checkpoints",
        s3_bucket="my-bucket",
        s3_prefix="project/checkpoints/",
        max_keep=3,
    )

    # Save
    ckpt_mgr.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=5,
        global_step=12000,
        metrics={"val_loss": 0.42, "val_auc": 0.88},
        config=config_dict,
    )

    # Load latest
    state = ckpt_mgr.load_latest(model, optimizer, scheduler)

    # Best model tracking
    if ckpt_mgr.is_best(metric_value=0.90, metric_name="val_auc"):
        ckpt_mgr.save_best(model, config=config_dict)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMeta:
    """Metadata stored alongside each checkpoint."""

    epoch: int = 0
    global_step: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    best_metric_name: str = "val_loss"
    best_metric_value: float = float("inf")
    timestamp: str = ""


class CheckpointManager:
    """Manage model checkpoints with local storage, S3 sync, and cleanup.

    Parameters
    ----------
    local_dir : str or Path
        Local directory for checkpoint files.
    s3_bucket : str, optional
        S3 bucket for remote backup.  If empty, S3 sync is disabled.
    s3_prefix : str, optional
        S3 key prefix for checkpoint files.
    max_keep : int
        Maximum number of checkpoints to retain (oldest pruned first).
        Does not count the ``best`` checkpoint.
    region : str
        AWS region for S3 operations.
    """

    CHECKPOINT_FILENAME = "checkpoint_epoch{epoch:04d}.pt"
    BEST_FILENAME = "best_model.pt"
    META_FILENAME = "checkpoint_meta.json"

    def __init__(
        self,
        local_dir: Union[str, Path] = "/opt/ml/checkpoints",
        s3_bucket: str = "",
        s3_prefix: str = "",
        max_keep: int = 3,
        region: str = "ap-northeast-2",
    ):
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.strip("/")
        self.max_keep = max_keep
        self.region = region

        self._s3_client = None  # lazy init
        self._best_value: Optional[float] = None
        self._best_higher_is_better: bool = False
        self._saved_checkpoints: List[Path] = []

        # Discover existing checkpoints
        self._scan_existing()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a checkpoint.

        The checkpoint is a self-contained dict with:

        * ``model_state_dict``
        * ``optimizer_state_dict`` (if provided)
        * ``scheduler_state_dict`` (if provided)
        * ``epoch``, ``global_step``
        * ``metrics``
        * ``config``

        Parameters
        ----------
        model : torch.nn.Module
        optimizer : torch.optim.Optimizer, optional
        scheduler : optional
            Any object with a ``state_dict()`` method.
        epoch : int
        global_step : int
        metrics : dict[str, float], optional
        config : dict, optional
            Model/pipeline configuration for reproducibility.

        Returns
        -------
        Path
            Local path of the saved checkpoint file.
        """
        metrics = metrics or {}
        filename = self.CHECKPOINT_FILENAME.format(epoch=epoch)
        path = self.local_dir / filename

        payload: Dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics,
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            payload["scheduler_state_dict"] = scheduler.state_dict()
        if config is not None:
            payload["config"] = config

        torch.save(payload, path)
        logger.info(f"Checkpoint saved: {path} (epoch={epoch}, step={global_step})")

        self._saved_checkpoints.append(path)

        # Upload to S3
        if self.s3_bucket:
            self._upload_to_s3(path, filename)

        # Prune old checkpoints
        self._prune()

        # Save meta
        self._save_meta(epoch, global_step, metrics)

        return path

    def save_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save as the best model checkpoint.

        Same format as :meth:`save` but stored under a fixed filename
        (``best_model.pt``) that is never pruned.

        Returns
        -------
        Path
        """
        metrics = metrics or {}
        path = self.local_dir / self.BEST_FILENAME

        payload: Dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics,
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            payload["scheduler_state_dict"] = scheduler.state_dict()
        if config is not None:
            payload["config"] = config

        torch.save(payload, path)
        logger.info(f"Best model saved: {path} (epoch={epoch})")

        if self.s3_bucket:
            self._upload_to_s3(path, self.BEST_FILENAME)

        return path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[Union[str, "torch.device"]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint.

        Tries the local directory first, then downloads from S3 if
        available.

        Parameters
        ----------
        model : torch.nn.Module
            Model to load weights into (in-place).
        optimizer : torch.optim.Optimizer, optional
        scheduler : optional
        map_location : str or device, optional
            Passed to ``torch.load``.

        Returns
        -------
        dict or None
            Checkpoint metadata (``epoch``, ``global_step``, ``metrics``,
            ``config``) or *None* if no checkpoint exists.
        """
        path = self._find_latest_local()

        if path is None and self.s3_bucket:
            path = self._download_latest_from_s3()

        if path is None:
            logger.info("No checkpoint found to resume from.")
            return None

        return self._load_from_path(path, model, optimizer, scheduler, map_location)

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[Union[str, "torch.device"]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load the best model checkpoint.

        Returns
        -------
        dict or None
        """
        path = self.local_dir / self.BEST_FILENAME

        if not path.exists() and self.s3_bucket:
            self._download_from_s3(self.BEST_FILENAME, path)

        if not path.exists():
            logger.info("No best-model checkpoint found.")
            return None

        return self._load_from_path(path, model, optimizer, scheduler, map_location)

    def load_from_path(
        self,
        path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[Union[str, "torch.device"]] = None,
    ) -> Dict[str, Any]:
        """Load a specific checkpoint file.

        Parameters
        ----------
        path : str or Path
            Checkpoint file path.

        Returns
        -------
        dict
            Checkpoint metadata.
        """
        return self._load_from_path(
            Path(path), model, optimizer, scheduler, map_location,
        )

    # ------------------------------------------------------------------
    # Best model tracking
    # ------------------------------------------------------------------

    def is_best(
        self,
        metric_value: float,
        metric_name: str = "val_loss",
        higher_is_better: bool = False,
    ) -> bool:
        """Check if *metric_value* beats the current best.

        Parameters
        ----------
        metric_value : float
            Current metric value.
        metric_name : str
            Metric name (for logging).
        higher_is_better : bool
            If True, larger values are better.

        Returns
        -------
        bool
        """
        self._best_higher_is_better = higher_is_better

        if self._best_value is None:
            self._best_value = metric_value
            logger.info(
                f"First {metric_name} recorded: {metric_value:.6f} (new best)"
            )
            return True

        if higher_is_better:
            improved = metric_value > self._best_value
        else:
            improved = metric_value < self._best_value

        if improved:
            old = self._best_value
            self._best_value = metric_value
            logger.info(
                f"New best {metric_name}: {metric_value:.6f} "
                f"(previous: {old:.6f})"
            )
            return True

        return False

    # ------------------------------------------------------------------
    # S3 integration
    # ------------------------------------------------------------------

    @property
    def _s3(self):
        """Lazy-initialized S3 client."""
        if self._s3_client is None:
            import boto3
            self._s3_client = boto3.client("s3", region_name=self.region)
        return self._s3_client

    def _upload_to_s3(self, local_path: Path, key_suffix: str) -> None:
        """Upload a file to S3."""
        key = f"{self.s3_prefix}/{key_suffix}" if self.s3_prefix else key_suffix
        try:
            self._s3.upload_file(str(local_path), self.s3_bucket, key)
            logger.debug(f"Uploaded {local_path} -> s3://{self.s3_bucket}/{key}")
        except Exception as e:
            logger.warning(f"S3 upload failed for {local_path}: {e}")

    def _download_from_s3(self, key_suffix: str, local_path: Path) -> bool:
        """Download a file from S3.

        Returns
        -------
        bool
            True if download succeeded.
        """
        key = f"{self.s3_prefix}/{key_suffix}" if self.s3_prefix else key_suffix
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._s3.download_file(self.s3_bucket, key, str(local_path))
            logger.debug(f"Downloaded s3://{self.s3_bucket}/{key} -> {local_path}")
            return True
        except Exception as e:
            logger.debug(f"S3 download failed for {key}: {e}")
            return False

    def _download_latest_from_s3(self) -> Optional[Path]:
        """Find and download the latest checkpoint from S3.

        Returns
        -------
        Path or None
        """
        try:
            prefix = f"{self.s3_prefix}/" if self.s3_prefix else ""
            response = self._s3.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=prefix,
            )
            contents = response.get("Contents", [])
            checkpoint_keys = [
                obj["Key"]
                for obj in contents
                if obj["Key"].endswith(".pt") and self.BEST_FILENAME not in obj["Key"]
            ]
            if not checkpoint_keys:
                return None

            # Sort by key name (contains epoch number)
            latest_key = sorted(checkpoint_keys)[-1]
            filename = latest_key.split("/")[-1]
            local_path = self.local_dir / filename
            self._s3.download_file(self.s3_bucket, latest_key, str(local_path))
            logger.info(f"Downloaded latest checkpoint from S3: {latest_key}")
            return local_path

        except Exception as e:
            logger.warning(f"Failed to list/download checkpoints from S3: {e}")
            return None

    # ------------------------------------------------------------------
    # Local file management
    # ------------------------------------------------------------------

    def _find_latest_local(self) -> Optional[Path]:
        """Find the most recent checkpoint file in *local_dir*."""
        candidates = sorted(
            self.local_dir.glob("checkpoint_epoch*.pt"),
            key=lambda p: p.name,
        )
        return candidates[-1] if candidates else None

    def _scan_existing(self) -> None:
        """Discover existing checkpoint files on startup."""
        self._saved_checkpoints = sorted(
            self.local_dir.glob("checkpoint_epoch*.pt"),
            key=lambda p: p.name,
        )
        if self._saved_checkpoints:
            logger.info(
                f"Found {len(self._saved_checkpoints)} existing checkpoint(s) "
                f"in {self.local_dir}"
            )

    def _prune(self) -> None:
        """Remove old checkpoints exceeding *max_keep*."""
        while len(self._saved_checkpoints) > self.max_keep:
            oldest = self._saved_checkpoints.pop(0)
            if oldest.exists():
                oldest.unlink()
                logger.debug(f"Pruned old checkpoint: {oldest}")

    def _save_meta(
        self,
        epoch: int,
        global_step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Persist checkpoint metadata to JSON for quick inspection."""
        from datetime import datetime

        meta = CheckpointMeta(
            epoch=epoch,
            global_step=global_step,
            metrics=metrics,
            best_metric_value=self._best_value if self._best_value is not None else float("inf"),
            timestamp=datetime.now().isoformat(),
        )
        meta_path = self.local_dir / self.META_FILENAME
        with open(meta_path, "w") as f:
            json.dump({
                "epoch": meta.epoch,
                "global_step": meta.global_step,
                "metrics": meta.metrics,
                "best_metric_value": meta.best_metric_value,
                "timestamp": meta.timestamp,
            }, f, indent=2)

        if self.s3_bucket:
            self._upload_to_s3(meta_path, self.META_FILENAME)

    def _load_from_path(
        self,
        path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        map_location: Optional[Union[str, "torch.device"]],
    ) -> Dict[str, Any]:
        """Load a checkpoint file and restore model/optimizer/scheduler state."""
        logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        # Restore model
        model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            if hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        meta = {
            "epoch": checkpoint.get("epoch", 0),
            "global_step": checkpoint.get("global_step", 0),
            "metrics": checkpoint.get("metrics", {}),
            "config": checkpoint.get("config"),
        }
        logger.info(
            f"Resumed from epoch={meta['epoch']}, step={meta['global_step']}"
        )
        return meta
