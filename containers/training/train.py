"""
SageMaker training entry point.

This script is executed inside the SageMaker training container.  It:

1. Parses SageMaker environment variables and hyperparameters.
2. Loads training / validation data from ``/opt/ml/input/data/``.
3. Builds the PLE model from the config.
4. Optionally resumes from checkpoint (Spot auto-resume).
5. Runs the training loop.
6. Saves the final model to ``/opt/ml/model/`` for SageMaker packaging.
7. Prints metrics to stdout for CloudWatch capture.

SageMaker environment variables
-------------------------------
SM_MODEL_DIR        /opt/ml/model
SM_CHANNEL_TRAIN    /opt/ml/input/data/train
SM_CHANNEL_VALIDATION  /opt/ml/input/data/validation
SM_OUTPUT_DATA_DIR  /opt/ml/output/data
SM_NUM_GPUS         number of GPUs
SM_HPS              JSON dict of hyperparameters
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Logging — SageMaker captures stdout/stderr for CloudWatch
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sagemaker-training")


# ---------------------------------------------------------------------------
# SageMaker environment helpers
# ---------------------------------------------------------------------------

def get_sm_env() -> Dict[str, str]:
    """Collect all SM_* environment variables into a dict."""
    return {k: v for k, v in os.environ.items() if k.startswith("SM_")}


def get_hyperparameters() -> Dict[str, Any]:
    """Parse hyperparameters from SM_HPS or /opt/ml/input/config/."""
    # Try SM_HPS first (set by SageMaker SDK)
    sm_hps = os.environ.get("SM_HPS")
    if sm_hps:
        return json.loads(sm_hps)

    # Fallback: read hyperparameters.json
    hp_path = Path("/opt/ml/input/config/hyperparameters.json")
    if hp_path.exists():
        with open(hp_path) as f:
            raw = json.load(f)
        # SageMaker stringifies all values
        return {k: _parse_hp_value(v) for k, v in raw.items()}

    return {}


def _parse_hp_value(v: str) -> Any:
    """Best-effort parse a stringified hyperparameter value."""
    if isinstance(v, str):
        lower = v.lower()
        if lower in ("true", "false"):
            return lower == "true"
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
    return v


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(channel_dir: str, config: Dict[str, Any]) -> TensorDataset:
    """Load Parquet files from a SageMaker input channel into a TensorDataset.

    Parameters
    ----------
    channel_dir : str
        Directory containing Parquet files (e.g. /opt/ml/input/data/train).
    config : dict
        Pipeline config dict with task/feature definitions.

    Returns
    -------
    TensorDataset
        Contains (features, *label_tensors_per_task).
    """
    import pandas as pd

    channel_path = Path(channel_dir)
    parquet_files = sorted(channel_path.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in {channel_dir}"
        )

    logger.info(f"Loading {len(parquet_files)} Parquet file(s) from {channel_dir}")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Extract features and labels
    tasks = config.get("tasks", [])
    label_cols = [t["label_col"] for t in tasks]
    feature_cols = [c for c in df.columns if c not in label_cols]

    features = torch.tensor(
        df[feature_cols].fillna(0).values, dtype=torch.float32,
    )

    label_tensors = []
    for task in tasks:
        col = task["label_col"]
        if col in df.columns:
            if task["type"] == "multiclass":
                label_tensors.append(
                    torch.tensor(df[col].fillna(0).values, dtype=torch.long)
                )
            else:
                label_tensors.append(
                    torch.tensor(df[col].fillna(0).values, dtype=torch.float32)
                )
        else:
            logger.warning(f"Label column '{col}' not found, using zeros")
            label_tensors.append(torch.zeros(len(df), dtype=torch.float32))

    return TensorDataset(features, *label_tensors)


# ---------------------------------------------------------------------------
# Metric printing (captured by SageMaker → CloudWatch)
# ---------------------------------------------------------------------------

def report_metrics(prefix: str, metrics: Dict[str, float], epoch: int) -> None:
    """Print metrics in the format SageMaker expects for CloudWatch.

    Each metric is printed as ``<name>=<value>`` on the same line.
    The METRIC_DEFINITIONS regex in the trainer picks these up.
    """
    parts = [f"epoch={epoch}"]
    for name, value in sorted(metrics.items()):
        parts.append(f"{prefix}_{name}={value:.6f}")
    line = " ".join(parts)
    logger.info(line)
    # Also print to stdout directly for SageMaker metric capture
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_names: list[str],
    epoch: int,
) -> Dict[str, float]:
    """Run one training epoch.

    Returns
    -------
    dict[str, float]
        Training metrics (loss, per-task losses).
    """
    from core.model.ple.model import PLEInput

    model.train()
    total_loss = 0.0
    task_losses_sum: Dict[str, float] = {name: 0.0 for name in task_names}
    n_batches = 0

    for batch in dataloader:
        features = batch[0].to(device)
        targets = {
            task_names[i]: batch[i + 1].to(device)
            for i in range(len(task_names))
        }

        inputs = PLEInput(features=features, targets=targets)
        output = model(inputs, compute_loss=True)

        if output.total_loss is None:
            continue

        optimizer.zero_grad()
        output.total_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += output.total_loss.item()
        if output.task_losses:
            for name, loss_val in output.task_losses.items():
                task_losses_sum[name] += loss_val.item()
        n_batches += 1

        # Update global step
        model.set_global_step(model.global_step + 1)

    if n_batches == 0:
        return {"loss": 0.0}

    metrics = {"loss": total_loss / n_batches}
    for name in task_names:
        metrics[f"loss_{name}"] = task_losses_sum[name] / n_batches
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    task_names: list[str],
) -> Dict[str, float]:
    """Run validation and compute metrics.

    Returns
    -------
    dict[str, float]
        Validation metrics including loss and per-task AUC (binary tasks).
    """
    from core.model.ple.model import PLEInput
    from sklearn.metrics import roc_auc_score

    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds: Dict[str, list] = {name: [] for name in task_names}
    all_targets: Dict[str, list] = {name: [] for name in task_names}

    for batch in dataloader:
        features = batch[0].to(device)
        targets = {
            task_names[i]: batch[i + 1].to(device)
            for i in range(len(task_names))
        }

        inputs = PLEInput(features=features, targets=targets)
        output = model(inputs, compute_loss=True)

        if output.total_loss is not None:
            total_loss += output.total_loss.item()
            n_batches += 1

        for name in task_names:
            if name in output.predictions:
                pred = output.predictions[name].cpu().numpy()
                tgt = targets[name].cpu().numpy()
                all_preds[name].append(pred)
                all_targets[name].append(tgt)

    if n_batches == 0:
        return {"loss": 0.0}

    metrics: Dict[str, float] = {"loss": total_loss / n_batches}

    # Per-task metrics
    for name in task_names:
        if all_preds[name]:
            preds = np.concatenate(all_preds[name]).squeeze()
            tgts = np.concatenate(all_targets[name]).squeeze()

            # AUC for binary tasks
            unique_labels = np.unique(tgts)
            if len(unique_labels) == 2:
                try:
                    metrics[f"auc_{name}"] = roc_auc_score(tgts, preds)
                except ValueError:
                    pass

            # MAE for regression
            try:
                metrics[f"mae_{name}"] = float(np.mean(np.abs(preds - tgts)))
            except Exception:
                pass

    # Aggregate AUC
    auc_values = [v for k, v in metrics.items() if k.startswith("auc_")]
    if auc_values:
        metrics["auc"] = float(np.mean(auc_values))

    return metrics


# ---------------------------------------------------------------------------
# Phase control
# ---------------------------------------------------------------------------

def apply_phase_config(
    model: nn.Module,
    phase: str,
    pretrained_uri: Optional[str] = None,
) -> None:
    """Configure model for the specified training phase.

    Parameters
    ----------
    model : nn.Module
        PLE model.
    phase : str
        ``"1"`` to freeze towers, ``"2"`` to unfreeze all.
    pretrained_uri : str, optional
        S3 URI or local path for pre-trained weights (phase 2).
    """
    if phase == "1":
        # Phase 1: freeze task towers, train shared experts + gating
        logger.info("Phase 1: Freezing task towers")
        for name, param in model.named_parameters():
            if "task_towers" in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"  Trainable: {trainable:,} / {total:,} parameters")

    elif phase == "2":
        # Phase 2: unfreeze everything
        logger.info("Phase 2: Unfreezing all parameters")
        for param in model.parameters():
            param.requires_grad = True

        # Load phase 1 weights if available
        if pretrained_uri:
            logger.info(f"Loading pre-trained weights from {pretrained_uri}")
            if pretrained_uri.startswith("s3://"):
                _download_pretrained_from_s3(pretrained_uri)
                local_path = "/tmp/pretrained/model.pth"
            else:
                local_path = pretrained_uri

            if os.path.exists(local_path):
                state = torch.load(local_path, map_location="cpu", weights_only=False)
                if "model_state_dict" in state:
                    model.load_state_dict(state["model_state_dict"], strict=False)
                else:
                    model.load_state_dict(state, strict=False)
                logger.info("Pre-trained weights loaded successfully")
    else:
        logger.info(f"Single-phase training (phase='{phase}')")


def _download_pretrained_from_s3(s3_uri: str) -> None:
    """Download pre-trained model from S3."""
    import boto3
    import tarfile

    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]

    local_tar = "/tmp/pretrained_model.tar.gz"
    local_dir = "/tmp/pretrained/"

    os.makedirs(local_dir, exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_tar)

    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(local_dir)

    logger.info(f"Extracted pre-trained model to {local_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """SageMaker training entry point."""
    start_time = time.time()

    # -- Environment --
    sm_env = get_sm_env()
    hp = get_hyperparameters()

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    val_dir = os.environ.get(
        "SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation",
    )
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    checkpoint_dir = os.environ.get(
        "SM_CHECKPOINT_DIR", "/opt/ml/checkpoints",
    )
    num_gpus = int(os.environ.get("SM_NUM_GPUS", "0"))

    logger.info("=" * 60)
    logger.info("SageMaker PLE Training Entry Point")
    logger.info("=" * 60)
    logger.info(f"GPUs available: {num_gpus}")
    logger.info(f"Model dir: {model_dir}")
    logger.info(f"Train dir: {train_dir}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")

    # -- Parse config from hyperparameters --
    config_str = hp.get("config", "{}")
    if isinstance(config_str, str):
        config = json.loads(config_str)
    else:
        config = config_str

    task_name = hp.get("task_name", config.get("task_name", "default"))
    batch_size = int(hp.get("batch_size", 2048))
    epochs = int(hp.get("epochs", 20))
    lr = float(hp.get("learning_rate", 1e-3))
    patience = int(hp.get("early_stopping_patience", 5))
    seed = int(hp.get("seed", 42))
    phase = str(hp.get("phase", "single"))
    freeze_towers = hp.get("freeze_towers", False)
    pretrained_uri = hp.get("pretrained_model_uri")

    logger.info(f"Task: {task_name}")
    logger.info(f"Phase: {phase}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    # -- Seed --
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -- Device --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # -- Load data --
    train_dataset = load_data(train_dir, config)
    logger.info(f"Train dataset: {len(train_dataset)} samples")

    val_dataset = None
    if os.path.isdir(val_dir):
        try:
            val_dataset = load_data(val_dir, config)
            logger.info(f"Validation dataset: {len(val_dataset)} samples")
        except FileNotFoundError:
            logger.warning("No validation data found, using train data split")

    # Split train data if no separate validation set
    if val_dataset is None:
        n = len(train_dataset)
        val_size = max(1, int(n * 0.1))
        train_size = n - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # -- Build model --
    from core.model.ple.model import PLEModel
    from core.model.ple.config import PLEConfig

    model_config = config.get("model", {})
    tasks = config.get("tasks", [])
    task_names = [t["name"] for t in tasks]

    # Determine input dimension from data
    sample_features = train_dataset[0][0] if hasattr(train_dataset, '__getitem__') else None
    if sample_features is not None:
        input_dim = sample_features.shape[0] if isinstance(sample_features, torch.Tensor) else len(sample_features)
    else:
        input_dim = model_config.get("expert_hidden_dim", 128)

    # Build PLEConfig
    ple_config = PLEConfig(
        input_dim=input_dim,
        task_names=task_names,
        num_shared_experts=model_config.get("num_shared_experts", 2),
        num_extraction_layers=model_config.get("num_layers", 2),
        dropout=model_config.get("dropout", 0.1),
    )

    # Set task overrides
    for t in tasks:
        ple_config.task_overrides[t["name"]] = {
            "task_type": t.get("type", "binary"),
            "output_dim": t.get("num_classes", 1),
        }

    model = PLEModel(ple_config).to(device)
    logger.info(model.summary())

    # -- Apply phase configuration --
    apply_phase_config(model, phase, pretrained_uri)

    # -- Optimizer & Scheduler --
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # -- Checkpoint manager --
    from core.training.checkpoint import CheckpointManager

    ckpt_mgr = CheckpointManager(
        local_dir=checkpoint_dir,
        max_keep=3,
    )

    # Resume from checkpoint if available (Spot restart)
    start_epoch = 0
    resume_state = ckpt_mgr.load_latest(model, optimizer, scheduler, map_location=device)
    if resume_state:
        start_epoch = resume_state["epoch"] + 1
        logger.info(f"Resuming from epoch {start_epoch}")

    # -- Training loop --
    best_val_loss = float("inf")
    no_improve_count = 0

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.set_epoch(epoch)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, task_names, epoch,
        )
        report_metrics("train", train_metrics, epoch)

        # Validate
        val_metrics = validate(model, val_loader, device, task_names)
        report_metrics("val", val_metrics, epoch)

        # Report learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"learning_rate={current_lr:.8f}", flush=True)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch}/{epochs-1} completed in {epoch_time:.1f}s — "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )

        # -- Checkpoint --
        ckpt_mgr.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=model.global_step,
            metrics=val_metrics,
            config=config,
        )

        # Best model tracking
        val_loss = val_metrics["loss"]
        if ckpt_mgr.is_best(val_loss, "val_loss", higher_is_better=False):
            ckpt_mgr.save_best(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=model.global_step,
                metrics=val_metrics,
                config=config,
            )
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Early stopping
        if no_improve_count >= patience:
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            break

    # -- Load best model for final save --
    ckpt_mgr.load_best(model, map_location=device)

    # -- Evaluate best model --
    final_metrics = validate(model, val_loader, device, task_names)
    report_metrics("final_val", final_metrics, epochs)

    # -- Save model for SageMaker packaging --
    os.makedirs(model_dir, exist_ok=True)

    # Save as both state_dict and full checkpoint
    model_path = os.path.join(model_dir, "model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "ple_config": {
            "input_dim": ple_config.input_dim,
            "task_names": ple_config.task_names,
            "num_shared_experts": ple_config.num_shared_experts,
            "num_extraction_layers": ple_config.num_extraction_layers,
        },
        "metrics": final_metrics,
        "task_name": task_name,
        "phase": phase,
    }, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save config separately for easy inspection
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Save evaluation report
    report_path = os.path.join(output_dir, "evaluation_report.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "task_name": task_name,
            "phase": phase,
            "final_metrics": final_metrics,
            "epochs_trained": epoch + 1,
            "total_time_seconds": time.time() - start_time,
        }, f, indent=2)

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Training complete in {total_time:.1f}s")
    logger.info(f"Final metrics: {final_metrics}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
