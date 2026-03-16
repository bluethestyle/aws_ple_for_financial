"""
Experiment tracking interface for PLE training.

Provides a unified API for logging metrics, parameters, and model artifacts.
Three implementations ship out of the box:

* :class:`LocalTracker` -- File-based logging (default, always works).
* :class:`SageMakerTracker` -- SageMaker Experiments integration.
* :func:`auto_tracker` -- Auto-detect: SageMaker if available, else local.

Usage::

    tracker = auto_tracker(experiment_name="my_experiment")
    tracker.log_params({"lr": 0.001, "batch_size": 4096})
    tracker.log_metric("train/loss", 0.42, step=100)
    tracker.end()
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Abstract interface
# ============================================================================


class ExperimentTracker(ABC):
    """Abstract interface for experiment tracking.

    Implementations must provide ``log_metric``, ``log_params``,
    ``log_artifact``, ``log_model``, and lifecycle methods ``start``/``end``.
    """

    @abstractmethod
    def start(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Begin a new tracking run."""

    @abstractmethod
    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """Log a single scalar metric."""

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log multiple metrics at once.

        Default implementation calls :meth:`log_metric` in a loop.
        Override for batch-optimised backends.
        """
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters (called once at start of run)."""

    @abstractmethod
    def log_artifact(self, local_path: Union[str, Path]) -> None:
        """Upload a file artifact (checkpoint, config, plot, ...)."""

    @abstractmethod
    def log_model(self, model: Any, artifact_name: str = "model") -> None:
        """Log a trained model (framework-specific serialisation)."""

    @abstractmethod
    def end(self) -> None:
        """End the current tracking run and flush buffers."""

    @property
    @abstractmethod
    def run_id(self) -> Optional[str]:
        """Return the current run identifier (or ``None``)."""

    @property
    def is_active(self) -> bool:
        """Whether a tracking run is currently open."""
        return self.run_id is not None


# ============================================================================
# Local file-based tracker
# ============================================================================


class LocalTracker(ExperimentTracker):
    """File-based experiment tracker.

    Writes metrics to a JSONL file and copies artifacts to a local directory.
    Always works -- no external services required.

    Directory layout::

        {base_dir}/{experiment_name}/{run_name}/
            params.json
            metrics.jsonl
            artifacts/
                best.pt
                ...
    """

    def __init__(self, base_dir: str = "experiments") -> None:
        self._base_dir = Path(base_dir)
        self._run_dir: Optional[Path] = None
        self._run_id: Optional[str] = None
        self._metrics_file = None

    def start(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        if run_name is None:
            run_name = f"run_{int(time.time())}"

        self._run_id = run_name
        self._run_dir = self._base_dir / experiment_name / run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "artifacts").mkdir(exist_ok=True)

        # Save tags
        if tags:
            with open(self._run_dir / "tags.json", "w") as f:
                json.dump(tags, f, indent=2)

        # Open metrics file
        self._metrics_file = open(self._run_dir / "metrics.jsonl", "a")

        logger.info("LocalTracker: run started at %s", self._run_dir)

    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        if self._metrics_file is None:
            return
        record = {"key": key, "value": value, "step": step, "ts": time.time()}
        self._metrics_file.write(json.dumps(record) + "\n")
        self._metrics_file.flush()

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._run_dir is None:
            return
        # Merge with existing params
        params_path = self._run_dir / "params.json"
        existing = {}
        if params_path.exists():
            with open(params_path) as f:
                existing = json.load(f)
        existing.update({k: _safe_json(v) for k, v in params.items()})
        with open(params_path, "w") as f:
            json.dump(existing, f, indent=2)

    def log_artifact(self, local_path: Union[str, Path]) -> None:
        if self._run_dir is None:
            return
        import shutil

        src = Path(local_path)
        dst = self._run_dir / "artifacts" / src.name
        if src.is_file():
            shutil.copy2(src, dst)
        elif src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)

    def log_model(self, model: Any, artifact_name: str = "model") -> None:
        if self._run_dir is None:
            return
        import torch

        model_path = self._run_dir / "artifacts" / f"{artifact_name}.pt"
        torch.save(model.state_dict(), model_path)
        logger.info("LocalTracker: model saved to %s", model_path)

    def end(self) -> None:
        if self._metrics_file is not None:
            self._metrics_file.close()
            self._metrics_file = None
        logger.info("LocalTracker: run ended (%s)", self._run_id)
        self._run_id = None

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id


# ============================================================================
# SageMaker Experiments tracker
# ============================================================================


class SageMakerTracker(ExperimentTracker):
    """SageMaker Experiments tracker.

    Wraps the ``sagemaker.experiments`` API.  Falls back to a no-op if
    the SageMaker SDK is not available or the run cannot be started.
    """

    def __init__(self) -> None:
        self._run = None
        self._run_id_str: Optional[str] = None

    def start(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        try:
            from sagemaker.experiments.run import Run
            from sagemaker.session import Session

            session = Session()
            self._run = Run(
                experiment_name=experiment_name,
                run_name=run_name,
                sagemaker_session=session,
            )
            self._run.__enter__()
            self._run_id_str = run_name or experiment_name
            logger.info("SageMakerTracker: run started (%s / %s)",
                        experiment_name, run_name)
        except Exception as e:
            logger.warning("SageMakerTracker: failed to start run: %s. "
                           "Metrics will not be logged.", e)
            self._run = None

    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        if self._run is None:
            return
        try:
            self._run.log_metric(name=key, value=value, step=step or 0)
        except Exception:
            pass  # non-critical

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._run is None:
            return
        try:
            for k, v in params.items():
                self._run.log_parameter(k, str(v))
        except Exception:
            pass

    def log_artifact(self, local_path: Union[str, Path]) -> None:
        if self._run is None:
            return
        try:
            self._run.log_file(str(local_path))
        except Exception:
            pass

    def log_model(self, model: Any, artifact_name: str = "model") -> None:
        # SageMaker model logging is handled by the training job output
        logger.debug("SageMakerTracker.log_model: model artifacts are "
                      "uploaded via SageMaker training job output path.")

    def end(self) -> None:
        if self._run is not None:
            try:
                self._run.__exit__(None, None, None)
            except Exception:
                pass
            self._run = None
        self._run_id_str = None
        logger.info("SageMakerTracker: run ended")

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id_str


# ============================================================================
# Auto-detection factory
# ============================================================================


def _is_sagemaker_environment() -> bool:
    """Detect whether we are running inside a SageMaker training job."""
    return (
        os.environ.get("SM_MODEL_DIR") is not None
        or os.environ.get("TRAINING_JOB_NAME") is not None
        or os.path.isdir("/opt/ml/model")
    )


def auto_tracker(
    experiment_name: str = "ple_training",
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    base_dir: str = "experiments",
) -> ExperimentTracker:
    """Create and start an appropriate experiment tracker.

    If running inside a SageMaker training job, returns a
    :class:`SageMakerTracker`.  Otherwise returns a :class:`LocalTracker`.

    Args:
        experiment_name: Name of the experiment.
        run_name: Optional run identifier.
        tags: Optional key-value tags for the run.
        base_dir: Base directory for the local tracker.

    Returns:
        An initialised and started :class:`ExperimentTracker`.
    """
    if _is_sagemaker_environment():
        tracker: ExperimentTracker = SageMakerTracker()
        logger.info("Auto-detected SageMaker environment.")
    else:
        tracker = LocalTracker(base_dir=base_dir)
        logger.info("Using local experiment tracker (base_dir=%s).", base_dir)

    tracker.start(experiment_name=experiment_name, run_name=run_name, tags=tags)
    return tracker


# ============================================================================
# Helpers
# ============================================================================


def _safe_json(value: Any) -> Any:
    """Convert a value to a JSON-safe type."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    return str(value)
