"""
Experiment Comparator -- compare runs without MLflow.

Reads the file-based directory layout produced by
:class:`~core.training.experiment.LocalTracker` and provides utilities
for listing, comparing, and ranking experiment runs.

Example::

    from core.training.experiment_comparator import ExperimentComparator

    cmp = ExperimentComparator("experiments")
    runs = cmp.list_runs("my_experiment")
    result = cmp.compare("my_experiment", ["run_a", "run_b"], "val/loss")
    print(result.best_run_name, result.ranking)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────


@dataclass
class RunSummary:
    """Summary of a single experiment run."""

    run_name: str
    experiment_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    final_metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: str = ""


@dataclass
class ComparisonResult:
    """Result of comparing multiple runs on a single metric."""

    experiment_name: str
    metric_key: str
    mode: str  # "min" or "max"
    runs: List[RunSummary] = field(default_factory=list)
    best_run_name: str = ""
    ranking: List[Tuple[str, float]] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# ExperimentComparator
# ──────────────────────────────────────────────────────────────────────


class ExperimentComparator:
    """Compare experiment runs stored by :class:`LocalTracker`.

    Expected directory layout::

        {experiments_dir}/{experiment_name}/{run_name}/
            params.json
            metrics.jsonl
            tags.json      (optional)
            artifacts/

    Parameters
    ----------
    experiments_dir : str
        Root directory where experiments are stored.
    """

    def __init__(self, experiments_dir: str = "experiments") -> None:
        self._base_dir = Path(experiments_dir)

    # ── Public API ────────────────────────────────────────────────────

    def list_runs(self, experiment_name: str) -> List[RunSummary]:
        """List all runs for an experiment.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment.

        Returns
        -------
        list of RunSummary
            Sorted by creation time (newest first).
        """
        exp_dir = self._base_dir / experiment_name
        if not exp_dir.is_dir():
            logger.warning("Experiment directory not found: %s", exp_dir)
            return []

        runs: List[RunSummary] = []
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            summary = self._load_run(experiment_name, run_dir.name)
            if summary is not None:
                runs.append(summary)

        # Sort by created_at descending
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs

    def compare(
        self,
        experiment_name: str,
        run_names: List[str],
        metric_key: str,
        *,
        mode: str = "min",
    ) -> ComparisonResult:
        """Compare specific runs on a single metric.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment.
        run_names : list of str
            Run names to compare.
        metric_key : str
            Metric key to rank by.
        mode : str
            ``"min"`` (lower is better) or ``"max"`` (higher is better).

        Returns
        -------
        ComparisonResult
        """
        runs: List[RunSummary] = []
        scored: List[Tuple[str, float]] = []

        for name in run_names:
            summary = self._load_run(experiment_name, name)
            if summary is None:
                logger.warning(
                    "Run '%s' not found in experiment '%s'; skipping.",
                    name, experiment_name,
                )
                continue
            runs.append(summary)
            value = summary.final_metrics.get(metric_key)
            if value is not None:
                scored.append((name, value))

        # Rank
        reverse = mode == "max"
        scored.sort(key=lambda t: t[1], reverse=reverse)

        best_name = scored[0][0] if scored else ""

        return ComparisonResult(
            experiment_name=experiment_name,
            metric_key=metric_key,
            mode=mode,
            runs=runs,
            best_run_name=best_name,
            ranking=scored,
        )

    def best_run(
        self,
        experiment_name: str,
        metric_key: str,
        mode: str = "min",
    ) -> Optional[RunSummary]:
        """Find the best run across all runs in an experiment.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment.
        metric_key : str
            Metric key to optimise.
        mode : str
            ``"min"`` or ``"max"``.

        Returns
        -------
        RunSummary or None
            Best run, or ``None`` if no runs have the requested metric.
        """
        all_runs = self.list_runs(experiment_name)
        if not all_runs:
            return None

        run_names = [r.run_name for r in all_runs]
        result = self.compare(experiment_name, run_names, metric_key, mode=mode)
        if not result.ranking:
            return None

        best_name = result.best_run_name
        for run in result.runs:
            if run.run_name == best_name:
                return run
        return None

    def diff_params(
        self,
        run_a: RunSummary,
        run_b: RunSummary,
    ) -> Dict[str, Dict[str, Any]]:
        """Show hyperparameter differences between two runs.

        Parameters
        ----------
        run_a : RunSummary
            First run.
        run_b : RunSummary
            Second run.

        Returns
        -------
        dict
            Mapping of param name to ``{"run_a": value, "run_b": value}``
            for every parameter that differs.
        """
        all_keys = set(run_a.params.keys()) | set(run_b.params.keys())
        diff: Dict[str, Dict[str, Any]] = {}

        for key in sorted(all_keys):
            val_a = run_a.params.get(key)
            val_b = run_b.params.get(key)
            if val_a != val_b:
                diff[key] = {
                    run_a.run_name: val_a,
                    run_b.run_name: val_b,
                }

        return diff

    def export_comparison(self, result: ComparisonResult) -> Dict[str, Any]:
        """Serialise a comparison result to a JSON-friendly dict.

        Parameters
        ----------
        result : ComparisonResult
            The result returned by :meth:`compare`.

        Returns
        -------
        dict
            JSON-serialisable report.
        """
        return {
            "experiment_name": result.experiment_name,
            "metric_key": result.metric_key,
            "mode": result.mode,
            "best_run_name": result.best_run_name,
            "ranking": [
                {"run_name": name, "value": value}
                for name, value in result.ranking
            ],
            "runs": [
                {
                    "run_name": r.run_name,
                    "experiment_name": r.experiment_name,
                    "params": r.params,
                    "final_metrics": r.final_metrics,
                    "tags": r.tags,
                    "created_at": r.created_at,
                }
                for r in result.runs
            ],
            "total_runs": len(result.runs),
        }

    # ── Internal helpers ──────────────────────────────────────────────

    def _load_run(
        self,
        experiment_name: str,
        run_name: str,
    ) -> Optional[RunSummary]:
        """Load a single run from disk."""
        run_dir = self._base_dir / experiment_name / run_name
        if not run_dir.is_dir():
            return None

        params = self._read_json(run_dir / "params.json")
        tags = self._read_json(run_dir / "tags.json")
        final_metrics = self._extract_final_metrics(run_dir / "metrics.jsonl")

        # created_at from directory mtime
        import os

        try:
            mtime = os.path.getmtime(run_dir)
            from datetime import datetime, timezone

            created_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except OSError:
            created_at = ""

        return RunSummary(
            run_name=run_name,
            experiment_name=experiment_name,
            params=params,
            final_metrics=final_metrics,
            tags=tags,
            created_at=created_at,
        )

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        """Read a JSON file, returning empty dict on failure."""
        if not path.is_file():
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Failed to read %s: %s", path, exc)
            return {}

    @staticmethod
    def _extract_final_metrics(path: Path) -> Dict[str, float]:
        """Extract the last-recorded value for each metric key from JSONL.

        The JSONL format matches :class:`LocalTracker`::

            {"key": "train/loss", "value": 0.42, "step": 100, "ts": ...}
        """
        if not path.is_file():
            return {}

        latest: Dict[str, float] = {}
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        key = record.get("key", "")
                        value = record.get("value")
                        if key and value is not None:
                            latest[key] = float(value)
                    except (json.JSONDecodeError, ValueError, TypeError):
                        continue
        except OSError as exc:
            logger.debug("Failed to read %s: %s", path, exc)

        return latest

    # ── Repr ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return f"ExperimentComparator(base_dir='{self._base_dir}')"
