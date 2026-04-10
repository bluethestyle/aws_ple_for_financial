"""
Change Detector for Ops/Audit Agents
======================================

Detects pipeline changes via push (events) and pull (periodic comparison) channels.
Emits standardized change events that trigger agent checklist re-evaluation.

Push sources:
    - git post-commit hook → code/config changes
    - _PipelineState.mark_complete() callback → pipeline stage completions
    - IngestionRunner completion → data changes

Pull sources:
    - Ingestion manifest diff → data volume/schema changes
    - CloudWatch metrics → serving metric shifts
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["ChangeDetector", "ChangeEvent"]

# Pipeline part classification by file path
_PATH_TO_PART: Dict[str, str] = {
    "core/ingestion/": "P1",
    "core/pipeline/": "P2",
    "core/feature/": "P2",
    "core/training/": "P3",
    "core/model/": "P3",
    "core/serving/": "P4",
    "core/recommendation/": "P5",
    "core/monitoring/": "P6",
    "core/compliance/": "P6",
    "core/agent/": "P6",
    "configs/": "P2",  # config changes primarily affect feature engineering
}


class ChangeEvent:
    """Standardized change event."""

    def __init__(
        self,
        change_type: str,  # "code", "config", "model", "data", "metric", "regulation"
        source: str,  # "git_post_commit", "pipeline_state", "manifest_diff", etc.
        details: Dict[str, Any],
        affected_parts: Optional[List[str]] = None,
    ) -> None:
        self.event_type = "change_detected"
        self.change_type = change_type
        self.source = source
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.details = details
        self.affected_parts = affected_parts or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "change_type": self.change_type,
            "source": self.source,
            "timestamp": self.timestamp,
            "details": self.details,
            "affected_parts": self.affected_parts,
        }


class ChangeDetector:
    """Detects and classifies pipeline changes.

    Args:
        repo_root: Git repository root path.
        manifest_dir: Directory containing ingestion manifests.
        event_callbacks: List of callbacks invoked on each detected change.
    """

    def __init__(
        self,
        repo_root: str = ".",
        manifest_dir: str = "outputs/manifests",
        event_callbacks: Optional[List[Callable[[ChangeEvent], None]]] = None,
    ) -> None:
        self._repo_root = Path(repo_root)
        self._manifest_dir = Path(manifest_dir)
        self._callbacks = event_callbacks or []
        self._event_queue: deque[ChangeEvent] = deque(maxlen=1000)
        self._last_manifest: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Push channel: receive events
    # ------------------------------------------------------------------

    def on_git_commit(self, commit_hash: str, changed_files: List[str]) -> ChangeEvent:
        """Handle git post-commit event.

        Args:
            commit_hash: The commit SHA.
            changed_files: List of changed file paths relative to repo root.
        """
        affected = self._classify_affected_parts(changed_files)

        # Separate code vs config changes
        config_files = [f for f in changed_files if f.endswith((".yaml", ".yml"))]
        code_files = [f for f in changed_files if f.endswith(".py")]

        change_type = "config" if config_files and not code_files else "code"

        event = ChangeEvent(
            change_type=change_type,
            source="git_post_commit",
            details={
                "commit_hash": commit_hash,
                "changed_files": changed_files,
                "code_files": code_files,
                "config_files": config_files,
                "diff_summary": f"{len(changed_files)} files changed",
            },
            affected_parts=affected,
        )
        self._emit(event)
        return event

    def on_pipeline_stage_complete(self, stage: str, artifacts: Dict[str, Any]) -> ChangeEvent:
        """Handle pipeline stage completion (callback from _PipelineState).

        Args:
            stage: Stage name (e.g., "stage_train", "stage_distill").
            artifacts: Stage artifacts dict.
        """
        stage_to_type = {
            "stage1_load": "data",
            "stage2_preprocessing": "data",
            "stage3_features": "data",
            "stage_train": "model",
            "stage_distill": "model",
            "stage_serving": "config",
        }
        change_type = stage_to_type.get(stage, "model")

        stage_to_part = {
            "stage1_load": "P1",
            "stage2_preprocessing": "P2",
            "stage3_features": "P2",
            "stage4_labels": "P2",
            "stage5_split": "P2",
            "stage6_normalization": "P2",
            "stage_train": "P3",
            "stage_analysis": "P3",
            "stage_distill": "P3",
            "stage_serving": "P4",
            "stage_cpe": "P5",
        }
        part = stage_to_part.get(stage, "P2")

        event = ChangeEvent(
            change_type=change_type,
            source="pipeline_state",
            details={
                "stage": stage,
                "artifacts": artifacts,
            },
            affected_parts=[part],
        )
        self._emit(event)
        return event

    def on_ingestion_complete(self, manifest: Dict[str, Any]) -> ChangeEvent:
        """Handle ingestion completion.

        Compares with previous manifest to detect volume/schema changes.
        """
        details: Dict[str, Any] = {"manifest_summary": {
            "total_domains": manifest.get("total_domains", 0),
            "total_rows": manifest.get("total_rows", 0),
            "domains_passed": manifest.get("domains_passed", 0),
        }}

        # Diff with previous manifest if available
        if self._last_manifest is not None:
            prev_rows = self._last_manifest.get("total_rows", 0)
            curr_rows = manifest.get("total_rows", 0)
            if prev_rows > 0:
                delta_pct = abs(curr_rows - prev_rows) / prev_rows
                details["row_count_delta_pct"] = round(delta_pct, 4)
                details["row_count_delta_abs"] = curr_rows - prev_rows

        self._last_manifest = manifest

        event = ChangeEvent(
            change_type="data",
            source="ingestion_complete",
            details=details,
            affected_parts=["P1", "P2"],
        )
        self._emit(event)
        return event

    # ------------------------------------------------------------------
    # Pull channel: periodic checks
    # ------------------------------------------------------------------

    def check_manifest_diff(self, current_path: str, previous_path: str) -> Optional[ChangeEvent]:
        """Compare two ingestion manifests for significant differences."""
        try:
            with open(current_path, encoding="utf-8") as f:
                current = json.load(f)
            with open(previous_path, encoding="utf-8") as f:
                previous = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Cannot compare manifests: %s", e)
            return None

        prev_rows = previous.get("total_rows", 0)
        curr_rows = current.get("total_rows", 0)

        if prev_rows == 0:
            return None

        delta_pct = abs(curr_rows - prev_rows) / prev_rows
        if delta_pct < 0.05:  # less than 5% change — not significant
            return None

        event = ChangeEvent(
            change_type="data",
            source="manifest_diff",
            details={
                "previous_rows": prev_rows,
                "current_rows": curr_rows,
                "delta_pct": round(delta_pct, 4),
            },
            affected_parts=["P1", "P2"],
        )
        self._emit(event)
        return event

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_affected_parts(self, changed_files: List[str]) -> List[str]:
        """Map changed file paths to affected pipeline parts (P1-P6)."""
        parts = set()
        for fpath in changed_files:
            fpath_normalized = fpath.replace("\\", "/")
            for prefix, part in _PATH_TO_PART.items():
                if fpath_normalized.startswith(prefix):
                    parts.add(part)
                    break
        return sorted(parts)

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _emit(self, event: ChangeEvent) -> None:
        """Queue event and notify callbacks."""
        self._event_queue.append(event)
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error("Change event callback failed: %s", e, exc_info=True)

    def get_pending_events(self) -> List[ChangeEvent]:
        """Get all pending events and clear the queue."""
        events = list(self._event_queue)
        self._event_queue.clear()
        return events

    @property
    def event_count(self) -> int:
        return len(self._event_queue)
