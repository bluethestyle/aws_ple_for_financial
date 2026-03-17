"""
Model Registry -- versioned model packaging with full traceability.

Packages teacher + student models together with:
- PLEConfig for teacher reconstruction
- Per-task feature selection results (selected_indices, selected_names)
- Per-task fidelity validation results (pass/fail, metrics)
- Training metrics and distillation audit trail
- Version manifest linking everything together

Storage: S3 with versioned directory structure.

Usage::

    registry = ModelRegistry(s3_base="s3://bucket/models/", region="ap-northeast-2")

    # Package a new model version
    version = registry.package(
        version="v2.1.0",
        teacher_state_dict=model.state_dict(),
        teacher_config=ple_config,
        training_metrics={"best_val_loss": 0.23, "avg_auc": 0.82},
        students={"ctr": lgbm_model_ctr, "churn": lgbm_model_churn},
        student_metadata={"ctr": {"num_trees": 500}},
        feature_selections={"ctr": feature_result_ctr},
        fidelity_results=[fidelity_ctr, fidelity_churn],
    )

    # Load for serving
    manifest = registry.load_manifest("v2.1.0")
    student = registry.load_student("v2.1.0", "ctr")
    features = registry.get_selected_features("v2.1.0", "ctr")

    # List versions
    versions = registry.list_versions()

    # Promote
    registry.promote("v2.1.0")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "ModelVersion",
    "ModelRegistry",
]


# ============================================================================
# Version manifest dataclass
# ============================================================================


@dataclass
class ModelVersion:
    """Metadata manifest for a single packaged model version.

    This is the single source of truth for everything in a version
    directory.  It is serialized as ``manifest.json``.

    Attributes:
        version: Semantic version string (e.g. ``"v2.1.0"``).
        created_at: ISO 8601 timestamp when the version was packaged.
        teacher_config_hash: SHA-256 of the PLEConfig JSON for
            reproducibility verification.
        teacher_metrics: Training metrics for the teacher model
            (e.g. ``{"best_val_loss": 0.23, "avg_auc": 0.82}``).
        student_tasks: List of task names with student models.
        fidelity_summary: Aggregate fidelity validation results
            (e.g. ``{"passed": 14, "failed": 2, "details": {...}}``).
        feature_selection_summary: Per-task count of selected features
            (e.g. ``{"ctr": 180, "churn": 150}``).
        promoted: Whether this version is the active serving version.
        promoted_at: ISO 8601 timestamp of promotion (``None`` if not
            promoted).
        metadata: Arbitrary extra metadata (git SHA, dataset ID, etc.).
    """

    version: str
    created_at: str
    teacher_config_hash: str
    teacher_metrics: Dict[str, float]
    student_tasks: List[str]
    fidelity_summary: Dict[str, Any]
    feature_selection_summary: Dict[str, int]
    promoted: bool = False
    promoted_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Model Registry
# ============================================================================


class ModelRegistry:
    """S3-based model registry with versioning and manifest tracking.

    Operates local-first: all artifacts are written to ``local_base``
    and then optionally uploaded to S3 when ``s3_base`` is configured.
    For loading, artifacts are downloaded from S3 to a local cache
    directory first.

    Directory structure per version::

        {version}/
            teacher/
                model.pth
                config.json
                training_metrics.json
            students/
                {task_name}/
                    model.lgbm
                    metadata.json
                    selected_features.json
                    fidelity.json
            manifest.json

    Parameters:
        s3_base: S3 URI prefix (e.g. ``"s3://bucket/models/"``).
            Leave empty for local-only operation.
        local_base: Local directory for storing version artifacts.
        region: AWS region for boto3 S3 client.
    """

    def __init__(
        self,
        s3_base: str = "",
        local_base: str = "models/",
        region: str = "ap-northeast-2",
    ) -> None:
        self._s3_base = s3_base.rstrip("/")
        self._local_base = local_base
        self._region = region

    # ------------------------------------------------------------------
    # Package
    # ------------------------------------------------------------------

    def package(
        self,
        version: str,
        teacher_state_dict: Optional[Dict] = None,
        teacher_config: Any = None,
        training_metrics: Optional[Dict[str, float]] = None,
        students: Optional[Dict[str, Any]] = None,
        student_metadata: Optional[Dict[str, Dict]] = None,
        feature_selections: Optional[Dict[str, Any]] = None,
        fidelity_results: Optional[List] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelVersion:
        """Package all artifacts into a versioned model directory.

        Writes teacher checkpoint, student LGBM models, feature
        selection results, fidelity validation results, and a top-level
        manifest to disk.  Uploads to S3 when ``s3_base`` is configured.

        Parameters:
            version: Semantic version string (e.g. ``"v2.1.0"``).
            teacher_state_dict: ``model.state_dict()`` from the PLE
                teacher.
            teacher_config: ``PLEConfig`` dataclass instance or dict.
            training_metrics: Teacher training metrics.
            students: ``{task_name: lgbm_booster}`` dict of trained
                student models.
            student_metadata: ``{task_name: metadata_dict}`` with
                per-student training details.
            feature_selections: ``{task_name: FeatureSelectionResult or
                dict}`` with per-task feature selection outcomes.
            fidelity_results: List of ``FidelityResult`` or dicts from
                distillation validation.
            metadata: Arbitrary metadata to attach to the manifest.

        Returns:
            A :class:`ModelVersion` manifest describing the packaged
            artifacts.
        """
        import torch

        version_dir = self._get_version_dir(version)
        training_metrics = training_metrics or {}
        students = students or {}
        student_metadata = student_metadata or {}
        feature_selections = feature_selections or {}
        fidelity_results = fidelity_results or []
        metadata = metadata or {}

        # -- Teacher artifacts ------------------------------------------------
        teacher_dir = os.path.join(version_dir, "teacher")
        os.makedirs(teacher_dir, exist_ok=True)

        config_hash = ""
        if teacher_config is not None:
            config_dict = self._to_dict(teacher_config)
            config_json = json.dumps(config_dict, sort_keys=True, default=str)
            config_hash = hashlib.sha256(config_json.encode()).hexdigest()
            self._write_json(config_dict, os.path.join(teacher_dir, "config.json"))

        if teacher_state_dict is not None:
            model_path = os.path.join(teacher_dir, "model.pth")
            torch.save(teacher_state_dict, model_path)
            logger.info("Teacher state_dict saved to %s", model_path)

        if training_metrics:
            self._write_json(
                training_metrics,
                os.path.join(teacher_dir, "training_metrics.json"),
            )

        # -- Student artifacts ------------------------------------------------
        students_dir = os.path.join(version_dir, "students")
        fidelity_by_task: Dict[str, Dict[str, Any]] = {}

        # Index fidelity results by task name
        for fr in fidelity_results:
            fr_dict = self._to_dict(fr)
            task = fr_dict.get("task_name", "unknown")
            fidelity_by_task[task] = fr_dict

        for task_name, model in students.items():
            task_dir = os.path.join(students_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)

            # Save LGBM model
            lgbm_path = os.path.join(task_dir, "model.lgbm")
            model.save_model(lgbm_path)
            logger.info("Student '%s' saved to %s", task_name, lgbm_path)

            # Save metadata
            meta = student_metadata.get(task_name, {})
            if not meta:
                # Auto-populate from model if possible
                meta = {
                    "task_name": task_name,
                    "num_trees": (
                        model.num_trees() if hasattr(model, "num_trees") else 0
                    ),
                    "num_features": (
                        model.num_feature() if hasattr(model, "num_feature") else 0
                    ),
                }
            self._write_json(meta, os.path.join(task_dir, "metadata.json"))

            # Save selected features
            if task_name in feature_selections:
                fs = feature_selections[task_name]
                fs_dict = self._to_dict(fs)
                # Normalize to a serving-friendly format
                selected = {
                    "indices": fs_dict.get("selected_indices", []),
                    "names": fs_dict.get("selected_names", []),
                    "count": fs_dict.get("selected_count", 0),
                    "original_count": fs_dict.get("original_count", 0),
                    "reduction_pct": fs_dict.get("reduction_pct", 0.0),
                    "selection_method": fs_dict.get("selection_method", ""),
                    "cumulative_threshold_used": fs_dict.get(
                        "cumulative_threshold_used", 0.0
                    ),
                }
                self._write_json(
                    selected, os.path.join(task_dir, "selected_features.json")
                )

            # Save fidelity
            if task_name in fidelity_by_task:
                self._write_json(
                    fidelity_by_task[task_name],
                    os.path.join(task_dir, "fidelity.json"),
                )

        # -- Fidelity summary -------------------------------------------------
        passed_count = sum(
            1 for fr in fidelity_results if self._to_dict(fr).get("passed", False)
        )
        failed_count = len(fidelity_results) - passed_count
        fidelity_summary: Dict[str, Any] = {
            "passed": passed_count,
            "failed": failed_count,
            "details": {
                self._to_dict(fr).get("task_name", "unknown"): {
                    "passed": self._to_dict(fr).get("passed", False),
                    "metrics": self._to_dict(fr).get("metrics", {}),
                    "failures": self._to_dict(fr).get("failures", []),
                }
                for fr in fidelity_results
            },
        }

        # -- Feature selection summary ----------------------------------------
        feature_selection_summary: Dict[str, int] = {}
        for task_name, fs in feature_selections.items():
            fs_dict = self._to_dict(fs)
            feature_selection_summary[task_name] = fs_dict.get("selected_count", 0)

        # -- Build manifest ---------------------------------------------------
        now = datetime.now(timezone.utc).isoformat()
        manifest = ModelVersion(
            version=version,
            created_at=now,
            teacher_config_hash=config_hash,
            teacher_metrics=training_metrics,
            student_tasks=sorted(students.keys()),
            fidelity_summary=fidelity_summary,
            feature_selection_summary=feature_selection_summary,
            promoted=False,
            promoted_at=None,
            metadata=metadata,
        )

        self._write_json(
            asdict(manifest), os.path.join(version_dir, "manifest.json")
        )
        logger.info(
            "Model version '%s' packaged: %d student tasks, fidelity %d/%d passed",
            version,
            len(students),
            passed_count,
            passed_count + failed_count,
        )

        # -- Upload to S3 if configured --------------------------------------
        if self._s3_base:
            s3_prefix = f"{self._s3_base}/{version}"
            self._upload_to_s3(version_dir, s3_prefix)

        return manifest

    # ------------------------------------------------------------------
    # Load manifest
    # ------------------------------------------------------------------

    def load_manifest(self, version: str) -> ModelVersion:
        """Load the manifest for a specific version.

        Downloads from S3 if the local copy is not available.

        Parameters:
            version: Version string (e.g. ``"v2.1.0"``).

        Returns:
            A :class:`ModelVersion` with all manifest fields.

        Raises:
            FileNotFoundError: If the manifest cannot be found locally
                or on S3.
        """
        version_dir = self._get_version_dir(version)
        manifest_path = os.path.join(version_dir, "manifest.json")

        if not os.path.exists(manifest_path) and self._s3_base:
            s3_prefix = f"{self._s3_base}/{version}"
            self._download_from_s3(s3_prefix, version_dir)

        data = self._read_json(manifest_path)
        return ModelVersion(
            version=data["version"],
            created_at=data["created_at"],
            teacher_config_hash=data.get("teacher_config_hash", ""),
            teacher_metrics=data.get("teacher_metrics", {}),
            student_tasks=data.get("student_tasks", []),
            fidelity_summary=data.get("fidelity_summary", {}),
            feature_selection_summary=data.get("feature_selection_summary", {}),
            promoted=data.get("promoted", False),
            promoted_at=data.get("promoted_at"),
            metadata=data.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # Load teacher
    # ------------------------------------------------------------------

    def load_teacher(
        self, version: str, device: str = "cpu"
    ) -> Tuple[Any, Any]:
        """Load teacher PLEModel from checkpoint.

        Reconstructs the PLEModel from the saved config and state_dict.

        Parameters:
            version: Version string.
            device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).

        Returns:
            ``(model, ple_config)`` tuple where ``model`` is the
            reconstructed PLEModel in eval mode.
        """
        import torch
        from core.model.ple.config import PLEConfig
        from core.model.ple.model import PLEModel

        version_dir = self._ensure_local(version)
        teacher_dir = os.path.join(version_dir, "teacher")

        # Load config
        config_data = self._read_json(os.path.join(teacher_dir, "config.json"))

        # Reconstruct PLEConfig -- filter to known fields
        known_fields = set(PLEConfig.__dataclass_fields__.keys())
        config_kwargs = {k: v for k, v in config_data.items() if k in known_fields}
        ple_config = PLEConfig(**config_kwargs)

        # Load state_dict
        model_path = os.path.join(teacher_dir, "model.pth")
        state_dict = torch.load(model_path, map_location=device, weights_only=False)

        # Reconstruct model
        model = PLEModel(ple_config)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        logger.info(
            "Teacher loaded from version '%s' (%d params)",
            version,
            sum(p.numel() for p in model.parameters()),
        )
        return model, ple_config

    # ------------------------------------------------------------------
    # Load student
    # ------------------------------------------------------------------

    def load_student(self, version: str, task_name: str) -> Any:
        """Load a specific LGBM student model.

        Parameters:
            version: Version string.
            task_name: Task identifier (e.g. ``"ctr"``).

        Returns:
            A LightGBM Booster loaded from the saved model file.

        Raises:
            FileNotFoundError: If the student model file does not exist.
        """
        import lightgbm as lgb

        version_dir = self._ensure_local(version)
        lgbm_path = os.path.join(
            version_dir, "students", task_name, "model.lgbm"
        )

        if not os.path.exists(lgbm_path):
            raise FileNotFoundError(
                f"Student model not found: {lgbm_path}"
            )

        model = lgb.Booster(model_file=lgbm_path)
        logger.info("Student '%s' loaded from version '%s'", task_name, version)
        return model

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    def get_selected_features(self, version: str, task_name: str) -> Dict:
        """Get feature selection result for a task.

        Returns the contents of ``selected_features.json`` which
        contains the indices and names of features the student model
        expects.  The serving layer MUST use this to slice the full
        644-dim feature vector before passing to the student.

        Parameters:
            version: Version string.
            task_name: Task identifier.

        Returns:
            Dict with keys ``"indices"``, ``"names"``, ``"count"``,
            ``"original_count"``, ``"reduction_pct"``.
        """
        version_dir = self._ensure_local(version)
        path = os.path.join(
            version_dir, "students", task_name, "selected_features.json"
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Selected features not found for task '{task_name}' "
                f"in version '{version}': {path}"
            )

        return self._read_json(path)

    # ------------------------------------------------------------------
    # Fidelity
    # ------------------------------------------------------------------

    def get_fidelity(
        self, version: str, task_name: Optional[str] = None
    ) -> Dict:
        """Get fidelity results.

        Parameters:
            version: Version string.
            task_name: If provided, returns fidelity for that task only.
                If ``None``, returns the fidelity summary from the
                manifest.

        Returns:
            Fidelity result dict.
        """
        if task_name is None:
            manifest = self.load_manifest(version)
            return manifest.fidelity_summary

        version_dir = self._ensure_local(version)
        path = os.path.join(
            version_dir, "students", task_name, "fidelity.json"
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Fidelity result not found for task '{task_name}' "
                f"in version '{version}': {path}"
            )

        return self._read_json(path)

    # ------------------------------------------------------------------
    # List / Latest / Promoted
    # ------------------------------------------------------------------

    def list_versions(self) -> List[ModelVersion]:
        """List all versions sorted by ``created_at`` descending.

        Scans the local base directory for version directories
        containing ``manifest.json``.

        Returns:
            List of :class:`ModelVersion` manifests, most recent first.
        """
        base = Path(self._local_base)
        if not base.exists():
            return []

        manifests: List[ModelVersion] = []
        for child in sorted(base.iterdir()):
            manifest_path = child / "manifest.json"
            if child.is_dir() and manifest_path.exists():
                try:
                    data = self._read_json(str(manifest_path))
                    manifests.append(
                        ModelVersion(
                            version=data["version"],
                            created_at=data["created_at"],
                            teacher_config_hash=data.get("teacher_config_hash", ""),
                            teacher_metrics=data.get("teacher_metrics", {}),
                            student_tasks=data.get("student_tasks", []),
                            fidelity_summary=data.get("fidelity_summary", {}),
                            feature_selection_summary=data.get(
                                "feature_selection_summary", {}
                            ),
                            promoted=data.get("promoted", False),
                            promoted_at=data.get("promoted_at"),
                            metadata=data.get("metadata", {}),
                        )
                    )
                except Exception:
                    logger.warning(
                        "Failed to load manifest from %s", manifest_path,
                        exc_info=True,
                    )

        # Sort by created_at descending
        manifests.sort(key=lambda m: m.created_at, reverse=True)
        return manifests

    def get_latest(self) -> Optional[str]:
        """Get the latest version string.

        Returns:
            Version string of the most recently created version, or
            ``None`` if no versions exist.
        """
        versions = self.list_versions()
        return versions[0].version if versions else None

    def get_promoted(self) -> Optional[str]:
        """Get the currently promoted (active) version.

        Returns:
            Version string of the promoted version, or ``None`` if no
            version is promoted.
        """
        versions = self.list_versions()
        for v in versions:
            if v.promoted:
                return v.version
        return None

    # ------------------------------------------------------------------
    # Promote
    # ------------------------------------------------------------------

    def promote(self, version: str) -> None:
        """Mark a version as promoted (active for serving).

        Demotes any previously promoted version by updating its
        ``manifest.json``.  Then marks the target version as promoted
        with a timestamp.

        If S3 is configured, the updated manifests are re-uploaded.

        Parameters:
            version: Version string to promote.

        Raises:
            FileNotFoundError: If the version manifest does not exist.
        """
        # Demote current champion
        current = self.get_promoted()
        if current and current != version:
            current_dir = self._get_version_dir(current)
            current_manifest_path = os.path.join(current_dir, "manifest.json")
            if os.path.exists(current_manifest_path):
                data = self._read_json(current_manifest_path)
                data["promoted"] = False
                data["promoted_at"] = None
                self._write_json(data, current_manifest_path)
                logger.info("Demoted version '%s'", current)

                if self._s3_base:
                    self._upload_file_to_s3(
                        current_manifest_path,
                        f"{self._s3_base}/{current}/manifest.json",
                    )

        # Promote target version
        version_dir = self._get_version_dir(version)
        manifest_path = os.path.join(version_dir, "manifest.json")

        if not os.path.exists(manifest_path):
            if self._s3_base:
                s3_prefix = f"{self._s3_base}/{version}"
                self._download_from_s3(s3_prefix, version_dir)
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(
                    f"Manifest not found for version '{version}': {manifest_path}"
                )

        now = datetime.now(timezone.utc).isoformat()
        data = self._read_json(manifest_path)
        data["promoted"] = True
        data["promoted_at"] = now
        self._write_json(data, manifest_path)

        if self._s3_base:
            self._upload_file_to_s3(
                manifest_path,
                f"{self._s3_base}/{version}/manifest.json",
            )

        logger.info("Promoted version '%s' at %s", version, now)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_version_dir(self, version: str) -> str:
        """Get local directory path for a version."""
        version_dir = os.path.join(self._local_base, version)
        os.makedirs(version_dir, exist_ok=True)
        return version_dir

    def _ensure_local(self, version: str) -> str:
        """Ensure version artifacts exist locally, downloading if needed.

        Returns the local version directory path.
        """
        version_dir = self._get_version_dir(version)
        manifest_path = os.path.join(version_dir, "manifest.json")

        if not os.path.exists(manifest_path) and self._s3_base:
            s3_prefix = f"{self._s3_base}/{version}"
            self._download_from_s3(s3_prefix, version_dir)

        return version_dir

    def _to_dict(self, obj: Any) -> Dict:
        """Convert a dataclass or dict to a plain dict."""
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        if hasattr(obj, "__dict__"):
            return vars(obj)
        return {"value": obj}

    def _read_json(self, path: str) -> Dict:
        """Read JSON from a local path.

        Parameters:
            path: Absolute or relative path to a JSON file.

        Returns:
            Parsed dict.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, data: Dict, path: str) -> None:
        """Write JSON to a local path.

        Creates parent directories as needed.

        Parameters:
            data: Dict to serialize.
            path: Target file path.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _upload_to_s3(self, local_dir: str, s3_prefix: str) -> None:
        """Upload an entire local directory to S3.

        Walks the directory tree and uploads each file, preserving the
        relative path structure under ``s3_prefix``.

        Parameters:
            local_dir: Local directory to upload.
            s3_prefix: S3 URI prefix (e.g. ``"s3://bucket/models/v2.1.0"``).
        """
        try:
            import boto3
        except ImportError:
            logger.warning("boto3 not available; skipping S3 upload")
            return

        s3 = boto3.client("s3", region_name=self._region)
        bucket, prefix = self._parse_s3_uri(s3_prefix)

        for root, _dirs, files in os.walk(local_dir):
            for fname in files:
                local_path = os.path.join(root, fname)
                rel_path = os.path.relpath(local_path, local_dir).replace("\\", "/")
                s3_key = f"{prefix}/{rel_path}" if prefix else rel_path

                s3.upload_file(local_path, bucket, s3_key)

        logger.info(
            "Uploaded %s to s3://%s/%s", local_dir, bucket, prefix,
        )

    def _upload_file_to_s3(self, local_path: str, s3_uri: str) -> None:
        """Upload a single file to S3.

        Parameters:
            local_path: Local file path.
            s3_uri: Full S3 URI (e.g.
                ``"s3://bucket/models/v2.1.0/manifest.json"``).
        """
        try:
            import boto3
        except ImportError:
            logger.warning("boto3 not available; skipping S3 upload")
            return

        s3 = boto3.client("s3", region_name=self._region)
        bucket, key = self._parse_s3_uri(s3_uri)
        s3.upload_file(local_path, bucket, key)

    def _download_from_s3(self, s3_prefix: str, local_dir: str) -> None:
        """Download all objects under an S3 prefix to a local directory.

        Parameters:
            s3_prefix: S3 URI prefix (e.g.
                ``"s3://bucket/models/v2.1.0"``).
            local_dir: Local directory to download into.
        """
        try:
            import boto3
        except ImportError:
            logger.warning("boto3 not available; skipping S3 download")
            return

        s3 = boto3.client("s3", region_name=self._region)
        bucket, prefix = self._parse_s3_uri(s3_prefix)

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel_path = key[len(prefix):].lstrip("/")
                if not rel_path:
                    continue

                local_path = os.path.join(local_dir, rel_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                s3.download_file(bucket, key, local_path)

        logger.info(
            "Downloaded s3://%s/%s to %s", bucket, prefix, local_dir,
        )

    @staticmethod
    def _parse_s3_uri(uri: str) -> Tuple[str, str]:
        """Parse an S3 URI into (bucket, key/prefix).

        Parameters:
            uri: S3 URI (e.g. ``"s3://bucket/path/to/object"``).

        Returns:
            ``(bucket, key)`` tuple.
        """
        path = uri.replace("s3://", "")
        parts = path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key
