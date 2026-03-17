"""
Lambda handler for model registration (Step Functions → ModelRegistry).

Invoked by the RegisterModel step in training_pipeline.json.
Repackages raw SageMaker Job outputs into the ModelRegistry versioned structure.

Event payload:
    {
        "action": "package_and_promote",
        "teacher_uri": "s3://bucket/models/training-.../output/model.tar.gz",
        "student_uri": "s3://bucket/models/training-.../students/",
        "registry_base": "s3://bucket/models/training-.../artifacts/",
        "version": "v-execution-id",
        "task_name": "financial_recommendation_ple",
        "auto_promote": true
    }

Returns:
    {
        "version": "v-...",
        "students_registered": 14,
        "promoted": true,
        "artifacts_uri": "s3://bucket/.../artifacts/v-.../",
        "manifest_uri": "s3://bucket/.../artifacts/v-.../manifest.json"
    }
"""

from __future__ import annotations

import io
import json
import logging
import os
import tarfile
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda entry point for model registration."""

    action = event.get("action", "package_and_promote")
    version = event.get("version", "unknown")
    teacher_uri = event.get("teacher_uri", "")
    student_uri = event.get("student_uri", "")
    registry_base = event.get("registry_base", "")
    auto_promote = event.get("auto_promote", True)

    logger.info("RegisterModel Lambda: action=%s, version=%s", action, version)

    import boto3
    s3 = boto3.client("s3")

    tmp = tempfile.mkdtemp(prefix="register_")

    try:
        # --- Extract teacher artifacts from model.tar.gz ---
        teacher_state = None
        teacher_config = None
        training_metrics = {}

        if teacher_uri:
            teacher_state, teacher_config, training_metrics = _extract_teacher(
                s3, teacher_uri, tmp,
            )

        # --- List student models from S3 ---
        students = {}
        student_metadata = {}

        if student_uri:
            students, student_metadata = _list_students(s3, student_uri)

        # --- Build version directory locally ---
        version_dir = Path(tmp) / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Teacher dir
        teacher_dir = version_dir / "teacher"
        teacher_dir.mkdir(exist_ok=True)

        if teacher_state is not None:
            # Save state_dict
            import torch
            torch.save(teacher_state, str(teacher_dir / "model.pth"))

        if teacher_config:
            with open(teacher_dir / "config.json", "w") as f:
                json.dump(teacher_config, f, indent=2, default=str)

        with open(teacher_dir / "training_metrics.json", "w") as f:
            json.dump(training_metrics, f, indent=2, default=str)

        # Student dirs
        students_dir = version_dir / "students"
        students_dir.mkdir(exist_ok=True)

        for task_name, model_s3_uri in students.items():
            task_dir = students_dir / task_name
            task_dir.mkdir(exist_ok=True)

            # Download model.lgbm
            _download_s3_file(s3, model_s3_uri, str(task_dir / "model.lgbm"))

            # Save metadata
            meta = student_metadata.get(task_name, {"task_name": task_name})
            meta["teacher_version"] = version
            meta["teacher_uri"] = teacher_uri
            with open(task_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2, default=str)

            # Copy selected_features.json if exists in raw output
            _try_copy_s3_file(
                s3, student_uri, task_name, "selected_features.json",
                str(task_dir / "selected_features.json"),
            )

            # Copy fidelity.json if exists
            _try_copy_s3_file(
                s3, student_uri, task_name, "fidelity.json",
                str(task_dir / "fidelity.json"),
            )

        # --- Build manifest ---
        fidelity_passed = 0
        fidelity_failed = 0
        feature_summary = {}

        for task_name in students:
            task_dir = students_dir / task_name
            fid_path = task_dir / "fidelity.json"
            if fid_path.exists():
                with open(fid_path) as f:
                    fid = json.load(f)
                if fid.get("passed", True):
                    fidelity_passed += 1
                else:
                    fidelity_failed += 1

            feat_path = task_dir / "selected_features.json"
            if feat_path.exists():
                with open(feat_path) as f:
                    feat = json.load(f)
                feature_summary[task_name] = feat.get("count", 0)

        from datetime import datetime, timezone
        import hashlib

        manifest = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "teacher_config_hash": hashlib.sha256(
                json.dumps(teacher_config or {}, sort_keys=True).encode()
            ).hexdigest()[:16],
            "teacher_metrics": training_metrics,
            "student_tasks": list(students.keys()),
            "fidelity_summary": {
                "passed": fidelity_passed,
                "failed": fidelity_failed,
            },
            "feature_selection_summary": feature_summary,
            "promoted": False,
            "promoted_at": None,
            "metadata": {
                "teacher_uri": teacher_uri,
                "student_uri": student_uri,
                "task_name": event.get("task_name", ""),
            },
        }

        with open(version_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        # --- Upload to S3 registry ---
        registry_prefix = registry_base.rstrip("/") + f"/{version}/"
        _upload_dir_to_s3(s3, str(version_dir), registry_prefix)

        logger.info(
            "Registered %s: %d students, fidelity %d/%d passed",
            version, len(students), fidelity_passed,
            fidelity_passed + fidelity_failed,
        )

        # --- Auto-promote if all fidelity passed ---
        promoted = False
        if auto_promote and fidelity_failed == 0:
            _promote_version(s3, registry_base, version)
            manifest["promoted"] = True
            manifest["promoted_at"] = datetime.now(timezone.utc).isoformat()
            # Re-upload manifest
            _upload_json(
                s3, manifest,
                registry_prefix + "manifest.json",
            )
            promoted = True
            logger.info("Auto-promoted %s to champion", version)

        return {
            "version": version,
            "students_registered": len(students),
            "promoted": promoted,
            "fidelity": {"passed": fidelity_passed, "failed": fidelity_failed},
            "artifacts_uri": registry_prefix,
            "manifest_uri": registry_prefix + "manifest.json",
        }

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_teacher(s3, teacher_uri, tmp_dir):
    """Extract teacher model from model.tar.gz."""
    state_dict = None
    config = None
    metrics = {}

    try:
        parts = teacher_uri.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]

        buf = io.BytesIO()
        s3.download_fileobj(bucket, key, buf)
        buf.seek(0)

        extract_dir = Path(tmp_dir) / "teacher_raw"
        extract_dir.mkdir(exist_ok=True)

        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            tar.extractall(str(extract_dir))

        # Load torch artifacts
        model_path = extract_dir / "model.pth"
        if model_path.exists():
            import torch
            state_dict = torch.load(str(model_path), map_location="cpu")

        config_path = extract_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        metrics_path = extract_dir / "training_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

    except Exception as e:
        logger.warning("Teacher extraction failed: %s", e)

    return state_dict, config, metrics


def _list_students(s3, student_uri):
    """List student model files from S3 prefix."""
    students = {}
    metadata = {}

    parts = student_uri.replace("s3://", "").split("/", 1)
    bucket, prefix = parts[0], parts[1]

    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(prefix):]
                parts_rel = rel.strip("/").split("/")

                if len(parts_rel) >= 2:
                    task_name = parts_rel[0]
                    filename = parts_rel[-1]

                    if filename == "model.lgbm":
                        students[task_name] = f"s3://{bucket}/{key}"
                    elif filename == "metadata.json":
                        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                        metadata[task_name] = json.loads(body.decode())
    except Exception as e:
        logger.warning("Student listing failed: %s", e)

    return students, metadata


def _download_s3_file(s3, s3_uri, local_path):
    """Download a single S3 file."""
    parts = s3_uri.replace("s3://", "").split("/", 1)
    s3.download_file(parts[0], parts[1], local_path)


def _try_copy_s3_file(s3, base_uri, task_name, filename, local_path):
    """Try to download a file from S3; silently skip if not found."""
    try:
        uri = base_uri.rstrip("/") + f"/{task_name}/{filename}"
        _download_s3_file(s3, uri, local_path)
    except Exception:
        pass


def _upload_dir_to_s3(s3, local_dir, s3_prefix):
    """Upload an entire local directory to S3."""
    parts = s3_prefix.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    for root, dirs, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir)
            s3_key = prefix.rstrip("/") + "/" + rel_path.replace("\\", "/")
            s3.upload_file(local_path, bucket, s3_key)


def _upload_json(s3, data, s3_uri):
    """Upload a JSON dict to S3."""
    parts = s3_uri.replace("s3://", "").split("/", 1)
    body = json.dumps(data, indent=2, default=str).encode("utf-8")
    s3.put_object(Bucket=parts[0], Key=parts[1], Body=body)


def _promote_version(s3, registry_base, version):
    """Write a _promoted marker file pointing to the active version."""
    marker_uri = registry_base.rstrip("/") + "/_promoted"
    _upload_json(s3, {"active_version": version}, marker_uri)
