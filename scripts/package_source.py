#!/usr/bin/env python3
"""Source code packaging for SageMaker jobs.

Builds a lightweight staging directory or tarball containing only the files
needed for SageMaker Training/Processing jobs.  Replaces the inline staging
logic previously embedded in run_sagemaker_teacher.py.

Usage::
    # As CLI
    python scripts/package_source.py --output outputs/source.tar.gz

    # As library
    from scripts.package_source import build_staging, build_tarball
    staging_dir = build_staging()
    tarball_path = build_tarball()
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("package_source")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories and files to include in the source package
_INCLUDE_DIRS = [
    "core",
    "configs/santander",
    "containers/training",
    "containers/evaluation",
]

# Additional top-level files to include (relative to project root)
_INCLUDE_FILES: list[str] = []

_DEFAULT_STAGING_DIR = PROJECT_ROOT / "outputs" / "_sagemaker_staging"
_DEFAULT_TARBALL_PATH = PROJECT_ROOT / "outputs" / "source.tar.gz"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _is_excluded(path: Path) -> bool:
    """Return True if a file should be excluded from the package."""
    parts = path.parts
    return "__pycache__" in parts or path.suffix == ".pyc"


def _copy_tree(
    src_dir: Path,
    dst_dir: Path,
    project_root: Path,
) -> int:
    """Recursively copy all non-excluded files from *src_dir* into *dst_dir*.

    Returns the number of files copied.
    """
    if not src_dir.is_dir():
        logger.warning("Source directory not found (skipping): %s", src_dir)
        return 0

    count = 0
    for fpath in src_dir.rglob("*"):
        if not fpath.is_file():
            continue
        if _is_excluded(fpath):
            continue
        rel = fpath.relative_to(project_root)
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(fpath), str(dst))
        count += 1
    return count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_staging(
    project_root: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Create a lightweight staging directory for SageMaker jobs.

    Copies:
      - containers/training/train.py  (training entry point)
      - containers/evaluation/        (eval entry point)
      - core/                         (model, training, data, pipeline)
      - configs/santander/            (pipeline.yaml, feature_groups.yaml)

    Excludes __pycache__ and .pyc files.

    Args:
        project_root: Absolute path to the project root.  Defaults to the
            parent of this script's directory.
        output_dir: Destination for the staging directory.  Defaults to
            ``outputs/_sagemaker_staging`` inside the project root.

    Returns:
        Absolute path (str) to the staging directory.
    """
    root = Path(project_root) if project_root else PROJECT_ROOT
    staging = Path(output_dir) if output_dir else _DEFAULT_STAGING_DIR

    # Clean previous staging
    if staging.exists():
        shutil.rmtree(str(staging))

    total_files = 0
    for rel_dir in _INCLUDE_DIRS:
        src = root / rel_dir
        n = _copy_tree(src, staging, root)
        total_files += n
        if n:
            logger.info("  Copied %d files from %s/", n, rel_dir)

    for rel_file in _INCLUDE_FILES:
        src = root / rel_file
        if not src.is_file():
            logger.warning("  SKIP (not found): %s", src)
            continue
        dst = staging / rel_file
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        total_files += 1

    total_size = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file())
    logger.info(
        "Staging complete: %d files, %.1f MB -> %s",
        total_files,
        total_size / (1024 * 1024),
        staging,
    )
    return str(staging)


def build_tarball(
    project_root: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Create a gzipped source tarball for SageMaker jobs.

    Calls :func:`build_staging` first to assemble the staging directory,
    then tars it up.

    Args:
        project_root: Absolute path to the project root.
        output_path: Destination for the .tar.gz file.  Defaults to
            ``outputs/source.tar.gz`` inside the project root.

    Returns:
        Absolute path (str) to the created tarball.
    """
    root = Path(project_root) if project_root else PROJECT_ROOT
    out_path = Path(output_path) if output_path else _DEFAULT_TARBALL_PATH

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build staging dir first
    staging = Path(build_staging(project_root=str(root)))

    logger.info("Creating tarball: %s", out_path)
    with tarfile.open(str(out_path), "w:gz") as tar:
        for fpath in sorted(staging.rglob("*")):
            if not fpath.is_file():
                continue
            arcname = fpath.relative_to(staging)
            tar.add(str(fpath), arcname=str(arcname))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("Tarball ready: %.1f MB -> %s", size_mb, out_path)
    return str(out_path)


def upload_source(
    tarball_path: str,
    s3_bucket: str,
    s3_prefix: str = "source",
) -> str:
    """Upload a source tarball to S3.

    Args:
        tarball_path: Local path to the source.tar.gz file.
        s3_bucket: Destination S3 bucket name.
        s3_prefix: S3 key prefix (e.g. ``"source/teacher_30ep"``).

    Returns:
        The full S3 URI of the uploaded file (``s3://bucket/prefix/source.tar.gz``).
    """
    import boto3

    local = Path(tarball_path)
    if not local.exists():
        raise FileNotFoundError(f"Tarball not found: {tarball_path}")

    s3 = boto3.client("s3")
    s3_key = f"{s3_prefix.rstrip('/')}/{local.name}"
    size_mb = local.stat().st_size / (1024 * 1024)

    logger.info(
        "Uploading %.1f MB -> s3://%s/%s",
        size_mb,
        s3_bucket,
        s3_key,
    )
    s3.upload_file(str(local), s3_bucket, s3_key)

    uri = f"s3://{s3_bucket}/{s3_key}"
    logger.info("Upload complete: %s", uri)
    return uri


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package source code for SageMaker training/evaluation jobs.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Output path for the tarball (default: outputs/source.tar.gz). "
            "Ignored when --staging-only is set."
        ),
    )
    parser.add_argument(
        "--staging-only",
        action="store_true",
        help="Only build the staging directory; do not create a tarball.",
    )
    parser.add_argument(
        "--upload",
        default=None,
        metavar="S3_BUCKET",
        help="Upload the tarball to this S3 bucket after building.",
    )
    parser.add_argument(
        "--s3-prefix",
        default="source",
        metavar="PREFIX",
        help="S3 key prefix for the uploaded file (default: source).",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        metavar="DIR",
        help="Project root directory (default: auto-detected from script location).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.staging_only:
        path = build_staging(project_root=args.project_root)
        logger.info("Staging directory: %s", path)
        return

    tarball = build_tarball(
        project_root=args.project_root,
        output_path=args.output,
    )

    if args.upload:
        upload_source(
            tarball_path=tarball,
            s3_bucket=args.upload,
            s3_prefix=args.s3_prefix,
        )


if __name__ == "__main__":
    main()
