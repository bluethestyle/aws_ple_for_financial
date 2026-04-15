"""
SageMaker / Local path resolver — single source of truth.

Resolves relative config/data paths to the correct absolute path
regardless of whether we're running inside SageMaker (/opt/ml/code)
or locally (CWD).

Usage::

    from containers.path_resolver import resolve_config_path

    config_path = resolve_config_path("configs/pipeline.yaml")
    # SageMaker → /opt/ml/code/configs/pipeline.yaml
    # Local     → ./configs/pipeline.yaml
"""

from __future__ import annotations

import os
from pathlib import Path

# SageMaker extracts source_dir to /opt/ml/code
_SM_CODE_DIR = Path("/opt/ml/code")


def is_sagemaker() -> bool:
    """Detect if running inside a SageMaker container."""
    return (
        os.environ.get("SM_MODEL_DIR") is not None
        or _SM_CODE_DIR.exists()
    )


def resolve_config_path(relative_path: str) -> str:
    """Resolve a relative config path to an absolute path.

    Search order:
      1. CWD / relative_path (works locally and in SageMaker if CWD is /opt/ml/code)
      2. /opt/ml/code / relative_path (SageMaker fallback)
      3. Return as-is (let caller handle FileNotFoundError)
    """
    p = Path(relative_path)

    # Already absolute and exists
    if p.is_absolute() and p.exists():
        return str(p)

    # Try CWD first
    cwd_path = Path.cwd() / relative_path
    if cwd_path.exists():
        return str(cwd_path)

    # Try SageMaker code dir
    sm_path = _SM_CODE_DIR / relative_path
    if sm_path.exists():
        return str(sm_path)

    # Return CWD-relative (caller will get a clear error)
    return str(cwd_path)


def resolve_data_path(env_var: str, fallback: str = "") -> str:
    """Resolve a SageMaker data channel path.

    Args:
        env_var: SM env var name (e.g., "SM_CHANNEL_TRAIN").
        fallback: Local fallback path.

    Returns:
        Resolved directory path.
    """
    return os.environ.get(env_var, fallback)
