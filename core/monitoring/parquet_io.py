"""s3://-aware Parquet row I/O for compliance archives (S7 fairness, S8 drift).

The fairness/drift archives are configured with ``s3://`` URIs in
pipeline.yaml, but the write side (FairnessMonitor._flush_parquet) and read
side (build_fairness_archive_source) used ``pathlib.Path`` directly. On a
local FS ``Path('s3://bucket/key')`` mangles the URI (``s3:\\bucket\\key`` on
Windows), ``exists()`` is always False, and the Lambda read-only FS makes the
local write raise — so the archive silently dropped to a 0.5 fallback in the
PromotionGate.

This module routes both sides through ``pyarrow.fs``, which natively resolves
``s3://`` (S3FileSystem) and local paths (LocalFileSystem), so the same row
helpers work in both environments. Small archives only — these read/rewrite
the whole file (append-by-rewrite), matching the prior behaviour.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _resolve(path: str):
    """Return ``(filesystem, normalized_path)`` for an s3:// or local path."""
    from pyarrow import fs as pafs

    if path.startswith("s3://"):
        # FileSystem.from_uri returns (S3FileSystem, 'bucket/key')
        filesystem, normalized = pafs.FileSystem.from_uri(path)
        return filesystem, normalized
    # Local path — keep it as given (absolute or relative) for LocalFileSystem.
    return pafs.LocalFileSystem(), path


def parquet_exists(path: str) -> bool:
    """True if a Parquet object exists at ``path`` (s3:// or local)."""
    try:
        from pyarrow import fs as pafs

        filesystem, normalized = _resolve(path)
        info = filesystem.get_file_info(normalized)
        return info.type != pafs.FileType.NotFound
    except Exception:
        logger.debug("parquet_exists check failed for %s", path, exc_info=True)
        return False


def read_parquet_rows(path: str) -> List[Dict[str, Any]]:
    """Read all rows from a Parquet file as a list of dicts.

    Returns ``[]`` if the file is missing or unreadable (callers treat an
    empty archive as "no data" and fall back).
    """
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        logger.debug("pyarrow not installed; read_parquet_rows returns []")
        return []
    try:
        if not parquet_exists(path):
            return []
        filesystem, normalized = _resolve(path)
        table = pq.read_table(normalized, filesystem=filesystem)
        return table.to_pylist()
    except Exception:
        logger.exception("failed to read parquet archive %s", path)
        return []


def write_parquet_rows(rows: List[Dict[str, Any]], path: str) -> bool:
    """Write ``rows`` to a Parquet file, creating parent dirs as needed.

    Returns True on success, False on failure (logged). Overwrites the whole
    file (append-by-rewrite); callers pass existing + new rows.
    """
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        logger.warning("pyarrow not installed; cannot write parquet %s", path)
        return False
    try:
        import os

        filesystem, normalized = _resolve(path)
        # Ensure the parent "directory" exists. Use os.path.dirname so a
        # Windows backslash path resolves its parent correctly (a plain
        # rsplit('/') misses '\\' separators). S3FileSystem.create_dir is a
        # prefix no-op; LocalFileSystem creates the folder.
        parent = os.path.dirname(normalized)
        if parent:
            try:
                filesystem.create_dir(parent, recursive=True)
            except Exception:
                logger.debug("create_dir skipped for %s", parent, exc_info=True)
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, normalized, filesystem=filesystem)
        return True
    except Exception:
        logger.exception("failed to write parquet archive %s", path)
        return False
