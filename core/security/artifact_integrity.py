"""Model / data artifact integrity verification (금융분야 AI 가이드라인 ⑦ 보안성).

External or upstream model artifacts (checkpoints, teacher weights, tarballs,
calibrators) are deserialized via ``torch.load`` / ``pickle`` / ``joblib`` —
all of which are arbitrary-code-execution (RCE) vectors when the artifact is
tampered with. This module provides the supply-chain integrity primitives the
guideline's "외부 모델·데이터 검증" requirement calls for:

* ``write_sidecar`` — emit a ``<file>.sha256`` digest next to an artifact on
  save, so downstream loads can verify it.
* ``verify_artifact`` — fail *closed* on checksum mismatch. Resolution order:
  explicit expected hash > sidecar file > (none → warn and allow, for
  backward compatibility with artifacts produced before sidecars existed).
* ``load_verified_torch`` — verify, then ``torch.load`` (defaulting to the
  safe ``weights_only=True`` deserializer).
* ``safe_extract_tar`` — reject path-traversal members before extraction.

The sha256 gate is format-agnostic and non-breaking: it closes the tamper
vector even where ``weights_only=True`` cannot be used (pickled non-tensor
payloads). Tightening individual load sites to ``weights_only=True`` is a
separate, staged change.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CHUNK = 1 << 20  # 1 MiB streaming read

__all__ = [
    "IntegrityError",
    "sha256_file",
    "sidecar_path",
    "write_sidecar",
    "read_sidecar",
    "verify_artifact",
    "load_verified_torch",
    "safe_extract_tar",
]


class IntegrityError(Exception):
    """Raised when an artifact fails integrity verification (fail-closed)."""


def sha256_file(path: Any) -> str:
    """Return the hex SHA-256 of a file, read in streaming chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar_path(path: Any) -> str:
    """Return the conventional ``<path>.sha256`` sidecar location."""
    return f"{os.fspath(path)}.sha256"


def write_sidecar(path: Any) -> str:
    """Compute the artifact digest and write it to ``<path>.sha256``.

    Returns the hex digest. Call this immediately after writing an artifact
    (``torch.save`` / ``pickle.dump`` / ``joblib.dump``).
    """
    digest = sha256_file(path)
    with open(sidecar_path(path), "w", encoding="utf-8") as fh:
        fh.write(digest + "\n")
    return digest


def read_sidecar(path: Any) -> Optional[str]:
    """Read the digest from ``<path>.sha256`` if present, else ``None``."""
    sp = sidecar_path(path)
    if not os.path.exists(sp):
        return None
    with open(sp, "r", encoding="utf-8") as fh:
        content = fh.read().strip()
    return content.split()[0] if content else None


def verify_artifact(path: Any, expected_sha256: Optional[str] = None) -> None:
    """Verify a file's integrity, failing *closed* on mismatch.

    Resolution order for the expected digest: the explicit ``expected_sha256``
    argument, then a ``<path>.sha256`` sidecar. When neither is available the
    artifact is loaded unverified (with a warning) so that artifacts produced
    before sidecars existed still load.

    Raises:
        IntegrityError: if a known expected digest does not match.
    """
    expected = expected_sha256 or read_sidecar(path)
    if expected is None:
        logger.warning(
            "artifact_integrity: no checksum available for %s; loading unverified",
            path,
        )
        return
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise IntegrityError(
            f"artifact integrity check failed for {os.fspath(path)}: "
            f"expected {expected}, got {actual}"
        )
    logger.debug("artifact_integrity: verified %s", path)


def load_verified_torch(
    path: Any,
    *,
    expected_sha256: Optional[str] = None,
    weights_only: bool = True,
    map_location: Any = "cpu",
    **kwargs: Any,
) -> Any:
    """Verify an artifact's checksum, then ``torch.load`` it.

    Defaults to ``weights_only=True`` (the safe deserializer). Pass
    ``weights_only=False`` only for trusted artifacts that carry non-tensor
    payloads — the checksum gate still applies.
    """
    import torch  # local import: keep torch off the import path of consumers

    verify_artifact(path, expected_sha256)
    return torch.load(
        path, map_location=map_location, weights_only=weights_only, **kwargs
    )


def safe_extract_tar(tar: Any, dest: Any) -> None:
    """Extract a tarfile after rejecting path-traversal members.

    Guards against ``../`` and absolute-path members that would write outside
    ``dest`` (a classic tarball escape used for RCE / overwrite attacks).

    Raises:
        IntegrityError: if any member resolves outside ``dest``.
    """
    dest_abs = os.path.abspath(os.fspath(dest))
    for member in tar.getmembers():
        target = os.path.abspath(os.path.join(dest_abs, member.name))
        if target != dest_abs and not target.startswith(dest_abs + os.sep):
            raise IntegrityError(
                f"unsafe tar member path (path traversal): {member.name}"
            )
    # Defense-in-depth: the stdlib 'data' filter (Python 3.12+) also blocks
    # unsafe members; fall back gracefully on older runtimes.
    try:
        tar.extractall(dest_abs, filter="data")
    except TypeError:
        tar.extractall(dest_abs)
