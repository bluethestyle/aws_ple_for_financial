"""
PII Encryptor -- SHA256 hashing with domain-specific salts.

One-way encryption: SHA256(salt + value) -> 32-byte digest.
No decryption. Downstream code maps the digest to a compact INT32
identifier via :class:`PIIIntegerIndexer`.

No pandas dependency (CLAUDE.md §3.3). Inputs are dict-of-lists
(column name -> list of raw values); outputs follow the same shape. The
legacy ``pd.Series`` / ``pd.DataFrame`` aliases that existed before
have been removed because the Lambda runtime does not ship pandas,
which caused ``PIIEncryptor unavailable`` warnings at production
cold-start.
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .domains import PIIDomain, resolve_domain
from .salt_manager import SaltManager

logger = logging.getLogger(__name__)

# One 32-byte null sentinel reused for every NaN/None input so the
# indexer can collapse them all to id = -1 downstream.
_NULL_DIGEST = b"\x00" * 32


def _is_null(value: Any) -> bool:
    """Pandas-free NaN detection.

    Accepts the standard null markers (``None``, float NaN, empty
    string); additional sentinels such as pandas' ``NaT`` are treated
    as truthy values because their ``__repr__`` is stable and rare in
    security-sensitive columns.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


class PIIEncryptor:
    def __init__(self, salt_manager: SaltManager):
        self._salt_manager = salt_manager

    # ------------------------------------------------------------------
    # Scalar
    # ------------------------------------------------------------------

    def hash_value(self, value: Any, domain: PIIDomain) -> bytes:
        """SHA256(salt + str(value)) -> 32-byte digest."""
        if _is_null(value):
            return _NULL_DIGEST
        salt = self._salt_manager.get_salt(domain)
        return hashlib.sha256(salt + str(value).encode("utf-8")).digest()

    # ------------------------------------------------------------------
    # Sequence
    # ------------------------------------------------------------------

    def hash_column(
        self,
        values: Iterable[Any],
        domain: PIIDomain,
    ) -> List[bytes]:
        """Vectorised hash over any iterable of raw values.

        Returns a list of 32-byte digests with the same length as the
        input.
        """
        salt = self._salt_manager.get_salt(domain)
        out: List[bytes] = []
        for v in values:
            if _is_null(v):
                out.append(_NULL_DIGEST)
            else:
                out.append(
                    hashlib.sha256(salt + str(v).encode("utf-8")).digest()
                )
        return out

    # ------------------------------------------------------------------
    # Columnar dict
    # ------------------------------------------------------------------

    def hash_columns(
        self,
        columns: Dict[str, Sequence[Any]],
        column_domain_map: Dict[str, PIIDomain],
    ) -> Dict[str, Sequence[Any]]:
        """Hash the specified PII columns in a ``{col: list}`` payload.

        Every column listed in ``column_domain_map`` is replaced with a
        new ``{col}_hashed`` column carrying the raw 32-byte digests; the
        original column is dropped. Columns not present in the payload
        are silently ignored so callers can pass a permissive schema.
        """
        result: Dict[str, Sequence[Any]] = dict(columns)
        for col, domain in column_domain_map.items():
            if col not in result:
                continue
            result[f"{col}_hashed"] = self.hash_column(result[col], domain)
            del result[col]
            logger.debug("Hashed column '%s' (domain=%s)", col, domain.value)
        return result

    def detect_and_hash_columns(
        self,
        columns: Dict[str, Sequence[Any]],
        pii_columns: Optional[List[str]] = None,
    ) -> Dict[str, Sequence[Any]]:
        """Auto-detect PII columns via :func:`resolve_domain` and hash.

        Columns that resolve to :attr:`PIIDomain.DEFAULT` are skipped.
        """
        if pii_columns is None:
            pii_columns = [
                c for c in columns
                if resolve_domain(c) != PIIDomain.DEFAULT
            ]
        column_domain_map = {c: resolve_domain(c) for c in pii_columns}
        return self.hash_columns(columns, column_domain_map)

    # ------------------------------------------------------------------
    # Single-row helpers (used by the Lambda predict inbound scan)
    # ------------------------------------------------------------------

    def hash_row(
        self,
        row: Dict[str, Any],
        column_domain_map: Dict[str, PIIDomain],
    ) -> Dict[str, Any]:
        """Hash a single-row dict in place.

        Convenience wrapper around :meth:`hash_columns` that avoids
        manual list/scalar plumbing for the serving path, which sees one
        customer at a time.
        """
        columns = {k: [v] for k, v in row.items()}
        hashed = self.hash_columns(columns, column_domain_map)
        return {k: (v[0] if isinstance(v, list) else v) for k, v in hashed.items()}
