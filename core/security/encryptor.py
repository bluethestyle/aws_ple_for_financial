"""
PII Encryptor -- SHA256 hashing with domain-specific salts.

One-way encryption: SHA256(salt + value) -> 32-byte digest.
No decryption. Integer indices are used for downstream operations.
"""
import hashlib, logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from .domains import PIIDomain, resolve_domain
from .salt_manager import SaltManager

logger = logging.getLogger(__name__)

class PIIEncryptor:
    def __init__(self, salt_manager: SaltManager):
        self._salt_manager = salt_manager

    def hash_value(self, value: Any, domain: PIIDomain) -> bytes:
        """SHA256(salt + str(value)) -> 32-byte digest."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return b'\x00' * 32  # null sentinel
        salt = self._salt_manager.get_salt(domain)
        return hashlib.sha256(salt + str(value).encode("utf-8")).digest()

    def hash_column(self, series: pd.Series, domain: PIIDomain) -> pd.Series:
        """Vectorized hashing of an entire column."""
        salt = self._salt_manager.get_salt(domain)
        def _hash(val):
            if pd.isna(val):
                return b'\x00' * 32
            return hashlib.sha256(salt + str(val).encode("utf-8")).digest()
        return series.map(_hash)

    def hash_dataframe(self, df: pd.DataFrame,
                       column_domain_map: Dict[str, PIIDomain]) -> pd.DataFrame:
        """Hash all specified PII columns in a DataFrame.

        Original columns are replaced with hashed versions.
        Column names get '_hashed' suffix.
        """
        result = df.copy()
        for col, domain in column_domain_map.items():
            if col not in result.columns:
                continue
            hashed = self.hash_column(result[col], domain)
            result[f"{col}_hashed"] = hashed
            result = result.drop(columns=[col])
            logger.debug("Hashed column '%s' (domain=%s)", col, domain.value)
        return result

    def detect_and_hash(self, df: pd.DataFrame,
                        pii_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Auto-detect PII columns and hash them.

        If pii_columns not provided, uses resolve_domain() on all columns.
        Columns that resolve to DEFAULT domain are skipped.
        """
        if pii_columns is None:
            pii_columns = [c for c in df.columns if resolve_domain(c) != PIIDomain.DEFAULT]

        column_domain_map = {c: resolve_domain(c) for c in pii_columns}
        return self.hash_dataframe(df, column_domain_map)
