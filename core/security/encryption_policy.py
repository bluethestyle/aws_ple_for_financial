"""
Encryption Policy -- column-level encryption rules.

Derives policies from SchemaRegistry pii tags or explicit YAML config.
Each policy specifies: column name, PII domain, action (hash_and_index / hash_only / drop).
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .domains import PIIDomain, resolve_domain

logger = logging.getLogger(__name__)


@dataclass
class ColumnEncryptionPolicy:
    """Encryption rule for a single column."""

    column_name: str
    domain: PIIDomain
    action: str = "hash_and_index"  # "hash_and_index" | "hash_only" | "drop"


@dataclass
class SourceEncryptionPolicy:
    """Encryption rules for all PII columns within a single data source."""

    source_name: str
    policies: Dict[str, ColumnEncryptionPolicy] = field(default_factory=dict)

    def get_columns_to_hash(self) -> List[str]:
        """Return columns that need SHA-256 hashing."""
        return [
            p.column_name
            for p in self.policies.values()
            if p.action in ("hash_and_index", "hash_only")
        ]

    def get_columns_to_index(self) -> List[str]:
        """Return columns that need integer indexing after hashing."""
        return [
            p.column_name
            for p in self.policies.values()
            if p.action == "hash_and_index"
        ]

    def get_columns_to_drop(self) -> List[str]:
        """Return columns that should be dropped entirely."""
        return [
            p.column_name
            for p in self.policies.values()
            if p.action == "drop"
        ]

    def get_column_domain_map(self) -> Dict[str, PIIDomain]:
        """Return {column_name: PIIDomain} for columns that are hashed."""
        return {
            p.column_name: p.domain
            for p in self.policies.values()
            if p.action in ("hash_and_index", "hash_only")
        }


# ── Factory functions ─────────────────────────────────────────────────


def derive_from_schema(schema_registry) -> Dict[str, SourceEncryptionPolicy]:
    """Auto-generate encryption policies from SchemaRegistry pii tags.

    For each source in the registry, finds columns with ``pii=True``
    and assigns domains via :func:`resolve_domain`.

    Parameters
    ----------
    schema_registry : SchemaRegistry
        A populated schema registry (from ``core.data.schema_registry``).

    Returns
    -------
    dict[str, SourceEncryptionPolicy]
        Mapping ``{source_name: SourceEncryptionPolicy}``.
    """
    policies: Dict[str, SourceEncryptionPolicy] = {}

    all_pii = schema_registry.all_pii_columns()
    for source_name, pii_cols in all_pii.items():
        source_policy = SourceEncryptionPolicy(source_name=source_name)
        for col_name in pii_cols:
            domain = resolve_domain(col_name)
            # Contact info (phone, email) should be dropped, not indexed
            action = "drop" if domain == PIIDomain.CONTACT else "hash_and_index"
            # Personal IDs (SSN) should also be dropped
            if domain == PIIDomain.PERSONAL_ID:
                action = "drop"
            source_policy.policies[col_name] = ColumnEncryptionPolicy(
                column_name=col_name,
                domain=domain,
                action=action,
            )
        if source_policy.policies:
            policies[source_name] = source_policy
            logger.debug(
                "Derived %d encryption policies for source '%s'",
                len(source_policy.policies),
                source_name,
            )

    logger.info(
        "Derived encryption policies for %d sources (%d total columns)",
        len(policies),
        sum(len(p.policies) for p in policies.values()),
    )
    return policies


def load_encryption_config(path: str) -> Dict[str, Any]:
    """Load encryption configuration from YAML.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
