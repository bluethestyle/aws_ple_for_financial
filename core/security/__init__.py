"""
Core Security Module
====================

PII hashing with domain-specific salts via AWS Secrets Manager.

Modules:
    domains           -- PIIDomain enum and column-to-domain resolution
    salt_manager      -- Salt retrieval from Secrets Manager (prod) or local (dev)
    encryptor         -- SHA256 one-way hashing of PII columns
    integer_indexer   -- PII integer indexing
    encryption_policy -- Column/source encryption policy derivation
    pipeline          -- Top-level encryption orchestrator
"""

from .domains import PIIDomain, COLUMN_DOMAIN_MAP, resolve_domain
from .salt_manager import SaltManager, LocalSaltManager
from .encryptor import PIIEncryptor
from .integer_indexer import PIIIntegerIndexer
from .encryption_policy import (
    ColumnEncryptionPolicy,
    SourceEncryptionPolicy,
    derive_from_schema,
    load_encryption_config,
)
from .pipeline import EncryptionPipeline
from .prompt_sanitizer import PromptSanitizer, SanitizeResult, Sensitivity

__all__ = [
    "PIIDomain",
    "COLUMN_DOMAIN_MAP",
    "resolve_domain",
    "SaltManager",
    "LocalSaltManager",
    "PIIEncryptor",
    "PIIIntegerIndexer",
    "ColumnEncryptionPolicy",
    "SourceEncryptionPolicy",
    "derive_from_schema",
    "load_encryption_config",
    "EncryptionPipeline",
    "PromptSanitizer",
    "SanitizeResult",
    "Sensitivity",
]
