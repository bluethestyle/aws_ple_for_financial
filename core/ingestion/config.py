"""
Ingestion Configuration -- dataclasses for domain source configs.

Loads from YAML (``configs/financial/ingestion.yaml``) and provides
typed access to S3 paths, file patterns, and enabled/disabled flags
for each domain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DomainSourceConfig:
    """Configuration for a single domain data source."""

    name: str
    s3_input_path: str = ""
    s3_output_path: str = ""
    file_pattern: str = "*.parquet"
    enabled: bool = True

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DomainSourceConfig(name={self.name!r}, "
            f"enabled={self.enabled}, "
            f"input={self.s3_input_path!r})"
        )


@dataclass
class IngestionConfig:
    """Top-level ingestion configuration.

    Holds per-domain source configs, schema/encryption paths, and the
    output S3 base prefix.

    Parameters
    ----------
    domains : dict[str, DomainSourceConfig]
        Mapping from domain name to its source configuration.
    schema_path : str
        Path to the schema YAML (for SchemaRegistry).
    encryption_config_path : str
        Path to the encryption YAML (for EncryptionPipeline).
    output_base_s3 : str
        Base S3 prefix for ingested output.
    """

    domains: Dict[str, DomainSourceConfig] = field(default_factory=dict)
    schema_path: str = "configs/financial/schema.yaml"
    encryption_config_path: str = ""
    output_base_s3: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> "IngestionConfig":
        """Load ingestion config from a YAML file.

        Expected YAML structure::

            output_base_s3: s3://bucket/ingested/
            schema_path: configs/financial/schema.yaml
            encryption_config_path: configs/financial/encryption.yaml
            domains:
              customer_master:
                s3_input_path: s3://bucket/raw/customer_master/
                enabled: true
              account:
                s3_input_path: s3://bucket/raw/account/
                enabled: true

        Parameters
        ----------
        path : str
            Path to the YAML config file.

        Returns
        -------
        IngestionConfig
        """
        import yaml

        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Ingestion config not found: {path}")

        with open(p, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        if raw is None:
            logger.warning("Empty ingestion config: %s", path)
            return cls()

        domains: Dict[str, DomainSourceConfig] = {}
        for domain_name, domain_def in raw.get("domains", {}).items():
            if isinstance(domain_def, dict):
                domains[domain_name] = DomainSourceConfig(
                    name=domain_name,
                    s3_input_path=domain_def.get("s3_input_path", ""),
                    s3_output_path=domain_def.get("s3_output_path", ""),
                    file_pattern=domain_def.get("file_pattern", "*.parquet"),
                    enabled=domain_def.get("enabled", True),
                )
            else:
                # Simple string: treat as s3_input_path
                domains[domain_name] = DomainSourceConfig(
                    name=domain_name,
                    s3_input_path=str(domain_def) if domain_def else "",
                )

        config = cls(
            domains=domains,
            schema_path=raw.get("schema_path", "configs/financial/schema.yaml"),
            encryption_config_path=raw.get("encryption_config_path", ""),
            output_base_s3=raw.get("output_base_s3", ""),
        )

        logger.info(
            "Loaded ingestion config from %s: %d domains (%d enabled)",
            path, len(domains), len(config.get_enabled_domains()),
        )
        return config

    def get_enabled_domains(self) -> List[DomainSourceConfig]:
        """Return list of enabled domain source configs."""
        return [d for d in self.domains.values() if d.enabled]

    def get_domain(self, name: str) -> Optional[DomainSourceConfig]:
        """Return config for a specific domain, or ``None``."""
        return self.domains.get(name)

    def get_input_path(self, domain_name: str) -> str:
        """Return the S3 input path for a domain.

        Parameters
        ----------
        domain_name : str
            Domain name.

        Returns
        -------
        str
            S3 input path, or empty string if not configured.
        """
        domain = self.domains.get(domain_name)
        return domain.s3_input_path if domain else ""

    def get_output_path(self, domain_name: str) -> str:
        """Return the S3 output path for a domain.

        Uses the domain-specific ``s3_output_path`` if set, otherwise
        constructs from ``output_base_s3 + domain_name/``.

        Parameters
        ----------
        domain_name : str
            Domain name.

        Returns
        -------
        str
            S3 output path.
        """
        domain = self.domains.get(domain_name)
        if domain and domain.s3_output_path:
            return domain.s3_output_path
        if self.output_base_s3:
            base = self.output_base_s3.rstrip("/")
            return f"{base}/{domain_name}/"
        return ""

    def __repr__(self) -> str:  # pragma: no cover
        enabled = len(self.get_enabled_domains())
        return (
            f"IngestionConfig(domains={len(self.domains)}, "
            f"enabled={enabled}, "
            f"output_base={self.output_base_s3!r})"
        )
