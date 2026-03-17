"""
Ingestion Runner -- orchestrates running multiple domain ingestors.

Usage::

    from core.ingestion.config import IngestionConfig
    from core.ingestion.runner import IngestionRunner

    config = IngestionConfig.from_yaml("configs/financial/ingestion.yaml")
    runner = IngestionRunner(config)

    # Run all enabled domains
    results = runner.run_all()

    # Run a single domain
    result = runner.run_domain("customer_master")

    # Generate manifest
    manifest = runner.generate_manifest(results)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import IngestionResult
from .config import IngestionConfig
from .registry import DomainRegistry

logger = logging.getLogger(__name__)


class IngestionRunner:
    """Orchestrator for running multiple domain ingestors.

    Parameters
    ----------
    config : IngestionConfig
        Ingestion configuration with domain paths and settings.
    schema_registry : SchemaRegistry, optional
        Schema registry instance for validation.
    encryption_pipeline : EncryptionPipeline, optional
        Encryption pipeline instance for PII processing.
    """

    def __init__(
        self,
        config: IngestionConfig,
        schema_registry: Optional[Any] = None,
        encryption_pipeline: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._schema_registry = schema_registry
        self._encryption_pipeline = encryption_pipeline

        # Import all domain modules to trigger registration.
        # This mirrors the generator pattern where importing the
        # subpackage activates @register decorators.
        try:
            from . import domains as _domains  # noqa: F401
        except ImportError:
            logger.debug(
                "No 'domains' subpackage found; "
                "domain ingestors must be registered manually."
            )

    def run_domain(self, domain_name: str) -> IngestionResult:
        """Run a single domain ingestor.

        Parameters
        ----------
        domain_name : str
            Name of the domain to ingest (must be registered in
            :class:`DomainRegistry`).

        Returns
        -------
        IngestionResult

        Raises
        ------
        KeyError
            If the domain is not registered.
        ValueError
            If the domain is not configured or has no input path.
        """
        if not DomainRegistry.is_registered(domain_name):
            raise KeyError(
                f"Domain '{domain_name}' is not registered. "
                f"Available: {DomainRegistry.list_available()}"
            )

        domain_config = self._config.get_domain(domain_name)
        if domain_config is None:
            raise ValueError(
                f"Domain '{domain_name}' has no configuration. "
                f"Add it to ingestion.yaml."
            )

        input_path = domain_config.s3_input_path
        if not input_path:
            raise ValueError(
                f"Domain '{domain_name}' has no s3_input_path configured."
            )

        output_path = self._config.get_output_path(domain_name)

        # Create ingestor instance
        ingestor = DomainRegistry.create(
            domain_name,
            config=self._config.domains.get(domain_name).__dict__
            if self._config.domains.get(domain_name)
            else {},
            schema_registry=self._schema_registry,
            encryption_pipeline=self._encryption_pipeline,
        )

        logger.info(
            "Running domain ingestor: %s (input=%s, output=%s)",
            domain_name, input_path, output_path,
        )

        return ingestor.run(input_path, output_path)

    def run_all(
        self,
        domains: Optional[List[str]] = None,
    ) -> List[IngestionResult]:
        """Run all enabled domains sequentially.

        Parameters
        ----------
        domains : list[str], optional
            Subset of domain names to run.  When ``None``, all enabled
            domains from the config are run (if they are registered).

        Returns
        -------
        list[IngestionResult]
            One result per domain, in execution order.
        """
        if domains is not None:
            domain_names = domains
        else:
            domain_names = [
                d.name for d in self._config.get_enabled_domains()
            ]

        results: List[IngestionResult] = []
        total_start = time.time()

        logger.info(
            "Starting ingestion for %d domains: %s",
            len(domain_names), domain_names,
        )

        for name in domain_names:
            if not DomainRegistry.is_registered(name):
                logger.warning(
                    "Domain '%s' is enabled but not registered; skipping.",
                    name,
                )
                continue

            try:
                result = self.run_domain(name)
                results.append(result)
            except Exception as exc:
                logger.exception(
                    "Failed to ingest domain '%s': %s", name, exc,
                )
                results.append(
                    IngestionResult(
                        source_name=name,
                        row_count=0,
                        column_count=0,
                        output_path="",
                        duration_seconds=0.0,
                        validation_passed=False,
                        validation_warnings=[f"Ingestion failed: {exc}"],
                    )
                )

        total_elapsed = time.time() - total_start
        succeeded = sum(1 for r in results if r.validation_passed)
        logger.info(
            "Ingestion complete: %d/%d succeeded (%.2fs total)",
            succeeded, len(results), total_elapsed,
        )

        return results

    def generate_manifest(
        self, results: List[IngestionResult],
    ) -> Dict[str, Any]:
        """Generate output manifest with all domain results.

        Parameters
        ----------
        results : list[IngestionResult]
            Results from :meth:`run_all` or individual runs.

        Returns
        -------
        dict
            Manifest with summary stats and per-domain details.
        """
        total_rows = sum(r.row_count for r in results)
        total_pii_encrypted = sum(r.pii_columns_encrypted for r in results)
        total_pii_dropped = sum(r.pii_columns_dropped for r in results)
        total_duration = sum(r.duration_seconds for r in results)

        manifest: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_domains": len(results),
            "domains_passed": sum(1 for r in results if r.validation_passed),
            "domains_failed": sum(
                1 for r in results if not r.validation_passed
            ),
            "total_rows": total_rows,
            "total_pii_encrypted": total_pii_encrypted,
            "total_pii_dropped": total_pii_dropped,
            "total_duration_seconds": round(total_duration, 3),
            "output_base_s3": self._config.output_base_s3,
            "domains": {},
        }

        for result in results:
            manifest["domains"][result.source_name] = {
                "row_count": result.row_count,
                "column_count": result.column_count,
                "output_path": result.output_path,
                "duration_seconds": result.duration_seconds,
                "validation_passed": result.validation_passed,
                "pii_columns_encrypted": result.pii_columns_encrypted,
                "pii_columns_dropped": result.pii_columns_dropped,
                "warnings": result.validation_warnings,
                "metadata": result.metadata,
            }

        return manifest

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"IngestionRunner(domains={len(self._config.domains)}, "
            f"registered={len(DomainRegistry.list_available())})"
        )
