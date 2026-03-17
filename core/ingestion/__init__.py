"""
Domain Ingestion Framework -- reads raw data, transforms, validates, encrypts.

Each domain (customer_master, account, card, transaction, etc.) implements
a concrete :class:`AbstractDomainIngestor` subclass registered via
``@DomainRegistry.register("name")``.

Pattern mirrors ``core.feature.generator`` / ``FeatureGeneratorRegistry``.

Available components
--------------------
* :class:`AbstractDomainIngestor` -- abstract base for all domain ingestors.
* :class:`DomainRegistry` -- plugin registry for domain ingestors (Pool).
* :class:`IngestionConfig` / :class:`DomainSourceConfig` -- typed config.
* :class:`IngestionRunner` -- orchestrator for running multiple domains.
* :class:`IngestionResult` -- structured outcome of an ingestion run.

Usage::

    from core.ingestion import (
        DomainRegistry,
        IngestionConfig,
        IngestionRunner,
    )

    config = IngestionConfig.from_yaml("configs/financial/ingestion.yaml")
    runner = IngestionRunner(config)
    results = runner.run_all()
"""

from __future__ import annotations

from .base import AbstractDomainIngestor, IngestionResult
from .config import DomainSourceConfig, IngestionConfig
from .registry import DomainRegistry
from .runner import IngestionRunner

__all__ = [
    "AbstractDomainIngestor",
    "DomainRegistry",
    "DomainSourceConfig",
    "IngestionConfig",
    "IngestionResult",
    "IngestionRunner",
]
