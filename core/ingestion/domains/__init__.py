"""
Domain ingestor implementations.

This package contains concrete AbstractDomainIngestor subclasses for each
data domain (customer_master, account, card, transaction, etc.).

In the AWS benchmark project, domain implementations live in the on-premises
repository and are not included here.  The Santander benchmark data flows
through the adapter layer instead.

To add a new domain ingestor for AWS deployment:

    1. Create a new file in this directory (e.g., ``my_domain.py``)
    2. Subclass ``AbstractDomainIngestor`` from ``core.ingestion.base``
    3. Register with ``@DomainRegistry.register("my_domain")``
    4. Add the domain to the ingestion YAML config

The abstract base, registry, config, and runner are all available in
``core.ingestion`` and ready to use.
"""
