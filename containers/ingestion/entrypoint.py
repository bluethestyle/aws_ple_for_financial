#!/usr/bin/env python3
"""
SageMaker Processing Job entry point for domain data ingestion.

Runs a single domain ingestor or all domains. Reads raw data from
/opt/ml/processing/input/, applies schema validation + PII encryption,
writes clean data to /opt/ml/processing/output/.

Environment variables:
  DOMAIN_NAME: Which domain to ingest (e.g. "customer_master", "all")
  CONFIG_PATH: Path to ingestion.yaml (default: configs/financial/ingestion.yaml)
  ENCRYPTION_CONFIG: Path to encryption.yaml (optional)
  USE_LOCAL_SALTS: "true" for development (no Secrets Manager)

Usage:
  python entrypoint.py  # uses DOMAIN_NAME env var
  python entrypoint.py --domain customer_master
  python entrypoint.py --domain all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("ingestion_entrypoint")


def main() -> None:
    """Entry point for the ingestion container."""
    parser = argparse.ArgumentParser(
        description="SageMaker Processing Job entry point for domain ingestion",
    )
    parser.add_argument(
        "--domain",
        default=os.environ.get("DOMAIN_NAME", "all"),
        help="Domain to ingest, or 'all' for every enabled domain.",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get(
            "CONFIG_PATH", "configs/financial/ingestion.yaml"
        ),
        help="Path to ingestion YAML config.",
    )
    parser.add_argument(
        "--input-dir",
        default="/opt/ml/processing/input",
        help="Local input directory (SageMaker mounts S3 here).",
    )
    parser.add_argument(
        "--output-dir",
        default="/opt/ml/processing/output",
        help="Local output directory (SageMaker uploads to S3).",
    )
    args = parser.parse_args()

    # Add project root to path so core.* imports resolve.
    # When SageMaker unpacks source_dir to /opt/ml/code the directory
    # is already on sys.path, but this handles local development too.
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))

    from core.ingestion.config import IngestionConfig
    from core.ingestion.runner import IngestionRunner

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Optional encryption pipeline
    # ------------------------------------------------------------------
    encryption_pipeline = None
    use_local = os.environ.get("USE_LOCAL_SALTS", "false").lower() == "true"
    encryption_config = os.environ.get("ENCRYPTION_CONFIG", "")

    if encryption_config or not use_local:
        try:
            from core.security.salt_manager import SaltManager, LocalSaltManager
            from core.security.integer_indexer import PIIIntegerIndexer
            from core.security.encryption_policy import derive_from_schema
            from core.security.pipeline import EncryptionPipeline
            from core.data.schema_registry import SchemaRegistry

            salt_mgr = LocalSaltManager() if use_local else SaltManager()
            indexer = PIIIntegerIndexer(
                os.path.join(args.output_dir, "pii_indices"),
            )

            # Derive policies from the schema file
            schema_path = "configs/financial/schema.yaml"
            if os.path.exists(schema_path):
                registry = SchemaRegistry(schema_path)
                policies = derive_from_schema(registry)
                encryption_pipeline = EncryptionPipeline(
                    salt_mgr, indexer, policies,
                )
                logger.info("Encryption pipeline initialized")
            else:
                logger.info(
                    "Schema file not found at %s; skipping encryption",
                    schema_path,
                )
        except Exception as exc:
            logger.warning(
                "Encryption pipeline init failed: %s "
                "(continuing without encryption)",
                exc,
            )

    # ------------------------------------------------------------------
    # Load ingestion config, override paths for SageMaker
    # ------------------------------------------------------------------
    config = IngestionConfig.from_yaml(args.config)

    # Override S3 paths with local SageMaker mount points so the runner
    # reads/writes via the Processing Job filesystem.
    for domain_cfg in config.domains.values():
        domain_cfg.s3_input_path = os.path.join(
            args.input_dir, domain_cfg.name,
        )
        domain_cfg.s3_output_path = os.path.join(
            args.output_dir, domain_cfg.name,
        )

    runner = IngestionRunner(
        config, encryption_pipeline=encryption_pipeline,
    )

    # ------------------------------------------------------------------
    # Run ingestion
    # ------------------------------------------------------------------
    start = time.time()

    if args.domain == "all":
        results = runner.run_all()
    else:
        results = [runner.run_domain(args.domain)]

    elapsed = time.time() - start

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    manifest = runner.generate_manifest(results)
    manifest_path = os.path.join(args.output_dir, "ingestion_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Ingestion complete in %.1fs", elapsed)
    for r in results:
        status = "PASS" if r.validation_passed else "WARN"
        logger.info(
            "  %s: %s (%d rows, %d cols, encrypted=%d)",
            r.source_name,
            status,
            r.row_count,
            r.column_count,
            r.pii_columns_encrypted,
        )
    logger.info("Manifest: %s", manifest_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
