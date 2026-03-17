#!/usr/bin/env python3
"""
Ingestion CLI -- run domain ingestion locally or submit SageMaker jobs.

Usage:
    # Local (development)
    python scripts/run_ingestion.py --local --domain customer_master \\
        --input ./data/raw/ --output ./data/ingested/

    python scripts/run_ingestion.py --local --domain all

    # SageMaker Processing Job
    python scripts/run_ingestion.py --sagemaker --domain all \\
        --config configs/financial/ingestion.yaml

    # Dry run (prints what would execute without launching anything)
    python scripts/run_ingestion.py --dry-run --domain all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("run_ingestion")


# ======================================================================
# Argument parsing
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run domain data ingestion locally or on SageMaker.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--local",
        action="store_true",
        help="Run ingestion locally (uses IngestionRunner directly).",
    )
    mode.add_argument(
        "--sagemaker",
        action="store_true",
        help="Submit a SageMaker Processing Job.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without running anything.",
    )

    parser.add_argument(
        "--domain",
        default="all",
        help="Domain to ingest, or 'all' (default: all).",
    )
    parser.add_argument(
        "--config",
        default="configs/financial/ingestion.yaml",
        help="Path to ingestion YAML config.",
    )

    # Local-mode overrides
    parser.add_argument(
        "--input",
        default="./data/raw",
        help="Local input directory (local mode only).",
    )
    parser.add_argument(
        "--output",
        default="./data/ingested",
        help="Local output directory (local mode only).",
    )
    parser.add_argument(
        "--use-local-salts",
        action="store_true",
        help="Use LocalSaltManager instead of AWS Secrets Manager.",
    )

    # SageMaker-mode options
    parser.add_argument(
        "--instance-type",
        default="ml.m5.xlarge",
        help="SageMaker instance type (SageMaker mode only).",
    )
    parser.add_argument(
        "--volume-size",
        type=int,
        default=30,
        help="EBS volume size in GB (SageMaker mode only).",
    )
    parser.add_argument(
        "--role-arn",
        default=os.environ.get("SAGEMAKER_ROLE_ARN", ""),
        help="SageMaker execution role ARN.",
    )
    parser.add_argument(
        "--input-s3",
        default="",
        help="S3 URI for raw input data (SageMaker mode only).",
    )
    parser.add_argument(
        "--output-s3",
        default="",
        help="S3 URI for ingested output data (SageMaker mode only).",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for SageMaker job to finish.",
    )
    parser.add_argument(
        "--source-dir",
        default="containers/ingestion/",
        help="Source directory for SageMaker managed-image mode.",
    )

    return parser


# ======================================================================
# Local execution
# ======================================================================

def run_local(args: argparse.Namespace) -> None:
    """Run ingestion using the IngestionRunner directly."""
    from core.ingestion.config import IngestionConfig
    from core.ingestion.runner import IngestionRunner

    config = IngestionConfig.from_yaml(args.config)

    # Override paths with local directories
    for domain_cfg in config.domains.values():
        domain_cfg.s3_input_path = os.path.join(args.input, domain_cfg.name)
        domain_cfg.s3_output_path = os.path.join(args.output, domain_cfg.name)

    # Set up encryption (optional)
    encryption_pipeline = None
    if not args.use_local_salts:
        encryption_pipeline = _build_encryption_pipeline(
            output_dir=args.output,
            use_local=args.use_local_salts,
        )

    runner = IngestionRunner(config, encryption_pipeline=encryption_pipeline)

    start = time.time()
    if args.domain == "all":
        results = runner.run_all()
    else:
        results = [runner.run_domain(args.domain)]
    elapsed = time.time() - start

    # Write manifest
    os.makedirs(args.output, exist_ok=True)
    manifest = runner.generate_manifest(results)
    manifest_path = os.path.join(args.output, "ingestion_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    _print_summary(results, elapsed, manifest_path)


# ======================================================================
# SageMaker execution
# ======================================================================

def run_sagemaker(args: argparse.Namespace) -> None:
    """Submit a SageMaker Processing Job for ingestion."""
    from core.pipeline.config import PipelineConfig

    try:
        import sagemaker
        from sagemaker.sklearn.processing import SKLearnProcessor
        from sagemaker.processing import ProcessingInput, ProcessingOutput
    except ImportError:
        logger.error(
            "sagemaker SDK is required for --sagemaker mode. "
            "Install with: pip install sagemaker"
        )
        sys.exit(1)

    if not args.role_arn:
        logger.error(
            "--role-arn is required for SageMaker mode "
            "(or set SAGEMAKER_ROLE_ARN env var)"
        )
        sys.exit(1)

    session = sagemaker.Session()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"ingestion-{args.domain}-{ts}"

    logger.info("Building SKLearnProcessor for ingestion job: %s", job_name)

    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=args.role_arn,
        instance_type=args.instance_type,
        instance_count=1,
        volume_size_in_gb=args.volume_size,
        max_runtime_in_seconds=3600,
        sagemaker_session=session,
        tags=[
            {"Key": "Pipeline", "Value": "ingestion"},
            {"Key": "Domain", "Value": args.domain},
        ],
    )

    # Input channels
    processing_inputs: List[ProcessingInput] = []
    input_s3 = args.input_s3
    if input_s3:
        processing_inputs.append(
            ProcessingInput(
                source=input_s3,
                destination="/opt/ml/processing/input",
                input_name="data",
            )
        )

    # Output channels
    processing_outputs: List[ProcessingOutput] = []
    output_s3 = args.output_s3
    if output_s3:
        processing_outputs.append(
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=output_s3,
                output_name="ingested",
            )
        )

    # Script arguments
    arguments = [
        "--domain", args.domain,
        "--config", args.config,
        "--input-dir", "/opt/ml/processing/input",
        "--output-dir", "/opt/ml/processing/output",
    ]

    # Environment variables
    env = {
        "DOMAIN_NAME": args.domain,
        "USE_LOCAL_SALTS": "false",
    }

    logger.info("Launching SageMaker Processing Job: %s", job_name)
    logger.info("  Instance:   %s", args.instance_type)
    logger.info("  Domain:     %s", args.domain)
    logger.info("  Source dir: %s", args.source_dir)
    logger.info("  Input S3:   %s", input_s3 or "(none)")
    logger.info("  Output S3:  %s", output_s3 or "(none)")
    logger.info("  Wait:       %s", not args.no_wait)

    processor.run(
        code="entrypoint.py",
        source_dir=args.source_dir,
        inputs=processing_inputs or None,
        outputs=processing_outputs or None,
        arguments=arguments,
        job_name=job_name,
        wait=not args.no_wait,
        logs=not args.no_wait,
    )

    logger.info("Job submitted: %s", job_name)
    if args.no_wait:
        logger.info(
            "Job is running asynchronously. "
            "Check status: aws sagemaker describe-processing-job "
            "--processing-job-name %s",
            job_name,
        )


# ======================================================================
# Dry run
# ======================================================================

def run_dry_run(args: argparse.Namespace) -> None:
    """Print the execution plan without running anything."""
    from core.ingestion.config import IngestionConfig

    config = IngestionConfig.from_yaml(args.config)
    enabled = config.get_enabled_domains()

    if args.domain == "all":
        domains = [d.name for d in enabled]
    else:
        domains = [args.domain]

    logger.info("=" * 60)
    logger.info("DRY RUN -- Ingestion Plan")
    logger.info("=" * 60)
    logger.info("Config:          %s", args.config)
    logger.info("Domains:         %s", domains)
    logger.info("Total enabled:   %d", len(enabled))
    logger.info("Output base S3:  %s", config.output_base_s3)
    logger.info("")

    for name in domains:
        d = config.get_domain(name)
        if d is None:
            logger.warning("  %s: NOT CONFIGURED", name)
            continue
        logger.info(
            "  %s: input=%s  output=%s  enabled=%s",
            name,
            d.s3_input_path or "(unset)",
            config.get_output_path(name) or "(unset)",
            d.enabled,
        )

    logger.info("")
    logger.info("No jobs were launched (dry run).")
    logger.info("=" * 60)


# ======================================================================
# Helpers
# ======================================================================

def _build_encryption_pipeline(
    output_dir: str,
    use_local: bool = False,
) -> Any:
    """Build an EncryptionPipeline if dependencies are available."""
    try:
        from core.security.salt_manager import SaltManager, LocalSaltManager
        from core.security.integer_indexer import PIIIntegerIndexer
        from core.security.encryption_policy import derive_from_schema
        from core.security.pipeline import EncryptionPipeline
        from core.data.schema_registry import SchemaRegistry

        salt_mgr = LocalSaltManager() if use_local else SaltManager()
        indexer = PIIIntegerIndexer(
            os.path.join(output_dir, "pii_indices"),
        )
        schema_path = "configs/financial/schema.yaml"
        if os.path.exists(schema_path):
            registry = SchemaRegistry(schema_path)
            policies = derive_from_schema(registry)
            pipeline = EncryptionPipeline(salt_mgr, indexer, policies)
            logger.info("Encryption pipeline initialized")
            return pipeline
        logger.info("Schema file not found; encryption disabled")
    except Exception as exc:
        logger.warning("Could not init encryption: %s", exc)
    return None


def _print_summary(
    results: list,
    elapsed: float,
    manifest_path: str,
) -> None:
    """Print ingestion summary to the console."""
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


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.local:
        run_local(args)
    elif args.sagemaker:
        run_sagemaker(args)
    elif args.dry_run:
        run_dry_run(args)


if __name__ == "__main__":
    main()
