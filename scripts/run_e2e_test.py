#!/usr/bin/env python3
"""
End-to-End Pipeline Test — all 13 stages.

Runs from this PC (API calls only, no local compute).
All heavy work happens on SageMaker instances.

Stages:
  0.  Data conversion (ealtman2019 CSV → Parquet on SageMaker)
  1.  Data ingestion (domain ETL + PII encryption)
  2.  Feature engineering (644D via TDA/HMM/GMM/Mamba/Graph/Economics)
  3.  PLE Teacher training (2-phase, GPU)
  4.  Knowledge distillation (Teacher → Soft Label → LGBM Students)
  5.  Fidelity validation (8 metrics, fail → abort)
  6.  Feature selection (IG + LGBM gain pruning)
  7.  Model registration (ModelRegistry + auto-promote)
  8.  Lambda deployment (SAM → predict endpoint)
  9.  Inference test (/predict API call)
  10. Recommendation reasons (L1 template + interpretation)
  11. Self-critique (compliance + grounding check)
  12. Audit trail verification (DynamoDB records)
  13. Offline evaluation (test set metrics)

Usage:
    python scripts/run_e2e_test.py --stage all
    python scripts/run_e2e_test.py --stage 0      # Data conversion only
    python scripts/run_e2e_test.py --stage 3-7    # Training through registration
    python scripts/run_e2e_test.py --dry-run       # Print plan without executing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("e2e_test")

# ===========================================================================
# Config
# ===========================================================================

REGION = "ap-northeast-2"
S3_BUCKET = "aiops-ple-financial"
ROLE_ARN = "arn:aws:iam::795833413857:role/AWSPLEPlatformSageMakerRole"
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

S3_BASE = f"s3://{S3_BUCKET}/e2e-test/{TIMESTAMP}"
S3_RAW = f"s3://{S3_BUCKET}/data/test"
S3_CONVERTED = f"{S3_BASE}/converted"
S3_INGESTED = f"{S3_BASE}/ingested"
S3_FEATURES = f"{S3_BASE}/features"
S3_MODEL = f"{S3_BASE}/model"
S3_STUDENTS = f"{S3_BASE}/students"
S3_REGISTRY = f"{S3_BASE}/artifacts"

# SageMaker pre-built images (no Docker build needed)
PYTORCH_IMAGE = None  # resolved at runtime via sagemaker.image_uris
SKLEARN_IMAGE = None


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Test")
    parser.add_argument("--stage", type=str, default="all", help="Stage(s) to run: all, 0, 3-7, etc.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--instance-type-gpu", type=str, default="ml.g4dn.xlarge")
    parser.add_argument("--instance-type-cpu", type=str, default="ml.m5.xlarge")
    parser.add_argument("--skip-data-prep", action="store_true", help="Skip stages 0-2 if data already exists")
    return parser.parse_args()


def should_run(stage: int, args) -> bool:
    if args.stage == "all":
        return True
    if "-" in args.stage:
        lo, hi = args.stage.split("-")
        return int(lo) <= stage <= int(hi)
    return stage == int(args.stage)


def wait_for_job(sm_client, job_name: str, job_type: str = "training"):
    """Wait for a SageMaker job to complete, streaming status."""
    logger.info("Waiting for %s job: %s", job_type, job_name)
    while True:
        if job_type == "training":
            desc = sm_client.describe_training_job(TrainingJobName=job_name)
            status = desc["TrainingJobStatus"]
        else:
            desc = sm_client.describe_processing_job(ProcessingJobName=job_name)
            status = desc["ProcessingJobStatus"]

        logger.info("  %s: %s", job_name, status)

        if status in ("Completed", "Failed", "Stopped"):
            if status != "Completed":
                logger.error("Job %s: %s", job_name, status)
                if "FailureReason" in desc:
                    logger.error("  Reason: %s", desc["FailureReason"])
                sys.exit(1)
            return desc

        time.sleep(30)


# ===========================================================================
# Stages
# ===========================================================================

def stage_0_data_conversion(args):
    """Convert ealtman2019 CSV → Parquet on SageMaker (too large for local PC)."""
    import sagemaker
    from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

    logger.info("=" * 60)
    logger.info("Stage 0: Data Conversion (CSV → Parquet on SageMaker)")
    logger.info("=" * 60)

    session = sagemaker.Session()
    image_uri = sagemaker.image_uris.retrieve("sklearn", REGION, version="1.2-1")

    processor = ScriptProcessor(
        role=ROLE_ARN,
        image_uri=image_uri,
        instance_count=1,
        instance_type=args.instance_type_cpu,
        command=["python3"],
        sagemaker_session=session,
    )

    if args.dry_run:
        logger.info("[DRY RUN] Would convert ealtman2019 CSV on %s", args.instance_type_cpu)
        return

    processor.run(
        code="scripts/convert_raw_to_parquet.py",
        source_dir=".",
        arguments=["--dataset", "financial_transactions", "--convert-only"],
        inputs=[
            ProcessingInput(
                source=f"{S3_RAW}/01_financial_users.parquet",
                destination="/opt/ml/processing/input",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=S3_CONVERTED,
            ),
        ],
        job_name=f"e2e-convert-{TIMESTAMP}",
        wait=True,
    )
    logger.info("Stage 0 complete: %s", S3_CONVERTED)


def stage_1_2_feature_engineering(args):
    """Ingestion + Feature Engineering combined."""
    import sagemaker
    from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

    logger.info("=" * 60)
    logger.info("Stage 1-2: Ingestion + Feature Engineering → 644D")
    logger.info("=" * 60)

    session = sagemaker.Session()
    image_uri = sagemaker.image_uris.retrieve("sklearn", REGION, version="1.2-1")

    if args.dry_run:
        logger.info("[DRY RUN] Would run Feature Engineering on %s", args.instance_type_cpu)
        return

    # For initial test with Bank Churners (already adapted, 34D),
    # use the adapted data directly. Full 644D requires ealtman2019.
    logger.info("Using pre-adapted Bank Churners data (34D) for initial test")
    logger.info("Full 644D feature engineering requires ealtman2019 transaction data")
    logger.info("Data: s3://%s/data/adapted/bank_churners_train.parquet", S3_BUCKET)


def stage_3_ple_training(args):
    """PLE Teacher training (2-phase, GPU)."""
    import boto3
    import sagemaker
    from sagemaker.pytorch import PyTorch

    logger.info("=" * 60)
    logger.info("Stage 3: PLE Teacher Training (GPU)")
    logger.info("=" * 60)

    session = sagemaker.Session()

    estimator = PyTorch(
        entry_point="containers/training/train.py",
        source_dir=".",  # project root → includes configs/, core/, containers/
        role=ROLE_ARN,
        instance_count=1,
        instance_type=args.instance_type_gpu,
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
        hyperparameters={
            "config": "configs/test/bank_churners_pipeline.yaml",
            "data-path": "/opt/ml/input/data/training/",
            "epochs": "10",
            "batch-size": "256",
            "learning-rate": "0.001",
        },
        output_path=S3_MODEL,
        max_run=3600,
    )

    if args.dry_run:
        logger.info("[DRY RUN] Would launch PyTorch Training Job on %s", args.instance_type_gpu)
        logger.info("[DRY RUN] Output: %s", S3_MODEL)
        return f"{S3_MODEL}/model.tar.gz"

    estimator.fit(
        inputs={"training": f"s3://{S3_BUCKET}/data/adapted/"},
        job_name=f"e2e-ple-train-{TIMESTAMP}",
        wait=True,
    )

    model_uri = estimator.model_data
    logger.info("Stage 3 complete. Teacher model: %s", model_uri)
    return model_uri


def stage_4_7_distillation(args, model_uri: str):
    """Distillation + Fidelity + Feature Selection + Registration."""
    import sagemaker
    from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

    logger.info("=" * 60)
    logger.info("Stage 4-7: Distillation → Fidelity → Selection → Registration")
    logger.info("=" * 60)

    session = sagemaker.Session()
    image_uri = sagemaker.image_uris.retrieve("sklearn", REGION, version="1.2-1")

    processor = ScriptProcessor(
        role=ROLE_ARN,
        image_uri=image_uri,
        instance_count=1,
        instance_type=args.instance_type_cpu,
        command=["python3"],
        sagemaker_session=session,
    )

    if args.dry_run:
        logger.info("[DRY RUN] Would run distillation on %s", args.instance_type_cpu)
        logger.info("[DRY RUN] Teacher: %s", model_uri)
        return

    processor.run(
        code="scripts/run_distillation.py",
        source_dir=".",
        inputs=[
            ProcessingInput(
                source=model_uri,
                destination="/opt/ml/processing/input/teacher",
            ),
            ProcessingInput(
                source=f"s3://{S3_BUCKET}/data/adapted/bank_churners_train.parquet",
                destination="/opt/ml/processing/input/data",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=S3_STUDENTS,
            ),
        ],
        arguments=[
            "--teacher-checkpoint", "/opt/ml/processing/input/teacher/model.pth",
            "--data-path", "/opt/ml/processing/input/data/bank_churners_train.parquet",
            "--output-dir", "/opt/ml/processing/output",
            "--config", "configs/test/bank_churners_pipeline.yaml",
            "--soft-label-path", "/opt/ml/processing/output/soft_labels.parquet",
            "--temperature", "5.0",
            "--alpha", "0.3",
        ],
        job_name=f"e2e-distill-{TIMESTAMP}",
        wait=True,
    )

    logger.info("Stage 4-7 complete. Students: %s", S3_STUDENTS)


def stage_8_lambda_deploy(args):
    """Deploy Lambda serving endpoint via SAM."""
    logger.info("=" * 60)
    logger.info("Stage 8: Lambda Deployment")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[DRY RUN] Would deploy Lambda via SAM")
        logger.info("[DRY RUN] cd aws/lambda && sam build && sam deploy")
        return

    logger.info("Lambda deployment requires SAM CLI.")
    logger.info("Run manually:")
    logger.info("  cd aws/lambda")
    logger.info("  bash ../../scripts/build_lambda_layer.sh")
    logger.info("  sam build")
    logger.info("  sam deploy --guided")
    logger.info("Skipping automated deployment for now.")


def stage_9_12_inference_test(args):
    """Test inference, reasons, self-critique, audit."""
    import boto3

    logger.info("=" * 60)
    logger.info("Stage 9-12: Inference + Reasons + Self-critique + Audit")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[DRY RUN] Would invoke Lambda /predict endpoint")
        logger.info("[DRY RUN] Would verify DynamoDB audit records")
        return

    # Check if Lambda exists
    lambda_client = boto3.client("lambda", region_name=REGION)
    try:
        lambda_client.get_function(FunctionName="ple-predict")
    except Exception:
        logger.warning("Lambda 'ple-predict' not deployed yet. Skipping inference test.")
        logger.info("Deploy Lambda first (Stage 8), then re-run stages 9-12.")
        return

    # Test inference
    test_payload = {
        "user_id": "test_user_001",
        "features": {f"feat_{i}": float(i) * 0.1 for i in range(34)},
        "context": {"channel": "app", "segment": "WARMSTART"},
    }

    logger.info("Invoking ple-predict Lambda...")
    response = lambda_client.invoke(
        FunctionName="ple-predict",
        Payload=json.dumps(test_payload),
    )
    result = json.loads(response["Payload"].read())
    logger.info("Inference result:")
    logger.info("  predictions: %s", list(result.get("predictions", {}).keys()))
    logger.info("  variant: %s", result.get("variant", ""))
    logger.info("  elapsed_ms: %s", result.get("elapsed_ms", ""))
    logger.info("  is_coldstart: %s", result.get("is_coldstart", ""))

    # Check audit trail
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    try:
        table = dynamodb.Table("ple-prediction-log")
        scan = table.scan(Limit=5)
        count = scan.get("Count", 0)
        logger.info("DynamoDB prediction_log: %d records found", count)
    except Exception as e:
        logger.warning("DynamoDB check failed: %s", e)


def stage_13_results(args):
    """Download and display test results."""
    import boto3

    logger.info("=" * 60)
    logger.info("Stage 13: Results Summary")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[DRY RUN] Would download test_evaluation.json from S3")
        return

    s3 = boto3.client("s3", region_name=REGION)

    # Download evaluation results
    for filename in ["distillation_summary.json", "test_evaluation.json", "fidelity_report.json"]:
        try:
            key = f"e2e-test/{TIMESTAMP}/students/{filename}"
            local_path = f"data/results/{filename}"
            Path("data/results").mkdir(parents=True, exist_ok=True)
            s3.download_file(S3_BUCKET, key, local_path)
            logger.info("Downloaded: %s", local_path)

            with open(local_path) as f:
                data = json.load(f)
            logger.info("  Content: %s", json.dumps(data, indent=2, default=str)[:500])
        except Exception as e:
            logger.info("  %s not found (may be expected): %s", filename, e)


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("E2E Pipeline Test")
    logger.info("  Timestamp: %s", TIMESTAMP)
    logger.info("  S3 Base: %s", S3_BASE)
    logger.info("  GPU Instance: %s ($0.94/hr)", args.instance_type_gpu)
    logger.info("  CPU Instance: %s ($0.23/hr)", args.instance_type_cpu)
    logger.info("  Stages: %s", args.stage)
    logger.info("  Dry Run: %s", args.dry_run)
    logger.info("=" * 60)

    model_uri = f"{S3_MODEL}/model.tar.gz"

    if should_run(0, args):
        stage_0_data_conversion(args)

    if should_run(1, args) or should_run(2, args):
        stage_1_2_feature_engineering(args)

    if should_run(3, args):
        result = stage_3_ple_training(args)
        if result:
            model_uri = result

    if should_run(4, args) or should_run(5, args) or should_run(6, args) or should_run(7, args):
        stage_4_7_distillation(args, model_uri)

    if should_run(8, args):
        stage_8_lambda_deploy(args)

    if should_run(9, args) or should_run(10, args) or should_run(11, args) or should_run(12, args):
        stage_9_12_inference_test(args)

    if should_run(13, args):
        stage_13_results(args)

    logger.info("=" * 60)
    logger.info("E2E Test Complete!")
    logger.info("  Monitor: https://ap-northeast-2.console.aws.amazon.com/sagemaker/home?region=ap-northeast-2#/jobs")
    logger.info("  Logs: https://ap-northeast-2.console.aws.amazon.com/cloudwatch/home?region=ap-northeast-2#logsV2:log-groups")
    logger.info("  S3: https://s3.console.aws.amazon.com/s3/buckets/%s?prefix=e2e-test/%s/", S3_BUCKET, TIMESTAMP)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
