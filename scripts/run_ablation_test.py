#!/usr/bin/env python3
"""
Ablation Test Orchestrator — submit & monitor SageMaker Jobs.

Runs a multi-phase ablation study on the ealtman2019 credit-card dataset:

    Phase 0  Data Preparation   (Processing Job, GPU)
    Phase 1  Feature Group Ablation  (9 Training Jobs, GPU)
    Phase 2  Expert Ablation         (7 Training Jobs, GPU)
    Phase 3  Hyperparameter Sensitivity (12 Training Jobs, GPU)
    Phase 4  Best-Config Full Pipeline  (Processing Job, CPU)
    Phase 5  Result Collection + HTML Report

Usage::

    # All phases
    python scripts/run_ablation_test.py --phase all

    # Single phase
    python scripts/run_ablation_test.py --phase 1

    # Dry run (print what would be submitted)
    python scripts/run_ablation_test.py --phase all --dry-run

    # Resume from a specific teacher checkpoint
    python scripts/run_ablation_test.py --phase 4 --model-uri s3://bucket/model.tar.gz
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("run_ablation_test")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGION = "ap-northeast-2"
S3_BUCKET = "aiops-ple-financial"
ROLE_ARN = "arn:aws:iam::795833413857:role/AWSPLEPlatformSageMakerRole"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = "configs/test/ealtman2019_pipeline.yaml"

# Instance types
GPU_INSTANCE = "ml.g4dn.xlarge"
CPU_INSTANCE = "ml.m5.xlarge"

# Framework versions (for PyTorch Estimator)
PYTORCH_VERSION = "2.1"
PY_VERSION = "py310"

# ---------------------------------------------------------------------------
# Feature-group ablation scenarios
# ---------------------------------------------------------------------------
FEATURE_ABLATION_SCENARIOS: List[Dict[str, Any]] = [
    {"name": "full",                 "remove": []},
    {"name": "no_tda",               "remove": ["tda_topology"]},
    {"name": "no_temporal",          "remove": ["base_temporal", "mamba_temporal"]},
    {"name": "no_graph",             "remove": ["graph_embeddings"]},
    {"name": "no_economics",         "remove": ["economics"]},
    {"name": "no_multidisciplinary", "remove": ["multidisciplinary"]},
    {"name": "no_hmm",              "remove": ["hmm_states"]},
    {"name": "no_merchant",          "remove": ["merchant_hierarchy"]},
    {
        "name": "base_only",
        "remove": [
            "tda_topology", "mamba_temporal", "hmm_states",
            "graph_embeddings", "economics", "multidisciplinary",
            "merchant_hierarchy", "model_derived", "gmm_clustering",
        ],
    },
]

# ---------------------------------------------------------------------------
# Expert ablation scenarios
# ---------------------------------------------------------------------------
ALL_SHARED_EXPERTS = [
    "deepfm", "temporal_ensemble", "hgcn",
    "perslay", "causal", "lightgcn", "optimal_transport",
]

EXPERT_ABLATION_SCENARIOS: List[Dict[str, Any]] = [
    {"name": "full_basket",  "shared": list(ALL_SHARED_EXPERTS)},
    {"name": "no_deepfm",    "shared": [e for e in ALL_SHARED_EXPERTS if e != "deepfm"]},
    {"name": "no_temporal",   "shared": [e for e in ALL_SHARED_EXPERTS if e != "temporal_ensemble"]},
    {"name": "no_hgcn",      "shared": [e for e in ALL_SHARED_EXPERTS if e != "hgcn"]},
    {"name": "no_perslay",    "shared": [e for e in ALL_SHARED_EXPERTS if e != "perslay"]},
    {"name": "no_causal",     "shared": [e for e in ALL_SHARED_EXPERTS if e != "causal"]},
    {"name": "mlp_only",      "shared": ["mlp"]},
]

# ---------------------------------------------------------------------------
# Hyperparameter sensitivity scenarios
# ---------------------------------------------------------------------------
HP_SCENARIOS: List[Dict[str, Any]] = []

for lr in [0.0001, 0.0005, 0.001, 0.005]:
    HP_SCENARIOS.append({"name": f"lr_{lr}", "learning_rate": lr})

for temp in [1.0, 3.0, 5.0, 10.0]:
    HP_SCENARIOS.append({"name": f"temp_{temp}", "temperature": temp})

for nl in [1, 2, 3]:
    HP_SCENARIOS.append({"name": f"layers_{nl}", "num_layers": nl})


# ===================================================================
# Argument parsing
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation Test Orchestrator for ealtman2019 pipeline",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["0", "1", "2", "3", "4", "5", "all"],
        help="Which phase to run (0-5 or 'all')",
    )
    parser.add_argument(
        "--instance-type-gpu",
        type=str,
        default=GPU_INSTANCE,
        help="GPU instance type for training jobs",
    )
    parser.add_argument(
        "--instance-type-cpu",
        type=str,
        default=CPU_INSTANCE,
        help="CPU instance type for processing jobs",
    )
    parser.add_argument(
        "--model-uri",
        type=str,
        default="",
        help="S3 URI to existing teacher model (skip training, jump to distill)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted without actually running",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit jobs and return immediately (don't wait for completion)",
    )
    parser.add_argument(
        "--skip-fidelity-gate",
        action="store_true",
        default=True,
        help="Skip fidelity gate in Phase 4 distillation (default: True)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retry attempts for processing jobs on failure (default: 3)",
    )
    return parser.parse_args()


# ===================================================================
# Source packaging
# ===================================================================

def _prepare_source_package(output_dir: str) -> str:
    """Package the project source into a tar.gz for SageMaker upload.

    Includes ``core/``, ``configs/``, ``containers/``, and ``scripts/``
    directories, excluding ``__pycache__``, ``.git``, and large data files.

    Returns the path to the created tar.gz file.
    """
    tar_path = os.path.join(output_dir, "source.tar.gz")
    root = str(PROJECT_ROOT)

    include_dirs = ["core", "configs", "containers", "scripts", "adapters"]
    include_files = ["setup.py", "setup.cfg", "pyproject.toml", "requirements.txt"]

    skip_patterns = {"__pycache__", ".git", ".eggs", "*.egg-info", "node_modules"}
    # Skip top-level data/ dir but NOT core/data/ (source code)
    skip_top_dirs = {"data"}

    def _should_skip(name: str) -> bool:
        for pat in skip_patterns:
            if pat in name:
                return True
        return False

    with tarfile.open(tar_path, "w:gz") as tar:
        for d in include_dirs:
            if d in skip_top_dirs:
                continue
            full_path = os.path.join(root, d)
            if os.path.isdir(full_path):
                for dirpath, dirnames, filenames in os.walk(full_path):
                    dirnames[:] = [dn for dn in dirnames if not _should_skip(dn)]
                    for fn in filenames:
                        if _should_skip(fn) or fn.endswith((".pyc", ".pyo")):
                            continue
                        filepath = os.path.join(dirpath, fn)
                        arcname = os.path.relpath(filepath, root)
                        tar.add(filepath, arcname=arcname)

        for f in include_files:
            full_path = os.path.join(root, f)
            if os.path.isfile(full_path):
                tar.add(full_path, arcname=f)

    logger.info("Source package created: %s (%.1f MB)",
                tar_path, os.path.getsize(tar_path) / 1024 / 1024)
    return tar_path


# ===================================================================
# Utilities
# ===================================================================

def _is_json_serializable(value: Any) -> bool:
    """Return True if value is safely JSON-serializable."""
    try:
        json.dumps(value, default=str)
        return True
    except (TypeError, ValueError):
        return False


def _sanitize_job_name(name: str) -> str:
    """Sanitize a SageMaker job name to meet API constraints.

    SageMaker job names must match: ^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$
    Max length 63 characters.
    """
    import re
    # Replace any non-alphanumeric/hyphen characters with hyphens
    sanitized = re.sub(r"[^a-zA-Z0-9-]", "-", name)
    # Collapse consecutive hyphens
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    # Strip leading/trailing hyphens
    sanitized = sanitized.strip("-")
    # Truncate to 63 characters (SageMaker limit)
    if len(sanitized) > 63:
        sanitized = sanitized[:63].rstrip("-")
    return sanitized


# ===================================================================
# S3 helpers
# ===================================================================

def _s3_base_path(timestamp: str) -> str:
    return f"s3://{S3_BUCKET}/ablation-test/{timestamp}"


def _upload_source_to_s3(local_path: str, s3_base: str) -> str:
    """Upload source package to S3 and return the S3 URI."""
    import boto3
    s3 = boto3.client("s3", region_name=REGION)
    key = f"{s3_base.replace(f's3://{S3_BUCKET}/', '')}/source/source.tar.gz"
    s3.upload_file(local_path, S3_BUCKET, key)
    uri = f"s3://{S3_BUCKET}/{key}"
    logger.info("Source uploaded to %s", uri)
    return uri


def _download_json_from_s3(s3_uri: str) -> Optional[dict]:
    """Download a JSON file from S3 and parse it."""
    import boto3
    try:
        s3 = boto3.client("s3", region_name=REGION)
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception as e:
        logger.warning("Failed to download %s: %s", s3_uri, e)
        return None


def _list_s3_keys(prefix: str) -> List[str]:
    """List all keys under an S3 prefix."""
    import boto3
    s3 = boto3.client("s3", region_name=REGION)
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    s3_prefix = prefix.replace(f"s3://{S3_BUCKET}/", "")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


# ===================================================================
# SageMaker Job submission helpers
# ===================================================================

def _wait_for_training_job(job_name: str) -> str:
    """Wait for a SageMaker Training Job to complete. Returns final status."""
    import boto3
    sm = boto3.client("sagemaker", region_name=REGION)
    logger.info("Waiting for training job: %s", job_name)

    while True:
        desc = sm.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        if status in ("Completed", "Failed", "Stopped"):
            logger.info("Job %s → %s", job_name, status)
            if status == "Failed":
                reason = desc.get("FailureReason", "unknown")
                logger.error("  Failure reason: %s", reason)
            return status
        time.sleep(30)


def _wait_for_processing_job(job_name: str) -> str:
    """Wait for a SageMaker Processing Job to complete. Returns final status."""
    import boto3
    sm = boto3.client("sagemaker", region_name=REGION)
    logger.info("Waiting for processing job: %s", job_name)

    while True:
        desc = sm.describe_processing_job(ProcessingJobName=job_name)
        status = desc["ProcessingJobStatus"]
        if status in ("Completed", "Failed", "Stopped"):
            logger.info("Job %s → %s", job_name, status)
            if status == "Failed":
                reason = desc.get("FailureReason", "unknown")
                logger.error("  Failure reason: %s", reason)
            return status
        time.sleep(30)


def _get_model_artifact_uri(job_name: str) -> Optional[str]:
    """Retrieve the S3 model artifact URI from a completed Training Job."""
    import boto3
    sm = boto3.client("sagemaker", region_name=REGION)
    desc = sm.describe_training_job(TrainingJobName=job_name)
    return desc.get("ModelArtifacts", {}).get("S3ModelArtifacts")


def _submit_training_job(
    job_name: str,
    s3_output: str,
    instance_type: str,
    hyperparameters: Dict[str, str],
    source_uri: str,
    data_uri: str,
    wait: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Submit a SageMaker PyTorch Training Job.

    Parameters
    ----------
    job_name : str
        Unique training job name.
    s3_output : str
        S3 URI for model output artifacts.
    instance_type : str
        SageMaker instance type.
    hyperparameters : dict
        Hyperparameters to pass to the training script.
    source_uri : str
        S3 URI to the source.tar.gz package.
    data_uri : str
        S3 URI to the training data.
    wait : bool
        Whether to block until job completes.
    dry_run : bool
        Print config without submitting.

    Returns
    -------
    dict
        Job metadata including name, status, model_uri.
    """
    result: Dict[str, Any] = {
        "job_name": job_name,
        "instance_type": instance_type,
        "hyperparameters": hyperparameters,
        "s3_output": s3_output,
        "data_uri": data_uri,
    }

    if dry_run:
        logger.info("[DRY RUN] Training Job: %s", job_name)
        logger.info("  Instance: %s", instance_type)
        logger.info("  Output: %s", s3_output)
        logger.info("  Hyperparameters: %s",
                     json.dumps(hyperparameters, indent=2, ensure_ascii=False))
        result["status"] = "DRY_RUN"
        return result

    import sagemaker
    from sagemaker.inputs import TrainingInput
    from sagemaker.pytorch import PyTorch

    session = sagemaker.Session()

    # Ensure PYTHONIOENCODING=utf-8 for Korean log messages
    env = {
        "PYTHONIOENCODING": "utf-8",
        "NCCL_DEBUG": "WARN",
    }

    # Package source code via _prepare_source_package (single packaging path)
    tmp_dir = tempfile.mkdtemp()
    source_tar = _prepare_source_package(tmp_dir)

    # Extract tar.gz to a temp dir for SageMaker Estimator source_dir
    import shutil
    pkg_dir = Path(tempfile.mkdtemp()) / "source"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source_tar, "r:gz") as tar:
        tar.extractall(str(pkg_dir))

    estimator = PyTorch(
        entry_point="containers/training/train.py",
        source_dir=str(pkg_dir),
        role=ROLE_ARN,
        instance_type=instance_type,
        instance_count=1,
        framework_version=PYTORCH_VERSION,
        py_version=PY_VERSION,
        output_path=s3_output,
        hyperparameters=hyperparameters,
        environment=env,
        use_spot_instances=True,
        max_run=14400,       # 4 hours
        max_wait=18000,      # 5 hours (spot wait)
        tags=[
            {"Key": "Project", "Value": "ablation-test"},
            {"Key": "Phase", "Value": hyperparameters.get("ablation_phase", "unknown")},
        ],
        sagemaker_session=session,
    )

    inputs = {
        "train": TrainingInput(
            data_uri,
            distribution="FullyReplicated",
        ),
    }

    job_name = _sanitize_job_name(job_name)
    logger.info("Submitting Training Job: %s", job_name)
    estimator.fit(inputs, job_name=job_name, wait=False)

    if wait:
        status = _wait_for_training_job(job_name)
        result["status"] = status
        if status == "Completed":
            result["model_uri"] = _get_model_artifact_uri(job_name)
    else:
        result["status"] = "InProgress"

    return result


def _submit_processing_job(
    job_name: str,
    script: str,
    s3_output: str,
    instance_type: str,
    arguments: List[str],
    inputs: Optional[List[Any]] = None,
    wait: bool = True,
    dry_run: bool = False,
    use_gpu: bool = False,
    max_retries: int = 0,
) -> Dict[str, Any]:
    """Submit a SageMaker Processing Job (SKLearn or PyTorch).

    Parameters
    ----------
    job_name : str
        Unique processing job name.
    script : str
        Python script to run (relative to source_dir).
    s3_output : str
        S3 URI for output.
    instance_type : str
        SageMaker instance type.
    arguments : list
        CLI arguments for the script.
    inputs : list, optional
        ProcessingInput objects.
    wait : bool
        Whether to block until job completes.
    dry_run : bool
        Print config without submitting.
    use_gpu : bool
        If True, use PyTorchProcessor with GPU deps instead of SKLearnProcessor.
    max_retries : int
        Number of retry attempts on failure (0 = no retries).

    Returns
    -------
    dict
        Job metadata.
    """
    result: Dict[str, Any] = {
        "job_name": job_name,
        "instance_type": instance_type,
        "script": script,
        "s3_output": s3_output,
    }

    if dry_run:
        logger.info("[DRY RUN] Processing Job: %s", job_name)
        logger.info("  Script: %s", script)
        logger.info("  Instance: %s", instance_type)
        logger.info("  Output: %s", s3_output)
        logger.info("  Arguments: %s", arguments)
        logger.info("  GPU mode: %s", use_gpu)
        result["status"] = "DRY_RUN"
        return result

    import sagemaker
    from sagemaker.processing import ProcessingInput, ProcessingOutput

    session = sagemaker.Session()

    # Package source code as tar.gz
    tmp_dir = tempfile.mkdtemp()
    source_tar = _prepare_source_package(tmp_dir)
    s3_source_key = f"ablation-test/{job_name}/source/source_pkg.tar.gz"
    s3_source = f"s3://{S3_BUCKET}/{s3_source_key}"
    import boto3 as _b3
    _b3.client("s3").upload_file(source_tar, S3_BUCKET, s3_source_key)
    logger.info("Source package uploaded: %s", s3_source)

    # Build pip install list based on whether we need GPU deps
    if use_gpu:
        pip_deps = [
            "pyyaml", "omegaconf", "lightgbm", "pyarrow", "scipy",
            "scikit-learn", "duckdb",
            # Feature generators (CPU-based, also needed in GPU jobs)
            "hmmlearn", "ripser", "giotto-ph", "persim",
            # GPU acceleration (on top of PyTorch base image)
            "cupy-cuda12x", "mamba-ssm",
        ]
    else:
        pip_deps = [
            "pyyaml", "omegaconf", "lightgbm", "pyarrow", "scipy",
            "scikit-learn", "duckdb", "torch",
            # Feature generators (CPU)
            "hmmlearn", "ripser", "giotto-ph", "persim",
        ]
    pip_install_str = ", ".join(f'"{d}"' for d in pip_deps)

    # Create wrapper script that unpacks source + installs deps + runs target
    wrapper = Path("_ablation_wrapper.py")
    wrapper.write_text(f"""\
import subprocess, sys, os, tarfile
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       {pip_install_str}])
src_tar = "/opt/ml/processing/input/source/source_pkg.tar.gz"
src_dir = "/opt/ml/processing/source"
os.makedirs(src_dir, exist_ok=True)
with tarfile.open(src_tar, "r:gz") as tar:
    tar.extractall(src_dir)
import glob
# Unpack any model.tar.gz in teacher input
teacher_dir = "/opt/ml/processing/input/teacher"
if os.path.isdir(teacher_dir):
    for tgz in glob.glob(os.path.join(teacher_dir, "*.tar.gz")):
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(teacher_dir)
        print(f"Unpacked teacher: {{os.listdir(teacher_dir)}}")
sys.path.insert(0, src_dir)
os.chdir(src_dir)
sys.argv = ["{script}"] + {arguments!r}
exec(open("{script}").read())
""")

    all_inputs = [
        ProcessingInput(
            source=s3_source,
            destination="/opt/ml/processing/input/source",
        ),
    ] + (inputs or [])

    outputs = [
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=s3_output,
        ),
    ]

    # Retry loop — attempt submission up to (1 + max_retries) times
    attempt = 0
    last_status = "Unknown"
    while attempt <= max_retries:
        attempt_job_name = _sanitize_job_name(
            job_name if attempt == 0 else f"{job_name}-r{attempt}"
        )
        attempt_suffix = f" (attempt {attempt + 1}/{max_retries + 1})" if max_retries > 0 else ""

        if use_gpu:
            from sagemaker.pytorch import PyTorchProcessor
            processor = PyTorchProcessor(
                role=ROLE_ARN,
                framework_version=PYTORCH_VERSION,
                py_version=PY_VERSION,
                instance_type=instance_type,
                instance_count=1,
                sagemaker_session=session,
                env={"PYTHONIOENCODING": "utf-8"},
            )
        else:
            from sagemaker.sklearn import SKLearnProcessor
            processor = SKLearnProcessor(
                role=ROLE_ARN,
                framework_version="1.2-1",
                instance_type=instance_type,
                instance_count=1,
                sagemaker_session=session,
                env={"PYTHONIOENCODING": "utf-8"},
            )

        logger.info("Submitting Processing Job: %s%s", attempt_job_name, attempt_suffix)
        processor.run(
            code=str(wrapper),
            inputs=all_inputs,
            outputs=outputs,
            job_name=attempt_job_name,
            wait=False,
        )

        if wait:
            last_status = _wait_for_processing_job(attempt_job_name)
            if last_status == "Completed" or attempt >= max_retries:
                result["status"] = last_status
                result["job_name"] = attempt_job_name
                if attempt > 0:
                    result["retry_attempts"] = attempt
                break
            logger.warning(
                "Processing Job %s failed — retrying (%d/%d)",
                attempt_job_name, attempt + 1, max_retries,
            )
            attempt += 1
        else:
            result["status"] = "InProgress"
            result["job_name"] = attempt_job_name
            break

    wrapper.unlink(missing_ok=True)
    return result


# ===================================================================
# Phase implementations
# ===================================================================

def run_phase0(s3_base: str, ts: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Phase 0: Data Preparation — run ealtman2019 adapter as Processing Job.

    Transforms 24M raw transactions into 2,000-user x ~469D feature matrix.
    """
    logger.info("=" * 70)
    logger.info("Phase 0: Data Preparation (ealtman2019 adapter)")
    logger.info("=" * 70)

    job_name = f"ablation-p0-data-prep-{ts}"
    s3_output = f"{s3_base}/phase0/data/"

    from sagemaker.processing import ProcessingInput

    result = _submit_processing_job(
        job_name=job_name,
        script="adapters/ealtman2019_adapter.py",
        s3_output=s3_output,
        instance_type=args.instance_type_cpu,  # Processing Job GPU quota=0; CPU + numpy fallbacks
        inputs=[
            ProcessingInput(
                source=f"s3://{S3_BUCKET}/data/raw/ealtman2019/",
                destination="/opt/ml/processing/input/raw",
            ),
        ],
        arguments=[
            "--input-dir", "/opt/ml/processing/input/raw",
            "--output-dir", "/opt/ml/processing/output",
        ],
        wait=not args.no_wait,
        dry_run=args.dry_run,
        use_gpu=False,
    )

    # Point data_uri at the output directory (not a single file) so that
    # training jobs receive both the parquet AND the event_sequences .npy files
    # produced by the adapter.
    result["data_uri"] = s3_output
    return result


def run_phase1(
    s3_base: str,
    ts: str,
    data_uri: str,
    source_uri: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Phase 1: Feature Group Ablation — 9 training scenarios.

    Each scenario removes one feature group to measure its contribution.
    """
    logger.info("=" * 70)
    logger.info("Phase 1: Feature Group Ablation (%d scenarios)",
                len(FEATURE_ABLATION_SCENARIOS))
    logger.info("=" * 70)

    results = []

    for scenario in FEATURE_ABLATION_SCENARIOS:
        name = scenario["name"]
        safe_name = name.replace("_", "-")
        job_name = f"ablation-p1-{safe_name}-{ts}"
        s3_output = f"{s3_base}/phase1/{name}/"

        hyperparameters = {
            "config": CONFIG_PATH,
            "ablation_phase": "1",
            "ablation_type": "feature_group",
            "ablation_scenario": name,
            "removed_feature_groups": json.dumps(scenario["remove"]),
            "epochs": "30",
            "batch_size": "128",
            "learning_rate": "0.001",
            "seed": "42",
            "_s3_output": f"{s3_base}/phase1/{name}",
        }

        logger.info("--- Scenario: %s (remove=%s) ---", name, scenario["remove"])

        result = _submit_training_job(
            job_name=job_name,
            s3_output=s3_output,
            instance_type=args.instance_type_gpu,
            hyperparameters=hyperparameters,
            source_uri=source_uri,
            data_uri=data_uri,
            wait=not args.no_wait,
            dry_run=args.dry_run,
        )
        result["scenario"] = name
        result["phase"] = 1
        results.append(result)

    return results


def run_phase2(
    s3_base: str,
    ts: str,
    data_uri: str,
    source_uri: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Phase 2: Expert Ablation — 7 training scenarios.

    Each scenario removes one shared expert from the PLE basket.
    """
    logger.info("=" * 70)
    logger.info("Phase 2: Expert Ablation (%d scenarios)",
                len(EXPERT_ABLATION_SCENARIOS))
    logger.info("=" * 70)

    results = []

    for scenario in EXPERT_ABLATION_SCENARIOS:
        name = scenario["name"]
        safe_name = name.replace("_", "-")
        job_name = f"ablation-p2-{safe_name}-{ts}"
        s3_output = f"{s3_base}/phase2/{name}/"

        hyperparameters = {
            "config": CONFIG_PATH,
            "ablation_phase": "2",
            "ablation_type": "expert",
            "ablation_scenario": name,
            "shared_experts": json.dumps(scenario["shared"]),
            "epochs": "30",
            "batch_size": "128",
            "learning_rate": "0.001",
            "seed": "42",
            "_s3_output": f"{s3_base}/phase2/{name}",
        }

        logger.info("--- Scenario: %s (experts=%s) ---", name, scenario["shared"])

        result = _submit_training_job(
            job_name=job_name,
            s3_output=s3_output,
            instance_type=args.instance_type_gpu,
            hyperparameters=hyperparameters,
            source_uri=source_uri,
            data_uri=data_uri,
            wait=not args.no_wait,
            dry_run=args.dry_run,
        )
        result["scenario"] = name
        result["phase"] = 2
        results.append(result)

    return results


def run_phase3(
    s3_base: str,
    ts: str,
    data_uri: str,
    source_uri: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Phase 3: Hyperparameter Sensitivity — 12 training scenarios.

    Sweeps learning_rate (4), temperature (4), and num_layers (3).
    """
    logger.info("=" * 70)
    logger.info("Phase 3: Hyperparameter Sensitivity (%d scenarios)",
                len(HP_SCENARIOS))
    logger.info("=" * 70)

    results = []

    for scenario in HP_SCENARIOS:
        name = scenario["name"]
        safe_name = name.replace("_", "-").replace(".", "-")
        job_name = f"ablation-p3-{safe_name}-{ts}"
        s3_output = f"{s3_base}/phase3/{name}/"

        hyperparameters = {
            "config": CONFIG_PATH,
            "ablation_phase": "3",
            "ablation_type": "hyperparameter",
            "ablation_scenario": name,
            "epochs": "30",
            "batch_size": "128",
            "learning_rate": str(scenario.get("learning_rate", 0.001)),
            "temperature": str(scenario.get("temperature", 5.0)),
            "num_layers": str(scenario.get("num_layers", 3)),
            "seed": "42",
            "_s3_output": f"{s3_base}/phase3/{name}",
        }

        logger.info("--- Scenario: %s ---", name)

        result = _submit_training_job(
            job_name=job_name,
            s3_output=s3_output,
            instance_type=args.instance_type_gpu,
            hyperparameters=hyperparameters,
            source_uri=source_uri,
            data_uri=data_uri,
            wait=not args.no_wait,
            dry_run=args.dry_run,
        )
        result["scenario"] = name
        result["phase"] = 3
        results.append(result)

    return results


def _select_best_config(
    phase1_results: List[Dict[str, Any]],
    phase2_results: List[Dict[str, Any]],
    phase3_results: List[Dict[str, Any]],
    s3_base: str,
) -> Dict[str, Any]:
    """Analyze Phase 1-3 results and select the best configuration.

    Collects eval_metrics.json from each completed scenario, ranks by
    primary metric, and returns the best combination of feature group,
    expert basket, and hyperparameters.

    Returns
    -------
    dict
        Best configuration with keys: feature_scenario, expert_scenario,
        learning_rate, temperature, num_layers, and the combined metric score.
    """
    best = {
        "feature_scenario": "full",
        "expert_scenario": "full_basket",
        "learning_rate": 0.001,
        "temperature": 5.0,
        "num_layers": 3,
        "best_metric": None,
    }

    # Collect Phase 1 metrics — find best feature config
    best_feature_score = -float("inf")
    for r in phase1_results:
        if r.get("status") != "Completed":
            continue
        s3_path = f"{s3_base}/phase1/{r['scenario']}/output/eval_metrics.json"
        metrics = _download_json_from_s3(s3_path)
        if metrics:
            # Use average primary metric across tasks as aggregate score
            score = metrics.get("aggregate_score", 0.0)
            if score > best_feature_score:
                best_feature_score = score
                best["feature_scenario"] = r["scenario"]

    # Collect Phase 2 metrics — find best expert config
    best_expert_score = -float("inf")
    for r in phase2_results:
        if r.get("status") != "Completed":
            continue
        s3_path = f"{s3_base}/phase2/{r['scenario']}/output/eval_metrics.json"
        metrics = _download_json_from_s3(s3_path)
        if metrics:
            score = metrics.get("aggregate_score", 0.0)
            if score > best_expert_score:
                best_expert_score = score
                best["expert_scenario"] = r["scenario"]

    # Collect Phase 3 metrics — find best HP config
    best_hp_score = -float("inf")
    for r in phase3_results:
        if r.get("status") != "Completed":
            continue
        s3_path = f"{s3_base}/phase3/{r['scenario']}/output/eval_metrics.json"
        metrics = _download_json_from_s3(s3_path)
        if metrics:
            score = metrics.get("aggregate_score", 0.0)
            if score > best_hp_score:
                best_hp_score = score
                scenario_name = r["scenario"]
                # Parse HP from scenario name
                if scenario_name.startswith("lr_"):
                    best["learning_rate"] = float(scenario_name.replace("lr_", ""))
                elif scenario_name.startswith("temp_"):
                    best["temperature"] = float(scenario_name.replace("temp_", ""))
                elif scenario_name.startswith("layers_"):
                    best["num_layers"] = int(scenario_name.replace("layers_", ""))

    best["best_metric"] = max(
        best_feature_score, best_expert_score, best_hp_score,
    ) if best_feature_score > -float("inf") else None

    logger.info("Best config selected:")
    logger.info("  Feature scenario: %s", best["feature_scenario"])
    logger.info("  Expert scenario:  %s", best["expert_scenario"])
    logger.info("  Learning rate:    %s", best["learning_rate"])
    logger.info("  Temperature:      %s", best["temperature"])
    logger.info("  Num layers:       %s", best["num_layers"])

    return best


def run_phase4(
    s3_base: str,
    ts: str,
    data_uri: str,
    source_uri: str,
    args: argparse.Namespace,
    best_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase 4: Full Pipeline with best config.

    Runs: Teacher Training -> Distillation -> Fidelity -> Feature Selection -> Eval.
    """
    logger.info("=" * 70)
    logger.info("Phase 4: Best Config Full Pipeline")
    logger.info("=" * 70)

    if best_config is None:
        best_config = {
            "feature_scenario": "full",
            "expert_scenario": "full_basket",
            "learning_rate": 0.001,
            "temperature": 5.0,
            "num_layers": 3,
        }

    # Step 4a: Train teacher with best config (or use provided model-uri)
    model_uri = args.model_uri
    teacher_result: Dict[str, Any] = {}

    if not model_uri:
        job_name = f"ablation-p4-teacher-{ts}"
        s3_output = f"{s3_base}/phase4/teacher/"

        # Resolve expert list from best scenario
        expert_scenario = next(
            (s for s in EXPERT_ABLATION_SCENARIOS
             if s["name"] == best_config["expert_scenario"]),
            EXPERT_ABLATION_SCENARIOS[0],
        )

        hyperparameters = {
            "config": CONFIG_PATH,
            "ablation_phase": "4",
            "ablation_type": "best_config",
            "shared_experts": json.dumps(expert_scenario["shared"]),
            "epochs": "30",
            "batch_size": "128",
            "learning_rate": str(best_config["learning_rate"]),
            "num_layers": str(best_config["num_layers"]),
            "seed": "42",
            "_s3_output": f"{s3_base}/phase4/teacher",
        }

        teacher_result = _submit_training_job(
            job_name=job_name,
            s3_output=s3_output,
            instance_type=args.instance_type_gpu,
            hyperparameters=hyperparameters,
            source_uri=source_uri,
            data_uri=data_uri,
            wait=not args.no_wait,
            dry_run=args.dry_run,
        )

        model_uri = teacher_result.get("model_uri", "")
        if not model_uri and not args.dry_run:
            logger.error("Teacher training failed — no model artifact")
            return {"status": "Failed", "teacher": teacher_result}
    else:
        logger.info("Using provided teacher model: %s", model_uri)
        teacher_result = {"model_uri": model_uri, "status": "Provided"}

    # Step 4b: Distillation + Fidelity + Feature Selection + Eval
    distill_job_name = f"ablation-p4-distill-{ts}"
    s3_distill_output = f"{s3_base}/phase4/distillation/"

    from sagemaker.processing import ProcessingInput

    distill_inputs = []
    if model_uri and not args.dry_run:
        distill_inputs.append(
            ProcessingInput(
                source=model_uri,
                destination="/opt/ml/processing/input/teacher",
            )
        )
        distill_inputs.append(
            ProcessingInput(
                source=data_uri,
                destination="/opt/ml/processing/input/data",
            )
        )

    distill_arguments = [
        "--teacher-checkpoint", "/opt/ml/processing/input/teacher/model.pth",
        "--data-path", "/opt/ml/processing/input/data/",
        "--output-dir", "/opt/ml/processing/output",
        "--config", CONFIG_PATH,
        "--soft-label-path", "/opt/ml/processing/output/soft_labels.parquet",
        "--temperature", str(best_config["temperature"]),
    ]
    if args.skip_fidelity_gate:
        distill_arguments.append("--skip-fidelity-gate")

    distill_result = _submit_processing_job(
        job_name=distill_job_name,
        script="scripts/run_distillation.py",
        s3_output=s3_distill_output,
        instance_type=args.instance_type_cpu,
        arguments=distill_arguments,
        inputs=distill_inputs if not args.dry_run else None,
        wait=not args.no_wait,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
    )

    return {
        "status": distill_result.get("status", "Unknown"),
        "teacher": teacher_result,
        "distillation": distill_result,
        "best_config": best_config,
        "model_uri": model_uri,
    }


def run_phase5(
    s3_base: str,
    ts: str,
    all_results: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Phase 5: Result Collection + HTML Report Generation.

    Collects all metrics from S3 and invokes generate_ablation_report.py.
    """
    logger.info("=" * 70)
    logger.info("Phase 5: Result Collection + HTML Report")
    logger.info("=" * 70)

    # Save the orchestration results as a manifest
    manifest_path = f"{s3_base}/manifest.json"
    manifest = {
        "timestamp": ts,
        "s3_base": s3_base,
        "phases": {},
    }

    for phase_key, phase_results in all_results.items():
        if isinstance(phase_results, list):
            manifest["phases"][phase_key] = [
                {
                    "scenario": r.get("scenario", ""),
                    "job_name": r.get("job_name", ""),
                    "status": r.get("status", "Unknown"),
                    "model_uri": r.get("model_uri", ""),
                }
                for r in phase_results
            ]
        elif isinstance(phase_results, dict):
            # Preserve all keys for dict-type phase results (phase0, phase4, phase5)
            # so downstream report generators can access data_uri, model_uri, best_config, etc.
            manifest["phases"][phase_key] = {
                k: v for k, v in phase_results.items()
                if _is_json_serializable(v)
            }

    if not args.dry_run:
        import boto3
        s3 = boto3.client("s3", region_name=REGION)
        manifest_key = manifest_path.replace(f"s3://{S3_BUCKET}/", "")
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            ContentType="application/json",
        )
        logger.info("Manifest saved to %s", manifest_path)

    # Generate HTML report
    report_output = f"docs/ablation_report_{ts}.html"
    logger.info("Generating HTML report: %s", report_output)

    if not args.dry_run:
        try:
            # Import and run the report generator in-process
            sys.path.insert(0, str(PROJECT_ROOT))
            from scripts.generate_ablation_report import generate_report

            generate_report(
                s3_base=s3_base,
                output_path=str(PROJECT_ROOT / report_output),
            )
            logger.info("Report generated: %s", report_output)
        except ImportError:
            logger.warning(
                "generate_ablation_report.py not importable — "
                "run it separately: python scripts/generate_ablation_report.py "
                "--s3-base %s --output %s",
                s3_base, report_output,
            )
    else:
        logger.info("[DRY RUN] Would generate report: %s", report_output)

    return {
        "manifest": manifest_path,
        "report": report_output,
    }


# ===================================================================
# Main orchestration
# ===================================================================

def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    s3_base = _s3_base_path(ts)

    logger.info("=" * 70)
    logger.info("Ablation Test Orchestrator")
    logger.info("  Timestamp:      %s", ts)
    logger.info("  S3 Base:        %s", s3_base)
    logger.info("  Phase:          %s", args.phase)
    logger.info("  GPU Instance:   %s", args.instance_type_gpu)
    logger.info("  CPU Instance:   %s", args.instance_type_cpu)
    logger.info("  Dry Run:        %s", args.dry_run)
    logger.info("  Model URI:      %s", args.model_uri or "(none)")
    logger.info("=" * 70)

    run_all = args.phase == "all"
    phase = args.phase

    # Prepare source package
    if not args.dry_run:
        tmp_dir = tempfile.mkdtemp(prefix="ablation_src_")
        source_tar = _prepare_source_package(tmp_dir)
        source_uri = _upload_source_to_s3(source_tar, s3_base)
    else:
        source_uri = f"{s3_base}/source/source.tar.gz"
        logger.info("[DRY RUN] Source package: %s", source_uri)

    # Default data URI — point to the directory so training jobs get both
    # the parquet features and the event_sequences .npy files.
    # Check env override for reusing a previous Phase 0 output.
    data_uri = os.environ.get(
        "ABLATION_DATA_URI",
        f"s3://{S3_BUCKET}/data/adapted/",
    )

    # Collect results from each phase
    all_results: Dict[str, Any] = {}

    # Phase 0: Data Preparation
    if run_all or phase == "0":
        p0_result = run_phase0(s3_base, ts, args)
        all_results["phase0"] = p0_result
        if p0_result.get("data_uri"):
            data_uri = p0_result["data_uri"]

    # Phase 1: Feature Group Ablation
    p1_results: List[Dict[str, Any]] = []
    if run_all or phase == "1":
        p1_results = run_phase1(s3_base, ts, data_uri, source_uri, args)
        all_results["phase1"] = p1_results

    # Phase 2: Expert Ablation
    p2_results: List[Dict[str, Any]] = []
    if run_all or phase == "2":
        p2_results = run_phase2(s3_base, ts, data_uri, source_uri, args)
        all_results["phase2"] = p2_results

    # Phase 3: Hyperparameter Sensitivity
    p3_results: List[Dict[str, Any]] = []
    if run_all or phase == "3":
        p3_results = run_phase3(s3_base, ts, data_uri, source_uri, args)
        all_results["phase3"] = p3_results

    # Phase 4: Best Config Full Pipeline
    if run_all or phase == "4":
        # Select best config from Phase 1-3 (or defaults if phases were skipped)
        best_config = _select_best_config(p1_results, p2_results, p3_results, s3_base)
        p4_result = run_phase4(s3_base, ts, data_uri, source_uri, args, best_config)
        all_results["phase4"] = p4_result

    # Phase 5: Result Collection + Report
    if run_all or phase == "5":
        p5_result = run_phase5(s3_base, ts, all_results, args)
        all_results["phase5"] = p5_result

    # Final summary
    logger.info("=" * 70)
    logger.info("Ablation Test Complete!")
    logger.info("  S3 Base:  %s", s3_base)
    logger.info("  Phases:   %s", list(all_results.keys()))
    total_jobs = sum(
        len(v) if isinstance(v, list) else 1
        for v in all_results.values()
    )
    logger.info("  Total Jobs Submitted: %d", total_jobs)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
