#!/bin/bash
# =============================================================================
# Deploy Lambda serving endpoint from SageMaker Notebook
# Run this script in a SageMaker notebook terminal
# =============================================================================
set -e

REGION="ap-northeast-2"
ACCOUNT_ID="795833413857"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/AWSPLEPlatformSageMakerRole"
S3_BUCKET="aiops-ple-financial"
STUDENTS_PATH="e2e-test/20260320-144441/students"
FUNCTION_NAME="ple-predict"
LAYER_NAME="ple-lgbm-layer"

echo "=== Step 1: Build Lambda Layer (lightgbm + numpy) ==="
LAYER_DIR=$(mktemp -d)
pip install lightgbm numpy -t "${LAYER_DIR}/python" -q
cd "${LAYER_DIR}"
zip -r9 lambda_layer.zip python/ -q
LAYER_SIZE=$(du -sh lambda_layer.zip | cut -f1)
echo "Layer size: ${LAYER_SIZE}"

echo "=== Step 2: Publish Lambda Layer ==="
LAYER_ARN=$(aws lambda publish-layer-version \
  --layer-name "${LAYER_NAME}" \
  --zip-file "fileb://lambda_layer.zip" \
  --compatible-runtimes python3.9 python3.10 python3.11 \
  --region "${REGION}" \
  --query "LayerVersionArn" --output text)
echo "Layer ARN: ${LAYER_ARN}"

echo "=== Step 3: Package Lambda Function ==="
FUNC_DIR=$(mktemp -d)
cd "${FUNC_DIR}"

# Clone the repo and copy predict.py
pip install awscli -q 2>/dev/null || true
git clone --depth 1 https://github.com/bluethestyle/aws_ple_for_financial.git repo
cp repo/core/serving/predict.py lambda_function.py

# Create a minimal handler wrapper
cat > handler.py << 'HANDLER_EOF'
"""Lambda handler for PLE prediction serving."""
import json
import os
import logging
import time
import tempfile
import tarfile

import boto3
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Globals for warm start
_MODELS = {}
_S3 = None


def _load_models():
    """Load LGBM models from S3 on cold start."""
    global _MODELS, _S3
    import lightgbm as lgb

    if _S3 is None:
        _S3 = boto3.client("s3")

    bucket = os.environ.get("MODEL_BUCKET", "aiops-ple-financial")
    prefix = os.environ.get("MODEL_PREFIX", "e2e-test/20260320-144441/students")

    # List task directories
    resp = _S3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/", Delimiter="/")
    task_dirs = []
    for cp in resp.get("CommonPrefixes", []):
        task_name = cp["Prefix"].rstrip("/").split("/")[-1]
        if task_name not in ("soft_labels.parquet", "fidelity_report.json",
                             "distillation_summary.json", "test_evaluation.json"):
            task_dirs.append(task_name)

    tmp_dir = tempfile.mkdtemp()
    for task_name in task_dirs:
        model_key = f"{prefix}/{task_name}/model.lgbm"
        local_path = os.path.join(tmp_dir, f"{task_name}.lgbm")
        try:
            _S3.download_file(bucket, model_key, local_path)
            _MODELS[task_name] = lgb.Booster(model_file=local_path)
            logger.info("Loaded model: %s", task_name)
        except Exception as e:
            logger.warning("Failed to load %s: %s", task_name, e)

    logger.info("Total models loaded: %d", len(_MODELS))


def handler(event, context):
    """Lambda handler for /predict endpoint."""
    start = time.time()

    # Cold start: load models
    if not _MODELS:
        _load_models()

    body = event if isinstance(event, dict) else json.loads(event.get("body", "{}"))
    user_id = body.get("user_id", "unknown")
    features = body.get("features", {})
    ctx = body.get("context", {})

    # Build feature vector
    feat_names = sorted(features.keys())
    feat_values = np.array([[features[k] for k in feat_names]])

    # Predict with each student model
    predictions = {}
    for task_name, model in _MODELS.items():
        try:
            pred = model.predict(feat_values)[0]
            if isinstance(pred, np.ndarray):
                predictions[task_name] = pred.tolist()
            else:
                predictions[task_name] = float(pred)
        except Exception as e:
            predictions[task_name] = {"error": str(e)}

    elapsed_ms = (time.time() - start) * 1000

    response = {
        "user_id": user_id,
        "predictions": predictions,
        "elapsed_ms": round(elapsed_ms, 2),
        "models_loaded": len(_MODELS),
        "features_received": len(features),
    }

    return {
        "statusCode": 200,
        "body": json.dumps(response, default=str),
    }
HANDLER_EOF

zip -r9 lambda_function.zip handler.py -q

echo "=== Step 4: Create/Update Lambda Function ==="
# Check if function exists
if aws lambda get-function --function-name "${FUNCTION_NAME}" --region "${REGION}" 2>/dev/null; then
    echo "Updating existing function..."
    aws lambda update-function-code \
      --function-name "${FUNCTION_NAME}" \
      --zip-file "fileb://lambda_function.zip" \
      --region "${REGION}" \
      --query "FunctionArn" --output text

    sleep 5

    aws lambda update-function-configuration \
      --function-name "${FUNCTION_NAME}" \
      --layers "${LAYER_ARN}" \
      --region "${REGION}" \
      --query "FunctionArn" --output text
else
    echo "Creating new function..."
    aws lambda create-function \
      --function-name "${FUNCTION_NAME}" \
      --runtime python3.10 \
      --handler handler.handler \
      --role "${ROLE_ARN}" \
      --zip-file "fileb://lambda_function.zip" \
      --layers "${LAYER_ARN}" \
      --timeout 60 \
      --memory-size 1024 \
      --environment "Variables={MODEL_BUCKET=${S3_BUCKET},MODEL_PREFIX=${STUDENTS_PATH}}" \
      --region "${REGION}" \
      --query "FunctionArn" --output text
fi

echo "=== Step 5: Create Function URL (public endpoint for testing) ==="
aws lambda create-function-url-config \
  --function-name "${FUNCTION_NAME}" \
  --auth-type NONE \
  --region "${REGION}" 2>/dev/null || true

aws lambda add-permission \
  --function-name "${FUNCTION_NAME}" \
  --statement-id "FunctionURLPublicAccess" \
  --action "lambda:InvokeFunctionUrl" \
  --principal "*" \
  --function-url-auth-type NONE \
  --region "${REGION}" 2>/dev/null || true

FUNC_URL=$(aws lambda get-function-url-config \
  --function-name "${FUNCTION_NAME}" \
  --region "${REGION}" \
  --query "FunctionUrl" --output text 2>/dev/null || echo "N/A")

echo ""
echo "=== Deployment Complete ==="
echo "Function: ${FUNCTION_NAME}"
echo "Layer: ${LAYER_ARN}"
echo "URL: ${FUNC_URL}"
echo ""
echo "Test with:"
echo "  curl -X POST ${FUNC_URL} -H 'Content-Type: application/json' -d '{\"user_id\": \"test_001\", \"features\": {\"feat_0\": 0.1, \"feat_1\": 0.2}}'"

# Cleanup
rm -rf "${LAYER_DIR}" "${FUNC_DIR}"
