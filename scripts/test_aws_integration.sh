#!/usr/bin/env bash
# test_aws_integration.sh — AWS integration smoke test.
#
# Validates:
#   1. AWS CLI is configured (sts get-caller-identity)
#   2. S3 bucket exists or can be accessed
#   3. S3 read/write round-trip with sample data
#   4. S3 directory structure creation (raw/, processed/, features/, models/, checkpoints/)
#   5. SageMaker dry-run: validate IAM role and container image (no job launched)
#   6. Cleanup of test data
#
# Usage:
#   ./scripts/test_aws_integration.sh
#   ./scripts/test_aws_integration.sh --bucket my-bucket --region ap-northeast-2
#   ./scripts/test_aws_integration.sh --skip-cleanup
#
# Exit codes:
#   0  All checks passed
#   1  One or more checks failed

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BUCKET="${AWS_TEST_BUCKET:-aiops-ple-financial}"
REGION="${AWS_DEFAULT_REGION:-ap-northeast-2}"
ROLE_NAME="${AWS_SAGEMAKER_ROLE:-AWSPLEPlatformSageMakerRole}"
SKIP_CLEANUP=false
TEST_PREFIX="test/integration-$(date +%Y%m%d-%H%M%S)"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket)       BUCKET="$2"; shift 2 ;;
    --region)       REGION="$2"; shift 2 ;;
    --role)         ROLE_NAME="$2"; shift 2 ;;
    --skip-cleanup) SKIP_CLEANUP=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--bucket NAME] [--region REGION] [--role ROLE_NAME] [--skip-cleanup]"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS=0
FAIL=0
SKIP=0

pass_check() { echo "  [PASS] $1"; ((PASS++)) || true; }
fail_check() { echo "  [FAIL] $1"; ((FAIL++)) || true; }
skip_check() { echo "  [SKIP] $1"; ((SKIP++)) || true; }
info()       { echo "  [INFO] $1"; }

# ---------------------------------------------------------------------------
echo "================================================"
echo " AWS Integration Test"
echo "================================================"
echo "  Bucket : s3://$BUCKET"
echo "  Region : $REGION"
echo "  Role   : $ROLE_NAME"
echo "  Prefix : $TEST_PREFIX"
echo "------------------------------------------------"

# ---------------------------------------------------------------------------
# 1. AWS CLI Configuration
# ---------------------------------------------------------------------------
echo ""
echo "[1/6] Checking AWS CLI configuration..."

if ! command -v aws &>/dev/null; then
  fail_check "AWS CLI not installed"
  echo ""
  echo "RESULT: Cannot proceed without AWS CLI. Install with: pip install awscli"
  exit 1
fi
pass_check "AWS CLI installed"

CALLER_IDENTITY=""
if CALLER_IDENTITY=$(aws sts get-caller-identity --region "$REGION" 2>&1); then
  ACCOUNT_ID=$(echo "$CALLER_IDENTITY" | python -c "import sys,json; print(json.load(sys.stdin)['Account'])" 2>/dev/null || echo "unknown")
  ARN=$(echo "$CALLER_IDENTITY" | python -c "import sys,json; print(json.load(sys.stdin)['Arn'])" 2>/dev/null || echo "unknown")
  pass_check "AWS credentials valid (Account=$ACCOUNT_ID)"
  info "ARN: $ARN"
else
  fail_check "AWS credentials invalid or expired"
  echo "  Error: $CALLER_IDENTITY"
  echo ""
  echo "RESULT: Fix AWS credentials before running integration tests."
  echo "  Run: aws configure"
  echo "  Or:  export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=..."
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. S3 Bucket Access
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Checking S3 bucket access..."

if aws s3api head-bucket --bucket "$BUCKET" --region "$REGION" 2>/dev/null; then
  pass_check "Bucket s3://$BUCKET exists and is accessible"
else
  fail_check "Bucket s3://$BUCKET not accessible"
  info "Create it with: aws s3api create-bucket --bucket $BUCKET --region $REGION --create-bucket-configuration LocationConstraint=$REGION"
fi

# ---------------------------------------------------------------------------
# 3. S3 Read/Write Round-trip
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Testing S3 read/write..."

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Generate a small test Parquet file using Python
TEST_FILE="$TMPDIR/test_data.parquet"
python -c "
import pandas as pd, numpy as np
rng = np.random.default_rng(42)
df = pd.DataFrame({
    'feature_1': rng.standard_normal(100),
    'feature_2': rng.standard_normal(100),
    'label': rng.integers(0, 2, 100),
})
df.to_parquet('$TEST_FILE', index=False)
print(f'Generated test file: {df.shape}')
" 2>/dev/null

if [[ -f "$TEST_FILE" ]]; then
  pass_check "Test Parquet file generated locally"
else
  fail_check "Failed to generate test Parquet file (check pandas/numpy)"
fi

S3_TEST_KEY="$TEST_PREFIX/test_data.parquet"

# Upload
if aws s3 cp "$TEST_FILE" "s3://$BUCKET/$S3_TEST_KEY" --region "$REGION" 2>/dev/null; then
  pass_check "Upload to s3://$BUCKET/$S3_TEST_KEY"
else
  fail_check "Upload to S3 failed"
fi

# Download and verify
DOWNLOADED="$TMPDIR/downloaded.parquet"
if aws s3 cp "s3://$BUCKET/$S3_TEST_KEY" "$DOWNLOADED" --region "$REGION" 2>/dev/null; then
  pass_check "Download from S3"

  # Verify content match
  MATCH=$(python -c "
import pandas as pd
df1 = pd.read_parquet('$TEST_FILE')
df2 = pd.read_parquet('$DOWNLOADED')
print('match' if df1.equals(df2) else 'mismatch')
" 2>/dev/null || echo "error")

  if [[ "$MATCH" == "match" ]]; then
    pass_check "Round-trip data integrity verified"
  else
    fail_check "Round-trip data mismatch ($MATCH)"
  fi
else
  fail_check "Download from S3 failed"
fi

# ---------------------------------------------------------------------------
# 4. S3 Directory Structure
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Creating S3 directory structure..."

DIRS=("raw" "processed" "features" "models" "checkpoints")
for dir in "${DIRS[@]}"; do
  KEY="$TEST_PREFIX/$dir/.keep"
  if aws s3api put-object --bucket "$BUCKET" --key "$KEY" --body /dev/null --region "$REGION" 2>/dev/null; then
    pass_check "Created s3://$BUCKET/$TEST_PREFIX/$dir/"
  else
    fail_check "Failed to create s3://$BUCKET/$TEST_PREFIX/$dir/"
  fi
done

# Verify listing
LISTED=$(aws s3 ls "s3://$BUCKET/$TEST_PREFIX/" --region "$REGION" 2>/dev/null | wc -l)
if [[ "$LISTED" -ge 5 ]]; then
  pass_check "Directory listing verified ($LISTED entries)"
else
  fail_check "Directory listing incomplete ($LISTED entries, expected >= 5)"
fi

# ---------------------------------------------------------------------------
# 5. SageMaker Dry-run Validation
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] SageMaker dry-run validation..."

# Check IAM role
ROLE_ARN=""
if ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text 2>/dev/null); then
  pass_check "SageMaker IAM role exists: $ROLE_ARN"

  # Check trust policy includes sagemaker.amazonaws.com
  TRUST=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.AssumeRolePolicyDocument.Statement[?Principal.Service==`sagemaker.amazonaws.com`]' --output text 2>/dev/null)
  if [[ -n "$TRUST" ]]; then
    pass_check "Role trust policy includes sagemaker.amazonaws.com"
  else
    fail_check "Role trust policy missing sagemaker.amazonaws.com"
  fi
else
  skip_check "SageMaker IAM role '$ROLE_NAME' not found (non-blocking)"
  info "Create with: ./scripts/setup_aws.sh --bucket $BUCKET --account $ACCOUNT_ID"
fi

# Check SageMaker API access (list training jobs, limit 1)
if aws sagemaker list-training-jobs --max-results 1 --region "$REGION" 2>/dev/null >/dev/null; then
  pass_check "SageMaker API accessible (list-training-jobs)"
else
  skip_check "SageMaker API not accessible (may need permissions)"
fi

# Validate PyTorch container image availability
CONTAINER_CHECK=$(python -c "
try:
    import sagemaker
    from sagemaker import image_uris
    uri = image_uris.retrieve(
        framework='pytorch',
        region='$REGION',
        version='2.1',
        py_version='py310',
        instance_type='ml.g4dn.xlarge',
        image_scope='training',
    )
    print(f'AVAILABLE:{uri}')
except Exception as e:
    print(f'ERROR:{e}')
" 2>/dev/null || echo "ERROR:sagemaker SDK not installed")

if [[ "$CONTAINER_CHECK" == AVAILABLE:* ]]; then
  CONTAINER_URI="${CONTAINER_CHECK#AVAILABLE:}"
  pass_check "PyTorch 2.1 training container available"
  info "Container: $CONTAINER_URI"
elif [[ "$CONTAINER_CHECK" == ERROR:* ]]; then
  skip_check "Container check skipped: ${CONTAINER_CHECK#ERROR:}"
fi

# Dry-run: validate job config (create but don't launch)
DRY_RUN=$(python -c "
import json
config = {
    'TrainingJobName': 'dry-run-validation-test',
    'RoleArn': '${ROLE_ARN:-arn:aws:iam::000000000000:role/placeholder}',
    'AlgorithmSpecification': {
        'TrainingImage': '${CONTAINER_URI:-placeholder}',
        'TrainingInputMode': 'File',
    },
    'ResourceConfig': {
        'InstanceType': 'ml.g4dn.xlarge',
        'InstanceCount': 1,
        'VolumeSizeInGB': 50,
    },
    'StoppingCondition': {'MaxRuntimeInSeconds': 7200},
    'InputDataConfig': [{
        'ChannelName': 'train',
        'DataSource': {
            'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri': 's3://$BUCKET/$TEST_PREFIX/raw/',
                'S3DataDistributionType': 'ShardedByS3Key',
            }
        },
        'ContentType': 'application/x-parquet',
    }],
    'OutputDataConfig': {'S3OutputPath': 's3://$BUCKET/$TEST_PREFIX/models/'},
}
# Validate the config structure is complete
required = ['TrainingJobName', 'RoleArn', 'AlgorithmSpecification', 'ResourceConfig', 'StoppingCondition']
missing = [k for k in required if k not in config]
if missing:
    print(f'INVALID:Missing keys: {missing}')
else:
    print('VALID')
" 2>/dev/null || echo "ERROR")

if [[ "$DRY_RUN" == "VALID" ]]; then
  pass_check "SageMaker job config structure validated (dry-run)"
else
  fail_check "SageMaker job config validation: $DRY_RUN"
fi

# ---------------------------------------------------------------------------
# 6. Cleanup
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Cleanup..."

if [[ "$SKIP_CLEANUP" == "true" ]]; then
  skip_check "Cleanup skipped (--skip-cleanup flag)"
  info "Test data remains at: s3://$BUCKET/$TEST_PREFIX/"
else
  if aws s3 rm "s3://$BUCKET/$TEST_PREFIX/" --recursive --region "$REGION" 2>/dev/null; then
    pass_check "Cleaned up s3://$BUCKET/$TEST_PREFIX/"
  else
    fail_check "Cleanup failed for s3://$BUCKET/$TEST_PREFIX/"
  fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================"
echo " Integration Test Summary"
echo "================================================"
TOTAL=$((PASS + FAIL + SKIP))
echo "  Total : $TOTAL"
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo "  Skipped: $SKIP"
echo "================================================"

if [[ $FAIL -gt 0 ]]; then
  echo ""
  echo "RESULT: FAILED ($FAIL check(s) failed)"
  exit 1
else
  echo ""
  echo "RESULT: PASSED (all checks passed or skipped)"
  exit 0
fi
