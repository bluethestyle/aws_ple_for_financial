#!/usr/bin/env bash
# Build + push the custom Mamba GPU training image to ECR.
#
# Why
# ---
# The AWS PyTorch 2.1 GPU DLC can't build mamba_ssm at job start
# (ninja missing, urllib fetch flaky, ~7 min build per job that
# fails ~half the time). This image bakes the wheels in once so
# the SageMaker training job picks them up immediately.
#
# Usage
# -----
#     bash scripts/build_mamba_image.sh
#
# Env overrides (defaults match aws.region / aws.s3_bucket layout):
#     REGION=ap-northeast-2
#     ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
#     IMAGE_REPO=ple-mamba-precompute
#     IMAGE_TAG=$(git rev-parse --short HEAD)
#
# Output: prints the resolved image URI on success — copy it to
# pipeline.yaml::aws.mamba_image_uri.
#
set -euo pipefail

REGION="${REGION:-ap-northeast-2}"
ACCOUNT_ID="${ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
IMAGE_REPO="${IMAGE_REPO:-ple-mamba-precompute}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d-%H%M%S)}"

ECR_HOST="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${ECR_HOST}/${IMAGE_REPO}:${IMAGE_TAG}"

DLC_HOST="763104351884.dkr.ecr.${REGION}.amazonaws.com"

echo "[mamba-image] region        = ${REGION}"
echo "[mamba-image] account       = ${ACCOUNT_ID}"
echo "[mamba-image] target URI    = ${IMAGE_URI}"
echo "[mamba-image] base DLC host = ${DLC_HOST}"

# 1) ECR login for the AWS DLC base image (different account 763104351884)
echo "[mamba-image] logging in to base DLC ECR …"
aws ecr get-login-password --region "${REGION}" \
    | docker login --username AWS --password-stdin "${DLC_HOST}"

# 2) ECR login for our own ECR (where we push the result)
echo "[mamba-image] logging in to target ECR …"
aws ecr get-login-password --region "${REGION}" \
    | docker login --username AWS --password-stdin "${ECR_HOST}"

# 3) Create our ECR repository if it doesn't exist
if ! aws ecr describe-repositories \
        --repository-names "${IMAGE_REPO}" \
        --region "${REGION}" >/dev/null 2>&1; then
    echo "[mamba-image] creating ECR repo ${IMAGE_REPO}"
    aws ecr create-repository \
        --repository-name "${IMAGE_REPO}" \
        --region "${REGION}" \
        --image-scanning-configuration scanOnPush=true >/dev/null
fi

# 4) Build
echo "[mamba-image] building image …"
docker build \
    --build-arg "REGION=${REGION}" \
    -f containers/mamba/Dockerfile \
    -t "${IMAGE_URI}" \
    .

# 5) Push
echo "[mamba-image] pushing to ECR …"
docker push "${IMAGE_URI}"

echo
echo "[mamba-image] DONE"
echo "[mamba-image] URI: ${IMAGE_URI}"
echo
echo "[mamba-image] Next: set the URI in pipeline.yaml:"
echo "    aws:"
echo "      mamba_image_uri: ${IMAGE_URI}"
