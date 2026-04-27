#!/usr/bin/env bash
# Build + push the Mamba custom image via AWS CodeBuild.
#
# Why
# ---
# Local Docker Desktop on Windows kept failing the ninja-build apt
# step with "tls: bad record MAC" (BuildKit network stack issue).
# CodeBuild runs the same Dockerfile on a clean Linux host with
# guaranteed IAM + ECR connectivity, so it works first try.
#
# What it does
# ------------
# 1. Creates the CodeBuild service role + ECR/CloudWatch policies
#    if missing (idempotent).
# 2. Creates the CodeBuild project ``ple-mamba-image-build`` if
#    missing.
# 3. Zips the repo's containers/ + scripts/ subset, uploads to
#    s3://{bucket}/{task}/codebuild/source-{ts}.zip.
# 4. Starts the build, polls every 30s, prints image URI on
#    success.
#
# Cost: ~$0.30 per build on BUILD_GENERAL1_MEDIUM (15-25 min).
#
# Usage:
#     bash scripts/build_mamba_image_codebuild.sh
#
set -euo pipefail

REGION="${REGION:-ap-northeast-2}"
ACCOUNT_ID="${ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
S3_BUCKET="${S3_BUCKET:-aiops-ple-financial}"
TASK="${TASK:-santander_ple}"
ROLE_NAME="${ROLE_NAME:-ple-mamba-codebuild-role}"
PROJECT_NAME="${PROJECT_NAME:-ple-mamba-image-build}"
IMAGE_REPO="${IMAGE_REPO:-ple-mamba-precompute}"
DLC_ACCOUNT="763104351884"

TS=$(date +%Y%m%d-%H%M%S)
SRC_KEY="${TASK}/codebuild/source-${TS}.zip"
SRC_S3="s3://${S3_BUCKET}/${SRC_KEY}"

echo "[codebuild] region=${REGION} account=${ACCOUNT_ID}"
echo "[codebuild] project=${PROJECT_NAME} role=${ROLE_NAME}"
echo "[codebuild] source S3 = ${SRC_S3}"

# ─── 1. IAM role ──────────────────────────────────────────────
# Pass JSON inline (single-line, escaped) instead of via file://
# to dodge the Windows MSYS path translation issue when this
# script is run from Git Bash on Windows: AWS CLI is a Windows
# python process and doesn't understand /tmp or /c/Users
# MSYS-style paths.
TRUST_JSON='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"codebuild.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
DLC_PULL_JSON='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["ecr:GetAuthorizationToken","ecr:BatchCheckLayerAvailability","ecr:GetDownloadUrlForLayer","ecr:BatchGetImage"],"Resource":"*"}]}'

ROLE_ARN=$(aws iam get-role --role-name "${ROLE_NAME}" --query 'Role.Arn' --output text 2>/dev/null || true)
if [ -z "${ROLE_ARN}" ]; then
    echo "[codebuild] creating IAM role ${ROLE_NAME} …"
    aws iam create-role \
        --role-name "${ROLE_NAME}" \
        --assume-role-policy-document "${TRUST_JSON}" >/dev/null
    aws iam attach-role-policy \
        --role-name "${ROLE_NAME}" \
        --policy-arn arn:aws:iam::aws:policy/AWSCodeBuildDeveloperAccess
    aws iam attach-role-policy \
        --role-name "${ROLE_NAME}" \
        --policy-arn arn:aws:iam::aws:policy/EC2InstanceProfileForImageBuilderECRContainerBuilds
    aws iam attach-role-policy \
        --role-name "${ROLE_NAME}" \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
    aws iam attach-role-policy \
        --role-name "${ROLE_NAME}" \
        --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
    aws iam put-role-policy \
        --role-name "${ROLE_NAME}" \
        --policy-name DLCPull \
        --policy-document "${DLC_PULL_JSON}"
    ROLE_ARN=$(aws iam get-role --role-name "${ROLE_NAME}" --query 'Role.Arn' --output text)
    echo "[codebuild] role created: ${ROLE_ARN} (waiting 10s for IAM eventual consistency …)"
    sleep 10
else
    echo "[codebuild] role already exists: ${ROLE_ARN}"
fi

# ─── 2. CodeBuild project ─────────────────────────────────────
if aws codebuild batch-get-projects --names "${PROJECT_NAME}" --region "${REGION}" --query 'projects[0].name' --output text 2>/dev/null | grep -q "${PROJECT_NAME}"; then
    echo "[codebuild] project ${PROJECT_NAME} already exists — updating source location"
    aws codebuild update-project \
        --name "${PROJECT_NAME}" \
        --region "${REGION}" \
        --source "type=S3,location=${S3_BUCKET}/${SRC_KEY},buildspec=containers/mamba/buildspec.yml" >/dev/null
else
    echo "[codebuild] creating project ${PROJECT_NAME} …"
    aws codebuild create-project \
        --region "${REGION}" \
        --name "${PROJECT_NAME}" \
        --source "type=S3,location=${S3_BUCKET}/${SRC_KEY},buildspec=containers/mamba/buildspec.yml" \
        --artifacts "type=NO_ARTIFACTS" \
        --environment "type=LINUX_CONTAINER,image=aws/codebuild/standard:7.0,computeType=BUILD_GENERAL1_MEDIUM,privilegedMode=true,environmentVariables=[{name=ACCOUNT_ID,value=${ACCOUNT_ID}},{name=IMAGE_REPO,value=${IMAGE_REPO}},{name=DLC_ACCOUNT,value=${DLC_ACCOUNT}}]" \
        --service-role "${ROLE_ARN}" \
        --timeout-in-minutes 60 >/dev/null
    echo "[codebuild] project created"
fi

# ─── 3. Package + upload source zip ───────────────────────────
ZIP_FILE="${TEMP:-/tmp}/mamba-image-source-${TS}.zip"
echo "[codebuild] packaging source → ${ZIP_FILE}"
rm -f "${ZIP_FILE}"
# Python's zipfile is portable across OSes; Git Bash on Windows
# doesn't ship the ``zip`` cli, so use python instead.
( cd "$(git rev-parse --show-toplevel)" && \
  python -c "
import zipfile, os
files = [
    'containers/mamba/Dockerfile',
    'containers/mamba/buildspec.yml',
]
with zipfile.ZipFile(r'${ZIP_FILE}', 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        zf.write(f, arcname=f)
print('zip written:', r'${ZIP_FILE}')
" )
ZIP_SIZE=$(du -h "${ZIP_FILE}" | cut -f1)
echo "[codebuild] zip size: ${ZIP_SIZE}"

echo "[codebuild] uploading to ${SRC_S3} …"
aws s3 cp "${ZIP_FILE}" "${SRC_S3}"

# Update project source so subsequent builds pick up the new zip
aws codebuild update-project \
    --name "${PROJECT_NAME}" \
    --region "${REGION}" \
    --source "type=S3,location=${S3_BUCKET}/${SRC_KEY},buildspec=containers/mamba/buildspec.yml" >/dev/null

# ─── 4. Start the build ───────────────────────────────────────
echo "[codebuild] starting build …"
BUILD_ID=$(aws codebuild start-build \
    --project-name "${PROJECT_NAME}" \
    --region "${REGION}" \
    --query 'build.id' --output text)
echo "[codebuild] build id: ${BUILD_ID}"
echo "[codebuild] CloudWatch console: https://${REGION}.console.aws.amazon.com/codesuite/codebuild/projects/${PROJECT_NAME}/build/${BUILD_ID//\//%2F}/log"

# ─── 5. Poll until done ───────────────────────────────────────
while true; do
    sleep 30
    STATUS=$(aws codebuild batch-get-builds --ids "${BUILD_ID}" --region "${REGION}" --query 'builds[0].buildStatus' --output text)
    PHASE=$(aws codebuild batch-get-builds --ids "${BUILD_ID}" --region "${REGION}" --query 'builds[0].currentPhase' --output text)
    echo "[codebuild] ${PHASE} / ${STATUS}"
    case "${STATUS}" in
        SUCCEEDED) break ;;
        FAILED|FAULT|TIMED_OUT|STOPPED)
            echo "[codebuild] build failed (${STATUS}) — see CloudWatch log"
            exit 1 ;;
    esac
done

# ─── 6. Resolve final image URI ──────────────────────────────
GIT_SHORT=$(git rev-parse --short HEAD)
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_REPO}:${GIT_SHORT}"
echo
echo "[codebuild] DONE"
echo "[codebuild] image URI: ${IMAGE_URI}"
echo
echo "[codebuild] Next: paste into pipeline.yaml::aws.mamba_image_uri:"
echo "    aws:"
echo "      mamba_image_uri: ${IMAGE_URI}"
