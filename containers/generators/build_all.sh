#!/usr/bin/env bash
# Build and push all feature generator container images to AWS ECR.
#
# Usage:
#   ./containers/generators/build_all.sh [AWS_ACCOUNT_ID] [AWS_REGION]
#
# Arguments:
#   AWS_ACCOUNT_ID  -- 12-digit AWS account ID (default: from AWS STS)
#   AWS_REGION      -- AWS region for ECR (default: ap-northeast-2)
#
# Prerequisites:
#   - Docker installed and running
#   - AWS CLI configured with ECR push permissions
#   - Run from the repository root directory
#
# The script builds images in dependency order:
#   1. feature-gen-base   (shared base)
#   2. feature-gen-tda    (inherits base)
#   3. feature-gen-graph  (inherits base)
#   4. feature-gen-nlp    (inherits base)

set -euo pipefail

# -- Configuration ---------------------------------------------------------

AWS_ACCOUNT_ID="${1:-$(aws sts get-caller-identity --query Account --output text)}"
AWS_REGION="${2:-ap-northeast-2}"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

IMAGES=(
    "feature-gen-base:containers/generators/base/Dockerfile"
    "feature-gen-tda:containers/generators/tda/Dockerfile"
    "feature-gen-graph:containers/generators/graph/Dockerfile"
    "feature-gen-nlp:containers/generators/nlp/Dockerfile"
)

TAG="latest"

# -- Functions -------------------------------------------------------------

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

ensure_ecr_repo() {
    local repo_name="$1"
    if ! aws ecr describe-repositories \
        --repository-names "${repo_name}" \
        --region "${AWS_REGION}" &>/dev/null; then
        log "Creating ECR repository: ${repo_name}"
        aws ecr create-repository \
            --repository-name "${repo_name}" \
            --region "${AWS_REGION}" \
            --image-scanning-configuration scanOnPush=true \
            --encryption-configuration encryptionType=AES256
    fi
}

# -- Main ------------------------------------------------------------------

log "AWS Account:  ${AWS_ACCOUNT_ID}"
log "AWS Region:   ${AWS_REGION}"
log "ECR Registry: ${ECR_REGISTRY}"
log ""

# Authenticate Docker with ECR
log "Authenticating Docker with ECR..."
aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# Build and push each image
for entry in "${IMAGES[@]}"; do
    IFS=':' read -r image_name dockerfile <<< "${entry}"
    ecr_uri="${ECR_REGISTRY}/${image_name}:${TAG}"

    log "=========================================="
    log "Building: ${image_name}"
    log "  Dockerfile: ${dockerfile}"
    log "  ECR URI:    ${ecr_uri}"
    log "=========================================="

    # Ensure the ECR repository exists
    ensure_ecr_repo "${image_name}"

    # Build (context is always repo root for COPY core/ to work)
    docker build \
        --tag "${image_name}:${TAG}" \
        --tag "${ecr_uri}" \
        --file "${dockerfile}" \
        .

    # Push to ECR
    log "Pushing ${ecr_uri}..."
    docker push "${ecr_uri}"

    log "Done: ${image_name}"
    log ""
done

log "All images built and pushed successfully."
log ""
log "Image URIs:"
for entry in "${IMAGES[@]}"; do
    IFS=':' read -r image_name _ <<< "${entry}"
    echo "  ${ECR_REGISTRY}/${image_name}:${TAG}"
done
