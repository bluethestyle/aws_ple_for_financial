#!/usr/bin/env bash
# setup_aws.sh — AWS 환경 초기 설정 (S3 버킷, IAM 역할)
#
# Usage:
#   ./scripts/setup_aws.sh --bucket my-ml-bucket --account 123456789012

set -euo pipefail

BUCKET=""
ACCOUNT=""
REGION="ap-northeast-2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket)  BUCKET="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --region)  REGION="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$BUCKET" || -z "$ACCOUNT" ]]; then
  echo "Usage: $0 --bucket <bucket-name> --account <aws-account-id>"
  exit 1
fi

echo "Setting up AWS resources..."
echo "  Bucket : $BUCKET"
echo "  Account: $ACCOUNT"
echo "  Region : $REGION"

# S3 버킷 생성
echo ""
echo "[1/3] Creating S3 bucket..."
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "  Bucket already exists: $BUCKET"
else
  aws s3api create-bucket \
    --bucket "$BUCKET" \
    --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION"
  # 버전 관리 활성화 (모델 체크포인트 보호)
  aws s3api put-bucket-versioning \
    --bucket "$BUCKET" \
    --versioning-configuration Status=Enabled
  echo "  Created: s3://$BUCKET"
fi

# 디렉토리 구조 초기화
echo ""
echo "[2/3] Initializing bucket structure..."
for prefix in data/raw data/processed features models experiments; do
  aws s3api put-object --bucket "$BUCKET" --key "$prefix/.keep" --body /dev/null
  echo "  s3://$BUCKET/$prefix/"
done

# SageMaker IAM 역할 생성
echo ""
echo "[3/3] Creating SageMaker IAM role..."
ROLE_NAME="AWSPLEPlatformSageMakerRole"
TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "sagemaker.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'

if aws iam get-role --role-name "$ROLE_NAME" 2>/dev/null; then
  echo "  Role already exists: $ROLE_NAME"
else
  aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "$TRUST_POLICY"
  aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
  aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn "arn:aws:iam::aws:policy/AmazonS3FullAccess"
  echo "  Created: $ROLE_NAME"
fi

ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/${ROLE_NAME}"

echo ""
echo "=================================="
echo " Setup complete!"
echo "=================================="
echo ""
echo "Add these to your config YAML:"
echo "  aws:"
echo "    s3_bucket: $BUCKET"
echo "    role_arn: $ROLE_ARN"
echo "    region: $REGION"
