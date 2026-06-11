#!/usr/bin/env bash
# setup_aws.sh — AWS 환경 초기 설정 (S3 버킷, IAM 역할)
#
# Usage:
#   ./scripts/setup_aws.sh --bucket my-ml-bucket --account 123456789012

set -euo pipefail

BUCKET=""
ACCOUNT=""
REGION="ap-northeast-2"
HMAC_SSM_PARAM="/ple/audit/hmac-secret-key"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket)  BUCKET="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --region)  REGION="$2"; shift 2 ;;
    --hmac-ssm-param) HMAC_SSM_PARAM="$2"; shift 2 ;;
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
echo "[1/4] Creating S3 bucket..."
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
echo "[2/4] Initializing bucket structure..."
for prefix in data/raw data/processed features models experiments; do
  aws s3api put-object --bucket "$BUCKET" --key "$prefix/.keep" --body /dev/null
  echo "  s3://$BUCKET/$prefix/"
done

# SageMaker IAM 역할 생성
echo ""
echo "[3/4] Creating SageMaker IAM role..."
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

# 감사 로그 HMAC 서명 키 (SSM SecureString) 프로비저닝
# audit_logger._get_hmac_secret 는 production 에서 이 파라미터(또는
# AUDIT_HMAC_SECRET_KEY env)가 없으면 서명을 거부(fail-closed)한다.
echo ""
echo "[4/4] Provisioning audit HMAC key in SSM Parameter Store..."
if aws ssm get-parameter --name "$HMAC_SSM_PARAM" --region "$REGION" >/dev/null 2>&1; then
  echo "  SSM parameter already exists: $HMAC_SSM_PARAM (left unchanged)"
else
  # 32-byte 랜덤 키 생성 (값은 출력하지 않음)
  HMAC_VALUE="$(openssl rand -base64 32)"
  aws ssm put-parameter \
    --name "$HMAC_SSM_PARAM" \
    --type SecureString \
    --value "$HMAC_VALUE" \
    --description "PLE audit-log HMAC signing key" \
    --region "$REGION" >/dev/null
  unset HMAC_VALUE
  echo "  Created SecureString: $HMAC_SSM_PARAM"
fi

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
echo ""
echo "Set these env vars on the serving Lambda / SageMaker containers:"
echo "  AUDIT_HMAC_SSM_PARAM=$HMAC_SSM_PARAM   # audit signing key (required in prod)"
echo "  ENVIRONMENT=production                  # enables fail-closed posture"
echo "  AUDIT_S3_BUCKET=$BUCKET                 # WORM audit-log destination"
echo ""
echo "Grant the execution role ssm:GetParameter on:"
echo "  arn:aws:ssm:${REGION}:${ACCOUNT}:parameter${HMAC_SSM_PARAM}"
