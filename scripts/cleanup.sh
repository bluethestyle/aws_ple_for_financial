#!/usr/bin/env bash
# cleanup.sh — 실험 후 AWS 자원 정리
#
# Usage:
#   ./scripts/cleanup.sh --endpoint ple-endpoint-20240101-120000
#   ./scripts/cleanup.sh --all-endpoints   # 모든 ple- 접두사 엔드포인트 삭제

set -euo pipefail

ENDPOINT=""
ALL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --all-endpoints) ALL=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if $ALL; then
  echo "Deleting all ple-* endpoints..."
  aws sagemaker list-endpoints --query "Endpoints[?starts_with(EndpointName, 'ple-')].EndpointName" \
    --output text | tr '\t' '\n' | while read -r name; do
    echo "  Deleting: $name"
    aws sagemaker delete-endpoint --endpoint-name "$name"
  done
  echo "Done."
elif [[ -n "$ENDPOINT" ]]; then
  echo "Deleting endpoint: $ENDPOINT"
  aws sagemaker delete-endpoint --endpoint-name "$ENDPOINT"
  echo "Done."
else
  echo "Usage:"
  echo "  $0 --endpoint <endpoint-name>"
  echo "  $0 --all-endpoints"
fi
