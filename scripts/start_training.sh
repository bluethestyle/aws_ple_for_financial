#!/usr/bin/env bash
# start_training.sh — SageMaker Training Job 시작
#
# Usage:
#   ./scripts/start_training.sh --config configs/examples/multitask_binary.yaml
#   ./scripts/start_training.sh --config configs/examples/multitask_binary.yaml --local

set -euo pipefail

CONFIG=""
MODE="sagemaker"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --local)  MODE="local"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Error: --config is required"
  echo "Usage: $0 --config configs/examples/multitask_binary.yaml [--local]"
  exit 1
fi

echo "=================================="
echo " AWS PLE Platform — Training"
echo "=================================="
echo " Config : $CONFIG"
echo " Mode   : $MODE"
echo "----------------------------------"

python - <<EOF
from core.pipeline.config import load_config
from core.pipeline.runner import PipelineRunner

config = load_config("$CONFIG")
runner = PipelineRunner(config)
result = runner.run(mode="$MODE")
print("Result:", result)
EOF
