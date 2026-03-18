#!/bin/bash
# =============================================================================
# Build Lambda Layer: lightgbm + numpy (predict Lambda only)
#
# Output: containers/lambda/layers/lgbm/python/
# Usage:  bash scripts/build_lambda_layer.sh
#
# The layer is referenced by aws/lambda/serving_stack.yaml:
#   LGBMLayer.ContentUri: ../../containers/lambda/layers/lgbm/
# =============================================================================

set -e

LAYER_DIR="containers/lambda/layers/lgbm/python"

echo "Building Lambda Layer: lightgbm + numpy"

# Clean
rm -rf containers/lambda/layers/lgbm
mkdir -p "$LAYER_DIR"

# Install into layer structure
# Use manylinux wheels for Lambda compatibility (Amazon Linux 2)
pip install \
    --platform manylinux2014_x86_64 \
    --implementation cp \
    --python-version 3.11 \
    --only-binary=:all: \
    --target "$LAYER_DIR" \
    lightgbm numpy

# Remove unnecessary files to reduce size
find "$LAYER_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$LAYER_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$LAYER_DIR" -name "*.pyi" -delete 2>/dev/null || true
find "$LAYER_DIR" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
rm -rf "$LAYER_DIR/numpy/tests" 2>/dev/null || true
rm -rf "$LAYER_DIR/numpy/doc" 2>/dev/null || true

# Check size
LAYER_SIZE=$(du -sm containers/lambda/layers/lgbm | cut -f1)
echo "Layer size: ${LAYER_SIZE}MB (limit: 250MB)"

if [ "$LAYER_SIZE" -gt 250 ]; then
    echo "ERROR: Layer exceeds 250MB limit"
    exit 1
fi

echo "Lambda Layer built at: containers/lambda/layers/lgbm/"
echo "Deploy with: sam build && sam deploy"
