# =============================================================================
# SageMaker PLE Training Container
#
# Based on the official PyTorch 2.1 SageMaker image with additional
# dependencies for the PLE multi-task learning platform.
#
# Build:
#   docker build -t ple-training:latest containers/training/
#
# Push to ECR:
#   aws ecr get-login-password --region ap-northeast-2 | \
#     docker login --username AWS --password-stdin <account>.dkr.ecr.ap-northeast-2.amazonaws.com
#   docker tag ple-training:latest <account>.dkr.ecr.ap-northeast-2.amazonaws.com/ple-training:latest
#   docker push <account>.dkr.ecr.ap-northeast-2.amazonaws.com/ple-training:latest
# =============================================================================

FROM 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Install the project package
# ---------------------------------------------------------------------------
WORKDIR /opt/ml/code

# Copy full project source so that `core.*` and `aws.*` imports resolve.
# SageMaker also copies source_dir into /opt/ml/code at runtime, but
# pre-installing ensures the package is importable during local testing.
COPY ../../core/       /opt/ml/code/core/
COPY ../../aws/        /opt/ml/code/aws/
COPY ../../pyproject.toml /opt/ml/code/pyproject.toml

# Install project in editable mode so all imports work
RUN pip install --no-cache-dir -e /opt/ml/code/ 2>/dev/null || true

# Copy the training entry point last (fastest cache bust)
COPY train.py /opt/ml/code/train.py

# ---------------------------------------------------------------------------
# SageMaker expects SAGEMAKER_PROGRAM to point to the entry script
# ---------------------------------------------------------------------------
ENV SAGEMAKER_PROGRAM=train.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure core/ is importable
ENV PYTHONPATH="/opt/ml/code:${PYTHONPATH}"

WORKDIR /opt/ml/code
