# =============================================================================
# PLE Inference Container
#
# Serves multi-task LGBM recommendations via FastAPI (ECS) or can be used
# as the base image for an AWS Lambda container.
#
# Build:
#   docker build -t ple-inference:latest containers/inference/
#
# Run locally (ECS mode):
#   docker run -p 8080:8080 \
#     -e MODEL_DIR=/opt/ml/model \
#     -e SERVING_CONFIG='{"serving":{"feature_store":"memory",...}}' \
#     ple-inference:latest
#
# Push to ECR:
#   aws ecr get-login-password --region ap-northeast-2 | \
#     docker login --username AWS --password-stdin <account>.dkr.ecr.ap-northeast-2.amazonaws.com
#   docker tag ple-inference:latest <account>.dkr.ecr.ap-northeast-2.amazonaws.com/ple-inference:latest
#   docker push <account>.dkr.ecr.ap-northeast-2.amazonaws.com/ple-inference:latest
# =============================================================================

FROM python:3.10-slim

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
WORKDIR /opt/ml/code

# Copy the core library so all imports resolve
COPY ../../core/       /opt/ml/code/core/
COPY ../../aws/        /opt/ml/code/aws/
COPY ../../pyproject.toml /opt/ml/code/pyproject.toml

# Install the project package
RUN pip install --no-cache-dir -e /opt/ml/code/ 2>/dev/null || true

# Copy inference entry points
COPY lambda_handler.py /opt/ml/code/containers/inference/lambda_handler.py
COPY app.py            /opt/ml/code/containers/inference/app.py

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV PYTHONPATH="/opt/ml/code:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=INFO

# Default model directory (mounted or downloaded at runtime)
ENV MODEL_DIR=/opt/ml/model

# ECS health check port
EXPOSE 8080

# ---------------------------------------------------------------------------
# Default command: start FastAPI (ECS mode)
# For Lambda, override CMD with the Lambda runtime interface client.
# ---------------------------------------------------------------------------
CMD ["uvicorn", "containers.inference.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
