# Base image for all feature generator containers.
#
# Provides Python 3.10, core data libraries (numpy, pandas, duckdb,
# pyarrow), and the feature engineering framework source code.
#
# Generator-specific images (TDA, graph, NLP) inherit from this base
# and add their own specialized dependencies.
#
# Build:
#   docker build -t feature-gen-base:latest -f containers/generators/base/Dockerfile .
#
# The build context MUST be the repository root so that COPY core/ works.

FROM python:3.10-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /opt/ml/code

# Install Python dependencies (cached layer)
COPY containers/generators/base/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy the core framework source code
COPY core/ /opt/ml/code/core/

# Copy the container entry point
COPY containers/generators/base/entrypoint.py /opt/ml/code/entrypoint.py

# SageMaker Processing Job convention:
#   Input:  /opt/ml/processing/input/
#   Output: /opt/ml/processing/output/
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/ml/code

ENTRYPOINT ["python", "/opt/ml/code/entrypoint.py"]
