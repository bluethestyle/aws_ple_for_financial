# Custom SageMaker training image for the Mamba GPU pre-compute job.
#
# Bakes mamba_ssm + causal-conv1d (the CUDA selective-scan kernels)
# into the image so the entry script doesn't have to compile them
# at job start.
#
# Base: SageMaker PyTorch 2.1 GPU training DLC for ap-northeast-2.
# Built once with scripts/build_mamba_image_codebuild.sh and pushed
# to ECR; launch_mamba_precompute() in aws/sagemaker/trainer.py
# reads ``aws.mamba_image_uri`` from pipeline.yaml and passes it as
# ``image_uri`` to the PyTorch estimator.
#
# Strategy: pre-built wheels from each project's GitHub Releases.
# The from-source path was tried in builds #5–#11 and hit:
#   * BuildKit "tls: bad record MAC" (local Docker on Windows)
#   * pip build-isolation pulling latest torch (cu130) → CUDA
#     mismatch with DLC nvcc (cu121)
#   * causal-conv1d ≥1.4.0 importing torch.library.custom_op
#     (PyTorch 2.4+ only; DLC has 2.1)
#   * sdist of causal-conv1d 1.2.x / 1.3.x missing csrc/*.cpp
# Pre-built wheels skip the entire compile path — they were built
# by the project authors against the exact PyTorch + CUDA combo
# we run, so the ABI matches end-to-end.

ARG REGION=ap-northeast-2
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Pre-built wheels — pinned to causal-conv1d 1.2.0.post2 and
# mamba-ssm 1.2.0.post1, both pre-built against torch 2.1 / cp310.
#
# CUDA: neither project publishes a cu121 wheel for the 1.2.x line
# (only cu118 + cu122 — verified via GitHub Releases API
# 2026-04-27). cu118 wheels link against libcudart.so.11.0 which
# the DLC doesn't ship, so build#13 failed at import:
#   ImportError: libcudart.so.11.0: cannot open shared object file
# cu122 wheels link against libcudart.so.12, which matches the
# DLC's cu121 libcudart (CUDA 12.x is binary-compatible at the
# library level within the same major).
#
# These versions still use torch.utils.cpp_extension (no
# torch.library.custom_op dependency) so they import cleanly under
# the DLC's PyTorch 2.1.
ARG CAUSAL_CONV1D_VERSION=1.2.0.post2
ARG MAMBA_SSM_VERSION=1.2.0.post1

RUN /opt/conda/bin/pip install --no-cache-dir \
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CAUSAL_CONV1D_VERSION}/causal_conv1d-${CAUSAL_CONV1D_VERSION}+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
        "https://github.com/state-spaces/mamba/releases/download/v${MAMBA_SSM_VERSION}/mamba_ssm-${MAMBA_SSM_VERSION}+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# Sanity-check the install at build time so a regression here
# fails the image build instead of a downstream training job.
RUN /opt/conda/bin/python -c "import mamba_ssm, causal_conv1d; print('mamba_ssm:', mamba_ssm.__version__, 'causal_conv1d:', causal_conv1d.__version__)"

# DuckDB / pyarrow are read by the entry script for parquet I/O.
RUN /opt/conda/bin/pip install --no-cache-dir \
        "duckdb>=0.10,<2.0" \
        "pyarrow>=14,<19"
