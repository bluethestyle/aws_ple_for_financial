# Custom SageMaker training image for the Mamba GPU pre-compute job.
#
# Bakes mamba_ssm + causal-conv1d (the CUDA selective-scan kernels)
# into the image so the entry script doesn't have to compile them
# at job start. Without this image the wheel build fails on the
# stock AWS PyTorch GPU DLC because:
#   * ninja is not preinstalled (distutils fallback is slow + fragile)
#   * causal-conv1d's setup.py downloads NVIDIA headers via urllib
#     which intermittently 403s on SageMaker hosts
#   * the from-source build takes ~7 minutes per job, which we don't
#     need since these wheels are stable
#
# Base: SageMaker PyTorch 2.1 GPU training DLC for ap-northeast-2.
# Built once with scripts/build_mamba_image.sh and pushed to ECR;
# launch_mamba_precompute() in aws/sagemaker/trainer.py reads
# ``aws.mamba_image_uri`` from pipeline.yaml and passes it as
# ``image_uri`` to the PyTorch estimator.

ARG REGION=ap-northeast-2
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System build prereqs the wheels need:
#   * ninja-build  — accelerates torch.utils.cpp_extension
#   * build-essential — gcc / g++ for the CUDA host compiler
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ninja-build \
        build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install ninja into the same conda environment SageMaker uses.
# The DLC's site-packages live under /opt/conda; pip-installed
# packages here are what the entry script will see at runtime.
RUN /opt/conda/bin/pip install --no-cache-dir ninja

# CUDA selective-scan wheels (causal-conv1d first — mamba-ssm
# imports it at module load time).
#
# CAUSAL_CONV1D_FORCE_BUILD=TRUE makes the build use the local
# CUDA toolkit instead of trying to download NVIDIA headers from
# the network (which has been flaky inside SageMaker).
ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    MAMBA_FORCE_BUILD=TRUE \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"

RUN /opt/conda/bin/pip install --no-cache-dir \
        "causal-conv1d>=1.1,<2.0" \
        "mamba-ssm>=1.2,<3.0"

# Sanity-check the install at build time so a regression here
# fails the image build instead of a downstream training job.
RUN /opt/conda/bin/python -c "import mamba_ssm, causal_conv1d; print('mamba_ssm:', mamba_ssm.__version__, 'causal_conv1d:', causal_conv1d.__version__)"

# DuckDB / pyarrow are read by the entry script for parquet I/O.
RUN /opt/conda/bin/pip install --no-cache-dir \
        "duckdb>=0.10,<2.0" \
        "pyarrow>=14,<19"
