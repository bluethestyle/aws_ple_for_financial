# TDA (Topological Data Analysis) feature generator container.
#
# Inherits from the base feature generator image and adds TDA-specific
# dependencies: ripser, persim, and giotto-tda for persistent homology
# computation on point-cloud data.
#
# Build:
#   docker build -t feature-gen-tda:latest -f containers/generators/tda/Dockerfile .
#
# Requires the base image to be built first:
#   docker build -t feature-gen-base:latest -f containers/generators/base/Dockerfile .

FROM feature-gen-base:latest

# Install TDA-specific dependencies
COPY containers/generators/tda/requirements.txt /tmp/tda-requirements.txt
RUN pip install --no-cache-dir -r /tmp/tda-requirements.txt \
    && rm /tmp/tda-requirements.txt

# The entrypoint is inherited from the base image
