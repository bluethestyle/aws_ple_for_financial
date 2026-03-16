# Graph embedding feature generator container.
#
# Inherits from the base feature generator image and adds graph/hyperbolic
# embedding dependencies: PyTorch, PyTorch Geometric, geoopt for
# Riemannian optimisation on the Poincare ball, and networkx for graph
# construction.
#
# Build:
#   docker build -t feature-gen-graph:latest -f containers/generators/graph/Dockerfile .
#
# Requires the base image to be built first:
#   docker build -t feature-gen-base:latest -f containers/generators/base/Dockerfile .

FROM feature-gen-base:latest

# Install graph-specific dependencies
COPY containers/generators/graph/requirements.txt /tmp/graph-requirements.txt
RUN pip install --no-cache-dir -r /tmp/graph-requirements.txt \
    && rm /tmp/graph-requirements.txt

# The entrypoint is inherited from the base image
