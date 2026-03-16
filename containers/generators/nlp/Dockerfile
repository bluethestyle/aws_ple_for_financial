# NLP feature generator container.
#
# Inherits from the base feature generator image and adds NLP
# dependencies: PyTorch, HuggingFace Transformers, and
# sentence-transformers for text embedding feature generation.
#
# Build:
#   docker build -t feature-gen-nlp:latest -f containers/generators/nlp/Dockerfile .
#
# Requires the base image to be built first:
#   docker build -t feature-gen-base:latest -f containers/generators/base/Dockerfile .

FROM feature-gen-base:latest

# Install NLP-specific dependencies
COPY containers/generators/nlp/requirements.txt /tmp/nlp-requirements.txt
RUN pip install --no-cache-dir -r /tmp/nlp-requirements.txt \
    && rm /tmp/nlp-requirements.txt

# The entrypoint is inherited from the base image
