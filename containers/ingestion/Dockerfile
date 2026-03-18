FROM python:3.11-slim

WORKDIR /opt/ml/code

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY containers/ingestion/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Project code
COPY core/ core/
COPY configs/ configs/
COPY containers/ingestion/ containers/ingestion/
COPY scripts/run_ingestion.py scripts/

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "containers/ingestion/entrypoint.py"]
