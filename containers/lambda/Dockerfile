FROM public.ecr.aws/lambda/python:3.10

# System deps for lightgbm (libgomp for OpenMP)
RUN yum install -y libgomp && yum clean all

# Copy pre-downloaded packages (built locally to avoid Docker network issues)
COPY containers/lambda/packages/ /tmp/packages/
RUN pip install --no-cache-dir --no-index --find-links /tmp/packages/ \
    lightgbm lancedb duckdb pyyaml && \
    rm -rf /tmp/packages/

# Copy application code (build context = project root)
COPY containers/lambda/predict.py ${LAMBDA_TASK_ROOT}/
COPY containers/lambda/l2a_consumer.py ${LAMBDA_TASK_ROOT}/
COPY containers/path_resolver.py ${LAMBDA_TASK_ROOT}/

# Copy core modules
COPY core/ ${LAMBDA_TASK_ROOT}/core/

# Copy configs
COPY configs/merged_config.json ${LAMBDA_TASK_ROOT}/configs/
COPY configs/feature_groups.yaml ${LAMBDA_TASK_ROOT}/configs/
COPY configs/financial/fact_extraction.yaml ${LAMBDA_TASK_ROOT}/configs/financial/

CMD ["predict.handler"]
