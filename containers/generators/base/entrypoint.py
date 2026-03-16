"""
Container entry point for SageMaker Processing Jobs.

This script is the generic entry point for all feature generator
containers.  It reads environment variables set by the
:class:`FeatureGroupPipeline` to determine which generator to run,
loads input data from the SageMaker processing input path, fits and
generates features, and writes the output to the processing output path.

Environment variables
---------------------
FEATURE_GROUP_NAME : str
    Name of the feature group being processed.
FEATURE_GROUP_TYPE : str
    ``"generate"`` or ``"transform"``.
FEATURE_GENERATOR : str
    Registry name of the feature generator (e.g. ``"tda_extractor"``).
FEATURE_GENERATOR_PARAMS : str (JSON)
    JSON-encoded dictionary of generator constructor parameters.

SageMaker paths
---------------
Input:  /opt/ml/processing/input/data.parquet
Output: /opt/ml/processing/output/features.parquet
"""

import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("container.entrypoint")


def main() -> None:
    """Entry point for the container."""
    start = time.time()

    # -- Read configuration from environment --
    group_name = os.environ.get("FEATURE_GROUP_NAME", "unknown")
    group_type = os.environ.get("FEATURE_GROUP_TYPE", "generate")
    generator_name = os.environ.get("FEATURE_GENERATOR", "")
    generator_params_json = os.environ.get("FEATURE_GENERATOR_PARAMS", "{}")

    logger.info(
        "Starting feature generation container: "
        "group=%s, type=%s, generator=%s",
        group_name, group_type, generator_name,
    )

    try:
        generator_params = json.loads(generator_params_json)
    except json.JSONDecodeError:
        logger.error("Failed to parse FEATURE_GENERATOR_PARAMS")
        generator_params = {}

    # -- Paths --
    input_dir = "/opt/ml/processing/input"
    output_dir = "/opt/ml/processing/output"
    input_path = os.path.join(input_dir, "data.parquet")
    output_path = os.path.join(output_dir, "features.parquet")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # -- Import framework (after PYTHONPATH is set) --
    from core.data.dataframe import df_backend
    from core.feature.generator import FeatureGeneratorRegistry

    # Trigger generator registration by importing the generators package
    import core.feature.generators  # noqa: F401

    # -- Load input data --
    logger.info("Reading input from %s", input_path)
    df = df_backend.read_parquet(input_path)
    logger.info("Input shape: %d rows x %d cols", len(df), len(df.columns))

    # -- Instantiate and run generator --
    if group_type == "generate" and generator_name:
        logger.info(
            "Building generator '%s' with params: %s",
            generator_name, generator_params,
        )
        gen = FeatureGeneratorRegistry.build(generator_name, **generator_params)
        gen.fit(df)
        result = gen.generate(df)
    else:
        logger.error(
            "Unsupported group_type '%s' or missing generator name",
            group_type,
        )
        sys.exit(1)

    # -- Write output --
    logger.info(
        "Writing output (%d rows x %d cols) to %s",
        len(result), len(result.columns), output_path,
    )
    df_backend.to_parquet(result, output_path)

    elapsed = time.time() - start
    logger.info(
        "Feature generation complete in %.2fs: "
        "group=%s, generator=%s, output_dim=%d",
        elapsed, group_name, generator_name, len(result.columns),
    )


if __name__ == "__main__":
    main()
