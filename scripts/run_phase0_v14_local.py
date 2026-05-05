"""Local Phase 0 v14 runner -- HMM/GMM/normalizer fixes verification.

Re-runs the 9-stage Phase 0 pipeline on benchmark_v12.parquet so the
following fixes propagate into ``outputs/phase0_v14/extracted/``:

* HMM mode_observation_cols (journey/lifecycle/behavior get distinct
  observation columns + per-mode random_state)
* GMM K=14 (was 20; eliminates 6 dead clusters)
* FeatureNormalizer probability_infix detection (gmm_cluster_prob_*,
  hmm_*_prob_* stay in [0,1] instead of being z-scored)

Output: ``outputs/phase0_v14/extracted/`` mirroring the v13 layout.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("phase0_v14")


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import adapters  # noqa: F401  registers santander adapter
    from core.pipeline.config import load_config
    from core.pipeline.runner import PipelineRunner

    config = load_config(
        "configs/pipeline.yaml",
        dataset_path="configs/datasets/santander.yaml",
    )

    output_dir = "outputs/phase0_v14/extracted"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Phase 0 v14 starting -> %s", output_dir)
    logger.info("Source: %s", config.data.source)

    start = time.time()
    runner = PipelineRunner(config)
    runner.run(output_dir=output_dir)
    elapsed = time.time() - start
    logger.info("Phase 0 v14 done in %.1f min", elapsed / 60.0)


if __name__ == "__main__":
    main()
