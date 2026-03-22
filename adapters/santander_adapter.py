"""
Santander Customer Transactions Adapter
========================================

Loads the santander_final.parquet dataset (941K users, 89 columns).
Already user-level — no aggregation needed.

Input:
    data/synthetic/santander_final.parquet  (or overridden by config)

Output (from load_raw):
    {"main": DataFrame}  -- 941,132 rows × 89 columns

The adapter is responsible ONLY for loading raw data. Feature engineering,
label derivation, sequence building, and encryption are handled by
downstream pipeline stages.
"""

from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

from core.pipeline.adapter import AdapterMetadata, AdapterRegistry, DataAdapter

logger = logging.getLogger(__name__)


@AdapterRegistry.register("santander")
class SantanderAdapter(DataAdapter):
    """DataAdapter for the Santander synthetic dataset.

    The dataset is already at user-level granularity with:
    - Scalar numeric columns (age, income, tenure, etc.)
    - String categorical columns (gender, segment, country, channel,
      age_group, income_group)
    - List-type sequence columns (txn_amount_seq, txn_count_seq,
      txn_mcc_seq, seq_saving, seq_current, seq_num_products,
      seq_acquisitions, seq_churns, etc.)
    - Binary product holding flags (prod_*)
    - Synthetic derived features (synth_*)
    """

    def load_raw(self) -> Dict[str, pd.DataFrame]:
        """Load santander_final.parquet. Already user-level, minimal preprocessing."""
        backend = self._select_backend()
        source = self.config.get("data", {}).get(
            "source", "data/synthetic/santander_final.parquet"
        )

        logger.info("SantanderAdapter: loading %s with backend=%s", source, backend)

        if backend == "cudf":
            import cudf

            df = cudf.read_parquet(source)
        elif backend == "duckdb":
            import duckdb

            con = duckdb.connect()
            df = con.execute(f"SELECT * FROM '{source}'").df()
            con.close()
        else:
            df = pd.read_parquet(source)

        self._metadata = AdapterMetadata(
            id_col="customer_id",
            entity_granularity="user",
            num_entities=len(df),
            num_raw_rows=len(df),
            source_files=[str(source)],
            backend_used=backend,
        )

        logger.info(
            "SantanderAdapter: loaded %d rows × %d cols (backend=%s)",
            len(df),
            len(df.columns),
            backend,
        )

        return {"main": df}


# ======================================================================
# Standalone entry point for SageMaker Processing Job
# ======================================================================

if __name__ == "__main__":
    import argparse
    import os
    import sys
    import shutil

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Santander data adapter")
    parser.add_argument("--input-dir", default="/opt/ml/processing/input/raw")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--pipeline", default="", help="Pipeline config path (unused, for compat)")
    parser.add_argument("--stages", default="1-6", help="Stages to run (unused, for compat)")
    cli_args = parser.parse_args()

    input_dir = cli_args.input_dir
    output_dir = cli_args.output_dir

    # Find the parquet file in input dir
    import glob
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    if not parquet_files:
        # Try subdirectories
        parquet_files = glob.glob(os.path.join(input_dir, "**", "*.parquet"), recursive=True)

    if not parquet_files:
        logger.error("No parquet files found in %s", input_dir)
        sys.exit(1)

    source = parquet_files[0]
    logger.info("Found source: %s", source)

    # Load via adapter
    config = {"data": {"source": source, "backend": ["duckdb", "pandas"]}}
    adapter = SantanderAdapter(config)
    raw_data = adapter.load_raw()
    df = raw_data["main"]

    # Save to output dir
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "santander_final.parquet")
    df.to_parquet(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)

    # Save metadata
    import json
    meta = adapter.metadata
    with open(os.path.join(output_dir, "adapter_metadata.json"), "w") as f:
        json.dump(meta.__dict__, f, indent=2, default=str)

    logger.info("Phase 0 complete. Output: %s", output_dir)
