# -*- coding: utf-8 -*-
"""Patch ``santander_final.parquet`` with a synthetic ``txn_day_offset_seq``.

Role in the data layer
----------------------
This script is the *fallback* day-offset producer in the Santander data
layer. The canonical producer is
``scripts/augment_santander_with_real_txns.py``, which generates real
inter-txn day gaps by linking each Santander customer to a peer in the
ealtman2019 pool. The augment input
(``data/santander_linked.parquet`` + ``data/새 폴더/01_financial_*.parquet``)
is not always available, so when the operator has only the bare
extracted parquet (``santander_final.parquet`` — produced by
``scripts/santander_temporal_extraction.py`` from ``train_ver2.csv``)
this patch runs as a one-shot synthesis step.

What it produces
----------------
For every row, ``len(txn_day_offset_seq) == len(txn_amount_seq)``.
Values are evenly-spaced integers in ``[0, day_window]`` (default 360).
The list is **right-aligned**: position 0 (oldest txn) gets
``day_window`` and position ``L-1`` (most recent txn) gets ``0`` — this
matches the convention the lag/rolling SQL generators assume after
``list_reverse``.

What it does NOT produce
------------------------
* Real txn calendar dates — these are placeholders. Lag positions
  derived from this column carry *structural* signal only.
* A new feature group / schema entry. The column name and dtype
  match what the lag/rolling generators already expect (see
  ``configs/santander/feature_groups.yaml``).

When to invoke
--------------
* New extraction (``santander_temporal_extraction.py``) finished and
  the augment pool is unavailable. Run once, upload the resulting
  ``santander_final_v2.parquet`` to S3, and bump
  ``configs/santander/pipeline.yaml > data.source`` to point at it.
* The patch is idempotent — if the input already carries
  ``txn_day_offset_seq`` it is overwritten with the synthetic series
  (a warning is logged).

Usage
-----
    python scripts/patch_santander_add_day_offset.py \
        --input  data/santander_final.parquet \
        --output data/santander_final_v2.parquet \
        --day-window 360
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import duckdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("patch_day_offset")


def patch(input_path: Path, output_path: Path, day_window: int = 360) -> None:
    """Add ``txn_day_offset_seq`` and write a new parquet.

    The synthesis rule:

      * For each row, ``len(txn_day_offset_seq) == len(txn_amount_seq)``.
      * Values are evenly spaced integers in ``[0, day_window]``: the
        last (most-recent) txn lands at offset 0 and the oldest at
        ``day_window``.  Right-aligned ordering matches the
        ``list_reverse`` convention the lag generator expects.
      * Empty-list rows produce an empty list — the lag generator
        handles that case via ``list_resize(..., k, pad_value)``.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        # Stash schema first — we want to preserve every existing
        # column verbatim and only append txn_day_offset_seq.
        cols = con.execute(
            f"DESCRIBE SELECT * FROM '{input_path}' LIMIT 1"
        ).fetchall()
        col_names = [c[0] for c in cols]
        if "txn_amount_seq" not in col_names:
            raise RuntimeError(
                "Input parquet does not have txn_amount_seq; cannot derive offsets"
            )
        if "txn_day_offset_seq" in col_names:
            logger.warning(
                "Input parquet already has txn_day_offset_seq — overwriting "
                "with the synthetic series anyway"
            )
            col_names = [c for c in col_names if c != "txn_day_offset_seq"]

        passthrough = ", ".join(f'"{c}"' for c in col_names)

        # Synthesis SQL:
        #   * range(len(txn_amount_seq)) gives [0, 1, ..., L-1]
        #   * map each i to ((L - 1 - i) * day_window / max(L - 1, 1))
        #     so position 0 (oldest) = day_window, position L-1
        #     (most recent) = 0
        #   * cast to INTEGER for compactness
        synth_sql = (
            "list_transform("
            "  range(len(txn_amount_seq)), "
            "  i -> CAST(((len(txn_amount_seq) - 1 - i) "
            f"* {day_window}) "
            "       / GREATEST(len(txn_amount_seq) - 1, 1) AS INTEGER)"
            ") AS txn_day_offset_seq"
        )

        sql = (
            f"COPY (\n"
            f"    SELECT {passthrough}, {synth_sql}\n"
            f"    FROM read_parquet('{input_path}')\n"
            f") TO '{output_path}' (FORMAT PARQUET)"
        )

        logger.info("Writing patched parquet -> %s", output_path)
        con.execute(sql)

        # Quick sanity: row count + average list length.
        rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{output_path}')"
        ).fetchone()[0]
        stats = con.execute(
            f"SELECT "
            f"  AVG(len(txn_day_offset_seq))::DOUBLE, "
            f"  MIN(len(txn_day_offset_seq)), "
            f"  MAX(len(txn_day_offset_seq)), "
            f"  COUNT(*) FILTER (WHERE len(txn_day_offset_seq) = 0) "
            f"FROM read_parquet('{output_path}')"
        ).fetchone()
        logger.info(
            "Patched parquet: rows=%d, day_offset len: avg=%.1f, min=%d, "
            "max=%d, empty_rows=%d",
            rows, stats[0], stats[1], stats[2], stats[3],
        )
    finally:
        con.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--day-window", type=int, default=360)
    args = p.parse_args()

    patch(args.input, args.output, day_window=args.day_window)
    return 0


if __name__ == "__main__":
    sys.exit(main())
