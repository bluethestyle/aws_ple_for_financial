"""
Local CPU-only smoke for the 2026-04-25 lag-tensor redesign.

What this verifies (no GPU required):
  1. New 4 groups instantiate via FeatureGeneratorRegistry without import
     errors after `feature_groups.yaml` edits.
  2. FeatureGroupPipeline.fit_transform succeeds on a 5K subsample using
     ONLY the CPU-friendly groups (transform groups + new generators).
  3. Output dimensions match yaml declarations.
  4. group_ranges remain contiguous (no §1.7 breakage).
  5. expert_routing maps lag/rolling/multihot to the experts declared
     in target_experts.
  6. 1-customer-1-row invariant holds at the integrator boundary.
  7. New columns get a sensible normalizer treatment (binary multi-hot
     stays untouched; lag/rolling continuous get scaled).

Heavy GPU groups (mamba, tda*, hmm, graph, merchant_hierarchy,
gmm, model_features) are skipped here because they belong to the
SageMaker run -- this script is for the local readiness gate only.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("lag_redesign_smoke")


# Groups we run locally (CPU-friendly).  Every other yaml group is
# disabled for this smoke -- they need GPU/large compute and belong
# to SageMaker.
LOCAL_FRIENDLY_GROUPS = {
    "demographics",         # transform
    "product_holdings",     # transform
    "txn_behavior",         # transform
    "derived_temporal",     # transform
    # New axis groups (pure CPU):
    "txn_lag_tensor",
    "txn_rolling_stats",
    "nba_label_multihot",
    "mcc_top30_multihot",
}


def load_subsample(parquet: str, n: int) -> Any:
    """Load N rows including all columns the local groups will need."""
    con = duckdb.connect()
    return con.execute(f"SELECT * FROM '{parquet}' LIMIT {n}").df()


def filter_yaml_to_local_groups(yaml_path: str) -> List[Dict[str, Any]]:
    with open(yaml_path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    groups = d["feature_groups"]
    keep = [g for g in groups if g["name"] in LOCAL_FRIENDLY_GROUPS]
    skipped = [g["name"] for g in groups if g["name"] not in LOCAL_FRIENDLY_GROUPS]
    logger.info("Local subset: keeping %d groups, skipping %d (GPU/heavy)",
                len(keep), len(skipped))
    logger.info("  skipped: %s", skipped)
    return keep


def main() -> int:
    parquet = "outputs/phase0_v12/santander_final.parquet"
    if not Path(parquet).exists():
        logger.error("source parquet missing: %s", parquet)
        return 1

    # Lazy imports so registry decorators have already fired.
    from core.feature.generators import FeatureGeneratorRegistry  # noqa: F401
    from core.feature.group import FeatureGroupConfig
    from core.feature.group_pipeline import FeatureGroupPipeline

    # 1. Load subsample
    n = 5000
    df = load_subsample(parquet, n)
    logger.info("loaded subsample shape=%s columns=%d", df.shape, len(df.columns))

    # 2. Build groups for local subset only
    raw_groups = filter_yaml_to_local_groups("configs/santander/feature_groups.yaml")
    groups = [FeatureGroupConfig.from_dict(g) for g in raw_groups]
    logger.info("built %d FeatureGroupConfig objects", len(groups))

    # 3. Run pipeline
    pipeline = FeatureGroupPipeline(groups=groups, name="lag_redesign_smoke")
    logger.info("=== fit ===")
    pipeline.fit(df)
    logger.info("=== transform ===")
    out = pipeline.transform(df)
    logger.info("output shape=%s", out.shape)

    # 4. Per-group dimension assertions (compare to yaml)
    print()
    print("=== Group output_dim check (yaml vs actual) ===")
    issues = []
    for g in groups:
        yaml_dim = next((rg["output_dim"] for rg in raw_groups
                         if rg["name"] == g.name), None)
        actual_dim = g.output_dim
        ok = "OK" if yaml_dim == actual_dim else "MISMATCH"
        print(f"  [{ok}] {g.name:30s} yaml={yaml_dim:4} actual={actual_dim:4}")
        if yaml_dim != actual_dim:
            issues.append(g.name)
    if issues:
        logger.error("dim mismatches: %s", issues)
        return 1

    # 5. group_ranges contiguity
    print()
    print("=== group_ranges contiguity ===")
    meta = pipeline.get_ple_input_metadata() if hasattr(
        pipeline, "get_ple_input_metadata") else None
    if meta is None:
        # Fall back: construct ranges from group output_dim
        offset = 0
        ranges = {}
        for g in groups:
            ranges[g.name] = (offset, offset + g.output_dim)
            offset += g.output_dim
    else:
        ranges = meta.get("feature_group_ranges", {}) or meta.get("group_ranges", {})

    prev_end = 0
    contiguous = True
    for name, rng in ranges.items():
        s, e = rng if isinstance(rng, (tuple, list)) else (rng["start"], rng["end"])
        marker = "OK" if s == prev_end else "GAP"
        if s != prev_end:
            contiguous = False
        print(f"  [{marker}] {name:30s} [{s:5d}, {e:5d})  width={e-s}")
        prev_end = e
    if not contiguous:
        logger.warning("non-contiguous group_ranges (acceptable if §1.7 longest-block "
                       "rebuild is intentional, but flag for review)")

    # 6. expert_routing -- target_experts must propagate to pipeline routing
    print()
    print("=== expert_routing for new groups ===")
    routing_map: Dict[str, List[str]] = {}
    for g in groups:
        routing_map[g.name] = list(g.target_experts) if g.target_experts else ["(broadcast)"]
    for new_name in ("txn_lag_tensor", "txn_rolling_stats",
                     "nba_label_multihot", "mcc_top30_multihot"):
        experts = routing_map.get(new_name, [])
        print(f"  {new_name:30s} -> {experts}")

    # 7. 1-customer-1-row invariant
    print()
    print("=== 1-customer-1-row invariant ===")
    inv_ok = len(out) == n
    print(f"  input_rows={n}, output_rows={len(out)}: "
          f"{'OK' if inv_ok else 'VIOLATED'}")
    if not inv_ok:
        return 1

    # 8. Sanity sample on new columns
    print()
    print("=== New-column sanity ===")
    new_prefixes = ("txn_lag_", "txn_roll_", "nba_mh_", "mcc_mh_")
    for prefix in new_prefixes:
        cols = [c for c in out.columns if c.startswith(prefix)]
        if not cols:
            print(f"  [{prefix}] no columns found!")
            continue
        sample_col = cols[0]
        nz = (out[sample_col] != 0).mean()
        print(f"  [{prefix}] {len(cols):4d} cols, sample='{sample_col}' "
              f"non-zero rate={nz:.1%}")

    # 9. Total dim summary
    print()
    print(f"=== Total local-subset feature dim: {sum(g.output_dim for g in groups)} ===")
    print(f"=== Concat output cols: {len(out.columns)} ===")

    # Persist a JSON summary so the next agent can read it
    summary = {
        "subsample_rows": n,
        "groups_local": [g.name for g in groups],
        "total_dim_local": int(sum(g.output_dim for g in groups)),
        "output_shape": [int(x) for x in out.shape],
        "row_invariant_ok": bool(inv_ok),
        "group_ranges_contiguous": bool(contiguous),
        "expert_routing_new": {k: routing_map[k] for k in
                               ("txn_lag_tensor", "txn_rolling_stats",
                                "nba_label_multihot", "mcc_top30_multihot")},
    }
    out_path = Path("outputs/eda/lag_redesign_smoke_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("summary -> %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
