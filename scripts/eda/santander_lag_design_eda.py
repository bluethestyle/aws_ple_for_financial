"""
Lag design EDA for Santander parquet.

Config-driven scan to support FE -> model input redesign (mirrors the on-prem
2026-04-25 lag design EDA on AWS Santander data).

Outputs JSON + HTML reports under outputs/eda/.

Usage:
    python scripts/eda/santander_lag_design_eda.py \
        --parquet outputs/phase0_v12/santander_final.parquet \
        --seq-columns txn_mcc_seq txn_amount_seq nba_label \
        --mcc-col txn_mcc_seq \
        --day-offset-col txn_day_offset_seq \
        --output-json outputs/eda/santander_lag_design.json \
        --output-html outputs/eda/santander_lag_design.html
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger("lag_design_eda")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _seq_distribution(con: duckdb.DuckDBPyConnection, parquet: str, col: str) -> dict[str, Any]:
    sql = f"""
    SELECT
        COUNT(*) AS n,
        AVG(len({col})) AS mean,
        STDDEV(len({col})) AS std,
        MIN(len({col})) AS min,
        MAX(len({col})) AS max,
        QUANTILE_CONT(len({col}), 0.10) AS p10,
        QUANTILE_CONT(len({col}), 0.25) AS p25,
        QUANTILE_CONT(len({col}), 0.50) AS p50,
        QUANTILE_CONT(len({col}), 0.75) AS p75,
        QUANTILE_CONT(len({col}), 0.90) AS p90,
        QUANTILE_CONT(len({col}), 0.95) AS p95,
        QUANTILE_CONT(len({col}), 0.99) AS p99,
        SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS null_cnt
    FROM '{parquet}'
    """
    row = con.execute(sql).fetchone()
    keys = ["n", "mean", "std", "min", "max",
            "p10", "p25", "p50", "p75", "p90", "p95", "p99", "null_cnt"]
    return {k: (float(v) if isinstance(v, (int, float)) and v is not None else v)
            for k, v in zip(keys, row)}


def _k_coverage(con: duckdb.DuckDBPyConnection, parquet: str, col: str,
                k_candidates: list[int]) -> dict[int, dict[str, float]]:
    """For each K return % of customers fully captured (len <= K) and % of
    total events captured (sum(min(len, K)) / sum(len))."""
    out: dict[int, dict[str, float]] = {}
    for k in k_candidates:
        sql = f"""
        SELECT
            SUM(CASE WHEN len({col}) <= {k} THEN 1 ELSE 0 END) * 100.0 / COUNT(*)
              AS pct_full_capture,
            SUM(LEAST(len({col}), {k})) * 100.0 / NULLIF(SUM(len({col})), 0)
              AS pct_event_capture
        FROM '{parquet}'
        """
        full, ev = con.execute(sql).fetchone()
        out[k] = {
            "pct_customers_fully_captured": float(full) if full is not None else 0.0,
            "pct_events_captured": float(ev) if ev is not None else 0.0,
        }
    return out


def _cold_heavy_split(con: duckdb.DuckDBPyConnection, parquet: str, col: str,
                      cold_thr: int, heavy_thr: int, cap: int) -> dict[str, float]:
    sql = f"""
    SELECT
        SUM(CASE WHEN len({col}) <= {cold_thr} THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_cold,
        SUM(CASE WHEN len({col}) >  {heavy_thr} THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_heavy,
        SUM(CASE WHEN len({col}) >= {cap} THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_cap_hit,
        SUM(CASE WHEN len({col}) =  0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_empty
    FROM '{parquet}'
    """
    cold, heavy, cap_hit, empty = con.execute(sql).fetchone()
    return {
        "pct_coldstart": float(cold or 0.0),
        "pct_heavytail": float(heavy or 0.0),
        "pct_cap_hit": float(cap_hit or 0.0),
        "pct_empty": float(empty or 0.0),
    }


def _category_coverage(con: duckdb.DuckDBPyConnection, parquet: str, col: str,
                       top_ns: list[int]) -> dict[str, Any]:
    """UNNEST a LIST column then compute Top-N coverage."""
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE _unnested AS
        SELECT UNNEST({col}) AS v FROM '{parquet}' WHERE {col} IS NOT NULL
    """)
    con.execute("""
        CREATE OR REPLACE TEMPORARY TABLE _ranked AS
        SELECT v, COUNT(*) AS cnt FROM _unnested GROUP BY v
    """)
    unique_cnt, total_cnt = con.execute(
        "SELECT COUNT(*), COALESCE(SUM(cnt), 0) FROM _ranked"
    ).fetchone()
    out: dict[str, Any] = {
        "column": col,
        "unique_values": int(unique_cnt),
        "total_events": int(total_cnt or 0),
    }
    for n in top_ns:
        sql = f"""
        SELECT COALESCE(SUM(cnt), 0) * 100.0 / NULLIF({total_cnt or 0}, 0) AS pct
        FROM (SELECT cnt FROM _ranked ORDER BY cnt DESC LIMIT {n}) t
        """
        pct = con.execute(sql).fetchone()[0]
        out[f"top_{n}_pct"] = float(pct or 0.0)
    return out


def _interval_distribution(con: duckdb.DuckDBPyConnection, parquet: str, col: str) -> dict[str, Any]:
    sql = f"""
    WITH u AS (
        SELECT UNNEST({col}) AS d FROM '{parquet}' WHERE {col} IS NOT NULL
    )
    SELECT
        COUNT(*) AS n,
        AVG(d) AS mean,
        STDDEV(d) AS std,
        MIN(d) AS min,
        MAX(d) AS max,
        QUANTILE_CONT(d, 0.10) AS p10,
        QUANTILE_CONT(d, 0.50) AS p50,
        QUANTILE_CONT(d, 0.90) AS p90,
        QUANTILE_CONT(d, 0.99) AS p99
    FROM u
    """
    row = con.execute(sql).fetchone()
    keys = ["n", "mean", "std", "min", "max", "p10", "p50", "p90", "p99"]
    return {k: (float(v) if isinstance(v, (int, float)) and v is not None else v)
            for k, v in zip(keys, row)}


def _format_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2f}%"


def _format_num(v: float | int | None, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:,.{digits}f}"


def _coverage_severity_class(pct: float) -> str:
    if pct >= 90:
        return "sev-low"
    if pct >= 80:
        return "sev-mid"
    if pct >= 60:
        return "sev-high"
    return "sev-critical"


def _verdict_from_coverage(coverage: dict[int, dict[str, float]]) -> tuple[int, str]:
    """Pick the smallest K whose pct_customers_fully_captured >= 90%; if none, return max K."""
    best_k = max(coverage.keys())
    for k in sorted(coverage.keys()):
        if coverage[k]["pct_customers_fully_captured"] >= 90.0:
            best_k = k
            break
    rationale = (f"K={best_k} captures "
                 f"{coverage[best_k]['pct_customers_fully_captured']:.1f}% of customers fully "
                 f"and {coverage[best_k]['pct_events_captured']:.1f}% of events.")
    return best_k, rationale


def write_html(results: dict[str, Any], path: str) -> None:
    title = "Santander Lag Design EDA"
    src = results["source"]
    rows = results["row_count"]
    seq_dist = results["sequence_distributions"]
    k_cov = results["k_coverage"]
    splits = results["cold_heavy_split"]
    mcc = results.get("category_coverage")
    intervals = results.get("inter_txn_intervals")
    config = results.get("config", {})

    parts: list[str] = []
    parts.append(f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>{title}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Malgun Gothic", sans-serif;
         max-width: 1180px; margin: 2rem auto; padding: 0 2rem;
         color: #1f2937; line-height: 1.7; word-break: keep-all; }}
  h1 {{ border-bottom: 3px solid #1d4ed8; padding-bottom: 0.6rem; color: #1e3a8a; }}
  h2 {{ border-bottom: 2px solid #e5e7eb; padding-bottom: 0.4rem; margin-top: 2.5rem; color: #111827; }}
  h3 {{ color: #374151; margin-top: 1.5rem; }}
  .meta {{ color: #6b7280; font-size: 0.9rem; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.92rem; }}
  th, td {{ border: 1px solid #d1d5db; padding: 0.5rem 0.7rem; text-align: left; vertical-align: top; }}
  th {{ background: #f3f4f6; font-weight: 600; color: #111827; }}
  tr:nth-child(even) td {{ background: #fafafa; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  code {{ background: #f3f4f6; padding: 0.1rem 0.4rem; border-radius: 3px; font-size: 0.9em; color: #be123c; }}
  .sev-critical {{ color: #dc2626; font-weight: 700; }}
  .sev-high     {{ color: #ea580c; font-weight: 600; }}
  .sev-mid      {{ color: #ca8a04; }}
  .sev-low      {{ color: #166534; font-weight: 600; }}
  .verdict {{ background: #1e293b; color: #f1f5f9; padding: 1.2rem 1.4rem; border-radius: 8px; margin: 1.5rem 0; }}
  .verdict h3 {{ color: #fbbf24; margin-top: 0; border: 0; }}
  .callout {{ border-left: 4px solid #1d4ed8; background: #eff6ff; padding: 0.8rem 1.2rem; margin: 1rem 0; border-radius: 0 6px 6px 0; }}
  .callout-warn {{ border-left-color: #ea580c; background: #fff7ed; }}
</style></head><body>""")

    parts.append(f"<h1>{title}</h1>")
    parts.append(f'<p class="meta"><strong>Source</strong>: <code>{src}</code> &middot; '
                 f'<strong>Rows</strong>: {rows:,} &middot; '
                 f'<strong>Cap (max_len)</strong>: {config.get("cap", "n/a")} &middot; '
                 f'<strong>Cold-start threshold</strong>: &le;{config.get("cold_thr", "n/a")} &middot; '
                 f'<strong>Heavy-tail threshold</strong>: &gt;{config.get("heavy_thr", "n/a")}</p>')

    parts.append('<div class="callout callout-warn">'
                 '<strong>⚠ Synthetic data caveat</strong> &mdash; Santander parquet 시퀀스는 '
                 f'Phase 0 단에서 <code>max_len={config.get("cap", "?")}</code> 으로 미리 truncate된 '
                 '합성 데이터입니다. 본 EDA의 K 권고는 "현 cap 한도 내 최적값" 이며, '
                 '실 운영 데이터(예: 우체국금융) 적용 시 동일 EDA 재실행 후 K 재결정이 필수입니다.</div>')

    # Verdict per primary sequence column (first one)
    primary = list(k_cov.keys())[0] if k_cov else None
    if primary:
        best_k, rationale = _verdict_from_coverage(k_cov[primary])
        parts.append(f'<div class="verdict"><h3>🎯 K 권고 (primary: <code>{primary}</code>)</h3>'
                     f'<p><strong>K = {best_k}</strong> &mdash; {rationale}</p></div>')

    # 1. Sequence length distribution
    parts.append("<h2>1. 시퀀스 길이 분포</h2>")
    parts.append("<table><thead><tr>"
                 "<th>Column</th><th>n</th><th>mean</th><th>std</th>"
                 "<th>min</th><th>p10</th><th>p50</th><th>p90</th><th>p95</th><th>p99</th><th>max</th>"
                 "</tr></thead><tbody>")
    for col, d in seq_dist.items():
        parts.append("<tr>"
                     f"<td><code>{col}</code></td>"
                     f"<td class=num>{_format_num(d['n'], 0)}</td>"
                     f"<td class=num>{_format_num(d['mean'], 2)}</td>"
                     f"<td class=num>{_format_num(d['std'], 2)}</td>"
                     f"<td class=num>{_format_num(d['min'], 0)}</td>"
                     f"<td class=num>{_format_num(d['p10'], 0)}</td>"
                     f"<td class=num>{_format_num(d['p50'], 0)}</td>"
                     f"<td class=num>{_format_num(d['p90'], 0)}</td>"
                     f"<td class=num>{_format_num(d['p95'], 0)}</td>"
                     f"<td class=num>{_format_num(d['p99'], 0)}</td>"
                     f"<td class=num>{_format_num(d['max'], 0)}</td>"
                     "</tr>")
    parts.append("</tbody></table>")

    # 2. K coverage
    parts.append("<h2>2. K 후보별 포착률</h2>")
    parts.append("<p><strong>pct_full_capture</strong>: 시퀀스 전체가 K 안에 들어오는 고객 비율. "
                 "<strong>pct_event_capture</strong>: 전체 이벤트 중 K 안에 보존되는 비율 "
                 "(<code>SUM(LEAST(len, K)) / SUM(len)</code>).</p>")
    for col, kdict in k_cov.items():
        parts.append(f"<h3><code>{col}</code></h3>")
        parts.append("<table><thead><tr>"
                     "<th>K</th><th>고객 완전 포착률</th><th>이벤트 보존률</th><th>판정</th>"
                     "</tr></thead><tbody>")
        for k in sorted(kdict.keys()):
            full = kdict[k]["pct_customers_fully_captured"]
            ev = kdict[k]["pct_events_captured"]
            cls = _coverage_severity_class(full)
            verdict = ("권장" if full >= 90 else
                       "차선" if full >= 80 else
                       "불충분" if full >= 60 else "기각")
            parts.append("<tr>"
                         f"<td class=num><strong>{k}</strong></td>"
                         f"<td class='num {cls}'>{_format_pct(full)}</td>"
                         f"<td class='num {cls}'>{_format_pct(ev)}</td>"
                         f"<td class={cls}>{verdict}</td>"
                         "</tr>")
        parts.append("</tbody></table>")

    # 3. Cold / heavy / cap split
    parts.append("<h2>3. Cold-start / Heavy-tail / Cap-hit 비율</h2>")
    parts.append("<table><thead><tr>"
                 "<th>Column</th><th>cold-start (≤thr)</th><th>heavy-tail (&gt;thr)</th>"
                 "<th>cap-hit</th><th>empty list</th>"
                 "</tr></thead><tbody>")
    for col, d in splits.items():
        parts.append("<tr>"
                     f"<td><code>{col}</code></td>"
                     f"<td class=num>{_format_pct(d['pct_coldstart'])}</td>"
                     f"<td class=num>{_format_pct(d['pct_heavytail'])}</td>"
                     f"<td class=num>{_format_pct(d['pct_cap_hit'])}</td>"
                     f"<td class=num>{_format_pct(d['pct_empty'])}</td>"
                     "</tr>")
    parts.append("</tbody></table>")

    # 4. Category coverage (MCC top-N)
    if mcc:
        parts.append("<h2>4. 카테고리 Top-N 커버리지</h2>")
        parts.append(f"<p>Column: <code>{mcc['column']}</code> &middot; "
                     f"unique values: <strong>{mcc['unique_values']:,}</strong> &middot; "
                     f"total events: <strong>{mcc['total_events']:,}</strong></p>")
        parts.append("<table><thead><tr><th>Top-N</th><th>커버리지</th></tr></thead><tbody>")
        for k, v in mcc.items():
            if k.startswith("top_"):
                cls = _coverage_severity_class(v)
                parts.append(f"<tr><td class=num><strong>{k}</strong></td>"
                             f"<td class='num {cls}'>{_format_pct(v)}</td></tr>")
        parts.append("</tbody></table>")

    # 5. Inter-txn intervals
    if intervals:
        parts.append("<h2>5. 거래 간 시간 간격 분포</h2>")
        parts.append(f"<p>Column: <code>{intervals['column']}</code> &middot; "
                     f"unit: 일 (txn_day_offset_seq의 elapsed days)</p>")
        d = intervals["stats"]
        parts.append("<table><thead><tr>"
                     "<th>n</th><th>mean</th><th>std</th>"
                     "<th>min</th><th>p10</th><th>p50</th><th>p90</th><th>p99</th><th>max</th>"
                     "</tr></thead><tbody><tr>"
                     f"<td class=num>{_format_num(d['n'], 0)}</td>"
                     f"<td class=num>{_format_num(d['mean'], 2)}</td>"
                     f"<td class=num>{_format_num(d['std'], 2)}</td>"
                     f"<td class=num>{_format_num(d['min'], 0)}</td>"
                     f"<td class=num>{_format_num(d['p10'], 0)}</td>"
                     f"<td class=num>{_format_num(d['p50'], 0)}</td>"
                     f"<td class=num>{_format_num(d['p90'], 0)}</td>"
                     f"<td class=num>{_format_num(d['p99'], 0)}</td>"
                     f"<td class=num>{_format_num(d['max'], 0)}</td>"
                     "</tr></tbody></table>")

    parts.append("<hr><p class=meta>Generated by <code>scripts/eda/santander_lag_design_eda.py</code></p>")
    parts.append("</body></html>")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, help="Path to parquet file")
    parser.add_argument("--seq-columns", nargs="+", required=True,
                        help="LIST columns to analyze for length distribution / K coverage")
    parser.add_argument("--k-candidates", nargs="+", type=int,
                        default=[30, 50, 100, 150, 180, 200, 300, 500])
    parser.add_argument("--cold-threshold", type=int, default=5)
    parser.add_argument("--heavy-threshold", type=int, default=100)
    parser.add_argument("--cap", type=int, default=200, help="Phase 0 max_len cap (for cap-hit pct)")
    parser.add_argument("--mcc-col", default=None,
                        help="LIST<INT> column for Top-N category coverage (e.g. txn_mcc_seq)")
    parser.add_argument("--top-n", nargs="+", type=int, default=[10, 30, 50, 100])
    parser.add_argument("--day-offset-col", default=None,
                        help="LIST<INT> column for inter-txn interval stats (e.g. txn_day_offset_seq)")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-html", required=True)
    args = parser.parse_args()

    if not Path(args.parquet).exists():
        raise SystemExit(f"[fail] parquet not found: {args.parquet}")

    con = duckdb.connect()
    logger.info("connecting to parquet: %s", args.parquet)

    results: dict[str, Any] = {
        "source": args.parquet,
        "row_count": int(con.execute(
            f"SELECT COUNT(*) FROM '{args.parquet}'").fetchone()[0]),
        "config": {
            "cap": args.cap,
            "cold_thr": args.cold_threshold,
            "heavy_thr": args.heavy_threshold,
            "k_candidates": args.k_candidates,
            "top_n": args.top_n,
        },
        "sequence_distributions": {},
        "k_coverage": {},
        "cold_heavy_split": {},
    }

    for col in args.seq_columns:
        logger.info("[1/3] length distribution: %s", col)
        results["sequence_distributions"][col] = _seq_distribution(con, args.parquet, col)
        logger.info("[2/3] K coverage:          %s", col)
        results["k_coverage"][col] = _k_coverage(con, args.parquet, col, args.k_candidates)
        logger.info("[3/3] cold/heavy split:    %s", col)
        results["cold_heavy_split"][col] = _cold_heavy_split(
            con, args.parquet, col, args.cold_threshold, args.heavy_threshold, args.cap)

    if args.mcc_col:
        logger.info("category coverage: %s", args.mcc_col)
        results["category_coverage"] = _category_coverage(
            con, args.parquet, args.mcc_col, args.top_n)

    if args.day_offset_col:
        logger.info("inter-txn intervals: %s", args.day_offset_col)
        results["inter_txn_intervals"] = {
            "column": args.day_offset_col,
            "stats": _interval_distribution(con, args.parquet, args.day_offset_col),
        }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("[ok] JSON: %s", args.output_json)

    write_html(results, args.output_html)
    logger.info("[ok] HTML: %s", args.output_html)


if __name__ == "__main__":
    main()
