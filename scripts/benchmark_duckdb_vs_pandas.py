"""
Benchmark: DuckDB vs pandas for data operations used in this project.

Measures wall-clock time (median of 3 runs) and peak RSS memory for:
  1. Parquet full read
  2. Group-by aggregation (COUNT, SUM, AVG, STDDEV)
  3. Column selection (50 of ~375 columns)
  4. Filter + aggregate (simulating temporal split logic)

Memory is measured via psutil RSS delta (before vs. peak-during-op) using a
background thread sampler.  This avoids the tracemalloc + pyarrow allocator
segfault on some platforms.

Output: console table + outputs/benchmark_duckdb_vs_pandas.md

Usage:
    python scripts/benchmark_duckdb_vs_pandas.py
    python scripts/benchmark_duckdb_vs_pandas.py --parquet path/to/file.parquet
    python scripts/benchmark_duckdb_vs_pandas.py --runs 5
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Guard: bail early if the parquet file is missing
# ---------------------------------------------------------------------------
DEFAULT_PARQUET = "outputs/phase0/santander_final.parquet"


def _resolve_parquet(cli_path: str) -> Optional[Path]:
    p = Path(cli_path)
    if not p.is_absolute():
        repo_root = Path(__file__).parent.parent
        p = repo_root / p
    if not p.exists():
        return None
    return p.resolve()


# ---------------------------------------------------------------------------
# Memory helpers — psutil RSS sampler (safe with pyarrow/DuckDB allocators)
# ---------------------------------------------------------------------------

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class _RssSampler:
    """Background thread that records peak RSS while a callable runs."""

    def __init__(self, interval: float = 0.05) -> None:
        self._proc = _psutil.Process(os.getpid()) if _HAS_PSUTIL else None
        self._interval = interval
        self._peak: float = 0.0
        self._baseline: float = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _sample_loop(self) -> None:
        while self._running:
            try:
                rss = self._proc.memory_info().rss
                if rss > self._peak:
                    self._peak = rss
            except Exception:
                pass
            time.sleep(self._interval)

    def __enter__(self) -> "_RssSampler":
        gc.collect()
        if self._proc is not None:
            self._baseline = self._proc.memory_info().rss
            self._peak = self._baseline
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    @property
    def peak_delta_mb(self) -> float:
        """Peak RSS above baseline in MB. Returns 0 if psutil unavailable."""
        if self._proc is None:
            return 0.0
        delta = max(0, self._peak - self._baseline)
        return delta / 1024 / 1024


# ---------------------------------------------------------------------------
# Core measurement helper
# ---------------------------------------------------------------------------

def _measure(fn: Callable[[], None]) -> Tuple[float, float]:
    """Run *fn*, return (elapsed_seconds, peak_rss_delta_MB)."""
    gc.collect()
    sampler = _RssSampler()
    with sampler:
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
    gc.collect()
    return elapsed, sampler.peak_delta_mb


def _run_n(fn: Callable[[], None], n: int = 3) -> Tuple[float, float]:
    """Run *fn* n times, return (median_elapsed_s, median_peak_MB)."""
    times: List[float] = []
    mems: List[float] = []
    for _ in range(n):
        elapsed, peak = _measure(fn)
        times.append(elapsed)
        mems.append(peak)
    return statistics.median(times), statistics.median(mems)


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

def bench_read_full_pandas(parquet_path: str) -> None:
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    _ = len(df)  # ensure materialised


def bench_read_full_duckdb(parquet_path: str) -> None:
    import duckdb
    conn = duckdb.connect(":memory:")
    norm = parquet_path.replace("\\", "/")
    df = conn.execute(f"SELECT * FROM read_parquet('{norm}')").fetchdf()
    _ = len(df)
    conn.close()


def bench_groupby_pandas(parquet_path: str) -> None:
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    # Simulate per-segment feature aggregation (mirrors runner.py GROUP BY logic)
    agg = df.groupby("segment").agg(
        n_rows=("customer_id", "count"),
        sum_income=("income", "sum"),
        avg_income=("income", "mean"),
        avg_products=("num_products", "mean"),
        avg_monetary=("synth_monetary", "mean"),
        stddev_income=("income", "std"),
    )
    _ = len(agg)


def bench_groupby_duckdb(parquet_path: str) -> None:
    import duckdb
    conn = duckdb.connect(":memory:")
    norm = parquet_path.replace("\\", "/")
    conn.execute(
        f"""
        SELECT
            segment,
            COUNT(*)                AS n_rows,
            SUM(income)             AS sum_income,
            AVG(income)             AS avg_income,
            AVG(num_products)       AS avg_products,
            AVG(synth_monetary)     AS avg_monetary,
            STDDEV(income)          AS stddev_income
        FROM read_parquet('{norm}')
        GROUP BY segment
        """
    ).fetchdf()
    conn.close()


# 50 column names present in santander_final.parquet
_COL50 = [
    "age", "income", "tenure_months", "gender", "segment", "country", "channel",
    "is_active", "age_group", "income_group",
    "prod_saving", "prod_guarantee", "prod_checking", "prod_derivados",
    "prod_payroll_acct", "prod_junior_acct", "prod_particular_acct",
    "prod_particular_plus", "prod_short_deposit", "prod_medium_deposit",
    "prod_long_deposit", "prod_e_account", "prod_funds", "prod_mortgage",
    "prod_pension_plan", "prod_loans", "prod_taxes", "prod_credit_card",
    "prod_securities", "prod_home_acct", "prod_payroll", "prod_pension_deposit",
    "prod_direct_debit", "prod_auto_debit", "num_products",
    "synth_monthly_txns", "synth_avg_amount", "synth_monthly_spend",
    "synth_unique_mcc", "synth_unique_merchants", "synth_morning_ratio",
    "synth_afternoon_ratio", "synth_evening_ratio", "synth_night_ratio",
    "synth_recency_days", "synth_frequency", "synth_monetary",
    "synth_stability", "synth_fraud_ratio", "total_acquisitions",
]


def bench_col_select_pandas(parquet_path: str) -> None:
    import pandas as pd
    df = pd.read_parquet(parquet_path, columns=_COL50)
    _ = df.shape


def bench_col_select_duckdb(parquet_path: str) -> None:
    import duckdb
    conn = duckdb.connect(":memory:")
    norm = parquet_path.replace("\\", "/")
    col_expr = ", ".join(f'"{c}"' for c in _COL50)
    conn.execute(
        f"SELECT {col_expr} FROM read_parquet('{norm}')"
    ).fetchdf()
    conn.close()


def bench_filter_agg_pandas(parquet_path: str) -> None:
    """Simulate temporal split: filter by date string, then aggregate."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    # Train split: snapshot_date < '2017-01' (approx 70% of data)
    mask = df["snapshot_date"] < "2017-01"
    train = df[mask]
    agg = train.groupby("segment").agg(
        n=("customer_id", "count"),
        avg_income=("income", "mean"),
        avg_tenure=("tenure_months", "mean"),
        sum_products=("num_products", "sum"),
    )
    _ = len(agg)


def bench_filter_agg_duckdb(parquet_path: str) -> None:
    """Same logic via SQL — filter + GROUP BY in a single scan."""
    import duckdb
    conn = duckdb.connect(":memory:")
    norm = parquet_path.replace("\\", "/")
    conn.execute(
        f"""
        SELECT
            segment,
            COUNT(*)            AS n,
            AVG(income)         AS avg_income,
            AVG(tenure_months)  AS avg_tenure,
            SUM(num_products)   AS sum_products
        FROM read_parquet('{norm}')
        WHERE snapshot_date < '2017-01'
        GROUP BY segment
        """
    ).fetchdf()
    conn.close()


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _fmt_time(s: float) -> str:
    if s < 1:
        return f"{s * 1000:.1f} ms"
    return f"{s:.2f} s"


def _fmt_mem(mb: float) -> str:
    if mb <= 0:
        return "n/a"
    if mb < 1:
        return f"{mb * 1024:.0f} KB"
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.1f} MB"


def _fmt_speedup(pd_t: float, dk_t: float) -> str:
    if dk_t <= 0:
        return "N/A"
    ratio = pd_t / dk_t
    return f"{ratio:.1f}x"


def build_markdown_table(results: List[Dict]) -> str:
    header = (
        "| Operation | pandas time | DuckDB time | Speedup "
        "| pandas peak mem | DuckDB peak mem |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in results:
        rows.append(
            f"| {r['op']} "
            f"| {_fmt_time(r['pd_t'])} "
            f"| {_fmt_time(r['dk_t'])} "
            f"| {_fmt_speedup(r['pd_t'], r['dk_t'])} "
            f"| {_fmt_mem(r['pd_m'])} "
            f"| {_fmt_mem(r['dk_m'])} |"
        )
    return header + "\n".join(rows)


def print_console_table(results: List[Dict]) -> None:
    col_w = [36, 14, 14, 10, 18, 18]
    headers = ["Operation", "pandas time", "DuckDB time", "Speedup",
               "pandas peak mem", "DuckDB peak mem"]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"

    def row_fmt(cells: List[str]) -> str:
        return "| " + " | ".join(
            str(c).ljust(w) for c, w in zip(cells, col_w)
        ) + " |"

    print(sep)
    print(row_fmt(headers))
    print(sep)
    for r in results:
        print(row_fmt([
            r["op"],
            _fmt_time(r["pd_t"]),
            _fmt_time(r["dk_t"]),
            _fmt_speedup(r["pd_t"], r["dk_t"]),
            _fmt_mem(r["pd_m"]),
            _fmt_mem(r["dk_m"]),
        ]))
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DuckDB vs pandas for Parquet operations."
    )
    parser.add_argument(
        "--parquet",
        default=DEFAULT_PARQUET,
        help=f"Path to input Parquet file (default: {DEFAULT_PARQUET})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Repetitions per operation (default: 3, median reported)",
    )
    args = parser.parse_args()

    parquet_path = _resolve_parquet(args.parquet)
    if parquet_path is None:
        print(
            f"[SKIP] Parquet file not found: {args.parquet}\n"
            "       Run Phase 0 pipeline first to generate the file.",
            file=sys.stderr,
        )
        sys.exit(0)

    if not _HAS_PSUTIL:
        print(
            "[WARN] psutil not available — memory columns will show 'n/a'.\n"
            "       Install with: pip install psutil",
            file=sys.stderr,
        )

    parquet_str = str(parquet_path)
    n = args.runs
    print(f"Benchmarking on: {parquet_str}")
    print(f"Rows: ~1 M, columns: ~375 | Runs per operation: {n} (median)\n")

    benchmarks = [
        (
            "Parquet full read",
            lambda: bench_read_full_pandas(parquet_str),
            lambda: bench_read_full_duckdb(parquet_str),
        ),
        (
            "Group-by aggregation",
            lambda: bench_groupby_pandas(parquet_str),
            lambda: bench_groupby_duckdb(parquet_str),
        ),
        (
            "Column selection (50/375 cols)",
            lambda: bench_col_select_pandas(parquet_str),
            lambda: bench_col_select_duckdb(parquet_str),
        ),
        (
            "Filter + aggregate (temporal split)",
            lambda: bench_filter_agg_pandas(parquet_str),
            lambda: bench_filter_agg_duckdb(parquet_str),
        ),
    ]

    results: List[Dict] = []
    for op_name, pd_fn, dk_fn in benchmarks:
        print(f"  [{op_name}]")
        print(f"    pandas ...", end="", flush=True)
        pd_t, pd_m = _run_n(pd_fn, n)
        print(f" {_fmt_time(pd_t)}, peak RSS delta {_fmt_mem(pd_m)}")

        print(f"    duckdb ...", end="", flush=True)
        dk_t, dk_m = _run_n(dk_fn, n)
        print(f" {_fmt_time(dk_t)}, peak RSS delta {_fmt_mem(dk_m)}")

        results.append(
            dict(op=op_name, pd_t=pd_t, pd_m=pd_m, dk_t=dk_t, dk_m=dk_m)
        )

    print()
    print_console_table(results)

    # Save markdown
    md_lines = [
        "# DuckDB vs pandas Benchmark",
        "",
        f"**Input file**: `{parquet_str}`  ",
        f"**Runs per op**: {n} (median reported)  ",
        f"**Data**: ~1 M rows x ~375 columns  ",
        "",
        build_markdown_table(results),
        "",
        "> **Memory**: peak RSS delta measured via `psutil` background sampler",
        "> (RSS before op subtracted from peak RSS during op).",
        "> DuckDB memory-maps Parquet files at OS level; Python-side RSS may",
        "> undercount true working set relative to pandas.",
    ]
    md_text = "\n".join(md_lines)

    repo_root = Path(__file__).parent.parent
    out_dir = repo_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_duckdb_vs_pandas.md"
    out_path.write_text(md_text, encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
