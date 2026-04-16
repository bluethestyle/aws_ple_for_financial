# DuckDB as the Data Engine for an ML Training Pipeline

> How a financial recommendation system replaced pandas with DuckDB SQL
> across the entire feature engineering pipeline — from Parquet to PyTorch tensors —
> on a single workstation.

**Repository**: [github.com/bluethestyle/aws_ple_for_financial](https://github.com/bluethestyle/aws_ple_for_financial)

---

## The Problem

This project is the AWS cloud extension of an on-premises financial
recommendation system running inside a Korean public financial institution.

**On-premises production scale**: 12 million customers, 734 features,
16 prediction tasks, 7 heterogeneous expert networks (PLE architecture).
The environment has Hive for data lake storage but no Spark or Impala.
All post-Hive processing — feature engineering, label derivation,
normalization, model input construction — runs on a single workstation
(RTX 4070 12 GB VRAM, 128 GB RAM) in an air-gapped network.
Requesting a Spark cluster or GPU infrastructure in a financial institution
is not a technical problem — it is a budget approval problem that does not
get approved.

With pandas or SQLite on this workstation, the pipeline hit severe bottlenecks:
OOM on wide joins, 40+ minute label derivation passes, and no memory headroom
for the PyTorch training that follows in the same process. Running Spark locally
was not viable — JVM overhead alone consumed too much of the 128 GB.

The question was not "how do we make it faster?" but
"how do we make it *possible* on this machine?"

**This case study** documents the DuckDB patterns from the AWS benchmark
version of the pipeline (1M synthetic customers, ~403 features after Phase 0,
open-source). The benchmarks below use this synthetic dataset.
The on-premises codebase (12M real customers, 734 features, 240+ DuckDB
source files) uses identical patterns and is being prepared for public
release separately. Production data is not included for regulatory reasons.

---

## The Solution: DuckDB as Pipeline Backbone

DuckDB replaced pandas across the entire data pipeline. Not as an analytics
add-on, but as the **primary data processing engine** from raw Parquet files
to the final PyTorch DataLoader.

Here is where DuckDB sits in the full lifecycle — from data generation
through training:

```
┌─────────────────────────────────────────────────────────────┐
│  DATA GENERATION (scripts/generate_benchmark_data.py)       │
│                                                             │
│  numpy random     numpy vectorized agg       Arrow → Parquet│
│  (raw matrices) → (synth_* features)       → (1.5 GB file) │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  TRAINING PIPELINE (core/pipeline/)                         │
│                                                             │
│  [1] DuckDB: read_parquet() ─────── Adapter                │
│  [2] DuckDB: SQL transforms ─────── Feature generators      │
│  [3] DuckDB: POSITIONAL JOIN ─────── Merge features         │
│  [4] DuckDB: SQL CASE/COALESCE ──── Label derivation (13×)  │
│  [5] DuckDB: register()+fetchnumpy() ─ 3-stage normalization│
│  [6] DuckDB: POSITIONAL JOIN ─────── Combine features+labels│
│  [7] .df() or .arrow() ─────────── → numpy → torch tensor  │
│                                                             │
│                        ▼                                    │
│                  PyTorch DataLoader                          │
└─────────────────────────────────────────────────────────────┘
```

pandas appears exactly once: at step [7], as the final conversion to numpy.
Everything upstream — including data generation itself — is SQL or Arrow.

---

## Pattern 1: Parquet-Native I/O

The adapter loads customer data directly from Parquet without a pandas intermediary:

```python
import duckdb

con = duckdb.connect()
df = con.execute(f"SELECT * FROM '{parquet_path}'").df()
```

The `QueryEngine` wrapper supports predicate pushdown and column selection,
so the full 403-column file is never materialized when only a subset is needed:

```python
sql = f"SELECT {col_expr} FROM read_parquet('{path}')"
if where:
    sql += f" WHERE {where}"
return self.query(sql)
```

Intermediate pipeline stages are checkpointed as ZSTD-compressed Parquet
without involving pandas:

```python
con.execute(f"COPY {table_name} TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
```

---

## Pattern 2: SQL Replaces 13 × groupby().apply(lambda)

The pipeline derives 13 labels (binary, multiclass, regression) from raw features.
The original pandas implementation ran 13 sequential `.apply(lambda)` calls
over 941K rows — each holding the full DataFrame in memory and serializing
row by row through Python.

The DuckDB version registers the DataFrame once and runs each derivation
as a SQL expression:

```python
con = duckdb.connect()
con.register("_label_src", df)   # zero-copy registration

# Bucket label — replaces pd.cut()
sql = """
    SELECT CASE
        WHEN val < 0   THEN 0
        WHEN val < 500 THEN 1
        ELSE 2
    END::INTEGER AS result
    FROM _label_src
"""
series = con.execute(sql).df()["result"]
```

We implemented **10 derivation types** as DuckDB SQL functions:

| Type | SQL Feature Used |
|---|---|
| `direct` | Simple column copy |
| `bucket` | `CASE WHEN` thresholds |
| `weighted_sum` | Arithmetic on `COALESCE`d columns |
| `string_map` | `CASE WHEN` string matching |
| `list_first` | `list[1]` indexing |
| `list_length` | `len(list)` |
| `list_intersect` | `list_intersect()` + `len()` |
| `sequence_last` | Negative list indexing `list[-1]` |
| `sequence_diversity_trend` | `list_distinct()` on list slices |
| `sequence_mode_shift` | Lateral `UNNEST` + `GROUP BY` + `ROW_NUMBER()` |

The last two deserve special mention. `sequence_mode_shift` detects whether
a customer's most frequent transaction category changed between the first
and second half of their history. In pandas, this requires unnesting a list
column into rows, grouping, counting, and ranking — a multi-step Python loop.
In DuckDB:

```sql
WITH unnested AS (
    SELECT _rid,
           UNNEST(seq_col[:len(seq_col) / 2]) AS val
    FROM _label_src
),
counted AS (
    SELECT _rid, val, COUNT(*) AS cnt
    FROM unnested GROUP BY _rid, val
),
ranked AS (
    SELECT _rid, val,
           ROW_NUMBER() OVER (PARTITION BY _rid ORDER BY cnt DESC, val) AS rn
    FROM counted
)
SELECT val FROM ranked WHERE rn = 1
```

One SQL query. No Python loop. Runs on 941K rows in seconds.

---

## Pattern 3: POSITIONAL JOIN for pandas-free Feature Merging

After feature generators (GMM clusters, TDA topology, HMM states, etc.)
produce new columns, their outputs must be merged back into the main
941K-row DataFrame. DuckDB's `POSITIONAL JOIN` merges by row position,
keeping the result in Arrow format without a pandas roundtrip:

```python
con = duckdb.connect()
con.register("_main", df_main.reset_index(drop=True))
con.register("_gen", df_generated[new_cols].reset_index(drop=True))

main_cols = ", ".join(f'_main."{c}"' for c in df_main.columns)
gen_cols  = ", ".join(f'_gen."{c}"' for c in df_generated.columns)

# .arrow() — stays in Arrow, no pandas conversion
merged = con.execute(
    f"SELECT {main_cols}, {gen_cols} FROM _main POSITIONAL JOIN _gen"
).arrow()
```

The same pattern combines the 403-column feature matrix with the 13-column
label matrix before DataLoader construction, with `pa.Table.take(indices)`
replacing `df.iloc[indices]` for train/val/test splits:

```python
tbl = con.execute(
    f"SELECT {feat_cols}, {lbl_cols} FROM _feat POSITIONAL JOIN _lbl"
).arrow()
train_tbl = tbl.take(train_indices)   # replaces df.iloc[train_idx]
```

**Why POSITIONAL JOIN, not `pd.concat`?** On 941K × 403 columns,
POSITIONAL JOIN → `.arrow()` runs in 1.08s vs. pandas concat in 2.07s (1.9×).
Memory consumption is similar — the advantage is **speed + pandas-free
consistency**, not memory reduction. DuckDB's real memory win comes from
aggregation/filtering (15 GB → 1 MB), not from merging.

---

## Pattern 4: Sliding Windows over 24M Rows

The pipeline constructs temporal transaction sequences by sliding a date window
over 24M transaction rows. DuckDB handles the per-window date filtering:

```python
con = duckdb.connect()
con.register("_seq_df", transactions)  # 24M rows, zero-copy

# Date range scan
date_range = con.execute("""
    SELECT MIN(tx_date) as min_dt, MAX(tx_date) as max_dt
    FROM _seq_df
""").fetchone()

# Per-window filter — only matching rows are materialized
window_df = con.execute(f"""
    SELECT customer_id, *
    FROM _seq_df
    WHERE tx_date >= '{window_start}' AND tx_date < '{window_end}'
""").df()
```

With pandas, filtering 24M rows per window was prohibitively slow.
DuckDB's columnar scan makes this feasible without an index.

---

## Pattern 5: The Bridge to PyTorch

The data loading layer detects DuckDB relations and avoids pandas entirely:

```python
if hasattr(df, 'fetchnumpy'):
    arrays = [np.asarray(df[c], dtype=np.float64) for c in columns]
    arr = np.column_stack(arrays).astype(np.float32)
    return torch.from_numpy(arr)
```

The preferred path uses Arrow:

```python
table = con.execute("SELECT feature_cols FROM features").arrow()
dataset = PLEDataset(df=table, feature_columns=spec, label_columns=labels)
```

The full data flow: **Parquet → DuckDB SQL → Arrow Table → numpy → PyTorch tensor**.
pandas is never in the critical path.

---

## The register/unregister Pattern

Every DuckDB interaction in the codebase follows the same idiom:

```python
con = duckdb.connect()
try:
    con.register("_table_name", df)          # zero-copy — no data copied
    result = con.execute("SELECT ...").df()  # or .arrow() / .fetchnumpy()
finally:
    con.unregister("_table_name")
    con.close()
```

`con.register()` exposes a pandas DataFrame, Arrow Table, or numpy array
to DuckDB SQL without copying the underlying buffers. The `finally` block
is important: the pipeline opens multiple DuckDB connections sequentially
within a single training job, and leaked connections accumulate memory.

This pattern appears in every pipeline stage — adapter, label deriver,
normalizer, sequence builder, feature merger, data loader — making
DuckDB's SQL engine a transparent layer between Python objects and
columnar operations.

---

## Pattern 6: Synthetic Data Generation

DuckDB is not just used for *processing* data — it is also used for
*generating* it. The project includes a synthetic benchmark data generator
(`scripts/generate_benchmark_data.py`) that produces 1M customers with
realistic transaction sequences. DuckDB drives the final stage of this pipeline.

### How synth_* Features Are Actually Computed

After generating raw transaction matrices (MCC codes, amounts, hours) per
customer via numpy, the aggregation is done entirely in numpy/Python — not in
DuckDB SQL. All `synth_*` scalar features (`synth_monetary`, `synth_frequency`,
`synth_recency_days`, `synth_unique_mcc`, time-of-day ratios, etc.) are computed
using vectorized numpy operations directly on the raw matrices. The results are
assembled into a PyArrow Table using `pa.table()`.

DuckDB is **not** used for `list_sum()`, `list_avg()`, or `list_distinct()`
aggregation at this stage. The speedup (~40×) comes from vectorized numpy, not
from DuckDB SQL aggregation.

Full data generation (1M customers including all 6 persona types, transaction
sequences, and label derivation) completes in ~13 minutes end-to-end.

### Parquet Output via Arrow

The final dataset (100+ scalar columns + LIST columns for sequences) is
assembled as a PyArrow Table and written directly — no pandas intermediate:

```python
import pyarrow as pa
import pyarrow.parquet as pq

# Build schema from scalar (numpy) + list (Python lists) columns
table = pa.table(column_dict, schema=schema)
pq.write_table(table, output_path, compression="snappy")
```

The 1M-customer dataset writes as a ~1.5 GB Snappy-compressed Parquet
file (105 raw columns; expands to 403 after Phase 0 feature engineering).
LIST columns (transaction sequences) are stored natively in Parquet's
nested type system, readable by DuckDB downstream without deserialization hacks.

This means the DuckDB → Arrow → Parquet chain runs end-to-end in both
directions: **generation** (numpy → Arrow → Parquet) and **consumption**
(Parquet → DuckDB SQL → Arrow → PyTorch).

---

## Pattern 7: Sequence Densification

`core/data/sequence_densifier.py` converts DuckDB LIST columns (produced by
aggregation queries) into flat, fixed-length feature vectors for model input,
using DuckDB-native operations throughout.

For datasets small enough to process in a single pass, the result table is
created directly:

```sql
CREATE OR REPLACE TABLE _densified_raw AS
SELECT
    passthrough_col,
    COALESCE(list_element(seq_col, len(seq_col) - 179), 0.0)::FLOAT AS seq_001,
    ...
    COALESCE(list_element(seq_col, len(seq_col) - 0  ), 0.0)::FLOAT AS seq_180
FROM raw_sequences
```

For large datasets (>500K rows by default), chunked processing uses
`ROW_NUMBER() OVER ()` to number rows, then `INSERT INTO` to batch-assemble
the result table:

```sql
-- Step 1: Add row numbers
CREATE OR REPLACE TABLE _numbered AS
SELECT *, ROW_NUMBER() OVER () AS _densify_rownum
FROM raw_sequences;

-- Step 2: First chunk creates the table
CREATE OR REPLACE TABLE _densified AS
SELECT ... FROM _numbered WHERE _densify_rownum > 0 AND _densify_rownum <= 500000;

-- Step 3: Subsequent chunks append
INSERT INTO _densified
SELECT ... FROM _numbered WHERE _densify_rownum > 500000 AND _densify_rownum <= 1000000;
```

Schema introspection uses `DESCRIBE` to identify passthrough (non-LIST)
columns automatically, without hard-coding column names:

```python
passthrough = conn.execute(
    f"SELECT column_name FROM (DESCRIBE SELECT * FROM {table}) "
    "WHERE column_type NOT LIKE 'LIST%'"
).fetchall()
```

This pattern generalises to any LIST column schema without changes to Python code.

---

## Production Features in QueryEngine

`core/data/query_engine.py` configures DuckDB as a production-grade engine with
several settings not visible in the per-pattern examples above.

**Resource limits** (config-driven via `query_engine.duckdb` in YAML):

```python
conn.execute(f"SET memory_limit='{memory_limit}'")  # default 4GB, env-overridable
conn.execute(f"SET threads={threads}")               # default 4
```

**S3/httpfs for cloud data access** — when `s3_region` or credentials are
configured, the engine installs `httpfs` and sets credentials so that
`read_parquet('s3://bucket/...')` works identically to local paths:

```python
conn.execute("INSTALL httpfs; LOAD httpfs;")
conn.execute(f"SET s3_region='{s3_region}'")
```

**Temp directory spillover** for datasets that exceed the memory limit:

```python
conn.execute(f"SET temp_directory='{temp_dir}'")
```

This means the same `QueryEngine` instance handles both local Parquet and S3
Parquet without code changes — only the YAML config differs between
on-premises and SageMaker runs.

---

## Schema Auto-Inspection in the Adapter

`adapters/santander_adapter.py` uses `DESCRIBE SELECT *` throughout to detect
column dtypes and build metadata automatically, rather than hard-coding column
lists:

```python
# Detect all columns with their types
col_meta = con.execute(
    f"SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM {_TBL})"
).fetchall()
```

This drives automatic routing of columns to normalizer stages (which columns
are binary, which are continuous, which are LIST types) based on the actual
schema of each loaded dataset — satisfying the config-driven requirement that
no column names be hard-coded in Python.

---

## Benchmarks

Measured on the AWS benchmark synthetic dataset (941K × 403 columns,
Phase 0 output from benchmark_v12) on the local development workstation
(RTX 4070, 64 GB RAM) — the same machine used for all development
before submitting to SageMaker. Production-scale benchmarks (12M × 734)
on the on-premises workstation (128 GB RAM) will be added when the
on-premises codebase is publicly released.
Reproducible with `scripts/benchmark_duckdb_vs_pandas.py`.

### Speed

| Operation | pandas | DuckDB | Speedup |
|---|---|---|---|
| Group-by aggregation (COUNT/SUM/AVG/STDDEV) | 17.8 s | 57.6 ms | **310×** |
| Filter + aggregate (temporal split) | 18.7 s | 52.7 ms | **355×** |
| 13-label derivation (full pipeline) | ~40 min | ~2 min | **~20×** |
| Feature matrix POSITIONAL JOIN (.arrow()) | 2.07 s | 1.08 s | **1.9×** |
| 1M-customer synth data generation | ~20 min | ~30 s | **~40× (vectorized numpy, not DuckDB SQL)** |
| Parquet full read (SELECT *) | 13.2 s | 176 s | **0.1× (pandas wins)** |
| Column selection (50 of 403 cols) | 243 ms | 908 ms | **0.3× (pandas wins)** |

The operations that dominate an ML pipeline — aggregation, filtering,
joining, label derivation — run **300×+ faster**.
Full materialization (`SELECT *`) is faster in pandas,
but the pipeline is designed to *never* do full materialization:
DuckDB handles all transforms, and `.df()` is called only at the
tensor construction boundary.

### Memory: The Real Story

| Operation | pandas peak RSS | DuckDB peak RSS | Ratio |
|---|---|---|---|
| Group-by aggregation | **15.2 GB** | **1.4 MB** | 10,857× |
| Filter + aggregate | **15.4 GB** | **1.1 MB** | 14,000× |
| Parquet full read | 15.1 GB | 14.4 GB | ~1× |

This is why DuckDB matters for this project. On the on-premises workstation
with 128 GB RAM, *every pandas operation* that touches the full dataset
(12M × 734) consumes far more than the 15 GB shown here at 1M scale.
With 16 label derivations, feature merging, normalization,
and tensor construction happening sequentially, pandas peak memory
leaves no headroom for PyTorch training that follows immediately after.

DuckDB performs the same aggregations in **1 MB**. This is not an optimization;
it is the difference between "the pipeline runs" and "the pipeline OOM-kills."

### What pandas Could Not Do on This Machine

- **941K × 403 sequential pipeline** — pandas requires the full DataFrame
  in memory for every operation. With 13 label derivations + feature merging
  + normalization happening in sequence, peak memory exceeds 40 GB.
  DuckDB's columnar engine keeps each operation under 2 MB, leaving headroom
  for the PyTorch training that follows in the same process.
- **24M-row sliding window sequences** — per-window `WHERE` filters over 24M
  rows. pandas loads the entire 24M DataFrame per window; DuckDB scans only
  matching rows.
- **Synthetic data generation** — the original Python-loop implementation
  required 20+ GB RAM for transaction sequence aggregation alone.
  Vectorized numpy reduced this to ~8 GB; DuckDB handles the final Parquet write
  via Arrow.
- **On-premises air-gapped environment** — Hive for storage but no Spark
  or Impala for processing. If it does not fit on the workstation (128 GB RAM),
  it does not run. DuckDB made it fit.

The same DuckDB-based pipeline was then ported to AWS SageMaker with zero
architecture changes — the SQL queries, `register()` patterns, and
`POSITIONAL JOIN` calls are identical between on-premises and cloud.
The on-premises codebase uses DuckDB in over 240 source files across
ingestion, feature engineering, model input, scoring, monitoring,
and serving — it is not a partial adoption but the sole data processing backend.

---

## Project Policy: pandas as Last Resort

The project's development guidelines (`CLAUDE.md`) enforce a strict
data backend hierarchy, born from the on-premises constraint:

```
cuDF (GPU)  →  DuckDB (CPU columnar)  →  pandas (≤10K rows only)
```

Specific rules:
- `pd.read_parquet()` is replaced by `duckdb.execute("SELECT * FROM 'file.parquet'")`.
- `df.groupby().apply(lambda)` is explicitly forbidden in code review.
- All aggregation and transformation go through SQL.
- `.df()` or `.fetchnumpy()` is called only at the tensor construction boundary.

The pandas fallback is retained for approximately 2 of 10 label derivation
types not yet ported to SQL, and for `<10K` row paths in unit tests.

---

## Key Takeaways

1. **DuckDB made an impossible pipeline possible.** The on-premises
   financial institution had Hive but no Spark or Impala.
   DuckDB replaced the entire post-Hive processing layer with
   `pip install duckdb` on a single workstation, enabling a production-scale
   ML pipeline (12M customers, 734 features) that pandas could not run
   on the same hardware.

2. **The real win is memory, not speed.** 310× on group-by is impressive,
   but 15 GB → 1 MB memory reduction is what kept the pipeline within
   128 GB RAM alongside PyTorch training on the on-premises workstation.
   On AWS, the same memory reduction translates directly to smaller
   (cheaper) instance types — RAM is the most expensive SageMaker dimension.

3. **POSITIONAL JOIN** keeps ML pipelines pandas-free. It replaces
   `pd.concat(axis=1)` with 1.9× speedup and Arrow-native output,
   maintaining consistency with the DuckDB-throughout philosophy.
   Memory is similar — the win is speed + no pandas dependency.

4. **Be honest about where pandas wins.** Full materialization is faster
   in pandas. Design your pipeline to minimize full materializations
   and maximize SQL-level operations.

5. **DuckDB pipelines port from on-prem to cloud with zero changes.**
   The same `register()` → SQL → `.df()` pattern runs identically
   on the air-gapped desktop and on AWS SageMaker.

---

*Built with DuckDB 1.2, Python 3.11, PyTorch 2.x. Full source at
[github.com/bluethestyle/aws_ple_for_financial](https://github.com/bluethestyle/aws_ple_for_financial).*
