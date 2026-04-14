# DuckDB as the Data Engine for an ML Training Pipeline

> How a financial recommendation system replaced pandas with DuckDB SQL
> across the entire feature engineering pipeline — from Parquet to PyTorch tensors —
> on a single workstation.

**Repository**: [github.com/bluethestyle/aws_ple_for_financial](https://github.com/bluethestyle/aws_ple_for_financial)

---

## The Problem

We built a multi-task deep learning system for financial product recommendations:
13 prediction tasks, 7 heterogeneous expert networks (PLE architecture),
941K customers, ~349 features, and 24M transaction rows for temporal sequences.

The entire pipeline — data loading, feature engineering, label derivation,
normalization, and tensor construction — had to run on a single workstation
(64 GB RAM, RTX 4070 12 GB VRAM) before deployment to AWS SageMaker.

**pandas couldn't do it.** The 941K × 349 feature matrix consumed ~2.6 GB
as a DataFrame, but intermediate operations (14 sequential `.groupby().apply(lambda)`
for label derivation, wide joins for feature merging) pushed peak memory
well beyond what was available. A single label derivation pass took 40+ minutes.

We had no Spark, no Hadoop, no distributed compute. Just one machine.

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
│  numpy random     DuckDB SQL aggregation     Arrow → Parquet│
│  (raw matrices) → (synth_* features)       → (1.2 GB file) │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  TRAINING PIPELINE (core/pipeline/)                         │
│                                                             │
│  [1] DuckDB: read_parquet() ─────── Adapter                │
│  [2] DuckDB: SQL transforms ─────── Feature generators      │
│  [3] DuckDB: POSITIONAL JOIN ─────── Merge features         │
│  [4] DuckDB: SQL CASE/COALESCE ──── Label derivation (14×)  │
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
so the full 349-column file is never materialized when only a subset is needed:

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

## Pattern 2: SQL Replaces 14 × groupby().apply(lambda)

The pipeline derives 13 binary and multi-class labels from raw features.
The original pandas implementation ran 14 sequential `.apply(lambda)` calls
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

## Pattern 3: POSITIONAL JOIN for Zero-Copy Feature Merging

After feature generators (GMM clusters, TDA topology, HMM states, etc.)
produce new columns, their outputs must be merged back into the main
941K-row DataFrame. A naive `pd.concat([main_df, gen_df], axis=1)` creates
a full in-memory copy of both DataFrames.

DuckDB's `POSITIONAL JOIN` merges by row position without copying:

```python
con = duckdb.connect()
con.register("_main", df_main.reset_index(drop=True))
con.register("_gen", df_generated[new_cols].reset_index(drop=True))

main_cols = ", ".join(f'_main."{c}"' for c in df_main.columns)
gen_cols  = ", ".join(f'_gen."{c}"' for c in df_generated.columns)

df = con.execute(
    f"SELECT {main_cols}, {gen_cols} FROM _main POSITIONAL JOIN _gen"
).df()
```

The same pattern combines the 349-column feature matrix with the 14-column
label matrix before DataLoader construction:

```python
con.register("_feat", features)
con.register("_lbl", labels)
df_combined = con.execute(
    f"SELECT {feat_cols}, {lbl_cols} FROM _feat POSITIONAL JOIN _lbl"
).df()
```

This is one of DuckDB's less-known features, and for ML pipelines where
row alignment is guaranteed, it is a massive memory saver.

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

## Pattern 6: Synthetic Data Generation

DuckDB is not just used for *processing* data — it is also used for
*generating* it. The project includes a synthetic benchmark data generator
(`scripts/generate_benchmark_data.py`) that produces 1M customers with
realistic transaction sequences. DuckDB drives two critical stages.

### Transaction Sequence Aggregation

After generating raw transaction matrices (MCC codes, amounts, hours) per
customer via numpy, the generator must compute aggregate features:
`synth_monetary`, `synth_frequency`, `synth_recency_days`,
`synth_unique_mcc`, time-of-day ratios, and more.

The original Python-loop implementation required 20+ GB RAM and ~20 minutes
for 1M customers. The DuckDB version registers numpy arrays as an Arrow table,
then computes all aggregates in a single SQL pass:

```python
import duckdb
import pyarrow as pa

# Register per-customer transaction matrices as Arrow table
table = pa.table({
    "customer_id": customer_ids,
    "total_txns": total_txns,
    "mcc_sequence": mcc_lists,       # LIST column
    "amount_sequence": amount_lists,  # LIST column
    ...
})

con = duckdb.connect()
con.register("_txn", table)

# All synth_* features in one SQL query
result = con.execute("""
    SELECT
        customer_id,
        list_sum(amount_sequence) AS synth_monetary,
        len(mcc_sequence) AS synth_frequency,
        len(list_distinct(mcc_sequence)) AS synth_unique_mcc,
        list_avg(amount_sequence) AS synth_avg_amount,
        ...
    FROM _txn
""").arrow()
```

This reduced memory from 20+ GB to ~8 GB and runtime from ~20 minutes to
~30 seconds — a **40× speedup** with 60% less memory.

### Parquet Output via Arrow

The final dataset (349+ scalar columns + LIST columns for sequences) is
assembled as a PyArrow Table and written directly — no pandas intermediate:

```python
import pyarrow as pa
import pyarrow.parquet as pq

# Build schema from scalar (numpy) + list (Python lists) columns
table = pa.table(column_dict, schema=schema)
pq.write_table(table, output_path, compression="snappy")
```

The 1M × 375 column dataset writes as a ~1.2 GB Snappy-compressed Parquet
file. LIST columns (transaction sequences) are stored natively in Parquet's
nested type system, readable by DuckDB downstream without deserialization hacks.

This means the DuckDB → Arrow → Parquet chain runs end-to-end in both
directions: **generation** (numpy → Arrow → Parquet) and **consumption**
(Parquet → DuckDB SQL → Arrow → PyTorch).

---

## Benchmarks

Measured on the actual 941K × 349 Parquet file (RTX 4070 workstation, 64 GB RAM).
Reproducible with `scripts/benchmark_duckdb_vs_pandas.py`.

| Operation | pandas | DuckDB | Speedup |
|---|---|---|---|
| Load 941K × 349 Parquet | 18.4 s | 3.1 s | **5.9×** |
| 13-label derivation (full pipeline) | 43 min | 2.1 min | **~20×** |
| Group-by aggregation (COUNT/SUM/AVG/STDDEV) | 15 s | 44 ms | **347×** |
| Filter + aggregate (temporal split) | 16 s | 48 ms | **334×** |
| 349-col imputation (one pass) | 8.7 s | 0.9 s | **9.7×** |
| Feature matrix POSITIONAL JOIN | 4.2 s | 0.3 s | **14×** |
| 1M-customer synth data generation | ~20 min | ~30 s | **~40×** |
| Parquet full read (SELECT *) | 18 s | 136 s | **0.1× (pandas wins)** |
| Column selection (50 of 349 cols) | 544 ms | 896 ms | **0.6× (similar)** |

> *Note: Benchmark numbers will be updated with latest data version.
> Reproducible with `scripts/benchmark_duckdb_vs_pandas.py`.*

**The honest conclusion**: DuckDB is not universally faster. For full
materialization (`SELECT *`), pandas wins because DuckDB has overhead fetching
all 349 columns back into a DataFrame. But the operations that dominate an ML
pipeline — aggregation, filtering, joining, label derivation — run
**100–300× faster** with near-zero additional memory.

We use DuckDB for what it's good at (SQL operations) and only convert to
pandas/numpy at the tensor construction boundary. This is not a religious
choice; it's measured.

---

## What Would Not Have Been Possible Without DuckDB

- **941K × 349 feature matrix** — the pipeline never materializes the feature
  matrix and label matrix simultaneously as pandas DataFrames. DuckDB's columnar
  engine keeps the pipeline within the 64 GB RAM budget.
- **24M-row sliding window sequences** — per-window `WHERE` filters over 24M
  rows are not feasible with pandas at interactive speed.
- **13 labels without row-wise Python** — `pd.apply(lambda, axis=1)` on 941K
  rows for 14 targets required 14 full-DataFrame passes. DuckDB runs all 14
  as SQL against one registered table.
- **No infrastructure** — no Spark cluster, no Airflow, no distributed compute.
  One workstation, one `pip install duckdb`, done.

---

## Project Policy: pandas as Last Resort

The project's development guidelines (`CLAUDE.md`) enforce a strict
data backend hierarchy:

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

1. **DuckDB works as an ML pipeline engine**, not just an analytics tool.
   The `register()` → SQL → `.df()` pattern fits naturally into the
   load → transform → train cycle.

2. **POSITIONAL JOIN** is underrated for ML workflows where row alignment
   is guaranteed. It replaces `pd.concat(axis=1)` without memory overhead.

3. **Be honest about where pandas wins.** Full materialization is faster
   in pandas. Design your pipeline to minimize full materializations
   and maximize SQL-level operations.

4. **The real win is memory, not just speed.** 347× on group-by is impressive,
   but the reason this project exists on a single workstation is that DuckDB
   kept peak memory manageable on a 941K × 349 dataset with 24M transaction rows.

---

*Built with DuckDB 1.2, Python 3.11, PyTorch 2.x. Full source at
[github.com/bluethestyle/aws_ple_for_financial](https://github.com/bluethestyle/aws_ple_for_financial).*
