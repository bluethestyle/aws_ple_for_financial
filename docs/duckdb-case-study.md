# DuckDB as the Data Engine for an ML Training Pipeline

> How a financial recommendation system replaced pandas with DuckDB SQL
> across the entire feature engineering pipeline — from Parquet to PyTorch tensors —
> on a single workstation.

**Repository**: [github.com/bluethestyle/aws_ple_for_financial](https://github.com/bluethestyle/aws_ple_for_financial)

---

## The Problem

This project is the AWS cloud extension of an on-premises financial
recommendation system running inside a Korean public financial institution.
The on-premises environment is extreme: an air-gapped network, a desktop PC
in a server room without dedicated cooling, an RTX 4070 (12 GB VRAM),
and 64 GB RAM. No Spark, no Hadoop, no distributed compute.
Requesting a GPU cluster in a financial institution is not a technical
problem — it is a budget approval problem that does not get approved.

The system processes 13 prediction tasks, 7 heterogeneous expert networks
(PLE architecture), 941K customers, ~403 features after Phase 0
feature engineering, and 24M transaction rows for temporal sequences.

**With pandas, this pipeline would not exist.**
The 941K × 403 feature matrix consumed ~3 GB as a DataFrame, but intermediate
operations — 13 sequential `.groupby().apply(lambda)` for label derivation,
wide joins for feature merging, sliding windows over 24M rows — pushed peak
memory well beyond 64 GB. A single label derivation pass took 40+ minutes.
On the on-premises desktop, pandas OOM-killed the process before Phase 0
could complete.

The question was not "how do we make it faster?" but
"how do we make it *possible* on this machine?"

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

The same pattern combines the 403-column feature matrix with the 13-column
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

The DuckDB-based aggregation step alone reduced from ~20 minutes to ~30 seconds
(**40× speedup**) with memory dropping from 20+ GB to ~8 GB.
Full data generation (1M customers including all 6 persona types,
transaction sequences, and label derivation) completes in ~13 minutes end-to-end.

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

## Benchmarks

Measured on the Phase 0 output (941K × 403 columns, benchmark_v12)
on the development workstation (RTX 4070, 64 GB RAM).
Reproducible with `scripts/benchmark_duckdb_vs_pandas.py`.

### Speed

| Operation | pandas | DuckDB | Speedup |
|---|---|---|---|
| Group-by aggregation (COUNT/SUM/AVG/STDDEV) | 17.8 s | 57.6 ms | **310×** |
| Filter + aggregate (temporal split) | 18.7 s | 52.7 ms | **355×** |
| 13-label derivation (full pipeline) | ~40 min | ~2 min | **~20×** |
| Feature matrix POSITIONAL JOIN | ~4 s | ~0.3 s | **~14×** |
| 1M-customer synth data generation | ~20 min | ~30 s | **~40×** |
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

This is why DuckDB matters for this project. On the on-premises desktop
with 64 GB RAM, *every pandas operation* that touches the full dataset
consumes 15 GB. With 13 label derivations, feature merging, normalization,
and tensor construction happening sequentially, pandas peak memory easily
exceeds 40 GB — leaving no headroom for PyTorch training that follows
immediately after.

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
  DuckDB SQL aggregation reduced this to ~8 GB.
- **On-premises air-gapped environment** — no Spark cluster, no Airflow,
  no cloud fallback. If it does not fit in 64 GB RAM on a single desktop PC,
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
   financial institution could not procure a Spark cluster or cloud access.
   DuckDB replaced the entire data processing layer with `pip install duckdb`
   on a single desktop PC, enabling a production-scale ML pipeline that
   pandas could not run on the same hardware.

2. **The real win is memory, not speed.** 310× on group-by is impressive,
   but 15 GB → 1 MB memory reduction is what kept the pipeline within
   64 GB RAM alongside PyTorch training.

3. **POSITIONAL JOIN** is underrated for ML workflows where row alignment
   is guaranteed. It replaces `pd.concat(axis=1)` without memory overhead.

4. **Be honest about where pandas wins.** Full materialization is faster
   in pandas. Design your pipeline to minimize full materializations
   and maximize SQL-level operations.

5. **DuckDB pipelines port from on-prem to cloud with zero changes.**
   The same `register()` → SQL → `.df()` pattern runs identically
   on the air-gapped desktop and on AWS SageMaker.

---

*Built with DuckDB 1.2, Python 3.11, PyTorch 2.x. Full source at
[github.com/bluethestyle/aws_ple_for_financial](https://github.com/bluethestyle/aws_ple_for_financial).*
