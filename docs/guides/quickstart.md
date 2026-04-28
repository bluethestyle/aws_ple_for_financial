# Quickstart Guide

This guide walks a new user from clone to a running end-to-end pipeline in about 15 minutes, entirely on a local machine (no AWS required for the first run).

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.11 recommended |
| pip or conda | Package manager |
| Git | To clone the repository |
| CUDA toolkit | Optional — GPU experts (Temporal, HGCN) benefit from it; CPU fallback works |
| AWS account | Optional — needed only for SageMaker / S3 workflows |

---

## 1. Install

```bash
git clone <repo-url> aws_ple_for_financial
cd aws_ple_for_financial

python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

# Core install (CPU, no AWS SDK)
pip install -e ".[dev]"

# With AWS support
pip install -r requirements-aws.txt

# With GPU support (cuDF / CUDA)
pip install -r requirements-gpu.txt
```

Key packages:

| Package | Purpose |
|---|---|
| `torch` | PLE model, 7 expert networks, loss functions |
| `duckdb` | Primary DataFrame backend — fast columnar Parquet I/O |
| `lightgbm` | Distilled student model (CPU inference) |
| `pandas`, `numpy`, `scikit-learn` | Small-scale utilities, final tensor conversion |
| `pyyaml` | Config loading |
| `boto3` | AWS SDK (optional) |

---

## 2. Generate Benchmark Data

The platform ships with a synthetic benchmark generator that produces a
realistic 1 M-customer financial dataset.

```bash
PYTHONPATH=. python scripts/generate_benchmark_data.py --n-customers 50000
# Use 1000000 for the full benchmark; 50000 is faster for a first run.
```

Output lands in `data/benchmark/`.

---

## 3. Phase 0 — Feature Engineering

Phase 0 reads raw Parquet files, runs 10 feature generators (TDA, HMM, HGCN,
Mamba, etc.), applies 3-stage normalization, and writes training-ready tensors.

DuckDB is the data backend throughout this phase — no pandas for large-scale
loads.

The adapter only converts raw data to a standardized DataFrame (CLAUDE.md §1.2);
Phase 0 as a whole is driven by `PipelineRunner` (`core/pipeline/runner.py`),
which runs preprocessing → feature generation → label derivation →
3-stage normalization → tensor save. Trigger Phase 0 through the training
entry point with `--phase0-only`, or let `train.py` execute it implicitly
when `outputs/phase0/*.pt` is missing:

```bash
PYTHONPATH=. python containers/training/train.py \
  --config  configs/pipeline.yaml \
  --dataset configs/datasets/santander.yaml \
  --phase0-only
```

Phase 0 produces:

```
outputs/phase0/
  train.pt / val.pt / test.pt   # Training-ready tensors
  feature_stats.json            # Per-feature statistics (zero-variance, NaN %)
  label_stats.json              # Class balance + positive rates
  feature_schema.json           # Column names, group ranges, scaler state
```

Before continuing, verify the output:

```bash
# Check for zero-variance columns and label distribution
python -c "
import json, pathlib
stats = json.loads(pathlib.Path('outputs/phase0/feature_stats.json').read_text())
labels = json.loads(pathlib.Path('outputs/phase0/label_stats.json').read_text())
print('Features after Phase 0:', stats.get('n_features'))   # expect ~403
print('Tasks:', list(labels.keys()))                         # expect 13 tasks
"
```

Expected: ~349 input features, ~403 after Phase 0 (log-transform copies added
by 3-stage normalization), 13 tasks.

---

## 4. Train

```bash
# Canonical split-config invocation (CLAUDE.md §1.1).
PYTHONPATH=. python containers/training/train.py \
  --config  configs/pipeline.yaml \
  --dataset configs/datasets/santander.yaml
```

To override hyperparameters without editing YAML:

```bash
PYTHONPATH=. python containers/training/train.py \
  --config  configs/pipeline.yaml \
  --dataset configs/datasets/santander.yaml \
  --hp '{"training": {"batch_size": 5632, "lr": 0.0005, "amp_enabled": true}}'
```

Default training settings (merged from `pipeline.yaml` + `datasets/santander.yaml`):

| Parameter | Value |
|---|---|
| `batch_size` | 5632 |
| `lr` | 0.0005 |
| `AMP (FP16)` | enabled |
| `Tasks` | 13 |
| `Experts` | 7 (DeepFM, Temporal, HGCN, PersLay, Causal, LightGCN, OT) |

Training logs GPU memory, data shape/dtype/NaN rates, label distribution, and
feature schema before the first epoch.

---

## 5. Evaluate

```bash
PYTHONPATH=. python scripts/eval_checkpoint.py \
  --checkpoint outputs/checkpoints/best.pt \
  --data-dir   outputs/phase0
```

Metrics are reported by task type — AUC for binary tasks, F1-macro for
multiclass, MAE for regression — and written to `outputs/eval_metrics.json`.

---

## 6. Run a Local Ablation (optional)

To reproduce the benchmark ablation table locally (no SageMaker):

```bash
PYTHONPATH=. python scripts/run_local_ablation.py
```

Results land in `outputs/ablation/`. The orchestrator auto-skips already-
completed scenarios based on `pipeline_state.json`.

---

## 7. Run on AWS (optional)

A full AWS submission consists of (a) one-time Mamba image build, (b)
GPU Mamba precompute → cached `embedding.parquet`, (c) CPU Phase 0,
(d) training. (a) and (b) are reused across many Phase 0 / training
runs. End-to-end commands:

```bash
# (a) One-time: build the Mamba GPU precompute image (CodeBuild, ~5 min, ~$0.10)
bash scripts/build_mamba_image_codebuild.sh
#     Then set aws.mamba_image_uri in BOTH
#       configs/pipeline.yaml
#       configs/santander/pipeline.yaml

# (b) Mamba precompute on g4dn.xlarge spot (~8.6 min, ~$0.026)
python scripts/submit_pipeline.py \
  --config configs/santander/pipeline.yaml \
  --mamba-precompute
#     Extract embedding.parquet from the resulting model.tar.gz and
#     upload to s3://aiops-ple-financial/santander_ple/mamba/embedding.parquet
#     Then set in configs/santander/feature_groups.yaml::mamba_temporal:
#       enabled: true
#       generator_params:
#         cached_embedding_uri: s3://.../mamba/embedding.parquet

# (c) Phase 0 on m5.4xlarge spot (~7 min, ~$0.04)
python scripts/submit_pipeline.py \
  --config configs/santander/pipeline.yaml \
  --phase0-only
#     MambaFeatureGenerator detects cached_embedding_uri and JOINs the
#     cached parquet via DuckDB+boto3 instead of running the GPU model.
#     Extract Phase 0 model.tar.gz contents and upload to
#       s3://aiops-ple-financial/santander_ple/phase0/<job>/extracted/

# (d) Training mode against the Phase 0 output prefix
python scripts/submit_pipeline.py \
  --mode training \
  --features-uri s3://aiops-ple-financial/santander_ple/phase0/<job>/extracted/
```

Use `--dry-run` with any of the above to print the launch config without
submitting (works for `--mamba-precompute` too). For the contiguous
9-stage OOM case, replace `--phase0-only` with `--phase0-split`. See
[deployment.md](./deployment.md) for instance-type / cost / IAM details.

---

## Configuration

Everything is config-driven (CLAUDE.md §1.1). Three YAML files form a
split-config and are deep-merged at runtime by `load_merged_config()`:

| Layer | File | Controls |
|---|---|---|
| 1 | `configs/pipeline.yaml` | Common model architecture, training HP, distillation, AWS, compliance, monitoring — shared across datasets |
| 2 | `configs/datasets/{name}.yaml` | Dataset-specific tasks, label derivation rules, adapter, data source, ablation scenarios |
| 3 | `configs/{name}/feature_groups.yaml` | Feature group definitions, generator input filters, expert routing (`target_experts`) — referenced via `feature_groups_file` in layer 2 |

Edit only these files to change tasks, hyperparameters, or expert assignments.
Do **not** hardcode values in Python scripts. See
[configuration_reference.md](./configuration_reference.md) for the full schema.

### Adapting to a new dataset

1. Copy the dataset template: `cp configs/datasets/example.yaml configs/datasets/<my_dataset>.yaml` and fill in tasks, labels, adapter, `data.source`, and `feature_groups_file`.
2. Copy an existing feature-groups file: `cp -r configs/santander configs/<my_dataset>` and edit `configs/<my_dataset>/feature_groups.yaml` for your columns + generators.
3. Implement a data adapter in `adapters/<my_dataset>_adapter.py` (copy `adapters/santander_adapter.py` as a starting point — adapter converts raw data to a standardized DataFrame; no feature engineering or label derivation in the adapter).
4. Run with the split-config invocation:
   ```bash
   PYTHONPATH=. python containers/training/train.py \
     --config  configs/pipeline.yaml \
     --dataset configs/datasets/<my_dataset>.yaml
   ```

---

## Architecture at a Glance

```
Customer Data (Parquet via DuckDB)
    |
    v
[Phase 0]  10 Feature Generators — ~349 in → ~403 out
           TDA · HGCN · Mamba · HMM · Chemical Kinetics · SIR · ...
    |
    v
[Train]    PLE + 7 Heterogeneous Experts + 13 Tasks
           DeepFM | Temporal | HGCN | PersLay | LightGCN | Causal | OT
           batch_size=5632 · lr=0.0005 · AMP FP16
    |
    v
[Distill]  Knowledge Distillation → 13 LightGBM students (CPU inference)
    |
    v
[Serve]    AWS Lambda · 3 serving agents · 2 ops/audit agents
```

The 13 tasks span binary classification (product holding, churn, NBA),
multiclass classification (risk tier, channel preference), and regression
(spend volume, CLV).

---

## Project Structure

```
aws_ple_for_financial/
  configs/santander/        pipeline.yaml + feature_groups.yaml (config-driven)
  core/model/ple/           PLE architecture (CGC gate, adaTT, gating)
  core/model/experts/       7 expert implementations
  core/feature/generators/  10 feature generators
  core/pipeline/            Phase 0: preprocessing, normalization, label derivation
  core/training/            Trainer, evaluator, callbacks
  core/recommendation/      Scoring, reason generation, compliance
  core/agent/               Ops/Audit agents (5-agent architecture)
  adapters/                 Data adapters (santander, ealtman2019)
  scripts/                  CLI scripts (generate, train, eval, ablation)
  containers/training/      train.py — SageMaker / local entry point
  aws/                      SageMaker, S3, Step Functions wrappers
  docs/                     Design docs, technical references
  paper/                    Research papers (Typst)
```

---

## Troubleshooting

**`ImportError: No module named 'duckdb'`**
DuckDB is the primary data backend. Install with `pip install duckdb>=0.10`.

**CUDA not available**
GPU experts fall back to CPU automatically. For GPU: install CUDA toolkit and
`pip install torch --index-url https://download.pytorch.org/whl/cu121`.

**`boto3` not installed**
AWS features (S3, SageMaker) require `pip install boto3`. Local mode works
without it.

**Windows sleep killing the process during overnight training**
Use `scripts/run_local_ablation.py`, which calls `SetThreadExecutionState` to
prevent Windows from suspending the process.

---

## Next Steps

| Goal | Where to look |
|---|---|
| Full configuration reference | `docs/guides/configuration_reference.md` |
| Feature engineering details | `docs/guides/feature_engineering.md` |
| SageMaker deployment | `docs/guides/deployment.md` |
| Ops/Audit agent design | `docs/design/11_ops_audit_agent.md` |
| DuckDB pipeline case study | `docs/duckdb-case-study.md` |
