# Getting Started with the AWS PLE Platform

This guide walks you through installing the platform, running the synthetic
example end-to-end, and understanding the output.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.11 recommended |
| pip or conda | Package manager |
| Git | To clone the repository |
| AWS account | **Optional** -- local mode works without AWS |
| CUDA toolkit | **Optional** -- GPU experts (Mamba, Temporal) benefit from it |

## Installation

### 1. Clone the repository

```bash
git clone <repo-url> aws_ple_for_financial
cd aws_ple_for_financial
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
# Core (CPU-only, no AWS SDK)
pip install -r requirements.txt

# With AWS support (SageMaker, S3, DynamoDB)
pip install -r requirements-aws.txt

# With GPU support (cuDF, CUDA)
pip install -r requirements-gpu.txt
```

Key packages installed:

| Package | Purpose |
|---|---|
| `torch` | PLE model, experts, loss functions |
| `lightgbm` | LGBM student model for distillation |
| `duckdb` | Default DataFrame backend (fast Parquet I/O) |
| `pandas`, `numpy`, `scikit-learn` | Data manipulation, transformers |
| `pyyaml` | Configuration loading |
| `boto3` | AWS SDK (optional) |

---

## Running the Synthetic Example

The platform ships with a self-contained example that generates synthetic data,
trains a multi-task model, and produces predictions -- all without AWS.

### Step 1: Run the example

```bash
python examples/synthetic/run.py
```

You can control the data size and output directory:

```bash
python examples/synthetic/run.py --n 50000 --output outputs/synthetic/
```

### Step 2: What happens behind the scenes

The script executes the following pipeline:

```
                                    examples/synthetic/run.py
                                             |
        1. Generate synthetic data           |
           (user_age, item_price, ...)       |
                    |                        |
        2. Save to Parquet                   |
                    |                        |
        3. Build PipelineConfig              |
           - 2 tasks: click (binary),        |
             convert (binary)                |
           - 4 numeric + 3 categorical       |
             features                        |
           - LGBM model architecture         |
                    |                        |
        4. PipelineRunner.run(mode="local")  |
           - Feature pipeline                |
             (StandardScaler, LabelEncoder)  |
           - Train/val split (80/10/10)      |
           - LGBM multi-task training        |
           - Evaluation metrics              |
                    |                        |
        5. Output results + model artifacts  |
```

### Step 3: Understand the output

After running, the `outputs/synthetic/` directory contains:

```
outputs/synthetic/
  data.parquet                # Generated synthetic data
  model/                      # Trained LGBM model artifacts
    click.model               # Click prediction model
    convert.model             # Conversion prediction model
  metrics/                    # Evaluation metrics
    evaluation.json           # AUC, accuracy, etc.
  pipeline/                   # Fitted pipeline artifacts
    metadata.json             # Pipeline metadata
```

Console output shows:

```
INFO  Generating 10,000 synthetic samples...
INFO  Data saved: outputs/synthetic/data.parquet | shape=(10000, 9)
INFO    click rate   : 0.523
INFO    convert rate : 0.181
INFO  Pipeline result: {...}
```

### Step 4: Examine the configuration

The synthetic example uses this configuration (from `examples/synthetic/run.py`):

```python
PipelineConfig(
    task_name="synthetic_multitask",
    tasks=[
        TaskSpec(name="click",   type="binary", loss="focal", loss_weight=1.0, label_col="clicked"),
        TaskSpec(name="convert", type="binary", loss="focal", loss_weight=1.5, label_col="converted"),
    ],
    data=DataSpec(source="outputs/synthetic/data.parquet", format="parquet"),
    features=FeatureSpec(
        numeric=["user_age", "item_price", "item_popularity", "days_since_last_visit"],
        categorical=["user_segment", "item_category", "platform"],
    ),
    model=ModelSpec(architecture="lgbm"),
    training=TrainingSpec(epochs=20, seed=42),
)
```

The YAML equivalent (in `configs/examples/multitask_binary.yaml`) demonstrates
the same configuration in file form -- useful for production workflows.

---

## Configuration

### Split-config pattern

The platform separates configuration into two layers:

| File | Purpose |
|---|---|
| `configs/pipeline.yaml` | Common defaults: model architecture, training HP, distillation, AWS |
| `configs/datasets/<name>.yaml` | Dataset-specific: tasks, labels, adapter, sequences, ablation |

Both files are deep-merged at runtime — dataset keys win on collision. This means you only describe _what is different_ in your dataset file; all model and training defaults come from `pipeline.yaml` automatically.

```
configs/
├── pipeline.yaml                 ← common (model, training, distillation, aws)
├── datasets/
│   ├── santander.yaml            ← benchmark dataset
│   └── example.yaml              ← template for new users
├── santander/
│   ├── feature_groups.yaml       ← feature group definitions
│   └── ...
└── financial/                    ← on-prem operation configs
```

### Creating a dataset config for your data

1. Copy the template:

```bash
cp configs/datasets/example.yaml configs/datasets/my_bank.yaml
```

2. Fill in the required sections (replace every `<PLACEHOLDER>`):
   - `dataset.name` and `adapter`
   - `tasks`: list of prediction targets with `label_col`, `type`, `loss`
   - `data.source`: path to your Parquet file or S3 URI
   - `features.numeric` / `features.categorical`: column names
   - `feature_groups_file`: path to your feature group YAML

3. Add a feature groups file at `configs/my_bank/feature_groups.yaml`
   (copy `configs/santander/feature_groups.yaml` as a starting point).

4. Implement a data adapter in `src/adapters/my_bank.py`
   (copy `src/adapters/santander.py` and adjust raw → standardized DataFrame).

### CLI usage

Pass both files to `train.py` or the orchestrator:

```bash
# Split-config pattern (recommended)
python containers/training/train.py \
  --config configs/pipeline.yaml \
  --dataset configs/datasets/my_bank.yaml

# Single-file pattern (backward compatible)
python containers/training/train.py \
  --config configs/santander/pipeline.yaml
```

For SageMaker, pass both as hyperparameters:

```python
hyperparameters = {
    "config": "configs/pipeline.yaml",
    "dataset_config": "configs/datasets/my_bank.yaml",
}
```

### Backward compatibility

Single-file configs (legacy pattern) still work. If `dataset_config` is not
provided, `train.py` loads the single config file as before. No migration is
required for existing setups.

---

## Next Steps

### Use your own data

1. Prepare your data as a Parquet file.
2. Copy `configs/datasets/example.yaml` → `configs/datasets/my_bank.yaml` and fill in tasks, features, and data source.
3. Run with `--config configs/pipeline.yaml --dataset configs/datasets/my_bank.yaml`.

### Add feature engineering

Feature groups let you generate new features (TDA, HMM, graph embeddings) or
transform existing columns (scaling, encoding). See the
[Feature Engineering Guide](feature_engineering.md).

### Use the PLE neural model instead of LGBM

Change `model.architecture` from `"lgbm"` to `"ple"` and configure experts.
See the [Model Architecture Guide](model_architecture.md).

### Deploy to AWS

The platform supports SageMaker Training (with Spot instances), Lambda and ECS
serving, and DynamoDB feature stores. See the
[Deployment Guide](deployment.md).

### Extend the platform

Add custom generators, transformers, experts, task heads, scorers, or filters
via the plugin registry pattern. See `configs/` directory for plugin configuration examples.

### Ops / Audit Agents

After pipeline execution completes, two autonomous diagnostic agents run asynchronously:

- **Ops Agent (OpsAgent)**: Monitors 7 checkpoints (from ingestion through A/B testing), analyzes cascading effects between anomalies, and generates reports in `finding + likely_cause + suggested_action` format.
- **Audit Agent (AuditAgent)**: Evaluates fairness, recommendation-reason quality, and regulatory compliance across 5 viewpoints, performing cross-protected-attribute analysis and 3-tier recommendation-reason quality verification.

Agent configuration: `configs/financial/agent.yaml`
Detailed design: `docs/design/11_ops_audit_agent.md`

---

## Project Structure Overview

```
aws_ple_for_financial/
  configs/                        # YAML configuration files
    pipeline.yaml                 # Common: model, training, distillation, aws
    datasets/                     # Dataset-specific configs
      santander.yaml              # Santander benchmark
      example.yaml                # Template for new datasets
    santander/                    # Santander feature definitions
      feature_groups.yaml         # Feature group definitions
    financial/                    # On-prem operation configs
    recommendation.yaml           # Scoring, filtering, reasons
    monitoring.yaml               # Fairness, drift, incidents
  core/                           # Core platform code
    feature/                      # Feature engineering layer
      generators/                 # Built-in generators (TDA, HMM, ...)
      transformers.py             # Built-in transformers
      group.py                    # FeatureGroupConfig + Registry
      group_pipeline.py           # FeatureGroupPipeline orchestrator
    model/                        # Model layer
      ple/                        # PLE architecture (CGC, adaTT, gating)
      experts/                    # Expert networks (DeepFM, Mamba, ...)
      lgbm/                       # LightGBM student model
    task/                         # Task heads (binary, regression, ...)
    recommendation/               # Scoring + filtering + reasons
    serving/                      # Lambda + ECS serving
    monitoring/                   # Fairness, drift, herding, incidents
    training/                     # Training loop, 2-phase, distillation
    data/                         # DataFrame backend (DuckDB/cuDF/pandas)
  aws/                            # AWS SDK wrappers (S3, SageMaker, Athena)
  containers/                     # Docker entry points
    training/train.py             # SageMaker training container
    inference/lambda_handler.py   # Lambda handler
    inference/app.py              # ECS FastAPI app
  examples/                       # Runnable examples
  tests/                          # Test suite
  docs/                           # Documentation
```

---

## Troubleshooting

**ImportError: No module named 'duckdb'**
DuckDB is the default DataFrame backend. Install it with `pip install duckdb`.
Alternatively, the platform falls back to pandas automatically.

**CUDA not available**
GPU experts (Mamba, Temporal Ensemble) will use CPU as fallback. Performance
will be slower but functionally identical. Install CUDA toolkit and
`pip install torch --index-url https://download.pytorch.org/whl/cu121` for
GPU support.

**boto3 not installed**
AWS features (S3, SageMaker, DynamoDB) require `pip install boto3`. Local mode
works without it.
