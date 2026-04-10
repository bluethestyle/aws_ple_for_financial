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

## Next Steps

### Use your own data

1. Prepare your data as a Parquet file.
2. Copy `configs/examples/multitask_binary.yaml` and edit:
   - `tasks`: define your prediction targets
   - `features.numeric` / `features.categorical`: list your columns
   - `data.source`: point to your file or S3 path
3. Run with `PipelineRunner`.

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
via the plugin registry pattern. See the
[Plugin Development Guide](plugin_development.md).

### 운영/감사 에이전트

파이프라인 실행 완료 후, 2개의 자율 진단 에이전트가 비동기로 작동합니다:

- **운영 에이전트 (OpsAgent)**: 7개 체크포인트(인제스천~A/B테스트)를 모니터링하고, 이상 징후 간 연쇄 영향을 분석하여 `finding + likely_cause + suggested_action` 형식으로 리포트를 생성합니다.
- **감사 에이전트 (AuditAgent)**: 공정성, 추천사유 품질, 규제 적합성을 5개 관점에서 점검하고, 교차 보호속성 분석 및 3-Tier 추천사유 품질 검증을 수행합니다.

에이전트 설정: `configs/financial/agent.yaml`
상세 설계: `docs/design/11_ops_audit_agent.md`

---

## Project Structure Overview

```
aws_ple_for_financial/
  configs/                        # YAML configuration files
    feature_groups.yaml           # Feature group definitions
    recommendation.yaml           # Scoring, filtering, reasons
    monitoring.yaml               # Fairness, drift, incidents
    examples/                     # Example pipeline configs
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
