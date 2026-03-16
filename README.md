# AWS PLE Platform

Modular multi-task learning platform built on **Progressive Layered Extraction (PLE)** and **LightGBM**, fully deployable on AWS with pay-as-you-go compute.

## Key Features

- **Domain-agnostic**: Define any ML problem via YAML config — no code changes needed for new tasks
- **Multi-task learning**: PLE architecture handles binary, multiclass, regression, and ranking tasks simultaneously
- **AWS-native**: S3 data lake → SageMaker Training (Spot) → ECS Fargate serving
- **Pay-per-use**: No always-on servers; compute spins up for training, saves results to S3, then terminates
- **Pluggable**: Custom data sources, feature transformers, and task heads via plugin registry

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Run a local example (no AWS needed)
python examples/synthetic/run.py --n 10000 --output outputs/synthetic/

# 3. Launch a SageMaker training job (requires AWS credentials)
./scripts/start_training.sh --config configs/examples/multitask_binary.yaml
```

## Architecture

```
Data (S3)
  └─► Feature Pipeline (SageMaker Processing)
        └─► PLE / LGBM Training (SageMaker Training Job — Spot)
              └─► Model Registry (S3 + SageMaker Model)
                    └─► Serving (ECS Fargate / SageMaker Serverless)
```

Orchestration via **AWS Step Functions** (pay-per-transition, no always-on Airflow cost).

## Defining a New Problem

```yaml
# configs/my_problem.yaml
task_name: my_churn_model
tasks:
  - name: churn
    type: binary
    loss: focal
    label_col: is_churned
  - name: ltv
    type: regression
    loss: huber
    label_col: revenue_30d

data:
  source: s3://my-bucket/data/
  format: parquet

model:
  architecture: ple
  num_experts: 4

aws:
  instance: ml.g4dn.xlarge
  spot: true
```

```bash
./scripts/start_training.sh --config configs/my_problem.yaml
```

## Examples

| Dataset | Task | Architecture | Status |
|---------|------|-------------|--------|
| MovieLens 1M | Rating prediction + Watch ratio (multi-task) | PLE | *Coming soon* |
| Criteo | CTR prediction (binary) | LGBM → PLE distillation | *Coming soon* |
| [Synthetic](examples/synthetic/) | Configurable multi-task demo | PLE | Available |

## Project Structure

```
├── core/               # Framework core (domain-agnostic)
│   ├── task/           # AbstractTask, TaskRegistry, TaskType
│   ├── feature/        # AbstractFeatureTransformer, FeaturePipeline
│   ├── model/          # PLE, LGBM model implementations
│   └── pipeline/       # Config-driven pipeline runner
├── aws/                # AWS integration layer
│   ├── s3.py           # S3DataStore
│   ├── sagemaker/      # Training & serving wrappers
│   ├── stepfunctions/  # Orchestration state machines
│   └── athena.py       # Query layer
├── plugins/            # Extension points for custom components
├── infrastructure/     # AWS CDK stacks (IaC)
├── configs/            # YAML problem definitions
├── examples/           # End-to-end runnable examples
└── scripts/            # AWS resource management scripts
```

## Cost Model

| Phase | Service | Est. Cost |
|-------|---------|-----------|
| Data storage | S3 | ~$0.023/GB/month |
| Feature prep | SageMaker Processing (Spot) | ~$0.30/hr |
| Training | SageMaker Training (Spot g4dn.xlarge) | ~$0.53/hr |
| Serving (batch) | Lambda + S3 | Near-zero |
| Serving (real-time) | ECS Fargate (on-demand) | Pay per request |
| Orchestration | Step Functions | $0.025/1000 transitions |

> All compute resources are terminated after job completion. No idle costs.
