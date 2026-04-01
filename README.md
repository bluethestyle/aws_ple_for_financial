# Heterogeneous Expert PLE for Financial Product Recommendation

An explainable multi-task learning system for financial product recommendation,
built on **Progressive Layered Extraction (PLE)** with **7 heterogeneous experts**
and **Adaptive Task Transfer (adaTT)**.

> **Core Insight**: The final deliverable of a recommendation system is not a probability --
> it is a *reason the customer can accept*. This architecture is designed so that
> the model structure itself generates business-interpretable explanations.

## Why Heterogeneous Experts?

Standard PLE uses identical MLP experts differentiated only by initialization.
We replace them with **structurally distinct** experts, each encoding a different inductive bias:

| Expert | Inductive Bias | Financial Pattern |
|--------|---------------|-------------------|
| DeepFM | Feature interaction | Income x product x channel crosses |
| Temporal Ensemble (Mamba + LNN + Transformer) | Multi-scale time series | Short/long/disrupted transaction patterns |
| Hyperbolic GCN | Tree hierarchy | Product category structure (MCC codes) |
| PersLay (TDA) | Topological persistence | Spending shape, consumption cycles |
| LightGCN | Graph convolution | Customer-product collaborative filtering |
| Causal (NOTEARS) | DAG constraint | Behavioral causation, not just correlation |
| Optimal Transport | Distribution matching | Customer segment distributional shifts |

Each expert's CGC gate weight is directly interpretable:
*"This recommendation is driven by your spending trend (Temporal, 35%) and product category fit (HGCN, 28%)"*
-- no post-hoc SHAP/LIME needed.

## Architecture

```
                        +----------------------------------------------+
                        |          PLE + adaTT Architecture             |
                        |                                               |
 316 Features --------->|  12 Feature Groups --> CGC Gate --> 7 Experts  |
 (11 disciplines)       |       |                    |         |        |
                        |       v                    v         v        |
                        |  4 Task Groups  <--  adaTT Transfer           |
                        |  (engagement / lifecycle / value / consumption)|
                        |       |                                       |
                        |       v                                       |
                        |  18 Task Towers --> Predictions               |
                        +----------------------------------------------+
                                   |
                         +---------v----------+
                         | Knowledge Distill. |
                         |  PLE -> LGBM (x18) |
                         +---------+----------+
                                   |
                         +---------v----------+
                         |  Lambda Serving    |
                         | + Reason Generation|
                         | (3-Agent Pipeline) |
                         +--------------------+
```

## Multi-Disciplinary Features (316D)

| Discipline | Method | Dim | What It Captures |
|------------|--------|-----|------------------|
| Topology | TDA Persistent Homology | 32 | Spending pattern shape, consumption cycles |
| Hyperbolic Geometry | Poincare Ball Embedding | 34 | Product hierarchy distance |
| Control Theory | Mamba (State Space) | 50 | Long-range behavioral dependencies |
| Stochastic Processes | HMM State Transitions | 25 | Latent lifecycle stages |
| Chemical Kinetics | Reaction Rate Modeling | 6 | Spending activation, dormancy reactivation |
| Epidemiology | SIR Compartment Model | 5 | Product adoption as contagion |
| Criminology | Routine Activity Theory | 5 | Transaction burstiness, circadian patterns |
| Signal Processing | FFT + Hilbert Transform | 8 | Spending periodicity, harmonic power |
| Economics | Friedman Permanent Income | 8 | Income stability decomposition |
| Graph Theory | LightGCN | 66 | Collaborative filtering signals |
| Statistics | GMM Soft Clustering | 22 | Probabilistic customer segmentation |

## Pipeline

```
Phase 0  Feature Engineering (DuckDB/cuDF, pandas-free)
           Adapter -> 10 Generators -> 3-Stage Normalization -> Leakage Validation

Phase 1-3  Training + Ablation (54 scenarios)
           Feature Group / Expert / Task x Structure ablation

Phase 4  Knowledge Distillation
           PLE teacher -> LGBM student (x18 tasks), IG-based feature selection

Phase 5  Serving + Compliance
           Lambda + 3-Agent Reason Generation + Monitoring (drift, fairness, audit)
```

## Config-Driven Design

Two YAML files control the entire system:

- `configs/santander/pipeline.yaml` -- model, training, deployment (18 tasks, 4 task groups, loss config)
- `configs/santander/feature_groups.yaml` -- feature generation, expert routing (12 groups, 10 generators)

Adding a new dataset, task, or expert requires **only configuration changes**.

## Regulatory Compliance

Designed for Korean FSS guidelines, EU AI Act, and Korean AI Basic Act:

| Requirement | System Component |
|-------------|-----------------|
| Explainability | Gate weights + 3-agent reason generation |
| Fairness | DI/SPD/EOD monitoring |
| Model Validation | Champion-Challenger, ablation evidence |
| Monitoring | PSI drift detection |
| Audit Trail | HMAC hash-chain immutable logs |
| Human Oversight | Sampling review, kill switch, opt-out |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Generate benchmark data (1M customers)
PYTHONPATH=. python scripts/generate_benchmark_data.py \
  --n-customers 1000000 --seed 42 --output data/benchmark/benchmark.parquet

# Phase 0 (feature engineering)
PYTHONPATH=. python adapters/santander_adapter.py \
  --input-dir data/benchmark --output-dir outputs/phase0

# Ablation (Docker mode)
ABLATION_USE_DOCKER=1 PYTHONPATH=. python scripts/run_local_ablation.py

# Or SageMaker cloud
python scripts/submit_pipeline.py --mode training --config configs/santander/pipeline.yaml
```

## Tech Stack

**Training**: PyTorch, SageMaker (Spot) | **Data**: DuckDB, cuDF, PyArrow (pandas-free) | **Serving**: Lambda + API Gateway | **Distillation**: LightGBM | **Reason Generation**: LLM agents + Safety Gate | **Config**: YAML-driven

## Project Structure

```
core/                       Framework core
  model/ple/                PLE, CGC gate, adaTT
  model/experts/            7 heterogeneous experts
  feature/generators/       10 generators (TDA, HMM, Mamba, Graph, GMM...)
  pipeline/                 Adapter, normalizer, leakage validator
  training/                 Trainer, callbacks
  recommendation/           Scoring, reason generation, compliance
adapters/                   Data adapters
aws/                        SageMaker, Step Functions
containers/training/        Docker + train.py
configs/santander/          pipeline.yaml, feature_groups.yaml
scripts/                    Ablation, benchmark, pipeline submission
paper/                      Research papers (Typst)
docs/design/                Architecture specs
```

## Papers

1. *Heterogeneous Expert PLE: An Explainable Multi-Task Architecture for Financial Product Recommendation* -- architecture + 54-scenario ablation
2. *From Prediction to Persuasion: Agentic Recommendation Reason Generation for Regulatory-Compliant Financial AI* -- distillation + reason generation + compliance

## Cost Model

| Phase | Service | Est. Cost |
|-------|---------|-----------|
| Feature engineering | SageMaker Processing (Spot) | ~$0.30/hr |
| Training | SageMaker Training (Spot g5.xlarge) | ~$1.00/hr |
| Ablation (54 scenarios) | 4x Spot parallel | ~$16 total |
| Serving | Lambda | Pay per request |

> All compute terminated after completion. No idle costs. No Kubernetes needed.

---

Built by a team of 3 with AI-augmented development (Claude, Anthropic).
