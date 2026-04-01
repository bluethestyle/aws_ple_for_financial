# Heterogeneous Expert PLE for Financial Product Recommendation

**금융 상품 추천 시스템** -- 고객에게 "왜 이 상품인지" 설명할 수 있�� AI

> A recommendation system that doesn't just predict what customers will buy —
> it explains *why*, in language that customers, bankers, and regulators understand.

**[Project Site](https://bluethestyle.github.io/aws_ple_for_financial/)** | [Paper 1 (Architecture)](paper/typst/paper1.pdf) | [Paper 2 (Serving)](paper/typst/paper2.pdf)

---

## What This Does

| Question | Answer |
|----------|--------|
| **What** | 18-task multi-task recommendation for banks/card companies |
| **How** | 7 structurally different AI experts, each seeing the customer through a different lens |
| **Why it matters** | Expert gate weights *are* the explanation -- "35% spending trend + 28% product fit" |
| **Regulation** | Korean FSS, EU AI Act, Korean AI Basic Act compliance built-in |
| **Serving** | Distilled to LGBM, runs on Lambda -- no GPU server needed |
| **Scale** | 1M customers, 316 features, 54 ablation scenarios |
| **Team** | Built by 3 people with AI-augmented development |

## Quick Overview

```
Customer Data (bank/card transactions)
    |
    v
[Phase 0] 10 Feature Generators (11 scientific disciplines, 316D)
    |       TDA, Hyperbolic GCN, Mamba, HMM, Chemical Kinetics, SIR, ...
    v
[Phase 1-3] PLE + 7 Heterogeneous Experts + 18 Tasks
    |         DeepFM | Temporal | HGCN | PersLay | LightGCN | Causal | OT
    v
[Phase 4] Knowledge Distillation -> LGBM (x18 tasks, CPU inference)
    |
    v
[Phase 5] Lambda Serving + 3-Agent Reason Generation + Safety Gate
    |
    v
"Your spending trend increased 15% over 3 months,
 and your product portfolio aligns with investment funds.
 We recommend a balanced growth fund."
```

## The 7 Experts

| Expert | What It Sees | Why It Matters |
|--------|-------------|----------------|
| **DeepFM** | Feature crosses | Income x product x channel interactions |
| **Temporal** (Mamba+LNN+Transformer) | Time patterns | Monthly trends + daily bursts + dormancy gaps |
| **Hyperbolic GCN** | Product hierarchy | "Savings" is closer to "deposits" than "loans" |
| **PersLay/TDA** | Behavioral shape | Spending cycles, consumption topology |
| **LightGCN** | Social graph | "Similar customers also hold this product" |
| **Causal** | Cause-effect | "Spending increase *causes* investment interest" |
| **Optimal Transport** | Distribution shift | "Moving from conservative to growth segment" |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | DuckDB, cuDF, PyArrow (pandas-free) |
| Training | PyTorch, SageMaker Spot |
| Feature Engineering | 10 GPU-accelerated generators |
| Serving | AWS Lambda (serverless, no GPU) |
| Distillation | LightGBM per-task students |
| Reason Generation | LLM agents with Safety Gate |
| Config | 2 YAML files drive everything |

## Getting Started

```bash
pip install -e ".[dev]"

# Generate benchmark data
PYTHONPATH=. python scripts/generate_benchmark_data.py --n-customers 1000000

# Feature engineering
PYTHONPATH=. python adapters/santander_adapter.py --input-dir data/benchmark --output-dir outputs/phase0

# Run ablation
ABLATION_USE_DOCKER=1 PYTHONPATH=. python scripts/run_local_ablation.py
```

## Project Structure

```
core/model/ple/          PLE architecture, CGC gate, adaTT
core/model/experts/      7 expert implementations
core/feature/generators/ 10 feature generators
core/recommendation/     Scoring, reason generation, compliance
adapters/                Data adapters
aws/                     SageMaker, Step Functions
configs/santander/       pipeline.yaml, feature_groups.yaml
paper/                   Research papers (Typst)
```

---

Built by a team of 3 with AI-augmented development (Claude, Anthropic).
