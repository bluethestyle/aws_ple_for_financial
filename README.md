# Heterogeneous Expert PLE for Financial Product Recommendation

**금융 상품 추천 시스템** -- 고객에게 "왜 이 상품인지" 설명할 수 있는 AI

> A recommendation system that doesn't just predict what customers will buy —
> it explains *why*, in language that customers, bankers, and regulators understand.

**[Project Site](https://bluethestyle.github.io/aws_ple_for_financial/)** | [Paper 1 (Architecture)](paper/typst/paper1.pdf) | [Paper 2 (Serving)](paper/typst/paper2.pdf)

---

## What This Does

| Question | Answer |
|----------|--------|
| **What** | 13-task multi-task recommendation for check card products |
| **How** | 7 structurally different AI experts, each seeing the customer through a different lens |
| **Why it matters** | Expert gate weights *are* the explanation -- "35% spending trend + 28% product fit" |
| **Regulation** | Korean FSS AI RMF, EU AI Act, Korean AI Basic Act compliance built-in |
| **Serving** | Distilled to LGBM, runs on Lambda -- no GPU server needed |
| **Scale** | 1M customers, ~349 features, 5-agent architecture (3 serving + 2 ops/audit) |
| **Team** | Built by 3 people with AI-augmented development (Claude Code) |

## Quick Overview

```
Customer Data (bank/card transactions)
    |
    v
[Phase 0] 10 Feature Generators (11 scientific disciplines, ~349D)
    |       TDA, Hyperbolic GCN, Mamba, HMM, Chemical Kinetics, SIR, ...
    v
[Phase 1-3] PLE + 7 Heterogeneous Experts + 13 Tasks
    |         DeepFM | Temporal | HGCN | PersLay | LightGCN | Causal | OT
    v
[Phase 4] Knowledge Distillation -> LGBM (x13 tasks, CPU inference)
    |
    v
[Phase 5] Lambda Serving + 3-Agent Reason Generation + Safety Gate
    |       + 2 Ops/Audit Agents (monitoring, regulatory compliance)
    v
"최근 3개월간 카드 사용이 15% 증가했고,
 교통·편의점 결제가 집중되어 있어
 통근형 체크카드를 추천드립니다."
```

## The 7 Experts

| Expert | What It Sees | Why It Matters |
|--------|-------------|----------------|
| **DeepFM** | Feature crosses | Income x product x channel interactions |
| **Temporal** (Mamba+LNN+Transformer) | Time patterns | Monthly trends + daily bursts + dormancy gaps |
| **Hyperbolic GCN** | Product hierarchy | "Savings" is closer to "deposits" than "loans" |
| **PersLay/TDA** | Behavioral shape | Spending cycles, consumption topology |
| **LightGCN** | Social graph | "Similar customers also hold this product" |
| **Causal** | Cause-effect | "Spending increase *causes* card upgrade interest" |
| **Optimal Transport** | Distribution shift | "Moving from basic to premium usage segment" |

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
core/pipeline/           Phase 0: preprocessing, label derivation, normalization
core/training/           Trainer, evaluator, callbacks, config
core/recommendation/     Scoring, reason generation, compliance
core/agent/              Ops/Audit agents, consensus, case store
adapters/                Data adapters
aws/                     SageMaker, Step Functions
configs/santander/       pipeline.yaml, feature_groups.yaml
docs/                    Design docs, technical references (KO/EN)
paper/                   Research papers (Typst)
```

## Documentation

| Category | Documents |
|----------|-----------|
| **Papers** | [Paper 1: Architecture (EN)](paper/typst/paper1.pdf) · [KO](paper/typst/paper1_ko.pdf) · [Paper 2: Serving & Ops (EN)](paper/typst/paper2.pdf) · [KO](paper/typst/paper2_ko.pdf) |
| **Architecture** | [Overview](docs/typst/en/architecture_overview_en.pdf) · [Expert Details](docs/typst/en/expert_details_en.pdf) · [Pipeline Guide](docs/typst/en/pipeline_guide_en.pdf) |
| **Technical Refs** | [PLE/adaTT](docs/typst/en/tech_ref_ple_adatt_en.pdf) · [Features](docs/typst/en/tech_ref_features_en.pdf) · [Causal/OT](docs/typst/en/tech_ref_causal_ot_en.pdf) · [Temporal](docs/typst/en/tech_ref_temporal_en.pdf) · [Distillation/Reason](docs/typst/en/tech_ref_distill_reason_en.pdf) |
| **Regulatory** | [Compliance Summary](docs/typst/en/regulatory_summary_en.pdf) · [Full Framework](docs/typst/en/regulatory_framework_en.pdf) |
| **Ops/Audit** | [Agent Design (4,500 lines)](docs/design/11_ops_audit_agent.pdf) |
| **Guides** | [Quickstart](docs/guides/quickstart.md) · [Config Reference](docs/guides/configuration_reference.md) · [Deployment](docs/guides/deployment.md) |
| **Case Studies** | [DuckDB as ML Pipeline Engine](docs/duckdb-case-study.md) · [AI Collaboration Guide](docs/typst/en/ai_collaboration_guide_en.pdf) |

All technical documents are available in both Korean and English (see `docs/typst/ko/` and `docs/typst/en/`).

## AI-Augmented Development

This system was built by a team of 3 (1 data scientist PM + 2 engineers) with no dedicated ML infrastructure budget, on a single consumer GPU (RTX 4070, 12GB).

| Tool | Role | Share |
|------|------|-------|
| Claude Code (Opus/Sonnet) | Architecture design, code implementation, debugging, documentation | ~90% |
| Gemini | Brainstorming, concept exploration, literature review | ~5% |
| Cursor | Real-time editing, GitHub integration | ~5% |

Our AI collaboration methodology: [AI Collaboration Guide (PDF)](docs/typst/en/ai_collaboration_guide_en.pdf)

---

Built by a team of 3 with AI-augmented development (Claude Code, Anthropic).
