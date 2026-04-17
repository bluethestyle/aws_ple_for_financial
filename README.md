# Heterogeneous Expert PLE for Financial Product Recommendation

**English** · [한국어](README_ko.md)

[![Paper 1 DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19621884.svg)](https://doi.org/10.5281/zenodo.19621884)
[![Paper 2 DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19622052.svg)](https://doi.org/10.5281/zenodo.19622052)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Built with Claude Code](https://img.shields.io/badge/Built_with-Claude_Code-8B5CF6)](https://claude.com/claude-code)
[![DuckDB](https://img.shields.io/badge/Data_Engine-DuckDB-FFF000)](https://duckdb.org/)

> A recommendation system that doesn't just predict what customers will buy —
> it explains *why*, in language that customers, bankers, and regulators understand.

**Preprints on Zenodo:**
- Paper 1 — [Heterogeneous Expert PLE: Architecture & Ablation](https://doi.org/10.5281/zenodo.19621884) ([local PDF](paper/typst/paper1.pdf))
- Paper 2 — [From Prediction to Persuasion: Agentic Reason Generation & Compliance](https://doi.org/10.5281/zenodo.19622052) ([local PDF](paper/typst/paper2.pdf))
- Paper 3 — [Loss Dynamics (work in progress)](paper/typst/paper3.pdf)

---

## What This Does

| Question | Answer |
|----------|--------|
| **What** | 13-task multi-task recommendation for check card products |
| **How** | 7 structurally different AI experts, each seeing the customer through a different lens |
| **Why it matters** | Expert gate weights *are* the explanation -- "35% spending trend + 28% product fit" |
| **Regulation** | Korean FSS AI RMF, EU AI Act, Korean AI Basic Act compliance built-in |
| **Serving** | Distilled to LGBM, runs on Lambda -- no GPU server needed |
| **Scale** | 1M customers, 349 features, 5-agent architecture (3 serving + 2 ops/audit) |
| **Team** | Built by 3 people with AI-augmented development (Claude Code) |

## Quick Overview

```
Customer Data (bank/card transactions)
    |
    v
[Phase 0] 10 Feature Generators (11 scientific disciplines, 349D)
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
| **Hyperbolic GCN** | Merchant hierarchy | MCC category tree in Poincaré space (27D) |
| **PersLay/TDA** | Behavioral shape | Spending cycles, consumption topology |
| **LightGCN** | Social graph | "Similar customers also hold this product" |
| **Causal** | Cause-effect | "Spending increase *causes* card upgrade interest" |
| **Optimal Transport** | Distribution shift | "Moving from basic to premium usage segment" |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | DuckDB (sole backend, 240+ files on-prem), cuDF, PyArrow — [pandas-free pipeline](docs/duckdb-case-study.md) |
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

# Run ablation (local, no Docker)
PYTHONPATH=. python scripts/run_local_ablation.py
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
| **Papers** | [Paper 1: Architecture (Zenodo DOI)](https://doi.org/10.5281/zenodo.19621884) · [local EN](paper/typst/paper1.pdf) · [KO](paper/typst/paper1_ko.pdf) · [Paper 2: Serving & Ops (Zenodo DOI)](https://doi.org/10.5281/zenodo.19622052) · [local EN](paper/typst/paper2.pdf) · [KO](paper/typst/paper2_ko.pdf) · [Paper 3: Loss Dynamics (WIP)](paper/typst/paper3.pdf) |
| **Architecture** | [Overview](docs/typst/en/architecture_overview_en.pdf) · [Expert Details](docs/typst/en/expert_details_en.pdf) · [Pipeline Guide](docs/typst/en/pipeline_guide_en.pdf) |
| **Technical Refs** | [PLE/adaTT](docs/typst/en/tech_ref_ple_adatt_en.pdf) · [Features](docs/typst/en/tech_ref_features_en.pdf) · [Causal/OT](docs/typst/en/tech_ref_causal_ot_en.pdf) · [Temporal](docs/typst/en/tech_ref_temporal_en.pdf) · [Distillation/Reason](docs/typst/en/tech_ref_distill_reason_en.pdf) |
| **Regulatory** | [Compliance Summary](docs/typst/en/regulatory_summary_en.pdf) · [Full Framework](docs/typst/en/regulatory_framework_en.pdf) |
| **Ops/Audit** | [Agent Design (4,500 lines)](docs/design/11_ops_audit_agent.pdf) |
| **Guides** | [Quickstart](docs/guides/quickstart.md) · [Config Reference](docs/guides/configuration_reference.md) · [Deployment](docs/guides/deployment.md) |
| **Case Studies** | [DuckDB as ML Pipeline Engine](docs/duckdb-case-study.md) · [AI Collaboration Guide](docs/typst/en/ai_collaboration_guide_en.pdf) |

All technical documents are available in both Korean and English (see `docs/typst/ko/` and `docs/typst/en/`).

## Citation

If you use this work, please cite the preprints:

```bibtex
@misc{jeong2026heteroexpertple,
  author       = {Jeong, Seonkyu and Sim, Euncheol and Kim, Youngchan},
  title        = {{Heterogeneous Expert PLE: An Explainable Multi-Task
                   Architecture for Financial Product Recommendation}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.19621884},
  url          = {https://doi.org/10.5281/zenodo.19621884}
}

@misc{jeong2026agenticreason,
  author       = {Jeong, Seonkyu and Sim, Euncheol and Kim, Youngchan},
  title        = {{From Prediction to Persuasion: Agentic Recommendation
                   Reason Generation for Regulatory-Compliant Financial AI}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.19622052},
  url          = {https://doi.org/10.5281/zenodo.19622052}
}
```

## Built with Claude Code

Every line of this system — architecture design, 7-expert model, agentic reason generation pipeline, regulatory compliance modules, 260+ technical documents, and both Zenodo preprints — was built by a 3-person team using **[Claude Code](https://claude.com/claude-code) (Anthropic)** as the primary development partner on personal subscriptions.

**The constraint**: no institutional funding, no dedicated ML infrastructure, a single consumer GPU (RTX 4070, 12GB VRAM), evenings and weekends only. **The result**: a 13-task multi-task learning system with regulatory-grade audit infrastructure, open-sourced with two Zenodo preprints.

| Tool | Role in this project | Share |
|------|----------------------|-------|
| **Claude Opus** | Architecture design, cross-disciplinary reasoning (topology ↔ finance, chemical kinetics ↔ spending dynamics), complex debugging, paper writing | — |
| **Claude Sonnet** | Parallel code implementation across the team, per-task ablation coding, 3-agent serving pipeline (Feature Selector / Reason Generator / Safety Gate), 2-agent ops pipeline (OpsAgent / AuditAgent), bilingual documentation | — |
| Claude (Opus + Sonnet combined) | Total share | **~90%** |
| Gemini | Brainstorming, concept exploration, literature review | ~5% |
| Cursor | Real-time editing, GitHub integration | ~5% |

**Production uses of Claude in the running system** (not just during development):

- **5-agent architecture on AWS Bedrock** (Sonnet): 3 customer-facing serving agents + 2 ops/audit agents, with 3-agent independent-voting consensus (AWS) or 2-Round hybrid deliberation (on-prem).
- **Safety Gate**: Sonnet validates every customer-facing reason against regulatory, suitability, hallucination, tone, and factuality criteria before the response leaves the Lambda handler.
- **Reason Generator**: Sonnet rewrites template-level L1 reasons into natural financial-honorific Korean at L2a, with output cached in DynamoDB for cache-hit 6 ms latency.

**Our methodology**: [AI Collaboration Guide (PDF, EN)](docs/typst/en/ai_collaboration_guide_en.pdf) · [Development Story (PDF, EN)](docs/typst/en/development_story_en.pdf) — full documentation of how a 3-person team with no institutional support collaborated with Claude across architecture, implementation, testing, and paper writing.

If you are from Anthropic and this project is of interest for a customer story, blog post, or conversation, please reach out to the corresponding author (ORCID: [0009-0005-3291-9112](https://orcid.org/0009-0005-3291-9112)).

---

Built by a team of 3 with AI-augmented development (Claude Code, Anthropic).
