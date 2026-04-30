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
| **Scale** | 1M customers, 1211 features (17 groups), 5-agent architecture (3 serving + 2 ops/audit) |
| **Team** | Built by 3 people with AI-augmented development (Claude Code) |

## Quick Overview

```
Customer Data (bank/card transactions)
    |
    v
[Phase 0] 11 generator types referenced in santander config (14 generator implementations available in core/feature/generators/) → 17 feature groups, 1211D
    |       TDA, Hyperbolic GCN, Mamba, HMM, LagExtractor, RollingStats, TopN MultiHot, ...
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
| Feature Engineering | 11 feature generators (GPU-accelerated where applicable) |
| Serving | AWS Lambda (serverless, no GPU) |
| Distillation | LightGBM per-task students |
| Reason Generation | LLM agents with Safety Gate |
| Config | 3-layer split-config (`pipeline.yaml` + `datasets/{name}.yaml` + `feature_groups.yaml`) |

## Getting Started

```bash
pip install -e ".[dev]"

# Generate benchmark data (1M synthetic customers)
PYTHONPATH=. python scripts/generate_benchmark_data.py --n-customers 1000000

# (Optional) Precompute Mamba temporal embeddings on a GPU SageMaker job.
# The mamba_ssm CUDA wheel does not build on the CPU m5.* instance used for
# Phase 0, so the SSM expert is run as a separate GPU job using a custom
# ECR image (containers/mamba/Dockerfile, cu122-torch2.1, prebuilt wheels).
# Output: s3://{bucket}/{task}/mamba/embedding.parquet — joined back into
# Phase 0 via feature_groups.yaml::mamba_temporal.cached_embedding_uri.
PYTHONPATH=. python scripts/submit_pipeline.py --mamba-precompute

# Run the full training pipeline (Phase 0 preprocessing + training).
# The adapter only converts raw data to a standardized DataFrame;
# PipelineRunner drives preprocessing, feature generation, 3-stage
# normalization, label derivation, and tensor save.
PYTHONPATH=. python containers/training/train.py \
  --config  configs/pipeline.yaml \
  --dataset configs/datasets/santander.yaml \
  --phase0-only            # Phase 0 only; drop this flag to continue into training

# Run ablation (local, no Docker)
PYTHONPATH=. python scripts/run_local_ablation.py
```

## Project Structure

```
core/model/ple/          PLE architecture, CGC gate, adaTT
core/model/experts/      7 expert implementations
core/feature/generators/ 11 feature generators (17 groups → 1211D)
core/pipeline/           Phase 0: preprocessing, label derivation, normalization
core/training/           Trainer, evaluator, callbacks, config
core/recommendation/     Scoring, reason generation, compliance
core/agent/              Ops/Audit agents, consensus, case store
adapters/                Data adapters
aws/                     SageMaker, Step Functions
configs/                 pipeline.yaml (common) + datasets/{name}.yaml (per-dataset) + feature_groups.yaml
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

## Advanced Claude Code Usage Patterns

This repository is also a working reference for non-trivial Claude Code workflows. What follows are the patterns we actually relied on day-to-day across ~3.5 months and 240+ source files. Each pattern links to the concrete artifact in this repo so it can be inspected rather than taken on trust.

### 1. CLAUDE.md as project-wide context engineering

[CLAUDE.md](CLAUDE.md) is not a README — it is a **binding ruleset** that every Claude Code session loads automatically. Six hardened sections, written as accumulated incident response:

- **§1 Config-Driven principle** — forbids hardcoded column names, boundary values, scenario lists, AWS constants. Every parameter must be read from `configs/pipeline.yaml` + `configs/datasets/*.yaml` via `load_merged_config()`.
- **§1.2 Separation of Concerns** — adapter / pipeline runner / config_builder / train.py each have a locked responsibility. "If a file exceeds 500 lines, the separation failed."
- **§1.3 Data leakage prevention** — scaler must fit TRAIN only, temporal split requires `gap_days`, `LeakageValidator` must run pre-training.
- **§1.7-1.10** — accumulated post-mortems (feature-group routing, metric aggregation, distillation thresholds, Champion-Challenger promotion). Each subsection starts with a date and a real incident.
- **§4 Code review criteria** — compile check, interface contract check, hardcoding scan, separation check. A task is not "done" until all four pass.
- **§6 Forbidden actions** — explicit kill list (SageMaker debugging, `--no-verify`, hardcoded dataset routing).

Adding rules to CLAUDE.md **after** a failure, not before, is the working pattern. It compounds.

### 2. Auto-memory for multi-month projects

The project uses Claude Code's auto-memory system (`~/.claude/projects/<project>/memory/`) as a persistent collaboration log. Sample entries (22 memory files, maintained across sessions):

- `feedback_no_hardcode_train.md` — "experiment parameters must be config-driven, no direct edits to train.py"
- `feedback_config_driven_strict.md` — "scheduler HPs were getting hardcoded in train.py; YAML merge must include every section"
- `feedback_dryrun_verify.md` — "dry-run must log actual HP values applied, not just confirm config loaded"
- `project_task_reduction.md` — "18 → 13 task reduction; deterministic-leakage labels removed"
- `feedback_gradsurgery.md` — "GradSurgery tested but NOT adopted; no improvement over adaTT-free PLE baseline, higher VRAM"
- `feedback_windows_sleep.md` — "overnight training killed by Windows sleep; `SetThreadExecutionState` is mandatory"
- `feedback_checkpoint_resume.md` — "filename pattern mismatch + epoch counting bug both fixed"

Memory entries include a `**Why:**` line (the originating incident) and a `**How to apply:**` line (when the rule should kick in). This turns individual corrections into durable context that survives the conversation window.

### 3. Parallel subagents for audit-style work

When a task is inherently parallel — checking N files for the same issue, syncing two language versions of a paper, reconciling interface contracts across a split codebase — we dispatched multiple subagents in one turn and then ran a **validator subagent** on the combined output. CLAUDE.md §5 codifies this:

> Parallel sub-agents run concurrently by default (one message, multiple Agent tool calls). After parallel work, a follow-up interface-contract validation agent MUST run to cross-check the results.

Concrete example: syncing the Korean papers with the English v1 canonical state ([commit `9becbc0`](https://github.com/bluethestyle/aws_ple_for_financial/commit/9becbc0)) — two parallel agents filled 8 content gaps and fixed 11 broken tables, then a third agent verified table structure and cross-file references. Neither of the first two would have caught the other's misses.

### 4. Plan mode for architecture-level decisions

Non-trivial implementation decisions route through the `Plan` subagent before any code is written. The separation: Plan produces a step-by-step plan, identifies critical files, and surfaces architectural trade-offs — the main session reviews and then executes. This avoided several "Claude implemented the wrong thing efficiently" failure modes that occurred in early sessions when we skipped planning.

### 5. Tests, failures, and explicit kill-lists

- **§1.4 pre-flight check** before every SageMaker job submission (cost: $0.50+ per submission). Four gates: Phase 0 output validation, generator input validation, label distribution check, dry-run + 50K subsample test. "SageMaker is not a debugger" is a hard rule.
- **§1.5 cost management** — profiler disabled, AMP mandatory, spot instance cap at 4 concurrent, `max_wait = max_run + 1h`. Each rule traces to a specific cost incident.
- **§1.6 orchestration cost efficiency** — state-file-based job skip, S3 result check, budget guard, failed-job eviction, warm pool.

### 6. Honest negative results documented in-repo

Two non-adoptions are preserved in both the codebase and Paper 1:

- **adaTT loss-level transfer** — degraded AUC by −0.019 in the 13-task heterogeneous setting (156 task-pair affinities cannot be estimated stably). Paper 1 §5 reports this as the headline *negative* finding. adaTT is still in the code for reproducibility; not used in production.
- **GradSurgery gradient projection** — tested as a replacement, matched the PLE-only baseline in accuracy but required significantly more VRAM. Memory entry `feedback_gradsurgery.md` records the decision not to adopt.

The pattern: when Claude proposes a fix that doesn't work, the fix stays in the ablation record (Paper 1 §5) and the decision is pinned to memory, so future sessions don't re-propose it.

### 7. Claude as production component, not just dev partner

Three points in the *running system* (not just development) use Claude via AWS Bedrock:

- **3-agent serving pipeline** (Feature Selector / Reason Generator / Safety Gate) — Sonnet, independent voting consensus on AWS, 2-Round hybrid deliberation on on-prem.
- **Safety Gate** — Sonnet validates every customer-facing reason against regulatory, suitability, hallucination, tone, and factuality criteria before the response leaves the Lambda handler.
- **Reason Generator** — Sonnet rewrites template-level L1 reasons into natural financial-honorific Korean at L2a, with DynamoDB caching for 6 ms cache-hit latency.

[Paper 2](https://doi.org/10.5281/zenodo.19622052) documents the full 5-agent architecture (3 serving + 2 ops/audit) with SR 11-7 model-risk-management mapping.

### Reproducing the workflow

| Artifact | What it shows |
|----------|---------------|
| [CLAUDE.md](CLAUDE.md) | Project ruleset loaded by every session |
| [`docs/typst/en/ai_collaboration_guide_en.pdf`](docs/typst/en/ai_collaboration_guide_en.pdf) | Full methodology write-up (EN) |
| [`docs/typst/en/development_story_en.pdf`](docs/typst/en/development_story_en.pdf) | Narrative of the 3.5-month build |
| [`configs/pipeline.yaml`](configs/santander/pipeline.yaml) | The config that enforces §1.1 config-driven rule |
| [Paper 1 §5 (Ablation)](paper/typst/paper1.pdf) | Honest record of adaTT/GradSurgery negative results |
| [`core/agent/`](core/agent/) | Production agent pipeline code |

### Scale note

The patterns above are validated twice — in this public AWS benchmark codebase (240+ DuckDB source files, this repo) and independently in a separate on-premises codebase at a Korean financial institution (12M real customers, 734 production features, not public for regulatory reasons). CLAUDE.md, the memory system, parallel subagents, and the explicit negative-results discipline transferred cleanly between the two. The on-prem repo's Claude Code conversation history spans the same ~3.5 months but is retained privately under the institution's data governance policy.

---

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

### Where to find us

- **Blog** — [bluethestyle.github.io](https://bluethestyle.github.io) — decision-journey notes covering the 3-month build, MRM / regulatory perspective, and paper walkthroughs (EN/KO pair)
- **Discussions** — [bluethestyle/aws_ple_for_financial/discussions](https://github.com/bluethestyle/aws_ple_for_financial/discussions) — technical questions, workflow, reproduction
- **Issues** — [bluethestyle/aws_ple_for_financial/issues](https://github.com/bluethestyle/aws_ple_for_financial/issues) — bug reports, reproducibility notes
- **ORCID** — [0009-0005-3291-9112](https://orcid.org/0009-0005-3291-9112)

---

Built by a team of 3 with AI-augmented development (Claude Code, Anthropic).
