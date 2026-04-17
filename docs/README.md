# Documentation Index

Navigation for all documentation in this repository. Published research papers are in [`paper/`](../paper/README.md); this directory contains implementation designs, technical references, and user guides.

## Structure

```
docs/
├── README.md              ← you are here
├── design/                ← implementation design notes (authoritative source)
├── typst/                 ← typeset technical references (EN/KO, generates PDFs)
├── guides/                ← user-facing quick guides
├── site/                  ← static site (GitHub Pages)
└── duckdb-case-study.md   ← standalone case study
```

## design/ — Implementation Design Notes

Numbered design documents describing each architectural layer. These are the source of truth for how the system is built.

| # | Document | Covers |
|---|----------|--------|
| 00 | [architecture_overview](design/00_architecture_overview.md) | System-wide architecture |
| 01 | [data_layer](design/01_data_layer.md) | Ingestion, adapters, benchmark |
| 02 | [feature_engineering](design/02_feature_engineering.md) | Phase 0, 10 generators, normalization |
| 03 | [model_architecture](design/03_model_architecture.md) | PLE, 7 experts, CGC gate |
| 04 | [training_pipeline](design/04_training_pipeline.md) | Training loop, champion-challenger promotion |
| 05 | [serving_and_testing](design/05_serving_and_testing.md) | Lambda serving, 3-layer fallback |
| 06 | [orchestration_and_audit](design/06_orchestration_and_audit.md) | SageMaker pipeline, audit logging |
| 07 | [cost_analysis](design/07_cost_analysis.md) | Spot pricing, budget management |
| 08 | [recommendation_intelligence](design/08_recommendation_intelligence.md) | Scoring, FD-TVS weighting |
| 09 | [compliance_governance](design/09_compliance_governance.md) | Regulatory mapping |
| 10 | [pool_basket_architecture](design/10_pool_basket_architecture.md) | Shared expert basket design |
| 11 | [ops_audit_agent](design/11_ops_audit_agent.md) | 5-agent architecture (3 serving + 2 ops) |
| 11 | [ops_audit_agent_onprem_handoff](design/11_ops_audit_agent_onprem_handoff.md) | On-prem vs AWS consensus |
| 12 | [rule_based_fallback](design/12_rule_based_fallback.md) | Layer 3 rule engine |
| — | [ablation_test_design](design/ablation_test_design.typ) | Ablation test design doc (typst) |
| — | [santander_ablation_design](design/santander_ablation_design.md) | Santander benchmark ablation plan |
| — | [onprem_experiment_design](design/onprem_experiment_design.md) | On-prem reference experiment plan |

## typst/ — Technical References (Typeset)

Detailed technical references in English and Korean. Each has a `.typ` source and generates a `.pdf`.

| Topic | EN | KO |
|-------|-----|-----|
| Architecture overview | [en](typst/en/architecture_overview_en.pdf) | [ko](typst/ko/architecture_overview.pdf) |
| Expert details | [en](typst/en/expert_details_en.pdf) | [ko](typst/ko/expert_details.pdf) |
| Pipeline guide | [en](typst/en/pipeline_guide_en.pdf) | [ko](typst/ko/pipeline_guide.pdf) |
| Development story | [en](typst/en/development_story_en.pdf) | [ko](typst/ko/development_story.pdf) |
| Regulatory framework | [en](typst/en/regulatory_framework_en.pdf) | [ko](typst/ko/regulatory_framework.pdf) |
| Regulatory summary | [en](typst/en/regulatory_summary_en.pdf) | [ko](typst/ko/regulatory_summary.pdf) |
| Tech ref: PLE/adaTT | [en](typst/en/tech_ref_ple_adatt_en.pdf) | [ko](typst/ko/tech_ref_ple_adatt.pdf) |
| Tech ref: Features | [en](typst/en/tech_ref_features_en.pdf) | [ko](typst/ko/tech_ref_features.pdf) |
| Tech ref: Causal/OT | [en](typst/en/tech_ref_causal_ot_en.pdf) | [ko](typst/ko/tech_ref_causal_ot.pdf) |
| Tech ref: Temporal | [en](typst/en/tech_ref_temporal_en.pdf) | [ko](typst/ko/tech_ref_temporal.pdf) |
| Tech ref: Distillation/Reason | [en](typst/en/tech_ref_distill_reason_en.pdf) | [ko](typst/ko/tech_ref_distill_reason.pdf) |
| AI collaboration guide | [en](typst/en/ai_collaboration_guide_en.pdf) | [ko](typst/ko/ai_collaboration_guide.pdf) |

## guides/ — Quick Guides

| Guide | Purpose |
|-------|---------|
| [quickstart](guides/quickstart.md) | Get running locally in 10 minutes |
| [configuration_reference](guides/configuration_reference.md) | Complete pipeline.yaml / feature_groups.yaml reference |
| [deployment](guides/deployment.md) | SageMaker + Lambda deployment |
| [feature_engineering](guides/feature_engineering.md) | Adding new features |
| [model_architecture](guides/model_architecture.md) | Extending the expert basket |

## Related

- [`../paper/README.md`](../paper/README.md) — research papers and source materials
- [`../README.md`](../README.md) — project top-level README with Zenodo DOI badges
- [duckdb-case-study.md](duckdb-case-study.md) — DuckDB as ML pipeline engine (standalone case study)
