# 이종 전문가 PLE: 금융 상품 추천 시스템

[English](README.md) · **한국어**

[![Paper 1 DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19621884.svg)](https://doi.org/10.5281/zenodo.19621884)
[![Paper 2 DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19622052.svg)](https://doi.org/10.5281/zenodo.19622052)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Built with Claude Code](https://img.shields.io/badge/Built_with-Claude_Code-8B5CF6)](https://claude.com/claude-code)
[![DuckDB](https://img.shields.io/badge/Data_Engine-DuckDB-FFF000)](https://duckdb.org/)

> 단순히 고객이 살 상품을 예측하는 것을 넘어,
> 고객·은행·규제기관 모두가 이해할 수 있는 언어로 *왜* 그 상품인지 설명하는 추천 시스템.

**Zenodo 프리프린트:**
- 논문 1 — [이종 전문가 PLE: 아키텍처 및 어블레이션](https://doi.org/10.5281/zenodo.19621884) ([로컬 PDF](paper/typst/paper1_ko.pdf))
- 논문 2 — [예측에서 설득으로: 에이전트 기반 추천사유 생성과 규제 준수](https://doi.org/10.5281/zenodo.19622052) ([로컬 PDF](paper/typst/paper2_ko.pdf))
- 논문 3 — [Loss Dynamics (작성 중)](paper/typst/paper3.pdf)

---

## 무엇을 하는가

| 질문 | 답 |
|----------|--------|
| **무엇을** | 체크카드 상품을 위한 13-태스크 멀티태스크 추천 |
| **어떻게** | 7개의 구조적으로 상이한 AI 전문가가 각자 다른 렌즈로 고객을 본다 |
| **왜 중요한가** | 전문가 게이트 가중치 자체가 설명이 된다 -- "소비 트렌드 35% + 상품 적합도 28%" |
| **규제** | 한국 금감원 AI RMF, EU AI Act, AI 기본법 준수 설계 |
| **서빙** | LGBM 증류 → Lambda 서빙 -- GPU 서버 불필요 |
| **규모** | 고객 100만 명, 피처 349차원, 5-에이전트 아키텍처 (서빙 3 + 운영·감사 2) |
| **팀** | 3인 팀, Claude Code 기반 AI 증강 개발 |

## 개요

```
고객 데이터 (은행/카드 거래)
    |
    v
[Phase 0] 10개 피처 생성기 (11개 과학 분야, 349차원)
    |       TDA, Hyperbolic GCN, Mamba, HMM, 화학 반응 속도론, SIR, ...
    v
[Phase 1-3] PLE + 7개 이종 전문가 + 13개 태스크
    |         DeepFM | Temporal | HGCN | PersLay | LightGCN | Causal | OT
    v
[Phase 4] 지식 증류 -> LGBM (태스크별 13개, CPU 추론)
    |
    v
[Phase 5] Lambda 서빙 + 3-에이전트 추천사유 생성 + 안전 게이트
    |       + 2 운영/감사 에이전트 (모니터링, 규제 준수)
    v
"최근 3개월간 카드 사용이 15% 증가했고,
 교통·편의점 결제가 집중되어 있어
 통근형 체크카드를 추천드립니다."
```

## 7개의 전문가

| 전문가 | 무엇을 보는가 | 왜 중요한가 |
|--------|-------------|----------------|
| **DeepFM** | 피처 교차 | 소득 x 상품 x 채널 상호작용 |
| **Temporal** (Mamba+LNN+Transformer) | 시간 패턴 | 월간 추세 + 일간 급증 + 휴면 구간 |
| **Hyperbolic GCN** | 가맹점 계층 | MCC 카테고리 트리를 Poincaré 공간에 (27D) |
| **PersLay/TDA** | 행동 형상 | 소비 주기, 소비 위상구조 |
| **LightGCN** | 소셜 그래프 | "비슷한 고객이 이 상품도 보유" |
| **Causal** | 원인-결과 | "소비 증가가 카드 업그레이드 관심을 *유발*" |
| **Optimal Transport** | 분포 변화 | "기본형에서 프리미엄 사용 세그먼트로 이동" |

## 기술 스택

| 계층 | 기술 |
|-------|-----------|
| 데이터 처리 | DuckDB (단일 백엔드, 온프렘 240+ 파일), cuDF, PyArrow — [pandas-free 파이프라인](docs/duckdb-case-study.md) |
| 학습 | PyTorch, SageMaker Spot |
| 피처 엔지니어링 | 10개 GPU 가속 생성기 |
| 서빙 | AWS Lambda (서버리스, GPU 없음) |
| 증류 | 태스크별 LightGBM 학생 |
| 추천사유 생성 | LLM 에이전트 + 안전 게이트 |
| 설정 | YAML 파일 2개가 전체 제어 |

## 시작하기

```bash
pip install -e ".[dev]"

# 벤치마크 데이터 생성
PYTHONPATH=. python scripts/generate_benchmark_data.py --n-customers 1000000

# 피처 엔지니어링
PYTHONPATH=. python adapters/santander_adapter.py --input-dir data/benchmark --output-dir outputs/phase0

# 어블레이션 실행 (로컬, Docker 없음)
PYTHONPATH=. python scripts/run_local_ablation.py
```

## 프로젝트 구조

```
core/model/ple/          PLE 아키텍처, CGC 게이트, adaTT
core/model/experts/      7개 전문가 구현
core/feature/generators/ 10개 피처 생성기
core/pipeline/           Phase 0: 전처리, 레이블 파생, 정규화
core/training/           Trainer, evaluator, callbacks, config
core/recommendation/     점수화, 추천사유 생성, 규제 준수
core/agent/              운영/감사 에이전트, 합의, 케이스 저장소
adapters/                데이터 어댑터
aws/                     SageMaker, Step Functions
configs/santander/       pipeline.yaml, feature_groups.yaml
docs/                    설계 문서, 기술 레퍼런스 (KO/EN)
paper/                   연구 논문 (Typst)
```

## 문서

| 분류 | 문서 |
|----------|-----------|
| **논문** | [논문 1: 아키텍처 (Zenodo DOI)](https://doi.org/10.5281/zenodo.19621884) · [로컬 EN](paper/typst/paper1.pdf) · [KO](paper/typst/paper1_ko.pdf) · [논문 2: 서빙 & 운영 (Zenodo DOI)](https://doi.org/10.5281/zenodo.19622052) · [로컬 EN](paper/typst/paper2.pdf) · [KO](paper/typst/paper2_ko.pdf) · [논문 3: Loss Dynamics (WIP)](paper/typst/paper3.pdf) |
| **아키텍처** | [개요](docs/typst/ko/architecture_overview.pdf) · [전문가 상세](docs/typst/ko/expert_details.pdf) · [파이프라인 가이드](docs/typst/ko/pipeline_guide.pdf) |
| **기술 레퍼런스** | [PLE/adaTT](docs/typst/ko/tech_ref_ple_adatt.pdf) · [피처](docs/typst/ko/tech_ref_features.pdf) · [Causal/OT](docs/typst/ko/tech_ref_causal_ot.pdf) · [Temporal](docs/typst/ko/tech_ref_temporal.pdf) · [증류/추천사유](docs/typst/ko/tech_ref_distill_reason.pdf) |
| **규제** | [준수 요약](docs/typst/ko/regulatory_summary.pdf) · [전체 프레임워크](docs/typst/ko/regulatory_framework.pdf) |
| **운영/감사** | [에이전트 설계 (4,500줄)](docs/design/11_ops_audit_agent.pdf) |
| **가이드** | [Quickstart](docs/guides/quickstart.md) · [Config Reference](docs/guides/configuration_reference.md) · [배포](docs/guides/deployment.md) |
| **케이스 스터디** | [ML 파이프라인 엔진으로서의 DuckDB](docs/duckdb-case-study.md) · [AI 협업 가이드](docs/typst/ko/ai_collaboration_guide.pdf) |

모든 기술 문서는 한국어/영문 두 버전으로 제공됩니다 (`docs/typst/ko/`, `docs/typst/en/`).

## 인용

본 작업을 활용하시는 경우 프리프린트를 인용해 주세요:

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

## Claude Code로 구축

본 시스템의 모든 코드 — 아키텍처 설계, 7-전문가 모델, 에이전트 기반 추천사유 생성 파이프라인, 규제 준수 모듈, 260개 이상의 기술 문서, 그리고 두 개의 Zenodo 프리프린트 — 는 3인 팀이 **[Claude Code](https://claude.com/claude-code) (Anthropic)** 를 개인 구독 기반의 주요 개발 파트너로 삼아 구축하였습니다.

**제약 조건**: 기관 자금 없음, 전용 ML 인프라 없음, 단일 소비자용 GPU (RTX 4070, 12GB VRAM), 저녁·주말만 활용. **결과**: 규제 수준 감사 인프라를 갖춘 13-태스크 멀티태스크 학습 시스템, 두 개의 Zenodo 프리프린트와 함께 오픈소스화.
