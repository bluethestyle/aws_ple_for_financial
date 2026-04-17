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

## Claude Code 고급 활용 패턴

이 저장소는 단순한 오픈소스 공개물을 넘어, **비자명한 Claude Code 워크플로우의 실전 참조 자료**로도 기능합니다. 아래 패턴들은 약 3.5개월, 240+ 소스 파일에 걸쳐 실제로 매일 의존한 패턴들이며, 각 패턴은 저장소 내 구체적 산출물에 링크되어 있어 말이 아니라 파일로 검증할 수 있습니다.

### 1. 프로젝트 전역 컨텍스트 엔지니어링으로서의 CLAUDE.md

[CLAUDE.md](CLAUDE.md) 는 README 가 아니라 **구속력 있는 규칙 세트** 이며, 모든 Claude Code 세션이 자동 로드합니다. 여섯 개의 단단한 섹션, 누적된 장애 대응의 결정체:

- **§1 Config-Driven 원칙** — 파이썬 코드에 컬럼명, 경계값, 시나리오 목록, AWS 상수 하드코딩 금지. 모든 파라미터는 `configs/pipeline.yaml` + `configs/datasets/*.yaml` 에서 `load_merged_config()` 로 읽어야 함.
- **§1.2 관심사 분리** — adapter / pipeline runner / config_builder / train.py 각각 책임이 잠겨 있음. "파일이 500줄을 넘으면 분리가 실패한 것" 이 기준.
- **§1.3 데이터 리키지 방지** — scaler는 TRAIN 에서만 fit, temporal split 에 `gap_days` 필수, 학습 전 `LeakageValidator` 호출 의무.
- **§1.7-1.10** — 누적된 post-mortem (피처 그룹 라우팅, 메트릭 집계, 증류 임계값, Champion-Challenger 승격). 각 하위 섹션은 날짜와 실제 사건으로 시작.
- **§4 코드 검수 기준** — 컴파일 검증, 인터페이스 계약 검증, 하드코딩 스캔, 관심사 분리 검증. 네 가지 모두 통과하기 전엔 "완료" 아님.
- **§6 금지사항** — 명시적 kill list (SageMaker 디버깅, `--no-verify`, 데이터셋-specific 하드코딩 라우팅).

**장애 이후에 규칙을 추가** 하는 것이 동작 원리입니다. 선제적 규칙이 아니라 사후 규칙이 누적됨.

### 2. 다개월 프로젝트를 위한 Auto-memory

프로젝트는 Claude Code의 auto-memory 시스템 (`~/.claude/projects/<project>/memory/`) 을 세션을 넘어 지속되는 협업 로그로 활용합니다. 예시 (22개 메모리 파일, 세션 간 유지):

- `feedback_no_hardcode_train.md` — "실험 파라미터는 config-driven, train.py 직접 수정 금지"
- `feedback_config_driven_strict.md` — "스케줄러 HP가 train.py에 하드코딩되고 있었음; YAML merge에 모든 섹션 포함 필수"
- `feedback_dryrun_verify.md` — "dry-run은 config 로드 확인만이 아니라 실제 적용된 HP 값을 로그에 남겨야"
- `project_task_reduction.md` — "18 → 13 태스크 축소, deterministic-leakage 레이블 제거"
- `feedback_gradsurgery.md` — "GradSurgery 실험했으나 채택 않음; adaTT-free PLE 대비 개선 없고 VRAM 부하 ↑"
- `feedback_windows_sleep.md` — "야간 학습이 Windows 절전 때문에 죽음; `SetThreadExecutionState` 필수"
- `feedback_checkpoint_resume.md` — "파일 패턴 불일치 + epoch counting 버그 모두 수정"

메모리 항목에는 `**Why:**` (원인 사건) 과 `**How to apply:**` (규칙이 적용되어야 할 때) 가 포함됩니다. 개별 교정이 대화 창을 넘어 지속되는 컨텍스트로 전환되는 구조.

### 3. 감사 스타일 작업을 위한 병렬 서브에이전트

본질적으로 병렬인 작업 — N 개 파일의 동일 이슈 점검, 두 언어 버전 논문 동기화, 분리된 코드베이스의 인터페이스 계약 재정합 — 에는 한 턴에 여러 서브에이전트를 디스패치한 후 **검증 서브에이전트** 를 합쳐 돌렸습니다. CLAUDE.md §5 에 명문화:

> 병렬 서브에이전트는 기본적으로 동시 실행 (한 메시지, 여러 Agent 도구 호출). 병렬 작업 후에는 반드시 인터페이스 계약 검증 에이전트를 추가로 실행하여 결과를 교차 확인.

구체 사례: 한국어 논문을 영문 v1 canonical 상태와 동기화 ([커밋 `9becbc0`](https://github.com/bluethestyle/aws_ple_for_financial/commit/9becbc0)) — 두 병렬 에이전트가 내용 공백 8곳을 채우고 깨진 표 11개를 수정, 그 후 세 번째 에이전트가 표 구조와 파일 간 참조를 검증. 첫 두 에이전트만으로는 서로의 누락을 잡지 못했음.

### 4. 아키텍처 수준 결정을 위한 Plan 모드

비자명한 구현 결정은 코드 작성 전에 `Plan` 서브에이전트를 경유합니다. 분리 구조: Plan이 단계별 계획 생성, 중요 파일 식별, 아키텍처 트레이드오프 제시 — 메인 세션은 이를 검토한 후 실행. 초기 세션에서 계획을 건너뛰었을 때 발생한 "Claude가 잘못된 것을 효율적으로 구현" 패턴을 여러 번 회피시켰음.

### 5. 테스트, 실패, 명시적 kill-list

- **§1.4 pre-flight check** — SageMaker Job 제출마다 ($0.50+ 비용). 4개 관문: Phase 0 출력 검증, generator 입력 검증, 레이블 분포 확인, dry-run + 50K 서브샘플 테스트. "SageMaker는 디버거가 아님" 이 확고한 규칙.
- **§1.5 비용 관리** — 프로파일러 비활성화, AMP 필수, spot 인스턴스 최대 4대 동시 실행, `max_wait = max_run + 1h`. 각 규칙은 특정 비용 사건에서 유래.
- **§1.6 오케스트레이션 비용 효율** — 상태 파일 기반 Job 스킵, S3 결과 확인, 예산 가드, 실패 Job 자동 eviction, Warm Pool.

### 6. 저장소에 문서화된 정직한 음성 결과

두 가지 미채택 사례가 코드베이스와 Paper 1 모두에 보존됨:

- **adaTT 손실 수준 전이** — 13-태스크 이질 설정에서 AUC −0.019 악화 (156 태스크 쌍 친화도를 안정적으로 추정 불가). Paper 1 §5 는 이를 headline *negative* finding 으로 보고. adaTT 는 재현성을 위해 코드에 남아있되 프로덕션 비사용.
- **GradSurgery 그라디언트 투영** — 대체품으로 실험, 정확도는 PLE-only baseline 과 동일하나 VRAM 부하 훨씬 큼. 메모리 항목 `feedback_gradsurgery.md` 에 미채택 결정 기록.

패턴: Claude 가 제안한 수정이 작동하지 않을 때, 그 수정은 ablation 기록 (Paper 1 §5) 에 남기고 결정은 메모리에 고정되어, 향후 세션이 재제안하지 않음.

### 7. 개발 파트너가 아닌 프로덕션 구성요소로서의 Claude

*실행 중인 시스템* 의 3 지점 (개발 단계만이 아닌) 이 AWS Bedrock 을 통해 Claude 를 사용:

- **3-에이전트 서빙 파이프라인** (Feature Selector / Reason Generator / Safety Gate) — Sonnet, AWS 에선 독립 투표 컨센서스, 온프렘에선 2-Round 혼합 숙의.
- **Safety Gate** — Sonnet 이 모든 고객 대상 추천 사유를 규제·적합성·환각·어조·사실성 기준으로 검증, Lambda 핸들러를 떠나기 전 관문.
- **Reason Generator** — Sonnet 이 L1 템플릿 수준 사유를 L2a 의 자연스러운 금융 경어체 한국어로 재작성, DynamoDB 캐싱으로 cache-hit 6 ms 레이턴시.

[Paper 2](https://doi.org/10.5281/zenodo.19622052) 에 SR 11-7 모델 리스크 관리 매핑이 포함된 전체 5-에이전트 아키텍처 (3 서빙 + 2 ops/audit) 문서화.

### 워크플로우 재현

| 산출물 | 보여주는 것 |
|--------|------------|
| [CLAUDE.md](CLAUDE.md) | 모든 세션이 로드하는 프로젝트 규칙 세트 |
| [`docs/typst/en/ai_collaboration_guide_en.pdf`](docs/typst/en/ai_collaboration_guide_en.pdf) | 방법론 전체 문서 (EN) |
| [`docs/typst/en/development_story_en.pdf`](docs/typst/en/development_story_en.pdf) | 3.5개월 빌드의 서사 |
| [`configs/pipeline.yaml`](configs/santander/pipeline.yaml) | §1.1 config-driven 규칙을 강제하는 설정 |
| [Paper 1 §5 (Ablation)](paper/typst/paper1.pdf) | adaTT/GradSurgery 음성 결과의 정직한 기록 |
| [`core/agent/`](core/agent/) | 프로덕션 에이전트 파이프라인 코드 |

### 스케일 주의

위 패턴들은 두 번 검증됨 — 공개 AWS 벤치마크 코드베이스 (240+ DuckDB 소스 파일, 본 저장소) 에서 한 번, 그리고 한국 금융기관의 별도 온프렘 코드베이스 (실고객 1200만, 프로덕션 피처 734, 규제 사유로 비공개) 에서 독립적으로 한 번. CLAUDE.md, 메모리 시스템, 병렬 서브에이전트, 명시적 음성 결과 훈련 규범이 두 환경 간에 그대로 전이됨. 온프렘 저장소의 Claude Code 대화 내역은 동일한 ~3.5개월에 걸쳐있으며 기관 데이터 거버넌스 정책에 따라 비공개로 보관.

---

## Claude Code로 구축

본 시스템의 모든 코드 — 아키텍처 설계, 7-전문가 모델, 에이전트 기반 추천사유 생성 파이프라인, 규제 준수 모듈, 260개 이상의 기술 문서, 그리고 두 개의 Zenodo 프리프린트 — 는 3인 팀이 **[Claude Code](https://claude.com/claude-code) (Anthropic)** 를 개인 구독 기반의 주요 개발 파트너로 삼아 구축하였습니다.

**제약 조건**: 기관 자금 없음, 전용 ML 인프라 없음, 단일 소비자용 GPU (RTX 4070, 12GB VRAM), 저녁·주말만 활용. **결과**: 규제 수준 감사 인프라를 갖춘 13-태스크 멀티태스크 학습 시스템, 두 개의 Zenodo 프리프린트와 함께 오픈소스화.
