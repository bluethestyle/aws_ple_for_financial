# 00. Architecture Overview — AWS PLE Platform

## 설계 철학

### 핵심 원칙
1. **Config-Driven**: YAML 하나로 데이터/태스크/모델/인프라를 정의 — 코드 변경 없이 새 문제 적용
2. **Registry Pattern**: Expert, Task, Feature, Model, Tower 모두 플러그인 방식 등록 — 확장 시 기존 코드 수정 불필요
3. **Pay-as-you-go**: 상시 서버 없음 — 학습/추론 시에만 리소스 할당, 완료 후 자동 해제
4. **Schema-First**: 데이터 스키마가 파이프라인 전체를 결정 — 스키마 변경 시 하위 파이프라인 자동 조정
5. **Audit by Design**: 모든 단계에서 데이터 리니지/실험 이력/결정 근거 기록 — pipeline_manifest, pipeline_state 추적
6. **Scale-Aware**: 트래픽/데이터 규모에 따라 config 한 줄로 인프라 전환 (Lambda↔ECS, Memory↔DynamoDB)
7. **5-Axis Feature Classification**: 모든 피처를 State/Snapshot/Timeseries/Hierarchy/Item 5축으로 분류 — Expert 라우팅의 기반
8. **Pool/Basket/Runtime 3계층**: 코드(Pool) → Config(Basket) → 학습(Runtime) 분리 — 도메인 전환 시 코드 수정 0
9. **Leakage Prevention**: 시퀀스 절단, 제품 재계산, 시간 기반 분할(gap_days), LeakageValidator 4중 검증

---

## 10+ Stage End-to-End 파이프라인

```
Stage 1           Stage 1.5          Stage 2            Stage 3
DataAdapter        TemporalPrep       SchemaClassifier   EncryptionPipeline
Raw Data Load      Leakage Prevention (5-axis)           (PII→SHA256→INT32)
┌──────────┐      ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ S3 Raw   │─────▶│ Seq truncate │──▶│ 5-Axis       │──▶│ core/        │
│ Parquet  │      │ (drop last   │   │ Classifier   │   │ security/    │
│ + Schema │      │  month)      │   │              │   │ pipeline.py  │
│ Registry │      │ Prod recomp  │   │ state        │   │              │
└──────────┘      │ from month16 │   │ snapshot     │   └──────┬───────┘
                  └──────────────┘   │ timeseries   │          │
                                     │ hierarchy    │          │
                                     │ item         │          │
                                     └──────────────┘          │
                                                               ▼
Stage 4            Stage 5           Stage 5.5          Stage 6
FeatureGroup +     LabelDeriver      LeakageValidator   SequenceBuilder
Normalization      (13 tasks)        (4-check)          (flat→3D)
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Per-Axis     │  │ Config-driven│  │ Sequence     │  │ event_seq.npy│
│ Feature Eng. │  │ label derive │  │ Correlation  │  │ session_seq  │
│ + PowerLaw   │  │ clip+log1p   │  │ Product      │  │ .npy         │
│ Scaler       │  │ 13 labels    │  │ Temporal     │  │              │
└──────────────┘  └──────────────┘  └──────────────┘  └──────┬───────┘
                                                              │
                                                              ▼
Stage 7            Stage 8           Stage 8.5          Stage 9
DataLoader         PLETrainer        Model Analysis     StudentTrainer
(temporal split)   (2-phase)         (Interpretability)  (Distillation)
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ PLEDataset   │  │ PLE + adaTT  │  │ IG, CCA,     │  │ PLE teacher  │
│ temporal     │  │ Uncertainty  │  │ Gate, HGCN,  │  │ → LGBM       │
│ split with   │  │ Weighting    │  │ Multi Interp │  │ students     │
│ gap_days=30  │  │ Per-task Loss│  │ Template Eng │  │ + fidelity   │
└──────────────┘  │ Evidential   │  │ XAI, Model   │  │ validation   │
                  │ SAE Reg.     │  │ Card         │  └──────┬───────┘
                  └──────────────┘  └──────────────┘         │
                                                              ▼
Stage 9.5          Stage 10
Context Vector     CPE + Agentic
Store (RAG)        Reason Orchestrator
┌──────────────┐  ┌──────────────┐
│ Embedding    │  │ L1+L2a+L2b   │
│ store for    │  │ orchestrator │
│ reason       │  │ FD-TVS score │
│ retrieval    │  │ DNA modifier │
└──────────────┘  │ Constraints  │
                  └──────────────┘

비동기 감시 계층 (Async Monitoring Layer) — 순차 스테이지와 무관하게 파이프라인 전체를 병렬 관측
┌─────────────────────────────────────────────────────────────────────┐
│  ├── OpsAgent    — 파이프라인 성능/안정성/비용 진단 (7 체크포인트)     │
│  └── AuditAgent  — 규제 준수/공정성/추천사유 품질 감사 (5 관점)        │
└─────────────────────────────────────────────────────────────────────┘
```

### Stage별 상세

| Stage | 이름 | 설명 | 구현 위치 | GPU 가속 |
|-------|------|------|-----------|----------|
| **1** | DataAdapter | DuckDB-native raw data load, AdapterRegistry 기반 데이터셋별 원시 로딩 | `core/pipeline/adapter.py` | - |
| **1.5** | TemporalPrep | 시퀀스 절단 (drop last month), prod_* 재계산 (month 16), cold start 고객 처리 | `core/pipeline/temporal_split.py` | - |
| **2** | SchemaClassifier (5-axis) | 모든 컬럼을 State/Snapshot/Timeseries/Hierarchy/Item 축으로 분류 | `core/pipeline/schema_classifier.py` | - |
| **3** | EncryptionPipeline | PII 컬럼 → SHA256 domain-specific salt → INT32 global index | `core/security/pipeline.py` | - |
| **4** | FeatureGroupPipeline + Normalization | 8개 Generator 실행 (cuDF primary, pandas fallback) + PowerLawAwareScaler | `core/feature/` | cuDF/cuPY |
| **5** | LabelDeriver (13 tasks) | Config-driven 레이블 생성 — direct/bucket/weighted_sum/product_group 등 | `core/pipeline/label_deriver.py` | - |
| **5.5** | LeakageValidator + auto-drop | 시퀀스/상관관계/제품/시간 4중 누수 검증, >0.95 상관 피처 자동 제거 | `core/pipeline/leakage_validator.py` | - |
| **6** | SequenceBuilder | Time-based + sliding window → 3D 텐서 (txn_day_offset_seq 기반) | `core/pipeline/sequence_builder.py` | - |
| **7** | DataLoader | Cross-sectional auto-detect → random split / temporal split, PyArrow 로딩 | `containers/training/train.py` | - |
| **8** | PLETrainer (2-phase) | PLE + adaTT + 7 heterogeneous experts, AMP FP32 loss, VRAM diagnostics | `core/training/trainer.py` | GPU required |
| **8.5** | Model Analysis | IG, CCA, Gate Analysis, HGCN Interpretable, Multi Interpreter, Template Engine, XAI Quality, Model Card | `core/analysis/` | GPU partial |
| **9** | StudentTrainer (Distillation) | PLE teacher → LGBM students (soft label + fidelity validation) | `core/distillation/` | - |
| **9.5** | Context Vector Store | 추천 사유 임베딩 저장소 (RAG retrieval) | `core/serving/context_store.py` | - |
| **10** | CPE + Agentic Orchestrator | FD-TVS scoring, DNA modifier, L1+L2a+L2b 추론, constraint engine | `core/serving/` | - |

---

## On-Prem (현재) vs AWS (목표) 아키텍처 대비

```
[On-Prem]                              [AWS]
──────────────────────                 ──────────────────────
로컬 파일 / GCS                        S3 Data Lake
    ↓                                      ↓
DuckDB (인프로세스)                     DuckDB (전 구간 사용, cuDF 가속)
    ↓                                      ↓
Airflow 86 DAGs (Docker)               Step Functions 5개
    ↓                                      ↓
로컬 GPU 학습                          SageMaker Training (Spot)
    ↓                                      ↓
MLflow (Docker)                        SageMaker Experiments
    ↓                                      ↓
FastAPI + Docker Compose               Lambda (기본) / ECS (대규모)
    ↓                                      ↓
Great Expectations (로컬)              CloudWatch + SageMaker Monitor
```

## 서비스별 매핑 상세

| 관점 | On-Prem (현재) | AWS (목표) | 전환 근거 |
|------|---------------|-----------|----------|
| **저장소** | 로컬 Parquet + GCS | S3 (버전 관리 활성화) | 내구성, 비용, IAM 통합 |
| **쿼리 엔진** | DuckDB 전용 | DuckDB 전용 (단일 머신 최강) | 수백 GB까지 DuckDB로 충분. TB급 필요 시 Athena 옵션 |
| **오케스트레이션** | Airflow 86 DAGs (상시 가동) | Step Functions 5개 (실행당 과금) | $300/월 → $0/월 |
| **학습** | 로컬 GPU (전용 머신) | SageMaker Training Job (Spot) | 70% 비용 절감, 자동 체크포인트 |
| **실험 관리** | MLflow (Docker) | SageMaker Experiments | 서버 유지 비용 제거 |
| **서빙** | FastAPI + Docker Compose (상시) | Lambda (기본) / ECS Fargate (대규모) | 규모별 자동 전환 |
| **피처 스토어** | 로컬 파일 | Lambda 메모리 (기본) / DynamoDB (대규모) | 규모별 자동 전환 |
| **데이터 검증** | Great Expectations (로컬) | SageMaker Processing + GX | 동일 로직, 실행 환경만 변경 |
| **감사/리니지** | 커스텀 audit_logger + DVC | CloudTrail + S3 버전관리 + SageMaker Lineage | AWS 네이티브 통합 |
| **모니터링** | 커스텀 drift_monitor | SageMaker Model Monitor + CloudWatch | 관리형 서비스, 알림 자동화 |
| **감시 에이전트** | 룰 엔진 전용 (결정론적), LLM 없음 | 룰 엔진 + Bedrock Sonnet 다이얼로그 + 3-에이전트 합의 + Solar 추천사유 | LLM 기반 자동 진단·규제 감사 |
| **암호화** | encryption_config.yaml 별도 | `core/security/` 통합 파이프라인 (SHA256 + INT32) | Stage 3에서 자동 처리 |

---

## 5-Axis Feature Classification 체계

파이프라인의 핵심 설계 결정 — 모든 피처를 5개 축으로 분류하고, 각 축에 대응하는 Feature Generator와 Expert를 매핑한다. **이 매핑은 현재 FeatureRouter를 통해 런타임에 실제로 강제된다** — `feature_groups.yaml`의 `target_experts` 선언이 전문가별 입력 차원을 결정한다.

```
            ┌────────────────────────────────────────────────────────┐
            │              5-Axis Feature Classification             │
            ├──────────┬──────────┬──────────┬──────────┬───────────┤
            │  State   │ Snapshot │Timeseries│Hierarchy │   Item    │
            │ (정적)    │ (장기)   │ (단기)    │ (구조)   │ (관계)    │
            ├──────────┼──────────┼──────────┼──────────┼───────────┤
 Features   │ 인구통계 │ TDA글로벌│ TDA로컬  │ Poincaré │ 고객×상품 │
            │ RFM      │ HMM 전이 │ Mamba SSM│ MCC L1/L2│ bipartite │
            │          │ 상품트렌드│ PatchTST │ 상품계층  │ LightGCN  │
            ├──────────┼──────────┼──────────┼──────────┼───────────┤
 Experts    │ DeepFM   │ PersLay  │ Temporal │ HGCN     │ LightGCN  │
            │ MLP      │ (global) │ Ensemble │          │           │
            │ AutoInt  │ Causal   │ Mamba    │          │           │
            │          │ OT       │ LNN      │          │           │
            ├──────────┼──────────┼──────────┼──────────┼───────────┤
 라우팅 차원 │ 109D     │ 32D      │ 129D     │ 34D      │ 66D       │
 (FeatureRouter) │ (deepfm) │ (perslay) │ (temporal) │ (hgcn) │ (lightgcn) │
            ├──────────┼──────────┼──────────┼──────────┼───────────┤
 GPU 가속   │ -        │ cuPY     │ GPU      │ GPU      │ GPU       │
            │          │ (TDA)    │ (Mamba)  │ (HGCN)   │ (GCN)     │
            └──────────┴──────────┴──────────┴──────────┴───────────┘
```

> causal (103D), optimal_transport (69D) 는 State+Snapshot 복합 입력. mlp (task expert) 는 51D 라우팅 입력.

### 축별 특성

| 축 | 시간 의존성 | 변화 속도 | 대표 데이터 | 처리 방식 |
|----|-----------|-----------|------------|----------|
| **State** | 없음 (정적) | 거의 변하지 않음 | 나이, 성별, 가입일 | 피처 상호작용 (FM) |
| **Snapshot** | 장기 (월/분기) | 느림 | 12개월 거래 위상, HMM 상태 | 장기 패턴 추출 |
| **Timeseries** | 단기 (일/주) | 빠름 | 최근 90일 거래 시퀀스 | 시퀀스 모델링 (SSM) |
| **Hierarchy** | 없음 (구조적) | 느림 | MCC 코드 계층, 상품 카테고리 | 쌍곡 임베딩 |
| **Item** | 없음 (관계적) | 중간 | 고객-상품 상호작용 | 그래프 협업 필터링 |

---

## 13-Task Architecture (4 Semantic Groups)

Santander 데이터셋 기준 13개 태스크가 4개 의미 그룹으로 구성된다.
> **2026-04-12 변경**: has_nba (binary) → nba_primary (multiclass) 통합. nba_primary class 0 = no NBA. 18→14→13 task로 축소. binary: 7개, multiclass: 3개, regression: 3개.

| Group | 질문 | 태스크 | 개수 |
|-------|------|--------|------|
| **engagement** | 고객이 반응하는가 | next_mcc, top_mcc_shift | 2 |
| **lifecycle** | 고객이 어디에 있는가 | churn_signal, product_stability, segment_prediction | 3 |
| **value** | 언제/어디서 가치를 만드는가 | mcc_diversity_trend, cross_sell_count | 2 |
| **consumption** | 무엇을 소비하는가 | nba_primary, will_acquire_{deposits,investments,accounts,lending,payments} | 6 |

### Logit Transfer 2 Edges

```
churn_signal → product_stability (이탈→안정성)
next_mcc → nba_primary (다음업종→다음상품)
```

> has_nba→nba_primary 전이 엣지 제거됨 (has_nba 통합으로 중복)

---

## 암호화 파이프라인 통합 (Stage 3)

`core/security/` 모듈이 Stage 3을 담당한다. 스키마의 `pii: true` 마킹에서 자동으로 암호화 정책을 유도한다.

```
Schema (pii: true)
    ↓ derive_from_schema()
EncryptionPolicy (per source, per column)
    ↓
EncryptionPipeline.process_source()
    ├── Step 1: Drop (phone, email, SSN 등 contact/personal_id)
    ├── Step 2: SHA256 Hash (domain-specific salt)
    │           PIIDomain: CUSTOMER, ACCOUNT, CARD, MERCHANT, ...
    │           SaltManager: AWS Secrets Manager or local
    ├── Step 3: Integer Index (hash BLOB → INT32 global index)
    │           PIIIntegerIndexer: append-only, Parquet 저장
    └── Step 4: Audit report
```

| 컴포넌트 | 파일 | 역할 |
|---------|------|------|
| `PIIDomain` | `core/security/domains.py` | 16개 PII 도메인 정의 + 컬럼 자동 매핑 |
| `SaltManager` | `core/security/salt_manager.py` | 도메인별 salt 관리 (Secrets Manager / 로컬) |
| `PIIEncryptor` | `core/security/encryptor.py` | SHA256(salt + value) 해싱 |
| `PIIIntegerIndexer` | `core/security/integer_indexer.py` | Hash → INT32 매핑 (S3 Parquet 영속) |
| `EncryptionPipeline` | `core/security/pipeline.py` | 전체 오케스트레이션 |

---

## 모델 아키텍처 핵심 구성요소

| 구성요소 | 설명 | 구현 위치 |
|---------|------|-----------|
| **Expert Basket** | Pool → Basket → CGC 3계층 선택 | `core/model/ple/experts.py` |
| **FeatureRouter** | feature_groups.yaml target_experts 기반 전문가별 이종 입력 슬라이싱 — build_model()에서 자동 생성 (활성화) | `core/model/ple/model.py` |
| **CGC + Attention** | 태스크별 Expert 가중 결합 + dim_normalize | `core/model/ple/experts.py` |
| **adaTT** | Adaptive Task Transfer (intra/inter group) | `core/model/ple/adatt.py` |
| **Logit Transfer** | 3-method dispatch (output_concat/hidden_concat/residual) | `core/model/ple/model.py` |
| **HMM Triple-Mode** | journey/lifecycle/behavior → 태스크 그룹별 라우팅 | `core/model/ple/model.py` |
| **Multidisciplinary Routing** | 24D → 4 x 6D per task group | `core/model/ple/model.py` |
| **Evidential Deep Learning** | Beta/Dirichlet/NIG 불확실성 (config-gated) | `core/model/layers/evidential.py` |
| **SAE Regularization** | Sparse Autoencoder (detached, config-gated) | `core/model/layers/sae.py` |
| **Per-task focal_alpha** | positive rate 기반 calibrated alpha | config per task |
| **Uncertainty Weighting** | Kendall et al. learnable log_var | `core/model/ple/loss_weighting.py` |
| **TowerRegistry** | standard/contrastive tower 플러그인 | `core/model/ple/model.py` |

---

## 해석 가능성 파이프라인 (Stage 8.5 → 10)

### Stage A: 모델 분석
| 분석 | 목적 | 출력 |
|------|------|------|
| **Integrated Gradients (IG)** | 피처 기여도 측정 | attribution scores |
| **Expert Redundancy CCA** | Expert 간 중복성 검출 | CCA correlation matrix |
| **CGC Gate Analysis** | 태스크별 Expert 가중치 분석 | attention heatmap |
| **HGCN Interpretable** | 계층 구조 설명 | hierarchy paths |

### Stage B: 추론 사유 생성
| 분석 | 목적 | 출력 |
|------|------|------|
| **Multi Interpreter** | 다학제 해석 통합 | structured reasons |
| **Template Reason Engine** | 자연어 추천 사유 | text templates |
| **XAI Quality Evaluator** | 설명 품질 평가 | quality scores |
| **Model Card** | 모델 문서 자동 생성 | model_card.json |

### Stage C: 서빙 파이프라인
| 컴포넌트 | 목적 | 출력 |
|---------|------|------|
| **CPE (Context-Personalized Engine)** | 개인화 스코어링 | FD-TVS scores |
| **Agentic Orchestrator** | L1+L2a+L2b 추론 체인 | final recommendations |
| **Context Vector Store** | RAG 기반 사유 검색 | context embeddings |

---

## Monitoring & Audit Artifacts

### 파이프라인 추적
| Artifact | 위치 | 용도 |
|---------|------|------|
| `pipeline_manifest.json` | output_dir/ | 전체 파이프라인 config 스냅샷 |
| `pipeline_state.json` | output_dir/ | Stage별 완료/실패 상태, resume 지원 |

### Per-stage 체크포인트
| Stage | Artifact | 형식 |
|-------|---------|------|
| Feature | `features.parquet` | Parquet |
| Label | `labels.parquet` | Parquet |
| Sequence | `sequences.npy`, `seq_lengths.npy` | NumPy |

### Audit Artifacts
```
audit/
├── schema/          ← 스키마 검증 결과
├── encryption/      ← PII 처리 감사 로그
├── scaler/          ← scaler_params.json
├── labels/          ← label_transforms.json
├── leakage/         ← LeakageValidator 결과
└── fidelity/        ← 증류 fidelity 검증
```

위 아티팩트들은 두 자율 에이전트에 의해 실시간으로 해석된다.

- **OpsAgent**: Collect → Diagnose → Report (finding + cause + action) 3단계 루프로 파이프라인 성능·안정성·비용을 7개 체크포인트에서 진단한다.
- **AuditAgent**: 공정성(fairness), 추천사유 품질(reason quality), 규제 준수(regulatory compliance) 등 5개 관점에서 감사 아티팩트를 검토한다.
- **3-에이전트 합의 (Sonnet×3)**: 환각(hallucination) 완화를 위해 동일 진단을 독립적으로 3회 실행 후 다수결로 결론을 확정한다.
- **DiagnosticCaseStore (LanceDB)**: 과거 진단 사례를 벡터 DB에 누적하여 유사 장애 패턴의 자동 검색 및 재사용을 지원한다.

→ 상세 설계: `docs/design/11_ops_audit_agent.md`

### Analysis Artifacts
```
analysis/
├── ig/              ← Integrated Gradients
├── cca/             ← Expert Redundancy CCA
├── gate/            ← CGC Gate weights
├── hgcn/            ← HGCN interpretable paths
├── multi/           ← Multi Interpreter
├── template/        ← Template Reason Engine
├── xai/             ← XAI Quality scores
└── model_card/      ← Model Card
```

### Serving Artifacts
```
serving/
├── cpe/             ← CPE scores
├── reasons/         ← Generated recommendation reasons
└── context_store/   ← Vector store embeddings
```

---

## cuDF/cuPY/DuckDB GPU+CPU 가속 체계

| Stage | 대상 | CPU 경로 | GPU 경로 | 가속 효과 |
|-------|------|---------|---------|----------|
| 1 | 데이터 로딩/집계 | DuckDB (primary) | cuDF (선택적) | DuckDB native adapter, pandas 없음 |
| 4 | Generator 실행 (TDA/HMM/GMM 등) | pandas fallback | cuDF primary (cuML for GMM) | Generator output: cuDF or pandas DataFrame |
| 4 | TDA persistence diagram | ripser (NumPy) | cuPY + ripser | 5-10x |
| 4 | StandardScaler (mean/std) | NumPy | cuPY | 3-5x on 100M+ rows |
| 7 | Training 데이터 로딩 | PyArrow (zero-copy parquet) | - | pandas 완전 제거 (hot path) |
| 8 | Model training | PyTorch CPU | PyTorch CUDA + AMP | Required, FP32 loss computation |

GPU 가속은 선택적이며, cuDF/cuPY 미설치 시 CPU 경로로 자동 폴백한다.
Phase 0 Generator는 cuDF primary → pandas fallback 패턴을 따른다.

---

## Santander 4-Dimension Ablation Framework

`scripts/run_santander_ablation.py`가 6-Phase 48 시나리오 ablation을 오케스��레이션한다. 모든 시나리오는 config에서 동적 생성된다 (bottom-up + top-down 학계 표준 설계):

| Phase | 내용 | Job 수 |
|-------|------|--------|
| **0** | Data Preparation (Processing Job) | 1 |
| **1** | Feature Group Ablation (full + base_only + bottom-up + top-down) | 16 |
| **2** | Expert Ablation (deepfm baseline + bottom-up + top-down + mlp_only) | 16 |
| **3** | Task x Structure Cross (4 tiers x 4 structures) | 16 |
| **4** | Best-Config Teacher + Distillation | 2 |
| **5** | Analysis + HTML Report | 1 |

4개 Ablation 차원:
1. **Feature** — Bottom-up (base+X) + Top-down (full-X) → 독립 기여 vs irreplaceability 측정
2. **Expert** — DeepFM baseline + pairwise 추가/제거 (피처-전문가 연동)
3. **Task Scaling** — 태스크 수 3→5→10→13, 구조 변형 (shared_bottom/ple_only/adatt_only/full)
4. **PLE-adaTT Structure** — Loss weighting, PLE depth, adaTT strength

Docker-based ablation runner가 SageMaker local mode를 지원한다 (`containers/training/Dockerfile`).

---

## PipelineRunner 통합 아키텍처

### 10-Stage PipelineRunner

`core/pipeline/runner.py`의 `PipelineRunner`가 전체 파이프라인을 단일 진입점으로 통합한다.

```
PipelineRunner.run()
    │
    ├── Stage 1: DataAdapter.load_raw()
    │   └── AdapterRegistry.build("santander", config)
    │       └── 941K user-level data 로드
    │
    ├── Stage 1.5: TemporalPrep
    │   ├── 시퀀스 절단 (17개월 → 16개월, drop last month)
    │   └── prod_* 컬럼 재계산 (month 16 state)
    │
    ├── Stage 2: SchemaClassifier (5-axis)
    │
    ├── Stage 3: EncryptionPipeline (SHA256 → INT32)
    │
    ├── Stage 4: FeatureGroupPipeline + Normalization
    │   └── TDA, HMM, Mamba, Graph generators + PowerLawAwareScaler
    │
    ├── Stage 5: LabelDeriver (13 tasks)
    │   └── direct/bucket/weighted_sum/product_group/sequence_next 등
    │
    ├── Stage 5.5: LeakageValidator
    │   └── sequence/correlation/product/temporal 4중 검증
    │
    ├── Stage 6: SequenceBuilder (flat → 3D tensors)
    │
    ├── Stage 7: DataLoader (temporal split, gap_days=30)
    │
    ├── Stage 8: PLETrainer (2-phase)
    │   └── PLE + adaTT + Evidential + SAE
    │
    ├── Stage 8.5: Model Analysis
    │   └── IG, CCA, Gate, HGCN, Multi Interpreter, Template, XAI, Model Card
    │
    ├── Stage 9: StudentTrainer (distillation)
    │   └── PLE teacher → LGBM students + fidelity validation
    │
    ├── Stage 9.5: Context Vector Store (RAG)
    │
    └── Stage 10: CPE + Agentic Reason Orchestrator
        └── FD-TVS scoring, DNA modifier, constraint engine
```

### DataAdapter 패턴

```python
# core/pipeline/adapter.py
class DataAdapter(ABC):
    """데이터셋별 원시 데이터 로딩 계약.
    각 데이터셋은 load_raw()를 구현하여 entity-level DataFrame을 반환.
    피처 엔지니어링은 수행하지 않음 — FeatureGroupPipeline 담당.
    """
    def load_raw(self) -> Dict[str, pd.DataFrame]: ...
```

**현재 등록된 어댑터:**

| 이름 | 파일 | 데이터 | 집계 방식 |
|------|------|--------|----------|
| `ealtman2019` | `adapters/ealtman2019_adapter.py` | 24M 신용카드 거래 (2K users, 6,146 cards) | DuckDB → ~469D features + 16 labels |
| `santander` | `adapters/santander_adapter.py` | 941K 사용자 × 89 컬럼 | DuckDB-native pipeline (pandas only at generator boundary), cold start 처리 포함 |

**Santander Adapter 특징:**
- Phase 0 `__main__` 블록이 전체 데이터를 DuckDB in-memory 테이블로 유지
- Generator 호출 시에만 pandas 변환 (generator interface boundary)
- Quality-gate, normalization, dtype downcasting, feature stats, label stats, final parquet writes 모두 DuckDB SQL
- Cold start 고객 처리: `is_cold_start` 플래그 + sequence-derived feature zeroing

### Training 진입점

`containers/training/train.py`는 SageMaker Training Job의 단일 진입점이다:

| 기능 | 설명 |
|------|------|
| **데이터 로딩** | PyArrow (zero-copy parquet), pandas 없음 (hot path) |
| **Split 전략** | Cross-sectional auto-detect (>80% 동일 date → random split) 또는 temporal split |
| **LeakageValidator** | 학습 전 자동 호출, >0.95 상관 피처 auto-drop |
| **VRAM 진단** | 학습 시작 전 GPU name/VRAM/compute capability 로깅, epoch별 VRAM 추적 |
| **AMP** | FP16 forward + FP32 loss computation (overflow 방지) |
| **Per-task val mask** | `data.split_strategy`에서 task별 val_method (temporal_latest/random) 지원 |

**Docker SageMaker Local Mode:** `containers/training/Dockerfile`로 로컬 GPU PC에서 SageMaker 환경 동일하게 실행 가능.

---

## 핵심 설계 결정 요약

| 결정 | 선택 | 근거 |
|------|------|------|
| 쿼리 엔진 | DuckDB 단일 (Athena는 옵션) | 단일 머신 최강, 수백 GB까지 충분 |
| 피처 분류 | 5-Axis (State/Snapshot/Timeseries/Hierarchy/Item) | Expert 라우팅의 명시적 기반 — FeatureRouter로 런타임 강제 (4.77M→~2.8M 감소) |
| 태스크 아키텍처 | 13 tasks in 4 semantic groups | adaTT intra/inter transfer 기반; has_nba → nba_primary 통합(2026-04-12) |
| 데이터 분할 | Cross-sectional auto-detect → random split / Temporal split + gap_days | 자동 감지 (>80% 동일 date → random) |
| 누수 방지 | 시퀀스절단 + prod재계산 + LeakageValidator | 4중 검증 |
| 모델 구조 | PLE + adaTT + Evidential + SAE | 불확실성 정량화 + 해석 가능성 |
| Loss 전략 | Per-task dispatch (focal_alpha calibrated) + Uncertainty weighting | config 선언적, 자동 밸런싱 |
| CGC | dim_normalize=True | Expert 출력 차원 불균형 보정 |
| Logit Transfer | 3-method dispatch (output_concat/hidden_concat/residual) | 태스크 관계별 최적 방법 |
| 서빙 | FD-TVS scoring + DNA modifier + constraint engine | 규제 준수 추천 |
| 해석 가능성 | 3-stage (A: 분석, B: 사유생성, C: 서빙) | 감사 가능한 추천 |
| Ablation | 4-Dimension (Feature/Expert/Task/Structure) × 48 scenarios | 체계적 실험 (bottom-up + top-down) |

---

## 설계서 구성

| 문서 | 내용 |
|------|------|
| [01_data_layer](01_data_layer.md) | DataAdapter, TemporalPrep, 암호화, 5-axis 분류, temporal split, LeakageValidator |
| [02_feature_engineering](02_feature_engineering.md) | 5축별 피처 매핑, TDA/HMM/Mamba/Graph/LightGCN, 정규화, LabelDeriver |
| [03_model_architecture](03_model_architecture.md) | PLE, Expert Basket, CGC, adaTT, Logit Transfer, HMM routing, Evidential, SAE, 13 tasks |
| [04_training_pipeline](04_training_pipeline.md) | PLETrainer, 4-Dimension Ablation, Santander 36-job, Distillation, Interpretability |
| [05_serving_and_testing](05_serving_and_testing.md) | Lambda↔ECS 자동 전환, 실시간 추론, A/B 테스트 |
| [06_orchestration_and_audit](06_orchestration_and_audit.md) | Step Functions 5개, 3계층 감사, E2E 리니지 |
| [07_cost_analysis](07_cost_analysis.md) | 규모별 비용 시뮬레이션, 손익분기점 분석 |
| [08_recommendation_intelligence](08_recommendation_intelligence.md) | FD-TVS 스코어링, 추천 사유 3계층, 규제 준수 |
| [09_compliance_governance](09_compliance_governance.md) | 감사 불변성, 36항목 레지스트리, 공정성, 쏠림, 킬스위치 |
| [10_pool_basket_architecture](10_pool_basket_architecture.md) | Pool/Basket/Runtime 3계층, Expert 11종, Feature Generator, Task Group |

---

### 온프레미스 환경

동일한 파이프라인이 로컬 GPU(RTX 4070, 64GB RAM)에서 실행된다. Bedrock 대신 Exaone 3.5 7.8B(사유 생성) + Qwen 2.5 14B Q4(에이전트 합의) 오픈소스 모델을 사용하며, 데이터가 외부로 전송되지 않아 데이터 보호가 구조적으로 보장된다. 코드와 config는 AWS와 동일하고 환경 변수로 자동 분기.
