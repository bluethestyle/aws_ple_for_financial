# AWS ↔ 온프렘 파이프라인 구현 비교 매트릭스

**작성일**: 2026-04-21
**대상**: `aws_ple_for_financial` (main @ 528be7a) vs `gotothemoon/workspace/code` (현 스냅샷)
**방법**: 모델 / 증류 / 서빙 / 감사 4개 레이어별 Sonnet 서브에이전트 병렬 조사 후 통합

**한 줄 요약**: AWS 는 *연구·MRM 인프라 깊이*, 온프렘은 *실운영·국내 규제 깊이*. 두 프로젝트는 중복이 아니라 *상보* 관계 — 병합할 것이 아니라 **양방향 이식 체크리스트**가 필요.

---

## 0. 레이어별 판정 요약

| 레이어 | AWS 우위 건수 | 온프렘 우위 건수 | 양쪽 동등 | 핵심 메시지 |
|---|---|---|---|---|
| **모델 아키텍처** | ~10 | ~4 | ~16 | AWS 가 Paper 3 Findings 8~13 (CEH/CG/CCP/W-amp) + Fusion 5종 + GradSurgery + FeatureRouter 보유. 온프렘은 LiquidNN, KD 모듈, ClusterTaskExpert, Sparse/Top-K gate 추가 |
| **훈련 + 증류** | 5 | 3 | 3 | AWS 가 Teacher threshold gating / 3-Layer FallbackRouter / Platt calibration 독점. 온프렘은 Uplift T-Learner, IG 기반 3-stage FS, DuckDB 통합 |
| **서빙 + 추천** | 1 | 9 | 3 | **온프렘 압도적 우위**. AI기본법/개보법/금소법/금감원 원칙이 모듈별 1:1 매핑. Human Review Queue, Dynamic Item Universe, Safety Gate 3-layer, Consent 4-모듈 모두 온프렘 전용 |
| **모니터링 + 감사** | 3 | 9 | 2 | **온프렘 압도적 우위**. FRIA (AI기본법 §35), EU AI Act Annex IV, 36-항목 compliance registry, MLflow+DVC, Data Lineage 722D 모두 온프렘. AWS 는 WORM 저장소 + `log_attribution`/`log_guardrail` + Agent 자연어 조회만 독점 |

---

## 1. 모델 아키텍처 레이어

### 1.1 AWS-Only 컴포넌트 (온프렘 이식 후보)

| 컴포넌트 | AWS 위치 | Paper 3 연계 | 이식 권장도 |
|---|---|---|---|
| **CEH attribution_head + target_mode (raw/demeaned/primary_task)** | `core/model/experts/causal.py` | Findings 9, 13 | **High** |
| **get_counterfactual (Pearl Rung 3)** | 동 | Finding 12 | **High** |
| **get_causal_coherence_score (CG v2)** | 동 | Findings 10, 11 | **High** |
| **w_init_scale config-driven** | 동 | Finding 11 | **High** |
| **NEAS** (neglected-expert aux supervision) | `core/model/ple/config.py::NEASConfig`, model.py | Finding 7 positive | **High** |
| **BRP / BRP-detached** | `core/model/ple/config.py::BRPConfig`, model.py | Finding 7 positive | **High** |
| **ECEB** (entropy-conditioned expert bank) | `core/model/ple/config.py::ECEBConfig` | Finding 7 negative | Low |
| **AdaTT-sp (표현-레벨)** | `core/model/ple/experts.py::CGCLayer(fusion_type="adatt_sp")` | Finding 7 negative | Low |
| **ResidualComplement / M1** | `CGCLayer(fusion_type="residual_complement")` | Finding 7 negative | Low |
| **GradSurgery** | `core/model/ple/grad_surgery.py` | Memory: 미채택 | **Skip** |
| **FeatureRouter** | `core/model/ple/feature_router.py` | 인프라 | Medium (CEH accessor 의존성) |

### 1.2 온프렘-Only 컴포넌트 (AWS 역수입 후보)

| 컴포넌트 | 온프렘 위치 | 가치 |
|---|---|---|
| **LiquidNeuralNetwork Expert** | `src/models/experts/liquid_neural_network.py` | 시계열 표현 추가 옵션 — 필요 시 이식 |
| **KnowledgeDistillation 모듈** | `src/models/knowledge_distillation.py` | Teacher-Student + Feature Projection — AWS containers/distillation 이 부분적으로 커버하지만 온프렘은 `nn.Module` 로 직접 구현 |
| **ClusterTaskExpert (GMM 클러스터 서브헤드)** | `src/models/experts/cluster_task_expert.py` | 메모리: "K×T 복잡도 폭발로 드랍" — 참조만 |
| **Sparse/Top-K gating + LoadBalance Loss** | `src/models/gating_networks.py` | MoE 전통적 load balancing — AWS 에 없음 |
| **Evidential 결측 마스크 방어 코드** | `src/models/layers/evidential_layer.py` (target ≥ 0 등) | 실데이터 안정성 — **AWS 로 역수입 가치 있음** |
| **HMM config 기반 동적 라우팅** (`set_hmm_routing`) | `src/models/experts/temporal_ensemble.py` | Config single-source-of-truth — AWS 역수입 가치 있음 |

### 1.3 양쪽 있지만 구조 다른 것

| 영역 | AWS | 온프렘 | 판정 |
|---|---|---|---|
| **Causal Expert 기본** | NOTEARS + recon_lambda + init 0.1 | 동등 (양쪽 Finding 8 패치 적용) | 동등 |
| **ExpertRegistry** | 데코레이터 패턴 | Factory + Registry 분리 | 온프렘 약간 우위 |
| **Evidential Layer** | 기본 | + valid_mask 결측 필터링 | **온프렘 우위** |
| **SparseAutoencoder** | + get_feature_attributions | 기본 | **AWS 우위** |
| **CGC Attention** | 어텐션 가중치 투명 노출 | Bayesian prior 테이블 하드코딩 | AWS 우위 |
| **adaTT** | 단일 `_apply_transfer` DRY | FALLBACK 기본값 + 버그픽스 태그 | 상보 |

---

## 2. 훈련 + 증류 레이어

### 2.1 AWS-Only (이식 권장)

| 컴포넌트 | AWS 위치 | 역할 | 이식 권장도 |
|---|---|---|---|
| **Teacher threshold gating** | `containers/distillation/threshold_gate.py` | task별 DISTILL/DIRECT/SKIP 자동 라우팅 (binary AUC>0.6, MC F1>2/K, reg R²>0.05) | **High** (SR 11-7 MRM) |
| **3-Layer FallbackRouter** | `core/recommendation/fallback_router.py` | distilled LGBM → direct LGBM → rule engine | **High** (서비스 연속성) |
| **Platt scaling / isotonic** | `containers/distillation/calibration.py` | 확률-필요 task 사후 보정 | **High** (확률 해석성) |
| **LeakageValidator (standalone)** | `core/pipeline/leakage_validator.py` | scaler train-only + temporal gap + sequence overlap | Medium (온프렘은 method 형태로 존재) |
| **Training loop 복원력** | `core/training/trainer.py` | OOM skip + NaN skip + GradScaler tail-flush + VRAM 진단 | Medium (운영 안정성) |

### 2.2 온프렘-Only (AWS 역수입 후보)

| 컴포넌트 | 온프렘 위치 | 가치 |
|---|---|---|
| **Uplift T-Learner** | `src/distillation/lgbm_student.py::train_uplift_student` | 처치 효과 추정 — AWS 에 없음. Paper 2 v2 에 Rung 2 관련 보강 가능 |
| **IG 기반 3-stage Feature Selection** | `src/distillation/feature_selector.py` (Stage1 IG top-200 → Stage2 LGBM gain bottom-30% 제거 → Stage3 도메인 필수 피처) | 더 정교한 feature selection. **AWS 역수입 가치 있음** |
| **DuckDB soft label 저장** | 동 | 대용량 처리 |
| **Airflow DAG pre-trigger 검증** | `src/airflow_dags/dag_weekly_model_competition.py` (drift + 성능 + 주기 3조건 AND) | 불필요 재학습 방지 |

### 2.3 알려진 리스크

- **온프렘 LGBM student random split (TODO C6)**: temporal split 미적용 → 잠재적 leakage. **즉시 수정 필요**
- **온프렘에 calibration 없음**: 확률값 필요 task 에 보정 미적용

---

## 3. 서빙 + 추천 레이어

### 3.1 AWS-Only (1개만)

| 컴포넌트 | AWS 위치 | 역할 |
|---|---|---|
| **Champion-Challenger ModelMonitor (온라인)** | `core/serving/model_monitor.py` | Two-proportion z-test 기반 실시간 승격 판정 |

### 3.2 온프렘-Only (AWS 역수입 강력 권장)

| 컴포넌트 | 온프렘 위치 | 법적 근거 | 역수입 우선순위 |
|---|---|---|---|
| **Human Review Queue** | `src/recommendation/human_review_queue.py` | 금융위 AI 가이드라인 제3원칙 (AI 보조, 사람 판단) | **High** |
| **Kill Switch** | `src/recommendation/kill_switch.py` | 운영 긴급 대응 | **High** |
| **Dynamic Item Universe Loader** | `src/recommendation/dynamic_item_universe_loader.py` | G3 캠페인 / G6 상품 실시간 로딩 | **High** |
| **Marketing Consent 4-모듈** | `marketing_consent.py` + `ai_decision_opt_out.py` + `profiling_rights_manager.py` + `explanation_sla_tracker.py` | 개보법 §22·§37의2, 신정법 §36의2, 정통망 §50, AI기본법 §31·§34 | **High** (규제) |
| **Explanation SLA Tracker** | `src/recommendation/explanation_sla_tracker.py` | 개보법 시행령 §44의2~4 (10일 SLA) | **High** (규제) |
| **Constraint 금소법 §17 매핑** | `checkcard_constraints.py` | 금융소비자보호법 적합성 원칙 | Medium |
| **L2a Safety Gate 3-layer (파싱→Rule→품질)** | `src/grounding/l2a_rewrite_engine.py` | 할루시네이션 방어 | Medium |
| **Audit Archive 확장 컬럼** (thinking_trace, hallucination_flags, tools_used) | `src/grounding/recommendation_audit_archiver.py` | 민원 대응 AI 판단 과정 재현 | **High** |
| **LLM_GENERATION_MARKER 자동 삽입** | `src/grounding/agentic_reason_orchestrator.py` | AI기본법 §31·§34조 (AI 생성 표시 의무) | **High** |

### 3.3 핵심 리스크 (AWS 에서)

- **Consent / Opt-out 이 `predict()` 내 optional injection**: 미주입 시 규제 위반. **의무화 필요**
- **Batch 추론이 단순 for문** (`predict_batch`): 대용량 처리 비현실적
- **아이템 유니버스 주입 방식**: caller 책임으로 남김 → 실운영 복잡도 상승

---

## 4. 모니터링 + 컴플라이언스 + 감사 레이어

### 4.1 AWS-Only (3개)

| 컴포넌트 | AWS 위치 | 핵심 가치 |
|---|---|---|
| **AuditLogger WORM 저장** | `core/monitoring/audit_logger.py` + S3 Object Lock (GOVERNANCE/COMPLIANCE, 2555일) | **불변 감사 기록** — 물리 변조 방지 |
| **log_attribution + log_guardrail** | 동 (Paper 2 v2) | per-prediction attribution_hash + coherence_score 저장. forensic replay 지원 |
| **OpsAgent / AuditAgent + Bedrock Dialog** | `core/agent/` | 자연어 기반 감사 쿼리 — "regulator queryability" |

### 4.2 온프렘-Only (AWS 역수입 강력 권장)

| 컴포넌트 | 온프렘 위치 | 법적 근거 / 가치 |
|---|---|---|
| **FRIA (Fundamental Rights Impact Assessment)** | `src/monitoring/fria_evaluator.py` | AI기본법 §35②③ (국가기관 강화), 시행령 §27, 5년 보존 |
| **EU AI Act Annex IV 12-항목 기술문서 매핑** | `src/monitoring/eu_ai_act_mapper.py` | Art.11 기술문서 요건, Art.43 내부 적합성 평가 시뮬 |
| **36-항목 Compliance Registry** | `src/monitoring/regulatory_compliance_checker.py` | A-group 18 + GAP 18, 금감원 현장 검사 대응 |
| **AI Risk Classifier (금감원 6-차원)** | `src/monitoring/ai_risk_classifier.py` | 금감원 AI RMF — 데이터민감도/자동화수준/영향범위/모델복잡도/외부의존도/공정성 |
| **Data Lineage 722D ↔ 64 Tables 매핑** | `src/monitoring/data_lineage_tracker.py` | AI기본법 §34 학습데이터 출처 + 가명처리 추적 |
| **MLflow + DVC Compliance Integration** | `src/monitoring/compliance_mlflow_logger.py`, `compliance_dvc_tracker.py` | 규제 산출물 버전관리 + 메트릭 시계열 |
| **ComplianceAuditStore** (DuckDB Parquet 중앙 보관) | `src/monitoring/compliance_audit_store.py` | regulator queryability via SQL |
| **Fairness metrics 영속화** (archive_metrics) | `src/monitoring/fairness_monitor.py` | 임의 기간 이력 쿼리 |
| **Drift DuckDB Parquet + 마크다운 리포트** | `src/monitoring/drift_detector.py` + `drift_impact_analyzer.py` | 감사 친화적 저장 |

### 4.3 양쪽 있지만 구조 다른 것

| 컴포넌트 | AWS | 온프렘 | 판정 |
|---|---|---|---|
| **Champion-Challenger 오프라인** | paired t-test + paired bootstrap(2000회), `auto_promote=True` 기본 | DuckDB 레지스트리, `auto_promote=False` 강제 (EU Art.14), Counterfactual IPS/SNIPS | **온프렘 우위** (인적 감독 + 관측편향 보정) |
| **AuditLogger 해시체인 검증** | `verify_chain` + `verify_chain_from_s3` 원격 검증 | `verify_chain` 로컬만 | AWS 우위 |

### 4.4 핵심 리스크 (온프렘에서)

- **AuditLogger 로컬 파일시스템만**: 물리 서버 접근자가 변조 가능. **WORM 저장소 추가 시급** (MinIO Object Lock 등)
- **log_attribution / log_guardrail 없음**: Paper 2 v2 의 CEH/CG 감사 통합 미반영
- **OpsAgent / AuditAgent 없음**: 감사인이 SQL/코드 직접 실행 필요

---

## 5. 종합 이식 로드맵

### 5.1 AWS → 온프렘 이식 (이미 `onprem_work_plan.md` 에 정리됨)

Phase 1A~1D 의 Must 항목:
- Causal expert 확장 (CEH / CG / CCP / W-amp / target_mode)
- NEAS + BRP-detached (Finding 7 positive recipes)
- FallbackRouter + teacher_threshold gating + Platt calibration
- AuditLogger `log_attribution` + `log_guardrail`
- LeakageValidator 독립 class 화

### 5.2 온프렘 → AWS 역수입 (신규 — 이 매트릭스로 발견됨)

| 우선순위 | 항목 | 대상 AWS 위치 | 이유 |
|---|---|---|---|
| **High** | Human Review Queue + Kill Switch | `core/serving/` 또는 `core/compliance/` | 금융위 AI 가이드라인 제3원칙 |
| **High** | Consent / Opt-out / Profiling Rights / SLA 4-모듈 | `core/compliance/` 확장 | 개보법, 신정법, AI기본법 |
| **High** | FRIA evaluator | `core/compliance/` 또는 `core/monitoring/` | AI기본법 §35 |
| **High** | 36-항목 Compliance Registry | `core/compliance/` | 금감원 현장 검사 |
| **High** | Dynamic Item Universe Loader | `core/recommendation/` | 캠페인 라이프사이클 |
| **High** | Audit Archive 확장 컬럼 (thinking_trace 등) | `core/recommendation/audit_archiver.py` | 민원 대응 재현성 |
| **High** | LLM_GENERATION_MARKER 자동 삽입 | `core/recommendation/reason/` | AI기본법 §31·§34 |
| **Medium** | IG 기반 3-stage Feature Selection | `core/training/feature_selector.py` | Feature 과다 선택 방지 |
| **Medium** | Evidential valid_mask 결측 필터링 | `core/model/layers/evidential.py` | 실데이터 안정성 |
| **Medium** | Uplift T-Learner | `core/training/` 또는 `containers/distillation/` | Paper 2 v2 Rung 2 보강 |
| **Medium** | HMM config 동적 라우팅 (`set_hmm_routing`) | `core/model/experts/temporal.py` | Config single-source |
| **Medium** | Airflow DAG pre-trigger 검증 패턴 | AWS 는 SageMaker 환경이라 직접 이식 X, 로직만 참고 | 불필요 재학습 방지 |
| **Low** | LiquidNN Expert | `core/model/experts/` | 시계열 표현 옵션 |

### 5.3 양쪽 모두 보강 필요

| 항목 | 양쪽 상태 | 조치 |
|---|---|---|
| **WORM 감사 저장소** | AWS S3 Object Lock ✓, 온프렘 파일시스템만 | 온프렘에 MinIO Object Lock 등 추가 |
| **log_attribution / log_guardrail** | AWS ✓, 온프렘 ✗ | 온프렘 AuditLogger 에 이식 |
| **Teacher threshold gating** | AWS ✓, 온프렘 ✗ | 온프렘 Distillation 에 이식 |
| **Platt calibration** | AWS ✓, 온프렘 ✗ | 온프렘 Distillation 에 이식 |
| **온프렘 LGBM student temporal split** | 온프렘 TODO C6 (random split) | **즉시 수정** (leakage 위험) |

---

## 6. 두 프로젝트의 본질적 포지셔닝

이 비교가 확인한 것:

1. **AWS 프로젝트 = 연구 + MRM 인프라 + Paper 생산**
   - Paper 3 Findings 1~13 전체 이식
   - Fusion 실험 5종 (NEAS/BRP/ECEB/AdaTT-sp/M1)
   - MRM/SR 11-7 관련 인프라 집약 (threshold gating, calibration, leakage validator)
   - WORM 저장소 + per-prediction 감사

2. **온프렘 프로젝트 = 실운영 + 국내 규제 + 인적 감독**
   - AI기본법 / 개보법 / 신정법 / 금소법 / 금감원 AI RMF 조항별 모듈
   - Human Review Queue + Kill Switch + Dynamic Item Universe
   - FRIA + 36-항목 레지스트리 + EU AI Act Annex IV
   - DuckDB Parquet 이력 저장 + MLflow + DVC

3. **상보 관계**: 병합하지 말 것. 각자의 강점을 양방향 이식하는 것이 합리적.
   - AWS → 온프렘: 연구 (CEH/CG/CCP/Fusion) + MRM 인프라 (threshold gating, fallback, calibration, WORM 개념)
   - 온프렘 → AWS: 국내 규제 (FRIA, AI기본법 매핑) + 실운영 (Human Review, Kill Switch, Dynamic Item Universe) + 인적 감독 강제

4. **v2 발표 시 협업 구조**:
   - Paper 1 v2 (AWS 쪽): 아키텍처 + 실데이터 ablation
   - Paper 2 v2 (AWS 쪽): 서빙/규제 + **온프렘 규제 모듈 역수입 내용 통합**
   - Paper 3 v2 (AWS 쪽): Loss dynamics + causal reinterpretation + 실데이터 재현
   - 온프렘 백서: 규제 대응 증거 문서 + AWS 연구 성과 참조

---

## 7. 후속 작업 연결

- **이 매트릭스** → `docs/pipeline_comparison_matrix.md` (현재 문서)
- **AWS → 온프렘 이식 계획** → `docs/onprem_work_plan.md`
- **실데이터 검증 체크리스트** → `docs/onprem_v2_validation_checklist.md`
- **온프렘 → AWS 역수입 작업** → *아직 없음, 필요 시 신규 문서 `docs/onprem_to_aws_import_plan.md` 작성*

역수입 문서는 v2 발표가 온프렘 테스트 결과 포함이라는 점 고려 시, 실제 이식 실행은 온프렘 검증 이후가 자연스러움. 지금은 이 매트릭스로 **gap 인식 + 우선순위 공유** 만 확보.
