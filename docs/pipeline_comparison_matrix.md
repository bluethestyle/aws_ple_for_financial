# AWS ↔ 온프렘 파이프라인 구현 비교 매트릭스

**작성일**: 2026-04-21
**대상**: `aws_ple_for_financial` (main @ 528be7a) vs `gotothemoon/workspace/code` (현 스냅샷)
**방법**: 모델 / 증류 / 서빙 / 감사 4개 레이어별 Sonnet 서브에이전트 병렬 조사 후 통합

**한 줄 요약**: AWS 는 온프렘의 **클라우드 확장 버전** — 두 체계는 원칙적으로 *동일해야* 하며, 차이는 (a) 클라우드 vs 온프렘 환경 고유 차이 또는 (b) **AWS 가 더 config-중심 / 모듈화를 지향해서 생긴 구조 차이** 또는 (c) **드리프트 (sync 필요)** 중 하나. 대부분의 gap 은 (c) = 양방향 sync 로 해결.

---

## 0. 포지셔닝 정정 (2026-04-21)

**원래 설계 의도**: AWS 는 온프렘의 **클라우드 확장 버전**. 두 체계는 원칙적으로 동일해야 한다.

**허용되는 차이는 3종류만**:
- **(a) 클라우드 vs 온프렘 인프라 차이** — SageMaker ↔ Airflow, S3 ↔ Parquet + DuckDB 쿼리, Bedrock API ↔ vLLM self-host, S3 Object Lock ↔ MinIO Object Lock
- **(b) 설계 의도 차이** — AWS 는 *더 config-중심 / 모듈화* 지향. 구조가 다르면 AWS 패턴이 "타겟"
- **(c) 드리프트** — 실수 또는 시간차로 한쪽만 구현됨. **양방향 sync 로 해결**

아래 건수는 대부분 (c) 드리프트. "AWS 우위 / 온프렘 우위" 라는 표현은 정확하지 않고, 정확히는 **"현재 어느 쪽에 구현됐나"** 를 나타냄.

| 레이어 | AWS 만 있음 (→ 온프렘으로 sync) | 온프렘만 있음 (→ AWS 로 sync) | 양쪽 구현 (API 정합성 체크) | 인프라 고유 차이 |
|---|---|---|---|---|
| **모델 아키텍처** | ~10 | ~4 | ~16 | (거의 없음 — 순수 코드) |
| **훈련 + 증류** | 5 | 3 | 3 | SageMaker HP ↔ Airflow DAG (인프라) |
| **서빙 + 추천** | 1 | 9 | 3 | Bedrock ↔ vLLM (LLM provider) |
| **모니터링 + 감사** | 3 | 9 | 2 | S3 WORM ↔ MinIO Object Lock |

**핵심**: 온프렘만 있는 항목 (서빙 9 + 감사 9) 이 특히 많다 = AWS 에 규제/운영 모듈이 아직 이식 안 됨 = **AWS 도 보강 필요**. 반대로 AWS 만 있는 Paper 3 Findings 8~13 및 Fusion / Teacher threshold gating / Platt 은 **온프렘도 보강 필요**.

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

### 5.4 Sprint 1~4 완료분 (온프렘 → AWS 역수입 중 Must 12항목) — 2026-04-21

M1~M12 (원래 "온프렘만" 표에 있던 항목들) 은 **`aws_build_plan.md` 의 4개 Sprint 에 걸쳐 AWS 로 이식 완료**했다. 아래 표는 "양쪽 구현" 상태로 이동한 흔적.

| # | 항목 | AWS 구현 위치 | 테스트 |
|---|---|---|---|
| M1 | Human Review Queue | `core/serving/review/human_review_queue.py` | `tests/test_sprint3_serving.py` (25 cases) |
| M2 | Kill Switch (in-memory + config) | `core/serving/kill_switch.py` (확장) | `tests/test_sprint3_serving.py` |
| M3 | Marketing Consent (5채널, config-driven) | `core/compliance/consent_manager.py` (확장) | `tests/test_consent_channels.py` (19 cases) |
| M4 | OptOutManager + explanation request | `core/compliance/rights/opt_out.py` | `tests/test_compliance_rights.py` |
| M5 | ProfilingWorkflow (access/correction/deletion) | `core/compliance/rights/profiling.py` | `tests/test_compliance_rights.py` |
| M6 | ExplanationSLATracker (10-일 SLA) | `core/compliance/rights/explanation_sla.py` | `tests/test_compliance_rights.py` (37 cases 통합) |
| M7 | KoreanFRIAAssessor (7-차원, 5-년 retention) | `core/compliance/fria_assessment.py` | `tests/test_compliance_sprint2.py` |
| M8 | ComplianceRegistry (36 항목, A+GAP) | `core/compliance/compliance_registry.py` | `tests/test_compliance_sprint2.py` |
| M9 | AIRiskClassifier (6-차원, 등급 escalation) | `core/compliance/ai_risk_classifier.py` | `tests/test_compliance_sprint2.py` (40 cases 통합) |
| M10 | DynamicItemUniverseLoader (DuckDB + TTL) | `core/recommendation/universe/dynamic_loader.py` | `tests/test_sprint3_serving.py` |
| M11 | Audit Archive 확장 5개 컬럼 | `core/recommendation/audit_archiver.py` (확장) | `tests/test_sprint3_serving.py` |
| M12 | LLM Generation Marker (idempotent) | `core/recommendation/reason/marker_applier.py` | `tests/test_sprint3_serving.py` (56 cases 통합) |

**통합**:
- `core/evaluation/promotion_gate.py`: FRIA + AI Risk 게이트가 `scripts/submit_pipeline.py::_decide_promotion` 에 optional post-check 로 연결 (`compliance.promotion_gate.enabled`=false 기본).
- `core/serving/predict.py`: 위 모든 hook 이 `RecommendationService.__init__` 의 optional 인자로 주입. 기본 None (기존 동작 보존). 주입 시 non-blocking metadata annotation 또는 (tier 2/3) blocking queue enqueue 로 동작.

**양쪽 구현 현황**:
- AWS: 완료 (244 테스트 PASS, 하드코딩 0건).
- 온프렘: 동일 기능이 별도 모듈 (예: `src/scoring/`, `src/monitoring/`) 로 존재. 구조가 다름 → Phase B 역이식 대상 (별도 sync).

### 5.5 Phase 2 Should 완료분 (2026-04-21)

Must 12항목 이식에 이어, Should 15항목 중 **9개** 를 추가 이식 완료. 남은 Should 6개는 학습 계열 (S2~S4), MLflow/DVC (S5), ComplianceAuditStore 통합 (S6), context 도구 (S12) 로 Paper 2 v2 의 코드 근거로 필수가 아니라 후속 Phase 로 보류.

| # | 항목 | AWS 구현 위치 | 테스트 |
|---|---|---|---|
| S1 | Human Fallback Router (Layer 4) | `core/recommendation/fallback_router.py` (확장) | `tests/test_phase2_should.py::TestS1HumanFallback` |
| S7 | Fairness metrics archive | `core/monitoring/fairness_monitor.py::archive_metrics` | `tests/test_phase2_should.py::TestS7FairnessArchive` |
| S8 | Drift DuckDB/Parquet persist + Markdown | `core/monitoring/drift_detector.py::archive_result, generate_markdown_report` | `tests/test_phase2_should.py::TestS8DriftPersist` |
| S9 | Lineage catalog 확장 | `core/monitoring/lineage_tracker.py::register_feature_mapping, load_mapping_from_yaml, coverage_report` | `tests/test_phase2_should.py::TestS9Lineage` |
| S10 | EU AI Act Annex IV 12-항목 | `core/compliance/annex_iv_mapper.py` (신규) | `tests/test_phase2_should.py::TestS10AnnexIV` |
| S11 | L2a Safety Gate 3-layer | `core/recommendation/reason/l2a_safety_gate.py` (신규) | `tests/test_phase2_should.py::TestS11SafetyGate` |
| S13 | 금소법 §17 적합성 필터 | `core/recommendation/constraint_engine.py::SuitabilityFilter` (확장) | `tests/test_phase2_should.py::TestS13Suitability` |
| S14 | Counterfactual C-C (IPS/SNIPS) | `core/evaluation/counterfactual_cc.py` (신규) | `tests/test_phase2_should.py::TestS14Counterfactual` |
| S15 | auto_promote=False 강제 | `core/evaluation/model_competition.py::CompetitionConfig.auto_promote` + `pipeline.yaml serving.competition` + `submit_pipeline.py` from_dict 주입 | `tests/test_phase2_should.py::TestS15AutoPromote` |

**누적 테스트 현황 (2026-04-21)**: Phase 1 Must 244 + Phase 2 Should 36 + 기존 regression (compliance_evaluators) 46 = **280/280 PASS**. 하드코딩 0건.

**남은 Should (2개, 후속 Phase)**:
- S5 MLflow + DVC Compliance — 외부 플랫폼 의존, infra 작업 별도 트랙
- S6 ComplianceAuditStore DuckDB 통합 — Sprint 0 foundation 과 legacy audit_store 병합 필요, 중복 리스크

### 5.6 Phase 2 Should 추가 완료분 (2026-04-21, 학습 계열 + context)

위 §5.5 의 9개에 이어 **S2/S3/S4/S12 를 추가 이식 완료**. 학습/reason 계열 4개로 Paper 2 v2 의 "실데이터 안정성 + 다학제 해석" 서술 근거 확보.

| # | 항목 | AWS 구현 위치 | 테스트 |
|---|---|---|---|
| S2 | IG 3-stage Feature Selection Stage 3 (mandatory feature 보장) | `core/training/feature_selector.py::select()` | `tests/test_phase2_remaining.py::TestS2FeatureSelectorStage3` |
| S3 | Evidential valid_mask 결측 방어 | `core/model/layers/evidential.py::forward(valid_mask=)` + auto-detect + task-type neutral fill | `tests/test_phase2_remaining.py::TestS3EvidentialValidMask` |
| S4 | HMM config-driven ensemble gating smoothing | `core/model/experts/temporal.py::set_hmm_routing()` + `hmm_routing` config block | `tests/test_phase2_remaining.py::TestS4HMMRouting` |
| S12 | Multidisciplinary interpreter hook + consultation context | `core/recommendation/reason/context_assembler.py::attach_interpreter` + `AssembledContext.multidisciplinary_insights` | `tests/test_phase2_remaining.py::TestS12MultidisciplinaryInterpreter` |

**누적 테스트 현황 (2026-04-21 2차)**: Phase 1 Must 244 + Phase 2 Should 36 + Phase 2 추가 23 + 기존 regression 46 = **303/303 PASS**.

---

## 6. 두 프로젝트의 본질적 포지셔닝 (정정본)

### 6.1 설계 의도

- **AWS 는 온프렘의 클라우드 확장 버전.** 별도 프로젝트가 아니라 *같은 시스템의 클라우드 대응*.
- **기본 원칙**: 두 체계는 기능적으로 동일해야 함.
- **허용되는 차이**:
  1. **인프라 고유 차이** (SageMaker vs Airflow, S3 vs DuckDB, Bedrock vs vLLM, S3 Object Lock vs MinIO Object Lock 등)
  2. **AWS 가 의도적으로 더 config-중심 / 모듈화 지향** — 구조가 다르면 AWS 패턴이 "타겟"
  3. 그 외는 **드리프트 = 실수/시간차 = 양방향 sync 로 해결**

### 6.2 이번 비교가 확인한 것

이 매트릭스가 보여준 gap 의 대부분은 (3) 드리프트입니다:

- **AWS 에 이식이 밀린 것**: 온프렘의 실운영/규제 모듈 (Human Review Queue, Kill Switch, Consent 4-모듈, FRIA, 36-항목 레지스트리, Dynamic Item Universe, Audit Archive 확장 컬럼, LLM marker, MLflow+DVC compliance)
  - 원래 온프렘에 먼저 들어간 것 — AWS 클라우드 버전이 만들어질 때 규제/운영 모듈이 아직 이식 안 됨. **이식 대상**
- **온프렘에 이식이 밀린 것**: AWS 의 최근 연구 성과 (Paper 3 Findings 8~13 = CEH/CG/CCP/W-amp/v3) 및 Fusion 기법 (NEAS/BRP-detached/ECEB/AdaTT-sp/M1) 및 MRM 인프라 (Teacher threshold gating, 3-Layer FallbackRouter, Platt calibration, LeakageValidator standalone class)
  - AWS 에서 실험적으로 먼저 구현된 것 — 온프렘은 아직 sync 안 됨. **이식 대상**
- **구조 차이가 있지만 기능 동등**: FD-TVS scoring (AWS monolithic `FDTVSScorer` 클래스 ↔ 온프렘 5-파일 분리)
  - AWS 는 config-중심 plugin 방식, 온프렘은 테스트 친화적 분리. **기능 동일, 구조 의도적 차이**.

### 6.3 인프라 고유 차이로 인정할 항목

| 항목 | AWS | 온프렘 | 같은 역할 |
|---|---|---|---|
| 오케스트레이션 | SageMaker 순차 실행 | Airflow DAG | 잡 실행 관리 |
| 데이터 저장소 | S3 Parquet + DynamoDB | Parquet (로컬/NFS) + DuckDB 쿼리 엔진 | 피처/메트릭 저장 (양쪽 모두 Parquet 포맷; AWS 는 S3 에, 온프렘은 파일시스템에. DuckDB 는 쿼리 레이어) |
| LLM 서빙 | Bedrock API (Claude) | vLLM self-host (Qwen3-8B-AWQ) | L2a rewrite |
| 감사 WORM | S3 Object Lock | *없음 → MinIO Object Lock 추가 필요* | 불변 감사 기록 |
| 모니터링 알림 | CloudWatch | Slack | 운영 알림 |
| 시크릿 관리 | SSM Parameter Store | 환경변수/파일 | HMAC 키 등 |

**주의**: 온프렘 "WORM 없음" 은 인프라 고유 차이가 아니라 **결함**. MinIO Object Lock 등 온프렘 대체재로 보강 필요.

### 6.4 설계 의도 차이 (AWS = 더 config-중심)

| 측면 | AWS 패턴 | 온프렘 패턴 | 결론 |
|---|---|---|---|
| Config 소스 | `PLEConfig` dataclass + YAML + SM_HPS 주입 | YAML dict 직접 참조 (`self.config.get(...)`) | AWS 패턴이 타겟 (type-safety + IDE 지원) |
| Plugin Registry | `ExpertRegistry`, `ScorerRegistry`, `FilterRegistry`, `TowerRegistry` 데코레이터 | Factory 패턴 | AWS 패턴이 타겟 |
| HP 주입 | `config_builder.py` 단일 진입점 | 분산 | AWS 패턴이 타겟 |
| Fusion 전환 | `fusion_type` 파라미터로 CGC 안에서 5종 전환 | 개별 구조 | AWS 패턴이 타겟 |

**의미**: 위 항목들에서 온프렘이 AWS 로 sync 할 때는 단순 이식이 아니라 **AWS config-중심 패턴으로 재구성** 필요.

### 6.5 양방향 sync 가 v2 목표

- Phase A (AWS → 온프렘): `onprem_work_plan.md` 참조. CEH / CG / CCP / Fusion / Teacher gating / Fallback / Calibration.
- Phase B (온프렘 → AWS): *신규 문서 필요 시*. Human Review / Kill Switch / Consent 4-모듈 / FRIA / 36-항목 레지스트리 / Dynamic Item Universe / Audit extended columns / LLM marker / MLflow+DVC.
- Phase C (양쪽 공통): 온프렘 WORM 저장소 추가, 온프렘 LGBM temporal split 수정, AuditLogger API 통일.

### 6.6 v2 발표 시 협업 구조 (정정)

두 시스템이 **같은 시스템의 두 deployment** 라는 관점에서:
- Paper 1/2/3 v2 는 AWS 쪽에서 작성하되, 내용은 **"온프렘 + AWS 공통 아키텍처"** 를 기술. 클라우드 고유 요소는 별도 노트.
- 온프렘 백서는 **실운영 증거 + 규제 대응 상세** 로 AWS paper 의 보충 역할.
- 두 프로젝트 모두 동일한 Findings 1~13 + 동일 규제 모듈 + 동일 서빙 흐름을 보유해야 함. 현재는 양쪽 모두 partial.

---

## 7. 후속 작업 연결

- **이 매트릭스** → `docs/pipeline_comparison_matrix.md` (현재 문서)
- **AWS → 온프렘 이식 계획** → `docs/onprem_work_plan.md`
- **실데이터 검증 체크리스트** → `docs/onprem_v2_validation_checklist.md`
- **온프렘 → AWS 역수입 작업** → *아직 없음, 필요 시 신규 문서 `docs/onprem_to_aws_import_plan.md` 작성*

역수입 문서는 v2 발표가 온프렘 테스트 결과 포함이라는 점 고려 시, 실제 이식 실행은 온프렘 검증 이후가 자연스러움. 지금은 이 매트릭스로 **gap 인식 + 우선순위 공유** 만 확보.
