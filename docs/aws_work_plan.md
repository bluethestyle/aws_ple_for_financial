# AWS 프로젝트 작업 계획 (온프렘 → AWS Sync)

**목적**: AWS 는 온프렘의 **클라우드 확장 버전** — 두 체계는 원칙적으로 동일해야 함. 이 문서는 **온프렘에 이식되어 있지만 AWS 에는 아직 없는 항목** 을 AWS 로 sync 하는 작업 계획.

**기준 시점**: 2026-04-21
**AWS 기준**: `aws_ple_for_financial` main 브랜치 (commit 35a95e9 이후)
**온프렘 기준**: `c:/Users/user/Desktop/ttm/gotothemoon/workspace/code/`

**진행 현황 (2026-04-21)**:
- ✅ **Phase 1 Must (M1~M12) 완료** — 모든 규제 핵심 모듈 AWS 이식.
- ✅ **Phase 2 Should 15/15 완료** — S1~S15 전체 완료.
- ✅ **Phase 3 Could 4/5 완료** — C1, C3, C4, C5. C2는 Won't (AWS SageMaker 네이티브).
- 누적 테스트: **606/606 PASS** (PR #1~#3 반영), 하드코딩 0건 (`ap-northeast-2` repo-wide 0 hits).
- **PromotionGate Live Wiring (PR #2/#3)**: `core/compliance/metadata_aggregator.py` (신규) 로 lineage / fairness / review queue / model registry / LLM config / static overrides 6 source 를 composite 하여 차원 점수 공급. `GateVerdict.details` 가 per-dimension 유도 트레일을 `AuditLogger.log_model_promotion` 의 HMAC + hash-chain 레코드에 임베드하고, 동시에 SageMakerComplianceTracker 에 `promotion_gate_verdict` artifact 로 기록. 이로써 `compliance.promotion_gate.enabled: true` 가 pipeline.yaml 기본값으로 전환 (conservative LIMITED collapse 리스크 해소).
- **S5 재정의**: 온프렘 MLflow+DVC 를 그대로 이식하지 않고, AWS 네이티브 서비스 (SageMaker Experiments + Model Registry + Lineage + S3 versioning) 기반으로 재설계. `core/compliance/sagemaker_compliance_tracker.py` 가 4개 규제 산출물 유형 (FRIA / AI Risk / Compliance Sweep / Promotion Gate) 을 Experiments TrialComponent 로 자동 기록.
- **S6 재정의**: Athena 도입은 과잉. DuckDB httpfs 확장이 s3:// URI 를 네이티브로 읽으므로, `core/compliance/audit_sql.py::ComplianceSQLHelper` 로 온프렘 DuckDB 경험을 AWS 에서 0 인프라 비용으로 유지 (regulator 쿼리 + cross-view JOIN).
- **C1 신규**: `core/evaluation/uplift_learner.py` (T-Learner + X-Learner + 평가 지표) — Paper 2 v2 Pearl Rung 2 공백 해소.
- **C4 신규**: `core/security/ai_security_checker.py` (prompt injection + output PII/leak 탐지 + provider wrapping) — 기존 `prompt_sanitizer.py` 를 보강.

**관련 문서**:
- `docs/pipeline_comparison_matrix.md` — 4-레이어 전수 비교 결과
- `docs/onprem_work_plan.md` — 역방향 작업 계획 (AWS → 온프렘 sync)
- `docs/onprem_v2_validation_checklist.md` — 실데이터 검증 체크리스트

---

## 0. 설계 원칙

AWS 로 이식할 때는 **AWS 의 config-중심 / 모듈화 패턴** 에 맞게 재구성한다. 온프렘 코드를 **그대로 복사하지 않는다**.

| 온프렘 스타일 | AWS 타겟 스타일 |
|---|---|
| YAML dict 직접 참조 (`self.config.get(...)`) | `dataclass + from_yaml` + `PLEConfig` 계층 구조 |
| Factory 함수 | `@Registry.register` 데코레이터 패턴 |
| 모듈별 독립 구현 | `config_builder.py` 단일 진입점에서 HP flag 통합 |
| Airflow DAG pre-trigger | SageMaker 잡 제출 pre-flight check |
| 환경변수 + 파일 | SSM Parameter Store + S3 설정 |

---

## 1. 우선순위별 이식 대상

### 1.1 Must (v2 Paper 2 규제 섹션 핵심)

이 항목들이 AWS 에 없으면 Paper 2 v2 에서 "규제 대응" 을 코드 근거로 기술 불가.

| # | 컴포넌트 | 상태 | AWS 구현 위치 | 법적 근거 |
|---|---|---|---|---|
| **M1** | Human Review Queue | ✅ 완료 | `core/serving/review/human_review_queue.py` | 금융위 AI 가이드라인 제3원칙 |
| **M2** | Kill Switch (DynamoDB + in-memory) | ✅ 완료 | `core/serving/kill_switch.py` (확장, `from_config`) | 운영 긴급 대응 |
| **M3** | Marketing Consent (5채널, config-driven) | ✅ 완료 | `core/compliance/consent_manager.py` (`ConsentConfig`) | 개보법 §22, 정통망 §50 |
| **M4** | AI Decision Opt-out + explanation | ✅ 완료 | `core/compliance/rights/opt_out.py` | 개보법 §37의2, AI기본법 §31 |
| **M5** | Profiling Rights Workflow | ✅ 완료 | `core/compliance/rights/profiling.py` | 신정법 §36의2 |
| **M6** | Explanation SLA Tracker (10-일) | ✅ 완료 | `core/compliance/rights/explanation_sla.py` | 개보법 시행령 §44의2~4 |
| **M7** | Korean FRIA Assessor (7-차원) | ✅ 완료 | `core/compliance/fria_assessment.py` | AI기본법 §35 + 시행령 §27 |
| **M8** | 36-항목 Compliance Registry | ✅ 완료 | `core/compliance/compliance_registry.py` | 금감원 AI RMF + 개보법 + 신정법 |
| **M9** | AI Risk Classifier (6-차원) | ✅ 완료 | `core/compliance/ai_risk_classifier.py` | 금감원 AI RMF |
| **M10** | Dynamic Item Universe Loader | ✅ 완료 | `core/recommendation/universe/dynamic_loader.py` | 캠페인 라이프사이클 |
| **M11** | Audit Archive 확장 (5 컬럼) | ✅ 완료 | `core/recommendation/audit_archiver.py` (확장) | 민원 대응 재현성 |
| **M12** | LLM Generation Marker | ✅ 완료 | `core/recommendation/reason/marker_applier.py` | AI기본법 §31·§34 |

**Phase 1 Must 완료** (244 tests PASS). 상세: `docs/aws_build_plan.md`.

### 1.2 Should (v2 Paper 2 보강 + 실운영 근거)

| # | 컴포넌트 | 상태 | AWS 구현 위치 | 가치 |
|---|---|---|---|---|
| **S1** | Human Fallback Router (Layer 4) | ✅ 완료 | `core/recommendation/fallback_router.py` (Layer 4 추가) | 고위험 건 자동 인적검토 |
| **S2** | IG 기반 3-stage Feature Selection | ✅ 완료 | `core/training/feature_selector.py::select()` Stage 3 mandatory feature 보장 | Feature 과다 선택 방지 |
| **S3** | Evidential valid_mask 결측 방어 | ✅ 완료 | `core/model/layers/evidential.py::forward(valid_mask=)` + auto-detect | 실데이터 안정성 |
| **S4** | HMM config 동적 라우팅 | ✅ 완료 | `core/model/experts/temporal.py::set_hmm_routing()` + config block | Config single-source |
| **S5** | SageMaker Experiments 기반 compliance tracker (MLflow+DVC 대체) | ✅ 완료 | `core/compliance/sagemaker_compliance_tracker.py` (신규) — `SageMakerComplianceTracker` + `InMemoryTrackerBackend` / `SageMakerTrackerBackend` + TrialComponent 규제 산출물 자동 기록 | 규제 산출물 버전관리 |
| **S6** | ComplianceAuditStore SQL-queryable 통합 | ✅ 완료 | `core/compliance/audit_sql.py::ComplianceSQLHelper` (신규) — DuckDB + httpfs 로 S3/로컬 Parquet 에 SQL. Athena 없이 온프렘 DuckDB 경험 유지. recent_opt_outs / consent_changes_for_user / sla_breaches / promotion_gate_history 편의 메서드 + cross-view JOIN 지원 | regulator queryability via SQL |
| **S7** | Fairness metrics 영속화 | ✅ 완료 | `core/monitoring/fairness_monitor.py::archive_metrics` | 이력 쿼리 |
| **S8** | Drift Parquet 저장 + Markdown 리포트 | ✅ 완료 | `core/monitoring/drift_detector.py::archive_result, generate_markdown_report` | 감사 친화 저장 |
| **S9** | Data Lineage 매핑 확장 API | ✅ 완료 | `core/monitoring/lineage_tracker.py::register_feature_mapping, load_mapping_from_yaml, coverage_report` | AI기본법 §34 학습데이터 출처 |
| **S10** | EU AI Act Annex IV 12-항목 매퍼 | ✅ 완료 | `core/compliance/annex_iv_mapper.py` (신규) | Art.11 기술문서 요건 |
| **S11** | L2a Safety Gate 3-layer | ✅ 완료 | `core/recommendation/reason/l2a_safety_gate.py` (신규) | 할루시네이션 방어 |
| **S12** | Consultation context + 다학제 해석기 | ✅ 완료 | `core/recommendation/reason/context_assembler.py::attach_interpreter + AssembledContext.multidisciplinary_insights` | Reason 근거 다양화 |
| **S13** | 금소법 §17 적합성 필터 | ✅ 완료 | `core/recommendation/constraint_engine.py::SuitabilityFilter` (확장) | 금융소비자보호법 |
| **S14** | Counterfactual C-C (IPS/SNIPS) | ✅ 완료 | `core/evaluation/counterfactual_cc.py` (신규) | 관측편향 보정 |
| **S15** | auto_promote=False 인적 감독 강제 | ✅ 완료 | `core/evaluation/model_competition.py::CompetitionConfig.auto_promote` + `pipeline.yaml serving.competition` | EU AI Act Art.14 |

**Phase 2 Should 15/15 완료** (105 tests PASS). S5 는 SageMaker Experiments 네이티브, S6 는 Athena 대신 DuckDB-over-Parquet. 두 항목 모두 온프렘 스택을 그대로 이식하지 않고 AWS 에 더 맞는 접근으로 재설계했지만 기능은 동등.

**Phase 3 Could + PromotionGate Live Wiring 후속 (PR #1~#3)**: 447 → 606 tests. 상세는 `docs/pipeline_comparison_matrix.md` §5.10.

### 1.3 Could (v2 Optional 보강)

| # | 컴포넌트 | 상태 | AWS 구현 위치 | 이유 |
|---|---|---|---|---|
| **C1** | Uplift T-Learner | ✅ 완료 | `core/evaluation/uplift_learner.py` (신규) | Paper 2 v2 Pearl Rung 2 (treatment effect / CATE) 공백 보강 |
| **C2** | Airflow DAG 조건부 재학습 | 🚫 Won't | — | AWS 는 SageMaker managed orchestration, 이식 불필요 |
| **C3** | LiquidNeuralNetwork Expert | ✅ 완료 | `core/model/experts/temporal.py::LiquidNeuralNetwork` (기존) | Paper 1 heterogeneous expert |
| **C4** | AI Security Checker (LLM 프롬프트 보안) | ✅ 완료 | `core/security/ai_security_checker.py` (신규) + 기존 `prompt_sanitizer.py` | Prompt injection / output PII leak 탐지 |
| **C5** | PortfolioTriageAgent | ✅ 완료 | `core/recommendation/reason/portfolio_triage.py` (기존) | Paper 1 5-agent architecture |

### 1.4 Won't (인프라 고유 — 이식 불필요)

| 항목 | 이유 |
|---|---|
| Airflow DAG 자체 | AWS 는 SageMaker managed orchestration |
| vLLM self-host | AWS 는 Bedrock API (대체재) |
| NFS 파일시스템 | AWS 는 S3 (대체재) |
| Slack 알림 | AWS 는 CloudWatch + EventBridge (대체재) |

---

## 2. v2 Paper 관점 우선순위

### 2.1 Paper 2 v2 기준 (규제 + 서빙)

Paper 2 는 "규제-준수적 금융 AI 서빙" 이 테마. 아래 항목들이 AWS 에 없으면 논문 claim 의 코드 근거 부재.

- **Must**: M1 (Human Review), M3~M6 (Consent 4-module), M7 (FRIA), M8 (36-항목 레지스트리), M9 (AI Risk Classifier), M12 (LLM 생성 표시)
- **Should**: S1 (Human Fallback), S10 (EU AI Act Annex IV), S15 (auto_promote=False)
- **Writable evidence 우선순위**: Must 전부 → v2 Paper 2 의 "규제 대응" 섹션이 실제 코드 근거를 확보.

### 2.2 Paper 3 v2 기준 (Loss dynamics + Causal)

Paper 3 는 AWS 연구가 원산지라 대부분 이미 AWS 에 있음. 온프렘 역수입 필요 항목:

- **Should**: S3 (Evidential 결측 방어) — 실데이터 안정성 근거
- **Should**: S4 (HMM 동적 라우팅) — config single-source 증거

Paper 3 핵심 주장에는 영향 없음. 실데이터 검증 안정성에 기여.

### 2.3 Paper 1 v2 기준 (아키텍처)

- **Could**: C3 (LiquidNeuralNetwork) — heterogeneous expert 주장의 추가 근거

---

## 3. 실행 phase 제안

### Phase 1 (Must 이식, 2.5~3주)

**Week 1**: Consent / Rights 4-모듈 + Human Review Queue
- M3, M4, M5, M6, M1 이식
- AWS `core/compliance/` 를 온프렘의 권리 관리 체계로 확장

**Week 2**: FRIA + Compliance Registry + AI Risk Classifier
- M7, M8, M9 이식
- `core/compliance/` 에 국내 규제 대응 모듈 완성

**Week 3**: Item Universe + Audit 확장 + LLM marker
- M10, M11, M12 이식
- 서빙-레이어 실운영 모듈 완성

### Phase 2 (Should 이식, 2~3주)

- Week 4: S1 (Human Fallback), S15 (auto_promote), S13 (금소법 §17)
- Week 5: S7 (Fairness 영속화), S8 (Drift), S9 (Lineage), S10 (EU AI Act Annex IV)
- Week 6: S2 (IG FS), S3 (Evidential), S4 (HMM), S11 (Safety Gate 3-layer), S12 (Context 도구)

### Phase 3 (Should 나머지, 1~2주)

- S5 (MLflow+DVC compliance), S6 (ComplianceAuditStore 통합), S14 (Counterfactual C-C)
- Paper 2 v2 문서 업데이트: 규제 섹션에 이식 완료 모듈 모두 반영

### Phase 4 (Could, 시간/필요 되면)

- C1 (Uplift), C2 (재학습 조건), C3 (LiquidNN), C4, C5

---

## 4. 공통 작업 (Phase C — 양쪽 모두 필요)

이 항목들은 `onprem_work_plan.md` 에서도 중복 언급됨. 한쪽에서 먼저 하면 되지만 **양쪽 모두 수정 필요**.

| # | 항목 | AWS 측 | 온프렘 측 |
|---|---|---|---|
| **C1** | WORM 감사 저장소 | 이미 S3 Object Lock ✓ | MinIO Object Lock 등 추가 필요 |
| **C2** | `log_attribution` / `log_guardrail` | AWS 이미 구현 ✓ | 온프렘 `audit_logger.py` 에 추가 필요 |
| **C3** | Teacher threshold gating | AWS 이미 구현 ✓ | 온프렘 `src/distillation/` 추가 필요 |
| **C4** | Platt calibration | AWS 이미 구현 ✓ | 온프렘 `src/distillation/` 추가 필요 |
| **C5** | 3-Layer FallbackRouter | AWS 이미 구현 ✓ | 온프렘에 이식 (`onprem_work_plan.md` 참조) |
| **C6** | LGBM student temporal split 수정 (TODO C6) | AWS N/A | 온프렘 **즉시 수정 필요** (leakage 위험) |
| **C7** | AuditLogger API 통일 | `verify_chain_from_s3` 등 원격 검증 API 유지 | 온프렘 로컬 검증 + 온프렘형 원격 검증 (MinIO 등) 맞춤 |

---

## 5. 이식 시 검증 체크리스트

각 Must / Should 항목 이식 후:

- [ ] `py_compile` 성공
- [ ] Unit test 추가 또는 기존 테스트 통과
- [ ] `core/compliance/__init__.py` 또는 해당 모듈 `__init__.py` 에 export 추가
- [ ] Paper 2 v2 해당 섹션 업데이트 (어느 코드 경로가 어느 규제 조항과 매핑되는지 명시)
- [ ] CLAUDE.md 에 신규 모듈 정책 추가 (필요 시)
- [ ] `docs/pipeline_comparison_matrix.md` 해당 행을 "양쪽 구현" 으로 업데이트

---

## 6. 완료 기준

- **Phase 1 (Must) 완료** → v2 Paper 2 의 "규제 대응" 섹션을 코드 근거로 기술 가능
- **Phase 1 + 2 완료** → AWS 가 온프렘과 **기능적 parity** 달성 (인프라 고유 차이 제외)
- **모든 Phase 완료** → 두 시스템이 "같은 시스템의 두 deployment" 주장이 코드 수준에서 성립

---

## 7. 공수 합계

| 범위 | 기간 | 설명 |
|---|---|---|
| Must 만 | 2.5 ~ 3주 | v2 Paper 2 규제 섹션 근거 확보 |
| Must + Should | 5 ~ 6주 | 양방향 parity 근사 |
| 전체 (Could 포함) | 7 ~ 8주 | 완전 sync |

**중요**: Phase 1 (Must) 가 가장 높은 ROI. v2 submission 타이밍에 맞춰 Phase 1 만이라도 완료하는 게 우선.
