# Claude Code Agent Guidelines — AIOps PLE Platform

## 1. 절대 준수사항 (MUST)

### 1.1 Config-Driven 원칙
- **모든 파라미터는 YAML config에서 읽는다.** Python 코드에 컬럼명, 경계값, 시나리오 목록, AWS 상수를 하드코딩하지 않는다.
- **Split-config 패턴**: `configs/pipeline.yaml`(공통: model, training, distillation, aws) + `configs/datasets/{name}.yaml`(데이터셋별: tasks, labels, ablation). `load_merged_config()`로 deep-merge하여 사용.
- 새 데이터셋 지원 시 `configs/datasets/example.yaml`을 복사하여 task/label만 정의한다. train.py, adapter, ablation script에 dataset-specific 코드를 넣지 않는다.
- `feature_groups.yaml`의 `input_filter`로 generator 입력을 선언한다. adapter에서 `product_cols`, `synth_cols` 같은 하드코딩 라우팅을 하지 않는다.

### 1.2 관심사 분리 (Separation of Concerns)
- **Adapter**: raw data → standardized DataFrame. 전처리/피처생성/레이블파생을 하지 않는다.
- **PipelineRunner**: 전처리 → 피처생성 → 레이블파생 → 정규화 → 텐서 저장 (Phase 0).
- **config_builder**: PLEConfig 빌드의 단일 진실 공급원 (`core/model/config_builder.py`). train.py와 PLEPredictor 모두 여기서 모델 구성을 읽는다.
- **train.py**: training-ready 데이터 로드 → config_builder로 모델 빌드 → 학습. 전처리 코드, 모델 빌드 인라인 코드를 넣지 않는다.
- **PLEPredictor** (`core/inference/predictor.py`): 체크포인트 로드 + 추론. 학습 코드 없음.
- **PLEEvaluator** (`core/evaluation/evaluator.py`): per-task 메트릭 계산. 학습/추론 코드 없음.
- 코드가 500줄을 넘으면 관심사 분리가 되지 않은 것이다.

### 1.3 데이터 리키지 방지
- **Scaler는 TRAIN split에서만 fit**한다. val/test는 train에서 fit된 scaler로 transform만 한다.
- **Temporal split 시 gap_days를 반드시 설정**한다 (최소 7일).
- **시퀀스 데이터의 마지막 timestep이 레이블과 겹치지 않는지 반드시 검증**한다.
- **LeakageValidator를 학습 전에 반드시 호출**한다.
- **피처의 단순 변환(bucketing, 선형 결합)으로 파생되는 레이블은 태스크로 사용하지 않는다.** 모델이 입력에서 레이블을 완벽 복원할 수 있어 증류/학습이 무의미하다 (예: income_tier, tenure_stage, spend_level, engagement_score 등 deterministic 변환 결과).

### 1.7 라우팅 및 메트릭 교훈 (2026-04-11)
- **Feature-group level routing**: expert_routing은 개별 컬럼이 아닌 feature group 이름을 기준으로 해야 한다. 컬럼 이름 기반 routing은 Phase 0에서 정규화/log 복사본 추가로 컬럼이 재배열될 때 오작동한다.
- **Group range contiguity**: feature_group_ranges는 연속된 블록이어야 한다. min~max index 방식은 3-stage 정규화가 생성한 _log 접미사 컬럼이 끝에 추가되면 range를 터트린다. 비연속 매칭 시 가장 긴 연속 블록을 사용해야 한다.
- **Metric aggregation by task type**: avg_auc는 binary, avg_f1_macro는 multiclass, avg_mae는 regression task 전용이어야 한다. 전 task 평균은 metric semantics가 호환되지 않아 의미가 없다.

### 1.8 증류 파이프라인 규칙 (2026-04-15)
- **Teacher threshold gating**: teacher 성능이 2x random baseline 미달인 task는 증류 대신 hard label로 직접 LGBM 학습 (MRM 안전장치). 임계값은 `distillation.teacher_threshold`에서 config-driven으로 관리.
- **Calibration은 필요한 task만 적용**: 확률값이 필요한 task(churn_signal 등)만 Platt scaling 적용. 순위 기반 추천 task는 calibration 불필요. `distillation.calibration.tasks`에서 관리.
- **eval_metrics는 best epoch마다 저장**: 학습 종료 시점만이 아닌, best val_loss 갱신 시마다 checkpoint 디렉토리에 eval_metrics.json 저장. Job 중단 시에도 결과 보존.
- **Checkpoint resume 시 epoch counting**: `remaining = target_epoch - current_epoch`. 총 시행횟수가 아닌 목표까지 남은 횟수만 진행.
- **3계층 서빙 폴백**: Layer 1 (PLE→LGBM 증류) → Layer 2 (LGBM 직접 학습) → Layer 3 (금융 DNA 기반 룰). 서비스 중단 없는 구조.

### 1.9 정규화 및 피처 범주화 규칙 (2026-04-15)
- **Normalizer에서 범주형 ID와 확률 컬럼을 StandardScaler에서 제외**한다. 예: customer_id 파생 정수, 이미 [0,1] 범위인 calibrated 확률 컬럼은 scaler를 적용하면 분포가 왜곡된다. `feature_groups.yaml`의 `exclude_from_scaler: [categorical_id, probability]` 필드로 선언.
- **피처 범주 맵은 feature_groups.yaml에서 자동 빌드**한다. `FEATURE_GROUP_COLUMN_PREFIXES` 같은 하드코딩 딕셔너리를 코드에 넣지 않는다. Phase 0 완료 후 feature_schema.json에 기록된 group→column 매핑을 읽는다.
- **FD-TVS 태스크 가중치는 동적(세그먼트 × 행동)**이어야 한다. `scoring.segment_task_weights`(세그먼트 기반 승수, 1.0~1.5 클리핑)와 `scoring.dynamic_weight_rules`(피처 임계값 기반 부스팅)를 반드시 config-driven으로 관리한다. 온프레미스의 상품 단위 가중치 방식을 태스크 단위로 개선한 설계이다.

### 1.10 Champion-Challenger 승격 원칙 (2026-04-17)
- **submit_pipeline은 항상 registry.package() 등록을 수행**한다. 학습/증류가 완료되고 fidelity가 실패해도 등록은 유지하여 추후 원인 분석이 가능하도록 한다.
- **승격 판정은 `scripts/submit_pipeline.py::_decide_promotion`의 단일 관문을 통과한다.** 외부 경로로 `registry.promote()`를 호출해서는 안 된다.
- **판정 순서 (단락 평가)**:
  1. `--force-promote` → 무조건 승격 (운영자 override, trigger="manual")
  2. champion 없음 → bootstrap 승격
  3. fidelity_summary.failed > 0 → reject (안전 floor, Competition 생략)
  4. `ModelCompetition.evaluate()` → promotion_approved 기반 promote/reject
- **오프라인 게이트만 자동**이다. 온라인 게이트(`ModelMonitor.evaluate_champion_challenger`)는 실 트래픽 누적 후 수동/스케줄 트리거로 실행하며, orchestrator에 자동 연결하지 않는다.
- **모든 판정은 `AuditLogger.log_model_promotion`으로 HMAC 서명 + hash chain 감사 로그에 기록**한다. promote/reject/bootstrap/force_promote 모두 동일하게 기록되어야 하며, 로깅 실패가 승격 자체를 차단하지 않도록 best-effort로 호출한다 (SR 11-7 MRM).
- **`ModelCompetition`의 default metric 설정**(primary=avg_auc, min_improvement=0.5%, max_degradation=2%, significance_level=0.05)은 코드에 하드코딩되어 있지만, 프로덕션 도입 시 pipeline.yaml `serving.competition.*` 섹션으로 이관 예정. 임의 수정 금지.

### 1.11 Compliance 모듈 원칙 (Sprint 0~4, 2026-04-21)
- **Sprint 0 Foundation이 모든 신규 컴플라이언스 기능의 의존성 기반**이다. 새 규제 기능은 반드시 `ComplianceRequest`/`ComplianceEvent` 타입, `ComplianceStore` 인터페이스, `SLATracker` 기반 위에 구축한다. 새 store, SLA, audit 메커니즘을 병행 작성하지 않는다.
- **`core/compliance/rights/` 서브패키지**는 사용자 권리 (opt-out, profiling, explanation SLA) 전용이다. 기존 `core/compliance/ai_opt_out.py`, `core/compliance/profiling_rights.py`는 legacy DynamoDB 경로로 유지하되, 새 코드는 rights/ 서브패키지를 사용한다.
- **한국 AI기본법 FRIA vs EU AI Act FRIA는 서로 다른 class**이다. `core/compliance/fria_assessment.py::KoreanFRIAAssessor` (AI기본법 §35, 7-차원, 5-년 retention)와 `core/monitoring/fria_evaluator.py::FRIAEvaluator` (EU AI Act Art. 9, 5-차원)를 혼용하지 않는다. 법적 기반이 다르므로 리포트를 통합해도 내부 저장은 분리한다.
- **promotion_gate는 `compliance.promotion_gate.enabled=false` 기본**이다. Dimension 점수 provider가 wire되지 않은 상태에서 기본 0.5를 쓰면 승격 경로가 conservative LIMITED로 수렴하므로 실제 게이트 역할을 못한다. 실 provider 연결 전에 enabled=true 변경 금지.
- **predict.py의 Sprint 1~3 hook은 모두 optional injection**이다. 아무 hook을 주입하지 않으면 기존 11단계 동작과 동일. 주입 시 대부분 non-blocking metadata annotation이며, 유일한 blocking 동작은 HumanReviewQueue (tier 2/3) 뿐이다.
- **Marker applier는 idempotent**이다. L2a LLM rewrite 결과에 이미 AI기본법 marker가 포함되어 있으면 `apply()`는 원본을 반환한다. 여러 단계에서 중복 적용되어도 안전.
- **ComplianceRegistry의 36 항목은 A-group 18개 + GAP 18개로 고정**한다. 새 체크 항목이 필요하면 GAP-group에 추가하고 Sprint X 진행 시 A-group으로 이동한다.

### 1.12 Phase 2 Should 정책 (2026-04-21)
- **`auto_promote=False` 는 pipeline.yaml 에서 강제**한다 (EU AI Act Art. 14 + SR 11-7). `CompetitionConfig` 의 코드 default 는 후행 호환성 때문에 True 로 두되, `pipeline.yaml::serving.competition.auto_promote=false` 를 통해 프로덕션 posture 을 강제한다. 모든 challenger 가 metrics gate 통과해도 `--force-promote` 수동 override 없이는 승격되지 않는다.
- **Layer 4 (Human Fallback) 는 opt-in**이다. `serving.review.tier_3_human_fallback=true` 일 때만 `FallbackRouter` 가 Layer 4 를 반환한다. Layer 4 verdict 를 받은 caller 는 반드시 `HumanReviewQueue` 에 enqueue 해야 하며, 라우터 자체는 큐에 넣지 않는다 (관심사 분리).
- **Fairness / Drift 영속화 경로는 config 플래그로만 활성**한다. `monitoring.fairness.archive_parquet_path`, `monitoring.drift.archive_parquet_path` 를 지정해야 Parquet 쓰기 경로가 작동한다. 미지정 시 in-memory archive 만 유지.
- **Counterfactual C-C 는 logged propensities 를 요구**한다. logged_propensities 가 없으면 IPS/SNIPS 실행 불가. `CounterfactualEvaluator.from_config(serving.counterfactual_cc)` 로 estimator (ips/snips), min_lift, bootstrap 파라미터를 config-driven 으로 관리.
- **Annex IV mapper 는 Article 9 EU-FRIA 와 별개**다. `core/compliance/annex_iv_mapper.py::AnnexIVMapper` 는 Art. 11 기술문서 증거 추적용이고, `core/monitoring/fria_evaluator.py::FRIAEvaluator` 는 Art. 9 위험 평가용이다. 혼용 금지.
- **SuitabilityFilter 는 `require_assessment=true` 가 기본**이다. `customer_risk_tolerance` 가 context 에 없으면 거부. 고령 (≥65) / 저소득 (<30M KRW) hard cap 은 config 로 조정하되 제거 금지 (금소법 §17 보호대상 규정).
- **L2aSafetyGate 는 LLM 후처리에만 적용**한다. L1 template 결과에는 적용하지 않음 (template 은 이미 검증됨). gate 실패 시 반드시 L1 fallback 으로 회귀해야 하며, 응답을 빈 문자열로 리턴하지 않는다.
- **Phase 2 남은 2개 항목 (S5, S6) 는 후속 트랙**이다. Paper 2 v2 코드 근거로 필수가 아니므로 우선순위 낮음. S5 (MLflow+DVC) 와 S6 (ComplianceAuditStore DuckDB 통합) 은 Sprint 0 foundation 과 중복 가능성이 있어 통합 설계를 먼저 검토해야 한다.

### 1.13 Phase 2 학습/Reason 레이어 정책 (2026-04-21 추가분)
- **Feature selector 는 3-stage 가 기본**이다 (`core/training/feature_selector.py::select`). Stage 3 (mandatory feature 보장) 는 `FeatureSelectionConfig.mandatory_features` 가 비어 있지 않을 때만 작동. 도메인 필수 feature (예: 규제/감사 관련) 를 Stage 1~2 가 드롭해도 Stage 3 가 복원한다. Stage 3 skip 금지 — 규제 리스크.
- **Evidential layer 는 valid_mask 전파가 필수**다 (`core/model/layers/evidential.py::forward`). `valid_mask=None` 인 경우 NaN/Inf 자동 감지하여 invalid 로 마킹하고 neutral prediction + max uncertainty 로 응답. Loss 계산 시 반드시 info 에 실린 `valid_mask` 를 참조해야 invalid 행을 gradient 에서 제외할 수 있다.
- **HMM 라우팅은 config-driven 이 원칙**이다 (`TemporalEnsembleExpert.set_hmm_routing`). 코드에서 `set_hmm_routing(True)` 를 직접 호출하지 말고 `config["hmm_routing"]["enabled"]=true` 로 제어한다. Transition matrix 는 buffer 로 등록되어 gradient 없음. `smoothing` 은 `[0, 1/(n_models-1)]` 로 clip 된다.
- **ContextAssembler interpreter 는 optional-injection**이다. `multidisciplinary_interpreter` 를 생성자 또는 `attach_interpreter(...)` 로 주입. interpreter 실패는 반드시 swallow (exception 을 reason generation flow 로 전파 금지). interpreter 반환 dict 의 값은 자동으로 str 변환되고 falsy value 는 필터된다.

### 1.14 Compliance tracking 원칙 (S5, 2026-04-21)
- **MLflow/DVC 를 직접 이식하지 않는다.** AWS 는 SageMaker Experiments + Model Registry + Lineage + S3 versioning 의 네이티브 등가물을 제공하므로, `core/compliance/sagemaker_compliance_tracker.py::SageMakerComplianceTracker` 를 사용한다.
- **4개 규제 산출물 유형만 기록**한다: `fria_assessment`, `ai_risk_assessment`, `compliance_registry_sweep`, `promotion_gate_verdict`. 그 외 임시 기록은 `log_custom_artifact` 로 타입 `custom` 에 태깅.
- **Tracker 실패는 반드시 swallow**한다. `put_artifact` 실패 시 `trial_component_arn=None` 만 남기고 원래 caller flow 를 중단시키지 않는다. 감사 누락은 별도 observability 로 감지.
- **Backend 기본은 `in_memory`** (단위 테스트 + 로컬 개발). 프로덕션 IAM + experiment 이름이 확정되면 `pipeline.yaml::compliance.tracking.backend=sagemaker` 로 전환. 직접 `boto3.client("sagemaker")` 를 주입하는 것도 가능 (테스트용).
- **TrialComponent 이름은 120자 제한**. SageMaker 의 하드 캡이므로 `<artifact_type>-<artifact_id>` 포맷을 `[:120]` 로 절삭하는 것이 표준. 로컬 검증 시 이 길이를 초과하면 프로덕션에서 실패.
- **ComplianceAuditStore 와 Compliance tracker 는 분리**한다. 전자는 **원천 이벤트** (consent 변경, opt-out 발생 등) 를 DynamoDB/S3 Parquet 에 저장. 후자는 **집계된 규제 산출물** (FRIA 결과, 승격 verdict 등) 을 SageMaker Experiments 에 기록. 용도가 다르므로 중복 저장이 아니다.

### 1.15 Compliance SQL 원칙 (S6, 2026-04-21)
- **Athena 를 기본으로 쓰지 않는다.** 온프렘의 DuckDB + Parquet 경험은 AWS 에서 DuckDB httpfs 확장 (`INSTALL httpfs; LOAD httpfs;`) 으로 그대로 재현된다. 쿼리당 $5/TB 의 Athena 비용을 감수할 만한 트래픽 (대시보드, QuickSight 연동 등) 이 생기기 전까지는 `core/compliance/audit_sql.py::ComplianceSQLHelper` 가 표준.
- **View 이름은 `[A-Za-z0-9_]+` 만 허용**한다. DuckDB `CREATE VIEW` 에 SQL injection 을 주지 않기 위해 `register_view()` 에서 엄격 검증. 쿼리는 항상 parameterised (`?` 플레이스홀더) 사용.
- **편의 메서드의 `view` 인자는 기본값 그대로 사용 권장**. `recent_opt_outs` 가 기본적으로 `opt_out` view, `sla_breaches` 가 `events` view 를 가정한다. pipeline.yaml 의 `compliance.audit_sql.paths` 가 동일 이름으로 등록되어 있어야 함.
- **`ComplianceSQLHelper` 는 on-demand 사용**이다. 장기 실행 서비스가 아니므로 context manager (`with ComplianceSQLHelper() as h:`) 또는 `close()` 호출로 DuckDB connection 을 명시적으로 닫는다.
- **DynamoDB (online 조회)** 와 **SQL helper (batch 조회)** 의 역할은 분리. DynamoDB 는 `ComplianceAuditStore` 가 담당. SQL helper 는 S3 Parquet archive 에 대한 SQL 만. 두 경로를 혼용해서 쓸 때도 서로의 데이터를 덮어쓰지 않아야 한다 (read-only join).

### 1.4 실험 전 검증 (Pre-flight Check)
- SageMaker Job 제출 전에 반드시 다음을 확인한다:
  1. **Phase 0 출력 검증**: feature_stats.json에서 zero-variance 컬럼, NaN 비율, 생성된 피처 컬럼 수 확인
  2. **Generator 입력 검증**: 각 generator가 받는 입력 컬럼의 dtype과 분포가 generator 설계 의도와 일치하는지 확인 (예: GMM은 continuous만, TDA는 시계열 또는 고차원 점구름)
  3. **레이블 분포 검증**: label_stats.json에서 class balance, positive rate 확인
  4. **Dry run**: `--dry-run` 플래그로 Job 구성을 먼저 확인한 후 제출
  5. **소규모 테스트**: 50K subsample로 end-to-end 성공을 확인한 후 전체 데이터 실행

### 1.5 비용 관리
- **ProfilerReport는 반드시 비활성화**한다 (`disable_profiler=True`를 estimator kwargs에 명시적으로 전달).
- **AMP (Mixed Precision)는 반드시 활성화**한다 — g4dn T4 GPU에서 ~2배 속도 향상.
- **batch_size는 VRAM과 데이터 크기에 맞게 최적화**한다 (941K 데이터 → 5632 권장).
- **spot 인스턴스는 동시 4대 이하**로 제한한다 — 8대 이상 시 같은 AZ 경쟁으로 중단 빈도가 급증한다.
- **max_wait는 max_run + 1시간**으로 설정한다 — 10시간 대기는 낭비.
- 실험 전 `aws ce get-cost-and-usage`로 현재 비용을 확인한다.
- **source 패키지는 1회만 빌드**하고 모든 Job에서 재사용한다.

### 1.6 오케스트레이션 비용 효율성
- **상태 파일 기반 자동 복구**: orchestrator가 죽어도 재시작 시 `pipeline_state.json`에서 완료된 Job을 감지하고 건너뛴다.
- **S3 결과 존재 확인**: Job 제출 전에 해당 시나리오의 `eval_metrics.json`이 S3에 이미 있는지 확인 → 있으면 스킵.
- **예산 가드**: 매 batch 제출 전에 `aws ce get-cost-and-usage`로 누적 비용을 확인, 설정된 예산 한도(`ablation.budget_limit`)를 초과하면 자동 중단.
- **실패 Job 자동 eviction**: Job이 예상 시간의 2배를 초과하면 자동 stop → 슬롯 반환.
- **Phase 0는 CPU 인스턴스** 사용 (pipeline.yaml `aws.cpu_instance_type`), GPU 인스턴스를 Phase 0에 낭비하지 않는다.
- **Warm Pool 활성화**: `keep_alive_period_in_seconds`를 설정하여 연속 Job 간 인스턴스 재사용.
- **비용 추정 vs 실제 비용 비교**: 실험 완료 후 `aws ce get-cost-and-usage`로 실제 비용을 확인하고, 추정치 대비 2배 이상 차이나면 원인을 분석하여 다음 실험에 반영.

## 2. 아키텍처 규칙

### 2.1 정규화 파이프라인 (3-stage)
```
Stage 1: 멱법칙 감지 (skew+kurt → log-log R²) + log1p 복사본 생성
Stage 2: StandardScaler (continuous 컬럼만, binary 제외, TRAIN fit only)
Stage 3: 멱법칙 _log 복사본은 스케일링하지 않음 (raw magnitude 보존)
```

### 2.2 모델 설정
- `task_loss_weights`: pipeline.yaml에서 읽어 PLEConfig에 전달
- `adaTT task_groups`: pipeline.yaml에서 읽어 `AdaTTConfig.from_pipeline_groups()` 호출
- `task_group_map`: adaTT groups에서 자동 빌드 (GroupEncoder + HMM routing 활성화)
- `logit_transfers`: pipeline.yaml task_relationships에서 읽어 전달
- `FocalLoss`: pre-activation logits를 전달 (double-sigmoid 방지)

### 2.3 Generator 입력 라우팅
```yaml
# feature_groups.yaml에서 선언 (하드코딩 금지)
generator_params:
  input_filter:
    dtype: continuous       # continuous | all_numeric
    exclude_binary: true    # GMM, TDA 등
    min_nunique: 20         # 이산 변수 제외
    include_prefix: [synth_] # 특정 컬럼 그룹만
```

## 3. 온프렘 프로젝트와의 일관성

### 3.1 반드시 유지해야 하는 온프렘 설계 요소
- 5축 피처 분류 (상태/스냅샷/시계열/계층/아이템)
- 피처 그룹별 전문가 라우팅 (target_experts)
- 태스크 그룹 기반 adaTT (intra/inter strength)
- 3단계 정규화 (log transform → scaler → raw copy)
- 로짓 전이 3가지 방식 (output_concat / hidden_concat / residual)
- Uncertainty weighting (Kendall et al.)
- 증류 시 LGBM gain importance 기반 피처 선택

### 3.2 AWS에서 달라도 되는 요소
- Airflow DAG → SageMaker 순차 실행 (오케스트레이션 방식)
- DuckDB 파일 기반 → S3 Parquet (저장소)
- Docker GPU 컨테이너 → SageMaker Training Job (실행 환경)

### 3.3 데이터 처리 백엔드 정책
- **pandas 직접 사용을 지양**한다. 대규모 데이터 로드/변환에 `pd.read_parquet()`, `pd.concat()`, `df.apply()` 등을 직접 쓰지 않는다.
- **우선순위**: cuDF (GPU) → DuckDB (CPU columnar) → pandas (최후 fallback, 10K 이하 소규모만)
- **Parquet 로드**: `duckdb.execute("SELECT ... FROM 'file.parquet'")` 사용. list/struct 컬럼도 네이티브 지원.
- **집계/변환**: SQL로 처리 (`GROUP BY`, `WINDOW`, `CASE WHEN`). pandas의 `groupby().apply(lambda)` 금지.
- **텐서 변환 직전**에만 `.df()` 또는 `.fetchnumpy()`로 pandas/numpy 변환.
- train.py `load_ready_data()`에서 `pd.read_parquet()` 대신 `duckdb.read_parquet()` 사용.

## 4. 코드 검수 기준 (커밋 전 필수)

모든 코드 작업 후, 커밋 전에 반드시 아래 **4단계 검수**를 통과해야 한다. 하나라도 빠지면 "완료"로 보고하지 않는다.

### 4.1 컴파일 검증
- 수정된 모든 `.py` 파일에 `py_compile.compile(f, doraise=True)` 실행
- 수정된 모든 `.yaml` 파일에 `yaml.safe_load()` 실행

### 4.2 인터페이스 계약 검증
- **파일 A가 저장하는 키 이름**과 **파일 B가 읽는 키 이름**이 일치하는지 확인
- 예: runner.py가 `feature_schema["group_ranges"]`로 저장하면, train.py도 `schema["group_ranges"]`로 읽어야 한다
- 병렬 에이전트 작업 후에는 반드시 **cross-file 키/필드 매핑 테이블**을 만들어 검증

### 4.3 하드코딩 스캔
- 실행 경로의 모든 파일에서 다음을 grep 검색:
  - **컬럼명**: `"customer_id"`, `"snapshot_date"`, `"income"`, `"tenure"`, `"prod_"`, `"synth_"`, `"seq_"`, `"nba_"` 등 dataset-specific 문자열
  - **AWS 상수**: `"ap-northeast"`, `"aiops-ple"`, `"ml.g4dn"`, `"795833"` 등
  - **매직넘버**: config에서 읽어야 하는 임계값, 경계값, 서브샘플 크기
- config fallback (`config.get("key", "하드코딩값")`)도 dataset-specific이면 안 된다
- `id_cols`, `date_cols`, `label_cols`는 반드시 config에서 읽는다

### 4.4 관심사 분리 검증
- train.py에 전처리 코드(fillna, LabelEncoder, scaler, label derivation)가 없는지 확인
- adapter에 generator 호출이 없는지 확인
- ablation script에 시나리오/전문가/태스크 목록이 하드코딩되지 않았는지 확인

### 4.5 "완료" 보고 기준
- 위 4단계를 **모두 통과한 후에만** "0건 잔여" 또는 "완료"로 보고한다
- 일부만 검증한 경우 "컴파일 OK, 인터페이스 미검증, 하드코딩 미스캔"으로 명시한다

### 4.6 에러 로깅 및 진단 기준
- **orchestrator**: Job 실패 시 FailureReason + CloudWatch 로그 URL을 함께 출력한다.
- **orchestrator**: Spot 중단과 알고리즘 에러를 구분하여 로깅한다 (SecondaryStatusTransitions 확인).
- **orchestrator**: 실패 사유를 state 파일에 저장한다 (재시작 시 이전 실패 원인 확인 가능).
- **orchestrator**: 실행 종료 시 성공/실패/중단 Job 목록 + 사유 + billable time 요약을 출력한다.
- **train.py**: `main()` 전체를 `try/except`로 감싸고 `logger.exception()`으로 full traceback을 출력한다.
- **train.py**: 학습 시작 전에 GPU 메모리, 데이터 shape/dtype/NaN 비율, 레이블 분포, 피처 스키마를 로깅한다.
- **model**: NaN/Inf loss 발생 시 **어떤 태스크**에서 발생했는지 로깅한다.
- **model**: gradient norm이 clip 임계값의 10배를 초과하면 경고 로깅한다.
- **adapter/generator**: 실패 시 `exc_info=True`로 full traceback을 포함한다.
- **validation**: metric 계산 실패 시 silent pass 대신 `logger.debug()`로 원인을 기록한다.

## 5. 서브에이전트 사용 규칙
- 서브에이전트는 **기본 sonnet 모델**을 사용한다. 사용자가 명시적으로 지정한 경우에만 opus를 사용한다.
- 병렬 서브에이전트 실행 시 각각의 역할을 명확히 구분한다.
- 서브에이전트 결과를 반드시 검수한 후 커밋한다.
- **병렬 에이전트 작업 후에는 반드시 인터페이스 계약 검증 에이전트를 추가로 실행한다.**

### 5.1 개발 환경 정책
- **코드 개발/디버깅은 로컬 GPU PC**(64GB RAM + RTX 4070)에서 수행한다.
- **SageMaker는 코드 디버깅에 사용하지 않는다** — 로컬에서 전체 데이터 테스트 통과 후 제출만 한다.
- 개발 순서: 로컬 테스트 → 전체 데이터 검증 → SageMaker 제출 (ablation 실행)
- SageMaker 제출 전에 로컬에서 최소 1 epoch end-to-end 성공을 확인한다.

## 6. 금지사항
- **SageMaker에서 코드 디버깅 금지** — 제출당 $0.50+ 비용 발생, 로컬에서 먼저 검증
- train.py에 전처리 코드를 넣지 않는다
- adapter에 generator 호출을 하드코딩하지 않는다
- FEATURE_GROUP_COLUMN_PREFIXES 같은 하드코딩 매핑을 만들지 않는다
- `_derive_santander_labels()` 같은 dataset-specific 함수를 train.py에 넣지 않는다
- 실험 결과 검증 없이 다음 Phase로 넘어가지 않는다
