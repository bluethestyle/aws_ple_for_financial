# 09. Compliance & Governance — 감사 추적, 규제 준수, 파이프라인 감시

## 개요

06장은 오케스트레이션(Step Functions)과 기본 감사 구조를 다뤘습니다.
이 장은 **규제 준수의 전체 범위** — 감사 로그 불변성, 규제 레지스트리, 공정성, 쏠림 탐지,
인시던트 보고, 거버넌스 보고서, 데이터 보존 정책까지 다룹니다.

```
규제 준수 인프라
├── 감사 추적 (Audit Trail)          → 불변 로그, 해시 체인, 7개 감사 테이블
├── 규제 레지스트리 (36항목)          → 자동 점검, 분기 보고
├── 공정성 모니터링 (Fairness)        → DI/SPD/EOD, 보호 속성 5개
├── 쏠림 탐지 (Herding)              → HHI/Gini/Entropy, 시스템 리스크
├── 드리프트 감시 (Drift)            → PSI, 연속 3일 임계 시 재학습
├── 인시던트 관리                     → CRITICAL(1h)/MAJOR(4h)/MINOR(24h)
├── 킬스위치                         → 긴급 모델 비활성화
├── 거버넌스 보고서                   → 월/분기 AI 거버넌스 위원회
├── 데이터 보존 정책                  → GDPR/금감원 카테고리별 보존 기간
└── 감사 패키지                       → 외부 감사 대응 (금감원 AI RMF)
```

---

## 현재 (On-Prem) 구현 요약

### 1. 감사 로그 — HMAC + 해시 체인 불변성 (`audit_logger.py`)
- 각 로그 엔트리에 HMAC-SHA256 서명
- 연속 엔트리 간 SHA256 해시 체인 (prev_hash) → 위변조 불가
- 일별 JSONL 파일 (YYYYMM/audit_YYYYMMDD.jsonl)
- `verify_chain()`: 전체 감사 로그 무결성 검증

### 2. 7개 테이블 규제 감사 스토어 (`compliance_audit_store.py`)
DuckDB 기반, fcntl 락 동시성 제어:
- **ks_audit**: 킬스위치 활성화/비활성화 이력
- **consent_audit**: 마케팅 동의 변경 이력 (부여/철회/갱신)
- **profiling_audit**: 정보주체 권리 행사 (열람/정정/삭제/제한/이동)
- **opt_out_audit**: AI 결정 거부 이력
- **incident_audit**: 규제 인시던트 (심각도별)
- **distillation_audit**: 교사-학생 모델 성능 갭
- **embedding_audit**: 임베딩 품질 메트릭

### 3. 36항목 규제 준수 레지스트리 (`regulatory_compliance_checker.py`)
- A그룹 (18항목, 구현 완료): 모델 카드, 학습 데이터 문서화, 안전-신뢰 문서, AI 공시, 설명 가능성, 옵트아웃, 킬스위치, 롤백, 공정성, 적합성 등
- GAP그룹 (18항목, 갭 분석): 드리프트 대응, SLA 추적, 인간 검토 큐, EU AI Act, PIA 등
- 분기별 자동 점검 → 준수율 산출 → 미준수 항목 조치 지시

### 4. 공정성 모니터링 (`fairness_monitor.py`)
보호 속성 5개: 연령대, 성별, 지역, 소득, 라이프스테이지
- DI (Disparate Impact): 0.8 ≤ DI ≤ 1.25
- SPD (Statistical Parity Difference): |SPD| ≤ 0.1
- EOD (Equal Opportunity Difference): |EOD| ≤ 0.1

### 5. 쏠림 탐지 (`herding_detector.py`)
- HHI (허핀달-허쉬만 지수): 시장 집중도
- Gini 계수: 추천 불평등
- Entropy: 다양성 측정
- 쏠림률: 동일 상품 추천 비율
- 심각도: none → low → medium → high → critical

### 6. 인시던트 분류/보고 (`incident_report_generator.py`)
- CRITICAL (1시간 대응): 킬스위치, DI<0.6, 보안 침해 → 과기부/금감원 보고
- MAJOR (4시간 대응): DI<0.8, 쏠림 critical, 모델 롤백 → 금감원/내부 AI위원회
- MINOR (24시간 대응): 드리프트 경고, 품질 저하

### 7. 거버넌스 보고서 (`governance_report_generator.py`)
월/분기 보고서 9개 섹션: 공정성, 드리프트, 인시던트, 모델 변경, 킬스위치, 추천 품질, 리스크, 감사 요약, 경영진 요약

### 8. 데이터 보존 정책 (`data_retention_policy.yaml`)
| 카테고리 | 보존기간 | 근거 |
|---------|---------|------|
| Raw Data | 30일 | GDPR 최소화 |
| Processed Features | 90일 | 재학습 지원 |
| Training Data | 365일 | 재현성 |
| Model Checkpoints | 365일 | 롤백 |
| Inference Results | 90일 | 분쟁 대응 |
| Audit Logs | 영구 | 불변 기록 |
| PII Data | 30일 | GDPR §17 |

### 9. 데이터 리니지 (`data_lineage_tracker.py`)
734D 피처 → 64개 소스 테이블 역추적, 추천 ID → IG top-5 → 원천 데이터

### 10. 추가 구성요소
- **킬스위치**: 글로벌/태스크별/클러스터별 모델 즉시 비활성화
- **학습 데이터 문서화**: AI 기본법 §34①⑤ 학습 데이터 투명성
- **감사 패키지 빌더**: 7개 섹션 통합 외부 감사 대응 패키지
- **DVC 버전 추적**: 규제 산출물 9개 유형 버전 관리
- **MLflow 규제 메트릭**: 쏠림, XAI 품질, 공정성, 준수율 시계열 추적

---

## AWS 설계 — 규제 준수 아키텍처

### 전체 구조

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AWS 규제 준수 인프라                               │
│                                                                     │
│  Layer 1: AWS 네이티브 (자동, 추가 비용 없음)                         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ CloudTrail        → 모든 AWS API 호출 자동 기록                 │ │
│  │ S3 Versioning     → 데이터/모델 변경 이력 보존                  │ │
│  │ S3 Object Lock    → 감사 로그 불변성 (WORM)                    │ │
│  │ KMS               → 암호화 키 관리 + 키 사용 감사               │ │
│  │ IAM Access Log    → 누가 어떤 리소스에 접근했는지               │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  Layer 2: 플랫폼 레벨 (반자동)                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ ComplianceAuditStore → 7개 감사 테이블 (DynamoDB)              │ │
│  │ AuditLogger          → HMAC + 해시 체인 (S3 + DynamoDB)       │ │
│  │ DataLineageTracker   → 피처→원천 추적 (S3 메타데이터)          │ │
│  │ ExperimentTracker    → 실험 메트릭 (SageMaker Experiments)     │ │
│  │ DriftDetector        → PSI 모니터링 (SageMaker Monitor)        │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  Layer 3: 비즈니스/규제 레벨 (명시적)                                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ RegulatoryComplianceChecker → 36항목 자동 점검                 │ │
│  │ FairnessMonitor             → DI/SPD/EOD 보호 속성 감시        │ │
│  │ HerdingDetector             → 시스템 리스크 쏠림 탐지           │ │
│  │ IncidentReporter            → 심각도별 자동 보고                │ │
│  │ GovernanceReportGenerator   → 월/분기 거버넌스 보고서           │ │
│  │ KillSwitch                  → 긴급 모델 비활성화               │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 감사 로그 — AWS에서의 불변성 보장

```
On-Prem:  로컬 JSONL + HMAC + 해시 체인
AWS:      S3 Object Lock (WORM) + DynamoDB 해시 체인

강화된 점:
  - S3 Object Lock: AWS 레벨에서 물리적 삭제 차단 (관리자도 못 삭제)
  - KMS 암호화: 감사 로그 자체를 암호화
  - CloudTrail: 감사 로그에 대한 접근도 감사 (메타 감사)
```

```yaml
# configs/audit.yaml
audit:
  # 감사 로그 저장소
  storage:
    backend: s3                       # s3 | dynamodb | local
    s3_bucket: aiops-ple-financial
    s3_prefix: audit/logs/
    object_lock: true                 # WORM 모드 (삭제 불가)
    retention_days: 2555              # 7년 (금융 규제 기준)

  # 해시 체인 설정
  integrity:
    hmac_enabled: true
    hmac_key_source: aws_ssm          # AWS Systems Manager Parameter Store
    hash_chain: true                  # 연속 엔트리 SHA256 체인

  # 규제 감사 스토어
  compliance_store:
    backend: dynamodb                 # dynamodb | duckdb
    tables:
      - ks_audit
      - consent_audit
      - profiling_audit
      - opt_out_audit
      - incident_audit
      - distillation_audit
      - embedding_audit
```

```python
# core/audit/audit_logger.py
class AuditLogger:
    """
    감사 로그 — HMAC + 해시 체인 + S3 Object Lock.
    On-Prem의 audit_logger.py를 AWS 네이티브로 전환.
    """

    def log(self, operation: str, input_data, output_data, metadata: dict = None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "input_hash": self._hash(input_data),
            "output_hash": self._hash(output_data),
            "metadata": metadata,
            "prev_hash": self._last_hash,
        }
        entry["hmac"] = self._sign(entry)
        self._last_hash = self._hash(json.dumps(entry))

        # S3 Object Lock으로 불변 저장
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"audit/logs/{date}/{uuid}.jsonl",
            Body=json.dumps(entry),
            ObjectLockMode="GOVERNANCE",
            ObjectLockRetainUntilDate=retain_until,
        )

    def verify_chain(self, date: str) -> bool:
        """특정 날짜의 전체 감사 로그 무결성 검증."""
        ...
```

---

### 규제 감사 스토어 — DynamoDB 전환

```
On-Prem:  DuckDB (단일 파일, fcntl 락)
AWS:      DynamoDB (서버리스, 자동 스케일링, 항목별 TTL)

장점:
  - 동시성 문제 해결 (DynamoDB 네이티브)
  - 서버리스 (관리 비용 0)
  - TTL 기반 자동 정리 (보존 정책 자동 적용)
  - 글로벌 테이블 (멀티리전 복제 가능)
```

```python
# core/audit/compliance_store.py
class ComplianceAuditStore:
    """
    7개 감사 테이블을 DynamoDB로 관리합니다.
    On-Prem의 DuckDB compliance_audit_store.py를 대체합니다.
    """

    TABLES = [
        "ks_audit",         # 킬스위치 활성화/비활성화
        "consent_audit",    # 마케팅 동의 변경
        "profiling_audit",  # 정보주체 권리 행사 (GDPR)
        "opt_out_audit",    # AI 결정 거부
        "incident_audit",   # 규제 인시던트
        "distillation_audit",  # 교사-학생 성능 갭
        "embedding_audit",     # 임베딩 품질
    ]

    def log_kill_switch(self, action, key, level, operator_id, reason):
        ...

    def log_consent_change(self, customer_id, consent_type, action, channel):
        ...

    def log_incident(self, event_type, severity, source_module, details):
        ...
```

---

### 36항목 규제 준수 자동 점검

```yaml
# configs/compliance_registry.yaml
compliance_registry:
  # A그룹 (구현 완료 항목)
  implemented:
    A01: {name: "모델 카드", check: file_exists, path: "docs/model_card.md"}
    A02: {name: "학습 데이터 문서화", check: file_exists, path: "docs/training_data.md"}
    A03: {name: "안전-신뢰 문서", check: file_exists, path: "docs/safety_trust.md"}
    A04: {name: "AI 공시", check: config_valid, key: "reason_generation.ai_disclosure"}
    A05: {name: "설명 가능성", check: module_exists, module: "core.recommendation.interpretation"}
    A06: {name: "옵트아웃 권리", check: table_exists, table: "opt_out_audit"}
    A07: {name: "인간 검토", check: config_valid, key: "human_review.enabled"}
    A08: {name: "킬스위치", check: module_exists, module: "core.serving.kill_switch"}
    A09: {name: "모델 롤백", check: s3_exists, prefix: "models/checkpoints/"}
    A10: {name: "헬스 체크", check: endpoint_alive, url: "/health"}
    A14:
      name: "공정성 모니터링"
      check: config_range
      validations:
        di_lower: {min: 0.7, max: 0.9}
        di_upper: {min: 1.1, max: 1.3}
        spd_max: {min: 0.05, max: 0.15}
        eod_max: {min: 0.05, max: 0.15}
    A18:
      name: "적합성 제약"
      check: config_range
      validations:
        churn_threshold: {min: 0.3, max: 0.8}
        fatigue_threshold: {min: 0.1, max: 0.5}

  # GAP그룹 (갭 분석 항목)
  gap_analysis:
    GAP01: {name: "드리프트 자동 대응", status: partial}
    GAP05: {name: "추천 사유 품질 검증", status: implemented}
    GAP10: {name: "SLA 추적", status: planned}
    GAP14: {name: "EU AI Act 고위험 분류", status: planned}
    GAP15: {name: "PIA (개인정보 영향 평가)", status: partial}

  # 점검 주기
  schedule:
    daily: [A08, A10]              # 킬스위치, 헬스 체크
    weekly: [A14, A05]             # 공정성, 설명 가능성
    quarterly: all                  # 전체 36항목
```

---

### 공정성 모니터링

```yaml
# configs/fairness.yaml
fairness:
  protected_attributes:
    - name: age_group
      groups: [youth, middle, pre_senior, senior]
      privileged: middle
    - name: gender
      groups: [M, F, unspecified]
      privileged: M
    - name: region_type
      groups: [metropolitan, urban, rural]
      privileged: metropolitan
    - name: income_tier
      groups: [low, middle, high]
      privileged: high
      # NOTE: income_tier는 공정성 모니터링용 보호속성 그룹 구분자(income 피처 → 3 구간)이다.
      # ML 예측 태스크가 아님 — income_tier를 예측 태스크로 사용하면 결정론적 leakage 발생.
    - name: life_stage
      groups: [6 classes]
      privileged: null              # 기준 그룹 없음 → 전체 쌍 비교

  thresholds:
    di_lower: 0.8                   # Disparate Impact 하한
    di_upper: 1.25                  # Disparate Impact 상한
    spd_max: 0.1                    # Statistical Parity Difference
    eod_max: 0.1                    # Equal Opportunity Difference

  actions:
    di_below_0.6: incident_critical   # DI < 0.6 → CRITICAL
    di_below_0.8: incident_major      # DI < 0.8 → MAJOR
    spd_above_0.1: incident_minor     # |SPD| > 0.1 → MINOR
```

```python
# core/monitoring/fairness_monitor.py
class FairnessMonitor:
    """
    보호 속성별 추천 공정성을 측정합니다.
    금감원 7대 원칙 ⑤ 금융안정성 + AI 기본법 §33·§34 준수.
    """

    def evaluate(self, predictions, user_data, config) -> FairnessReport:
        results = []
        for attr in config.protected_attributes:
            di = self._disparate_impact(predictions, user_data, attr)
            spd = self._statistical_parity(predictions, user_data, attr)
            eod = self._equal_opportunity(predictions, user_data, attr)
            results.append(FairnessResult(attribute=attr.name, di=di, spd=spd, eod=eod))

            # 임계값 위반 시 인시던트 생성
            if di < 0.6:
                self.incident_reporter.report("fairness_violation", "critical", ...)
        return FairnessReport(results)
```

---

### 쏠림(Herding) 탐지

```python
# core/monitoring/herding_detector.py
class HerdingDetector:
    """
    추천 쏠림 탐지 — 동일 상품이 과도하게 추천되는 시스템 리스크 방지.
    """

    def detect(self, recommendations: list[dict]) -> HerdingReport:
        product_counts = Counter(r["product_id"] for r in recommendations)
        total = len(recommendations)

        hhi = sum((c / total) ** 2 for c in product_counts.values())
        gini = self._gini_coefficient(list(product_counts.values()))
        entropy = self._entropy(list(product_counts.values()))
        herding_rate = max(product_counts.values()) / total

        severity = self._classify_severity(hhi, gini, herding_rate)
        # critical → 인시던트 보고 + 킬스위치 검토
        return HerdingReport(hhi=hhi, gini=gini, entropy=entropy,
                             herding_rate=herding_rate, severity=severity)
```

---

### 드리프트 감시 — SageMaker Model Monitor 통합

```
On-Prem:  커스텀 PSI 계산 + 로컬 Parquet 저장
AWS:      SageMaker Model Monitor (기본) + 커스텀 PSI (확장)

SageMaker Model Monitor:
  - 베이스라인 자동 생성 (학습 데이터 기준)
  - 추론 입력 분포 자동 모니터링
  - 위반 시 CloudWatch 알람 → SNS → Lambda

커스텀 확장:
  - PSI 연속 3일 임계 시 자동 재학습 트리거
  - 피처별 개별 PSI 추적 (734D 각각)
```

```yaml
# configs/monitoring.yaml
drift:
  backend: sagemaker_monitor        # sagemaker_monitor | custom_psi | both
  psi_thresholds:
    warning: 0.1
    critical: 0.25
  auto_retrain:
    enabled: true
    consecutive_critical_days: 3    # 3일 연속 critical → 재학습
    trigger: step_functions          # Step Functions training_pipeline 시작
```

---

### 인시던트 관리

```yaml
# configs/incident.yaml
incident:
  severity_levels:
    critical:
      response_time: 1h
      triggers: [kill_switch, di_below_0.6, security_breach]
      report_to: [과기부, 금감원, CISO]
      auto_action: kill_switch_activate
    major:
      response_time: 4h
      triggers: [di_below_0.8, herding_critical, model_rollback]
      report_to: [금감원, AI거버넌스위원회]
      auto_action: notify_only
    minor:
      response_time: 24h
      triggers: [drift_warning, quality_drop, herding_high]
      report_to: [ML팀]
      auto_action: log_only

  # AWS 알림
  notification:
    sns_topic: arn:aws:sns:ap-northeast-2:ACCOUNT:ai-incidents
    slack_webhook: ${SLACK_WEBHOOK_URL}         # 선택적
```

```python
# core/monitoring/incident_reporter.py
class IncidentReporter:
    def report(self, event_type, severity, source_module, details):
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"

        # 1. 감사 스토어 기록
        self.compliance_store.log_incident(...)

        # 2. 심각도별 자동 조치
        if severity == "critical":
            self.kill_switch.activate(scope="global")
            self.sns.publish(topic=self.config.sns_topic, message=...)

        # 3. 거버넌스 보고서에 포함
        self.governance_queue.add(incident_id)
```

---

### 킬스위치

```yaml
# configs/kill_switch.yaml
kill_switch:
  levels:
    global: false                   # 전체 모델 비활성화
    per_task:                       # 태스크별 비활성화
      click: false
      purchase: false
    per_cluster:                    # 클러스터별 비활성화
      cluster_5: false

  fallback:
    strategy: rule_based            # 킬스위치 시 규칙 기반 추천으로 대체
    # strategy: previous_model      # 이전 모델로 롤백
    # strategy: disable             # 추천 비활성화
```

```python
# core/serving/kill_switch.py
class KillSwitch:
    """
    긴급 모델 비활성화.
    DynamoDB에 상태 저장 → 서빙 시 매 요청마다 확인.
    Lambda/ECS 어디서든 동일하게 동작.
    """

    def is_active(self, task: str = None, cluster: int = None) -> bool:
        state = self.dynamodb.get_item(Key={"key": "kill_switch"})
        if state["global"]:
            return True
        if task and state.get("per_task", {}).get(task, False):
            return True
        if cluster and state.get("per_cluster", {}).get(f"cluster_{cluster}", False):
            return True
        return False

    def activate(self, scope="global", task=None, cluster=None, reason=""):
        # 상태 변경 + 감사 로그 기록
        self.compliance_store.log_kill_switch(
            action="activate", key=scope, level=..., operator_id=..., reason=reason
        )
```

---

### 거버넌스 보고서

```
월/분기 자동 생성 → S3 저장 → AI 거버넌스 위원회 배포

보고서 9개 섹션:
1. 공정성 요약 (DI/SPD/EOD by 보호 속성)
2. 드리프트 요약 (기간 내 PSI 통계)
3. 인시던트 요약 (CRITICAL/MAJOR/MINOR 건수 + 상세)
4. 모델 변경 이력 (학습/배포/롤백)
5. 킬스위치 이력 (활성화/비활성화 횟수)
6. 추천 품질 (L1/L2 사유 품질 지표)
7. 리스크 변동 (쏠림 추이, 리스크 수준)
8. 감사 스토어 요약 (7개 테이블 건수)
9. 경영진 요약 (자동 생성 서술형)
```

```python
# core/monitoring/governance_report.py
class GovernanceReportGenerator:
    def generate(self, period: str = "monthly") -> GovernanceReport:
        return GovernanceReport(
            fairness=self.fairness_monitor.summary(period),
            drift=self.drift_detector.summary(period),
            incidents=self.incident_reporter.summary(period),
            model_changes=self.experiment_tracker.summary(period),
            kill_switch=self.compliance_store.ks_summary(period),
            reason_quality=self.reason_quality.summary(period),
            herding=self.herding_detector.summary(period),
            audit_summary=self.compliance_store.all_tables_summary(period),
            executive_summary=self._generate_executive_summary(...),
        )
```

---

### 데이터 보존 정책 — S3 Lifecycle 자동 적용

```yaml
# configs/data_retention.yaml
retention:
  policies:
    raw_data:
      retention_days: 30
      action: delete
      s3_lifecycle: true              # S3 Lifecycle Rule 자동 생성
      regulatory_basis: "GDPR 최소화 원칙"

    processed_features:
      retention_days: 90
      action: archive_to_glacier      # Glacier로 이동 (비용 절감)
      s3_lifecycle: true

    training_data:
      retention_days: 365
      action: archive_to_glacier
      regulatory_basis: "재현성 보장"

    model_checkpoints:
      retention_days: 365
      action: keep
      regulatory_basis: "롤백 지원"

    inference_results:
      retention_days: 90
      action: delete
      regulatory_basis: "분쟁 대응"

    audit_logs:
      retention_days: 2555            # 7년
      action: immutable               # S3 Object Lock
      regulatory_basis: "금융 규제 7년 보존"

    pii_data:
      retention_days: 30
      action: encrypted_delete
      regulatory_basis: "GDPR §17 삭제권"
```

```python
# infrastructure/s3_lifecycle.py (CDK)
# 위 config를 읽어 S3 Lifecycle Rules를 자동 생성합니다.
# 각 prefix별 보존 기간 + 아카이브 전략이 자동 적용됩니다.
```

---

### 데이터 리니지 — S3 메타데이터 기반

```python
# core/audit/lineage_tracker.py
class DataLineageTracker:
    """
    피처 → 원천 데이터 역추적.
    추천 ID → IG top-K → 피처 범위 → 소스 테이블 + 컬럼.
    """

    def trace_recommendation(self, recommendation_id, ig_scores, config) -> LineageRecord:
        top_features = np.argsort(ig_scores)[-5:]
        sources = []
        for feat_idx in top_features:
            range_info = self.reverse_mapper.find_range(feat_idx)
            sources.append({
                "feature_index": feat_idx,
                "feature_range": range_info.name,
                "source_tables": range_info.source_tables,
                "source_columns": range_info.source_columns,
            })
        return LineageRecord(
            recommendation_id=recommendation_id,
            top_features=sources,
            model_version=config.model_version,
            pipeline_execution_id=config.execution_id,
        )
```

---

### 파이프라인 감시 — Step Functions + CloudWatch 통합

```
Step Functions (각 상태 머신)
    ↓ 성공/실패 이벤트
EventBridge
    ├── 성공 → CloudWatch 메트릭 기록
    └── 실패 → SNS 알림 + 인시던트 자동 생성
          ↓
    Lambda (인시던트 분류)
          ├── CRITICAL → 킬스위치 검토 + 과기부/금감원
          ├── MAJOR → AI위원회 알림
          └── MINOR → 로그만

CloudWatch 대시보드:
  - 파이프라인 성공/실패율
  - 학습 메트릭 추이 (loss, AUC)
  - 서빙 레이턴시 (p50, p95, p99)
  - 공정성 지표 추이 (DI, SPD, EOD)
  - 쏠림 지표 추이 (HHI, herding_rate)
  - 드리프트 PSI 히트맵
  - 비용 추이
```

---

## 적용 규제 요약

| 규제 | 관련 조항 | 설계 반영 |
|------|---------|----------|
| **AI 기본법** | §31 (AI 생성물 표시) | AI 공시 + 추천 사유 L1/L2 |
| | §33 (고위험 AI 거버넌스) | 36항목 레지스트리 + 거버넌스 보고서 |
| | §34 (위험 관리 기록) | 감사 로그 불변성 + 7개 감사 테이블 |
| **금소법** | §19 (설명의무) | 피처 역매핑 + 태스크별 해석 |
| **금감원 AI RMF** | ①합법성 | 36항목 자동 점검 |
| | ②안전·신뢰 | 킬스위치 + 인시던트 보고 |
| | ④신뢰성 | 드리프트 감시 + 자동 재학습 |
| | ⑤금융안정성 | 공정성 DI/SPD/EOD + 쏠림 탐지 |
| **GDPR** | §17 (삭제권) | 30일 PII 보존 + 암호화 삭제 |
| | §22 (자동화 거부) | opt_out_audit 테이블 |
| | §35 (DPIA) | PIA GAP-15 항목 |
| **개보법** | §28의2 (가명정보) | 가명 처리 기록 감사 |

---

## 현재 vs AWS — 규제 준수 비교

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| 감사 로그 불변성 | HMAC + 해시 체인 (로컬) | S3 Object Lock (WORM) + HMAC | 물리적 삭제 방지 |
| 감사 스토어 | DuckDB (fcntl 락) | DynamoDB (서버리스) | 동시성 + 관리 비용 0 |
| 암호화 키 관리 | 환경변수 | AWS KMS + SSM | 키 감사 + 자동 순환 |
| 드리프트 | 커스텀 PSI | SageMaker Monitor + 커스텀 | 관리형 + 커스텀 결합 |
| 데이터 보존 | yaml 정의 (수동 정리) | S3 Lifecycle (자동) | 보존 정책 자동 적용 |
| 인시던트 알림 | 로그만 | SNS → Slack/Email 자동 | 즉시 대응 |
| 거버넌스 보고서 | 로컬 JSON/HTML | S3 + 자동 배포 | 위원회 접근성 |
| API 감사 | 없음 | CloudTrail (자동) | 전체 API 호출 기록 |

---

### 운영/감사 에이전트 통합

09장의 모든 감사/거버넌스 컴포넌트는 AuditAgent의 도구(tool)로 래핑되어 자동 점검에 사용된다:

| 컴포넌트 | 에이전트 도구 | 역할 |
|---|---|---|
| FairnessMonitor | `evaluate_fairness` | AV1 공정성 (단일+교차 보호속성) |
| HerdingDetector | `detect_herding` | AV2 집중도 |
| SelfChecker + XAIQualityEvaluator | `check_reason_quality` + `evaluate_xai_quality` | AV3 추천사유 품질 |
| RegulatoryComplianceChecker + EUAIActMapper + FRIAEvaluator | `run_regulatory_checks` + `evaluate_eu_ai_act` + `evaluate_fria` | AV4 규제 적합성 |
| DataLineageTracker | `trace_feature_lineage` | AV5 데이터 계보 |
| AuditLogger | `verify_audit_chain` | 감사 로그 무결성 검증 |
| GovernanceReportGenerator | 9개 섹션에 에이전트 결과 공급 | 월간 통합 리포트 |

48개 체크리스트 항목이 주기적으로 자동 실행되며, WARN/FAIL 항목은 3-에이전트 합의(Sonnet×3)를 거쳐 마이너리티 리포트를 포함한 진단 결과를 산출한다. 진단 이력은 DiagnosticCaseStore(LanceDB)에 축적되어 유사 케이스 검색과 대응 효과 추적에 활용된다.

상세 설계: `docs/design/11_ops_audit_agent.md`
구현: `core/agent/` 패키지
