# Operational Scaling Guide: 운영 단계별 확장 가이드

이 문서는 서비스 성장에 따른 인프라/운영 확장 결정 포인트를 정리합니다.
코드는 이미 준비되어 있으며, **트래픽 규모에 따라 설정만 변경**하면 됩니다.

---

## Phase 1: 초기 (일 1,000건 미만)

### Lambda Cold Start 대응
- **현재**: `ReservedConcurrency: 100` (동시 실행 상한만, 비용 0)
- **대응**: EventBridge 5분 warm-up ping 추가
  ```
  aws/eventbridge/retrain_schedule.json에 warm-up rule 추가
  ScheduleExpression: rate(5 minutes)
  Input: {"action": "warmup"}
  ```
- **비용**: ~$0/월
- **효과**: 대부분의 요청이 warm 응답 (~50ms)

### Feature Store
- **현재**: `MemoryFeatureStore` (Parquet → 메모리)
- **기준**: 사용자 수 500만 이하면 유지
- **비용**: Lambda 메모리 내 포함 (추가 비용 0)

### 모델 서빙
- **현재**: Lambda 단일 champion 모델
- **A/B 테스트**: 비활성 (`AB_ENABLED=false`)
- **모니터링**: prediction_log만 기록 (Champion-Challenger 미실행)

---

## Phase 2: 성장 (일 10,000~100,000건)

### Lambda Cold Start 대응
- **전환**: Provisioned Concurrency 도입
  ```yaml
  # serving_stack.yaml
  ProvisionedConcurrency: 5              # 영업시간
  # + Application Auto Scaling
  # 09:00-22:00 KST: 10개
  # 22:00-09:00 KST: 2개
  ```
- **비용**: ~$10-15/월
- **효과**: cold start 완전 제거

### A/B 테스트 활성화
- **전환**: Lambda 환경변수 변경만으로 활성화
  ```
  AB_ENABLED=true
  AB_CHALLENGER_VERSION=v-new-model
  AB_CHALLENGER_WEIGHT=0.1    # 10% 카나리 시작
  ```
- **비용**: 모델 2개 메모리 로드 (Lambda 메모리 1024→2048MB 고려)

### Champion-Challenger 자동 평가
- **전환**: ModelMonitor.evaluate_champion_challenger() 주기 실행
- **방법**: EventBridge 주간 스케줄 → Lambda 호출
- **기준**: min_samples=5,000 이상 시 통계 검정

### 자동 재학습 루프 활성화
- **현재**: auto_retrain_trigger Lambda + EventBridge 매일 (이미 구현)
- **전환**: drift_report_generator Lambda도 활성화
- **조건**: PSI > 0.25 또는 30일 경과 시 자동 트리거

---

## Phase 3: 대규모 (일 100,000건 이상)

### Feature Store 전환
- **전환**: `DynamoDBFeatureStore` (자동 전환)
  ```yaml
  # ServingConfig
  feature_store: auto
  auto_feature_threshold: 5_000_000  # 500만 사용자 이상 시 자동 전환
  ```
- **비용**: DynamoDB PAY_PER_REQUEST (읽기 건당 ~$0.0000025)

### Lambda → ECS 전환 검토
- **기준**: `ServingConfig.auto_threshold: 150_000_000` (월 1.5억 건)
- **현재**: Lambda 단일로 충분 (일 10만건 = 월 300만건)
- **ECS 전환 시점**: 월 1.5억건 초과 시
  ```yaml
  serving:
    mode: auto            # 자동 판단
    auto_threshold: 150_000_000
  ```

### 모델 Layer 패키징 (선택)
- **목적**: S3 다운로드 시간 제거 (cold start 추가 최적화)
- **조건**: 16개 LGBM 모델 합계 < 250MB
- **방법**: Lambda Layer로 모델 포함, /opt/models/ 경로에서 직접 로드

---

## Phase 4: 엔터프라이즈 (월 1억건 이상)

### 실시간 Feature Store (Kafka/Flink)
- **현재**: 배치 피처만 지원 (일 1회 갱신)
- **필요 시점**: 실시간 피처(최근 1시간 거래 패턴)가 모델 성능에 유의미한 차이를 보일 때
- **구현**: Kinesis Data Streams → Flink → DynamoDB (실시간 피처 업데이트)
- **비용**: 상당 (Kinesis + Flink 클러스터)
- **판단**: 배치 피처만으로 AUC > 0.8 유지되면 불필요

### Agentic MLOps (Level 2)
- **현재**: 규칙 기반 자동 재학습 (Level 1)
- **전환 조건**: 운영 6개월 이상, 재학습 이력 데이터 충분
- **구현**: LLM Agent가 drift report 해석 → 재학습 판단
  ```
  현재: PSI > 0.25 → 무조건 재학습
  Agent: "PSI 0.22인데 계절성인가? → 이전 3월 패턴과 비교 → 재학습 불필요"
  ```
- **주의**: Agent 판단의 감사 추적 필수 (AI기본법)

### LLM 사유 배치 추론
- **현재**: L2a SQS 건별 처리
- **전환**: Bedrock Batch Inference API (50% 비용 할인)
- **조건**: L2a 건수가 월 10만건 이상일 때 비용 효과

---

## 비용 요약

| Phase | 일 요청 | Lambda | DynamoDB | 기타 | 월 합계 |
|-------|---------|--------|----------|------|---------|
| 1 | ~1K | ~$1 | ~$1 | - | **~$5** |
| 2 | ~50K | ~$10 + PC $15 | ~$5 | A/B 모니터링 | **~$40** |
| 3 | ~300K | ~$50 + PC $20 | ~$30 | 재학습 SageMaker | **~$200** |
| 4 | ~3M | ECS ~$200 | ~$100 | Kinesis/Flink | **~$1,000+** |

PC = Provisioned Concurrency

---

## 규제 확장

### AI기본법 시행 (2026.01.22)
- **유예기간**: 최소 1년 (2027.01까지 계도기간)
- **현재 준수 상태**: 90%+ (이의제기 재심사, 모니터링 owner 완료)
- **Phase 2에서 추가**: FairnessMonitor 주간 실행, RegulatoryChecker 분기 실행

### 금감원 AI RMF
- **성격**: 자율규제 (법적 강제력 없음)
- **현재**: monitoring.yaml에 owner/schedule 정의 완료
- **Phase 2에서 추가**: 거버넌스 보고서 자동 생성 (GovernanceReportGenerator 활성화)

---

## 변경 이력

| 일자 | 항목 | 비고 |
|------|------|------|
| 2026-03-18 | 최초 작성 | 4개 Phase 정의 |
