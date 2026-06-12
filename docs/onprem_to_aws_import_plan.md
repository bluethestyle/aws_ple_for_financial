# 온프렘 → AWS 에이전트 이식 계획 (2026-06-12)

**목적**: 온프렘 repo 의 2026-05-07 이후 89커밋을 전수 분석해 도출한 AWS 이식 후보
15건의 실행 계획과 상태 추적. 전 항목이 양쪽 코드 라인 단위 실측 + 적대적 검증을
통과한 것만 수록 (추정 항목 없음).

**분석 기준 시점**: 2026-06-12
**온프렘 기준**: `c:/Users/user/Desktop/ttm/gotothemoon/workspace/code` (HEAD d74f012c)
**선행 진단 2가지**:
1. 6/9 `ed37117` 로 이식한 `triage()` 와 동적 도구선택조차 **프로덕션 호출자 0건인
   dead code** 였다 — `pipeline_reports.py` 에 배선이 없었음 (PORT-04 에서 해소).
2. 온프렘 `8f006186` 커밋이 "AWS 역이식 권장"으로 명시한 순수 버그 3건이 AWS 에
   잔존했다 (Phase 1 에서 해소).

**관련 문서**: `docs/aws_work_plan.md` (모듈 단위 sync 트랙),
`docs/pipeline_comparison_matrix.md` (4-레이어 전수 비교).

---

## 1. 상태 보드

| ID | 항목 | 우선순위 | 근거 (온프렘 커밋) | 상태 | AWS 커밋 |
|---|---|---|---|---|---|
| P1-버그×3 | diagnoser len(int) / intersectional dataclass 직렬화 / jargon findall | P1 | 8f006186 | ✅ 완료 | bf28627 |
| PORT-02 | tool routing CP5/CP6 도구명 AWS 스키마 정합 (+CP7 신설) | P2 | 2026-06-09 sync 잔여 | ✅ 완료 | d2f4a1e |
| PORT-04 | investigate() 이식 + run_pipeline_reports verdict 분기 배선 | P2 | 7bedef70 | ✅ 완료 | 7b66666 |
| PORT-03 | verify_grounding (할루시네이션 더블체크) | P2 | e7c12a36 | ✅ 완료 | 0949555 |
| PORT-05 | triage 프롬프트 금융 AIOps 심각도 4요소 (AWS 규제 어휘) | P2 | c3df4317 | ✅ 완료 | 49916ba |
| PORT-07 | scripts/agent_healthcheck.py (AWS 스택 치환) | P2 | d74f012c | ✅ 완료 | 6c82d0f |
| PORT-08 | HITL consumer 진입점 + 큐 백엔드 영속화 | P2 | 8f38dece | ✅ 완료 | 14aa2df |
| PORT-06 | CloudWatch LogAnalyzer + query_cloudwatch_logs 도구 | P2 | b67db29a~939960a0 | ✅ 완료 | 7159f64 |
| PORT-09 | 동기화 문서 정비 (본 문서 + matrix/work_plan 정정) | P2 | — | ✅ 완료 | (본 커밋) |
| PORT-10 | _embed_finding → Bedrock Titan Embed v2 옵트인 provider | P3 | 온프렘 bge-m3 등가 | 진행 예정 | — |
| PORT-12 | LeakageValidator 누수 분류학 명문화 (docstring/docs) | P3 | — | 진행 예정 | — |
| PORT-13 | quality_gate degenerate 가드 + schema_changed 경고 게이트 | P3 | — | 진행 예정 | — |
| PORT-11 | 충족도 매트릭스 문서 | P3 | — | ⏸ 별도 트랙 (V2 publish/outreach) | — |
| PORT-15 | kill_switch TTL 캐시 | P3 | — | ⛔ 보류 — fail-closed 의미론 충돌 검토 전 도입 금지 | — |

상태 갱신 규칙: 항목 완료 시 이 표의 상태/커밋 칼럼만 갱신한다. 세부 구현 설명은
각 커밋 메시지가 단일 진실 공급원.

## 2. 핵심 배선 구조 (Phase 2 결과)

```
run_pipeline_reports (scripts/submit_pipeline.py Step 5)
 ├─ ops/audit 보고서 생성 (기존)
 ├─ [REPORTS_LLM_TRIAGE_ENABLED=1] verdict 분기            ← PORT-04
 │    ├─ ops RED / audit HIGH  → BedrockDialogSession.investigate()
 │    ├─ ops YELLOW / audit MEDIUM → triage()  (금융 4요소)  ← PORT-05
 │    └─ [REPORTS_LLM_VERIFY_GROUNDING≠off] grounding_check ← PORT-03
 │       → 보고서 JSON llm_followup 키
 ├─ [REPORTS_LOG_ANALYSIS_ENABLED=1] CloudWatch 로그 증분 분석 ← PORT-06
 │    ERROR=자동 FIX_NOW+investigate, WARNING만 LLM triage
 │    → 보고서 JSON log_analysis 키
 └─ S3 업로드 (첨부 결과 포함)
```

LLM 비용 옵트인 env (전부 기본 off):

| env | 기능 | 기본 |
|---|---|---|
| `REPORTS_LLM_TRIAGE_ENABLED` | verdict 분기 investigate/triage | off |
| `REPORTS_LLM_VERIFY_GROUNDING` | 결론↔도구결과 대조 (위 플래그 안에서) | on (off 로 끄기) |
| `REPORTS_LOG_ANALYSIS_ENABLED` | CloudWatch 로그 증분 분석 | off (LLM 없이 룰 분류) |

## 3. 이식 제외 항목과 근거 (환경 특이)

| 온프렘 항목 | 제외 근거 |
|---|---|
| GPU 가드 (`gpu_guard.py`, llm_should_skip) | 로컬 RTX 4070 학습/LLM 경합 보호 장치. AWS 는 Bedrock(관리형)이라 경합 없음 |
| qwen/exaone 모델 벤치마크, /no_think 등 소형모델 트랩 대응 | Ollama 로컬 모델 특이. Bedrock Claude 는 tool-calling/결론 누락 트랩 빈도가 다름 (_synthesize 만 이식) |
| lancedb 백엔드 마이그레이션 세부 | AWS case_store 는 동일 backend=auto 패턴 기보유 |
| Ollama 연결/모델 헬스체크 항목 | PORT-07 에서 Bedrock model id 유효성 점검으로 치환 |
| Airflow DAG 배선 (dag_audit_agent_monitoring 등) | AWS 는 submit_pipeline Step 5 가 동일 역할 (CLAUDE.md §3.2 의도적 차이) |
| SMTP 알림 | AWS 는 send_notification 도구 (SNS) 경로 |
| query_local_metrics 도구 | AWS 는 query_cloudwatch_metrics / query_cloudwatch_logs (PORT-06) 가 등가 |

## 4. 검증 기준

- 시작 베이스라인: 698 passed / 0 fail (2026-06-12). 모든 Phase 커밋 후 전체
  테스트 무회귀 확인.
- 신규 LLM 기능은 전부 env 옵트인 기본 off (위 표) — CI/dry-run 에서 Bedrock
  호출 0.
- `scripts/agent_healthcheck.py` 기본 모드가 도구 스키마↔구현↔라우팅 정합을
  상시 점검 (PORT-02 회귀 방지 런타임 가드).
