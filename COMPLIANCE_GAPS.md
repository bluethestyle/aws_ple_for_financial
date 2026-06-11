# 규제 정합성 — 코드 개선 가능 항목 (Quick Wins)

> 작성: 2026-06-08 · 출처: FSS supplementary 자료(`outreach/04_fss_supplementary.typ`) 규제 매핑 정합성 점검
> 범위: 규제 매핑상 코드가 **미지원**이나 **단기 코드 작업으로 충족 가능**한 항목만. 연구·검증 단계 또는 조직 결정 사항은 마지막 §에 분리.

정합성 점검 결과, 규제 인용·대부분의 시스템 대응은 코드와 일치했습니다. 아래 2건이 "코드 미지원 → 단기 개선 가능"으로 확인된 quick win 입니다.

---

## 1. 자동화평가 **재산출(recompute) 요구권** 미지원 — ✅ 해결 (2026-06-09)

> **해결**: cloud `RequestType.RECOMPUTE` 추가 + `ProfilingWorkflow.request_recompute()` / `fulfill_recompute()`(주입형 `recompute_provider`로 serving 재실행 디커플) + profiling 30일 SLA 공유. 정정→재산출 audit 링크(`correction_request_id`). 온프렘은 `profiling_rights_manager.py`의 `RECALCULATION` + `trigger_recommendation_recalculation()`로 이미 구현돼 있었음(미러 `src/compliance/types.py`에 `RECOMPUTE` 상수 동기화). 테스트 `TestProfilingRecompute` 추가.

| | |
|---|---|
| **규제 근거** | 신용정보법 §36의2 (자동화평가 대응권: 설명·정정·삭제·**재산출** 요구), 개인정보보호법 §37의2 (자동화 결정 거부·설명) |
| **현황(코드)** | `core/compliance/types.py` `RequestType` = `OPT_OUT`, `OPT_OUT_REVOKE`, `PROFILING_ACCESS`, `PROFILING_CORRECTION`, `PROFILING_DELETION`, `EXPLANATION`. **재산출(recompute) 유형 없음.** 정정(`PROFILING_CORRECTION`)은 있으나, 정정 후 자동 재평가/재추천을 트리거하는 경로가 없음. |
| **갭** | 정보주체가 기초정보를 정정한 뒤 "자동평가/추천 재실행(재산출)"을 요구하는 흐름이 없어, §36의2의 네 권리 중 재산출만 미충족. |
| **개선 방안** | ① `RequestType.RECOMPUTE` 추가 → ② `core/compliance/rights/`에 recompute 핸들러 신설(정정된 입력으로 추천/스코어 재실행 → 결과·근거 갱신 → audit 기록) → ③ `SLATracker`에 recompute SLA 정의 연결. |
| **영향 파일** | `core/compliance/types.py`, `core/compliance/rights/` (신규 `recompute.py` 또는 `profiling.py` 확장), `core/compliance/sla_tracker.py`, 예측 경로(`core/serving/predict.py`) 재호출 연계 |
| **작업 규모** | **M** (예측 재실행 경로 연계 필요) |
| **우선순위** | 중 — 상품 추천은 §36의2 준용 대상이나, 네 권리 완결성 확보 차원 |

---

## 2. 신용정보법 §36의2 **전용 설명 요소** 미구분 — ✅ 해결 (2026-06-09)

> **해결**: cloud `RequestType.CREDIT_EXPLANATION` 분기 + `CreditExplanationElements` dataclass(평가 실시 여부·결과·주요 기준·기초정보 lineage 구조화) + `OptOutManager.request_credit_explanation()` / `build_credit_explanation_elements()`. `mark_explanation_provided()`가 두 설명 유형 모두 처리(sla_name 분기). 온프렘은 `ai_decision_opt_out.py`의 `LayeredExplanationResponse`(`personal_info_mapping`=피처→원천, `decision_factors`=주요 기준)로 이미 구조화돼 있었음. 테스트 `TestCreditExplanation` 추가.

| | |
|---|---|
| **규제 근거** | 신용정보법 §36의2 — 설명 대상이 PIPA보다 구체적: *자동화평가 실시 여부 · 결과 · 주요 기준 · 사용된 기초정보* |
| **현황(코드)** | `ExplanationSLATracker`가 **내부 SLA 10일**로 wired(`core/compliance/rights/explanation_sla.py`). 법정 응답기한은 개인정보보호법 시행령 **§44의3⑤(30일)** 이고 10일은 이보다 엄격한 내부 목표치(과준수). `RequestType.EXPLANATION` 단일 유형으로, 신용정보법 §36의2 고유 disclosure 항목(평가 주요 기준·기초정보)을 구조화하지 않음. |
| **갭** | 신용정보법 §36의2 설명 요구를 PIPA 설명과 동일하게 처리 → §36의2가 요구하는 "주요 기준·기초정보" 구성요소가 응답 페이로드에 명시 구조로 없음. |
| **개선 방안** | 설명 응답 템플릿에 §36의2 요소(평가 실시 여부, 주요 기준, 사용된 기초정보 = 피처→원천 lineage)를 포함. 필요 시 `RequestType.CREDIT_EXPLANATION` 분기로 PIPA 설명과 구분. |
| **영향 파일** | `core/compliance/rights/explanation_sla.py`, `core/compliance/rights/opt_out.py`(설명 핸들러), `core/recommendation/reason/`(근거 페이로드), `DataLineageTracker` 연계 |
| **작업 규모** | **S** (기존 lineage·근거 산출물 재구성 수준) |
| **우선순위** | 중 |

---

## 참고 — 코드 quick win 이 **아닌** 항목 (별도 트랙)

정합성 점검에서 "코드 미지원/부분 지원"으로 나왔으나, 단기 코드 작업 범위를 벗어나는 것:

- **CEH(인과 설명 head) · Evidential 불확실성 · Causal Guardrail 활성화** — config로 켤 수 있으나 운영 적용에는 연구·검증이 선행되어야 함(현재 기본 비활성/연구 단계). 코드 추가가 아니라 **검증 과제**.
- **S3 버킷 레벨 Object Lock(COMPLIANCE 모드) + 버저닝 IaC** — put-object 단위 retention은 코드에 있으나 버킷 레벨 프로비저닝(완전 불변)이 IaC에 없음. **인프라 코드(CDK/CloudFormation) 작업** — 중간 규모, 별도 트랙.
- **금융분야 AI 가이드라인 7대 원칙 ① 거버넌스 위원회 공식 설치** — **조직 결정** 사항(코드 무관).

---

## 정합성 점검에서 **이상 없음**으로 확인된 매핑 (참고)

- 금소법 §17 적합성 → `SuitabilityFilter`(`core/recommendation/constraint_engine.py`) ✓
- PIPA §37의2 거부권 → `AIOptOut` + `core/compliance/rights/opt_out.py` ✓
- AI 기본법 §35 FRIA → `KoreanFRIAAssessor` ✓ (준용·선제)
- EU AI Act §9/§11 → `FRIAEvaluator` / `AnnexIVMapper` ✓
- SR 11-7 MRM → `_decide_promotion` + `AuditLogger`(HMAC·해시체인) + 증류 fidelity safety floor ✓
