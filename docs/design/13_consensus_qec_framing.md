# 13. 컨센서스의 양자오류정정(QEC) 재해석 — 개념 토대 + AWS↔온프렘 교차 비교

> 작성일 2026-06-16. 온프렘(gotothemoon) 세션에서 넘어온 QEC 재해석을, **AWS 코드 실제와 대조**해 반영한 문서.
> QEC 프레임은 양 프로젝트가 공유하되, 본 문서의 모든 서술은 AWS 코드의 실제 동작 기준이다(온프렘 서술 복붙 아님).
>
> AWS 합의 소스:
> - `core/agent/consensus.py` — `ConsensusArbiter`, `ConsensusResult`, `_classify` (결정규칙)
> - `core/agent/pipeline_reports.py` — Bedrock provider + Ops/Audit arbiter 배선, 실제 perspective
> - `core/agent/ops/reporter.py`, `core/agent/audit/reporter.py` — `_consensus.evaluate()` 호출부

---

## 1. 프레임 — 컨센서스 = 양자오류정정, 그리고 model capability가 설계를 가른다

작은 LLM의 출력은 **측정 전의 노이지 물리 큐빗**과 같다. 출력은 확률변수이고, 샘플링하는 순간이 측정이며 collapse다. 컨센서스가 푸는 문제는 양자컴퓨팅의 QEC가 푸는 문제와 동형이다 — **신뢰 못 할 물리 큐빗 여러 개로 신뢰할 수 있는 논리 큐빗 하나를 만드는 것**.

가져오는 것은 양자 *speedup*이 아니라 **QEC의 엔지니어링 규율**이다. 구체적으로 threshold theorem(임계 정리), 상관 노이즈(correlated error), syndrome(오류 서명) 측정. 얽힘, 유니타리성, 지수적 상태공간 같은 양자 우위의 원천은 넘어오지 않는다 — 우리 투표자들은 양자 상관이 아니라 고전 상관으로 묶여 있다.

### 1.1 왜 AWS와 온프렘 방식이 다른가 — threshold theorem 축

이 프레임은 단순 비유가 아니라 **두 합의 방식이 왜 다른지를 예측한다.** QEC의 1번 정리(**threshold theorem**): 물리 큐빗 에러율 *p*가 임계값 *p_th* **아래일 때만** 큐빗을 늘리고 코드를 중첩(concatenate)할수록 논리 에러가 내려간다. *p*가 임계 위면 큐빗을 더 넣을수록 **더 나빠진다.**

AWS와 온프렘 합의 구조의 차이는 임의 설계 취향이 아니라, **각 모델이 임계선의 어느 쪽에 있느냐의 직접적 귀결**이다.

| | **AWS** | **온프렘** |
|---|---|---|
| 물리 큐빗(모델) | 고성능 Sonnet — per-vote 에러 **임계 한참 아래** | 한정적 sLLM(Qwen 14B 등) — per-vote 에러 **임계에 근접** |
| 필요한 중복도 | 낮음 → **3표 독립 투표로 충분** | 높음 → **5~7표 필요** |
| 코드 구조 | 단일 라운드(독립 투표만) | **2-Round Hybrid = 코드 중첩(concatenation)**: Round1 독립 + Round2 심의로 한 번 더 에러 억제 |

- **AWS가 1라운드 3표로 끝낼 수 있는 이유**: Sonnet의 단일 투표 에러가 이미 임계 아래라, 적은 중복으로도 논리 에러가 충분히 낮다. 심의 라운드가 불필요하다.
- **온프렘이 5~7표 + 2라운드를 *써야 하는* 이유**: sLLM 단일 투표 에러가 임계에 가까워, (a) 물리 큐빗을 늘리고(5~7표) (b) 코드를 중첩(Round2 심의)해야 논리 에러가 내려간다. 단 순수 델파이(순차 심의)는 수렴 편향으로 소수의견을 지우므로, Round1 독립(마이너리티 확정)을 보존한 하이브리드로 절충한 것이다.

즉 **"고성능 Sonnet이냐, 한정적 sLLM이냐"가 곧 threshold theorem 축이고, 그것이 3표-단일라운드 ↔ 5~7표-2라운드라는 구조 차이를 만든다.**

#### 설계 가드레일 (AWS)
AWS의 "3표 독립, 심의 없음"은 **Sonnet이 임계 아래라는 가정** 위에 선다. 따라서:
- 이 가정은 **검증 대상**이다 — Sonnet이 ops/audit 판정에서 실제로 임계 아래인지(단일 투표 에러율)를, §4 탈상관 실측과 함께 측정해야 한다.
- **모델을 약한 것으로 다운그레이드하면(예: 비용 절감용 Haiku 등) 이 구조가 더 이상 안전하지 않을 수 있다.** 그 경우 온프렘 방향(투표자 증설 + 심의 라운드)으로 이동해야 한다 — 모델 교체는 quorum/라운드 재설계를 *동반*해야 하며, 단순 model_id 교체로 끝나지 않는다.

---

## 2. AWS 코드 실제 — "코드 존재 ≠ 실가동" 대조

온프렘에서 PASS측 비대칭이 코드에 *없었던* 것이 발견됐으므로, AWS도 동일한 검증을 거쳤다. 결과:

**합의는 배선되어 발화한다 — 단 조건부다.**

- `pipeline_reports.py`의 `_build_ops_arbiter()`(L1144)와 `_build_audit_arbiter()`(L1160)가 `ConsensusArbiter`를 생성해 각각 Ops/Audit reporter에 주입한다(`consensus_arbiter=...`).
- reporter는 **attention 항목(WARN/FAIL)마다** `self._consensus.evaluate()`를 호출한다(`ops/reporter.py:153-176`, `audit/reporter.py:~150`). `apply_to: checklist_warn_fail`(agent.yaml)과 일치 — 정상 항목에는 합의를 돌리지 않는다.
- **조건부 발화**: `_build_sonnet_provider()`(L1102)가 boto3/Bedrock import에 실패하면 `None`을 반환하고, arbiter도 `None`이 된다. 그러면 reporter의 `if self._consensus and attention:`이 거짓이 되어 **합의는 발화하지 않고**, reporter는 룰 엔진 severity만으로 판정한다. 즉 합의는 **Bedrock 가용 + WARN/FAIL 항목 존재** 시에만 작동하는 *조건부 보강층*이다(핵심 룰 판정은 합의 없이도 성립).

**실제 perspective는 docstring 기본값(alpha/beta/gamma)이 아니다.**

`consensus.py`의 모듈 레벨 `_PERSPECTIVES`(alpha 보수 / beta 통계 / gamma 비즈니스)는 *기본 폴백값*일 뿐, 프로덕션 배선은 이를 override한다. 실제 패널은 **이해관계자 역할** 기반이다.

| 패널 | perspective (config 키) | 역할 |
|---|---|---|
| **Ops** (`_OPS_PERSPECTIVES`, L1016) | `sre` | 인프라 가용성, 콜드스타트, p50/p95 지연, 에러율, 롤백 |
| | `mlops` | 학습 수렴(loss/val_auc/epochs), 증류 피델리티, 피처 상태 |
| | `biz` | 지연/가용성/정확도 → 고객 전환·이탈·만족 KPI |
| **Audit** (`_AUDIT_PERSPECTIVES`, L1040) | `regulator` | 금소법·AI기본법, EU AI Act, FRIA, 제13조 설명의무 |
| | `risk` | 편향(DI), 드리프트, 설명가능성, SR 11-7 MRM |
| | `audit_trail` | HMAC 서명, hash chain, 증빙 보존, 서명 체인 |

**실행은 순차다.** 두 arbiter 모두 `parallel: False`로 생성된다(L1154, L1171). 메커니즘(`consensus.py`)은 `ThreadPoolExecutor` 물리 병렬을 *지원*하지만, 배선된 Ops/Audit arbiter는 **순차 실행**한다. (온프렘 핸드오프의 "AWS = ThreadPoolExecutor 물리 병렬" 서술은 메커니즘의 *능력*이지 실제 배선이 아니다.)

**다양성 축은 perspective 분리 단독이다.** `_single_vote`는 `generate()`에 temperature를 전달하지 않으므로, 모든 투표가 provider 기본값(`_BedrockProvider.generate`의 `temperature=0.3`)으로 고정된다. 온프렘의 temperature 0.3~0.7 변동과 달리 **AWS는 temperature 변동이 없고, 다양성은 전적으로 시스템 프롬프트(perspective) 차이에서 나온다.**

**model/region**: pipeline.yaml의 `llm_provider.bedrock.models.agent_consensus.model_id` / `aws.region`를 해석하며, 미설정 시 폴백은 `ap-northeast-2` / `global.anthropic.claude-sonnet-4-6`(L1064-1065).

---

## 3. AWS 비대칭 정족수 = biased-noise code (cat qubit)

결정규칙(`_classify`, `consensus.py:272-340`)은 **단순 다수결이 아니다.** 발화 확인됨:

1. **FAIL이 1표라도 있으면 → 최종 FAIL** (`consensus_type="majority"`), 나머지 의견은 minority_report로 보존.
2. **PASS는 만장일치(3/3)일 때만 통과** (`consensus_type="consensus"`).
3. 만장일치 WARN → WARN.
4. 그 외 불일치(2P+1W 등) → **WARN** + minority_report (PASS는 만장일치가 아니면 통과 불가).

### QEC 해석
이것은 **편향 노이즈 코드**다. cat qubit이 노이즈가 위상플립 쪽으로 강하게 편향돼 있을 때 그 방향에 보호를 몰아주듯, AWS 정족수는 **비싼 에러 방향에 보호를 몰아준다.** FAIL측은 1표에 에스컬레이션(민감 정족수), PASS측은 만장일치 요구(보수 정족수).

### 정당화 — cost-ratio 명문화 (지금까지 암묵적이던 것)

`_classify`의 주석은 규칙을 *기술*하지만 왜 비대칭인지의 근거는 코드에 명시돼 있지 않다. 명문화한다:

> 비대칭은 **오류 비용비**에 근거한다. 운영/감사 맥락에서
> **false PASS**(위험·규제위반·드리프트를 정상으로 통과시킴)의 비용은
> **false FAIL**(정상 항목을 보류·재검토로 돌림)의 비용보다 **압도적으로 크다.**
> 따라서 PASS측에는 보수적 정족수(만장일치), FAIL측에는 민감 정족수(1표 에스컬레이션)를 둔다.
> 이 cost-ratio가 quorum 설계의 안전성 근거이며, **비용비가 달라지면 quorum도 재조정해야 한다.**

cat qubit이 *실측된* 노이즈 편향에 코드를 맞추듯, 이 정족수도 false PASS/false FAIL 비용비에 묶여 있어야 원리적으로 정당하다. 현재 비용비는 정성적(false PASS ≫ false FAIL)이며, 정량화는 후속 과제다.

### 온프렘 정합성 (Part A)
온프렘 2-Round Hybrid에는 이 **PASS측 비대칭(만장일치만 PASS, 그 외 WARN)이 누락**돼 있었고, 온프렘이 이번에 이식해 AWS와 정합화했다. **AWS가 이 규칙의 기준(reference)이다.** 본 문서가 cost-ratio 근거를 명문화함으로써, AWS↔온프렘 양쪽 결정규칙의 정합성이 추적 가능해진다.

---

## 4. Perspective 분리 = Pauli twirling 부분 구현

QEC에서 계통오차(coherent error)는 반복해도 평균으로 사라지지 않고 누적된다. **Pauli twirling / randomized compiling**은 계통오차를 랜덤화해 stochastic noise로 바꾼 뒤에야 평균이 작동하게 한다.

AWS의 perspective 분리(Ops sre/mlops/biz, Audit regulator/risk/audit_trail)는 서로 다른 시스템 프롬프트로 상관 오차를 **일부 탈상관**시키려는 twirling의 부분 구현이다. LLM 앙상블의 지배적 실패는 세 투표자가 *같은 틀린 답*을 자신 있게 내는 상관 할루시네이션인데, 관점을 직교화하면 그 공유 편향이 부분적으로 랜덤화된다.

### 한계 — 명문화
- **known-answer 프로브로 투표자 간 상관행렬을 실측하지 않는다.** 따라서 탈상관 효과는 *설계 가정*이지 검증된 값이 아니다.
- 프롬프트 순서, few-shot 예시, 프레이밍의 직교화는 적용돼 있지 않다.
- → AWS도 온프렘 Part C(탈상관 측정)의 채택 검토 대상이다(§6).

---

## 5. minority_report 보존 = deferred measurement (측정 유예)

`ConsensusResult.minority_report`는 식별되면 삭제되지 않는다(`consensus.py` docstring + `_classify`). 그리고 reporter가 **모든 vote의 full reasoning + minority_report를 report 항목에 영속화**한다(`ops/reporter.py:146-176`의 주석: *"per-persona argument IS the audit trail … the dissenting opinions that would otherwise disappear behind 'majority'"*).

### QEC 해석
이것은 **principle of deferred measurement(측정 유예 원칙)** 다 — 소수 분기를 마지막(사람 검토)까지 collapse하지 않는다. 다수결이 소수의견을 "majority" 뒤로 지워버리는 대신, AWS는 소수 분기를 끝까지 보존해 *"왜 이 소수의견이 사라졌나"* 라는 규제기관 질문에 답을 남긴다.

---

## 6. AWS가 채택 검토할 온프렘 신규 (교차 비교)

### Syndrome 측정층 (온프렘 Part B) — AWS 채택 후보 ★
QEC는 논리 큐빗을 직접 측정(붕괴)하지 않고, 안실라(ancilla)로 **stabilizer(패리티 체크)만** 측정해 데이터를 건드리지 않고 오류 서명만 뽑는다.

컨센서스 응용: PASS/WARN/FAIL을 **재투표**하지 말고, 답을 안 건드리는 **결정론적 안실라**로 오류 서명만 측정한다. AWS에 이미 있는 재료:
- **룰 엔진 판정** — `evaluate()`에 이미 `rule_engine_verdict`로 전달됨. "합의 결과가 룰 엔진과 모순되는가?" 체크 가능.
- **grounding validator**(`core/agent/audit/grounding_validator.py`) — "인용한 수치가 measurements에 실재하는가?" = 안실라.
- **단조성/invariant 체크** — 방향성 위반 탐지.

이 체크들의 패턴(syndrome)이 에러 *종류*를 국소화하며, 답 재생성보다 싸고 신뢰도 높다. **AWS도 룰 엔진 + grounding validator가 이미 있으므로 신규 LLM 호출 0회로 얹을 수 있다.** "투표자 늘리기"가 아니라 "오류 서명 측정". 신규 개발이 가장 작은 채택 후보.

### 투표자 탈상관 측정 (온프렘 Part C 후속)
AWS는 perspective 분리가 이미 있어 이 축에서는 온프렘보다 앞서 있다. 다만 **상관행렬 실측 + 프롬프트 순서/few-shot/프레이밍 직교화는 양쪽 다 미완**이다(§4 한계). 공통 후속 과제.

---

## 7. AWS ↔ 온프렘 컨센서스 비교표

| 항목 | **AWS independent_voting** | **온프렘 2-Round Hybrid** |
|---|---|---|
| 모델 | Bedrock Sonnet ×3 | 로컬 GPU (Qwen 계열 등) |
| 실행 방식 | **순차** (`parallel:False`; 병렬 지원하나 미사용) | Round1 독립 투표 + Round2 순차 심의 |
| 에이전트 수 | 3 (Ops·Audit 각 패널) | Round1 5(고위험 7) + Round2 2 |
| 다양성 축 | **perspective 분리 단독** (이해관계자 역할; temperature 0.3 고정) | temperature 변동(0.3~0.7) + perspective |
| 결정규칙 | **비대칭 정족수** (FAIL 1표 / PASS 만장일치) — **기준(reference)** | AWS 비대칭 이식 완료(Part A) |
| Syndrome 측정 | 미구현 — **채택 후보**(룰 엔진+grounding validator 재료 보유) | 신규 도입(Part B) |
| 탈상관 실측 | 미구현 (perspective 분리는 보유) | 신규/진행(Part C) |
| 소수의견 보존 | `minority_report` + 전 vote reasoning 영속화 | HITL + 영속화(Part C/14) |
| cost-ratio 명문화 | **본 문서에서 명문화** | — |
| 실가동 조건 | Bedrock 가용 + WARN/FAIL 항목 존재 시 (없으면 룰엔진 폴백) | (온프렘 문서 참조) |

---

## 8. Cross-reference

**온프렘 설계 전문** (온프렘 레포 기준 경로 — 본 워크스페이스에서는 직접 접근 불가):
- `docs/design/12_consensus_syndrome_measurement.md` — syndrome 측정 + 비대칭(Part A·B)
- `docs/design/13_consensus_voter_decorrelation.md` — 투표자 탈상관 + HITL(Part C)
- `docs/design/14_consensus_output_persistence_and_traceability.md` — 영속화 + 규제 추적성

**AWS 관련 문서/코드**:
- `docs/design/11_ops_audit_agent.md`, `docs/design/11_ops_audit_agent_onprem_handoff.md`
- `core/agent/consensus.py` (`_classify` 결정규칙), `core/agent/pipeline_reports.py` (배선·perspective)
- `core/agent/ops/reporter.py`, `core/agent/audit/reporter.py` (호출부)
- `configs/financial/agent.yaml` (`agent.consensus`)

**이 프레이밍이 반영된 surface** (2026-06-16):
- Paper 2 §"Environment-Adaptive Consensus Mechanism" — design rationale 1문단(threshold theorem + biased-noise code + cost-ratio) + 본 문서 링크 (EN/KO)
- 규제기관 자료(`docs/typst/{en,ko}/regulatory_framework`, `regulatory_summary`) — 양자 용어 없이 실질(비대칭 fail-safe 비용근거 + 소수의견 감사추적)만 보강
- `core/agent/consensus.py` docstring — "in parallel"이 모듈 기본값임을 명시(실배선은 순차) + 실제 perspective + biased-noise code 근거 + 본 문서 링크

---

## 9. 후속 작업 (AWS)
- [ ] **cost-ratio 정량화**: false PASS / false FAIL 비용비를 정성→정량으로. 그 위에서 quorum 재검토.
- [ ] **Syndrome 측정층 시제**(Part B 이식): 룰 엔진 모순 체크 + grounding validator + invariant 안실라를 합의 앞단에 결정론적 층으로. 신규 LLM 호출 0회 목표.
- [ ] **투표자 상관행렬 실측**(Part C): known-answer 프로브로 perspective 간 상관 측정, 직교화 효과 검증.
- [x] consensus.py docstring "in parallel"→모듈 기본값 명시(실배선 순차) + 본 문서 링크 (2026-06-16 완료).

### 9.1 미해결 정합 플래그 (코드 대조 중 발견 — 단독 수정 보류)
- **Paper 2 결과표의 `α/β/γ` agent 표기 vs 실제 stakeholder 패널** (Ops: SRE/MLOps/Biz, Audit: Regulator/Risk/AuditTrail): 해당 프로덕션 테스트가 실제 패널로 돌았는지 vs 구 alpha/beta/gamma 기본값으로 돌았는지 **provenance 확인 후** 라벨 정합 필요(데이터 왜곡 방지). 확인 전까지 표는 손대지 않음.
- **온프렘 합의 모델 표기 드리프트**: Paper 2 = Exaone 3.5 2.4B / `regulatory_framework` = Qwen 2.5 14B Q4. **온프렘 세션이 권위** — 온프렘 확정 후 AWS 측 표기를 일괄 정합.
