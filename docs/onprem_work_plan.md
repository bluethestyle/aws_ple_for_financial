# On-Prem 프로젝트 작업 계획

**목적**: AWS 프로젝트에서 확정된 Findings 1~13 을 온프렘(gotothemoon) 프로젝트로 이식하고 실데이터 검증 준비. 이 문서는 **온프렘 쪽에서 실행할 작업**의 계획서.

**기준 시점**: 2026-04-20
**AWS 기준**: `aws_ple_for_financial` main 브랜치 (commit b58f07a 이후)
**온프렘 기준**: `c:/Users/user/Desktop/ttm/gotothemoon/workspace/code/`
**온프렘 ablation 설계서 최종 수정**: 2026-04-13 (AWS Findings 8~13 반영 전)

---

## 1. 코드 동기화 Gap 분석

### 1.1 Causal Expert (`src/models/experts/causal_expert.py`)

| AWS 기능 | 온프렘 상태 | 필요 작업 |
|---|---|---|
| `recon_lambda=0.5` (Finding 8) | ✅ 있음 | — |
| W init scale = 0.1 하드코딩 | ✅ 있음 | — |
| `w_init_scale` config-driven (Finding 11) | ❌ 없음 | config 파라미터로 분리 |
| CEH `attribution_head` + `_last_attribution` 캐시 | ❌ 없음 | 이식 |
| `ceh_target_mode` {raw, demeaned, primary_task} | ❌ 없음 | 이식 |
| `get_last_attribution()` 공개 accessor | ❌ 없음 | 이식 |
| `get_causal_latent()` 공개 accessor | ❌ 없음 | 이식 |
| `get_causal_coherence_score()` | ❌ 없음 | 이식 |
| `get_counterfactual()` (CCP) | ❌ 없음 | 이식 |
| `set_attr_target_external()` (v3) | ❌ 없음 | 이식 |

### 1.2 PLE Model (`src/models/ple_model.py`)

| AWS 기능 | 온프렘 상태 | 필요 작업 |
|---|---|---|
| `get_ceh_attribution()` 헬퍼 | ❌ 없음 | 이식 |
| `get_causal_coherence()` 헬퍼 | ❌ 없음 | 이식 |
| `_inject_ceh_v3_target()` (v3 gradient 주입) | ❌ 없음 | 이식 (선택 — v3 negative 결과 고려 시 생략 가능) |
| DAG regularization loss (`get_dag_regularization()`) | 확인 필요 | grep 결과 없음 → 학습 루프가 recon_lambda 를 사용하고 있는지 재확인 |
| CEH attribution loss 통합 | ❌ 없음 | 이식 |

### 1.3 Monitoring (`src/monitoring/`)

| AWS 기능 | 온프렘 상태 | 필요 작업 |
|---|---|---|
| `AuditLogger` 기본 | ✅ 있음 (`audit_logger.py`) | — |
| `log_model_inference` | ✅ 있음 | — |
| `log_attribution` (Paper 2 v2) | ❌ 없음 | 이식 |
| `log_guardrail` (Paper 2 v2) | ❌ 없음 | 이식 |
| `log_model_promotion` (Champion-Challenger) | 확인 필요 | 있으면 유지, 없으면 이식 |
| `CausalGuardrail` class (`causal_guardrail.py`) | ❌ 없음 | 이식 |
| HMAC + hash chain 인프라 | 확인 필요 | 있을 가능성 높음, 호환성 점검 |

### 1.4 Config (`configs/model_config.yaml`)

| 필드 | 온프렘 상태 | 필요 작업 |
|---|---|---|
| `model.causal.recon_lambda: 0.5` | ✅ 있음 | — |
| `model.causal.w_init_scale: 0.1` (기본) | ❌ 없음 | 필드 추가 |
| `model.causal.ceh.enabled: false` | ❌ 없음 | 서브블록 추가 |
| `model.causal.ceh.hidden_dim: 64` | ❌ 없음 | 추가 |
| `model.causal.ceh.loss_weight: 0.1` | ❌ 없음 | 추가 |
| `model.causal.ceh.target_mode: "raw"` | ❌ 없음 | 추가 |
| `model.causal.ceh.primary_task_name: "churn_signal"` | ❌ 없음 | 추가 |

### 1.5 설정 (`src/config/` or config builder)

AWS 의 `config_builder.py` 에 해당하는 온프렘 위치 확인 필요. HP flag 패턴:
- `use_ceh`
- `ceh_target_mode`
- `ceh_primary_task`
- `w_init_scale`
- `recon_lambda` (HP 노출)

---

## 2. Ablation 프레임워크 Gap

### 2.1 현재 온프렘 Ablation (`ablation_test/configs/experiments.yaml`)

45+ 개 시나리오, 우선순위 P0~P7:
- **P0**: BASELINE (1개)
- **P1**: Feature Group Leave-One-Out (F1~F7, 7개)
- **P2**: Expert Leave-One-Out (추정)
- **P3~P6**: FSP / X / P 시리즈
- **P7**: DeepFM-from Build-Up (B0~B6)

**구조적 중심**: feature × expert 조합. **Findings 1~13 에 해당하는 실험은 없음**.

### 2.2 신규 필요 시나리오 (AWS Findings → 온프렘 ablation)

| AWS Finding | 필요 온프렘 시나리오 이름(예) | 설정 포인트 |
|---|---|---|
| **F1** UW bug | `EXP-UW1` (UW off vs on) | loss_weighting.strategy |
| **F2** Softmax vs sigmoid | `EXP-GATE1` (gate_type) | cgc.gate_type |
| **F3** UW convergence | `EXP-UW2` (SB vs PLE UW) | 구조 + UW |
| **F4** Epoch budget | `EXP-EP1` (10 vs 30 epoch) | training.epochs |
| **F5** GTE mixed-group | `EXP-GTE1` | use_group_task_expert |
| **F6** Gate entropy | `EXP-ENT1` (분석 전용) | 기존 baseline 재분석 |
| **F7** Fusion recipes | `EXP-FUSE1~6` (BRP-detached, NEAS, 조합) | fusion flags |
| **F8** W=0 patch | `EXP-DAG1` (recon_lambda on/off) | causal.recon_lambda |
| **F9** CEH v1/v2 | `EXP-CEH1` (off), `EXP-CEH2` (raw), `EXP-CEH3` (demeaned) | ceh.target_mode |
| **F10** CG v1/v2 | 분석 전용 (학습 후 post-hoc) | — |
| **F11** W-amp | `EXP-WAMP1` (w_init_scale=0.3, recon_lambda=2.0) | causal.{w_init_scale, recon_lambda} |
| **F12** CCP | 분석 전용 (W-amp 체크포인트에서 post-hoc) | — |
| **F13** CEH v3 | `EXP-CEH4` (primary_task target) | ceh.target_mode |

**설계 권장**: 기존 P0~P7 체계에 **P8: Loss/Gate Dynamics** (F1~F6) 와 **P9: Causal Expert Axis** (F8~F13) 를 추가. Feature/Expert ablation (기존 P0~P7) 과 직교.

### 2.3 기존 Ablation 설계서 개정

`Ablation_테스트_설계서.typ` (2026-04-13 작성, 907 라인):

| 섹션 | 현재 상태 | 필요 작업 |
|---|---|---|
| "AWS 합성 데이터 핵심 발견" (L322) | **5개 finding** 만 열거 | 13개 로 확장 or 3-테마 재구성 |
| "온프렘 검증 목표" (L332) | 5개 finding 기반 | 13개 반영 |
| "온프렘 구조 ablation 최소 세트" (L341) | 8 시나리오 (구조+기본) | F8~F13 반영한 확장 시나리오 |
| "시나리오 8: Causal Confidence Gate" (L358) | **AWS CG v2 와 다른 설계** | 둘 중 선택: (a) CG v2 로 교체 or (b) Confidence Gate + CG v2 병렬 비교 |

**중요**: 시나리오 8 의 "Causal Confidence Gate" 는 AWS CG v2 (z-space Mahalanobis) 와 다른 접근. AWS 실험에서 CG v2 가 더 강하게 작동함이 확인됐으므로, 온프렘에서는 **CG v2 를 기본 채택** 하고 Confidence Gate 는 보조 비교로만 유지 권장.

---

## 3. 설계서 / 문서 업데이트

### 3.1 `Ablation_테스트_설계서.typ`

- AWS Findings 섹션: 5 → 13 개로 확장. 3-테마 (loss/fusion/causal) 구성.
- 검증 목표 재정의: "mechanistic replication" + "real-only signal 탐색" 강조.
- 시나리오 8 (Causal Confidence Gate) 처리 결정: CG v2 로 교체 vs 병렬.
- 신규 시나리오 P8, P9 추가 (Loss/Gate Dynamics + Causal Axis).
- "Primary AUC 에서 ±0.5% 내외가 대부분이다" 경고 추가.

### 3.2 온프렘 `CLAUDE.md`

현재 매우 짧음 (Airflow DAG 규칙 정도). AWS 의 CLAUDE.md (정책 + 인사이트) 수준으로 확장 권장. 특히:
- Pearl 사다리 프레임워크 (Rung 1/2/3) 기록
- W 학습 상태 모니터링 의무 (||W||_F, active edges)
- Audit log 기본 패턴

### 3.3 백서 / 운영 가이드

| 문서 | 업데이트 필요 |
|---|---|
| `AIOps 추천 시스템 구축 백서 v3.typ` | CEH/CG 반영 여부 확인 |
| `AIOps 추천 시스템 운영 가이드.typ` | audit log 신규 entry 타입 반영 |
| `STUDY_GUIDE.typ` | Pearl 사다리 / Axis 3 설명 추가 |

---

## 4. 실행 순서 (제안)

### Phase 1: 코드 이식 (1~2주)

1. **Causal Expert 확장** (1~2일)
   - `w_init_scale` config 분리
   - `attribution_head` + CEH 캐시 이식
   - `target_mode` 분기 로직 이식
   - 공개 accessor 5종 (`get_last_attribution`, `get_causal_latent`, `get_causal_coherence_score`, `get_counterfactual`, `set_attr_target_external`)
2. **PLE Model 확장** (1일)
   - CEH attribution loss 통합
   - DAG regularization loss 학습 루프 연결 확인
   - `get_ceh_attribution`, `get_causal_coherence` 헬퍼
   - `_inject_ceh_v3_target` 은 v3 negative 고려 시 생략 가능
3. **Config 확장** (0.5일)
   - `model_config.yaml` ceh 서브블록
   - config builder HP flags (use_ceh, ceh_target_mode, w_init_scale, recon_lambda)
4. **Monitoring 확장** (1일)
   - `AuditLogger.log_attribution`
   - `AuditLogger.log_guardrail`
   - `CausalGuardrail` class 이식
5. **Unit Tests 이식** (1일)
   - `test_ceh_audit.py`, `test_causal_guardrail.py`, `test_cg_serving.py`
   - `test_causal_counterfactual.py`
   - 온프렘 테스트 체계에 맞추어 변환

### Phase 2: Ablation 프레임워크 확장 (0.5~1주)

1. **신규 시나리오 정의** (1일)
   - P8 (Loss/Gate Dynamics): EXP-UW1~3, EXP-GATE1, EXP-EP1, EXP-GTE1
   - P9 (Causal Axis): EXP-DAG1, EXP-CEH1~3, EXP-WAMP1, (optional EXP-CEH4)
2. **config_mutator 확장** (1일)
   - 새 HP 경로 지원 (causal.ceh.*, causal.w_init_scale)
3. **Post-hoc 분석 스크립트 이식** (1일)
   - `eval_ceh_attribution.py`, `eval_causal_guardrail.py`, `eval_causal_counterfactual.py`
   - 온프렘 데이터 schema 에 맞게 조정

### Phase 3: 설계서 / 문서 업데이트 (1~2일)

1. `Ablation_테스트_설계서.typ` 개정 (13 findings 반영)
2. 백서 / 운영 가이드 sync
3. 온프렘 `CLAUDE.md` 확장

### Phase 4: 파이프라인 smoke test + 실데이터 검증 (이 문서의 범위 밖)

→ `docs/onprem_v2_validation_checklist.md` 참조.

---

## 5. 우선순위와 스코프 관리

### 5.1 필수 (Must)

- Causal Expert 확장 + config + audit log 이식
- F8 (W=0 patch) + F9 (CEH v2) + F10~12 (CG, W-amp, CCP) ablation 시나리오
- Ablation 설계서 Findings 섹션 업데이트 (5→13)

### 5.2 권장 (Should)

- F1~F7 시나리오 이식 (실데이터 mechanistic replication 용)
- 백서 / 운영 가이드 sync
- Post-hoc 분석 스크립트 이식

### 5.3 선택 (Optional)

- F13 (CEH v3) 이식 — negative 결과라서 생략 가능
- "Causal Confidence Gate" (기존 시나리오 8) 과 CG v2 의 병렬 비교
- Cross-seed 반복 (industrial track 이라 필수 아님)

---

## 6. AWS 측 의존성 (필요 시 수정)

### 6.1 온프렘-고유 설계가 AWS 로 역수입될 가능성

| 온프렘 전용 | AWS 에 역수입할지 |
|---|---|
| Airflow DAG 오케스트레이션 | ❌ (AWS 는 SageMaker 순차 실행, 의도적 분리) |
| DuckDB 파일 기반 저장소 | ❌ (AWS 는 S3 Parquet) |
| Docker GPU 컨테이너 운영 | ❌ (AWS 는 SageMaker managed) |
| `Causal Confidence Gate` (시나리오 8) | ⚠️ AWS CG v2 로 대체됨. 온프렘 실데이터 결과에 따라 역수입 결정 |

### 6.2 AWS → 온프렘 이식 시 주의사항

- AWS `feature_schema.json` 의 group_ranges 매핑은 온프렘 `feature_groups.yaml` 기반 group 매핑과 **scheme 가 다를 수 있음**. FeatureRouter 의 `_idx_causal` 생성 경로 재확인 필요.
- AWS `configs/pipeline.yaml` ↔ 온프렘 `configs/model_config.yaml` — 필드명 불일치 가능. config-driven 원칙은 양쪽 모두 존재.
- AWS 의 CEH 는 `ceh_input_dim=102` (causal 의 routed slice). 온프렘은 `input_dim=771` (v3.16) 인데, FeatureRouter 로 routed slice 가 다른 크기일 수 있음. 이식 시 head output dim 자동 결정 로직 확인.

---

## 7. 검증 로그 (Phase 1 끝나면 채우기)

| 항목 | 상태 | 비고 |
|---|---|---|
| Causal Expert CEH 이식 | ☐ | |
| PLE Model CEH loss 통합 | ☐ | |
| Config ceh 서브블록 | ☐ | |
| AuditLogger log_attribution/log_guardrail | ☐ | |
| CausalGuardrail class | ☐ | |
| Unit tests 통과 | ☐ | |
| Ablation 신규 시나리오 등록 | ☐ | |
| Ablation 설계서 개정 | ☐ | |
| 온프렘 smoke test (1 epoch) 통과 | ☐ | |
| Finding 8 재현 (W=0 → patched) | ☐ | |
| Finding 9 재현 (CEH v2 variance ratio) | ☐ | |
| Finding 11 재현 (W-amp 10×+) | ☐ | |
| Finding 12 재현 (CCP mediation ratio) | ☐ | |

---

## 8. 리스크 및 불확실성

1. **온프렘 PLE 모델 구조 차이**: AWS 는 `extraction_layers.0.shared_experts` 경로, 온프렘은 `ple_cluster_adatt.py` 에서 별도 구조. CEH/CG accessor 가 올바른 expert 를 찾는지 재확인 필요.
2. **피처 차원 불일치**: AWS causal input_dim=102, 온프렘 771. attribution_head 의 output_dim 을 자동 결정해야 함. 이미 AWS 코드가 `input_dim` 파라미터로 처리 — 그대로 이식 시 OK.
3. **실데이터 prior 분포 차이**: CG μ/σ 캘리브레이션을 온프렘 배치에서 다시 해야 함. threshold 도 재산출.
4. **DAG regularization 학습 루프 연결**: 온프렘 `ple_model.py` 에 get_dag_regularization 호출이 없을 가능성. 없으면 학습에서 recon_lambda 가 작동 안 해 W=0 붕괴 재현됨. 반드시 확인.

---

## 9. 완료 기준

- Phase 1 + 2 완료 + 온프렘 smoke test 통과 → **논문 v2 준비 진입 상태**
- Phase 3 완료 + 실데이터 1~2 개 Finding 재현 → **논문 v2 초안 작성 가능**
- `docs/onprem_v2_validation_checklist.md` Tier 3 진입 → **v2 submission 가시화**
