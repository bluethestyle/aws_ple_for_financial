# On-Prem v2 Validation Checklist

**목적**: AWS 프로젝트에서 확정된 아키텍처 + 발견사항 13개를 온프렘 (우체국 등) 실데이터로 재검증. v2 논문 발표 자료.

**현재 상태 (2026-04-20)**: 온프렘 파이프라인 정상작동 점검 단계. 아래 체크리스트는 파이프라인 안정화 후 순차 진행.

**전제 교훈**: AWS 실험 대부분이 primary AUC 에서 ±0.5% 내외 변화였음. 실데이터 검증은 **mechanistic replication** + **real-only signal 탐색** 두 목표. Primary task 에서 극적 차이 기대 X.

---

## Tier 0: Pipeline Smoke Test (선행 조건)

체크리스트 전체가 의미 있으려면 먼저 파이프라인 자체가 실데이터에서 돌아가야 함.

| 검증 항목 | 측정 | Pass 기준 | 비고 |
|---|---|---|---|
| **0.1** Phase 0 실데이터 처리 | feature_stats.json NaN 비율, zero-variance 컬럼 | NaN < 10%, zero-var 컬럼 없음 | adapter 가 실데이터 스키마에 맞는지 |
| **0.2** Feature group 라우팅 | feature_schema.json 의 group_ranges 연속성 | 모든 group 이 연속 블록 | 3-stage 정규화 후 _log 컬럼 재배치 OK |
| **0.3** 1 epoch end-to-end | loss 유한, NaN 없음, 모든 태스크 predict 생성 | Primary AUC > random (>0.5) | 학습 신호 도달 여부 |
| **0.4** Leakage validator | scaler train-only fit, temporal gap, sequence overlap | 3개 체크 모두 통과 | 학습 시작 전 gate |

이 단계에서 실패하면 뒤 scenarios 는 의미 없음. 먼저 해결 후 진행.

---

## Tier 1: 핵심 아키텍처 기본 검증 (Must)

온프렘 v2 의 "정상 동작" 근거. 이 단계가 깨끗해야 v2 논문 submission 가능.

### 1.1 Baseline teacher 학습
| 시나리오 | Scenario name | Config | 측정 | AWS 비교값 |
|---|---|---|---|---|
| Baseline teacher_full | `teacher_full` | default | Primary AUC, F1 macro, NDCG@3, MAE | AUC 0.6870 |
| 10 epoch 수렴 확인 | | | train/val loss curve | 30 epoch 에서 수렴 |
| Gate entropy | analyze_gate_weights.py | | extraction ratio 분포 | 0.33~0.88 |

**Pass**: primary AUC > Random baseline × 1.5, loss 수렴 확인, NaN/Inf 없음.

### 1.2 Distillation smoke test
| 항목 | 측정 | Pass 기준 |
|---|---|---|
| Teacher threshold gating | 각 task 의 DISTILL/DIRECT/SKIP 라우팅 | 모든 13 task 라우팅 결정 기록 |
| LGBM student 학습 | distill 완료 후 per-task AUC | Teacher 대비 gap < 0.05 on DISTILL tasks |
| 3-layer fallback | Layer 3 rule engine 동작 | SKIP 태스크 가 Layer 3 로 routing |

### 1.3 CPU 서빙 경로
| 항목 | 측정 | Pass 기준 |
|---|---|---|
| LGBM predict latency | 단일 요청 end-to-end ms | warm < 150ms (AWS Lambda baseline) |
| 3-agent reason pipeline | Feature Selector → Reason Generator → Safety Gate | 정상 응답, Safety Gate 통과 |
| Audit log emission | log_model_inference HMAC 서명 | Chain verifier 통과 |

---

## Tier 2: Paper 3 핵심 Finding 재현 (Should)

AWS 에서 발견한 패턴이 실데이터에서도 재현되는지. 이게 v2 논문의 "mechanistic replication" 증거.

### 2.1 Findings 1~6 (Loss dynamics / Gating)

| Finding | 시나리오 | 측정 | Pass 기준 |
|---|---|---|---|
| **F1** UW bug | UW on vs off | multiclass task AUC/F1 | UW on 이 +0.01 이상 |
| **F2** Softmax vs sigmoid | CGC gate: softmax / sigmoid | aggregate AUC | softmax 우위 (heterogeneous mix) |
| **F3** UW convergence | shared-bottom vs PLE 의 UW weight | W_uw trajectory 비교 | 두 구조 모두 동일 수렴점 |
| **F4** Epoch budget | 10 epoch vs 30 epoch | primary AUC curve | 10 에선 차이 없음, 30 에선 드러남 |
| **F5** GTE pre-gating | GTE on/off (mixed-type group) | 다중 클래스 task AUC | GTE on 이 성능 저하 |
| **F6** Gate entropy | CGC extraction entropy | per-task ratio | 0.3~0.9 분포, attention 레벨은 ~1.0 |

**Pass**: 6개 중 5개 이상에서 AWS 와 같은 방향성 (수치 크기는 달라도 됨).

### 2.2 Finding 7 (Fusion augmentation)

| 후보 | 시나리오 | 측정 | Pass 기준 |
|---|---|---|---|
| BRP-detached | `ablation_brp_detached` | aggregate AUC + F1 macro + NDCG@3 | F1 macro +0.005 이상 |
| NEAS | `ablation_neas` | aggregate AUC | Δ AUC > +0.0005 |
| NEAS × BRP-detached | 합성 | aggregate AUC | 개별보다 낮거나 유사 (non-additive 재현) |

### 2.3 Findings 8~11 (Causal expert 구조)

| Finding | 시나리오 | 측정 | Pass 기준 |
|---|---|---|---|
| **F8** W=0 collapse patch | Default train vs patched (recon_lambda=0.5, init 0.1) | ||W||_F | default < 0.05, patched > 0.3 |
| **F9** CEH v2 demeaned | `teacher_ceh_demeaned` | variance ratio, top-10 overlap | ratio > 0.5, overlap < 0.4 |
| **F10** CG v1 vs v2 | W-recon vs z-Mahalanobis | OOD TPR @ 5% FPR | v1 chance, v2 > 80% |
| **F11** W-amplification | `teacher_ceh_w_amp` | ||W||_F, edges%, primary AUC 유지 | W 10배 이상, AUC 불변 |

**Pass**: 4개 모두 방향성 일치. 수치는 데이터 특성상 변동 허용.

### 2.4 Finding 12 (CCP Pearl Rung 3)

| 시나리오 | 측정 | Pass 기준 |
|---|---|---|
| Baseline CCP mediation | 5개 intervention × 모든 causal latent 차원 | mediation ratio median |
| W-amp CCP mediation | | median > 20%, p95 > 50% |

---

## Tier 3: 실데이터 전용 신규 검증 (Should, AWS 에선 불가능)

이게 실데이터 검증의 **진짜 가치**. AWS 합성 데이터로는 측정 불가능한 것들.

### 3.1 CEH Attribution 도메인 타당성 (Human Eval)

| 항목 | 측정 | Pass 기준 |
|---|---|---|
| Top-K 피처 도메인 매치 | N=50 샘플, 도메인 전문가 리뷰 | ≥70% 가 "납득 가능" 판정 |
| Attribution 일치도 (Int. Gradients 등) | Spearman(CEH top-K, IG top-K) | mean > 0.4 |
| Confounding 여부 | 전문가가 "이 피처 말 안 됨" 지적 건수 | < 20% |

### 3.2 CG 실제 OOD 탐지

| OOD 유형 | 측정 | Pass 기준 |
|---|---|---|
| Temporal drift | Q1 에서 fit, Q2/Q3 traffic CG score | p95 traffic 이 Q1 p99 이상 |
| Subgroup 불균형 | 저수입/신규/장기고객 각 segment CG score | segment 간 score 분포 유의미 차이 |
| 합성 adversarial | FGSM-style perturbation | TPR @ 5% FPR > 50% |
| Concept drift (label shift) | 규제 변경 후 예측 분포 | CG score 일부 상승 |

### 3.3 Pearl Rung 2 (Treatment Effect) — 실 offer 데이터 있다면

| 시나리오 | 측정 | 요구사항 |
|---|---|---|
| 상품 offer → acquisition 상승 | ATE 추정 (causal expert 경로) | 실제 offer record + outcome |
| 온프렘 vs 모델 ATE 비교 | 추정 ATE 와 domain 추정 | domain analyst 와 consistency |
| Counterfactual offer | "offer X 안 했다면?" — CCP 로 예측 | 예측 vs 실제 non-offer group |

### 3.4 Fairness 실데이터

| Protected attribute | 측정 | Pass 기준 |
|---|---|---|
| 연령 그룹 | Disparate Impact (DI), SPD, EOD | DI ∈ [0.8, 1.25] (4/5 rule) |
| 성별 | DI, SPD, EOD | 동일 |
| 소득 티어 | DI, SPD, EOD | 동일 |
| 지역 | DI, SPD, EOD | 동일 |
| 거래 기간 | DI, SPD, EOD | 동일 |

### 3.5 Distillation 품질 (실 피처)

| 항목 | 측정 | Pass 기준 |
|---|---|---|
| Binary tasks | Teacher vs student AUC gap | ≤ 0.03 |
| Ranking | NDCG@3 gap | ≤ 0.02 |
| Regression | MAE ratio (student/teacher) | ≤ 1.3 |
| Calibration | ECE after Platt scaling | < 0.05 |

### 3.6 서빙 운영 지표

| 항목 | 측정 | Pass 기준 |
|---|---|---|
| Lambda cold start | 최초 요청 latency | < 10s |
| Lambda warm | 95th percentile | < 300ms |
| L2a Bedrock cache hit | cache hit rate | > 60% (정상 운영 후) |
| Audit log latency overhead | request 별 추가 지연 | < 30ms |

---

## Tier 4: Regulatory Story 증거 축적 (Nice-to-have)

금감원/GARP 제출용 증거. 실데이터 위에서 의미 있어지는 것들.

### 4.1 감사로그 End-to-End

| 항목 | 측정 | Pass 기준 |
|---|---|---|
| Per-prediction 감사 pair (CEH + CG) | 24h 샘플 트래픽 | 100% entry 포함, chain verifier 통과 |
| HMAC 검증 | SSM key rotation 시뮬 | key rotation 후에도 past entry 검증 가능 |
| Tamper detection | 의도적 변조 inject | verifier 가 변조 지점 특정 |
| Regulator query scenario | 3종 query (특정 고객 / 기간 / task) | 모두 응답, 재현 가능 |

### 4.2 SR 11-7 MRM 컴플라이언스

| 요구사항 | 구현 | 검증 |
|---|---|---|
| 모델 재검증 (validation) | Champion-Challenger 오프라인 게이트 | 승격 결정 audit log |
| Ongoing monitoring | Drift + Fairness + OpsAgent | 주간 리포트 생성 |
| Model inventory | versioning + registry | 모든 배포 모델 기록 |
| Independent validation | AuditAgent 자동 리포트 | 승격/거부 근거 텍스트 |

### 4.3 EU AI Act + GDPR Art.22

| 조항 | 커버 메커니즘 | 실데이터 검증 |
|---|---|---|
| Art. 9 Risk mgmt | CG + Ongoing monitor | 실제 드리프트 시나리오 대응 |
| Art. 10 Data governance | Phase 0 validator | 실데이터 quality report |
| Art. 13 Transparency | CEH + Reason Gen | Top-K + 자연어 reason 동시 제공 |
| Art. 14 Human oversight | HITL review queue | 운영자 검토 로그 |
| GDPR Art. 22 | CEH attribution + audit log | "왜 이 결정" 규제기관 쿼리 재현 |

---

## Tier 5: Negative Result 재확인 (Optional)

AWS 에서 negative 였던 것들이 실데이터에서도 negative 인지만 확인. 시간/비용 되면.

| Finding | 재확인 필요성 | 방법 |
|---|---|---|
| **F13** CEH v3 primary-task target | 낮음 | 생략 권장. Target 설계 민감성 교훈만 유지 |
| Finding 7 representation-additive fusions (adaTT, M1, ECEB, MV BRP) | 낮음 | 생략 권장. AWS 에서 명확히 음성 |
| GTE pre-gating (F5) | 중간 | 실데이터 task 구성이 다르면 재확인 가치 |

---

## 추천 실행 순서

1. **Week 1**: Tier 0 완료 (smoke test) + Tier 1.1 baseline
2. **Week 2**: Tier 1.2~1.3 + Tier 2.1 (Findings 1~6)
3. **Week 3**: Tier 2.2~2.4 (Findings 7~12)
4. **Week 4**: Tier 3.1~3.2 (human eval + 실 OOD) — **v2 논문의 차별화 지점**
5. **Week 5+**: Tier 3.3~3.6 + Tier 4 (시간 여유 있을 때)
6. **v2 submission**: Tier 3 완료 후 Tier 4 부분 포함해서 발표

## 필요 자원

- **도메인 전문가**: Tier 3.1 (CEH human eval) — 최소 2명, 각 50 샘플 리뷰 (1~2일)
- **Offer/outcome 로그**: Tier 3.3 (Rung 2) — 해당 데이터 없으면 이 항목 생략 가능
- **운영 트래픽**: Tier 3.2 (OOD) + Tier 4.1 (audit log) — 최소 1~2주 축적
- **Fairness-labeled 데이터**: Tier 3.4 — protected attributes 보장된 subset

## 우선순위가 낮아도 기록해야 하는 것

실데이터 검증 중 발견되는 **예상 외 패턴** 은 무조건 기록. v2 논문의 주요 차별화는 "AWS 에서 예측한 것 + 실데이터에서 발견한 것" 의 대비.

## v2 논문 포함 예상 구조

- Paper 1 v2: 아키텍처 + **실데이터에서의 ablation 결과**
- Paper 2 v2: 서빙/규제 + **audit log E2E 실 시나리오** + **human eval 결과**
- Paper 3 v2: Loss dynamics + causal reinterpretation + **실데이터 재현 + Rung 2 결과 (가능 시)**
