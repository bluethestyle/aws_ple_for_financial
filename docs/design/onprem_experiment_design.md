# 온프렘 실험 설계 — Production Data Validation

## 1. 목적

AWS 합성 데이터 ablation(Santander 941K x 316D)에서 프레임워크 작동을 검증했다.
온프렘 실험의 목적은 **실제 금융 운영 데이터(12M x 700D+)에서 아키텍처 효과를 검증**하는 것이다.

### 합성 데이터에서 확인된 것 (arXiv v1)
- Expert별 차등 기여 존재 (TDA→investments, HGCN→deposits, Causal→churn)
- FeatureRouter 작동 (파라미터 ~34% 절감, 성능 유지)
- Pool→Basket→CGC 3-tier selection 구조 작동
- HGCN/LightGCN 제거 시 AUC 폭락 → 구조적으로 필수

### 합성 데이터에서 미검증 (arXiv v2에서 추가)
- Temporal expert 효과 (합성 시계열에 패턴 없어서 negative transfer)
- Full basket이 단일 expert보다 나은지 (10ep 수렴 부족)
- 절대 성능 수준 (합성 AUC 0.57)
- YAML primary_tasks 매핑과 실제 기여 일치 여부

---

## 2. 데이터 환경

| 항목 | 합성 (AWS) | 실제 (온프렘) |
|------|-----------|-------------|
| 고객 수 | 941K | ~12M |
| 피처 차원 | 316D | 700D+ |
| 거래 데이터 | synth_* (시뮬레이션) | 실제 거래 시계열 |
| 시계열 길이 | 17개월 (고정) | 수 년 |
| 상품 계층 | 24개 상품 (flat) | 다단계 계층 |
| 고객 그래프 | 합성 co-holding | 실제 거래 네트워크 |
| 인프라 | RTX 4070 12GB | 온프렘 GPU 서버 |

---

## 3. 실험 시나리오 (3개, 필수)

시나리오당 2-3일 예상, **총 7-9일**.

### Scenario 1: shared_bottom (baseline)
```yaml
use_ple: false
use_adatt: false
epochs: 30-50  # 데이터 규모에 맞게 조정
amp: false
```
**목적**: PLE/expert 구조 없는 바닥선.
**검증**: 이것보다 full system이 나아야 아키텍처 정당성 확보.

### Scenario 2: full basket (ple_sigmoid_adatt)
```yaml
use_ple: true
use_adatt: true
gate_type: sigmoid
epochs: 30-50
amp: false
```
**목적**: 전체 시스템 (7 experts + PLE + adaTT + FeatureRouter).
**검증**:
- full > shared_bottom → **아키텍처 효과 입증**
- Gate weight 분석 → **expert별 태스크 차등 기여** (추가 학습 없이 추출)
- Per-task AUC/F1/MAE → **태스크별 어떤 expert가 기여하는지**

### Scenario 3: deepfm+temporal
```yaml
shared_experts: ["deepfm", "temporal_ensemble"]
epochs: 30-50
amp: false
```
**목적**: 합성 데이터에서 negative transfer였던 temporal expert가 실제 시계열에서 역전되는지.
**검증**:
- AUC > deepfm_base 수준 → **"합성 데이터 한계 입증, 실제 시계열에서 temporal expert 유효"**
- 여전히 negative → **"temporal expert 설계 재검토 필요"** (이것도 논문에 가치 있는 결과)

---

## 4. Gate Weight 분석 (추가 학습 불필요)

Scenario 2 (full basket) 학습 완료 후, 모델의 CGC gate weight를 추출하면
**expert별 태스크 기여를 시나리오 추가 없이 분석**할 수 있다.

```python
# Validation set에서 gate weight 수집
for batch in val_loader:
    outputs = model(batch)
    gate_weights = model.extraction_layers[0]._last_gate_weights
    # gate_weights[task_idx] = (batch, num_experts) softmax/sigmoid weights
```

### 기대 분석 결과물
1. **Task x Expert 히트맵** — 각 태스크에 어떤 expert가 높은 가중치를 받는지
2. **Expert 전문화 정도** — gate entropy가 낮으면 특정 expert에 집중 (전문화)
3. **YAML primary_tasks 매핑 검증** — 설계 의도와 실제 gate weight 일치 여부

이 분석이 논문의 "structural interpretability" 주장의 핵심 근거.

---

## 5. 평가 지표

### 태스크 유형별 primary 지표

| 유형 | 태스크 예시 | Primary | Secondary |
|------|-----------|---------|-----------|
| Binary | churn, has_nba, will_acquire_* | **AUC** | F1 (optimal threshold) |
| Multiclass | nba_primary, segment, income_tier | **F1 macro** | Accuracy |
| Regression | product_stability, engagement | **MAE** | RMSE |

### 비교 방법
- **Aggregate**: 전체 AUC, F1m avg, MAE avg
- **Per-task**: 태스크별 지표 테이블 (행=시나리오, 열=태스크)
- **Delta table**: shared_bottom 대비 개선량
- **Gate weight heatmap**: full basket에서만 추출

---

## 6. 성공 기준

논문에 쓸 수 있는 최소 조건:

| 주장 | 성공 기준 |
|------|----------|
| 아키텍처 효과 | full basket AUC > shared_bottom (통계적 유의) |
| Expert 차등 기여 | gate weight 히트맵에서 expert별 편중 확인 |
| Temporal expert 기여 | deepfm+temporal AUC > deepfm_base (합성 대비 역전) |
| Graceful degradation | full basket에서 특정 expert gate weight가 0에 수렴하지 않음 |

### 실패해도 논문에 쓸 수 있는 것
- full ≤ shared_bottom → "복잡한 아키텍처가 항상 유리하지 않다" (negative result)
- temporal 여전히 negative → "시계열 expert 설계 재검토 필요"
- gate weight가 uniform → "expert 전문화가 발현되지 않음, 더 강한 inductive bias 필요"

---

## 7. 체크리스트 (실행 전)

- [ ] 온프렘 gotothemoon 레포에서 config 확인 (pipeline.yaml, feature_groups.yaml)
- [ ] AWS 프레임워크의 adapter를 온프렘 데이터 포맷에 맞게 조정
- [ ] feature_group_ranges 오버랩 없는지 검증
- [ ] 1 epoch dry-run으로 메모리/시간 측정
- [ ] checkpoint 경로를 시나리오별로 분리 (`SM_CHECKPOINT_DIR=$RESULTS/$NAME/checkpoints`)
- [ ] AMP=false 확인 (FP32)
- [ ] temporal split + gap_days 설정 확인
- [ ] LeakageValidator 실행

---

## 8. 일정 (예상)

| 주차 | 작업 |
|------|------|
| Day 1 | 환경 세팅, dry-run, 메모리/시간 측정 |
| Day 2-4 | Scenario 1: shared_bottom |
| Day 5-7 | Scenario 2: full basket (ple_sigmoid_adatt) |
| Day 8-9 | Scenario 3: deepfm+temporal |
| Day 10 | Gate weight 분석 + 결과 정리 |
| Day 11 | arXiv v2 업데이트 |

---

## 9. AWS 합성 결과 참조 (비교용)

### Joint Ablation (FP32, 10ep, Santander 941K x 316D)

| Scenario | AUC | 비고 |
|----------|-----|------|
| full-temporal | 0.5753 | temporal 빼면 1위 |
| deepfm+tda | 0.5746 | TDA가 가장 효과적 |
| deepfm+lightgcn | 0.5738 | graph expert 효과 |
| deepfm+hgcn | 0.5697 | hierarchy 효과 |
| deepfm_all | 0.5659 | 피처만 추가 |
| deepfm+causal | 0.5640 | causal 소폭 기여 |
| deepfm_base | 0.5632 | baseline |
| deepfm+temporal | 0.5507 | **negative transfer** |
| full | 0.5470 | 수렴 부족 |

### 합성 데이터 한계
- AUC 0.57 수준 — 합성 피처 신호가 약함
- temporal negative transfer — synth_* 컬럼에 실제 시계열 패턴 없음
- full basket 수렴 부족 — 10ep로는 7 expert 동시 최적화 어려움
- 이상의 한계는 실제 데이터에서 해소될 것으로 기대
