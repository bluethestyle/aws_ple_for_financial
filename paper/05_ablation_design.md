# Ablation Study Design

## 1. 목적

"이 컴포넌트를 빼면 성능이 X% 떨어진다"를 정량적으로 보여서:
- 복잡한 아키텍처의 정당성 증명
- 각 expert/feature group의 marginal contribution 정량화
- PLE+adaTT 구조 자체의 효과 분리
- 논문 모형 검증 문서의 핵심 근거

## 2. 23 Scenarios (14 joint + 9 structure)

### Joint Ablation (14 scenarios)

**Feature Group Ablation (bottom-up & top-down):**
- full: 모든 피처 그룹 사용
- base_only: demographics + product_holdings만 (base features)
- base+tda, base+hmm, base+mamba, base+graph
- base+hierarchy, base+gmm, base+model_derived
→ 각 그룹의 독립적 기여도 측정
- full-tda, full-hmm, full-mamba, full-graph, full-model_derived
→ 각 그룹의 marginal contribution 측정 (다른 그룹 존재 하에서)

### Structure Ablation (9 scenarios)

**Expert basket + structural variants:**
- full_basket (all 7 experts): DeepFM, Temporal Ensemble, HGCN, PersLay, Causal, LightGCN, OT
- deepfm_only, deepfm+temporal, deepfm+hgcn, deepfm+perslay
- deepfm+causal, deepfm+lightgcn, deepfm+ot
- mlp_only (minimal baseline)
- shared_bottom (no PLE/adaTT)

> **주의 (FeatureRouter):** FeatureRouter 활성화로 각 expert의 입력 차원이 균일하지 않다. Expert ablation 시나리오의 파라미터 수는 uniform 349D 기준이 아니라 라우팅된 실제 입력 차원 기준으로 산출된다 (예: hgcn은 34D hierarchy 전용, perslay는 32D TDA 전용). Full basket 전체 모델 파라미터는 ~2.8M이며, expert 제거 시 해당 expert의 전용 파라미터가 제거된다.

## 3. 학습 설정

```yaml
training_defaults:
  epochs: 3 (Phase 1+2 각각, 총 6 epochs/scenario)
  batch_size: 6144
  learning_rate: 0.0005
  amp: true
  early_stopping_patience: 3
  num_workers: 2
  pin_memory: true
  drop_last: true
```

- 데이터: 1M customers, 349 features (Phase 0 v12 전체 피처 공간), 13 tasks
- **FeatureRouter 활성화**: 각 expert는 전체 349D가 아닌 지정된 피처 그룹만 수신
  - deepfm=109D, temporal_ensemble=129D, causal=103D, optimal_transport=69D
  - lightgcn=66D, hgcn=34D, perslay=32D
  - 모델 파라미터: 4.77M → ~2.8M (감소)
- GPU: RTX 4070 12GB (로컬) / g5.xlarge A10G 24GB (클라우드)
- 시나리오당 ~30분, 전체 ~24시간 (순차 실행)

## 4. 기대 결과

### AUC 범위
- XGBoost ceiling: churn_signal=0.65 (has_nba는 nba_primary로 통합; nba_primary NDCG@3으로 평가)
- PLE 딥모델 예상: 0.65-0.80 (비선형 상호작용 학습)
- 시나리오 간 델타: 0.02-0.10 (의미있는 차이)

### 핵심 분석 포인트
1. **Feature contribution**: TDA/HMM/Mamba 등 고급 피처가 base 대비 얼마나 기여?
2. **Expert specialization**: 각 expert가 특정 task group에 특화?
3. **Structure effect**: PLE+adaTT가 shared_bottom 대비 얼마나 개선?
4. **Scaling behavior**: 태스크 수 증가 시 구조별 성능 변화 패턴
5. **Graceful degradation**: 컴포넌트 제거 시 급격한 성능 저하 없는지 (금융 안정성)

## 5. 결과 활용

### 논문 Table
- Feature group importance ranking (top-down delta 기준)
- Expert specialization matrix (expert × task group AUC)
- Structure comparison (PLE+adaTT vs alternatives, task 수별)

### 논문 Figure
- Ablation waterfall chart (기여도 누적)
- Task × Structure heatmap
- adaTT affinity matrix visualization

### 증류 연계
- 기여도 높은 expert/feature → 증류 시 우선 보존
- 기여도 낮은 컴포넌트 → 프로덕션 모델에서 pruning 가능
