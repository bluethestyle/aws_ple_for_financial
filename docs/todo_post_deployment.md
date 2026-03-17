# Post-Deployment TODO: 실험 후 확인/조치 항목

SageMaker에서 첫 파이프라인 실행 후 확인해야 할 항목들.
코드는 준비되어 있으나 **실제 데이터/학습 결과를 보고 판단**해야 하는 것들.

---

## 1. TemporalEnsemble Gate Entropy 모니터링

**현재 상태**: `aux_losses["temporal_gate_entropy"]`로 매 step 로깅 중 (regularization 없음)

**확인 사항**:
- gate entropy가 0에 가까워지면 gate collapse 발생 (3-branch 중 1개만 선택)
- 정상 범위: entropy ≈ 1.0~1.6 (3-branch uniform이면 log(3) ≈ 1.1)

**조치 기준**:
- entropy < 0.3 이 지속되면 → `cgc.entropy_lambda` 패턴으로 entropy regularization 추가
- 특정 branch가 항상 0 weight면 → 해당 branch의 LR을 높이거나 별도 warmup

---

## 2. Expert별 Learning Rate 튜닝

**현재 상태**: 모든 expert가 동일 LR (1e-3). `expert_lr_overrides` 인프라는 trainer에 구현 완료.

**실험 후 확인**:
- CausalExpert의 DAG adjacency matrix W가 수렴하는지 (get_causal_graph() 모니터링)
- OptimalTransport의 ground cost M이 안정적인지
- Temporal의 3-branch가 균형 있게 학습되는지

**조치 예시** (training config YAML에 추가):
```yaml
optimizer:
  expert_lr_overrides:
    causal: {lr: 0.0003, weight_decay: 0.001}
    optimal_transport: {lr: 0.0005}
    temporal_ensemble: {lr: 0.001}
```

---

## 3. CausalExpert DAG 수렴 확인

**현재 상태**: `get_dag_regularization()`이 total_loss에 추가됨, `aux_losses["dag_regularization"]`으로 로깅.

**확인 사항**:
- dag_regularization 값이 epoch 진행에 따라 감소하는지
- `expert.get_causal_graph()` 결과가 sparse한 DAG를 형성하는지
- acyclicity term (tr(e^{W∘W}) - d)이 0에 수렴하는지

**조치 기준**:
- DAG reg가 감소하지 않으면 → `dag_lambda` 값 증가 (기본 0.01 → 0.1)
- 너무 빠르게 sparse해지면 → `sparsity_lambda` 감소

---

## 4. brand_prediction Contrastive Tower 검증

**현재 상태**: `ContrastiveTower` + InfoNCE loss 설정. loss 함수 구현은 별도 확인 필요.

**확인 사항**:
- InfoNCE loss가 `_compute_task_losses`에서 올바르게 처리되는지
- L2-normalized embedding이 의미 있는 클러스터를 형성하는지
- 128클래스 recall@k 평가

**조치 기준**:
- InfoNCE loss 미구현 시 → `core/training/losses.py`에 추가 필요
- softmax 128 대비 성능이 낮으면 → standard tower로 롤백

---

## 5. GroupEncoder 그룹 간 성능 차이 확인

**현재 상태**: 4개 GroupEncoder (engagement, lifecycle, value, consumption)가 독립 학습.

**확인 사항**:
- 그룹별 validation metric 분포 (특정 그룹만 성능 저하?)
- GroupEncoder 파라미터 norm 비교 (특정 그룹이 under/overfitting?)

**조치 기준**:
- 특정 그룹 성능 저조 → 해당 GroupEncoder의 hidden_dim 조정 또는 LR 별도 설정
- 모든 그룹 유사 → GroupEncoder를 하나로 합치는 것도 고려 (파라미터 절감)

---

## 6. 2-Phase Training 전환점 최적화

**현재 상태**: Phase1=30, Phase2=20 (고정)

**확인 사항**:
- Phase1에서 val_loss가 언제 plateau 되는지
- Phase2 시작 시 loss spike가 발생하는지
- Phase2에서 task-specific metric (AUC, F1)이 실제로 향상되는지

**조치 기준**:
- Phase1이 15 epoch에 수렴 → epochs 줄이고 Phase2에 배분
- Phase2에서 성능 악화 → freeze 범위 조정 (CGC만 unfreeze 등)

---

## 7. ClusterEmbedding 효과 검증

**현재 상태**: n_clusters=0 (비활성). 활성화 시 GMM 20 clusters → 32D embedding.

**확인 사항**:
- ClusterEmbedding 없이 먼저 baseline 확보
- 이후 n_clusters=20으로 활성화하고 성능 비교

**조치 기준**:
- 성능 향상 없음 → 비활성 유지 (파라미터 절감)
- 특정 태스크(life_stage, nba 등)에서만 효과 → 태스크별 선택적 적용 고려

---

## 8. Poincaré Projection 후 HGCN 입력 검증

**현재 상태**: graph.py에서 Poincaré projection 후 HGCN에 전달. HGCN은 유클리드 MLP로 정제.

**확인 사항**:
- Poincaré ball projection된 벡터의 norm 분포 (전부 < 1인지)
- HGCN output이 downstream 태스크에 기여하는지 (CGC attention weight 확인)

**조치 기준**:
- HGCN attention weight가 항상 0에 가까움 → expert 제거 또는 input 재설계
- norm > 0.99인 포인트가 많음 → curvature 조정
