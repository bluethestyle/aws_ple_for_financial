# 04. Training Pipeline — PLETrainer, 4-Dimension Ablation, Distillation, Interpretability

## 10+ Stage 파이프라인에서의 위치

Training Pipeline은 Stage 8 ~ Stage 10을 담당한다:

```
Stage 8:   PLETrainer (2-phase training)
Stage 8.5: Model Analysis (IG, CCA, Gate, HGCN, Multi, Template, XAI, Model Card)
Stage 9:   StudentTrainer (PLE → LGBM distillation)
Stage 9.5: Context Vector Store (RAG)
Stage 10:  CPE + Agentic Reason Orchestrator
```

---

## PipelineRunner 통합 진입점

`containers/training/train.py`에 두 가지 진입 경로가 존재한다:

### 1. Legacy 경로 — `main()`
기존 SageMaker Training Job에서 직접 호출하는 경로. Parquet 로드 → PLETrainer 실행.

### 2. Pipeline 경로 — `main_pipeline(config_path)`
`--pipeline config.yaml` 플래그로 활성화. PipelineRunner가 Stage 1~10 전체를 오케스트레이션.

```bash
# Legacy (기존 호환)
python train.py

# Pipeline (PipelineRunner 경유)
python train.py --pipeline configs/santander/pipeline.yaml
```

`main_pipeline()`은 다음을 수행한다:
1. `load_config(config_path)` — YAML 로드
2. SageMaker HP 오버라이드 적용 (`SM_HPS` 환경변수)
3. `PipelineRunner(config).run(output_dir=SM_MODEL_DIR)` — 전체 파이프라인 실행

---

## 학습 파이프라인 흐름

```
configs/santander/pipeline.yaml
         ↓
Step Functions (오케스트레이션)
         ↓
   ┌──────────────────────────────────────────────────────────────┐
   │                     PLETrainer 2-Phase                       │
   │                                                              │
   │  ┌─────────────┐                                            │
   │  │ Phase 1     │ SageMaker Training Job #1                  │
   │  │ Joint       │ - Spot Instance (g4dn.xlarge)              │
   │  │ Training    │ - 체크포인트 → S3 (Spot 중단 대비)            │
   │  │ (30 epoch)  │ - Per-task loss: build_loss() 팩토리         │
   │  │             │ - Uncertainty weighting: Kendall et al.     │
   │  │             │ - Evidential + SAE (config-gated)           │
   │  └──────┬──────┘                                            │
   │         │ S3에 Phase1 체크포인트 저장                          │
   │         ▼                                                    │
   │  ┌─────────────┐                                            │
   │  │ Phase 2     │ SageMaker Training Job #2                  │
   │  │ Tower       │ - Phase1 체크포인트 로드                     │
   │  │ Fine-tune   │ - Extraction layers freeze                  │
   │  │ (20 epoch)  │ - Task Tower + adaTT 미세 조정               │
   │  └──────┬──────┘                                            │
   │         │                                                    │
   │         ▼                                                    │
   │  ┌─────────────┐                                            │
   │  │ 평가         │ SageMaker Processing Job                   │
   │  │ + 등록       │ - 테스트셋 평가 (temporal split)             │
   │  │              │ - Champion/Challenger 비교                  │
   │  │              │ - 합격 시 Model Registry 등록                │
   │  └─────────────┘                                            │
   └──────────────────────────────────────────────────────────────┘
```

---

## PLETrainer — 단일 학습 경로

### 제거된 레거시 코드 경로

| 제거 항목 | 이유 |
|----------|------|
| `TensorDataset` 경로 | → `PLEDataset` 단일 경로만 유지 |
| `train.py` 내 인라인 학습 루프 | → `PLETrainer` 단일 경로만 유지 |
| `_source_pkg/` 정적 복사 | → 동적 패키징만 사용 |

### PLETrainer 핵심 기능

```python
# core/training/trainer.py
class PLETrainer:
    """
    PLE 2-Phase Training, AMP, gradient accumulation, callbacks.

    핵심 기능:
    1. Phase 1/2 자동 전환 (freeze/unfreeze 관리)
    2. Per-task loss dispatch (build_loss 팩토리)
    3. Uncertainty weighting (Kendall et al.) — learnable log_var
    4. Mixed Precision (fp16 AMP) + Dynamic loss scaling
    5. Gradient clipping + accumulation
    6. Expert별 learning rate override
    7. S3 체크포인트 (Spot 중단 자동 대비)
    8. SageMaker Experiments 메트릭 로깅
    9. Callback system (EarlyStopping, LRScheduler, Checkpoint, MetricLogger)
    10. Per-task metric computation (AUC, MAE, F1, Recall@K)
    """
```

### Per-Task Loss 계산 흐름

```
Task Towers (18개 출력)
    ↓
┌──────────────────────────────────────────────────┐
│ Per-task loss 계산                                 │
│                                                    │
│ for task in tasks:                                │
│   loss_fn = build_loss(task.loss, **task.params)  │
│   raw_loss = loss_fn(pred, label)                 │
│                                                    │
│ task_losses = {                                    │
│   "has_nba": 0.234,          # focal(alpha=0.90)  │
│   "churn_signal": 0.567,     # focal(alpha=0.85)  │
│   "product_stability": 1.23, # huber              │
│   "nba_primary": 0.890,      # ce(auto weights)   │
│   ...                         # 18 tasks total     │
│ }                                                  │
├──────────────────────────────────────────────────┤
│ Auxiliary losses                                    │
│ + Evidential loss (regression tasks)               │
│ + SAE reconstruction loss (detached)               │
│ + CGC entropy regularization                       │
├──────────────────────────────────────────────────┤
│ Uncertainty weighting (Kendall et al.)             │
│                                                    │
│ loss_total = Sigma [exp(-log_var_k) * L_k          │
│              + log_var_k / 2]                      │
│              + sae_weight * sae_loss               │
│              + evidential_loss                      │
│              + cgc_entropy_reg                      │
├──────────────────────────────────────────────────┤
│ Optimizer step (AMP + gradient accumulation)       │
│ log_var도 함께 업데이트 (learnable parameter)        │
└──────────────────────────────────────────────────┘
```

---

## Training Config (Santander)

```yaml
training:
  experiment_name: santander_ple
  batch_size: 2048             # 941K users / 2048 = ~460 steps/epoch
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.01
  seed: 42

  phase1:
    epochs: 30
    description: "Joint training of all tasks"

  phase2:
    epochs: 20
    freeze_extraction: true
    description: "Fine-tune towers with frozen extraction"

  early_stopping:
    patience: 7
    metric: val_loss

  scheduler:
    type: cosine
    warmup_epochs: 3

  loss_weighting:
    strategy: uncertainty       # Kendall et al.
```

---

## Interpretability Pipeline (Stage 8.5)

### Stage A: 모델 분석

| 분석 | 입력 | 출력 | 목적 |
|------|------|------|------|
| **Integrated Gradients** | 모델 + 피처 | 피처별 attribution scores | 어떤 피처가 예측에 기여하는가 |
| **Expert Redundancy CCA** | Expert 출력 벡터 | CCA correlation matrix | Expert 간 중복성 감지 |
| **CGC Gate Analysis** | CGC attention weights | 태스크×Expert 히트맵 | 태스크별 Expert 의존도 |
| **HGCN Interpretable** | 쌍곡 임베딩 | 계층 경로 | 계층 구조 기반 설명 |

### Stage B: 추론 사유 생성

| 분석 | 입력 | 출력 | 목적 |
|------|------|------|------|
| **Multi Interpreter** | IG + Gate + HGCN + domain features | 구조화된 추천 사유 | 다학제 통합 해석 |
| **Template Reason Engine** | Multi Interpreter 출력 | 자연어 추천 사유 | 고객 대면 설명 |
| **XAI Quality Evaluator** | 생성된 설명 | 품질 점수 | 설명 충실도/일관성 |
| **Model Card** | 전체 분석 | model_card.json | 감사용 모델 문서 |

### Stage C: 서빙 파이프라인 (Stage 9.5-10)

| 컴포넌트 | 입력 | 출력 | 목적 |
|---------|------|------|------|
| **Context Vector Store** | 추천 사유 + 임베딩 | 벡터 저장소 | RAG 기반 유사 사유 검색 |
| **CPE** | 모델 예측 | FD-TVS composite scores | 개인화 스코어링 |
| **Agentic Orchestrator** | CPE + reasons + constraints | 최종 추천 | L1+L2a+L2b 추론 체인 |

---

## Knowledge Distillation (Stage 9)

```
PLE Teacher (GPU, 18 tasks)
    ↓ Forward pass on full dataset
    ↓ Soft labels (temperature=5.0) + hard labels
    ↓ S3에 저장
    ↓
LGBM Students (CPU, per-task)
    ├── loss = alpha * hard_loss + (1-alpha) * soft_loss
    │   (alpha=0.3: 30% hard + 70% soft)
    ├── num_leaves: 127, n_estimators: 500
    ├── Per-task fidelity validation (AUC gap < threshold)
    └── 경량 모델 저장 (~5ms inference)
```

### Fidelity Gate

`--skip-fidelity-gate` 플래그로 ablation 시 fidelity 검증 비활성화 가능.

---

## 4-Dimension Ablation 설계

### 개요 — Santander 4-Dimension Ablation

`scripts/run_santander_ablation.py`가 6-Phase, 36 SageMaker Job 체계적 ablation을 실행한다.

```
┌──────────────────────────────────────────────────────────────────┐
│                    4-Dimension Ablation Framework                │
│                                                                  │
│  Dim 1: Feature Ablation                                        │
│    5-Axis별 피처 그룹 제거 → 축 기여도 측정                        │
│    full / no_tda / no_temporal / no_graph / no_hmm /            │
│    no_demographics / base_only                                   │
│                                                                  │
│  Dim 2: Expert Ablation                                         │
│    개별/그룹 Expert 제거 (피처-전문가 연동)                        │
│    full_basket / no_deepfm / no_temporal / no_hgcn /            │
│    no_perslay / no_causal / no_lightgcn / no_ot                 │
│                                                                  │
│  Dim 3: Task x Structure Cross                                  │
│    태스크 수 스케일링 (4 → 8 → 18) + PLE/adaTT 변형             │
│    PLE+adaTT / PLE only / adaTT only / baseline                 │
│                                                                  │
│  Dim 4: Structure Ablation                                      │
│    Loss weighting: Uncertainty vs GradNorm vs DWA vs Fixed      │
│    PLE stacking depth: 1 → 2 → 3 layers                        │
│    adaTT intra/inter strength 변형                               │
└──────────────────────────────────────────────────────────────────┘
```

### 6-Phase 실행 계획

| Phase | 내용 | Job 수 | Instance |
|-------|------|--------|----------|
| **0** | Data Preparation (Processing Job) | 1 | CPU |
| **1** | Feature Group Ablation | 10 | GPU (g4dn.xlarge) |
| **2** | Expert Ablation | 7 | GPU |
| **3** | Task x Structure Cross (핵심 실험) | 16 | GPU |
| **4** | Best-Config Teacher + Distillation | 2 | GPU + CPU |
| **5** | Analysis + HTML Report | 1 | CPU |
| **합계** | | **36+** | |

### Dim 1: Feature Ablation

```yaml
ablation:
  feature_groups:
    - name: full
      remove: []
    - name: no_tda
      remove: [tda_global, tda_local]
    - name: no_temporal
      remove: [mamba_temporal]
    - name: no_graph
      remove: [graph_collaborative, product_hierarchy]
    - name: no_hmm
      remove: [hmm_states]
    - name: no_demographics
      remove: [demographics]
    - name: base_only
      remove: [tda_global, tda_local, mamba_temporal, hmm_states,
               graph_collaborative, product_hierarchy, gmm_clustering, model_derived]
```

### Dim 2: Expert Ablation (피처-전문가 연동)

```yaml
ablation:
  experts:
    - name: full_basket
      shared: [deepfm, temporal_ensemble, hgcn, perslay, causal, lightgcn, optimal_transport]
    - name: no_deepfm
      shared: [temporal_ensemble, hgcn, perslay, causal, lightgcn, optimal_transport]
    - name: no_temporal
      shared: [deepfm, hgcn, perslay, causal, lightgcn, optimal_transport]
    # ... 개별 Expert 제거
```

### Dim 3: Task x Structure Cross

```yaml
# 태스크 수 x 구조 변형 = 4 x 4 = 16 실험
task_subsets:
  4_tasks: [has_nba, churn_signal, product_stability, nba_primary]
  8_tasks: [has_nba, churn_signal, product_stability, nba_primary,
            spend_level, cross_sell_count, engagement_score, next_mcc]
  18_tasks: all

structure_variants:
  - name: ple_adatt         # PLE(stacked CGC) + adaTT — full
  - name: ple_only          # PLE만 (adaTT 없음)
  - name: adatt_only        # adaTT만 (단일 MLP shared)
  - name: baseline          # 단순 멀티태스크 MLP
```

### Ablation HP (Hyperparameters)

```python
ablation_hps = {
    "--removed-feature-groups": "쉼표 구분 제거 feature group",
    "--removed-axes": "쉼표 구분 제거 axis",
    "--removed-experts": "쉼표 구분 제거 expert",
    "--num-active-tasks": "활성 태스크 수 (4/8/18)",
    "--disable-adatt": "adaTT 비활성화",
    "--disable-ple-stacking": "PLE stacking 비활성화 (단일 CGC)",
    "--loss-weighting-strategy": "uncertainty/gradnorm/dwa/fixed",
    "--ple-num-layers": "PLE stacking depth (1/2/3)",
    "--skip-fidelity-gate": "Fidelity gate 비활성화 (증류 ablation용)",
}
```

---

## SageMaker Training 래퍼

```python
# aws/sagemaker/trainer.py
class SageMakerTrainer:
    """
    2-Phase Training을 SageMaker Job 2개로 분리 실행.
    PLETrainer만 사용 (레거시 인라인 루프 제거됨).
    데이터 로딩: PLEDataset → DataLoader 단일 경로.
    소스 패키징: 동적 패키징만 (_source_pkg/ 제거됨).
    """
```

---

## Pipeline State Tracking

`core/pipeline/runner.py`의 `_PipelineState`가 Stage별 완료 상태를 추적:

```json
// pipeline_state.json
{
  "completed_stages": ["adapter", "temporal_prep", "schema", "encryption", "features", "labels", "leakage", "sequences", "dataloader", "training"],
  "artifacts": {
    "features": {"path": "features.parquet", "rows": 941132, "dim": 512},
    "labels": {"path": "labels.parquet", "tasks": 18},
    "training": {"best_val_loss": 0.234, "epochs": 50}
  },
  "start_time": "2026-03-20T23:07:04"
}
```

resume 지원: 이미 완료된 stage는 skip하고 이어서 실행.

---

## 실험 관리

| 항목 | MLflow (현재) | SageMaker Experiments (AWS) |
|------|-------------|---------------------------|
| 서버 | Docker 자체 호스팅 (상시 가동) | AWS 관리형 (비용 0) |
| 메트릭 로깅 | `mlflow.log_metric()` | `sagemaker.experiments` |
| 모델 저장 | MLflow Model Registry | S3 + 메타데이터 |
| 하이퍼파라미터 | 수동 로깅 | YAML에서 자동 추출 |
| 실험 비교 | MLflow UI | SageMaker Studio |

---

## Champion/Challenger 평가

```
현재 모델 (Champion) vs. 새 모델 (Challenger)
    ↓
┌─────────────────────────────────────────┐
│ 평가 기준                                │
│                                         │
│ 1. 주요 메트릭 (태스크별)                 │
│    - Binary: AUC-ROC > champion - 0.01  │
│    - Regression: MAE < champion + 5%    │
│    - Multiclass: F1-macro > champion    │
│                                         │
│ 2. 추론 성능                             │
│    - Latency p99 < 100ms               │
│    - Throughput > 1000 req/s            │
│                                         │
│ 3. 안정성                                │
│    - 테스트셋 분포 PSI < 0.1             │
│    - 예측 분포 변화 < 10%                │
│                                         │
│ → 모두 통과: 자동 등록                    │
│ → 하나라도 실패: 알림 → 수동 검토         │
└─────────────────────────────────────────┘
```

---

## Spot 인스턴스 전략

```
비용 비교:
  On-Demand g4dn.xlarge: $0.526/hr
  Spot g4dn.xlarge:      ~$0.16/hr (70% 절감)

Spot 중단 대비:
  1. 매 epoch마다 S3에 체크포인트 저장
  2. SageMaker CheckpointConfig 설정 → 자동 재개
  3. max_wait = max_run + 1hr (Spot 재할당 대기)

Santander 학습 기준 (50 epochs, ~4시간):
  On-Demand: $2.10
  Spot:      ~$0.64
```

---

## 현재 vs AWS — 학습 파이프라인 비교

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| 실행 환경 | 고정 GPU 서버 | SageMaker Spot (필요 시만) | 비용 70% 절감 |
| 2-Phase | trainer.py 내부 로직 | SageMaker Job 2개 분리 | 각 Phase 독립 재실행 가능 |
| 데이터 로딩 | TensorDataset + PLEDataset 이중 | **PLEDataset 단일** + temporal split | 레거시 제거, 누수 방지 |
| 학습 루프 | 인라인 루프 + PLETrainer 이중 | **PLETrainer 단일** (AMP/callbacks/체크포인트) | 일관성 |
| 태스크 수 | 16개 | **18개** (Tier 5 txn-based NBA 추가) | 거래 시퀀스 활용 |
| Loss 함수 | 코드 내 하드코딩 | **build_loss() + focal_alpha calibrated** | positive rate 반영 |
| Loss 가중치 | 불확실성 (미활성화) | **Uncertainty weighting 활성화** | 자동 밸런싱 |
| 모델 구조 | PLE + adaTT | **+ Evidential + SAE + HMM routing + MD routing** | 불확실성 + 해석 가능성 |
| Logit Transfer | 단일 방법 | **3-method dispatch (5 edges)** | 관계 유형별 최적화 |
| 해석 가능성 | 없음 | **3-stage (A:분석, B:사유, C:서빙)** | 감사 가능한 추천 |
| 증류 | distillation.py 단일 | **config 기반 + fidelity gate** | 품질 보증 |
| Ablation | 없음 | **4-Dimension × 36 SageMaker jobs** | 체계적 실험 |
| 파이프라인 추적 | 없음 | **pipeline_state.json + resume** | 재현성, 장애 복구 |
| 재현성 | 느슨 (코드+데이터 별도) | YAML config + S3 + temporal split | 완전 재현 |
