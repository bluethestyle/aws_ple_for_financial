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

## Training 진입점

`containers/training/train.py`의 `main()`이 SageMaker Training Job의 단일 진입점이다.

### 데이터 로딩: PyArrow (no pandas in hot path)

```python
def load_ready_data(channel_dir: str) -> dict:
    """Load training-ready artifacts produced by Phase 0.
    Returns PyArrow Tables (zero-copy from parquet). No pandas in the hot path.
    """
    # DuckDB DESCRIBE로 scalar column 탐지 (list/struct 자동 제외)
    # PyArrow native pq.read_table (no pandas intermediary)
    # NaN diagnostics, dtype summary 자동 로깅
```

반환 구조:
| Key | Type | 설명 |
|-----|------|------|
| `features` | `pyarrow.Table` | normalized numeric features |
| `labels` | `pyarrow.Table` | derived labels |
| `sequences` | `np.ndarray` | padded 3D tensor |
| `feature_schema` | `dict` | column names, group_ranges, expert_routing |
| `label_schema` | `dict` | task definitions |
| `split_indices` | `dict` | train/val/test row indices |

### Split Strategy: Cross-sectional Auto-detect

```python
# If >80% of rows share the same date → cross-sectional → random split
# Otherwise → temporal split (DuckDB SQL, gap_days from config)
```

- **Cross-sectional** (Santander): 단일 snapshot_date → seeded random split
- **Multi-date**: DuckDB SQL로 temporal split 수행 (gap_days 적용)

### Label Deduplication

Label columns이 features Arrow Table에도 포함된 경우, features에서 label columns을 **DROP before merge**하여 중복 방지:

```python
if labels is None:
    label_cols_present = [t["label_col"] for t in tasks if t["label_col"] in features.column_names]
    labels = features.select(label_cols_present)
    features = features.select([c for c in features.column_names if c not in label_cols_present])
```

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
    4. Mixed Precision (fp16 AMP) + FP32 loss computation (overflow 방지)
    5. Gradient clipping + accumulation
    6. Expert별 learning rate override
    7. S3 체크포인트 (Spot 중단 자동 대비)
    8. SageMaker Experiments 메트릭 로깅
    9. Callback system (EarlyStopping, LRScheduler, Checkpoint, MetricLogger)
    10. Per-task metric computation (AUC, MAE, F1, Recall@K)
    11. VRAM diagnostics per epoch (alloc/reserved/peak MB)
    12. Per-task validation masks (temporal_latest / random)
    """
```

### VRAM Diagnostics per Epoch

매 epoch 종료 시 GPU 메모리 사용량을 로깅한다:

```python
# core/training/trainer.py
_alloc = torch.cuda.memory_allocated() / 1e6
_reserved = torch.cuda.memory_reserved() / 1e6
_peak = torch.cuda.max_memory_allocated() / 1e6
# epoch_record에 gpu_memory_allocated_mb, gpu_memory_reserved_mb, gpu_memory_peak_mb 기록
```

### LeakageValidator + Auto-drop

`train.py`에서 학습 전 LeakageValidator를 호출하고, >0.95 상관 피처를 자동 제거한다:

```python
validator = LeakageValidator(correlation_threshold=0.95)
result = validator.validate(features_pd, labels_pd, config)
if not result.passed:
    # regex로 leaking feature name 추출 → Arrow Table에서 제거
    features = features.select([c for c in features.column_names if c not in drop_cols])
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

`scripts/run_santander_ablation.py`가 6-Phase, 48 시나리오 ablation을 오케스트레이션한다. 모든 시나리오는 `pipeline.yaml` + `feature_groups.yaml`에서 동적 생성된다 (하드코딩 없음).

```
┌──────────────────────────────────────────────────────────────────┐
│                    4-Dimension Ablation Framework                │
│                    (학계 표준 Bottom-up + Top-down)              │
│                                                                  │
│  Dim 1: Feature Group Ablation (16 scenarios)                   │
│    Bottom-up: base_only, base+X (각 advanced 그룹 하나씩 추가)   │
│    Top-down: full-X (각 그룹 하나씩 제거 → irreplaceability)     │
│    해석: bottom-up 기여 vs top-down irreplaceability 비교        │
│                                                                  │
│  Dim 2: Expert Ablation (16 scenarios)                          │
│    DeepFM baseline + bottom-up (deepfm+X pairwise 추가)         │
│    Top-down: full-X (각 expert 하나씩 제거)                      │
│    mlp_only (minimal baseline)                                   │
│                                                                  │
│  Dim 3: Task x Structure Cross (16 scenarios)                   │
│    태스크 수 스케일링 (4 → 8 → 15 → 18)                         │
│    구조 변���: shared_bottom / ple_only / adatt_only / full       │
│                                                                  │
│  Dim 4: Structure Ablation (Phase 3에 포함)                     │
│    Loss weighting: Uncertainty vs GradNorm vs DWA vs Fixed      │
│    PLE stacking depth: 1 → 2 → 3 layers                        │
│    adaTT intra/inter strength 변형                               │
└──────────────────────────────────────────────────────────────────┘
```

### 6-Phase 실행 계획

| Phase | 내용 | Scenario 수 | Instance |
|-------|------|------------|----------|
| **0** | Data Preparation (Processing Job) | 1 | CPU (ml.m5.xlarge) |
| **1** | Feature Group Ablation (bottom-up + top-down) | 16 | GPU (g4dn.xlarge) |
| **2** | Expert Ablation (bottom-up + top-down) | 16 | GPU |
| **3** | Task x Structure Cross (4 tiers x 4 variants) | 16 | GPU |
| **4** | Best-Config Teacher + Distillation | 2 | GPU + CPU |
| **5** | Analysis + HTML Report | 1 | CPU |
| **합계** | | **48+** | |

### Dim 1: Feature Ablation (동적 생성)

Config에서 `ablation.feature_scenarios: auto`로 설정하면 `feature_groups.yaml`에서 자동 생성:

```python
# scripts/run_santander_ablation.py
# base_groups: [demographics, product_holdings]  (항상 포함)
# advanced_groups: 나머지 모든 enabled feature groups

scenarios = [
    {"name": "full", "remove": []},
    {"name": "base_only", "remove": advanced_groups},
    # Bottom-up: base + 하나씩
    {"name": "base+txn_behavior", "remove": [all advanced except txn_behavior]},
    {"name": "base+tda_global", "remove": [all advanced except tda_global]},
    # ...
    # Top-down: full - 하나씩
    {"name": "full-txn_behavior", "remove": ["txn_behavior"]},
    {"name": "full-tda_global", "remove": ["tda_global"]},
    # ...
]
```

**해석 프레임워크:**
- Bottom-up 기여 = `base+X` 성능 - `base_only` 성능 (독립 기여)
- Top-down irreplaceability = `full` 성능 - `full-X` 성능 (대체 불가 정보)

### Dim 2: Expert Ablation (동적 생성)

Config에서 `ablation.expert_scenarios: auto`로 설정하면 `model.expert_basket.shared`에서 자동 생성:

```python
# base_expert: deepfm (ablation.base_expert)
# minimal_expert: mlp (ablation.minimal_expert)

scenarios = [
    {"name": "deepfm_only", "experts": ["deepfm"]},
    {"name": "deepfm+temporal", "experts": ["deepfm", "temporal_ensemble"]},
    # ... (각 expert pairwise 추가)
    {"name": "full-deepfm", "experts": [7 experts minus deepfm]},
    # ... (각 expert 제거)
    {"name": "mlp_only", "experts": ["mlp"]},
]
```

Feature-expert 연동: 피처 그룹이 제거될 때 해당 expert만 받는 expert도 자동 비활성화.

> **FeatureRouter 활성화 이후 해석 유의사항**: Expert 제거는 해당 Expert에 라우팅되던 **피처 경로도 함께 제거**한다. 즉, Expert Ablation 결과는 "Expert 구조의 기여"와 "해당 피처 그룹의 기여"를 동시에 측정한다. Dim 1 Feature Ablation 결과와 반드시 교차 비교하여 두 효과를 분리 해석해야 한다.

### Dim 3: Task x Structure Cross

4개 태스크 티어 x 4개 구조 변형 = 16 시나리오 (full 중복 제거 시 15):

```yaml
task_tiers:            # ablation.task_tiers
  tasks_4: [has_nba, churn_signal, product_stability, nba_primary]
  tasks_8: [+spend_level, cross_sell_count, engagement_score, next_mcc]
  tasks_15: [+Tier 3 Product Group + Segmentation]
  tasks_18: all

structure_variants:    # ablation.structure_variants
  shared_bottom: {use_ple: false, use_adatt: false}
  ple_only: {use_ple: true, use_adatt: false}
  adatt_only: {use_ple: false, use_adatt: true}
  full: {use_ple: true, use_adatt: true}
```

### Docker-based Ablation Runner

`containers/training/Dockerfile`로 로컬 GPU PC에서 SageMaker 환경을 동일하게 재현할 수 있다. 로컬에서 end-to-end 검증 후 SageMaker에 제출하는 워크플로우를 따른다.

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
| 데이터 로딩 | TensorDataset + PLEDataset 이중 | **PyArrow zero-copy** + cross-sectional auto-detect split | pandas 없음, 자동 split 감지 |
| 학습 루프 | 인라인 루프 + PLETrainer 이중 | **PLETrainer 단일** (AMP/callbacks/체크포인트) | 일관성 |
| 태스크 수 | 16개 | **18개** (Tier 5 txn-based NBA 추가) | 거래 시퀀스 활용 |
| Loss 함수 | 코드 내 하드코딩 | **build_loss() + focal_alpha calibrated** | positive rate 반영 |
| Loss 가중치 | 불확실성 (미활성화) | **Uncertainty weighting 활성화** | 자동 밸런싱 |
| 모델 구조 | PLE + adaTT | **+ 7 heterogeneous experts + Evidential + SAE + AMP FP32 loss** | 불확실성 + 해석 가능성 |
| Expert 입력 차원 | 전체 피처 브로드캐스트 | **FeatureRouter 활성화 — Expert별 이종 입력 차원** (deepfm=109D, temporal=129D, hgcn=34D, perslay=32D, causal=103D, lightgcn=66D, ot=69D; 파라미터 4.77M→~2.8M 감소) | 불필요한 피처 제거로 Expert 전문성 강화 |
| Logit Transfer | 단일 방법 | **3-method dispatch (5 edges)** | 관계 유형별 최적화 |
| 해석 가능성 | 없음 | **3-stage (A:분석, B:사유, C:서빙)** | 감사 가능한 추천 |
| 증류 | distillation.py 단일 | **config 기반 + fidelity gate** | 품질 보증 |
| Ablation | 없음 | **4-Dimension × 48 scenarios** (bottom-up + top-down) | 체계적 실험, Docker local mode |
| 파이프라인 추적 | 없음 | **pipeline_state.json + resume** | 재현성, 장애 복구 |
| 재현성 | 느슨 (코드+데이터 별도) | YAML config + S3 + temporal split | 완전 재현 |
