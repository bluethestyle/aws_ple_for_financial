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

### Config 로딩: split-config 패턴 (2026-04-15)

`train.py`는 두 가지 config 로딩 패턴을 지원한다:

| 패턴 | HP `config` | HP `dataset_config` | 설명 |
|------|------------|---------------------|------|
| **Split (권장)** | `configs/pipeline.yaml` | `configs/datasets/santander.yaml` | 공통 + 데이터셋 별도 파일 deep-merge |
| **Single (하위 호환)** | `configs/santander/pipeline.yaml` | (없음) | 단일 파일 그대로 로드 |

```python
# containers/training/train.py 내부 로직
config_str = hp.get("config", "{}")
dataset_config_str = hp.get("dataset_config", "")

if dataset_config_str:
    # Split-config pattern: deep-merge common + dataset-specific
    config = _load_merged_config(config_path, dataset_config_path)
else:
    # Legacy single-file pattern
    config = yaml.safe_load(open(config_path))
```

`deep_merge` 규칙: dict는 재귀 병합, list·scalar는 dataset 파일 값이 우선한다.
dataset 파일에 없는 키(model architecture, training HP, aws 설정 등)는
`pipeline.yaml` 기본값이 자동 적용된다.

### train.py 리팩토링 (2026-04-14)

Model build 로직(435줄)이 `core/model/config_builder.py`로 추출되었다. train.py는 2317줄 → 1882줄로 감소하였으며, PLEConfig 생성은 `build_ple_config()`를 호출하는 방식으로 위임된다.

```python
# containers/training/train.py (리팩토링 후)
from core.model.config_builder import build_ple_config

def main():
    ...
    ple_config = build_ple_config(pipeline_cfg, feature_schema, label_schema)
    model = PLEModel(ple_config)
    ...
```

`build_ple_config()`는 train.py와 PLEPredictor 양쪽에서 공유하는 단일 진실 공급원(single source of truth)이다. PLEConfig가 두 곳에서 다르게 재구성되는 동기화 오류를 원천 차단한다.

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
configs/pipeline.yaml  +  configs/datasets/santander.yaml
         ↓  (deep_merge: load_merged_config)
합산된 config (dataset 키가 우선)
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
   │  │ (10 epoch)  │ - Per-task loss: build_loss() 팩토리         │
   │  │             │ - Uncertainty weighting: Kendall et al.     │
   │  │             │ - Evidential + SAE (config-gated)           │
   │  └──────┬──────┘                                            │
   │         │ S3에 Phase1 체크포인트 저장                          │
   │         ▼                                                    │
   │  ┌─────────────┐                                            │
   │  │ Phase 2     │ SageMaker Training Job #2                  │
   │  │ Tower       │ - Phase1 체크포인트 로드                     │
   │  │ Fine-tune   │ - Extraction layers freeze                  │
   │  │ (10 epoch)  │ - Task Tower + adaTT 미세 조정               │
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
Task Towers (13개 출력)
    ↓
┌──────────────────────────────────────────────────┐
│ Per-task loss 계산                                 │
│                                                    │
│ for task in tasks:                                │
│   loss_fn = build_loss(task.loss, **task.params)  │
│   raw_loss = loss_fn(pred, label)                 │
│                                                    │
│ task_losses = {                                    │
│   "churn_signal": 0.567,     # focal(alpha=0.85)  │
│   "product_stability": 1.23, # huber              │
│   "nba_primary": 0.890,      # ce(auto weights)   │
│   ...                         # 13 tasks total     │
│ }                                                  │
├──────────────────────────────────────────────────┤
│ Auxiliary losses                                    │
│ + Evidential loss (regression tasks)               │
│ + SAE reconstruction loss (detached)               │
│ + CGC entropy regularization                       │
├──────────────────────────────────────────────────┤
│ Uncertainty weighting (Kendall et al.)             │
│                                                    │
│ loss_total = Sigma [loss_weight_k *                │
│              (exp(-log_var_k) * L_k                │
│              + log_var_k / 2)]                     │
│              + sae_weight * sae_loss               │
│              + evidential_loss                      │
│              + cgc_entropy_reg                      │
│                                                    │
│ ※ Bug fix (2026-04-13): 이전 구현은 uncertainty    │
│   weighting 활성 시 task별 loss_weight를 무시했다.  │
│   수정 후: loss_weight * (precision * L + log_var) │
│   형태로 적용하며, log_var는 [-4, 4] clamp.         │
│   이것이 ablation에서 가장 큰 단일 개선이었다.       │
├──────────────────────────────────────────────────┤
│ GradSurgery (tested, not adopted — 실험 전용)       │
│                                                    │
│ - backward() 직후, optimizer.step() 직전에 동작    │
│ - retain_graph=True는 grad_interval=10 step마다만  │
│   사용하여 VRAM 오버헤드를 최소화                   │
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
  batch_size: 5632             # 941K users / 5632 = ~167 steps/epoch (optimized for VRAM)
  epochs: 10                   # ablation 기준 10 epoch (변경: 50 → 10, 2026-04-13)
  learning_rate: 0.0005
  weight_decay: 0.01
  seed: 42

  phase1:
    epochs: 10                 # ablation 기준 (변경: 30 → 10, 2026-04-13)
    description: "Joint training of all tasks"

  phase2:
    epochs: 10
    freeze_extraction: true
    description: "Fine-tune towers with frozen extraction"

  early_stopping:
    patience: 7
    metric: val_loss

  scheduler:
    type: cosine
    warmup_epochs: 3            # adaTT warmup default: 3 (변경: 10 → 3, 2026-04-13)

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
PLE Teacher (GPU, 13 tasks)
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

`scripts/run_santander_ablation.py`가 23 시나리오 ablation (14 joint + 9 structure; v1 paper canonical)을 오케스트레이션한다. 모든 시나리오는 `configs/pipeline.yaml` (공통) + `configs/datasets/santander.yaml` (Santander 특화) + `configs/santander/feature_groups.yaml`에서 동적 생성된다 (하드코딩 없음).

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
│    태스크 수 스케일링 (3 → 5 → 10 → 13)                         │
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
  tasks_3: [churn_signal, product_stability, nba_primary]          # has_nba 통합됨 (2026-04-12)
  tasks_5: [+cross_sell_count, next_mcc, top_mcc_shift, mcc_diversity_trend]  # was tasks_8 when has_nba existed
  tasks_10: [+Tier 3 Product Group + Segmentation일부]
  tasks_13: all

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
    "--num-active-tasks": "활성 태스크 수 (3/5/10/13)",
    "--disable-adatt": "adaTT 비활성화",
    "--disable-ple-stacking": "PLE stacking 비활성화 (단일 CGC)",
    "--loss-weighting-strategy": "uncertainty/gradnorm/dwa/fixed",
    "--ple-num-layers": "PLE stacking depth (1/2/3)",
    "--skip-fidelity-gate": "Fidelity gate 비활성화 (증류 ablation용)",
}
```

---

## 신규 모듈 (2026-04-14)

### core/model/config_builder.py

PLEConfig 생성의 단일 진실 공급원. train.py와 PLEPredictor 양쪽에서 import하여 사용한다.

```python
# core/model/config_builder.py
def build_ple_config(pipeline_cfg: dict, feature_schema: dict, label_schema: dict) -> PLEConfig:
    """
    pipeline.yaml + feature_schema.json + label_schema.json으로부터
    PLEConfig를 완전히 재구성한다.
    - task_loss_weights: pipeline.yaml에서 읽어 전달
    - adaTT task_groups: AdaTTConfig.from_pipeline_groups() 호출
    - task_group_map: adaTT groups에서 자동 빌드
    - logit_transfers: pipeline.yaml task_relationships에서 읽어 전달
    - feature_group_ranges: feature_schema["group_ranges"]에서 읽어 전달
    """
```

### core/inference/predictor.py

체크포인트를 로드하고 PLEConfig를 재구성하여 AMP 추론을 수행하는 서빙 컴포넌트.

```python
# core/inference/predictor.py
class PLEPredictor:
    """
    pipeline.yaml + feature_schema.json에서 build_ple_config()를 호출하여
    PLEConfig를 재구성한다 (train.py와 동일 경로 — 동기화 오류 불가).
    AMP (fp16) 추론 지원. batch 단위 또는 단건 추론 모두 지원.
    """
    def load_checkpoint(self, checkpoint_path: str): ...
    def predict(self, features: np.ndarray) -> dict[str, np.ndarray]: ...
```

### core/evaluation/evaluator.py

태스크 유형별 분리 집계를 포함한 평가 컴포넌트.

```python
# core/evaluation/evaluator.py
class PLEEvaluator:
    """
    Per-task metrics:
      - Binary  → AUC, confusion matrix
      - Multiclass → F1 macro, F1 per-class
      - Ranking  → NDCG@K, Recall@K
      - Regression → MAE, RMSE

    집계:
      avg_auc       → binary tasks only
      avg_f1_macro  → multiclass tasks only
      avg_mae       → regression tasks only
    (전 task 단일 평균 사용 금지 — metric semantics 충돌)
    """
    def evaluate(self, model, dataloader, tasks) -> dict: ...
    def save_metrics(self, metrics: dict, path: str): ...  # eval_metrics.json
```

### containers/evaluation/eval_entry.py

SageMaker Processing Job용 평가 진입점. SM_CHANNEL_TRAIN(피처·레이블)과 SM_CHANNEL_MODEL(체크포인트)을 읽어 PLEEvaluator를 실행하고 결과를 S3에 저장한다.

### scripts/run_sagemaker_teacher.py

3개 시나리오를 병렬 Spot 학습으로 제출하는 오케스트레이션 스크립트. 제출 전 S3에 `eval_metrics.json`이 존재하면 해당 시나리오를 스킵한다.

### scripts/run_sagemaker_eval.py

체크포인트를 S3에 업로드하고 평가 Job을 제출한 뒤 결과를 로컬로 다운로드하는 스크립트.

### scripts/package_source.py

소스 패키징을 재사용 가능하게 분리한 유틸리티. staging 디렉토리 생성 → tarball 빌드 → S3 업로드의 3단계를 수행한다. 모든 Job이 동일 패키지를 재사용하여 1회 빌드 원칙을 준수한다.

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

### Metric Definitions (SageMaker 콘솔 자동 추적)

SageMaker Estimator에 regex 패턴을 등록하면 CloudWatch 및 SageMaker Experiments에 자동으로 메트릭이 기록된다:

```python
metric_definitions = [
    {"Name": "train:loss",      "Regex": r"train_loss=([0-9.]+)"},
    {"Name": "val:loss",        "Regex": r"val_loss=([0-9.]+)"},
    {"Name": "val:avg_auc",     "Regex": r"val_avg_auc=([0-9.]+)"},
    {"Name": "val:avg_f1",      "Regex": r"val_avg_f1_macro=([0-9.]+)"},
    {"Name": "val:avg_mae",     "Regex": r"val_avg_mae=([0-9.]+)"},
    {"Name": "epoch",           "Regex": r"epoch=([0-9]+)"},
]
```

### SageMaker Experiments — Region 자동 감지

Experiments tracker 초기화 시 region을 하드코딩하지 않고 boto3 session에서 자동 감지한다:

```python
import boto3
region = boto3.session.Session().region_name  # 환경 변수 / IMDSv2 자동 조회
run = sagemaker.experiments.Run(..., sagemaker_session=Session(boto_session=boto3.Session(region_name=region)))
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
    "labels": {"path": "labels.parquet", "tasks": 13},
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

> **Metric Aggregation 원칙 (2026-04-11)**: ablation 및 Champion/Challenger 비교에서 전 task 단일 평균을 사용하지 않는다. `avg_auc`는 binary task 전용, `avg_f1_macro`는 multiclass task 전용, `avg_mae`는 regression task 전용으로 분리 집계한다. 세 지표를 별도 열로 표시해야 metric semantics 충돌을 방지할 수 있다.

### 승격 의사결정 흐름 (오프라인 게이트)

증류가 끝난 신규 모델은 `scripts/submit_pipeline.py::_register_model`에서 다음 순서로 판정한다. 어떤 결과라도 `core.monitoring.audit_logger.AuditLogger.log_model_promotion`을 통해 HMAC 서명 + hash chain이 연결된 감사 로그에 기록된다 (SR 11-7 MRM).

```
Distillation 완료
    ↓
ModelRegistry.package(version=…)   ← 항상 등록
    ↓
┌──────────────────────────────────────────────┐
│ ① --force-promote 플래그가 있는가?            │
│     yes → 현 champion 강등 + 신규 promote     │
│           (decision="force_promote", 수동)    │
└──────────────────────────────────────────────┘
    ↓ no
┌──────────────────────────────────────────────┐
│ ② 현재 champion이 등록되어 있는가?             │
│     no  → 신규 promote (bootstrap)           │
│           (decision="bootstrap")             │
└──────────────────────────────────────────────┘
    ↓ yes
┌──────────────────────────────────────────────┐
│ ③ fidelity_summary.failed > 0?               │
│     yes → register only (decision="reject",  │
│           reason="N fidelity failures")      │
└──────────────────────────────────────────────┘
    ↓ no
┌──────────────────────────────────────────────┐
│ ④ ModelCompetition.evaluate(champ, chall)    │
│    - 주요 metric 개선 ≥ min_improvement(0.5%) │
│    - 보조 metric 하락 ≤ max_degradation(2%)   │
│    - paired bootstrap 유의성 (선택)           │
│                                              │
│  promotion_approved=True                     │
│     → 신규 promote (decision="promote")      │
│  promotion_approved=False                    │
│     → register only (decision="reject")      │
└──────────────────────────────────────────────┘
```

**Decision matrix 요약**

| 조건 | 결과 | audit decision |
|---|---|---|
| `--force-promote` | 항상 승격 (수동 override) | `force_promote` |
| champion 없음 | bootstrap 승격 | `bootstrap` |
| fidelity 실패 ≥ 1건 | 등록만 (안전 floor) | `reject` |
| Competition 승인 | 자동 승격 | `promote` |
| Competition 거부 | 등록만 (승격 안 함) | `reject` |

**Safety floor**: fidelity gate는 ModelCompetition과 독립적으로 동작한다. 학습 metric이 champion보다 좋더라도 student↔teacher fidelity가 실패한 버전은 자동 승격 대상에서 제외된다. `--force-promote`로만 강제 가능.

**감사 로그 구조**

모든 승격 판정은 다음 형식으로 S3 WORM (또는 local fallback)에 추가된다:

```json
{
  "operation": "model_promotion:promote",
  "user": "system",
  "status": "SUCCESS",
  "metadata": {
    "champion_version": "v.20260415.170000",
    "challenger_version": "v.20260417.101600",
    "decision": "promote",
    "reason": "Challenger improves primary metric 'avg_auc' by 0.012 (>= 0.005)…",
    "comparison": {"avg_auc": {"champion": 0.80, "challenger": 0.812, "delta": 0.012}},
    "significance": {"bootstrap_pvalue": 0.018, "is_significant": true},
    "trigger": "auto"
  },
  "prev_hash": "…",
  "hmac": "…"
}
```

### 온라인 게이트 (향후 확장)

DynamoDB prediction log가 충분히 누적되면 `core.serving.model_monitor.ModelMonitor.evaluate_champion_challenger`를 사용한 **태스크별 온라인 재평가**를 추가할 수 있다. 오프라인 게이트가 학습 metric을 기반으로 "승격 자격"을 정하고, 온라인 게이트는 실제 트래픽에서 "유지 자격"을 검증한다. 현재는 뼈대만 존재하고 orchestrator에는 연결되지 않은 상태.

---

## Checkpoint Resume 수정사항 (2026-04-14)

체크포인트 재개 로직에서 발견된 두 가지 버그를 수정하였다.

### 파일 패턴 인식

CheckpointManager가 기존에는 `checkpoint_epoch*.pt` 패턴만 인식하였다. 수정 후 `epoch_*.pt` 및 `best.pt` 패턴도 함께 인식한다:

```python
# core/training/checkpoint.py
CHECKPOINT_PATTERNS = [
    "checkpoint_epoch*.pt",   # 기존
    "epoch_*.pt",             # 추가
    "best.pt",                # 추가
]
```

### Epoch 카운팅

재개 시 남은 epoch 수 계산 오류를 수정하였다:

```python
# 수정 전 (버그): target_epoch를 처음부터 다시 학습하는 횟수로 해석
remaining = target_epoch

# 수정 후 (정상): 현재까지 완료된 epoch을 빼서 잔여 epoch 계산
remaining = target_epoch - current_epoch
```

### eval_metrics.json 저장 시점

기존에는 모든 epoch 완료 후 마지막에 한 번만 저장되었다. Spot 중단 시 최종 결과가 유실되는 문제를 방지하기 위해 **best epoch 갱신 시마다** 저장하도록 변경하였다:

```python
# core/training/trainer.py (callback 내부)
if val_loss < self.best_val_loss:
    self.best_val_loss = val_loss
    self._save_checkpoint(epoch, tag="best")
    self._save_eval_metrics(epoch)   # best 갱신마다 즉시 저장
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
  4. eval_metrics.json을 best epoch 갱신마다 저장 (중단 시 결과 보존)
  5. CheckpointManager: checkpoint_epoch*.pt + epoch_*.pt + best.pt 모두 인식

Santander ablation 기준 (10 epochs, ~1시간):
  On-Demand: $0.53
  Spot:      ~$0.16
```

### Windows 절전 방지 (로컬 야간 학습)

로컬 GPU PC에서 overnight ablation 실행 시 Windows 절전 모드가 프로세스를 종료하는 문제가 있다. orchestrator 스크립트에서 `SetThreadExecutionState`를 호출하여 절전을 방지한다:

```python
# scripts/run_santander_ablation.py (orchestrator 진입 시 호출)
import ctypes
ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040
ctypes.windll.kernel32.SetThreadExecutionState(
    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
)
# 실험 완료 후 해제
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
```

---

## 현재 vs AWS — 학습 파이프라인 비교

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| 실행 환경 | 고정 GPU 서버 | SageMaker Spot (필요 시만) | 비용 70% 절감 |
| 2-Phase | trainer.py 내부 로직 | SageMaker Job 2개 분리 | 각 Phase 독립 재실행 가능 |
| 데이터 로딩 | TensorDataset + PLEDataset 이중 | **PyArrow zero-copy** + cross-sectional auto-detect split | pandas 없음, 자동 split 감지 |
| 학습 루프 | 인라인 루프 + PLETrainer 이중 | **PLETrainer 단일** (AMP/callbacks/체크포인트) | 일관성 |
| 태스크 수 | 16개 | **13개** (has_nba → nba_primary 통합; Tier 5 txn-based NBA 포함) | 거래 시퀀스 활용, 중복 제거 |
| Loss 함수 | 코드 내 하드코딩 | **build_loss() + focal_alpha calibrated** | positive rate 반영 |
| Loss 가중치 | 불확실성 (미활성화) | **Uncertainty weighting 활성화** | 자동 밸런싱 |
| 모델 구조 | PLE + adaTT | **+ 7 heterogeneous experts + Evidential + SAE + AMP FP32 loss** | 불확실성 + 해석 가능성 |
| Expert 입력 차원 | 전체 피처 브로드캐스트 | **FeatureRouter 활성화 — Expert별 이종 입력 차원** (deepfm=168D, temporal=139D, hgcn=27D, perslay=32D, causal=161D, lightgcn=100D, ot=127D; 파라미터 4.77M→~2.8M 감소) | 불필요한 피처 제거로 Expert 전문성 강화 |
| Logit Transfer | 단일 방법 | **3-method dispatch (5 edges)** | 관계 유형별 최적화 |
| 해석 가능성 | 없음 | **3-stage (A:분석, B:사유, C:서빙)** | 감사 가능한 추천 |
| 증류 | distillation.py 단일 | **config 기반 + fidelity gate** | 품질 보증 |
| Ablation | 없음 | **4-Dimension × 23 scenarios** (14 joint + 9 structure; v1 paper canonical) | 체계적 실험, Docker local mode |
| 파이프라인 추적 | 없음 | **pipeline_state.json + resume** | 재현성, 장애 복구 |
| 재현성 | 느슨 (코드+데이터 별도) | YAML config + S3 + temporal split | 완전 재현 |
