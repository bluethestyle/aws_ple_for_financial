# 04. Training Pipeline — PLETrainer 단일 경로, 4-Dimension Ablation 설계

## 9-Stage 파이프라인에서의 위치

Training Pipeline은 Stage 9를 담당한다.

---

## 학습 파이프라인 흐름

```
configs/training.yaml + pipeline.yaml
         ↓
Step Functions (오케스트레이션)
         ↓
   ┌──────────────────────────────────────────────────────────────┐
   │                     PLETrainer 단일 경로                      │
   │                                                              │
   │  ┌─────────────┐                                            │
   │  │ Phase 1     │ SageMaker Training Job #1                  │
   │  │ Shared      │ - Spot Instance (g4dn.xlarge)              │
   │  │ Experts     │ - 체크포인트 → S3 (Spot 중단 대비)            │
   │  │ (15 epoch)  │ - Per-task loss: build_loss() 팩토리         │
   │  │             │ - Uncertainty weighting: Kendall et al.     │
   │  └──────┬──────┘                                            │
   │         │ S3에 Phase1 체크포인트 저장                          │
   │         ▼                                                    │
   │  ┌─────────────┐                                            │
   │  │ Phase 2     │ SageMaker Training Job #2                  │
   │  │ Task Heads  │ - Phase1 체크포인트 로드                     │
   │  │ Fine-tune   │ - Shared Expert freeze (선택적)              │
   │  │ (8 epoch)   │ - Task Tower + adaTT 미세 조정               │
   │  └──────┬──────┘                                            │
   │         │                                                    │
   │         ▼                                                    │
   │  ┌─────────────┐                                            │
   │  │ 평가         │ SageMaker Processing Job                   │
   │  │ + 등록       │ - 테스트셋 평가                              │
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
    4. Mixed Precision (fp16 AMP)
    5. Gradient clipping + accumulation
    6. Expert별 learning rate override
    7. S3 체크포인트 (Spot 중단 자동 대비)
    8. SageMaker Experiments 메트릭 로깅

    사용법:
        config = TrainingConfig.from_yaml("training.yaml")
        model = PLEModel(ple_config)
        trainer = PLETrainer(model, config, device=torch.device("cuda"))
        results = trainer.train(train_loader, val_loader)
    """
```

### Per-Task Loss 계산 흐름

```python
# PLETrainer._compute_loss() 내부
def _compute_loss(self, predictions, labels, task_configs):
    task_losses = {}

    for task in task_configs:
        pred = predictions[task.name]
        label = labels[task.label_col]

        # 1. Per-task loss dispatch
        loss_fn = self._loss_fns[task.name]  # build_loss()로 사전 생성
        raw_loss = loss_fn(pred, label)

        task_losses[task.name] = raw_loss

    # 2. Uncertainty weighting (Kendall et al.)
    # loss_total = Σ [exp(-log_var_k) * L_k + log_var_k / 2]
    total_loss = self._loss_weighting(task_losses)

    return total_loss, task_losses
```

---

## Training Config

```yaml
# configs/training.yaml
training:
  seed: 42

  # ── Phase 1: Shared Expert 학습 ──
  phase1:
    epochs: 15
    batch_size: 4096
    optimizer:
      type: adamw
      lr: 0.0005
      weight_decay: 0.01
    scheduler:
      type: cosine_warmup
      warmup_epochs: 5
      T_0: 10
      T_mult: 2
    gradient:
      clip_norm: 5.0
      accumulation_steps: 4     # effective batch: 4096 × 4 = 16384
    mixed_precision: true       # fp16

    # Expert별 학습률 (선택적)
    expert_lr_overrides:
      temporal_ensemble: 0.0003 # 시퀀스 Expert는 보수적으로
      hgcn: 0.0005
      causal: 0.0001            # DAG 제약 → 매우 보수적

  # ── Phase 2: Task Head 미세 조정 ──
  phase2:
    epochs: 8
    freeze_shared: true         # Shared Expert 동결
    batch_size: 4096
    optimizer:
      type: adamw
      lr: 0.0002                # Phase1보다 낮은 lr
      weight_decay: 0.01
    scheduler:
      type: cosine_warmup
      warmup_epochs: 2

  # ── 조기 종료 ──
  early_stopping:
    patience: 5
    metric: val_loss
    mode: min

  # ── 체크포인트 ──
  checkpoint:
    save_every_n_epochs: 5
    max_keep: 3
    save_best: true
    s3_path: s3://aiops-ple-financial/models/checkpoints/

  # ── Loss 가중치 전략 ──
  loss_weighting:
    strategy: uncertainty        # 기본: Kendall et al. uncertainty weighting
    # uncertainty: 태스크별 learnable log_var → 자동 밸런싱
    # fixed: pipeline.yaml의 loss_weight 고정 사용
    # gradnorm: Chen et al. (ICML 2018) gradient 균형
    # dwa: Dynamic Weight Averaging

  # ── Per-task Loss 함수 ──
  # pipeline.yaml의 tasks[].loss 필드에서 지정
  # build_loss() 팩토리가 focal/huber/mse/ce/infonce 디스패치
  # multiclass 태스크는 class_weights 자동 계산

# ── SageMaker 설정 ──
aws:
  instance_type: ml.g4dn.xlarge
  use_spot: true
  max_run_seconds: 14400        # 4시간
  volume_size_gb: 50
```

---

## 4-Dimension Ablation 설계

### 개요

4개 차원에서 체계적으로 ablation 실험을 설계:

```
┌──────────────────────────────────────────────────────────────────┐
│                    4-Dimension Ablation Framework                │
│                                                                  │
│  Dim 1: Feature Ablation                                        │
│    5-Axis별 피처 그룹 제거 → 축 기여도 측정                        │
│                                                                  │
│  Dim 2: Expert Ablation                                         │
│    개별/그룹 Expert 제거 → Expert 기여도 측정                      │
│    (피처-전문가 연동: 피처 제거 시 대응 Expert도 함께 비활성화)      │
│                                                                  │
│  Dim 3: Task Ablation                                           │
│    태스크 수 스케일링 (4 → 8 → 16) + 구조 변형                    │
│    PLE only / adaTT only / PLE+adaTT / baseline                 │
│                                                                  │
│  Dim 4: Structure Ablation                                      │
│    Loss weighting: Uncertainty vs GradNorm vs DWA vs Fixed      │
│    PLE stacking depth: 1 → 2 → 3 layers                        │
│    adaTT intra/inter strength 변형                               │
└──────────────────────────────────────────────────────────────────┘
```

### Dim 1: Feature Ablation (5-Axis 기반)

```yaml
# Ablation 시나리오
feature_ablation:
  # 축별 제거
  - name: no_state
    remove_axes: [state]
    description: "State 축 전체 제거 (314D) → RFM/demographics 없이"

  - name: no_snapshot
    remove_axes: [snapshot]
    description: "Snapshot 축 전체 제거 (139D) → 장기 TDA/HMM 없이"

  - name: no_timeseries
    remove_axes: [timeseries]
    description: "Timeseries 축 전체 제거 (214D) → 단기 시퀀스 없이"

  - name: no_hierarchy
    remove_axes: [hierarchy]
    description: "Hierarchy 축 전체 제거 (41D) → 계층 구조 없이"

  - name: no_item
    remove_axes: [item]
    description: "Item 축 전체 제거 (64D) → 협업 필터링 없이"

  # 개별 그룹 제거
  - name: no_tda
    remove_groups: [tda_global, tda_local]
    description: "TDA 피처 전체 제거 (70D)"

  - name: no_graph
    remove_groups: [graph_embeddings, merchant_hierarchy]
    description: "Graph 피처 전체 제거 (41D)"
```

### Dim 2: Expert Ablation (피처-전문가 연동)

```yaml
expert_ablation:
  # 피처 제거 시 대응 Expert도 함께 비활성화
  - name: no_graph_coupled
    remove_groups: [graph_embeddings]
    remove_experts: [lightgcn, hgcn]
    description: "Graph 피처 + LightGCN/HGCN Expert 동시 제거"

  - name: no_tda_coupled
    remove_groups: [tda_global, tda_local]
    remove_experts: [perslay]
    description: "TDA 피처 + PersLay Expert 동시 제거"

  - name: no_temporal_coupled
    remove_groups: [base_temporal, mamba_temporal]
    remove_experts: [temporal_ensemble, mamba]
    description: "Temporal 피처 + Temporal/Mamba Expert 동시 제거"

  # 개별 Expert 제거 (피처는 유지)
  - name: no_causal
    remove_experts: [causal]
    description: "Causal Expert만 제거 → DAG 인과 구조 기여도"

  - name: no_ot
    remove_experts: [optimal_transport]
    description: "OT Expert만 제거 → 최적 수송 기여도"
```

### Dim 3: Task Ablation (태스크 수 스케일링 + 구조 변형)

```yaml
task_ablation:
  # 태스크 수 스케일링
  task_subsets:
    4_tasks: [ctr, churn, ltv, life_stage]
    8_tasks: [ctr, churn, ltv, life_stage, channel, nba, spending_category, merchant_affinity]
    16_tasks: all

  # 구조 변형 (각 태스크 서브셋에 대해 4가지 구성 실행)
  structure_variants:
    - name: ple_adatt         # PLE(stacked CGC) + adaTT — full
      ple_stacking: true
      adatt: true

    - name: ple_only          # PLE만 (adaTT 없음)
      ple_stacking: true
      adatt: false

    - name: adatt_only        # adaTT만 (단일 MLP shared)
      ple_stacking: false
      adatt: true

    - name: baseline          # 단순 멀티태스크 MLP
      ple_stacking: false
      adatt: false

  # 총 시나리오: 3 subsets × 4 variants = 12 실험
  # 기대 결과: 태스크 수 증가 → PLE+adaTT 조합의 이점 커짐
```

### Dim 4: Structure Ablation (Loss/Layer/adaTT 변형)

```yaml
structure_ablation:
  # Loss weighting 비교
  loss_weighting_variants:
    - {strategy: uncertainty, description: "Kendall et al. (기본)"}
    - {strategy: gradnorm, description: "Chen et al. gradient 균형"}
    - {strategy: dwa, description: "Dynamic Weight Averaging"}
    - {strategy: fixed, description: "고정 가중치 (pipeline.yaml loss_weight)"}

  # PLE stacking depth
  ple_layers: [1, 2, 3]

  # adaTT strength 변형
  adatt_variants:
    - {intra: 0.5, inter: 0.1, description: "약한 전이"}
    - {intra: 0.7, inter: 0.3, description: "기본 (현재 설정)"}
    - {intra: 0.9, inter: 0.5, description: "강한 전이"}
```

### Ablation 구현 HP (Hyperparameters)

```python
# train.py에 추가되는 HP
ablation_hps = {
    # Feature ablation
    "--removed-feature-groups": "쉼표로 구분된 제거할 feature group 이름",
    "--removed-axes": "쉼표로 구분된 제거할 axis (state,snapshot,...)",

    # Expert ablation
    "--removed-experts": "쉼표로 구분된 제거할 expert 이름",

    # Task ablation
    "--num-active-tasks": "활성 태스크 수 (4/8/16)",
    "--active-tasks": "쉼표로 구분된 활성 태스크 이름",

    # Structure ablation
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

    내부적으로 PLETrainer만 사용 (레거시 인라인 루프 제거됨).
    PLETrainer가 optimizer, scheduler, AMP, freeze/unfreeze를 일괄 관리.
    데이터 로딩은 PLEDataset → DataLoader 단일 경로.
    소스 패키징은 동적 패키징만 (_source_pkg/ 제거됨).
    """

    def launch(self) -> dict:
        # Phase 1
        phase1_result = self._launch_phase(
            phase="phase1",
            hyperparameters=self.config.training.phase1,
            checkpoint_s3=self.config.training.checkpoint.s3_path,
        )

        # Phase 2 (Phase1 체크포인트를 입력으로)
        phase2_result = self._launch_phase(
            phase="phase2",
            hyperparameters=self.config.training.phase2,
            model_uri=phase1_result["s3_model_uri"],
        )

        return phase2_result
```

---

## 실험 관리

| 항목 | MLflow (현재) | SageMaker Experiments (AWS) |
|------|-------------|---------------------------|
| 서버 | Docker 자체 호스팅 (상시 가동) | AWS 관리형 (비용 0) |
| 메트릭 로깅 | `mlflow.log_metric()` | `sagemaker.experiments` |
| 모델 저장 | MLflow Model Registry | S3 + 메타데이터 |
| 하이퍼파라미터 | 수동 로깅 | YAML에서 자동 추출 |
| 실험 비교 | MLflow UI | SageMaker Studio |

```python
# core/experiment/tracker.py
class ExperimentTracker:
    """
    MLflow / SageMaker Experiments 추상화.

    로컬 개발: MLflow (docker 불필요, 파일 백엔드)
    AWS 실행: SageMaker Experiments (자동)
    """

    def __init__(self, backend="auto"):
        if backend == "auto":
            self.backend = "sagemaker" if self._in_sagemaker() else "mlflow"

    def log_metric(self, name: str, value: float, step: int = None): ...
    def log_params(self, params: dict): ...
    def log_model(self, model_path: str, metadata: dict): ...
```

---

## Champion/Challenger 평가

```
현재 모델 (Champion)
    vs.
새 모델 (Challenger)
    ↓
┌─────────────────────────────────────────┐
│ 평가 기준                                │
│                                         │
│ 1. 주요 메트릭 (태스크별)                 │
│    - Binary: AUC-ROC > champion - 0.01  │
│    - Regression: MAE < champion + 5%    │
│    - Multiclass: F1-macro > champion    │
│    - Contrastive: Recall@K > champion   │
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

학습 4시간 기준:
  On-Demand: $2.10
  Spot:      ~$0.64  ← 이 비용으로 학습 1회
```

---

## 현재 vs AWS — 학습 파이프라인 비교

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| 실행 환경 | 고정 GPU 서버 | SageMaker Spot (필요 시만) | 비용 70% 절감 |
| 2-Phase | trainer.py 내부 로직 | SageMaker Job 2개 분리 | 각 Phase 독립 재실행 가능 |
| 데이터 로딩 | TensorDataset + PLEDataset 이중 경로 | **PLEDataset 단일 경로** | 레거시 제거 |
| 학습 루프 | 인라인 루프 + PLETrainer 이중 경로 | **PLETrainer 단일 경로** | AMP/체크포인트 일관성 |
| Loss 함수 | 코드 내 하드코딩 | **build_loss() 팩토리** (focal/huber/mse/ce/infonce) | config 선언적 지정 |
| Loss 가중치 | 불확실성 기반 (미활성화) | **Uncertainty weighting 활성화** (Kendall et al.) | 태스크 간 자동 밸런싱 |
| Class weight | 수동 | **자동 계산** (multiclass) | 불균형 자동 대응 |
| 패키징 | `_source_pkg/` 정적 복사 | 동적 패키징만 | 스테일 코드 위험 제거 |
| 체크포인트 | 로컬 디스크 | S3 (Spot 중단 자동 대비) | 내구성, 재개 |
| 실험 관리 | MLflow Docker | SageMaker Experiments | 서버 유지비 0 |
| Ablation | 없음 | **4-Dimension** (feature/expert/task/structure) | 체계적 실험 설계 |
| 재현성 | 느슨 (코드+데이터 별도) | YAML config + S3 데이터 버전 | 완전 재현 |
