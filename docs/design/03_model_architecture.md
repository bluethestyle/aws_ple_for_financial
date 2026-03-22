# 03. Model Architecture — PLE, Expert→5축 라우팅, Per-task Loss, Uncertainty Weighting

## 9-Stage 파이프라인에서의 위치

Model Architecture는 Stage 9 (Training)의 모델 구조를 정의한다.

---

## 전체 구조

```
5-Axis Features (Stage 5-6 출력)
    ↓ FeatureRouter (축별 분배)
┌─────────────────────────────────────────────────────────────────────────┐
│                              PLEModel                                   │
│                                                                         │
│  ┌──────────────────────────────────────┐  ┌──────────────────────────┐ │
│  │        SharedExpertPool              │  │    TaskExpertPool        │ │
│  │        (Registry 기반, 5축 라우팅)    │  │    (Task Group별)        │ │
│  │                                      │  │                          │ │
│  │  State축    ──▶ DeepFM, MLP, AutoInt │  │  engagement: [deepfm]   │ │
│  │  Snapshot축 ──▶ PersLay, Causal, OT  │  │  lifecycle:  [causal]   │ │
│  │  Timeseries축──▶ Temporal, Mamba     │  │  value:      [mlp]      │ │
│  │  Hierarchy축──▶ HGCN                 │  │  consumption:[deepfm]   │ │
│  │  Item축    ──▶ LightGCN             │  │                          │ │
│  └─────────────────┬────────────────────┘  └───────────┬──────────────┘ │
│                    │                                    │                │
│                    ▼                                    ▼                │
│          ┌──────────────────────────────────────────────────┐           │
│          │   CGC Layer (Customized Gate Control)            │           │
│          │   Expert 출력을 태스크별 softmax 가중 결합        │           │
│          │   + CGC Attention (per-task expert weighting)    │           │
│          └────────────────────┬─────────────────────────────┘           │
│                               │                                         │
│                               ▼                                         │
│          ┌──────────────────────────────────────────────────┐           │
│          │   adaTT (Adaptive Task Transfer)                 │           │
│          │   태스크 간 지식 전이 (intra/inter group)         │           │
│          │   + Logit Transfer (source→target)               │           │
│          └────────────────────┬─────────────────────────────┘           │
│                               │                                         │
│                               ▼                                         │
│          ┌──────────────────────────────────────────────────┐           │
│          │   Task Towers (16개)                             │           │
│          │   Per-task loss dispatch: build_loss() 팩토리     │           │
│          │   Uncertainty weighting: Kendall et al.          │           │
│          └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Expert → 5-Axis Feature 라우팅

### FeatureRouter

```python
# core/model/ple/feature_router.py
class FeatureRouter:
    """
    5-Axis 피처를 Expert에 라우팅.

    feature_groups.yaml의 target_experts 매핑을 기반으로
    각 Expert가 받을 피처 슬라이스의 인덱스를 사전 계산.
    """

    def __init__(self, feature_schema: FeatureSchema, expert_names: list[str]):
        # Expert별 입력 피처 인덱스 사전 계산
        self._expert_indices: dict[str, list[int]] = {}
        for expert_name in expert_names:
            indices = feature_schema.get_expert_feature_indices(expert_name)
            self._expert_indices[expert_name] = indices

    def route(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """전체 피처 텐서에서 Expert별 서브셋 추출."""
        return {
            name: x[:, indices]
            for name, indices in self._expert_indices.items()
        }
```

### Expert Pool — 11종 등록

| # | 등록 이름 | 파일 | 주요 Axis | 설명 |
|---|-----------|------|-----------|------|
| 1 | `mlp` | `core/model/experts/mlp.py` | State | 기본 MLP 베이스라인 |
| 2 | `deepfm` | `core/model/experts/deepfm.py` | State | FM + Deep 피처 상호작용 |
| 3 | `mamba` | `core/model/experts/mamba.py` | Timeseries | Selective SSM |
| 4 | `temporal_ensemble` | `core/model/experts/temporal.py` | Timeseries | Mamba + Transformer 앙상블 |
| 5 | `causal` | `core/model/experts/causal.py` | Snapshot | NOTEARS DAG 인과 구조 |
| 6 | `optimal_transport` | `core/model/experts/ot.py` | Snapshot | Sinkhorn 최적 수송 |
| 7 | `hgcn` | `core/model/experts/hgcn.py` | Hierarchy | 쌍곡 그래프 합성곱 |
| 8 | `perslay` | `core/model/experts/perslay.py` | Snapshot | 위상 데이터 분석 (TDA global) |
| 9 | `lightgcn` | `core/model/experts/lightgcn.py` | Item | 경량 그래프 합성곱 |
| 10 | `autoint` | `core/model/experts/autoint.py` | State | Self-Attention 상호작용 |
| 11 | `xdeepfm` | `core/model/experts/xdeepfm.py` | State | CIN + Deep |

### 현재 금융 Basket 설정

```yaml
# configs/financial/pipeline.yaml
expert_basket:
  shared_experts:
    - hgcn               # Hierarchy축
    - perslay             # Snapshot축
    - deepfm              # State축
    - temporal_ensemble   # Timeseries축
    - lightgcn            # Item축
    - causal              # Snapshot축
    - optimal_transport   # Snapshot축
  task_experts:
    - mlp                 # 범용
  expert_configs:
    hgcn:
      hidden_dim: 128
      dropout: 0.2
    perslay:
      use_raw_diagram: false
      tda_dim: 70
    deepfm:
      embedding_dim: 16
      hidden_dims: [256, 128, 64]
    lightgcn:
      hidden_dim: 128
      num_layers: 3
```

### Expert Registry 구현

```python
# core/model/experts/registry.py
class ExpertRegistry:
    """
    Expert 플러그인 레지스트리.

    기본 제공: deepfm, mlp, temporal_ensemble, mamba, hgcn, lightgcn,
              causal, optimal_transport, perslay, autoint, xdeepfm
    커스텀 추가: @ExpertRegistry.register("name")

    Expert 인터페이스:
    - __init__(self, input_dim, config) → 초기화
    - forward(self, x, **kwargs) → (batch, output_dim) 텐서
    """

    _registry: dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(expert_cls):
            cls._registry[name] = expert_cls
            return expert_cls
        return decorator

    @classmethod
    def build_pool(cls, config: list[dict], input_dim: int) -> nn.ModuleList:
        """config에서 enabled=True인 Expert만 빌드."""
        experts = []
        for spec in config:
            if not spec.get("enabled", True):
                continue
            expert_cls = cls._registry[spec["type"]]
            experts.append(expert_cls(input_dim, spec))
        return nn.ModuleList(experts)

    @classmethod
    def total_output_dim(cls, config: list[dict]) -> int:
        """활성화된 Expert의 출력 차원 합계."""
        return sum(
            spec["output_dim"]
            for spec in config
            if spec.get("enabled", True)
        )
```

---

## Per-Task Loss Dispatch

### build_loss() 팩토리

```python
# core/task/losses.py
def build_loss(loss_type: str, **kwargs) -> nn.Module:
    """Config에서 지정된 loss type에 따라 적절한 loss 함수를 빌드.

    지원 타입:
    - focal   → FocalLoss(alpha, gamma) — binary 불균형
    - huber   → nn.SmoothL1Loss(beta=delta) — 회귀, 이상치 강건
    - mse     → nn.MSELoss() — 회귀 기본
    - ce      → nn.CrossEntropyLoss(weight=class_weights) — multiclass
    - infonce → InfoNCELoss(temperature) — contrastive learning

    Parameters
    ----------
    loss_type : str
        Loss 함수 타입 (위 5종 중 하나)
    **kwargs : dict
        focal: alpha, gamma
        huber: delta
        ce: class_weights (auto 또는 수동)
        infonce: temperature, embedding_dim
    """
    if loss_type == "focal":
        return FocalLoss(
            alpha=kwargs.get("alpha", 0.25),
            gamma=kwargs.get("gamma", 2.0),
        )
    elif loss_type == "huber":
        return nn.SmoothL1Loss(beta=kwargs.get("delta", 1.0))
    elif loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "ce":
        weight = kwargs.get("class_weights")  # None이면 균등
        return nn.CrossEntropyLoss(weight=weight)
    elif loss_type == "infonce":
        return InfoNCELoss(
            temperature=kwargs.get("temperature", 0.07),
            embedding_dim=kwargs.get("embedding_dim", 128),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
```

### Class Weight 자동 계산

```python
# core/task/class_weights.py
def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """학습 데이터의 클래스 분포에서 자동으로 class weight를 계산.

    weight_c = N / (C * n_c)
    - N: 전체 샘플 수
    - C: 클래스 수
    - n_c: 클래스 c의 샘플 수

    PLEModel.set_class_weights()에서 호출되어
    multiclass 태스크의 CrossEntropyLoss에 주입됨.
    """
    counts = torch.bincount(labels.long(), minlength=num_classes).float()
    counts = counts.clamp(min=1)  # division by zero 방지
    weights = len(labels) / (num_classes * counts)
    return weights
```

### Task Config

```yaml
# configs/financial/pipeline.yaml — 태스크별 loss 설정
tasks:
  # Binary — FocalLoss
  - name: ctr
    type: binary
    loss: focal              # build_loss("focal", alpha=0.25, gamma=2.0)
    loss_weight: 1.0
    label_col: label_ctr

  # Regression — HuberLoss (이상치 강건)
  - name: ltv
    type: regression
    loss: huber              # build_loss("huber", delta=1.0)
    loss_weight: 1.5
    label_col: label_ltv

  # Multiclass — CrossEntropy + auto class_weights
  - name: nba
    type: multiclass
    loss: ce                 # build_loss("ce", weight=auto_class_weights)
    loss_weight: 2.0
    label_col: label_nba
    num_classes: 12

  # Contrastive — InfoNCE
  - name: brand_prediction
    type: contrastive
    loss: infonce             # build_loss("infonce", temperature=0.07)
    loss_weight: 2.0
    label_col: label_next_brand
    num_classes: 128
    temperature: 0.07
```

---

## Uncertainty Weighting (Kendall et al., CVPR 2018)

### 원리

태스크별 learnable `log_var` 파라미터를 도입하여, 불확실성이 높은 태스크의 가중치를 자동으로 낮춘다.

```
loss_total = Σ_k [ exp(-log_var_k) * L_k + log_var_k / 2 ]

- log_var_k: 태스크 k의 learnable 파라미터 (학습 중 자동 조정)
- exp(-log_var_k): 정밀도 (precision) — 불확실성 높으면 가중치 낮음
- log_var_k / 2: 정규화 항 — log_var가 무한히 커지는 것 방지
```

### 구현

```python
# core/model/ple/loss_weighting.py
class UncertaintyWeighting(BaseLossWeighting):
    """Kendall et al. (CVPR 2018) uncertainty weighting.

    각 태스크의 homoscedastic uncertainty를 학습하여
    태스크 간 loss 스케일을 자동으로 밸런싱.

    Parameters
    ----------
    num_tasks : int
        태스크 수 (16)
    init_log_var : float
        log_var 초기값 (기본 0.0 → 초기 가중치 1.0)
    """

    def __init__(self, num_tasks: int, init_log_var: float = 0.0):
        super().__init__()
        self.log_vars = nn.Parameter(
            torch.full((num_tasks,), init_log_var)
        )

    def forward(self, task_losses: dict[str, torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for i, (name, loss) in enumerate(task_losses.items()):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i] / 2
        return total

    def get_weights(self) -> dict[str, float]:
        """현재 학습된 태스크별 가중치 반환 (모니터링용)."""
        with torch.no_grad():
            precisions = torch.exp(-self.log_vars).cpu().numpy()
        return {f"task_{i}": float(p) for i, p in enumerate(precisions)}
```

### Loss 계산 전체 흐름

```
Task Towers (16개 출력)
    ↓
┌──────────────────────────────────────────────────┐
│ Per-task loss 계산                                 │
│                                                    │
│ for task in tasks:                                │
│   raw_loss = build_loss(task.loss)(pred, label)   │
│   # focal / huber / mse / ce / infonce            │
│                                                    │
│ task_losses = {                                    │
│   "ctr": 0.234,                                   │
│   "cvr": 0.567,                                   │
│   "ltv": 1.234,                                   │
│   "nba": 0.890,                                   │
│   ...                                              │
│ }                                                  │
├──────────────────────────────────────────────────┤
│ Uncertainty weighting                              │
│                                                    │
│ loss_total = Σ [exp(-log_var_k) * L_k             │
│               + log_var_k / 2]                     │
│                                                    │
│ 불확실성이 높은 태스크(ltv 등) → 자동 가중치 감소    │
│ 안정적인 태스크(ctr 등) → 자동 가중치 증가           │
├──────────────────────────────────────────────────┤
│ Optimizer step                                     │
│ log_var도 함께 업데이트 (learnable parameter)        │
└──────────────────────────────────────────────────┘
```

### Loss Weighting 전략 선택

```yaml
# configs/training.yaml
training:
  loss_weighting:
    strategy: uncertainty    # 기본값 (권장)
    # 대안:
    # fixed    — task config의 loss_weight 고정 사용
    # gradnorm — Chen et al. (ICML 2018) gradient 균형
    # dwa      — Dynamic Weight Averaging (Liu et al.)
    # manual   — 수동 설정
```

---

## Task 모듈화

### Task Groups

```yaml
# configs/financial/pipeline.yaml
task_groups:
  - name: engagement
    tasks: [ctr, cvr]
    task_experts: [deepfm]
    adatt_intra_strength: 0.8
    adatt_inter_strength: 0.3

  - name: lifecycle
    tasks: [churn, retention, life_stage, ltv]
    task_experts: [causal, mlp]
    adatt_intra_strength: 0.6

  - name: value
    tasks: [balance_util, channel, timing]
    task_experts: [mlp]
    adatt_intra_strength: 0.7

  - name: consumption
    tasks: [nba, spending_category, consumption_cycle,
            spending_bucket, merchant_affinity, brand_prediction]
    task_experts: [deepfm, xdeepfm]
    adatt_intra_strength: 0.7
```

### adaTT (Adaptive Task Transfer)

```
Task Group 내 전이 (intra): 강도 높음 (0.6-0.8)
  ctr ↔ cvr (engagement 내)
  churn ↔ retention ↔ ltv (lifecycle 내)

Task Group 간 전이 (inter): 강도 낮음 (0.3)
  engagement → lifecycle (CTR 정보가 churn 예측에 도움)
  lifecycle → consumption (LTV가 소비 패턴에 영향)
```

### 클러스터 유연성

```yaml
clustering:
  method: gmm              # gmm | kmeans | none
  num_clusters: 20          # 도메인에 따라 조정
  cluster_features: auto    # 어떤 피처로 클러스터링할지 (auto = 전체)
```

---

## 모델 빌드 자동화

```python
# core/model/builder.py
class ModelBuilder:
    """
    Config에서 모델 전체를 자동 빌드.

    ExpertRegistry + TaskRegistry + ModelRegistry를 조합하여
    어떤 Expert/Task 조합이든 config만으로 모델을 구성.
    5-Axis Feature Router도 자동 생성.
    """

    def build(self, model_config, task_config, feature_schema) -> nn.Module:
        input_dim = feature_schema.input_dim

        # 1. Feature Router (5축 → Expert 라우팅)
        router = FeatureRouter(feature_schema, expert_names=[...])

        # 2. Expert Pool 빌드 (Registry 기반)
        shared_experts = ExpertRegistry.build_pool(
            model_config.shared_experts, input_dim
        )

        # 3. Task Head 빌드 (per-task loss 포함)
        tasks = []
        for task_spec in task_config.tasks:
            loss_fn = build_loss(task_spec.loss, **task_spec.loss_params)
            task = TaskRegistry.build(task_spec, loss_fn=loss_fn)
            tasks.append(task)

        # 4. Loss Weighting (Uncertainty 기본)
        loss_weighting = create_loss_weighting(
            strategy=model_config.loss_weighting.strategy,
            num_tasks=len(tasks),
        )

        # 5. PLE 모델 조립
        return PLEModel(
            ple_config=PLEConfig(...),
            feature_router=router,
            tasks=tasks,
            loss_weighting=loss_weighting,
        )
```

---

## 지식 증류 (PLE → LGBM)

```
PLE Teacher (GPU 학습)
    ↓ Soft Labels 생성
    ↓ S3에 저장
LGBM Student (CPU 학습)
    ↓ Soft Labels로 학습
    ↓ 경량 모델 저장
서빙 (실시간: LGBM, 배치: PLE)
```

```yaml
# configs/distillation.yaml
distillation:
  teacher:
    model_path: s3://bucket/models/ple-latest/
    tasks: [click, purchase, revenue]
  student:
    architecture: lgbm
    temperature: 3.0
    alpha: 0.7
  output:
    path: s3://bucket/models/lgbm-distill/
```

---

## 현재 vs AWS — 모델 아키텍처 비교

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| Expert 종류 | 8종 하드코딩 | ExpertRegistry 11종 (동적) | 도메인별 Expert 선택적 활성화 |
| Expert 라우팅 | 암묵적 (코드 내 분기) | **5-Axis FeatureRouter** (명시적) | 피처→Expert 매핑 투명화 |
| 태스크 정의 | active_tasks.py 16개 고정 | YAML tasks: 배열 | 코드 변경 없이 태스크 추가/삭제 |
| Loss 함수 | 코드 내 하드코딩 | **build_loss() 팩토리** (focal/huber/mse/ce/infonce) | config에서 선언적 지정 |
| Loss 가중치 | 불확실성 기반 (미활성화) | **Uncertainty weighting 활성화** (Kendall et al.) | 태스크 간 자동 밸런싱 |
| Class weight | 수동 설정 | **자동 계산** (set_class_weights) | multiclass 불균형 자동 대응 |
| 클러스터 | GMM 20개 고정 | config에서 method/수 조정 | 도메인별 최적화, 또는 비활성화 |
| adaTT 그룹 | 4그룹 하드코딩 | YAML task_groups (1등 시민) | adaTT + loss + monitoring 공통 참조 |
| 입력 차원 | 734D 하드코딩 | feature_schema.input_dim 동적 | 피처 변경 시 자동 반영 |
| 모델 빌드 | 코드에서 직접 조립 | ModelBuilder (config → model) | 선언적, 재현 가능 |
| 증류 | distillation.py 단일 | config 기반 teacher/student 조합 | 유연한 압축 전략 |
