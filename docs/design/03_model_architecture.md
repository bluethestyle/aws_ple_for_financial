# 03. Model Architecture — PLE, Expert, Task, adaTT 모듈화

## 현재 (On-Prem) 분석

### 아키텍처 요약
```
PLE-Cluster-adaTT
├── Shared Experts (8종, 512D 출력)
│   ├── PersLay (64D) — 위상 데이터 분석
│   ├── DeepFM (64D) — 피처 상호작용
│   ├── Temporal Ensemble (64D) — Mamba + LNN + Transformer
│   ├── Unified H-GCN (128D) — 쌍곡 그래프 합성곱
│   ├── LightGCN (64D) — 협업 필터링
│   ├── Causal (64D) — 인과 구조 학습
│   └── OT (64D) — 최적 수송
├── Task Experts (16 태스크 × 20 클러스터 = 320 subhead)
├── adaTT (적응적 태스크 전이)
└── Task Towers (16개 출력)
```

### 문제점
1. **Expert가 금융 도메인에 종속**: H-GCN의 MCC 계층, PersLay의 금융 거래 위상 등
2. **태스크 16개가 하드코딩**: active_tasks.py에 CTR, CVR, churn 등 고정
3. **클러스터 수(20) 고정**: GMM 클러스터링이 금융 고객 세분화에 특화
4. **Expert 간 의존관계 암묵적**: 코드로만 파악 가능

### 유지할 패턴
- **Expert Registry**: 팩토리 패턴 + config 기반 활성화 → 그대로 확장
- **2-Phase Training**: 공유 Expert → 태스크 전용 → 효과 검증됨
- **adaTT**: 태스크 간 지식 전이 → 범용화 가능
- **CGC (Customized Gate Control)**: PLE 핵심 → 모델 코어에 유지

---

## AWS 설계 — 모듈형 모델 아키텍처

### 전체 구조

```
Config (YAML)
    ↓ 파싱
┌─────────────────────────────────────────────────────┐
│                    PLEModel                         │
│                                                     │
│  ┌─────────────────────────┐  ┌──────────────────┐ │
│  │   SharedExpertPool      │  │  TaskExpertPool   │ │
│  │   (Registry 기반 로드)   │  │  (Config 기반)    │ │
│  │                         │  │                   │ │
│  │  ┌───────┐ ┌────────┐  │  │  ┌──────────────┐│ │
│  │  │Expert1│ │Expert2 │  │  │  │Task1 Subhead ││ │
│  │  │(64D)  │ │(64D)   │  │  │  │Task2 Subhead ││ │
│  │  └───────┘ └────────┘  │  │  │...           ││ │
│  │  ...                    │  │  └──────────────┘│ │
│  └────────────┬────────────┘  └────────┬─────────┘ │
│               │                        │            │
│               ▼                        ▼            │
│         ┌──────────────────────────────────┐        │
│         │   CGC Layer (Gating Network)     │        │
│         │   Expert 출력을 태스크별 가중 결합  │        │
│         └──────────────┬───────────────────┘        │
│                        │                            │
│                        ▼                            │
│         ┌──────────────────────────────────┐        │
│         │   adaTT (Adaptive Task Transfer) │        │
│         │   태스크 간 지식 전이              │        │
│         └──────────────┬───────────────────┘        │
│                        │                            │
│                        ▼                            │
│         ┌──────────────────────────────────┐        │
│         │   Task Towers                    │        │
│         │   (TaskRegistry에서 동적 빌드)    │        │
│         └──────────────────────────────────┘        │
└─────────────────────────────────────────────────────┘
```

### Expert 모듈화

```yaml
# configs/model.yaml — Expert 설정
model:
  architecture: ple

  shared_experts:
    # 기본 Expert (항상 사용 가능)
    - type: deepfm
      enabled: true
      output_dim: 64
      params:
        field_groups: auto       # 스키마에서 자동 추론
        embedding_dim: 16

    - type: mlp
      enabled: true
      output_dim: 64
      params:
        hidden_dims: [256, 128]
        dropout: 0.1

    # 시퀀스가 있을 때만 활성화
    - type: temporal
      enabled: ${has_sequences}  # 조건부 활성화
      output_dim: 64
      params:
        sub_experts: [mamba, transformer]
        d_model: 128

    # 그래프 데이터가 있을 때만 활성화
    - type: hgcn
      enabled: false             # 기본 비활성화
      output_dim: 128
      params:
        curvature: 1.0

    # 커스텀 Expert (플러그인)
    - type: my_custom_expert     # @ExpertRegistry.register("my_custom_expert")
      enabled: false
      output_dim: 64

  # Expert 출력 차원 = sum(enabled expert output_dims)
  # 위 예시: 64 + 64 + 64 = 192D (hgcn, custom 비활성화)
```

### Expert Registry 구현

```python
# core/model/experts/registry.py
class ExpertRegistry:
    """
    Expert 플러그인 레지스트리.

    기본 제공: deepfm, mlp, temporal, hgcn, lightgcn, causal, ot, perslay
    커스텀 추가: @ExpertRegistry.register("name")

    Expert는 반드시 다음 인터페이스를 따릅니다:
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
        """config에서 enabled=True인 Expert만 빌드하여 ModuleList 반환."""
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

### Task 모듈화

```yaml
# configs/tasks.yaml — 태스크 설정 (모델 config에서 분리 가능)
tasks:
  # 태스크 정의 — 어떤 조합이든 가능
  - name: click
    type: binary
    loss: focal
    loss_weight: 1.0
    label_col: clicked
    focal_alpha: 0.25
    focal_gamma: 2.0

  - name: purchase
    type: binary
    loss: focal
    loss_weight: 1.5
    label_col: purchased
    focal_alpha: 0.20       # 극단적 불균형 → alpha 낮춤

  - name: revenue
    type: regression
    loss: huber
    loss_weight: 1.0
    label_col: revenue_30d
    huber_delta: 1.0

  - name: category
    type: multiclass
    loss: ce
    loss_weight: 0.8
    label_col: category_id
    num_classes: 50

  - name: brand_similarity
    type: contrastive
    loss: infonce
    loss_weight: 2.0
    embedding_dim: 128
    temperature: 0.07

# 태스크 그룹 (adaTT 전이 단위)
task_groups:
  engagement: [click, purchase]           # 그룹 내 전이 강도 높음
  value: [revenue, category]
  discovery: [brand_similarity]

# adaTT 설정
adatt:
  intra_group_strength: 0.7    # 그룹 내 전이 가중치
  inter_group_strength: 0.3    # 그룹 간 전이 가중치
  negative_transfer_threshold: -0.1
```

### 클러스터 유연성

```yaml
# configs/model.yaml — 클러스터 설정
clustering:
  method: gmm              # gmm | kmeans | none
  num_clusters: 20          # 도메인에 따라 조정
  cluster_features: auto    # 어떤 피처로 클러스터링할지 (auto = 전체)
  # method: none이면 클러스터 없이 단일 헤드
```

```python
# 클러스터 없이 동작 (단순 PLE)
if config.clustering.method == "none":
    # 클러스터 subhead 없이 단일 Task Expert
    task_expert = TaskExpert(input_dim, config)
else:
    # 클러스터별 subhead (현재 On-Prem 구조)
    task_expert = ClusterTaskExpert(input_dim, config)
```

### 모델 빌드 자동화

```python
# core/model/builder.py
class ModelBuilder:
    """
    Config에서 모델 전체를 자동 빌드합니다.

    ExpertRegistry + TaskRegistry + ModelRegistry를 조합하여
    어떤 Expert/Task 조합이든 config만으로 모델을 구성합니다.
    """

    def build(self, model_config, task_config, feature_schema) -> nn.Module:
        input_dim = feature_schema.input_dim

        # 1. Expert Pool 빌드
        shared_experts = ExpertRegistry.build_pool(
            model_config.shared_experts, input_dim
        )
        expert_output_dim = ExpertRegistry.total_output_dim(
            model_config.shared_experts
        )

        # 2. Task Head 빌드
        tasks = []
        for task_spec in task_config.tasks:
            task_cfg = TaskConfig(
                name=task_spec.name,
                task_type=TaskType(task_spec.type),
                loss_type=LossType(task_spec.loss),
                loss_weight=task_spec.loss_weight,
                num_classes=task_spec.get("num_classes", 1),
                label_col=task_spec.label_col,
            )
            task = TaskRegistry.build(task_cfg, tower_input_dim=expert_output_dim)
            tasks.append(task)

        # 3. PLE 모델 조립
        ple_config = PLEConfig(
            input_dim=input_dim,
            num_tasks=len(tasks),
            num_shared_experts=len(shared_experts),
            expert_hidden_dim=expert_output_dim,
            num_layers=model_config.num_layers,
        )

        return PLEModel(ple_config, tasks)
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
    tasks: [click, purchase, revenue]   # teacher의 출력 중 어떤 태스크를 증류
  student:
    architecture: lgbm
    temperature: 3.0                    # soft label smoothing
    alpha: 0.7                          # soft vs hard label 비율
  output:
    path: s3://bucket/models/lgbm-distill/
```

---

## 현재 vs AWS — 모델 아키텍처 비교

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| Expert 종류 | 8종 하드코딩 | ExpertRegistry (동적) | 도메인별 Expert 선택적 활성화 |
| 태스크 정의 | active_tasks.py 16개 고정 | YAML tasks: 배열 | 코드 변경 없이 태스크 추가/삭제 |
| 클러스터 | GMM 20개 고정 | config에서 method/수 조정 | 도메인별 최적화, 또는 비활성화 |
| adaTT 그룹 | 4그룹 하드코딩 | YAML task_groups | 태스크 조합에 따라 유연하게 |
| 입력 차원 | 734D 하드코딩 | feature_schema.input_dim 동적 | 피처 변경 시 자동 반영 |
| 모델 빌드 | 코드에서 직접 조립 | ModelBuilder (config → model) | 선언적, 재현 가능 |
| 증류 | distillation.py 단일 | config 기반 teacher/student 조합 | 유연한 압축 전략 |
