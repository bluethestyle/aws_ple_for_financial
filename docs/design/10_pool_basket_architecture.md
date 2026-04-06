# 10. Pool / Basket / Runtime 3계층 아키텍처

## 1. 개요

### 3계층 패턴

```
Pool (등록 목록)        Basket (Config 선택)       Runtime (가중 실행)
┌─────────────────┐    ┌──────────────────┐      ┌──────────────────┐
│ 모든 구성 요소가  │    │ YAML config으로   │      │ 실행 시점에       │
│ Registry에 등록  │───▶│ 부분 집합 선택    │────▶│ 가중치 기반 결합  │
│ (플러그인 방식)   │    │ (도메인별 교체)   │      │ (학습/추론)       │
└─────────────────┘    └──────────────────┘      └──────────────────┘
     코드 영역              Config 영역              모델 영역
```

이 패턴은 Expert, Feature Generator, Task 세 축에 동일하게 적용된다.

| 계층 | 역할 | 변경 주체 |
|------|------|-----------|
| **Pool** | 사용 가능한 모든 구성 요소를 Registry에 등록 | 개발자 (코드 추가) |
| **Basket** | 특정 파이프라인에 사용할 부분 집합을 YAML로 선택 | 운영자 (config 교체) |
| **Runtime** | 선택된 구성 요소를 가중 결합하여 실행 | 모델 (학습 중 자동) |

### 도입 배경

원본 On-Prem 시스템에서 도메인 전환(금융 -> 이커머스 등)을 하려면 Expert 종류, 태스크 목록, 피처 파이프라인 코드를 전부 수정해야 했다. 3계층 패턴을 도입하면:

1. **Pool에 모든 구성 요소를 등록**해 두고
2. **Basket(config)만 교체**하면 도메인이 바뀌고
3. **코드 수정은 0**

이것이 `00_architecture_overview.md`의 핵심 원칙 1번(Config-Driven)과 2번(Registry Pattern)을 구체적으로 실현하는 패턴이다.

---

## 2. Expert Pool/Basket (구현 완료)

### Expert Pool -- 11종 등록 목록

Expert Pool은 `@ExpertRegistry.register` 데코레이터로 등록된 모든 전문가 네트워크의 전체 목록이다.

| # | 등록 이름 | 클래스 | 파일 | 주요 입력 | 설명 |
|---|-----------|--------|------|-----------|------|
| 1 | `mlp` | `MLPExpert` | `core/model/experts/mlp.py` | 전체 피처 (316D, 라우팅 없음) | 기본 MLP 베이스라인 (태스크 전용) |
| 2 | `deepfm` | `DeepFMExpert` | `core/model/experts/deepfm.py` | State 피처 (162D) | FM + Deep 피처 상호작용 |
| 3 | `mamba` | `MambaExpert` | `core/model/experts/mamba.py` | 시퀀스 | Selective State Space Model |
| 4 | `temporal_ensemble` | `TemporalEnsembleExpert` | `core/model/experts/temporal.py` | Timeseries 피처 (127D) | Mamba + Transformer 앙상블 |
| 5 | `causal` | `CausalExpert` | `core/model/experts/causal.py` | Snapshot+State 피처 (158D) | NOTEARS DAG 인과 구조 학습 |
| 6 | `optimal_transport` | `OptimalTransportExpert` | `core/model/experts/ot.py` | Snapshot+State 피처 (124D) | Sinkhorn 최적 수송 |
| 7 | `hgcn` | `UnifiedHGCNExpert` | `core/model/experts/hgcn.py` | Hierarchy 피처 (34D) | 쌍곡 그래프 합성곱 |
| 8 | `perslay` | `PersLayExpert` | `core/model/experts/perslay.py` | TDA 피처 (32D) | 위상 데이터 분석 |
| 9 | `lightgcn` | `LightGCNExpert` | `core/model/experts/lightgcn.py` | Item 그래프 피처 (66D) | 경량 그래프 합성곱 (협업 필터링) |
| 10 | `autoint` | `AutoIntExpert` | `core/model/experts/autoint.py` | 전체 피처 | Self-Attention 상호작용 |
| 11 | `xdeepfm` | `XDeepFMExpert` | `core/model/experts/xdeepfm.py` | 전체 피처 | CIN + Deep |

구현 위치: `core/model/experts/registry.py` -- `ExpertRegistry` 클래스

### Expert Basket -- Config으로 부분 집합 선택

Basket은 Pool에서 특정 파이프라인에 사용할 전문가만 골라낸다. YAML config 한 줄로 교체 가능.

구현 위치:
- Config: `core/model/ple/config.py` -- `ExpertBasketConfig` 데이터클래스
- 로직: `core/model/ple/experts.py` -- `ExpertBasket` 클래스
- 연동: `PLEConfig.expert_basket` 필드에서 참조

```python
@dataclass
class ExpertBasketConfig:
    shared_experts: List[str] = field(default_factory=list)   # Pool에서 선택
    task_experts: List[str] = field(default_factory=list)     # Pool에서 선택
    expert_configs: Dict[str, dict] = field(default_factory=dict)  # 개별 오버라이드
```

### Runtime -- CGC Gating 가중 합산

Basket에서 선택된 전문가들은 `CGCLayer`에 주입되고, 태스크별 Gating Network가 런타임에 가중치를 학습한다.

```
Expert Pool (11종)
    │
    ▼  ExpertBasket 선택
Expert Basket (7 shared + 2 task)
    │
    ▼  FeatureRouter (활성화 — build_model()에서 자동 생성)
        feature_groups.yaml target_experts 선언 → 전문가별 이종 입력 차원
        deepfm: 162D │ temporal_ensemble: 127D │ hgcn: 34D │ perslay: 32D
        causal: 158D │ lightgcn: 66D │ optimal_transport: 124D
        mlp (task): 316D (전체 입력, 라우팅 없음)
    │
    ▼  CGCLayer.forward()
CGC Gating (태스크별 softmax 가중 합산)
    │
    ▼
Task Tower 입력
```

> **파라미터 효과**: FeatureRouter 활성화로 모델 파라미터 4.77M → 3.16M (34% 감소). 각 전문가가 관련 피처 축만 수신하므로 노이즈 입력이 제거된다.

### 현재 금융 파이프라인 설정

`configs/financial/pipeline.yaml`에 정의된 현재 설정:

```yaml
expert_basket:
  shared_experts:          # 7종 선택 (Pool 11종 중)
    - hgcn
    - perslay
    - deepfm
    - temporal_ensemble
    - lightgcn
    - causal
    - optimal_transport
  task_experts:            # 2종 선택
    - mlp
    - deepfm
  expert_configs:          # 개별 오버라이드
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

> **FeatureRouter 자동 생성**: `build_model()`이 `feature_groups.yaml`의 각 feature group에 선언된 `target_experts` 필드를 읽어 `FeatureRouter`를 자동으로 구성한다. 별도 코드 수정 없이 YAML 선언만으로 전문가별 입력 피처 집합이 결정된다.
>
> ```yaml
> # feature_groups.yaml 예시 — FeatureRouter 라우팅 선언
> - name: base_rfm
>   target_experts: [deepfm, causal, optimal_transport]   # 이 그룹 피처 → 해당 전문가에게만 전달
>
> - name: tda_topology
>   target_experts: [perslay]                              # TDA 피처 → PersLay 전용
>
> - name: temporal_features
>   target_experts: [temporal_ensemble]                   # 시계열 피처 → TemporalEnsemble 전용
>
> - name: hierarchy_embedding
>   target_experts: [hgcn]                                # 계층 피처 → HGCN 전용
> ```

미선택 전문가 (`mamba`, `autoint`, `xdeepfm`, `mlp`(shared에서)) 는 Pool에 등록되어 있지만 이 파이프라인에서는 비활성화 상태다. 도메인 전환 시 Basket만 교체하면 된다.

---

## 3. Feature Generator Pool/Basket (부분 구현, 확장 예정)

### 현재 상태

Feature Generator는 이미 Pool/Basket 구조가 부분적으로 존재한다:
- **Pool**: `FeatureGeneratorRegistry`에 Generator를 등록 (`core/feature/generator.py`)
- **Basket**: `feature_groups.yaml`이 어떤 Generator를 사용할지 선택 (Basket 역할)
- **Runtime**: `FeatureGroupPipeline`이 선택된 Generator를 실행

### Pool -- 현재 등록된 5개 Generator

| # | 등록 이름 | 파일 | 출력 | 설명 |
|---|-----------|------|------|------|
| 1 | `tda_extractor` | `core/feature/generators/tda.py` | 70D | 위상 데이터 분석 (Persistence Diagram) |
| 2 | `hmm_triple_mode` | `core/feature/generators/hmm.py` | 48D | HMM 상태 추정 (journey/lifecycle/behavior) |
| 3 | `hyperbolic_embedding` | `core/feature/generators/graph.py` | 20D | 쌍곡 공간 그래프 임베딩 |
| 4 | `temporal_pattern` | `core/feature/generators/temporal.py` | 가변 | 시계열 집계 + 주기 인코딩 |
| 5 | `multidisciplinary` | `core/feature/generators/multidisciplinary.py` | 24D | 화학동역학, 전염병확산, 간섭, 범죄패턴 |

### Basket -- feature_groups.yaml

`configs/financial/feature_groups.yaml`이 현재 Basket 역할을 수행한다. 15개 feature group을 정의하며, 각 group이 `generator` 필드로 Pool의 Generator를 참조한다:

```yaml
# generate 타입 -- Generator가 피처를 생성
- name: tda_topology
  group_type: generate
  generator: tda_extractor        # Pool에서 선택
  generator_params: { ... }

# transform 타입 -- 기존 컬럼을 변환 (Generator 불필요)
- name: base_rfm
  group_type: transform
  transformers: [quantile_transformer]
```

### 원본 프로젝트에서 추가 필요한 Generator

원본 On-Prem 시스템에는 현재 AWS 프로젝트에 미구현된 Generator가 있다:

| Generator | 역할 | 출력 예시 | 추가 시점 |
|-----------|------|-----------|-----------|
| `economics_extractor` | MPC, 소득 탄력성, 항상소득 가설 | 17D | feature_groups.yaml에 참조 있음, Generator 구현 필요 |
| `coldstart_generator` | 신규 고객 피처 (제한된 이력으로 추정) | 가변 | 신규 고객 데이터 확보 후 |
| `segmentation_generator` | 고객 세분화 피처 | 가변 | GMM 외 세분화 필요 시 |
| `snapshot_generator` | 월별 도메인 스냅샷 | 가변 | 배치 파이프라인 구축 후 |
| `model_feature_extractor` | 이전 모델 출력을 피처로 활용 | 27D | feature_groups.yaml에 참조 있음, Generator 구현 필요 |

### 확장 시 구현 방향

현재 구조를 그대로 활용하면 된다:

1. 새 Generator를 `core/feature/generators/` 에 추가하고 `@FeatureGeneratorRegistry.register`로 등록 (Pool 확장)
2. `feature_groups.yaml`에 해당 Generator를 참조하는 group 추가 (Basket 변경)
3. 코드 수정 없이 config만으로 파이프라인에 반영

```python
# 새 Generator 추가 예시
@FeatureGeneratorRegistry.register("economics_extractor")
class EconomicsGenerator(BaseFeatureGenerator):
    def generate(self, df: DataFrame) -> DataFrame:
        ...
```

### 구현 시점

실제 데이터로 파이프라인을 실행한 후, 누락된 Generator가 확인될 때 추가한다. 현재는 `feature_groups.yaml`에 Generator 이름만 참조되어 있고, 실제 Generator 클래스가 없는 것들이 있지만 파이프라인 미실행 상태에서는 문제없다.

---

## 4. Task Pool/Basket (설계 완료, 구현 예정)

### 현재 상태

| 구성 요소 | 현재 구현 | Pool/Basket 대응 |
|-----------|-----------|------------------|
| `TaskRegistry` | `core/pipeline/task.py` | Pool 역할 (태스크 타입 등록) |
| `pipeline.yaml` tasks | `configs/financial/pipeline.yaml` | Basket 역할 (16개 태스크 선택) |
| `AdaTTConfig.task_groups` | `core/model/ple/config.py` | Task Group 정의 (adaTT 전용) |

### 부족한 점

1. **Task Group이 adaTT config에만 존재**: `task_groups`가 `AdaTTConfig` 내부에 정의되어 있어 adaTT 전용. Loss weighting, monitoring 등 다른 모듈에서 참조 불가.
2. **active/disabled 토글 없음**: 태스크를 일시 비활성화하려면 `pipeline.yaml`에서 해당 항목을 삭제해야 함. 실험 시 불편.

### 구현 방향

```
현재:                                    목표:
┌───────────────┐                       ┌───────────────┐
│ pipeline.yaml │                       │ pipeline.yaml │
│   tasks: [...]│                       │   tasks: [...]│
│               │                       │               │
│ model:        │                       │ task_groups:  │ ◀── 1등 시민으로 승격
│   adatt:      │                       │   engagement: │
│     task_groups│                      │     members:  │
│               │                       │     active: T │
└───────────────┘                       │   lifecycle:  │
                                        │     members:  │
                                        │     active: T │
                                        └───────────────┘
                                              │
                                        adaTT, loss_weighting,
                                        monitoring 모두 참조
```

#### TaskGroupConfig 도입

```python
@dataclass
class TaskGroupConfig:
    """Task Group -- 1등 시민 config."""
    name: str
    members: List[str]       # 태스크 이름 목록
    active: bool = True      # 그룹 단위 활성/비활성
    intra_strength: float = 0.7
```

#### pipeline.yaml에 정의

```yaml
task_groups:
  engagement:
    members: [ctr, cvr, engagement, uplift]
    active: true
    intra_strength: 0.7

  lifecycle:
    members: [churn, retention, life_stage, ltv]
    active: true
    intra_strength: 0.7

  value:
    members: [balance_util, channel, timing]
    active: true
    intra_strength: 0.5

  consumption:
    members: [nba, spending_category, consumption_cycle,
              spending_bucket, merchant_affinity, brand_prediction]
    active: true
    intra_strength: 0.6
```

#### 공통 참조 구조

```
task_groups (pipeline.yaml)
    │
    ├── adaTT        -- intra/inter group 전이 강도
    ├── loss_weighting -- 그룹별 loss 가중치 전략
    └── monitoring    -- 그룹별 성능 대시보드
```

### 구현 로드맵

| 단계 | 내용 | 시점 |
|------|------|------|
| 1단계 | `TaskGroupConfig` 도입 + adaTT 연동 | 즉시 가능 |
| 2단계 | `active/disabled` 토글 + loss_weighting 연동 | 파이프라인 실행 후 |

---

## 5. 3계층 통합 비전

### 도메인 전환 시나리오: 금융 -> 이커머스

```
변경 대상          금융                          이커머스
─────────────────────────────────────────────────────────────
Expert Basket     hgcn, perslay, deepfm, ...    deepfm, autoint, xdeepfm, ...
Feature Basket    tda_topology, economics, ...  click_sequence, search_query, ...
Task Basket       ctr, churn, ltv, nba, ...     ctr, cvr, add_to_cart, ...
코드 수정          0                             0
```

### Config 구조 예시

```yaml
# configs/ecommerce/pipeline.yaml -- 이커머스 도메인
task_name: ecommerce_recommendation_ple

tasks:
  - {name: click, type: binary, loss: focal, loss_weight: 1.0, label_col: clicked}
  - {name: purchase, type: binary, loss: focal, loss_weight: 1.5, label_col: purchased}
  - {name: add_to_cart, type: binary, loss: focal, loss_weight: 1.0, label_col: added_cart}
  - {name: revenue, type: regression, loss: huber, loss_weight: 1.5, label_col: revenue_30d}

task_groups:
  engagement:
    members: [click, add_to_cart]
    active: true
  conversion:
    members: [purchase, revenue]
    active: true

model:
  architecture: ple
  expert_basket:
    shared_experts:
      - deepfm          # 피처 상호작용 (범용)
      - autoint          # Self-Attention (범용)
      - xdeepfm          # CIN (범용)
      - temporal_ensemble # 시퀀스 (클릭 이력)
    task_experts:
      - mlp
      - deepfm
    expert_configs:
      deepfm:
        embedding_dim: 16
        hidden_dims: [256, 128]
      temporal_ensemble: {}
```

```yaml
# configs/ecommerce/feature_groups.yaml -- 이커머스 피처
feature_groups:
  - name: user_profile
    group_type: transform
    columns: [age, gender, region, ...]
    output_dim: 20
    target_experts: [deepfm]

  - name: click_sequence
    group_type: generate
    generator: temporal_pattern
    generator_params: {window_days: 30, output_dim: 64}
    output_dim: 64
    target_experts: [temporal_ensemble]

  - name: item_embedding
    group_type: generate
    generator: hyperbolic_embedding
    generator_params: {hierarchy_sources: [category, brand], curvature: 1.0}
    output_dim: 32
    target_experts: [autoint]
```

핵심: **코드 한 줄 변경 없이**, config 파일만 `configs/financial/` -> `configs/ecommerce/`로 교체하면 전혀 다른 도메인의 추천 시스템이 구성된다.

---

## 6. 구현 로드맵

| Phase | 내용 | 상태 | 대상 파일 |
|-------|------|------|-----------|
| **Phase 1** | Expert Pool/Basket | 완료 | `core/model/experts/registry.py`, `core/model/ple/config.py`, `core/model/ple/experts.py` |
| **Phase 2** | Task Group 통합 (`TaskGroupConfig` + adaTT 연동) | 다음 | `core/model/ple/config.py`, `configs/financial/pipeline.yaml` |
| **Phase 3** | Feature Generator 확장 (미구현 Generator 추가) | 데이터 준비 후 | `core/feature/generators/` |
| **Phase 4** | Task active/disabled 토글, Feature Pool 정규화 | 파이프라인 실행 후 | `core/pipeline/config.py`, `core/feature/generator.py` |

```
Phase 1 (완료)          Phase 2 (다음)           Phase 3              Phase 4
─────────────────      ─────────────────       ─────────────────    ─────────────────
Expert Pool/Basket     TaskGroupConfig         Feature Generator    Task 토글
ExpertBasketConfig     pipeline.yaml에          economics, cold-    active/disabled
ExpertBasket           task_groups 정의          start 등 Generator   Feature Pool
CGCLayer 연동          adaTT/loss 공통 참조      Registry 등록        정규화
```

---

## 7. 설계 결정 기록

### 결정 1: Feature/Task Basket을 지금 정식으로 만들지 않는 이유

**현재 config이 이미 Basket 역할을 수행한다.**

| 축 | Pool | 현재 Basket 역할 | 정식 Basket 클래스 필요? |
|----|------|------------------|------------------------|
| Expert | `ExpertRegistry` (11종) | `ExpertBasketConfig` | 완료 |
| Feature | `FeatureGeneratorRegistry` (5종) | `feature_groups.yaml` | 아직 불필요 |
| Task | `TaskRegistry` | `pipeline.yaml` tasks | 아직 불필요 |

Feature와 Task는 현재 YAML config 자체가 Basket 역할을 충분히 수행하고 있다. 별도의 `FeatureBasketConfig`, `TaskBasketConfig` 같은 추상화 클래스를 추가하면:

- 파이프라인이 실행되지 않은 상태에서 **과잉 추상화**
- 실제 요구사항 없이 만든 추상화는 **잘못된 방향으로 고착될 위험**
- YAML config 교체만으로 도메인 전환이 가능한데 **불필요한 간접 계층 추가**

### 결정 2: 추가 시점 판단 기준

정식 Basket 클래스를 도입해야 하는 시점:

1. **실제 파이프라인 실행 후** 도메인 전환을 시도했을 때 config만으로 부족한 경우
2. **검증 로직이 필요할 때** -- 예: "선택한 Generator가 Pool에 등록되어 있는지" 런타임 검증 (Expert Basket이 이미 하는 것)
3. **Introspection이 필요할 때** -- 예: "현재 파이프라인에서 활성화된 피처 그룹 목록"을 프로그래밍 방식으로 조회

Expert Basket은 1, 2, 3 모두 필요했기 때문에 먼저 구현했다. Feature와 Task는 아직 1번 단계에 도달하지 않았다.

### 결정 3: Task Group을 adaTT에서 분리하는 이유

현재 `task_groups`는 `AdaTTConfig` 내부에 있어서 adaTT 모듈만 참조할 수 있다. 하지만 Task Group은 본질적으로 **태스크의 의미적 분류**이므로:

- `loss_weighting` -- 그룹별 loss 전략 차별화
- `monitoring` -- 그룹별 성능 대시보드
- `evaluation` -- 그룹별 메트릭 집계
- `active/disabled` -- 그룹 단위 실험 토글

이 모든 모듈이 공통으로 참조해야 한다. 따라서 `TaskGroupConfig`를 `pipeline.yaml` 최상위로 승격시키는 것이 Phase 2의 핵심이다.
