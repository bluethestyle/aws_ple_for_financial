# 03. Model Architecture — PLE, Expert Basket, CGC, adaTT, Logit Transfer, Evidential, SAE

## 10+ Stage 파이프라인에서의 위치

Model Architecture는 Stage 8 (PLETrainer)의 모델 구조를 정의한다.

---

## 전체 구조

```
5-Axis Features (Stage 4 출력)
    ↓ FeatureRouter (축별 분배)
┌─────────────────────────────────────────────────────────────────────────┐
│                              PLEModel                                   │
│                                                                         │
│  ┌──────────────────────────────────────┐  ┌──────────────────────────┐ │
│  │       SharedExpertPool (Basket)      │  │   TaskExpertPool         │ │
│  │       (Pool→Basket→CGC 3계층)        │  │   (Group별 배정)          │ │
│  │                                      │  │                          │ │
│  │  State축    ──▶ DeepFM              │  │  engagement: [deepfm]   │ │
│  │  Snapshot축 ──▶ PersLay, Causal, OT │  │  lifecycle:  [causal]   │ │
│  │  Timeseries축──▶ Temporal Ensemble  │  │  value:      [mlp]      │ │
│  │  Hierarchy축──▶ HGCN                │  │  consumption:[deepfm]   │ │
│  │  Item축    ──▶ LightGCN            │  │                          │ │
│  └─────────────────┬────────────────────┘  └───────────┬──────────────┘ │
│                    │                                    │                │
│                    ▼                                    ▼                │
│          ┌──────────────────────────────────────────────────┐           │
│          │   CGC Layer (Customized Gate Control)            │           │
│          │   Expert 출력을 태스크별 softmax 가중 결합        │           │
│          │   + CGC Attention (dim_normalize=True)           │           │
│          │   + Entropy regularization                       │           │
│          └────────────────────┬─────────────────────────────┘           │
│                               │                                         │
│                               ▼                                         │
│          ┌──────────────────────────────────────────────────┐           │
│          │   HMM Triple-Mode Projection                     │           │
│          │   journey  → value,consumption groups             │           │
│          │   lifecycle → lifecycle group                      │           │
│          │   behavior  → engagement group                     │           │
│          └────────────────────┬─────────────────────────────┘           │
│                               │                                         │
│                               ▼                                         │
│          ┌──────────────────────────────────────────────────┐           │
│          │   Multidisciplinary Per-Task Routing              │           │
│          │   24D → 4 subgroups (6D each) → 4 task groups    │           │
│          │   engagement←chemical, lifecycle←epidemic,        │           │
│          │   value←crime, consumption←interference           │           │
│          └────────────────────┬─────────────────────────────┘           │
│                               │                                         │
│                               ▼                                         │
│          ┌──────────────────────────────────────────────────┐           │
│          │   adaTT (Adaptive Task Transfer)                 │           │
│          │   intra-group 전이 (0.6-0.8 strength)            │           │
│          │   inter-group 전이 (0.3 strength)                │           │
│          └────────────────────┬─────────────────────────────┘           │
│                               │                                         │
│                               ▼                                         │
│          ┌──────────────────────────────────────────────────┐           │
│          │   Logit Transfer (3-method dispatch)             │           │
│          │   5 edges, strength=0.5                          │           │
│          │   output_concat / hidden_concat / residual       │           │
│          └────────────────────┬─────────────────────────────┘           │
│                               │                                         │
│                               ▼                                         │
│          ┌──────────────────────────────────────────────────┐           │
│          │   Task Towers (18개) — TowerRegistry             │           │
│          │   standard: binary/regression/multiclass          │           │
│          │   contrastive: brand_prediction 등                │           │
│          ├──────────────────────────────────────────────────┤           │
│          │   Evidential Deep Learning (config-gated)         │           │
│          │   regression → NIG(mu,v,alpha,beta) 불확실성       │           │
│          ├──────────────────────────────────────────────────┤           │
│          │   SAE Regularization (detached, config-gated)     │           │
│          │   shared expert concat → sparse autoencoder       │           │
│          ├──────────────────────────────────────────────────┤           │
│          │   Per-task loss: build_loss() + focal_alpha cal.  │           │
│          │   Uncertainty weighting: Kendall et al.           │           │
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
    def route(self, x: torch.Tensor, expert_name: str) -> torch.Tensor:
        """전체 피처 텐서에서 해당 Expert의 서브셋 추출."""
```

### Expert Pool — 11종 등록

| # | 등록 이름 | 파일 | 주요 Axis | 설명 |
|---|-----------|------|-----------|------|
| 1 | `mlp` | `core/model/experts/mlp.py` | State | 기본 MLP 베이스라인 |
| 2 | `deepfm` | `core/model/experts/deepfm.py` | State | FM + Deep 피처 상호작용 |
| 3 | `mamba` | `core/model/experts/mamba.py` | Timeseries | Selective SSM (S6, O(n)) |
| 4 | `temporal_ensemble` | `core/model/experts/temporal.py` | Timeseries | Mamba + PatchTST + LNN 앙상블 |
| 5 | `causal` | `core/model/experts/causal.py` | Snapshot | NOTEARS DAG 인과 구조 |
| 6 | `optimal_transport` | `core/model/experts/ot.py` | Snapshot | Sinkhorn 최적 수송 |
| 7 | `hgcn` | `core/model/experts/hgcn.py` | Hierarchy | 쌍곡 그래프 합성곱 |
| 8 | `perslay` | `core/model/experts/perslay.py` | Snapshot | 위상 데이터 분석 (TDA global) |
| 9 | `lightgcn` | `core/model/experts/lightgcn.py` | Item | 경량 그래프 합성곱 |
| 10 | `autoint` | `core/model/experts/autoint.py` | State | Self-Attention 상호작용 |
| 11 | `xdeepfm` | `core/model/experts/xdeepfm.py` | State | CIN + Deep |

### Dual-Registry 아키텍처

```
Expert Pool Registry (core.model.experts.registry.ExpertRegistry)
    └── AbstractExpert(input_dim, config)
    └── 11종 등록: deepfm, mlp, mamba, temporal_ensemble, ...

Expert PLE Registry (core.model.ple.experts.ExpertRegistry)
    └── BaseExpert(input_dim, output_dim, dropout)
    └── CGCLayer 기본 expert 생성용

Expert Basket (core.model.ple.experts.ExpertBasket)
    └── Pool Registry → Basket 선택 → CGCLayer.shared_experts에 주입
    └── Basket validates all expert names exist in pool
```

### 현재 Santander Basket 설정

```yaml
# configs/santander/pipeline.yaml
expert_basket:
  shared:
    - deepfm               # State축
    - temporal_ensemble     # Timeseries축
    - hgcn                  # Hierarchy축
    - perslay               # Snapshot축
    - causal                # Snapshot축
    - lightgcn              # Item축
    - optimal_transport     # Snapshot축
  task: [mlp]               # 범용 per-task
```

---

## CGC Layer + CGC Attention

### CGCLayer (core/model/ple/experts.py)

```python
class CGCLayer(nn.Module):
    """Customized Gate Control -- PLE 핵심 빌딩 블록.
    각 태스크에 대해 gating network가 shared + task-specific expert 출력을 결합.
    FeatureRouter 지원: 각 shared expert가 서로 다른 피처 서브셋을 받을 수 있음.
    """
```

### CGCAttention (dim_normalize=True)

```python
class CGCAttention(nn.Module):
    """Per-task attention over shared expert outputs.
    dim_normalize=True: expert 출력 차원이 다를 때 sqrt(mean_dim/dim) 스케일링.
    bias_high/bias_low: domain-relevant expert에 초기 바이어스 주입.
    entropy_regularization(): expert collapse 방지.
    """
```

Santander config에서 `cgc.dim_normalize: true` 활성화 — Expert 출력 차원 불균형 보정.

### Stacked PLE Layers

```yaml
ple:
  num_layers: 3           # 3개 CGC layer stacking
  extraction_dim: 64
  num_shared_experts: 7
  num_task_experts: 1
```

- Layer 0: Expert Basket의 이종 Expert (deepfm, hgcn, perslay, ...) + FeatureRouter
- Layer 1-2: 동종 MLP Expert (extraction_dim 입출력)

---

## HMM Triple-Mode Projection

3개 HMM 모드를 태스크 그룹별로 라우팅:

```python
_HMM_GROUP_MODE_MAP = {
    "engagement": "behavior",     # 행동 모드 → 반응/활성
    "lifecycle": "lifecycle",     # 생애주기 모드 → 이탈/유지
    "value": "journey",           # 여정 모드 → 가치/소비
    "consumption": "journey",     # 여정 모드 → 소비 패턴
}
```

각 모드별 16D → `_task_expert_output_dim` 으로 projection 후 tower input에 additive fusion.

---

## Multidisciplinary Per-Task Routing

24D multidisciplinary 피처를 4개 태스크 그룹에 6D씩 라우팅:

```yaml
multidisciplinary_routing:
  engagement: [0,1,2,3,4,5]        # chemical_kinetics 6D
  lifecycle: [6,7,8,9,10,11]        # epidemic_diffusion 6D
  value: [12,13,14,15,16,17]        # crime_pattern 6D
  consumption: [18,19,20,21,22,23]  # interference 6D
```

Per-group Linear projection → tower input에 additive fusion.

---

## adaTT (Adaptive Task Transfer)

### Task Group 정의

| Group | 태스크 | intra 강도 | inter 강도 |
|-------|--------|-----------|-----------|
| **engagement** | has_nba, engagement_score, next_mcc, top_mcc_shift | 0.8 | 0.3 |
| **lifecycle** | churn_signal, product_stability, tenure_stage, segment_prediction | 0.7 | 0.3 |
| **value** | spend_level, income_tier, mcc_diversity_trend | 0.6 | 0.3 |
| **consumption** | nba_primary, cross_sell_count, will_acquire_* (5개) | 0.7 | 0.3 |

### adaTT Config

```python
class AdaptiveTaskTransfer(nn.Module):
    """
    Intra-group: 같은 그룹 내 태스크 간 강한 전이 (0.6-0.8)
    Inter-group: 다른 그룹 간 약한 전이 (0.3)
    Negative transfer threshold: 성능 저하 시 전이 자동 차단
    EMA decay: 전이 가중치 안정화
    Warmup/freeze epochs: 초기 안정화
    """
```

---

## Logit Transfer (3-Method Dispatch)

### 5개 전이 엣지

| Source | Target | Method | 의미 |
|--------|--------|--------|------|
| `engagement_score` | `has_nba` | output_concat | 활성도 → 가입 (선행지표) |
| `has_nba` | `nba_primary` | output_concat | 가입 여부 → 어떤 상품 |
| `churn_signal` | `product_stability` | output_concat | 이탈 → 상품 안정성 |
| `spend_level` | `cross_sell_count` | output_concat | 소비수준 → 교차판매 |
| `next_mcc` | `nba_primary` | output_concat | 다음 업종 → 다음 상품 |

### 3가지 전이 방법

```python
def _build_logit_transfer(self) -> None:
    """
    residual:      source output → Linear → tower_dim, 잔차 합산
    output_concat: source output과 tower input concat → Linear → tower_dim
    hidden_concat: source pre-tower hidden과 tower input concat → Linear → tower_dim
    """
```

`logit_transfer_strength: 0.5` — 전이 비율.

---

## Evidential Deep Learning (Config-Gated)

`core/model/layers/evidential.py`

### 지원 분포

| Task Type | 분포 | 파라미터 | 불확실성 |
|-----------|------|---------|---------|
| Binary | Beta(alpha, beta) | alpha, beta | 1/(alpha+beta) |
| Multiclass | Dirichlet(alpha_1..K) | alpha_k | K/sum(alpha) |
| **Regression** | Normal-Inverse-Gamma | mu, v, alpha, beta | beta/(v*(alpha-1)) |

### Config

```yaml
evidential:
  enabled: true
  kl_lambda: 0.01
  annealing_epochs: 10    # KL 항 선형 어닐링
```

- regression 태스크에만 적용 (product_stability, cross_sell_count, engagement_score, mcc_diversity_trend)
- EvidentialLayer가 tower 출력 후 NIG 파라미터로 변환
- 에피스테믹 불확실성 정량화

---

## SAE Regularization (Detached, Config-Gated)

`core/model/layers/sae.py`

Anthropic-style Sparse Autoencoder:
- shared expert concatenated output에 적용
- **detached**: 메인 모델 gradient에 영향 없음 (분석용 sidecar)
- Tied weights, pre-bias centering, ReLU activation
- Loss = MSE(reconstruction) + l1_lambda * |latent|_1

```yaml
sae:
  enabled: true
  weight: 0.01            # total loss에 기여하는 비율
  expansion_factor: 4     # latent_dim = input_dim * 4
  l1_lambda: 0.001
```

---

## Per-Task Loss Dispatch

### build_loss() 팩토리

| loss type | nn.Module | 용도 | 태스크 타입 |
|-----------|-----------|------|-----------|
| `focal` | FocalLoss(alpha, gamma) | 불균형 이진 분류 | binary |
| `huber` | SmoothL1Loss(beta=delta) | 이상치 강건 회귀 | regression |
| `mse` | MSELoss | 기본 회귀 | regression |
| `ce` | CrossEntropyLoss(weight) | 다중 클래스 (auto class_weights) | multiclass |
| `infonce` | InfoNCELoss(temperature) | 대조 학습 | contrastive |

### Per-Task Focal Alpha Calibration

Binary 태스크의 focal_alpha는 양성 비율(positive rate)에 기반하여 calibrated:

```yaml
# 예: has_nba (2.98% positive)
- name: has_nba
  loss: focal
  loss_params:
    alpha: 0.90    # 높은 alpha → 양성 샘플 가중치 증가
    gamma: 2.0
```

---

## Uncertainty Weighting (Kendall et al., CVPR 2018)

```python
class UncertaintyWeighting(BaseLossWeighting):
    """
    loss_total = Sigma_k [ exp(-log_var_k) * L_k + log_var_k / 2 ]
    - log_var_k: 태스크 k의 learnable 파라미터
    - 불확실성 높은 태스크 → 자동 가중치 감소
    - 안정적인 태스크 → 자동 가중치 증가
    """
```

대안:
- `fixed`: pipeline.yaml의 loss_weight 고정 사용
- `gradnorm`: Chen et al. (ICML 2018) gradient 균형
- `dwa`: Dynamic Weight Averaging

---

## TowerRegistry — 태스크 타워 플러그인

```python
class TowerRegistry:
    """
    standard: MLP → sigmoid/softmax/None
    contrastive: MLP → L2-normalize (InfoNCE용)
    """
```

### Task Tower Config

```yaml
task_tower:
  default_dims: [64, 32]
  dropout: 0.1
```

태스크별 tower_type/tower_dims 오버라이드 가능 (`task_overrides`).

---

## FD-TVS Scoring + Constraint Engine

### FD-TVS Composite Score

```yaml
scoring:
  method: fd_tvs
  weights:
    has_nba: 0.20
    nba_primary: 0.30
    cross_sell_count: 0.15
    churn_signal: 0.15
    product_stability: 0.10
    engagement_score: 0.10
```

### DNA Modifier

```yaml
dna_modifier:
  segment_weights:
    "01-TOP": 1.3
    "02-PARTICULARES": 1.0
    "03-UNIVERSITARIO": 0.8
    "UNKNOWN": 0.7
```

### Constraint Engine

| 제약 | 설명 |
|------|------|
| **Fatigue** | 7일 내 최대 5회 메시지 |
| **Eligibility** | min_score > 0.05, max_churn_prob < 0.6 |
| **Owned Product** | prod_* prefix로 이미 보유한 상품 제외 |
| **Product Tier** | standard 3개월, growth 6개월, premium 12개월 최소 가입기간 |
| **Top-K Diversity** | MMR (lambda=0.5)로 다양성 보장 |

---

## 지식 증류 (PLE → LGBM)

```
PLE Teacher (GPU 학습)
    ↓ Soft Labels 생성 (temperature=5.0)
    ↓ S3에 저장
LGBM Student (CPU 학습)
    ↓ alpha=0.3 (30% hard + 70% soft)
    ↓ 경량 모델 저장
서빙 (실시간: LGBM ~5ms, 배치: PLE)
```

```yaml
distillation:
  temperature: 5.0
  alpha: 0.3
  lgbm:
    num_leaves: 127
    learning_rate: 0.05
    n_estimators: 500
```

---

## 모델 빌드 자동화 — PLEModel.__init__()

```python
class PLEModel(nn.Module):
    def __init__(self, config: PLEConfig):
        # 1. Expert Basket (optional) → Pool Registry에서 선택
        self._build_extraction_layers()       # Stacked CGC + FeatureRouter
        self._build_cgc_attention()           # Per-task attention (dim_normalize)
        self._build_task_experts()            # GroupTaskExpertBasket or MLP fallback
        self._build_hmm_projectors()          # 3 modes × projection
        self._build_adatt()                   # Adaptive Task Transfer
        self._build_logit_transfer()          # 3-method dispatch, 5 edges
        self._build_multidisciplinary_routing()  # 24D → 4 × 6D
        self._build_task_towers()             # TowerRegistry (standard/contrastive)
        self._build_evidential_layers()       # NIG for regression (config-gated)
        self._build_sae()                     # Sparse Autoencoder (config-gated)
        self._build_task_loss_fns()           # build_loss() per task
        self._build_loss_weighting()          # Uncertainty / GradNorm / DWA / Fixed
```

---

## PLEInput 데이터 컨테이너

```python
@dataclass
class PLEInput:
    features: torch.Tensor                    # (batch, input_dim)
    feature_group_ranges: Optional[Dict]      # group→(start,end) for routing
    cluster_ids: Optional[torch.Tensor]       # (batch,) cluster assignment
    cluster_probs: Optional[torch.Tensor]     # (batch, n_clusters) soft probs
    targets: Optional[Dict[str, torch.Tensor]]# {task_name: label}
    hyperbolic_features: Optional[torch.Tensor]  # (batch, 20) HGCN
    tda_features: Optional[torch.Tensor]         # (batch, 70) TDA
    collaborative_features: Optional[torch.Tensor]# (batch, 64) LightGCN
    hmm_journey: Optional[torch.Tensor]          # (batch, 16)
    hmm_lifecycle: Optional[torch.Tensor]        # (batch, 16)
    hmm_behavior: Optional[torch.Tensor]         # (batch, 16)
    event_sequences: Optional[torch.Tensor]      # (batch, seq_len, feat_dim)
    session_sequences: Optional[torch.Tensor]    # (batch, seq_len, feat_dim)
    multidisciplinary_features: Optional[torch.Tensor]  # (batch, 24)
    sample_weights: Optional[torch.Tensor]       # (batch,) importance weights
    # ... and more
```

---

## 현재 vs AWS — 모델 아키텍처 비교

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| Expert 종류 | 8종 하드코딩 | ExpertRegistry 11종 (동적) | 도메인별 Expert 선택적 활성화 |
| Expert 라우팅 | 암묵적 (코드 내 분기) | **5-Axis FeatureRouter** (명시적) | 피처→Expert 매핑 투명화 |
| Expert 선택 | 전체 사용 | **Pool→Basket→CGC 3계층** | Config-driven subset selection |
| CGC | 기본 | **dim_normalize=True** + entropy regularization | Expert 출력 균형 + collapse 방지 |
| 태스크 수 | 16개 고정 | **18개** (config 확장 가능) | Tier 5 txn-based NBA 추가 |
| 태스크 그룹 | 4그룹 하드코딩 | YAML task_groups (4 semantic groups) | adaTT + routing 공통 참조 |
| HMM Routing | 없음 | **Triple-Mode → task group 라우팅** | 모드별 특화 정보 전달 |
| Multidisciplinary | Flat 피처 | **24D → 4×6D per-task-group 라우팅** | 학제별 도메인 전문성 |
| Logit Transfer | 단일 방법 | **3-method dispatch** (output/hidden/residual) | 관계 유형별 최적화 |
| Evidential | 없음 | **Evidential Deep Learning (config-gated)** | 에피스테믹 불확실성 정량화 |
| SAE | 없음 | **Sparse Autoencoder (detached, config-gated)** | 기계적 해석 가능성 |
| Loss 함수 | 코드 내 하드코딩 | **build_loss() 팩토리 + focal_alpha calibrated** | config 선언적, 양성비율 반영 |
| Scoring | 없음 | **FD-TVS + DNA modifier + constraints** | 규제 준수 추천 |
| Tower | 단일 MLP | **TowerRegistry** (standard/contrastive) | 태스크 유형별 최적 tower |
| 입력 차원 | 734D 하드코딩 | feature_schema.input_dim 동적 | 피처 변경 시 자동 반영 |
