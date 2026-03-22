# 02. Feature Engineering — 5-Axis 피처 매핑, Generator, 정규화

## 9-Stage 파이프라인에서의 위치

Feature Engineering은 Stage 5~6을 담당한다:

```
Stage 5: Feature Engineering per axis (TDA, HMM, Mamba, Graph, LightGCN 등)
Stage 6: Feature Integration + Normalization (power-law auto-detect → log1p+raw → StandardScaler)
```

---

## FeatureGroupPipeline — 피처 엔지니어링 오케스트레이터 (Step 12-13)

`core/pipeline/features.py`의 `FeatureGroupPipeline`이 Stage 5 피처 엔지니어링의 주 진입점이다. 기존 `FeaturePipelineBuilder`는 개별 generator 호출에 사용되지만, 전체 오케스트레이션은 `FeatureGroupPipeline`이 담당한다.

### FeatureGroupPipeline vs Adapter 역할 분리

| 역할 | DataAdapter (Stage 1) | FeatureGroupPipeline (Stage 5) |
|------|----------------------|-------------------------------|
| DuckDB 집계 | O (24M → 2,000 user) | X |
| RFM 기본 피처 | O (base aggregation) | X |
| TDA (global/local) | X | O — Generator 호출 |
| HMM 상태 추정 | X | O — Generator 호출 |
| Mamba SSM | X | O — Generator 호출 |
| Graph 임베딩 | X | O — Generator 호출 |
| GMM 클러스터링 | X | O — Generator 호출 |

### TDA Global/Local 구분

TDA 피처는 두 개의 별도 Generator 호출로 분리된다:

- **TDA Global** (Snapshot 축, 36D+10D): 12개월 장기 윈도우 Persistence Diagram
- **TDA Local** (Timeseries 축, 24D): 90일 단기 윈도우 Persistence Diagram

### cuML Fallback

scikit-learn transformer가 대규모 데이터에서 느린 경우, cuML이 설치되어 있으면 자동으로 GPU 가속 경로를 사용한다:

```
cuML StandardScaler  →  sklearn StandardScaler  (fallback)
cuML PCA             →  sklearn PCA             (fallback)
```

---

## 5-Axis Feature Group 매핑 (Santander 데이터)

### 전체 구조

```
5-Axis Classification
    ↓
    ├── State (정적)
    │   ├── base_rfm (34D) — 인구통계, RFM 프로필
    │   ├── base_category (64D) — 업종별 소비 패턴
    │   ├── multi_source (91D) — 예금/카드/멤버십/투자/디지털
    │   ├── extended_source (84D) — 보험/상담/캠페인/해외결제
    │   ├── economics (17D) — MPC, 소득 탄력성, 항상소득 가설
    │   └── multidisciplinary (24D) — 화학동역학/전염병확산/간섭/범죄패턴
    │
    ├── Snapshot (장기)
    │   ├── tda_topology [global] (36D+10D) — 12개월 Persistence Diagram + Phase Transition
    │   ├── hmm_states (48D) — HMM Triple-Mode 상태 전이 (journey/lifecycle/behavior)
    │   ├── gmm_clustering (22D) — 장기 고객 세그먼트 (20 soft + entropy + dominant)
    │   ├── model_derived [HMM/LNN] (23D) — HMM summary(5) + LNN(18)
    │   └── 상품 트렌드 — 월별 상품 보유 변화 패턴
    │
    ├── Timeseries (단기)
    │   ├── tda_topology [local] (24D) — 90일 Persistence Diagram (short window)
    │   ├── base_temporal (60D) — 시계열 집계 + 주기 인코딩
    │   ├── mamba_temporal (50D) — Mamba SSM 시퀀스 출력
    │   ├── base_txn_stats (80D) — 최근 기간 거래 통계
    │   └── PatchTST (미래) — Patch 기반 시계열 트랜스포머
    │
    ├── Hierarchy (구조)
    │   ├── merchant_hierarchy (21D) — MCC L1(4) + L2(4) + 브랜드 임베딩(8) + 통계(4) + radius(1)
    │   ├── graph_embeddings (20D) — Poincare 쌍곡 임베딩 (MCC/상품/지역 계층)
    │   └── product_hierarchy — 상품 카테고리 트리 (gotothemoon 분석 기반)
    │
    └── Item (관계)
        ├── customer_product_bipartite — 고객×상품 상호작용 그래프
        └── lightgcn_collaborative (64D) — LightGCN 협업 필터링 임베딩
```

### 차원 요약

| Axis | Feature Groups | Total Dim | Target Experts |
|------|---------------|-----------|----------------|
| **State** | 6 groups | ~314D | DeepFM, MLP, AutoInt |
| **Snapshot** | 4-5 groups | ~139D | PersLay, Causal, OT |
| **Timeseries** | 4-5 groups | ~214D | Temporal Ensemble, Mamba |
| **Hierarchy** | 2-3 groups | ~41D | HGCN |
| **Item** | 2 groups | ~64D | LightGCN |
| **합계** | | ~772D+ | 11종 Expert Pool |

---

## Axis별 Feature Generator 상세

### State 축 — 정적 속성

State 축 피처는 기존 테이블에서 추출하는 **transform** 타입이 대부분이다.

```yaml
# State 축 feature groups
- name: base_rfm
  axis: state
  group_type: transform
  output_dim: 34
  target_experts: [deepfm, mlp]
  description: "인구통계 + RFM 프로필 (가입일, 나이, 거래빈도/금액/최근성)"

- name: economics
  axis: state
  group_type: generate
  generator: economics
  output_dim: 17
  target_experts: [deepfm]
  description: "MPC(한계소비성향), 소득 탄력성, 항상소득 가설 기반 피처"
```

### Snapshot 축 — 장기 패턴

```yaml
# Snapshot 축 — TDA Global (long window)
- name: tda_global
  axis: snapshot
  group_type: generate
  generator: tda
  generator_params:
    window_days: 365          # 12개월 장기 윈도우
    output_dim: 46            # long(36) + phase_transition(10)
    method: persistence_diagram
  target_experts: [perslay]
  description: "12개월 거래 위상 구조 — Persistence Diagram의 장기 토폴로지"

# Snapshot 축 — HMM 상태 전이
- name: hmm_states
  axis: snapshot
  group_type: generate
  generator: hmm
  generator_params:
    modes: [journey, lifecycle, behavior]
    state_dim: 16
  output_dim: 48
  target_experts: [temporal_ensemble]
  description: "HMM Triple-Mode 상태 추정 — 고객 여정/생애주기/행동 패턴"

# Snapshot 축 — 상품 트렌드 (gotothemoon 분석 기반)
- name: product_trend
  axis: snapshot
  group_type: generate
  generator: product_trend
  generator_params:
    lookback_months: 12
    product_count: 24         # Santander 24개 상품
  output_dim: 48              # 24 상품 × 2 (보유변화 + 보유기간)
  target_experts: [deepfm]
  description: "월별 상품 보유 변화 패턴 (gotothemoon ind_actividad_cliente 분석)"
```

### Timeseries 축 — 단기 시퀀스

```yaml
# Timeseries 축 — TDA Local (short window)
- name: tda_local
  axis: timeseries
  group_type: generate
  generator: tda
  generator_params:
    window_days: 90           # 90일 단기 윈도우
    output_dim: 24
    method: persistence_diagram
  target_experts: [temporal_ensemble]
  description: "최근 90일 거래 위상 — 단기 행동 변화 감지"

# Timeseries 축 — Mamba SSM
- name: mamba_temporal
  axis: timeseries
  group_type: generate
  generator: mamba
  generator_params:
    d_model: 128
    d_state: 16
    expand: 2
    output_dim: 50
  target_experts: [temporal_ensemble]
  description: "Mamba Selective State Space Model — O(n) 시퀀스 처리"

# Timeseries 축 — PatchTST (향후)
- name: patchtst_temporal
  axis: timeseries
  group_type: generate
  generator: patchtst
  generator_params:
    patch_length: 16
    stride: 8
    d_model: 128
    num_heads: 8
    output_dim: 64
  target_experts: [temporal_ensemble]
  enabled: false              # 구현 후 활성화
  description: "Patch 기반 시계열 트랜스포머 — 장거리 의존성 포착"
```

### Hierarchy 축 — 구조적 계층

```yaml
# Hierarchy 축 — Poincare 임베딩
- name: graph_embeddings
  axis: hierarchy
  group_type: generate
  generator: graph
  generator_params:
    hierarchy_sources: [mcc, product, region]
    curvature: 1.0
    mcc_dim: 8
    product_dim: 8
    region_dim: 4
  output_dim: 20
  target_experts: [hgcn]
  description: "쌍곡 공간 임베딩 — MCC/상품/지역 계층 구조 보존"

# Hierarchy 축 — MCC L1/L2
- name: merchant_hierarchy
  axis: hierarchy
  group_type: generate
  generator: merchant_hierarchy
  generator_params:
    mcc_level1_dim: 4
    mcc_level2_dim: 4
    brand_embed_dim: 8
    agg_stats_dim: 4
    hierarchy_radius_dim: 1
  output_dim: 21
  target_experts: [hgcn]
  description: "가맹점 MCC 코드 계층 + 브랜드 임베딩 + 소비 반경"

# Hierarchy 축 — 상품 카테고리 트리 (gotothemoon 분석)
- name: product_hierarchy
  axis: hierarchy
  group_type: generate
  generator: product_hierarchy
  generator_params:
    # gotothemoon의 24개 금융 상품을 계층 트리로 구성
    # Level 1: 예금/대출/투자/보험/카드/기타
    # Level 2: 세부 상품 (ahor_fin, cco_fin, ...)
    hierarchy_depth: 2
    embedding_dim: 16
  output_dim: 16
  target_experts: [hgcn]
  description: "금융 상품 카테고리 계층 임베딩 (Santander 24개 상품)"
```

### Item 축 — 관계적 상호작용

```yaml
# Item 축 — 고객×상품 Bipartite Graph
- name: customer_product_bipartite
  axis: item
  group_type: generate
  generator: bipartite_graph
  generator_params:
    # gotothemoon의 ind_* 컬럼 (24개 상품 보유 여부)에서 bipartite 구성
    # 고객 노드: customer_id
    # 상품 노드: 24개 금융 상품 (ahor_fin, cco_fin, ...)
    # 엣지: 보유 여부 (0/1) + 보유 기간 weight
    customer_col: customer_id_idx     # Stage 4에서 INT32로 변환된 ID
    product_cols_prefix: "ind_"       # ind_ahor_fin, ind_cco_fin, ...
    edge_weight: holding_duration     # 보유 기간을 엣지 가중치로
  output_dim: 64
  target_experts: [lightgcn]
  description: "고객-상품 bipartite graph → LightGCN 협업 필터링"

# Item 축 — LightGCN 협업 필터링 임베딩
- name: lightgcn_collab
  axis: item
  group_type: generate
  generator: lightgcn
  generator_params:
    num_layers: 3
    embedding_dim: 64
    # 입력: customer_product_bipartite에서 생성된 graph
    graph_source: customer_product_bipartite
  output_dim: 64
  target_experts: [lightgcn]
  description: "LightGCN 경량 그래프 합성곱 — 유사 고객의 상품 선호도 전파"
```

---

## Feature Generator Registry

### 현재 등록된 Generator (Pool)

| # | 등록 이름 | 파일 | Axis | Output | 설명 |
|---|-----------|------|------|--------|------|
| 1 | `tda` | `core/feature/generators/tda.py` | Snapshot/Timeseries | 70D | Persistence Diagram (short+long) |
| 2 | `hmm` | `core/feature/generators/hmm.py` | Snapshot | 48D | HMM Triple-Mode 상태 |
| 3 | `graph` | `core/feature/generators/graph.py` | Hierarchy | 20D | Poincare 쌍곡 임베딩 |
| 4 | `temporal_pattern` | `core/feature/generators/temporal.py` | Timeseries | 가변 | 시계열 집계 + 주기 인코딩 |
| 5 | `multidisciplinary` | `core/feature/generators/multidisciplinary.py` | State | 24D | 화학/전염병/간섭/범죄 |
| 6 | `mamba` | (등록 예정) | Timeseries | 50D | Mamba SSM |
| 7 | `economics` | (등록 예정) | State | 17D | MPC, 소득 탄력성 |
| 8 | `merchant_hierarchy` | (등록 예정) | Hierarchy | 21D | MCC L1/L2 + 브랜드 |
| 9 | `gmm` | (등록 예정) | Snapshot | 22D | GMM 클러스터링 |
| 10 | `model_features` | (등록 예정) | Snapshot | 27D | HMM summary + Bandit + LNN |
| 11 | `product_trend` | (신규) | Snapshot | 48D | 월별 상품 보유 변화 |
| 12 | `product_hierarchy` | (신규) | Hierarchy | 16D | 상품 카테고리 트리 |
| 13 | `bipartite_graph` | (신규) | Item | 64D | 고객×상품 bipartite |
| 14 | `lightgcn` | (신규) | Item | 64D | LightGCN 협업 필터링 |
| 15 | `patchtst` | (신규, 향후) | Timeseries | 64D | Patch 시계열 트랜스포머 |

### Generator 추가 방법

```python
# Pool 확장 — 새 Generator 등록
@FeatureGeneratorRegistry.register("product_trend")
class ProductTrendGenerator(BaseFeatureGenerator):
    """월별 상품 보유 변화 패턴 (gotothemoon ind_actividad_cliente 분석)"""

    def generate(self, df: DataFrame, params: dict) -> DataFrame:
        # 24개 상품의 월별 보유 변화 계산
        ...

# Basket 변경 — feature_groups.yaml에 추가
# → 코드 수정 0
```

---

## Stage 6: Feature Integration + Normalization

### Power-law 자동 감지 + StandardScaler

Phase 1 리팩토링의 핵심 변경: QuantileTransform + Raw Power-Law 이중 파이프라인 → **StandardScaler 단일 파이프라인**으로 교체.

```
Numeric Features (all axes merged)
    ↓
[Skewness 검사]─── skewness > 2.0 ──▶ log1p 변환 (power-law 자동 감지)
    ↓                                      ↓
    │ (원본 유지)                      (log1p 변환본)
    ↓                                      ↓
    └──── log1p + raw 병렬 결합 ──────────┘
                   ↓
          StandardScaler (z-score: mean=0, std=1)
                   ↓
        scaler_params.json 저장 (mean, std, log1p_cols)
```

### 구현

```python
# core/feature/normalizer.py
class PowerLawAwareScaler:
    """
    Power-law 자동 감지 + StandardScaler.

    동작:
    1. fit 시 각 컬럼의 skewness 계산
    2. skewness > 2.0인 컬럼에 log1p 변환 (원본은 유지)
    3. log1p 변환본 + 원본을 병렬로 결합
    4. 전체에 StandardScaler 적용 (z-score)
    5. get_params() → scaler_params.json 저장

    cuPY 가속: cupy 설치 시 skewness/mean/std 계산을 GPU에서 수행.
    """

    def fit(self, df):
        # Power-law 자동 감지
        self._log1p_cols = [
            c for c in self.numeric_cols if df[c].skew() > 2.0
        ]
        # log1p 변환 + 원본 병렬
        df_extended = self._extend_with_log1p(df)
        # StandardScaler fit
        self.mean = df_extended.mean()
        self.std = df_extended.std()
        return self

    def transform(self, df):
        df_extended = self._extend_with_log1p(df)
        return (df_extended - self.mean) / (self.std + 1e-8)

    def _extend_with_log1p(self, df):
        """log1p 컬럼을 '{col}_log1p'로 추가, 원본은 유지."""
        result = df.copy()
        for col in self._log1p_cols:
            result[f"{col}_log1p"] = np.log1p(result[col])
        return result

    def get_params(self) -> dict:
        return {
            "mean": self.mean.to_dict(),
            "std": self.std.to_dict(),
            "log1p_cols": self._log1p_cols,
        }
```

### cuPY 가속

```python
# GPU 가속 경로 (cuPY 설치 시)
if HAS_CUPY:
    import cupy as cp
    skewness = cp.asnumpy(cp.array(df.values).mean(axis=0))  # GPU에서 계산
    mean = cp.asnumpy(cp.array(df.values).mean(axis=0))
    std = cp.asnumpy(cp.array(df.values).std(axis=0))
```

---

## SageMaker Processing에서 실행

```
Step Functions
    ↓
SageMaker Processing Job
    ├── 입력: s3://bucket/data/encrypted/       (Stage 4 출력)
    ├── 코드: core/feature/pipeline_builder.py
    ├── config: configs/financial/feature_groups.yaml
    ├── 엔진: DuckDB (인프로세스, 메모리 효율적)
    ├── GPU: cuPY/cuDF optional (ml.g4dn.xlarge)
    └── 출력: s3://bucket/features/v{version}/
         ├── train.parquet
         ├── val.parquet
         ├── test.parquet
         ├── schema.json              ← 실제 생성된 피처 메타데이터 (axis별 dim 포함)
         ├── scaler_params.json       ← StandardScaler mean/std + log1p 컬럼 목록
         ├── label_transforms.json    ← 회귀 레이블 clip/log1p 파라미터
         ├── item_universe/           ← Stage 8 출력
         │   ├── product_hierarchy.json
         │   └── customer_product_graph.parquet
         └── transformers/            ← fit된 scaler/encoder pickle
             ├── transformer_00_PowerLawAwareScaler.pkl
             └── transformer_01_LabelEncoder.pkl
```

---

## 피처 차원 관리 — 동적 계산

```python
# core/feature/schema.py
class FeatureSchema:
    """
    피처 차원을 하드코딩하지 않고 config + axis 분류에서 동적으로 계산.

    On-Prem에서 644D, 734D 등이 코드 곳곳에 하드코딩된 문제를 해결.
    """

    @property
    def input_dim(self) -> int:
        """전체 모델 입력 차원."""
        return sum(g.output_dim for g in self.groups if g.enabled)

    @property
    def axis_dims(self) -> dict[str, int]:
        """축별 피처 차원."""
        dims = defaultdict(int)
        for g in self.groups:
            if g.enabled:
                dims[g.axis] += g.output_dim
        return dict(dims)
        # → {"state": 314, "snapshot": 139, "timeseries": 214, "hierarchy": 41, "item": 64}

    @property
    def expert_input_dims(self) -> dict[str, int]:
        """Expert별 입력 차원 (target_experts 매핑 기반)."""
        dims = defaultdict(int)
        for g in self.groups:
            if g.enabled:
                for expert in g.target_experts:
                    dims[expert] += g.output_dim
        return dict(dims)
```

---

## Item Universe + Product Hierarchy (Stage 8)

### Item Universe 개념

gotothemoon 원본 프로젝트 분석에서 도출된 개념. Santander 데이터의 24개 금융 상품(`ind_*` 컬럼)을 하나의 "Item Universe"로 정의하고, 고객×상품 관계를 bipartite graph로 모델링한다.

```
Item Universe (24 금융 상품)
┌──────────────────────────────────────────────────┐
│ 예금 계열                                         │
│   ahor_fin, aval_fin, cco_fin, cder_fin           │
│ 대출 계열                                         │
│   hip_fin, pres_fin, reca_fin, valo_fin           │
│ 투자 계열                                         │
│   fond_fin, plan_fin, viv_fin                     │
│ 카드 계열                                         │
│   tjcr_fin, ctma_fin, ctop_fin, ctpp_fin          │
│ 디지털/기타                                       │
│   ecue_fin, dela_fin, deme_fin, deco_fin          │
│   nom_pens, nomina, recibo, direct_debit          │
└──────────────────────────────────────────────────┘
```

### Product Hierarchy Config

```json
{
  "hierarchy": {
    "level_1": {
      "savings": ["ahor_fin", "aval_fin", "cco_fin"],
      "loans": ["hip_fin", "pres_fin", "reca_fin"],
      "investments": ["fond_fin", "plan_fin", "viv_fin", "valo_fin"],
      "cards": ["tjcr_fin", "ctma_fin", "ctop_fin", "ctpp_fin"],
      "digital": ["ecue_fin", "dela_fin", "deme_fin", "deco_fin"],
      "recurring": ["nom_pens", "nomina", "recibo", "direct_debit"]
    },
    "cross_sell_rules": {
      "savings → cards": 0.8,
      "loans → insurance": 0.6,
      "cards → digital": 0.7
    }
  }
}
```

이 계층 정보는 Hierarchy 축의 `product_hierarchy` Generator와 Item 축의 `bipartite_graph` Generator 양쪽에서 참조된다.

---

## 현재 vs AWS — 피처 엔지니어링 비교

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| 피처 분류 | 없음 (flat 목록) | **5-Axis** (State/Snapshot/Timeseries/Hierarchy/Item) | Expert 라우팅 명시적 기반 |
| 피처 정의 | 코드 내 하드코딩 | YAML 선언형 (feature_groups.yaml) | 코드 변경 없이 피처 추가/삭제 |
| 차원 관리 | 644D, 734D 하드코딩 | `FeatureSchema.input_dim` 동적 계산 + axis_dims | 피처 변경 시 자동 반영 |
| 정규화 | QuantileTransform + Raw Power-Law | **StandardScaler + power-law 자동 감지** (skew>2.0 → log1p+raw 병렬) | 단일 경로, 분포 보존 |
| TDA | 단일 윈도우 | **Global(365일)/Local(90일) 분리** → Snapshot/Timeseries 축 | 장기/단기 위상 분리 |
| HMM | 코드에 직접 구현 | Generator Registry (hmm_triple_mode) | 선택적 활성화 |
| Graph | HGCN만 | **Poincare + LightGCN** (Hierarchy + Item 축) | 계층+협업 분리 |
| Item Universe | 없음 | **고객×상품 bipartite graph** (gotothemoon 분석) | 상품 추천 핵심 |
| 도메인 피처 | 코드에 직접 구현 | Plugin Registry (선택적) | 도메인 무관하게 on/off |
| 피처 버전 | 없음 | features/v{version}/ | 재현성, 롤백 |
| GPU 가속 | 없음 | cuPY (TDA, scaler), cuDF (preprocessing) | 5-10x 가속 |
| 서빙 일관성 | scaler pickle 수동 관리 | transformers/ + scaler_params.json 자동 저장 | 학습-추론 불일치 방지 |
