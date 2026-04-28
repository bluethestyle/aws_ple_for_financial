# 02. Feature Engineering — 5-Axis 피처 매핑, Generator, 정규화, LabelDeriver

## 10+ Stage 파이프라인에서의 위치

Feature Engineering은 Stage 4 ~ Stage 6을 담당한다:

```
Stage 4:   FeatureGroupPipeline + Normalization (per axis generators + PowerLawAwareScaler)
Stage 5:   LabelDeriver (13 tasks, config-driven derivation)
Stage 5.5: LeakageValidator (sequence/correlation/product/temporal)
Stage 6:   SequenceBuilder (time-based + sliding window → sequences.npy, seq_lengths.npy)
```

---

## FeatureGroupPipeline — 피처 엔지니어링 오케스트레이터

`core/pipeline/features.py`의 `FeatureGroupPipeline`이 Stage 4 피처 엔지니어링의 주 진입점이다.

### Expert Routing — Feature Group 단위 원칙 (2026-04-11)

**expert_routing은 개별 컬럼 이름이 아닌 feature group 이름을 기준으로 해야 한다.** Phase 0의 3-stage 정규화는 power-law 컬럼마다 `_log` 접미사 복사본을 추가하므로, 컬럼 이름 기반 routing은 이 재배열 이후 잘못된 슬라이스를 Expert에 전달한다.

`feature_group_ranges`는 연속된 컬럼 블록으로 저장해야 한다 (contiguous blocks). min~max index 방식은 정규화 후 삽입된 `_log` 컬럼이 블록 경계를 넘어갈 때 인접 group의 컬럼을 포함하는 오류를 일으킨다. 비연속 매칭 발생 시 가장 긴 연속 블록을 사용하거나 컬럼 이름 리스트로 range를 저장한다.

### FeatureGroupPipeline vs Adapter 역할 분리

| 역할 | DataAdapter (Stage 1) | FeatureGroupPipeline (Stage 4) |
|------|----------------------|-------------------------------|
| DuckDB 집계 | O (24M → 2,000 user) | X |
| RFM 기본 피처 | O (base aggregation) | X |
| TDA (global/local) | X | O — Generator 호출 |
| HMM 상태 추정 | X | O — Generator 호출 |
| Mamba SSM | X | O — Generator 호출 |
| Graph 임베딩 | X | O — Generator 호출 |
| GMM 클러스터링 | X | O — Generator 호출 |
| Normalization | X | O — PowerLawAwareScaler |

### TDA Global/Local 구분

TDA 피처는 두 개의 별도 Generator 호출로 분리된다:

- **TDA Global** (Snapshot 축, 36D+10D): 12개월 장기 윈도우 Persistence Diagram
- **TDA Local** (Timeseries 축, 24D): 90일 단기 윈도우 Persistence Diagram

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
    │   └── PatchTST — Patch 기반 시계열 트랜스포머
    │
    ├── Hierarchy (구조)
    │   ├── merchant_hierarchy (27D) — MCC 계층의 Poincaré 쌍곡 임베딩 (27D). HGCN 라우팅
    │   ├── graph_embeddings (20D) — Poincare 쌍곡 임베딩 (MCC/상품/지역 계층)
    │   └── product_hierarchy — 상품 카테고리 트리 (24개 금융 상품)
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

```yaml
- name: base_rfm
  axis: state
  group_type: transform
  output_dim: 34
  target_experts: [deepfm, mlp]

- name: economics
  axis: state
  group_type: generate
  generator: economics
  output_dim: 17
  target_experts: [deepfm]

- name: multidisciplinary
  axis: state
  group_type: generate
  generator: multidisciplinary
  output_dim: 24
  target_experts: [deepfm]
  # 4 subgroups (6D each): chemical_kinetics, epidemic, crime, interference
  # Per-task-group routing: engagement←chemical, lifecycle←epidemic, etc.
```

### Snapshot 축 — 장기 패턴

```yaml
- name: tda_global
  axis: snapshot
  generator: tda
  generator_params:
    window_days: 365
    output_dim: 46        # long(36) + phase_transition(10)
  target_experts: [perslay]

- name: hmm_states
  axis: snapshot
  generator: hmm
  generator_params:
    modes: [journey, lifecycle, behavior]   # Triple-Mode
    state_dim: 16
  output_dim: 48          # 3 modes x 16D
  target_experts: [temporal_ensemble]
  # HMM Triple-Mode → task group routing:
  #   journey  → value, consumption groups
  #   lifecycle → lifecycle group
  #   behavior  → engagement group

- name: gmm_clustering
  axis: snapshot
  generator: gmm
  generator_params:
    n_components: 20       # K=20 clusters
    covariance_type: full
    # BIC-based validation: warns if K is suboptimal
    # Cold-start fallback: uniform distribution for small data
  output_dim: 22           # K soft probs(20) + entropy(1) + dominant(1)
  target_experts: [deepfm, mlp]
  # NOTE: GMM soft labels (not KMeans) — posterior probabilities + Shannon entropy
```

### Timeseries 축 — 단기 시퀀스

```yaml
- name: tda_local
  axis: timeseries
  generator: tda
  generator_params:
    window_days: 90
    output_dim: 24
  target_experts: [temporal_ensemble]

- name: mamba_temporal
  axis: timeseries
  generator: mamba
  generator_params:
    d_model: 128
    d_state: 16
    expand: 2
    output_dim: 50
  target_experts: [temporal_ensemble]
```

### Hierarchy 축 — 구조적 계층

```yaml
- name: graph_embeddings
  axis: hierarchy
  generator: graph
  generator_params:
    hierarchy_sources: [mcc, product, region]
    curvature: 1.0
  output_dim: 20
  target_experts: [hgcn]

- name: merchant_hierarchy
  axis: hierarchy
  generator: merchant_hierarchy
  generator_params:
    mcc_hierarchy_path: configs/mcc_hierarchy.yaml   # ISO 18245, L1/L2/L3
    n_svd_components: 8       # Brand SVD embedding dim
    poincare_dim: 27          # Poincaré hyperbolic embedding of MCC hierarchy
  output_dim: 27              # Poincaré 27D embedding (쌍곡 공간, MCC 계층 구조 인코딩)
  target_experts: [hgcn]
  # MCC Hierarchy: 10 L1 groups, ~30 L2 subcategories, 109 L3 codes in dataset
  # NOTE: txn_mcc_seq는 generator 호출 전에 마지막 1원소 제거 (next_mcc 누수 방지)

- name: product_hierarchy
  axis: hierarchy
  generator: product_hierarchy
  generator_params:
    hierarchy_depth: 2        # Level 1: 예금/대출/투자/보험/카드/기타
    embedding_dim: 16         # Level 2: 24개 세부 상품
  output_dim: 16
  target_experts: [hgcn]
```

### Item 축 — 관계적 상호작용

```yaml
- name: customer_product_bipartite
  axis: item
  generator: bipartite_graph
  generator_params:
    customer_col: customer_id_idx
    product_cols_prefix: "ind_"
    edge_weight: holding_duration
  output_dim: 64
  target_experts: [lightgcn]

- name: lightgcn_collab
  axis: item
  generator: lightgcn
  generator_params:
    num_layers: 3
    embedding_dim: 64
  output_dim: 64
  target_experts: [lightgcn]
```

---

## Feature Generator Registry

### 8 구현 완료 Generator

모든 Generator는 `core/feature/generators/` 디렉토리에 구현되어 ���으며, `FeatureGeneratorRegistry`에 ���록된다. cuDF primary / pandas fallback 패턴을 따른다. Generator output은 cuDF DataFrame 또는 pandas DataFrame이다.

| # | 등록 이름 | 파일 | Axis | Output | GPU 가속 | 설명 |
|---|-----------|------|------|--------|---------|------|
| 1 | `tda` | `core/feature/generators/tda.py` | Snapshot/Timeseries | 70D | cuPY + ripser | Persistence Diagram (short+long) |
| 2 | `hmm` | `core/feature/generators/hmm.py` | Snapshot | 48D | hmmlearn | HMM Triple-Mode 상태 |
| 3 | `mamba` | `core/feature/generators/mamba.py` | Timeseries | 50D | GPU (mamba-ssm) | Mamba SSM 시퀀스 |
| 4 | `graph` | `core/feature/generators/graph.py` | Hierarchy | 20D | - | Poincare 쌍곡 임베딩 |
| 5 | `gmm` | `core/feature/generators/gmm.py` | Snapshot | 22D | cuML (optional) | GMM soft labels (K=20, not KMeans) |
| 6 | `model_derived` | `core/feature/generators/model_features.py` | Snapshot | 27D | - | GMM soft probs(5D) + Bandit(4D) + LNN(18D) |
| 7 | `economics` | `core/feature/generators/economics.py` | State | 17D | - | Income decomposition(8D) + Financial behavior(9D) |
| 8 | `merchant_hierarchy` | `core/feature/generators/merchant_hierarchy.py` | Hierarchy | 27D | - | MCC 계층의 Poincaré 쌍곡 임베딩 (27D). txn_mcc_seq는 next_mcc 누수 방지를 위해 마지막 원소 제거 후 전달 |

### 추가 Generator (보조)

| # | 등록 이름 | 파일 | Axis | Output | 설명 |
|---|-----------|------|------|--------|------|
| 9 | `temporal_pattern` | `core/feature/generators/temporal.py` | Timeseries | 가변 | 시계열 집계 + 주기 인코딩 |
| 10 | `multidisciplinary` | `core/feature/generators/multidisciplinary.py` | State | 24D | 화학/전염병/간섭/범죄 |
| 11 | `phase_transition` | `core/feature/generators/phase_transition.py` | Snapshot | 10D | Phase transition features |

### GPU Utility Layer

`core/feature/generators/gpu_utils.py`가 모든 Generator에 공통 GPU 유틸리티를 제공:
- Device detection (`get_device(prefer_gpu=True)`)
- Adaptive batch sizing
- OOM-retry decorator (GPU 메모리 부족 시 자동 CPU fallback)
- `has_cudf()`, `has_cuml()`, `has_cupy()` lazy import 체크

---

## Stage 4: Feature Integration + Normalization

### Canonical 3-Stage Normalization (CLAUDE.md §2.1, §1.9)

```
Numeric Features (all axes merged)
    ↓
┌─ Stage 1 ─── 멱법칙 자동 감지 (skew + kurt 스크린 → log-log R²) ──▶ log1p(x) 복사본 생성
│                                                                     (`{col}_log` 컬럼으로 피처 행렬 말미에 append)
│
├─ Stage 2 ─── TRAIN split only fit StandardScaler (z-score: mean=0, std=1)
│   · `exclude_from_scaler: [categorical_id, probability]` 선언 컬럼 제외
│     (ID 버킷 정수, [0,1] 확률 컬럼은 scaling 시 분포 왜곡되므로 skip — CLAUDE.md §1.9)
│   · 연속형 (non-binary, non-categorical-ID, non-probability) 컬럼에만 적용
│
└─ Stage 3 ─── Stage 1 에서 생성된 `_log` 컬럼은 **raw magnitude 보존**
                (scaler 재적용 금지 — heavy-tail 신호 유지)
    ↓
    scaler_params.json 저장 (mean, std, log1p_cols, exclude_cols)
    ↓
[Stage 6 완료 후] PipelineRunner._rebuild_group_ranges_post_normalization()
    · Stage 1 이 `_log` 컬럼을 말미에 append 하고 컬럼 순서를 재배열하므로,
      pre-normalization 의 `feature_group_ranges` 는 stale 상태가 된다
    · 각 feature group 의 original + `_log` offspring 을 post-normalization
      컬럼 순서에서 재위치 → longest contiguous block 을 새 range 로 emit
    · 비연속 매칭 발생 시 WARNING, 14 regression test 로 잠금
    · 상세: CLAUDE.md §1.7, commit `ec8587b` (2026-04-21), 테스트
      `tests/test_group_ranges_rebuild.py`
```

### 구현

```python
# core/feature/normalizer.py
class PowerLawAwareScaler:
    """
    3-Stage 정규화의 Stage 1+2+3 을 래핑한 클래스.

    Stage 1: fit 시 각 컬럼의 skew+kurt → log-log R² 검정으로 멱법칙 감지.
             감지된 컬럼에 log1p 변환본(`{col}_log`)을 피처 행렬 말미에 append.
    Stage 2: TRAIN split 에서만 fit. `exclude_from_scaler` 규칙에 따라
             categorical_id / probability sub-kind 를 skip. 연속형에만 z-score.
    Stage 3: Stage 1 에서 생성된 `_log` 컬럼은 raw magnitude 를 보존 — 재스케일 금지.
    get_params() → scaler_params.json 저장 (mean, std, log1p_cols, exclude_cols).
    cuPY 가속: cupy 설치 시 skew/mean/std 계산을 GPU 에서 수행.
    """
```

> **왜 Stage 3 에서 `_log` 를 보존하는가** — FeatureRouter 가 expert 마다 두 뷰 (centered 원본 + tail-preserving log) 를 모두 받을 수 있도록 설계된 구조이기 때문이다. `_log` 를 Stage 2 scaler 에 포함시키면 heavy-tail 신호가 평탄화되어 DeepFM / OT 같은 분포 의존 expert 의 성능이 저하된다.

---

## Schema-Invariant Audit (Phase 0 v3, 2026-04-28)

> 관련 commit: `2c93b1f`, `4b1a40c`, `dfd116c`, `ecfdeb9`, `c6dbe3e`, `9308916`, `88f7a7b` (모두 `main`, 미push). 검증 Job: `santander-ple-phase0-0428-1628`.

### Schema Invariant (스키마 불변식)

Phase 0 산출물의 모든 다운스트림 컴포넌트 (`FeatureRouter`, expert dispatch, group-level routing, IG attribution) 가 의존하는 단일 invariant 는 다음과 같다:

```
feature_groups.yaml registry order
  == Stage 3 (FeatureGroupPipeline) concat order
  == features.parquet column order
  == sequential feature_group_ranges (no gaps, no overlaps)
```

이 등식이 성립하면 `PipelineRunner._rebuild_group_ranges_post_normalization` 은 **no-op 으로 수렴**한다 (입력 range == 출력 range). 위반 시 FeatureRouter 가 out-of-bounds index 로 `features` tensor 를 슬라이스하고, GPU 경로에서 `device-side assert triggered` 만 떠서 `CUDA_LAUNCH_BLOCKING=1` 없이는 디버깅이 불가능하다.

### 발견된 3 종 위반 경로 + 픽스

#### 1) Placeholder `output_columns` 가 실제 generator output 을 가린 경우 (commit `2c93b1f`)

`FeatureGroupConfig.__post_init__` 는 dataclass 생성 시 `output_columns` 가 비어 있으면 placeholder 를 자동으로 채운다:

```python
# core/feature/group_pipeline.py (수정 전)
self.output_columns = [f"{name}_{i}" for i in range(output_dim)]
# → ["mamba_temporal_0", "mamba_temporal_1", ..., "mamba_temporal_49"]
```

`FeatureGroupPipeline.fit()` 안의 3 군데 post-fit assignment 가 `if not group.output_columns:` guard 로 보호되고 있었는데, placeholder 가 이미 채워져 있으니 guard 가 **항상 False** 였다. 그래서 generator 의 실제 output 컬럼명 (`mamba_d0..mamba_d49`, `tda_global_h0_*` 등) 은 끝까지 group 에 등록되지 않았다.

Stage 6 의 `_rebuild_group_ranges_post_normalization` 은 이 placeholder 들을 post-normalization `feature_cols` 에서 lookup 하는데 — 당연히 매치되지 않으니 — **모든 group 을 silently 드롭**하고 stale Stage 3 range 를 그대로 유지했다. 픽스는 guard 를 모두 제거하고 항상 `gen.output_columns` 로 덮어쓰는 것이다.

#### 2) Stage 6 normalizer 의 컬럼 reorder (commit `c6dbe3e`)

`PowerLawAwareScaler.transform_sql` / `transform` 은 projection 을 **고정 bucket 순서**로 emit 했다:

```
[continuous block] → [binary block] → [categorical_int block]
                  → [probability block] → [_log copies]
```

`demographics` 처럼 group 안에 dtype 이 섞여 있으면 (continuous 9 + binary 2 — `gender`, `is_active`) bucket 순서로 재배열되어 binary 컬럼이 원래 위치에서 **약 500 자리 떨어진 곳**으로 흩어진다. Stage 3 는 그룹별 contiguous 였는데 Stage 6 가 깨버린 것.

픽스는 `feature_cols` 를 입력 순서로 순회하면서 각 컬럼의 projection 을 in-place 로 emit 하도록 변경. `_log` 복사본만 진짜 추가분이므로 tail 에 append 하는 동작은 유지.

#### 3) Stage 6 입력 `feature_cols` 가 adapter dictate 순서를 상속 (commit `88f7a7b`)

Normalizer 가 order-preserving 으로 고쳐져도, **입력으로 받는 `feature_cols`** 자체가 DuckDB DESCRIBE 순서 (= adapter 가 컬럼을 만든 순서) 였다. Adapter 는 group registry 와 무관하게 dtype/생성 시점 기준으로 컬럼을 emit 하므로 demographics 의 5 declared cols 가 흩어졌다.

`PipelineRunner._reorder_feature_cols_by_group` helper 를 추가하여, 각 enabled group 의 `output_columns` 를 registry 순서로 먼저 배치하고, 어떤 group 에도 routed 되지 않은 passthrough 컬럼은 tail 에 모은다. 이 시점에 invariant 가 처음으로 성립한다.

### 보조 픽스

- **`_compute_dim_ranges` 는 YAML 의 `output_dim` 대신 `len(group.output_columns)` 사용** (commit `4b1a40c`). YAML 값과 generator 실제 출력이 어긋날 때 range 가 허위 길이로 계산되는 것을 방지.
- **`feature_groups.yaml::output_dim` 을 generator 실측에 맞춰 재정렬** (commit `dfd116c`):
  - demographics 38 → 11
  - tda_global 36 → 16
  - tda_local 24 → 16
  - hmm_states 48 → 25
  - product_hierarchy 32 → 34
  - graph_collaborative 64 → 66
- **`FeatureSpec.meta_cols` / `TaskSpec.derive` 를 dataclass 에 추가** (commit `ecfdeb9`). 둘 다 YAML 에는 선언되어 있었지만 dataclass 가 받지 못해 load 시 silent drop. 결과적으로 `snapshot_date` 와 `has_nba` (= `nba_primary.derive.filter_col`) 가 features.parquet 에 새어들어가고 있었다.

### 방어 가드 (Defensive Guards)

침묵 실패를 fail-fast 로 전환하기 위해 두 군데에 가드를 추가:

```python
# core/model/config_builder.py
# end > n_feat_cols 인 group_range 는 build 시 drop + WARN
ranges = [(g, s, e) for (g, s, e) in ranges if e <= n_feat_cols]

# core/model/feature_router.py (FeatureRouter.route)
# slice index 가 features.shape[-1] 를 넘으면 raise (CUDA assert 대신)
assert idx.max() < features.shape[-1], "router index OOB ..."
```

### 검증 결과 (Phase 0 v3, Job `santander-ple-phase0-0428-1628`)

- `feature_columns` 총 **1211 개**.
- 17 개 group 모두 **연속 (sequential), 갭 없음**:
  ```
  demographics             [   0,    5)
  product_holdings         [   5,   29)
  ...
  mcc_top30_multihot       [1174, 1205)
  ```
- 6 개 trailing passthrough cols: `gender`, `segment`, `country`, `channel`, `age_group`, `income_group` (Stage 2 가 auto-encode 한 categorical 들로, 어떤 group 의 `columns:` 에도 declared 되지 않음).
- `snapshot_date` / `has_nba` 는 **features.parquet 에서 제외**.
- `demographics` 의 declared 5 cols 가 모두 contiguous block 에 포함.

이 시점부터 `_rebuild_group_ranges_post_normalization` 은 input range 와 동일한 output 을 emit 하며 (no-op), expert routing / IG attribution / group-level FRIA 분석 이 모두 동일한 인덱스 공간을 공유한다.

---

## Santander 데이터 피처 상세 (configs/santander/)

### Categorical Embeddings

| 컬럼 | Cardinality | Embedding Dim | 비고 |
|------|-------------|---------------|------|
| `gender` | 2 | 4 | F, M |
| `segment` | 4 | 4 | 01-TOP, 02-PARTICULARES, 03-UNIVERSITARIO, UNKNOWN |
| `country` | 118 | 16 | 118개국 (ES 96%) |
| `channel` | 163 | 16 | 163개 채널 코드 |
| `age_group` | 5 | 4 | young/adult/middle/senior/elderly |
| `income_group` | 5 | 4 | low/mid/high/very_high/unknown |

### Numeric Features

| 컬럼 | 범위 | 특이사항 |
|------|------|---------|
| `age` | 18-100 | 정상 범위 |
| `income` | 0-28.8M EUR | 25.4% missing (0=결측) |
| `tenure_months` | -999999~256 | sentinel 처리 필요 |
| `is_active` | 0/1 | 42% active |
| `num_products` | 0-15 | avg 1.3 |

---

## 13-Task Label Architecture

### 4-Tier 태스크 구조

| Tier | 설명 | 태스크 수 | 예시 |
|------|------|----------|------|
| **Tier 1** | Core targets (직접 레이블) | 3 | churn_signal, product_stability, nba_primary |
| **Tier 2** | Derived targets (규칙 유도) | 1 | cross_sell_count |
| **Tier 3** | Product group + segmentation | 6 | will_acquire_{deposits,investments,accounts,lending,payments}, segment_prediction |
| **Tier 5** | Transaction-based NBA | 3 | next_mcc, mcc_diversity_trend, top_mcc_shift |

> **참고**: has_nba (binary)는 2026-04-12에 nba_primary (multiclass)로 통합됨 — nba_primary의 class 0이 "NBA 없음"을 나타내므로 has_nba는 중복 태스크로 제거됨. 18→14→13 태스크로 축소.

### Per-task Focal Alpha Calibration

Binary 태스크의 `focal_alpha`는 positive rate에 따라 calibrated:

| 태스크 | Positive Rate | focal_alpha | 근거 |
|--------|-------------|-------------|------|
| `churn_signal` | 5.1% | 0.85 | 매우 불균형 |
| `will_acquire_deposits` | ~1% | 0.95 | 극단적 불균형 |
| `will_acquire_investments` | ~2% | 0.90 | 매우 불균형 |
| `will_acquire_accounts` | ~5% | 0.85 | 불균형 |
| `will_acquire_lending` | ~1% | 0.90 | 극단적 |
| `will_acquire_payments` | ~3% | 0.85 | 매우 불균형 |
| `top_mcc_shift` | ~10% | 0.70 | 경도 불균형 |

### 4 Semantic Groups (adaTT Transfer Units)

```yaml
task_groups:
  engagement:   [next_mcc, top_mcc_shift]
  lifecycle:    [churn_signal, product_stability, segment_prediction]
  value:        [mcc_diversity_trend, cross_sell_count]
  consumption:  [nba_primary, will_acquire_deposits,
                 will_acquire_investments, will_acquire_accounts,
                 will_acquire_lending, will_acquire_payments]
```

---

## Item Universe + Product Hierarchy

### Item Universe (24 금융 상품)

```
┌──────────────────────────────────────────────────┐
│ 예금: saving, guarantee, checking, derivados     │
│ 계좌: payroll_acct, junior_acct, particular_acct │
│       particular_plus, e_account, home_acct      │
│ 예탁: short_deposit, medium_deposit, long_deposit│
│ 투자: funds, securities, pension_plan,           │
│       pension_deposit                             │
│ 대출: mortgage, loans                            │
│ 결제: credit_card, direct_debit, auto_debit      │
│ 세금: taxes                                      │
│ 보수: payroll                                    │
└──────────────────────────────────────────────────┘
```

### Product Hierarchy Config

```json
{
  "hierarchy": {
    "level_1": {
      "deposits": ["saving", "short_deposit", "medium_deposit", "long_deposit"],
      "accounts": ["checking", "payroll_acct", "junior_acct", "particular_acct", "..."],
      "investments": ["funds", "securities", "pension_plan", "pension_deposit"],
      "lending": ["mortgage", "loans"],
      "payments": ["credit_card", "direct_debit", "auto_debit"]
    }
  }
}
```

---

## 피처 차원 관리 — 동적 계산

```python
# core/feature/schema.py
class FeatureSchema:
    @property
    def input_dim(self) -> int:
        return sum(g.output_dim for g in self.groups if g.enabled)

    @property
    def axis_dims(self) -> dict[str, int]:
        dims = defaultdict(int)
        for g in self.groups:
            if g.enabled:
                dims[g.axis] += g.output_dim
        return dict(dims)
        # → {"state": 314, "snapshot": 139, "timeseries": 214, "hierarchy": 41, "item": 64}
```

---

## SageMaker Processing에서 실행

```
Step Functions
    ↓
SageMaker Processing Job
    ├── 입력: s3://bucket/data/encrypted/
    ├── 코드: core/feature/pipeline_builder.py
    ├── config: configs/santander/feature_groups.yaml
    ├── 엔진: DuckDB (인프로세스)
    ├── GPU: cuPY/cuDF optional (ml.g4dn.xlarge)
    └── 출력: s3://bucket/features/v{version}/
         ├── features.parquet
         ├── labels.parquet
         ├── sequences.npy
         ├── seq_lengths.npy
         ├── feature_schema.json
         ├── label_schema.json
         ├── split_indices.json
         ├── scaler_params.json
         ├── label_transforms.json
         ├── item_universe/
         │   ├── product_hierarchy.json
         │   └── customer_product_graph.parquet
         └── transformers/
             └── transformer_00_PowerLawAwareScaler.pkl
```

---

## 현재 vs AWS — 피처 엔지니어링 비교

| 항목 | 현재 (On-Prem) | AWS (설계) | 변경 이유 |
|------|---------------|-----------|----------|
| 피처 분류 | 없음 (flat 목록) | **5-Axis** (State/Snapshot/Timeseries/Hierarchy/Item) | Expert 라우팅 명시적 기반 |
| 피처 정의 | 코드 내 하드코딩 | YAML 선언형 (feature_groups.yaml) | 코드 변경 없이 피처 추가/삭제 |
| 차원 관리 | 644D, 734D 하드코딩 | `FeatureSchema.input_dim` 동적 계산 + axis_dims | 피처 변경 시 자동 반영 |
| 정규화 | QuantileTransform + Raw Power-Law | **StandardScaler + power-law 자동 감지** (skew>2.0 → log1p+raw) | 단일 경로, 분포 보존 |
| TDA | 단일 윈도우 | **Global(365일)/Local(90일) 분리** → Snapshot/Timeseries 축 | 장기/단기 위상 분리 |
| HMM | 코드에 직접 구현 | Generator Registry (hmm_triple_mode) | 선택적 활성화 |
| Graph | HGCN만 | **Poincare + LightGCN** (Hierarchy + Item 축) | 계층+협업 분리 |
| Item Universe | 없음 | **고객x상품 bipartite graph** (24 금융 상품) | 상품 추천 핵심 |
| 레이블 생성 | 코드 하드코딩 | **LabelDeriver** (config-driven 13 tasks) | 선언적, 재현 가능 |
| 누수 방지 | 없음 | **LeakageValidator** (4-check) + temporal split | 자동 누수 감지 |
| 피처 버전 | 없음 | features/v{version}/ | 재현성, 롤백 |
| Cold Start | 없음 | **is_cold_start flag + sequence-derived feature zeroing** | cold start 고객 대응 |
| GPU 가속 | 없음 | cuDF primary (generators), cuPY (TDA, scaler), cuML (GMM) | 5-10x 가속 |
