# Project Background & Evolution

## 1. 프로젝트 출발점

### 배경 (2025년 말)
타겟: 은행, 카드사, 금융지주 — 예금/카드/대출/투자 등 종합 금융 상품 추천.
기존 금융 상품 추천 시스템은 ALS(Alternating Least Squares) 기반 협업 필터링으로 운영되고 있었다.
MLOps 도입을 결정하면서 ALS를 대체할 차세대 모델을 검토하기 시작했다.

### 초기 모델 후보 검토
1. **DL 모델군**: DeepFM, Wide&Deep, AutoInt 등 추천 특화 딥러닝
2. **GBM 모델군**: XGBoost, LightGBM, CatBoost 등 tabular 강자
3. **앙상블 접근**: DL + GBM 조합으로 상호 보완

앙상블 방식을 고민하던 중, 금융 포트폴리오 최적화에서 쓰이는
**Black-Litterman 모델**을 추천에 적용하려는 시도가 있었다.

- 아이디어: 각 모델(DL, GBM 등)의 예측을 "전문가 의견(view)"으로 취급하고,
  각 의견의 불확실성(리스크)에 비중을 두어 베이지안 업데이트로 통합.
  이 결합된 사후 분포가 각 태스크의 최종 예측으로 업데이트되는 구조를 구상.
- 기대: 전문가(모델) 간 의견 불일치를 불확실성으로 자연스럽게 표현하고,
  신뢰도 높은 모델의 의견에 가중치를 더 주는 adaptive 앙상블

### Black-Litterman이 드랍된 이유
설계 단계에서 핵심 한계가 확인됨:
- **비즈니스 해석 불가**: 베이지안 업데이트 과정에서 각 모델의 기여가 혼합되면서,
  "왜 이 상품을 추천했는가"를 명확한 비즈니스 맥락으로 설명하기 어려움.
  금감원 규제 환경에서 이는 치명적 약점.
- 금융 포트폴리오(연속 비중 배분)와 상품 추천(이산 선택)의 구조적 차이
- View matrix(각 모델의 불확실성 추정)의 자동화가 어렵고 주관적
- 멀티태스크(이탈/추천/세그먼트/가치 등)를 하나의 BL 프레임워크로 통합 시
  태스크 간 관계를 표현할 수단이 부족

### 2가지 추가 설계 원칙

기존 추천 모델의 "모래상자 흔들기"(통계적 상관만으로 추천)를 넘어서기 위해
인과적 설득력을 확보하는 2가지 축을 추가:

**축 1: 다학제 피처 다양화**
동일 데이터에서 여러 학문 분야의 관점으로 시사점을 추출.
현장 경험과 학술 연구에서 검증된 방법론들을 피처 생성에 도입:

| 학문 분야 | 도입 요소 | 금융 고객에서의 의미 |
|-----------|----------|-------------------|
| 위상수학 | TDA Persistent Homology | 소비 패턴의 구조적 형태 |
| 쌍곡기하학 | Hyperbolic GCN | 상품 계층 구조의 왜곡 없는 임베딩 |
| 확률과정 | HMM 상태 전이 | 고객 생애주기 단계 추적 |
| 제어이론 | Mamba (State Space) | 장기 행동 의존성 |
| 경제학 | 항상소득/일시소득, 한계효용 | 소비 여력의 구조적 분해 |
| 금융공학 | 리스크 지표, Bandit 탐색/활용 | 상품 탐색 성향 |
| 그래프이론 | LightGCN | 유사 고객군 행동 전이 |
| 통계학 | GMM 군집화 | 연성 세그먼테이션 |
| 인과추론 | Causal DAG | 행동 간 인과 방향 |

핵심: 이 관점들은 서로 중복되지 않는다. 위상수학의 "형태"와 경제학의 "효용"은
같은 고객의 완전히 다른 측면이며, 각각이 독립적으로 기여한다.
(ablation으로 "대체 불가능한 정보"임을 증명)

**축 2: 추천사유 생성을 위한 피처의 이중 역할**

다학제 피처의 가치는 학습 성능만이 아니다:
- 학습 관점: AUC 기여 (있으면 좋고)
- 설명 관점: 추천사유의 재료 (이게 핵심)

ablation에서 TDA 피처를 빼도 AUC가 0.01밖에 안 떨어질 수 있지만,
"고객님의 소비 패턴이 지속적으로 안정적인 형태를 보이고 있어"라는
설명은 TDA 없이는 생성 불가하다.

따라서 모든 피처에 **비즈니스 맥락 역매핑**을 구축:
- hmm_lifecycle_prob_growing → "성장 단계 고객"
- mamba_temporal_d3 → "최근 3개월 소비 증가 추세"
- hgcn_hierarchy_d5 → "투자 상품군과 가까운 포지션"
- synth_stability → "안정적 거래 패턴"

이 역매핑된 맥락들을 LLM Agent가 조합하여 자연어 추천사유를 생성:
```
[모델 추론] → [IG Attribution: 상위 피처] → [비즈니스 역매핑]
→ [LLM Agent: 맥락 조합 → 자연어 추천사유]

"고객님은 현재 자산 성장 단계에 있으시고,
 최근 소비가 증가하는 추세입니다.
 보유하신 상품 구조상 투자 펀드가 자연스러운
 다음 단계이며, 안정적인 거래 패턴을 고려할 때
 중위험 균형형 펀드를 추천드립니다."
```

넓은 피처가 학습 성능에 큰 영향을 못 주더라도,
인간을 설득할 때에는 충분한 재료가 된다.

**축 3: 인과적 설명 요소 강화**

다학제 피처와 추천사유 생성만으로는 "상관이 있다"는 설명은 되지만
"A 때문에 B"라는 인과적 설명은 부족하다.
금감원과 고객 모두 "왜"를 물을 때, 상관이 아닌 인과를 기대한다.

이를 위해 아키텍처에 인과 설명 요소를 추가:
- **Causal Expert**: DAG 제약으로 피처 간 인과 방향을 구조적으로 강제.
  "소비 증가 → 상품 관심 → 구매"라는 방향성을 모델이 학습.
- **Optimal Transport Expert**: 고객 세그먼트 간 분포 이동을 Wasserstein distance로 측정.
  "이 고객이 A 세그먼트에서 B 세그먼트로 이동 중"이라는 변화 방향 포착.
- **logit transfer (3방식)**: 태스크 간 인과 관계를 output_concat / hidden_concat / residual로 표현.
  "이탈 가능성이 높아졌기 때문에 → 리텐션 상품 추천"이라는 태스크 간 인과 연결.

이 인과적 요소들이 추천사유 생성 시 단순 "상관" 설명을 "인과" 설명으로 격상:
- 상관: "비슷한 고객들이 이 상품을 구매했습니다"
- 인과: "최근 소비 패턴 변화가 투자 관심 증가로 이어지고 있어, 이 시점에 투자 상품이 적합합니다"

**축 4: 이종 전문가에 의한 구조적 설명 가능성** (아래 상세)

### "효율적 앙상블"에 대한 재정의
Black-Litterman 드랍 후, 질문을 재구성:
- "여러 모델을 어떻게 잘 섞을 것인가" (앙상블)
  → "하나의 모델 안에서 여러 전문가가 협업하는 구조는 없는가" (MoE/PLE)

기존 앙상블의 문제:
- 모델 N개를 각각 학습 → 추론 시 N개 모두 실행 → 서빙 비용 N배
- 모델 간 중복 학습 (같은 피처를 여러 모델이 반복 처리)
- 모델별 관리/업데이트 포인트 N개 → 운영 부담

하드웨어/인력 제약 하에서의 현실적 요구:
- GPU 1-4장 수준의 하드웨어로 학습/서빙 가능해야
- 모델 관리 포인트를 최소화 (하나의 모델, 하나의 config)
- 새 태스크/피처 추가 시 전체를 재설계하지 않아야

### MTL → MMoE → PLE 선택 과정
앙상블을 모델 외부가 아니라 **모델 내부에서 수행**하는 방향으로 전환.
Multi-Task Learning (MTL) 계열을 검토:

1. **Shared-Bottom**: 전문가 1개를 전 태스크가 공유 → negative transfer 심각
2. **MMoE (Multi-gate Mixture of Experts)**: 전문가 N개 + 태스크별 gate
   → 개선되지만, 모든 expert가 모든 태스크에 노출 → 전문화 부족
3. **PLE (Progressive Layered Extraction)**: shared expert + task-specific expert 분리
   → 태스크별 전용 expert가 있어 negative transfer 최소화
   → CGC gate로 shared/specific expert를 태스크별로 adaptive 조합

**PLE가 선택된 이유:**
- expert network = 내부 앙상블 (전문가별 특화)
- CGC gate = 태스크별 전문가 가중치 → 설명 가능성 (gate weight로 기여도 분해)
- 단일 모델 학습/배포 → 관리 포인트 1개, 서빙 비용 고정

### 태스크 그룹 설계: GMM 서브헤드 → 태스크 그룹으로 전환

초기 설계에서는 **GMM 클러스터별 태스크 서브헤드**를 두려 했다:
- 고객 클러스터 K개 → 클러스터별 전용 task head K개
- 아이디어: 세그먼트마다 다른 예측 로직

하지만 클러스터 수가 늘어날수록 복잡도가 K × T (클러스터 × 태스크)로 폭발.
K=20, T=18이면 서브헤드 360개 → 관리 불가, 과적합 위험.

**방향 전환: 태스크 그룹 기반 서브헤드**
클러스터가 아니라 **태스크를 그룹으로** 나누고, 그룹에 서브헤드를 부여.
그룹핑 기준은 고객을 이해하는 **금융적 DNA 관점**:

| Task Group | 금융 DNA | 포함 태스크 | 의미 |
|------------|---------|-----------|------|
| engagement | 고객이 무엇을 하는가 | has_nba, engagement_score, next_mcc, top_mcc_shift | 행동/활동 |
| lifecycle | 고객이 어디에 있는가 | churn_signal, tenure_stage, segment_prediction | 생애주기 단계 |
| value | 고객이 얼마나 가치있는가 | income_tier, spend_level, cross_sell_count, product_stability | 경제적 가치 |
| consumption | 고객이 무엇을 살 것인가 | will_acquire_* (5개), nba_primary | 구매 의향 |

이 그룹핑이 이후 모든 구조에 일관되게 적용:
- CGC gate: 그룹별 expert 가중치 학습
- adaTT: 그룹 내/간 transfer strength 차등화
- logit transfer: 그룹 간 자연스러운 경험 이전

### adaTT 결합 + 로짓 전이
PLE만으로는 태스크 간 관계가 gate를 통한 간접적 공유에 한정.
실제 금융에서는 태스크 간 명시적 상호작용이 중요:
- 이탈 가능성 ↔ 상품 구매 의향 (역상관)
- 고객 세그먼트 → 추천 상품 카테고리 (선행 관계)
- 소비 패턴 → 교차 판매 기회 (인과 관계)

adaTT(Adaptive Task Transfer)를 결합하여:
- 태스크 그룹핑을 그대로 살려서 **그룹 내 연관성(intra)과 그룹 간 연관성(inter)에 차이**
  - intra-group: 강한 전이 (같은 DNA 관점의 태스크 간)
  - inter-group: 약한 전이 (다른 DNA 관점 간 간섭 최소화)
- affinity matrix를 데이터로부터 자동 학습

**로짓 전이(logit transfer)**: 소비자 경험의 자연스러운 이전을 모델에 반영
- has_nba → nba_primary: "가입 여부 → 어떤 상품" (순차적 경험)
- engagement_score → has_nba: "활성도 → 가입 가능성" (선행 지표)
- spend_level → will_acquire_*: "소비 수준 → 구매 의향" (구매력 전이)
- 3가지 전이 방식: output_concat / hidden_concat / residual

이것이 Black-Litterman이 외부에서 하려던 것(전문가 의견의 불확실성 기반 통합)을
모델 내부 구조로, 그리고 데이터 기반으로 해결한 셈이다.
동시에 태스크 그룹이 금융적 DNA에 기반하므로,
전이 관계 자체가 비즈니스 해석 가능하다:
"이 고객의 활성도가 높아지고 있어서(engagement) → 신규 가입 가능성이 올라갔고(lifecycle 전이)
→ 소비 패턴상 투자 상품이 적합합니다(consumption 전이)"

## 2. 온프렘 초기 버전 (v1-v2)

### 초기 설계 (온프렘 Airflow + DuckDB)
- Airflow DAG로 파이프라인 오케스트레이션
- DuckDB 파일 기반 데이터 저장/처리
- Docker GPU 컨테이너에서 학습
- 5축 피처 분류: 상태/스냅샷/시계열/계층/아이템

### 온프렘에서 확립된 핵심 설계 원칙 (유지)
- 피처 그룹별 전문가 라우팅 (target_experts)
- 태스크 그룹 기반 adaTT (intra/inter strength)
- 3단계 정규화 (log transform → scaler → raw copy)
- 로짓 전이 3방식 (output_concat / hidden_concat / residual)
- Uncertainty weighting (Kendall et al.)
- 증류 시 IG 기반 피처 선택

### 온프렘의 한계
- 단일 GPU 서버로 대규모 실험 불가
- ablation 48개 시나리오를 순차 실행하면 수일 소요
- 서빙 인프라 부재 (학습만 하고 배포는 수동)
- 재현성: 환경 의존적, 다른 연구자가 재현 어려움

## 3. AWS 마이그레이션 (v3.x — 현재)

### 온프렘과 클라우드의 관계
온프렘 시스템은 **그대로 운용** 중이며, 클라우드 버전은 **별도로 마이그레이션에 대비**하여 구축.
두 시스템은 병행 운영 구조:
- 온프렘: 현재 운영 환경. 규제상 데이터 이동이 어려운 코어 시스템.
- 클라우드(AWS): 마이그레이션 대비 + ablation 실험 + 논문용 재현 환경.
- 아키텍처 철학(PLE+adaTT, config-driven, expert routing 등)은 동일하게 유지.
- 인프라만 Airflow/DuckDB → SageMaker/S3로 전환.

향후 금융 클라우드 규제가 더 완화되면 클라우드로 전면 전환 가능하도록 설계.
온프렘에서 검증된 모델 아키텍처를 그대로 클라우드에서 재현할 수 있음이 핵심.

### 왜 AWS/SageMaker를 선택했는가
- ablation 병렬 실행 (Spot 4대 → 48시나리오를 4시간에)
- end-to-end 파이프라인: 학습 → 증류 → Lambda 서빙
- 논문의 "프로덕션 배포 가능" 주장을 뒷받침
- 금융사 현실에 맞는 아키텍처 (K8s 불필요, 서버리스 서빙)

### AWS에서 달라진 것
- Airflow DAG → SageMaker Pipelines / Step Functions
- DuckDB 파일 → S3 Parquet (저장소)
- Docker GPU 컨테이너 → SageMaker Training Job
- 수동 서빙 → Lambda + API Gateway

### AWS에서 유지한 것
- 모델 아키텍처 전체 (PLE, adaTT, experts, task groups)
- Config-driven 원칙 (YAML에서 모든 파라미터)
- 관심사 분리 (Adapter → Pipeline → Trainer)

## 4. 주요 기술적 전환점

### 데이터 처리 백엔드 전환
**pandas → DuckDB/cuDF/PyArrow**

초기에는 pandas 중심이었으나, 1M row 데이터에서 메모리 문제 발생.
단계적으로 전환:
1. Phase 0 (데이터 전처리): DuckDB SQL 네이티브 (pandas 제거)
2. Generators: cuDF GPU 가속 (10개 전부 전환)
3. train.py: PyArrow 로딩 → numpy → torch tensor (pandas import 없음)
4. Normalizer: CuPy/numpy 직접 계산 (sklearn 최소화)
5. LeakageValidator: numpy.corrcoef 직접 (pandas .corr() 제거)

정책: cuDF (GPU) → DuckDB (CPU columnar) → pandas (최후 fallback, 10K 이하만)

### 합성 벤치마크 데이터 구축
**왜 합성 데이터가 필요했는가**

Santander 실데이터로 초기 테스트 시 문제 발견:
- will_acquire_* 태스크의 AUC가 0.98-0.99 → 시나리오 간 델타가 무의미하게 작음
- 레이블이 product ownership에서 직접 파생 → 사실상 leakage
- 데이터가 cross-sectional (단일 스냅샷) → temporal split 의미 없음

해결: Gaussian Copula + Latent Variable 기반 합성 데이터 생성
- 6 GMM-fitted personas (실데이터 calibration 기반)
- 5D latent vector (관측 불가 → AUC ceiling 통제)
- Variance budget으로 태스크별 난이도 설계
- Label noise로 feature-latent 상관 우회 방지

### Label Leakage 발견 및 수정
**3차례의 leakage 발견**

1. **has_nba_1 duplicate**: label derivation 시 기존 has_nba 컬럼이 남아서 _1 suffix로 중복 생성 → corr=1.0. EXCLUDE로 기존 컬럼 먼저 제거 후 재파생.

2. **Ground truth 파일 로드**: benchmark_ground_truth.parquet이 benchmark_v2.parquet보다 glob 정렬에서 앞 → Phase 0가 latent/persona 데이터로 실행. ground_truth 파일을 하위 디렉토리로 분리.

3. **Generator label leakage**: GMM/model_derived generator가 label 컬럼(has_nba, product_stability 등)을 입력으로 사용 → AUC 1.0. run_generators_duckdb()에 label_cols 자동 제외 추가. exclude_columns config 키도 구현 (기존에 dead code였음).

### GPU 활용 최적화
**37% → 98% GPU utilization**

1. num_workers=0 → 2 (DataLoader CPU prefetch)
2. batch_size 4096 → 6144 (VRAM 97% 활용)
3. Batch streaming: CPU tensor → GPU per batch (전체 데이터 GPU preload 제거)
4. VRAM diagnostics per epoch (allocated/reserved/peak 로깅)
5. 모든 DataLoader 파라미터를 pipeline.yaml config에서 참조

## 5. 아키텍처 결정의 흐름

```
ALS (기존)
  ↓ "MLOps 도입, 차세대 모델 필요"
DL + GBM 앙상블 검토
  ↓ "단순 앙상블로는 멀티태스크 통합 어려움"
Black-Litterman 시도
  ↓ "금융 포트폴리오 ≠ 상품 추천, 설계 단계에서 드랍"
PLE + adaTT 선택
  ↓ "온프렘에서 프로토타입 검증"
AWS 마이그레이션
  ↓ "ablation 병렬화 + end-to-end 서빙 필요"
벤치마크 데이터 + Ablation 실행 (현재)
  ↓ (예정)
증류 + Lambda 서빙 + 논문 작성
```

## 6. 현재 상태 (2026-03-31)

### 완료
- Config-driven 전체 파이프라인 (YAML 기반, 하드코딩 제거)
- DuckDB/cuDF/PyArrow 네이티브 데이터 처리 (pandas 제거)
- 1M 벤치마크 데이터 생성 (Copula + latent + variance budget)
- Label leakage 수정 및 검증 (XGBoost AUC ceiling 확인)
- GPU 최적화 (98% utilization, 12GB VRAM 활용)
- 48개 ablation 시나리오 설계 및 실행 중

### 진행 중
- Ablation 48 시나리오 실행 (~24시간 소요)

### 남은 작업
- Ablation 결과 분석 및 논문 표/그래프 생성
- Phase 4: Knowledge distillation (PLE → LGBM)
- Phase 5: Lambda 서빙 + 추천사유 생성
- DDP wrapper (multi-GPU 선택적 지원)
- Spark Ingestion layer (대규모 데이터 확장)
- 논문 작성
