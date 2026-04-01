# Paper 1 Outline — Architecture & Ablation

## Title (후보)
- "Heterogeneous Expert PLE: An Explainable Multi-Task Architecture for Financial Product Recommendation"
- "Beyond Homogeneous Experts: Multi-Disciplinary Feature Engineering and Structural Explainability for Financial Recommendation"
- "From Correlation to Persuasion: Explainable Financial Recommendation via Heterogeneous Expert Progressive Layered Extraction"

---

## Abstract (~250 words)
- 문제: 금융 추천은 성능뿐 아니라 설명 가능성/규제 준수가 필수. 기존 모델은 사후적 설명(SHAP/LIME)에 의존.
- 제안: 이종 전문가 PLE+adaTT — 7가지 다른 inductive bias를 가진 expert를 shared basket에 배치. gate weight 자체가 비즈니스 해석 가능한 설명을 생성.
- 방법: 다학제 피처(위상수학, 쌍곡기하학, 경제학 등) + 금융 DNA 기반 태스크 그룹 + LGBM 증류 + 추천사유 생성
- 결과: 48개 ablation 시나리오에서 각 expert/feature group의 독립적 기여 증명. graceful degradation 확인. 금감원/EU AI Act 요건 충족.
- 의의: 학습→증류→서빙→설명까지 end-to-end 파이프라인. 소규모 팀(3명)+AI 에이전트로 구축.

---

## 1. Introduction (~2 pages)

### 1.1 문제 정의
- 금융 상품 추천의 특수성: 성능 < 설명 가능성 + 규제 준수 + 안정성
- 기존 접근의 한계:
  - 단일 모델: 멀티태스크 불가, 설명 불가
  - 모델 앙상블: 관리 포인트 N개, 서빙 비용 N배, "MLP 3번이 기여" = 의미 없는 설명
  - 사후 설명(SHAP/LIME): 모델과 분리, 불안정, 추가 비용
- 핵심 질문: "모델 자체가 인간을 설득할 수 있는 설명을 구조적으로 생성할 수 있는가?"

### 1.2 핵심 통찰
- 설득의 대상은 항상 사람 (고객/행원/금감원)
- 사람은 확률이 아니라 이야기로 설득됨
- 기존 모델의 "모래상자 흔들기"(통계적 상관)는 인과적 설득력 부족
- 각기 다른 관점의 전문가가 다른 종류의 "왜"에 답할 수 있다면 → 구조적 설명

### 1.3 제안 아키텍처 요약
- 이종 전문가 PLE + adaTT + 금융 DNA 태스크 그룹
- 4가지 구조적 특성: 견고한 설명, 안정적 내결함성, 유연한 확장성, 통합 관리 가능성

### 1.4 Contributions
1. **Heterogeneous Shared Expert Basket**: PLE에 7가지 다른 inductive bias의 expert를 배치 (기존: 동종 MLP)
2. **Inherent Explainability**: gate weight가 비즈니스 맥락의 설명을 직접 생성 (사후 SHAP 불필요)
3. **Multi-disciplinary Feature Engineering**: 위상수학/쌍곡기하학/경제학/인과추론 등 9개 학문 분야의 피처
4. **Financial DNA Task Grouping**: 4축(engagement/lifecycle/value/consumption) 태스크 그룹 + adaTT 차등 전이
5. **End-to-end Pipeline**: 학습→증류→서버리스 서빙→추천사유 생성, 단일 config 관리
6. **Regulatory Compliance**: 금감원 가이드라인 + EU AI Act 요건의 명시적 아키텍처 매핑
7. **Reproducible Benchmark**: Gaussian Copula + latent variable 기반 합성 데이터 (variance budget으로 난이도 통제)

---

## 2. Related Work (~2 pages)

### 2.1 Multi-Task Learning for Recommendation
- Shared-Bottom → MMoE → PLE 발전 과정
- adaTT, ESMM, STAR 등 최근 MTL 추천

### 2.2 Mixture of Experts
- MoE (Shazeer 2017), Switch Transformer
- 기존 MoE의 한계: 전부 동종 MLP → 이종 expert 연구 부재

### 2.3 Explainability in Recommendation
- SHAP/LIME의 한계 (사후적, 불안정)
- 구조적 설명 가능성 접근법
- 금융 도메인 XAI 요구사항

### 2.4 Financial Recommendation Systems
- 은행/카드사 추천 시스템 사례
- 규제 환경 (금감원, EU AI Act)

### 2.5 Individual Expert Technologies
- TDA, Hyperbolic GCN, Mamba, LNN, Causal Inference 등
- 각 기술의 금융 적용 사례

---

## 3. Architecture (~4 pages, 핵심 섹션)

### 3.1 Design Philosophy
- "인간을 설득하는 추천" — 4축 설계 원칙
- ALS → Black-Litterman(드랍) → 앙상블 재정의 → PLE+adaTT 선택 과정
- 하드웨어 제약 하에서의 효율적 표현력 확보 전략

### 3.2 Data 5-Axis Classification
- 상태/스냅샷/단기시계열/장기시계열/단절시계열/계층/관계/위상/인과
- 각 축 → 최적 feature generator → 최적 expert 매핑
- [Figure 1: 데이터 축 → Feature Group → Expert 매핑 다이어그램]

### 3.3 Multi-disciplinary Feature Engineering
- 9개 학문 분야별 피처 생성 전략
- [Table 1: 학문 분야 × Generator × Expert × 금융 해석 매핑]
- 피처의 이중 역할: 학습 성능 + 추천사유 재료

### 3.4 Heterogeneous Expert Basket
- 7 experts의 inductive bias와 파라미터 효율성
- [Figure 2: PLE 아키텍처 — 이종 expert basket + CGC gate + task towers]
- Temporal Ensemble 내부 앙상블 (Mamba + LNN + Transformer)
- 각 expert가 다른 종류의 "왜"에 답하는 구조

### 3.5 Financial DNA Task Grouping
- GMM 서브헤드(드랍) → 태스크 그룹 전환 배경
- 4축 그룹: engagement, lifecycle, value, consumption
- [Table 2: 태스크 그룹 × 태스크 × 설명]

### 3.6 adaTT + Logit Transfer
- 그룹 내/간 transfer strength 차등화
- 로짓 전이 3방식 (output_concat / hidden_concat / residual)
- 소비자 경험의 자연스러운 이전 반영
- [Figure 3: adaTT affinity matrix 시각화]

### 3.7 Inherent Explainability
- Gate weight → 비즈니스 맥락 해석
- SHAP/LIME 대비 장점 (안정성, 비용, 일관성)
- 피처 → 비즈니스 역매핑 → LLM Agent 추천사유 생성
- [Figure 4: 추천사유 생성 파이프라인 (3-agent)]

---

## 4. Training Pipeline & Reproducibility (~1.5 pages)

### 4.1 Config-Driven Pipeline
- pipeline.yaml + feature_groups.yaml로 전체 파이프라인 제어
- Adapter → Pipeline → Trainer 관심사 분리
- 새 데이터셋/태스크 추가 시 config만 변경

### 4.2 Data Processing Backend
- DuckDB/cuDF/PyArrow 네이티브 (pandas-free)
- Phase 0: 피처 생성 + 정규화 + leakage 검증
- Generator 입력에서 label 자동 제외 (leakage 방지)

### 4.3 Deployment Considerations (간략)
- LGBM 증류 + Lambda 서빙 (상세는 논문 2에서)
- [Figure 5: 전체 파이프라인 흐름도 (Phase 0→학습→증류→서빙)]

---

## 5. Experiments (~4 pages)

### 5.1 Benchmark Data
- Gaussian Copula + Latent Variable 합성 데이터
- 4-Layer generative model 설명
- Variance budget으로 태스크별 난이도 통제
- XGBoost AUC ceiling 검증
- [Table 4: 레이블별 variance budget + AUC ceiling]

### 5.2 Experimental Setup
- 데이터: 1M customers, 316 features, 18 tasks
- 하드웨어: RTX 4070 12GB (로컬) / g5.xlarge A10G 24GB (클라우드)
- 학습 설정: epochs, batch_size, lr, AMP, early stopping
- 평가 메트릭: AUC (binary), F1 (classification), MAE/RMSE/R² (regression)

### 5.3 Feature Group Ablation (RQ1: 각 피처 그룹이 기여하는가?)
- Bottom-up: base + one group
- Top-down: full - one group
- [Table 5: Feature ablation 결과]
- [Figure 6: Feature contribution waterfall chart]

### 5.4 Expert Ablation (RQ2: 각 expert가 전문화되는가?)
- Bottom-up: deepfm + one expert
- Top-down: full - one expert
- [Table 6: Expert ablation 결과]
- [Figure 7: Expert × Task Group AUC heatmap]

### 5.5 Task × Structure Cross Ablation (RQ3: PLE+adaTT 구조가 효과적인가?)
- 4 task tiers × 4 structures
- [Table 7: Structure comparison]
- [Figure 8: Task 수 증가에 따른 구조별 성능 변화]

### 5.6 Graceful Degradation (RQ4: 안정적인가?)
- Expert 제거 시 성능 저하 폭 분석
- 급격한 성능 저하 없이 나머지가 보완하는지
- [Figure 9: Degradation curve per expert removal]

### 5.7 Explainability Analysis (RQ5: gate weight가 의미있는 설명을 제공하는가?)
- Gate weight 분포 분석: expert별 태스크 전문화 패턴
- SHAP 대비 gate-based 설명의 안정성 비교 (입력 perturbation 실험)
- (추천사유 생성 품질 평가는 논문 2에서 상세)

---

## 6. Discussion (~1.5 pages)

### 6.1 Findings Summary
- RQ1-5에 대한 답변 종합

### 6.2 Practical Implications
- 금융사 도입 시 고려사항
- 소규모 팀 + AI 에이전트 개발 방법론
- SageMaker vs K8s 인프라 선택 기준
- TCO 분석

### 6.3 Limitations
- 합성 데이터 vs 실데이터 gap
- 단일 GPU 학습 한계 (DDP 미구현)
- 추천사유 생성의 LLM 의존성
- 규제 환경 변화에 따른 아키텍처 수정 필요성

### 6.4 Future Work
- DDP multi-GPU 지원
- Spark Ingestion layer
- 실데이터 검증
- Online A/B 테스트
- 추천사유 생성 및 규제 대응 상세 (→ 논문 2 예고)

---

## 7. Conclusion (~0.5 page)
- 핵심 메시지 재강조: "인간을 설득하는 추천을 위한 구조적 설명 가능성"
- 기여 요약
- 금융 AI의 방향성 제시

---

## Appendix

### A. Feature Groups Configuration (재현성)
### B. 10 Generators Technical Details
### C. Full Ablation Results (all 54 scenarios)
### D. Regulatory Compliance Mapping (detailed)
### E. Synthetic Data Generation Algorithm
### F. AI-Augmented Development Methodology

---

## Figures & Tables Summary

| # | Type | 내용 | 섹션 |
|---|------|------|------|
| Fig 1 | 다이어그램 | 데이터 5축 → Feature Group → Expert 매핑 | 3.2 |
| Fig 2 | 다이어그램 | PLE 아키텍처 (이종 expert + gate + towers) | 3.4 |
| Fig 3 | Heatmap | adaTT affinity matrix | 3.6 |
| Fig 4 | 다이어그램 | 추천사유 3-agent 파이프라인 | 3.7 |
| Fig 5 | 다이어그램 | 서빙 아키텍처 (Lambda + distillation) | 4.3 |
| Fig 6 | Waterfall | Feature group contribution | 5.3 |
| Fig 7 | Heatmap | Expert × Task Group AUC | 5.4 |
| Fig 8 | Line chart | Task 수 vs 구조별 성능 | 5.5 |
| Fig 9 | Bar chart | Graceful degradation curve | 5.6 |
| Tab 1 | 매핑 | 학문 분야 × Generator × Expert × 해석 | 3.3 |
| Tab 2 | 매핑 | 태스크 그룹 × 태스크 × 설명 | 3.5 |
| Tab 3 | 매핑 | 규제 요건 × 아키텍처 대응 | 4.4 |
| Tab 4 | 설정 | 레이블별 variance budget + AUC ceiling | 5.1 |
| Tab 5 | 결과 | Feature ablation AUC | 5.3 |
| Tab 6 | 결과 | Expert ablation AUC | 5.4 |
| Tab 7 | 결과 | Structure comparison AUC | 5.5 |

---

## 예상 분량
- 본문: ~16 pages (2-column ACM/IEEE format)
- Appendix: ~4 pages
- References: ~40 papers
- 총: ~20 pages

## 타겟 학회/저널 후보
- **KDD** (ACM SIGKDD): 산업 트랙 — end-to-end 시스템 + ablation
- **RecSys** (ACM): 추천 시스템 특화 — 아키텍처 + 설명 가능성
- **AAAI** (AI for Financial Services 워크숍): 금융 AI 특화
- **arxiv preprint**: 빠른 공개 + 학회 제출 전 피드백
