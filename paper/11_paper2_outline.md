# Paper 2 Outline — Recommendation Reason Generation & Regulatory Compliance

## Title (후보)
- "From Prediction to Persuasion: Agentic Recommendation Reason Generation for Regulatory-Compliant Financial AI"
- "Explaining Financial Recommendations: Knowledge Distillation, Multi-Agent Reason Generation, and Regulatory Compliance"
- "Beyond SHAP: Structural Explainability and Agentic Reason Generation for Financial Product Recommendation"

## 전제
- 논문 1에서 제안한 이종 expert PLE+adaTT 모델이 학습 완료된 상태
- 논문 1의 gate weight 기반 구조적 설명 가능성을 활용
- 논문 1을 참조 ("our prior work [X]")

---

## Abstract (~250 words)
- 문제: 금융 추천에서 "왜 이 상품인가"를 고객/행원/감독기관에 설명해야 함. 기존 SHAP/LIME은 사후적이고 불안정. 모델 출력(확률)을 사람이 이해하는 언어로 변환하는 체계적 파이프라인이 부재.
- 제안: (1) PLE teacher → LGBM student 증류로 GPU-free 추론 (2) 피처 비즈니스 역매핑 + 3-agent 추천사유 생성 (3) Safety Gate로 규제 준수 자동 검증
- 결과: 증류 후 AUC drop < X%, 추천사유 human evaluation Y점, 금감원/EU AI Act 요건 전 항목 충족
- 의의: 예측→설명→설득까지의 full chain을 구현한 최초의 금융 추천 시스템

---

## 1. Introduction (~2 pages)

### 1.1 문제 정의
- 금융 추천의 최종 산출물은 확률이 아니라 "고객이 납득할 수 있는 이유"
- 설득의 3대 대상: 고객(신뢰→전환), 행원(영업 근거), 금감원(규제 준수)
- 기존 설명 가능성의 한계:
  - SHAP/LIME: 사후적, 불안정, 피처 이름이 비즈니스 의미 없음 ("feature_237이 0.12 기여")
  - 규칙 기반 템플릿: 유연성 부족, 맥락 무시
  - LLM 직접 생성: 할루시네이션 위험, 규제 위반 가능

### 1.2 핵심 질문
- Q1: 복잡한 DL 모델의 지식을 어떻게 경량 모델에 증류하면서 설명 가능성을 보존하는가?
- Q2: 피처 기여도를 어떻게 비즈니스 맥락의 자연어 설명으로 변환하는가?
- Q3: 생성된 추천사유가 규제 요건을 자동으로 준수하도록 어떻게 보장하는가?

### 1.3 Contributions
1. **IG-guided Knowledge Distillation**: teacher의 피처 중요도를 기준으로 student 입력 선택 → 설명력 보존
2. **Feature Business Reverse-Mapping**: 모든 피처에 비즈니스 맥락 역매핑 (interpretation_registry)
3. **3-Agent Reason Generation Pipeline**: 피처 선별 → 사유 생성 → 안전성 검증 역할 분리
4. **Regulatory Compliance by Design**: 금감원 가이드라인 + EU AI Act 요건을 파이프라인에 내장
5. **Human Evaluation Protocol**: 추천사유 품질의 체계적 평가 방법론

---

## 2. Related Work (~1.5 pages)

### 2.1 Knowledge Distillation for Recommendation
- Hinton (2015), CKD, 추천 특화 증류
- DL → tree-based 증류 사례

### 2.2 Recommendation Explanation Generation
- 템플릿 기반 vs 생성 기반
- LLM을 활용한 추천 설명 최근 연구
- 금융 도메인 특화 설명 생성

### 2.3 Responsible AI in Finance
- 금감원 AI 가이드라인 (2021) + 모형 리스크 관리 지침
- EU AI Act (2024) Article 13 투명성
- 한국 AI 기본법 (2024.12)
- FAccT/AIES 관련 연구

### 2.4 LLM Safety & Grounding
- 할루시네이션 방지
- RAG / grounding 기법
- Safety gate / content filtering

---

## 3. Knowledge Distillation (~2 pages)

### 3.1 Teacher-Student Architecture
- Teacher: PLE 14-task, 7 experts, 316 features (논문 1)
- Student: LGBM × 14 tasks, CPU 추론
- 제외된 4개 태스크 (income_tier, tenure_stage, spend_level, engagement_score): 피처의 deterministic 변환이므로 레이블로 부적합 (모델이 입력에서 완벽 복원 가능 → 리키지)
- [Figure 1: Teacher → Student 증류 구조]

### 3.2 IG-based Feature Selection
- Integrated Gradients로 teacher의 피처 중요도 계산
- 상위 k개 피처만 student에 전달
- ablation 결과(논문 1)와 IG 피처 중요도의 상관 분석
- [Table 1: IG top features vs ablation contribution ranking]

### 3.3 Soft Label Distillation
- Temperature scaling
- 태스크별 독립 증류 vs joint 증류
- [Table 2: 증류 후 AUC drop (태스크별)]

### 3.4 Teacher-Student Lifecycle
- Teacher: 주/월 1회 재학습 (SageMaker)
- Student: 일 1회 재증류 (새 soft label)
- Champion-Challenger 자동 비교
- [Figure 2: Teacher-Student 재학습 주기 다이어그램]

---

## 4. Recommendation Reason Generation (~3 pages, 핵심 섹션)

### 4.1 Feature Business Reverse-Mapping
- interpretation_registry: 피처명 → 비즈니스 맥락
- 매핑 구조: 카테고리, 방향성, 단위, 템플릿
- [Table 3: 피처 역매핑 예시 (10개)]
  ```
  hmm_lifecycle_prob_growing → "고객 성장 단계 확률" → "자산이 늘어나는 단계"
  mamba_temporal_d3 → "3개월 소비 트렌드" → "최근 소비가 증가하는 추세"
  ```

### 4.2 3-Agent Pipeline Architecture
- Agent 1 (Feature Selector): IG attribution + 설명력 관점 피처 보충 선별
- Agent 2 (Reason Generator): 역매핑된 맥락 → 자연어 추천사유
- Agent 3 (Safety Gate): 규제 위반/할루시네이션/부적절 투자 권유 검증
- [Figure 3: 3-Agent 파이프라인 상세 다이어그램]

### 4.3 Context Assembly
- 고객 프로필 맥락 + 피처 기여도 맥락 + 상품 속성 맥락
- Context vector store에서 유사 추천 사례 검색 (few-shot grounding)
- [Figure 4: Context assembly 흐름]

### 4.4 Safety Gate Design
- 위험성 점검: 부적절한 투자 권유 (적합성 원칙 위반)
- 규제 위반 점검: 금소법, 금융상품판매법
- 할루시네이션 점검: 없는 사실 언급, 수치 왜곡
- 통과 시 → 서빙, 불통과 시 → 템플릿 기반 안전 사유로 폴백
- [Table 4: Safety Gate 검증 항목 × 판단 기준]

### 4.5 Caching & Performance
- 동일 패턴 추천사유 캐싱 (reason_cache)
- LLM 호출 최소화 전략
- 캐시 적중률 분석

---

## 5. Regulatory Compliance (~2 pages)

### 5.1 금감원 가이드라인 매핑
- 설명 가능성, 공정성, 모형 검증, 모니터링, 감사 추적
- [Table 5: 금감원 요건 × 시스템 대응 상세]

### 5.2 EU AI Act 매핑
- Article 13 투명성, Article 14 인간 감독, Article 15 정확성/견고성
- [Table 6: EU AI Act 조항 × 시스템 대응]

### 5.3 한국 AI 기본법 (2024.12)
- 고위험 AI 분류 기준
- 금융 추천이 고위험에 해당하는지 분석

### 5.4 Monitoring & Governance
- 드리프트 감시 (PSI)
- 공정성 모니터링 (DI/SPD/EOD)
- 쏠림 탐지 (HHI/Gini/Entropy)
- 거버넌스 보고서 자동 생성
- Human-in-the-Loop 원칙
- [Figure 5: 모니터링/거버넌스 아키텍처]

---

## 6. Experiments (~3 pages)

### 6.1 Distillation Experiments
- AUC drop per task (teacher vs student)
- 피처 수 vs AUC trade-off curve
- [Table 7: 증류 결과 (teacher AUC / student AUC / gap)]

### 6.2 Reason Generation Quality
- Human evaluation protocol:
  - 평가자: 금융 도메인 전문가 N명
  - 평가 기준: 정확성, 자연스러움, 설득력, 규제 적합성
  - 비교 대상: (1) 템플릿 기반 (2) SHAP 기반 (3) 3-agent 기반
- [Table 8: Human evaluation 결과 (방식별 × 평가 기준)]
- 추천사유 샘플 제시 + 분석

### 6.3 Safety Gate Evaluation
- 위험 사유 탐지율 (recall)
- 오탐지율 (false positive)
- 폴백 발생 빈도
- [Table 9: Safety Gate precision/recall]

### 6.4 Serving Performance
- Lambda 추론 latency (추론 + attribution + 사유 생성)
- 캐시 적중률에 따른 latency 변화
- 비용 분석: Lambda vs 상시 GPU 서버
- [Table 10: 서빙 성능 + 비용 비교]

### 6.5 Regulatory Compliance Audit
- 금감원 요건 체크리스트 통과 여부
- 감사 로그 무결성 검증
- 공정성 메트릭 결과

---

## 7. Discussion (~1 page)

### 7.1 Findings
### 7.2 Practical Deployment Considerations
- 금융사 도입 시 LLM 선택 (온프렘 vs API)
- 추천사유 품질 유지 전략
- 규제 변화 대응

### 7.3 Limitations
- LLM 의존성 (비용, latency, 할루시네이션 잔존 위험)
- Human evaluation 규모 한계
- 한국/EU 외 규제 미검토

### 7.4 Future Work
- 실데이터 + 실고객 대상 A/B 테스트
- Multi-lingual 추천사유
- 규제 자동 업데이트 파이프라인

---

## 8. Conclusion (~0.5 page)

---

## Figures & Tables Summary

| # | Type | 내용 | 섹션 |
|---|------|------|------|
| Fig 1 | 다이어그램 | Teacher → Student 증류 구조 | 3.1 |
| Fig 2 | 다이어그램 | Teacher-Student 재학습 주기 | 3.4 |
| Fig 3 | 다이어그램 | 3-Agent 추천사유 파이프라인 | 4.2 |
| Fig 4 | 다이어그램 | Context assembly 흐름 | 4.3 |
| Fig 5 | 다이어그램 | 모니터링/거버넌스 아키텍처 | 5.4 |
| Tab 1 | 비교 | IG top features vs ablation ranking | 3.2 |
| Tab 2 | 결과 | 증류 AUC drop (태스크별) | 3.3 |
| Tab 3 | 예시 | 피처 역매핑 (10개) | 4.1 |
| Tab 4 | 기준 | Safety Gate 검증 항목 | 4.4 |
| Tab 5 | 매핑 | 금감원 요건 × 대응 | 5.1 |
| Tab 6 | 매핑 | EU AI Act × 대응 | 5.2 |
| Tab 7 | 결과 | 증류 teacher/student AUC | 6.1 |
| Tab 8 | 결과 | Human evaluation | 6.2 |
| Tab 9 | 결과 | Safety Gate precision/recall | 6.3 |
| Tab 10 | 비교 | 서빙 성능 + 비용 | 6.4 |

---

## 예상 분량
- 본문: ~14 pages
- Appendix: ~3 pages
- References: ~35 papers
- 총: ~17 pages

## 타겟 학회/저널
- **FAccT** (ACM Fairness, Accountability, Transparency): 공정성/투명성 중심
- **AAAI** (AI for Financial Services 워크숍): 금융 AI 특화
- **CIKM** (ACM): 정보/지식 관리 — 추천+설명
- **Expert Systems with Applications** (Elsevier): 산업 응용 저널
