// ============================================================
// Paper 3: 대규모 MTL에서의 손실 동역학과 게이트 선택
// ============================================================

#set document(
  title: "동질 태스크를 넘어선 멀티태스크 학습 확장: 12-태스크 금융 추천에서의 손실 동역학과 게이트 선택",
  author: ("Seonkyu Jeong", "Euncheol Sim", "Youngchan Kim"),
)

#set page(
  paper: "us-letter",
  margin: (x: 1.8cm, y: 2cm),
  numbering: "1",
)

#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.6em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

// Bibliography setup
#set bibliography(style: "ieee")

// Title
#align(center)[
  #text(size: 14.5pt, weight: "bold")[
    동질 태스크를 넘어선 멀티태스크 학습 확장: \
    12-태스크 금융 추천에서의 \
    손실 동역학과 게이트 선택
  ]

  #v(0.8em)

  #text(size: 11pt)[
    Seonkyu Jeong#super[1], Euncheol Sim#super[1], Youngchan Kim#super[1]
  ]

  #v(0.3em)

  #text(size: 9pt, style: "italic")[
    #super[1]Independent Research \
    교신저자: Seonkyu Jeong (ORCID: #link("https://orcid.org/0009-0005-3291-9112")[0009-0005-3291-9112])
  ]

  #v(1em)
]

// Abstract
#block(
  width: 100%,
  inset: (x: 1em),
)[
  #text(weight: "bold")[초록.]
  MMoE와 PLE 같은 멀티태스크 학습(MTL) 아키텍처는 거의 전적으로
  2--4개의 동질 태스크(예: CTR + CVR)에서만 검증되어 왔다.
  본 연구에서는 PLE를 *13개의 이종 태스크*
  --- 7개 이진, 3개 다중분류, 3개 회귀 ---
  로, 구조적으로 서로 다른 7개의 전문가와 100만 명의 합성 고객을 가진
  금융 상품 추천 시스템에서 확장하며 얻은 실증적 발견을 보고한다
  (최종 v14 벤치마크는 누수 스캔으로 segment_prediction 을 제외한
  12개 --- 7 binary, 2 multiclass, 3 regression --- 를 사용하며,
  각 Finding 의 태스크 구성은 해당 실행 시점을 따른다).
  14개의 발견은 하나의 줄기로 모인다 --- *MTL을 동질 태스크 영역 너머로
  밀어붙일 때 무엇이 조용히 실패하며, 그것을 잡아내려면 어떤 측정 규율이
  필요한가* --- 이를 네 가지 주제로 정리한다.
  *손실 동역학과 게이팅* (Finding 1--6): uncertainty-weighting 구현 버그가
  소수 유형(minority-type) 태스크를 조용히 억제하며(수정 시 +0.018 NDCG\@3),
  softmax 게이팅이 이종 태스크 혼합에서 sigmoid를 능가하여
  동질 설정에서의 우위를 *역전*시킨다. 학습된 uncertainty weight는 아키텍처 전반에서
  동일하게 수렴하고, shared-bottom 베이스라인은 확장된 에폭 예산에 과적합하는 반면
  게이트형 PLE 변형은 정규화 없이 이를 흡수한다. GroupTaskExpert 사전 게이팅은
  혼합 유형 그룹에서 다중분류 성능을 저하시키며, 게이트 엔트로피 분석은
  추출 계층(extraction-layer)의 전문화(엔트로피 비율 0.33--0.88)를 보이는 한편
  attention 수준 집계는 균등 평균으로 붕괴하고(비율 1.00), 회귀 태스크가 존재하면
  복합 검증 손실(composite val-loss)은 신뢰할 수 없는 체크포인트 신호가 된다.
  *Fusion augmentation 트레이드오프* (Finding 7): 9-way 비교가 분리된 축(disjoint axes)에서
  두 가지 긍정적 레시피를 분리해 낸다 --- gradient isolation을 동반한 출력 공간 부스팅
  (BRP-detached, 어려운 태스크 구제)과 inverse-gate 보조 감독(NEAS, aggregate-AUC 향상) ---
  그러나 이 둘은 가산적으로 *결합되지 않는다*.
  *인과 전문가 재해석* (Finding 8--13): 인과 전문가의 인접 행렬 $bold(W)$가
  검토한 모든 학습 체크포인트에서 0으로 붕괴하여, 그 forward를 평범한 MLP와 동등하게 만든다.
  2단계 패치(NOTEARS reconstruction loss + 초기화 재스케일)는 태스크 메트릭 비용 없이
  DAG 학습을 복원한다. 기능적 DAG를 소비 가능한 출력으로 라우팅하면
  Causal Explainability Head(CEH, 샘플 단위 attribution, Pearl Rung 1)와
  Causal Guardrail(CG, 3개의 합성 프로브에서 100% TPR / 5% FPR로 z-공간-Mahalanobis OOD 탐지)이 얻어진다.
  init $0.1 arrow 0.3$ + $lambda_"recon" 0.5 arrow 2.0$ 를 통한 W-증폭은
  주 태스크 비용 없이 인접 행렬을 Frobenius 노름 기준 $14 times$ 키워,
  "장식용 DAG(decorative DAG)"가 아키텍처 제약이 아니라 학습 선택의 산물임을 입증한다.
  또한 반사실 프로브(CCP, Pearl Rung 3)는 베이스라인 W 스케일에서 DAG가
  반사실 효과의 $0.16%$만 운반하지만 증폭된 스케일에서는 매개 비율(mediation ratio)이
  중앙값 $32%$, 95번째 백분위수 $61%$로 상승함을 보인다 --- 이는 검증된
  결과가 아니라 향후 과제로 남기는 예비 신호로서, Pearl의 Rung 3가 증폭된
  교사 위에서 성립 가능할 수 있음을 시사한다. 그리고 인과 인코더 출력 타깃을
  특정 task-logit gradient 타깃으로 대체한 CEH v3 변형은 정직한 음성 결과다 ---
  학습 신호에도 불구하고 헤드는 전역 중요도 패턴으로 재붕괴하며,
  attribution의 타깃 설계가 v1$arrow$v2 전환이 시사한 것보다 더 민감함을 보여준다.
  *서빙 측 증류 제어* (Finding 14): 12-태스크 교사를 태스크별 gradient-boosted 학생으로 증류하면
  운영 게이트를 통과하지만(AUC 격차 $<= 0.0125$, 학생 보정 오차 $<= 0.0114$),
  행동 유사성은 구조적 하한을 보인다(교사--학생 일치도 $0.75$--$0.82$,
  증류된 7개 태스크 전체에서 ranking correlation $0.78$--$0.91$). 이 하한은
  동일한 예측에 대해 채점된 세 세대의 게이트 의미론 전반에서 불변이다.
  따라서 두 메트릭 계열에 걸친 단일 계층(single-tier) 게이트는 항상 실패하며 ---
  항상 실패하는 게이트는 우회를 유발하므로 --- 제어는 차단형 운영 계층과
  정보 제공형 행동-진단 계층으로 분리된다.
  우리는 이러한 관찰을 동질 태스크 영역을 넘어 MTL을 확장하는 실무자를 위한
  실천적 가이드라인으로 정리한다.

  #v(0.3em)
  #text(weight: "bold")[키워드:]
  멀티태스크 학습, Progressive Layered Extraction, Mixture of Experts,
  gate selection, uncertainty weighting, 손실 동역학, 이종 태스크
]

// Switch to 2-column layout for body
#show: rest => columns(2, rest)

= 서론

멀티태스크 학습(multi-task learning, MTL)은 관련 태스크를 공동으로 최적화함으로써
파라미터 효율성과 긍정적 전이(positive transfer)를 약속한다 @caruana1997.
추천 시스템에서 MTL은 표준적 관행으로 자리 잡았다:
MMoE @ma2018 는 태스크 충돌을 다루기 위해 multi-gate mixture-of-experts를 도입하였고,
PLE @tang2020 는 공유 전문가와 태스크별 전문가를 분리하는 progressive extraction layer를 추가하였으며,
AdaTT @li2023 는 적응적 태스크 간 전이를 가능하게 하였다.

그러나 추천을 위해 발표된 거의 모든 MTL 아키텍처는
*2--4개의 동질적 태스크* --- 일반적으로 클릭률(click-through rate, CTR)과
전환율(conversion rate, CVR)이며, 둘 다 기울기 방향이 정렬된
이진 분류 문제이다 --- 위에서 검증된다.
우리가 PLE를 *13개의 이종 태스크*
--- 이진 7개, 다중클래스 3개(4~50개 클래스), 회귀 3개 ---
로 확장하려 시도했을 때, 기존 문헌이 다루지 않는 방식으로
여러 가정이 무너졌다.

13개 태스크 구성은 설계상의 선호가 아니라 제약 조건이었다.
금융 규제는 서로 구별되는 예측 대상을 의무화한다:
적합성 평가, 보호 속성 전반에 걸친 공정성 모니터링,
이탈 조기경보, 상품 단위 가입 성향이
각각 별도의 지도학습 신호를 요구한다.
특히 2026년 6월 22일 시행된 금융위 「금융분야 인공지능
가이드라인」@koreafsc2024 은 신뢰성 원칙(§4.4)에서 전역(global),
국소(local) 설명가능성을, 보호 속성 기반 공정성 모니터링을
권고하여, 이러한 구별되는 예측 대상의 규제적 동기를 시행된 규범으로
뒷받침한다(다만 가이드라인은 자율 적용이며 특정 태스크 구성을
강제하지는 않는다).
한편 제한된 인프라 --- 단일 데스크톱 GPU(12GB VRAM)와
3인 팀 --- 는 태스크별로 별도 모델을 유지하는 것을 불가능하게 한다.
그 결과는 대규모 CTR 팀은 진입할 이유가 없지만
(이들은 태스크별 모델을 감당할 수 있다) 자원이 제약된 규제 산업은
어쩔 수 없이 들어가게 되는 영역(regime)이다.

본 논문은 이러한 확장 경험에서 얻은 14가지 실증적 발견을
네 가지 주제로 정리하여 보고한다: 손실 동역학과 게이팅
(Findings 1--6), 융합 증강 트레이드오프(Finding 7),
인과 전문가 재해석(Findings 8--13), 그리고 서빙 측
증류 제어(Finding 14). 우리는 state-of-the-art 성능을
주장하지 않는다. 대신 MTL이 동질적 태스크 영역을 넘어
밀어붙여졌을 때 나타나는 *현상과 실무적 지침*을 기록한다.

*동반 논문 대비 위치.*
본 연구는 동일한 12개 태스크 벤치마크, 동일한 7개 전문가 PLE 백본,
동일한 v14 phase0 데이터 파이프라인을 공유하는 두 동반 논문 사이에 위치한다.#footnote[여기서 "12개 태스크"와 "7개 전문가"는 합성 v14 벤치마크 기준이다. 운영 배포는 별개의 태스크 명명 체계를 쓰며 활성 태스크 수가 라벨 가용성에 따라 변동하고(2026-07 실행 기준 11개), 전문가도 LightGCN 신호를 unified HGCN 입력으로 흡수한 6-전문가 변형으로 가동된다 --- Paper 1의 운영 데이터 예비 검증 절 참조.]
본 논문이 맡는 줄기는 좁고 분명하다: *MTL을 동질 태스크 영역 너머로
밀어붙일 때 무엇이 조용히 실패하며, 그것을 잡아내는 측정 규율은 무엇인가.*
Paper 1 (Heterogeneous Expert PLE: An Explainable Multi-Task
Architecture for Financial Product Recommendation)#footnote[DOI: #link("https://doi.org/10.5281/zenodo.19621884")[10.5281/zenodo.19621884]] 은
아키텍처와 주요 어블레이션 발견의 시스템 수준 요약을 제시한다.
본 논문은 그 요약의 *상세 실증 동반판*으로서, (a) A--E 전반에 걸친 완전한
가설 식별(신호 정제, epoch 예산, 앙상블, 게이트 과적합,
정규화)과 12개 태스크 전체의 태스크별 v14
수치(Findings 1--6) --- Paper 1과 겹치는 부분은 수치를 다시 도출하지 않고
Paper 1을 인용한다 --- (b) "CGC 위에서 어떤 종류의
융합 증강이 작동하는가"의 설계 공간을 분리하는 9방향 융합
메커니즘 비교(Finding 7), (c) Paper 1에는 대응물이 없는
인과 전문가 재해석 흐름(Findings 8--13), 그리고 (d)
서빙 스택의 증류 충실도 게이트에 대한 측정 측 분석(Finding 14)을 제공한다.
이 게이트의 거버넌스 대응물 --- 감사 추적 의미론과 운영자 override
리스크 --- 는 Paper 2의 fidelity-gate 의미론 논의에서 다룬다.
Paper 2 (From Prediction to Persuasion: Agentic Recommendation Reason
Generation for Regulatory-Compliant Financial AI)#footnote[DOI: #link("https://doi.org/10.5281/zenodo.19622052")[10.5281/zenodo.19622052]] 는
자립적이다: 그 GDPR 제22조 / EU AI Act 제13조 논증은 증류된 학생의
SHAP attribution, 3-에이전트 추천사유 파이프라인, 그리고 HMAC 서명
해시 체인 감사 로그가 떠받치며, 본 논문의 어떤 출력에도 의존하지 않는다.
다만 우리 발견 중 두 가지는 그 감사 로그에 *선택적 포렌식 보강*으로
제공될 수 있다 --- Causal Explainability Head가 생산하는 표본별 피처
attribution 벡터(Finding 9)와 Causal Guardrail이 생산하는 예측별 신뢰성
점수(Findings 10--11)이며, Paper 2는 이를 `log_attribution` /
`log_guardrail` 레코드로 라우팅할 수 있다. 이는 의존이 아니라 보강임을
분명히 한다: 예측별 인과 감사는 서빙되는 학생이 아니라 PLE 교사에서
돌기 때문에, 이를 프로덕션 트래픽에 배선하는 것 --- 그리고 신뢰성 신호를
날카롭게 할 W-증폭(Finding 11) --- 은 제공된 기능이 아니라 향후 과제다.

우리의 기여:

- Kendall 등의 uncertainty weighting이 태스크별 손실 가중치가
  생략되었을 때 어떻게 조용히 실패하는지에 대한 진단(Section 4.1).
- *게이트 유형 선택이 아키텍처의 정교함이 아니라 태스크 유형의 동질성에
  좌우된다*는 증거(Section 4.2).
- uncertainty 가중치가 아키텍처 전반에 걸쳐 동일하게 수렴하여
  그 보호 역할이 제한됨을 보이는 실증(Section 4.3).
- 구조 비교에서 epoch 예산 민감도 분석(Section 4.4).
- 혼합 유형 그룹에 대한 사전 게이팅 태스크 그룹화(GTE)에 관한
  경고성 발견(Section 4.5).
- CGC extraction-layer 전문화가 실재하며 태스크 의존적인 반면
  attention 수준 집계는 균일 평균화로 붕괴함을 보이는 게이트 엔트로피
  분석, 그리고 회귀와 분류 태스크가 공존할 때 val-loss가
  오해를 일으키는 체크포인트 기준임을 보이는 실증
  (Section 4.6).
- CGC 기준선 위에서 융합 증강의 9방향 비교로,
  *서로소(disjoint)인 축 위의 두 가지 긍정적 레시피*를 식별하고
  이들이 *가산적이지 않음*을 보인다. 다섯 가지 표현 가산형
  융합(loss-level adaTT, AdaTT-sp, complementary-gate recovery,
  uncertainty-conditioned expert bank, MV BRP)은 모두 개입의
  침습성에 따라 AUC를 단조적으로 저하시킨다. *공유 전문가 기울기
  분리를 동반한 출력 공간 부스팅*(BRP-detached)은 종합 AUC에서
  CGC와 동률이며($Delta = -0.0007$; best epoch는 기준선을
  $+0.0008$ 초과) F1 macro를 $+0.007$, NDCG\@3 를 $+0.015$
  끌어올리고 가장 어려운 다중클래스 태스크에서 $+$256%의 상대적
  구제를 유지한다. *학습 시점 부하 분산 정규화*(NEAS ---
  inverse-gate 집계에 대한 보조 지도)는 종합 AUC를 실제로
  높인 이 계열 최초의 메커니즘이며($Delta = +0.0011$),
  단조 증가 궤적과 거의 균일한 태스크별 향상을 보인다.
  두 긍정적 레시피를 쌓으면 NEAS의 AUC 이득이 붕괴하는데,
  공유 전문가가 일반론자(NEAS)인 동시에 1차-보조 전문가
  (BRP-detached)일 수 없기 때문이다. 지침은 목적별로 다르다:
  종합 AUC와 태스크 간 견고성에는 NEAS, 어려운 태스크 구제에는
  BRP-detached이며, 둘은 쌓지 않는다(Section 4.7).
- 인과 전문가의 학습 가능한 인접 행렬 $bold(W)$ 가 우리가 검토한
  모든 학습된 체크포인트(로컬 4개, 온프렘 업스트림 2개)에서
  0으로 붕괴하여, NOTEARS 정규화에도 불구하고 전문가의
  forward pass가 평범한 MLP와 동등해진다는 것을 보이는 구조적
  진단. 이 실패는 saddle-point 문제이며(task-loss와 reconstruction
  기울기 모두 초기화 스케일에서 사라지는 $bold(W)$ 인자를 가진다),
  두 부분으로 구성된 패치 아래에서 해결된다: 원논문의
  reconstruction 항($||bold(z) - bold(z) bold(W)^2||_F^2$)을 추가하고
  초기화를 $0.01$ 에서 $0.1$ 로 재조정한다. 패치 이후 전문가는
  유효한 희소 DAG($bold(W)$ Frobenius $0.34$, $7.3%$ 엣지 활성,
  $h(bold(W)) = 0$)를 학습하지만, 종합 태스크 지표는 변하지 않는다 ---
  DAG는 구조적으로 존재하지만 현재 아키텍처에서는 예측으로
  라우팅되지 않는다(Section 4.8).
- 이제 기능하는 DAG를 소비 가능한 출력으로 *라우팅*하는 두 가지
  Axis-3 후보:

  Causal Explainability Head (CEH)는 인과 전문가의 출력을
  표본별 피처별 attribution 벡터로 매핑하는 작은 MLP로,
  gradient $times$ input 에 대한 MSE로 학습된다.
  MV 결과는 1차 AUC를 잡음 범위 내에서 보존하고
  (패치 이후 softmax 기준선 대비 $+0.0015$), DAG 자체를
  미미하게 강화하며($bold(W)$ Frobenius $0.338 arrow 0.366$, 희소 엣지
  $7.3% arrow 8.5%$), 다운스트림 감사 로그 영속화를 위한 실시간
  attribution 벡터를 생산한다. 사후 품질 평가는 원시 grad $times$
  input 타깃 아래에서 head가 거의 전역적인 중요도 벡터로
  붕괴함을 드러냈다(표본 간/표본 내 분산비 $0.055$, 표본 전반
  상위 10개 피처 중첩 $0.791$). 최소 개입 반복(v2, "demeaned target")은
  지도 신호에서 배치 평균을 빼서 1차 태스크를 교란하지 않으면서
  표본별 식별력을 복원한다(분산비 $0.055 arrow 0.719$, 상위 10개
  중첩 $0.791 arrow 0.281$, 1차 AUC는 잡음 범위 내에서 불변,
  Section 4.9.4). 추가 반복(v3, "primary-task-logit target")은
  attribution 지도 신호를 모델 자신의 logit으로 대체하여
  head를 v1 기준선 아래로 다시 붕괴시켰는데
  (분산비 $0.719 arrow 0.043$), 이는 demeaned target이 견고하게
  효과적이라기보다 좁게 효과적임을 입증한다
  (Section 4.13). 감사 로그 통합은 이후 구현되었다 ---
  표본별 attribution 벡터는 동반 서빙 스택(Paper 2 v2)의
  `AuditLogger.log_attribution` 으로 공급된다 ---
  반면 교차 데이터셋 재현은 여전히 보류 상태이다.

  Causal Guardrail (CG)은 인과 전문가의 잠재 $bold(z)$ 에서
  도출되는 예측별 신뢰성 플래그이다. W-reconstruction
  공식(CG v1)은 학습된 $bold(W)$ 가 너무 작아 우연 수준의 식별에서
  실패한다. z-공간 Mahalanobis 공식(CG v2)은 세 가지 유형의
  합성 OOD를 $100%$ TPR과 $5%$ FPR로 탐지한다(Section 4.10).
  후속 W-amplification 실험(W init $0.1 arrow 0.3$, $lambda_"recon"
  0.5 arrow 2.0$)은 1차 태스크 비용 없이 인접 행렬을 Frobenius
  norm 기준 $14 times$, 활성 엣지를 $8.5%$ 에서 $59.5%$ 로 키워,
  CG v1을 부분적으로 고치되 잠재 공간 CG v2를 여전히 우위로
  남긴다(Section 4.11). Finding 11의 확정된 결과: Finding 8의
  "장식용 DAG"는 아키텍처적 제약이 아니라 학습 선택의
  산물이다.

- Pearl Rung 3을 직접 시험하는 Counterfactual Probe(CCP, Finding 12):
  $"do"(z_j = v)$ 개입 아래에서 amplified DAG는 반사실 효과의
  중앙값 $32%$ 와 95번째 백분위수 $61%$ 를 운반하는 반면, 기준
  교사에서는 $0.16%$ 에 그친다 ---
  이는 $200 times$ 의 도약으로, 우리는 이를 amplified 교사 위에서의 Rung 3에
  대한 *예비 실현가능성 신호*로 읽으며, 검증된 결과가 아니라 향후 과제로
  보고한다. 기준 체크포인트에서 DAG는
  수치적으로 Rung 3 주장을 지지할 수 없다(Section 4.12).

- v2의 demeaned grad $times$ input 지도를 모델 자신의 1차 태스크
  logit 기울기로 타깃을 대체하는 CEH v3 변형(Finding 13,
  *정직한 부정 결과*); head는 전역 중요도 패턴으로 재붕괴하며
  (분산비 $0.719 arrow 0.043$), 이는 v1$arrow$v2
  개선이 견고한 설계 원리라기보다 좁게 타깃 의존적임을 입증하고
  CEH 탐색의 해당 분기를 종결한다(Section 4.13).

- *동일한* 교사--학생 예측에 대해 모두 채점된, 세 세대에 걸친
  연속적인 증류 충실도 보고서(Finding 14)에 대한 재분석으로,
  교차 아키텍처 PLE $arrow.r$ LightGBM 쌍에 내재된
  행동 유사성 하한(7개 증류 태스크 전반에서 교사--학생 일치도
  $0.75$--$0.82$, 순위 상관 $0.78$--$0.91$)을
  깔끔하게 통과하는 운영 지표(AUC 격차 $<= 0.0125$, 학생 보정
  오차 $<= 0.0114$) 아래에서 분리해 낸다. 그리고 충실도 게이트가
  차단형 운영 계층과 정보성 행동 계층으로 분할되어야 한다는
  제어 설계상의 결론 ---
  항상 실패하는 게이트는 우회되는 게이트이기 때문이다
  (Section 4.14).

본 시스템, 데이터 생성기, 어블레이션 스크립트는 공개되어 있다.#footnote[
  https://github.com/bluethestyle/aws\_ple\_for\_financial
]

= 관련 연구

== 추천에서의 멀티태스크 학습

Shared-bottom 네트워크 @caruana1997 는 태스크가 충돌할 때 부정적 전이를 겪는다.
MMoE @ma2018 는 공유 전문가 풀에 대한 태스크별 게이팅으로 이를 완화하며,
PLE @tang2020 는 공유 extraction layer와 태스크별 extraction layer를 한층 더 분리한다.
AdaTT @li2023 는 적응적 태스크 간 전이 강도를 추가한다.

공통된 맥락: 모든 평가가 *동일한 유형*의 2--4개 태스크를 사용한다.
MMoE의 Census 실험은 이진 태스크 2개를 사용한다.
Tencent에서 수행된 PLE의 프로덕션 평가는 태스크 2개(CTR + VCR)를 사용한다.
AdaTT의 Alibaba 실험은 밀접하게 관련된 engagement 태스크 3개를 사용한다.
이진, 다중클래스, 회귀가 혼합된 태스크를 우리가 보고하는 규모로
평가한 발표된 PLE/MMoE 연구는 존재하지 않는다.

== MTL에서의 손실 균형

Kendall 등 @kendall2018 은 등분산(homoscedastic) uncertainty weighting을 도입하여,
손실 스케일을 자동으로 균형 맞추는 태스크별 정밀도(precision) 파라미터를 학습한다.
GradNorm @chen2018gradnorm 은 기울기 norm에 기반하여 손실 가중치를 동적으로 조정한다.
MGDA @sener2018multi 는 각 스텝에서 다목적 최적화 문제를 푼다.

이 방법들은 모든 태스크가 유사한 손실 크기를 공유하는 시나리오를 위해
설계되고 그 위에서 시험된다. 이진 교차 엔트로피(스케일 ~0.5),
다중클래스 교차 엔트로피(50개 클래스에 대해 스케일 ~3.9),
회귀 MSE(스케일 ~0.01--1.0)가 공존할 때, 이 균형
방법들의 암묵적 가정은 재검토할 가치가 있다.

== 게이트 설계: Softmax vs.\ Sigmoid

표준 PLE와 MMoE는 *softmax* 게이트를 사용하여, 경쟁적이고 합이 1이 되는
전문가 선택을 강제한다. Nguyen 등 @sigmoid_moe2024 은 sigmoid 게이팅
--- 각 전문가가 경쟁 없이 독립적으로 기여하도록 허용 ---
이 전문가 간 경쟁을 제거함으로써 더 높은 표본 효율성을 달성함을 입증하였다.

그러나 이 발견은 동질적 태스크 집합 위에서 확립되었다.
우리는 태스크가 이종일 때 sigmoid 이점이 *역전*됨을 보이는데,
독립적 전문가 활성화가 기울기가 큰 이진 태스크로 하여금
다중클래스 태스크가 의존하는 전문가를 오염시키도록 허용하기 때문이다.

= 아키텍처

== 이종 전문가 바스켓을 갖춘 PLE

본 연구의 PLE 구현은 Tang et al. @tang2020 을 따르되 핵심적인 수정을 가한다.
$K$개의 동일한 MLP 전문가 대신, 각각 서로 다른 귀납 편향(inductive bias)을
인코딩하는 7개의 *구조적으로 이질적인* 전문가를 사용한다.

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 5pt,
    align: (left, left, right),
    stroke: 0.5pt,
    table.header([*Expert*], [*Architecture*], [*Input Dim*]),
    [DeepFM], [Factorization Machine + DNN], [977D],
    [Temporal], [Mamba + LNN + Transformer], [116D],
    [HGCN], [Hyperbolic GCN (Poincaré)], [58D],
    [PersLay], [Topological (TDA)], [32D],
    [Causal], [NOTEARS DAG], [129D],
    [LightGCN], [Graph Convolution], [955D],
    [Optimal Transport], [Sinkhorn matching], [95D],
  ),
  caption: [전문가 바스켓. 각 전문가는 FeatureRouter를 통해 서로 다른
  피처 부분집합을 받는다. 전체 입력 공간: 1211D (Phase 0 v3/v4, 17 groups).],
) <tab:experts>

*FeatureRouter* 는 각 전문가에게 지정된 피처 그룹을 할당하며,
이는 하드코딩이 아니라 YAML 구성으로 선언된다.
전문가별 입력 차원의 합(2419D)이 전체 피처 공간(1211D)을 초과하는 이유는,
여러 피처 그룹이 상호 보완적인 귀납 편향을 가진 전문가들 사이에서 공유되기 때문이다.

== CGC 게이팅

Customized Gate Control(CGC) 모듈은 각 태스크 $t$ 에 대해 $K$개의 전문가에 걸친
어텐션을 계산한다.

$ g_t = "softmax"(W_t dot.c h_t) in RR^K $
또는
$ g_t = sigma(w_t dot.c h_t) slash sum_j sigma(w_j dot.c h_j) $

여기서 $h_t$ 는 공유 표현(shared representation)이다.
*Softmax* 게이트는 경쟁적이고 합이 1이 되는 할당을 강제한다.
*Sigmoid* 게이트는 독립적 평가를 허용하며 사후에 정규화한다.
이 둘 사이의 선택은 본 논문의 핵심 발견 중 하나이다 (Section 4.2).

== Uncertainty Weighting

Kendall et al. @kendall2018 을 따라, 태스크별로 로그 분산 파라미터
$log sigma^2_t$ 를 학습한다. 가중 손실은 다음과 같다.

$ cal(L)_t^"uw" = w_t dot.c (1 / (2 sigma_t^2) dot.c cal(L)_t + 1/2 log sigma_t^2) $ <eq:uw-correct>

여기서 $w_t$ 는 구성에서 정의되는 태스크별 손실 가중치이다.
정밀도(precision) $1/(2 sigma_t^2)$ 와 정규화항 $1/2 log sigma_t^2$ 는
안정적인 범위로 클램핑된다.
$log sigma^2_t in [-4, 4]$, precision $in [10^(-3), 100]$.

결정적인 세부 사항은 $w_t$ 가 *전체 표현식에 곱해져야 한다*는 점이며,
$cal(L)_t$ 에만 곱해서는 안 된다. $w_t$ 를 누락한 것이 Section 4.1 에서 보고하는 버그이다.

== AdaTT 태스크 그룹

태스크는 4개의 Financial DNA 그룹으로 조직된다.

#figure(
  table(
    columns: (auto, 1fr, auto),
    inset: 5pt,
    align: (left, left, left),
    stroke: 0.5pt,
    table.header([*Group*], [*Tasks*], [*Type mix*]),
    [Engagement], [next\_mcc, top\_mcc\_shift], [1 mc, 1 bin],
    [Lifecycle], [churn\_signal, product\_stability, segment\_prediction#super[†]], [1 bin, 1 reg, 1 mc#super[†]],
    [Value], [mcc\_diversity\_trend], [1 reg],
    [Consumption], [nba\_primary, cross\_sell\_count, will\_acquire\_\* (5 tasks)], [1 mc, 1 reg, 5 bin],
  ),
  caption: [AdaTT 전이를 위한 Financial DNA 태스크 그룹화로,
  `configs/santander/pipeline.yaml` 의 `task_groups` 를 따른다.
  표의 13-태스크 구성(7 binary, 3 multiclass, 3 regression)은
  v12 계열 실행(Finding 1--3, 5 및 게이트 엔트로피 분석)에 사용된 것이다.
  v13 전처리 개정의 누수 스캔에서 `segment_prediction`(†)이 제거되어,
  v13/v14 실행(Finding 4, 6--7 및 Finding 14 증류 라운드)은 나머지
  12개 태스크(7 binary, 2 multiclass, 3 regression)를 사용한다.],
) <tab:taskgroups>

AdaTT @li2023 는 그룹 내 및 그룹 간 전이 강도를 학습한다.
Consumption 그룹은 5개의 이진 태스크를 하나의 멀티클래스 및 하나의 회귀 태스크와
혼합하고 있는데 --- 이 설계는 Section 4.5 에서 관련성을 갖게 된다.

= 결과 및 분석

모든 실험은 *benchmark_v12* 합성 생성 스키마를 공유한다.
고객 100만 명이며, 생성 스키마는 13개 태스크(7 binary, 3 multiclass, 3 regression)를 정의한다.
v13 전처리 개정부터는 누수 위험으로 `segment_prediction` 이 제외되어,
v13/v14 실행은 12개 태스크(7 binary, 2 multiclass, 3 regression)로 학습한다. 본 논문에서는
세 가지 버전 라벨이 반복적으로 등장하며 서로 혼동되기 쉬우므로, 여기서 그
지시 대상을 명확히 한다. *benchmark_v12* 는 데이터 *생성(generation)* 스키마를
가리킨다. *Phase 0 v3/v4* 는 *피처 엔지니어링 빌드(feature-engineering build)*
버전(Section 3 에서 설명한 1211-피처 / 17-그룹 빌드)을 가리킨다. *v13/v14
phase0* 은 벤치마크가 SageMaker 파이프라인으로 이행하면서 적용된 연속적인
*전처리 개정(preprocessing revisions)* 을 가리킨다. v13 은 세 가지 결함(HMM
상태 중복, GMM $K = 20$ 의 죽은 클러스터, 확률 컬럼이 스케일러를 통과한
문제)을 포함하며, v14 가 이를 수정한다(HMM mode-split, GMM $K = 14$, 확률
컬럼을 스케일링에서 제외, 1210 features).

표에는 두 개의 실행 계열(run family)이 등장하며, 모든 캡션에는 그 출처가
라벨로 표기되어 있다. *Local v12 runs* (Findings 1--3 및 5; RTX 4070, 실행
아티팩트 기준 349 model-input features)은 batch size 5632, learning rate
0.0005, AMP (FP16), warm restart를 갖춘 cosine annealing($T_0 = 10$),
uncertainty weighting을 사용하며, 표별로 명시된 대로 10 또는 20 epochs 로
실행되었다. *v14 phase0 SageMaker runs* (Findings 4, 6, 7;
ml.g4dn.2xlarge, fp32 DuckDB streaming)는 명시된 대로 15 또는 30 epochs 로
실행된다. Findings 8--13 은 local v12 체크포인트에 대한 진단 분석과 명시된
타겟 SageMaker 재실행으로 구성되며, Finding 14 는 v14 증류 라운드의
fidelity 보고서를 재분석한다. 본 논문에서 보고하는 모든 실행은 단일 시드(42)
기반이며, 시드 간 비교 주장은 하지 않는다(Limitations 참조).
지표는 태스크 유형별로 보고한다: Avg AUC (binary), Avg F1-macro
(multiclass), Avg MAE (regression), NDCG\@3 및 Acc\@3 (ranking).

== Finding 1: 침묵하는 Uncertainty Weighting 버그 <find1>

Kendall et al. 의 uncertainty weighting을 온프렘 코드베이스에서
AWS 구현으로 이식하는 과정에서 미묘한 버그가 유입되었다.
온프렘 코드는 @eq:uw-correct 를 올바르게 구현한다.

```python
# On-premises (correct)
loss = loss_weight * (precision * task_loss + log_var)
```

AWS 이식 버전은 태스크별 `loss_weight` 와 클램핑을 누락하였다.

```python
# AWS port (buggy)
loss = precision * task_loss + log_var / 2
```

이 누락은 두 가지 결과를 초래한다.
(1) `pipeline.yaml` 의 `loss_weight` 파라미터 --- 이진(~0.5)과 50-클래스
멀티클래스(~3.9) 태스크 사이의 cross-entropy 스케일 차이를 보정한다 ---
가 조용히 무시된다.
(2) 클램핑이 없으면 극단적인 로그 분산 값이 정밀도를 수치적으로
불안정한 영역으로 밀어 넣을 수 있다.

그 결과, 원시 손실이 이진 태스크보다 ~8배 큰 멀티클래스 태스크는
정밀도 파라미터가 유사한 값으로 수렴할 때 비례적으로 *더 적은* 그래디언트를
받게 된다.
수치적으로 우세한(13개 중 7개) 이진 태스크는 순수한 그래디언트 양만으로도
멀티클래스 학습을 추가로 억제한다.

#figure(
  table(
    columns: (auto, auto),
    inset: 5pt,
    align: (left, right),
    stroke: 0.5pt,
    table.header([*Metric*], [*$Delta$ (fixed $-$ buggy)*]),
    [Avg F1-macro (multiclass)], [+0.031],
    [NDCG\@3], [+0.018],
  ),
  caption: [Uncertainty-weighting 버그 수정의 영향(shared-bottom
  baseline, 초기 benchmark_v12 ablation 라운드). 프로젝트 실험 기록에는
  영향받은 두 지표 계열의 델타만 보존되어 있었으며, 버그 구성의 실행
  아티팩트는 보존되지 않았으므로 절댓값은 재현하지 않는다. Binary AUC 는
  본질적으로 영향을 받지 않았다 --- 이진 태스크는 억제된 유형이 아니었다.
  두 델타 모두 ablation 연구에서 측정된 어떤 아키텍처 수준의 변화보다도 크다.],
) <tab:bugfix>

*교훈*: Uncertainty weighting 구현은 수렴 여부만 테스트할 것이 아니라
*원래의* 수학적 정식화에 대해 검증되어야 한다.
버그가 있는 버전도 여전히 수렴한다 --- 다만 다수 태스크 유형에 유리한
차선의 손실 균형으로 수렴할 뿐이다.

== Finding 2: 게이트 선택은 태스크 동질성에 의존한다 <find2>

기존 문헌은 PLE/MoE 아키텍처에서 sigmoid 게이팅이 softmax 보다 우수하다고
시사한다 @sigmoid_moe2024.
본 연구의 결과는 이것이 태스크가 동질적일 때에만 성립함을 보인다.
13개의 이종 태스크에서는 softmax 가 ranking 지표에서 sigmoid 를 능가한다.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 5pt,
    align: (left, right, right, right, right, right),
    stroke: 0.5pt,
    table.header([*Variant*], [*Val Loss*], [*Avg AUC*], [*Avg F1m*], [*Avg MAE*], [*NDCG\@3*]),
    [Shared Bottom], [14.641], [0.6691], [0.2025], [0.9594], [0.6849],
    [PLE Softmax], [14.637], [0.6716], [*0.2026*], [*0.9567*], [*0.7002*],
    [PLE Sigmoid], [*14.635*], [*0.6724*], [0.2021], [0.9605], [0.6943],
  ),
  caption: [13개 이종 태스크에서의 게이트 유형 비교
  (benchmark_v12 local runs, 20 epochs $= 2 T_0$, seed 42, final
  epoch). Bold = 컬럼별 최고. Softmax 가 멀티클래스 및 ranking 컬럼(F1-macro,
  NDCG\@3)을 차지하고, sigmoid 는 집계 AUC 와 val loss 에서 NDCG\@3 격차보다
  여러 배 작은 마진으로 앞선다($+0.0008$ AUC vs.\ $-0.0059$ NDCG\@3).],
) <tab:gatetype>

sigmoid 게이트 위에 loss-level adaTT 를 쌓은 네 번째 변형은 20-epoch
실행이 완료되지 않았으므로 20-epoch 표에서 제외한다. 10-epoch 예산에서는
plain sigmoid 대비 집계 AUC 에 영향이 없는 null effect 이다($0.6728 arrow.r 0.6717$,
$Delta = -0.0011$) --- 이는 Finding 7 의 nine-way 비교가 토대로 삼는 결과이다.

*여기서 softmax 가 이기는 이유*: softmax 게이트에서는 각 태스크의 전문가에
대한 어텐션의 합이 1이 되어 *경쟁적 할당(competitive allocation)* 이
형성된다. 한 이진 태스크가 Expert A 를 가중치 0.6 으로 차지하면, 다른
전문가들에는 0.4 만 남는다.
이 경쟁은 태스크 유형별로 전문가를 *분리(isolate)* 하여,
고그래디언트 이진 태스크가 멀티클래스 태스크가 의존하는 전문가를
오염시키는 것을 방지한다.

sigmoid 게이트는 각 전문가가 독립적으로 기여하도록 허용한다.
이는 모든 태스크의 그래디언트 크기가 유사할 때(동질적 경우) 더 풍부한
전문가 조합을 가능하게 하므로 유익하다.
그러나 7개의 이진 태스크가 고그래디언트 신호를 생성하고 3개의 멀티클래스
태스크가 상대적으로 작은 그래디언트를 생성하는 상황에서는,
sigmoid 가 이진 그래디언트를 *모든* 전문가에 동시에 흐르도록 허용하여
더 약한 멀티클래스 신호를 압도한다.

판단 기준은 아키텍처의 정교함이 아니라
*태스크 유형 동질성(task-type homogeneity)* 이다:
- 동질적 태스크(모두 이진, 또는 모두 회귀): sigmoid 선호(더 풍부한 혼합).
- 이종 태스크(혼합 유형): softmax 선호(그래디언트 분리).

== Finding 3: Uncertainty 가중치는 아키텍처에 무관하게 동일하게 수렴한다 <find3>

놀라운 관찰: epoch 10 에서 학습된 uncertainty 가중치는 shared-bottom 과
PLE-softmax 사이에서 *사실상 동일하다*.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 5pt,
    align: (left, right, right, right),
    stroke: 0.5pt,
    table.header([*Task*], [*Shared-Bottom*], [*PLE Softmax*], [*Δ*]),
    [will\_acquire\_payments (binary)], [0.3981], [0.3974], [$-0.0007$],
    [nba\_primary (7-class)], [0.3353], [0.3354], [+0.0001],
    [next\_mcc (50-class)], [0.3360], [0.3361], [+0.0001],
    [cross\_sell\_count (regression)], [0.6652], [0.6645], [$-0.0007$],
    [churn\_signal (binary)], [0.3432], [0.3439], [+0.0007],
  ),
  caption: [epoch 10 에서 학습된 uncertainty 가중치(선택된 태스크,
  benchmark_v12 local 10-epoch runs). 차이는 아키텍처와 무관하게 소수점
  넷째 자리 수준이다.],
) <tab:uw-convergence>

이는 uncertainty weighting 이 *손실 스케일 정규화(loss-scale normalization)* ---
각 태스크의 손실을 비교 가능한 크기로 매핑하는 것 --- 를 수행할 뿐,
그래디언트 간섭에 대한 *구조적 보호(structural protection)* 는 제공하지
않음을 의미한다.
구조적 보호는 게이팅에서 비롯된다(Section 4.2).
거의 동일한 uncertainty 가중치 하에서도, @tab:gatetype 의 20-epoch 비교에서
softmax 는 여전히 NDCG\@3 0.7002 를 달성하여 shared-bottom 의 0.6849 를
앞서며, 이 +0.0153 개선은 순수하게 게이트 구조에 기인한다.

*함의*: 실무자는 태스크 유형 충돌을 다루는 데 uncertainty weighting 에만
의존해서는 안 된다. 이는 스케일을 균형 맞추지만 그래디언트 오염을 방지하지는
못한다. 소수 태스크 유형을 보호하는 실제 메커니즘은 게이트 설계이다.

== Finding 4: Epoch 예산 민감도 <find4>

v14 SageMaker 매칭 페어 연구(ml.g4dn.2xlarge, 두 예산 모두 동일한 fp32
DuckDB-streaming 파이프라인)는 10 epochs 에서 관찰된 작은 AUC 격차가
underfitting 아티팩트인지 진정한 plateau 인지를 직접 검증한다. 매칭된 설정은
local fp64 baseline 과 SageMaker fp32 추론 사이의 정밀도 경로(precision-path)
드리프트로부터 epoch 예산 효과를 분리하므로, @tab:epoch 의 10ep / 15ep 델타는
오직 학습 기간에만 기인한다.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 5pt,
    align: (left, right, right, right, right),
    stroke: 0.5pt,
    table.header(
      [*Epoch*], [*Shared-Bottom*], [*PLE Softmax*], [*PLE Full*], [*PLE Full+adaTT*],
    ),
    [10], [0.8197], [*0.8233*], [0.8216], [0.8223],
    [15], [0.8015], [*0.8249*], [0.8203], [0.8213],
    [Δ (15$-$10)], [$-$0.0182], [$+$0.0016], [$-$0.0013], [$-$0.0010],
  ),
  caption: [v14 phase0 에서 매칭된 10 / 15 epochs 의 Avg AUC (1M rows,
  ml.g4dn.2xlarge SageMaker spot, fp32 DuckDB stream). PLE 변형들은
  plateau 에 도달하는($|Delta| <= 0.002$) 반면 shared\_bottom 은 $-$0.0182
  하락하며 val\_loss 가 16.58 $arrow.r$ 17.64 로 상승한다 --- 공유 표현이
  일반화보다 학습 노이즈를 더 빨리 암기한다.],
) <tab:epoch>

매칭 페어 표는 이전 초안의 local-only 해석에 두 가지 정정을 가한다.

*정정 1: PLE softmax 는 이미 10 epochs 에서 shared\_bottom 을 능가한다.*
PLE softmax(0.8233)는 동일한 SageMaker 인프라에서 10 epochs 시점에
shared\_bottom(0.8197)을 $+$0.0036 AUC 앞서며, 이는 shared\_bottom 이
앞서는 것처럼 보였던 이전의 local-10-epoch 판독과 모순된다.
local 순위는 epoch 예산 결핍이 아니라 정밀도 경로 아티팩트(fp64
pq.read\_table vs.~fp32 DuckDB)였으며, 게이트 구조의 결함이 아니다.

*정정 2: 과적합하는 것은 PLE 가 아니라 shared\_bottom 이다.* 10 에서
15 epochs 로 연장하면 shared\_bottom 은 $-$0.0182 AUC 만큼 열화되는 반면
세 PLE 변형은 모두 평탄하게 유지된다($|Delta| <= 0.002$). 따라서 "예산을
최소 $2 times T_0$ 로 보장하라"는 권고는 공유 표현 변형에 대해서는
역전이 필요하다: 이들은 통제되지 않은 shared bottom 이 노이즈 암기
영역에 진입하는 순간 *조기 종료(early stopping)* 가 필요하다.
PLE+CGC 의 $T times K$ 게이트 파라미터($T = 13$ tasks, $K = 7$ experts)는
과적합 없이 예산 연장을 흡수하는 구조적 정규화기(regularizer) 역할을 한다.

*PLE-softmax-reg* 변형(dropout 0.3, weight\_decay $1 times 10^(-4)$)은
AUC 0.8256, val\_loss 16.57 로 추가 개선되며(정규화하지 않은 PLE-softmax
baseline 의 0.8249 / 17.09 대비), plateau 에서조차 게이트 변형이
통상적 정규화로 회복 가능한 calibration-loss 여유를 남겨두고 있음을 보인다.

*지침(개정)*: v14 phase0 스케일(1M rows, 1210 features, fp32 streaming)에서
MTL 구조를 비교할 때, 10-epoch 매칭 비교만으로도 게이트 아키텍처와 공유
아키텍처를 구별하기에 충분하다. 더 긴 예산(15+)은 과적합을 방지하기 위해
shared\_bottom 에 정규화가 필요하지만, PLE 변형은 정규화 없이도 연장을
견딘다.

== Finding 5: GTE 사전 게이팅이 혼합 유형 그룹의 성능을 저하시킨다 <find5>

GroupTaskExpert(GTE)는 PLE 게이팅이 일어나기 *전에* 전문가 표현을 태스크 그룹별로 분할하는
사전 게이팅 계층을 추가한다.
그 동기는 그룹 내 전문가 공유를 강화하려는 데 있다.

그러나 태스크 그룹이 혼합 유형을 포함할 때
--- Consumption 그룹은 이진 태스크 5개, 다중클래스 태스크 1개
(`nba_primary`), 회귀 태스크 1개(`cross_sell_count`)로 구성된다 ---
GTE는 양립 불가능한 손실 유형 간에 공유 표현 학습을 강제한다:

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 5pt,
    align: (left, right, right),
    stroke: 0.5pt,
    table.header([*Variant*], [*NDCG\@3*], [*Avg AUC*]),
    [PLE Softmax], [0.7002], [0.6716],
    [PLE Sigmoid + GTE], [0.6632], [0.6720],
    [Δ], [$-0.0370$], [+0.0004],
  ),
  caption: [GTE 영향(benchmark_v12 로컬 실행, 20 epoch). GTE가 혼합 유형 그룹에
  사전 게이트 표현을 공유하도록 강제하면 NDCG\@3 이 크게 하락하는 반면, 집계 AUC 는
  변하지 않는다. 이 하락은 게이트 유형이 아니라 GTE 에 기인한다: 순수 PLE
  sigmoid 의 NDCG\@3 은 0.6943(@tab:gatetype)이므로, GTE 는 자체 게이트 유형
  기준선과 비교해도 $-0.0311$ 의 비용을 치른다.],
) <tab:gte>

그 메커니즘은 다음과 같다: GTE 는 태스크별 PLE 게이팅이 전문가들을 차별화하기 *전에*
그룹 수준에서 전문가 출력을 풀링한다.
Consumption 그룹 내에서 이진 태스크 5개의 높은 기울기 신호가 풀링된 표현을 지배하며,
이 그룹의 소수 유형 태스크 --- `nba_primary`(NDCG\@3 의 배후에 있는 다중클래스 랭킹 태스크)와
회귀 태스크 `cross_sell_count` --- 는 이미 이진 결정 경계 쪽으로 편향된 표현을 받게 된다.
GTE 풀링 *이후*에 작동하는 PLE 게이팅은 이 손상을 되돌릴 수 없다.

*지침*: GTE(또는 유사한 사전 게이팅 메커니즘)를 위한 태스크 그룹은 비즈니스 의미가 아니라
태스크 유형 기준으로 동질적이어야 한다.
비즈니스적으로 의미 있는 그룹화가 유형을 혼합한다면,
GTE 를 생략하고 전문가 간 할당은 PLE 게이팅에만 의존하라.

== Finding 6: 게이트 엔트로피와 손실–메트릭 디커플링 <find6>

=== CGC 게이트 엔트로피 분석

PLE 의 CGC 게이팅이 실제로 전문가를 *어떻게* 할당하는지 이해하기 위해,
교사 학습 종료 시점(30 epoch)에서 각 태스크의 게이트 가중치 분포
$g_t in RR^K$ 의 Shannon 엔트로피 비율을 계산한다.
엔트로피 비율은 다음과 같이 정의된다:

$ E_t = H(g_t) / H_"max" = -( sum_k g_{t,k} log g_{t,k} ) / log K $

여기서 $H_"max" = log K$ 는 $K$ 개 전문가에 대한 최대 엔트로피이다.
$E_t = 1$ 은 전문가가 완전히 균일하게 활용됨을 의미하며,
$E_t = 0$ 은 단일 전문가가 모든 가중치를 차지함을 의미한다.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 5pt,
    align: (left, right, right, left),
    stroke: 0.5pt,
    table.header([*Task*], [*Layer 1*], [*Layer 2*], [*Pattern*]),
    [top\_mcc\_shift], [0.347], [---], [Single-expert dominance],
    [product\_stability], [0.431], [---], [Single-expert dominance],
    [segment\_prediction], [---], [0.332], [Single-expert dominance],
    [cross\_sell\_count], [0.570], [0.614], [Moderate diversity],
    [churn\_signal], [0.691], [0.860], [Moderate → full diversity],
    [nba\_primary], [0.851], [0.839], [Full expert utilization],
    [will\_acquire\_payments], [0.882], [---], [Full expert utilization],
  ),
  caption: [태스크 및 PLE 계층별 CGC 게이트 엔트로피 비율(교사 모델,
  30 epoch --- `segment_prediction` 제거 전의 v12 계열 13-태스크 구성).
  낮은 엔트로피($E_t < 0.45$)는 1--2 개 전문가가 지배함을 나타내고,
  높은 엔트로피($E_t > 0.80$)는 7개 전문가 모두가 의미 있게 기여함을 나타낸다.
  본 엔트로피 패턴(0.33--0.88)은 합성 벤치마크 특정값이다 --- Paper 1의
  운영 실데이터 재측정에서는 분포가 더 균일했다(0.67--0.97, 지배 전문가가
  뚜렷한 태스크는 소수). 붕괴 부재 결론 자체는 양쪽에서 공통이다.],
) <tab:gate-entropy>

엔트로피 비율은 행동적으로 구분되는 세 가지 태스크 군집을 드러낸다:

*단일 전문가 지배*($E_t$ 0.33--0.43): `top_mcc_shift`
(MCC 카테고리 이동 예측)와 `segment_prediction`(4-클래스 고객 세그먼트 ---
명명된 세그먼트 3개와 UNKNOWN)
같은 태스크는 1--2 개 전문가가 담당한다. 이러한 태스크는 단일 특화 전문가 ---
거래 피처에 대한 DeepFM, 혹은 계층적 세그먼트에 대한 HGCN --- 가
거의 최적으로 처리하는 단순한 패턴을 인코딩하는 것으로 보인다.
낮은 엔트로피는 실패 모드가 아니라 효율적인 라우팅이다.

*중간 다양성*($E_t$ 0.57--0.72): `cross_sell_count`
(카운트 회귀)와 일부 이진 획득 태스크 같은 태스크는 3--4 개 전문가를 활용한다.
이러한 태스크는 거래 신호(DeepFM)와
시퀀스 수준 패턴(Temporal)을 모두 요구할 가능성이 높아, 부분적인 전문가 분산을 설명한다.

*전체 전문가 활용*($E_t$ 0.85--0.88): `nba_primary`(7-클래스 next
best action)와 `will_acquire_payments`(이진)는 7개 전문가 모두를 적극적으로 사용한다.
이러한 태스크는 단일 아키텍처 귀납적 편향으로는 완전히 포착되지 않는,
복잡하고 다면적인 고객 행동 패턴을 인코딩한다.

=== 어텐션 붕괴: 구조적 사각지대

어텐션 집계 수준(공유 전문가 집계)에서는 13개 태스크 전부가
*정확히 1.000* 의 엔트로피 비율을 보인다. 어텐션 메커니즘이 차별화를
학습하지 못한 것 --- 전문가 풀에 대한 단순 평균처럼 작동한다.

이는 구조적으로 중요하다. CGC 추출 계층 게이트(표 @tab:gate-entropy)는
*모델이 차별화된 전문가 선호를 학습할 수 있음*을 입증하지만,
이 능력은 어텐션 집계 수준에서는 완전히 부재한다.
두 가지 설명이 가능하다:

+ *기울기 결핍(gradient starvation)*: 어텐션 파라미터는 태스크별 타워 헤드를
  통과한 후에야 기울기를 받는데, 그 헤드들은 이미 추출 계층 게이트를 통해 특화되어 있다.
  신호가 어텐션 계층에 도달할 무렵에는 태스크별 구분이
  상류에서 이미 충분히 처리되었을 수 있다.

+ *매개변수화 병목*: 어텐션 쿼리 차원이 전문가 임베딩 공간에 비해 작다면,
  어텐션은 7개의 이종 전문가에 걸쳐 태스크별 선호를 형성할
  충분한 용량을 갖지 못한다.

어느 경우든 어텐션 구성요소는 의미 있는 라우팅을 수행하지 않으면서 파라미터만 추가한다.
이는 아키텍처 단순화의 후보이며 ---
어텐션 집계를 고정 평균이나 태스크별 학습된 스칼라로 대체하는 것 --- 향후 연구로 남긴다.

=== 30 epoch 에서의 손실–메트릭 디커플링

학습을 10 에서 30 epoch 로 연장하면($T_0 = 10$, 코사인 웜
재시작) 복합 손실 모니터링의 근본적 긴장이 드러난다.
v14 SageMaker 재실행(ml.g4dn.2xlarge, fp32 DuckDB-stream 파이프라인,
PLE softmax, seed 42)은 더 높은 절대 AUC 크기에 도달함으로써 v13 궤적을
정련하면서도 동일한 디커플링 패턴을 유지한다.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 5pt,
    align: (left, right, right, right, right),
    stroke: 0.5pt,
    table.header([*Epoch*], [*Val Loss*], [*Avg AUC*], [*NDCG\@3*], [*Avg MAE*]),
    [1],  [19.71], [0.7032], [0.6321], [0.4322],
    [5],  [17.51], [0.7903], [*0.7904*], [0.3717],
    [10], [16.44], [0.8142], [0.7655], [0.3382],
    [13], [*16.20*], [0.8239], [0.7134], [*0.3291*],
    [15], [16.38], [*0.8272*], [0.7707], [0.3216],
    [20], [18.43], [0.8196], [0.7702], [0.3102],
    [21], [17.35], [0.8210], [0.7692], [0.3203],
    [25], [17.92], [0.8147], [0.7565], [0.3199],
    [30], [19.04], [0.8133], [0.7726], [0.3166],
  ),
  caption: [v14 phase0 에서 30 epoch 에 걸친 손실–메트릭 디커플링(교사 = PLE
  softmax, 고객 1M, $T_0=10$ 코사인 웜 재시작).  Val loss 는 epoch 13 에서
  최솟값(16.20)에 도달한 뒤 상승하며(과적합 시작);
  Avg AUC 는 epoch 15 에서 정점(0.8272)을 찍고 epoch 30 까지 $-$1.4\%p 하락하며;
  NDCG\@3 은 놀랍도록 이른 epoch 5 에서 정점(0.7904)을 찍고 epoch 30 까지
  부분적으로 회복한다(0.7726, 정점 대비 $-$1.8\%p).  Avg MAE 는 epoch 20 까지 계속
  개선되다가 안정화된다.  Bold = 메트릭 정점.],
) <tab:loss-metric-decouple>

v14 궤적은 서로 다른 epoch 에서 발생하는 세 가지 태스크 유형별 정점을 보인다:

- *Avg AUC*(이진 태스크): epoch 15 에서 정점(0.8272)을 찍은 뒤 epoch 30 까지
  0.8133 으로 하락($-$1.4\%p).  이는 v13 의
  $-$0.4\%p 하락보다 더 두드러지는데, 더 깨끗한 v14 피처가 학습 초기에 이진
  천장을 더 높이 끌어올려, 두 번째 코사인 사이클에서 과적합의 여지를
  더 많이 남기기 때문이다.
- *NDCG\@3*(랭킹 품질): 놀랍도록 이른 epoch 5 에서 정점(0.7904)을
  찍는다 --- 어떤 코사인 재시작보다도 한참 앞선다.  추천
  랭킹 품질은 초기 고-LR 단계에 민감하다: 코사인
  스케줄이 epoch 10 까지 LR 을 감쇠시키면서, 게이트의 전문가
  선호가 AUC 는 계속 개선되는 와중에도 rank-aware NDCG 를 해치는 방향으로 굳어진다.
  epoch 15 이후 NDCG 는
  0.71--0.79 사이에서 단조적 방향 없이 진동한다.
- *Avg MAE*(회귀 태스크): epoch 20 까지 0.4322 → 0.3102 로 꾸준히
  개선되다가 정체된다.  복합 val\_loss 는 그 반대의
  U자형을 보이는데(epoch 13 에서 16.20 최솟값), MAE 가 포화됨에 따라 이진 태스크
  교차 엔트로피가 지배하기 때문이다.

근본 원인은 v13 에서 이어진다: *복합 손실에서의 태스크 유형 지배*.
각 태스크 유형이 서로 다른 epoch 에서 정점을 찍으므로(NDCG\@3 은 5,
AUC 는 15, MAE 는 20+), 단일 메트릭 기반 체크포인트 선택은 세 태스크 유형 중
둘을 희생한다.  v14 의 더 넓은 절대 크기
($Delta$ AUC 정점-최종 $-$1.4\%p, $Delta$ NDCG 정점-최종
$-$1.8\%p)는 이 트레이드오프를 완화하는 것이 아니라 더 날카롭게 만든다.

=== 태스크 유형 전반의 코사인 재시작 진동

코사인 웜 재시작($T_0 = 10$, $T_"mult" = 2$)은 사이클 경계에서 학습률
스파이크를 만든다(epoch 10 → 두 번째 사이클 11--30 은 epoch 20 의
중간 지점에서 다시 감쌈).  v14 에서 NDCG\@3 은 이러한
경계에서 강한 진동을 보인다:

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 5pt,
    align: (left, right, left),
    stroke: 0.5pt,
    table.header([*Epoch (event)*], [*NDCG\@3*], [*Change*]),
    [5 (cycle 1 mid)],   [*0.7904*], [global peak — early high-LR phase],
    [10 (cycle 1 end)],  [0.7655], [$-$2.5\%p (drift before restart)],
    [11 (restart 1)],    [0.7201], [$-$4.5\%p (sharpest drop)],
    [13 (val\_loss min)], [0.7134], [recovery deferred],
    [15 (AUC peak)],     [0.7707], [recovers $+$5.7\%p but below epoch 5],
    [21 (LR mini-bump)], [0.7692], [stable through small perturbation],
    [28 (late cycle 2)], [0.7874], [secondary peak, $-$0.3\%p vs.~epoch 5],
    [30 (final)],        [0.7726], [stable below initial peak],
  ),
  caption: [v14 에서 코사인 재시작 경계의 NDCG\@3 진동.
  전역 정점은 epoch 5(사이클 1 의 고-LR 단계)에 있다 ---
  v13 궤적에서 예상되는 사이클 1 종료 지점이 *아니다*.
  첫 재시작은 epoch 11 에서 NDCG\@3 을 $-$4.5\%p 떨어뜨리며; epoch 15 의
  0.7707 로의 회복은 부분적이다.  늦은 사이클 2(epoch 28)는
  전역 정점에 다시 근접하지만 이를 초과하지는 못한다.],
) <tab:cosine-oscillation>

v14 패턴은 태스크 유형 전반에 걸쳐 비대칭적이다.  Avg MAE 는 두 번째 코사인
사이클 내내 계속 개선되는데(epoch 20 까지 0.4322 → 0.3102),
이는 회귀 손실 지형이 매끄러워 옵티마이저가
각 LR 스파이크 이후 저-MAE 영역으로 되돌아오기 때문이다.  이진 AUC 는
재시작 1 에서 작은 하락($-$0.4\%p)을 보였다가 epoch 15 에서 새로운
정점으로 강하게 회복한다.  NDCG\@3 은 가장 큰 교란($-$4.5\%p,
재시작 1)을 겪는데, 이는 랭킹 메트릭이 점수의 상대적
순서에 민감하고, LR 재시작이 모델이 재수렴하기 전에 점수
척도를 일시적으로 뒤흔들기 때문이다.  무엇보다 중요한 것은 v14
NDCG\@3 *전역 정점*이 사이클 1 의 고-LR 단계
(epoch 5) 안에 위치하며 --- AUC 정점(epoch 15)보다 한참 앞선다는 점으로 ---
이는 rank-aware 다중클래스 태스크가 동일한 복합 손실 스케줄에서
이진 분류보다 더 이른 최적 체크포인트를 가짐을 확인해 준다.

*시사점*: 랭킹에 민감한 응용에서는
$T_"mult" = 1$(일정한 사이클 길이)의 코사인 재시작을
$T_"mult" = 2$(사이클 길이 배증) 혹은 학습률 워밍업-후-감쇠
(재시작 없음)로 대체하여, 30 epoch 예산에 걸쳐 평가해야 한다.

=== 체크포인트 선택 기준

이러한 발견은 서로 다른 유형의 태스크가 복합 손실을 공유할 때
*val loss 가 유효하지 않은 체크포인트 기준*임을 함께 확립한다.
올바른 접근은 메트릭 의미를 태스크 유형별로 가중하는
복합 체크포인트 메트릭이다:

$ "score"_"ckpt" = alpha dot.c "AvgAUC" + beta dot.c "NDCG@3" + gamma dot.c (1 - "NormMAE") $

여기서 $alpha, beta, gamma$ 는 태스크 수에 비례하는 것이 아니라
태스크 유형을 동등하게 가중하도록 설정한다(예: $alpha = beta = gamma = 1/3$).
@tab:loss-metric-decouple 의 v14 궤적에서는 세 태스크 유형 모두에 대해
최적인 단일 epoch 가 없다 --- NDCG\@3 은 epoch 5 에서 정점을, val
loss 는 epoch 13 에서 최저를, AUC 는 epoch 15 에서 정점을 찍고, MAE 는 epoch 20 까지
계속 개선된다. val-loss 모니터는 epoch 13 을 고정하여,
2 epoch 뒤의 AUC 정점과 이른 랭킹
최적점을 모두 희생하게 된다; 유형 가중 복합 메트릭은 그 트레이드오프를
손실 척도 안에 감추는 대신 명시적으로 드러낸다.

*지침*: 이종 MTL 시스템에 회귀 태스크가 존재할 때,
(1) 학습 시작 전에 태스크 유형 전반에 걸친 복합 체크포인트 메트릭을 정의하고,
(2) val loss 를 모니터링하는 대신 매 epoch 체크포인트를 저장하여 사후에 선택하며,
(3) val loss 는 주된 정지 기준이 아니라 진단 지표(회귀 진행도를 나타내는)로
취급하라.

== Finding 7: 분리된 축 위의 두 가지 긍정적 융합 레시피, 합성 시 비가산적 <find7>

Finding 2에서 loss-level `adaTT` 변형이 12-태스크 규모에서 집계 AUC에
영향을 주지 않음($Delta = -0.001$, null)을 확인한 후, CGC의 게이팅된
선택을 넘어서는 유용한 신호를 추출할 수 있는 융합 증강(fusion
augmentation)의 형태가 (있다면) 무엇인지 규명하기 위해 동일 벤치마크에서
8개의 추가 메커니즘을 평가하였다. 이 9-way 비교는 v14 phase0에서 세 개의
영역으로 정리된다. M1 complement를 예외로 하는 *표현-가산적(representation-additive)
융합*(loss-level adaTT, AdaTT-sp, ECEB, BRP-MV)은 primary 표현에 residual을
주입하거나 residual-error gradient를 shared 전문가로 전파한다. 이 네 가지는
집계 AUC를 노이즈 수준($-$0.001 ~ $-$0.003)으로 저하시킨다. *M1 complement*는
AUC를 *상승*시키는($Delta = +0.0006$) v14 특이적 예외로, v13 순위($-0.0053$)를
역전시킨다. *gradient 격리를 동반한 출력 공간 부스팅*(BRP-detached)은
NDCG\@3을 $+0.025$ 끌어올리지만 v14에서 집계 AUC를 $-0.0046$ 저하시켜 v13의
*tied* 판정($-$0.0007)에서 후퇴하는데, 더 깨끗한 v14 피처가 CGC 베이스라인을
0.8234(v13의 0.6728 대비)까지 끌어올려 이전에 손익분기였던 residual bank가
이제 더 강한 primary tower와 경쟁하기 때문이다. *학습 시점 부하 분산
정규화*(NEAS --- inverse-gate auxiliary supervision)는 v14에서 v13의 양의
AUC 판정을 재현하여($Delta = +0.0017$, v13의 $+0.0011$ 대비), aux supervision을
통해 전문가 붕괴(expert collapse)를 방지하는 메커니즘이 phase0 세대 전반에서
견고함을 확인한다. 두 비자명 레시피는 분리된 축에서 작동한다 --- 출력 공간에서의
오류 보정(BRP-detached) 대 게이트에서의 전문가 붕괴 방지(NEAS) --- 그러나 이
둘을 쌓은 아홉 번째 실험은 *비가산적(non-additive)* 결과를 낳는다: NEAS의 AUC
상승이 사라지고($Delta = -0.0026$) BRP-detached의 NDCG\@3 이점도 약화된다(단독
0.7871 대비 0.7579). shared 전문가가 NEAS를 위한 일반화와 BRP-detached가 의존하는
primary-지원 최적점으로의 특화를 동시에 달성할 수 없기 때문이다.

=== 다섯 메커니즘, 하나의 집계-AUC 결론

*Loss-level adaTT*(Paper 1에서 보고된 변형)는 gradient 코사인 유사도에서
추정한 전이 가중치를 사용하여 가중 cross-task 손실 항
$L_i + lambda sum_(j != i) w(i arrow.r j) L_j$ 을 추가한다.

*AdaTT-sp* @li2023 는 native-expert residual을 추가한다: CGC 게이트가 가중합을
산출한 후, 해당 태스크 자신의 task-specific 전문가들의 평균 출력을 학습 가능한
스칼라로 스케일링하여 재주입한다.

*Residual complement (M1)* 은 본 논문에서 도입한 것으로, primary 게이팅 출력을
보존하면서 동일한 전문가 출력에 보완 가중치 $(1 - "gate_weights")$(전문가 축에
대해 clamp 및 재정규화)를 residual로 적용하고 학습 가능한 스칼라로 스케일링한다.
의도는 cross-task 혼합 없이, 게이트가 down-weight한 전문가로부터 intra-task
신호를 회복하는 것이다.

*ECEB (Error-Conditioned Expert Bank, MV)* 는 본 논문에서 도입한 것으로, 위
세 가지의 공유 구조에서 벗어나도록 특별히 설계되었다: residual을 게이트의
출력이 아니라 게이트의 *엔트로피*에서 도출한다. 구체적으로, 회복 경로는
태스크 무관(task-agnostic) 합의(게이트 가중 없는 전체 전문가 출력 평균)이며,
forward 시점에 태스크별 학습 가능 스칼라 $sigma(w_t)$ 와 정규화된 게이트
엔트로피 $H(g_t)/log N$(샘플별)의 곱으로 스케일링된다. 게이트가 확신할 때(낮은
엔트로피) 회복은 0으로 수렴하고, 분산되어 있을 때 회복이 활성화된다. 설계상
ECEB은 "게이트에서 도출된 residual" 요인을 제거한다.

*BRP (Boosting-Residual Path, MV)* 역시 본 논문에서 도입한 것으로,
표현-가산적 구조 자체를 제거한다. 태스크별 residual 전문가 bank가 마지막 CGC
계층의 `shared_concat`(게이트 우회 피처 뷰)을 입력으로 받아 primary tower의
출력 형태에 맞는 logit residual을 산출한다. residual은 primary의 *detached*
예측 오류, 즉 $y - "activation"("stop_grad"("primary"))$ 에 대한 MSE로 학습된다
(단일 단계 부스팅). primary 경로는 ground truth만으로 학습하며, 결합 출력
$"primary" + sigma(lambda_t) dot "residual"$ 은 추론과 평가에만 사용된다.
primary 표현은 결코 건드리지 않는다.

*BRP-detached* 는 BRP의 한 줄 수정으로, residual bank에 raw `shared_concat`
대신 `shared_concat.detach()` 를 공급한다. 이는 primary 경로, 파라미터 수,
학습 스케줄을 변경하지 않은 채 residual-MSE gradient가 shared 전문가로 역류하는
것을 차단한다. 이 수정은 아래 BRP의 태스크별 분석에서 동기를 얻었다.

*NEAS (Neglected-Expert Auxiliary Supervision)* 는 마지막 CGC 계층 전문가
출력의 *inverse-gate-weighted aggregation* 을 소비하는 태스크별 auxiliary
head를 추가한다. auxiliary 타겟은 primary 태스크 레이블이며, auxiliary
손실(`aux_weight = 0.05` 로 스케일링)은 학습 시에만 총 손실에 더해진다. 추론은
auxiliary head를 전혀 사용하지 않는다. 이 메커니즘은 게이트가 특정 태스크에서
de-emphasis하는 neglected 전문가들이 예측적 표현을 유지하도록 명시적으로 강제하여
전문가 붕괴를 완화한다. NEAS는 위의 모든 residual 메커니즘과 구조적으로
독립적이며, residual을 주입하지도 primary 출력을 수정하지도 않는다.

*NEAS + BRP-detached* 는 두 긍정적 메커니즘을 쌓아 가산성(additivity)을
테스트한다. 둘 다 단독 설정으로 활성화하며, 그 외 수정은 없다.

12-태스크 벤치마크 결과는 9개 시나리오 전부에 대해 v14 SageMaker
재실행(15 에포크, ml.g4dn.2xlarge, fp32 DuckDB-stream 파이프라인, seed=42)으로
보고한다. 원래의 9-way 비교는 v13 phase0(HMM duplication, GMM K=20 dead
clusters, prob-column scaler 적용)에서 수행되었으나, paper 1 / paper 3 v14
베이스라인과 정합을 맞추기 위해 이제 v14 phase0에서 완전히 갱신되었다.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    stroke: 0.5pt,
    inset: 5pt,
    table.header(
      [*Fusion*], [*Final AUC*], [*F1 macro*], [*NDCG\@3*], [*$Delta$ AUC*]
    ),
    [Shared-Bottom (no gate)], [0.8015],   [*0.1426*], [0.7149], [(reference)],
    [CGC gate (sigmoid baseline)], [0.8234], [0.1404], [0.7619], [---],
    [PLE-full (sigmoid+5 toggles)], [0.8204], [0.1306], [*0.7871*], [$-$0.0030],
    [PLE-full $+$ adaTT], [0.8202], [0.1311], [0.7697], [$-$0.0032],
    [M1 complement],     [*0.8240*], [0.1366], [0.7685], [*$+$0.0006*],
    [ECEB (MV)],         [0.8231], [0.1325], [0.7625], [$-$0.0003],
    [BRP (MV)],          [0.8216], [0.1327], [0.7679], [$-$0.0018],
    [BRP-detached],      [0.8188], [0.1393], [0.7871], [$-$0.0046 (degraded)],
    [NEAS],              [0.8251], [0.1325], [0.7692], [$+$0.0017 (positive)],
    [NEAS + BRP-detached], [0.8208], [0.1330], [0.7579], [$-$0.0026 (non-additive)],
  ),
  caption: [NDCG\@3을 복원한 융합 메커니즘 비교.
  모든 행: v14 SageMaker ml.g4dn.2xlarge, 15 에포크.  $Delta$ AUC는 각 변형을
  CGC sigmoid 베이스라인(2행; v13의 PLE+CGC 기준에 해당하는 v14 등가물)과
  비교한다.  v14에서 네 가지 패턴이 나타난다:
  *(i)* M1 complement은 집계 AUC를 *상승*시키는 유일한 메커니즘으로
  ($Delta = +$0.0006) — 더 깨끗한 phase0 피처(HMM mode-split, GMM K=14,
  prob-column scaler 제외)에 기인하는, v13의 M1 순위($-$0.0053)에 대한 v14
  특이적 역전이다;
  *(ii)* PLE-full과 BRP-detached가 최고 NDCG\@3(0.7871)에서 동률이며,
  PLE-full은 GTE + LT + HMM-projector 스택을 통해, BRP-detached는
  residual 전용 출력 공간 보정을 통해 이에 도달한다; *(iii)* shared\_bottom의
  F1 macro(0.1426)는 최저 AUC(0.8015)에도 불구하고 v14 컬럼에서 최고이다 ---
  no-gate 베이스라인은 게이팅된 변형들이 softmax로 재분배하여 흩어버리는
  소수 클래스 신호를 보존한다; *(iv)* NEAS는 v14에서 v13의 양의 AUC 판정을
  재현하지만($Delta = +$0.0017, v13의 $+$0.0011 대비), BRP-detached는 v14에서
  저하된다($Delta = -$0.0046, v13의 $-$0.0007 tied 대비). 더 깨끗한 v14 피처가
  CGC 베이스라인을 더 높이 끌어올려(0.8234 vs v13의 0.6728), 이전에 손익분기였던
  residual bank가 이제 더 강한 primary tower와 경쟁하기 때문이다. 비가산적
  합성(10행) 역시 저하되나 BRP-detached 단독보다는 덜 심한데, 이는 NEAS가
  BRP-detached의 손실을 부분적으로 상쇄한다는 점과 일치한다.]
) <tab:fusion9way>

M1이 에포크 1(학습 전)에 최고 AUC를 보이고 이후 단조 감소한다는 것은,
학습 가능한 회복 가중치를 학습하는 것이 오히려 성능을 적극적으로 저하시킴을
나타낸다 --- random initialisation이 수렴된 가중치보다 덜 해로운 동작점이다.

=== 태스크별 분해와 두 이상치

집계 델타는 모든 변형에 걸쳐 노이즈 수준($<= 0.005$)이지만, 태스크별 분해는
세 가지 양상을 드러낸다:

- *게이트 포화 태스크*(top_mcc_shift,
  mcc_diversity_trend)는 낮은 게이트 엔트로피(ratio $< 0.55$)를 가지며 모든
  회복 메커니즘에 둔감하다($abs(Delta) <= 0.003$).
- *강한 primary를 가진 게이트 분산 태스크*(churn_signal,
  will_acquire_lending)는 높은 게이트 엔트로피(ratio $> 0.82$)를 가지며 가장
  큰 M1 저하를 보인다($-$0.020 및 $-$0.009).
- *단일 양의 이상치*는 next_mcc(50개 클래스, near-random base F1
  $approx$ 0.01)로, 세 가지 회복 변형 모두 $+$0.005 ~ $+$0.008 만큼 개선한다.
  이 이득은 기준 대비로는 크지만 절대적으로는 작으며, 우리는 이를 진정한 회복
  효과가 아니라 near-floor 출발점에 기인한 것으로 본다.

나머지 7개 태스크는 노이즈 범위 내에 있다($abs(Delta) <= 0.005$).

=== 게이트 엔트로피 상관: 약한 신호

게이트 엔트로피가 회복 이득을 구조적으로 예측하는지 검증하기 위해,
joint_full 체크포인트의 마지막 CGC 계층에서 태스크별 게이트 가중치를 추출하여
태스크 수준 엔트로피와 각 변형의 $Delta$ 를 상관시켰다:

- Loss-level adaTT: $r = -0.31$
- AdaTT-sp: $r = -0.32$
- M1 complement: $r = -0.40$

세 상관 모두 음수(엔트로피가 높을수록 $arrow.r$ 더 큰 해)이며 부호가 일관되지만,
$n = 13$, $p approx 0.18$ 로 어느 것도 통상적 유의성에 도달하지 못한다. 두
이상치 --- churn_signal과 next_mcc --- 는 게이트 엔트로피보다 태스크 특이적
요인(churn_signal의 레이블 구성, next_mcc의 near-floor base rate)으로 더 잘
설명된다. 따라서 게이트 엔트로피를 이 벤치마크에서 회복 이득의 구조적 예측자로
주장할 수 없다.

=== BRP, BRP-detached, NEAS: 메커니즘 진단

첫 네 가지 증강 변형(loss-level adaTT, AdaTT-sp, M1, ECEB)은 모두 primary
표현에 residual을 가산적으로 주입하며 단조 저하를 일으킨다. BRP는 이 계열에서
residual을 출력 공간에 두는 유일한 변형이다. 그럼에도 BRP의 집계 AUC는 5개의
비베이스라인 실행 중 *최저*이지만($Delta = -0.0078$), 그 하락은 그 다섯 중
*최고*의 F1 macro 및 NDCG\@3(CGC 대비 +0.0115 및 +0.0219)을 동반한다. 따라서
BRP 결과는 처음에는 성공이라기보다 태스크 균형 트레이드오프처럼 보였다.

태스크별 분해는 더 구체적인 이야기를 들려준다. next_mcc(50-class, baseline
macro-F1이 near-random 0.0100)는 BRP에서 0.0440(상대 +340%)로 개선된다 ---
어려운 태스크 구제 효과(hard-task rescue)이다. 반대 방향으로, churn_signal ---
베이스라인에서 가장 높은 binary AUC(0.6868)를 가진 태스크 --- 는 BRP에서
0.6512($-$0.036)로 하락하며, 이 단일 태스크가 집계 AUC 손실의 대부분을 차지한다.
churn_signal을 제외하면 BRP의 binary AUC는 베이스라인 근처에 위치하고, 나머지
다섯 binary 태스크는 각각 $-$0.001 ~ $-$0.010 사이를 잃는다.

메커니즘은 shared-expert gradient leak이다. BRP의 residual bank가
`shared_concat`을 소비하고, residual-MSE gradient가 shared 전문가로 역전파된다
--- detach된 것은 residual의 *타겟*뿐이었고 그 입력이 아니었다. primary가 이미
천장 근처인 태스크에서는 shared 전문가가 primary-지원 최적점으로 수렴해 있었고,
추가적인 MSE 압력이 그것을 그 최적점에서 끌어낸다. primary가 고전하는
태스크(next_mcc)에서는 residual이 primary가 추출하지 못한 신호를 공급한다. 따라서
집계 하락은 출력 공간 부스팅의 알고리즘적 한계가 아니다 --- shared 표현을 통해
residual bank를 학습시킨 구현 아티팩트이다.

BRP-detached는 이를 직접 검증한다. residual bank의 입력에서 `shared_concat`을
`shared_concat.detach()`로 교체하면 --- 파라미터 변경 없음, 학습 스케줄 변경
없음 --- 다음 태스크별 패턴이 나타난다:

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    stroke: 0.5pt,
    inset: 5pt,
    table.header(
      [*Task*], [*CGC*], [*BRP*], [*BRP-detached*], [*Verdict*]
    ),
    [churn_signal (AUC)],             [0.6868], [*0.6512*], [*0.6852*], [restored],
    [will_acquire_lending (AUC)],     [0.6549], [0.6453],   [0.6553],   [restored],
    [will_acquire_deposits (AUC)],    [0.6534], [0.6493],   [0.6536],   [restored],
    [will_acquire_investments (AUC)], [0.6754], [0.6719],   [0.6764],   [restored],
    [next_mcc (F1 macro)],            [0.0100], [*0.0440*], [*0.0356*], [retained (+256%)],
    [remaining 7 tasks],              [---],    [$plus.minus 0.002$], [$plus.minus 0.002$], [unchanged],
  ),
  caption: [BRP가 지표를 실질적으로 변화시킨 태스크 부분집합에 대한 BRP 및
  BRP-detached 대 CGC 비교. `shared_concat`을 detach하면 모든 쉬운 태스크의
  AUC 손실이 복원되고(churn_signal의 $-$0.036이 $-$0.002로 하락) next_mcc 구제
  효과의 대부분이 유지된다(상대 +340%가 +256%로 감소하나 여전히 CGC의
  near-random 베이스라인을 압도).]
) <tab:brp-pertask>

따라서 진단은 태스크 수준에서 확인된다. BRP의 쉬운 태스크 AUC 손실은
residual-MSE gradient가 shared 전문가를 재구성한 데서 비롯되었다. 입력을
detach하면 그 채널이 차단되고, shared 전문가는 primary-지원 최적점에 머물며,
residual bank는 자신의 파라미터만으로 어려운 태스크 보정을 학습한다.

*NEAS*는 다른 경로를 취한다. residual이 아니라 전문가 출력의
*inverse-gate-weighted aggregation* 에 auxiliary supervision 신호를 붙인다:
각 태스크의 aux head는 게이트가 de-emphasis한 전문가들을 주로 사용하여 primary
레이블을 예측해야 한다. 이는 태스크 수준 게이트가 집중하더라도 모든 shared
전문가가 예측적으로 남도록 gradient 압력을 생성한다. 그 효과는 전문가 붕괴에
대한 학습 시점 정규화이다. NEAS의 궤적은 10 에포크 전반에 걸쳐 단조 상승하며,
태스크별 상승은 12개 중 11개 태스크에 퍼진다(7개 binary 중 6개가 $+$0.0004 ~
$+$0.0029 개선; nba_primary는 F1을 $+0.0056$ 상승). BRP와 달리 NEAS는 어느 단일
어려운 태스크에서도 큰 구제를 만들지 않는다 --- next_mcc의 F1은 NEAS 단독으로는
0.0100에서 0.0107로만 이동하며, BRP-detached의 $+$0.0256과 대비된다 --- 그
메커니즘이 targeted-correction이 아니라 prevention-of-loss이기 때문이다.

따라서 두 긍정적 레시피는 *분리된 축*에서 작동한다:

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: 0.5pt,
    inset: 5pt,
    table.header(
      [*Dimension*], [*BRP-detached*], [*NEAS*]
    ),
    [Where it acts],                 [Output-space residual],           [Shared-expert gradients via aux loss],
    [Training signal],               [MSE on primary's detached error], [Task loss on inverse-gate aggregation],
    [Inference overhead],            [Non-zero (residual expert)],       [Zero (training-only regulariser)],
    [Parameter addition],            [0.36M (residual bank)],            [0.17M (aux heads)],
    [Aggregate $Delta$ AUC (v14)],   [$-0.0046$ (degraded)],             [$+0.0017$ (positive)],
    [Aggregate $Delta$ AUC (v13)],   [$-0.0007$ (tied)],                 [$+0.0011$ (positive)],
    [Typical per-task pattern],      [One big rescue ($+$256% next_mcc)], [Many small lifts ($plus.minus 0.003$)],
    [Failure mode if stacked],       [NDCG\@3 advantage erased],         [AUC lift erased],
  ),
  caption: [이 벤치마크에서 식별된 두 비자명 융합 레시피의 구조적 비교. NEAS의
  양의 판정은 v13과 v14 phase0 세대 전반에서 견고하다. BRP-detached의 v13 tied
  판정은 v14에서 저하되는데, 더 깨끗한 v14 피처가 CGC 베이스라인을 더 높이
  끌어올려(0.8234 vs 0.6728) residual bank가 더 강한 primary tower와 경쟁하기
  때문이다. 태스크별 패턴 설명과 stacking 실패 모드는 정성적으로 이어진다 ---
  아래의 태스크별 분석은 해당 진단이 원래 수행된 v13 phase0에서 보고된다.]
) <tab:two-recipes>

=== 결론: 세 구조적 영역, 비가산적 합성

9개 실행에 걸쳐, 세 구조적 영역이 CGC 게이팅을 갖춘 이종 전문가 PLE 위에서의
융합 증강 공간을 조직한다:

+ *표현-가산적 융합은 집계 AUC에서 실패한다.* 다섯 변형 --- loss-level
  adaTT, AdaTT-sp, M1 complement, ECEB, 그리고 BRP-MV(그 `shared_concat`
  입력이 residual gradient를 shared 전문가로 전파) --- 는 모두 residual-error
  신호를 primary-표현 경로로 주입하거나 전파한다. 다섯 모두 AUC를 저하시키며,
  개입의 침습성(invasiveness)과 저하 크기 사이에 단조 관계가 있다($-0.001$ ~
  $-0.008$). residual의 정의(게이트 역수, 자기 전문가, uncertainty-gated
  consensus, 부스팅 오류)는 이를 바꾸지 않는다. 중요한 것은 메커니즘이
  primary-지원 표현에 손을 뻗는지 여부이다.
+ *gradient 격리를 동반한 출력 공간 부스팅: v13에서 tied, v14에서 저하.*
  BRP-detached는 residual을 출력 공간에 두고 primary의 detached 오류에 대한
  부스팅 보정으로 학습시키면서, residual-MSE gradient가 shared 전문가에 결코
  도달하지 않도록 `shared_concat` 입력을 detach한다. v13 phase0에서 결과는
  집계 AUC에서 CGC와 동률이었고($Delta = -0.0007$; best epoch $+0.0008$) F1
  macro를 $+0.007$ / NDCG\@3을 $+0.015$ 끌어올렸으며, 가장 어려운 다중클래스
  태스크에서 상대 $+256$% 구제를 보였다. v14에서는 판정이 바뀐다:
  $Delta$ AUC = $-0.0046$ (degraded), 그러나 NDCG\@3은 PLE-full과 함께
  0.7871($+0.025$ vs CGC)로 공동 최고에서 동률이다. 더 깨끗한 v14 피처가 CGC
  primary tower의 천장을 높여, residual bank가 v13에서 활용할 수 있었던 여유
  공간(head-room)을 좁힌다.
+ *학습 시점 부하 분산 정규화: v13과 v14에서 모두 긍정.* NEAS는 전문가 출력의
  inverse-gate-weighted aggregation에 태스크별 auxiliary head를 붙여 primary
  레이블에 대해 학습시킨다. 집계 AUC를 v13에서 $+0.0011$, v14에서 $+0.0017$
  끌어올리며 --- phase0 세대 전반에 걸쳐 양의 AUC 판정을 재현하는 유일한
  메커니즘이다. 이 메커니즘은 추론 시 제로 오버헤드(학습 전용 정규화)이며
  0.17M 파라미터만 추가한다.

*두 긍정적 레시피는 분리된 축에서 작동하며*(출력 공간에서의 오류 보정 대
게이트에서의 전문가 붕괴 방지) 독립적으로 재현 가능하다. 이 둘을 쌓은 아홉
번째 실험은 *비가산적* 결과를 낳는다: NEAS의 집계 AUC 상승이 사라지고
($Delta = -0.0006$), BRP-detached의 어려운 태스크 구제는 부분적으로 살아남는다
(next_mcc F1 $+0.0250$ vs 단독 $+0.0256$). 메커니즘 수준의 설명은, NEAS가
shared 전문가를 *일반론자(generalists)* 쪽으로 밀어붙이는(각 전문가가 inverse-gate
재가중 하에서 예측해야 함) 반면, BRP-detached는 primary 태스크 손실을 통해
shared 전문가를 *primary-지원 최적점*에 붙들어 둔다는 것이다. shared 전문가는
유한한 자원이며 두 압력을 동시에 만족시킬 수 없다.

따라서 실무적 해석은 stack-everything이 아니라 목표별(per-objective)이다:

- 집계 AUC와 균일한 cross-task 견고성이 가장 중요하다면 *NEAS*를 사용하라
  (제로 추론 오버헤드, v13에서 $+0.0011$ AUC / v14에서 $+0.0017$, 단조 학습
  궤적).
- 다중클래스 순위 품질(NDCG\@K)이 가장 중요하다면 *BRP-detached*가 두 phase0
  세대 모두에서 경쟁력이 있어 NDCG\@3을 v13에서 $+0.015$, v14에서 $+0.025$
  끌어올린다 --- 단, 더 깨끗한 피처가 residual bank의 유효 여유 공간을 줄인
  트레이드오프로서 v14의 AUC 하락($Delta = -0.0046$)을 감수해야 한다.
- 이 둘을 있는 그대로 *쌓지 말라*. 그렇게 하면 기대한 가산적 상승을 만들지
  못한 채 NEAS의 AUC 이득을 지운다. shared-expert 충돌을 해소하는 메커니즘
  (예: NEAS와 BRP-detached 압력을 학습 단계에 걸쳐 번갈아 적용하는 스케줄러,
  또는 두 head 사이의 파라미터 공유 어댑터)은 자연스러운 후속 연구이나 여기서는
  다루지 않는다.

== Finding 8: 인과 전문가의 인접 행렬은 죽은 파라미터였다 <find8>

인과 전문가의 역할을 재해석하는 후속 연구를 위한 사전
진단 과정에서, 베이스라인 아키텍처 자체에서 예기치 못한
실패가 드러났다. 인과 전문가의 학습 가능한 인접 행렬
$bold(W) in RR^(32 times 32)$ --- 이 전문가가 학습해야 하는
DAG 구조 --- 가 우리가 검사한 모든 체크포인트에서 사실상
0으로 collapse 해 있었던 것이다. 이 현상은 특정 시나리오의
아티팩트가 아니라 일반적이다:

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: 0.5pt,
    inset: 5pt,
    table.header([*Checkpoint*], [*$bold(W)$ Frobenius*], [*Entries* $abs(W) > 0.01$]),
    [struct_13_ple_sigmoid (CGC baseline)], [0.0001], [0%],
    [struct_13_residual_complement (M1)],  [0.0001], [0%],
    [struct_13_eceb (MV)],                 [0.0003], [0%],
    [struct_13_brp_detached],              [0.0001], [0%],
    [upstream on-prem implementation],     [0.0001], [0%],
  ),
  caption: [독립적으로 학습된 다섯 개 체크포인트에 걸친 인과
  전문가의 인접 행렬. 포팅 이전의 upstream on-prem 구현에서
  나온 두 개를 포함한다. 모든 경우에서 $bold(W)$의 Frobenius
  norm 은 무작위 init 스케일보다 낮으며, 단 하나의 비대각
  엔트리도 크기가 0.01 을 넘지 않는다. 학습 후에는 이 전문가가
  학습해야 할 DAG 가 존재하지 않는다.]
) <tab:W-collapse>

=== 근본 원인: DAG 를 우회하는 잔차

이 전문가의 forward pass 는 다음과 같다.

$ bold(z)_"hat" = bold(z) + bold(z) bold(W)^2 $

여기서 $bold(z) = "feature_compressor"(bold(x))$ 이며
$bold(z)_"hat"$ 는 하류의 `causal_encoder` MLP 로 입력된다. 잔차
항 $bold(z)$ 는 $bold(W)$ 와 무관하게 잠재 내용 전체를 전달하므로,
태스크 손실에는 $bold(W)$ 를 0 에서 밀어낼 구조적 유인이 없다.
NOTEARS 의 비순환성(acyclicity) 및 희소성(sparsity) 정규화는
모두 $bold(W)$ 가 조밀해지지 않도록 페널티를 주지만, 어느 쪽도
$bold(W) = 0$ 에 페널티를 주지 않는다 --- 사실 희소성 항은
그 지점에서 *최소화*된다. 따라서 결합된 목적함수의 전역
최적점은 $bold(W) = 0$ 이며, 학습은 안정적으로 거기로 수렴한다.

학습된 gradient 도 동일한 문제를 안고 있다.
$partial bold(z)_"hat" slash partial bold(W)$ 를 통한 태스크 손실
기여와 NOTEARS 재구성 gradient(추가될 경우) 모두 $bold(W)$
자체에 비례한다.

$ (partial) / (partial bold(W)) "trace" ((bold(W) dot.circle bold(W))^k)
    thin prop thin bold(W) $

0 에 가까운 초기화는 0 에 가까운 gradient 를 낳으므로,
$bold(W) = 0$ 은 최적화기가 스스로 빠져나올 수 없는 *saddle
point* 이다.

=== 패치: 재구성 손실 + 초기화 재스케일

두 가지 변경, 둘 다 load-bearing 이다.

+ *재구성 정규화.* 원본 NOTEARS 논문은
  $||bold(X) - bold(X) bold(W)||_F^2$ --- 명시적 재구성 신호 ---
  를 최소화한다. 우리는 압축 잠재(compressed-latent) 버전을
  `get_dag_regularization()` 의 세 번째 항으로 채택한다.

  $ cal(L)_"recon" = "mean" ((bold(z) - bold(z) bold(W)^2)^2), quad
    lambda_"recon" = 0.5 $

  이는 원본 논문이 의존하는, $bold(W)$ 에 대한 직접적 압력을
  다시 도입한다.

+ *초기화 재스케일.* 초기값 $bold(W) tilde cal(N)(0, 0.01^2)$
  은 너무 작았다. 그 $bold(W)^2$ 엔트리는 $10^(-4)$ 규모에
  위치하는데, 이는 태스크 손실이든 재구성 손실이든 gradient 가
  사실상 0 이 되는 스케일이다. init 을 $cal(N)(0, 0.1^2)$ 로
  재스케일하면 $bold(W)^2$ 이 $O(10^(-2))$ 스케일을 유지하면서
  (여전히 잔차 경로에 대한 작은 섭동), 초기 학습 동안 gradient
  를 전파할 만큼의 크기를 갖게 된다.

어느 한 변경만으로는 불충분하다. init 재스케일 없는 재구성은
10-에포크 SageMaker 런에서 검증한 결과 $bold(W) approx 0$ 을
남겼고, 재구성 없는 init 재스케일은 표준 NOTEARS 압력은
복원하지만 gradient 소실 문제는 그대로 남긴다.

=== 패치 후 검증

두 변경을 모두 적용한 10-에포크 SageMaker 런은 이 코드베이스
에서 처음으로 비자명한 DAG 를 산출한다.

- $bold(W)$ Frobenius: $bold(0.338)$ (init 스케일은 0.1 이었음)
- 0.01 에서의 $abs(W)$-임계값 희소성: 7.3%
- 비순환성 $h(bold(W)) = 0$ (유효한 DAG)
- 평균 self-loop 강도: $0.000$ (의도대로 대각이 억제됨)
- $W_(i j)^2$ 기준 상위 엣지: `var_23 -> var_13` (0.019),
  `var_9 -> var_13` (0.009), `var_15 -> var_11` (0.007)

희소성 비율은 목표 범위(논문 권장 5--15%) 안에 있으며, 상위
엣지들은 일관된 sink(`var_13`)를 보이는데, 이는 잠재 DAG 가
나타내야 할 종류의 허브 구조다.

=== 종합 태스크 지표는 변하지 않는다

collapse 를 패치해도 하류 태스크 성능은 변하지 *않는다*. 동일한
SageMaker softmax-gate 10-에포크 런에서, 종합 지표는 패치 이전
softmax 베이스라인과 노이즈 범위 안에 있다.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    stroke: 0.5pt,
    inset: 5pt,
    table.header(
      [*Run*], [*AUC*], [*F1 macro*], [*NDCG\@3*], [*MAE*]
    ),
    [Pre-patch local (softmax, $bold(W) approx 0$)],  [0.6729], [0.2009], [0.6814], [0.9598],
    [Post-patch SageMaker (softmax, $bold(W)$ learned)], [0.6719], [0.2042], [0.6875], [0.9597],
    [$Delta$],                                        [$-0.001$], [$+0.003$], [$+0.006$], [$0$],
  ),
  caption: [W-collapse 패치 후 태스크 지표 변화. AUC, F1 macro,
  NDCG\@3, MAE 전반의 차이는 9-way fusion 비교(Finding 7)에서
  관측된 노이즈 대역 안에 있다. 구조적 버그는 실재했으나, 그
  해결 자체가 태스크 개선으로 이어지지는 않는다.]
) <tab:W-patch-metrics>

=== 시사점

진단적 발견은 구조적이지만 지표상의 발견은 null 이다. 두 가지
해석이 가능하다.

+ 인과 전문가는 주로 자신의 `causal_encoder` MLP 를 통해 예측에
  기여해 왔으며, 이 MLP 는 의미 있는 $bold(W)$ 없이도 적합되는
  일반적인 비선형 변환이다. 의미 있는 $bold(W)$ 를 추가하면
  잠재 경로($bold(z)_"hat" = bold(z) + bold(z) bold(W)^2$)가 조금
  바뀌지만, 하류 encoder 가 적응하므로 앙상블의 최종 예측은
  동일하게 보인다. 현재 아키텍처에서 DAG 는 예측에 기능적으로
  쓰이는 것이 아니라 *장식적(decorative)*이다.
+ DAG 에 의존하는 설명가능성 주장(attribution, 반사실(counterfactual)
  probe, reason-code 생성)에 대해서는 이것이 중요하다. 패치
  이전에는 DAG 가 부재했으므로, `get_causal_graph()` 위에 세운
  그러한 주장은 init 스케일에서 사실상 노이즈를 가져오고 있었다.
  패치 이후에는 DAG 가 존재하고 구조화되어 있지만, 그 자체가
  예측 경로로 라우팅되지는 않는다.

이는 학습된 DAG 가 전문가의 내부 표현뿐 아니라 예측까지
load-bearing 한 경로를 갖도록 인과 전문가의 역할을 재정의하는
별도의 구조적 연구(본 논문 범위 밖)를 동기화한다. 후보로는 DAG
를 직접 소비하는 attribution head, 인과 전문가에서 태스크별
게이트로 가는 라우팅 신호, 반사실 probe head 등이 있다. 여기서
보고한 W-collapse 패치는 그러한 탐색들의 전제 조건이다 --- 패치
없이는 라우팅할 DAG 자체가 존재하지 않기 때문이다.

== Finding 9: Causal Explainability Head (CEH) — 첫 번째 Axis-3 시도 <find9>

인과 전문가의 역할을 재배선하는 여러 Axis-3 후보(CEH / CG /
CTGR / CRCG / CCP) 중 첫 번째로, CEH 는 인과 전문가 위에 샘플별,
피처별 attribution 벡터를 추가한다. 동기는 명확하다. Finding 8
이 전문가에게 실제 DAG 를 부여했으니, 이제 그 DAG 가 전문가의
내부를 떠나 소비자에게 도달해야 한다. CEH 는 "DAG 가 존재한다"
에서 "DAG 가 감사 로그가 영속화할 수 있는 예측 단위 출력을
갖는다"로 가는 가장 짧은 경로다. 이 감사 로그는 동반 서빙 논문(Paper 2)에
있으며, 여기서 CEH 는 그 논문의 *선택적 포렌식 보강*을 산출할 뿐 Paper 2가
의존하는 기능은 아니다.

=== 설계

작은 MLP head 가 전문가의 64-dim 출력을 다시 $"input_dim"$ 폭의
attribution 벡터로 매핑하며, 입력에 대한 전문가 자신의 스칼라
출력의 gradient $times$ input saliency 베이스라인과 정렬되도록
MSE 로 학습된다.

$ cal(L)_"attr" = "mean"(abs("head"("output") - (gradient_x "output".sum() dot.circle x))^2) $

gradient $times$ input 은 `requires_grad=True` 로 복제한 입력
복사본에 대해 추가 forward pass 한 번으로 계산되므로, 주
forward 그래프는 건드리지 않는다. 비용은 인과 전문가 내부에서
$approx$ 14% 추가 연산이다(shared-expert 시간의 약 $1 slash 7$).
주 예측 경로는 변하지 않으며, CEH 는 학습 시점의 정규화이자
추론 시점의 부가 출력이다.

추론: `expert._last_attribution` 이 길이 $"input_dim"$ 의 샘플별
벡터를 노출한다. 감사 로그와의 파이프라인 측 통합(HMAC 서명
영속화, SR 11-7 MRM)은 `AuditLogger.log_attribution` 을 통해
Paper 2 v2 에서 제공된다.

=== MV 결과 (SageMaker teacher_full + CEH, 10 epochs, softmax gate)

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt,
    inset: 5pt,
    table.header(
      [*Metric*], [*Pre-patch (W ≈ 0)*], [*Post-patch (no CEH)*], [*Post-patch + CEH*]
    ),
    [Primary AUC],                [0.6729], [0.6719], [*0.6734*],
    [F1 macro],                   [0.2009], [0.2042], [0.1994],
    [NDCG\@3],                    [0.6814], [0.6875], [0.6842],
    [MAE],                        [0.9598], [0.9597], [0.9609],
    [$bold(W)$ Frobenius],        [0.0001], [0.338],  [*0.366*],
    [$abs(W) > 0.01$ sparse edges], [0%], [7.3%], [*8.5%*],
    [Attribution head trained],   [n/a],   [n/a],    [*yes*],
  ),
  caption: [CEH MV 결과. Primary AUC 는 노이즈 범위 안에서
  보존되며, $bold(W)$ Frobenius 와 sparse-edge 비율은 모두 패치
  후 no-CEH 런보다 약간 높다. 이는 attribution-head gradient 가
  DAG 를 강화하는 추가적 구조 신호를 제공함을 시사한다.
  attribution head 학습은 head 의 가중치와 bias 에 대한 사후
  검사(zero-init 에서 출발한 비영 bias)로 검증되었다.]
) <tab:ceh-mv>

attribution head 의 layer-3 bias(init = $bold(0)$)는 원소당
$||bold(b)|| = 0.03 plus.minus 0.04$ 로, layer-0 bias 는
$0.08 plus.minus 0.08$ 로 이동했는데, 이는 MSE 정렬 손실이
head 에 gradient 를 기여했다는 직접적 증거다.

=== MV 가 검증하지 못하는 것

CEH MV 는 다음을 확인한다.

+ head 가 주 경로를 교란하지 않고 학습된다.
+ W-collapse 패치(Finding 8)가 추가 attribution-학습 신호
  아래에서도 그대로 유지된다 --- 실제로 DAG 의 Frobenius norm 과
  sparse-edge 수가 모두 약간 증가한다.
+ 예측 단위 attribution 벡터가 존재하며 하류에서 소비될 수 있다.

CEH MV 가 아직 검증하지 *못하는* 것은 다음과 같다.

- *Attribution 품질.* head 는 구성상 gradient $times$ input
  타겟에 적합된다. 학습된 출력이 샘플별 신호를 담는지, 아니면
  단지 전역 중요도 패턴을 재현하는지는 전용 사후 평가가
  필요했다. 그 평가(Section 4.9.3)는 원본(raw) 타겟 아래에서
  거의 전역적인 collapse 를 보였고, collapse 를 해소한 v2
  iteration(Section 4.9.4)을 동기화했다.
- *감사 로그 유용성.* 배선 자체는 이후 Paper 2 v2
  (`AuditLogger.log_attribution`, HMAC 서명 해시 체인 예측 단위
  레코드)에서 제공되었다. 본 논문이 검증하지 않는 것은 영속화된
  attribution 이 규제기관에 *유용*한지 여부인데, 이는 아래에서
  평가하는 품질 차원에 달려 있다.
- *하류 지표 영향.* CEH 의 작은 AUC 상승은 패치 후 softmax 대비
  $+0.0015$ 로, 9-way 비교의 노이즈 대역 안에 있다. 우리는 이를
  유의미한 개선으로 주장하지 않는다.

=== Attribution 품질 평가 (사후)

MV 만으로는 head 가 *학습됨*만 보이므로, 학습된 head 가 의미
있는 샘플별 설명을 산출하는지 아니면 단지 평활화된 전역 패턴을
산출하는지 묻기 위해 5,000 개 검증 샘플에 대한 전용 사후 평가를
수행했다. 네 가지 측정:

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    stroke: 0.5pt,
    [*Measurement*], [*Value*], [*Interpretation*],
    [Spearman corr. (CEH vs. grad $times$ input)], [mean $0.259$, median $0.252$], [Partial fit to training target],
    [Between-sample / within-sample variance], [$0.055$], [Attribution varies much more within a sample than across samples],
    [Mean top-10 feature overlap across samples], [$0.791$], [Different samples share $~80%$ of their top features],
    [Stability under input noise ($sigma = 0.05$)], [Spearman $0.985$], [Very stable (trivially consistent with near-global output)],
  ),
  caption: [패치 후 CEH 체크포인트에서의 CEH attribution 품질
  측정. 종합하면, 낮은 between-over-within 비율($0.055$)과 높은
  top-10 overlap($0.791$)은 head 가 작은 샘플별 섭동만 있는 전역
  중요도 벡터로 대체로 collapse 했음을 가리킨다 --- 예측 단위
  attribution head 가 산출해야 하는 것의 정반대다.],
)<tbl:ceh-quality>

그룹별 attribution mass 도 이 평탄함을 뒷받침한다. CEH 는 동일
샘플에서 grad $times$ input 의 $44.7%$ 대비 txn\_behavior 에
$32.1%$ 를 할당하며, product\_holdings($12.6%$ vs. $5.5%$)와
product\_hierarchy($11.3%$ vs. $3.8%$)에 과대 가중한다. 학습된
분포는 자신의 학습 타겟보다 명백히 더 평탄하다.

*해석.* head 는 타겟에 약하게만($rho approx 0.26$) 적합하며 그
타겟의 샘플별 성분을 거의 전적으로 버린다. 가장 유력한 메커니즘:
타겟 자체 --- causal encoder 의 합산 출력의 grad $times$ input
--- 가 큰 샘플 불변 성분을 갖고 있어서, 64-hidden single-layer
MLP 가 샘플별 잔차를 무시한 채 그 전역 성분을 낮은 손실로 포착할
수 있다는 것이다. head 가 망가진 것이 아니라, 얇은 MLP 가 샘플별
판별을 강제당하기에는 타겟이 너무 평탄하다.

*이것이 배제하는 것/포함하는 것.*

- 포함(Rules in): 인프라 경로(Finding 9 MV)는 옳다 --- 기능적
  DAG 가 주 예측이나 DAG 자체를 불안정하게 하지 않으면서
  attribution 소비자에게 입력을 공급한다.
- 배제(Rules out): 현재 타겟 설계(causal-encoder 출력 합의
  grad $times$ input)는 규제기관이 활용 가능한 샘플별(국소) 설명
  --- 금융위 가이드라인 §4.4 가 권고하는 국소 설명에 대응하는 후보 ---
  을 산출하기에 *충분하지 않다*. 샘플별 attribution 품질에 의존하는
  추가 Axis-3 후보(특히 CRCG)는 타겟 정련 없이 이 베이스라인에
  대해 평가되어서는 안 된다.

*타겟 정련 후보.*

+ *Demeaned 타겟:* grad $times$ input 을 감독으로 쓰기 전에 배치
  평균을 뺀다. head 가 전역 패턴을 재학습하는 대신 전역 패턴으로
  부터의 샘플별 편차를 학습하도록 강제한다. 가장 작은 코드
  변경이며, "타겟이 너무 평탄하다" 가설을 직접 검증한다.
  *실행함; 아래 Section 4.9.4 참조.*
+ *Primary-prediction 타겟:* causal encoder 출력의 grad 를 특정
  태스크 logit(예: churn\_signal)의 grad 로 교체한다. 태스크별
  head 비용을 치르는 대신, 하류 소비자가 실제로 설명을 요구하는
  대상에 attribution 을 정렬한다.
+ *더 크고/깊은 head:* hidden dim 을 두 배로 하고 layer 를 하나
  추가하여, 한계 요인이 타겟이 아니라 64-hidden 병목인지 검증한다.
+ *DAG-path 타겟:* 학습된 $bold(W)$ 를 사용해 피처-출력 영향
  경로를 감독으로 구성하여, 국소 gradient 기반에서 구조적-그래프
  기반 attribution 으로 전환한다.

=== Iteration v2: Demeaned 타겟이 collapse 를 해소한다

"타겟이 너무 평탄하다" 가설의 최소 개입 검증으로 후보
1(demeaned 타겟)을 실행했다. 설정 플래그
`ceh.target_mode: "demeaned"` 는 MSE 이전에 grad $times$ input
감독에서 피처별 배치 평균을 빼며, 그 외 모든 것은 v1 MV 런과
동일하다(동일 아키텍처, 동일 하이퍼파라미터, 동일 데이터,
SageMaker g4dn.xlarge 에서 동일 10 에포크). 5,000 개 검증
샘플에 대한 동일한 사후 평가는 다음을 산출한다.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    stroke: 0.5pt,
    [*Measurement*], [*Raw (v1)*], [*Demeaned (v2)*], [*Change*],
    [Between-sample / within-sample variance], [$0.055$], [$0.719$], [$13 times$ larger],
    [Top-10 feature overlap across samples], [$0.791$], [$0.281$], [$65%$ smaller],
    [Stability under input noise ($sigma = 0.05$)], [Spearman $0.985$], [Spearman $0.953$], [Still stable],
    [Primary AUC (churn\_signal)], [$0.6866$], [$0.6870$], [Within noise],
  ),
  caption: [CEH attribution 품질, v1(raw grad $times$ input
  타겟) vs. v2(demeaned 타겟). 두 판별력 측정값은 한 자릿수
  이상으로 변하는 반면 primary AUC 는 변하지 않는다 --- head 가
  이제 전역 중요도 벡터를 재현하는 대신 샘플별 편차를 학습하며,
  하류 태스크에 비용이 들지 않는다.],
)<tbl:ceh-v1-v2>

이 결과는 Section 4.9.3 의 가설을 확인한다. collapse 는
head 용량이나 아키텍처 한계가 아니라 학습 타겟 아티팩트였다.
최소 개입(세 줄과 설정 플래그 하나)이 샘플별 판별력을 복원한다.

그룹별 attribution mass 도 demeaned 타겟 아래에서 상당히
재균형된다. txn\_behavior 는 $32.1%$ 에서 $18.3%$ 로,
gmm\_clustering 은 $28.5%$ 에서 $21.4%$ 로 떨어지는 반면,
product\_hierarchy 는 $11.3%$ 에서 $30.3%$ 로,
product\_holdings 는 $12.6%$ 에서 $21.3%$ 로 뛴다. demeaned
head 는 raw head 를 지배했던 전역적으로 고분산인 그룹보다,
샘플별 구별 신호를 담는 피처 그룹(상품 taxonomy)을 선호한다.

*유의점.*

- *raw* grad $times$ input 에 대한 Spearman 상관은 v1 과 v2
  사이에서 $0.259$ 에서 $0.096$ 으로 떨어진다. 이는 예상된 것이며
  회귀(regression)가 아니다. v2 head 는 demeaned grad $times$
  input 에 대해 감독되므로, raw 버전은 더 이상 그 학습 타겟이
  아니다. 품질은 @tbl:ceh-v1-v2 의 타겟 독립적 지표(분산 비율,
  top-K overlap, 안정성)로 측정된다.
- 샘플별 판별력은 규제기관이 활용 가능한 설명을 위한 필요
  조건이지 충분 조건은 아니다. 샘플 사례에 대한 인간 평가와 도메인
  기대치와의 정렬은 본 논문 범위 밖이며, 감사 로그 통합은 동반
  논문의 v2 감사 인프라에 구현되어 있다.
- v2 는 Santander 에서의 단일 시드 10-에포크 런으로 보고한다.
  시드 간 안정성과 데이터셋 간 재현은 여기서 검증하지 않는다.

*다음 iteration 방향(미실행).* Primary-task-gradient 타겟이
자연스러운 다음 검증으로 남는다. causal-encoder 출력 합을 특정
태스크 logit 으로 교체하면 하류 소비자가 실제로 묻는 것("이
고객은 왜 높은 churn 점수를 받았는가?")에 attribution 을
정렬한다. demeaned 인프라는 그대로 유지되며, gradient 를 취하는
스칼라만 바뀐다. 더 큰 head 와 DAG-path 타겟은 우선순위가 낮은
대안으로 남는다.

=== 왜 CEH 가 먼저인가 (CG / CTGR / CRCG / CCP 가 아니라)

다섯 개 Axis-3 후보 중 CEH 가 구조적 풋프린트가 가장 작다. MSE
감독 신호를 갖는 단일 MLP head 하나로, 태스크 라우팅 변경(CTGR
대비)도, 서빙 시점 경로 분기(CG 대비)도, 모듈 간 배선(Paper 2 의
reason generator 에 엮이는 CRCG 대비)도 없다. 이는 더 무거운
재설계에 착수하기 전에, 가능한 한 가장 깨끗한 검증으로 기본
전제 --- 하류 소비자가 이제 기능하는 DAG 에서 의미 있는 무언가를
추출할 수 있다는 것 --- 를 확인하게 해준다. 후속 후보들은
no-causal-routing-at-all 이 아니라 CEH 의 베이스라인에 대해
평가된다.

== Finding 10: Causal Guardrail (CG) --- 두 번째 Axis-3 시도 <find10>

CEH 가 "모델은 왜 이것을 추천했는가?"에 답한다면, CG(Causal
Guardrail)는 인접한 질문 "이 추천을 신뢰할 수 있는가?"에
답한다 --- 의심스러운 입력을 조용히 통과시키지 않고 폴백이나
사람 검토로 라우팅하는 예측 단위 신뢰성 신호다. SR 11-7 "known
limitations" 보고와 EU AI Act 제9조 위험 관리는 모두 이를 위한
런타임 메커니즘을 요구한다. 2026년 6월 시행된 금융위 「금융분야
인공지능 가이드라인」@koreafsc2024 도 금융안정성 원칙(§5.2)의 사후
긴급정지와 보조수단성 원칙(§3)의 고위험 결정에 대한 사람 개입(HITL)을
요구하는데, CG 는 이러한 런타임 폴백 메커니즘에 대응하는 예측 단위
신뢰성 신호의 설계 후보이다(운영 배선은 향후 과제다). 인과 전문가에서의
자연스러운 hook 은 그 DAG 구조다.

=== MV 정식화 v1 (W-Reconstruction) 은 실패한다

첫 번째 정식화는 Finding 8 의 NOTEARS 재구성 잔차를 재사용했다.
학습은 $bold(W)$ 를 0 saddle 밖으로 유지하기 위해
$||bold(z) - bold(z) bold(W)^2||_F^2$ 를 최적화하며, 서빙
시점에 샘플별 버전
$||bold(z)_i - bold(z)_i bold(W)^2||^2 / ||bold(z)_i||^2$ 는
*학습된 DAG* 가 입력의 압축 인과 상태를 얼마나 잘 재구성하는지에
대한 직접적 척도다. 잘 적합된 입력은 깨끗하게 재구성되고, OOD
입력은 그렇지 않아야 한다.

teacher_ceh_demeaned 체크포인트에서 5,000 개 검증 샘플과 세 개의
합성 OOD probe([-3,3] 균등 무작위, 컬럼 치환, 극단 꼬리
백분위수)로 평가했다. 모든 분포가 1.0 근처의 좁은 창 안에
집중되었다(in-dist median 0.9995, p99 1.0055; OOD median
0.9998--0.9972). in-distribution 집합에서 교정한 p95 임계값에서
OOD probe 의 TPR 은 6.8% / 8.1% / 0% 로 --- 5% FPR 베이스라인
대비 사실상 우연 수준이었다.

실패 메커니즘은 Finding 8 의 동일한 "장식적 DAG" 관찰이다.
학습된 $bold(W)$ 가 충분히 작아서($8 times 8$ 행렬에서
$||bold(W)||_F approx 0.36$ 이므로 $bold(W)^2$ 는 Frobenius 로
$O(0.13)$) $bold(z) bold(W)^2$ 가 $bold(z)$ 를 거의 섭동하지
않는다. 입력이 DAG 에 적합하든 아니든 잔차 비율이
$approx 1.0$ 이므로, 신호가 퇴화(degenerate)한다.

=== MV 정식화 v2 (z-Space Mahalanobis) 는 작동한다

$bold(W)$ 자체가 판별을 추동하기에는 너무 약하므로, *인과
잠재(causal latent)* $bold(z)$ 가 독립적으로 분포 신호를 담는지
검증했다. $bold(mu), bold(sigma)$ 를 in-distribution 배치에
대한 $bold(z)$ 의 차원별 평균과 표준편차라 하면, 점수는 표준화
제곱합 $d_i = sum_j ((z_(i,j) - mu_j) / sigma_j)^2$ 이다.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: 0.5pt,
    [*Probe*], [*median*], [*p95*], [*p99*], [*max*],
    [In-distribution], [$23.6$], [$62.8$], [$268.1$], [$983.6$],
    [Uniform random], [$749.5$], [$1200.9$], [$1458.5$], [$1936.7$],
    [Column-permuted], [$537.4$], [$1076.0$], [$1336.1$], [$1873.9$],
    [Extreme-tail], [$479.3$], [$479.3$], [$479.3$], [$479.3$],
  ),
  caption: [5,000 개 검증 샘플과 세 개의 합성 OOD probe 에 대한
  z-space Mahalanobis CG v2 점수 분포. 모든 OOD median 이
  in-distribution $"p99"$ 위에 놓여, 거의 완벽한 분리를 보인다.],
)<tbl:cg-v2-dists>

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    stroke: 0.5pt,
    [*Probe*], [*TPR \@ ID p95*], [*FPR \@ ID p95*],
    [Uniform random], [$100.0%$], [$5.0%$],
    [Column-permuted], [$100.0%$], [$5.0%$],
    [Extreme-tail], [$100.0%$], [$5.0%$],
  ),
  caption: [권장 CG v2 임계값(in-distribution $"p95" = 62.8$)에서의
  OOD 탐지율. 세 probe 유형 모두 예상 거짓 양성률에서 플래그된다.],
)<tbl:cg-v2-rates>

따라서 CG 는 DAG 가중치를 통해서는 실패하지만 인과 잠재를 통해서는
작동한다. 잠재는 W-reconstruction 이 노출하지 않는 구조를 담고
있다. 서빙 통합: 호출자가 참조 모집단에서 $bold(mu), bold(sigma)$
를 한 번 계산해 캐시하고, 추론 시점에 샘플별 점수를 임계 처리하여
통과(pass-through)와 폴백을 결정한다.

=== CG MV 가 검증하는 것 / 못하는 것

검증된 것:
+ z-space 정식화는 질적으로 서로 다른 세 OOD probe 를 사실상
  $100%$ TPR, $5%$ FPR 로 탐지한다 --- 이 primitive 는 작동한다.
+ v1 vs v2 비교는 신호가 실제로 *어디에* 존재하는지를 분리한다:
  학습된 DAG 가중치가 아니라 인과 잠재다. $bold(W)$ 가 구조적으로
  존재하지만 현재 아키텍처에서 과소 활용된다는 Finding 8 의 관찰을
  강화한다.
+ 학습 비용 0: CG 는 기존 teacher_ceh_demeaned 체크포인트에 대한
  사후 분석이다. 새 SageMaker job 이 없다.

아직 검증되지 않은 것:
- *실세계 OOD*: 세 probe 는 합성이다. 운영 guardrail 은 현실적인
  분포 드리프트(시간적 이동, 하위 그룹 불균형, 적대적 섭동)에도
  발화해야 한다. 이는 Paper 2 의 모니터링 책임이며 여기서
  검증하지 않는다.
- *하류 지표 영향*: CG 는 플래그를 산출한다. 플래그된 예측을
  폴백으로 라우팅하는 것이 종단 간 태스크 지표나 calibration 을
  개선하는지는 아직 측정하지 않았다. 3-layer 폴백 라우터와의 통합이
  요구된다.
- *임계값 드리프트*: 참조 배치에서 계산한 $bold(mu), bold(sigma)$
  는 입력 분포가 변화함에 따라 주기적 재교정이 필요할 것이다.
  여기서 검증하지 않는다.

=== 나머지 Axis-3 후보에 대한 시사점

v1/v2 대비는 CTGR 와 CRCG 에 무엇을 기대할지 명확히 한다. 둘 다
*학습된 인과 구조*($bold(W)$)가 단지 존재하는 것을 넘어 현재
아키텍처에서 정보를 담고 있어야 한다는 점에 의존한다. Finding 10
실험은 $bold(W)$ 가 현재 구조적 용도로 쓰기에 충분히 강하지
않다는 직접적 증거다. CTGR 의 라우팅과 CRCG 의 reason 경로는 CG
v1 을 좌초시킨 동일한 약한 신호를 물려받을 것이다. 그 후보들을
실행할 가치가 있으려면 W 를 증폭(더 큰 init, 더 강한 recon
lambda, 또는 DAG-routed 잔차 경로)하거나, 아니면 CTGR/CRCG 가
가중치가 아니라 잠재에서 끌어오도록 재설계되어야 한다 --- 여기서의
v1$arrow.r$v2 전환을 본떠서. Finding 11(아래)은 증폭 실험을 실행하며
부분적으로 혼재된 답을 산출한다.

== Finding 11: W-증폭 테스트 <find11>

Finding 10은 CG v1이 실패하는 이유에 대해 두 가지 가설을 제시하였다:
(a) 학습된 $bold(W)$가 단순히 너무 작아 의미 있는 재구성을
구동하지 못한다, 또는 (b) $||bold(z) - bold(z) bold(W)^2||^2$
형식 자체가 $bold(W)$의 크기와 무관하게 구조적으로 제한되어 있다.
우리는 한 시나리오(teacher_ceh_w_amp)를 두 가지 증폭 손잡이로
재학습하여 (a)를 검증하였다: $bold(W)$ init scale
$0.1 arrow.r 0.3$, $lambda_"recon"$ $0.5 arrow.r 2.0$. 나머지
하이퍼파라미터는 teacher_ceh_demeaned와 동일하다.

=== 학습 측 결과: W는 증폭되고 주 태스크는 무손상

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: 0.5pt,
    [*Metric*], [*Baseline*], [*W-amp*], [*Change*],
    [$||bold(W)||_F$], [$0.363$], [$5.028$], [$approx 14 times$],
    [Active edges ($|W| > 0.01$)], [$8.5%$], [$59.5%$], [$7.0 times$],
    [Max $|W_(i j)|$], [$0.11$], [$0.77$], [$7.0 times$],
    [Primary AUC (churn\_signal)], [$0.6870$], [$0.6865$], [within noise],
    [Loss], [$25.62$], [$25.61$], [within noise],
  ),
  caption: [W-증폭의 학습 측 결과. 학습된 인접 행렬은 모든
  구조적 척도에서 한 자릿수만큼 성장하지만 주 태스크 지표는
  보존된다. Finding 8의 "장식적 DAG" 관찰은 직접적이고 저렴하게
  뒤집힌다.],
)<tbl:wamp-training>

증폭은 태스크 지표 관점에서 무비용이다: 더 커진 $bold(W)$가
$bold(W)^2$를 Frobenius $0.13$에서 $2.19$로 끌어올려 SCM 잔차
$bold(z) + bold(z) bold(W)^2$가 이제 의미 있게 비자명한 섭동을
기여하지만, 주 태스크 AUC는 단지 $0.0005$만 이동한다(within
noise). 장식적 DAG 체제는 학습 선택의 결과(너무 작은 init과
너무 약한 recon 항)였을 뿐, 아키텍처적 제약이 아니었다.

=== CG v1은 개선되나 여전히 아키텍처적으로 제한됨

14$times$ 더 커진 $bold(W)$로 CG v1의 판별력은 실제로 증가하지만
v2의 한계에는 근접하지 못한다:

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    stroke: 0.5pt,
    [*Probe*], [*Baseline TPR \@ ID p95*], [*W-amp TPR \@ ID p95*],
    [Uniform random], [$6.8%$], [$22.7%$],
    [Column-permuted], [$8.1%$], [$18.6%$],
    [Extreme-tail], [$0.0%$], [$0.0%$],
  ),
  caption: [W-증폭 전후의 CG v1 OOD 탐지(FPR은 $5%$로 고정).
  두 가지 probe 유형에서 판별력이 우연 수준에서 약하지만
  비영(non-zero)으로 개선되나, extreme-tail은 무관하게 영에
  머물러 구조적 한계를 드러낸다.],
)<tbl:wamp-v1>

extreme-tail 결과가 핵심 신호다. 이 probe는 모든 표본을 동일한
99분위 벡터로 설정하므로 모든 표본이 동일한 $bold(z)$를 가지며
따라서 동일한 잔차를 갖는다 --- $||bold(z) - bold(z) bold(W)^2||$는
점 추정치여서 구성상 판별이 불가능하다. 이 형식은 입력의 *분포*에
대한 어떤 관점도 제공하지 못하는 *표본 단위* 척도이므로, 그 어떤
W 증폭으로도 이를 고칠 수 없다. 반면 CG v2의 z-공간 Mahalanobis는
명시적으로 기준 분포로부터의 거리이므로, 퇴화된 OOD 경우를
간단히 처리한다($100%$ TPR).

=== CG v2는 영향받지 않음

v2의 판별력은 증폭 후에도 변하지 않는다: 권장 $"p95"$ 임계값에서
세 가지 probe 전반에 걸쳐 $100%$ TPR. z-공간 경로는 $bold(W)$의
크기에 의존하지 않으며, 증폭은 그것을 돕지도 해치지도 않는다.

=== Finding 11이 정리한 것 / 남겨둔 것

정리된 것:
+ $bold(W)$는 아키텍처적으로 *운명지어진 것이 아니다*. init $0.3$
  + $lambda_"recon" = 2.0$로, 이는 $10$ epoch 안에 비자명한
  희소 구조로 성장하며 주 태스크 비용은 영이다.
+ Finding 10에서의 CG v1 실패는 *부분적으로* 작은 $bold(W)$
  때문이었고(이제 ~$15$--$20$ TPR 점을 기여), *부분적으로*
  형식의 한계(extreme-tail probe가 구조적 한계를 드러냄) 때문이었다.
+ 잠재 공간 CG(v2)는 OOD 탐지에서 여전히 우월하다. W 기반
  가드레일은 기껏해야 보완물이지 대체물이 아니다.

남은 것:
- 증폭된 $bold(W)$가 CTGR과 CRCG를 풀어주는가, 아니면 그
  후보들도 유사한 구조적 한계에 부딪히는가? Finding 11은 CG
  질문만 직접 답한다. 학습 측 증폭은 CTGR/CRCG가 이제 적어도
  Finding 10의 전제조건에 가로막히지 않고 *시도될* 수 있음을
  의미한다.
- 더 공격적인 증폭(init $0.5$, $lambda_"recon" = 5$, 희소성
  제거)이 v1을 사용 가능한 수준으로 끌어올리는가, 아니면 주
  태스크 AUC를 무너뜨리는가? 검증하지 않았다. $0.3 / 2.0$ 설정은
  단일 작업 최소 개입으로 선택되었다.
- 잠재 공간 CG의 $100%$ TPR이 합성 probe가 아니라 *현실적인*
  드리프트(시간적, 하위집단, 적대적)에서도 살아남는가? Paper 2
  모니터링 범위이며, 여기서는 검증하지 않았다.

=== 남은 Axis-3 후보에 대한 갱신된 권장사항

Finding 10은 $bold(W)$를 증폭하거나 형제 후보들을 잠재 중심으로
재설계할 것을 권장하였다. Finding 11은 첫 번째 옵션을 택하고 그것이
부분적으로 작동함을 보인다: $bold(W)$는 이제 의미 있지만, 둘 다
사용 가능할 때 잠재 기반 형식이 여전히 W 기반 형식을 한계 측면에서
앞선다. 따라서 CTGR / CRCG / CCP에 대한 실무적 권장사항은 다음과 같다:

+ 이들을 W-증폭 교사 위에서 실행하라. 그래야 W가 구조적 정보를
  *운반할 수 있는* 상태가 된다.
+ 동시에 각 후보에 대해 잠재 기반 대안을 평가하라. 둘 다 측정
  가능한 모든 경우에서 W 기반 버전이 선호된다고 가정하기보다
  잠재 버전을 이겨야 할 기준선으로 취급하라.

== Finding 12: Counterfactual Probe (CCP) --- Pearl Rung 3 실행 가능성 (예비) <find12>

인과 전문가의 본래 동기 --- "A가 B와 상관관계가 있다"가 아니라
"A가 B를 유발한다"는 설명 @pearl2009causality --- 은 Pearl
인과 위계의 Rung 3, 즉 반사실(counterfactual)에 대한 접근을
요구한다: "피처 $j$가 달랐다면 출력은 무엇이었을까?" Finding 9
(CEH attribution)는 Rung 1(관찰)에 위치한다. Finding 10(CG)은
사다리 밖(안전 모니터링)이다. 반면 CCP는 학습된 DAG가 반사실
질의를 매개하는지를 직접 검증한다.

_이 발견은 합성 벤치마크에서의 단일 시드 사후(post-hoc) 프로브로,
예비적이다. 우리는 이를 검증된 Rung 3 결과가 아니라 실현가능성 신호이자
향후 과제 방향으로 보고한다 --- 아래 수치는 증폭된 DAG가 무엇을 운반하는지를
정량화할 뿐, 시스템이 프로덕션에서 반사실 설명을 제공한다는 주장이 아니다._

=== 형식화

인과 잠재 $bold(z) = "compressor"(x)$와 SCM 잔차
$bold(z) bold(W)^2$를 가진 표본에 대해, 사실적(factual) 출력은
$"encoder"(bold(z) + bold(z) bold(W)^2)$이다. 개입
$"do"(z_j = v)$는 $z'_j = v$이고 $k eq.not j$에 대해 $z'_k = z_k$인
$bold(z)'$를 산출한다. 우리는 세 가지 출력을 계산한다:

- *Factual*: $"encoder"(bold(z) + bold(z) bold(W)^2)$.
- *Direct-only*: $"encoder"(bold(z)' + bold(z) bold(W)^2)$ ---
  개입된 잠재가 직접 공급되지만 $bold(W)^2$-매개 항은 개입 이전
  값으로 고정된다. 이는 비인과적("그냥 피처를 바꾸는") 효과를
  분리한다.
- *Full-CF*: $"encoder"(bold(z)' + bold(z)' bold(W)^2)$ --- 개입이
  두 경로 모두를 통해 전파된다. 이것이 학습된 DAG 하에서의 반사실
  예측이다.

*매개 비율(mediation ratio)* $||"full CF" - "direct only"|| / ||"full CF"
- "factual"||$ 은 반사실 효과 중 DAG가 운반하는 비율을 측정한다.
이 비율이 영에서 떨어져 유계일 때 정확히 Pearl의 Rung 3가
실행 가능하다.

=== 평가: Pearl Rung 3는 W-증폭 이후에만 실행 가능

probe를 1,000개 검증 표본, 32개 인과 잠재 차원 전체, 세 가지
개입 값($v in {-2, 0, 2}$)에 대해 실행한 결과:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: 0.5pt,
    [*Metric (median over $32 times 3$ cells)*],
    [*Baseline (demeaned)*], [*W-amp*], [*Ratio*],
    [$||bold(W)||_F$], [$0.363$], [$5.028$], [$14 times$],
    [Direct effect $||bold(y)' - bold(y)||$], [$3.66$], [$3.45$], [similar],
    [Total effect $||bold(y)_"cf" - bold(y)||$], [$3.66$], [$3.54$], [similar],
    [Mediated effect $||bold(y)_"cf" - bold(y)'||$], [$0.003$], [$0.887$], [$269 times$],
    [*Mediation ratio*], [$0.16%$], [$32.1%$], [$200 times$],
  ),
  caption: [두 체크포인트에 대한 CCP 평가. 기준 W 스케일에서
  매개는 전체 반사실 효과의 $0.16%$로 --- DAG가 본질적으로 아무것도
  운반하지 않으며, 이는 Findings 10--11의 장식적 DAG 관찰과
  일치한다. W가 증폭되면 매개는 중앙값에서 $32%$, 95분위에서
  $61%$로 상승하여, Finding 11의 증폭 레시피 하에서 학습될 때
  학습된 DAG가 반사실 질의를 수행할 수 *있을 수 있음*을 시사한다 ---
  검증된 결과가 아니라 향후 과제로 표시되는 예비 신호다.],
)<tbl:ccp-mediation>

=== 해석

기준 결과는 "장식적 DAG" 관찰의 직접적인 운영적 귀결이다:
$bold(W)^2$가 작으면 잠재 값 하나를 바꿔도 매개 항
$bold(z) bold(W)^2$가 거의 섭동되지 않으므로, 반사실은 "직접
공급에서 피처 하나만 바꾸는 것" --- 퇴화된 Rung 1 연산이지
Rung 3 질의가 아닌 것 --- 으로 붕괴한다. 증폭된 체크포인트는
중앙값에서 효과의 $32%$를 DAG를 통해 라우팅한다. 직접 효과가
여전히 지배적인데, 이는 부분 매개가 Pearl의 이론에서 예측하는
바와 같으나 Rung 3는 더 이상 막혀 있지 않다.

따라서 Finding 12는 이 프로젝트의 Axis-3 후보 하에서 인과
전문가에 대한 Pearl 사다리 그림을 그려 본다(Rung 2는 여전히 미평가):

- Rung 1 (관찰 / 연관): CEH attribution (Finding 9).
- 사다리 밖 (안전): CG coherence (Findings 10--11).
- Rung 3 (반사실): W-증폭 DAG를 통한 CCP 매개
  (Finding 12, 본 절).
- Rung 2 (개입 / 처치 효과): 아직 평가되지 않음. 자연스러운
  다음 단계는 교사에 대한 미니배치 개입 실험(예: 반사실 상품
  제안 효과)이겠으나, 이는 여기서 사용된 합성 벤치마크를 넘어서는
  인과적으로 해석 가능한 학습 데이터를 요구한다.

CCP는 기존의 두 체크포인트에 대해 전적으로 사후(post-hoc)
분석으로 실행되었다. MV 비용은 Findings 9--11 위에 추가되는
SageMaker 지출이 영이다.

== Finding 13: CEH v3 주-태스크-그래디언트 타깃 --- 부정 결과 <find13>

Finding 9의 "아직 검증되지 않음" 목록은 우리가 실행하지 않은
후보 하나로 마무리되었다: attribution 타깃의 스칼라를 인과
인코더의 집계 출력에서 특정 태스크 logit으로 교체하는 것. 그
동기는 다운스트림 소비자가 태스크 용어로 설명을 요구하므로("이
고객이 왜 높은 churn 점수를 받았는가?") 태스크에 정렬된 타깃이
전문가의 내부 출력에 정렬된 타깃보다 더 유용한 표본 단위 신호를
줄 것이라는 점이었다.

=== 구현

CausalExpert에 새로운 `target_mode = "primary_task"`. 타깃은
인과 전문가가 아니라 PLE 학습 루프가 계산한다: 입력의 grad-enabled
복제본에 대한 추가 forward pass가 태스크 logit을 산출하고, 입력의
인과 부분집합에 대한 그 그래디언트를 `torch.autograd.grad`로
계산한 뒤, 입력과 곱하고, 배치 전반에 걸쳐 컬럼별로 demean한다(v2와
동일). attribution 헤드의 main-forward 출력은 보존되어 attribution
손실이 main backward pass에 깔끔하게 통합되도록 한다.

이 구성은 또한 W-증폭을 켜둔 채로 유지하여(init $0.3$,
$lambda_"recon" = 2.0$) 학습된 DAG가 구조적으로 의미 있게 한다
(Finding 11). 특정 태스크는 ``churn\_signal``, 이 프로젝트의
규제-anchoring 이진 태스크이다.

=== 결과: 헤드가 거의-전역으로 재붕괴

5,000개 검증 표본에 대해 동일한 사후 품질 평가를 실행한 결과:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: 0.5pt,
    [*Metric*], [*v1 raw*], [*v2 demeaned*], [*v3 primary-task*],
    [Spearman (CEH vs. raw grad $times$ input)], [$0.259$], [$0.096$], [$-0.035$],
    [Between-sample / within-sample variance], [$0.055$], [$0.719$], [$0.043$],
    [Top-10 feature overlap across samples], [$0.791$], [$0.281$], [$0.799$],
    [Stability under noise ($sigma = 0.05$)], [$0.985$], [$0.953$], [$0.998$],
    [Primary AUC (churn\_signal)], [$0.6866$], [$0.6870$], [$0.6873$],
  ),
  caption: [동일한 평가에 대한 CEH v1, v2(demeaned), v3
  (primary-task-gradient, demeaned). v3는 주 태스크를 해치지 않고
  다른 타깃으로 학습되지만, 그 표본 단위 판별력은 v1 수준으로
  다시 붕괴한다 --- top-10 overlap $0.799$와 분산 비율 $0.043$은
  attribution 헤드가 표본별 편차가 아니라 전역 중요도 패턴을
  재학습했음을 가리킨다.],
)<tbl:ceh-v3-comparison>

헤드 파라미터 노름은 v2와 v3 사이에서 실제로 성장하므로
($||"head".0||_F = 3.42 arrow.r 4.54$; $||"head".3||_F =
4.15 arrow.r 5.78$) 학습 신호는 흐르고 있었다. 헤드는 미학습
상태가 아니다. 주 태스크 성능이 희생되지도 않았다. 실패는 인프라가
아니라 타깃 설계에 있다.

=== 전문가-출력 타깃이 성공한 곳에서 태스크-logit 타깃이 실패한 이유

그럴듯한 메커니즘: 태스크 logit은 최종 선형 계층
$bold(W)_"task" dot bold("repr") + bold(b)_"task"$가 산출한다.
입력에 대한 그 그래디언트는 $bold(W)_"task" dot
(partial bold("repr") / partial bold(x))$이다. 선형 계층의 가중치
$bold(W)_"task"$는 상수 방향이며, 표본별 변동은 오직 입력에 대한
표현의 Jacobian에서만 온다. 이 그래디언트를 $bold(x)$와 곱하고
배치 전반에 걸쳐 demean하면, 살아남은 표본 단위 신호가 의미 있는
표본 단위 출력을 산출하기에는 헤드의 용량 대비 너무 작아 보인다.
헤드는 demean된 타깃의 전역 구조를 가장 잘 설명하는 평평한 패턴으로
수렴한다.

반면 v2 타깃은 *인과 인코더*의 집계 출력의 그래디언트로 --- 훨씬
넓은 표현적 병목과 더 풍부한 표본별 상호작용 항을 갖는다. 그
demean된 버전은 헤드가 학습할 만큼 충분한 표본별 구조를 유지한다.

=== 이것이 배제하는 것 / 인정하는 것

인정: CEH 인프라(attribution 헤드 + 타깃 주입 + demean)는 타깃
출처에 무관하다. v3 인프라는 기계적으로 작동한다.

배제: *태스크-logit 그래디언트가 전문가-출력 그래디언트보다
자동으로 더 나은 CEH 타깃인 것은 아니다.* v2를 구한 demean 단계는
이 타깃으로 일반화되지 않는다. 타깃 설계는 v1$arrow.r$v2 전환이
시사한 것보다 더 민감하다. v2의 성공은 일반적인 레시피가 아니라
특정한 sweet spot이다.

=== 후보 다음 단계 (미실행)

+ *Demean 없는 태스크 logit.* demean 단계가 태스크 그래디언트에
  특유한 구조를 과도하게 빼고 있을 수 있다. demean하지 않은 태스크
  타깃(전역 붕괴에 대한 L2 정규화 포함)이 명백한 다음 반복이다.
+ *다중 태스크 그래디언트 집계.* $sum_t w_t nabla_bold(x) text("logit")_t$
  로, 여기서 $w_t$는 태스크 가중 벡터다. 단일 logit의 전역 구조가
  지배하는 것을 피하기 위해 여러 태스크 logit에 걸쳐 감독을
  분산한다.
+ *Integrated Gradients를 타깃으로.* 경로-적분 attribution은
  단일 점 grad $times$ input보다 더 판별적인 것으로 알려져 있다.

비용: SageMaker 작업 1회(\$0.13). 부정 결과는 타깃 설계 탐색
공간의 정직하게 닫힌 분기로 취급한다.

== Finding 14: 교차 아키텍처 증류에서의 행동 유사도 바닥, 그리고 2계층 fidelity 게이트 <find14>

서빙 스택은 12-task PLE 교사를 태스크별 LightGBM 학생으로
증류한다 --- 이는 *교차 아키텍처(cross-architecture)* 쌍이다:
부드러운 mixture-of-experts 교사 대 결정 경계가 축에 정렬된
계단 함수인 트리 앙상블 --- 그리고 배포 전에 각 학생을
fidelity 게이트로 검증한다. v14 증류 라운드(교사 = Finding 4의
정규화된 PLE-softmax 변형)는 세 세대의 fidelity 보고서를 연달아
산출했고(run artefact에서 v3 $arrow.r$ v4 $arrow.r$ v6),
모두 *동일한* 교사 및 학생 예측 파일에 대해 채점되었다: 학생은
세대 간에 재학습되지 않았으며, 게이트의 측정 의미론만 바뀌었다.
따라서 이 보고서 3종은 게이트 그 자체에 관한 작은 통제된
데이터셋이며, 본 절은 추가 학습 연산 없이 측정 측면에서 이를
재분석한다. 동반 논문은 동일 에피소드의 거버넌스 측면 ---
감사 추적 가독성과 운영자 override 위험 --- 을 fidelity 게이트
의미론 논의에서 다룬다; 두 서술은 중복이 아니라 상호 보완적이다.

=== 동일 예측에 대한 세 게이트 세대

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    stroke: 0.5pt,
    inset: 5pt,
    table.header([*게이트*], [*범위*], [*cal\_gap 의미론*], [*판정*]),
    [v3], [10 tasks (incl.\ 3 fallback-routed)], [$|"ECE"_"teacher" - "ECE"_"student"|$], [0 / 10 pass],
    [v4], [7 distilled tasks], [student ECE], [0 / 7 pass],
    [v6], [7 distilled tasks], [student ECE], [Tier 1: 6 / 7 pass; Tier 2: 7 / 7 flagged],
  ),
  caption: [동일한 v14 교사--학생 예측에 대해 채점한 세 세대의
  fidelity 게이트. 태스크별 metric 값은 세 보고서에서 부동소수점
  재평가 잡음 수준까지 일치한다; 범위, metric 정의, tier 의미론만
  다르다. 학생 집단은 결코 바뀌지 않았다 --- 바뀐 것은 측정이다.],
) <tab:fidelity-generations>

세 가지 결함이 순차적으로 발견되어 수정되었다
(`core/training/distillation_validator.py`; 공개 저장소의 커밋
`0d6dc34` 와 `d7cb38b`):

+ *범위.* v3 게이트는 학습된 학생이 있는 모든 태스크를 채점했는데,
  여기에는 교사가 신뢰할 수 없었기 때문에 서빙 폴백이 소프트 레이블
  증류로부터 명시적으로 *벗어나게* 라우팅한 세 태스크가 포함되어
  있었다. 그 결과 게이트는 그 학생들이 교사와 일치하지 않는다는
  이유로 페널티를 부과했다 --- 바로 라우팅이 도입하도록 설계된
  바로 그 divergence를 두고서. 가장 극명한 사례는 nba\_primary
  이다: 폴백 학생은 ground truth에 대해 F1-macro $0.983$ 을
  기록하는 반면 교사는 $0.278$ 이며, v3 게이트는
  "f1\_macro\_gap" $0.705$ 를 이유로 이를 실패 처리한다. 수정은
  게이트의 범위를 `distill_tasks` 집합으로 한정한다(v4/v6
  보고서는 명시적인 `gate_scope: "distilled_tasks_only"` 필드를
  담고 있다).

+ *Calibration 의미론.* v3는 `calibration_gap` 을
  $|"ECE"("teacher") - "ECE"("student")|$ 로 정의했는데, 이는
  교사 자체가 잘 보정되지 않은 경우(불균형 이진 태스크에 대한
  focal-loss 학습에서 전형적이다) 잘 보정된 학생을 실패시킨다.
  이를 ground-truth 레이블에 대한 학생 *자신의* ECE로
  재정의하면 churn\_signal의 값이 $0.2452$ 에서 $0.0045$ 로
  무너지는데 --- 모델 변경 없이 정의 수정만으로 일어난
  $54 times$ 의 변화이다.

+ *Tier 혼동.* 두 수정 이후에도 v4 게이트는 여전히 $0 / 7$ 을
  통과시켰다: 모든 이진 태스크에서 모든 운영 metric이 임계값을
  넘겼으나, 게이트는 행동 유사도 metric(agreement, ranking
  correlation)이 동시에 넘기기를 요구했다. v6 게이트는 두 계열을
  blocking 운영 tier와 informational 진단 tier로 분리하며
  (`informational_failures`, `tier2_violation_count` 로 집계),
  판정은 6 / 7 이 된다.

=== 행동 유사도 바닥

@tab:fidelity-floor 은 증류된 여섯 이진 태스크에 대한 v6
태스크별 값을 보고한다. 운영 metric은 임계값을 큰 폭으로
넘긴다 --- 최악의 경우 AUC 격차 $0.0125$, 최악의 경우 학생 ECE
$0.0114$ 로 $0.05$ 임계값 대비 --- 반면 행동 metric은 사실상
모든 태스크에서 임계값 아래의 좁은 대역에 머무른다: agreement는
$0.85$ 대비 $0.754$--$0.820$, ranking correlation은 $0.90$
대비 $0.870$--$0.913$(여섯 중 둘만 이를 넘긴다).

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: 0.5pt,
    inset: 5pt,
    table.header([*Task*], [*AUC gap*], [*Student ECE*], [*Agreement*], [*Rank corr.*]),
    [churn\_signal], [0.0033], [0.0045], [0.8134], [0.8949],
    [will\_acquire\_deposits], [0.0068], [0.0047], [0.7545], [0.8703],
    [will\_acquire\_investments], [0.0016], [0.0038], [0.8186], [0.8895],
    [will\_acquire\_accounts], [0.0125], [0.0114], [0.7999], [0.9014],
    [will\_acquire\_lending], [0.0001], [0.0038], [0.8204], [0.8847],
    [will\_acquire\_payments], [0.0116], [0.0076], [0.7973], [0.9135],
    [(threshold)], [$<= 0.05$], [$<= 0.05$], [$>= 0.85$], [$>= 0.90$],
  ),
  caption: [증류된 여섯 이진 태스크에 대한 태스크별 fidelity
  metric (v6 보고서; v4는 동일, v3는 정의상 calibration
  열에서 차이가 난다 --- 본문 참조). 운영 metric(왼쪽 쌍)은
  $4 times$--$10 times$ 의 여유를 두고 통과한다; 행동
  metric(오른쪽 쌍)은 태스크 전반에 걸쳐 균일하게 임계값 바로
  아래의 바닥에 머무른다. 태스크 전반의 균일성과 세 측정 세대에
  걸친 불변성은 이 바닥을 임의의 개별 학생의 속성이 아니라
  PLE $arrow.r$ LightGBM 아키텍처 쌍의 속성으로 특징짓는다.],
) <tab:fidelity-floor>

그 메커니즘은 아키텍처 쌍 그 자체이다: 학생은 교사의 전역
랭킹을 일치시킬 수 있으나(거의 0에 가까운 AUC 격차가 그 증거다)
개별 점 추정치에서는 불일치하는데, 트리 앙상블이 교사의 부드러운
결정 표면을 보간(interpolate)할 수 없기 때문이다. 이 바닥은
재학습으로 고칠 결함이 아니다 --- 동일 예측에 대한 세 세대의
재측정이 이를 움직이지 못했고, 학생 계열을 바꾸지 않는 한 어떤
학생 측 변경으로도 움직이지 못할 것이다.

일곱 번째 증류 태스크인 cross\_sell\_count(count regression)는
남아 있는 유일한 Tier-1 실패이며, 회귀 분기에 잔존하는 의미론
문제를 드러낸다. 이것의 blocking metric은
`rmse_gap` $= 0.3177 > 0.10$ 이다 --- 그러나 이 격차는 교사
RMSE($1.5245$)와 학생 RMSE($1.2068$) 사이의 *절대 차이*이다:
학생은 자신의 교사보다 RMSE가 $0.32$ 만큼 *더 나은데도*, 게이트는
벗어났다는 이유로 이를 차단한다. 회귀 분기는 여전히 학생 자신의
품질이 아니라 교사와의 유사도를 채점하는데 --- 이는 calibration
수정이 이진 분기에서 제거한 바로 그 혼동이다. 우리는 이를 판정이
아니라 미해결 항목으로 표시한다; deviation bound에는 정당한
용도가 있으나(집계 RMSE에서 교사를 이긴 학생이라도 교사의 검증이
한 번도 시험하지 않은 방식으로 오류를 재분배했을 수 있다), 이
비대칭성은 상속된 기본값이 아니라 명시적 결정을 받을 자격이 있다.

=== 항상 실패하는 게이트에서 2계층으로

이 finding의 요점은 통제 설계의 귀결이다. 운영 metric과 행동
metric을 함께 통과하도록 요구하는 단일 계층 게이트는, 이
아키텍처 쌍에 대해서는 *항상 실패하는* 게이트이다 --- 그리고
항상 실패하는 게이트는 효력을 유지하지 못한다. 이를 우회하는
운영 경로(`skip_fidelity_gate=true`)는 통제 전체를 비활성화하며,
여기에는 통과하고 있었고 진정으로 load-bearing이었던 운영 검사도
포함된다. 행동 임계값을 관측된 바닥까지 완화하는 것은 상호
보완적인 이유로 거부되었다: 그것은 metric이 drift 신호로서
가지는 잔여 가치를 지워버린다. 채택된 설계는 각 metric이 무엇을
측정하는지에 따라 판정을 분리한다:

- *Tier 1 (blocking, 운영)*: 프로덕션 scorer가 망가졌는가?
  AUC 격차와 학생 ECE(이진), F1-macro 격차(다중클래스),
  MAE/RMSE 격차(회귀). 실패는 배포를 차단한다.
- *Tier 2 (informational, 진단)*: 학생이 교사와 얼마나
  유사한가? Agreement, ranking correlation, Jensen-Shannon
  divergence, quartile agreement. 위반은 기록되고 추세가
  추적되며, 결코 차단하지 않는다 --- 그러나 급격한 붕괴(예:\
  agreement가 $0.5$ 아래로 떨어지는 경우)는 여전히
  `tier2_violation_count` 추세에서 이상으로 표면화된다.

=== Finding 계보에서의 위치

Finding 14는 이 논문을 연 silent-failure 계보의 서빙 측
구성원이다. Finding 1에서는 학습이 건강해 보이는 동안 통제가
조용히 *불활성(inert)*이었다 --- 손실 가중치가 무시되었고 모든
run이 그럴듯하게 수렴했다. 여기서는 프로덕션 품질이 건강한
동안 통제가 요란하게 *실패(failing)*하고 있었다. 둘 다 모델이
아니라 측정 의미론의 실패이며, 둘 다 집계 대시보드에는 보이지
않았다: Finding 1은 아무것도 경보하지 않았기 때문에, Finding
14는 모든 것이 경보했기 때문이다. Finding 6의 교훈이 스택의 한
단계 위에서 재현된다: 태스크 유형이 metric 의미론에서 다를 때
합성 val-loss가 무효한 checkpoint 기준인 것과 똑같이, metric
계열이 *통제* 의미론에서 다를 때 합성 단일 계층 게이트는 무효한
배포 기준이다. 그리고 집계 AUC가 태스크별 구조를 가릴 수 있다는
Finding 7의 경계가, 운영 집계가 표본별 행동 divergence를 가리는
형태로 서빙 시점에 다시 나타난다. 동반 논문과의 분업은
의도적이다: Paper 2는 게이트 에피소드가 거버넌스에 대해 무엇을
의미하는지 --- 감사 추적 의미론, 규제기관 가독성, override
위험 --- 를 문서화하는 반면, 본 절은 그것이 측정에 대해 무엇을
의미하는지를 문서화한다: 바닥이 어디서 오는지, 어떤 metric
정의가 잘못되었는지, 그리고 게이트가 효력을 유지하려면 무엇을
분리해야 하는지.

= 논의

== 실무 가이드라인 요약

본 절에서는 열네 가지 발견을 네 주제로 묶어, 실무자를 위한 열두 가지
가이드라인으로 정리한다.

=== 손실 동역학과 게이팅 (발견 1--6)

+ *게이트 선택은 아키텍처가 아니라 태스크 유형 구성에 달려 있다.*
  서로 다른 손실 유형이 섞인 이질적 태스크 조합에는 softmax 를,
  동질적 태스크(전부 이진, 또는 전부 회귀)에는 sigmoid 를 사용한다.
  이것이 단일 항목으로는 가장 큰 영향을 미치는 설계 의사결정이다.

+ *uncertainty weighting 은 필요조건이지 충분조건이 아니다.*
  손실 스케일을 정규화한다 --- 이것이 없으면 멀티클래스 태스크가
  소리 없이 억제된다. 그러나 gradient 오염을 막지는 못한다.
  실제 격리는 게이트 구조가 제공한다. 구현을 @eq:uw-correct 와
  대조해 검증하되, 특히 태스크별 $w_t$ 와 clamping 을 확인한다.

+ *사전 게이팅 그룹에 태스크 유형을 섞지 말라.*
  GTE, task-group attention 및 유사 메커니즘은 그룹 내 동질성을
  전제한다. 이진 태스크와 회귀 태스크를 섞은 비즈니스 의미 단위
  그룹은 소수 유형을 열화시킨다.

+ *checkpoint 지표로 val loss 가 아닌 복합 지표를 사용하라.*
  회귀 태스크와 분류 태스크가 복합 손실을 공유할 때, 회귀 개선이
  val loss 를 계속 끌어내리는 동안 분류/랭킹 지표는 정체되거나
  퇴보한다. 학습 전에 유형 가중 복합 지표(Avg AUC + NDCG\@3 +
  정규화 MAE)를 정의하고 이를 기준으로 checkpoint 한다 (4.6절).

+ *게이트 엔트로피는 아키텍처적 낭비를 드러낸다.*
  모든 태스크가 균일한 attention 수준 엔트로피(비율 = 1.000)를
  보인다면, attention 집계는 라우팅을 수행하는 것이 아니라 평균을
  내고 있는 것이다. 게이팅 메커니즘에 성능 향상을 귀속시키기 전에
  extraction 수준과 attention 수준 양쪽의 엔트로피 비율을 감사하라.

=== Fusion 증강 (발견 7)

+ *fusion 레시피를 목적에 맞추되, 쌓지 말라.*
  태스크별 균일한 향상을 동반한 aggregate-AUC 이득에는 inverse-gate
  보조 supervision(NEAS)을 우선한다. aggregate AUC 동등성을 대가로
  한 어려운 태스크 구제에는 shared-expert gradient 격리를 동반한
  output-space boosting(BRP-detached)을 우선한다. 두 레시피는 서로
  분리된 축 --- gate 수준 부하 분산 대 output 수준 오차 보정 ---
  에서 작동하며, 둘을 쌓으면 aggregate-AUC 이득이 붕괴한다. shared
  expert 가 generalist(NEAS)인 동시에 primary 를 보조하는
  specialist(BRP-detached)일 수는 없기 때문이다.

=== Causal expert 재해석 (발견 8--13)

+ *causal DAG 를 소비하기 전에 실제로 학습되고 있는지 검증하라.*
  학습 중 0 으로 표류하는 $bold(W)$ 는 소리 없는 실패 모드이다.
  NOTEARS acyclicity 와 sparsity 페널티는 $bold(W) = 0$ 에서
  자명하게 충족되며, SCM 잔차 forward
  ($bold(z) + bold(z) bold(W)^2$)는 $bold(z)$ 를 통해 여전히
  primary 신호를 전파한다. 최종 태스크 지표뿐 아니라 학습 시점의
  Frobenius norm, active-edge 개수, acyclicity 값 $h(bold(W))$ 를
  모니터링하라.

+ *장식적 DAG 는 학습 선택의 문제이지 아키텍처적 제약이 아니다.*
  init scale $0.1$, reconstruction-loss 가중치 $0.5$ 에서 $bold(W)$
  는 $8 times 8$ 행렬에서 Frobenius ~$0.36$ 에 안착한다 --- "존재하나
  의미를 갖기엔 너무 약하다". init 을 $0.3$ 으로, $lambda_"recon"$ 을
  $2.0$ 으로 올리면 학습된 행렬이 primary-task 비용 없이 $14 times$
  로 증폭된다. $bold(W)$ 가 구조 정보를 담고 있음에 의존하는 하류
  용도는 default 가 아니라 증폭된 학습 실행에 대해 평가해야 한다.

+ *attribution-head 타깃 설계에서 sample-variance 붕괴를 점검하라.*
  sample-invariant 한 내용이 큰 타깃(예: 집계 출력의 raw gradient
  $times$ input)은 얇은 MLP 가 전역 패턴을 재학습하고 sample 단위
  잔차를 무시하게 만든다. per-prediction 설명 가능성을 주장하기 전에
  attribution 이 sample 마다 변하는지(between/within 분산 비율,
  sample 간 top-$K$ overlap)를 검증하라. 최소 수정 --- 타깃의 batch
  평균을 빼는 것 --- 은 sample 단위 편차 학습을 강제하며 (본 사례에서)
  분산 비율을 태스크 비용 없이 $0.055$ 에서 $0.719$ 로 끌어올렸다.

+ *guardrail 에는 weight-space 보다 latent-space 정식화를 우선하라.*
  per-sample W-reconstruction 잔차는 분포 인식 능력이 없다. 모든
  sample 이 동일한 latent 를 산출하면 잔차는 한 점이 되어 guardrail
  이 발동할 수 없다. 캐시된 in-distribution 참조 batch 로부터의
  Mahalanobis 스타일 거리는 구성상 분포를 인식하며, 본 설정에서 세
  probe 에 걸쳐 5% FPR 에서 100% OOD TPR 을 달성하였다. latent
  정식화를 default 로 하고, weight-space 버전은 보완으로만 사용하라.

+ *추론 시점의 causal mediation 에 의존하기 전에 $bold(W)$ 를
  증폭하라.* Pearl 의 Rung 3 counterfactual --- "피처 $j$ 가 달랐다면
  출력이 어떻게 되었을까?" --- 는 학습된 DAG 가 실제로 개입을
  $bold(W)^2$ 를 통해 전파할 것을 요구한다. default 학습 레시피에서
  DAG 는 counterfactual 효과의 $1%$ 미만을 운반하며(발견 12
  baseline), 발견 11 의 증폭 레시피에서는 mediation 비율이 중앙값
  $32%$, 95 백분위수 $61%$ 로 상승한다. counterfactual 추론을
  표방하는 모든 하류 기능(규제기관 대면 설명, treatment-effect 추정,
  what-if 시뮬레이션)은 증폭된 teacher 에서 평가해야 한다. default
  teacher 에서 평가하면 그 주장이 소리 없이 direct-feed 관찰 probe
  수준으로 격하된다.

=== 서빙 측 증류 제어 (발견 14)

+ *fidelity 게이트를 지표 의미별로 분할하고, 항상 실패하는 게이트를
  유효 상태로 남겨두지 말라.* cross-architecture 증류에서는
  behavioural-similarity floor(teacher--student 일치도, ranking
  correlation)가 깔끔하게 통과하는 운영 품질(AUC 격차, student
  calibration) 아래에 존재할 것으로 예상해야 한다. student 계열이
  teacher 의 결정 곡면을 보간할 수 없기 때문이다. similarity 계열에
  배포를 차단하면 영구적 실패가 보장되며, 운영 검사까지 무력화하는
  전면 override 를 부른다. ground truth 에 대한 student 고유 품질에만
  차단하고, similarity 는 추세 진단으로 기록하라. 그리고 각 게이트
  지표의 정의에서 similarity 와 quality 의 혼동을 감사하라. teacher 가
  품질 기준이 아닌데 $|"teacher" - "student"|$ 를 채점하는 게이트는,
  teacher 보다 나은 student 를 도리어 실패 처리한다.

== 한계

*합성 데이터*: 모든 실험은 통제된 노이즈 프로파일을 가진 합성
벤치마크를 사용한다. 실제 프로덕션 데이터는 라벨 희소성, 클래스
불균형, 비정상 분포로 인해 다른 gradient 동역학을 보일 수 있다.
프로덕션 결과가 확보되는 대로 보완할 계획이다.

*단일 expert basket*: 본 연구의 발견은 7개의 이질적 expert 를 가진
PLE 에 특정된다. 동질 expert PLE 는 다른 게이트 동역학을 보일 수 있다.

*epoch 예산과 태스크 유형의 상호작용*: 발견 4 는 10-epoch 비교가
시기상조일 수 있음을 인정한다. 발견 6 은 이를 30 epoch 로 확장하여,
추가 epoch 이 회귀에는 도움이 되지만 분류와 랭킹에는 해롭다는 것을
확인한다. 30 epoch 에서의 cross-architecture 비교(예: PLE 대\
shared-bottom)는 여전히 미완으로 남아 있다.

*단일 데이터셋 규모*: 1M 고객은 중견 금융기관을 대표하지만, 태스크
gradient 동역학이 다른 인터넷 규모 데이터셋(1억+ 사용자)으로는
발견이 일반화되지 않을 수 있다. 특히 발견 8--13 은 단일 seed 와 단일
데이터셋(Santander)에서 보고되며, cross-seed 안정성과 cross-dataset
재현은 보류되어 있다.

*CG 를 위한 합성 OOD probe (발견 10--11)*: Causal Guardrail 평가는
세 종의 합성 out-of-distribution probe(uniform random,
column-permuted, extreme-tail)를 사용한다. 실세계 분포 드리프트(시간적
변동, 하위그룹 불균형, 적대적 교란)는 구조와 난도 모두에서 다를
것으로 예상된다. 합성 probe 에서 CG v2 의 5% FPR 에서 100% TPR 은
sanity-check 상한이지, 프로덕션 준비된 수치가 아니다.

*attribution 의 의미성은 인간 평가되지 않았다*: CEH v2 (발견 9)는
sample 간 변별력을 가진 sample 단위 attribution 을 생산하지만(분산
비율 0.719, top-10 overlap 0.281), 그 결과로 도출된 top-$K$ 피처가
도메인 전문가의 기대 또는 대안적 attribution 방법(Integrated
Gradients, DAG-path traversal)과 정합하는지는 평가하지 않았다. 인간
평가 패스는 향후 과제이다.

*잔여 Axis-3 후보 (CTGR, CRCG)*: 발견 10--11 은 전제조건 ---
$bold(W)$ 가 학습 가능하다는 것 --- 을 확립하며, CCP 는 그 이후 증폭된
teacher 에서 평가되었다(발견 12). 남은 두 후보 CTGR 과 CRCG 는 아직
평가되지 않았다. CG 의 v1$arrow.r$v2 전환은 또한 구체적 예측(latent
기반 정식화는 weight 기반 정식화와 나란히 평가되어야 한다)을
도입하는데, 이는 이들에 대해 검증되지 않았다.

*fidelity floor 의 특수성 (발견 14)*: behavioural-similarity
floor(일치도 $0.75$--$0.82$, ranking correlation $0.78$--$0.91$,
증류된 7개 태스크 전반)는 한 teacher--student 쌍(이질 expert PLE
$arrow.r$ LightGBM)에 대해 하나의 벤치마크에서 측정되었다. 2계층 제어
원칙은 일반화되지만, 수치 floor 는 그렇지 않다.

*deterministic 태스크 재주입은 아직 측정되지 않았다*: 13-태스크
구성(이후 v13 개정에서 `segment_prediction` 제거로 12-태스크)은
deterministic 태스크를 라벨 누출 위험으로 제거한 18-태스크
초안에서 파생되었다. 이를 재주입하면 MTL 학습이 오염되는지(gradient
지배, gate 왜곡)는 새로운 학습 실행을 요하는 미해결 실험 문제이며,
여기서는 어떤 주장도 하지 않는다.

*범위 경계 --- evidential uncertainty head*: 위의 causal-expert
재해석 흐름과 별개로, 구현체는 입력 피처가 무효(NaN/Inf 탐지)이거나
호출자가 제공한 `valid_mask` 가 행을 신뢰할 수 없는 것으로 표시할 때
최대 불확실성을 갖는 sample 단위 중립 예측을 방출하는
`EvidentialLayer`(`core/model/layers/evidential.py`)를 탑재한다. 이
레이어는 `valid_mask` 를 forward 로 전파하여 하류 손실이 무효 행을
gradient 업데이트에서 제외할 수 있게 한다. 이 head 는 Causal
Guardrail(CG)에 대해 상보적 역할을 한다. CG 는 causal latent 공간에서
*out-of-distribution* 입력을 표시하는 반면, evidential head 는 피처
수준에서 *수치적으로 무효한* 입력을 표시한다. 이 evidential head 에
대한 경험적 발견을 여기서 의도적으로 보고하지 않는데, Axis-3 재해석
프로그램에서 비롯된 것이 아니기 때문이다. 독자가 그 역할을 CG 와
혼동하지 않도록 언급할 뿐이다.

== 동반 논문과의 관계

본 논문은 동일 프로젝트의 두 동반 논문을 보완하며, 본 논문의 기여는
나열된 카탈로그가 아니라 하나의 줄기로 읽는 것이 옳다. *논문 1*(아키텍처와
ablation)은 이질 expert PLE 설계를 확립하고 joint feature+expert
ablation 으로 expert 특화를 검증한다. 우리가 상세히 보고하는 손실 동역학과
게이팅 결과(발견 1--6)는 논문 1 이 요약한 ablation 의 완전한 실증 기록이며,
둘이 겹치는 부분은 다시 도출하지 않고 논문 1 의 수치를 인용한다. *논문
2*(서빙, 추천사유 생성, 규제 준수)는 자립적이다. 그 감사 추적은 증류된
학생의 SHAP attribution, 3-에이전트 추천사유 파이프라인, HMAC 서명 해시
체인으로 충족되며, GDPR 제22조 / EU AI Act 제13조 논증은 본 논문의 어떤
출력에도 의존하지 않는다.

그 배경 위에서 본 논문이 맡는 줄기는 *MTL을 동질 태스크 영역 너머로
밀어붙일 때 무엇이 조용히 실패하며, 그것을 잡아내는 측정 규율은
무엇인가*이다. 조용한 실패의 계보는 학습이 건강해 보이는 동안 비활성인
uncertainty-weighting 제어(발견 1)에서, 복합 검증 손실이 가리는 게이트
엔트로피와 checkpoint 병리(발견 6)를 지나, 인접행렬이 조용히 0 으로
최적화되는 인과 전문가(발견 8), 그리고 프로덕션 품질이 건강한데도 영구히
실패하는 증류 fidelity 게이트(발견 14)로 이어진다. 각각은 모델이 아니라
측정 의미론의 실패이며, 실무자가 자연히 지켜볼 지표에는 보이지 않는다.

기능적 DAG 위에 세운 두 Axis-3 출력 --- CEH 표본별 attribution
벡터(발견 9)와 CG z-space coherence score(발견 10--11) --- 는 논문 2 의
감사 인프라(`log_attribution` / `log_guardrail`)에 *선택적 포렌식 보강*으로
라우팅될 수 있다. 우리는 이 프레이밍에 신중하다. 이는 전제조건이 아니라
보강이다. 논문 2 의 규제 논증은 이것 없이도 성립하며, 예측별 인과 감사는
서빙되는 학생이 아니라 PLE 교사에서 돌기 때문에 이 레코드를 프로덕션
트래픽에 내보내는 것은 향후 과제다. counterfactual-probe 방향 --- 증폭된
교사 위에서 Pearl Rung 3 를 시험하는 것(발견 12) --- 역시 향후 과제로
남기며, 여기서는 검증된 결과가 아니라 실현가능성 신호로만 보고한다.

발견 14 는 성격이 다른 동반 논문 대응물을 갖는다. cross-architecture
증류에서의 fidelity-게이트 의미에 관한 논문 2 의 논의는, 측정 측면이
4.14절에서 분석되는 바로 그 결함-수정 시퀀스의 거버넌스 귀결 ---
감사 추적의 가독성, 규제기관 대면 tier 의미, operator-override 위험
--- 을 문서화한다. 두 서술은 기저 산출물(세 세대의 fidelity 리포트)을
공유하지만, 어느 쪽도 다른 쪽의 분석을 중복하지 않는다.

= 결론

멀티태스크 학습을 2--4개의 동질 태스크에서 13개의 이질 태스크로
확장하면, 주로 동질적 구성에서 평가되어 온 기존 문헌이 다루지 않는 네
가지 동역학 계열이 부상한다.

*손실 동역학과 게이팅*(발견 1--6): 게이트 유형 선택 --- softmax 대\
sigmoid --- 은 아키텍처적 선호가 아니라 태스크들이 손실 유형을
공유하는지에 달려 있다. uncertainty weighting 은 스케일을 정규화하지만
gradient 를 격리하지 못하며 아키텍처 전반에서 동일하게 수렴한다. GTE
같은 사전 게이팅 메커니즘은 유형 동질 그룹을 요구한다. 공유 표현
baseline 은 게이트 변형이 정규화 없이 흡수하는 확장된 epoch 예산에
과적합한다. 게이트 엔트로피 분석은 extraction 수준 게이팅이
특화(엔트로피 비율 $0.33$--$0.88$)되는 반면 attention 수준 집계는
균일 평균으로 붕괴(비율 $1.00$)됨을 보인다. 그리고 복합 val loss 는
회귀 태스크가 존재하는 순간 신뢰할 수 없는 checkpoint 신호가 된다.
그 지속적 개선이 분류와 랭킹 열화를 가리기 때문이다.

*fusion 증강 trade-off*(발견 7): CGC 위에 representation 수준과
output 수준 fusion 을 비교한 9-way 비교는 설계 공간을 세 영역으로
매핑한다. representation-additive fusion --- 손실 수준 adaTT,
AdaTT-sp, M1 complement, ECEB, MV BRP --- 은 잔차 오차 신호를
primary-representation 경로로 전파하며 aggregate AUC 를 일률적으로
열화시키되, 그 크기는 개입의 침습성에 비례해 커진다. shared-expert
gradient 격리를 동반한 output-space boosting(BRP-detached)은
aggregate AUC 에서 CGC 와 동률을 이루는 한편 F1 macro 와 NDCG\@3 을
끌어올리고 가장 어려운 멀티클래스 태스크에서 +256% 상대 구제를
유지한다. 학습 시점 부하 분산 정규화(NEAS)는 이 계열에서 실제로
aggregate AUC 를 끌어올린($Delta = +0.0011$) 거의 균일한 태스크별
향상을 동반한 첫 메커니즘이다. 두 긍정 레시피는 분리된 축에서
작동하며 가산적이지 않다 --- 둘을 쌓으면 NEAS 의 AUC 이득이 붕괴하는데,
shared expert 가 generalist(NEAS)인 동시에 primary 를 보조하는
specialist(BRP-detached)일 수 없기 때문이다. 실무 지침은 목적별로
달라진다.

*causal expert 재해석*(발견 8--13): causal expert 의 인접행렬
$bold(W)$ 는 우리가 살펴본 모든 학습된 checkpoint 에서 0 으로
붕괴하여, expert 를 평범한 MLP 와 동등하게 만들었다 --- 발견 1 과 같은
조용한 실패 패턴이, 아키텍처의 한 층 아래에서 일어난 것이다. 2부
패치(NOTEARS reconstruction loss + 초기화 재스케일)는 primary-task 비용
없이 DAG 학습을 복원하지만, DAG 는 처음에는 "장식적"이다 --- 구조적으로
유효하나 예측으로 라우팅되지 않는다. 기능적 DAG 위에 우리는 두 Axis-3
출력을 세운다. sample 단위 attribution 벡터를 산출하는 Causal
Explainability Head(CEH; 첫 정식화는 전역 중요도 패턴으로 붕괴했고, 최소
개입 "demeaned target" 변형이 sample 단위 변별력을 복원했다)와, 신뢰성
플래그를 산출하는 Causal Guardrail(CG; W-reconstruction 정식화는 우연
수준 변별에서 실패했고, z-space Mahalanobis 정식화가 세 합성 OOD probe
에서 $5%$ FPR 에서 $100%$ TPR 을 달성했다)이다. W-증폭 실험은 장식적
DAG 가 아키텍처적 제약이 아니라 학습 선택의 산물임을 확립한다 --- init
$0.1 arrow.r 0.3$ 과 $lambda_"recon" 0.5 arrow.r 2.0$ 이
$||bold(W)||_F$ 를 태스크 비용 없이 14 배로 키운다. 이 두 출력은 동반
서빙 논문의 감사 인프라에 부하를 지는 의존이 아니라 *선택적 포렌식
보강*으로 제공된다. 논문 2 의 규제 논증은 이것 없이도 완전하다. 증폭된
teacher 위의 counterfactual probe(발견 12)는 예비적인 Pearl Rung 3
실현가능성 신호 --- $"do"(z_j = v)$ 개입 하 매개된 counterfactual 효과의
중앙값 $32%$ 대 baseline 의 $0.16%$ --- 를 돌려주며, 이는 단일 사후
합성-개입 분석에 기반하므로 검증된 결과가 아니라 향후 과제의 동기로
보고한다.

*서빙 측 증류 제어*(발견 14): 12-태스크 teacher 를 태스크별
gradient-boosted student 로 cross-architecture 증류하면 모든 운영 품질
검사(AUC 격차 $<= 0.0125$, student calibration 오차 $<= 0.0114$)를
통과하는 한편, teacher--student behavioural similarity 는 구조적
floor(일치도 $0.75$--$0.82$, ranking correlation $0.78$--$0.91$, 증류된
7개 태스크 전반)에 머문다. 이 floor 는 동일한 예측에 대해 채점된 세
세대 연속의 게이트 의미로도 움직일 수 없었는데 --- 어떤 student 의
속성이 아니라 아키텍처 쌍의 속성이기 때문이다. 두 지표 계열에 걸친
단일 tier 게이트는 영구적으로 실패하며 operator bypass 를 부른다.
채택된 2계층 설계는 운영 결함에만 배포를 차단하고 similarity 지표는
기록된 진단으로 강등한다. 이 교훈은 발견 1 이 연 고리를 닫는다.
거기서는 학습이 건강해 보이는 동안 제어가 소리 없이 불활성이었고,
여기서는 프로덕션 품질이 건강한 동안 제어가 요란하게 실패하고 있었다.
둘 다 모델의 실패가 아니라 측정 의미의 실패이다.

이 발견들은 새로운 알고리즘이 아니라 실무적 진단이며, 그 위에 놓인
최소 실행 가능 후보들이다. 우리는 이것이 다른 실무자들이 MTL 을
실세계의 이질적 태스크 포트폴리오로 확장할 때 같은 함정을 다시
발견하는 것을 막아주기를, 그리고 긍정 레시피들(NEAS, BRP-detached,
CEH v2, CG v2, W-증폭 teacher)을 추가 seed 와 데이터셋에 걸쳐 검증하는
후속 연구를 촉발하기를 바란다.

// ============================================================
= 저자 기여

*정선규* (PM / Lead Architect / 데이터 사이언티스트):
연구를 착안하고, ablation 프레임워크를 설계하고, 열네 가지 발견을 모두
식별하고, 원고를 집필. AI 증강 개발 방법론을 주도.

*심은철*: 데이터 파이프라인, 피처 엔지니어링, ablation 실행.

*김영찬*: 모델 학습, 수학적 검증, 손실 가중치 구현.

모든 저자는 빠른 피드백 사이클의 Scrum 스프린트로 협업하였다.

== 연구비

본 연구는 외부 연구비를 일절 받지 않았다.
AI 개발 도구, AWS SageMaker 클라우드 학습, 운영 경비 등 모든 비용은
제1저자의 개인 자금으로 충당하였다.
개발은 데스크톱급 GPU 1대(NVIDIA RTX 4070, 12GB VRAM)에서 수행하였다.

// ============================================================

#bibliography("references.bib")
