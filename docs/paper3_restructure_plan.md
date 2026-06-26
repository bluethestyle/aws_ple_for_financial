# Paper 3 재구성 실행 계획 (silent-failure 재포지션)

> 작성 2026-06-16. 근거: 워크플로 wf_253aa6ab-523 (Paper2 의존감사 + 14-Finding 분류 + 교차참조 무결성 + 최소실험 + 적대검증).
> 사용자 확정: ① silent-failure 논지로 재포지션 ② F8 척추 유지·F9–11 Paper2-bridge·F12 future-work 컷.

## 0. 핵심 결론 (왜 이렇게)

- **Paper 2 자립 확정** (적대검증 refuted=false/high). 규제 논증(Art.13/GDPR Art.22)은 3-agent reason + LGBM SHAP 감사로그 + HMAC 해시체인으로 충족. 인과 아크 출력(CEH/CG)은 본문 스스로 "by-product / intentionally minimal", 서빙 경로(LGBM 학생)에서 발화하지도 않음(teacher-in-serving = future work). → Paper 2 의존은 *논증*이 아니라 *광고된 기여 목록*(Contribution #5/#7) 수사 의존.
- **F1–F6 = Paper 1 ablation 본체** (적대검증 refuted=true/high). 본문 line 121 "extends Paper 1's summary". F2 역전·메커니즘 서사·F1 버그 델타·F3 수렴이 Paper 1 초록/기여/§5.4/결론에 글자 그대로 중복. → **F2를 독립 *신규* 논문으로 분리 = 자기표절.** 따라서 in-place 재포지션 + 수치는 Paper 1 인용 귀속.
- **정정**: 이전 대화의 "softmax +0.0153 우위"는 softmax-vs-*shared-bottom*. 실제 softmax-vs-*sigmoid* NDCG@3 격차는 +0.0059(부호 혼재: sigmoid가 AUC +0.0008·val-loss 이김). 단일시드.

## 1. Finding 분류 (재포지션 기준)

| 분류 | Finding | 처리 |
|---|---|---|
| **독립 척추 (silent-failure 서사)** | F1(침묵 버그) → F6(체크포인트·attention 병리) → F8(decorative DAG) → F14(충실도 바닥) | 헤드라인. 단일 논지로 묶음. |
| 척추 보강 (Paper1 인용 귀속) | F2, F3, F4, F5 | 서사 내 증거로 유지, 중복 수치는 Paper1 인용. |
| 음성결과 (독립 유지) | F13 (attribution target 민감도) | 방법론 caution으로 유지. |
| 가이드라인 강등 | F7 (9-way fusion, 노이즈바닥·버전 verdict flip) | "발견"→실무 가이드 박스. **2026-06-19: v14 3-way(§7)가 M1 부호 취약성($+$0.0006 → $-$0.0010)을 독립 재현 → 강등 근거 보강(버그 아님).** |
| **Paper2-bridge enhancement** | F9(CEH), F10·F11(CG) | "optional forensic enhancement, not prerequisite"로 명시 재라벨. |
| **future-work 컷** | F12 (CCP / Pearl Rung 3) | 본문 결과에서 빼고 viability signal로 future-work 언급. (타 논문 미인용 → 컷 안전) |

## 2. 하드 제약 (반드시 준수)

1. **Finding 번호 재배열 절대 금지.** Paper1이 "Findings 8 and 11"·"Findings 1-7", Paper2가 "Finding 14"를 *번호로* 인용. 재포지션 = 강조/배치/서사만 변경, F1..F14 번호·라벨(`<find8>` 등) 불변.
2. **EN+KO 6파일 동기화.** paper{1,2,3}.typ + paper{1,2,3}_ko.typ. EN 수정마다 KO 동일 줄 미러.
3. **섹션 번호 cross-ref 주의.** Paper2가 "companion §3.4/§3.7" 하드코드 참조 → Paper3 섹션 밀림 금지(F12 컷 시 §4.13/§4.14 번호 유지되도록 라벨/번호 확인).
4. **지금 Zenodo 재발행 안 함.** Layer 2(운영데이터)와 합쳐 1회 재발행.

## 3. 두 층 시퀀싱 (운영데이터 반영)

- **Layer 1 — 구조/서사 (지금, 수치 무관)**: 아래 작업 목록. 로컬 .typ 작업본에만.
- **Layer 2 — 운영기준 수치 + 단일 재발행 (운영데이터 도착 시)**: 표에 운영 컬럼 추가, 한계 절 갱신, 그때 Paper1/2/3 동시 V2 재발행 + Paper3 첫 DOI 발급 후 Paper1/2에 역주입.
- E1(F2 cross-seed 3)·E2(all-binary 동질 셋)는 Layer 2에서 "핵심 시나리오 운영 검증"과 함께 재판단(중복 가능).

## 4. 작업 목록 (Layer 1, 순서)

**EN 레이어 완료 (2026-06-16) — 3편 내부 정합.**

- [x] (P3 EN) Relationship to Companion Papers 절 재작성 — silent-failure 논지 + Paper2 자립 명시 + F9-11 enhancement + F12 future-work
- [x] (P3 EN) Conclusion 인과 문단 재작성 — F9-11 enhancement·F12 viability-signal
- [x] (P3 EN) Abstract 오프닝 → silent-failure 논지 1문장 + CCP 단정 톤다운
- [x] (P3 EN) Intro "Position relative to companion papers" 재작성 ("extends"→"detailed empirical companion", F2-F5 Paper1 인용 귀속, Paper2 자립+F9-11 optional enhancement, F11 W-amp future-work)
- [x] (P3 EN) Contribution 목록 CCP 단정 톤다운 (preliminary viability signal)
- [x] (P3 EN) §4.12(F12 CCP) — 헤딩 "(Preliminary)" + 상단 caveat 문단 + caption 톤다운 + "completes"→"sketches" (표·수치 보존)
- [x] (P3 EN) F9 인라인 "Paper2 optional enhancement, not dependency" 1구절
- [x] (Paper2 EN) Abstract (5)·Contribution "Causal Audit Pair"·Conclusion → optional enhancement + teacher-path future-work caveat (3곳)
- [x] (Paper1 EN) audit surface "pairs CEH/CG"→optional, "feed Paper2"→"can optionally enrich" (2곳). 645-651 silent-W→0 서술은 정확+논지 일치라 유지.

**KO 레이어 완료 (2026-06-16) — EN/KO 재동기화됨.**

- [x] (P3 KO) paper3_ko.typ 미러 13곳 (abstract 논지+CCP, intro 포지셔닝+Paper2 자립, contribution CCP, F9 inline, §4.12 (예비)+caveat+caption, Relationship 절, Conclusion 인과 문단)
- [x] (Paper2 KO) paper2_ko.typ 4곳 (abstract (5), contribution, Discussion 발견5 "갭을 닫는다"→"포렌식 깊이 더한다", conclusion bullet)
- [x] (Paper1 KO) paper1_ko.typ 2곳
- [x] (검증) **typst EN/KO 6편 전부 OK 컴파일** (라벨/figure/footnote 무결성 확인)

> 참고: EN Paper2도 Discussion 'Finding 5: ...closes the gap' 헤딩을 'adds per-decision forensic depth'로 톤다운(원래 3곳 계획 → EN/KO 각 4곳으로 확장).

## 5. 키스톤 voice 샘플

Layer 1 첫 편집은 "Relationship to Companion Papers" 절 + Conclusion 인과 문단(재구성이 가장 집약되는 두 선언 지점). 사용자 voice 승인 후 나머지 전파.

## 6. 남은 것 = Layer 2 (운영데이터 도착 시)

- 표에 운영기준 수치 컬럼 추가, 한계 절 갱신
- Paper 1/2/3 동시 V2 재발행 (Zenodo 새 버전) + Paper 3 첫 DOI 발급 후 Paper 1/2에 역주입
- E1/E2(F2 cross-seed)는 "DOI 기록" 목적상 불필요 판정 — 단일시드·합성 한계는 한계 절에 명시 유지

## 7. OCP (직교여공간보정) + CCA 측정 — 2026-06-19 추가, 2026-06-26 커밋(50b02b6) (Layer 2 후보, paper3 표 미반영)

> OCP = orthogonal-complement residual recovery. `residual_orthogonal` fusion: gate primary 방향을 각 expert 출력에서 사영 제거 후 (1−gate) residual 합산. M1(complement)이 전체 complement를 더하는 것과 대비 — primary와 직교하는 새 신호만. 구현·테스트·런칭·커밋 완료(commit 50b02b6, [[project-ocp-experiment]] 메모리).

**구현/인프라 (커밋됨 50b02b6, 수치 무관 — Layer 1 성격):**
- OCP `residual_orthogonal` 커밋됨 (이전 docstring "future extension" → 실구현, `core/model/ple/experts.py`). 활성화: `residual_recovery.enabled=true`(기본 false) + `residual_method='orthogonal'` 둘 다 필요.
- CCA `ExpertRedundancyAnalyzer`가 이전엔 호출부 0건 dead code였으나 `train.py` eval에 config-gated(flag default-on) 배선·커밋됨 → 모든 학습 job이 실제로 `expert_redundancy.json` 산출. dead→live.

**3-way 결과 (provenance: phase0_v14, 15ep, batch2048, seed 42 단일, N=40960, 7 shared experts):**

| variant | mean ρ (CCA) | aggregate | avgAUC(7) | avgF1mac(2) | avgMAE(3) |
|---|---|---|---|---|---|
| baseline cgc | 0.1982 | 0.8228 | 0.8228 | 0.1280 | 0.3202 |
| M1 complement | 0.1973 | 0.8219 | 0.8219 | 0.1306 | 0.3178 |
| OCP orthogonal | **0.1927** | 0.8219 | 0.8219 | **0.1317** | **0.3121** |

- expert가 baseline에서 이미 직교(ρ 0.198, 대부분 LOW, perslay·OT≈0). OCP가 직교화 최대(−0.0055, M1의 6×)하나 헤드룸 작아 task 이득 미미. AUC −0.001(paper3 residual-recovery null 일관), minority task(F1/MAE)만 OCP 소폭 최고.

**⚠ paper3.typ 표 미반영 사유 (provenance/무결성):**
- 기존 §multi-mechanism 표의 M1 = avgAUC **0.8240**, Δ**+0.0006**(상승)인데, 본 run의 M1 = **0.8219**, baseline 대비 **−0.0010**(하락). **부호까지 반대** — struct_13 별도 run vs phase0_v14 run, 직접 병합 시 내부 모순.
- F7(9-way fusion)은 §1 분류상 "실무 가이드 박스" 강등 중이고, 본 run의 M1 부호 뒤집힘은 그 강등 근거인 "버전별 verdict flip"을 **독립 재현·corroborate**한다(버그 아님, §1 표 참조). 단일 seed.
- → OCP 수치는 **multi-seed + matched-provenance run 전까지 paper3 표에 넣지 않음.**

**Layer 2 통합 후보 (운영데이터/multi-seed 후):**
- (P1) §CGC Gate Design "non-redundant" 전제를 CCA 측정(baseline mean ρ 0.198)으로 백업 — 각주. *EN+KO 동기화.* (단일 seed caveat 명시) — **본 작업에서 paper1 로컬 작업본에 footnote 선반영, Zenodo 미발행.**
- (P3) F7 가이드 박스에 OCP 1줄: "직교화 최대지만 task 이득 없음 — expert 이미 직교라 헤드룸 부재" (multi-seed 확정 후).
