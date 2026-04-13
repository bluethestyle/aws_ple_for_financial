// =============================================================================
//  Feature Engineering Technical Reference — AWS PLE for Financial
//  v1.0 · 2026-04-01
// =============================================================================

// ─────────────────────────── Color Palette ───────────────────────────
#let anthropic-bg = rgb("#F0EFEA")
#let anthropic-text = rgb("#141413")
#let anthropic-accent = rgb("#CC785C")
#let anthropic-muted = rgb("#6B7280")
#let anthropic-rule = rgb("#D1D5DB")

// Legacy aliases for component compatibility
#let navy   = anthropic-text
#let teal   = anthropic-accent
#let amber  = anthropic-accent
#let indigo = anthropic-accent
#let rose   = anthropic-accent
#let slate  = anthropic-muted

// ─────────────────────────── Page Setup ───────────────────────────
#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
  fill: anthropic-bg,
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: "New Computer Modern", fill: anthropic-muted, tracking: 0.12em)
      #smallcaps[Feature Engineering Technical Reference]
      #h(1fr)
      #smallcaps[AWS PLE for Financial · v1.0]
      #v(4pt)
      #line(length: 100%, stroke: 0.4pt + anthropic-rule)
    ]
  },
  footer: context {
    let pg = counter(page).get().first()
    if pg > 1 [
      #line(length: 100%, stroke: 0.3pt + anthropic-rule)
      #v(4pt)
      #set text(size: 8pt, font: "New Computer Modern", fill: anthropic-muted)
      #h(1fr)
      — #pg —
      #h(1fr)
    ]
  },
)

// ─────────────────────────── Base Text ──────────────────────────
#set text(font: "New Computer Modern", size: 10pt, fill: anthropic-text, lang: "en")
#set par(justify: true, leading: 0.8em, spacing: 1.5em)
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

// ─────────────────────────── Code Blocks ───────────────────────────
#show raw.where(block: true): it => {
  block(
    width: 100%,
    inset: 12pt,
    radius: 5pt,
    fill: rgb("#f8fafc"),
    stroke: 0.5pt + anthropic-rule,
    text(size: 8.5pt, it),
  )
}
#show raw.where(block: false): box.with(
  fill: rgb("#f1f5f9"),
  inset: (x: 3pt, y: 1pt),
  outset: (y: 2pt),
  radius: 2pt,
)

// ─────────────────────── Equation Block Spacing ──────────────────────────
#show math.equation.where(block: true): it => {
  v(4pt)
  it
  v(4pt)
}

// ─────────────────────────── Heading Styles ──────────────────────────
#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  v(0.6cm)
  set par(first-line-indent: 0pt)
  block(width: 100%)[
    #text(size: 20pt, fill: anthropic-text, weight: "bold")[#it.body]
    #v(4pt)
    #line(length: 100%, stroke: 1pt + anthropic-accent)
  ]
  v(0.4cm)
}
#show heading.where(level: 2): it => {
  v(0.4cm)
  set par(first-line-indent: 0pt)
  block[
    #text(size: 14pt, fill: anthropic-text, weight: "bold")[#it.body]
  ]
  v(0.15cm)
}
#show heading.where(level: 3): it => {
  v(0.2cm)
  set par(first-line-indent: 0pt)
  block[
    #text(size: 10pt, fill: anthropic-text, weight: "bold")[#it.body]
  ]
  v(0.1cm)
}

// ───────────────────── Custom Components ────────────────────────────
#let note(title, body, accent: anthropic-accent) = {
  block(
    width: 100%,
    inset: (left: 14pt, right: 14pt, top: 10pt, bottom: 10pt),
    stroke: (left: 2pt + accent),
    [
      #text(weight: "bold", size: 9.5pt, fill: accent)[#title]
      #v(2pt)
      #text(size: 9.5pt)[#body]
    ],
  )
}
#let warn(title, body) = note(title, body, accent: anthropic-accent)
#let eq-highlight(body) = {
  block(
    width: 100%,
    inset: 14pt,
    stroke: (left: 2pt + anthropic-accent),
    body,
  )
}
#let chip(label, color: anthropic-accent) = {
  box(
    inset: (x: 6pt, y: 2pt),
    radius: 3pt,
    fill: color.lighten(88%),
    text(8pt, weight: "bold", fill: color.darken(10%))[#label],
  )
}
#let dim-label(body) = text(size: 8.5pt, fill: anthropic-muted)[#body]

// ───────────────── Table Base Style Function ────────────────────────
#let styled-table(cols, ..args) = {
  set text(size: 9pt)
  table(
    columns: cols,
    inset: 8pt,
    stroke: 0.4pt + anthropic-rule,
    fill: (_, y) => if y == 0 { anthropic-accent.lighten(88%) } else if calc.odd(y) { luma(252) },
    ..args,
  )
}


// =====================================================================
//  Cover Page
// =====================================================================

#set page(header: none, footer: none)

#v(3cm)
#align(center)[
  #text(
    size: 10pt,
    fill: anthropic-muted,
    tracking: 0.5em,
    weight: "regular",
  )[#upper[Tech Reference]]
  #v(0.5cm)

  #text(size: 26pt, fill: anthropic-text, weight: "bold")[
    Feature Engineering\
    Technical Reference
  ]

  #v(0.3cm)
  #line(length: 30%, stroke: 0.5pt + anthropic-rule)
  #v(0.3cm)

  #text(size: 11pt, fill: anthropic-muted)[
    11 Disciplines · Economics · Chemistry · Epidemiology · Criminology · Wave · TDA · HMM · GMM · Mamba · Graph · 3-Stage Normalization
  ]

  #v(2cm)

  #grid(
    columns: (1fr, 1fr),
    align(left)[
      #text(10pt, fill: anthropic-muted)[Project] \
      #text(12pt, fill: anthropic-text, weight: "bold")[AWS PLE for Financial] \
      #text(9pt, fill: anthropic-muted)[PLE-adaTT Multi-Task Recommendation]
    ],
    align(right)[
      #text(10pt, fill: anthropic-muted)[Version] \
      #text(12pt, fill: anthropic-text, weight: "bold")[v1.0] \
      #text(9pt, fill: anthropic-muted)[2026-04-01]
    ],
  )

  #v(1cm)
  #line(length: 100%, stroke: 0.5pt + anthropic-rule)
  #v(0.5cm)

  #block(width: 85%, stroke: (left: 2pt + anthropic-accent), inset: (left: 14pt, right: 14pt, top: 8pt, bottom: 8pt))[
    #text(size: 9.5pt, fill: anthropic-muted)[
      This document describes the complete feature engineering pipeline of the AWS PLE for Financial project.
      It covers the theoretical foundations, mathematical definitions, output feature specifications of
      11 cross-disciplinary feature generators (Economics, Chemical Kinetics, SIR, Crime Pattern, Wave Interference,
      TDA, HMM, GMM, Mamba, Graph, Base), and the 3-Stage Normalization Pipeline.
    ]
  ]
]

#v(1fr)
#pagebreak()

#set page(
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: "New Computer Modern", fill: anthropic-muted, tracking: 0.12em)
      #smallcaps[Feature Engineering Technical Reference]
      #h(1fr)
      #smallcaps[AWS PLE for Financial · v1.0]
      #v(4pt)
      #line(length: 100%, stroke: 0.4pt + anthropic-rule)
    ]
  },
  footer: context {
    let pg = counter(page).get().first()
    if pg > 1 [
      #line(length: 100%, stroke: 0.3pt + anthropic-rule)
      #v(4pt)
      #set text(size: 8pt, font: "New Computer Modern", fill: anthropic-muted)
      #h(1fr)
      — #pg —
      #h(1fr)
    ]
  },
)

// =====================================================================
//  Table of Contents
// =====================================================================
#outline(title: [Table of Contents], indent: 1.5em, depth: 3)

#v(12pt)
#warn[Design vs. Implementation Dimensions][
  This document is written based on the *full-bank design (734D)*. The current Santander benchmark implementation uses *350D (13 feature groups)*. For the actual implementation dimension specifications, refer to `outputs/phase0/feature_schema.json`. The Appendix "Design vs. Implementation Dimension Mapping" details the per-group differences.
]

// =====================================================================
//  1. Feature Design Philosophy
// =====================================================================
= Feature Design Philosophy

== Rationale for a Multidisciplinary Approach

Traditional feature engineering views data through a single lens: statistics. Means, variances, frequencies, and correlations are powerful, yet they reveal only a fraction of the structure inherent in the data. This system simultaneously applies the mathematical frameworks of 11 academic disciplines to extract different facets of the data.

#styled-table(
  (1fr, 1.2fr, 1.5fr),
  table.header([*Discipline*], [*Patterns Captured*], [*What Statistics Alone Cannot See*]),
  [Economics], [Income structure, elasticity, time preference], [Permanent vs. transitory income decomposition, causal consumption direction],
  [Chemical Kinetics], [Speed, barriers, and catalysts of transformation], [Energy barriers of state transitions, second-order derivatives (acceleration)],
  [Epidemiology (SIR)], [Adoption diffusion, immunity/churn], [Population-level dynamics, transmission threshold $R_0$],
  [Criminology], [Routines, deviations, anomalous patterns], [Circular nature of time, Burstiness],
  [Wave Physics], [Periodic decomposition, phase synchronization], [FFT frequency domain, PLV phase coupling],
  [TDA], [Topological shape, holes, connectivity], [Coordinate-invariant structure, multi-resolution observation],
  [HMM], [Hidden state transitions], [Probabilistic inference of unobservable latent stages],
  [GMM], [Soft clustering], [Probabilistic type assignment, uncertainty quantification],
  [Mamba SSM], [Long-range temporal dependencies in time series], [Nonlinear temporal representation via selective state spaces],
  [Graph (LightGCN)], [Collaborative filtering signals], [Multi-hop indirect preferences, hierarchical structure],
  [Base Statistics], [Demographics, holdings, transactions], [RFM, category distribution, channel diversity],
)

== Structural Isomorphism

In multidisciplinary feature engineering, "analogy" is not mere metaphor — it is grounded in *structural isomorphism*. Two systems may differ in their surface objects (molecules vs. consumers), yet when the _relational structure_ among those objects is mathematically identical, a structural isomorphism holds.

#note[Core Principle][
  The "time for reactant concentration to halve" in chemistry and the "time for transaction frequency to halve" in finance both share the same mathematical structure: exponential decay. When the equations are identical, the patterns they capture are identical. What the objects represent does not affect the validity of the equations.
]

== The Dual Role of Features

Features in this system serve two simultaneous roles.

+ *Predictive input*: As input tensors to the PLE-adaTT model, they contribute to the prediction of 13 tasks.
+ *Expert routing signal*: The soft assignment probabilities ($gamma_(n k)$) from GMM are used as routing signals that weighted-combine the 20 sub-heads of the GroupTaskExpertBasket, and the 48D HMM output is fed to a dedicated Projector through a separate input path.

#note[Expert Routing Granularity][
  Expert routing (`target_experts` in `feature_groups.yaml`) is declared at the *feature-group level*, not the column level. Each expert receives the full subset of its assigned feature groups as sliced by `FeatureRouter`. Column-level routing is not supported and would break the config-driven design.
]

== Overall Feature Tensor Composition

#styled-table(
  (1.2fr, 0.6fr, 2fr),
  table.header([*Feature Block*], [*Dim.*], [*Components*]),
  [Base], [238D], [RFM(34) + Category(64) + Transaction Stats(76) + Product Diversity(12) + ...],
  [Multi-Source], [91D], [Deposit + Credit + Investment + Digital, etc.],
  [Extended-Source], [84D], [Insurance + Refund + Consultation + STT, etc.],
  [Domain], [159D], [TDA(70) + GMM(22) + Mamba(50) + Economics(17)],
  [Model-Derived], [27D], [HMM summary(5) + Bandit/MAB(4) + LNN(18)],
  [Multidisciplinary], [24D], [Chemical(6) + SIR(5) + Crime(5) + Wave(8)],
  [Merchant Hierarchy], [27D], [MCC Poincaré embeddings (hierarchy) + brand embeddings (was 21D stats only)],
  [*Total (normalized)*], [*644D*], [],
  [Raw power-law copy], [90D], [log1p originals of power-law columns (unscaled)],
  [*Main Tensor Total*], [*734D*], [644D normalized + 90D raw power-law],
)

#v(4pt)
#dim-label[Separate input: HMM Triple-Mode 48D + Hyperbolic 20D = 68D (separate input path)]


// =====================================================================
//  2. Economics Features (17D)
// =====================================================================
= Economics Features (Economics, 17D)

#chip[Domain Features] #chip(color: indigo)[Friedman PIH] #chip(color: amber)[17D / 734D]

== Theoretical Foundation: Friedman's Permanent Income Hypothesis

Friedman's (1957) Permanent Income Hypothesis (PIH) decomposes observed income into a permanent component and a transitory component.

#eq-highlight[
  $ Y_t = Y_t^P + Y_t^T $

  Here $Y_t^P$ is permanent income (long-run stable) and $Y_t^T$ is transitory income (temporary fluctuations). The consumption function follows:
  $ C_t = k(r, w, u) dot Y_t^P $

  Core implication: consumers spend in proportion to their permanent income; transitory income flows into savings and investment.
]

=== Income Decomposition Estimation Methods

#styled-table(
  (1fr, 2fr, 0.8fr),
  table.header([*Method*], [*Formula*], [*Complexity*]),
  [Moving Average], [$hat(Y)_t^P = 1/L sum_(i=0)^(L-1) Y_(t-i)$, $L=12$], [Low],
  [HP Filter], [$min_tau {sum_t (Y_t - tau_t)^2 + lambda sum_t [(tau_(t+1)-tau_t) - (tau_t - tau_(t-1))]^2}$], [Medium],
  [Kalman Filter], [State: $Y_(t+1)^P = Y_t^P + eta_t$, Obs: $Y_t = Y_t^P + epsilon_t$], [High],
)

The first-order condition of the HP Filter yields $(I + lambda D^top D)tau = Y$, a positive-definite banded linear system solvable in $O(T)$ via Cholesky decomposition. Following the Ravn-Uhlig (2002) standard, $lambda = 14400$ is used for monthly data.

== Income Decomposition Output (8D)

#styled-table(
  (1.5fr, 2fr, 2fr),
  table.header([*Feature*], [*Formula*], [*Financial Interpretation*]),
  [`permanent_income_avg`], [$"mean"(hat(Y)^P)$], [Long-run stable income level],
  [`permanent_income_stability`], [$sigma(hat(Y)^P) / mu(hat(Y)^P)$], [CV of permanent income; low value implies stable employment],
  [`permanent_income_growth`], [$(hat(Y)_T^P - hat(Y)_1^P) / hat(Y)_1^P$], [Income trajectory (indicator for card tier upgrade)],
  [`permanent_income_trend`], [REGR_SLOPE / polyfit], [Robust long-term growth direction],
  [`transitory_income_avg`], [$"mean"(hat(Y)^T)$], [Bonus frequency indicator (theoretically $approx 0$)],
  [`transitory_income_volatility`], [$sigma(hat(Y)^T)$], [Magnitude of income uncertainty],
  [`transitory_income_max`], [$max(hat(Y)^T)$], [Proxy for the largest bonus event],
  [`bonus_frequency`], [$"count"(hat(Y)^T > 0.5 hat(Y)^P) / N$], [Proportion of periods with a large bonus],
)

== Microeconomic Behavioral Features (9D)

=== Income Elasticity

#eq-highlight[
  $ epsilon_Y = frac(partial Q, partial Y) dot frac(Y, Q) = frac(d ln Q, d ln Y) $

  - $epsilon_Y > 1$: propensity for luxury consumption (spending grows faster than income)
  - $0 < epsilon_Y < 1$: propensity for necessity consumption
  - $epsilon_Y < 0$: inferior good (consumption declines as income rises)
]

Implemented as the discrete arc elasticity: $hat(epsilon)_Y = 1/T sum_(t=1)^T (Delta S_t / S_(t-1)) / (Delta Y_t / Y_(t-1))$

=== Consumption Smoothing

Hall (1978) combined the PIH with rational expectations to show that optimal consumption follows a random walk.

$ C_t = C_(t-1) + epsilon_t, quad epsilon_t tilde "WN"(0, sigma^2) $

`consumption_smoothing` $= mu / sigma$ (inverse CV, the Sharpe ratio of consumption). Higher values indicate closer proximity to the theoretical optimum.

=== Time Discounting

$ V_0 = sum_(t=0)^T beta^t u(C_t), quad 0 < beta < 1 $

`discount_rate_proxy` $=$ first-half spending / second-half spending. A larger value implies a smaller $\beta$ and stronger present bias — useful for identifying customers who prefer immediate rewards (same-day cashback).

=== Other Behavioral Features

#styled-table(
  (1.5fr, 2fr, 2fr),
  table.header([*Feature*], [*Formula*], [*Interpretation*]),
  [`spending_diversification`], [$H = -sum_i s_i ln(s_i)$], [Shannon entropy; category diversity],
  [`category_hhi`], [$"HHI" = sum_i s_i^2$], [Herfindahl index; spending concentration],
  [`savings_propensity`], [$(Y - C) / Y$], [Savings propensity; reflects expected future utility],
  [`price_sensitivity`], [Refund ratio proxy], [Behavioral proxy for price elasticity],
  [`cross_category_elasticity`], [Time variation in number of categories], [Cross-elasticity proxy],
)

Shannon entropy and HHI are special cases of Rényi entropy $H_alpha = 1/(1-alpha) ln(sum s_i^alpha)$: Shannon corresponds to $alpha -> 1$, and HHI to $alpha = 2$ ($H_2 = -ln("HHI")$).


// =====================================================================
//  3. Chemical Kinetics Features (6D)
// =====================================================================
= Chemical Kinetics Features (Chemical Kinetics, 6D)

#chip[Multidisciplinary] #chip(color: indigo)[Arrhenius Equation] #chip(color: amber)[6D / 24D]

== Theoretical Foundation

In the physicochemical worldview, every transformation (reaction) occurs on an energy landscape. For a system to move from one state to another, it must surmount an energy barrier; the height of that barrier determines the difficulty of the transition, while the frequency of surmounting it determines the rate.

=== Arrhenius Equation

#eq-highlight[
  $ k = A e^(-E_a \/ R T) $

  The category-switching frequency ($k$) increases exponentially when the entry barrier ($E_a$) decreases or consumer activity ($T$) increases.
]

=== Half-Life

The half-life of a first-order reaction, $T_(1\/2) = ln 2 \/ k$, corresponds to the median transaction interval. Customers with a short half-life exhibit a rapid transaction cycle.

=== Second-Order Finite Difference

Spending acceleration is the discrete second-order derivative:

$ f''(t) approx f(t+Delta t) - 2f(t) + f(t-Delta t) $

Derived by cancellation of the $f'$ term in the Taylor expansion, it distinguishes acceleration (positive) from deceleration (negative).

== Output Features (6D)

#styled-table(
  (1.5fr, 2fr, 2fr),
  table.header([*Feature*], [*Definition*], [*Financial Interpretation*]),
  [`new_category_activation_rate`], [New MCCs in last 30 days / active MCCs], [Proxy for inverse activation energy],
  [`spending_half_life`], [Median transaction interval (days)], [Chemical half-life $T_(1\/2) = ln 2 / k$],
  [`spending_acceleration`], [$f(t+1) - 2f(t) + f(t-1)$], [Second-order finite difference: acceleration (+) or deceleration (−)],
  [`dormancy_reactivation_rate`], [MCC present in W1, absent in W2, reappearing in W3], [Catalytic reactivation rate],
  [`catalyst_sensitivity`], [Daily avg. spend in early month / daily avg. spend in late month], [Payday catalyst elasticity],
  [`saturation_proximity`], [$"max" / ("avg" + "std")$], [Proximity to spending ceiling],
)


// =====================================================================
//  4. SIR Epidemiology Features (5D)
// =====================================================================
= SIR Epidemiology Features (Epidemiology, 5D)

#chip[Multidisciplinary] #chip(color: indigo)[Compartmental Model] #chip(color: amber)[5D / 24D]

== Theoretical Foundation: Kermack-McKendrick Model

#eq-highlight[
  $ frac(d S, d t) = -beta S I, quad frac(d I, d t) = beta S I - gamma I, quad frac(d R, d t) = gamma I $

  Basic reproduction number: $R_0 = beta / gamma$. When $R_0 > 1$, adoption spreads in a self-reinforcing manner; when $R_0 < 1$, it naturally dies out.
]

== Financial Domain Mapping

#styled-table(
  (0.8fr, 1fr, 2fr),
  table.header([*Compartment*], [*Epidemiology*], [*Financial Interpretation*]),
  [S (Susceptible)], [Uninfected], [Unused categories among the top-15 MCC by population],
  [I (Infected)], [Currently spreading], [Daily avg. frequency in last 30 days > prior period],
  [R (Recovered)], [Immune/recovered], [Categories used in the past but inactive in last 30 days],
)

The infection classification criterion applies a period-length correction factor of $30/(L-30)$.

== Output Features (5D)

#styled-table(
  (1.5fr, 2.5fr),
  table.header([*Feature*], [*Description*]),
  [`sir_susceptible`], [S ratio. High value indicates a wide recommendation opportunity space (exploration-ready)],
  [`sir_infected`], [I ratio. High value indicates active adoption underway (optimal timing for cross-sell)],
  [`sir_recovered`], [R ratio. High value indicates contraction (target for retention campaigns)],
  [`sir_r0`], [Basic reproduction number. Degree of self-reinforcing adoption spread],
  [`sir_infection_rate`], [Adoption speed. Corresponds to $beta S$],
)

== Customer Profiling

#styled-table(
  (1fr, 0.5fr, 0.5fr, 0.5fr, 2fr),
  table.header([*Profile*], [*S*], [*I*], [*R*], [*Interpretation*]),
  [Exploration-Ready], [High], [Low], [Low], [Maximum recommendation opportunity space],
  [Active Adoption], [Mid], [High], [Low], [Optimal moment for cross-selling],
  [Stable Usage], [Low], [Low], [Low], [Loyal customer],
  [Contraction], [Low], [Low], [High], [Target for churn-prevention campaign],
)


// =====================================================================
//  5. Criminology Features (5D)
// =====================================================================
= Criminology Features (Crime Pattern / Routine Activity, 5D)

#chip[Multidisciplinary] #chip(color: indigo)[Routine Activity Theory] #chip(color: amber)[5D / 24D]

== Theoretical Foundation

In Cohen & Felson's (1979) Routine Activity Theory, behavior is governed by daily routines, and deviations from those routines create opportunities for anomalous events.

=== Burstiness (Barabasi, 2005)

#eq-highlight[
  $ B = frac(sigma_tau - mu_tau, sigma_tau + mu_tau) in [-1, 1] $

  - $B = -1$: perfectly regular intervals (recurring payments)
  - $B = 0$: Poisson process (random)
  - $B = +1$: extreme clustering (explosive shopping)
]

Human behavior follows not a Poisson process but a heavy-tailed distribution characterized by short bursts followed by long rests.

=== Circular Statistics

Essential for analyzing transaction timestamps. The distance between 11 PM and 1 AM is 2 hours, not 22.

#eq-highlight[
  $ overline(bold(R)) = (1/n sum cos theta_i, 1/n sum sin theta_i), quad "CV" = 1 - |overline(bold(R))| $

  where $theta = 2 pi h \/ 24$. Euclidean distance distorts circular time, whereas circular statistics correctly measures concentration.
]

== Output Features (5D)

#styled-table(
  (1.5fr, 2.5fr),
  table.header([*Feature*], [*Description*]),
  [`txn_burstiness`], [Burstiness of transaction intervals. Regularity vs. explosiveness of spending rhythm],
  [`time_circular_variance`], [Circular variance of transaction timestamps. Concentration across time-of-day],
  [`routine_stability`], [Routine stability. Consistency of day-of-week patterns],
  [`breakpoint_count`], [Number of breakpoints. How many times the spending routine was disrupted],
  [`anomaly_score`], [Anomaly pattern score. Intensity of deviation from routine],
)


// =====================================================================
//  6. Wave Interference Features (8D)
// =====================================================================
= Wave Interference Features (Wave Interference, 8D)

#chip[Multidisciplinary] #chip(color: indigo)[Spectral Analysis] #chip(color: amber)[8D / 24D]

== Theoretical Foundation

In the wave-physics worldview, when multiple waves superpose, their phase relationship produces either constructive interference (amplitude increase) or destructive interference (amplitude decrease).

=== KL Divergence

#eq-highlight[
  $ D_"KL"(P||Q) = sum P(x) ln frac(P(x), Q(x)) $

  The KL divergence between weekday and weekend spending distributions quantifies, in _bits_, how differently a customer behaves across the two contexts. By Gibbs' inequality (a consequence of Jensen's inequality), it is always non-negative.
]

=== Phase Locking Value (PLV)

A concept borrowed from functional connectivity in neuroscience.

$ "PLV" = frac(1, T) |sum_(t=1)^T e^(j(phi_x (t) - phi_y (t)))| $

It measures the consistency of the phase difference between the spending rhythms of two categories, extracted via the Hilbert transform.

=== Cross-spectral Coherence

$ C_(x y)(f) = frac(|S_(x y)(f)|^2, S_(x x)(f) dot S_(y y)(f)) $

A frequency-resolved correlation that identifies at which periodicities categories co-synchronize.

== Output Features (8D)

#styled-table(
  (1.5fr, 2.5fr),
  table.header([*Feature*], [*Description*]),
  [`weekday_weekend_kl`], [KL divergence between weekday and weekend spending distributions],
  [`spectral_entropy`], [Normalized Shannon entropy of the power spectrum. Predictability of spending periodicity],
  [`dominant_period`], [Period (days) of the strongest frequency component identified by FFT],
  [`phase_locking_value`], [PLV. Phase synchronization of spending rhythms across categories],
  [`cross_coherence`], [Cross-spectral coherence. Per-frequency category correlation],
  [`hhi_shift`], [Change in HHI. Trend toward spending concentration or diversification],
  [`category_sync_ratio`], [Synchronization ratio. Proportion of category pairs in phase],
  [`interference_strength`], [Interference strength. Degree of constructive/destructive interference across categories],
)

== Information-Theoretic Justification

The four multidisciplinary modules capture nearly orthogonal projections:
- *Chemical Kinetics*: the _differential structure_ of time (first- and second-order derivatives)
- *Epidemiological Diffusion*: the _state-space transition structure_ (S -> I -> R)
- *Crime Patterns*: the _statistical texture_ of time series (periodicity, clustering, dispersion)
- *Wave Interference*: the _frequency-domain spectral structure_ (FFT, coherence, phase)

Cross-module combinations reveal patterns invisible to individual modules: for example, high `catalyst_sensitivity` + high `burstiness` = payday-burst spender (optimal target for early-month promotional campaigns).


// =====================================================================
//  7. TDA Features (70D)
// =====================================================================
= TDA Features (Topological Data Analysis, 70D)

#chip[Domain Features] #chip(color: indigo)[Persistent Homology] #chip(color: amber)[70D / 159D]

== Core Idea

#note[Core Principle of TDA][
  By growing balls of increasing radius around data points, we track _when_ topological characteristics such as connected components, holes, and voids _appear and disappear_. Long-lived characteristics are signal; short-lived ones are noise.
]

Strengths of TDA:
+ *Coordinate invariance*: topological characteristics are preserved under rotation and continuous deformation of the data. Robust to feature scaling.
+ *Multi-resolution observation*: all scales are considered simultaneously, rather than a single threshold.
+ *Noise resistance*: by retaining only long-lived characteristics, spurious structure is naturally filtered out.

== Betti Numbers and Homology

#eq-highlight[
  $ beta_k = "rank"(H_k (X)) $

  $H_k (X)$ is the $k$-dimensional homology group of space $X$.

  - $beta_0$: number of connected components (how many clusters the points form)
  - $beta_1$: number of one-dimensional holes (how many loops/cycles exist)
  - $beta_2$: number of two-dimensional voids (how many enclosed cavities exist)
]

Homology identifies chains that have no boundary and are not themselves the boundary of anything else:

$ H_k = "Ker"(partial_k) slash "Im"(partial_(k+1)) $

=== Financial Domain Interpretation

- *Large $H_0$*: spending falls into several disconnected clusters (living expenses vs. travel vs. education)
- *Non-trivial $H_1$*: periodic circular pattern (month-start card payment → mid-month food → month-end leisure → repeat)
- *Non-trivial $H_2$*: "hollow" pattern in 3D structure (central void in the amount–category–time space)

== Vietoris-Rips Complex and Persistent Homology

This system employs the Vietoris-Rips complex:

$ sigma = {x_0, ..., x_k} in "VR"_epsilon (X) quad "iff" quad d(x_i, x_j) <= epsilon, forall i,j $

It is a "2× radius" approximation of the Čech complex, but computationally far more efficient (requires only pairwise distance comparisons).

Persistent Homology observes _all scales simultaneously_, resolving the sensitivity to a single threshold. Points far from the diagonal in a Persistence Diagram represent strong signals.

=== Stability Theorem

$ d_B ("Dgm"(f), "Dgm"(g)) <= ||f - g||_infinity $

Perturbations to the input are bounded by the maximum change in the filtration function. This mathematically guarantees the noise robustness of topological features.

== Output Features (70D)

#styled-table(
  (1fr, 0.6fr, 2fr),
  table.header([*Subgroup*], [*Dim.*], [*Description*]),
  [`tda_short`], [24D], [Short-term topological patterns from 90-day app logs],
  [`tda_long`], [36D], [Long-term topological patterns from 12-month financial transactions],
  [`phase_transition`], [10D], [Topological change detection across time windows],
)

#note[Relationship to the PersLay Expert][
  The TDA features in this document (offline 70D) and the PersLay Expert (online 64D) both leverage Persistent Homology but serve different roles. The 70D features are part of the 734D main tensor and are precomputed during batch preprocessing. The PersLay Expert learns end-to-end from Persistence Diagrams inside the PLE model.
]


// =====================================================================
//  8. HMM Features (25D = 5D summary + separate 48D → 53D total)
// =====================================================================
= HMM Features (Hidden Markov Model, 5D summary + 48D separate)

#chip[Model-Derived + Separate Input] #chip(color: indigo)[Triple-Mode HMM] #chip(color: amber)[53D total]

== Core Idea: Hidden State Inference

What can be directly observed from card transaction data is limited to transaction amount, frequency, and category diversity. Yet even an identical 100,000-won monthly spend may belong to two entirely different states: "a customer exploring a new service" versus "a last-gasp spend before churn." The HMM *probabilistically infers* unobservable latent states from observed data.

=== Markov Property

$ P(q_(t+1) | q_t, q_(t-1), ..., q_1) = P(q_(t+1) | q_t) $

The next state depends only on the current state. This simplification bounds the parameter count to $O(N^2)$, enabling stable learning across hundreds of thousands of customers.

== Triple-Mode Architecture

#styled-table(
  (1fr, 0.8fr, 0.8fr, 2fr),
  table.header([*Mode*], [*States*], [*Output*], [*Patterns Captured*]),
  [Journey], [5], [16D], [Daily/weekly customer journey (AWARENESS → CONSIDERATION → PURCHASE → ...)],
  [Lifecycle], [5], [16D], [Monthly/yearly lifecycle (NEW → GROWING → MATURE → AT_RISK → ...)],
  [Behavior], [6], [16D], [Monthly behavioral pattern types (frugal, investment-oriented, consumption-oriented, etc.)],
)

The 48D output from the three modes is supplied as a separate input to the PLE's HMM Triple-Mode Projector.

== Three Core Algorithms

#styled-table(
  (0.8fr, 1.5fr, 1fr, 1.5fr),
  table.header([*Problem*], [*Question*], [*Algorithm*], [*Usage*]),
  [Evaluation], [What is the likelihood of a sequence?], [Forward], [Model quality validation, anomaly detection],
  [Decoding], [What is the optimal state sequence?], [Viterbi], [State sequence extraction → meta features],
  [Learning], [What are the optimal parameters?], [Baum-Welch (EM)], [Optimize transition/emission distributions],
)

=== Forward-Backward

$ alpha_t (i) = P(o_1, ..., o_t, q_t = S_i | lambda), quad gamma_t (i) = P(q_t = S_i | bold(O), lambda) $

$gamma_t(i)$ is the soft state-assignment probability and constitutes the core of the features. A probability vector such as "80% active, 15% growing, 5% at-risk" is itself a rich feature.

== HMM Summary (Main Tensor 5D)

#styled-table(
  (1.5fr, 2.5fr),
  table.header([*Feature*], [*Description*]),
  [`hmm_dominant_state`], [State ID with the highest $gamma_t(i)$ aggregated across the three modes],
  [`hmm_state_duration`], [Consecutive duration (months) spent in the current dominant state],
  [`hmm_transition_stability`], [Inverse of the state-transition frequency over the last N months],
  [`hmm_transition_entropy`], [$H = -sum_j a_(i j) log a_(i j)$. High value implies difficulty predicting the next state],
  [`hmm_state_change_rate`], [Number of state changes / total observation period],
)

#note[HMM vs. GMM][
  HMM performs _time-series_ state inference: "What _stage_ is this customer currently in, and where will they move next?" GMM performs _cross-sectional_ clustering: "What _type_ is this customer?" Because the transition matrix $bold(A)$ captures temporal dynamics, the HMM can distinguish "currently growing" from "on the verge of churn" even for identical spending patterns. The two modules are supplied to the PLE through separate input paths and are mutually complementary.
]


// =====================================================================
//  9. GMM Features (22D)
// =====================================================================
= GMM Features (Gaussian Mixture Model, 22D)

#chip[Domain Features] #chip(color: indigo)[EM Algorithm + BIC] #chip(color: amber)[22D / 159D]

== Theoretical Foundation

#eq-highlight[
  $ p(bold(x)) = sum_(k=1)^K pi_k cal(N)(bold(x) | bold(mu)_k, bold(Sigma)_k) $

  where $pi_k >= 0$, $sum_k pi_k = 1$, and each component is a multivariate Gaussian.
  Architecture: $K = 20$ clusters, $D = 40$ input dimensions, `covariance_type = "full"`.
]

Core assumption: "The observed data were generated from $K$ subpopulations, and each data point's group membership is unobserved (a latent variable)."

== EM Algorithm

=== E-Step (Posterior Responsibility)

$ gamma_(n k) = frac(pi_k cal(N)(bold(x)_n | bold(mu)_k, bold(Sigma)_k), sum_j pi_j cal(N)(bold(x)_n | bold(mu)_j, bold(Sigma)_j)) $

A direct application of Bayes' theorem: the prior $pi_k$ and likelihood $cal(N)_k$ are combined to yield the posterior $gamma_(n k)$.

=== M-Step

$gamma_(n k)$ is used as a weight to update $bold(mu)_k$, $bold(Sigma)_k$, and $pi_k$.

=== Convergence Guarantee

By Jensen's inequality, EM monotonically non-decreases the ELBO. Global optimality is not guaranteed; this is mitigated with `n_init=10`.

== Model Selection: BIC

$ "BIC" = -2 ln hat(L) + k ln(n) $

With hundreds of thousands of data points, BIC's $k ln(n)$ penalty prevents overfitting more effectively than AIC's $-2 ln hat(L) + 2k$.

== Output Features (22D)

#styled-table(
  (1.5fr, 0.6fr, 2.5fr),
  table.header([*Feature*], [*Dim.*], [*Description*]),
  [`cluster_prob_00` -- `19`], [20D], [Soft assignment probabilities $gamma_(n k)$ (sum = 1.0)],
  [`cluster_id`], [1D], [Hard assignment $arg max_k gamma_(n k)$],
  [`cluster_entropy`], [1D], [$H_n = -sum_k gamma_(n k) ln(gamma_(n k) + epsilon)$],
)

Entropy interpretation:
- $H approx 0$: a clear behavioral archetype (high-confidence classification)
- $H = ln(20) approx 2.996$: uniform distribution (cold-start / unclassifiable)

== GMM vs. K-Means: Why Soft Assignment?

#styled-table(
  (1fr, 1.5fr, 1.5fr),
  table.header([*Perspective*], [*K-Means*], [*GMM*]),
  [Assignment], [Hard: one-hot (1 bit)], [Soft: probability vector (~4.32 bits max)],
  [Boundary customers], [Arbitrary assignment, unstable], [Probability spread across adjacent clusters],
  [Cluster shape], [Spherical (Euclidean)], [Ellipsoidal (Mahalanobis, full covariance)],
  [Uncertainty], [None], [Entropy-based confidence],
  [PLE role], [Single sub-head activation], [$gamma_(n k)$-weighted ensemble combining 20 sub-heads],
)

The Mahalanobis distance $d_M = sqrt((bold(x)-bold(mu))^top bold(Sigma)^(-1)(bold(x)-bold(mu)))$ generalizes Euclidean distance ($bold(Sigma) = bold(I)$) by accounting for feature correlations and scale differences.


// =====================================================================
//  10. Mamba Time-Series Features (50D)
// =====================================================================
= Mamba Time-Series Features (Selective State Space, 50D)

#chip[Domain Features] #chip(color: indigo)[Mamba SSM + PCA] #chip(color: amber)[50D / 159D]

== Core Idea

Simple aggregate statistics (sum, mean, standard deviation) completely destroy the *temporal ordering* of a time series. Shuffle the values randomly and the sum and mean remain unchanged. Mamba SSM captures nonlinear long-range dependencies in time series via a selective state-space mechanism.

== Selective State Space Model

Mamba is based on the discretization of the Continuous State Space Model (SSM):

#eq-highlight[
  Continuous: $h'(t) = bold(A) h(t) + bold(B) x(t), quad y(t) = bold(C) h(t)$

  Discretization: $overline(bold(A)) = exp(Delta bold(A)), quad overline(bold(B)) = (Delta bold(A))^(-1)(exp(Delta bold(A)) - bold(I)) dot Delta bold(B)$

  Key point: Because $Delta_k$ varies *dynamically* with the input, the model adaptively updates its state even for non-stationary time series.
]

=== Advantages over Prior Methods

#styled-table(
  (1fr, 1fr, 1fr, 1fr),
  table.header([*Perspective*], [*RNN*], [*Transformer*], [*Mamba*]),
  [Long-range dependency], [Vanishing gradient], [Resolved via self-attention], [Resolved via selective SSM],
  [Sequence length], [$O(T)$ sequential], [$O(T^2)$ memory], [$O(T)$ linear],
  [Non-stationarity], [Difficult to adapt], [Dependent on positional encoding], [$Delta_k$ dynamic adaptation],
)

== Pipeline

+ MCC 15D $times$ 180-day input sequence
+ Mamba encoder (`d_model=256`) extracts hidden representations
+ PCA $arrow.r$ 50D compression
+ Integrated into the Domain block of the 734D main tensor

#warn[Distinction from the Online Mamba Expert][
  These features (offline 50D) use `d_model=256` with a 15D MCC $times$ 180-day input and are precomputed _before_ PLE training. The Temporal Expert's Mamba inside PLE uses `d_model=128` with a 16D card $times$ 180-step input and is trained end-to-end. No weight sharing.
]

== Four Perspectives of Time-Series Analysis

Mamba 50D is a _learned representation_ that nonlinearly integrates information from the following four perspectives:

#styled-table(
  (1fr, 1.5fr, 1.5fr),
  table.header([*Perspective*], [*Key Question*], [*Technique*]),
  [Time domain], [How do values change over time?], [Autocorrelation, change points, moving average],
  [Frequency domain], [What periodicities repeat?], [FFT, spectral analysis],
  [Distribution/shape], [What shape does the value distribution take?], [Skewness, kurtosis, tail probability],
  [Information theory], [How complex and predictable is the series?], [Entropy, permutation entropy],
)


// =====================================================================
//  11. Graph Features (LightGCN + Hyperbolic, 66D embeddings)
// =====================================================================
= Graph Features (LightGCN + Hyperbolic GCN)

#chip[Offline Precomputed] #chip(color: indigo)[Graph Neural Network] #chip(color: amber)[64D + H-GCN aggregation]

== LightGCN: Collaborative Filtering

=== Message Passing

#eq-highlight[
  $ bold(e)_u^((k+1)) = sum_(i in cal(N)_u) frac(1, sqrt(|cal(N)_u|) dot sqrt(|cal(N)_i|)) dot bold(e)_i^((k)) $

  Symmetric normalization $tilde(A) = D^(-1\/2) A D^(-1\/2)$ simultaneously suppresses the influence of popular items (sender) and high-degree receivers.
]

=== Layer Combination

$ bold(e)_u^"final" = frac(1, L+1) sum_(k=0)^L bold(e)_u^((k)) $

Uniform average of the 0-hop (self) through 1-, 2-, 3-hop neighbors. Empirically superior without learnable attention weights, and prevents overfitting.

=== BPR Loss

$ cal(L)_"BPR" = -sum_((u, i^+, i^-)) log sigma(hat(y)_(u i^+) - hat(y)_(u i^-)) + lambda ||Theta||^2 $

A pairwise ranking loss that optimizes relative ranking rather than absolute scores.

== Hyperbolic GCN: Hierarchical Structure

=== Poincaré Ball Model

#eq-highlight[
  $ BB_c^d = { bold(x) in RR^d : c||bold(x)||^2 < 1 } $

  *Exponential map*: $exp_bold(0)(bold(v)) = tanh(sqrt(c)||bold(v)||) dot bold(v) / (sqrt(c)||bold(v)||)$

  *Poincaré distance*: $d_(BB)(bold(x), bold(y)) = 1/sqrt(c) "arccosh"(1 + frac(2c||bold(x)-bold(y)||^2, (1-c||bold(x)||^2)(1-c||bold(y)||^2)))$
]

Near the boundary the denominator approaches 0, so the hyperbolic distance explodes even when the Euclidean distance is small. This is the mechanism that naturally encodes hierarchical depth.

=== Financial Domain Justification

The MCC taxonomy (Root → L1(8) → L2(~100) → Brand(~50K) → Branch(~500K)) is inherently a tree structure. Hyperbolic space matches tree branching through exponential volume growth, embedding ~550K nodes in an 8D Poincaré Ball. Nickel & Kiela (2017) result: 5D hyperbolic > 200D Euclidean (WordNet hierarchy).

== Dual GCN Architecture

#styled-table(
  (1fr, 1.5fr, 1.5fr),
  table.header([*Property*], [*LightGCN*], [*H-GCN*]),
  [Nodes], [Customers + merchants (bipartite graph)], [Merchants only (MCC tree)],
  [Edges], [Customer–merchant transactions], [Parent–child hierarchy + brand co-visitation],
  [Space], [$RR^(64)$ Euclidean], [$BB^8$ Poincaré Ball],
  [Learning objective], ["Who likes what"], ["How are merchants structurally related"],
  [Output], [Customer embedding 64D (direct)], [Merchant embeddings → per-customer aggregation (indirect)],
)

*2-Stage Pipeline*: Stage 1 (offline) graph training → save embedding Parquet. Stage 2 (online) lookup + lightweight MLP adaptation. No graph propagation at inference — VRAM-friendly on a single GPU.


// =====================================================================
//  12. Other Features (Base + Product + Transaction Behavior)
// =====================================================================
= Other Features (Base Demographics, Product Holdings, Transaction Behavior)

#chip[Base Features] #chip(color: indigo)[238D + 91D + 84D] #chip(color: amber)[413D / 644D]

== Base Features (238D)

The foundational feature block of traditional financial ML.

#styled-table(
  (1.2fr, 0.6fr, 2fr),
  table.header([*Subgroup*], [*Dim.*], [*Description*]),
  [RFM (Recency, Frequency, Monetary)], [34D], [Multi-dimensional decomposition of transaction recency, frequency, and amount],
  [Category], [64D], [Spending share/amount/frequency per MCC category],
  [Transaction Stats], [76D], [Transaction statistics: mean, median, std. dev., skewness, kurtosis, etc.],
  [Product Diversity], [12D], [Product holding diversity, cross-holding patterns],
  [Channel Behavior], [18D], [Online/offline/mobile channel ratios],
  [Temporal Pattern], [22D], [Transaction distribution by day-of-week and time-of-day],
  [Demographics], [12D], [Age, gender, region, and other demographic attributes],
)

== Multi-Source Features (91D)

#styled-table(
  (1.2fr, 0.6fr, 2fr),
  table.header([*Subgroup*], [*Dim.*], [*Description*]),
  [Deposit], [~25D], [Deposit balance, inflow/outflow patterns, interest rate sensitivity],
  [Credit], [~20D], [Credit rating, utilization ratio, delinquency history],
  [Investment], [~15D], [Investment product holdings, risk preference],
  [Digital Engagement], [~31D], [App logs, login frequency, feature usage],
)

== Extended-Source Features (84D)

#styled-table(
  (1.2fr, 0.6fr, 2fr),
  table.header([*Subgroup*], [*Dim.*], [*Description*]),
  [Insurance], [~20D], [Insurance product holdings, subscription history],
  [Refund/Cancellation], [~15D], [Refund frequency, cancellation patterns],
  [Consultation/STT], [~25D], [Consultation history, speech recognition analysis],
  [External Signals], [~24D], [Signals from external data integrations],
)

== Model-Derived Features (27D)

#styled-table(
  (1.2fr, 0.6fr, 2fr),
  table.header([*Subgroup*], [*Dim.*], [*Description*]),
  [HMM Summary], [5D], [Compressed representation of Triple-Mode 48D (dominant state, duration, stability, entropy, change rate)],
  [Bandit/MAB], [4D], [Exploration-exploitation balance indicators from Multi-Armed Bandit],
  [LNN Statistics], [18D], [Hand-crafted statistical features: moving average, volatility, autocorrelation, etc.],
)

== Merchant Hierarchy Features (27D)

Coordinates and embeddings reflecting the MCC hierarchy (Root → L1 → L2 → Brand). Includes MCC Poincaré embeddings extracted from H-GCN (Phase 0 v3/v4; was 21D stats-only prior to v3) and brand similarity based on co-visitation.


// =====================================================================
//  13. 3-Stage Normalization
// =====================================================================
= 3-Stage Normalization Pipeline

#chip[Preprocessing] #chip(color: indigo)[Power-law -> Scaler -> Raw Copy] #chip(color: amber)[644D + 90D]

== Design Principles

The Normalization Pipeline consists of three stages. Its core objective is to *preserve the original magnitude information of power-law-distributed features* while transforming them to a scale suitable for model training.

== Stage 1: Power-Law Detection + log1p Copy Generation

#eq-highlight[
  *Power-Law Detection Criteria*:
  + Candidate selection based on skewness and kurtosis
  + Power-law determination via $R^2$ of log-log regression

  *If power-law is confirmed*: generate a `log1p` copy of the column for separate preservation
  $ x_"raw" = log(1 + x) $
]

Power-law distributions (transaction amounts, balances, etc.) compress most values near zero when StandardScaler is applied due to outliers, resulting in information loss. The log1p transform mitigates this problem.

== Stage 2: StandardScaler (TRAIN fit only)

#eq-highlight[
  $ z = frac(x - mu_"train", sigma_"train") $

  *Applied to*: continuous columns only (excluding binary columns)
  *Core rule*: the Scaler must be fit exclusively on the TRAIN split. Val/test sets are only transformed using the scaler fit on train.
]

#warn[Data Leakage Prevention][
  Fitting the scaler on val/test data allows future information to leak into training, inflating estimated performance beyond its true value. This is the most common leakage pattern; LeakageValidator must verify this before training begins.
]

== Stage 3: Preservation of Raw Power-Law Copy

The `_log` copies of power-law columns (generated in Stage 1) are preserved *without scaling*, retaining their original raw magnitude.

#note[Why Preserve the Raw Copy?][
  StandardScaler redistributes information but *destroys absolute magnitude information*. For example, after z-score transformation, the difference between "monthly spend of 1,000,000 KRW" and "monthly spend of 10,000,000 KRW" is reduced to a relative position. However, for certain tasks (e.g., LTV prediction, spending_bucket classification), the absolute amount is critical. The 90D raw power-law copy provides this information directly to the model.
]

== Final Tensor Composition

#styled-table(
  (1fr, 0.8fr, 2fr),
  table.header([*Segment*], [*Dim.*], [*Processing*]),
  [Normalized continuous], [~554D], [Stage 2 StandardScaler applied],
  [Binary (pass-through)], [~90D], [No scaling (0/1 preserved)],
  [Raw power-law copy], [90D], [Stage 1 log1p only; Stage 2 not applied],
  [*Total*], [*734D*], [644D normalized + 90D raw],
)

== Normalization Pipeline Flow Summary

#figure(
  placement: auto,
  diagram(
    node-stroke: 0.6pt + luma(80),
    edge-stroke: 0.7pt + luma(80),
    node-corner-radius: 3pt,
    spacing: (10pt, 16pt),
    node((0,0), [Raw features], fill: luma(245), width: 32mm),
    edge((0,0), (0,1), "->"),
    node((0,1), [\[Stage 1\] Power-Law Detection \ (skew+kurt → log-log R²)], fill: rgb("#d6e6f0"), width: 58mm),
    edge((0,1), (0,2), "->", label: [power-law → generate log1p copy (90D raw)], label-side: right),
    node((0,2), [\[Stage 2\] StandardScaler \ fit on TRAIN only \ (continuous only; binary excluded)], fill: rgb("#d6e6f0"), width: 58mm),
    edge((0,2), (0,3), "->"),
    node((0,3), [\[Stage 3\] raw power-law copy \ preserved as-is], fill: rgb("#d6e6f0"), width: 58mm),
    edge((0,3), (0,4), "->"),
    node((0,4), [\[Final\] 644D normalized ⊕ 90D raw \ = 734D main tensor], fill: rgb("#e8f5e9"), width: 62mm),
  ),
  caption: [3-stage normalization pipeline: power-law detection → StandardScaler → raw copy preservation.],
)


// =====================================================================
//  Appendix: Design vs. Implementation Dimension Mapping
// =====================================================================
= Appendix: Design vs. Implementation Dimension Mapping

#warn[Note][This Appendix summarizes the dimensional differences between the full-bank design (734D) and the current Santander benchmark implementation (350D, Phase 0 v3/v4). Implementation dimensions can be verified in `outputs/phase0/feature_schema.json`.]

#styled-table(
  (1.2fr, 1fr, 1fr, 2fr),
  table.header([*Feature Groups*], [*Design (734D)*], [*Implementation (350D)*], [*Notes*]),
  [TDA], [70D], [32D], [tda\_global 16D + tda\_local 16D],
  [HMM], [48D + 5D (separate)], [25D], [main tensor only],
  [Base (Profile, etc.)], [238D], [47D], [Demographics, RFM, Financial Summary reduced],
  [Graph], [unspecified], [66D], [added as independent group in implementation],
  [Merchant / Hierarchy], [27D], [34D], [MCC Poincaré embeddings + brand embeddings (Phase 0 v3/v4)],
  [GMM], [22D], [53D], [number of clusters and derived features expanded],
  [Others (Economics, SIR, etc.)], [335D], [93D], [Mamba, Wave, Crime, etc.; expanded vs. prior],
  [*Total*], [*734D*], [*350D*], [13 feature groups],
)


// =====================================================================
//  Appendix: References
// =====================================================================
= References

== Economics
- Friedman, M. (1957). _A Theory of the Consumption Function_. Princeton UP.
- Hall, R. (1978). Stochastic Implications of the Life Cycle-PIH. _JPE_.
- Hodrick, R. & Prescott, E. (1997). Postwar U.S. Business Cycles. _JMCB_.
- Kalman, R. (1960). A New Approach to Linear Filtering. _J. Basic Engineering_.
- Kahneman, D. & Tversky, A. (1979). Prospect Theory. _Econometrica_.

== Multidisciplinary
- Arrhenius, S. (1889). Reaction rates of sucrose inversion.
- Kermack, W. & McKendrick, A. (1927). Mathematical Theory of Epidemics. _Proc. Royal Society_.
- Cohen, L. & Felson, M. (1979). Routine Activity Approach. _ASR_.
- Barabasi, A.-L. (2005). The origin of bursts and heavy tails. _Nature_.
- Shannon, C. (1948). A Mathematical Theory of Communication. _Bell System Technical Journal_.
- Kullback, S. & Leibler, R. (1951). On Information and Sufficiency.

== TDA
- Carlsson, G. (2009). Topology and Data. _Bulletin of the AMS_.
- Edelsbrunner, H., Letscher, D. & Zomorodian, A. (2002). Topological Persistence and Simplification.
- Cohen-Steiner, D., Edelsbrunner, H. & Harer, J. (2007). Stability of Persistence Diagrams. _DCG_.
- Carriere, M. et al. (2020). PersLay. _AISTATS_.

== HMM
- Baum, L. & Petrie, T. (1966). Statistical Inference for Probabilistic Functions of Finite State Markov Chains. _AMS_.
- Rabiner, L. (1989). A Tutorial on HMM. _Proc. IEEE_.
- Dempster, A., Laird, N. & Rubin, D. (1977). Maximum Likelihood from Incomplete Data via EM. _JRSS-B_.

== GMM
- Pearson, K. (1894). Contributions to the Mathematical Theory of Evolution. _Phil. Trans. Royal Society A_.
- Schwarz, G. (1978). Estimating the Dimension of a Model. _Annals of Statistics_.
- Bishop, C. (2006). _Pattern Recognition and Machine Learning_, Ch. 9.

== Time Series / Mamba
- Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
- Cooley, J. & Tukey, J. (1965). An Algorithm for the Machine Calculation of Complex Fourier Series. _Math. Comp._

== Graph
- He, X. et al. (2020). LightGCN: Simplifying and Powering GCN for Recommendation. _SIGIR_.
- Chami, I. et al. (2019). Hyperbolic Graph Convolutional Neural Networks. _NeurIPS_.
- Nickel, M. & Kiela, D. (2017). Poincare Embeddings for Learning Hierarchical Representations. _NeurIPS_.
- Rendle, S. et al. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. _UAI_.
