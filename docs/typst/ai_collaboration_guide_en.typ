// ─────────────────────────────────────────────
// AI Collaboration Guidelines: A Framework for Expanding Knowledge and Solving Problems
// Typst Web App Compatible
// ─────────────────────────────────────────────

#let navy = rgb("#1B2A4A")
#let burgundy = rgb("#6B2D3E")
#let gold = rgb("#B8860B")
#let cream = rgb("#FAF6F0")
#let dark-cream = rgb("#F0EBE1")
#let ink = rgb("#2C2C2C")
#let muted = rgb("#5A5A5A")
#let light-rule = rgb("#C4B8A8")

// ── Page setup ──
#set page(
  paper: "a4",
  margin: (top: 2.8cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
  fill: cream,
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.5pt, font: "New Computer Modern", fill: muted, tracking: 0.12em)
      #smallcaps[AI Collaboration Guidelines]
      #h(1fr)
      #smallcaps[A Framework for Expanding Knowledge and Solving Problems]
      #v(4pt)
      #line(length: 100%, stroke: 0.4pt + light-rule)
    ]
  },
  footer: context {
    let pg = counter(page).get().first()
    if pg > 1 [
      #line(length: 100%, stroke: 0.3pt + light-rule)
      #v(4pt)
      #set text(size: 8pt, font: "New Computer Modern", fill: muted)
      #h(1fr)
      — #pg —
      #h(1fr)
    ]
  },
)

// ── Base text ──
#set text(
  font: "New Computer Modern",
  size: 10.5pt,
  fill: ink,
  lang: "en",
)

#set par(
  justify: true,
  leading: 0.78em,
  first-line-indent: 1.2em,
)

// ── Heading styles ──
#show heading.where(level: 1): it => {
  v(0cm)
  set par(first-line-indent: 0pt)
  align(center)[
    #block(width: 100%)[
      #v(6pt)
      #line(length: 40%, stroke: 0.8pt + gold)
      #v(10pt)
      #text(
        size: 10pt,
        fill: gold,
        tracking: 0.35em,
        weight: "regular",
      )[#upper(smallcaps[Part])]
      #v(2pt)
      #text(
        size: 18pt,
        fill: navy,
        weight: "bold",
      )[#it.body]
      #v(10pt)
      #line(length: 40%, stroke: 0.8pt + gold)
      #v(6pt)
    ]
  ]
  v(0.6cm)
}

#show heading.where(level: 2): it => {
  v(0.0cm)
  set par(first-line-indent: 0pt)
  block[
    #text(
      size: 10pt,
      fill: burgundy,
      tracking: 0.25em,
      weight: "regular",
    )[#upper(smallcaps[Section])]
    #v(-2pt)
    #text(size: 13.5pt, fill: navy, weight: "bold")[#it.body]
    #v(0pt)
    #line(length: 100%, stroke: (paint: light-rule, thickness: 0.4pt, dash: "loosely-dotted"))
  ]
  v(0.0cm)
}

#show heading.where(level: 3): it => {
  v(0.0cm)
  set par(first-line-indent: 0pt)
  block[
    #text(size: 11pt, fill: burgundy, weight: "bold", style: "italic")[#it.body]
  ]
  v(0.15cm)
}

// ── Custom components ──

// Ornamental divider
#let ornament() = {
  v(0.4cm)
  align(center)[
    #text(size: 11pt, fill: gold)[✦ #h(6pt) ◆ #h(6pt) ✦]
  ]
  v(0.4cm)
}

// Small ornament
#let small-ornament() = {
  v(0.2cm)
  align(center)[
    #text(size: 8pt, fill: light-rule)[— ✦ —]
  ]
  v(0.2cm)
}

// Epigraph / pull quote
#let epigraph(body, source: none) = {
  v(0.3cm)
  set par(first-line-indent: 0pt)
  block(
    width: 100%,
    inset: (left: 2cm, right: 2cm, top: 0.4cm, bottom: 0.4cm),
  )[
    #align(center)[
      #text(size: 12pt, fill: navy, style: "italic")[#body]
      #if source != none [
        #v(4pt)
        #text(size: 8pt, fill: muted, tracking: 0.1em)[— #source]
      ]
    ]
  ]
  v(0.3cm)
}

// Level indicator box
#let level-box(level, title, desc, examples) = {
  set par(first-line-indent: 0pt)
  v(0.0cm)
  block(
    width: 100%,
    stroke: (left: 2.5pt + burgundy, rest: 0.3pt + light-rule),
    radius: (right: 3pt),
    fill: dark-cream,
    inset: (left: 14pt, right: 14pt, top: 10pt, bottom: 10pt),
  )[
    #grid(
      columns: (auto, 1fr),
      gutter: 10pt,
      align(center + horizon)[
        #box(
          width: 28pt,
          height: 28pt,
          radius: 50%,
          fill: navy,
          stroke: 0.5pt + gold,
        )[
          #align(center + horizon)[
            #text(size: 13pt, fill: cream, weight: "bold")[#level]
          ]
        ]
      ],
      [
        #text(size: 12.5pt, fill: navy, weight: "bold")[#title]
        #v(2pt)
        #text(size: 10pt, fill: muted, style: "italic")[#desc]
        #v(2pt)
        #for ex in examples [
          #text(size: 10.0pt, fill: ink)[• #ex] #linebreak()
        ]
      ],
    )
  ]
  v(0.0cm)
}

// Scenario flow block
#let scenario(title, bad-label, bad-flow, good-label, good-flow) = {
  set par(first-line-indent: 0pt)
  v(0.0cm)
  block(
    width: 100%,
    stroke: 0.4pt + light-rule,
    radius: 4pt,
    fill: white,
    inset: 0pt,
  )[
    // Title bar
    #block(
      width: 100%,
      fill: navy,
      radius: (top: 4pt),
      inset: (x: 14pt, y: 8pt),
    )[
      #text(size: 12pt, fill: cream, weight: "bold", tracking: 0.05em)[#title]
    ]
    #v(-0.5cm)
    // Body
    #block(inset: (x: 10pt, y: 10pt))[
      // Bad flow
      #block(
        width: 100%,
        stroke: 0.3pt + rgb("#D4A0A0"),
        radius: 3pt,
        fill: rgb("#FFF8F6"),
        inset: (x: 10pt, y: 8pt),
      )[
        #text(size: 11pt, fill: burgundy, weight: "bold", tracking: 0.15em)[#upper[#bad-label]]
        #v(3pt)
        #text(size: 10pt, fill: ink)[#bad-flow]
      ]
      #v(-4pt)
      // Good flow
      #block(
        width: 100%,
        stroke: 0.3pt + rgb("#A0B8A0"),
        radius: 3pt,
        fill: rgb("#F6FBF6"),
        inset: (x: 10pt, y: 8pt),
      )[
        #text(size: 11pt, fill: rgb("#2D5A2D"), weight: "bold", tracking: 0.15em)[#upper[#good-label]]
        #v(3pt)
        #text(size: 10pt, fill: ink)[#good-flow]
      ]
    ]
  ]
  v(0.0cm)
}

// Principle table row
#let principle-row(name, desc, practice) = {
  (
    [#text(fill: navy, weight: "bold", size: 9.5pt)[#name]],
    [#text(size: 9pt)[#desc]],
    [#text(size: 8.5pt, fill: muted, style: "italic")[#practice]],
  )
}

// Template box
#let template-box(title, body) = {
  set par(first-line-indent: 0pt)
  v(-0.0cm)
  block(
    width: 100%,
    stroke: (left: 2pt + gold, rest: 0.3pt + light-rule),
    radius: (right: 2pt),
    fill: rgb("#FFFDF7"),
    inset: (x: 12pt, y: 8pt),
  )[
    #text(size: 11pt, fill: gold, weight: "bold")[#title]
    #v(2pt)
    #text(size: 10pt, fill: ink)[#body]
  ]
  v(-0.0cm)
}

// Anti-pattern box
#let anti-pattern(number, title, desc, alternative) = {
  set par(first-line-indent: 0pt)
  v(0.2cm)
  block(
    width: 100%,
    inset: (left: 0pt, right: 0pt, y: 0pt),
  )[
    #grid(
      columns: (22pt, 1fr),
      gutter: 8pt,
      align(center)[
        #text(size: 16pt, fill: burgundy, weight: "bold")[#number]
      ],
      [
        #text(size: 11pt, fill: navy, weight: "bold")[#title]
        #v(2pt)
        #text(size: 10pt, fill: ink)[#desc]
        #v(3pt)
        #text(size: 10pt, fill: muted)[💡 ]
        #text(size: 10pt, fill: burgundy, style: "italic")[#alternative]
      ],
    )
  ]
  v(2pt)
  line(length: 100%, stroke: 0.2pt + light-rule)
  v(0.1cm)
}

// Dialogue pattern
#let pattern-block(title, principle, examples) = {
  set par(first-line-indent: 0pt)
  v(0.15cm)
  block(width: 100%)[
    #text(size: 10pt, fill: burgundy, style: "italic")[#principle]
    #v(6pt)
    #block(
      width: 100%,
      fill: dark-cream,
      radius: 3pt,
      inset: (x: 12pt, y: 10pt),
      stroke: 0.3pt + light-rule,
    )[
      #text(size: 10pt, fill: navy, weight: "bold", tracking: 0.1em)[Practical Examples]
      #v(4pt)
      #for ex in examples [
        #text(size: 10pt, fill: ink)[• #ex]
        #v(2pt)
      ]
    ]
  ]
  v(0.15cm)
}


// ═══════════════════════════════════════════════
// TITLE PAGE
// ═══════════════════════════════════════════════

#set page(header: none, footer: none)

#v(3cm)

#align(center)[
  #line(length: 50%, stroke: 0.6pt + gold)
  #v(0.6cm)

  #text(
    size: 10.0pt,
    fill: gold,
    tracking: 0.5em,
    weight: "regular",
  )[#upper[A Practical Guide to]]
  #v(0.4cm)

  #text(
    size: 28pt,
    fill: navy,
    weight: "bold",
  )[AI Collaboration Guidelines]
  #v(0.2cm)

  #line(length: 20%, stroke: 0.5pt + light-rule)
  #v(0.2cm)

  #text(
    size: 14pt,
    fill: burgundy,
    style: "italic",
  )[A Framework for Expanding Knowledge and Solving Problems]

  #v(0.5cm)
  #text(size: 10pt, fill: muted, tracking: 0.2em)[ON THE ART OF INQUIRY\ IN THE AGE OF ARTIFICIAL INTELLIGENCE]

  #v(1cm)
  #line(length: 50%, stroke: 0.6pt + gold)
]

#v(2cm)

#align(center)[
  #block(
    width: 70%,
    inset: (x: 1.5cm, y: 1cm),
  )[
    #set par(first-line-indent: 0pt)
    #text(size: 10pt, fill: ink, style: "italic")[
      "A good tool extends your hands,\ but a good question extends your thinking."
    ]
    #v(0.5cm)
    #text(size: 9pt, fill: muted)[
      This guide addresses principles for leveraging AI\ not merely as an information retrieval tool,\ but as a thinking partner.\ Before prompt techniques, it begins with the fundamental questions:\ "What should I ask?" and "How should I collaborate?"
    ]
  ]
]

#v(1fr)

#align(center)[
  #text(size: 10pt, fill: muted, tracking: 0.15em)[A Practical Guide for Finance, Marketing, and Business Planning Professionals]
  #v(0.3cm)
  #line(length: 30%, stroke: 0.3pt + light-rule)
]

#pagebreak()

// ═══════════════════════════════════════════════
// TABLE OF CONTENTS
// ═══════════════════════════════════════════════

#set page(
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 9pt, font: "New Computer Modern", fill: muted, tracking: 0.12em)
      #smallcaps[AI Collaboration Guidelines]
      #h(1fr)
      #smallcaps[A Framework for Expanding Knowledge and Solving Problems]
      #v(4pt)
      #line(length: 100%, stroke: 0.4pt + light-rule)
    ]
  },
  footer: context {
    let pg = counter(page).get().first()
    if pg > 1 [
      #line(length: 100%, stroke: 0.3pt + light-rule)
      #v(4pt)
      #set text(size: 11pt, font: "New Computer Modern", fill: muted)
      #h(1fr)
      — #pg —
      #h(1fr)
    ]
  },
)

#v(1.0cm)
#align(center)[
  #text(size: 8pt, fill: gold, tracking: 0.4em)[#upper[Contents]]
  #v(0.2cm)
  #text(size: 16pt, fill: navy, weight: "bold")[Table of Contents]
  #v(0.3cm)
  #line(length: 30%, stroke: 0.5pt + gold)
]
#v(0.8cm)

#set par(first-line-indent: 0pt)

#{
  let toc-entry(num, title, target, sub-items) = {
    v(0.1cm)
    block[
      #link(target)[
        #grid(
          columns: (auto, 1fr),
          gutter: 8pt,
          text(size: 10.5pt, fill: navy, weight: "bold")[#num],
          text(size: 10.5pt, fill: navy, weight: "bold")[#title],
        )
      ]
      #v(1pt)
      #for (item-text, item-target) in sub-items [
        #h(24pt)
        #link(item-target)[#text(size: 11pt, fill: muted)[#item-text]]
        #linebreak()
      ]
    ]
    v(0.0cm)
    line(length: 100%, stroke: 0.2pt + light-rule)
  }

  toc-entry([Preface], [Why Prompt Techniques Alone Are Not Enough], <preface>, ())
  toc-entry([I], [The Anatomy of a Good Question], <part1>, (
    ([1.1  The Four Layers of Questions], <sec1-1>),
    ([1.2  Three Conditions for a Good Question], <sec1-2>),
  ))
  toc-entry([II], [Dialogue Patterns for Expanding Thought], <part2>, (
    ([2.1  Reverse Questioning: From Conclusions to Premises], <sec2-1>),
    ([2.2  Boundary Exploration: Thinking Under Extreme Conditions], <sec2-2>),
    ([2.3  Analogical Transfer: Viewing Through Another Domain's Lens], <sec2-3>),
    ([2.4  Temporal Shift: Perspectives from Past and Future], <sec2-4>),
  ))
  toc-entry([III], [Collaboration Strategies for Problem Solving], <part3>, (
    ([3.1  Stepwise Decomposition Dialogue], <sec3-1>),
    ([3.2  Requesting Counterarguments and Steelmanning], <sec3-2>),
    ([3.3  Iterative Refinement], <sec3-3>),
    ([3.4  Accumulating and Managing Context], <sec3-4>),
  ))
  toc-entry([IV], [Anti-Patterns to Avoid], <part4>, ())
  toc-entry([V], [Question Design for Real-World Scenarios], <part5>, (
    ([A. Market Analysis and New Business Planning], <scen-a>),
    ([B. Credit Assessment and Risk Evaluation], <scen-b>),
    ([C. Marketing Campaign Planning], <scen-c>),
    ([D. Executive Report Writing], <scen-d>),
    ([E. Business Process Improvement], <scen-e>),
  ))
  toc-entry([VI], [Summary of Core Principles], <part6>, ())
  toc-entry([Appendix], [Situational Question Templates], <appendix>, ())
}

#pagebreak()


// ═══════════════════════════════════════════════
// PREFACE
// ═══════════════════════════════════════════════

#v(1.2cm)
#align(center)[
  #text(size: 10pt, fill: gold, tracking: 0.4em)[#upper[Preface]]
  #v(0.15cm)
  #text(size: 16pt, fill: navy, weight: "bold")[Preface] <preface>
  #v(0.1cm)
  #text(size: 10pt, fill: muted, style: "italic")[Why Prompt Techniques Alone Are Not Enough]
  #v(0.3cm)
  #line(length: 30%, stroke: 0.5pt + gold)
]
#v(0.6cm)

Most AI training focuses on "how to ask." Assign a role, provide examples, instruct step-by-step thinking. These techniques are useful, but they are merely means to an end.

The real issue lies in *"what to ask"* and *"why you ask it."* Two people using the same AI can get vastly different results: one receives search-engine-level answers while the other achieves breakthroughs in thinking. The difference comes not from prompt craftsmanship but from the #text(fill: navy, weight: "bold")[structure of questions] and the #text(fill: navy, weight: "bold")[attitude toward collaboration].

This guide addresses principles for leveraging AI not as a simple information retrieval tool but as a #text(fill: burgundy, style: "italic")[thinking partner]. It is structured with practical examples immediately applicable in finance, marketing, business planning, and other professional settings.

#ornament()


#pagebreak()

// ═══════════════════════════════════════════════
// PART I
// ═══════════════════════════════════════════════

= The Anatomy of a Good Question <part1>

== The Four Layers of Questions <sec1-1>

Every question belongs to one of the following four levels. As the level rises, the value derived from conversations with AI increases exponentially.

#level-box(
  [1],
  [Fact Retrieval — What],
  [Simple information queries that a search engine could handle. AI is nothing more than a fast search engine.],
  (
    ["What is the COFIX rate?"],
    ["Summarize the DSR regulations."],
    ["What is this year's household loan growth rate?"],
  ),
)

#level-box(
  [2],
  [Structural Understanding — How],
  [Questions that explore mechanisms and causality. They seek the connections between concepts.],
  (
    ["Through what pathways does a change in the COFIX rate affect our deposit product margins?"],
    ["How does cohort analysis work in calculating customer churn rates?"],
    ["What is the mechanism by which cash flow indicators are prioritized over financial ratios in corporate credit assessment?"],
  ),
)

#level-box(
  [3],
  [Judgment and Comparison — Why / Which],
  [Exploring the basis for choices and trade-offs. At this level, AI begins to function as a thinking partner.],
  (
    ["In SME credit assessment, which is more effective for predicting defaults: quantitative metrics or qualitative evaluation?"],
    ["New customer acquisition vs. cross-selling to existing customers — what are the prerequisites and limitations of each?"],
    ["Compare the trade-offs between expanding digital channels and strengthening face-to-face sales."],
  ),
)

#level-box(
  [4],
  [Questioning Premises and Reframing — What if / What's wrong],
  [Questioning and reconstructing the thinking framework itself. This level generates the greatest value.],
  (
    ["What if the assumption 'young customers = digital preference' is fundamentally wrong?"],
    ["Under what scenarios could the causal relationship between customer satisfaction and churn rate be spurious?"],
    ["Will the market premises behind this new business initiative still hold in three years?"],
  ),
)

#small-ornament()

== Three Conditions for a Good Question <sec1-2>

=== First, state the boundaries of your own understanding

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  fill: rgb("#FFF8F6"),
  stroke: 0.3pt + rgb("#D4A0A0"),
  radius: 3pt,
  inset: (x: 12pt, y: 8pt),
)[
  #text(size: 10pt, fill: burgundy, weight: "bold")[Bad example] #h(8pt)
  #text(size: 10pt, fill: ink)["Create a marketing strategy for me."]
]
#v(4pt)
#block(
  width: 100%,
  fill: rgb("#F6FBF6"),
  stroke: 0.3pt + rgb("#A0B8A0"),
  radius: 3pt,
  inset: (x: 12pt, y: 8pt),
)[
  #text(size: 10pt, fill: rgb("#2D5A2D"), weight: "bold")[Good example] #h(8pt)
  #text(size: 10pt, fill: ink)["We are cross-selling credit cards to payroll-transfer customers aged 30\~40, but the conversion rate has plateaued at 2%. I cannot tell whether it is a timing issue or a targeting issue — from what angle should I diagnose this?"]
]
#set par(first-line-indent: 1.2em)

#v(6pt)
The latter is better because the questioner has clearly stated "what they already know and where they are stuck." AI can then explore beyond that boundary.

=== Second, constraints give direction to thinking

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  fill: rgb("#FFF8F6"),
  stroke: 0.3pt + rgb("#D4A0A0"),
  radius: 3pt,
  inset: (x: 12pt, y: 8pt),
)[
  #text(size: 10pt, fill: burgundy, weight: "bold")[Bad example] #h(8pt)
  #text(size: 10pt, fill: ink)["Tell me how to improve profitability."]
]
#v(4pt)
#block(
  width: 100%,
  fill: rgb("#F6FBF6"),
  stroke: 0.3pt + rgb("#A0B8A0"),
  radius: 3pt,
  inset: (x: 12pt, y: 8pt),
)[
  #text(size: 10pt, fill: rgb("#2D5A2D"), weight: "bold")[Good example] #h(8pt)
  #text(size: 10pt, fill: ink)["With an annual marketing budget of \$500K, customer touchpoints limited to app push and SMS, and regulatory restrictions on product comparison language — what is a realistic approach to grow our deposit balances by 15%?"]
]
#set par(first-line-indent: 1.2em)

#v(6pt)
Constraints do not narrow the answer — they reveal the essence of the problem. Paradoxically, the more constraints you specify, the more actionable the answer becomes.

=== Third, leave room for the answer to change your mind

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  fill: rgb("#FFF8F6"),
  stroke: 0.3pt + rgb("#D4A0A0"),
  radius: 3pt,
  inset: (x: 12pt, y: 8pt),
)[
  #text(size: 10pt, fill: burgundy, weight: "bold")[Confirmation bias] #h(8pt)
  #text(size: 10pt, fill: ink)["Verify that this new business plan is sound."]
]
#v(4pt)
#block(
  width: 100%,
  fill: rgb("#F6FBF6"),
  stroke: 0.3pt + rgb("#A0B8A0"),
  radius: 3pt,
  inset: (x: 12pt, y: 8pt),
)[
  #text(size: 10pt, fill: rgb("#2D5A2D"), weight: "bold")[Reframing] #h(8pt)
  #text(size: 10pt, fill: ink)["Identify the three key assumptions in this business plan, and evaluate the impact on the entire venture if each one turns out to be wrong."]
]
#set par(first-line-indent: 1.2em)

#v(6pt)
When a question is open to "under what conditions is this valid?" rather than the binary "right or wrong," the conversation becomes discovery rather than confirmation.

#pagebreak()

// ═══════════════════════════════════════════════
// PART II
// ═══════════════════════════════════════════════

= Dialogue Patterns for Expanding Thought <part2>

== Reverse Questioning <sec2-1>

=== Tracing back from conclusions to premises

Most people ask "How do I do this?" A more productive approach goes in the opposite direction.

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  fill: dark-cream,
  stroke: 0.3pt + light-rule,
  radius: 3pt,
  inset: (x: 14pt, y: 10pt),
)[
  #text(size: 10pt, fill: muted)[Typical flow] #h(12pt)
  #text(size: 10pt, fill: ink)[Premise → Strategy → Execution → Result]
  #v(4pt)
  #text(size: 10pt, fill: burgundy)[Reverse questioning] #h(12pt)
  #text(size: 10pt, fill: ink, style: "italic")["For this outcome to happen, what premises must hold true? Under what conditions would those premises break down?"]
]
#set par(first-line-indent: 1.2em)

#pattern-block(
  [Reverse Questioning],
  [Many practical decisions rest on unverified assumptions. This pattern helps unearth those 'hidden assumptions.'],
  (
    ["What economic assumptions underpin the judgment that this loan portfolio is safe?"],
    ["If this quarter's results were good, how do we distinguish whether it was our strategy or market conditions?"],
    ["Could the KPI used to declare this campaign a success be the wrong metric entirely?"],
  ),
)

== Boundary Exploration <sec2-2>

=== Thinking under extreme conditions

A system's true nature is revealed not under normal conditions but at its extremes. This is also the essence of risk management.

#pattern-block(
  [Boundary Exploration],
  [Exploring extreme conditions reveals structural vulnerabilities and dependencies that remain invisible under normal circumstances.],
  (
    ["If the base rate spikes by 200bp, which segment in our customer portfolio would collapse first?"],
    ["If this product attracts 10x the expected subscribers, where does the operation break first?"],
    ["If a competitor drops fees to zero, which part of our business model survives?"],
    ["If the marketing budget is halved, which single campaign should we keep and why?"],
  ),
)

== Analogical Transfer <sec2-3>

=== Viewing through another domain's lens

Looking at a complex problem only from within the same frame creates blind spots. Transplanting a proven structure from an entirely different industry into your problem is a practical tool.

#pattern-block(
  [Analogical Transfer],
  [Analogy is not decoration — it is a method for discovering structural isomorphism between problems.],
  (
    ["What would it look like to apply Netflix's recommendation engine principles to financial product recommendations?"],
    ["How would an airline mileage lock-in strategy translate to deposit products?"],
    ["Could a hospital's patient triage system be applied to credit assessment prioritization?"],
    ["What insights can subscription economy churn-prevention strategies offer for term-deposit maturity retention?"],
  ),
)

== Temporal Shift <sec2-4>

=== Viewing from past and future vantage points

Being trapped in the present perspective causes you to miss the full context of a problem. Shifting the time axis can remove short-term bias.

#pattern-block(
  [Temporal Shift],
  [Explore what long-term consequences today's decisions might produce, and what lessons analogous historical cases can teach.],
  (
    ["Looking back three years from now, what part of this channel strategy would we regret most?"],
    ["Has this kind of rate movement happened before? How did the industry respond back then?"],
    ["Will this customer behavior pattern still hold in five years, or will it fade with generational change?"],
    ["Once the digital transformation we are pursuing is complete, which investments will seem unnecessary in hindsight?"],
  ),
)

#pagebreak()

// ═══════════════════════════════════════════════
// PART III
// ═══════════════════════════════════════════════

= Collaboration Strategies for Problem Solving <part3>

== Stepwise Decomposition Dialogue <sec3-1>

Do not throw a complex problem at AI all at once. The process of decomposing the problem together with AI is itself the key to the solution.

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  fill: dark-cream,
  stroke: 0.4pt + light-rule,
  radius: 4pt,
  inset: (x: 14pt, y: 12pt),
)[
  #text(size: 11pt, fill: navy, weight: "bold", tracking: 0.1em)[Five-Step Decomposition Framework]
  #v(6pt)
  #grid(
    columns: (24pt, 1fr),
    gutter: 6pt,
    text(size: 14pt, fill: gold, weight: "bold")[1], text(size: 10pt, fill: ink)[#strong[Define the Problem] — "What exactly am I trying to solve?"],
    text(size: 14pt, fill: gold, weight: "bold")[2], text(size: 10pt, fill: ink)[#strong[Identify Constraints] — "What conditions limit the solution space?"],
    text(size: 14pt, fill: gold, weight: "bold")[3], text(size: 10pt, fill: ink)[#strong[Decompose] — "What independently solvable sub-problems can I identify?"],
    text(size: 14pt, fill: gold, weight: "bold")[4], text(size: 10pt, fill: ink)[#strong[Explore Each] — Deep conversation for each sub-problem],
    text(size: 14pt, fill: gold, weight: "bold")[5], text(size: 10pt, fill: ink)[#strong[Integrate and Verify] — "When I combine the partial solutions, is the whole really solved?"],
  )
]
#set par(first-line-indent: 1.2em)

#v(0.3cm)

#text(size: 11pt, fill: navy, weight: "bold")[Practical Example — Corporate Customer Retention:]

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  stroke: (left: 2pt + burgundy),
  inset: (left: 12pt, y: 4pt),
)[
  #set text(size: 10pt)
  #strong[Step 1] What exactly does "corporate customer churn" mean? Transaction cessation? Balance decline? Primary banking relationship shift? \
  #strong[Step 2] Constraints? Limited sales force, difficult access to decision-makers, pricing competition limits \
  #strong[Step 3] Sub-problems: Churn signal detection / High-risk customer prioritization / Retention offer design / Effectiveness measurement \
  #strong[Step 4] Explore each sub-problem in a separate conversation turn with depth \
  #strong[Step 5] When integrating, verify whether the field sales team can actually execute the process
]
#set par(first-line-indent: 1.2em)

#v(5em)
== Requesting Counterarguments and Steelmanning <sec3-2>

Seeking "agreement" from AI about your own idea is the most common mistake.

#epigraph(
  ["Construct the argument of the smartest person who would say this cannot work."],
)

The key is requesting a *steelman* — the strongest possible version of the opposing argument. The question above elicits far sharper feedback than simply asking "Why won't this work?"

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  fill: dark-cream,
  radius: 3pt,
  stroke: 0.3pt + light-rule,
  inset: (x: 12pt, y: 10pt),
)[
  #text(size: 11pt, fill: navy, weight: "bold", tracking: 0.1em)[Usage Examples]
  #v(4pt)
  #text(size: 10pt, fill: ink)[
    • "Build the strongest counterargument against this business plan." \
    • "What would be the most rational objection a CFO would raise against this budget allocation?" \
    • "I am trying to sell this investment — what are the three sharpest questions the opposition would ask?"
  ]
]
#set par(first-line-indent: 1.2em)

#v(4pt)

This technique is also highly effective for simulating the anticipated questions of senior executives before drafting a report.

#v(2em)
== Iterative Refinement <sec3-3>

Do not expect a perfect result from a single question. A conversation is an exploration process, not a vending machine.

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  fill: dark-cream,
  stroke: 0.3pt + light-rule,
  radius: 3pt,
  inset: (x: 12pt, y: 10pt),
)[
  #text(size: 11pt, fill: navy, weight: "bold", tracking: 0.1em)[Types of Follow-Up Questions at Each Turn]
  #v(6pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    text(size: 10pt, fill: ink)[• "Elaborate on this part." #text(fill: muted)[ — Depth expansion]],
    text(size: 10pt, fill: ink)[• "What about a different angle?" #text(fill: muted)[ — Breadth expansion]],
    text(size: 10pt, fill: ink)[• "Would this actually work in practice?" #text(fill: muted)[ — Reality check]],
    text(size: 10pt, fill: ink)[• "Is there a logical leap here?" #text(fill: muted)[ — Logic check]],
    text(size: 10pt, fill: ink)[• "Explain it to an exec in one sentence?" #text(fill: muted)[ — Essence extraction]],
    [],
  )
]
#set par(first-line-indent: 1.2em)

#v(6pt)

#text(size: 11pt, fill: navy, weight: "bold")[Practical Example — New Product Design (5-Turn Conversation):]

#set par(first-line-indent: 0pt)
#block(
  width: 100%,
  stroke: (left: 2pt + gold),
  inset: (left: 12pt, y: 4pt),
)[
  #set text(size: 11pt)
  #text(fill: gold, weight: "bold")[Turn 1] "I want to design a savings + insurance bundled product for dual-income couples in their 30s — help me set the direction" \
  #text(fill: gold, weight: "bold")[Turn 2] "What tangible benefit does the customer actually feel? The differentiation versus competing products seems weak" \
  #text(fill: gold, weight: "bold")[Turn 3] "From a regulatory perspective, what are the pitfalls and prohibited structures for bundled sales?" \
  #text(fill: gold, weight: "bold")[Turn 4] "What alternative structures maximize customer benefit without circumventing regulations?" \
  #text(fill: gold, weight: "bold")[Turn 5] "If I structure this as an internal proposal, what should the outline look like?"
]
#set par(first-line-indent: 1.2em)

#v(2em)
== Accumulating and Managing Context <sec3-4>

Effective collaborators build systems for accumulating context in their AI conversations.

#set par(first-line-indent: 0pt)
#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  block(
    fill: dark-cream,
    stroke: 0.3pt + light-rule,
    radius: 3pt,
    inset: (x: 10pt, y: 8pt),
    width: 100%,
  )[
    #text(size: 11pt, fill: navy, weight: "bold")[Leverage Memory Features]
    #v(3pt)
    #text(size: 10pt, fill: ink)[Set your role, domain, and tech stack as baseline context in the AI. No need to repeat your introduction every time.]
  ],
  block(
    fill: dark-cream,
    stroke: 0.3pt + light-rule,
    radius: 3pt,
    inset: (x: 10pt, y: 8pt),
    width: 100%,
  )[
    #text(size: 11pt, fill: navy, weight: "bold")[Separate by Project]
    #v(3pt)
    #text(size: 10pt, fill: ink)[Keep separate conversations for credit, marketing, regulatory review, and other topics. This increases focus.]
  ],
  block(
    fill: dark-cream,
    stroke: 0.3pt + light-rule,
    radius: 3pt,
    inset: (x: 10pt, y: 8pt),
    width: 100%,
  )[
    #text(size: 11pt, fill: navy, weight: "bold")[State Intermediate Conclusions]
    #v(3pt)
    #text(size: 10pt, fill: ink)[Every 3\~5 turns, declare "To summarize so far..." This simultaneously organizes both the AI's and your own thinking.]
  ],
  block(
    fill: dark-cream,
    stroke: 0.3pt + light-rule,
    radius: 3pt,
    inset: (x: 10pt, y: 8pt),
    width: 100%,
  )[
    #text(size: 11pt, fill: navy, weight: "bold")[Explicitly Update Premises]
    #v(3pt)
    #text(size: 10pt, fill: ink)[Clearly state premise changes: "Earlier I assumed rates would hold steady, but let's switch to a rate-hike scenario."]
  ],
)
#set par(first-line-indent: 1.2em)

#pagebreak()

// ═══════════════════════════════════════════════
// PART IV
// ═══════════════════════════════════════════════

= Anti-Patterns to Avoid <part4>

#anti-pattern(
  [①],
  [Reinforcing Confirmation Bias],
  [Asking "This strategy is right, isn't it?" turns AI into an agreement machine. Since AI inherently tends to align with the user's direction, you must consciously counteract this.],
  ["List at least three possible causes for this quarter's underperformance and compare their explanatory power."],
)

#anti-pattern(
  [②],
  [Excessive Delegation],
  ["Just write the report for me" fails to leverage AI's strengths. AI produces its best work when given clear constraints and evaluation criteria. Setting direction is the human's role.],
  ["The audience is five Risk Management Committee members; the focus is construction-sector loan defaults. Structure it for this context."],
)

#anti-pattern(
  [③],
  [Expecting Perfection in One Shot],
  [Expecting a complete "marketing strategy" from a single prompt yields textbook-level, superficial answers. The more complex the problem, the more effective multi-turn conversations become.],
  [Plan for a minimum of 3\~5 turns of iterative dialogue before you begin.],
)

#anti-pattern(
  [④],
  [Uncritical Acceptance],
  [Accepting AI output at face value is the most dangerous pattern. In finance especially, "plausible errors" can be fatal. Numbers, regulatory citations, and market trends must always be cross-checked against primary sources.],
  ["Why did you reach this conclusion?", "What is the source of this figure?", "Is there a missing premise in this logic?"],
)

#anti-pattern(
  [⑤],
  [Confusing the Relationship with the Tool],
  [AI is a capable thinking tool, not an authoritative judge. "AI analyzed it this way" cannot serve as evidence in a report. AI output is a 'hypothesis to be examined.'],
  [Final judgment and responsibility always rest with the human.],
)

#pagebreak()

// ═══════════════════════════════════════════════
// PART V
// ═══════════════════════════════════════════════

= Question Design for Real-World Scenarios <part5>

.<scen-a>
 #scenario(
  [A. Market Analysis and New Business Planning],
  [Superficial Approach],
  ["Tell me about Gen-Z finance trends" → Accept generalizations → Paste into report],
  [Thought-Expanding Approach],
  [
    ① "What is _structurally_ different about Gen-Z financial behavior compared to older generations?" #text(fill: muted, size: 10pt)[ — Structural analysis] \
    ② "What specific implications does that difference have for our deposit product design?" #text(fill: muted, size: 10pt)[ — Contextual application] \
    ③ "How do we determine whether this trend is a temporary fad or a structural shift?" #text(fill: muted, size: 10pt)[ — Premise verification] \
    ④ "If it is a structural shift, what should our product lineup look like in three years?" #text(fill: muted, size: 10pt)[ — Temporal shift] \
    ⑤ "If a competitor is betting in the exact opposite direction, what would their rationale be?" #text(fill: muted, size: 10pt)[ — Steelman]
  ],
)

<scen-b>
#scenario(
  [B. Credit Assessment and Risk Evaluation],
  [Superficial Approach],
  ["Analyze this company's financial statements" → Accept ratio analysis → Draft assessment report],
  [Thought-Expanding Approach],
  [
    ① "What hidden risk signals might lurk behind this company's financial numbers?" #text(fill: muted, size: 10pt)[ — Hidden assumptions] \
    ② "What common patterns appear among defaulted companies of similar industry and size?" #text(fill: muted, size: 10pt)[ — Pattern comparison] \
    ③ "Under which macroeconomic scenario would this company become most vulnerable?" #text(fill: muted, size: 10pt)[ — Boundary exploration] \
    ④ "Build three arguments that my judgment 'this company is safe' is wrong" #text(fill: muted, size: 10pt)[ — Counter confirmation bias] \
    ⑤ "What would credit terms look like if we incorporated those risks?" #text(fill: muted, size: 10pt)[ — Action link]
  ],
)

#pagebreak()
<scen-c>
#scenario(
  [C. Marketing Campaign Planning],
  [Superficial Approach],
  ["Give me card promotion ideas" → Accept ideas → Execute],
  [Thought-Expanding Approach],
  [
    ① "What were the success factors behind the highest-converting campaign in the past three months?" #text(fill: muted, size: 10pt)[ — Historical basis] \
    ② "Will those success factors hold this time? What has changed in the market?" #text(fill: muted, size: 10pt)[ — Premise validity] \
    ③ "With the same budget, what entirely different approach would be possible?" #text(fill: muted, size: 10pt)[ — Alternative exploration] \
    ④ "Name three scenarios where this campaign fails, and the early warning signs for each" #text(fill: muted, size: 10pt)[ — Risk assessment] \
    ⑤ "Also examine whether we might have chosen the wrong KPI to judge success or failure" #text(fill: muted, size: 10pt)[ — Measurement frame]
  ],
)

#text[ ] <scen-d>
#scenario(
  [D. Executive Report Writing],
  [Superficial Approach],
  ["Write a performance report" → Submit output],
  [Thought-Expanding Approach],
  [
    ① "What is the first question an executive reading this report would ask?" #text(fill: muted, size: 10pt)[ — Reader's perspective] \
    ② "If our answer to that question is weak, what data should we supplement?" #text(fill: muted, size: 10pt)[ — Evidence strengthening] \
    ③ "Are there any 'looks good on the surface but is actually a problem' metrics in this report?" #text(fill: muted, size: 10pt)[ — Beyond the surface] \
    ④ "Is there room to interpret the conclusion in the opposite way? How would we respond to that counterargument?" #text(fill: muted, size: 10pt)[ — Counterargument prep] \
    ⑤ "Compress the core message into one sentence?" #text(fill: muted, size: 10pt)[ — Essence extraction]
  ],
)

#text[ ] <scen-e>
#scenario(
  [E. Business Process Improvement],
  [Superficial Approach],
  ["Tell me how to improve work efficiency" → Accept generalizations],
  [Thought-Expanding Approach],
  [
    ① "Among our team's weekly recurring tasks, which has the lowest value relative to time spent?" #text(fill: muted, size: 10pt)[ — Current-state diagnosis] \
    ② "Is the original purpose of this task still valid today?" #text(fill: muted, size: 10pt)[ — Raison d'etre check] \
    ③ "What side effects could arise if we eliminated it?" #text(fill: muted, size: 10pt)[ — Elimination cost] \
    ④ "How could we achieve the same purpose while cutting the time to one-third?" #text(fill: muted, size: 10pt)[ — Constrained optimization] \
    ⑤ "How do other industries handle a similar process?" #text(fill: muted, size: 10pt)[ — Analogical transfer]
  ],
)

#pagebreak()

// ═══════════════════════════════════════════════
// PART VI
// ═══════════════════════════════════════════════

= Summary of Core Principles <part6>

#v(0.3cm)

#set par(first-line-indent: 0pt)

#{
  let header-cell(body) = table.cell(
    fill: navy,
    inset: (x: 10pt, y: 8pt),
  )[#align(center)[#text(size: 11pt, fill: cream, weight: "bold", tracking: 0.05em)[#body]]]

  let body-cell(body) = table.cell(
    inset: (x: 10pt, y: 8pt),
  )[#text(size: 10pt, fill: ink)[#body]]

  let alt-cell(body) = table.cell(
    fill: dark-cream,
    inset: (x: 10pt, y: 8pt),
  )[#text(size: 10pt, fill: ink)[#body]]

  table(
    columns: (0.46fr, 1.01fr, 1fr),
    stroke: 0.4pt + light-rule,
    align: center+horizon,
    header-cell[Principle], header-cell[Description], header-cell[Practical Application],

    body-cell[#text(fill: navy, weight: "bold")[Awareness of\ Ignorance]],
    body-cell[Knowing what you don't know is the starting point],
    body-cell[#text(style: "italic", fill: muted)[Start with "The part I'm uncertain about is..."]],

    alt-cell[#text(fill: navy, weight: "bold")[Questioning\ Premises]],
    alt-cell[Examine the premise of the question, not just the answer],
    alt-cell[#text(style: "italic", fill: muted)["Examine the possibility that this\ analysis's premise is wrong"]],

    body-cell[#text(fill: navy, weight: "bold")[The Paradox\ of Constraints]],
    body-cell[The more constraints,\ the more actionable the answer],
    body-cell[#text(style: "italic", fill: muted)[Specify real-world conditions:\ budget, staffing, regulations, timeline]],

    alt-cell[#text(fill: navy, weight: "bold")[Iterative\ Refinement]],
    alt-cell[Converge through repeated dialogue],
    alt-cell[#text(style: "italic", fill: muted)[Plan for a minimum of 3~5 turns\ before starting]],

    body-cell[#text(fill: navy, weight: "bold")[Critical\ Acceptance]],
    body-cell[AI output is a hypothesis, not a conclusion],
    body-cell[#text(style: "italic", fill: muted)[Always cross-check numbers and\ regulations against primary sources]],

    alt-cell[#text(fill: navy, weight: "bold")[Accumulating\ Context]],
    alt-cell[The value of continuous conversation\ exceeds the sum of one-off questions],
    alt-cell[#text(style: "italic", fill: muted)[Use memory, project separation,\ and mid-session summaries]],

    body-cell[#text(fill: navy, weight: "bold")[Openness\ to Change]],
    body-cell[A good question is one where the\ questioner is ready to be changed],
    body-cell[#text(style: "italic", fill: muted)[Start with the premise\ "I might be wrong"]],
  )
}

#set par(first-line-indent: 1.2em)

#pagebreak()
#ornament()

// ═══════════════════════════════════════════════
// APPENDIX
// ═══════════════════════════════════════════════

#v(0.0cm)
#text[ ] <appendix>
#align(center)[
  #text(size: 8pt, fill: gold, tracking: 0.4em)[#upper[Appendix]]
  #v(0.15cm)
  #text(size: 16pt, fill: navy, weight: "bold")[Appendix: Situational Question Templates]
  #v(0.15cm)
  #text(size: 9pt, fill: muted, style: "italic")[Replace the content in [ ] brackets with your own situation.]
  #v(0.3cm)
  #line(length: 30%, stroke: 0.5pt + gold)
]
#v(0.4cm)

#template-box(
  [Unearthing Hidden Premises],
  [What are three assumptions that must hold true for [current project/strategy] to succeed? And what is the probability of each assumption breaking down?],
)

#template-box(
  [Counterargument Simulation],
  [I am about to pursue [decision/proposal]. Construct the most rational opposing argument against it.],
)

#template-box(
  [Extreme Scenario],
  [Build a scenario in which [current system/strategy] fails completely. What combination of conditions would it take?],
)

#template-box(
  [Analogical Transfer],
  [Is there a case in another industry that is structurally similar to [current problem]? How did they solve it?],
)

#template-box(
  [Temporal Shift],
  [Looking back [N] years from now, what would be the biggest regret about [current decision]?],
)

#template-box(
  [Essence Extraction],
  [If I had to convey everything we have discussed to a decision-maker in a single sentence, what would it be?],
)

#template-box(
  [Measurement Frame Review],
  [We are measuring performance by [this metric] — under what circumstances could this metric diverge from the actual goal?],
)

#pagebreak()
#ornament()

// ═══════════════════════════════════════════════
// CLOSING
// ═══════════════════════════════════════════════

#v(0.0cm)
#align(center)[
  #text(size: 8pt, fill: gold, tracking: 0.4em)[#upper[Epilogue]]
  #v(0.15cm)
  #text(size: 14pt, fill: navy, weight: "bold")[Closing Remarks]
  #v(0.15cm)
  #text(size: 10pt, fill: burgundy, style: "italic")[The Role of the Questioning Practitioner]
  #v(0.3cm)
  #line(length: 30%, stroke: 0.5pt + gold)
]
#v(0.5cm)

No matter how advanced AI becomes, there are things it cannot replace: #text(fill: navy, weight: "bold")[defining what constitutes an important problem], #text(fill: navy, weight: "bold")[judging whether an answer works in our real-world context], and #text(fill: navy, weight: "bold")[bearing responsibility for the final decision].

A good question is not a technique — it is an attitude. The most productive AI collaboration is not "a conversation where I get what I already know organized" but "a conversation where I discover what I did not know."\

And that discovery begins with the honest acknowledgment: "I do not yet know enough about this problem."

Tools improve every day. But the ability to ask good questions is something no tool can do for you. That ability is, in the age of AI, the most powerful competitive advantage a practitioner can possess.
