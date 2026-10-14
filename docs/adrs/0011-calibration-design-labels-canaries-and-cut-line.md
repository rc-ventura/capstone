# ADR-011: Calibration design — what "correct" means, where the labeled answers come from, how the canaries are built, how the cut line is chosen

**Status**: Accepted (D-16 to D-19). **Addendum 2026-10-12** records observations and predictions (it changes no decision). **Amendment 2026-10-12** (D-20 to D-23) closes the open items of the addendum and **replaces one canary type of D-18**; with it, the P9 decisions are closed.
**Date**: 2026-10-12
**Related**: [Roadmap §5 decisions D-16 to D-19](../roadmap/ckpt-0.6.1b-metrics-roadmap.md) · [ADR-008](./0008-answer-correctness-signals-and-calibration.md) (the signals, C1 and C2, whose calibration this record designs) · [ADR-010](./0010-text-normalization-for-lexical-correctness-signals.md) (the cleaning rule) · [ADR-007](./0007-abstention-detection-and-reporting.md) (the labeling session is shared) · Reading notes: [Adlakha 2024](../research/fichamentos/adlakha-2024-correctness-faithfulness-iqa.md) · [Magesh 2025](../research/fichamentos/magesh-2025-hallucination-free.md) · [CheckList (Ribeiro 2020)](../research/fichamentos/ribeiro-2020-checklist-behavioral-testing.md) · [Cao 2025](../research/fichamentos/cao-2025-negation-blindness-embeddings.md) · [Ho 2025 and Wang 2023](../research/fichamentos/qa-eval-judge-vs-f1.md)
**Code**: none yet. This record fixes the **design of the calibration** (it runs once, after EXP-0) before any real answer exists. **Evidence (reproducible)**: [`evidence/0011-label-supply-simulation.py`](./evidence/0011-label-supply-simulation.py) (+ [`.out.txt`](./evidence/0011-label-supply-simulation.out.txt)) and, for the addendum, [`evidence/0011-observations-after-decision.py`](./evidence/0011-observations-after-decision.py) (+ [`.out.txt`](./evidence/0011-observations-after-decision.out.txt)).

---

> **This record is written for any reader, not only engineers.** The short version: ADR-008 says that, after the first real run (EXP-0), the author will label 100–150 real answers as right or wrong and check whether our cheap scoring rules ("rulers": token-F1, word Recall, embedding cosine) agree with those labels (test **C1**) and whether they notice an answer that says the opposite (test **C2**). That plan left four practical questions open: (1) what exactly counts as "right"? (2) where do enough *wrong* answers come from, since a good system produces few? (3) how are the "opposite" test answers (called *canaries*) made? (4) how is the score line below which a ruler calls an answer wrong chosen? Each is answered below, with the evidence and the arithmetic that supports it. They are fixed now, **before** seeing any data, for the same reason as the pass marks of ADR-008: a rule adjusted after the labels exist would be tuned to the result.

## Context

The calibration compares two things for each real answer: **the author's label** (right = 1, wrong = 0) and **the ruler's score**. ADR-008 §5 fixed the pass marks: **C1** Spearman ≥ 0.60 between score and label (*Spearman*: +1 when every right answer scores above every wrong one, 0 when the scores say nothing about the labels); **C2** ≥ 80% of 10–20 *canaries* (right answers turned into their opposite, so known to be wrong) fall below the *cut line* (the score under which a ruler declares an answer wrong). ADR-008 §4 also says "label 100–150 real answers, only where the gold paper was retrieved".

Four things were left undefined, and each can silently change what the pass marks mean:

1. **The labeling rule.** A multi-document question has a 54-word reference with several facts. If the answer is correct about paper A and silent about paper B, is it right or wrong?
2. **The supply of wrong answers.** C1 needs wrong answers to separate from right ones, and the labeled set may contain almost none (shown below).
3. **The construction of the canaries.** "Turn an answer into its opposite" can be done in many ways; testing only one would measure only that one.
4. **The cut line.** C2 depends on it, and a line set too high would let C2 pass without the ruler being any good.

## Decision

**D-16 — Labeling rule.** Label an answer **right (1)** if it does **not contradict** the reference and **answers the main point** of the question; otherwise **wrong (0)**. An answer that covers only part of a multi-part question is still right if what it says is correct; the author marks the flag **`partial`**. The existing flag **`contradicts_reference`** (ADR-008 §4) stays. Arxiv ids and citations are ignored (ADR-010; citation correctness is `citation_accuracy`'s job); extra, unrequested content is ignored here (faithfulness has its own judge). Missing parts are the `completeness_judge`'s dimension, not this one.

**D-17 — Where the labeled answers come from.** Labeling is done in this order, always **blind to the rulers' scores**:

1. Label the real answers of the two EXP-0 rounds, only where the gold paper was retrieved (about 59 questions × 2 rounds = 118 answers).
2. If the wrong answers are **fewer than 30% of the labeled answers**, add answers to the **same questions** from a deliberately **worse version of the system** (the gold paper still in the context, so a failure is the generator's), until the wrong answers reach 30% **or the labeled set reaches 200 answers**, whichever comes first.
3. Every added answer is tagged `source=worse_version`, so any result can be reported with and without them.
4. If the cap is reached below 30%, C1 is reported as measured and ADR-008 §6, path 2 applies (the LLM judge runs every round; the pass marks may be reopened with the evidence).

Which model or prompt makes the worse version is **not decided here**; it must keep the gold paper in context and fail often enough that the cap is reachable (see "How the simulation works", Part 4). The numbers 30% and 200 are **engineering choices** from the simulation and the labeling effort, not from a paper. *(2026-10-12: confirmed by the author; the type of the worse version is fixed by D-20 and the treatment of refusals by D-21.)*

**D-18 — Canaries.** Build **20 canaries = 4 types × 5**, each made from an answer the author labeled **right**:

| Type | Original | Canary |
|---|---|---|
| Negation | "the method improves precision" | "the method does **not** improve precision" |
| Number swap | "reduces cost by 30%" | "reduces cost by **3%**" |
| Direction reversed | "increases success" | "**decreases** success" |
| Paper swap | "paper A proposes a model" | "paper **B** proposes a model" |

C2 stays exactly as ADR-008 wrote it (**≥ 16 of 20** below the cut line). Reported next to it, as **diagnostics only**: the rate per type, and a paired check ("is the canary's score lower than the score of the answer it came from?"). The canaries are **not** part of the labeled set used for C1. A draft is prepared and the author reviews each one. *(2026-10-12: the fourth type, "paper swap", is replaced by a **key-term swap** — see D-22 in the Amendment.)*

**D-19 — Cut line.** For each ruler, the cut line is the score that maximizes **(wrong answers caught) + (right answers kept) − 1** on the author's labels, where "caught" means the score is **below** the line and "kept" means at or above it (Youden's index). **Canaries never take part in choosing it.** Ties go to the **lowest** line (it rejects fewer right answers). The line is frozen before the canaries are scored, and **every C2 result is reported with the share of right answers that fall below the line**.

Worked example (invented numbers, for illustration only): 10 right answers scoring 0.97 0.95 0.93 0.91 0.90 0.88 0.86 0.84 0.82 0.74 and 10 wrong answers scoring 0.86 0.82 0.78 0.75 0.72 0.70 0.66 0.62 0.58 0.50.

| Line | Wrong caught | Right kept | Score |
|---|---|---|---|
| 0.70 | 4 of 10 | 10 of 10 | 0.4 + 1.0 − 1 = **0.4** |
| **0.80** | 8 of 10 | 9 of 10 | 0.8 + 0.9 − 1 = **0.7** |
| 0.90 | 10 of 10 | 5 of 10 | 1.0 + 0.5 − 1 = **0.5** |
| 0.99 | 10 of 10 | 0 of 10 | 1.0 + 0 − 1 = **0** |

The "− 1" makes a useless line score 0: a line so high that everything falls below catches every wrong answer but keeps no right one (1 + 0 − 1 = 0), and a line so low that nothing falls below does the opposite (0 + 1 − 1 = 0). That is why raising the line is not rewarded.

## Evidence

> Convention: **(F)** = measured or stated by a source; **(I)** = my interpretation or arithmetic.

### D-16 — the labeling rule

- **(F) Adlakha 2024** (read in full): the annotator's task "is to assess if the model response is correct, i.e., it is factually accurate and satisfies the information need underlying the user's query". Two annotators per answer, a third for ties; agreement 92.42%, Fleiss' κ 76.4%; of 1,200 answers 961 were judged correct and 239 incorrect. **Limit:** their references have 2–11 words, so the partially-answered case almost never arose; ours average 54 words with several facts.
- **(F) Magesh 2025** classifies every answer into exactly one of *accurate*, *incomplete* or *hallucinated*: "incomplete" is a separate dimension from "wrong".
- **(F) Our plan** already has the `completeness_judge` ("were all the points of the gold synthesis present?", `ckpt-0.6-plan.md` §3.2).
- **(I)** Counting a missing part as "wrong" here would count it twice. The `partial` flag keeps the information, so a stricter reading can be applied later without relabeling.

### D-17 — where the labeled answers come from

- **(I) The number of answers.** Eligible questions: 40 `answerable` + 10 `persona` + 9 `multi-doc` known-item where the gold paper was retrieved (baseline hit@5 = 0.90 on the 10, roadmap P5; the 5 open-topic questions have no gold paper until the mini-pooling) = **59**. EXP-0 runs twice (`docs/plan.md` CKPT-0), so **118** answers; each question appears twice, so they are **not 118 independent cases**.
- **(F) Adlakha 2024** had 20% wrong answers (239 of 1,200) because it **pooled four models of different quality**. A single strong generator with the gold paper present will probably err less, which is why the supply is a design question.
- **(I) The simulation below** shows what share of wrong answers C1 needs.

#### How the simulation works and what it shows

**The question.** If a ruler separates right answers from wrong ones with a given quality, and the labeled set has a given share of wrong answers, how often does it reach C1 (Spearman ≥ 0.60)?

**The model**, kept simple so each step can be checked by hand:
1. Each answer is right (1) or wrong (0); *p* is the share of wrong ones and *n* the number of labeled answers.
2. A ruler gives a score to each answer. The scores of right answers are drawn from a bell curve centered at 0; those of wrong answers from the same curve shifted lower by *d*.
3. *d* is set from the ruler's quality, measured as the probability that a randomly chosen right answer scores above a randomly chosen wrong one (the AUC). Three qualities: **0.80 (decent)**, **0.90 (good: a right-versus-wrong pair is ordered correctly 9 times in 10)** and **0.95 (very good)**. For two unit-width bell curves, AUC = Φ(*d*/√2), so *d* = √2 · Φ⁻¹(AUC).
4. Spearman is computed between score and label, exactly as Adlakha did, and the experiment is repeated **4,000 times** for each cell (fixed seed). The table reports the share of repetitions with Spearman ≥ 0.60.

**Part 2 — the ceiling.** If the ruler were *perfect*, Spearman would still be limited by *p*: the ranks are uniform, the label is 1 above a rank and 0 below, which gives Cov(rank, label) = *p*(1−*p*)/2 and so **ρ = √(3·*p*·(1−*p*))**. The script checks the formula against a simulation with 1,000 answers and they match to three decimals (0.377, 0.520, 0.693, 0.794, 0.849, 0.866 for *p* = 5%, 10%, 20%, 30%, 40%, 50%). With 10% wrong the best possible Spearman is **0.52**, below the 0.60 required: C1 is unreachable *whatever the ruler*, simply for lack of wrong answers.

**Part 3 — the chance of passing C1** (share of 4,000 repetitions with ρ ≥ 0.60):

| Share of wrong answers | n = 118, AUC 0.80 | n = 118, AUC 0.90 | n = 118, AUC 0.95 | n = 200, AUC 0.90 |
|---|---|---|---|---|
| 10% | 0.00 | 0.00 | 0.00 | 0.00 |
| 20% | 0.00 | 0.19 | 0.82 | 0.10 |
| 25% | 0.01 | 0.55 | 0.98 | 0.51 |
| **30%** | 0.03 | **0.78** | 1.00 | 0.83 |
| 40% | 0.09 | 0.94 | 1.00 | 0.98 |

**How to read it.** (a) What matters is the **share** of wrong answers, not their count: the same share gives nearly the same chance whether *n* is 118 or 250 (a first draft of this analysis used a count of 35 wrong answers, which equals 30% only when *n* = 118; it was replaced by a share for that reason). (b) A **good** ruler (0.90) passes C1 about 0.8 of the time at 30% wrong, about half the time at 25%, and about one time in five at 20%. (c) A merely **decent** ruler (0.80) essentially never passes at any share up to 40%, which is C1 doing its job. (d) That is why 30% is the target.

**Part 4 — the effort.** If the real answers have *k* wrong ones and the worse version errs a share *e* of the time, adding *m* answers gives (*k* + *e*·*m*) / (118 + *m*) = 0.30, so *m* = (0.30·118 − *k*) / (*e* − 0.30). Total answers the author labels:

| Real wrong answers *k* | worse version wrong 40% | 50% | 70% |
|---|---|---|---|
| 6 | 412 | 265 | 192 |
| 12 | 352 | 235 | 176 |
| 24 | 232 | 175 | 146 |

With the cap of 200 (so at most 82 extra answers), the target is reachable only if the worse version errs about 60% of the time or more when the real system errs about 10% (k = 12), and about 45% or more when the real system errs about 20% (k = 24). This is the price of the target and is accepted explicitly.

**Limits of the simulation.** Bell-curve scores and independent answers; real scores are not normal and the answers come in pairs, so read the numbers as an order of magnitude, not as a prediction.

### D-18 — the canaries

- **(F) CheckList (Ribeiro 2020)**, read in full except the second half of §4. Its *directional expectation* tests perturb an input so that the output must move in a known direction (here: the correctness score must go down) and need no new human label. It reports a failure rate **per test**, with no pass mark.
- **(F)** Fine-tuned BERT and RoBERTa, 91.5% and 93.9% accurate on the SST-2 subset that contains negation, "fail simple negation MFTs"; commercial sentiment models fail "near 100%" when the negation comes at the end of the sentence or far from the sentiment word, much less on "The food is not poor.".
- **(F)** "A more diverse set of inputs is particularly helpful when a small set of test cases could miss a failure, e.g. if a model works for some forms of negation but not others."
- **(F) Cao 2025 and our own test** (ADR-008): embedding cosine gave a negated answer 0.94, above every correct paraphrase (0.59–0.84).
- **(I)** A ruler that catches "not" but misses a swapped number would pass a single-type test and still be unsafe. With 5 canaries per type one canary moves a type by 20 points, so the per-type figures are diagnostics, not pass marks. The 80% of C2 is still the author's choice, not a source's.

### D-19 — the cut line

- **(F) Ho 2025** (fichamento [A]): binarizing F1 at different thresholds moves the mean correlation with humans from 0.451 (threshold 0.3) to 0.237 (0.8). **Wang 2023** ([B]): BERTScore is "highly sensitive to threshold settings". The cut line is not a detail: it changes the result.
- **(F) Cao 2025**: the cosine scale is arbitrary and specific to the embedding model, so a fixed number shared by all rulers has no meaning.
- **(I)** Choosing the line with the labels only, and testing it on independent canaries, avoids tuning C2 to itself. Youden's index is a standard criterion and the worked example above shows why raising the line is not rewarded; **no paper I read supports this particular rule**. Fluss, Faraggi & Reiser 2005 study how to estimate this cutoff, but the full text is behind a paywall and I did not read it; from the abstract the plain empirical estimate is noisy in small samples, and ours will be small.

## Alternatives considered

### D-16 alternatives

- **Strict ("right only if it covers everything").** Matches the "satisfies the information need" wording more literally, but counts missing parts twice (with the `completeness_judge`) and yields more wrong answers. **Why not chosen:** double counting.
- **Three levels (right, partial, wrong).** More information, but C1 and the reference study use binary labels. **Why not chosen:** the `partial` flag keeps the information without changing the metric.

### D-17 alternatives

- **Label only the real EXP-0 answers.** Represents real use. **Why not chosen:** probably few wrong answers, so C1 fails for lack of them (Part 2–3).
- **Always add worse-version answers.** Guarantees wrong answers. **Why not chosen:** needless labeling when the real system already errs enough, and more distance from real use.
- **The extended set (about 200 questions).** More questions. **Why not chosen:** it does not exist yet and the plan only uses it at CKPT-7.

### D-18 alternatives

- **20 canaries of one type ("not").** Simple. **Why not chosen:** measures one form of contradiction only (CheckList).

### D-19 alternatives

- **A fixed number for all rulers (e.g. 0.5).** Simple. **Why not chosen:** the scales differ and the choice moves results a lot (Ho, Wang, Cao).
- **A line chosen so that 80% of the canaries pass.** Would make C2 pass by construction. **Not allowed.**

## Consequences

### Accepted

- The labeling rule, the supply of wrong answers, the canaries and the cut line are fixed **before** any real answer exists.
- C1 can be reached by a good ruler (about 0.8 of the time at the target share) instead of being doomed by lack of wrong answers.
- Every added answer and every canary is tagged, so results can be shown with and without them.

### Trade-offs

- **The labeled set may mix the real system with a worse version.** The Spearman and the cut line then describe that mixture, not production; a change of generator or embedding model forces a recalibration (ADR-008, condition 3).
- **More labeling work.** Up to 200 answers instead of 100–150, plus 20 canaries reviewed.
- **The target may not be reachable:** if the cap is hit below 30%, C1 is reported as measured and ADR-008's path 2 applies.
- **Two numbers are our choices** (30%, 200), and the 80% of C2 and the 0.60 of C1 stay the author's choices (ADR-008).
- **One labeler**, so no agreement figure between annotators can be reported (the reference study had two plus a tie-break).
- **Open, not decided here:** how to label a *refusal* of an answerable question (the abstention labeling of ADR-007 shares the session and will define it); the choice of the worse version.
- **The simulation is only an order of magnitude** (normal scores, independent answers).

### Conditions that invalidate this decision

This decision should be **revisited** if:

1. The first real answers show far more wrong answers than expected (20% or more): the worse version may not be needed.
2. The labeled set cannot reach 30% within 200 answers: reopen with the evidence of the calibration (ADR-008, path 2).
3. The generator, the embedding model or the golden set changes (the labels, canaries and cut lines are void).
4. The author finds the `partial` cases too frequent for the rule "partial is still right".
5. A study appears that measures these rulers on answers and references like ours.

### Migration path when needed

1. **Relax or tighten the target share and the cap** (a decision about numbers only, taken before the labeling).
2. **Apply a stricter reading of "right"** using the `partial` flag, with no relabeling.
3. **Add canary types** (for instance entity swap), declared before the labeling.
4. **Recalibrate** after any change of generator, embedding model or golden set.

## Addendum (2026-10-12): observations after the decision, and predictions registered before EXP-0

> **This addendum changes no decision.** D-16 to D-19 stand exactly as written above. It records (1) observations that came up when the author questioned the design right after it was accepted, (2) the points that are still open, and (3) **predictions made before any real answer exists**, so that after EXP-0 they can be compared with what actually happens (the author's proposal: keep the design, observe, and turn the gap between prediction and result into a lesson learned). Every number below comes from the recorded output of [`evidence/0011-observations-after-decision.py`](./evidence/0011-observations-after-decision.py); a re-run gives the same values (fixed seed).

### How the addendum came about

After D-16 to D-19 were accepted, the author asked: (a) is enriching the labeled set with answers from a worse version of the system a workaround to reach the pass mark? (b) if the marks are arbitrary choices, why not lower the Spearman from 0.60 to 0.50? (c) are the canaries over-engineering, and can deterministic metrics detect such small edits at all? And proposed to leave the design as it is and observe.

### Observation 1 — The Spearman measures the mix of right and wrong answers as much as the ruler (the bias of D-17)

**(I, own simulation, Part A)** The *same* ruler, one that gives the higher score to the right answer in 90 of 100 right-versus-wrong pairs (AUC 0.90), gets a different Spearman depending only on the share of wrong answers (n = 118):

| Share of wrong answers | Mean Spearman of the same ruler (AUC 0.90) |
|---|---|
| 10% | 0.42 |
| 20% | 0.56 |
| 30% | 0.63 |

Consequences:
- Enriching the set with wrong answers raises the chance of passing C1 **without improving the ruler**. Passing at 30% wrong and failing at 10% do not, by themselves, say the ruler is good or bad. The author's suspicion is correct in effect.
- What still supports D-17: the 0.60 was borrowed from Adlakha 2024, whose set had about 20% wrong, so a comparable mix is a like-for-like comparison; and the use of the number is to **compare configurations** in the experiments, which needs a ruler that works over a range of qualities.
- What is **not** shown: that 0.60 at 30% wrong is the quality of ruler the project needs. The number is borrowed.
- Guard rails already in D-17: the decision to enrich depends only on the **count** of wrong labels (made blind to the scores), every added answer is tagged `worse_version`, and results are reported with and without them.

Alternatives discussed and **not adopted**: **(A)** keep D-17 and report both results (what D-17 does); **(B)** make the C1 on real answers only the official one and use the worse version as a separate, unlabeled check ("does the ruler score the worse version lower than the real one?"); **(C)** do not mix anything, accept that C1 probably fails and go to ADR-008's path 2. A note on the statistic: the Spearman against a yes/no label depends on how common the wrong answers are; a measure that does not (the AUC) exists, but ADR-008 keeps the Spearman, and the AUC is used here only to describe ruler quality in words.

### Observation 2 — Lowering 0.60 to 0.50

**(I, own simulation, Part A)** Chance of passing the cut 0.50 / the cut 0.60 (3,000 repetitions, n = 118):

| Wrong answers | Ruler AUC 0.80 | AUC 0.85 | AUC 0.90 | AUC 0.95 |
|---|---|---|---|---|
| 10% | 0.00 / 0.00 | 0.00 / 0.00 | 0.02 / 0.00 | 0.19 / 0.00 |
| 20% | 0.13 / 0.00 | 0.45 / 0.02 | 0.89 / 0.20 | 1.00 / 0.81 |
| 30% | 0.38 / 0.03 | 0.82 / 0.24 | 0.99 / 0.77 | 1.00 / 1.00 |

Reading: with 10% wrong the best possible Spearman is 0.52, so lowering to 0.50 does not remove the scarcity problem (only a nearly perfect ruler passes). With 20% wrong, the cut 0.50 asks for a ruler of about AUC 0.85 and the cut 0.60 for about 0.93. Choosing between 0.50 and 0.60 is therefore choosing **what ruler quality is required**, and the Spearman only translates that quality, with a translation that depends on the share of wrong answers. A change is legitimate only if it is made **before** the data and justified by the quality needed; no source links ruler quality to the ≥ 15% gates of `docs/plan.md`, so that remains a judgment. **The 0.60 stands.**

### Observation 3 — Deterministic metrics cannot detect the canaries, and what that means for C2

**(F, offline pre-check, Part B)** Each canary is a real `multi-doc` gold answer with one minimal edit; it is scored as the answer against the original gold (a perfect answer scores 1.0):

| Canary type | Cases found | Mean Recall | Mean F1 | Share with score ≥ 0.9 |
|---|---|---|---|---|
| Negation ("does not fix") | 7 | 0.980 | 0.961 | 1.00 |
| Number swap ("three" → "seven") | 3 | 0.979 | 0.979 | 1.00 |
| Direction reversed | 6 | 0.979 | 0.979 | 1.00 |
| Paper swap (only the ids) | 15 | 1.000 | 1.000 | 1.00 |

A correct paraphrase scored F1 0.21 / Recall 0.25 in the 2026-10-10 local test. A cut line high enough to reject these canaries (above about 0.96) would also reject almost every correct answer that is not a copy of the reference. The reason is arithmetic: one changed word in a reference of about 50 words moves the score by about 2%, and a negation only **adds** a word, so the Recall does not even move. The embedding cosine did not help either in our local test (the negated answer scored 0.94, above every correct paraphrase, 0.59–0.84).

Consequences:
1. **C2 for token-F1, Recall and cosine is a predictable failure.** The canaries' real purpose is to test the **LLM judge**.
2. **Inconsistency recorded, not resolved.** After the ADR-010 cleaning (D-15) the paper-swap canary is **identical** to its original in 15 of 15 cases, so no cheap ruler can detect it, and D-16 tells the author to ignore ids when labeling, so by that rule the swapped answer would not even be "wrong". With 5 of the 20 canaries undetectable by construction, a cheap ruler can reach at most 15 of 20 = 75%, below the 80% of C2, whatever its quality.
3. **On over-engineering:** the per-type rates and the paired check of D-18 are diagnostics that add effort; they stay as written.
4. **The cosine, measured with the real model (2026-10-12, [live check](./evidence/0010-embedding-cosine-live-check.py), part 4):** the cosine between a canary and the gold it came from is **0.978** on average for negation (n = 7, worst 0.945), **0.997** for number swap (n = 3) and **0.997** for direction reversed (n = 6); the cosine between two *unrelated* golds is 0.389 on average (0.19–0.74). Every canary scores above the highest correct-paraphrase value of the 2026-10-10 local test (0.84), so Prediction 3 is **supported by this measurement**, though it stays a prediction until the real C2 with the cut line.

### Observation 4 — The labeled set will probably be used mainly to validate the LLM judge

If Observation 3 holds, the main correctness number will be the LLM judge (ADR-008, path 2), and the calibration mostly checks the judge. For a yes/no verdict what matters is **how many wrong answers** the set contains, because that sets how precisely we learn what share the judge catches. **(F, Part C, Wilson 95% interval)**, for a judge that catches 90% of the wrong answers it sees:

| Wrong answers | Caught | 95% interval |
|---|---|---|
| 12 | 11 of 12 | 65% – 99% |
| 20 | 18 of 20 | 70% – 97% |
| 30 | 27 of 30 | 74% – 97% |
| 40 | 36 of 40 | 77% – 96% |
| 50 | 45 of 50 | 79% – 96% |

The canaries add known-wrong cases but they are synthetic and easier than natural errors, so they do not replace real wrong answers. An idea **not adopted**: aim D-17 at a **count** of about 30 wrong answers (which would also drop the need for a worse version that errs 60% or more) instead of a 30% share. D-17 is unchanged.

### Open items (closed on 2026-10-12 by D-20 to D-23 — see the Amendment)

| # | Open question | Discussed options | Recommendation given in the discussion (not adopted) |
|---|---|---|---|
| 1 | Type of "worse version" (D-17) | A weaker model · the same model with a degraded prompt · the same model with a context full of irrelevant abstracts (gold paper present) | A weaker model, after a pilot of about 20 answers to measure its error rate. Pairing each question with another question's answer was rejected: trivially separable, it would inflate C1. |
| 2 | Refusals of answerable questions | Leave them out of the calibration set (they are already counted by `over_refusal_rate`, ADR-007) · label them wrong | Leave them out: they share almost no words with the reference, so they would inflate C1. Cost: fewer wrong answers in the set. |
| 3 | Paper-swap canary vs. D-15/D-16 (Observation 3, point 2) | Replace it with a **key-term swap** (e.g. "planning" → "memory") · keep it for the judge only · use 3 types (C2 would be 12 of 15) | Key-term swap. |
| 4 | Tie-break between token-F1 and Recall if both pass | Highest Spearman, ties to Recall · fixed choice | Highest Spearman; ties to the Recall, which does not depend on answer length. Probably moot. |
| 5 | Does the judge see the arxiv ids? | — | Belongs to P11 and batch 0.6.3. |

Considered and **not worth deepening now**: an NLI (contradiction-detection) model as a rung before the judge (no evidence read on long answers), rule-based checks of numbers and negation words (brittle), changing 0.60 to 0.50 (Observation 2).

### Predictions registered before EXP-0

Written on 2026-10-12, when **no real answer exists**. After EXP-0 and the labeling, each is compared with the result and a lesson learned is written (skill `learning-lesson`), including the cases where the prediction fails.

| # | Prediction | Basis | Would be falsified if… |
|---|---|---|---|
| 1 | Among the real labeled answers (gold paper retrieved) the wrong answers are **between 5% and 15%**, so below 30% and the enrichment of D-17 is triggered. | hit@5 = 100% on 50 questions and a strong generator; Adlakha's 20% came from weaker, pooled models | wrong ≥ 30%, or fewer than 4 of 118 |
| 2 | **Token-F1 and Recall do not pass C2** (fewer than 16 of 20 canaries below their cut line). | Part B | either reaches 16 of 20 |
| 3 | **The embedding cosine does not pass C2.** | local test: negation 0.94 above correct paraphrases | it reaches 16 of 20 |
| 4 | **On the real answers alone no cheap ruler reaches Spearman 0.60**; the best lands at about 0.2–0.5. | Part A (10–15% wrong) | any cheap ruler ≥ 0.60 on real answers alone |
| 5 | **The main correctness number is the LLM judge** (ADR-008, path 2). | predictions 2–4 | a cheap ruler passes both C1 and C2 |
| 6 | **Refusals of answerable questions are rare** (under 10% of the real answers). **Weak basis:** a modern model with the gold paper present, but the prompt tells it to say "I don't know" and Adlakha saw 25–41% with older models. | — | 10% or more |

## Amendment (2026-10-12): decisions that close P9

The author approved, one by one, the recommendations that had been given in the discussion of the addendum. D-16 to D-19 are **kept**, except that D-22 replaces the fourth canary type of D-18. Predictions 1 to 6 of the addendum are **unchanged**.

**D-17 confirmed.** The enrichment with a worse version stays as written (label the real answers first, blind to the scores; if fewer than 30% are wrong, add worse-version answers up to 30% or 200 labeled answers).

**D-20 — Type of the worse version.** The worse version is a **weaker model than `gpt-4o-mini`**, given the **same context as the real system, with the gold paper present** (a failure must be the generator's). It is built **only if needed**, i.e. after the real answers are labeled and fewer than 30% are wrong. Before fixing the model, a **pilot of about 20 answers** from a candidate is run on eligible questions and labeled by the author under D-16; the candidate is accepted if at least about **60%** of those answers are wrong (the level the cap of 200 needs, see Part 4 of the simulation). If no candidate reaches it, a candidate is combined with a context padded with irrelevant abstracts (the gold paper still present). The pilot's labeled answers count among the extra answers. Rejected: pairing each question with the answer to another question (wrong by construction and trivially separable, it would inflate C1). *The model itself is chosen at the pilot, not here.*

**D-21 — Refusals of answerable questions.** While labeling, the author marks an answer **`refusal`** when the system declines to answer a question whose gold paper was retrieved. These answers are **left out of the calibration set** (they are not labeled right or wrong, and they do not enter C1, the cut line or C2) and are **counted and reported next to it**. The over-refusal is measured by `over_refusal_rate` (ADR-007), and the author's `refusal` marks also serve as the reference for the heuristic-versus-judge comparison that ADR-007 already asks for in the shared labeling session. Reason: a refusal shares almost no words with the reference, so every ruler separates it trivially and it would inflate C1 without testing anything. Cost accepted: fewer wrong answers in the set.

**D-22 — Canary types (amends D-18).** The four types are **negation, number swap, direction reversed and key-term swap**, five each (20 in total). The **key-term swap** replaces one salient content word (a method, metric, dataset or component name) by another of the same kind, e.g. "varies three components — *planning*, action space, …" → "— *memory*, action space, …". It replaces the paper swap because, after the ADR-010 cleaning, a paper swap is identical to its original (15 of 15 cases) and D-16 ignores ids. Effects: C2 stays "at least 16 of 20 below the cut line"; the ceiling of 15 of 20 = 75% that the paper-swap type imposed on the cheap rulers **disappears** (all 20 canaries are visible in the text); token-F1, Recall and cosine are still expected to miss all four types (Prediction 2 and 3), so the main use of the canaries remains testing the LLM judge. Everything else in D-18 (built from answers the author labeled right, outside the C1 set, draft reviewed by the author, per-type rate and paired check as diagnostics) is unchanged.

**D-23 — Tie-break between token-F1 and Recall.** If both pass C1 and C2, the main number is the one with the **higher Spearman** on the author's labels; if the two Spearman values differ by **less than 0.05**, the **Recall**, because it does not depend on the length of the answer (a prompt change that makes answers longer or shorter moves the F1 without changing correctness). The 0.05 is an engineering choice: with about 120 labels a measured 0.60 could lie anywhere from about 0.47 to 0.70 (ADR-008), so a smaller difference is noise. By Prediction 2 this rule will probably never be used.

**Deferred, not a P9 decision.** Whether the judge sees the arxiv ids belongs to P11 and to batch 0.6.3 (the judge's prompt).

**What remains for P9.** The embedding-cosine evaluator was **built on 2026-10-12** (id removal only, `text-embedding-3-small` pinned, embeddings cached in the process) and checked against the real API; what is left is, after EXP-0, the calibration itself. The author also approved, after the live check, two clarifications recorded in ADR-010 (D-24: the cosine also drops the brackets left empty by the id removal; D-25: the embedding cache is kept on disk).

## References

- Roadmap: [`ckpt-0.6.1b-metrics-roadmap.md`](../roadmap/ckpt-0.6.1b-metrics-roadmap.md) — §5 decisions D-16 to D-19; §7 P9
- [ADR-008](./0008-answer-correctness-signals-and-calibration.md) (C1, C2, the calibration), [ADR-010](./0010-text-normalization-for-lexical-correctness-signals.md) (cleaning rule), [ADR-007](./0007-abstention-detection-and-reporting.md) (shared labeling session)
- Reading notes: [`adlakha-2024-correctness-faithfulness-iqa.md`](../research/fichamentos/adlakha-2024-correctness-faithfulness-iqa.md) (arXiv:2307.16877) · [`magesh-2025-hallucination-free.md`](../research/fichamentos/magesh-2025-hallucination-free.md) · [`ribeiro-2020-checklist-behavioral-testing.md`](../research/fichamentos/ribeiro-2020-checklist-behavioral-testing.md) (arXiv:2005.04118; <https://aclanthology.org/2020.acl-main.442/>) · [`cao-2025-negation-blindness-embeddings.md`](../research/fichamentos/cao-2025-negation-blindness-embeddings.md) · [`qa-eval-judge-vs-f1.md`](../research/fichamentos/qa-eval-judge-vs-f1.md) (Ho et al. 2025, Wang et al. 2023)
- Not read (paywall): Fluss, R., Faraggi, D., Reiser, B. (2005). *Estimation of the Youden Index and its associated cutoff point*. Biometrical Journal 47, 458–472 — <https://onlinelibrary.wiley.com/doi/10.1002/bimj.200410135>
- Evidence script and recorded output: [`evidence/0011-label-supply-simulation.py`](./evidence/0011-label-supply-simulation.py), [`.out.txt`](./evidence/0011-label-supply-simulation.out.txt)
- Addendum evidence: [`evidence/0011-observations-after-decision.py`](./evidence/0011-observations-after-decision.py), [`.out.txt`](./evidence/0011-observations-after-decision.out.txt)
