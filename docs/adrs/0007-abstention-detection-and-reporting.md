# ADR-007: Abstention — how the system says "I don't know", and how we measure it

**Status**: Accepted
**Date**: 2026-10-06
**Related**: [Roadmap §6 P7/P8 and §5 decisions D-5, D-6](../roadmap/ckpt-0.6.1b-metrics-roadmap.md) · [Reading notes: Abstention/Refusal benchmarks](../research/fichamentos/abstention-benchmarks.md) · [decisions.md, 2026-10-06](../decisions.md) · [ADR-001](./0001-golden-set-provenance-data-model.md) (the dataset field that says when refusing is the right answer)
**Code**: `evaluators.py` (`abstained`, `wilson_ci`, `abstention_summary`, `f1_summary_evaluator`); tests in `tests/test_evaluators.py`

---

> **This record is written for any reader, not only engineers.** Every technical term is explained where it first appears. The short version: the old way of checking *whether the assistant admitted "I don't know"* missed any paraphrase, and the old way of *scoring* that behavior could be cheated by a system that refuses everything. This record explains why, what we decided instead, and what evidence supports it.

## Context

### What the product must do, and what "abstention" means here

The product is a chat that answers questions using **only** 250 paper abstracts stored in its own database. When the answer is not in those abstracts, the assistant must **admit it** — say some version of "I don't know" — instead of inventing. We call that **abstention**.

Refusing matters in both directions:

- If it **answers anyway**, it makes things up (the dangerous failure).
- If it **refuses too much**, the product is useless: even questions it could answer get a shrug.

Because of that, the project has a release requirement: **at least 90% of the questions that cannot be answered must be correctly refused**. The test set has 100 questions: 25 should be refused (they have no answer in the abstracts), 75 should be answered. Each question carries an explicit label saying which case it is.

### Problem 1 — how we *detected* a refusal was broken

The code that decided "did the assistant refuse?" checked whether the answer **begins with one of two exact phrases**:

```python
answer.strip().startswith(("I don't know", "No papers found"))
```

The problem: the instruction we give the assistant **never told it to use those phrases**. It only says "if you don't know, say so" — so the assistant phrases the refusal however it wants. The check then concludes "it did not refuse":

| What the assistant actually said | Is it a refusal? | What the old check concluded |
|---|---|---|
| "I don't know." | yes | refusal ✔ |
| "I'm sorry, the context doesn't say." | yes | **answered** ✘ |
| "No papers found in this period." | yes | refusal ✔ (by luck: the phrase exists only in the reference data) |
| "The retrieved abstracts don't cover this." | yes | **answered** ✘ |

This was reproduced in the project: `"I'm sorry, the context doesn't say."` scored **0** when it should score **1**. The practical consequence: an assistant that refuses all 25 unanswerable questions correctly — each in its own words — would measure **0%** and we would conclude the product is broken when it is working. **The measurement was measuring our two phrases, not the system.**

### Problem 2 — how we *scored* refusals could be cheated

The old score was one number: **accuracy** — the share of correct calls. The problem: two **opposite** ways of being wrong get mixed into that single number:

- refusing when it should answer (the "useless product" error), and
- answering when it should refuse (the "making things up" error).

Two deliberately dumb systems show how the single number lies:

| Dumb system | Old score (accuracy) | "Refused when it should" (the release requirement) | Refused when it should have answered |
|---|---|---|---|
| **Never refuses** | **75%** (looks decent) | 0% | 0% |
| **Refuses everything** | 25% (looks bad) | **100% — passes the requirement!** | **100%** (product useless) |

The refuse-everything system **passes the release requirement at 100%** and is completely useless. One number cannot tell those cases apart.

### Problem 3 — the sample is small, so every point must be shown honestly

There are only 25 should-refuse questions (15 in one group, 10 in another). At that size, **one single question changes the score by 6.7 points**: getting 14 of 15 right is 93.3% (passes the 90% requirement); 13 of 15 is 86.7% (fails). It would be dishonest to report "93%" as if it were a precise fact — it might easily be luck.

### What the research we read says (the "fichamentos")

We read the four main published benchmarks on this exact problem. What they contribute, in plain words:

1. **A benchmark from Meta (AbstentionBench, 35,000 questions on exactly "when should a model say I don't know")**: they **tried** matching phrases and rejected it, spending a whole appendix explaining why. Their solution: a **second model acting as a referee** — it reads the answer and says "refusal: yes/no" — checked against 300 examples labeled by humans (agreement: 88%). They also state that the referee must run at its most deterministic setting ("temperature 0", i.e., always giving the same output for the same input, like a calculator) for this to be reliable.
2. **A caution from another paper (UAEval4RAG)**: they had **humans double-check other humans** on "is this a refusal or not?" — and the humans agreed only **76% of the time**. Even people disagree about what counts as refusing. So a referee's score must be validated on **our own answers**, not imported from a paper.
3. **A point in favor of cheap phrase-checks (OR-Bench)**: a keyword check agreed with an expensive referee within 2.4% — **but** that was for *safety* refusals, which use predictable phrases ("I cannot help with that"). In our case the assistant has no fixed phrase, so that agreement has to be **measured**, not assumed.
4. **The trade-off is real (RefusalBench)**: the two errors pull in opposite directions (correlation −0.78); one model refused **14.6× more than necessary**. This is why the release requirement cannot stand alone — a system can always pass it by refusing everything. The same paper shows the clean fix on the product side: instruct the model to emit a fixed refusal *code* instead of free text.

## Decision

Two linked decisions, one about **detecting** refusals and one about **scoring** them.

### How refusals are detected (D-5) — a cheap first check, a referee as the reference

1. **The cheap check comes first: `abstained(answer)`.** A small, deterministic, free function. Before comparing, it tidies the text up (lowercase, fix typographic apostrophes, collapse spaces) and then looks for **two families of phrasings**:
   - the **speaker owns the ignorance**: "I don't know", "I cannot find"…
   - the **source lacks the answer**: "no papers found", "not in the context"…
   One deliberate narrowness: "don't know" **by itself** is not a match, so an answer that reports *the paper's* uncertainty ("the authors don't know whether…") stays an **answer**, not a refusal. Another one: the check **understands English only** — see the trade-offs below for what happens with other languages and why we did not solve it by translating the list.
2. **The referee is the reference, not the check.** A second model (different from the one that generates answers), at the deterministic setting, reading **only the answer** and answering "refusal: yes/no". It enters the pipeline at the next checkpoint (0.6.3) and will be validated against **~50 real answers labeled by hand by the project's author** — we validate on our data because even humans only agree 76% with each other.
3. **We measure how often the cheap check and the referee disagree**, and that disagreement number goes into the baseline experiment report (EXP-0). Target: they agree ≥ 90% of the time. If they don't, the referee becomes the primary detector.
4. **We do not force a magic refusal word — yet.** Instructing the model to emit a fixed refusal code would make detection trivial, but it **changes the product**, and the purpose of the baseline is to measure the product as it is today. The fixed-code idea stays scheduled as a controlled experiment (checkpoint 6).

### How refusals are scored (D-6) — three numbers instead of one, always with counts

The single accuracy number is replaced by an **abstention report** with, for each group of questions (out-of-corpus, period-without-papers, etc.) and overall:

| Number | Plain meaning | Role |
|---|---|---|
| `abstention_recall` | Of the questions that **should** be refused: how many actually were | **The release requirement** (≥ 90%) |
| `over_refusal_rate` | Of the questions that **could** be answered: how many were refused anyway | **Mandatory companion**: improving the first number by refusing more is not a win if this one worsens (the refuse-everything system now shows its 100% here) |
| `abstention_f1` | A single balance number of the two | Informational only — one number is exactly what hid the trade-off |

Every number is printed **with its raw count and a confidence interval** (e.g. "14 of 15; plausible range 70%–99%"), so nobody mistakes 93% for a precise fact. The interval is a **Wilson** interval — explained below, because it was questioned and the answer is part of this record.

The old mixed `abstention_accuracy` number was **removed** from the summary evaluator where it lived; that evaluator now reports only the word-overlap metric it was named after.

The numeric tolerance for the over-refusal companion ("how much worse is too much") is deliberately **not fixed now**: it will be set against the baseline's confidence interval once the baseline exists — fixing a number before having anything to compare against would be arbitrary.

### Why Wilson and not Student's t (a question the author asked; recorded because it is a real doubt)

Both exist because **small samples break the naive error bar** — but they solve **different problems**, because the data is different.

- **Student's t** is for the **average of measurements on a continuous scale** (weights, durations, grades) where the true spread is unknown and must be estimated from the same small sample. Estimating the spread adds its own uncertainty, so the correction widens the interval (the smaller the sample, the wider).
- A **proportion** ("14 of 15 refused") has **no separate spread to estimate**: the spread is a direct consequence of the proportion itself. There is nothing for Student's t to correct — using it here is using the tool of a neighboring problem.
- What Wilson does instead: rather than plugging the observed fraction into the error formula (which produces absurdities — for 14/15 it yields a range of **81% to 106%**, i.e., more than 100%), it **asks the question backwards**: "which true values are compatible with having seen 14 in 15?" and solves for the answer. The result stays inside 0–100% and behaves at the extremes (0/15, 15/15).

Reference points, pinned by tests: 10 of 10 → **[72%; 100%]**; 14 of 15 → **[70%; 99%]**.

One honest caveat, true for Wilson and every interval of this kind: they assume each observation is independent. In our adjudication pilot, several judgments come from the same question and are not fully independent — the interval is honest about sampling luck, not about that clustering.

## Alternatives considered

### Alternative A: a better phrase-check only (no referee)

Improve the list of refusal phrasings and stop there.

**Why not chosen**:
- No finite list covers every way of saying "I don't know" (the Meta benchmark's appendix is the standing objection).
- Without a referee, we could never **know** how much the list misses — the disagreement measurement is precisely what makes the cheap check defensible.

**Advantages** (partially leveraged):
- Free and instant — kept as the always-on detector; the referee audits it.

### Alternative B: force a fixed refusal code in the assistant's instructions now

Instruct the model to answer with a machine-readable code (like `REFUSE_MISSING_INFO`) instead of free text; detection becomes trivial parsing.

**Why not chosen**:
- It **changes the product**, and the baseline exists to measure the current prompt before optimizing it.
- Kept as a scheduled experiment (checkpoint 6), where its effect on answer quality can be measured properly.

**Advantages** (not yet leveraged):
- Removes the ambiguity at the source; the referee would become a simple reader.

### Alternative C: keep the two exact phrases (status quo)

**Why not chosen**:
- Demonstrated failure: a perfectly legitimate refusal scored 0 — the release requirement was measuring our two phrases, not the system.

### Alternative D: keep the single accuracy score (status quo)

**Why not chosen**:
- The refuse-everything system passes the release requirement at 100% while being useless; the never-refuse system looks respectable at 75% while making things up. One number cannot separate the two failures it mixes.

### Alternative E: adopt the full scorecard of the RefusalBench paper

Six error categories plus a combined score (a simple average of two rates).

**Why not chosen**:
- Categorizing **why** the system refuses is not a requirement of this project (we have two reasons, by construction: not in the corpus; no papers in the period).
- Their combined single score is exactly the kind of number that hides the trade-off — the paper's own reading notes advise against using it as a gate.

## Consequences

### Accepted

- The release requirement can no longer be satisfied by refusing everything: the companion number travels with it and would expose the trick.
- Refusals phrased in the assistant's own words now count (16 paraphrase forms tested), while answers that merely report the paper's own uncertainty are not mistaken for refusals (10 such cases tested).
- Every score is published with its count and its plausible range, so a 1-in-15 swing is visible rather than hidden.
- The always-on detection costs nothing and runs offline; the referee is a bounded, one-off calibration (~50 hand-labeled answers).

### Trade-offs

- The cheap check understands **only English**. The assistant's instructions fix no answer language, but today's test set is entirely in English — so the gap is real but *latent*: if someone asks in Portuguese ("Desculpe, não encontrei nada sobre isso no contexto.") the refusal reads as an answer and the recall score would sink through no fault of the product. We deliberately did **not** patch this by translating the phrase list: a translated list is the original trap on a new axis — a list that never completes, now multiplied by every language. The real answers are the ones already in this design: the referee (a language model reads any language natively, no list needed) and, if the product ever serves non-English questions, a product decision on the answer language — either fix it in the assistant's instructions (a checkpoint-6 experiment, since it changes the product) or accept multilingual output and make the referee the primary detector. A test pins today's behavior (`test_abstained_is_english_scoped`) so the boundary is explicit, not silent.
- The cheap check has one **known, accepted imperfection**: a hedge inside a real answer ("I don't know whether the paper tests this, but the results suggest…") is counted as a refusal. Accepted deliberately — it will be **measured** by the referee comparison rather than hidden.
- The referee adds model cost at the next checkpoint plus a one-off human labeling effort, and the labels only hold while the golden set is stable.
- Three numbers per group instead of one: more to read, but each means exactly one thing.
- Confidence intervals assume independence between observations; judgments clustered on one question violate that. The interval is honest about sampling luck, not about clustering (stated above, repeated here because it is a real limitation).

### Conditions that invalidate this decision

This decision should be **revisited** if:

1. The cheap check and the referee agree on **less than 90%** of the baseline run — the referee becomes the primary detector.
2. The assistant's instructions change to fix a refusal phrase or emit a refusal code (the checkpoint-6 experiment wins) — detection becomes trivial and this record is superseded.
3. The golden set is regenerated — the ~50 validation labels must be redone before the referee is trusted again.
4. Setting the over-refusal tolerance against the baseline's confidence interval proves impractical at EXP-0 — a fixed percentage-point threshold should then be agreed explicitly.
5. The product starts serving questions in **languages other than English** (the test set today is English-only, so the current scores never see the gap) — then either the answer language is fixed in the assistant's instructions, or the referee becomes the primary detector.

### Migration path when needed

Ordered from least to most disruptive:

1. **Referee becomes the primary detector**; the phrase-check stays as a free diagnostic and as the disagreement measure. No data changes.
2. **Fixed refusal code in the assistant's instructions** plus a tolerant reader — a product change, to be decided at checkpoint 6 with its own before/after comparison.
3. **Full re-calibration after any dataset regeneration**: relabel ~50 answers, re-measure the disagreement, then trust the referee again.

## References

- Roadmap: [`ckpt-0.6.1b-metrics-roadmap.md`](../roadmap/ckpt-0.6.1b-metrics-roadmap.md) — sections P7/P8 with the full evidence tables and the decision log entries D-5/D-6
- Reading notes (fichamentos): [`abstention-benchmarks.md`](../research/fichamentos/abstention-benchmarks.md) (AbstentionBench, RefusalBench, UAEval4RAG, OR-Bench) · [`yu-2024-rag-eval-survey.md`](../research/fichamentos/yu-2024-rag-eval-survey.md) (the survey's refusal rate is one-sided) · [`magesh-2025-hallucination-free.md`](../research/fichamentos/magesh-2025-hallucination-free.md) (a useful refusal explains why; a lazy one just refuses)
- `docs/decisions.md` — 2026-10-06 observation: the regression rule ("a gain in refusal recall cannot come from an increase in over-refusal") and the detection rule
- [ADR-001](./0001-golden-set-provenance-data-model.md) — where the "should this question be refused?" label lives in the dataset
