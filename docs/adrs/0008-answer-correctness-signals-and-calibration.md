# ADR-008: Answer correctness — free signals first, embedding cosine next, an LLM judge only if the calibration demands it

**Status**: Accepted
**Date**: 2026-10-11
**Related**: [Roadmap §7 P9 and decisions D-12, D-13](../roadmap/ckpt-0.6.1b-metrics-roadmap.md) · Reading notes: [Adlakha 2024](../research/fichamentos/adlakha-2024-correctness-faithfulness-iqa.md) · [Badshah 2025](../research/fichamentos/badshah-2025-reference-guided-verdict.md) · [SemScore 2024](../research/fichamentos/aynetdinov-2024-semscore.md) · [Cao 2025](../research/fichamentos/cao-2025-negation-blindness-embeddings.md) · [Ho + Wang (QA judge vs F1)](../research/fichamentos/qa-eval-judge-vs-f1.md) · [RAGChecker](../research/fichamentos/ru-2024-ragchecker.md) · [ADR-007](./0007-abstention-detection-and-reporting.md) (abstention, the sibling decision) · [ADR-010](./0010-text-normalization-for-lexical-correctness-signals.md) (the text-normalization rule that §2 left open)
**Code**: `evaluators.py` (`token_f1_summary`, `token_f1`, `word_recall`, `normalize_text`; slice P9a, implemented 2026-10-12). **Still not implemented**: the embedding-cosine evaluator, the canaries and the calibration. This record fixes the design and the pass marks before any real answer exists.

---

> **This record is written for any reader, not only engineers.** Every technical term is explained where it first appears. The short version: the only number we have for "is the answer right?" counts shared words, which punishes correct answers phrased differently and can reward a wrong answer that reuses the reference's words. This record explains what we tried, what the research does and does not tell us about *our* kind of answers, and the order in which we will use cheap measurements before paying for an LLM judge — with the pass marks fixed **before** any data is seen.

## Context

### What "correct" means here, and where it is measured

Each golden question carries a **reference answer** written beforehand. "Correct" means the system's answer says the same thing as that reference. This is measured on the slices that have a real reference answer: `answerable` (40), `persona` (10) and `multi-doc` (15), that is, **65 examples per run**. The 25 questions that should be refused (ADR-007) and the 10 `format` questions have their own evaluators and are not scored for correctness.

### Problem 1 — the only number we have counts shared words

The current score, **token-F1**, compares the words of the answer with the words of the reference (the share of words in common, balanced between "how much of the reference was covered" and "how much of the answer was useful"). Reproduced offline on 2026-10-10 against the reference `"LLM agents often hallucinate tool calls [2609.12345]."`:

| Answer | token-F1 |
|---|---|
| Correct paraphrase, no citation | 0.21 |
| Correct paraphrase, same citation | 0.62 |
| **Wrong** (opposite meaning, same words + citation) | **0.88** |
| Wrong (unrelated claim), cites the same paper | 0.24 |

Three separate problems show up:

1. **A correct paraphrase scores low and a wrong answer that reuses the words scores high.** These are constructed examples: they prove it *can* happen, not how often.
2. **Citations count as words.** `[2609.12345]` becomes the tokens `2609` and `12345`, so citing the right paper adds points by itself (0.21 → 0.62 for the same paraphrase).
3. **The average mixes in the refusal questions.** The 25 should-refuse examples have references like `"I don't know."`; a correct refusal in other words scores 0.33, or 0.00 against `"No papers found in this period."`. Refusals are already measured properly by ADR-007.

### Problem 2 — no study covers our kind of answer

What the research says, in plain words (reading notes linked above; read in full, not only abstracts):

- **Adlakha et al. 2024** (1,200 human-labeled answers from instruction-following models over retrieved passages): token-F1 rejected **39%** of the answers humans judged correct but accepted only **3.8%** of those judged incorrect. So the dominant failure is *underestimating* correct answers, not rewarding wrong ones. A simpler metric, **Recall** (the share of the reference's words that appear in the answer), agreed better with humans than F1 (agreement 0.60 vs 0.47, same measure as in §5 below); a GPT-3.5 judge reached 0.61 and a GPT-4 judge 0.67.
- **Ho et al. 2025 and Wang et al. 2023**: LLM judges agree with humans much better than F1 on short extractive answers; but on **long** answers a GPT-3.5 judge *lost* to a simple containment check (69.5% vs 82.3% accuracy on one system). A judge is not automatically better and needs local checking.
- **Badshah & Sajjad 2025**: a judge that reads question + answer + reference and returns true/false agreed with humans at Cohen's κ of 0.63–0.93 (100 samples per dataset); they warn that **noisy references mislead the judge**, and ours are synthetic (written by an LLM from one abstract and reviewed as questions/gold, not as generated answers).

The catch: those studies use **short references** (2–11 words in Adlakha). Our references are full sentences — the 15 `multi-doc` reference answers average **53 words** (44–62, measured 2026-10-10). **No published number tells which metric is best for three-sentence answers against synthetic references.** The choice has to be tested locally.

### Problem 3 — the cheap semantic option has a documented blind spot

An obvious cheaper alternative to a judge is **embedding cosine**: turn the answer and the reference into lists of numbers with an embedding model and measure how close the two lists are (1.0 = same direction, near 0 = unrelated).

- **SemScore (Aynetdinov & Akbik 2024)** used exactly this and ranked 12 LLMs like humans did (Kendall τ 0.879, comparable to an LLM-judge metric at 0.855 and better than ROUGE-L at 0.788) — but on the ranking of **12 systems**, not on single answers, with one main annotator.
- **Cao 2025** shows that embedding models give a sentence and its **negation** a high cosine (0.6–1 in every similarity group; in 99.27% of the pairs humans rated dissimilar, the negation was closer than the unrelated sentence), and that a cosine value means different things in different models.
- **Our own test (2026-10-10)** with the project's embedding model (`text-embedding-3-small`), six constructed answers and three control pairs:

| Answer | token-F1 | Recall | Cosine | Human would say |
|---|---|---|---|---|
| Correct paraphrase | 0.24 | 0.33 | 0.59 | correct |
| Correct and verbose | 0.48 | 1.00 | 0.83 | correct |
| **Opposite meaning** | 0.83 | 0.83 | **0.94** | **wrong** |
| Unrelated topic | 0.00 | 0.00 | 0.11 | wrong |
| Correct and short | 0.44 | 0.33 | 0.72 | correct |
| Correct paraphrase, other wording | 0.36 | 0.67 | 0.84 | correct |

Negation control pairs: 0.92 ("reduces" vs "does not reduce") and 0.87 ("outperforms" vs "does not outperform"); unrelated sentences: 0.006. (F1 and Recall computed after removing citations.) Reading: the cosine separates paraphrase from unrelated topics well (0.59–0.84 vs 0.11), but gave the **highest** score to the answer that contradicts the reference. The same happens with the word metrics. Six constructed answers prove the blind spot exists; they do not measure how often it matters.

### Problem 4 — the sample is small

With 65 examples per run, a 70% rate carries a margin of about ±11 points (Wilson interval, own arithmetic). Decisions of 10–15% between configurations sit near the noise, whatever the metric. How to handle gates statistically is a separate point (P15), not decided here.

## Decision

### 1. Scope

Correctness is computed only for `answerable`, `persona` and `multi-doc`. Refusal and format keep their own evaluators.

### 2. Three cheap signals, every run

| Signal | What it is | Cost |
|---|---|---|
| `token_f1_summary` | token-F1 after removing citations and normalizing the text — **the exact rule is fixed in [ADR-010](./0010-text-normalization-for-lexical-correctness-signals.md)** (renamed from `f1_summary`) | free, offline |
| word Recall | share of the reference's words present in the answer (same normalization, ADR-010) | free, offline |
| embedding cosine | cosine between the answer and the reference (arXiv ids removed, nothing else — ADR-010) with `text-embedding-3-small` **pinned** (the model of the retrieval index); reference embeddings cached | cents |

Until the calibration below has run, these are reported as **diagnostics, never as "the quality of the answers"**. Changing the embedding model invalidates its pass marks and forces a new calibration (its scale is model-specific — Cao 2025).

### 3. The LLM judge is conditional

A `correctness_judge` (an LLM that reads question + reference + answer and returns correct/incorrect) is **not built before the calibration**. Policy:

- If a cheap signal passes both pass marks (§5), the judge does **not** run in routine rounds. It still confirms **every gate decision** (differences of ±10–15% between configurations) and large changes of generator or prompt: no gate rests on cheap signals alone.
- If no cheap signal passes, the judge runs **every round**, on all examples, and is calibrated with the same labels (no extra labeling).

### 4. Calibration — once, after the EXP-0 run produces real answers

1. The author labels **100–150 real answers** as correct or incorrect, adding a flag **"contradicts the reference"** when applicable. Only answers where the gold paper was retrieved (hit@k = 1) are labeled, so a failure is the generator's and not the retriever's (design of Adlakha 2024).
2. **10–20 canaries** are built: real correct answers turned into their opposite (negation, swapped number). We know they are wrong; they test whether a signal *can* detect contradiction. The flag in step 1 measures how *often* contradictions occur.
3. The labeling session is shared with the validation of the abstention judge (~50 answers, ADR-007) and the judge-calibration point P11.

### 5. Two pass marks, fixed before looking at any data

| Mark | Question it answers | Pass mark | Where the number comes from |
|---|---|---|---|
| **C1 — agreement** | Does the signal order answers like the author's labels? | **Spearman ≥ 0.60** between the signal's scores and the author's correct/incorrect labels | Reference level only: in Adlakha 2024 (Tab. 3) Recall reached 60.0 against human labels (F1 47.3, GPT-3.5 judge 61.4, GPT-4 judge 67.5). That paper sets **no cut**; using 0.60 as the cut is **our choice**. Spearman is the measure that paper used because binary labels create many ties. |
| **C2 — contradiction** | Does the signal notice when an answer says the opposite? | **≥ 80% of the canaries fall below the cut line** | **Author-approved choice, no source.** The *cut line* is the score under which a signal calls an answer wrong; it is set from the author's labels and then frozen. |

*Spearman* is a "concordance of order" number from −1 to +1: +1 means the signal's order matches the author's, 0 means no relation. In a 6-answer illustration the word and cosine signals scored 0.00–0.21 because one wrong answer got a top score, and 0.71–0.73 without it.

### 6. How the main number is chosen

The **simplest** signal that passes **both** marks becomes the main correctness number, in this order: word metrics → embedding cosine → LLM judge. Expected outcomes (not guaranteed):

1. A cheap signal passes both → it is the main number; the judge only confirms gates.
2. A cheap signal passes C1 but fails C2 (what the test above suggests for the cosine) → the judge runs every round.
3. Not even the judge passes → the problem is probably in the references: audit them before trusting any number.

### 7. Reporting

Signals appear as diagnostics until calibrated; the main number is reported with its count and a Wilson interval; any disagreement between the judge and the author is audited against the reference, because a wrong reference would look like a system failure.

### Out of scope

The negation adapter of Cao 2025, a vote of three judges, claim-level metrics (RAGChecker style) and BERTScore.

## Alternatives considered

### Alternative A: keep token-F1 as the quality number

**Why not chosen**:
- Reproduced failures (Problem 1); with a longer or shorter answer after a prompt change, F1 moves without correctness changing, which would distort the ±10–15% gates.

**Advantages** (partially leveraged):
- Free and deterministic — kept as a diagnostic.

### Alternative B: fix token-F1 and stop there

**Why not chosen**:
- Leaves the report without any signal that reads meaning; paraphrase and negation stay wrong.

**Advantages** (leveraged): the citation/normalization fix is part of §2.

### Alternative C: word metrics only (F1 + Recall), no judge, no embeddings

**Why not chosen**:
- Neither handles paraphrase well (Recall 0.25–0.33 on correct paraphrases in our test) or negation (Recall 0.83 on the opposite answer). Adlakha's good result for Recall was with 2–11-word references.

**Advantages** (leveraged): free; the length-insensitive Recall complements F1.

### Alternative D: an LLM judge as the main metric from day one

**Why not chosen**:
- Reads meaning, but costs one call per example per run, is unvalidated until labels exist, and a GPT-3.5 judge barely beat free Recall in Adlakha (0.61 vs 0.60) and lost to a lexical check on long answers in Wang et al.
- Removes the free baseline against which a judge should be compared, and is built before knowing whether cheaper signals suffice.

**Advantages** (kept in reserve): the only option that can detect contradiction reliably — it is the planned fallback in §3 and §6.

### Alternative E: heavier variants

- **Three-judge majority vote** (Badshah 2025): 3× the calls; against the best single judge it was higher in 5 of 9 rows, equal in 2, lower in 2 (own arithmetic from Tab. 2).
- **Claim-level metric** (RAGChecker): best correlation with human preference (61.93 vs 43.10 for ROUGE-L) but on another task (preference between pairs) and with high cost.
- **BERTScore**: worse than the sentence-level cosine in two of the sources read (Adlakha ρ 31.9; Wang et al.: worst evaluator).
- **Negation adapter** (Cao 2025): re-weights embedding dimensions to notice negation; needs ~200–500 training triples and a maintained component; tested only on open-source models. Kept as a possible mitigation if the cosine passes C1 but fails C2 and the judge turns out not to be wanted.

### Alternative F: per-example cascade (judge only on "suspicious" examples)

**Why not chosen**:
- Negations get a *high* cosine (0.94 in our test), so they do not look suspicious; the trigger would miss exactly the case it is meant to catch. A separate negation-cue detector would add complexity to save cents.

## Consequences

### Accepted

- The cheap signals run from day one; the judge is built only if the evidence asks for it.
- The same 100–150 labels calibrate every signal, including a judge added later: deciding late costs no extra labeling.
- The pass marks are fixed before the data, so the choice of main metric cannot be tuned to the results.
- The contradiction case, the most dangerous error, has its own test (canaries) instead of being assumed away.

### Trade-offs

- **A validated correctness number only exists after the EXP-0 run and the author's labeling.** Until then the report carries diagnostics only.
- **Both pass marks are choices, not findings.** 0.60 is borrowed as a reference level from Adlakha's table (the paper does not set a cut); 80% of canaries has no source.
- **Spearman against yes/no labels has a ceiling below 1** that depends on the share of correct answers (own simulation on 2026-10-11, not from a paper: with 50% correct a perfect signal reaches at most 0.87, with 80% 0.69, with 90% 0.52). Adlakha's set had about 80% correct. If our system is much more accurate, 0.60 becomes very demanding even for a perfect signal.
- **The calibration is statistically loose.** With about 120 labels, a measured 0.60 could be anywhere from about 0.47 to 0.70 (Fisher interval, own arithmetic); what limits it is the number of *incorrect* answers.
- **Canaries test whether a signal can detect contradiction, not how often contradictions occur**; frequency comes from the author's flag.
- **The judge will probably be needed**: the local test suggests the cosine fails C2. The decision does not avoid the judge; it avoids building it before knowing.
- **One labeler**, so no agreement figure between annotators can be reported.
- **Synthetic references** may be wrong; a wrong reference looks like a failure of the system, hence the audit of disagreements.
- The embedding model is pinned; changing it means recalibrating.

### Conditions that invalidate this decision

This decision should be **revisited** if:

1. The share of correct answers in the labeled set is so high (about 85–90% or more) that the Spearman ceiling makes C1 unreachable even for a perfect signal — the pass mark must then be reopened, with evidence.
2. The calibration shows that references are wrong often — fix the references before trusting any signal.
3. The generator or the embedding model changes — the cut line and the pass marks must be recalibrated.
4. The golden set is regenerated — the labels and the canaries are void and must be redone (same condition as ADR-007).
5. Contradictions turn out to be frequent in real answers — the judge becomes mandatory every round, and the negation adapter may be reconsidered.
6. A study appears that measures these signals on answers and references like ours.

### Migration path when needed

Ordered from least to most disruptive:

1. **Judge enters every round**: build `correctness_judge`, calibrate it with the existing labels, keep the cheap signals as diagnostics. No data changes.
2. **Reopen the pass marks** (for example a measure that does not depend on the share of correct answers) with the evidence collected in the calibration.
3. **Negation adapter** on the embedding signal, if the judge is judged too costly and the cosine otherwise passes.
4. **Full recalibration** after any change of generator, embedding model or golden set.

## References

- Roadmap: [`ckpt-0.6.1b-metrics-roadmap.md`](../roadmap/ckpt-0.6.1b-metrics-roadmap.md) — §7 (P9 evidence), decision log D-12/D-13
- Reading notes (fichamentos, read in full): [`adlakha-2024-correctness-faithfulness-iqa.md`](../research/fichamentos/adlakha-2024-correctness-faithfulness-iqa.md) (arXiv:2307.16877, TACL 2024) · [`badshah-2025-reference-guided-verdict.md`](../research/fichamentos/badshah-2025-reference-guided-verdict.md) (arXiv:2408.09235) · [`aynetdinov-2024-semscore.md`](../research/fichamentos/aynetdinov-2024-semscore.md) (arXiv:2401.17072) · [`cao-2025-negation-blindness-embeddings.md`](../research/fichamentos/cao-2025-negation-blindness-embeddings.md) (arXiv:2504.00584) · [`qa-eval-judge-vs-f1.md`](../research/fichamentos/qa-eval-judge-vs-f1.md) (Ho et al. 2025, Wang et al. 2023) · [`ru-2024-ragchecker.md`](../research/fichamentos/ru-2024-ragchecker.md)
- [ADR-007](./0007-abstention-detection-and-reporting.md) — abstention, the sibling decision; the labeling session is shared
- Local tests of 2026-10-10 (token-F1, Recall and `text-embedding-3-small` cosine on six constructed answers; length of the `multi-doc` reference answers) and the simulation of 2026-10-11 (Spearman ceiling with yes/no labels) are scratch computations, not stored in the repository
