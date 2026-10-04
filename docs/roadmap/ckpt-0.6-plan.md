# CKPT-0.6 Plan — `evaluators.py` (evaluator catalog)

> Status: **awaiting approval**. No code will be changed before the "go".
> Parent plan: `docs/plan.md` §2b (ground-truth map) + §4 (catalog).
> Dependencies already in place: reviewed golden set (100 ex), `utils.chunk_id`, ADR-001.

## 1. Context

EXP-0 (CKPT-0.8) needs evaluators to measure the two surfaces of plan §3:
**retrieval** (recall@k, hit@k, MRR, precision@k) and **generation** (faithfulness,
citation accuracy, completeness, format, abstention, specificity, relevance). Without them,
the golden set produces no metric at all — it is the capstone's measuring instrument.

## 2. Data contract (what the 0.7 runner will produce)

Each evaluated example reaches the evaluator with:

- `outputs` (from the target/runner): `{"answer": str, "retrieved_chunk_ids": list[str],
  "retrieved_context": list[str], "retrieved_arxiv_ids": list[str]}`
  (`app.arxiv_copilot` currently returns only `answer`; the runner (0.7) exposes the retrieved docs)
- `reference` (from the golden set): `outputs.{answer, gold_chunk_ids, gold_arxiv_ids,
  should_abstain}` + `metadata.slice`

## 3. Exact scope — what each evaluator measures and why

### 3.1 Surface 1 — Retrieval (code, reference-based)

**`retrieval_recall_at_k`** — *coverage of the required documents.*
What it is: fraction of the gold documents that appear in the retrieved top-k.
Measures: answering completely requires the right docs to be present; if the system never
retrieves them, generation is doomed before it starts.
Why (FP2 — retrieval loss / R5 weak ranking): it is the central metric of the CKPT-1/2/3 gates
("does the new k/model/chunk cover more of the required docs?").

**`retrieval_hit_rate`** — *did it find at least one?*
What it is: 1 if ≥1 gold doc is in the top-k, else 0.
Measures: cases where part of the evidence suffices (fallback), and gives a cheap binary metric
that is easy to read.
Why: complements recall — distinguishes "found nothing" from "found part".

**`retrieval_mrr`** — *how high in the ranking the first gold doc appears.*
What it is: Mean Reciprocal Rank = 1/(position of the 1st gold doc). MRR=1 if gold is 1st,
1/2 if 2nd, etc.
Measures: position, not just presence — docs at the bottom of the page tend to be ignored
by the generator.
Why (R5 — weak ranking): it is the direct target of the reranker gate (CKPT-4) and of the
`deep-hit` slice (docs that rank low).

**`retrieval_precision_at_k`** — *fraction of the top-k that is gold.*
What it is: of the k retrieved, how many are gold-labeled.
Measures: noise — how much irrelevant material competes for the generator's attention.
Why (FP3 — irrelevant retrieval/noise): exposes noise injection (e.g. the dedup collapse noted in decisions.md). **Honesty constraint (pooling bias)**: only
computed where there is a non-empty, adjudicable gold; in examples with no judgment of
irrelevance for the other docs, treating non-gold as an error would be false.

### 3.2 Surface 2 — Generation

**`faithfulness_judge`** (LLM, ref-free) — *is the answer supported by the retrieved
context?*
What it is: a judge verifies the answer claim by claim against `retrieved_context` (not against
the reference answer).
Measures: hallucination in the RAG answer — the most dangerous mode, when the system appears to
"cite" but invents.
Why (FP4): RAGAS-style metric; independent of a correct reference answer — even with a wrong
gold, an answer faithful to the context is correct behavior of the component.

**`citation_accuracy`** (code + LLM, ref-free) — *does the citation exist AND support the claim?*
What it is: a regex extracts `[2609.xxxxvN]` from the answer (code); for each citation, a judge
checks whether the chunk of that `arxiv_id` in the retrieved context supports the cited sentence.
Measures: misgrounding — the system cites a source that does not say what the answer asserts
(Magesh et al. 2025, §4).
Why (FP4 in its most specific form): the product promises "cited Q&A" — a false citation
breaks the value proposition.

**`completeness_judge`** (LLM, hybrid) — *were all parts of the question covered?*
What it is: ref-free in the other slices (were all sub-questions answered?);
reference-based in `multi-doc` (are all the points of the gold synthesis present?).
Measures: incomplete consolidation — answered only partially.
Why (FP7): it is the target of the decomposition experiments (CKPT-5).

**`abstention_quality`** (LLM, ref-based) — *abstained when it should / answered when it could?*
What it is: a judge compares the answer's behavior with the gold's `should_abstain`.
Measures: safety hedge vs. usefulness — over-abstaining is as bad as hallucinating.
Why (FP1): correct abstention on `unanswerable`/`stale` is release-blocking
(plan §5 CKPT-6: 90% gate).

**`specificity_judge`** (LLM, ref-based) — *does the level of detail match the requested audience?*
What it is: a judge compares the answer with the gold for the required audience (`lay` vs `phd`).
Measures: density calibration — neither too vague nor too dense.
Why (FP6): aims directly at the Feature C product.

**`answer_relevance`** (LLM, ref-free) — *does the answer address the question asked?*
What it is: a judge evaluates the answer's relevance/on-topic-ness vs. the question (RAGAS-style).
Measures: drift — a correct answer but about something else.
Why: cross-cutting monitoring metric (CKPT-8 uses it in production).

### 3.3 Formats and aggregates

**`format_validator`** (code, ref-based) — *does the output obey the requested format?*
What it is: deterministic validation by type — markdown table (right columns), JSON
(exact keys), parseable CSV, YAML, exact number of bullets/words/sentences + citation.
Measures: adherence to the format instruction (no judge — pure code).
Why (FP5): Features B/D promise structured outputs (comparison tables).

**`f1_summary_evaluator`** (code, summary, ref-based) — *substantive vs. hallucinated aggregate.*
What it is: a summary evaluator (P5 of the course): computes an aggregate F1 between the set of
substantive answers vs. hallucinated answers/abstentions over the gold labels collected from the
whole experiment.
Measures: global balance between usefulness and safety in an experiment.
Why: a single reading of "substantive quality vs. risk" per experiment
configuration — used in the EXP-0-baseline report.

---

**Evaluator × slice matrix** (who runs where — from the §2b ground-truth map):

| Slice | Retrieval | Generation |
|---|---|---|
| `answerable` | recall, hit, MRR, precision | faithfulness, citation, relevance |
| `unanswerable` | (n/a — empty gold by design) | abstention, faithfulness |
| `multi-doc` known | recall, hit, precision | completeness, faithfulness, citation |
| `multi-doc` open | **skip until adjudication** | completeness, faithfulness, citation |
| `format` | n/a | format_validator |
| `persona` | recall (shared source) | specificity, faithfulness |
| `stale` | (empty gold pre-refresh) | abstention |

**Retrieval semantics (decision already documented in the review addendum / lesson L1):**
recall/hit/MRR count only gold labels; retrieved docs outside the gold are
**unjudged**, never automatically irrelevant — this eliminates pooling bias.
`precision_at_k` is only computable where the gold is non-empty. `open_topic` examples
(`retrieval_gold=pending-adjudication`) are **skipped** from the retrieval metrics until the
EXP-0 mini-pooling.

## 4. LLM judges — configuration

- Judge model: `GENERATION_MODEL` (gpt-4o-mini), `temperature=0`, structured output
  (JSON with `{"score": bool|float, "reason": str}`). Cheap; variance will be measured in
  EXP-0 round 2 (for that, the prompt must be deterministic).
- Each judge = 1 function `(question, answer, context, reference) -> EvaluationResult`.
- `citation_accuracy`: step 1 code (extracts `[2609.xxxxvN]` from the answer via regex),
  step 2 judge per citation (does the `arxiv_id` chunk support the claim?).

## 5. Internal mini-checkpoints (one commit each, you review between them)

| Batch | Content | Test |
|---|---|---|
| **0.6.1** | 4 code-based retrieval evaluators | pure offline tests (no LLM calls at all): slices of synthetic examples with controlled golds, covering edge cases (empty gold, open-topic skip, dedup, partial top-k) |
| **0.6.2** | `format_validator` + `f1_summary_evaluator` | offline tests; dogfood against the 10 golds of the `format` slice (must pass 10/10) |
| **0.6.3** | 6 LLM judges | tests with canary pass/fail cases on both sides (e.g. faithful answer × hallucinated answer) — ~12 LLM calls in total |

## 6. Out of scope

- Runner (`experiments/run_experiment.py`) → 0.7. Real run → EXP-0 (0.8).
- Formal judge calibration with human agreement (kappa) → 0.8, before freezing
  the baseline (uses your review labels).

## 7. Verification

1. Offline test suite: `python -m pytest tests/test_evaluators.py -q` (or a
   single script if you prefer not to add pytest to the project — decision below).
2. Each judge validated on a real pass case AND a real fail case (failure = wrong score is a bug).
3. `format_validator` dogfood: 10/10 of the `format` slice golds pass.
4. Zero traced judge calls outside the tests (cost control).

## 8. Open decision (small)

- **Tests**: add `pytest` as a dev-dependency, or keep the current pattern of inline
  validation scripts? I recommend pytest (the catalog will grow until CKPT-8).
