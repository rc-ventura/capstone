# Living Roadmap — CKPT-0.6.1b: fixing and hardening the metrics

> **Iterative document (version 4, 2026-10-06).** Rewritten at every discussion round; the current version is always this one.
> Writing rules: every metric cited comes with what it measures in parentheses when that is not obvious; every "design
> error" is explained in plain language, with a numeric example, **alternatives researched before the recommendation**
> and the decision recorded in the §5 log. P6 was decided on 2026-10-04, P7/P8 on 2026-10-06 (§6); P9 and later are **proposals for discussion**.

## 1. Current state

| Point | Status | Notes |
|---|---|---|
| P1 Shifted MRR (gold per paper) | ✅ done, committed (`907fa1a`) | `evaluators.py` |
| P2 precision = 0 (gold per paper) | ✅ done, committed (`907fa1a`) | same |
| P3 chunk × paper levels mixed (cause of P1/P2) | ✅ done, committed (`907fa1a`, oracle swap `a9c375a`) | helper `_ranked_and_gold`; `ir-measures` oracle in tests (L-3) |
| P4 `f1_summary_evaluator` with invalid format | ✅ done, committed (`907fa1a`) | signature + return `{"results": [...]}` |
| P5 hit@k ≡ recall@k with 1 gold | ✅ **implemented**, committed (`28dd93f`) | paper as the default level (D-1), `k`, `hit_at_ks`, `primary_retrieval_metrics` (D-2) |
| P6 `precision_at_k` with a 1-paper gold | ✅ **decided 2026-10-04; code + tests implemented, committed (`5faf69e`, `a2b7e30`)** | `is_gold_complete` (derived, no stored flag) + `precision_at_k` returns `None` unless the gold is complete (D-8). Deferrals D-10; pilot D-9/D-11 (§6) |
| EXP-0 pilot mini-pooling | 🟢 **unblocked — next** (CKPT-0.8) | protocol and decision rule in §6 (P6, "Decision and execution"); P7/P8 done (2026-10-06) |
| **D-7 baseline = one record per abstract** | ✅ **implemented** (2026-10-04, ADR-005) | `config.BASELINE`/`CHUNKED_500_0`, `utils.chunk_documents(…, 0, 0)`, new cache; golden builder pinned to the 500/0 arm |
| P7 abstention detection via `startswith` | ✅ **decided 2026-10-06 (D-5); implemented 2026-10-06** | `abstained()` — normalization + two marker families (16 paraphrase tests, 10 "know"-containing non-refusals); judge at 0.6.3; divergence in EXP-0; marker `REFUSE_*` stays a CKPT-6 experiment |
| P8 abstention as accuracy | ✅ **decided 2026-10-06 (D-6); implemented 2026-10-06** | `abstention_summary`: `abstention_recall` (gate) + `over_refusal_rate` (guardrail) + `abstention_f1` (diagnostic), per slice, raw counts + Wilson 95% CI; `abstention_accuracy` removed from `f1_summary_evaluator`; over-refusal tolerance fixed at EXP-0 vs the baseline CI |
| P9–P11 | 🟠 draft, **updated with the reading notes (fichamentos)** (§7) | discuss after P5–P8 |
| P12–P17 (new, arising from the reading notes) | 🆕 listed in §7 | triage pending |
| Reading notes | ✅ **15/15 written** | §4 |

Verification of the batch already done: `uv run pytest` → 35 green tests; MRR (Mean Reciprocal Rank: 1 ÷ position of the first
relevant document) from 0.167 → 1.0 and precision@k (fraction of the top-k that is relevant) from 0.0 → 0.2 in the cases that
reproduced the bug; 600 random cases agreed with the reference oracle (`ranx` at the time; now `ir-measures`, see L-3).

## 2. Metrics glossary (what each one measures)

| Metric | What it measures, in one sentence |
|---|---|
| **hit@k** (hit) | Did the top-k bring **at least one** correct document? It is 1 (yes) or 0 (no). In IR standards it is called *success@k*. |
| **recall@k** (coverage) | Of **all** correct documents, what fraction appeared in the top-k? (2 correct, found 1 → 0.5) |
| **MRR** (Mean Reciprocal Rank — "how close to the top the first correct one is") | 1 ÷ position of the first correct one. 1st → 1.0; 2nd → 0.5; 3rd → 0.33; 5th → 0.2; absent → 0. |
| **precision@k** (purity) | Of what was retrieved, what fraction is correct? (5 retrieved, 1 correct → 0.2) |
| **nDCG** (normalized Discounted Cumulative Gain — quality of the whole ranking) | Rewards correct documents at the top and accepts degrees of relevance. With 1 binary relevant document it equals 1/log₂(position+1): 1st → 1.0; 2nd → 0.63; 3rd → 0.5; 5th → 0.39. |
| **Hole@k** (hole rate) | Fraction of the top-k that **no one judged** (neither gold nor marked irrelevant). Measures how much of what the system retrieved is "in the dark". |
| **distinct_papers@k** (diversity) | Distinct papers ÷ k. 5 chunks of the same paper → 0.2. Measures redundancy, **not** relevance. |
| **abstention recall** (correct abstention) | Of the examples where the system **should** abstain, how many it abstained on. It is the plan's gate (≥ 90% on `unanswerable`). |
| **over-refusal rate** (refusing when it should answer) | Of the **answerable** examples, how many the system refused. Measures uselessness. |
| **under-refusal rate** (answering when it should abstain) | Of the **unanswerable** examples, how many the system answered anyway (= 1 − abstention recall). Measures hallucination risk. |
| **Classification F1** (balance between hits and false alarms) | Harmonic mean of the precision and recall of a class. |
| **token-F1** (word overlap) | F1 over the words of the generated answer × the reference answer. |
| **faithfulness** (fidelity to the context) | Does the answer only claim what the retrieved context supports? |
| **Cohen's κ** (agreement beyond chance) | How much two raters (e.g., LLM judge and human) agree after discounting agreement by luck. |
| **Wilson CI** (confidence interval for proportions) | Plausible range of the true rate given a small n. 10 hits out of 10 → [72%; 100%]. |

**Test oracle:** a reference implementation used only to check that ours gives the same result. It is not part of the product.

## 3. Why does the project use chunk IDs **and** paper IDs? (user's question 3)

**Corpus facts (measured in `resources/*.parquet`):** 250 papers, 843 chunks, from 2 to 5 chunks per paper
(mean 3.4), ~420 characters per chunk. Each paper is only the *abstract*, cut into pieces of ~500 characters.

**The retriever returns chunks**, not papers. What varies is **how the gold of each example was built**
(ADR-001, ADR-003):

| Slice | How the question was born | What we know for sure | Gold that can be recorded |
|---|---|---|---|
| `answerable` (40), `persona` (10) | An LLM read **one chunk** and generated the question (*synthetic-by-construction*) | Which **exact passage** answers it | `gold_chunk_ids` (1 chunk) |
| `multi-doc` known-item (10) | "Compare paper A and paper B" | Which **papers** — but not which chunk of each | `gold_arxiv_ids` (2–3 papers) |
| `multi-doc` open-topic (5) | Thematic question, without naming papers | Nothing yet | empty until human adjudication (EXP-0) |
| `unanswerable` (15), `stale` (10), `format` (10) | Outside the corpus / no retrieval | Nothing to retrieve | empty |

This is not an architecture choice: it is a consequence of two ways of building the dataset. The evaluators need to
compare **at the level where the gold exists**, and mixing the two levels caused bugs P1–P3.

**New reading after the reading notes (decision D-1):** the 1-chunk gold is, in fact, an **incomplete set of
relevant documents** — the other 2–4 chunks of the same paper may also answer. Therefore:
- metrics at **chunk level** are a **floor** (they only count the exact passage);
- metrics at **paper level** are a **ceiling** (any chunk of the paper counts).

**DECIDED (2026-10-03)** — measurements and reasoning in the lesson [`retrieval_unit_and_gold_granularity.md`](../learning-lessons/retrieval_unit_and_gold_granularity.md):
- **D-7: the baseline becomes one index record per abstract** (the whole abstract is the "chunk"). The 500-character chunking
  becomes an **experimental arm** (EXP-3: "does splitting the abstract help or hurt?"). Reason: the product cites papers and reads only abstracts;
  the median abstract has 255 tokens (fits comfortably in the embedding); the top-5 now always has 5 distinct papers.
- **D-1: the retrieval gold is the paper.** With one record per abstract, chunk = paper, so there is no "book + page" nor a list of
  chunks. For the 500-character configuration (EXP-3 arm) the evaluator keeps the **chunk level as a diagnostic** ("did it find the exact
  passage?"), never compared across chunking configurations (the `chunk_id` is a hash of the text).
- **Contingency (yours):** if the metrics are harmed by the golden set's bias (questions generated from 500-character
  chunks; the whole abstract lost a little on hit@1: 0.90 versus 1.00), **regenerate `answerable`/`persona` from the whole
  abstract** instead of from chunks.
- Left out: unjudged relevant documents in **other** papers (only the EXP-0 mini-pooling reveals them).

## 4. Reading notes (full reading of the papers) — 15/15 written

> **Related lessons:** [Retrieval Unit and Gold Granularity for RAG over Abstracts: Chunk vs. Paper vs. Whole Abstract](../learning-lessons/retrieval_unit_and_gold_granularity.md) (D-1, D-7, EXP-3) · [Golden Dataset Construction for RAG: Synthetic vs. Adjudication vs. Calibration](../learning-lessons/golden_dataset_construction_and_human_calibration.md) (L1/L2/L3).

Folder: `docs/research/fichamentos/`. All follow: reference · problem · method · contributions · numbers with
section/table · limitations · **reflections anchored in our plan** · confidence of the reading.

| Reading note | Confidence | Used in |
|---|---|---|
| [`yu-2024-rag-eval-survey.md`](../research/fichamentos/yu-2024-rag-eval-survey.md) | full (HTML; without Appendix 0.A) | metrics catalog |
| [`es-2023-ragas.md`](../research/fichamentos/es-2023-ragas.md) | full (HTML) | P9, judges |
| [`saad-falcon-2023-ares.md`](../research/fichamentos/saad-falcon-2023-ares.md) | full (HTML) | P11, calibration |
| [`ru-2024-ragchecker.md`](../research/fichamentos/ru-2024-ragchecker.md) | full (HTML; Tab. 16 and figures not seen) | D-1, P6, P9 |
| [`barnett-2024-seven-failure-points.md`](../research/fichamentos/barnett-2024-seven-failure-points.md) | full (v1) | FP→metric mapping (P13) |
| [`magesh-2025-hallucination-free.md`](../research/fichamentos/magesh-2025-hallucination-free.md) | full (**preprint v1**; published version not read) | P12, P7/P8, P11 |
| [`gao-2023-alce.md`](../research/fichamentos/gao-2023-alce.md) | full (v2) | P12 |
| [`zheng-2023-llm-as-judge.md`](../research/fichamentos/zheng-2023-llm-as-judge.md) | full (v4) | P11, P14 |
| [`wataoka-2024-self-preference.md`](../research/fichamentos/wataoka-2024-self-preference.md) | full (v2); Fig. 1b GPT-4 only | P11 |
| [`qa-eval-judge-vs-f1.md`](../research/fichamentos/qa-eval-judge-vs-f1.md) | full (body); appendices C–E of 2305.12421 not read | P9 |
| [`ir-pooling-incomplete-judgments.md`](../research/fichamentos/ir-pooling-incomplete-judgments.md) | full; Büttcher tables partial | D-1, P5, P6 |
| [`thakur-2021-beir.md`](../research/fichamentos/thakur-2021-beir.md) | full (v4) | P5, P6 |
| [`bassani-2022-ranx.md`](../research/fichamentos/bassani-2022-ranx.md) | **PARTIAL** (full paper blocked: poster, docs and code) | L-3 |
| [`abstention-benchmarks.md`](../research/fichamentos/abstention-benchmarks.md) | full (AbstentionBench, RefusalBench) + partial extras | P7, P8 |
| [`zhou-2023-ifeval.md`](../research/fichamentos/zhou-2023-ifeval.md) | full (v1); checker code not read | P10 |

## 5. Decision log

| ID | Date | Decision / question | Basis | Status |
|---|---|---|---|---|
| L-1 | 2026-10-03 | Always compare **at the gold level**; deduplicate papers preserving order | Bugs P1–P3; `decisions.md` (same paper 3× in the top-5) | Implemented |
| L-2 | 2026-10-03 | The summary evaluator follows the LangSmith contract: `(outputs, reference_outputs)` → `{"results":[{key,score}]}` | `langsmith` 0.13.0 (`extra="forbid"`) + course | Implemented |
| L-3 | 2026-10-03 | Test oracle for the retrieval formulas: **`ir-measures`** (wraps `trec_eval`; dev-only). `ranx` removed (22 extra packages vs 4; `uv.lock` −469/+27 lines). Oracle test extended to the `k` cutoff and to paper gold derived from chunk ids. | `bassani-2022-ranx.md` (partial) + empirical test below | ✅ **Decided and implemented** (user approved 2026-10-03; not an ADR: dev-only tooling) |
| L-4 | 2026-10-03 | Retrieval evaluators: **default level = paper** (`level="paper"`), `level="chunk"` as a diagnostic, `k` truncates the raw list before dedup | D-1/D-2 | Implemented |
| D-1 | 2026-10-03 | Retrieval gold **per paper**; exact chunk only as a diagnostic (config 500/0). No adjudication of siblings in EXP-0. | §3; measurements in the [lesson](../learning-lessons/retrieval_unit_and_gold_granularity.md) | ✅ **Decided** (implemented in P5; recorded in [ADR-006](../adrs/0006-retrieval-gold-granularity-and-metrics.md)) |
| D-2 | 2026-10-03 | Gold of 1 paper → report **hit@k + MRR** (+ hit@1/3/5 curve); gold of 2+ papers → **recall@k** (primary) + hit@k + MRR | P5 | ✅ **Decided** (`primary_retrieval_metrics`, `hit_at_ks`; recorded in [ADR-006](../adrs/0006-retrieval-gold-granularity-and-metrics.md)) |
| D-3 | — | Run BM25 in parallel with every new retriever to detect lexical bias of the synthetic gold? And/or put BM25's top-5 in the EXP-0 pool? | `thakur…` reflection 4 (analogy); NIST §6 (pool from more than one retriever type) | Proposal — **decide after the pilot** (D-9); no second retriever exists until CKPT-1 |
| D-4 | 2026-10-04 | `precision_at_k` only with complete gold; `distinct_papers@k` separately? | P6 | ✅ **Decided**: first part → D-8 (implemented); second part deferred to EXP-3 (D-10) |
| D-5 | 2026-10-06 | Abstention: **D — hybrid A+B**. A (heuristic `abstained()` with two marker families, tests: 16 paraphrases + 10 non-refusals containing "know") as the cheap floor; B (`abstention_quality` LLM judge, T=0, answer-only, validated on ~50 real stratified answers) as the ground truth at 0.6.3; EXP-0 reports the heuristic×judge divergence (target ≥ 90%). C (structured `REFUSE_*` marker) stays a CKPT-6 experiment — the baseline must measure the real prompt. | `abstention-benchmarks.md` reflections 3–4 (AbstentionBench C.3.1 rejected string matching; OR-Bench: cheap check ≈ judge only for predictable safety phrases); user approved 2026-10-06 | ✅ **Decided and implemented** (A implemented 2026-10-06; B at 0.6.3; divergence at EXP-0). Recorded in [ADR-007](../adrs/0007-abstention-detection-and-reporting.md) |
| D-6 | 2026-10-06 | Abstention gate = **B**: `abstention_recall` (the ≥90% gate on `unanswerable`) **with** a mandatory `over_refusal_rate` guardrail, per slice, raw counts + **Wilson** 95% CI; `abstention_f1` diagnostic only. Numeric over-refusal tolerance: defined at EXP-0 against the baseline's CI. Wilson, not Student's t: a proportion has no separate variance to estimate (σ² = p(1−p)), so the small-sample fix is inverting the score test, not widening with t(n−1) — see the P8 decision note in §6 | `abstention-benchmarks.md` reflections 1–2/5–8 (RefusalBench FRR/MRR trade-off r = −0.78; GPT-4o refused 14.6× more than needed; n=15 → one example = 6.7 p.p.); user approved 2026-10-06 (Wilson-vs-t rationale added the same day) | ✅ **Decided and implemented** (2026-10-06). Recorded in [ADR-007](../adrs/0007-abstention-detection-and-reporting.md) |
| D-7 | 2026-10-03 | **Baseline = one record per abstract**; 500/0 chunking becomes an EXP-3 arm. Measured: retrieval ≈ equal (same hit@5; 5 distinct papers vs 3.8), 2.6× input tokens, 2.2× cost, +8% latency. **Contingency:** if the golden set's bias (generated from chunks) harms the metrics, regenerate `answerable`/`persona` from the whole abstract | [lesson](../learning-lessons/retrieval_unit_and_gold_granularity.md) | ✅ **Approved and implemented** (2026-10-04); recorded in [ADR-005](../adrs/0005-retrieval-unit-whole-abstract.md) |
| D-8 | 2026-10-04 | `precision_at_k` is defined only where the gold is **complete**: `is_gold_complete` = multi-doc known-item ("relevant = the papers named in the question", by definition) **or** `review.state == "adjudicated"`; otherwise `None`. **Derived from existing metadata, not a stored flag.** | P6; NIST/Büttcher (unjudged ≠ irrelevant; treating it as irrelevant is a floor); ADR-001 (no two sources of truth); the dataset sync only creates missing examples, so a new field would not reach the 100 existing ones without a remote update | ✅ **Decided and implemented** (committed `5faf69e`). For adjudicated examples the gold is complete only **with respect to the judged pool** |
| D-9 | 2026-10-04 | EXP-0 mini-pooling scope = the 5 open-topic + a **pilot** (10 random `answerable`/`persona`, stratified 8+2, fixed seed, plus up to 4 `answerable` cases where the gold was not rank 1) ≈ 80 judgments, **not** all 50 single-gold examples (≈ 225). Known-item multi-doc is not pooled (D-8 definition). | Facts A/B/C below (§6); `ir-pooling…` (≤ ~200–250 judgments is feasible, so the argument is order of work, not feasibility) | ✅ **Decided** (protocol in §6; runs at CKPT-0.8) |
| D-10 | 2026-10-04 | `distinct_papers@k` **deferred to EXP-3** (always 5.00 on the whole-abstract baseline; no paper defines it, it is our own diagnostic). `Hole@k` **built together with the mini-pooling**, and the dataset must also store papers judged **irrelevant** (it keeps only positives today). | Measured: 5.00 in all three slices; BEIR §6/Tab. 4 defines Hole@10 (one dataset, 50 queries, top-10, single round); before judgments Hole@5 would be 0.8 on every 1-paper example | ✅ **Decided**. Deadline for `Hole@k`: before the first retriever comparison (CKPT-1) |
| D-11 | 2026-10-04 | Pilot decision rule, fixed **before** looking at the data. *x* = judged-relevant ÷ judged candidates in the 10-random sample: *x* ≤ 10% → keep the single gold; *x* ≥ 25% → judge the remaining 40 examples (pilot judgments are reused); 10–25% → review the cases and decide. | Engineering judgment, **no source gives these numbers**. Reference: *x* = 25% means one extra answering paper per question on average (true precision@5 ceiling 0.40 instead of 0.20) | ✅ **Decided** |

**L-3 note (empirical test done in a throwaway environment, without touching the project):**

| | `ranx` (installed today) | `ir-measures` + `pytrec-eval-terrier` |
|---|---|---|
| New packages | **22** (numba, llvmlite, matplotlib, seaborn, ir-datasets…) | **4** (ir-measures, numpy, pytrec-eval-terrier, scipy) |
| Installs on the project's Python 3.13 | yes | **yes** (tested: resolves and runs) |
| `success@k` vs `recall@k` with 1 gold | `hit_rate` ≡ recall | `Success@5 = R@5 = 1.0` in both tested cases (confirms P5 in practice) |
| `precision@k` | ÷ k | ÷ k (`P@5 = 0.2` with 1 hit) |
| Basis | poster: "tested against trec_eval"; paper not read | wrapper of `trec_eval` itself (community reference) |
Both agree with our formulas in the tested cases. Recommendation: **switch to `ir-measures`** (same guarantee
with ~5× fewer dependencies) and keep `ranx` out of the lock file (**done 2026-10-03**; the property test only changed its `_oracle` helper). Only `ranx` provides the paired
tests (t, Fisher, Tukey) — if you want statistical significance (P15), reassess.

### Corrections to existing documents (proposals; **only C-3, partly, applied on 2026-10-04**)

| # | Where | What is there | Correction | Source |
|---|---|---|---|---|
| C-1 | `docs/learning-lessons/golden_dataset_…` (§Design consequence) | "bpref-style tolerance" | bpref requires **judged** non-relevant documents (our gold only has positives) and overestimates systems outside the pool; the correct tolerance is "do not penalize unjudged" | `ir-pooling…` reflection 2 |
| C-2 | same lesson, L3 (line 161) | the human review of the golden set becomes the "judge calibration set" | The review judged **questions/gold**, not **generated answers**; calibrating judges requires labeling real answers (see P11) | `results/ckpt-0.5-review.md`; ARES; Zheng |
| C-3 | `docs/plan.md` / 0.6 §3.1 | FP3 = "noise"; `faithfulness`/`citation_accuracy` = FP4 | In Barnett, FP3 = consolidation limit (retrieved but did not fit in the context); FP4 = omission (answer in the context, not extracted). `precision_at_k` does not measure FP3 | `barnett-2024…`. **Partly applied (2026-10-04):** `precision_at_k` relabeled as top-k purity / a cause of FP4 in `docs/plan.md` §4, `ckpt-0.6-plan.md` §3.1/§3.3 and the docstring; the FP4 labels of `faithfulness_judge`/`citation_accuracy` are still pending. FP3 itself returns with the CKPT-4 reranker (measure "gold retrieved but not in the prompt") |
| C-4 | `docs/research/rag_failure_modes_review.md` | "15k docs"; "17–34%" | Barnett: 4,017 docs in §4.3 (§1 says 15,000: an inconsistency in the paper); Magesh v1: 17–33% | `barnett…`, `magesh…` |
| C-5 | roadmap §6 (old draft) and plan 0.6 | `ranx` "validated at ECIR/CIKM/SIGIR" | Only ECIR 2022 deals with evaluation; CIKM 2022 = fusion (`ranx.fuse`), SIGIR 2023 = run repository (`ranxhub`) | `bassani…` |
| C-6 | P9 draft / rationale | "token-F1 punishes the correct answer and rewards the wrong one" | The papers only show that it **underestimates correct answers**; 0.22/0.40/0.85 are Pearson means over 1,288 binary judgments of **extractive** QA | `qa-eval-judge-vs-f1.md` |
| C-7 | draft "ARES: 59.3 p.p." | 59.3 | The paper uses 59.3 (§1) and 59.9 (§5.1); the average in Tab. 1 gives 59.9 | `saad-falcon-2023-ares.md` |
| C-8 | plan 0.6 §4 (judge format) | JSON `{"score","reason"}` | The score comes **before** the justification, nullifying the effect of explaining first (Zheng); but "giving reasons" made GPT-3.5 worse in Wang → test locally | `zheng…`, `qa-eval…` |

**Findings from the `intro-to-langsmith` course (module 2):** the course's summary evaluator is a **classification F1** (TP/FP/FN),
with signature `(outputs: list[dict], reference_outputs: list[dict])`; ours computed token-F1. The course's judge is tested
with a pair of **opposite** meaning ("Yes… integrated" × "No… NOT integrated"). The course reads `outputs["output"]` and ours
reads `outputs["answer"]` → the 0.7 runner needs to return `answer`. The course has no retrieval metrics.

---

## 6. Rewritten proposals: P5, P6, P7, P8

> Format of each point: **(1) what is implemented · (2) the problem in plain language · (3) evidence ·
> (4) what the literature says (with reading note) · (5) alternatives · (6) recommendation · (7) steps · (8) what I need from you.**

---

### P5 — hit@k and recall@k are the same number when the gold has 1 document

**(1) Implemented.** `hit_rate` and `recall_at_k` in `evaluators.py`, computed separately and meant to be output as two columns.

**(2) In plain language.** hit@k asks "did the top-k bring any correct document?". recall@k asks "what fraction of the
correct documents appeared?". If there is only **one** correct document, the "fraction" can only be 0/1 or 1/1 — exactly the hit answer.
This is **not an arithmetic bug**; it is redundancy that can mislead the reading of the report (two columns "agreeing" that are actually
one).

| Situation (gold = 1 chunk) | hit@5 | recall@5 |
|---|---|---|
| Correct chunk in the top-5 | 1 | 1 |
| Correct chunk outside the top-5 | 0 | 0 |
| (multi-doc, 2 gold papers, found 1) | 1 | **0.5** ← only here do they differ |

**(3) Evidence.** `answerable` (40) + `persona` (10) = **50 of the 100 examples** have a 1-chunk gold. Also confirmed empirically
with `ir-measures`: `Success@5 = R@5 = 1.0` and both `0.0` with the gold absent. Subtle effect: since the gold is "this exact chunk", hit
becomes 0 even if the system brings **another chunk of the same paper** that answers — the metric measures "found the exact passage", not
"found useful evidence" (that is D-1).

**(4) Literature.**
- `thakur-2021-beir.md`: BEIR **does not define hit rate**; it discards precision/recall for being "rank unaware" (they do not consider position) and adopts nDCG@10. With 1 binary relevant document, nDCG@k = 1/log₂(position+1), that is, an MRR with a softer discount (agent's derivation, checked against the glossary table).
- `bassani-2022-ranx.md` (code) and `ir-measures`: name `success` and `recall` separately (trec_eval convention), although they coincide with 1 relevant document.
- `ir-pooling…` (B&V §4): with few relevant documents per query all measures become unstable; with 40–50 examples with a binary outcome the sampling noise is large. **No source recommends MRR for a single gold** — that is a field convention, not a result.

**(5) Alternatives.**
| | What it would do | Pros | Cons |
|---|---|---|---|
| A | hit@k + MRR on a single gold; recall only on multi-gold | no duplication; MRR brings the position | requires a rule "which metric is primary per slice" |
| B | **hit@1 / hit@3 / hit@5** curve | reports the effect of k (CKPT-1 experiment) almost for free | 3 numbers per example |
| C | nDCG@k instead of MRR | BEIR standard | redundant with MRR on a binary 1-item gold; only worthwhile with graded relevance |
| D | expand the gold (adjudicated siblings, D-1) | recall becomes informative again and the "floor" rises | human cost (~200–250 judgments, estimate) |
| E | keep both and just document | zero effort | misleading reading |

**(6) Recommendation (D-2): A + B now; D in parallel via D-1.** nDCG is left for when graded relevance exists (CKPT-4).

**(7) Steps (if approved).**
1. `hit_rate(..., k: int | None = None)`: truncates `ranked[:k]` before comparing (today it uses the whole top-k received). Same for MRR/recall for consistency.
2. Helper `primary_retrieval_metrics(reference_outputs) -> list[str]`: `["hit", "mrr"]` if `|gold| == 1`; `["recall", "hit", "mrr"]` if `|gold| ≥ 2`. The runner (0.7) uses this to decide what to report per slice.
3. Docstrings and `docs/plan.md` §3: "recall and hit only diverge with |gold| ≥ 2".
4. Tests: recall≡hit identity with a single gold; divergence with 2 golds; `hit@1/3/5` on known lists. The oracle (L-3) covers `Success@k`.

**(8) What I need from you:** do you agree with A+B? Do you want the hit@1/3/5 curve or only hit@5?

**Decision and execution (2026-10-03).** D-2 approved (with D-1 and D-7, see §3 and §5). Implemented in `evaluators.py` (**not committed**):
- `level="paper"` (default) and `level="chunk"` (diagnostic) in `hit_rate`, `recall_at_k`, `mrr`, `precision_at_k`; optional `k` (truncates the raw top-k before dedup);
- `hit_at_ks(..., ks=(1, 3, 5))` — the hit@1/3/5 curve;
- `primary_retrieval_metrics(reference_outputs)` — 1 gold paper → `["hit", "mrr"]`; 2+ → `["recall", "hit", "mrr"]`; no gold → `[]`.
- Tests: 46 green (includes the real "checkpoint handoff" case: MRR 1.0 at paper level and 0.5 at chunk level) and the `ir-measures` oracle (see L-3), now also covering the `k` cutoff, the `hit_at_ks` curve and paper gold derived from chunk ids, and `level="chunk"` for the diagnostic.
- Checked on the real golden set (baseline retriever, k=5): `answerable` (n=40) hit@1/3/5 = 1.000, MRR 1.000, exact-chunk diagnostic 0.975; `persona` (n=10) same, with diagnostic 1.000; `multi-doc` (n=10) recall@5 0.667, hit@1/3/5 = 0.70 / 0.70 / 0.90, MRR 0.740.
- **Effect on interpretation:** in `answerable`/`persona` hit@k is already **saturated at 100%** on the baseline: these slices do not discriminate retrieval in EXP-0 (the curve is only informative in `multi-doc`, `deep-hit` and after the index change).

---

### P6 — `precision_at_k` with a 1-chunk gold: ceiling of 1/k and penalizes what was not judged

> **Note (2026-10-03, corrected 2026-10-04):** with the whole-abstract baseline (D-7) there are no sibling chunks, so the **sibling** part of P6 disappears in the baseline. The other part does **not**: with a 1-paper gold the other papers of the top-5 are still unjudged, so `precision@5` still cannot exceed 0.2. The earlier note said the remaining scope was only `multi-doc` and the 500/0 arm; that was too optimistic. Closed on 2026-10-04: see "Decision and execution" at the end of this section.

**(1) Implemented.** `precision_at_k` runs on any example with a non-empty gold and counts everything outside the gold as an error. The
docstring admits the restriction ("meaningful only where the gold is the complete relevant set"), but the code does not enforce it.
After P1–P3 the denominator at paper level is the number of **distinct** papers retrieved.

**(2) In plain language.** Precision asks "of what the system brought, how much was worth having?". But we only know that **1 chunk**
was relevant — the other 4 of the top-5 may be relevant too (3.4 chunks per paper on average; `decisions.md` already recorded the
same paper 3× in the top-5). Counting them as "errors" is treating **"unjudged" as "irrelevant"** — and with a 1-chunk gold the maximum
possible score is 1/5 = 0.2, even with perfect retrieval.

**(3) Evidence.** Reproduced: 1-chunk gold with a hit at 1st → `precision = 0.2` (the maximum possible). Today plan 0.6 §3.1 still says
precision measures "noise (FP3)"; Barnett defines FP3 differently (see C-3).

**(4) Literature.**
- `ir-pooling…` (NIST 2007 §7; Büttcher 2007 §5.2): treating unjudged as irrelevant gives a **lower bound** (conservative); **ignoring** unjudged documents (bpref, P@k(j), condensed lists) **is not neutral** and overestimates. Quote: "Ignoring the unjudged documents… assumes… the same proportion of relevant… — an assumption that is simply wrong."
- `thakur-2021-beir.md` (§6, Tab. 4): when the system retrieves something **without a judgment**, the score drops through no fault of its own, and the effect is uneven (BM25 +0.012 vs ANCE +0.081 when annotating the holes). BEIR's solution: **annotate the holes** and measure Hole@10.
- `barnett-2024…`: FP3 = "retrieved but did not make it into the context" (consolidation limit) — with k=5 chunks of ~420 characters it is practically nonexistent in our system. **`precision_at_k` does not measure FP3.**
- `ru-2024-ragchecker.md`: *context precision* by **content** (a chunk is relevant if it supports some claim of the reference answer), not by ID — it works around sibling chunks and ID breakage from re-chunking, **but RAGChecker's diagnostic metrics were not validated with humans** in the paper.
- `saad-falcon-2023-ares.md`: treats passages from the **same document** as hard negatives — exactly the premise that our corpus of fragmented abstracts refutes.

**(5) Alternatives.**
| | What it would do | Pros | Cons |
|---|---|---|---|
| A | `precision_at_k` only with a **complete gold** (flag `gold_complete`); `None` otherwise | honest; follows NIST/Büttcher | the metric disappears in `answerable` |
| B | `distinct_papers@k` as a separate metric (diversity) | measures the observed collapse (same paper 3×) without pretending to measure relevance | does not say whether the content is worth having |
| C | **judge the holes** (Hole@k) in EXP-0 and only then compute chunk precision | is what the literature recommends; raises the gold floor | human cost |
| D | context precision by content (RAGChecker/RAGAS) | independent of ID and of chunking | LLM cost; not validated on humans; would use gpt-4o-mini = generator |
| E | keep as is | zero effort | misleading metric |

**(6) Recommendation (D-4): A + B now; C in EXP-0; D only as a diagnostic for the 5 `open_topic` examples** (which have no gold until adjudication).
In addition: **rename the metric's role** in the plan from "noise/FP3" to "top-k purity (only with a complete gold)".

**(7) Steps.**
1. `build_golden_dataset.py` writes `reference.gold_complete` (`True` only for multi-doc known-item); idempotent migration of the 100 examples.
2. `precision_at_k` returns `None` if `gold_complete` is false/absent.
3. New `distinct_papers_at_k(retrieved_chunk_ids, retrieved_arxiv_ids)` (pure, no gold).
4. `hole_rate_at_k(ranked, judged_ids)` (prepared; used in EXP-0).
5. Short ADR (**renumbered 008 — ADR-007 was taken by abstention, 2026-10-06**): "precision semantics" + correction C-3 in the plan. Tests: 1-chunk gold ⇒ `None`; known multi-doc ⇒ a value; same paper 3× ⇒ `distinct_papers@5 = 0.6`.

**(8) What I need from you:** (i) do you agree with renaming precision's role? (ii) do you accept judging the holes in EXP-0 (D-1/C)? (iii) does diversity (B) become an official metric or only a diagnostic?

**Decision and execution (2026-10-04).**

*The problem, restated.* The synthetic gold of `answerable`/`persona` came from generating a question out of one chunk, so the gold is that one source paper. With the whole-abstract baseline (D-7) the sibling-chunk case is gone, but the gold can still be incomplete: other papers may also answer the question and nobody judged them. `precision_at_k` reads "not in gold" as an error, so with a 1-paper gold and k=5 it cannot exceed 1/5 = 0.2. If a fraction *x* of the other four papers does answer, the true ceiling is (1 + 4*x*) / 5: *x* = 10% → 0.28; 25% → 0.40; 50% → 0.60 (own arithmetic). The anti-pooling-bias rule ("unjudged ≠ irrelevant") already protected hit/MRR/recall, which never penalize extra papers; `precision_at_k` was the one metric that broke it. The plans also contradicted themselves ("precision only in `unanswerable`/`stale`", whose gold is empty and therefore skipped).

*Three facts that ground the decisions (what each one is):*
- **A — observation, bounds the stakes.** Baseline hit@5 is 100% on `answerable` (40/40) and `persona` (10/10) (lesson, whole-abstract table). An incomplete gold therefore cannot hurt hit@5 there; it can only distort the position metrics hit@1 (0.90) and MRR (0.934), e.g. when another valid paper is ranked above the gold. The reason for the 100% is a hypothesis, untested (questions generated from the text itself share its vocabulary; ADR-006 trade-offs). Its value would change if the questions were regenerated (C).
- **B — structural limit of the method.** Candidates for the mini-pooling come from the baseline's top-5, so a paper that only another retriever would find is never shown to the judge and never enters the gold; the baseline's own picks all get judged. BEIR Tab. 4 shows the effect after judging the holes: nDCG@10 ANCE 0.654 → 0.735 vs BM25 0.656 → 0.668. No choice of scope removes it; only the source of the candidates does (BM25 in the pool, D-3, or judging each new retriever's holes at CKPT-1).
- **C — sequencing constraint.** A judgment is "this candidate answers *this* question". Regenerating `answerable`/`persona` (the D-7 contingency, open) voids the judgments, so judge a small sample before judging everything.

*Implemented (2026-10-04; committed in `5faf69e` + `a2b7e30`).*
- `evaluators.is_gold_complete(reference_outputs, metadata)` and `precision_at_k(..., metadata=None)`, which returns `None` unless the gold is complete (D-8). Without `metadata` the result is `None` (safe default).
- Tests: 62 green (new: `None` for a 1-paper gold, `None` without metadata, defined for known-item and adjudicated, `is_gold_complete` rules; the `ir-measures` oracle now passes adjudicated metadata).
- Text corrections: `docs/plan.md` §4 evaluator catalog, `ckpt-0.6-plan.md` §3.1/§3.3, C-3 row (partly), the note above.
- **Changed from steps (7):** the completeness flag is derived, not stored (no migration of the 100 LangSmith examples; D-8). Steps 3–4 (`distinct_papers@k`, `hole_rate_at_k`) are deferred (D-10). Step 5 (ADR-**008**; renumbered because [ADR-007 was taken by abstention](../adrs/0007-abstention-detection-and-reporting.md), 2026-10-06) is still pending: write it together with the adjudication data model (judged-relevant and judged-irrelevant ids), after the pilot.

*Pilot protocol (CKPT-0.8; decisions D-9, D-11).*
1. Candidates for a question = the baseline's top-5 (whole abstract, k=5) minus the gold; for open-topic all 5 (no gold yet).
2. Sample: the 5 open-topic (≈ 25 judgments); 10 random `answerable`/`persona`, stratified 8+2, fixed seed (≈ 40); up to 4 `answerable` cases where the gold was not rank 1 (≈ 16). Total ≈ 80 judgments (own estimate: 4 candidates per single-gold example).
3. For each candidate you read the question and the abstract and mark "answers" / "does not answer". Relevant ones go to `gold_arxiv_ids`; irrelevant ones are stored too (needed by `Hole@k`); the example becomes `review.state = adjudicated`.
4. Report separately: *x* from the 10 random examples (with a Wilson interval; with 40 judgments the interval is wide, e.g. 4/40 ≈ 4%–23%, own calculation, and the candidates of one question are not independent); the 4 rank>1 cases (is the hit@1 loss real or an artifact of the incomplete gold? it does not tell whether the cause is the whole-abstract index or the question-generation bias); open-topic on their own.
5. Apply the rule fixed in D-11. If the rule says "judge the remaining 40", the pilot judgments are reused.
6. D-3 (BM25 in the pool) is decided after the pilot: if almost no baseline candidate answers, that is a sign (inference, not measured) that these questions have few relevant papers beyond the gold, which lowers the weight of the pool bias.
7. `Hole@k` is built with these judgments and must exist before the first retriever comparison (CKPT-1). Each new dense retriever may need its own holes judged: BEIR measured 6.4% (BM25) to 31.8% (TAS-B) of the top-10 without judgment; applied to our top-5 that is 0.3–1.6 papers per question, i.e. about 15–80 judgments per 50 questions per retriever (own arithmetic; different dataset, may not transfer).

---

### P7 — Detecting abstention via `startswith`

**(1) Implemented.** Inside `f1_summary_evaluator`: `answer.strip().startswith(("I don't know", "No papers found"))`.

**(2) In plain language.** The evaluator only recognizes a refusal if the answer **starts** with one of two exact phrases. But the prompt
(`app.py`, `PROMPT_V1`) only says "say you don't know", without fixing a phrase, and **never** instructs "No papers found" (that phrase only exists in the `stale` reference answer).
Any paraphrase ("I'm sorry, the context doesn't say…") counts as **not** abstaining.

**(3) Evidence.** Reproduced: `"I'm sorry, the context doesn't say."` → `abstention_accuracy = 0.0` when it should be 1.0.

**(4) Literature.**
- `abstention-benchmarks.md` (AbstentionBench, Meta FAIR 2025): defines abstention **broadly** (not answering directly, expressing uncertainty, caveats, partial answer); it spent an appendix justifying **not** using string matching and adopted an **LLM judge (Llama 3.1 8B, T=0)** validated on **300** annotated, stratified pairs (88% accuracy).
- Same note (RefusalBench): eliminates the ambiguity at the source — the model emits **only** a `REFUSE_*` code; a judge merely reads it.
- Same note (OR-Bench, partial reading): keyword matching diverges ≤ 2.4% from a GPT-4 judge — **but on safety refusals**, with standardized phrases; it does not hold for an open-ended "I don't know".
- `magesh-2025…`: distinguishes an **informative refusal** (explains why) from a **default refusal**; specialized human labeling.

**(5) Alternatives.**
| | What it would do | Pros | Cons |
|---|---|---|---|
| A | robust heuristic (set of paraphrases + normalization) | free, deterministic, testable offline | never covers everything |
| B | **LLM judge** (`abstention_quality`, 0.6.3) as ground truth, validated on a stratified human sample | it is the benchmarks' standard | cost; needs its own validation (their 88% does not transfer) |
| C | structured marker in the prompt (`REFUSE`/JSON) | eliminates ambiguity | **changes the product**: the baseline must measure the real prompt |
| D | hybrid A + B | A as a cheap baseline and as a divergence measure | two implementations |

**(6) Recommendation (D-5): D — B as ground truth, A as a baseline/cheap guard; C only as a CKPT-6 experiment.** The judge should see only the answer
(first test whether passing `should_abstain` to the judge biases it); T=0.

**(7) Steps.**
1. `abstained(answer) -> bool` (pure function): normalization (case, typographic quotes) + patterns ("don't know", "do not know", "cannot find", "no papers found", "not in the context"…); test table with ≥ 15 refusal paraphrases and ≥ 10 non-refusal answers that contain "know".
2. `f1_summary_evaluator` starts using `abstained` (minimal change).
3. In 0.6.3, an `abstention_quality` judge with structured output; **your** labeling of ~50 real, stratified answers (`should_abstain × prediction`) validates the judge (together with the P11 calibration).
4. The EXP-0 report shows the **heuristic × judge divergence** (target ≥ 90% agreement).

**(8) What I need from you:** do you accept D? How many real answers are you willing to label (the literature uses 300 pairs; I proposed ~50 just for abstention, within a larger sample in P11)?

**Decision and execution (2026-10-06; D-5).** Approved: **D — hybrid A+B; C stays a CKPT-6 experiment**; ~50 real answers for the judge's calibration. Implemented (2026-10-06): `abstained(answer)` with normalization (case, typographic apostrophes, whitespace) and **two marker families**: first-person ("i dont know", "i cannot find"…) and source-negative ("no papers found", "not in the context", "isnt in the"…). Design detail that the tests pin: "dont know" alone is **not** a marker, so an answer reporting the *paper's* uncertainty ("the authors dont know whether…") stays an answer. Tests: 16 refusal paraphrases (including the reproduced evidence "I'm sorry, the context doesn't say."), 10 non-refusals containing "know", the documented first-person-hedge false positive, and empty/None inputs. **Changed from steps (7):** step 2 ("f1_summary_evaluator starts using abstained") became moot — P8 step 2 moved abstention out of `f1_summary_evaluator` entirely, so `abstained()` is consumed by `abstention_summary`. Steps 3–4 (judge at 0.6.3; divergence report in EXP-0) remain scheduled. **Scope:** `abstained()` is **English-only** (the golden set is English; `PROMPT_V1` fixes no answer language — a non-English refusal reads as an answer). Not fixed by translating the marker list (the original trap on a new axis); the answers are the judge (natively multilingual) or a prompt-level language contract (CKPT-6). Pinned by `test_abstained_is_english_scoped`; recorded in ADR-007 trade-offs.

---

### P8 — Abstention measured as accuracy (and not as recall + over/under-refusal)

**(1) Implemented.** `abstention_accuracy = hits / total` over `should_abstain`, alongside token-F1 in `f1_summary_evaluator`.

**(2) In plain language.** There are **two opposite errors**, and accuracy mixes them into a single number:
- **under-refusal** (answering when it should abstain): answering something that was **not** in the corpus → hallucination risk;
- **over-refusal** (refusing when it should answer): refusing something **answerable** → the product becomes useless.
The plan's gate ("correct abstention ≥ 90% on `unanswerable`") is, in practice, the **abstention recall**. But recall alone is **gameable**.

**(3) Evidence (with the numbers from our core golden set, 100 examples).** 25 should abstain (`unanswerable` 15 + `stale` 10); 75 should not.
| "Dumb" system | Accuracy | Abstention recall (gate) | Over-refusal |
|---|---|---|---|
| Never abstains | **75%** (looks good) | **0%** | 0% |
| Always abstains | 25% | **100%** (passes the gate!) | **100%** |
With n=15 at the gate, 14/15 = 93.3% passes and 13/15 = 86.7% fails: **a single example** decides.

**(4) Literature.**
- `abstention-benchmarks.md` (AbstentionBench): reports **recall (primary)**, precision and F1; it focuses on recall because precision ≈ 1 in their models. It supports recall as the gate, **not** F1 as the primary metric.
- Same note (RefusalBench): separates **False Refusal Rate** (= over-refusal) and **Missed Refusal Rate** (= under-refusal; acronym "MRR" in the paper — **do not confuse** it with Mean Reciprocal Rank), plus **Refusal Detection F1**; shows the trade-off (correlation −0.78) and that GPT-4o refuses 62.8% of the answerable and lets 4.3% of the unanswerable through. No frontier model exceeds 73% refusal accuracy **with** the right category. They propose the **CRS** (simple mean of two accuracies) — the note recommends **not** adopting it as the gate, because it hides the trade-off.
- `yu-2024-rag-eval-survey.md`: the survey's "Rejection Rate" is **one-sided** (only one side).
- `magesh-2025…`: a refusal counts as "incomplete", not as a hallucination.
- No source discusses **small samples**: the decision on CI (Wilson) is **an engineering one**, not from the literature.

**(5) Alternatives.**
| | Metric | Pros | Cons |
|---|---|---|---|
| A | accuracy (current) | simple | mixes the two errors; biased by the 75/25 imbalance |
| B | **abstention recall (gate) + over-refusal as a guardrail + diagnostic F1, per slice** | aligned with AbstentionBench/RefusalBench; separates the risks | 3 numbers |
| C | full RefusalBench (FRR, Missed Refusal Rate, Detection F1, CRS, category) | richer | categorization is not a requirement; CRS hides the trade-off |
| D | single number (CRS) | easy to compare | hides the trade-off (the note advises against it) |

**(6) Recommendation (D-6): B**, using names derived from RefusalBench (`abstention_recall`, `over_refusal_rate`, `abstention_f1` diagnostic),
**per slice** (`unanswerable` × `stale`; AbstentionBench's concept of "stale" is a different one — see `abstention-benchmarks.md` reflection 6) and **always with raw counts + Wilson CI**.
Proposed gate: `abstention_recall ≥ 90%` on `unanswerable` **and** `over_refusal_rate` not worsening beyond the baseline's CI (numeric threshold: to be defined with you).

**(7) Steps.**
1. New summary evaluator `abstention_summary(outputs, reference_outputs, examples)` (the `examples` argument gives access to `metadata.slice`), returning `{"results": [...]}` (L-2 contract): `abstention_recall`, `over_refusal_rate`, `abstention_f1` — total and per slice.
2. Uses `abstained()` from P7; `f1_summary_evaluator` loses the `abstention_accuracy` key (only token-F1 remains, which P9 reassesses).
3. Counts (`n`, `k`) and Wilson CI in each result's `comment`.
4. Tests with synthetic confusion matrices (TP/FP/TN/FN), including the two "dumb" systems from the table.
5. Entry in `decisions.md`: regression rule "a recall gain cannot come from an increase in over-refusal".

**(8) What I need from you:** (i) do you accept B as the design? (ii) what tolerance for worsening of over-refusal (e.g., ≤ 1 example? ≤ 5 p.p.?) (iii) do you agree to keep F1 only as a diagnostic?

**Decision and execution (2026-10-06; D-6).** Approved: **B**, with F1 as a diagnostic; the numeric over-refusal tolerance is **defined at EXP-0 against the baseline's CI** (fixing a number before a baseline exists would be arbitrary — the rule is "must not worsen beyond the baseline's confidence interval"). Implemented (2026-10-06): `abstention_summary(outputs, reference_outputs, examples)` reports, total and per `metadata.slice` (suffix `__<slice>`, via langsmith's supported `examples` arg), with raw counts + Wilson 95% CI in each result's `comment`:
- `abstention_recall` — the gate (≥ 90% on `unanswerable`);
- `over_refusal_rate` — the guardrail; regression rule recorded in `decisions.md`: *a recall gain cannot come from an increase in over-refusal*;
- `abstention_f1` — diagnostic only.
`f1_summary_evaluator` lost its `abstention_accuracy` key (token-F1 remains there; P9 reassesses it). The LangSmith contract is pinned by a test (`EvaluationResult` extra="forbid", the two "dumb systems" of the evidence table — the one that refuses everything now shows recall 100% **with** over_refusal 100%).

**Wilson vs Student's t (author's question, 2026-10-06).** They answer the same worry — with n ~ 10–25 the naive interval (p̂ ± 1.96·se, "Wald") is too narrow — but they fix *different* problems, because the data type differs. Student's t corrects the interval for a **mean of continuous data** when the variance is unknown and must be estimated from the sample itself; the extra variability of σ̂ makes the tails heavier (df = n−1), and the interval widens. A **proportion has no separate variance to estimate**: the variance is a function of the very parameter (σ² = p(1−p)), so there is nothing for the t-curve to correct — applying t to a binomial proportion is a category error (approximations exist; none is standard practice). The correct small-sample fix is what Wilson does: **invert the score test** — instead of plugging p̂ into the standard error, solve for which p values are compatible with the observed k/n. This also keeps the interval inside [0, 1] and stays honest for extreme p (0/15, 15/15), where Wald produces nonsense: at 14/15 it returns [81%; 106%] — a precision above 100%. Sanity anchors: Wilson(10/10) = [72%; 100%], Wilson(14/15) = [70%; 99%] — one example swings the point estimate by 6.7 p.p., which is *why* the CI travels with every number. Caveat shared by Wilson, Wald and any binomial interval: the trials are assumed independent, and candidates within one question are not (§6, pilot note 4) — the CI is honest about sampling noise, not about clustering.

---

## 7. What the reading notes changed in P9–P11, and new points

### Update of the P9–P11 drafts (discussion after P5–P8)
- **P9 (token-F1):** the evidence is **weaker** than I said (C-6). `ru-2024-ragchecker.md` (Tab. 2/5, Pearson with humans): claim-level metric 61.93; RAGAS answer-similarity 48.31; ROUGE-L 43.10; BLEU 35.14; BERTScore 33.51; human×human 70.09 (base judge Llama3-70B). `qa-eval-judge-vs-f1.md` (Wang 2023): the GPT-3.5 judge did **not** beat lexical comparison on long answers (69.5% vs 82.3% on BingChat). Provisional direction: a **locally calibrated** reference-based correctness judge + token-F1 only as a diagnostic.
- **P10 (format_spec):** `zhou-2023-ifeval.md` supports a structured spec, but the `instruction_id`+`kwargs` pair belongs to the **dataset**, not to the paper; the `kwargs` are a "union" record that is almost entirely `None` → prefer atomic requirements with per-type parameters. The 8 transformations of the *loose* mode **do not** fit (removing the 1st/last line breaks a table/CSV/quote header). With n=10, the Wilson CI of 10/10 = [72%; 100%]: the `format` slice is a **regression test**, not a rate estimate.
- **P11 (judge ≠ generator):** `wataoka-2024…`: GPT-4's bias = 0.520, tied to the text's **perplexity**, not to authorship; GPT-4/3.5 were left out of the perplexity analysis; **no one tested gpt-4o-mini**. `abstention-benchmarks.md` (RefusalBench): self-evaluation 91.0% vs 82.1% cross-evaluation; κ between judges as low as 0.061. `zheng-2023…`: self-enhancement observed (GPT-4 +10%, Claude-v1 +25%) but the authors **cannot conclude**. → separate `JUDGE_MODEL`, from **another family**.
  **New discovery (C-2):** the "100 human labels" reviewed the golden set, **not generated answers**. ARES uses ≥150 points labeling **system outputs** and shows that with 100–150 the discriminative power is limited (Tab. 3). To calibrate judges (κ) **you need to label real answers** after the baseline (~100–150, faithfulness + abstention + citation). Magesh gives a reference point: κ 0.77 on 48 items (legal domain).

### New points arising from the reading notes (triage pending)
| ID | Point | Source | Summary |
|---|---|---|---|
| P12 | `citation_accuracy` is incomplete | `magesh…`, `gao-2023-alce.md` | Covers only *misgrounded* (the citation does not support the claim). Missing: *ungrounded* (claim without a citation) and *fabricated* (ID not retrieved). ALCE: citation **recall** (is the claim supported by the citations?) and **precision** (is each citation necessary?) per sentence; human×ALCE κ 0.698 / 0.525. Abstention without a citation would give recall 0 → exempt it. Citation alone is gameable (with the "top-1 passage" shortcut correctness drops only 5 pts; what gives it away is fluency) → report it together with correctness. |
| P13 | FP→metric mapping | `barnett-2024…` | C-3. Barnett's evidence is weak (3 case studies, no counts per FP): use it as a **coverage taxonomy**, not as proof of prevalence. |
| P14 | Format and order of the judges' prompt | `zheng…`, `qa-eval…` | C-8. Test explain-before-scoring, include the gold answer (Zheng Tab. 4: 70% → 15% failures in math with reference-guided). |
| P15 | Statistical significance of the gates (≥15%/≥10%) | `ir-pooling…`, `bassani…` | B&V: δ of ~8–18% for 95% confidence with 50–100 topics and many relevant documents; with ~50 binary examples δ tends to be larger. `ranx.compare` offers t/Fisher/Tukey but **without correction for multiple comparisons**. Propose a paired CI/bootstrap. |
| P16 | Lexical bias of the synthetic gold | `thakur…` (analogy), `ir-pooling…` | Questions generated from the chunk share vocabulary with it; this may favor BM25/hybrid over dense in CKPT-1/4. **No source studies synthetic gold** — extrapolation. Mitigation: D-3 (BM25 in parallel) + paraphrased questions. |
| P17 | New RAGChecker diagnostics | `ru-2024-ragchecker.md` | *Context utilization* (found it but did not use it) ≈ accuracy conditioned on hit@k=1 vs 0 (cheap); *noise sensitivity*; *hallucination vs self-knowledge*; utilization/noise/faithfulness trilemma when optimizing the prompt (CKPT-6). Diagnostic metrics **not validated** with humans in the paper. |

---

## 8. Historical record: original plans P1–P4 (executed) and drafts P9–P11

> Kept as they were in version 1 for traceability. P5–P8 were **replaced** by §6; the old section
> "Scientific grounding (summaries)" was **replaced by the reading notes** (§4).
> Mentions of `ranx` as the oracle below are historical: it was swapped for `ir-measures` on 2026-10-03 (L-3).

## P1 + P2 + P3 — Normalize the comparison level (chunk × paper) — ✅ EXECUTED

### What is implemented
`evaluators.py` receives `retrieved_chunk_ids` and `retrieved_arxiv_ids` and the gold can be `gold_chunk_ids`
(slice `answerable`) **or** `gold_arxiv_ids` (multi-doc, persona). The four functions do
`gold = chunk_gold or arxiv_gold` and then treat the two lists of retrieved items as a single universe:
- `recall_at_k` / `hit_rate`: `set(chunk) | set(arxiv)` — works by accident (IDs of different formats do not collide).
- `mrr`: `enumerate([*chunk_ids, *arxiv_ids], 1)` — **concatenates** the two lists.
- `precision_at_k`: `retrieved = chunk_ids if chunk_ids else arxiv_ids` — picks the list by presence, not by the gold's type.

### Evidence of the error (reproduced offline)
```
ids=[a..e]; chunks=["a:h",...]; ref={"gold_arxiv_ids":["a"]}   # gold paper in 1st place
mrr(chunks, ids, ref)            -> 0.1667   (expected 1.0 — rank 6 instead of 1)
precision_at_k(chunks, ids, ref) -> 0.0      (expected 0.2 — compares chunk IDs against paper gold)
```
Coverage: `tests/test_evaluators.py` has `test_recall_uses_arxiv_ids_for_multidoc`, but
**no MRR/precision test with gold by `arxiv_id`** — that is why it passed.
Aggravating factor (decisions.md, 2026-09-27): the baseline top-5 returns the same paper up to 3×, so the rank
at paper level needs an **order-preserving dedup** (otherwise MRR/precision at paper level end up inflated/deflated).

### Solution (step by step)
1. Write the helper `_ranked_at_gold_level(retrieved_chunk_ids, retrieved_arxiv_ids, reference_outputs) -> tuple[list[str], set[str]] | None`:
   - non-empty chunk gold → `(retrieved_chunk_ids, gold_chunks)` (chunk level);
   - else non-empty paper gold → `(dedup_ordenado(retrieved_arxiv_ids), gold_papers)` (paper level, `dict.fromkeys` to preserve rank);
   - else `None`.
2. If the runner (0.7) passes only `retrieved_chunk_ids`, derive `arxiv_id` with `chunk_id.split(":")[0]` when `retrieved_arxiv_ids` comes empty (0.6 §2 contract documented in the docstring).
3. Rewrite the four metrics on top of the helper: recall = `|gold ∩ ranked| / |gold|`; hit = `any`; MRR = `1/rank` of the 1st gold in `ranked`; precision = `|gold ∩ ranked| / len(ranked)` (P6 refines the denominator).
4. New tests (offline, in `tests/test_evaluators.py`): MRR and precision with paper gold at 1st/3rd/absent; top-k with the same paper 3× (dedup); `arxiv_id` fallback derived from the chunk ID; mixed gold (chunk takes precedence).
5. **Independent oracle**: add `ranx` (or `ir_measures`) as a **dev** dependency and a property test that generates random runs/qrels and checks `recall/mrr/hit/precision` against the lib. Do not use the lib at runtime: LangSmith evaluators are called per example (1 query), and building `Qrels/Run` per call is overhead with no gain; the lib comes in as proof of correctness.

### Why it improves
- Removes the cause (mixed granularity), not just the two symptoms — any new metric (nDCG in CKPT-4) inherits the correct behavior.
- The multi-doc/persona baseline stops being underestimated ~6× in MRR; the rerank gate (CKPT-4) measures the real effect.
- The `ranx` oracle makes the defense simple: "implementation checked against the reference lib".

### Verification
`uv run pytest tests/test_evaluators.py` green; the two evidence commands return `1.0` and `0.2`; property test with ≥200 random cases agrees with `ranx`.

---

## P4 — `f1_summary_evaluator` with a return format invalid for LangSmith — ✅ EXECUTED

> **Correction (2026-10-03):** the original text below says the extra key is "ignored". Verified in `langsmith` 0.13.0: it is **rejected** (`EvaluationResult` has `extra="forbid"`), the runner catches the exception and **discards the whole evaluator**; moreover the signature `(results)` did not even pass the validation of `summary_evaluators=`.

### What is implemented
Returns `{"key": "f1_summary", "score": ..., "abstention_accuracy": ...}` (an extra key in a dict).

### Evidence of the error
Real test output: `{'key': 'f1_summary', 'score': 0.33, 'abstention_accuracy': 0.0}`. LangSmith summary evaluators
consume `key`/`score` (or `{"results": [...]}`); the extra key **does not become a metric** in the experiment — abstention would be
computed and silently discarded. (Confirm the exact format in the installed `langsmith` version via context7 before coding.)

### Solution
1. Read the docs for the installed version (`uv pip show langsmith`; context7) for the return format of summary evaluators with multiple metrics.
2. Split into **separate, single-responsibility summary evaluators**: `token_f1_summary` (P9), `abstention_summary` (P8) — or a single one returning `{"results": [{"key":..., "score":...}, ...]}` if that is the supported format.
3. A test that validates the *shape* (list/dict with a `key` and a numeric `score`) and, preferably, a minimal integration test with `langsmith.evaluate` over an in-memory dataset (no network, via `upload_results=False` if available).

### Why it improves
Each metric becomes a visible/comparable column in LangSmith — a requirement of `decisions.md` ("metric before → after"). Today abstention, which is release-blocking (90% gate), would not show up.

### Verification
Run the evaluator in a local experiment and see both metrics listed; shape test green.

---


## P9 — Token-F1 vs gold answer as a quality metric

### What is implemented
SQuAD-style (bag-of-words) `_token_f1(pred, ref)` comparing the generated answer with the gold answer, aggregated in `f1_summary`. The plan (`ckpt-0.6-plan.md` §3.3) describes something else ("F1 between noun × hallucinated"), so **code and plan diverge**.

### Evidence
"Agents hallucinate tools [2609.12345]." vs "LLM agents often hallucinate tool calls." (the correct answer) → **0.36**. Paraphrase is punished; the tokens of the citation `[2609.12345]` inflate the denominator; the prompt limits answers to 3 sentences, which makes F1 sensitive to length.

### Solution
1. Decide and fix the divergence: the plan now says "token-F1 = **diagnostic metric of lexical coverage**", not of quality.
2. Pre-process before comparing: remove citations (`\[\d{4}\.\d{4,5}v\d+\]`) and stopwords; rename `f1_summary` → `token_f1_summary`.
3. Quality now comes from **two semantic signals**: (a) a reference-based `correctness_judge` (LLM, answer vs gold answer; via `openevals.create_llm_as_judge` or an own prompt in 0.6.3) and (b) embedding similarity (`compare_semantic_similarity`, already cited in plan §2b) as a cheap alternative.
4. Calibrate against the golden set's human labels (100/100 reviewed): correlation (Spearman) token-F1 × human judgment vs correctness_judge × human — the result decides whether token-F1 leaves the main report.

### Why it improves
It swaps a metric that punishes correct answers for one that measures what the product promises; human calibration turns the choice into a defensible empirical decision.

### Verification
Normalization tests (citation removed, case, punctuation); correlation reported in `results/`.

---

## P10 — `format_validator` coupled to the question text

### What is implemented
`format_validator(question, answer)` dispatches by substring ("markdown table", "JSON array"…) and extracts columns/keys/counts with regex over the question (`_check_*`). An unknown format returns `{"score": 0, "reason": ...}` **without `key`**; there is dead code (`if "No other text" ...: pass` in `_check_json`).

### Evidence
Any question rewrite (e.g., `regenerated: v2` in `build_golden_dataset.py:635`, or the `stale`/persona of the ADR-004 axis) can break the parse and become `score 0` — a **false negative attributed to the system**. The "format not detected" fallback scores 0 instead of being excluded, contaminating the `format` slice.

### Solution
1. In `build_golden_dataset.format_examples`, write a **structured spec** in `metadata.format_spec`, e.g.: `{"type":"json_object","keys":["title","year"],"exact_keys":true}` / `{"type":"markdown_table","columns":[...],"rows":3}` / `{"type":"bullets","count":3,"max_words":12}`.
2. `format_validator(answer, format_spec)` validates only against the spec (no regex over the question); the question remains only for display.
3. Missing/unknown spec ⇒ `score=None` (skipped, like `is_retrieval_evaluable`) and always with `key="format_valid"`.
4. Idempotent dataset migration (the builder is already idempotent — add a metadata update on the 10 existing examples) + remove the dead code.
5. Keep `test_format_golds_dogfood_all_pass` (10/10) and add a "missing spec ⇒ None" case.

### Why it improves
It validates the contract the dataset declares, not an interpretation of free text; eliminates false negatives from rewrites and aligns with the IFEval standard ("programmatically verifiable instruction").

### Verification
Dogfood 10/10 with specs; a question-rewrite test does not change the result.

---

## P11 — Judge and generator use the same model (gpt-4o-mini)

### What is implemented
`config.GENERATION_MODEL` (`gpt-4o-mini`) generates the answers; the plan (`ckpt-0.6-plan.md` §4) defines the **same model** as the judge, `temperature=0`. `config.RERANK_MODEL` is also the same.

### Evidence
There is no numeric error to reproduce — it is a known methodological risk (self-preference/self-enhancement bias of LLM judges). The plan already foresees measuring variance between rounds and calibrating with kappa in 0.8, but does **not** separate the judge's model.

### Solution
1. New variable `JUDGE_MODEL = os.getenv("JUDGE_MODEL", ...)` in `config.py`, **distinct** from `GENERATION_MODEL`; default: a stronger model or one from another family (the user's decision, see open question).
2. All judges (0.6.3) read `JUDGE_MODEL`; record `judge_model` in the experiment metadata.
3. Calibration (0.8): judge×human kappa **per judge** on the 100 labels; acceptance criterion documented (e.g., κ ≥ 0.6) before freezing the baseline.
4. Sensitivity round: re-evaluate the same set of answers with 2 judges and report the difference in the EXP-0 report.

### Why it improves
It removes the most common objection to evals with an LLM judge ("the model evaluates itself") and makes the gates' deltas attributable to the system, not to the judge's bias.

### Verification
`config.JUDGE_MODEL` appears in the runs' metadata; calibration report with κ per judge.

---

## Done criteria for the batch
1. `uv run pytest` green, including the `ranx` oracle (P3) and the tests for P4/P7/P8/P10.
2. Reproduction of the 5 evidence commands returns the expected values (1.0, 0.2, valid shape, abstention by paraphrase, normalized token-F1).
3. `docs/plan.md` §3/§4 and `ckpt-0.6-plan.md` §3 updated to reflect the new semantics (recall×hit, precision, F1/abstention, token-F1).
4. ADR-008 (precision semantics; renumbered — ADR-007 was taken by abstention, 2026-10-06) and an entry in `docs/decisions.md` → Observations describing the correction of the IR bugs **before** EXP-0.
5. One commit per point (messages `fix(evaluators): ...`), without mixing with the uncommitted work of CKPT-0.6.

---

