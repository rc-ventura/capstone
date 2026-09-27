# Golden Dataset Construction for RAG: synthetic-by-construction vs. human adjudication vs. judge calibration

**Context:** CKPT-0.5.2b (golden set, `multi-doc` slice). While reviewing how retrieval gold labels
were built (questions naming the required arXiv IDs), the question came up: *"this is synthetic —
in production you'd need human annotation of some examples and a calibrated LLM judge, right? What
does the literature say (e.g. RAGAS)?"* This document records the research and the design
conclusions, with **auditable quotes** for every claim.
**Date:** 2026-09-24
**Future intent:** (1) Add a mini-pooling human-adjudication step to EXP-0 (CKPT-0.8). (2) Formalize
the human spot-check as the judge-calibration set in CKPT-0.6. (3) Rely on CKPT-8's annotation queue
as the production mechanism that grows the golden set with real judgments.

---

## Mental Model: three layers, not one

```
┌─────────────────────────────────────────────────────────────────┐
│ GOLD LABELS FOR RAG EVAL — three complementary provenance layers│
├─────────────────────────────────────────────────────────────────┤
│ L1 SYNTHETIC-BY-CONSTRUCTION  — question generated FROM the doc │
│    gold retrieval label is perfect BY DESIGN (RAGAS-style)      │
│    but the query distribution is artificial                     │
├─────────────────────────────────────────────────────────────────┤
│ L2 POOLED HUMAN ADJUDICATION  — union of retriever top-k pools, │
│    humans judge ONLY the pool (TREC). Real queries, incomplete  │
│    judgments → metrics must tolerate unjudged docs              │
├─────────────────────────────────────────────────────────────────┤
│ L3 JUDGE CALIBRATION SET      — small (~100-200) human-labeled  │
│    examples used to ALIGN the LLM judge (ARES PPI, LangSmith    │
│    Align Evals) — humans calibrate the judge, not the dataset   │
└─────────────────────────────────────────────────────────────────┘
```

| Layer | What it gives | What it costs | What it doesn't give |
|-------|---------------|---------------|----------------------|
| L1 synthetic-by-construction | Perfect gold retrieval labels at scale | Cheap; LLM calls only | Realism of query distribution |
| L2 pooled human adjudication | Realistic queries, real relevance signal | Human time on the pool only | Completeness — unjudged docs ≠ irrelevant |
| L3 judge calibration set | Trustworthy automated judges | ~100-200 human labels | Dataset-wide ground truth |

---

## The research (claim → quote → source)

### 1. TREC pooling: golden sets were never exhaustively annotated — and metrics must tolerate that

> "For creating the qrels, TREC, CLEF and NTCIR all adopt a mechanism called *pooling*… Hence, despite
> the fact that pooling is a very efficient way of collecting relevant documents, qrels formed through
> pooling are possibly *incomplete*. That is, there may exist relevant documents within the document
> collection, which none of the participating systems managed to retrieve and therefore are outside
> the qrels (Voorhees 2002)."
— Sakai, *On information retrieval metrics designed for evaluation with incomplete relevance
assessments*, Information Retrieval Journal (2008).
<https://link.springer.com/article/10.1007/s10791-008-9059-7>

> "For each search, the set of the (usually 100) top ranked documents is created. The union of the
> created sets forms the pool, say Pt, of documents to be relevance judged with respect to t."
— same article, describing NIST/TREC procedure.

> "We show that current evaluation measures are not robust to substantially incomplete relevance
> judgments. A new measure is introduced that is both highly correlated with existing measures when
> complete judgments are available and more robust to incomplete judgment sets."
— Buckley & Voorhees, *Retrieval Evaluation with Incomplete Information*, NIST.
<https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=150469>

**Design consequence:** if some docs judged relevant sit outside the gold labels, `precision@k` that
counts every non-gold doc as noise is *biased* — the classic pooling bias. Our `multi-doc` and
`answerable` evaluators should privilege recall/hit/MRR against gold labels and treat extra retrieved
docs as *unjudged*, not as errors (bpref-style tolerance). Full `precision@k` is only safe on
`unanswerable`/`stale`, where the gold is "nothing should match".

### 2. RAGAS testset generation is synthetic-by-construction — and documents its own limitation

> "Different types of queries requires different contexts to be synthesized. To solve this problem,
> Ragas uses a Knowledge Graph based approach to Test set Generation."
— RAGAS docs, *Testset Generation for RAG*.
<https://docs.ragas.io/en/latest/concepts/test_data_generation/rag/>

> "Synthetic questions may not reflect real user query patterns"
— Dataset card of a RAGAS-generated offline eval set (listed as an explicit caveat).
<https://huggingface.co/datasets/likhitjuttada/rag-offline-evalset>

**Design consequence:** synthetic generation is the standard bootstrap mechanism (our `answerable`
and `persona` slices), with a *documented* realism gap — exactly why our plan keeps 40+ hand-written
adversarial examples (`docs/plan.md` §8 risk: "Synthetic questions too easy").

### 3. MultiHop-RAG — the reference multi-doc RAG dataset — did NOT hand-label retrieval gold either

> "MultiHop-RAG: a QA dataset to evaluate retrieval and reasoning across documents with metadata in
> the RAG pipelines. It contains 2556 queries, with evidence for each query distributed across 2 to
> 4 documents."
— <https://huggingface.co/datasets/yixuantt/MultiHopRAG>

> "Using GPT-4 as a data generator, we then take an extensive procedure to construct a diverse set of
> multi-hop queries, each requiring the retrieval and reasoning over multiple documents."
— <https://github.com/yixuantt/MultiHop-RAG>

> "We detail the procedure of building the dataset, utilizing an English news article dataset as the
> underlying RAG knowledge base."
— Tang & Yang, *MultiHop-RAG* (arXiv:2401.15391) abstract.
<https://arxiv.org/abs/2401.15391>

**Design consequence:** our `multi-doc` gold (2–3 named paper IDs + LLM-free synthesis) follows the
published state of the art: construction-derived evidence labels, not human annotation. The
"known-item" weakness (named IDs make retrieval lexically easier) is a *conscious trade-off*, not a
mistake — and it mirrors the product's Features B/D, where users name papers A, B, C.

### 4. Judges are not a substitute for humans — they are calibrated BY a small human set

> "To mitigate potential prediction errors, ARES utilizes a small set of human-annotated datapoints
> for prediction-powered inference (PPI). Across eight different knowledge-intensive tasks in KILT,
> SuperGLUE, and AIS, ARES accurately evaluates RAG systems while using only a few hundred human
> annotations during evaluation."
— Saad-Falcon et al., *ARES* (NAACL 2024) abstract.
<https://aclanthology.org/2024.naacl-long.20/>

> "The quality of judge alignment depends entirely on the quality of your ground truth labels. In
> production scenarios, involve a principal domain expert… You don't need every example labeled - a
> representative sample (100-200 examples covering diverse scenarios) is sufficient for reliable
> alignment."
— RAGAS how-to, *Align LLM-as-Judge with human judgments*.
<https://docs.ragas.io/en/v0.3.7/howtos/applications/align-llm-as-judge/>

> "To build an evaluator you can trust for shipping decisions, you need systematic alignment to human
> corrections. Prompt iteration alone won't close the gap between a technically correct evaluator and
> a reliable one."
— LangChain, *How to Calibrate LLM-as-Judge with Human Corrections* (Mar 2026).
<https://www.langchain.com/resources/llm-as-a-judge>

> "An LLM judge is only useful when its scores reflect the decisions people would make for the same
> task… Treat alignment as something to measure for a specific judge configuration and target, not as
> an inherent property of the underlying model. Use a labeled golden set to develop the judge as you
> would any other measured system."
— TruLens, *LLM Judge Alignment*.
<https://www.trulens.org/component_guides/evaluation/llm_judge_alignment/>

**Design consequence:** humans don't label the whole test set — they label a *calibration set*. Our
planned 5-sample-per-slice human spot-check (`docs/plan.md` §8) is the seed of that set.

### 5. The production flywheel: traces → datasets → evals → better traces

> "Running these judges in production helps you build a data flywheel. Production traces feed your
> AI observability layer… Those insights inform datasets. Datasets power evaluations. Evaluations
> drive improvements. Improvements generate better new traces, and the cycle continues."
— LangChain, *How to Calibrate LLM-as-Judge with Human Corrections*.
<https://www.langchain.com/resources/llm-as-a-judge>

**Design consequence:** synthetic bootstraps the dataset; *production* corrects and expands it. Our
CKPT-8 annotation queue (`user 👎 ∪ online-eval low score`, `docs/plan.md` §5) is exactly this
flywheel — the "human annotation of some examples" intuition is architected into the capstone.

---

## How this lands in this project

| Literature layer | Where it lives here |
|---|---|
| L1 synthetic-by-construction | `answerable` (40, questions generated from chunks), `persona` (10), `multi-doc` known-item (10 — `build_golden_dataset.py`) |
| L1+L2 hybrid | `multi-doc` open-topic (5 questions without named IDs): retrieval gold intentionally **empty until adjudicated** in the EXP-0 mini-pooling |
| L2 pooled adjudication | **Planned for EXP-0 (CKPT-0.8):** for each golden question, run retrieval top-k, list unjudged candidates, human adjudicates relevant/not, promote judged-relevant docs into `gold_arxiv_ids`. A 250-paper TREC-pooling in miniature |
| L3 judge calibration set | The per-slice human spot-check (`docs/plan.md` §8) formalized as the judge-alignment set when `evaluators.py` lands (CKPT-0.6) |
| Production flywheel | CKPT-8 annotation queue + online evals (`docs/plan.md` §5) |

**Guardrail adopted from the pooling literature:** retrieval metrics on `multi-doc`/`answerable` count
gold-labeled docs only; extra retrieved docs are treated as *unjudged*, never auto-counted as
irrelevant. `precision_at_k` is reserved for `unanswerable`/`stale`, whose gold is "no doc should
match".

---

## Relation to next steps

- **docs/plan.md §2** — golden set design already mixes synthetic + hand-written adversarial
  construction; this lesson grounds *why* that mix is right.
- **CKPT-0.6 (`evaluators.py`)** — implement gold-only recall/hit/MRR semantics per the guardrail above.
- **CKPT-0.8 (EXP-0)** — add the mini-pooling human adjudication pass before freezing baseline labels.
- **CKPT-8** — annotation queue is the production incarnation of L2/L3.
