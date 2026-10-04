# Retrieval Unit and Gold Granularity for RAG over Abstracts: Chunk vs. Paper vs. Whole Abstract

**Context:** CKPT-0.6.1b (retrieval metrics fix). While discussing D-1 ("is the gold the chunk or the paper?"), a product
observation came up: the user is a scientist, the system cites **papers** (`PROMPT_V1`: "cite the arXiv ID for each
claim"), and a 1-chunk gold looks like a design error — perhaps 1 chunk per abstract solves it. This document records what
was **measured on the project's data** (core golden set of 100 examples, baseline k=5, `text-embedding-3-small`, `gpt-4o-mini`),
what the 15 reading notes say (and do not say) about it, and the decisions that remain open.
**Date:** 2026-10-03
**Future intent:** (1) Close **D-1** (gold per paper as the primary level) before implementing P5/P6. (2) Treat the
**indexing unit** as a separate decision (**D-7**) and test it only after the judges exist, comparing answer quality,
cost and latency together. (3) Re-evaluate the design of **EXP-3** (chunk 500/0 × 1000/200), which currently depends on chunk IDs.

---

## Mental Model: three "units" that are not the same thing

```
 PRODUCT UNIT      what the user sees and what supports the answer  →  the PAPER (cited arxiv_id)
 RETRIEVAL UNIT    what the index stores and the retriever returns  →  today: CHUNK of ~500 characters
 GOLD UNIT         what the reference answer says is "correct"      →  today: 1 CHUNK (the one that generated the question)

 Real example — paper 2609.19636v1 split into 4 chunks (500/0), question generated from chunk 2:

   chunk 1 ─ "…An SFT checkpoint and an RL checkpoint are then scored… mixes two changes: where the"  ← ends mid-sentence
   chunk 2 ─ "agent arrives, and what it does… We introduce checkpoint handoff…"       ◄── GOLD (generated the question)
   chunk 3 ─ "policy arrives at a state the environment confirms…"
   chunk 4 ─ "SOLVE gaps predict… so long-horizon evaluation can report arrival and completion…"   (also covers the protocol)

   retriever top-5:  #1 chunk 4 (same paper)   #2 chunk 2 (GOLD)   #3–#5 other papers
```

A 1-chunk gold lists **one** correct document, not **all** the correct ones. This is what makes chunk-level a **floor** (it
only counts the exact passage) and paper-level a **ceiling** (any chunk of the paper counts).

| Metric level | What it measures | When it breaks |
|---|---|---|
| **Chunk** (floor) | Did it find the exact passage the question came from? | If the chunking changes, the IDs cease to exist; penalizes a sibling that also answers |
| **Paper** (ceiling) | Did it find the right paper, in any passage? | Does not distinguish finding the passage that answers from finding another passage of the paper |

---

## Corpus facts (measured on `resources/arxiv_text-embedding-3-small_500-0.parquet`)

| Item | Value |
|---|---|
| Papers / chunks | 250 / 843 (2 to 5 chunks per paper, mean 3.37) |
| Chunk size | ~421 characters on average |
| Whole abstract (chars) | min 703 · **median 1,429** · p90 1,820 · max 2,010 |
| Whole abstract (tokens, cl100k) | min 124 · **median 255** · p90 337 · max 412 |
| `text-embedding-3-small` limit | 8,191 tokens → the whole abstract fits in 1 chunk with plenty of room |

---

## Evidence 1 — The 1-chunk gold hides siblings, but today it barely distorts

Sample: the 50 examples with a 1-chunk gold (`answerable` 40, `persona` 10), baseline retriever, k=5.

| Measure | Chunk (floor) | Paper (ceiling) |
|---|---|---|
| hit@5 (does the top-5 contain the gold?) | 50/50 = 100% | 50/50 = 100% |
| hit@1 | 48/50 | 50/50 |
| MRR (1 ÷ position of the 1st hit) | 0.980 | 1.000 |
| precision@5 (fraction of the top-5 that is gold)* | 0.200 | 0.276 |

\* At the paper level the denominator is the number of **distinct** papers retrieved.

- Only **2 of 50** examples have a different MRR between the levels (a sibling ranked 1st and the gold chunk 2nd). Gold chunk position: 48× at 1st, 2× at 2nd.
- Siblings per example (other chunks of the same paper): mean 2.44 · dist. {1: 4, 2: 20, 3: 26}.
- In **38/50** the top-5 contains a sibling of the gold; in **39/50** some paper repeats; mean of **3.80 distinct papers** in 5 positions.
- **Parallel finding:** with hit@5 = 100%, EXP-0 **does not discriminate** retrieval on `answerable`/`persona`. This is consistent with the `deep-hit` slice (hard questions, found in EXP-0) being the place where retrieval changes will show up. The gold almost always ranking 1st (48/50) is compatible with questions that are easy by construction — **hypothesis**, not tested.

## Evidence 2 — The `chunk_id` breaks if the chunking changes (affects EXP-3)

`utils.chunk_id(doc) = f"{arxiv_id}:{sha1(page_content)[:12]}"`. The plan (CKPT-3) compares **chunk 500/overlap 0 × 1000/overlap 200**.
Test on paper `2609.20822v1` (re-chunking of the concatenated text of the current chunks):

| Chunk size | Number of chunks | IDs in common with the other sizes |
|---|---|---|
| 500 | 4 | none |
| 300 | 7 | none |
| 800 | 3 | none |

In EXP-3 the 50 `gold_chunk_ids` would match nothing. Only the `arxiv_id` survives. This reinforces paper-level as the level that is comparable **across** experiments.

## Evidence 3 — Whole abstract as the unit: similar quality, with a bias to consider

Temporary index (1 chunk per abstract; nothing written to `resources/`), same questions, gold at paper level, k=5.

| Slice (n) | Index | hit@1 | hit@3 | hit@5 | MRR | recall@5 | Distinct papers in top-5 |
|---|---|---|---|---|---|---|---|
| answerable (40) | chunks 500/0 | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 | 3.80 |
| | whole abstract | 0.90 | 0.97 | 1.00 | 0.934 | 1.000 | **5.00** |
| multi-doc known (10) | chunks 500/0 | 0.70 | 0.70 | 0.90 | 0.740 | 0.667 | 4.30 |
| | whole abstract | 0.70 | 0.70 | 0.90 | 0.750 | 0.617 | **5.00** |
| persona (10) | chunks 500/0 | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 | 3.80 |
| | whole abstract | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 | **5.00** |

- Same hit@5; the whole abstract **always returns 5 distinct papers** (the per-chunk index repeats papers and wastes slots).
- The whole abstract loses a little on hit@1 (`answerable`: 4/40 examples). **Bias to consider:** the questions were generated **from a 500-character chunk**, so the per-chunk index has a matching advantage. Hypothesis (not tested): with questions generated from the whole abstract, or written by a human, the difference tends to vanish.
- **Small n** (10 in `multi-doc`; 4 examples of difference in `answerable`): indication, not proof.

## Evidence 4 — Cost and latency of sending 5 whole abstracts to the generator

`gpt-4o-mini`, `PROMPT_V1`, temperature 0, 20 questions per arm (12 `answerable` + 8 `multi-doc`), order alternated between arms.

| | Chunks 500/0 | Whole abstract | Ratio |
|---|---|---|---|
| Input tokens (mean / max) | 606 / 722 | 1,571 / 1,887 | **2.6×** |
| Output tokens (mean) | 50.1 | 54.6 | ~1.1× |
| Median / p90 latency | 1.03 s / 1.44 s | 1.11 s / 1.59 s | +8% |
| Cost per query (generation)* | ~US$ 0.00012 | ~US$ 0.00027 | 2.2× |

\* Price used: US$ 0.15 / 1M input tokens and US$ 0.60 / 1M output tokens (third-party pages: devtk.ai, OpenRouter) — **confirm on the OpenAI site** before budgeting.

- Latency depends more on the **output** (~50 tokens, fixed by the 3-sentence prompt) than on the input; n=20 → large noise, treat as an order of magnitude. Far from the 4 s limit of the CKPT-1 gate.
- The judges (faithfulness, citation) also read the context: evaluation cost rises in the same proportion.
- **Unmeasured hypothesis:** with whole abstracts, **k=3** (~1,060 tokens, arithmetic estimate) could cover as much as k=5 chunks (3.8 distinct papers). Worth testing in CKPT-1.

---

## What the literature (reading notes) says — and the gap

| Source (reading note in `docs/research/fichamentos/`) | What it supports | Limit |
|---|---|---|
| `thakur-2021-beir` | SciFact (5,183 abstracts) and SCIDOCS (25,657) use the **abstract as the document** | Does not compare whole abstract vs. chunk |
| `gao-2023-alce` | Without gold at passage granularity, the oracle is built **by content**, not by ID | Wikipedia, 100-word passages |
| `magesh-2025-hallucination-free` | Judges the citation by **cited document**, with no chunk gold | Legal domain |
| `ru-2024-ragchecker` | recall@k/MRR by chunk ID "depend on annotated chunks and rigid chunking"; chunk 150→300 tokens: claim recall 70.3→77.6 | Long documents, k=20 — does not transfer to abstracts of 255 tokens |
| `barnett-2024-seven-failure-points` (R7) | A small chunk prevents answering, a large one brings noise; calls for systematic evaluation | Gives no number or rule |
| `ir-pooling-incomplete-judgments` | Treating unjudged as irrelevant gives a **floor**; the way out is to judge the "holes" | Does not address passage vs. document |

**Gap:** no reading note compares **whole abstract vs. sub-abstract** as the indexing unit, nor studies synthetic gold (a question generated from the chunk itself). The decision rests on the measurements in this document, not on a paper that prescribes it. Targeted literature research on chunking of short texts remains an optional step.

---

## Design consequences (guardrails)

1. **Always compare at the level where the gold exists** (chunk if `gold_chunk_ids`, otherwise paper) — already implemented (P1–P3, `_ranked_and_gold`).
2. **Candidate for primary level: the paper** (D-1) — holds for any indexing unit and survives EXP-3; the chunk becomes a diagnostic, never compared across chunking configurations.
3. **Paper-level gold does not solve everything:** other papers in the corpus may also answer (unjudged relevant). Only the EXP-0 mini-pooling reveals this.
4. **Separate D-1 (gold level) from D-7 (indexing unit).** The first can close now; the second changes the baseline and the design of EXP-3 and needs evidence about the **answer**.
5. **Do not adjudicate sibling chunks in EXP-0 for hit/MRR:** floor ≈ ceiling today (2/50 diverge). Re-evaluate only if chunk-level precision (P6) is needed or if k decreases (hit@1 already separates 48 from 50).

### Alternatives considered

| | D-1 (gold level) | Against |
|---|---|---|
| A | Chunk only | Breaks in EXP-3; counts a relevant sibling as an error |
| B | Paper only | Loses "did it find the passage that answers?" |
| **C** | **Both, as a [floor, ceiling] range; paper = comparable level** | Two numbers per example |
| D | Chunk + adjudicate siblings | ~200–250 judgments (agent estimate, not checked); small gain today |
| E | Gold by passage **text**, matched by overlap | More complex; RAGChecker diagnostics without human validation |

| | D-7 (indexing unit) | Against |
|---|---|---|
| 1 | Chunks 500/0 (current plan) | Repeats papers, cuts sentences, fragile EXP-3 |
| 2 | **Whole abstract** | 2.6× tokens, 2.2× cost; golden set generated from chunks; changes the CKPT-3 design |
| 3 | Hybrid (index per abstract, chunks as an EXP-3 variant) | Two implementations |

The baseline with 500-character chunks was a **deliberate** choice (corpus sized in characters to yield 800–1,200
chunks; see `utils.chunk_documents` and `docs/plan.md` §1): it provokes failures for CKPT-3 to study. Switching to 1 chunk/abstract
redefines the experiment ("does splitting the abstract help or hurt?").

---

## Decision made (2026-10-03)

After this analysis the user approved:

1. **Baseline = one index record per abstract** (D-7): the whole abstract is the "chunk". 500-character chunking becomes an **experimental arm** (EXP-3: "does splitting the abstract help or hurt?").
2. **Retrieval gold = the paper** (D-1). With one record per abstract, chunk = paper: there is no "book + page" and no list of chunks per paper. The chunk level stays as a **diagnostic** only in the 500/0 configuration.
3. **Metrics by question type** (D-2): gold of 1 paper → hit@k + MRR (+ hit@1/3/5 curve); gold of 2+ papers → recall@k + hit@k + MRR.
4. **Contingency:** if the golden-set bias (questions generated from 500-character chunks) harms the metrics of the whole-abstract index, **regenerate `answerable`/`persona` from the whole abstract**.
5. **On citing "paper Y says Z":** it does not depend on the chunk. The generator can return a literal sentence from the abstract together with the `arxiv_id`; `citation_accuracy` (P12) verifies that the sentence exists and supports the claim. Today the prompt cites only the `arxiv_id`.

Why decide **before** EXP-0: once the baseline is recorded, switching the unit costs far more. The accepted risks: ×2.2 cost per query (cents), answer quality still **not measured**, and the CKPT-3 design changes.

**Implemented:** the evaluators (paper as the default level, `k`, `hit_at_ks`, `primary_retrieval_metrics`) and, on 2026-10-04, the whole-abstract index (`config.BASELINE`; the 500/0 arm is `config.CHUNKED_500_0`; new cache with the same 250 papers; paper-level gold derived at evaluation time). **Finding while implementing:** the frozen corpus can no longer be rebuilt from arXiv (the snapshot-pin skip budget is exhausted), so the corpus is now described by a git-versioned manifest (`resources/corpus_manifest.json`: ids + abstract digests) and rebuilt by arXiv id with verification; see ADR-005 and `docs/decisions.md` (Observations, 2026-10-04).

---

## Limits of this evidence

- One retriever, baseline only, 50–60 queries; differences of 2–4 examples.
- I did not judge whether the sibling chunks **actually answer** (the 2 divergent cases are in the example above for human judgment).
- Latency: n=20 per arm, single calls; price from third-party pages.
- The index comparison favors the per-chunk index (questions generated from chunks).
- **Answer** quality (faithfulness, correctness) was not measured — it depends on the judges (CKPT-0.6.3) and on calibration with real labeled answers.
- The measurement scripts were left in a temporary session folder and **were not saved to the repository**; reproducing requires rewriting them (candidate for `experiments/`).

---

## Relation to ADRs and next steps

- **ADR-005** ([retrieval unit](../adrs/0005-retrieval-unit-whole-abstract.md)) and **ADR-006** ([gold granularity and metric policy](../adrs/0006-retrieval-gold-granularity-and-metrics.md)) — record the decisions taken from this lesson.
- **ADR-001 / ADR-003** — origin of the two gold types (chunk vs. paper): `answerable`/`persona` by chunk; `multi-doc` by paper.
- **`docs/roadmap/ckpt-0.6.1b-metrics-roadmap.md`** — decisions **D-1**, **D-2**, **D-7** (new), **P5**, **P6**; log §5.
- **`docs/plan.md`** — CKPT-3 (chunking) and §2b (gold map) with a note pointing to this lesson.
- **Previous lesson:** [Golden Dataset Construction for RAG: Synthetic vs. Adjudication vs. Calibration](./golden_dataset_construction_and_human_calibration.md) (L1/L2/L3; pooling bias; the "bpref-style tolerance" cited there was contested in `ir-pooling-incomplete-judgments.md`).
- **Next steps:** implement the index change (D-7, its own slice); close P6 (postponed); run EXP-0 with the new baseline; if the golden-set bias shows up, regenerate `answerable`/`persona` from the abstract; consider targeted research on chunking of short texts.
