# CKPT-0.5 Plan — Golden Dataset (core split)

> Status: **awaiting approval**. No code will be changed before the "go".
> Parent plan: `docs/plan.md` §2 (Golden Set) + §5 (CKPT-0 contract).

## 1. Context — where we are

CKPT-0 (Foundation & Baseline) is being delivered in small batches:

| Batch | Deliverable | Commit |
|---|---|---|
| 0.1 | project scaffold | `e787586` |
| 0.2 | `config.py` + `RunConfig` | `27e3130` |
| 0.3 | arXiv ingestion + vector store (parquet) + `main.py` | `6b14177` |
| 0.4 | traced RAG pipeline (`app.py`, 3 tracing methods) | `bade39e` |
| **0.5** | **`build_golden_dataset.py` (core split)** | **this batch** |
| 0.6 | `evaluators.py` (§4 catalog) | next |
| 0.7 | `experiments/run_experiment.py` + EXP-0 round 1 | |
| 0.8 | EXP-0 round 2 (judge variance) + `deep-hit` slice + `results/EXP-0-baseline.md` | |

## 2. What to build (exact scope of this batch)

1. **`build_golden_dataset.py`** — builds the `arxiv-copilot-golden` dataset (`core` split) with **6 slices and ~100 examples**.
2. **`utils.py`** — two small additions:
   - `chunk_id(doc)` — deterministic chunk identity (see §4.1);
   - corpus snapshot pin (see §4.2).
3. **`config.py`** — `CORPUS_SNAPSHOT_DATE` + per-slice target counts.
4. **`main.py`** — wire `build_golden_dataset()` into the flow (the 0.4 stub already announces this).

**Out of scope (explicit)**: `evaluators.py`, experiment runner, `deep-hit` slice (defined empirically in EXP-0 — `docs/plan.md` §2), `extended` split (~200 ex; only used in CKPT-7 — becomes its own batch before it), any change to `app.py`.

After this batch the core split will have 100 examples — a number that matches the "~100" of §2 (the §2 table sums to 115 because it includes the 15 `deep-hit` examples, which only exist after EXP-0).

## 3. Why (rationale)

- **Plan contract**: CKPT-0 requires the golden set to be visible in LangSmith with all slices before EXP-0 (`docs/plan.md` §5, verification). The slices exist because each FP in `docs/research/rag_failure_modes_review.md` needs (a) an evaluator, (b) a slice, (c) an experiment — this batch delivers (b).
- **`deep-hit` deliberately postponed**: the §2 table defines it as "found empirically: questions whose answer doc ranks low in baseline". Building it now would be a guess. It is born from EXP-0 (batch 0.8), preserving the honesty of the label.
- **Extended split postponed**: §2/§8 reserve it for CKPT-7; spending ~200 LLM generations now would incur cost ahead of any consumer.
- **Ground-truth map already defined** (`docs/plan.md` §2b): this batch only needs to materialize the labels the table requires — nothing more.

## 4. Design decisions (with codebase evidence)

### 4.1 Deterministic `chunk_id` — no cache rebuild

**Problem**: §2 requires "Source chunk ID" as the gold retrieval label, but the `SKLearnVectorStore` generates random UUID IDs per chunk (verified: the parquet's `ids` = `f665f189-…`). Retrieval labels are useless without a stable, reproducible identity.

**Decision**: `utils.chunk_id(doc) = f"{arxiv_id}:{sha1(page_content)[:12]}"`. Derived from already-persisted fields (no schema change, no parquet rebuild). The same function is used by the dataset builder and, later (0.6), by the retrieval evaluators when inspecting retrieved docs — practical collision zero (distinct chunks have distinct content).

Rejected alternative: stamping `chunk_id` in the metadata in `chunk_documents` + cache rebuild — same guarantee, but burns the current snapshot (see 4.2) and re-embeds 843 chunks for no gain.

### 4.2 Corpus snapshot pin (`stale` slice)

**Evidence**: the cached corpus covers papers published between **2026-09-16 and 2026-09-17**; today (batch build) is 2026-09-24. That is, there is ~1 week of real papers published *after* the corpus content.

**Problem**: today `fetch_arxiv_corpus` always takes the 250 most recent. Any future rebuild (CKPT-2 swaps the embedding → new parquet) would pull new papers and *accidentally invalidate* the `stale` slice.

**Decision** (recommendation to the capstone): `config.CORPUS_SNAPSHOT_DATE = "2026-09-17"` (= max `published` of the current cache). `fetch_arxiv_corpus` now filters client-side `published.date() <= snapshot`, with oversampling (fetches ~2× the results, filters, cuts to 250). The existing cache **remains valid** (its max is already 09/17). Thus:

- **pre-refresh** (now until CKPT-7): `stale` questions about papers ≥ 2026-09-18 → gold = abstention ("No papers found in this period");
- **refresh drill (CKPT-8)**: bump the date → real re-index → gold becomes a summary of the new papers. The drill is executable by command, it does not depend on real time passing.

Rejected alternatives: postponing the slice to CKPT-8 (would leave CKPT-0 without coverage of FP1′/R3 and of §2b); niche questions absent without a date (tests "out of corpus", not temporal *staleness* — does not exercise R3).

### 4.3 Construction per slice

| Slice | n | Provenance | How | Gold labels |
|---|---|---|---|---|
| `answerable` | 40 | synthetic-by-construction (§2) | stratified sampling of chunks (fixed seed, max. 1–2 questions per paper), 1 LLM call per chunk → `{question, answer}` JSON | `gold_chunk_ids=[source chunk_id]`, concise `answer` |
| `unanswerable` | 15 | hand-written | topics outside the corpus (quantum physics, genomics, astrobiology, finance, medicine…) | `gold_chunk_ids=[]`, `answer="I don't know…"`, `should_abstain=true` |
| `multi-doc` | 15 | **hybrid (decision 2026-09-26)**: 10 known-item with IDs in the question (product Features B/D) + 5 open-topic without IDs (realistic production queries) | pairs/trios of real corpus papers on adjacent topics; the question requires joint synthesis | known-item: `gold_arxiv_ids=[2–3 ids]`; open-topic: `gold_arxiv_ids=[]` **pending** until mini-pooling adjudication in EXP-0 (§8), flag `retrieval_gold=pending-adjudication` in the metadata. Both: `answer` = synthesis with the expected points |
| `format` | 10 | hand-written | explicit format instructions (markdown table with columns X; JSON conforming to schema; list with exactly N items) | `answer` = exemplary formatted output; no retrieval gold |
| `persona` | 10 | derived from `answerable` | 5 base questions × 2 audiences (`pm`, `phd`) = 10 twin examples; gold calibrated per audience via LLM | same as `answerable` + `audience` in the input; pairs linked by `base_id` in the metadata |
| `stale` | 10 | hand-written | "What's new about \<topic present in the corpus\> published after September 17, 2026?" — a corpus topic to tempt the retriever into answering with old material | `gold_chunk_ids=[]`, `answer="No papers found in this period"`, `should_abstain=true` |

**Example schema (kv, uniform)**:
- `inputs`: `{"question": str}` (+ `"audience": "pm"|"phd"` only in `persona`);
- `outputs`: `{"answer": str, "gold_chunk_ids": list[str], "gold_arxiv_ids": list[str], "should_abstain": bool}`;
- `metadata`: `{"slice": str, "provenance": "synthetic"|"curated", "base_id"?: str}` — correction 2026-09-27: the plan's "hand-written" slices are in fact written by AI and **approved by a human** (spot-check/batch approval); the human anchor enters via review, mini-pooling adjudication and judge calibration, not via authorship. Renamed to `"curated"` for provenance honesty;
- `split=["core"]`.

A uniform schema (fields always present, empty when N/A) avoids branching in the batch 0.6 evaluators.

**Gold answer generation**: `GENERATION_MODEL` (gpt-4o-mini), temperature 0, with a human spot-check of 5 samples/slice before commit approval — this is the mitigation already foreseen in the §8 risk table. Estimated cost: ~60 calls → < $0.05.

### 4.4 Idempotency

§2 contract: check `client.has_dataset(...)` before writing. Implementation: if the dataset exists, list the examples of the `core` split once, count by `metadata.slice`, and create **only missing/incomplete slices**; final report with counts per slice. Second run = noop (except the report).

## 5. Batch verification (how we will know it is done)

1. `python build_golden_dataset.py` creates the dataset; run 2× → 2nd run noop.
2. Dataset visible in the `capstone-arxiv-copilot` project with `split=core` and the 6 slices (CKPT-0 §5 verification).
3. Counts per slice match table 4.3 (total 100).
4. **Human spot-check** (you approve before the batch merge): 5 samples printed per synthetic/derived slice (`answerable`, `persona`) — questions actually answerable from the source chunk, gold answers correct.
5. Baseline cache untouched (parquet does not regenerate) and `python main.py` still runs end to end — the fetch change must not break the hot path.
6. No change in pipeline behavior (`app.py` is not touched).

## 6. Expected commits of the batch (even smaller batches)

1. `chore: pin corpus snapshot date (2026-09-17) + deterministic chunk_id` — config + utils.
2. `feat: golden dataset builder, core split — 6 slices (~100 examples)` — build_golden_dataset.py + wiring in main.py + approved spot-check.

## 7. Risks specific to this batch

| Risk | Mitigation |
|---|---|
| Synthetic questions too easy ("too easy", §8) | sampling with max. 1–2/paper + spot-check; diversity measured in EXP-0 |
| Hand-written `multi-doc`/`stale` questions with my bias | provenance in the metadata; you review in the spot-check |
| Date pin changing the corpus of future configs | additive skip-budget (≥ ~1 week margin) guarantees 250 papers ≤ snapshot; an explicit guard raises an error if it shrinks — see `config.py` |
| Hash-based `chunk_id` diverging from what the evaluator computes | the single function in `utils.py` is the source of truth (dataset builder + 0.6 evaluators import from it) |

## 8. Addendum — human adjudication and judge calibration (research 2026-09-24)

Full rationale with auditable citations:
`docs/learning-lessons/golden_dataset_construction_and_human_calibration.md`. Derived decisions:

1. **Mini-pooling in EXP-0 (CKPT-0.8)**: for each golden question, run top-k retrieval, list
   candidates outside the gold for human adjudication; docs judged relevant enter
   `gold_arxiv_ids`. TREC pooling in miniature — constructed golds are not exhaustively reviewed.
2. **Retrieval metric semantics (applies to 0.6)**: in `answerable`/`multi-doc`, docs
   retrieved outside the gold are **unjudged**, never counted as irrelevant (pooling bias
   documented in the literature); recall/hit/MRR count only gold. Full `precision@k` only in
   `unanswerable`/`stale`, whose gold is "nothing should match".
3. **Human spot-check (§5 item 4) takes on a dual function**: dataset quality validation AND
   seed of the LLM judges' calibration set (ARES/RAGAS-Align/TruLens: ~100–200 human labels
   align the judge — not the whole dataset).
