# Capstone v2: arXiv Research Copilot — RAG Failure-Mode Battleground

> A **product** — a cited Q&A assistant over ~250 arXiv cs.AI papers — engineered so that every documented RAG failure point (see `rag_failure_modes_review.md`) can be **detected** (golden set + evaluators), **measured** (monitoring), and **mitigated** (experiments). Planning is organized by experiments; each experiment is a **checkpoint** with a decision gate.

---

## 1. Product Definition

**User**: engineers/researchers who need fast, trustworthy answers about recent AI research.

**Job-to-be-done**: *"Answer my questions about recent AI papers with real citations — and tell me when you don't know."*

**Corpus**: ~250–300 arXiv cs.AI abstracts (~800–1200 chunks @ 500 chars), parquet-cached, publication-dated (enables the `stale` slice).

**Corpus size rationale**: with a tiny corpus, retrieval recalls ~100% and FP2 never manifests. 250+ papers make retrieval realistic enough for experiments to discriminate.

### Features — each one surfaces specific failure points

| Feature | Behavior | Failure points exercised | Justification |
|---|---|---|---|
| **A. Cited Q&A** | Answer + paper citation per claim | FP4, hallucination-with-source | Core RAG capability; citation is where Magesh et al. misgrounding manifests |
| **B. Structured comparisons** | "Compare methods of papers A, B, C" → table | FP5, FP3, FP7 | Multi-doc retrieval + format constraint in one feature |
| **C. Audience-adaptive answers** | "Explain like a PM" vs "like a PhD" | FP6 | **Reframed as specificity tuning**: same question, different granularity → tests whether the model over-summarizes (too vague) or over-specifies (too dense) for the requested level. FP6 is about calibration, not persona. |
| **D. Multi-part synthesis** | "Techniques AND limitations of A, B, C?" | FP7, FP3 | Natural multi-doc scenario where completeness and consolidation fail |
| **E. Honest abstention** | Out-of-corpus → "I don't know" | FP1 | Most critical failure mode in high-stakes domains; abstention quality is a release-blocking metric |
| **F. Multi-turn threads** | Follow-ups with conversation memory | FP2, FP3 (at turn N>1) | **Reframed as thread context degradation**: later turns lose retrieval context as conversation history grows → retrieval quality degrades across the thread. Thread-level online evals (M1 L4) measure per-turn retrieval drift. |

---

## 2. Golden Set — two-tier adversarial design

### Why two tiers

Slices of n=6–12 have no statistical power: one example flipping = 8–17% metric swing, while LLM-judge noise is 5–10%. Decision gates asking for "≥10% improvement" are unmeasurable at those sizes. The two-tier design separates **cheap directional experiments** (core) from **statistical validation** (extended).

| Tier | Total | Used by | Purpose |
|---|---|---|---|
| **Core** | ~100 | EXP-0 through EXP-6 (every experiment) | Directional signal — cheap to run, fast feedback |
| **Extended** | ~200 | EXP-7 (final A/B pairwise, CKPT-7) | Statistical validation — enough power to detect 10% deltas |

### Core set (~100 examples)

| Slice | Size | Targets | Construction | Gold retrieval label | Gold answer |
|---|---|---|---|---|---|
| `answerable` | ~40 | baseline | **Synthetic-by-construction**: question generated *from* a chunk → that chunk is the gold retrieval label | Source chunk ID | Concise answer extracted from source chunk (LLM-generated, human-verified on 5-sample spot check) |
| `unanswerable` | ~15 | FP1 | Hand-written: topics outside corpus (quantum, biology) | None (no doc should match) | `"I don't know"` (explicit abstention label) |
| `deep-hit` | ~15 | FP2, R5 | Found empirically: questions whose answer doc ranks low in baseline | Answer doc ID (identified during EXP-0) | Answer extracted from the low-ranking doc |
| `multi-doc` | ~15 | FP3, FP7 | Hand-written: requires 2–3 specific papers jointly | List of 2–3 gold doc IDs | Synthesis answer listing all expected points from all required docs |
| `format` | ~10 | FP5 | Table / JSON-schema / list instructions | N/A (format is the test, not retrieval) | Correctly formatted output matching the requested schema |
| `persona` | ~10 | FP6 | Same question, PM vs PhD audience | Same as `answerable` source | Two gold answers: one calibrated for PM, one for PhD |
| `stale` | ~10 | FP1′ / R3 | "What's new in \<recent month\>?" — fails until re-index | None (pre-refresh); new docs (post-refresh) | `"No papers found in this period"` (pre-refresh); summary of new papers (post-refresh) |

### Extended set (~200 examples)

Same 7 slices, expanded to statistical power. Used only for CKPT-7 (final pairwise A/B).

| Slice | Core | Extended | Rationale for size |
|---|---|---|---|
| `answerable` | 40 | 80 | Baseline needs the largest n; 80 gives ±5% granularity |
| `unanswerable` | 15 | 25 | Abstention is release-blocking; 25 gives ±4% granularity |
| `deep-hit` | 15 | 25 | Primary target of CKPT-4/5; 25 gives ±4% granularity |
| `multi-doc` | 15 | 25 | Primary target of CKPT-5; 25 gives ±4% granularity |
| `format` | 10 | 20 | Code-based evaluator (no judge noise); 20 is sufficient |
| `persona` | 10 | 15 | LLM-judge but binary-ish (PM vs PhD); 15 gives ±6.7% |
| `stale` | 10 | 15 | Time-sensitive; 15 gives ±6.7% |
| **Total** | **115** | **205** | |

### Statistical power note

Even the extended set is small by clinical-trial standards. The design assumes:
- LLM-judge evaluators have ~5–10% inter-run variance (measured during EXP-0 by running the same evaluator twice).
- Decision gates on the core set require **≥15% delta** (not 10%) to be credible.
- Decision gates on the extended set (CKPT-7 only) require **≥10% delta** with pairwise comparison (which controls for judge bias by comparing both arms in the same judge call).
- The pairwise design (CKPT-7) is the statistical backstop: it doesn't need absolute scores, only relative preference, which has lower variance.

**Idempotent dataset creation**: check `client.has_dataset("arxiv-copilot-golden")` before writing. Core and extended are stored as dataset splits: `splits=["core"]` and `splits=["extended"]`.

---

## 2b. Ground Truth Map — what gold label exists per slice per evaluator

Every slice now has explicit gold labels. The map below defines which evaluators are reference-based (need gold) vs. reference-free (judge against retrieved context only).

| Slice | Gold retrieval label | Gold answer | Evaluators that use gold retrieval | Evaluators that use gold answer | Reference-free evaluators (no gold needed) |
|---|---|---|---|---|---|
| `answerable` | Source chunk ID | Concise answer from source chunk | recall@k, hit@k, MRR, precision@k | `compare_semantic_similarity` (vs. gold answer) | `faithfulness_judge`, `citation_accuracy`, `answer_relevance` |
| `unanswerable` | None (no doc should match) | `"I don't know"` | precision@k (should be ~0) | `abstention_quality` (checks abstention matches gold) | `faithfulness_judge` (should flag ungrounded answers) |
| `deep-hit` | Answer doc ID (from EXP-0) | Answer from low-ranking doc | recall@k, hit@k, MRR | `compare_semantic_similarity` | `faithfulness_judge`, `citation_accuracy` |
| `multi-doc` | List of 2–3 gold doc IDs | Synthesis listing all expected points | recall@k, hit@k (all gold docs in top-k?) | `completeness_judge` (checks all points from gold answer present) | `faithfulness_judge`, `citation_accuracy` |
| `format` | N/A | Correctly formatted output | N/A | `format_validator` (code: checks output matches schema) | N/A |
| `persona` | Same as `answerable` source | Two gold answers (PM + PhD) | recall@k (same as answerable) | `specificity_judge` (checks output matches requested audience's gold) | `faithfulness_judge` |
| `stale` | None (pre-refresh) / new docs (post-refresh) | `"No papers found"` (pre) / new paper summary (post) | precision@k (pre-refresh: should be ~0) | `abstention_quality` (pre-refresh: should abstain) | `faithfulness_judge` (post-refresh: should be grounded in new docs) |

**Key design decisions**:
- `faithfulness_judge` is **always reference-free** — it checks if the answer is grounded in retrieved context, not against a gold answer. This is the RAGAS-style faithfulness metric.
- `completeness_judge` is **reference-based on `multi-doc` only** — it needs the gold synthesis answer to know what "complete" means. On other slices it's reference-free (checks if all sub-questions in the query were addressed).
- `abstention_quality` is **reference-based** — it needs the gold answer to know whether the system *should* have abstained.
- `citation_accuracy` is **reference-free** — it checks if the cited doc exists in the corpus AND supports the claim (the Magesh et al. misgrounding check).

---

## 3. Instrumentation — two failure surfaces measured separately

Per the review (Suthar; Barnett): retrieval and generation are measured independently, in that order.

```
arxiv_copilot (chain, root)
├── rewrite_query (chain, toggleable)        ← EXP-5 arm
├── retrieve_documents (retriever)
│      metrics: recall@k · hit@k · MRR · precision@k   ← SURFACE 1
├── rerank (chain, toggleable)               ← EXP-4 arm
└── generate_response (chain)
       └── call_openai (llm)
              metrics: faithfulness · citation accuracy · answer relevance
                       completeness · format validity · abstention quality  ← SURFACE 2
```

**Run metadata on every run**: `app_version`, `embedding_model`, `k`, `chunk_strategy`, `rerank`, `query_rewrite`, `prompt_version` — enables slicing any metric by any configuration.

---

## 4. Evaluator Catalog

### Provenance

The course (`intro-to-langsmith`) teaches 7 evaluators, none RAG-specific. The evaluators below extend beyond the course. Each is tagged with its provenance and the course pattern it follows.

**Course patterns** (from `module_2/evaluators.ipynb`, `experiments.ipynb`, `pairwise_experiments.ipynb`, `summary_evaluators.ipynb`):
- **P1**: code-based, reference-based (e.g., `correct_label`, `is_concise_enough`)
- **P2**: LLM-as-judge, reference-based (e.g., `compare_semantic_similarity`)
- **P3**: LLM-as-judge, reference-free (e.g., `summary_score_evaluator`)
- **P4**: LLM-as-judge, pairwise reference-free (e.g., `ranked_preference`)
- **P5**: code-based summary evaluator, reference-based (e.g., `f1_score_summary_evaluator`)

| Evaluator | Type | Course pattern | Provenance | Measures | FP | Reference? |
|---|---|---|---|---|---|---|
| `retrieval_recall_at_k` | code | P1 (code + reference) | IR standard (Respan, Redis) | gold doc in top-k? | FP2, FP3 | Reference (gold doc ID) |
| `retrieval_hit_rate` | code | P1 | IR standard | ≥1 gold doc in top-k | FP2 | Reference (gold doc ID) |
| `retrieval_mrr` | code | P1 | IR standard | rank of first gold doc | FP2, R5 | Reference (gold doc ID) |
| `retrieval_precision_at_k` | code | P1 | IR standard (Redis) | noise ratio in top-k | FP3 | Reference (gold doc ID) |
| `faithfulness_judge` | LLM-as-judge | P3 (LLM-judge, ref-free) | RAGAS / literature — **not in course** | grounded in retrieved context | FP4 | **Reference-free** |
| `citation_accuracy` | code + LLM | P1 + P3 hybrid | **Novel** — inspired by Magesh et al. (2025) misgrounding finding | citation exists AND supports claim | FP4 | **Reference-free** |
| `completeness_judge` | LLM-as-judge | P3 (ref-free) / P2 (ref-based on multi-doc) | **Novel** — extends course P3 to multi-part queries | all sub-questions answered | FP7 | Hybrid: ref-free on single-Q, ref-based on `multi-doc` |
| `format_validator` | code | P1 (code + reference) | **Novel** — code-based schema check | valid table/JSON per schema | FP5 | Reference (gold formatted output) |
| `abstention_quality` | LLM-as-judge | P2 (LLM-judge + reference) | **Novel** — extends course P2 to abstention | abstains when it should, answers when it can | FP1 | Reference (gold answer: abstain or not) |
| `specificity_judge` | LLM-as-judge | P2 (LLM-judge + reference) | **Novel** — extends course P2 to audience calibration | matches requested audience | FP6 | Reference (gold answer per audience) |
| `answer_relevance` | LLM-as-judge | P3 (LLM-judge, ref-free) | RAGAS / literature — **not in course** | on-question, useful | — | **Reference-free** |
| `f1_summary_evaluator` | code, summary | P5 (code summary + reference) | Adapted from course `f1_score_summary_evaluator` (toxicity → RAG quality) | aggregate substantive-vs-hallucinated F1 | cross-cutting | Reference (gold labels) |

### What the course teaches vs. what we extend

| Course evaluator | Course pattern | Our adaptation |
|---|---|---|
| `correct_label` (exact match) | P1 | → `format_validator` (exact schema match instead of exact string match) |
| `compare_semantic_similarity` (LLM vs. reference) | P2 | → `specificity_judge`, `abstention_quality` (same pattern: LLM judges output vs. gold answer, different criteria) |
| `summary_score_evaluator` (LLM, ref-free) | P3 | → `faithfulness_judge`, `answer_relevance` (same pattern: LLM judges output against context/query, no gold answer) |
| `ranked_preference` (pairwise, ref-free) | P4 | → CKPT-7 pairwise A/B (unchanged pattern) |
| `f1_score_summary_evaluator` (code summary, ref) | P5 | → `f1_summary_evaluator` (same pattern, different domain) |
| `is_concise_enough` (code, ref) | P1 | → `retrieval_recall_at_k` etc. (same pattern: code checks output against reference) |

**What's genuinely new** (not derivable from course patterns):
- `faithfulness_judge` and `answer_relevance` — RAGAS-style context-grounding checks. The course doesn't teach groundedness/faithfulness at all. These require a judge prompt that checks if each claim in the answer is supported by the retrieved context.
- `citation_accuracy` — the Magesh et al. misgrounding check: does the cited source *support* the claim? This is a two-step evaluator (code: does citation exist? + LLM: does it support?) with no course analog.

API contract reminders (from course notebooks):
- per-example evaluators → `evaluate(..., evaluators=[...])`
- aggregate evaluators → `evaluate(..., summary_evaluators=[...])` (list inputs)
- pairwise evaluators receive `outputs: list[dict]`, return one score per experiment.

---

## 5. Checkpoints — experiments as the unit of planning

Every checkpoint follows the same contract:

```
HYPOTHESIS → CONFIG CHANGE → TARGET SLICES → EVALUATORS → DECISION GATE → ARTIFACT
```

The **decision gate** promotes or rejects a configuration. All gates also require **no regression** on non-target slices (Δfaithfulness > −2pts, Δcost < budget). Each gate's outcome is logged in `decisions.md` (date, config, metric before/after, verdict).

**Golden set tier per checkpoint**: EXP-0 through EXP-6 run on the **core set** (~100). EXP-7 (CKPT-7) runs on the **extended set** (~200). Core-set gates require **≥15% delta** to be credible (slices are small). Extended-set gates (pairwise) require **≥10% delta** or ≥60% win rate.

### CKPT-0 — Foundation & Baseline (Module 0 + 1 + 2)

**Build**: `config.py`, `utils.py` (arXiv load → chunks → embeddings → parquet), `app.py` (all 3 tracing methods: `@traceable`, `wrap_openai`, `trace` context manager; run types chain/llm/retriever; metadata; threads), `build_golden_dataset.py`, `evaluators.py`.

**EXP-0 (baseline measurement)**: core golden set × baseline config (`k=5, text-embedding-3-small, chunk 500/0, prompt v1, no rewrite, no rerank`). **Run twice** to measure LLM-judge inter-run variance — this establishes the noise floor for all later decision gates.

**Gate**: none — produces the baseline metrics table per FP/slice + judge variance report.
**Verification**: `python app.py` traces to project `capstone-arxiv-copilot`; dataset visible with all slices; EXP-0 results table printed with per-slice breakdown; judge variance ≤ 10% (if higher, increase `num_repetitions` in evaluator runs).
**Artifact**: `results/EXP-0-baseline.md` — the "what's broken right now" report + judge noise floor.

### CKPT-1 — Retrieval depth (FP2, FP3)

**EXP-1**: k ∈ {3, 5, 10} × golden set. Metrics: recall@k, hit@k, MRR, p95 latency, cost/query.

**Gate**: promote the k with best recall@k under latency < 4s and cost < budget. Expected: recall curve flattens; pick the knee.
**Artifact**: recall@k curve + decision log.

### CKPT-2 — Embedding model (FP2, FP6)

**EXP-2**: `text-embedding-3-small` vs `text-embedding-3-large` (vs multilingual if queries in PT). Same dataset, same k from CKPT-1.

**Gate**: promote large only if recall@5 gain ≥ 3pts → else keep small (5× cheaper).
**Artifact**: recall-vs-embedding-cost decision table (a real eng memo).

### CKPT-3 — Chunking strategy (FP3, R4, FP4)

**EXP-3**: chunk 500/overlap 0 vs 1000/overlap 200 on the *winning* embedding. Watch precision@k drop (noise) vs recall@k gain (coverage) — the Redis-documented tradeoff — and faithfulness on the `answerable` slice.

**Gate**: promote the config with best recall@k that does not regress faithfulness by >2pts.

### CKPT-4 — Reranker (FP2, R5) ← INCLUDED

**EXP-4**: rerank off vs **LLM-rerank** (no new dependencies): retrieve top-20 → `gpt-4o-mini` scores each candidate 0–10 for relevance → reorder → top-5 to generation.

Measured on `deep-hit` slice: hit@5 and MRR, plus added latency & cost/query (which the monitor must also track in production).

**Gate**: adopt rerank if MRR gain ≥ 15% on `deep-hit` (core set) AND added latency < 800ms/query.

### CKPT-5 — Query rewriting & decomposition (FP2, FP7)

**EXP-5**: four arms `{rewrite off/on} × {decompose off/on}`:
- rewrite: expand short user questions into retrieval-friendly queries
- decompose: split multi-part questions into sub-queries, union results

Target slices: `deep-hit` (rewrite), `multi-doc` (decompose). Metrics: recall@k + completeness_judge.

**Gate**: adopt each arm independently if it improves its target slice ≥15% (core set) without hurting `answerable`.

### CKPT-6 — Prompt variants via Prompt Hub (FP1, FP4, FP5, FP6) — Module 3

Push to Prompt Hub, version, and evaluate:
- `copilot-prompt-v1`: baseline (course-style 3-sentence)
- `copilot-prompt-v2`: v1 + "answer ONLY from context; if not present, say you don't know"
- `copilot-prompt-v3`: v2 + mandatory per-claim citations + format schema + audience instruction

Metrics per slice: `abstention_quality` on `unanswerable`, `faithfulness` + `citation_accuracy` on `answerable`, `format_validator` on `format`, `specificity_judge` on `persona`.

**Gate**: highest-scoring version that keeps correct abstention ≥ 90% on `unanswerable` (abstention regressions are release-blocking).

### CKPT-7 — Final A/B (pairwise) — Module 2 L5

**EXP-7**: best-configured system vs baseline, pairwise LLM judge (`Preference` 1/2/0 → `[1,0]/[0,1]/[0,0]`). Runs on **extended set** (~200 examples) for statistical power.

**Gate**: best config must win ≥ 60% of non-tie comparisons on extended set → ship candidate.

### CKPT-8 — Production (Module 4 + 5)

**Build**:
- `server.py` (FastAPI): `POST /ask` (pre-generated `run_id` + presigned feedback URL in response, threads via `thread_id`), `POST /feedback`, `GET /feedback-url/{run_id}`
- Online evaluators configured as rules on 1–5% of production traffic: `faithfulness_judge`, `format_validator`, `abstention_quality` (all reference-free)
- `feedback.py`: annotation queue fed by `user 👎 ∪ online-eval low score`
- `monitor.py`: north-star metrics with trend-by-config —
  retrieval hit-rate · faithfulness · abstention rate · p95 latency · cost/query · error rate — plus alert thresholds and failure-pattern analysis via tree filters
- **Corpus refresh drill**: bump corpus, re-run `stale` slice, record before/after

**Gate**: monitors live + at least one user-feedback round-trip visible in LangSmith + one routed annotation queue.

---

## 6. Module Coverage

| Module | Where it lives |
|---|---|
| M0 RAG foundation | CKPT-0: arXiv corpus, vector store |
| M1 Tracing | CKPT-0: 3 tracing methods, run types, metadata, threads |
| M2 Testing | Golden set slices, evaluator catalog, EXP-0–7, summary + pairwise |
| M3 Prompt engineering | CKPT-6: Prompt Hub versions compared by experiments |
| M4 Human feedback | CKPT-8: presigned URLs, feedback API, annotation queue |
| M5 Production monitoring | CKPT-8: online evals, filtering, monitor.py, refresh drill |

---

## 7. Directory Structure

```
capstone/
├── docs/
│   ├── plan.md                    ← this file (v2)
│   ├── rag_failure_modes_review.md ← research foundation
│   └── decisions.md               ← experiment decision log (gates)
├── main.py                        # CKPT-0 entry point — orchestrates the modules below
├── config.py                      # shared config + experiment flags
├── utils.py                       # arXiv indexing + vector store(s) (library, no script logic)
├── app.py                         # traced RAG pipeline (rewrite/retrieve/rerank/generate)
├── build_golden_dataset.py        # sliced golden set construction
├── evaluators.py                  # evaluator catalog
├── experiments/
│   ├── run_experiment.py          # experiment runner (--exp EXP-N)
│   └── pairwise.py                # final pairwise runner
├── prompt_management.py           # Prompt Hub push/pull/compare
├── feedback.py                    # feedback + presigned URLs + queue routing
├── server.py                      # FastAPI production service
├── monitor.py                     # north-star metrics + alerts
├── results/                       # per-checkpoint result reports
├── pyproject.toml                 # uv-managed deps (adds fastapi, uvicorn at CKPT-8)
└── resources/                     # parquet caches per embedding/chunk config
```

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Judge cost across 12 evaluators × 100 core examples × 7 experiments + 200 extended × CKPT-7 | Cache judge calls; run expensive judges only on target slices per experiment; core set for EXP-0–6, extended only for CKPT-7 |
| LLM-judge inter-run variance drowns signal on small slices | EXP-0 runs twice to measure noise floor; core-set gates require ≥15% delta (above measured noise); pairwise (CKPT-7) as statistical backstop |
| Small slices (n=10–15 on core) have marginal statistical power | Document as directional indicators; extended set (n=15–25) for final validation; pairwise comparison controls for judge bias |
| Synthetic questions too easy | 40+ hand-written adversarial examples across non-answerable slices; verify synthetic p50 similarity to corpus < threshold |
| `deep-hit` slice found via the baseline itself | Acceptable — document provenance; it's still a valid regression target for later configs |
| Gold answer construction for all slices is labor-intensive | LLM-generated gold answers with human spot-check on 5-sample per slice; `unanswerable` and `stale` gold answers are trivial ("I don't know" / "no papers found") |
| `faithfulness_judge` and `answer_relevance` have no course analog | Documented in §4 as RAGAS-derived; judge prompts written from first principles and validated against EXP-0 baseline (a system that hallucinates should score low) |
| arXiv rate limits | Parquet cache; corpus built once per config |
| Too-small corpus makes FP2 invisible | 250–300 papers minimum (rationale §1) |

---

## 9. Run Order

```
CKPT-0:  python main.py   (orchestrates utils.py → app.py → build_golden_dataset.py)
         python experiments/run_experiment.py --exp EXP-0
CKPT-1…5: python experiments/run_experiment.py --exp EXP-N   (record to decisions.md)
CKPT-6:  python prompt_management.py
CKPT-7:  python experiments/pairwise.py
CKPT-8:  uvicorn server:app  +  python monitor.py --window 1h
```

**Definition of done**: every FP in §2 of `rag_failure_modes_review.md` has (a) a measuring evaluator, (b) a golden-set slice, (c) a mitigation experiment with a logged decision, and (d) a production metric watching it.
