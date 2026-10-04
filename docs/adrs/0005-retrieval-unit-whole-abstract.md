# ADR-005: Retrieval unit for the baseline index: one record per whole abstract

**Status**: Accepted (implementation pending)
**Date**: 2026-10-03
**Related**: [Lesson: Retrieval Unit and Gold Granularity](../learning-lessons/retrieval_unit_and_gold_granularity.md) · [Roadmap decision D-7](../roadmap/ckpt-0.6.1b-metrics-roadmap.md) · [ADR-006](./0006-retrieval-gold-granularity-and-metrics.md)

---

## Context

The product is a research assistant over arXiv cs.AI papers, of which it reads **only the abstract**. It cites **papers**
(`PROMPT_V1` in `app.py`: "cite the arXiv ID for every claim"), never chunks or pages.

The baseline defined in `docs/plan.md` (CKPT-0) indexes **chunks of 500 characters, overlap 0**. The corpus was sized in
characters on purpose (~800–1,200 chunks) so that retrieval failure modes manifest, and CKPT-3 later compares 500/0 against
1000/200. Measured on the actual corpus (`resources/arxiv_text-embedding-3-small_500-0.parquet`):

| Fact | Value |
|---|---|
| Papers / chunks | 250 / 843 (2 to 5 chunks per paper, mean 3.37) |
| Whole abstract, tokens (`cl100k_base`) | mean 260.5, median 255, min 124, max 412 |
| Whole abstract, tokens (`o200k_base`, `gpt-4o-mini`) | mean 258.9, max 415 |
| Embedding input limit (`text-embedding-3-small`) | 8,191 tokens: a whole abstract fits with a wide margin |

Consequences of chunking abstracts at 500 characters:

- A 255-token text is cut into 3–4 pieces, sometimes mid-sentence.
- The top-5 repeats papers: in 39 of 50 single-gold examples some paper appears more than once (3.80 distinct papers in 5 slots).
- `chunk_id = arxiv_id:sha1(page_content)[:12]` changes with any chunking change, so chunk-level gold labels cannot be
  compared across chunking configurations (checked on one paper: 500 → 4 chunks, 300 → 7, 800 → 3, with no shared ids).
  This makes the planned EXP-3 (500/0 vs 1000/200) unmeasurable at chunk level.

Measured comparison of both indexes on the same golden-set questions (paper-level gold, k=5, `text-embedding-3-small`):

| Slice (n) | Index | hit@1 | hit@5 | MRR | Distinct papers in top-5 |
|---|---|---|---|---|---|
| answerable (40) | chunks 500/0 | 1.00 | 1.00 | 1.000 | 3.80 |
| | whole abstract | 0.90 | 1.00 | 0.934 | 5.00 |
| multi-doc known-item (10) | chunks 500/0 | 0.70 | 0.90 | 0.740 | 4.30 |
| | whole abstract | 0.70 | 0.90 | 0.750 | 5.00 |
| persona (10) | both | 1.00 | 1.00 | 1.000 | 3.80 vs 5.00 |

Cost and latency of generation (`gpt-4o-mini`, `PROMPT_V1`, temperature 0; 20 questions per arm, order alternated):

| | Chunks 500/0 | Whole abstract | Ratio |
|---|---|---|---|
| Input tokens (mean) | 606 | 1,571 | 2.6× |
| Output tokens (mean) | 50.1 | 54.6 | ~1.1× |
| Median / p90 latency | 1.03 s / 1.44 s | 1.11 s / 1.59 s | +8% |
| Cost per query* | ~US$ 0.00012 | ~US$ 0.00027 | 2.2× |

\* At US$ 0.15 / 1M input tokens and US$ 0.60 / 1M output tokens (third-party price pages; confirm on OpenAI's site before budgeting).

The golden set slices `answerable` and `persona` were generated **from 500-character chunks**, which favors the chunk index
in the comparison above (the 0.90 hit@1 of the whole-abstract index is likely partly an artifact of that).

The decision is taken **before EXP-0** because no baseline has been recorded yet: changing the unit later is far more expensive.

## Decision

The baseline index holds **one record per paper, containing the whole abstract** (the abstract is the "chunk").

- **Chunking 500/0 (and 1000/200 if still relevant) becomes an experimental arm of CKPT-3.** EXP-3 is reframed as
  "does splitting the abstract help or hurt?", compared at paper level (see ADR-006).
- **Contingency:** if the golden set generated from chunks biases the metrics against the whole-abstract index (hit@1 below
  the chunk index for reasons unrelated to retrieval quality), **regenerate the `answerable` and `persona` questions from the
  whole abstract** instead of from chunks.
- Quotations in answers (citing "paper Y says Z") do not depend on chunks: the generator can return a literal sentence of the
  abstract next to the `arxiv_id`, and the citation evaluator checks that it exists in the abstract.

Implementation (pending, next slice): an "abstract" indexing strategy in `utils.py`/`config.py` with its own cache, gold derived
from the `arxiv_id`, and updated baseline definitions in `docs/plan.md` (CKPT-0 and §1).

## Alternatives considered

### Alternative A: Keep chunks 500/0 as the baseline (current plan)

Keep the plan unchanged and only fix evaluation by scoring at paper level.

**Why not chosen**:
- Splits a 255-token text into fragments and repeats papers in the top-5 (3.80 distinct of 5).
- Keeps the "book + page" complexity (paper level plus a chunk-level diagnostic) in every report.
- Keeps EXP-3 fragile: it cannot be compared at chunk level because ids change with the chunking.

**Advantages** (not leveraged):
- Plan and golden set stay consistent with each other (questions were generated from these chunks).
- Cheaper per query (2.2×) and slightly lower input size.
- The baseline stays deliberately adversarial for the failure modes CKPT-3 studies (FP3, R4, FP4).

### Alternative B: Index both (whole abstract and 500/0 chunks) from the start

**Why not chosen**:
- Two pipelines and two caches before any evidence says both are needed.
- The chunked arm can be added at CKPT-3 with the cache that already exists in `resources/`.

**Advantages** (not leveraged):
- The comparison is available immediately.

## Consequences

### Accepted

- Chunk = paper: no "page" level, no per-paper chunk lists, and the retrieval gold is simply the paper (ADR-006).
- The top-5 always holds 5 distinct papers.
- The retrieval unit equals the citation unit of the product.
- EXP-3 no longer depends on chunk ids.

### Trade-offs

- Generation input is 2.6× larger: ~2.2× the cost per query (cents) and about +8% median latency (n=20, treat as an order of magnitude).
  Judges that read the context (faithfulness, citation) scale the same way, so each experiment run costs roughly 2× more.
- **Answer quality has not been measured** (faithfulness, correctness with five whole abstracts vs five fragments). It needs the
  judges of CKPT-0.6.3 and calibration with labelled real answers.
- The golden set bias above: `answerable` hit@1 0.90 vs 1.00 may be partly an artifact.
- The baseline is less adversarial for chunking-related failure modes; CKPT-3 changes shape.
- With 250 documents `hit@5` is already 100% on `answerable` and `persona` for both indexes: those slices do not discriminate retrieval at baseline.
- Hypothesis, **not measured**: with whole abstracts, k=3 (~1,060 input tokens, arithmetic estimate) could cover as much as k=5 chunks. To be tested in CKPT-1.

### Conditions that invalidate this decision

This decision should be **revisited** if:

1. Answer quality with whole-abstract context is worse than with chunk context by more than the judge noise floor.
2. The hit@1 gap persists after regenerating `answerable`/`persona` from the whole abstract.
3. The corpus grows beyond abstracts (full text or long documents), where one record per document no longer fits an embedding input.
4. Cost or latency at production volume becomes a constraint that the quality gain does not justify.
5. A larger corpus shows embedding dilution (whole-abstract vectors matching specific details worse than chunks).

### Migration path when needed

1. Run the chunked configuration as the experimental arm of CKPT-3 (the 500/0 cache already exists in `resources/`).
2. If the golden set is the cause of a regression, regenerate `answerable`/`persona` from whole abstracts (contingency above).
3. To revert, switch the baseline indexing strategy back to 500/0 in `config.py`; paper-level evaluation (ADR-006) works for either unit.

## References

- Lesson: [Retrieval Unit and Gold Granularity for RAG over Abstracts](../learning-lessons/retrieval_unit_and_gold_granularity.md)
- Roadmap: [`ckpt-0.6.1b-metrics-roadmap.md`](../roadmap/ckpt-0.6.1b-metrics-roadmap.md) (D-1, D-7)
- ADR-006 (gold granularity and metrics), ADR-003 (hybrid multi-doc slice), ADR-002 (corpus snapshot pin)
- `docs/plan.md`: CKPT-0, §1 (corpus sizing), CKPT-3 (chunking)
- Reading notes: `docs/research/fichamentos/thakur-2021-beir.md` (abstracts as retrieval documents in SciFact/SCIDOCS), `ru-2024-ragchecker.md`, `barnett-2024-seven-failure-points.md` (R7: chunk size trade-off, no rule given)
