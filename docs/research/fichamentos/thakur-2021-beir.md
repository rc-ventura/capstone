# Thakur et al. 2021 — BEIR: heterogeneous zero-shot retrieval benchmark (and the annotation-bias study)

- **Full reference and link:** Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., Gurevych, I. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. NeurIPS 2021, Datasets and Benchmarks Track. arXiv:2104.08663 — <https://arxiv.org/abs/2104.08663>. Code: <https://github.com/UKPLab/beir>.
- **Reading confidence:** **full.** I read the PDF `arxiv.org/pdf/2104.08663` — I confirmed it is **v4** (24 pages, 21/Oct/2021; the header still says "Preprint. Under review"), including Tables 1, 2 and 4 (extracted with layout) and appendices D, F, G and H. Page citations refer to this v4. I did not read the BEIR code nor the leaderboard; numbers from earlier versions (v1, 21 pages) may differ.

> Convention: **(F)** = what the source states; **(I)** = my interpretation.

## 1. Problem the source addresses
(F, §1, p.1–2) Neural retrieval models were evaluated in homogeneous settings (same training and test dataset, e.g., MS MARCO/NQ), which says nothing about **out-of-domain generalization**. Existing benchmarks (MultiReQA, KILT) cover a single task or domain. BEIR proposes a **zero-shot** benchmark with 18 datasets from 9 tasks and compares 10 systems from 5 architectures. As a second finding (and the one that interests us most), the paper shows that the datasets have **lexical annotation bias** that disadvantages non-lexical retrievers (§6).

## 2. Method (with the exact definitions)
(F)
- **Scope (§2, §3):** "document" is used as "cover term for text of any length" (§2); queries are also of any length. Datasets selected by four criteria: diverse tasks, diverse domains, sufficient difficulty, diverse annotation strategies (§3). Single format (corpus, queries, qrels). The 18 zero-shot + MS MARCO (in-domain, outside the average).
- **Statistics (Tab. 1, p.4):** e.g., SciFact — corpus of 5,183 abstracts, 300 queries, **1.1 relevant/query**; SCIDOCS — 25,657 documents, 1,000 queries, 4.9 relevant/query, task "citation prediction"; TREC-COVID — 171,332 docs, 50 queries, **493.5** relevant/query (3 levels); NQ — 1.2; ArguAna — 1.0.
- **Systems (§4):** lexical (BM25, Anserini, k1=0.9 b=0.4); sparse (DeepCT, SPARTA, docT5query); dense (DPR, ANCE, TAS-B, **GenQ** = TAS-B fine-tuned on **synthetic queries generated from the documents themselves** with T5, 5 per doc, ceiling of 100 thousand docs); late-interaction (ColBERT); re-ranking (BM25 + MiniLM cross-encoder top-100). Documents truncated to **512 word pieces** in the neural models.
- **Main metric (§4, p.5):** **nDCG@10** (normalized Discounted Cumulative Gain: rewards relevant items in the top positions and accepts degrees of relevance; in trec_eval, gain/log₂(position+1), normalized by the ideal order), computed by the Python interface of the official trec_eval. The authors' literal justification: "Decision support metrics such as Precision and Recall which are both rank unaware are not suitable. Binary rank-aware metrics such as MRR and MAP fail to evaluate tasks with graded relevance judgements."
- **Recall@k (Appendix G, p.20):** `R@k = (1/|Q|) Σ_i |top-k(A_i) ∩ A*_i| / |A*_i|` (of all the relevant items of the query, what fraction appeared in the top-k) and **Capped Recall@k**: denominator `min(k, |A*_i|)` — prevents queries with more than k relevant items (e.g., 500) from having a low ceiling (500 relevant ⇒ R@100 max. 0.2). **BEIR does not define hit rate/success@k nor MRR@k as a reported metric**; the software "provides you with all IR-based metrics from Precision, Recall, MAP (Mean Average Precision), MRR (Mean Reciprocal Rate) to nDCG" (§3.2, p.5).
- **Annotation-bias study (§6, p.8–9):** the authors compute the **Hole@10** of each system on TREC-COVID — fraction of the top 10 results **without a judgment** (never seen by the annotators) — and **manually annotate the holes**, without knowing which system retrieved them ("to avoid a preference bias"), following the original guidelines; they recompute nDCG@10.

## 3. Main contributions
- Unified zero-shot benchmark, with a standard format and Python package (integrates Sentence-Transformers, Anserini, DPR, ColBERT, Elasticsearch…).
- Evidence that in-domain performance does **not predict** generalization; BM25 is a strong baseline.
- Comparison of 5 retriever families with cost (latency and index size, §5.1, Tab. 3).
- **Quantification of the selection/lexical bias of the judgments** (Tab. 4) — the most relevant contribution for us.
- Metric standard: nDCG@10 as a single metric comparable across tasks, and Capped Recall@k as a comparable recall.
- Analysis of length preference (TAS-B prefers short docs; Appendix H: effect of cosine similarity × dot product).

## 4. Key results
- **Tab. 2 (p.7) — zero-shot nDCG@10** (BM25 / BM25+CE / TAS-B / ANCE / docT5query): SciFact 0.665 / 0.688 / 0.643 / 0.507 / 0.675; SCIDOCS 0.158 / 0.166 / 0.149 / 0.122 / 0.162; TREC-COVID 0.656 / 0.757 / 0.481 / 0.654 / 0.713; NQ 0.329 / 0.533 / 0.463 / 0.446 / 0.399. Row "Avg. Performance vs. BM25" (Tab. 2, in the column order DeepCT, SPARTA, docT5query, DPR, ANCE, TAS-B, GenQ, ColBERT, BM25+CE): −27.9%, −20.3%, +1.6%, −47.7%, −7.4%, −2.8%, −3.6%, +2.5%, +11%.
- **Enumerated findings (§5, p.6–7):** (1) in-domain ≠ generalization: BM25 loses 7–18 points on MS MARCO but is strong on BEIR; (2) BM25+CE re-ranking beats BM25 on **16/18** datasets; (3) docT5query beats BM25 on 11/18; (4) GenQ helps in specialized domains (scientific, finance) and worsens on Wikipedia; (5) TAS-B retrieves short documents (median 10 words on TREC-COVID vs 160 for ANCE, Fig. 4).
- **Tab. 4 (p.9) — Hole@10 on TREC-COVID:** BM25 **6.4%**; docT5query **2.8%**; BM25+CE 1.6%; ColBERT 12.4%; SPARTA 12.4%; ANCE **14.4%**; DeepCT 19.4%; DPR 30.6%; TAS-B **31.8%**. **980 query-document pairs** were annotated.
- **nDCG@10 before → after annotating the holes (Tab. 4):** BM25 0.656 → 0.668; docT5query 0.713 → 0.714; BM25+CE 0.757 → 0.760; **ANCE 0.654 → 0.735** (from slightly below to 6.7 points above BM25); **ColBERT 0.677 → 0.735** (+5.8); DPR 0.332 → 0.445; TAS-B 0.481 → 0.555; DeepCT 0.406 → 0.472; SPARTA 0.538 → 0.624.
- **The authors' conclusion (§6):** "Even though many systems contributed to the TREC-COVID annotation pool, the annotation pool is still biased towards lexical approaches."

## 5. Limitations
- **From the authors:** need for "better unbiased datasets" and for diverse pools (§6–§7); English only; truncation to 512 word pieces; GenQ limited to 100 thousand docs due to resource constraints (§4).
- **Mine (I):**
  1. The hole study covers **one dataset** (TREC-COVID, 50 queries), **top-10 only** and a single round; there is no significance test or repetition — the values are illustrative, not a law.
  2. The recommendation "nDCG@10 instead of P/R" comes from a benchmark with graded relevance and hundreds of relevant items in some datasets; it **does not automatically apply** to a top-5 that goes entirely into an LLM's prompt (where the internal order of the top-k weighs less).
  3. BEIR does not discuss **granularity** (passage × document): each dataset uses the unit that came ready-made.
  4. No BEIR dataset has a **synthetic gold generated from the document itself**; GenQ generates synthetic queries but only to *train* retrievers, not for the evaluation gold. Therefore BEIR **does not answer** our concern "synthetic gold ⇒ lexical bias"; it only shows the analogous effect for a gold made with lexical search.
  5. Table 4 in this version does not include the GenQ column.

## 6. Reflections anchored in OUR project

| # | Point / file | Reflection | The source… |
|---|---|---|---|
| 1 | **P5 / D-2**, `hit_rate` vs `recall_at_k` (`evaluators.py`) | With |gold|=1: `recall@k = |top-k ∩ gold|/1 = 1[gold∈top-k] = hit@k` (my derivation). BEIR **does not report hit rate**; it defines Recall@k and **Capped Recall@k** (denominator `min(k,|A*|)`), the latter irrelevant for us (gold ≤3 and k=5 ⇒ cap = |gold|). For a single gold, BEIR uses nDCG@10 — which, with 1 binary relevant item, becomes `1/log₂(position+1)` (my derivation: position 1 → 1.0; 2 → 0.63; 3 → 0.50; 5 → 0.39 vs MRR 1; 0.5; 0.33; 0.2): a monotonic function of the position like MRR, only with a smoother discount. Hence, it supports keeping **hit@k + MRR** (or nDCG@k) on single gold and recall only on multi-gold. | **(i) SUPPORTS** (P5) |
| 2 | **Rank-unaware (P/R) vs rank-aware** — choice of metrics | BEIR rejects P and R for being "rank unaware". This **challenges** presenting `recall@5`/`precision@5` as the main metric; but (I) our consumer is the LLM, which receives the 5 chunks at once — the order within the top-5 matters less than in BEIR's ranking of 10+ results. We would adopt: hit@k (the "does the context have the answer?" bar) + MRR (the "is it at the top?" bar) — without stacking recall; nDCG only when there are degrees of relevance (CKPT-4). | **(ii) CHALLENGES** partially |
| 3 | **D-1 / holes in the 1-chunk gold**, `precision_at_k` (P6) | Tab. 4 shows the exact mechanism: when the evaluated system retrieves something **that nobody judged**, the score drops without the system having erred — and the effect is **uneven**: BM25 +0.012, ANCE +0.081. Our synthetic gold is a pool of **1 item**; the top-5 with siblings from the same paper has a potentially high "Hole@5". We would adopt: reporting **Hole@5** (fraction of the top-5 that is neither gold nor judged) and **judging the holes** in EXP-0 (BEIR annotated 980 pairs for 50 queries ≈ 20/query; for us ≤ ~200–250 items, my estimate: ~50 queries with chunk gold × siblings from the same paper). | **(i) SUPPORTS** (P6, EXP-0 mini-pooling) |
| 4 | **Lexical bias of the synthetic gold** (lesson L1, golden set ADR) | BEIR is **analogous and indirect** evidence: gold derived from lexical search (BioASQ, Signal-1M, TREC-COVID) favors lexical retrievers (§6). Our gold comes from an LLM that read the chunk ⇒ likely vocabulary overlap (I, not tested in any source). Concrete risk: in CKPT-1/4, a dense retriever or a reranker may look worse than BM25/hybrid on the synthetic golden set because of the gold, not the retriever. Mitigation (mine): compare BM25 × dense on the same examples and track each one's Hole@k; include paraphrased/human questions (the `persona` slice already helps). | **(ii) CHALLENGES** the neutrality of the synthetic gold (by analogy) |
| 5 | **D-1** (chunk × paper level) | BEIR uses **the unit that the dataset provides** (abstracts in SciFact/SCIDOCS, without decomposing into passages); it **does not discuss** passage × document nor multi-level. **Neutral** source; the two scientific-abstract datasets (SciFact: 1.1 relevant/query; SCIDOCS: 4.9) are the most similar to our corpus (cs.AI abstracts) and show low nDCG@10 even for the best systems (SCIDOCS ≤ 0.166), a warning that absolute metrics on abstracts are hard to interpret without a BM25 baseline alongside. | **(iii) NEUTRAL** (gap) |
| 6 | **L-3** — ranx as oracle | BEIR computes its metrics with the **official trec_eval via the Python interface** (§4, p.5). This supports using an implementation derived from trec_eval (ranx or `pytrec_eval`) as an oracle: it is the de facto standard in the field. It does not decide between them. See `bassani-2022-ranx.md`. | **(i) SUPPORTS** (oracle concept = trec_eval) |
| 7 | **`unanswerable`/`stale` slice** | BEIR has no unanswerable queries in its datasets; **neutral** source for abstention. | **(iii) NEUTRAL** |

**What we would adopt from BEIR:** (1) a BM25 baseline in parallel to any new retriever, to detect lexical bias in the gold; (2) hole rate (Hole@k) and judging of the holes in EXP-0; (3) at least one rank-aware metric (MRR already exists; nDCG@k when there are degrees); (4) the trec_eval standard as the reference for correctness.
**What we would NOT adopt:** (1) nDCG@10 as the single metric — our k=5 and binary relevance make it redundant with MRR; (2) Capped Recall — it has no effect in our regime; (3) extrapolating BEIR's model rankings to our domain (cs.AI abstracts) — the paper itself says that in-domain performance/generalization do not transfer.

### Answers to the questions in the request (BEIR part)
- **(a)** BEIR **does not define hit rate/success@k**; it defines Recall@k (and Capped) and adopts **nDCG@10** as the single metric, rejecting P/R for being rank-unaware and MRR/MAP for not handling graded relevance. With 1 relevant item, hit@k and recall@k are numerically equal (my derivation); BEIR does not discuss this case.
- **(b)** It does not deal with questions generated from the document itself; it shows lexical bias in qrels derived from term search (Tab. 4, §6).
- **(c)** Stance: **treating unjudged as irrelevant is the practice**; the authors propose **judging the holes** and measuring Hole@10, not using bpref/condensed lists.
- **(d)** No discussion of granularity; unit = document, "a cover term for text of any length" (§2).

## 7. Useful quotations
1. §4 (p.5): "Decision support metrics such as Precision and Recall which are both rank unaware are not suitable."
2. §6 (p.8): "All other unseen documents are assumed to be irrelevant. This is a source for selection bias [39]: A new retrieval system might retrieve vastly different results than the system used for the annotation."
3. §6 (p.9): "Even though many systems contributed to the TREC-COVID annotation pool, the annotation pool is still biased towards lexical approaches."
