# Ru et al. 2024 — RAGChecker (claim-level RAG diagnosis)

- **Full reference:** Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang, Jiayang Cheng, Cunxiang Wang, Shichao Sun, Huanyu Li, Zizhao Zhang, Binjie Wang, Jiarong Jiang, Tong He, Zhiguo Wang, Pengfei Liu, Yue Zhang, Zheng Zhang. *RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation.* arXiv:2408.08067. <https://arxiv.org/abs/2408.08067> · Code: <https://github.com/amazon-science/RAGChecker>
  *(The author list and venue — NeurIPS 2024, Datasets & Benchmarks — come from my prior knowledge/the roadmap and were **not verified** in the text I read; confirm before citing at a defense.)*
- **Reading confidence:** **complete.** I read the raw text of the arXiv HTML version: Sections 1–5, Tables 1–3, and Appendices A–I (benchmark, formulas, meta-evaluation, setup, per-dataset results — Tables 4–12 seen up to Recreation —, diagnosis, limitations). **I did not see:** figures (Fig. 1 and 4–10 only through the captions), Tables 13–15 (remaining domains) and Table 16 (validation of RefChecker with Llama3). The PDF could not be decoded; I used the HTML.

> Convention: **[Paper]** = stated in the text/tables; **[My reading]** = my interpretation.

---

## 1. Problem the paper addresses

**[Paper]** Three challenges (Section 1): **(1) modular complexity** — RAG has a retriever and a generator, and it is necessary to understand the origin of errors; **(2) metric limitation** — recall@k and MRR "depend on annotated chunks and a rigid chunking approach, missing out on the full semantic scope of the knowledge base"; BLEU/ROUGE/BERTScore and LLM judges "perform well with concise answers but fail to detect finer distinctions in longer responses"; **(3) metric reliability** little explored (alignment with humans).

Response: **claim-level** evaluation (atomic claim) with extraction and entailment checking, producing **overall**, **retriever** and **generator** metrics.

## 2. Method (step by step, exact formulas)

### 2.1 Formulation and inputs (Section 3, 3.1)

RAG = {R, G}. Given a query q and documents D: it retrieves top-k chunks {chunkⱼ} = R(q, D, k) and generates the answer m = G({chunkⱼ}, q). Each benchmark sample is ⟨q, D, gt⟩: query, documents (segmented into chunks of the same size in tokens) and a **complete and correct reference answer, in long form**.

**Two personas** (Section 3, "Design Principle"): the *user* who wants a single number to compare systems; the *developer* who wants to find the **cause** of errors — retrieval error (incomplete/irrelevant context) or generator error (fails to identify/use the relevant information).

### 2.2 Claim-based evaluation (Section 3.2)

Two components: **(1) text→claims extractor** (decomposes a text T into claims {cᵢ}); **(2) entailment checker** (decides whether a claim c is *entailed* (∈) or *not* (∉) in a reference text *Ref*). Implementation: **Llama3-70B** as extractor and checker, via **RefChecker** (Section 4.1; validated on the RefChecker benchmark — Appendix G; Table 16 values not read).

### 2.3 Metrics — definitions (Section 3.3 and formulas in Appendix B)

Notation: m = model answer; gt = reference answer; {chunkⱼ} = retrieved chunks. A chunk is **relevant (r-chunk)** if it **contains at least one claim of the gt** (∃i: cᵢ(gt) ∈ chunkⱼ); the others are **irrelevant (irr-chunk)**.

**Overall metrics (user view; 3.3.1):**
- **Precision** = |{cᵢ(m) | cᵢ(m) ∈ gt}| / |{cᵢ(m)}| — *(of the claims the answer made, what fraction is confirmed by the reference answer)*.
- **Recall** = |{cᵢ(gt) | cᵢ(gt) ∈ m}| / |{cᵢ(gt)}| — *(of the claims in the reference answer, what fraction the answer covered)*.
- **F1** = harmonic mean of the two — *(single overall score)*.

**Retriever metrics (3.3.2):**
- **Claim Recall** = |{cᵢ(gt) | cᵢ(gt) ∈ {chunkⱼ}}| / |{cᵢ(gt)}| — *(of the claims needed for the reference answer, what fraction is **present** in the retrieved chunks; measures whether the retriever brought the necessary information, **regardless of ID or position**)*.
- **Context Precision** = |{r-chunkⱼ}| / k — *(of the k retrieved chunks, how many have at least one useful claim; defined at the **chunk** level, not the claim level, for interpretability — the paper notes that a chunk can mix useful and misleading information, so the ceiling of claim-level precision is < 100%)*.

**Generator metrics (3.3.3) — all over the claims of the answer, except Context Utilization:**
- **Faithfulness** = |{cᵢ(m) | cᵢ(m) ∈ {chunkⱼ}}| / |{cᵢ(m)}| — *(fraction of the answer's claims supported by the chunks; faithfulness to the context)*.
- **Relevant Noise Sensitivity** = |{cᵢ(m) | cᵢ(m) ∉ gt **and** cᵢ(m) ∈ {r-chunkⱼ}}| / |{cᵢ(m)}| — *(fraction of **wrong** claims that came from a **relevant** chunk: the generator was misled by noise mixed with useful information)*.
- **Irrelevant Noise Sensitivity** = same with ∈ {irr-chunkⱼ} — *(wrong claims coming from an irrelevant chunk)*.
- **Hallucination** = |{cᵢ(m) | cᵢ(m) ∉ gt **and** cᵢ(m) ∉ {chunkⱼ}}| / |{cᵢ(m)}| — *(wrong claims that are **not** in any chunk: invention by the generator)*.
- **Self-knowledge** = |{cᵢ(m) | cᵢ(m) ∈ gt **and** cᵢ(m) ∉ {chunkⱼ}}| / |{cᵢ(m)}| — *(**correct** claims that did not come from any chunk: the generator got it right using its own knowledge; lower is better when dependence on the context alone is expected)*.
- **Context Utilization** = |{cᵢ(gt) | cᵢ(gt) ∈ {chunkⱼ} **and** cᵢ(gt) ∈ m}| / |{cᵢ(gt) | cᵢ(gt) ∈ {chunkⱼ}}| — *(of the reference-answer claims that the retriever **managed to bring**, how many the answer actually incorporated: "found and used")*.

### 2.4 Benchmark and systems (Section 4.1; Appendices A and D)

- **Benchmark:** **4,162 queries in 10 domains** (Table 1): ClapNQ 300, NovelQA 280, RobustQA Writing/BioASQ(511)/Finance/Lifestyle/Recreation/Science/Technology (500 each, BioASQ 511) and KIWI 71 (NLP papers, 429 documents). Short answers from the datasets were converted into **long answers** (GPT-4 + quality control with RefChecker) — Appendix A.
- **8 systems** = {BM25, E5-Mistral} × {GPT-4, Mixtral-8x7B, Llama3-8B, Llama3-70B}. Chunks of **300 tokens**, overlap **0.2**, **k = 20**, temperature 0 (Appendix D).

### 2.5 Meta-evaluation with humans (Section 4.2; Appendix C)

- **280 instances** = 28 system pairs × 10 domains; each instance is a **pair of answers** from two systems to the same question.
- Two annotators per instance, who choose among 5 options (significantly better … significantly worse) on **correctness, completeness and overall**; 10 annotators in total (7 internal, 3 graduate students; US$ 255 in total — Appendix C). Annotators see GPT-4-generated critiques comparing each answer with the reference answer.
- **Human agreement:** proportion with |hᵢ − hᵢ′| ≤ 1 = **90.95%**.
- Metric: correlation (Pearson/Spearman) between the human preference difference and the normalized difference of the metric's score.
- Baselines: 10 metrics from TruLens, RAGAS, ARES and CRUD-RAG (+ BLEU, ROUGE-L, BERTScore); **Llama3-70B-Instruct backbone** "when applicable" for all (embeddings: each one's default backbone).

## 3. Main contributions

1. **Claim-level** framework with metrics separated by component (retriever × generator) + an overall metric — Section 3.
2. New metrics for the generator: **context utilization, relevant/irrelevant noise sensitivity, self-knowledge** (in addition to faithfulness and hallucination) — Section 3.3.3.
3. **Human meta-evaluation** (280 instances) showing higher correlation than the other metrics — Section 4.2/Table 2.
4. **Benchmark** of 4,162 queries, 10 domains, long answers — Table 1/Appendix A.
5. Diagnostic findings (retriever↔generator trade-offs; effect of k, chunk, overlap, prompt) — Sections 4.3–4.4/Appendix F.

## 4. Key results (with location)

**Table 2 (Section 4.2) / Table 5 (Appendix C)** — correlation with human preference (Pearson; Correctness / Completeness / Overall):

| Metric | Correctness | Completeness | Overall |
|---|---|---|---|
| **RAGChecker** (same metric as the human) | **49.66** | **60.67** | **61.93** |
| RAGAS Answer Similarity | 41.07 | 53.16 | 48.31 |
| ROUGE-L | 31.75 | 47.88 | 43.10 |
| CRUD-RAG Recall | 30.93 | 45.11 | 41.25 |
| BLEU-avg | 38.89 | 32.13 | 35.14 |
| TruLens Answer Relevance | 35.01 | 37.24 | 35.15 |
| BERTScore | 30.34 | 37.93 | 33.51 |
| TruLens Groundedness (Tab. 5) | 21.11 | 14.01 | 19.45 |
| ARES Answer Relevance | 18.63 | 20.13 | 17.81 |
| ARES Answer Faithfulness (Tab. 5) | 9.46 | 10.25 | 8.80 |
| RAGAS Faithfulness (Tab. 5) | 8.22 | 4.90 | 7.83 |
| RAGAS Answer Relevance (Tab. 5) | 11.59 | 9.39 | 10.27 |
| *Human × human (ceiling)* | *63.67* | *71.91* | *70.09* |

Spearman (overall): RAGChecker 60.90; RAGAS Answer Similarity 57.23; human 68.89 (Table 5).

**Table 3 (Section 4.3)** — averages over the 10 domains (Prec / Rec / F1 | CR / CP | CU / NS(I) / NS(II) / Hallu / SK / Faith | #claims):

| System | Prec | Rec | F1 | CR | CP | CU | NS(I) | NS(II) | Hallu | SK | Faith | #claims |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25_GPT-4 | 61.0 | 49.7 | 50.3 | 74.0 | 52.3 | 61.4 | 26.2 | 4.1 | 8.7 | 3.4 | 87.9 | 12 |
| E5-Mistral_GPT-4 | 62.0 | 53.0 | **52.7** | 83.5 | 61.8 | 60.4 | 28.9 | 3.5 | 5.7 | 1.4 | 92.9 | 12 |
| E5-Mistral_Llama3-8b | 53.8 | 48.3 | 45.0 | 83.5 | 61.8 | 55.0 | 33.5 | 5.5 | 6.6 | 0.8 | 92.7 | 11 |
| E5-Mistral_Llama3-70b | 60.6 | 50.4 | 50.2 | 83.5 | 61.8 | 57.6 | 31.7 | 4.3 | 3.3 | 0.8 | 95.9 | 10 |
| BM25_Llama3-70b | 59.1 | 44.9 | 46.3 | 74.0 | 52.3 | 56.2 | 30.4 | 5.3 | 5.1 | 1.7 | 93.2 | 9 |

(The remaining 3 systems are in the original Table 3; omitted here.) Textual insights (Section 4.3): (i) the retriever matters consistently; (ii) Llama3-70B > Llama3-8B on all generator metrics; (iii) **context utilization correlates strongly with the overall F1**; (iv) more informative context improves faithfulness and reduces hallucination/self-knowledge; (v) **the retriever's claim recall has a trade-off with the generator's noise sensitivity** (more recall → more noise accompanying the useful information); (vi) **relevant noise** weighs more than irrelevant — "generators demonstrate a chunk-level faithfulness"; (vii) open-source models "are faithful but tend to trust the context blindly".

**Section 4.4 / Appendix F** — configuration adjustments (3 systems × 3 domains: Writing, Finance, KIWI):
- **k from 5→20:** claim recall 61.5→77.6; faithfulness 88.1→92.2; noise sensitivity 34.0→35.4; F1 51.7→53.4. **Chunk size 150→300:** claim recall 70.3→77.6; F1 52.6→53.4.
- **Overlap:** context precision 69.3→71.1, but claim recall 77.8→78.1 — more chunks with the *same* information; "Chunk Overlap Does Not Matter a Lot".
- **Prompt with explicit requirements:** faithfulness 92.2→93.6; context utilization 59.2→63.7, **but** noise sensitivity 35.4→38.1 — a "trilemma" among context utilization, noise sensitivity and faithfulness.

## 5. Limitations

**[Paper] (Appendix H):** (1) the **retriever** metrics are "less insightful" than the generator's (they focus on claim recall and context precision; they do not capture density, diversity, coherence); (2) they **do not distinguish Neutral from Contradiction** in the checker's results; (3) benchmark is **text-only and English-only**, reused from existing datasets.

**[My reading]**
- **Validation only of the overall metrics:** the meta-evaluation (Section 4.2) compares correlation with preference on *correctness/completeness/overall*; the **diagnostic** metrics (claim recall, context precision, context utilization, noise sensitivity, hallucination, self-knowledge) were **not** validated against humans. Take them with caution.
- **Not a like-for-like comparison:** all baselines use Llama3-70B; **ARES** here is not the domain-fine-tuned DeBERTa judge of the original paper, and *faithfulness* metrics (RAGAS/ARES/TruLens) were not designed to predict "correctness vs. reference answer" (the human target) — the very low correlations (7.83; 8.80) reflect in part this **target mismatch**, not necessarily a failure of the metric.
- **Dependence on the long reference answer:** the basis of all metrics is the list of claims of the `gt`; short or incomplete reference answers limit claim recall/precision. The long reference answer was **generated by GPT-4** from short answers (Appendix A.2), with RefChecker's own filter.
- **Extractor/checker = LLM:** extraction/entailment errors propagate; the validation of RefChecker with Llama3 (Appendix G, Table 16) was not read by me.
- **Cost:** the checker runs per claim × chunk (k = 20 per query) — expensive at scale.
- **Abstention absent:** there are no unanswerable queries; with an empty gt, claim recall/recall/CU are undefined (0/0). *(My inference.)*
- Humans see GPT-4-generated critiques (Appendix C) — possible anchoring of the judgment.

## 6. Reflections anchored in OUR project

**R1 — Claim recall / context precision × `gold_chunk_ids` (D-1, P6, P1–P3; `evaluators.py: precision_at_k, recall_at_k`).** *CHALLENGES gold by ID and SUPPORTS the anti-pooling semantics.* RAGChecker labels the **relevance of chunks by content** (a chunk is relevant if it *entails* ≥1 claim of the gt), instead of assuming "non-gold = irrelevant". This attacks exactly the problem of P6 — **sibling** chunks of the same paper that also answer — **without** manually adjudicating all chunks, and is an automated answer to the "expanded chunk gold" of D-1. The paper also explicitly criticizes recall@k/MRR for depending on "annotated chunks" (Section 1). **Condition:** the checker (LLM) would need to be **validated on a sample of our own** (e.g., 30–50 chunk×claim pairs labeled by a human), because the paper does **not** validate the diagnostic metrics against humans. *Proposal, not decision:* adopt as a **diagnostic parallel** to the ID-based metrics in EXP-0, measuring the disagreement (no. of "sibling" chunks that the checker marks as relevant and the ID-based gold marks as errors).

**R2 — Fragility of `gold_chunk_ids` to chunking changes (CKPT-3; `utils.chunk_id`).** *CHALLENGES our design by chunk ID.* The plan foresees gates on "new chunk" (CKPT-3). In `utils.py`, `chunk_id(doc) = f"{arxiv_id}:{sha1(page_content)[:12]}"` — it is a **hash of the content**; any re-chunking changes the ID and invalidates `gold_chunk_ids`, while the `arxiv_id` prefix survives. RAGChecker is robust to this (relevance by claim) and Section 4.4 tests chunk size/overlap precisely. **Implication for D-1:** for the chunking experiments, use the **paper level** (or claim level) as the comparable measure across configurations; keep chunk ID only for the baseline. This favors the "two levels" option **with the comparison across chunking configurations done at the paper/claim level**.

**R3 — Precision inflated by redundancy (P1–P3 dedup, P6 `distinct_papers@k`; `decisions.md`: same paper 3× in the top-5).** *SUPPORTS.* In Appendix F, increasing overlap raises context precision (69.3→71.1) **without** raising claim recall (77.8→78.1), because of "the retrieval of more chunks that contain the same segment of useful information". It is the analog of our observed collapse (same paper repeated). It reinforces: (i) deduplicate by paper before measuring; (ii) treat `precision_at_k` in isolation as misleading; (iii) report `distinct_papers@k` alongside.

**R4 — *Context utilization*: separating "did not find" from "found and did not use" (D-2, P5; EXP-0 report, plan §3 matrix).** *SUPPORTS and adds a diagnostic we lack.* The paper shows (Section 4.3) that context utilization correlates strongly with the overall F1. **Cheap version, without claim extraction:** report answer quality (correctness/completeness) **conditioned** on hit@k = 1 versus hit@k = 0. This answers the survey question (retrieval ↔ generation relationship) and gives meaning to the hit/recall pair when |gold| = 1 (D-2): instead of reporting hit and recall as two "pieces of evidence", report hit **and** the answer's hit rate given hit. **Full version** (claims) only if cost allows (see R7).

**R5 — Hallucination × Self-knowledge refine `faithfulness_judge` and the reading of abstention (P7, P8, `unanswerable`/`stale`).** *SUPPORTS/refines.* The plan's `faithfulness_judge` treats "not supported by the context" as a single block. RAGChecker separates **claim unsupported and wrong (hallucination)** from **claim unsupported but correct (self-knowledge)**. For `unanswerable`, a non-abstaining answer with unsupported claims is *under-refusal*; knowing whether they are false (hallucination) or true (self-knowledge) distinguishes "made it up" from "answered from parametric knowledge" — useful information for the release-blocking abstention gate. Limitation: it needs a gt to decide "correct", and `unanswerable` has an empty gt (R6).

**R6 — Abstention stays outside RAGChecker (P7, P8).** *NEUTRAL.* Without unanswerable queries, the metrics were not defined for an empty gt; they do **not** replace our abstention P/R/F1 nor `abstention_quality`. Use RAGChecker only on the slices with a gt (answerable, multi-doc, persona).

**R7 — Cost and P11 (judge ≠ generator; `config.py`).** *PARTIALLY CHALLENGES.* Each claim is checked against each chunk (k = 20 in the paper; our k is smaller, ~5, with ~100 examples: feasible). But RAGChecker uses **Llama3-70B** to extract/check, different from the generators; in our plan the judge would be the **same** gpt-4o-mini as the generator (P11). If we adopt claim-level, it is worth using a model **distinct** from the generator for the checker; the paper (Appendix G) validates Llama3-70B on the RefChecker benchmark — not gpt-4o-mini.

**R8 — P9 (token-F1 × semantic judgment; `f1_summary_evaluator`).** *SUPPORTS the switch, with a caveat.* In Table 2/5, lexical/embedding metrics vs. reference answer correlate weakly with humans: BLEU 35.14, ROUGE-L 43.10, BERTScore 33.51 (Overall Pearson), against RAGChecker 61.93 and human ceiling 70.09. This is evidence (on **long answers**, 10 domains) in favor of replacing token-F1 with a semantic/claim-level judgment. **Caveats:** (a) RAGChecker's reference answer is long and made of several claims; our gold answers are short (1–3 sentences, the prompt limits to 3 sentences), where the difference between token-F1 and claim-F1 is smaller and the effect is uncertain; (b) calibration with **our** labels is missing (P9 step 4). The **Overall F1 of claims** is, in essence, the semantic version of what our token-F1 tries to be — a natural candidate to replace it, but only after local validation.

**R9 — CU/NS/Faithfulness trilemma and CKPT-6 (prompt; P8 over/under-refusal).** *SUPPORTS by analogy.* The paper shows that a prompt that asks for more faithfulness and use of the context improves CU (59.2→63.7) but worsens noise sensitivity (35.4→38.1) — "makes it difficult to improve all aspects simultaneously". This reinforces our P8 rule (record over-refusal when improving abstention recall) as an instance of the same principle: **every prompt change must report the opposing metrics together**. It is not evidence about abstention itself.

**R10 — Choice of k (CKPT-1).** *SUPPORTS the experiment design.* k 5→20 improves claim recall and faithfulness, at the cost of more noise sensitivity and saturation (Section 4.4). Our k gate must also look at the **answer**, not just hit@k; the paper uses data with long answers and k = 20, so the values **do not transfer** to our case (abstracts of ~420 characters, k ≈ 5).

**What we would adopt:** (1) context utilization in a cheap version (answer conditioned on hit@k) — R4; (2) hallucination × self-knowledge as a refinement of `faithfulness_judge` — R5; (3) **claim recall / context precision by entailment as a diagnostic parallel** to the ID-based metrics, after validation on a sample of our own — R1/R2; (4) paper/claim level to compare chunking configurations — R2. **What we would NOT adopt:** (1) the full RAGChecker pipeline as the main metric now (cost, long reference answer, unvalidated diagnostics, checker ≠ our model choice); (2) the RAGAS/ARES faithfulness correlations from Table 2 as proof that they are bad (different human target); (3) transferring the paper's numeric k/chunk values (distinct domain and chunk size).

## 7. Useful quotations

1. *"traditional metrics like recall@k and MRR for retrievers depend on annotated chunks and a rigid chunking approach, missing out on the full semantic scope of the knowledge base."* — Section 1, challenge (2) "metric limitation".
2. *"A retrieved chunk is called relevant chunk (r-chunk), if any ground-truth claim is entailed in it."* — Section 3.3.2 (Retriever Metrics).
3. *"the diagnostic metrics for the retriever component are less insightful compared to those for the generator."* — Appendix H (Limitations).
