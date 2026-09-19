# RAG Failure Modes — Bibliographic Review

> Research foundation for the capstone: the documented bottlenecks of RAG architectures in production, their metrics, and their mitigations. This document is the substrate for the capstone design — every failure point listed here becomes a test (offline eval), a metric (online monitoring), and a mitigation (experiment) in the final product.

---

## Primary Sources

| Source | Contribution |
|--------|--------------|
| **Barnett et al. (2024)** — *Seven Failure Points When Engineering a Retrieval Augmented Generation System* (CAIN 2024, Deakin University) [arXiv:2401.05856](https://arxiv.org/abs/2401.05856) | Canonical catalog of 7 failure points from 3 real-world case studies (research, education, biomedical) + empirical experiment on BioASQ (15k docs, 1k Q&A pairs) |
| **TrustNLP (2026)** — *A Systematic Taxonomy of Failure Modes in RAG Systems* | 33 failure modes across 7 pipeline stages; finds representation, evaluation and agentic orchestration failures under-investigated despite production frequency |
| **clawRxiv (2026)** — *A Taxonomy of Failure Modes in RAG Systems* | 14 failure modes on 3 orthogonal axes (retrieval, fusion, generation); instrumented 7 open-source stacks, 4,812 questions, 3 domains |
| **Respan** — *RAG Evaluation in Production: 6 Metrics That Matter* | Practitioner method at ~80M LLM requests/day: 6 metrics, 2 failure surfaces, golden-set construction from production logs |
| **QubitTool** — *Production RAG Evaluation: Metrics and Release Gates* | Layered evaluation model (retrieval / generation / end-to-end / safety / operations) |
| **Redis** — *RAG Metrics* | How chunk size forces a precision/recall tradeoff; metric-architecture interaction |
| **RaftLabs** — *Why RAG Systems Fail: 7 Failure Modes and Fixes* | Practitioner view: retrieval and data quality account for the majority of "wrong answer" tickets |
| **Suthar** — *RAG Debugging: Retrieval vs Generation Failure* | Diagnostic discipline: measure retrieval first, generation second — same symptom, completely different fixes |

---

## The 7 Failure Points (Barnett et al. 2024)

Organized by pipeline stage. The figure in the paper places them across the **index process** (development time) and the **query process** (runtime).

### Stage 1 — Indexation / Corpus (before any query)

**FP1 — Missing Content**
The question has no answer in the corpus. Ideally the system says "I don't know" — but for questions *related* to the content without answers, the system is fooled into giving a response. The most dangerous case: looks like success, is actually pure hallucination.

**FP1′ — Truncated relevant span** (R4, clawRxiv)
The right document is indexed, but the relevant span lies **outside the chunk window** — the answer was cut mid-way by chunking.

### Stage 2 — Retrieval (query time)

**FP2 — Missed the Top Ranked Documents**
The answer *is* in the document, but it didn't rank in the top-k. K is selected based on performance/cost, not coverage guarantees.

**FP3 — Not in Context — Consolidation Strategy Limitations**
Documents with the answer *were* retrieved from the database, but a consolidation process (context window limits, reranking) dropped them before the LLM saw them.

**Extended retrieval modes** (clawRxiv):
- **R1 — Empty retrieval**: zero documents returned
- **R2 — Off-topic retrieval**: documents returned but no relevant chunks present
- **R3 — Stale retrieval**: relevant chunks present but superseded by newer documents not retrieved
- **R5 — Reranker mis-ordering**

### Stage 3 — Generation (the LLM writing)

**FP4 — Not Extracted**
The answer *is present in the context*, but the LLM failed to extract it. Typically occurs when there is too much noise or contradicting information in the context.

**FP5 — Wrong Format**
The question involved extracting information in a certain format (table, list, JSON) and the LLM ignored the instruction.

**FP6 — Incorrect Specificity**
The answer is returned but is not specific enough — or too specific — for the user's need. Also occurs when users are unsure how to ask and phrase questions too generally.

**FP7 — Incomplete**
Answers are not incorrect but miss part of the information even though it was in the context. Example: "What are the key points covered in documents A, B and C?" answered only partially.

**Hallucination despite correct retrieval** — Stanford researchers found leading legal AI tools still hallucinated on **17–34% of queries** despite retrieving real sources (Magesh et al., 2025). Grounding does not eliminate hallucination. The study also documents a second, more pernicious failure mode: *misgrounded* responses, where the cited source is real but does not support the claim — a failure that a superficial "is this a real citation?" check cannot catch.

### Stage 4 — System / Operations

| Bottleneck | Source |
|------------|--------|
| **Latency + cost under concurrency** — RAG systems struggle with concurrent users due to rate limits and LLM cost; lesson: prepopulate semantic cache with frequently asked questions | Barnett (AI Tutor case) |
| **Stale knowledge base** — "a RAG system is only as current as the documents behind it, and nobody owns keeping them fresh" | RaftLabs |
| **Security** — jailbreaks bypass the RAG system and hit safety training; fine-tuned LLMs can reverse safety training and must be re-tested | Barnett |
| **Chunk size forces a precision/recall tradeoff** — smaller chunks reduce noise but fragment information; larger chunks preserve context but dilute signal. No universally optimal range | Redis |

---

## Lessons Learned (Barnett et al., Section 6)

| Failure Points | Lesson | Detail | Case Study |
|----------------|--------|--------|------------|
| FP4 | Larger context gets better results | 8K vs 4K contexts enabled more accurate responses | AI Tutor |
| FP1 | Semantic caching drives cost and latency down | Prepopulate cache with frequent questions | AI Tutor |
| FP5–7 | Jailbreaks bypass the RAG system | Test all fine-tuned LLMs for RAG safety | AI Tutor |
| FP2, FP4 | Adding metadata improves retrieval | File name + chunk number in retrieved context helped extraction | AI Tutor |
| FP2, FP4 | Open-source embedding models perform well on small text | Performed as well as closed-source alternatives | BioASQ, AI Tutor |
| FP2–7 | RAG systems require continuous calibration | Unknown input at runtime requires constant monitoring | AI Tutor, BioASQ |
| FP1, FP2 | Implement a RAG pipeline for configuration | Calibrate chunk size, embedding strategy, chunking strategy, retrieval strategy, consolidation strategy, context size, prompts | All 3 |
| FP2, FP4 | Bespoke-assembled RAG pipelines are suboptimal | End-to-end training enhances domain adaptation | BioASQ, AI Tutor |
| FP2–7 | Testing performance characteristics is only possible at runtime | Offline evaluation techniques are promising but premised on labelled data | All 3 |

---

## The 5 Recurring Insights Across All Sources

1. **"Most RAG failures start in retrieval, not the model"** (RaftLabs). Retrieval and data quality account for the majority of "wrong answer" tickets — far more than model choice.

2. **"Measure retrieval on its own, then generation on its own, in that order"** (Suthar). The two surfaces fail in ways that are indistinguishable from the outside and require completely different fixes. The classic mistake: rewriting the prompt for six weeks while retrieval is quietly broken.

3. **"Validation of a RAG system is only feasible during operation"** and **"robustness evolves rather than designed in at the start"** (Barnett, the paper's two key takeaways). Offline evaluation cannot fully validate; monitoring is not optional.

4. **Two failure surfaces, measured separately** (Respan, Redis, QubitTool):

   ```
   Retrieval:  recall@k · precision@k · MRR · nDCG · context recall · context precision
   Generation: faithfulness · answer relevance · citation accuracy
   End-to-end: task success · abstention quality
   Safety:     PII leakage · unauthorized retrieval · unsafe actions
   Operations: p95 latency · cost per request · error rate · token/context budget
   ```

5. **Golden set of 100–300 questions from real production logs** (Respan). Synthetic questions are a backstop, never the foundation. Refresh quarterly. Run offline evals on every retriever/prompt change; run online evals on a 1–5% sample of production traffic. The two catch different failure modes.

---

## Pipeline Map

```
INDEX TIME                       QUERY TIME
───────────                      ──────────────────────────────────────────
                                         RETRIEVAL              GENERATION
FP1  missing content        ┌──►  FP2 missed top-k ────────►  FP4 not extracted
FP1′ stale knowledge base   │    FP3 not-in-context           FP5 wrong format
R4   truncated chunk span   │    R1  empty retrieval          FP6 incorrect specificity
     bad chunking           │    R2  off-topic retrieval      FP7 incomplete
                             │    R3  stale retrieval          hallucination w/ correct source
                             │    R5  reranker mis-ordering
                             │
                             └──────────► OPERATIONS: p95 latency, cost, rate limits
                                          SAFETY: jailbreaks, PII leakage
```

---

## Implications for the Capstone Design

The research constrains the capstone design in four ways:

1. **Retrieval and generation must be instrumented separately.** Same symptom (wrong answer), different fixes. The application must trace retrieval and generation as distinct, measurable steps (LangSmith child runs).

2. **The golden set must be built to cover specific failure points**, not generic Q&A. A dataset that only contains answerable questions cannot detect FP1 (missing content). The dataset needs adversarial coverage: answerable, unanswerable, multi-part, format-constrained, stale.

3. **Every failure point becomes a triple**: an offline test (evaluator), an online metric (monitoring rule), and a mitigation experiment (before/after comparison in LangSmith).

4. **Monitoring is part of the product, not an afterthought.** Per Barnett's key takeaway, validation is only feasible during operation — so the capstone product ships with its monitoring loop from day one.

---

## References

- Barnett, S., Kurniawan, S., Thudumu, S., Brannelly, Z., & Abdelrazek, M. (2024). *Seven Failure Points When Engineering a Retrieval Augmented Generation System*. 3rd International Conference on AI Engineering — Software Engineering for AI (CAIN 2024). [arXiv:2401.05856](https://arxiv.org/abs/2401.05856)
- Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C. D., & Ho, D. E. (2025). *Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools*. Journal of Empirical Legal Studies. [DOI:10.1111/jels.12413](https://doi.org/10.1111/jels.12413) · [PDF](https://law.stanford.edu/wp-content/uploads/2024/05/Legal_RAG_Hallucinations.pdf)
- *A Systematic Taxonomy of Failure Modes in Retrieval-Augmented Generation Systems*. TrustNLP 2026. [aclanthology.org/2026.trustnlp-main.27](https://aclanthology.org/2026.trustnlp-main.27.pdf)
- *A Taxonomy of Failure Modes in Retrieval-Augmented Generation Systems*. clawRxiv 2026. [clawrxiv.io/abs/2604.02033](https://clawrxiv.io/abs/2604.02033)
- *Classifying and Addressing the Diversity of Errors in Retrieval-Augmented Generation Systems*. EACL 2026. [aclanthology.org/2026.eacl-long.147](https://aclanthology.org/2026.eacl-long.147.pdf)
- Respan. *RAG Evaluation in Production: 6 Metrics That Matter (2026)*. [respan.ai/articles/rag-evaluation](https://www.respan.ai/articles/rag-evaluation)
- QubitTool. *Production RAG Evaluation: Metrics and Release Gates*. [qubittool.com/blog/rag-evaluation-production-guide](https://qubittool.com/blog/rag-evaluation-production-guide)
- Redis. *RAG Metrics*. [redis.io/blog/rag-metrics](https://redis.io/blog/rag-metrics/)
- RaftLabs. *Why RAG Systems Fail: 7 Failure Modes and Fixes*. [raftlabs.com/blog/why-rag-systems-fail](https://www.raftlabs.com/blog/why-rag-systems-fail)
- Suthar, R. *RAG Debugging: Retrieval vs Generation Failure*. [ruchitsuthar.com/blog/software-architecture/rag-retrieval-vs-generation-failure](https://ruchitsuthar.com/blog/software-architecture/rag-retrieval-vs-generation-failure/)
- Digital Applied. *RAG System Metrics: Recall, Precision, Faithfulness 2026*. [digitalapplied.com/blog/rag-system-metrics-recall-precision-faithfulness-2026](https://www.digitalapplied.com/blog/rag-system-metrics-recall-precision-faithfulness-2026)
