# Barnett et al. 2024 — Seven failure points when engineering a RAG system

- **Full reference:** Barnett, S., Kurniawan, S., Thudumu, S., Brannelly, Z., Abdelrazek, M. (2024). *Seven Failure Points When Engineering a Retrieval Augmented Generation System*. 3rd International Conference on AI Engineering — Software Engineering for AI (CAIN 2024), Lisbon. arXiv:2401.05856 [cs.SE]. Link: <https://arxiv.org/abs/2401.05856> (PDF: <https://arxiv.org/pdf/2401.05856>).
- **Reading confidence:** **complete, but only of arXiv version v1** (2024-01-11, 6 pages), read from the text extracted from the PDF (WebFetch returned the binary PDF; I used `pdftotext`). I read the whole paper (abstract, §1–§7, Tables 1–2, Figure 1 only through the caption; the figure image was not inspected). I did not check later versions or the version published by ACM, so numbers and wording may differ in them. The labels "(a) the paper states" and "(b) my interpretation" are marked in the text.

## 1. Problem the paper addresses

**(a) The paper states:** software engineers are adding semantic search to applications with RAG (retrieving documents and passing them to an LLM), but there was no account of what breaks in practice. The paper declares itself "an experience report" (abstract) on RAG failures in 3 case studies (research, education, biomedical) and says it is, "to the best of our knowledge", the first empirical insight into the challenges of building robust RAGs (§1). It delivers: (1) a catalog of failure points (FP), (2) an account of 3 cases, (3) a research agenda (§1, "Contributions").

Two research questions (§1): **RQ1** "What are the failure points that occur when engineering a RAG system?" (answered in §5, with the BioASQ experiment) and **RQ2** "What are the key considerations when engineering a RAG system?" (answered in §6, with the lessons from the 3 cases).

**(b) My interpretation:** it is a software engineering paper (CAIN), not an evaluation paper. It catalogs *where* RAG fails, but proposes no metrics and does not measure the prevalence of each failure. This matters for our use (see §6 of this reading note).

## 2. Method (step by step, with the exact definitions)

**Reference architecture (§3, Figure 1).** Two processes:
- **Index** (development time): each document is split into *chunks*, each chunk becomes an *embedding* (a numeric vector that represents the meaning of the text) and is stored in a database. Decisions: chunk size ("If chunks are too small certain questions cannot be answered, if the chunks are too long then the answers include generated noise", §3.1) and embedding model (changing it requires re-indexing everything).
- **Query** (runtime): question → rewritten into a general query (*rewriter*) → embedding → top-k by similarity (e.g., cosine) → *re-ranker* → **Consolidator** (reduces what fits in the prompt because of token and rate limits) → *Reader* (the LLM that filters noise, obeys the format and produces the answer) (§3.2).

**Case studies (§4, Table 1).** Three systems:
| Case | Domain | Doc type | Dataset size | Stages | In use? |
|---|---|---|---|---|---|
| Cognitive Reviewer | Research | PDFs | "(Any size)" | Chunker, Rewriter, Retriever, Reader | yes (*) |
| AI Tutor | Education | videos, HTML, PDF | 38 | Chunker, Rewriter, Retriever, Reader | yes (*) (pilot with 200 students, §4.2) |
| BioASQ | Biomedical | scientific PDFs | 4017 | Chunker, Retriever, Reader | experiment |

Only BioASQ has open data (figshare, footnote 5); the other two were omitted "due to confidentiality concerns" (§4).

**Only quantitative experiment (§4.3).** BioASQ: they downloaded 4017 open-access documents and 1000 questions, indexed everything and generated the answers with GPT-4. Evaluation: OpenAI's "OpenEvals" technique (automatic LLM-based evaluator); then, **manual inspection of 40 cases and of "all issues that the OpenEvals flagged as inaccurate"**. Conclusion of §4.3: the automatic evaluation was "more pessimistic than a human rater" in that domain. Threat to validity declared by the authors: the reviewers were not biomedical experts, so the LLM may know more than they do.

**The seven failure points (§5, textual, my translation in quotes, English original right after):**
- **FP1 Missing Content** — the question cannot be answered from the documents. In the happy case the system says "Sorry, I don't know"; but "for questions that are related to the content but don't have answers the system could be fooled into giving a response".
- **FP2 Missed the Top Ranked Documents** — the answer is in the document but it was not ranked high enough to be returned; in practice the top-K is returned, with K chosen "based on performance".
- **FP3 Not in Context — Consolidation strategy Limitations** — documents with the answer *were retrieved* from the database but "did not make it into the context" for generating the answer; it happens when many documents come back and a consolidation process selects.
- **FP4 Not Extracted** — the answer *is in the context*, but the LLM did not extract it; "typically" because of too much noise or contradictory information in the context.
- **FP5 Wrong Format** — the question asked for a format (table, list) and the LLM ignored the instruction.
- **FP6 Incorrect Specificity** — the answer comes, but it is not specific enough or is too specific; it happens when the designers have a desired outcome (e.g., teachers for students, where specific educational content is expected "not just the answer") and also when the user does not know how to ask and is too general.
- **FP7 Incomplete** — answers that are "not incorrect", but that omit part of the information that was in the context. Example: "What are the key points covered in documents A, B and C?"; the authors suggest asking separately.

**How each FP is (not) measured.** The paper **defines no metric, threshold or detection procedure for any of the 7 FPs**. The closest: in §3 it says RAGs are hard to test because there is no data and it must be obtained by synthetic generation or a pilot with little testing; in §6.3, that application-specific question–answer pairs and quality metrics are needed, that using LLMs is expensive, introduces latency and changes with each version; it cites G-Eval (Liu et al. 2023) as an offline evaluation technique that looks promising but is premised on having labeled Q&A pairs (Table 2, last row; paraphrase).

## 3. Main contributions (list)

1. Catalog of 7 failure points organized by the Index/Query pipeline (§5, Figure 1).
2. Experience report of 3 cases (2 in operation at Deakin University) (§4).
3. Table of 9 lessons (Table 2) linking each lesson to FPs and cases.
4. Research agenda: chunking and embeddings (§6.1), RAG vs. fine-tuning (§6.2), testing and monitoring (§6.3).
5. Two key conclusions (abstract): "validation of a RAG system is only feasible during operation" and "the robustness of a RAG system evolves rather than designed in at the start".

## 4. Key results (with numbers and where they are)

**(a) What the paper states, with the n of each piece of evidence:**
- **Quantitative evidence:** only BioASQ (§4.3): 4017 documents, 1000 Q&A pairs, 40 cases manually inspected plus all those marked as incorrect by the automatic evaluator. **There is no count per FP, no overall hit rate, and no results table.** The only reported result is qualitative: the automatic evaluator was more pessimistic than the human.
- **Internal inconsistency:** §1 (RQ1) says the experiment involved "15,000 documents and 1000 question and answer pairs"; §4.3 and Table 1 say 4017 documents. The project's `docs/research/rag_failure_modes_review.md` repeats "15k docs". I could not resolve which is right from v1 alone.
- **Lessons (Table 2) — what is a "lesson" and not measured evidence:**
  | Lesson | FP | Case | Nature |
  |---|---|---|---|
  | Larger context gives better results ("8K vs 4K", contrary to prior work with GPT-3.5) | FP4 | AI Tutor | observation in 1 system, no number reported |
  | Semantic cache reduces cost and latency | FP1 | AI Tutor | recommendation |
  | Jailbreaks bypass the RAG | FP5–7 | AI Tutor | recommendation supported by external literature |
  | Metadata (file name, chunk no.) improves retrieval | FP2, FP4 | AI Tutor | qualitative observation |
  | Open-source embeddings were as good as closed ones on short text | FP2, FP4–7 | BioASQ, AI Tutor | observation, no number |
  | RAG requires continuous calibration | FP2–7 | AI Tutor, BioASQ | opinion/lesson |
  | Implement a configurable pipeline | FP1, FP2 | all 3 | lesson |
  | Custom-assembled pipelines are suboptimal (end-to-end training helps) | FP2, FP4 | BioASQ, AI Tutor | supported by Siriwardhana et al. 2023 |
  | Performance testing is only possible at runtime | FP2–7 | Cognitive Reviewer, AI Tutor | lesson |

  **My reading:** almost all rows of Table 2 are qualitative reports; none comes with an effect size.

**(b) My interpretation of the weight of the evidence:** the FPs are a **qualitative taxonomy** derived from 3 systems, useful for test coverage (covering each failure mode), but they are neither an estimate of prevalence nor proof of cause.

## 5. Limitations (the author's and the ones I identify)

**From the author:**
- Threat to validity in BioASQ: non-expert reviewers (§4.3).
- Two of the three cases cannot be opened due to confidentiality (§4), hence they are not reproducible.
- §6.3 acknowledges that the question of generating realistic domain questions and answers "remains an open problem" and that testing RAG requires data that usually does not exist.

**Mine (marked as interpretation):**
- Without a count per FP, we cannot say which FPs dominate. The statements about FP1–FP7 as "most common" in our review documents come from other sources (RaftLabs, Suthar), not from this paper.
- Inconsistency 15,000 × 4017 documents (see §4).
- The catalog mixes pipeline stages with symptoms in the answer: FP1 is a property of the corpus relative to the question; FP2 and FP3 are retrieval losses; FP4–FP7 are failures of the reader (LLM). There is no treatment of **hallucination with the right evidence in the context** (an LLM that asserts something the context does not say), only of omission (FP4/FP7). This is a gap relative to Magesh et al. 2025, which Magesh himself notes when saying that his typology collapses some of Barnett's FPs (paraphrase of Magesh) and introduces new ones.
- No comparison with a no-RAG baseline, no statistical analysis, no confidence intervals.
- The lesson "Larger context get better results" comes from one system and contradicts Liu et al. 2023 (*Lost in the middle*); the paper merely notes the contrast, without testing the cause.

## 6. Reflections anchored in OUR project

Reference for the points: P1–P11, D-1, D-2, L-3 in `docs/roadmap/ckpt-0.6.1b-metrics-roadmap.md`; plan in `docs/roadmap/ckpt-0.6-plan.md` (§3) and `docs/plan.md`.

**R1 — The mapping "FP3 = noise in the context" does not match Barnett's definition. (CHALLENGES what is written.)**
Plan 0.6 §3.1 (`retrieval_precision_at_k`) says "FP3 — irrelevant return/noise" and roadmap P6 says that `distinct_papers@k` and precision "answer FP3 ('noise in the context')". In Barnett (§5), **FP3 is "Not in Context — consolidation strategy limitations"**: the right document *was retrieved* but *did not make it into the context* because of consolidation, reranking or token limit. Noise appears in the paper as the **cause of FP4**. **Affects:** `evaluators.py::precision_at_k`, `docs/plan.md` §3, `ckpt-0.6-plan.md` §3.1, roadmap P6.
For our system (top-5 chunks of ~420 characters, no consolidation step), FP3 probably almost never occurs; it would only come back with a reranker that cuts documents (CKPT-4) or with context truncation. **Recommendation:** rename the label of `precision_at_k` (and of `distinct_papers@k`) to "noise in the context — cause of FP4" and treat FP3 as "retrieved but discarded", measurable only when a selection step exists between retrieval and prompt (e.g., `gold_chunk ∈ retrieved` but `gold_chunk ∉ prompt`).

**R2 — `faithfulness_judge` and `citation_accuracy` labeled "FP4" are poorly anchored. (CHALLENGES.)**
Plan 0.6 §3.2 puts both under FP4. Barnett's FP4 is **omission** (the answer was in the context and the LLM did not extract it). Faithfulness and citation accuracy measure the opposite: a **claim without support in the context** (commission/hallucination). **Affects:** `faithfulness_judge`, `citation_accuracy`, and the gap that FP4 really asks for: a metric of **"correct answer given that the gold chunk was retrieved"** (for example, correctness conditioned on `hit@k=1`) and/or completeness at the chunk level. **What we would adopt:** keep both judges (for Magesh and ALCE), but fix the label to "hallucination with context present (outside Barnett's catalog)" and add the conditional measure for FP4. **Neutral** as to the usefulness of the judges; it only challenges the label.

**R3 — The remaining mappings are coherent. (SUPPORTS.)**
FP1→`abstention_quality` + `unanswerable`/`stale` slices (P7, P8); FP2→`recall@k`/`hit@k`/`MRR` and `deep-hit` slice; FP5→`format_validator` (P10); FP6→`specificity_judge` and `persona` slice; FP7→`completeness_judge` and `multi-doc` slice. The paper's FP7 example ("What are the key points covered in documents A, B and C?") is exactly our `multi-doc`. The paper's recommendation to ask separately supports the decomposition experiment (CKPT-5).

**R4 — "Validation only in operation": the paper SUPPORTS monitoring, but it is an argument, not a result.**
This supports CKPT-8 (online evaluators) and the principle of `rag_failure_modes_review.md`. But the basis is anecdotal (AI Tutor in a pilot). **Caution:** do not cite it as an "empirical finding" in a report; it is an engineering lesson.

**R5 — There is no estimate of prevalence per FP: the paper does not help size the golden set slices. (NEUTRAL.)**
The 100 examples (40 answerable, 15 unanswerable, 10 stale, 15 multi-doc, 10 format, 10 persona; see roadmap §3) were sized by our own decision. Nothing in Barnett says how many cases of each FP are needed. With slices of 10–15 items, the confidence intervals are wide (my interpretation), so report a CI per slice (cf. P11 and calibration).

**R6 — Automated evaluator more pessimistic than a human rater (§4.3): SUPPORTS human calibration of the judge (P11, 0.8).**
The only evaluation data point in the paper is that an LLM judge diverged from the human. It is a small n (40 cases), with a non-expert reviewer. **Affects:** judge×human κ calibration with the 100 reviewed labels. It reinforces the decision not to freeze the baseline before measuring κ per judge.

**R7 — Chunking: the paper says small chunks prevent answering and large chunks bring noise (§3.1) and asks for systematic evaluation (§6.1). (SUPPORTS the question, gives no answer.)**
Our corpus has 843 chunks of ~420 characters (2–5 per paper). The paper provides no number or rule. It relates to D-1 (gold per chunk or per paper): since an abstract is cut into 2–5 pieces, the "right chunk" may be different from the chunk that partially answers. Nothing in Barnett decides D-1.

**What we would adopt:** (1) FP1–FP7 as a **coverage checklist** for the golden set and as the vocabulary of the report; (2) fix the FP3/FP4 labels (R1, R2); (3) the idea of measuring retrieval and generation separately (2-process structure).
**What we would NOT adopt and why:** (1) treating the catalog as complete, since it does not cover hallucination with context present nor false citation (see Magesh); (2) citing "15k documents" (see §4); (3) using Table 2 as quantitative evidence, since it is qualitative; (4) inheriting the FP1–FP7 order as priority, since the paper does not order by frequency.

## 7. Useful quotations (short verbatim excerpts)

1. "validation of a RAG system is only feasible during operation" — Abstract.
2. "Documents with the answer were retrieved from the database but did not make it into the context for generating an answer." — §5, FP3.
3. "Here the answer is present in the context, but the large language model failed to extract out the correct answer." — §5, FP4.
