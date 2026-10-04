# Es et al. 2023 — RAGAS (RAG evaluation without reference answers)

- **Full reference:** Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert. *Ragas: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217 (version read: v2, 2025-04-28). <https://arxiv.org/abs/2309.15217> · Code: <https://github.com/explodinggradients/ragas> · Dataset: <https://huggingface.co/datasets/explodinggradients/WikiEval>
  *(The paper is a short demo paper; the publication venue — EACL 2024, demos — comes from my prior knowledge and was **not verified** in this reading.)*
- **Reading confidence:** **full.** I read the complete raw text of the arXiv HTML version (v2): abstract, Sections 1–6, Table 1 and Appendix A (Tables 2–4 with examples). The paper is short (≈ 34 thousand characters). **I did not see:** figures (there are no relevant figures in the extracted text). The reading tool could not decode the PDF; I used the HTML.

> Convention: **[Paper]** = stated in the text; **[My reading]** = my interpretation.

---

## 1. Problem the paper addresses

**[Paper]** Evaluating RAG requires looking at several dimensions: the retriever's ability to find relevant *and focused* passages, the LLM's ability to use them *faithfully*, and the quality of the generation itself (Abstract). Common evaluations have limits: perplexity does not always predict downstream performance and requires probabilities that closed models do not expose; QA tends to use short-answer/extractive datasets, which are not very representative (Section 1). Goal: a set of **reference-free** metrics — "without having to rely on ground truth human annotations" — to speed up evaluation cycles (Abstract, Section 3: "we focus on metrics that are fully self-contained and reference-free").

## 2. Method (step by step, exact formulas)

Setting (Section 3): given a question *q*, the system retrieves context *c(q)* and generates answer *a_s(q)*. Three quality aspects, all measured **by LLM prompting** (gpt-3.5-turbo-16k; embeddings: text-embedding-ada-002). No reference answer is used.

### 2.1 Faithfulness — Section 3

*(measures whether everything the answer claims can be deduced from the retrieved context; it is the antidote to hallucination)*

1. An LLM decomposes the answer into a set of **statements** *S(a_s(q))* (prompt: "Given a question and answer, create one or more statements from each sentence in the given answer").
2. For each statement *sᵢ*, the LLM checks whether it can be inferred from the context (function *v(sᵢ, c(q))*; verification prompt with a Yes/No verdict and a brief explanation per statement).
3. **F = |V| / |S|**, where |V| = number of supported statements and |S| = total number of statements.

### 2.2 Answer relevance — Section 3, Eq. (1)

*(measures whether the answer addresses exactly the question asked, without being incomplete or redundant; it does **not** assess whether it is true)*

1. Given *a_s(q)*, the LLM generates **n questions** *qᵢ* that are plausible for that answer (prompt: "Generate a question for the given answer").
2. Embeddings (ada-002) of all the questions; **sim(q, qᵢ)** = cosine between the original question and each generated question.
3. **AR = (1/n) · Σᵢ sim(q, qᵢ)**.

The paper explicitly says that this assessment "does not take into account factuality, but penalises cases where the answer is incomplete or where it contains redundant information".

### 2.3 Context relevance — Section 3, Eq. (2)

*(measures how **focused** the retrieved context is: what fraction of it is actually needed to answer; penalizes context with redundant/irrelevant information)*

1. The LLM extracts from *c(q)* the subset of "crucial" sentences **S_ext** for answering *q* (the prompt asks for relevant sentences without altering them; if there are none, it returns "Insufficient Information").
2. **CR = number of extracted sentences / total number of sentences in c(q)**.

**[My reading]** This is **not** retrieval recall or precision in the IR sense: it does not compare against gold documents, does not consider rank position, and does not measure whether *all* the necessary information is present — it only measures the *proportion of the delivered context that the LLM judges useful*. A context that contains a single useful sentence and nothing else would have CR = 1.0 even if the rest of the answer were missing.

### 2.4 Validation: WikiEval (Section 4)

- **Construction:** 50 Wikipedia pages about events since early 2022 (beyond the model's cutoff), prioritizing pages with recent edits. For each page, ChatGPT suggests a question answerable from the introductory section and answers it using that section as context.
- **Annotation:** two people annotated the three dimensions; agreement ~**95%** (faithfulness and context relevance) and ~**90%** (answer relevance); disagreements resolved by discussion.
- **Tasks constructed (paired):**
  - *Faithfulness:* compare the standard answer with the one ChatGPT generated **without** context; the human says which is more faithful to the page.
  - *Answer relevance:* compare the standard answer with a deliberately **incomplete** answer (ChatGPT instructed to answer incompletely).
  - *Context relevance:* compare the original context with a context **expanded** with sentences from Wikipedia backlinks (or completed by ChatGPT, for pages without backlinks).

### 2.5 Baselines (Section 5)

- **GPT Score:** asks ChatGPT for a score from 0 to 10 per dimension (prompt with the definition); ties broken randomly.
- **GPT Ranking:** asks ChatGPT to choose the preferred answer/context, with the definition in the prompt.

Agreement metric: **accuracy** = fraction of instances in which the candidate preferred by the metric coincides with the one preferred by the annotators.

## 3. Main contributions

1. Three **reference-free** metrics defined operationally (F=|V|/|S|; AR via generated questions + cosine; CR via sentence extraction) — Section 3.
2. **WikiEval**, a public dataset with human judgments on the three dimensions — Section 4.
3. Integration with llama-index and LangChain (Section 1), which explains the library's practical adoption.
4. Evidence that, on WikiEval, the metrics agree more with humans than two LLM-judge baselines (Table 1).

## 4. Key results

**Table 1 (Section 5)** — accuracy of agreement with humans in paired comparisons, WikiEval:

| Method | Faithfulness | Answer Relevance | Context Relevance |
|---|---|---|---|
| **Ragas** | **0.95** | **0.78** | **0.70** |
| GPT Score | 0.72 | 0.52 | 0.63 |
| GPT Ranking | 0.54 | 0.40 | 0.52 |

Text of Section 5: for faithfulness the Ragas predictions are "in general highly accurate"; for answer relevance it is lower, "largely due to the fact that the differences between the two candidate answers are often very subtle"; context relevance was "the hardest quality dimension to evaluate" — ChatGPT "often struggles" when selecting the crucial sentences, "especially for longer contexts".

**Appendix A (Tables 2–4):** examples. In particular, Table 3 shows as "low answer relevance" an answer that says the date/time "have not been provided" and then adds generic content.

**External corroboration (another paper, for context):** in RAGChecker (Table 5 of the appendix), RAGAS metrics correlate weakly with human preference for correctness/completeness: Faithfulness Pearson 8.22 / 4.90 / 7.83 and Answer Relevance 11.59 / 9.39 / 10.27 (correctness / completeness / overall). **Caveat:** the human target there (correctness/completeness vs. reference answer) is different from what these metrics are meant to measure.

## 5. Limitations

**[Paper]** Those in Section 5: answer relevance has lower agreement due to subtle differences; context relevance is the hardest (sentence-selection errors in long contexts). There is no formal limitations section.

**[My reading]**
- **Small sample and easy task:** n = 50 questions, all generated and answered by ChatGPT, with **constructed contrasts** (answer with × without context; complete × incomplete; clean context × polluted context). The 0.95 accuracy on faithfulness is on a paired binary choice with a large contrast — it is not equivalent to the accuracy of an *absolute* judge on real answers, which are almost all "partially faithful".
- **Same family generating and judging:** ChatGPT answers, judged by gpt-3.5-turbo; the paper does not discuss self-preference (self-enhancement). Relevant for P11.
- **No validation of real-world use:** no analysis of variance across runs, prompt sensitivity or cost.
- **Answer relevance and abstention:** by definition (generating questions from the answer), an "I don't know" answer generates generic questions and scores low; the "low answer relevance" example in Table 3 contains exactly "have not been provided". The paper does not discuss abstention.
- **Context relevance and missing content:** since CR is a ratio of useful sentences to total, it does not detect *missing* information — the opposite of a recall.
- **Dependence on sentence segmentation:** with chunks of ~420 characters (2–3 sentences), the granularity would be coarse. *(Inference: the paper does not discuss chunk size.)*

## 6. Reflections anchored in OUR project

**R1 — Context relevance does NOT replace the ID-based metrics (P1–P3, P5, P6, D-1; `hit_rate`, `recall_at_k`, `mrr`, `precision_at_k`).** *NEUTRAL/CHALLENGES the idea of using them as an alternative.* CR does not use a reference answer (good: it dispenses with `gold_chunk_ids`), but for that very reason it **does not answer** "was the chunk that answers in the top-k?" nor "at what position?". In our corpus, with a 1-chunk gold and top-5, an ideal context would have 1 useful chunk out of 5: CR tends to have a ceiling close to 1/k — the same problem as P6 (1/k ceiling of `precision_at_k`), *[My reading; it is an inference, not measured]*. Hence CR would inherit P6's defect without delivering the ranking advantage of MRR.

**R2 — Where we would adopt CR (ckpt-0.6-plan §3.1; `multi-doc open-topic` slice, 5 examples with `retrieval_gold=pending-adjudication`).** *SUPPORTS as a complement.* For these 5 examples, the plan **skips** them in the retrieval metrics until the EXP-0 mini-pooling. CR (or better, an entailment-based variant — see the RAGChecker reading note) would give a **provisional signal of context focus** without gold. It also serves as a **noise diagnostic** (FP3) across all slices: "how much of the context delivered to the generator is useless" — relevant for the choice of k in CKPT-1 and for the dedup collapse (same paper 3× in the top-5). Flagged as a diagnostic, **not** as a gate.

**R3 — `faithfulness_judge` (ckpt-0.6-plan §3.2, FP4).** *SUPPORTS.* The plan already describes "judge checks claim by claim of the answer against `retrieved_context`" — this is exactly F = |V|/|S| in two steps (extract statements; verify each one). The difference to decide is the format: RAGAS uses the **proportion** of supported statements; the plan speaks of an output `{"score": bool|float, ...}`. I recommend the float (proportion) because it is more informative than a bool and allows aggregation per slice. For `unanswerable`/`stale`, the faithfulness of the abstention answer is trivially high (nothing is asserted) — do not confuse it with abstention hit (P8).

**R4 — `answer_relevance` (ckpt-0.6-plan §3.2).** *SUPPORTS partially; CHALLENGES applying it to all slices.* The definition via generated questions + cosine is cheap and deterministic (no 1–10 score judge), an alternative to a free-form LLM judge. **But** it penalizes correct abstention (see Limitations). The plan's matrix already restricts `answer_relevance` to `answerable`, which is consistent; I reinforce: **never** apply it to `unanswerable`/`stale`, otherwise a correct abstention appears as an "irrelevant answer" and distorts P8. We would adopt the embedding-based computation as a **cheap baseline** and compare it with the LLM judge in calibration (0.8).

**R5 — Judge validation (P11, P9; calibration 0.8).** *CHALLENGES the strength of the evidence.* RAGAS shows that ref-free metrics *can* agree with humans, but under favorable conditions (n=50, pairs with a strong contrast, 2 annotators). It does not serve as proof that a gpt-4o-mini judge works in **our** domain (cs.AI abstracts, chunks of ~420 characters). It reinforces the decision to calibrate with κ on our set and **not** rely on "RAGAS has already been validated". It also shows the P11 risk in pure form: the same ChatGPT generates and judges.

**R6 — Reference-free × our gold (ckpt-0.6-plan §3.2; `completeness_judge`, `abstention_quality`).** *NEUTRAL.* Our golden set **has** a reference answer (answer, gold IDs, `should_abstain`); for correctness, completeness and abstention we use a reference. RAGAS covers only what dispenses with a reference (faithfulness, relevance, focus) — that is why it is a complement to our catalog, not an alternative.

**R7 — P9 (token-F1).** *NEUTRAL.* RAGAS does not offer a correctness metric against a reference answer (only *answer relevance*, which ignores factuality). It does not help decide token-F1 × correctness judge; see the RAGChecker and QA-judge reading notes.

**What we would adopt:** (1) the decomposition into statements + verification for `faithfulness_judge`, with score = proportion; (2) CR as a noise diagnostic and provisional signal for `open-topic`; (3) AR via embeddings as a cheap baseline for `answer_relevance`, only on `answerable`. **What we would NOT adopt:** (1) CR as a substitute for hit/recall/MRR or as a retrieval gate; (2) AR on the abstention slices; (3) the 0.95 accuracy as justification of judge reliability (R5).

## 7. Useful quotations

1. *"We say that the answer a_s(q) is faithful to the context c(q) if the claims that are made in the answer can be inferred from the context."* — Section 3, "Faithfulness".
2. *"our assessment of answer relevance does not take into account factuality, but penalises cases where the answer is incomplete or where it contains redundant information."* — Section 3, "Answer relevance".
3. *"We found context relevance to be the hardest quality dimension to evaluate. In particular, we observed that ChatGPT often struggles with the task of selecting the sentences from the context that are crucial, especially for longer contexts."* — Section 5 (after Table 1).
