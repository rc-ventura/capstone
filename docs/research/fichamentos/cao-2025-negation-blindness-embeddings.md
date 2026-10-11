# Cao 2025 — Negation blindness in universal text embeddings

- **Full reference and link:** Cao, H. (2025). *Semantic Adapter for Universal Text Embeddings: Diagnosing and Mitigating Negation Blindness to Enhance Universality*. arXiv:2504.00584 (**v2, 16 Jul 2025**; text read = v2; Amadeus France; venue not stated in the text) — <https://arxiv.org/abs/2504.00584>.
- **Reading confidence:** **full** (all 8 pages, `pdftotext`; body, Tables 1–4, Figures 1–3 read through captions and the text that cites them; reference list only skimmed). I did not run any code. The paper does **not** test OpenAI embeddings, so nothing here is a measurement of `text-embedding-3-small`.
- **Why it was added (2026-10-10):** found in a web search on whether embedding cosine can tell an answer from its negation. It is the systematic counterpart of our local test (negated answer 0.94, negation pairs 0.92 and 0.87).

> Convention: **(F)** = what the source states or measures; **(I)** = my interpretation or arithmetic.

## 1. Problem the source addresses

(F) Text-embedding cosine is widely used for semantic similarity, "also used as evaluation metrics to measure the answer correctness of RAGs or LLMs" (§3, citing RAGAS and Adlakha et al.), yet there is "few study trying to understand what kind of semantic information is encoded". The paper treats **negation** as a stress test: do state-of-the-art "universal" embeddings tell a sentence from its negated version?

## 2. Method

- (F) **Data (§3.1):** STSB (human similarity 0–5, rescaled to 0–1), training split divided into five groups by human similarity (0–0.2, …, 0.8–1). For each S1 a **negated version** NegS1 is produced with a rule-based sentence-level negation tool (simple verbal negations, e.g. "The horse is white" → "The horse is not white").
- (F) **Models (§3.2):** eight open-source universal embeddings (gte-base/large-en-v1.5, bge-base/large-en-v1.5, e5-base/large-v2, multilingual-e5-large-instruct, mxbai-embed-large-v1) and two contextual baselines (`all-mpnet-base-v2`, `all-roberta-large-v1`). Three LLM-based embeddings in §5.4.
- (F) **Diagnosis (§3.3):** within each group, compare Sim_12 = cos(S1, S2) with Sim_1neg1 = cos(S1, NegS1).
- (F) **Tasks (§5):** (a) identify the paraphrase of S1 among [S2, NegS1] using STSB group 5; (b) **SemAntoNeg**, three candidates (negation, antonym, double negation), where random guessing is 33.33%.
- (F) **Mitigation (§4):** a "negation adapter": per-dimension re-weighting of the embedding learned from [text, paraphrase, negation] triplets, applied only for negation-sensitive tasks.

## 3. Main contributions

- (F) A systematic diagnosis of "negation blindness" in universal embeddings.
- (F) A lightweight re-weighting adapter (200–500 triplets, minutes on one CPU) that improves negation awareness without fine-tuning.

## 4. Key results

- **Negated pairs look similar (Fig. 2, §3.3):** for all groups and all universal embeddings, Sim_1neg1 lies in **[0.6, 1]**. For human-dissimilar pairs (group 1), S1 is more similar to its **negation** than to S2 in **99.27%** of cases on average; in group 4 (roughly similar) in **51.57%**; in group 5 (near paraphrases) S1 is closer to the paraphrase than to the negation in **77.25%**, so the negation wins in about one case in four.
- **The cosine scale is arbitrary (§3.3):** for pairs with human similarity 0–0.2 most models give 0.2–0.8 and the E5 series 0.6–1. "a cosine value of 0.5 cannot be interpreted as 'moderately similar' across different models, as each model's absolute scale is arbitrary."
- **Paraphrase-vs-negation accuracy (Tab. 1, 2):** STSB (simple negation): universal embeddings 77.25% on average, contextual 72.63% (`all-mpnet-base-v2` **71.45%**). **SemAntoNeg (complex negation):** universal 58.64% on average, contextual 33.32%; `all-mpnet-base-v2` **31.32%**, below random (33.33%); best open model multilingual-e5 68.87%.
- **Adapter (Tab. 1–3):** +4.68 points on STSB (universal), and with only 200 training samples +14.52 points on SemAntoNeg (universal average 59.22 → 73.74); a −1 point cost in correlation with human similarity.
- **LLM-based embeddings (Tab. 4):** gte-Qwen2-7B-instruct reaches 81.78% on SemAntoNeg without the adapter; SFR-Embedding-2_R 58.52%, gte-Qwen2-1.5B-instruct 55.03%.

## 5. Limitations

- **From the authors:** only the open-source models listed were tested; fine-tuning for negation is rejected as contrary to "universality"; English only; the gains of the adapter vary by model (e.g. +0.05 for gte-base on SemAntoNeg with weights learned on STSB).
- **Mine (I):**
  1. **Short isolated sentences with rule-based negation.** Our answers are three sentences with citations; whether a negation inside a longer text weighs less on the vector (dilution) is plausible but not tested here.
  2. **No OpenAI embeddings** (`text-embedding-3-*`), so the numbers are not about our model; our local test (6 answers + 3 control pairs, 2026-10-10) is the only measurement on it.
  3. **The adapter needs access to the vectors and training triplets**; it was tested on open models and measured only on a paraphrase-vs-negation choice, not on RAG answer correctness.
  4. Single-author preprint; no confidence intervals beyond standard deviations over 10 resamplings for SemAntoNeg.

## 6. Reflections anchored in OUR project

Reference: roadmap P9 (ladder with two acceptance criteria), the canary tests foreseen in `ckpt-0.6-plan.md` §5.

| # | Point / file | Reflection | The source… |
|---|---|---|---|
| 1 | **Contradiction detection as a separate acceptance criterion** | Our local result (negated answer 0.94, higher than every correct paraphrase 0.59–0.84; negation pairs 0.92 and 0.87) matches the published pattern: negated text stays at 0.6–1. So a cosine cannot be expected to flag an answer that contradicts the reference; the second criterion must be tested explicitly (canaries) rather than assumed. | **(i) SUPPORTS** |
| 2 | **Thresholds must be calibrated locally** | The scale is model-specific ("a cosine value of 0.5 cannot be interpreted … across different models"). A threshold chosen on one embedding model is meaningless for another. | **(i) SUPPORTS** pre-registering thresholds per model |
| 3 | **Frequency of the failure** | The paper shows the blind spot exists; it does not say how often negation or contradiction occurs in RAG answers. That is what the "contradicts the reference" mark in our labeling measures. | **(iii) NEUTRAL**, the gap our labeling fills |
| 4 | **Negation adapter** | A possible mitigation if embeddings otherwise pass and we want to avoid the judge: it needs ~200–500 [text, paraphrase, negation] triplets (our canaries could provide them) and access to the vectors. It adds a trained component to maintain. | **(iii) NEUTRAL**, not adopted now (over-engineering until the calibration says it is needed) |
| 5 | **Pointer** | The paper cites Anschütz et al. 2023, *This is not correct! Negation-aware evaluation of language generation systems* (arXiv:2307.13989), about negation-aware evaluation of generation. **Not read.** | pointer only |

**What we would adopt:** negation/number-swap **canaries** as an explicit test of any embedding or lexical signal; pinning the embedding model; calibrating thresholds locally. **What we would NOT adopt:** the adapter, for now; the accuracy figures as predictions for `text-embedding-3-small`.
