# Loukina, Zechner & Chen 2014 — Automatic evaluation of spoken summaries: the case of language assessment

- **Full reference and link:** Loukina, A., Zechner, K., Chen, L. (2014). *Automatic evaluation of spoken summaries: the case of language assessment*. Proceedings of the 9th Workshop on Innovative Use of NLP for Building Educational Applications, Baltimore, pp. 68–78 — <https://aclanthology.org/W14-1809/>.
- **Reading confidence:** **full** (all 11 pages, `pdftotext`; body, Tables 1–4, discussion). The text extraction of Tables 2–4 was linear, so row/column assignment was checked against the sentences that cite them.
- **Why it was added (2026-10-11):** for the P9 normalization decision. Unlike the news-summarization studies, it measures ROUGE against human scores **per individual response** (Spearman), on short **abstractive** texts (about 72 words), and it compares stopword removal and a token-versus-type variant. That is the closest evaluation unit to our C1 (answer-by-answer agreement) among the sources found.

> Convention: **(F)** = what the source states or measures; **(I)** = my interpretation or arithmetic.

## 1. Problem the source addresses

(F) Can ROUGE, built for automatic summaries, score the content coverage of **spoken summaries** written by non-native speakers, one response at a time, without manual annotation per item? Which ROUGE variant and which pre-processing agree best with expert raters, and how many reference summaries are needed?

## 2. Method

- (F) **Data:** 5,934 spoken responses (1,611 speakers, 24 prompts) from a pilot of the TOEFL Junior test; average 72 words (σ = 29); scored 1–4 holistically by two raters (inter-rater κ = 0.69, Tab. 4 caption). Texts come from an ASR system (word error rate 26.5–29.4%) and from manual transcription.
- (F) **Metrics:** ROUGE-1…4, ROUGE-SU1–4 and a geometric mean, computed against up to 10 references drawn from responses scored 4; baseline cosine on tf-idf vectors (CVA). Pre-processing grid: all words / **non-stop** / **types** (each word counted once) / non-stop types, raw or lemmatized: 72 variants per response.
- (F) **Agreement:** Spearman ρ between the metric and the first rater's score, per response; non-parametric bootstrap for the differences; selection on a training partition, confirmation on an unseen test partition.

## 3. Main contributions

- (F) ROUGE-1 against four references reaches ρ ≈ 0.65 with experts (0.67 on manual transcripts), above the cosine baseline (0.51 / 0.45) and above a naive ROUGE-1 with one reference and token counts (0.55 / 0.59) (Tab. 3).
- (F) Using **types instead of tokens** improves agreement; the choice of reference responses does not matter once enough (never more than 4) are used; lemmatization makes no difference.

## 4. Key results

- **Stopwords (Tab. 2, averaged over ROUGE-1 and ROUGE-SU1–4; ASR / manual transcripts):** all words 0.573 / 0.606; **non-stop 0.585 / 0.600**; non-stop types 0.601 / 0.617; **types 0.622 / 0.634**.
  - (I) Removing stopwords moved ρ by **+0.012 (ASR) and −0.006 (manual)**: no consistent effect, and not tested as a separate contrast.
  - (F) For metrics on consecutive n-grams with n > 2 "the removal of stop-words led to further drops in performance" (p = 4.4 × 10⁻⁵). In the discussion: "the removal of stop-words sometimes led to a decrease in performance."
  - (I) **Types versus tokens** moved ρ by **+0.049 (ASR) and +0.028 (manual)** relative to all-words tokens, and the paper reports it as significant for unigram metrics (p = 0.00017).
- (F) Metric type: ROUGE-1 and ROUGE-SU were best; n-grams longer than 2 did worse (the authors: "our data represents abstractive summaries, poor performances of longer n-grams is not surprising").
- (F) The metric agreed better on "summarize what you heard" prompts (ρ̄ = 0.653) than on "describe the pictures" prompts (0.437), where responses vary more.
- (F) Discussion: "Recall-based approaches are sensitive to the length of candidate responses"; a response can get inflated scores by repeating an n-gram that occurs several times in the reference. Counting types was introduced to address this.

## 5. Limitations

- **From the authors:** bag-of-words, "doesn't distinguish between syntactic roles"; negations and subject-object relations are lost; the human score is holistic (pronunciation, grammar and content), so ROUGE correlates partly because fluent speakers also cover more content (r = 0.62 with pronunciation accuracy).
- **Mine (I):**
  1. **Spoken, non-native, ASR-noisy responses** scored for proficiency, not factual QA over abstracts with a written reference.
  2. **References are other student responses** scored 4, not an LLM-written gold answer.
  3. **The stopword contrast is averaged and indirect:** it is "all" versus "non-stop" inside a grid of 72 variants, with no interval for that contrast alone.
  4. **Holistic target:** agreement with a score that mixes several skills is not agreement with "is the answer correct".
  5. **No contradiction test:** the authors' own limitation about negation applies to everything measured.

## 6. Reflections anchored in OUR project

Reference: P9 normalization (decision D-14, ADR-010), ADR-008 §2 and §5 (C1), roadmap §7.

| # | Point / file | Reflection | The source… |
|---|---|---|---|
| 1 | **Stopword removal** | At answer level on abstractive ~72-word texts, removing stopwords changed ρ by +0.012 and −0.006: no usable effect. Together with `owczarzak-2012-rouge-accuracy-tac.md` it argues against paying for a stopword list. | **(i) SUPPORTS** SQuAD-style normalization |
| 2 | **Tokens versus types in the overlap** | Counting each word once raised agreement with experts by 0.03–0.05 ρ. Our `_token_f1` and the planned Recall use a clipped multiset (`Counter` intersection), like ROUGE. Whether a "types" variant is better for our references (about 54 tokens with many repeated function words) is **not shown for our regime**. | **(iii) NEUTRAL**, but a new design variable for a later round |
| 3 | **Recall and length** | A recall-oriented signal can be pushed up by repetition or length; ours reports Recall as a diagnostic next to F1 (which penalizes length), as ADR-008 §2 already does. | **(i) SUPPORTS** reporting both |
| 4 | **Answer-level unit** | A ρ ≈ 0.55–0.65 for a free lexical metric against expert scores on individual abstractive responses is the same order as the C1 pass mark (0.60, borrowed from Adlakha). It is a second, independent reference level; it does not set a cut. | **(iii) NEUTRAL**, context for C1 |
| 5 | **Few references suffice** | Optimal references never exceeded 4. We have one gold answer per example; the paper hints that a second reference could help, but that is a dataset change, not now. | **(iii) NEUTRAL** |

**What we would adopt:** nothing new to implement now; the finding supports "no stopword list" and flags the token-versus-type variant as a candidate to test **inside the calibration, declared beforehand** (not chosen after seeing labels). **What we would NOT adopt:** the ρ values as expected performance (spoken proficiency, ASR, different references); "types" as the default without a test on our labels.

## 7. Useful quotations (verbatim from the text read)

- "…the removal of stop-words sometimes led to a decrease in performance."
- "Recall-based approaches are sensitive to the length of candidate responses."
- "…a recall-based approach may produce inflated scores by assigning higher metric values to a response which contains multiple repetitions of the same n-gram as long as the n-gram occurs several times in the reference response."
