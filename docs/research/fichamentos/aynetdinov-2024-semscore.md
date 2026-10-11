# Aynetdinov & Akbik 2024 — SemScore: evaluating instruction-tuned LLMs with embedding similarity

- **Full reference and link:** Aynetdinov, A., Akbik, A. (2024). *SemScore: Automated Evaluation of Instruction-Tuned LLMs based on Semantic Textual Similarity*. arXiv:2401.17072 (**v2, 5 Feb 2024**; text read = v2; short paper, venue not stated in the text) — <https://arxiv.org/abs/2401.17072>.
- **Reading confidence:** **full** (all 9 pages, `pdftotext`; body, Tables 1–6, appendices A.1–A.4). I did not run any code.
- **Why it was added (2026-10-10):** found in a web search for work that used **embedding similarity** to evaluate LLM answers (the "middle rung" of the P9 ladder: free metrics → embedding cosine → LLM judge). It is the only paper I found that evaluates a plain cosine between response and reference embeddings against human judgment.

> Convention: **(F)** = what the source states or measures; **(I)** = my interpretation or arithmetic.

## 1. Problem the source addresses

(F) Human evaluation of instruction-tuned LLMs does not scale, and n-gram metrics (BLEU, ROUGE-L) correlate poorly with humans: a response rated best by a human can get a very low BLEU/ROUGE-L because it shares few words with the single gold response (Tab. 1). It is also unclear which of the many existing metrics correlates best with humans.

## 2. Method

- (F) **Data:** the 252 instructions of Wang et al. 2023c (varied tasks: blog posts, text completion, coding, formal logic, …) with human-expert target responses; 12 models (GPT-3 variants, text-davinci-001/002/003, GPT-3.5-turbo, GPT-4, LLaMA, Alpaca).
- (F) **Human ranking:** four-category ratings A–D (Wang et al. 2023c), extended by the authors to four more models; "the majority of our evaluation was carried out by one human expert, while the second human expert evaluated a sample" (Appendix A.4: κ = 0.63; computer-science PhD students). The ratings are averaged per model to rank the 12 models.
- (F) **Metrics compared (§3):** BLEU, ROUGE-L, BERTScore (deberta-xlarge-mnli), BLEURT, BARTScore (two variants), DiscoScore, G-Eval (an LLM-judge metric, with GPT-4, gpt-3.5-turbo or gpt-3.5-turbo-instruct as backbone), and the proposed **SemScore**.
- (F) **SemScore:** embed the model response and the target response separately with `all-mpnet-base-v2` (a sentence transformer, MPNet-Base, 768 dimensions) and take the **cosine similarity** of the two vectors.
- (F) **Agreement measure (§3.3):** Kendall τ and Pearson r between the **averaged** metric score and the **averaged** human score, over the 12 models. This is agreement on the ranking of *systems*, not on individual answers.

## 3. Main contributions

- (F) A human-judged ranking of 12 LLMs and a comparison of 9 metrics against it.
- (F) SemScore, "a straightforward but remarkably effective evaluation metric", with no prompt and no API cost, claimed to correlate best with human evaluation.

## 4. Key results

- **Table 3 (Kendall τ / Pearson r with the human ranking):** SemScore **0.879 / 0.970**; G-Eval-4* 0.855 / 0.863; G-Eval-3.5* 0.855 / 0.831; BERTScore 0.848 / 0.944; G-Eval-3.5-instruct 0.840 / 0.911; ROUGE-L 0.788 / 0.933; BARTScore 0.788 / 0.621; BARTScore-para 0.697 / 0.884; BLEU 0.667 / 0.865; BLEURT 0.485 / 0.485; DiscoScore 0.364 / 0.583. (* the backbone model's own responses are excluded, so these two are not computed on the same 12 systems as the rest.)
- **Choice of embedding model matters (Tab. 4, ablation with deberta-xlarge-mnli embeddings):** SemScore with mean pooling τ 0.870 / r 0.929, with the CLS token τ 0.756 / r 0.892, vs BERTScore on the same model 0.848 / 0.944. The authors recommend `all-mpnet-base-v2` and ask future work to name the transformer it uses.
- **Per-task breakdown (Tab. 5):** SemScore's correlation varies a lot by task type, from 0.333 (Overleaf) and 0.548 (Grammarly) to 1.0 on several others.
- **Examples (Tab. 6):** a human-rated "A" code snippet gets SemScore 0.49 (the authors suspect the natural-language comment in the target dominates the embedding); a response with an extra, wrong genre ("Thriller") gets 0.913 and is rated B: "SemScore does not penalize the extra genre enough".

## 5. Limitations

- **From the authors:** dependence on the underlying embedding model; the need for "at least one gold-standard target output … human created or at least human-vetted"; the small size of the evaluation set and its lack of focus on traditional NLP tasks.
- **Mine (I):**
  1. **System-level, not answer-level.** With 12 systems, a τ gap of 0.879 vs 0.855 (SemScore vs G-Eval-4) is not a meaningful difference (no interval or test is given). The result says "embedding cosine ranks systems about as well as the other metrics", not "it judges single answers correctly".
  2. **One main annotator**, κ = 0.63 on a sample.
  3. **Different task:** instruction-following with heterogeneous tasks and 2023 models, not factual QA over retrieved abstracts with citations.
  4. **No negation or contradiction test**, and the extra-content example shows extra wrong content is barely penalized.
  5. **Not our embedding model.** All numbers are for `all-mpnet-base-v2`; the project uses `text-embedding-3-small`.

## 6. Reflections anchored in OUR project

Reference: roadmap P9 (ladder: lexical → embeddings → judge), gates in plan §2 and CKPT-6.

| # | Point / file | Reflection | The source… |
|---|---|---|---|
| 1 | **Gates between configurations** | What the gates (≥10–15% between configs) need is a metric that *ranks configurations* like a human would. At that level an embedding cosine matched an LLM-judge metric and beat ROUGE-L and BLEU in a small study. It does not show answer-level correctness. | **(i) SUPPORTS** embeddings as a candidate, **(iii) NEUTRAL** on answer-level use |
| 2 | **Cost and reproducibility** | No prompt, no judge-model drift, a single pinned embedding model; the authors stress reproducibility vs a proprietary evaluator that "may be updated over time". | **(i) SUPPORTS** the "cheap rung before the judge" |
| 3 | **Pin the embedding model** | The ablation (τ 0.756–0.879 depending on the encoder and pooling) shows the metric depends on the encoder. Thresholds calibrated on `text-embedding-3-small` are invalid if the model changes. | **(i) SUPPORTS** pinning the model and recalibrating on change |
| 4 | **References must be human-vetted** | Our references are LLM-generated from a chunk; the 100/100 review covered questions and gold, not generated answers (roadmap C-2). | **(i) SUPPORTS** the audit of references |
| 5 | **Extra content is not penalized** | Like Recall, a cosine tolerates a verbose answer; but it also barely penalizes wrong additions. Wrong additions are the job of `faithfulness_judge`, not of the correctness signal. | **(iii) NEUTRAL**, a reminder of scope |

**What we would adopt:** cosine(answer, reference) as the middle rung, with the embedding model pinned; checking, in the local calibration, agreement at the level of configurations too. **What we would NOT adopt:** the τ/r values as expected performance (different model, different task); any claim that the cosine detects wrong answers (see `cao-2025-negation-blindness-embeddings.md`).
