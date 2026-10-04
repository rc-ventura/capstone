# QA: LLM judge vs. EM/F1 — Ho et al. (2504.11972) and Wang et al. 2023 (2305.12421)

Double reading note: one part per paper (A and B) + final comparison (C) with the metric recommendation for our project.

- **Full references and links:**
  - **[A]** Ho, X.; Huang, J.; Boudin, F.; Aizawa, A. *Reassessing Extractive QA Datasets at Scale: LLM-as-a-Judge and In-Depth Analyses* (title of the v1/original abstract, used in the roadmap: *LLM-as-a-Judge: Reassessing the Performance of LLMs in Extractive QA*). arXiv:2504.11972 (**v3, 2026-05-29**; text read = v3). <https://arxiv.org/abs/2504.11972> · data/code: <https://github.com/Alab-NII/llm-judge-extract-qa>. Venue: not stated in the text read.
  - **[B]** Wang, C.; Cheng, S.; Guo, Q.; Yue, Y.; Ding, B.; Xu, Z.; Wang, Y.; Hu, X.; Zhang, Z.; Zhang, Y. *Evaluating Open-QA Evaluation*. NeurIPS 2023, Datasets and Benchmarks Track. arXiv:2305.12421 (v4, 2023-10-23). <https://arxiv.org/abs/2305.12421> · <https://github.com/wangcunxiang/QA-Eval>. (Note: it is not "Li et al."; the first author is Cunxiang Wang.)
- **Reading confidence:** **full** for both (full body + tables; for [B] the body and Tabs. 1–7 were read; **[B]'s appendices C–E** — Tab. 8 precision/recall, Tab. 11 of errors by category, Tab. 12 of prompts, definitions in E.4 — **were not read in detail**; for [A] I did not read appendices A–C beyond what the body references: annotation guideline, F1 thresholds and examples of self-preference). Figures: captions/text only. **Version:** [A]'s numbers are those of v3; since the title changed, the v1 numbers may differ (the v3 abstract repeats 0.22/0.40/0.85).
- **Convention:** **[Paper]** = stated/measured; **[Interpretation]** = my reading.

---

# PART A — Ho et al. (2504.11972): LLM judge vs EM/F1 in extractive QA

## A1. Problem the paper addresses

EM (Exact Match: the answer is *identical* to a reference answer, after normalization) and token F1 (word overlap between prediction and reference) "often fail to reflect true model performance" in extractive QA, underestimating correct answers written in a different form (Fig. 1: gold "EPA", prediction "Environmental Protection Agency (EPA)" → EM=false, F1=0.4, judge=true). Prior studies with LLM-as-a-judge in QA (Kamalloo 2023; Verga 2024; Adlakha 2024) do not cover well: **answer types**, **self-preference** and **robustness to prompt variations** (§1). The paper carries out this study on **four datasets** and **several families** of LLMs.

## A2. Method

- **Datasets (§3, Tab. 1):** Quoref, DROP, HotpotQA, 2WikiMultiHopQA; ~1,000 samples from each (**4,193 in total**: Quoref 1,051; DROP 1,018; HotpotQA 1,124; 2Wiki 1,000), with 7 answer types obtained by rules (place, name, job, date, number, year, string). It **excludes** boolean questions (EM/F1 already work). Criteria: use EM/F1, not solved (<95%), have context, diverse answers.
- **QA models (§4.1):** 8 instruction-tuned models — Mistral-7B v0.1, Mixtral 8x7B, Qwen 2 (7B, 72B), Gemma 2 (9B, 27B), Llama 3.1 (8B, 70B); zero-shot CoT prompt (Appendix B.1).
- **Judges:** Mistral-7B-Instruct-v0.3, **Llama 3.3 70B**, **Qwen 2.5 72B**; **6-shot** (4 hand-picked examples + 2 from Verga et al. 2024); input: question, gold, prediction and context; output: **CORRECT** or **INCORRECT** (binary); greedy decoding (§4.2).
- **Human label (§5.1):** 200 instances (50/dataset), each with the 8 answers from the 8 models; 2 authors as annotators, **1 annotator per sample**, with discussion of ambiguous cases; those with an incorrect gold were excluded → **161 samples × 8 = 1,288 answers** labeled Correct/Incorrect.
- **Agreement metric:** **Pearson correlation** between the human label (0/1) and the judge's output (0/1) (measures how closely the judge follows the human; 1 = perfect). For EM/F1 the same correlation is computed with F1 **binarized** at threshold **0.5** (0.3/0.4/0.6/0.7/0.8 tested).
- **Self-preference (§6.2):** 4 QA models (Llama 3.1 8B/70B, Qwen 2 7B/72B) × 7 judges. **Own definition:** there is bias when **all the other judges** say INCORRECT and **only the judge that is the QA model itself** says CORRECT (threshold 100%; also 83% and 67%); reported as % among the answers with EM false. **Sibling-preference:** same, with a judge from the same family (Llama 3.3 70B for Llama 3.1; Qwen 2.5 for Qwen 2).
- **Prompt robustness (§6.3, Tab. 7):** 6-shot (initial), zero-shot, 2-shot, no context, word swap.
- **Practical use:** the judge is applied **only to the answers with EM false** (those with EM=1 receive score 1); the comparison with "all samples" has minimal gaps (§5.2, Appendix C.2).

## A3. Main contributions

1. Large-scale study (4 datasets, 8 QA models, 3 judges) showing that the LLM judge correlates far more with humans than EM/F1 does.
2. First analysis **by answer type**.
3. Test of **self-preference and sibling-preference**: no evidence in the extractive regime.
4. Robustness to prompt variations; zero-shot without context is often the best.
5. Open data and code.

## A4. Key results

- **The numbers "0.22 / 0.40 / 0.85" (Abstract; Tab. 2; §5.1):** are **mean Pearson correlations** with the human label (1,288 answers), **over 8 QA models**: **EM = 0.220** (mean of 7: Llama 3.1 70B gives NaN because there is no EM=1, and the column is constant), **F1 = 0.404** (binarized at 0.5), and judge: **Mistral-7B v0.3 = 0.653; Llama 3.3 70B = 0.750; Qwen 2.5 72B = 0.847 (≈0.85)**. By QA model, Qwen 2.5's correlation reaches **0.945** (Llama 3.1 8B) and **0.908** (Gemma 2 27B), but only **0.723** for Mixtral (Tab. 2). "up to 0.85" is, therefore, the **mean of the best judge**, not a maximum.
- **The F1 threshold changes the result:** mean correlations of binarized F1: 0.3 → 0.451; 0.4 → 0.432; 0.5 → 0.404 (base); 0.6 → 0.379; 0.7 → 0.242; 0.8 → 0.237 (§5.1). Even the most lenient (0.451) is below the worst judge (0.653).
- **EM/F1 vs judge gap on answers with EM=0 (Tab. 3):** the judge gives a much higher score; **largest gap = 81.9 points** (EM vs Llama-70B-judge on HotpotQA, for Mixtral 8x7B); smallest = 15.9 (2Wiki, Gemma 2 9B). Example (Tab. 3): HotpotQA, Mixtral 8x7B: **EM 7.1 / F1 26.1 / judge 84.6–89.0** (Mistral 85.9; Llama 89.0; Qwen 84.6).
- **By answer type (Tab. 4, Qwen judge):** date 1.000 (n=16), number 0.899 (336), name 0.862 (464), string 0.862 (160), place 0.771 (240), **job 0.352 (72)**. The judge is "less strict than humans" on answers with several professions (Tab. 5, example 1: gold "actor" vs prediction with several professions → human F, model T).
- **Self-preference (Tab. 6):** threshold 100%: Llama 3.1 8B **5.77%**; Llama 3.1 70B 0.26%; Qwen 2 7B 0.63%; Qwen 2 72B 0.14% → the authors observe "no self-preference bias"; with thresholds 83%/67% Llama 3.1 8B rises to 12.04%/14.77%. **Sibling-preference:** less pronounced than self-preference (Appendix C.3).
- **Prompts (Tab. 7, mean correlation):** Qwen 2.5: 6-shot 0.847; zero-shot 0.877; 2-shot 0.850; **no context 0.909**; word swap 0.854; variance 0.0007 (Mistral 0.0037; Llama 0.0014).

## A5. Limitations

**From the authors (Limitations section):** only **open-source** LLMs; ~1,000 samples/dataset (not the full sets); **small human evaluation**: 161 samples (1,288 answers), **one annotator per answer**.

**Which I identify [Interpretation]:**
- **Extractive/short regime, with verifiable gold**: the authors themselves restrict the conclusions to "extractive or short-form QA" (§1). **Nothing** about **generated/long** answers, although they claim that the problem "becomes even more pronounced with generative AI models" (§1), without measuring it.
- **Annotators = authors**, with no inter-annotator κ (each sample has 1 label).
- **Pearson correlation between two binary vectors** (equivalent to the phi coefficient), with F1 **dichotomized** at 0.5: the EM/F1 result (0.22/0.40) depends on this arbitrary binarization; a *continuous* F1 against the continuous human score was not analyzed.
- **Weak definition of self-preference**: it requires unanimity of the other judges (conservative), does not compare with an **expected baseline** and uses **only 4 QA models** (all from 2 families). Llama 3.1 8B reaches 5.77% (12–15% with looser thresholds) and the text calls it "relatively small". There is no bias is a generous reading; **"not detected by this criterion"** would be more correct.
- The judges in the main analysis are **newer/larger** than the QA models (Llama 3.3 70B and Qwen 2.5 72B vs Llama 3.1 and Qwen 2); the "same model" test only covers 4 models (Llama 3.1 8B/70B, Qwen 2 7B/72B) judging their own answers.
- Zero-shot without context being the best **suggests that the task is easy** (comparing two short answers); it does not transfer to judgments with a lot of information (faithfulness).

## A6. Reflections anchored in OUR project

**R-A1 — Token-F1 as a quality metric (P9; `_token_f1` and `f1_summary_evaluator` in `evaluators.py`; ckpt-0.6 §3.3).** Position: **(i) SUPPORTS** the distrust of token-F1, **(with a scope caveat)**. The 0.40 (F1) vs 0.85 (judge) are **real**, but they come from short **extractive** QA with 8 small/medium models and **1,288** judgments; **their task is the opposite of ours**: 3-sentence answers generated by `gpt-4o-mini`, with citations, and a *synthetic-by-construction* reference answer. The roadmap's caveat (§D) is correct; I add: **(a)** "up to 0.85" is the mean of the best judge (0.847), not a guaranteed floor; **(b)** their F1 was binarized at 0.5 (and ranges from 0.24 to 0.45 depending on the threshold); **(c)** the paper shows **underestimation** (false negatives) by EM/F1, **not** that F1 rewards incorrect answers through overlap — **this second claim in roadmap §B/P9 is not demonstrated by this paper** (nor by [B], which observes rare FPs from the lexical metric). **We would adopt** the evidence to justify *downgrading* token-F1 to a "lexical diagnostic" and the local test (Spearman/κ on the labels). **We would not adopt** "0.85" as our target.

**R-A2 — Reference-based quality judge (P9; `correctness_judge`).** Position: **(i) SUPPORTS.** The paper's design is practically what we plan: the judge receives **question + gold + prediction**, **binary** output. Operational lessons: (a) **zero-shot ≥ 6-shot** and **no context ≥ with context** (Tab. 7) — for a correctness judge against the reference answer, the retrieved context is dispensable (and avoids mixing with faithfulness); (b) the variance across prompts is low (0.0007–0.0037) for strong judges, **high** for the 7B (0.0037): confirms that the judge **should not be too small**; (c) **"job"/multi-value types** (corr. 0.352) are the weak point: in our corpus, `multi-doc` and `completeness` questions (several items) are the analogue — **require a rubric "all points of the gold present"**, not just "matches the gold".

**R-A3 — Judge = generator (P11; `JUDGE_MODEL`).** Position: **(ii) CHALLENGES** the urgency partially, **but without refutation power** (see A5): "no self-preference" in extractive QA with short reference answers **does not contradict** Wataoka (style preference in open-ended pairwise). **We would adopt** their *measurement design* as a cheap one: compare, **for the same set of answers**, the verdict of the generator's judge vs. judges from another family, and count disagreements of the type "only the model's own judge finds it correct" (simple version of Tab. 6). **We would not adopt** the conclusion of no bias.

**R-A4 — Hybrid cheap→expensive use (cost).** Position: **(iii) NEUTRAL.** The paper applies the judge **only to EM false**. For us, the cost of 100 examples is irrelevant, so we **would not adopt** the saving; **but** the idea of a *deterministic filter first* is useful for **abstention** (P7: abstention rule + judge), already planned.

---

# PART B — Wang et al. 2023 (2305.12421): "Evaluating Open-QA Evaluation" (EVOUNA)

## B1. Problem the paper addresses

Open-QA is used to estimate the **factuality** of LLMs, but it is evaluated by **Exact Match / containment**, which "does not adequately account for the variation in expression of the answers" (e.g. "Lionel Messi" vs "Messi") and is "inapplicable" to detailed LLM answers (§1). The authors create the **QA-Eval** task (evaluating QA *evaluators*) and the **EVOUNA** dataset to measure which evaluators agree with the human.

## B2. Method

- **Data (§3.1, §4, Tab. 1):** Natural Questions (dev; **3,610 → 3,020** after filtering temporal questions/bad gold) and TriviaQA (**2,000 → 1,938**). **5 Open-QA systems**: DPR+FiD (retriever-reader with T5-large), GPT-3.5 (text-davinci-003, T=0), ChatGPT-3.5 (gpt-3.5-turbo, T=0), ChatGPT-4 and BingChat (via web, Apr/2023). Each answer is labeled by the human as correct/incorrect (data: 5 × (3,020+1,938) answers).
- **Human annotation (§4):** done **by the authors** with guidelines (Appendix C.2); **Cohen's κ** between annotators on 500 samples of each subset: **86.4–100** (Tab. 2; NQ-BingChat 86.4 the lowest; TQ-FiD 100).
- **Evaluators compared (§3.3):**
  - **Lexical Matching:** for DPR+FiD, **EM** (the output is identical to a gold); for the LLMs' long answers, **containment** — correct if **some gold appears inside** the answer. **[Important interpretation]**: it is **not token-F1**; there is no token F1 in this paper. The "F1" in the tables is the evaluator's **classification Macro-F1** against the human label (mean of the F1 of the "correct" and "incorrect" classes).
  - **BERTScore** (similarity through contextual embeddings; ref = question+gold; hyp = question+answer; **threshold τ = 0.5**). BART-Score and GPT-Score were discarded because they give a continuous score without separating correct/incorrect.
  - **GPT-3.5 (text-davinci-003)** as judge, T=0, Yes/No prompt with question + list of golds + answer.
  - **"Another Human"** as a reference (ceiling).
- **Evaluator metrics:** **accuracy** and **Macro-F1** vs. the human label (binary task); plus the **ranking** of the 5 systems (Tab. 6).
- **Prompt engineering (§6.4, Tab. 7):** ignore background information, give reasons, **CoT**, **In-Context Learning** (4 examples chosen by k-means, across domains).
- **Normalization:** removes symbols/sources from BingChat's answers and closing questions ("Do you want to know more about xx?") so as not to induce the judge to answer instead of judging (§4).

## B3. Main contributions

1. **QA-Eval** task and **EVOUNA** dataset (human labels on 5 systems × NQ/TQ).
2. Evidence that **EM/containment underestimates** and **swaps the ranking** of the systems.
3. Comparison of **3 families** of evaluator (lexical, neural, LLM) with an **error taxonomy** (categorized by hand).
4. Analysis of the effect of prompt type on the judge.

## B4. Key results

- **EM/containment underestimates (Tab. 4):** NQ — DPR+FiD **human 68.9 vs lexical 59.2**; GPT-3.5 65.5 vs 50.7; ChatGPT-3.5 73.0 vs 57.9; ChatGPT-4 78.8 vs 61.8; BingChat 79.9 vs 65.4. TriviaQA: 81.5 vs 73.5; 78.4 vs 71.0; 84.45 vs 76.7; 90.2 vs 82.1; 89.6 vs 81.6. **Gap** human − lexical (my arithmetic, Tab. 4): from **7.4 to 17.0 points** (largest: ChatGPT-4 on NQ, 17.0).
- **Evaluator hit vs human (Tab. 5, acc/Macro-F1; NQ):**

| NQ subset | Lexical | BERTScore | GPT-3.5 (judge) | Another human |
|---|---|---|---|---|
| FiD | 89.7/92.0 | 75.1/83.5 | **93.6/95.3** | 96.3/97.4 |
| GPT-3.5 | **84.8/86.9** | 69.5/77.6 | 84.1/87.2 | 96.8/97.8 |
| ChatGPT-3.5 | 80.3/84.9 | 72.8/81.2 | **82.2/86.9** | 95.6/96.5 |
| ChatGPT-4 | **82.5/87.6** | 76.0/84.3 | 80.9/86.9 | 96.6/97.9 |
| BingChat | **82.3/87.8** | 67.5/77.6 | 69.5/77.2 | 95.5/97.2 |

  → **GPT-3.5 judge's accuracy minus the lexical one (NQ, my arithmetic):** FiD +3.9; GPT-3.5 −0.7; ChatGPT-3.5 +1.9; ChatGPT-4 −1.6; **BingChat −12.8**. That is, the judge gains little on short/medium answers and **loses** on long answers loaded with extra information (ChatGPT-4, BingChat). On TriviaQA (accuracy): judge better on FiD (95.7 vs 91.8), ChatGPT-3.5 (92.5 vs 92.3) and ChatGPT-4 (92.4 vs 91.1); lexical better on GPT-3.5 (92.3 vs 91.2) and **BingChat (89.8 vs 80.9)**.
- **BERTScore is the worst** in most cases (§6.3: "poorest performance among the three"): sensitive to the **threshold** ("Many datasets are highly sensitive to threshold settings") and to **extended answers** with extra information; it distributes errors equally between FP and FN (§6.1, Fig. 1).
- **Error profile (§6.1, §6.3; Fig. 1–2):** lexical and GPT-3.5 have **few false positives** — lexical "It often marks answers that humans consider correct as incorrect, but rarely does the opposite"; typical lexical errors: paraphrase, synonym, **structural variation** ("8 September 2010" × "September 8, 2010"). The GPT-3.5 judge suffers with **extra information and BingChat's formatting** ("Excluding BingChat data significantly improves GPT-3.5’s performance"), errs on **paraphrase** (the judge's most common error, §6.3), **overgeneralization**, and sometimes **uses its own knowledge and ignores the gold** (it accepted "Bidhan Chandra Roy" for gold "Prafulla Chandra Ghosh").
- **Ranking (Tab. 6):** **no evaluator reproduces the human ranking** of the 5 systems. NQ: human → GPT-4 1st, BingChat 2nd, GPT-3.5 3rd, FiD 4th, ChatGPT-3.5 5th; lexical → BingChat 1st, GPT-4 2nd, FiD 3rd, ChatGPT-3.5 4th, GPT-3.5 5th; the GPT-3.5 judge places BingChat **last** (§5.2).
- **Prompts (Tab. 7, NQ, acc/F1):** BingChat: original 69.5/77.2; **CoT 80.4/87.1**; ICL 75.3/82.5; ignore context 65.7/73.4; **give reasons 55.6/62.2**. ChatGPT-4: original 80.9/86.9; CoT 86.0/91.2; reasons 64.3/71.2. But **CoT worsens on NQ-FiD** (93.6 → 90.2): the effect depends on the distribution (§6.4).

## B5. Limitations

**From the authors (§7):** data via API/web change (not reproducible); **no GPT-4 as evaluator** (API limit); only part of NQ and TriviaQA labeled; **gold with errors** (kept).

**Which I identify [Interpretation]:**
- **Weak and old judge** (text-davinci-003): the results do **not** represent modern judges; [A], with much larger judges, shows a much more favorable picture for the LLM.
- **Human label from the authors themselves**, high κ agreement, but not an independent ceiling.
- **"Long" answers here are still factoids** (an entity/date surrounded by text); they are not *synthetic 3-sentence answers* with several claims like those of our RAG. Footnote 3: "We concentrate on objective question answering with relatively short answers in this work".
- **"Lexical matching" ≠ token-F1:** the conclusion that lexical rarely gives FP holds for **gold containment**, not for token F1, which mixes word precision and coverage and **gives partial credit** to wrong answers with a lot of overlap.
- **Internal inconsistency:** the human scores in **Tab. 6** do not match those in **Tab. 4** (e.g. NQ GPT-3.5: 70.3 vs 65.5; ChatGPT-3.5: 63.0 vs 73.0; DPR+FiD: 69.7 vs 68.9). The claims about *ranking* (§5.2) depend on Tab. 6; **treat with caution** (I found no explanation in the text read).
- The judge's original prompt asks only "Yes/No" (no reasoning); Tab. 7 tests 4 variants in a single family.

## B6. Reflections anchored in OUR project

**R-B1 — "LLM judge is better than token-F1" is conditional (P9).** Position: **(ii) CHALLENGES** the roadmap's simplified reading §B/P9 ("0.22/0.40 vs 0.85"). On **long/informative** answers, the 2023 judge did **not** beat lexical containment (Tab. 5). **What this requires:** the judge needs a **good, validated prompt and model** (the choice of judge matters as much as "using a judge"), and **local calibration is not optional** — it reinforces the roadmap's decision to measure Spearman/κ on the labels before retiring token-F1.

**R-B2 — Lexical false negatives vs false positives (P9; `f1_summary_evaluator`).** Position: **(i) SUPPORTS** the roadmap's diagnosis (paraphrase punished), **(ii) CHALLENGES** the premise that lexical rewards incorrect answers: here lexical almost never gives FP. **[Interpretation]:** for our case, where the gold is one sentence and the answer has 3 sentences, token-F1's dominant problem is **sensitivity to length/citations** (roadmap example: 0.36 for a correct answer) — **false negative**, consistent with both papers. The risk of *false positives* appears with **abstentions vs abstention gold** and with long answers that contain the gold's words; both locally testable.

**R-B3 — BERTScore/embeddings as a cheap alternative (P9 §3(b): `compare_semantic_similarity` / embedding similarity).** Position: **(ii) CHALLENGES.** Wang et al.: BERTScore was the **worst evaluator**, sensitive to threshold and to extended answer with extra information — exactly the answer format of our RAG (3 sentences + citations). The LangSmith course, by contrast, uses a semantic-similarity **judge** (score 1–10) and not pure embedding. **We would adopt** embeddings **only as a secondary signal** (diagnostic), **never** as a substitute for the correctness judge; **we would not adopt** a fixed threshold (0.5) without calibration.

**R-B4 — Correctness judge prompt (P9/`correctness_judge`; 0.6.3).** Position: **(i) SUPPORTS** partially. Wang: **CoT and ICL improve on long answers** (BingChat 69.5 → 80.4 with CoT); **give reasons worsens** (55.6) — but Zheng et al. recommend explanation before the verdict. **[Interpretation]:** the effect is unstable and depends on the model; **test 2–3 variants** (direct verdict; CoT before the verdict; reasons after the verdict) **on the 100 labels** and choose by κ. **We would not adopt** give reasons by default.

**R-B5 — System ranking (plan §2 gates of ≥10–15% improvement; EXP-0).** Position: **(iii) NEUTRAL with a warning.** No evaluator reproduced the human ranking in Tab. 6 (which has an inconsistency with Tab. 4; see B5). **Consequence [Interpretation]:** when deciding gates between configurations, what matters is the evaluator's **ranking correlation** with the human, not just the per-example hit; also report **ranking/delta agreement** in the 0.8 calibration.

---

# PART C — Final comparison and recommendation

## C1. Comparative table

| Aspect | **[A] Ho et al. (2504.11972, v3)** | **[B] Wang et al. (2305.12421, NeurIPS'23)** |
|---|---|---|
| Task | **Extractive** QA with context (Quoref, DROP, HotpotQA, 2Wiki) | **Open-QA without context** (NQ, TriviaQA) — system answers, some **long** |
| Gold | span/short, 1 gold | list of golds; LLM answers **longer and more verbose** than the gold |
| No. of samples evaluated by humans | 1,288 answers (161 items × 8 models) | NQ 3,020 + TQ 1,938, ×5 systems (each answer labeled); κ measured on 500/subset |
| Annotator | 2 authors, **1 per item**; no κ | authors; **κ 86.4–100** (Tab. 2) |
| "Lexical baseline" | **EM** and **token F1** (binarized at 0.5) | **EM** (FiD) and **gold containment** (LLMs); there is **no** token-F1 |
| Judges | Mistral 7B v0.3, Llama 3.3 70B, Qwen 2.5 72B (**open, 2024–25**) | GPT-3.5 text-davinci-003 (**2022–23**), BERTScore |
| Agreement metric | **Pearson** judge×human (mean per QA model) | **Accuracy and Macro-F1** vs human; **ranking** of the systems |
| Main result | EM 0.22; F1 0.40; judge **0.65–0.85** | lexical underestimates (up to −17 pts); judge ≈ lexical on long answers; BERTScore worst |
| Long/generative answers | **Not covered**; only says it is worse | Partially covered (LLMs' long answers), but factoids |
| Self-preference | Tested, **there is none** (fragile definition) | **Not** tested |
| Alternatives to token-F1 | LLM judge; cites **BEM** (Bulian et al. 2022) only as related work (§2) | **BERTScore** (worst), lexical containment, LLM judge; BART/GPT-Score discarded |
| Embeddings/NLI | **Not tested** | BERTScore (contextual embedding) tested; **NLI not tested** |

**They agree on:** (1) EM/lexical **underestimates** paraphrased correct answers; (2) lexical almost never produces false positives; (3) an LLM judge captures paraphrase/synonym; (4) answer types or format (job/multi-value in [A]; answer with extra information in [B]) are the judge's weak points. **They diverge on:** **how much better** the judge is — because of the **judge used** (modern and large in [A]; old in [B]) and the **metric** (Pearson over 0/1 vs accuracy/Macro-F1 vs ranking). **Neither** evaluates **long generative answers with several claims and citations** nor **abstention**.

## C2. Under what conditions token-F1/EM fails and the LLM judge helps (answers to the team-lead's questions)

- **EM/F1 failure:** paraphrase/synonym/form ("EPA" × "Environmental Protection Agency"; "8 September 2010" × "September 8, 2010"), **answer longer than the gold** (extra information), correct answer with **additional context** ("93.5" × "93.5%"; [A] Tab. 2, NaN note) — **false negatives** (Ho §1, Fig. 1; Wang §6.3).
- **Where the judge helps most:** short, verifiable gold (number, date, name) — [A] Tab. 4; **strong** judge.
- **Where the judge is weak:** multi-value/ambiguous answers (job 0.352, [A]); long answers with extra information and formatting ([B], BingChat 69.5%); when it uses its own knowledge and ignores the gold ([B] §6.3).
- **The numbers 0.22/0.40/0.85:** dataset = **4 extractive datasets** (Quoref, DROP, HotpotQA, 2Wiki); **8 QA models**; judges **Mistral-7B/Llama-3.3-70B/Qwen-2.5-72B**; **1,288 judgments** (161 items × 8 answers), **Pearson**, F1 binarized at 0.5. They cover **extractive**, **not long generative**.

## C3. Best "answer quality vs reference answer" metric for our case (anchored reflection; P9, D-1 not affected)

**[Interpretation, based on the evidence above; not a recommendation of the paper.]**

1. **Main metric: reference-based `correctness_judge`** — the LLM judge receives **question + reference answer + generated answer** (without retrieved context, as in [A] Tab. 7), with an **explicit rubric**: *(a)* the key facts of the reference answer are present and without contradiction; *(b)* **correct** extra information is allowed; *(c)* an abstention answer is handled separately (P7/P8). Output on a short scale (pass/partial/fail or 0–1) — for 3-sentence answers with several points, **partial** avoids [A]'s multi-value failure. **Judge model from a different family** than the generator (Wataoka; P11) and **strong** (the evidence from [A] vs [B] shows that a weak judge loses to lexical).
2. **Calibrate before trusting:** κ and agreement per judge against the human, with a trivial baseline and a human ceiling (Zheng); compare **with the normalized token-F1** (Spearman/AUC against the human label) as P9 already foresees — this comparison is what **neither paper does for generative answers**.
3. **Token-F1 → diagnostic** (lexical coverage), with preprocessing (remove citations `[2609.xxxxx]`; P9).
4. **Embeddings/BERTScore:** diagnostic only (Wang: worst, sensitive to threshold and to extended answers). **NLI** (e.g.: claim-by-claim *entailment* check): **not covered** by these papers; remains as an alternative to investigate in the RAGAS/RAGChecker reading notes, not here.
5. **Optional deterministic filter** (gold-entity containment) as a cheap pre-screen, as in [A] (judge only on EM false) — useful for `unanswerable` and for the report, **not** for the main score.

## C4. What we would adopt / would not adopt (summary)

- **We would adopt:** reference-based judge, strong and from another family; local test on the 100 labels; answer types/slices in the report (like [A]'s Tab. 4: `answerable` vs `multi-doc`); test of 2–3 prompt formats; report the ranking correlation of the arms.
- **We would not adopt:** "0.85" as a target; BERTScore with a fixed threshold; [A]'s conclusion "no self-preference"; giving reasons after the verdict by default; generalizing [B] (2022 judge) to modern judges.

---

## 7. Useful quotations (max. 3 in total)

1. "As shown in Table 2, EM and F1 scores correlate less with human judgments than any LLM-as-a-judge model." — [A] Ho et al., §5.1.
2. "It often marks answers that humans consider correct as incorrect, but rarely does the opposite." — [B] Wang et al., §6.3 (Lexical Matching).
3. "the LLM-evaluators tends to perform much worse on long answers with much additional information." — [B] Wang et al., §1 (Introduction).
