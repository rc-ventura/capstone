# Zheng et al. 2023 — LLM-as-a-Judge, MT-Bench and Chatbot Arena

- **Full reference and link:** Zheng, L.; Chiang, W.-L.; Sheng, Y.; Zhuang, S.; Wu, Z.; Zhuang, Y.; Lin, Z.; Li, Z.; Li, D.; Xing, E. P.; Zhang, H.; Gonzalez, J. E.; Stoica, I. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023, Datasets and Benchmarks Track. arXiv:2306.05685 (v4, 2023-12-24). <https://arxiv.org/abs/2306.05685> · code/data: <https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge>
- **Reading confidence:** **complete.** Full text of the PDF v4 (body + Appendices A–F, including the prompts of Figs. 5–10 and Tables 9–15). Figures (win-rate curves) read only through the captions and the text that discusses them. Location: section/table; page only when clear.
- **Convention of this reading note:** **[Paper]** = what the paper states/measures. **[Interpretation]** = my reading, not the paper's.

## 1. Problem the paper addresses

Classic benchmarks (MMLU, HELM etc.) measure knowledge on closed questions and **do not distinguish** aligned models (RLHF/instruction-tuning) from base models, even though humans clearly prefer the former (§1, Fig. 1, Tab. 8). Evaluating human preference on open-ended questions is expensive and slow; overlap metrics (BLEU/ROUGE) are "also ineffective for these questions" (open-ended answers without a reference answer) (§3). The question: **can a strong LLM replace the human as an evaluator of chatbots, and with what biases?** The paper studies biases (position, verbosity, self-enhancement, limited reasoning), mitigations, and measures judge×human **agreement** on two new benchmarks (MT-bench and Chatbot Arena).

## 2. Method (step by step)

**2.1 New data (§2, §4.1, Appendix C)**
- **MT-bench:** 80 multi-turn questions (8 categories × 10: writing, roleplay, extraction, reasoning, math, coding, STEM, humanities), hand-written. 6 models answer (GPT-4, GPT-3.5, Claude-v1, Vicuna-13B, Alpaca-13B, LLaMA-13B). **58 "expert" annotators** (mostly graduate students, US$ 20 per 20 questions), each judging ≥20 questions → ~3K votes.
- **Chatbot Arena:** anonymous voting on battles; ~30K votes in 1 month; random sample of **3K single-turn votes** (2,114 unique IPs) for the agreement study.

**2.2 Three judge modes (§3.1)**
- **Pairwise comparison** (the judge sees question + two answers and says A, B or tie; prompt Fig. 5).
- **Single-answer grading** (score 1–10 for one answer; "short explanation first, then `[[rating]]`"; prompt Fig. 6).
- **Reference-guided grading** (the judge receives a reference solution; prompt Fig. 8).
- Trade-offs [Paper]: pairwise grows quadratically with the no. of models; single-answer "may be unable to discern subtle differences between specific pairs" and absolute scores "fluctuate more" if the judge changes.

**2.3 How each bias was measured (§3.3 and Appendix D.1)**
- **Position bias** (preferring the 1st or the 2nd answer, regardless of content). Construction: for each 1st-turn question of MT-bench, two *almost identical* answers (GPT-3.5 called 2× with T=0.7); each judge evaluates both orders. **Consistency** = % of cases in which the judge gives the same verdict when the order is swapped (measures stability; 100% = no position bias). Also "Biased toward first/second" and "Error" (invalid format). The "rename" prompt renames "Assistant A/B" to separate position bias from name bias.
- **Verbosity bias** (preferring a longer answer even when it is not better). "**Repetitive list**" attack: 23 MT-bench answers that contain a numbered list; GPT-4 rewrites the list without adding information and the result is *prepended* to the original list (5 items → 10). The attack "works" if the judge deems the new version better. **Failure rate** = % of successful attacks. Calibration: judges give a tie for two identical answers.
- **Self-enhancement** (preferring the model's own answers). Measured only **statistically**: win rate (without ties) of 6 models under each judge and under humans (Fig. 3b).
- **Limited capability in math/reasoning:** 10 math questions, LLaMA-13B vs Vicuna-13B, positions swapped (20 cases). **Failure** = GPT-4 says an *incorrect* answer is correct (Tab. 4).

**2.4 Mitigations (§3.4)**
- *Swapping positions:* call the judge 2× swapping the order; only declare a win if the same answer wins in both orders, otherwise tie ("conservative" approach, used in the experiments). "Aggressive" alternative: random position, which works at scale.
- *Few-shot judge:* 3 examples (A better / B better / tie) generated with GPT-3.5, Vicuna and judged by GPT-4.
- *CoT judge:* the judge solves the question on its own before judging (prompt Fig. 7).
- *Reference-guided judge:* the judge generates its answer independently and it is inserted as a reference in the prompt (Fig. 8).
- *Judge fine-tuning:* Vicuna-13B fine-tuned with 20K Arena votes (Appendix F).
- *Multi-turn:* showing the full conversation in a single prompt (rather than one prompt per turn) avoids reference errors (§3.5, Fig. 16).

**2.5 Judge×human agreement protocol (§4.1, Appendix D.3)**
- **Agreement** = probability that two *distinct* individuals of each type (e.g., GPT-4 and a randomly drawn human) agree on the vote for a randomly drawn question (measures how much the judge coincides with a typical human).
- **S1:** includes non-tie votes, ties and "inconsistent" (due to position bias, counted as a tie); **agreement of two random judges R = 33%**.
- **S2:** only non-tie votes; **R = 50%**.
- Also: **human-majority** (majority vote of the humans; GPT-4's agreement ceiling is the human-majority×human agreement; a vote tie counts ½) and average per-model *win rate*.
- [Paper, App. D.3] human×human agreement may be an underestimate because three humans "A, A, B" agree with each other in only 1/3 of the pairs.

## 3. Main contributions

1. Systematic study of LLM-as-a-judge, with a taxonomy of 3 modes (pairwise / single / reference-guided) and four named limitations.
2. **MT-bench** (80 multi-turn questions, 3K expert votes) and **Arena data** (30K conversations with human preference) made public.
3. Evidence that **GPT-4 reaches >80% agreement** with humans, at the level of human×human agreement (S2).
4. Simple, tested mitigations (position swap, few-shot, CoT, reference-guided) and a fine-tuned open-source judge (Vicuna-13B).
5. Argument for **hybrid** evaluation: capability benchmarks (MMLU etc.) + preference benchmarks with an LLM judge (§5, Tab. 8: "no single benchmark can determine model quality").

## 4. Key results

**Position bias (Tab. 2, §3.3; 80 cases)**

| Judge | Prompt | Consistency | Favors 1st | Favors 2nd | Error |
|---|---|---|---|---|---|
| Claude-v1 | default | 23.8% | 75.0% | 0.0% | 1.2% |
| Claude-v1 | rename | 56.2% | 11.2% | 28.7% | 3.8% |
| GPT-3.5 | default | 46.2% | 50.0% | 1.2% | 2.5% |
| GPT-3.5 | rename | 51.2% | 38.8% | 6.2% | 3.8% |
| GPT-4 | default | 65.0% | 30.0% | 5.0% | 0.0% |
| GPT-4 | rename | 66.2% | 28.7% | 5.0% | 0.0% |

- "Only GPT-4 outputs consistent results in more than 60% of cases" (§3.3). Claude-v1 also shows **name bias** (favors "Assistant A", Tab. 2 rename).
- Other prompts (Appendix D.1, Tab. 9): "score" (two absolute scores) raises GPT-3.5's consistency (55.0%) but **reduces** that of Claude-v1 (20.0%) and GPT-4 (51.2%); "short" (without anti-bias instructions): GPT-4 62.5%.
- By category (Tab. 10, GPT-4 default): **lower** consistency in writing (42.0%), STEM (44.0%), humanities (36.0%); **higher** in math and coding (86.0%). By model pair (Tab. 11): it almost disappears when the models differ a lot (GPT-3.5 vs LLaMA-13B: 98.8%) and is higher when they are close (GPT-3.5 vs Claude-v1: 67.5%).
- Few-shot (Tab. 12): consistency Claude-v1 23.8→63.7%; GPT-3.5 46.2→55.0% (the bias migrates from the 1st to the 2nd position: 28.7% favor the 2nd); **GPT-4 65.0→77.5%**. But "high consistency may not imply high accuracy", long prompts make the API ~**4× more expensive** (§3.4) and the agreement with humans of few-shot GPT-4 was **similar** to zero-shot (Appendix D.2).

**Verbosity bias (Tab. 3):** failure rate under the *repetitive list* attack on 23 answers — Claude-v1 **91.3%**, GPT-3.5 **91.3%**, GPT-4 **8.7%**.

**Self-enhancement (§3.3, Fig. 3b):** GPT-4 favors itself with a win rate **10% higher** than that of humans; Claude-v1, **25% higher**; **GPT-3.5 does not favor itself**. The authors themselves: "due to limited data and small differences, our study cannot determine whether the models exhibit a self-enhancement bias" and admit that a controlled study is difficult.

**Math/reasoning (Tab. 4):** failure rate with 10 questions × 2 positions = 20 judgments: **default 14/20 (70%) → CoT 6/20 (30%) → reference-guided 3/20 (15%)**. The text (§3.4) cites "from 70% to 15%". Even with CoT, the judge repeats the error of the presented answers (Fig. 15): "may still be misled by the context".

**Judge×human agreement (Tab. 5, MT-bench, 1st turn; Tab. 13 in Appendix D.3)**

| Pair | S1 (R=33%) | S2 (R=50%) |
|---|---|---|
| GPT-4 pairwise × human | 66% | **85%** |
| GPT-4 single × human | 60% | **85%** |
| human × human | 63% | **81%** |
| GPT-4 pairwise × GPT-4 single | 70% | 97% |

2nd turn (Tab. 5b): GPT-4 pair×human S2 85%; human×human S2 **82%**. Arena (Tab. 6): GPT-4×human S1 **64%**, S2 **87%**; human×human is not measured in the Arena. Against the humans' majority vote (Tab. 13): GPT-4 pair S2 85% (1st and 2nd turn). Agreement **rises from ~70% to ~100%** as the win-rate difference between the compared models increases (Fig. 2, §4.2). In disagreements, humans found GPT-4's judgment reasonable in **75%** and changed their vote in **34%** (§4.2).

**Note:** Tab. 6 also shows that GPT-4 produces many more non-tie votes than GPT-3.5/Claude ("more affirmative and less suffered from position bias"), with similar non-tie agreement.

**Vicuna-13B as judge (Appendix F, Tab. 15):** zero-shot, consistency 11.2–16.2% and format error 22.5–78.8%; fine-tuned on 20K Arena votes (3 classes A/B/tie): consistency **65.0%**, error 0%, agreement **56.8%** (S1) and **85.5%** (S2), vs GPT-4 66% and 87%.

**Complementary benchmarks (Tab. 8):** MT-bench (GPT-4 single-answer score, scale 10) separates Vicuna-7B-selected (5.95; MMLU 37.3) from LLaMA-13B (2.61; MMLU 47.0), showing that MMLU and preference measure different things.

## 5. Limitations

**From the author (§6, §3.3, §3.4):**
- Focuses on *helpfulness*, barely addresses safety/honesty; several dimensions (accuracy, relevance, creativity) are **merged into a single score**.
- Self-enhancement **inconclusive** (limited data, no controlled experiment).
- Few-shot may introduce new biases and costs ~4×.
- The origin of position bias is only conjectured (training data or causal architecture).
- Proposed solutions are "preliminary".

**Ones I identify [Interpretation]:**
- **Artificial position test:** *almost identical* pairs from a single family (GPT-3.5 at T=0.7) — the text itself says the test is "challenging" and that the bias is smaller when there is a large difference (Tab. 11). The consistency of 23.8%–65.0% **is not** an estimate of the bias on real pairs.
- **Small samples:** 80 cases in Tab. 2; 23 in Tab. 3; **10 questions** (20 judgments) in Tab. 4. No confidence intervals.
- **Agreement metric without formal prevalence correction**, only the reference "R = 33% / 50%". There is no Cohen's κ. S2 agreement excludes ties and inconsistencies, which **inflates** the number (Tab. 5 shows G4-Pair×human S1 = 66%, well below the 85%).
- The intro cites "§4.2, Table 4" for agreement, but the table is **5** (editorial oversight).
- The reference for math was **generated by the judge itself** (reference-guided), so it still inherits the judge's errors; with a human reference answer the effect may be larger.
- Human annotators are graduate students (sample bias) and, in the Arena, uncontrolled anonymous voters.
- 2023 judges (GPT-4, GPT-3.5, Claude-v1); the numbers **do not** transfer to `gpt-4o-mini`.

## 6. Reflections anchored in OUR project

**R1 — Judge = generator (P11; `config.py`, 6 judges of `ckpt-0.6-plan.md` §4).** Paper's position: **(iii) cautious NEUTRAL**, with a touch of (i). The numbers (+10% GPT-4, +25% Claude-v1) *suggest* self-enhancement, but the paper itself says it cannot conclude, and GPT-3.5 does not favor itself. Therefore **Zheng alone neither supports nor overturns** the judge/generator separation; what supports it is Wataoka (see the dedicated reading note). **What we would adopt:** a separate `JUDGE_MODEL`, recorded in the metadata (already in P11). **What we would NOT do:** cite "GPT-4 favors itself by 10%" as proof of bias (it is a claim of the paper with the author's caveat).

**R2 — Calibration against a human at stage 0.8 (Cohen's κ, `ckpt-0.6-plan.md` §6; glossary "κ").** Position: **(i) SUPPORTS** calibrating. Zheng does not prescribe κ; it uses **S1/S2 agreement against a random baseline** and **human×human as ceiling**. **We would adopt:** (a) report raw agreement *and* κ per judge (our labels are imbalanced: ~25 `should_abstain` vs ~90 answerable, P8, where raw agreement misleads); (b) include a **trivial baseline** (a judge that always answers "pass") next to the random one; (c) measure the **human×human ceiling** (if there are 2 annotators on some sample) — without it we cannot say whether κ=0.6 is "good". **Warning [Interpretation]:** the roadmap speaks of calibrating against "the 100 human labels"; if these labels come from the *golden set review* (question/reference answer quality) and are not **judgments of generated answers**, Zheng's protocol requires labeling a sample of **system outputs**. This needs to be confirmed before 0.8.

**R3 — Output format of the judges `{"score": ..., "reason": ...}` (`ckpt-0.6-plan.md` §4; all 6 judges).** Position: **(ii) PARTLY CHALLENGES** the design. Zheng's prompts (Figs. 5–7) instruct to **explain before giving the score** (e.g., single-answer prompt: "Begin your evaluation by providing a short explanation. … After providing your explanation, please rate the response"). In a JSON with `score` before `reason`, the model decides the verdict *before* reasoning, nullifying the CoT effect. **[Interpretation]:** put `reason` **before** `score` in the schema. **Caveat:** Wang et al. 2023 (reading note `qa-eval-judge-vs-f1.md`, Tab. 7) found that giving reasons *worsened* GPT-3.5 on QA-eval, and CoT improved only on long answers — the effect depends on the judge/task; **test it in the sensitivity round**, do not assume.

**R4 — Reference-guided and the quality judge vs. reference answer (P9; planned `correctness_judge`; `completeness_judge` on multi-doc).** Position: **(i) SUPPORTS**. Tab. 4: giving the reference drops the failure from 70% to 15%. Our case is *more favorable* than the paper's (the reference answer is human-reviewed, 100/100, and not generated by the judge) — but the `answerable`/`persona` are *synthetic-by-construction* (ADR-001/003), so the reference answer reflects the wording of the LLM that read the chunk; a judge with a reference answer may punish correct but different answers. **We would adopt:** passing the gold answer to the `completeness`/`correctness` judge; **not** passing it to `faithfulness` (ref-free by design, against `retrieved_context`).

**R5 — Verbosity (`completeness_judge`, `specificity_judge`, P9).** Position: **(ii) CHALLENGES** the plan to use a small model. In the *repetitive list* attack, Claude-v1 and GPT-3.5 failed 91.3%; only GPT-4 resisted (8.7%). `gpt-4o-mini` is the kind of lower-capability model for which the literature shows fragility (but it **was not tested** here). Our prompt limits the answer to 3 sentences, which helps, however completeness/specificity are exactly the criteria that "reward" more detail. **We would adopt:** a cheap sanity test inspired by §3.3 — take ~10 answers, add a redundant paraphrase and verify that the score **does not rise**. **We would not adopt** the exact attack (numbered lists are not our format).

**R6 — Position bias and pairwise judges (course `module_2/pairwise_experiments.ipynb`, gates ≥15%/≥10% of plan §2).** Position: **(iii) NEUTRAL** for the 6 planned judges (all **single-answer**: position does not apply). **If** we compare baseline×variant with a *pairwise* judge (as in the course, with gpt-4o), the paper says to **swap the order and require consistency** (§3.4) — the course does not do this (to check). Without it, the difference between arms may be a position artifact (favors the 1st in 30–75% of cases depending on the judge, Tab. 2). Single-answer is enough for our design; Zheng says GPT-4 single-answer "matches both pairwise GPT-4 and human preferences very well" (Tab. 5) but absolute scores are "likely to fluctuate more" if the judge changes — **relevant for the sensitivity round with 2 judges** (P11): absolute scores from two judges **are not comparable in value**, only in ranking/agreement.

**R7 — Separate dimensions (6 judges).** Position: **(i) SUPPORTS**. The author's limitation no. 1 is the single score that mixes dimensions; our catalog (faithfulness, citation, completeness, abstention, specificity, relevance) already separates them.

**R8 — Judge misled by context (faithfulness/citation).** Position: **(iii) NEUTRAL / warning.** §3.3: GPT-4 solves the question on its own but errs when *judging* when the presented answers contain convincing errors. **[Interpretation]:** in `faithfulness_judge`, a hallucinated but fluent answer may be accepted; therefore validation must include negative cases (already planned in 0.6.3: "pass and fail").

**Summary — what we would adopt:** `JUDGE_MODEL` ≠ `GENERATION_MODEL`; gold in the correctness/completeness prompt; `reason` before `score` (to validate); agreement + κ + trivial baseline; human×human ceiling; order swap if we ever use pairwise. **What we would NOT adopt:** few-shot as default (4× the cost, no agreement gain, Appendix D.2); judge fine-tuning (Appendix F; wrong scale for 100 examples); the "10-point" score as an absolute score comparable across different judges; the premise of "GPT-4 ≈ human" for `gpt-4o-mini`.

## 7. Useful quotations

1. "GPT-4 favors itself with a 10% higher win rate; Claude-v1 favors itself with a 25% higher win rate. However, they also favor other models and GPT-3.5 does not favor itself." — §3.3 (Self-enhancement bias), p. 5.
2. "Due to limited data and small differences, our study cannot determine whether the models exhibit a self-enhancement bias." — §3.3, p. 5.
3. "A conservative approach is to call a judge twice by swapping the order of two answers and only declare a win when an answer is preferred in both orders." — §3.4 (Swapping positions), p. 6.
