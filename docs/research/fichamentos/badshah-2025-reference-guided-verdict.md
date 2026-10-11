# Badshah & Sajjad 2025 — Reference-guided verdict: LLMs-as-judges for free-form QA

- **Full reference and link:** Badshah, S., Sajjad, H. *Reference-Guided Verdict: LLMs-as-Judges in Automatic Evaluation of Free-Form QA*. arXiv:2408.09235 (**v3, 10 Nov 2025**; text read = v3) — <https://arxiv.org/abs/2408.09235>. Venue: not stated in the text read.
- **Reading confidence:** **full** (all 17 pages, `pdftotext`; body, Tables 1–7, Appendices A–D; figures through captions and the text that cites them). Secondary note: it is a small study (100 samples per dataset), kept because it reports Cohen's κ for judge-vs-human agreement with a reference answer, which is the design our `correctness_judge` would use.
- **Why it was added (2026-10-10):** found in a web search for evidence on P9b/P11 (reference-guided correctness judge, κ with humans, self-preference when a reference is present).

> Convention: **(F)** = what the source states or measures; **(I)** = my interpretation or arithmetic.

## 1. Problem the source addresses

(F) EM/F1 penalize correct free-form answers that are phrased differently, and human evaluation is slow and expensive. The paper proposes a **reference-guided verdict**: an LLM judge receives the question, the candidate's answer and a reference answer and returns True/False; verdicts of several judges are combined by **majority vote**.

## 2. Method

- (F) **Judges and candidates (App. B.1):** Mistral-7B-Instruct-v0.3, GPT-3.5-turbo, Llama-3.1-70B-Instruct, each used both as candidate and as judge; temperature 0.
- (F) **Data (App. B.2):** TruthfulQA (validation, generation subset), TriviaQA (unfiltered.nocontext, validation), HotpotQA (distractor, validation); **100 random samples per dataset**, 3 candidates per dataset (900 answers).
- (F) **Prompt (App. B.3, D):** zero-shot, role "impartial judge", binary verdict plus a brief explanation; no few-shot or chain-of-thought.
- (F) **Human labels (App. B.4):** three NLP graduate students, binary 1/0 "alignment with the reference answer and contextual relevance", majority vote as ground truth.
- (F) **Agreement (App. B.5):** percent agreement, Fleiss' κ (many raters), **Cohen's κ** between each judge (or the judge majority) and the human majority.

## 3. Main contributions

- (F) A zero-shot, reference-guided, binary judge protocol evaluated against human labels with κ.
- (F) Evidence that aggregating three judges by majority vote improves agreement on most rows.
- (F) A prompt-sensitivity ablation and a stability check (five repeated calls at T=0).

## 4. Key results

- **Judge vs human, Cohen's κ (Tab. 2):** single judges **0.63–0.93** (lowest: GPT-3.5-Judge on TruthfulQA/GPT-3.5, 0.63; highest: Llama-3.1-70B-Judge on TriviaQA/GPT-3.5, 0.93). Majority of the three judges vs human majority: **0.72–0.96**.
- **Task difficulty (App. C.2–C.3):** agreement is lowest on TruthfulQA, where the reference answers are adversarial ("I have no comment", "It is unknown who said…"); humans' percent agreement is 82–99%, judges' 72–96%; Fleiss' κ among humans 0.74–0.97 vs 0.61–0.80 among judges (Tab. 5–6).
- **Does the ensemble help? (I, my arithmetic from Tab. 2):** comparing the ensemble κ with the **best single judge** in each of the 9 rows, it is higher in 5, equal in 2 and lower in 2 (TruthfulQA/GPT-3.5: 0.72 vs 0.76; TriviaQA/Llama-70B: 0.79 vs 0.86). The paper's statement "alignment improves in most cases" holds against the average judge, not always against the best one.
- **Stability (§4.3):** five repeated calls at T=0 on TruthfulQA give the same verdicts; the explanations change.
- **Prompt sensitivity (§4.3, Tab. 3, 7; TruthfulQA, Mistral-7B candidate):** a prompt without guidelines ("open") changed Mistral-Judge's verdicts in 18% of cases; ensemble-vs-human κ: open 0.66, detailed 0.73, **close (verdict only, no explanation) 0.79**; Fleiss' κ among judges 0.55 / 0.51 / **0.62** (humans 0.74).
- **Self-enhancement (§4.2):** "we did not observe significant instances of self-enhancement bias" when a reference answer is provided; no statistical test is reported.

## 5. Limitations

- **From the authors:** accuracy depends on reference quality — "the assumption that all reference answers are correct may not always hold; noisy or incomplete references could mislead the evaluation process"; binary verdicts oversimplify partial correctness; sensitivity to prompt design; small slices of three datasets; the cost of running several judges; a limited set of models.
- **Mine (I):** (1) n = 100 per dataset and candidate, and no confidence intervals for the κ values, which are then uncertain by several hundredths; (2) human labels come from three students with no external check; (3) old and small judges (Mistral-7B, GPT-3.5); (4) the ablation changes one prompt factor at a time on one dataset and one candidate, so "verdict only is best" is a hypothesis, not a result; (5) "no self-enhancement" is a qualitative observation over three models, not a measurement.

## 6. Reflections anchored in OUR project

| # | Point / file | Reflection | The source… |
|---|---|---|---|
| 1 | **P9b — `correctness_judge` design** | Same design as ours: question + answer + reference, binary verdict, no context. κ with humans of 0.63–0.93 (single) is the order of magnitude to *expect*, not to assume: short-form QA, 2023-era models, 100-sample labels. | **(i) SUPPORTS** the design, **(iii) NEUTRAL** on the expected κ |
| 2 | **Noisy references** | Our references are synthetic (an LLM wrote them from a chunk) and the 100/100 human review covered questions and gold, not generated answers (roadmap C-2). The paper's own limitation says noisy references mislead the judge, and agreement was lowest where references were adversarial. A wrong reference becomes a false failure attributed to the system. | **(i) SUPPORTS** a spot-audit of judge disagreements against the reference |
| 3 | **Majority vote of three judges** | Gain over the best single judge is mixed (5 higher, 2 equal, 2 lower of 9 rows) and the cost is three calls per example. Not worth it for the baseline; reconsider only if a single judge's κ on our labels is below target. | **(ii) CHALLENGES** adopting an ensemble now |
| 4 | **P11 — urgency of judge ≠ generator for a reference-based judge** | With a reference the authors saw no significant self-enhancement; `qa-eval-judge-vs-f1.md` (Ho et al.) found a small self-preference only for Llama-3.1-8B (5.77% at the strict threshold). Both are weak evidence (no tests, few models). Keeping the judge from another family is cheap insurance; it need not block the correctness judge. | **(ii) CHALLENGES** the urgency, **(iii) NEUTRAL** on the default |
| 5 | **P14 — judge prompt** | The verdict-only prompt had the best κ in this ablation, while Zheng et al. recommend explanation before the verdict and Wang et al. found "give reasons" worse. Three sources, three tendencies, all small: test 2–3 variants on our labels. | **(iii) NEUTRAL** |
| 6 | **Cost and repeatability** | One call per example at T=0 is enough: verdicts were stable across five runs. | **(i) SUPPORTS** a single call per example |

**What we would adopt:** the reference-guided binary protocol with a single judge call; an audit of disagreements against the reference; κ (not percent agreement) as the reported agreement.
**What we would NOT adopt:** the three-judge ensemble for the baseline; the κ ranges as promised agreement.
