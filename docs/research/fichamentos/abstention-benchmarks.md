# Abstention and selective refusal — AbstentionBench (Kirichenko et al. 2025), RefusalBench (Muhamed et al. 2025) and extras (UAEval4RAG, OR-Bench)

> Single reading note, one section per paper + comparison. Convention: **[Paper]** = what the text states (with section/table/figure);
> **[My reading]** = my interpretation, not the paper's. Text read from the PDF (arXiv v1/v3 indicated below),
> extracted with `pdftotext`; **numbers that exist only inside chart figures were read only when they also appear in text/caption/table**.

## Index
1. AbstentionBench — full reading
2. RefusalBench — full reading
3. Comparison AbstentionBench × RefusalBench × project
4. Extra A: UAEval4RAG (Peng et al. 2025) — partial reading
5. Extra B: OR-Bench (Cui et al. 2025) — partial reading
6. Synthesis for the roadmap (P7, P8, P11)

---

# 1. Kirichenko et al. 2025 — AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions

- **Full reference and link:** Polina Kirichenko\*, Mark Ibrahim\*, Kamalika Chaudhuri, Samuel J. Bell\* (FAIR at Meta). *AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions*. arXiv:2506.09038v1 [cs.AI], 10 Jun 2025. <https://arxiv.org/abs/2506.09038> · code <https://github.com/facebookresearch/AbstentionBench>
- **Reading confidence:** **full** (body §1–5 + Appendices A–F, incl. judge prompt, Tables 1–6). Caveats: (i) charts (Fig. 2–8, S1–S16) read through captions and text, not through bar values; (ii) this is v1 — later revisions may exist; (iii) in Appendix C.3.3 the number of annotators per pair appears as "[NUMBER REDACTED]" in the PDF.

## 1. Problem the paper addresses
**[Paper]** LLMs need to know *when not to answer* (underspecified, ill-posed, no known answer), but "abstention remains understudied, without a systematic evaluation framework for modern LLMs" (Abstract). Existing datasets each cover a single, isolated type of problem (§2). Central question: does reasoning fine-tuning (o1, R1, s1) help or hurt abstention?

## 2. Method (step by step)
**[Paper]**
1. **Definition of abstention (§3, p.3):** "a response that refrains from directly answering the question, such as by expressing a lack of knowledge, communicating uncertainty or caveats, or highlighting unanswerable aspects of the prompt. This can include simple statements such as 'I don't know'… but can also include detailed responses providing partial answers". *(= a broad, semantic definition; it is NOT a prefix/regex.)*
2. **Collection (§3.1, App. C.2.1):** search on Semantic Scholar ("LLM abstention/abstain/uncertainty") → 183 papers → 82 candidate datasets → manual review (abstention desirable in ≥1 sample, public, license ok) → **17 datasets**; + 3 own reasoning variants (GSM8K-Abstain, GPQA-Abstain, MMLU-Math-Abstain: they duplicate the question and **remove the context** to make it unanswerable) + UMWP. The Abstract says "20 datasets"; the appendix adds up to 17+3+UMWP (the count does not match exactly — **[My reading]** minor counting discrepancy, does not affect the conclusions). Each dataset is capped at 3,500 samples (C.2.2). Over **35k questions** (Fig. 1).
3. **Each sample** = prompt (question + optional context) + binary label `should_abstain` + optional reference answers (C.2.2). That is, answerable *and* unanswerable questions coexist in the same base.
4. **Six scenarios (§3.2):** Answer Unknown, False Premise, **Stale**, Subjective, Underspecified Context, Underspecified Intent. "Stale" = "questions regarding recent events that occurred after model pretraining, such that answers contained in the training data may be stale".
   - Implementation of Stale in FreshQA (C.2.2): they compare two snapshots (v10282024, before pretraining, and v12182024, after the models' maximum cutoff); answers that **changed** between them are "should abstain".
5. **Abstention detection (§3.4, C.3):** **LLM-as-judge** = **Llama 3.1 8B Instruct**, prompt adapted from Brahman et al. (CoCoNot), "Yes"/"No" output, **greedy decoding (T=0), "crucial for high performance"** (C.3.1). They reject embedding similarity (used in earlier work) because it does not capture the diversity of abstentions. The prompt (C.3.2) has one description per scenario of "appropriate abstention" × "NOT an abstention" and **also receives the `[GROUND TRUTH ABSTENTION LABEL]` and the reference answers** ("they can be noisy, so mostly rely on the [QUESTION]…"); "accuracy or verbosity of the answer does not matter".
6. **Correctness judge** (only for non-abstaining answers that have a reference answer): also Llama 3.1 8B, prompt from Thakur et al.; "correct"/"incorrect" output; invalid answers are filtered out (C.3.5).
7. **Judge validation (C.3.3–C.3.4):** 300 prompt–response pairs (3 prompts per general-domain benchmark × {Llama 3.1 70B, GPT-4o}), **stratified** sampling by `should_abstain` × first-pass judge prediction; the authors label independently (full abstention / partial / none); non-unanimous ones are discussed → consensus; 50/50 validation/test (validation to iterate on the prompt, test for the final number).
8. **Generation:** max. 4k tokens, **temperature 0.8**, top-p 0.95, fixed seed (C.1); o1 with T=1.0. Reasoning: 4k "thinking" tokens + 4k for the final answer; only the final answer is evaluated by default (§4.3).

**Metrics (§3.4, p.5)** — the judge's label compared against the sample's `should_abstain`:
- **Abstention recall** = proportion of the *unanswerable* samples in which the model actually abstained. *(In plain language: of the questions where it should say "I don't know", in how many it did.)* Cited as "the proportion of responses where the model correctly abstained".
- **Abstention precision** "to account for over-abstention" — of the times it abstained, how many it should have. *(Measures over-refusal.)*
- **Abstention F1** — harmonic mean of both.
- **Response accuracy** — correctness (by the correctness judge) on the answers that did *not* abstain.
- Decision: "as we find that models generally exhibit high abstention precision, we focus on abstention recall."
- No formulas are written; the definitions are verbal.

## 3. Main contributions
1. A single abstention benchmark, 20 datasets, 6 scenarios, >35k questions (Fig. 1).
2. Finding: **abstention does not scale with model size** (Fig. 3b; Llama 8B/70B/405B ≈ equal) and **better accuracy ≠ better abstention** (Fig. 4).
3. Finding: **reasoning fine-tuning worsens abstention by −24% on average** (Abstract/§1; Fig. 6a), including on math/science; a larger "reasoning budget" improves accuracy but does not help/worsens abstention (Fig. 7).
4. **RLVR** (final stage of Tülu 3) degrades abstention (Fig. 5c); SFT/DPO help, except on *underspecified context*.
5. Reasoning chains *contain* uncertainty, but the final answer is definitive (Fig. 8a, Fig. S15).
6. A *system prompt* describing the scenarios raises abstention without degrading precision much (Fig. 8b, S16) — "unlikely to fundamentally address a lack of reasoning about uncertainty" (§4.4).
7. Abstention judge validated against humans (Tab. 1) and a *fast subset* of 100 questions/benchmark (App. E, Tab. 6).

## 4. Key results
- **Judge (Tab. 1, C.3.4):** Llama 3.1 8B — acc 0.88 / F1 0.85 / precision 0.86 / recall 0.83; Llama 3.3 70B — 0.88 / 0.83 / 0.94 / 0.75; GPT-4o — 0.89 / 0.85 / 0.96 / 0.77. They chose the 8B for cost and because it has the best recall (§3.4 cites only "88% accuracy").
- **Average per model (Tab. 4, App. D.1; mean accuracy / mean abstention recall):** DeepSeek-R1-Distill-Llama-70B 0.81 / **0.46**; o1 0.80 / 0.66; s1.1-32B 0.80 / **0.43**; Llama 3.3 70B Instruct 0.78 / 0.66; GPT-4o 0.75 / 0.69; Qwen2.5-32B 0.75 / **0.71** (best mean recall); Llama 3.1 8B Instruct 0.70 / 0.66. Text: "reasoning models… are the top 3 performing LLMs [in accuracy]. At the same time, DeepSeek R1 Distill and s1.1 are close to the worst models in terms of abstention".
- **Precision (App. D.1, Fig. S1):** "on most datasets the precision is close to 1 for most models—i.e., models rarely over-abstain"; the ranking by F1 "mostly agrees" with the ranking by recall (Fig. S2). Over-abstention exceptions: precision degraded on *stale* after Tülu's post-training (App. D.3, Fig. S5) and when the judge sees the reasoning chain (Fig. S15).
- **Reasoning budget (Fig. 7, §4.3):** s1.1 with 512→4096 tokens: accuracy rises; abstention recall "either not improving (GSM8K-Abstain) or worsening (UMWP)".
- **Stale (Fig. 4, §4.1; Tab. 6):** on FreshQA, "improving correctness correlates with degraded abstention". Tab. 6 (Llama 3.1 8B Instruct): *stale* recall = 0.74 on the *fast subset* vs 0.95 on the full set — the author himself says stale is "saturated" and is the exception where the subset does not reproduce the full set.
- **System prompt (§4.4, App. D):** raises recall in standard and reasoning models "without a significant degradation in abstention precision".
- **By scenario (§4.1, Fig. 2/3c):** nearly perfect on BIG-Bench Known Unknowns, nearly zero on MediQ; "underspecification, subjectivity, and false assumptions are key challenges"; "Answer Unknown" is the scenario in which models do well ("with the exception of questions with unknown answers, frontier LLMs struggle across all other abstention scenarios", §1).

## 5. Limitations
**[Paper] (App. A/B):** English only; scenarios may not all be covered; **leakage** (the CoCoNot train split is in Tülu → confounds OLMo/Tülu); finite set of models; "any LLM judge will be imperfect"; public data → risk of inflation over time (no private test set).
**[My reading]**
- The judge was validated on 300 pairs **from general-domain benchmarks only**, over 2 generator models (Llama 70B, GPT-4o); the use of the same judge on the math/science datasets and on reasoning models was not explicitly validated.
- The judge prompt **receives the `should_abstain` label** (even though described as "noisy"): this may *inflate* agreement with the reference label and dilutes the independence between "judge" and "reference answer". The paper does not measure this influence.
- "Partial" abstention (answers, but with a caveat) counts as abstention under the broad definition; this is looser than our literal reference answer ("I don't know.").
- Generation at T=0.8, one sample per question, and **I did not see confidence intervals/bootstrap** or significance tests in the figures/tables I read.
- Explicit focus on *recall* because precision is high **in their set**, where most unanswerable samples are designed to provoke abstention; the answerable/unanswerable proportion per dataset varies and is not reported in an aggregate form that would allow computing a "naive" accuracy.

## 6. Reflections anchored in OUR project
| # | Point / file | Verdict | Reflection |
|---|---|---|---|
| 1 | **P8** — `f1_summary_evaluator` reports abstention *accuracy* | **SUPPORTS the spirit, qualifies the format** | The paper **does not use accuracy** as an abstention metric; it uses recall (primary), precision and F1. This supports abandoning accuracy. But it **does not support "F1 as the main metric"**: they *focus on recall* because precision ≈ 1. Our plan's gate ("correct abstention ≥ 90% on `unanswerable`") **is a recall** — aligned with their practice. |
| 2 | **P8** — over/under-refusal | **SUPPORTS** | They treat over-abstention through *precision* (Fig. S1, S5, S15) — that is, they keep the "refusing the answerable" side in the report, even if secondary. Recommendation: gate = recall on `unanswerable` + **precision/over-refusal as a guardrail (regression must not get worse)**, especially when changing the prompt (§4.4: a restrictive prompt does not degrade precision much *for them*; in our case it needs to be measured). |
| 3 | **P7** — `startswith(("I don't know","No papers found"))` | **CHALLENGES** | The paper spent an entire appendix justifying why *not* to use string/embedding matching (C.3.1) and adopted an LLM judge + human validation (300 pairs). Our reference answer ("I don't know."/"No papers found in this period.") and our prompt (`PROMPT_V1`: "say you don't know" with no fixed phrase) make `startswith` fragile exactly as they describe. **What we would adopt:** a judge (already planned in `abstention_quality`, 0.6.3) with validation on a human sample stratified by `should_abstain × prediction` (the C.3.3 recipe; fits within our 25 should-abstain + a sample of answerable ones). **What we would not adopt:** an 8B judge without our own validation — their 88% is on general English/diverse benchmarks, not on short answers from an abstracts RAG. |
| 4 | **P7** — judge T=0 | **SUPPORTS** | "greedy decoding… crucial" (C.3.1) coincides with plan 0.6 §4 (`temperature=0`). |
| 5 | **Slice `unanswerable`** (15, outside the corpus) | **SUPPORTS (≈ Answer Unknown)** | It is the scenario in which models do *best* in the benchmark (§1) → expect high recall; if our baseline comes out low, it may be the *prompt*/detection (P7) and not capability. |
| 6 | **Slice `stale`** ("What's new in <month>?") | **NEUTRAL/CHALLENGES the analogy** | AbstentionBench's "Stale" = the *answer changed after the model's cutoff* (FreshQA, two snapshots). Ours is different: the RAG corpus is fixed (250 abstracts) and the question asks for a period without documents → it is **absence in the corpus** (closer to Out-of-Database/Answer Unknown, see UAEval4RAG below) than "outdated model data". We cannot reuse their stale numbers as a reference. Transferable lesson: on FreshQA, getting more right correlated with abstaining less (Fig. 4) — a RAG that *retrieves* something vaguely from the month and answers may "look correct". |
| 7 | **P11** — judge = generator (gpt-4o-mini) | **NEUTRAL/weakly SUPPORTS** | Their judge (Llama 8B) is not the main generator model, but it also evaluates Llama 8B as a generator (Fig. 3b/S4) — they do not discuss self-preference. There is no evidence here for/against; the evidence is in RefusalBench (section 2). |
| 8 | **P8/gate** — sample size | **CHALLENGES (gap)** | 15 `unanswerable` → one error = 6.7 p.p.; a 90% gate ≈ accepts 1 error (93.3%) and fails with 2 (86.7%). The paper offers no guidance for small samples (no CIs). Our decision: report raw count + CI (Wilson/bootstrap) — **an engineering decision, not the literature's**. |
| 9 | System prompt for abstention (CKPT-6) | **SUPPORTS** | §4.4: a prompt describing scenarios raises recall; but "unlikely to fundamentally address" the lack of reasoning about uncertainty. Do not adopt it as a "solution", only as a measured lever with a precision guardrail. |

**What we would adopt:** (a) recall as the gate; (b) precision/over-refusal as a guardrail; (c) LLM judge with T=0 + stratified human sample to validate (including measuring how much `startswith` diverges from the judge); (d) report by scenario/slice, not just the mean.
**What we would NOT adopt:** (a) the simple mean of recall across scenarios as a single number; (b) passing the `should_abstain` label to the judge without testing the effect (for our literal reference answer, the judge should judge only the *response*); (c) transposing "Stale" as equivalent to our `stale`.

## 7. Useful quotations (max. 3)
1. "We define abstention as a response that refrains from directly answering the question, such as by expressing a lack of knowledge, communicating uncertainty or caveats, or highlighting unanswerable aspects of the prompt." — §3, p.3.
2. "However, as we find that models generally exhibit high abstention precision, we focus on abstention recall." — §3.4, p.5.
3. "We use greedy decoding (i.e. temperature = 0) for judge inference following prior works, and found this to be crucial for high performance of the judge." — App. C.3.1.

---

# 2. Muhamed et al. 2025 — RefusalBench: Generative Evaluation of Selective Refusal in Grounded Language Models

- **Full reference and link:** Aashiq Muhamed, Leonardo F. R. Ribeiro, Markus Dreyer, Virginia Smith, Mona T. Diab (CMU; Amazon AGI). *RefusalBench: Generative Evaluation of Selective Refusal in Grounded Language Models*. arXiv:2510.10390v1 [cs.CL], 12 Oct 2025. <https://arxiv.org/abs/2510.10390>
- **Reading confidence:** **full** (body §1–6, appendix on metric definitions, human validation C.2, analyses F, prompts I). Caveats: (i) figures/heatmaps (Fig. 2, 5–11, 18–30) extracted as scrambled text; I used only numbers that also appear in running text/captions; (ii) the PDF's column order comes out mixed up — appendix section numbering inferred from content; (iii) this is v1 (Oct/2025 preprint, no peer review as far as I read).

## 1. Problem the paper addresses
**[Paper]** In RAG systems, the model must **selectively refuse** when the retrieved context has defects (ambiguous, contradictory, missing information…), but it fails systematically: it either refuses "over 60%" of the answerable, or answers confidently despite defects (§1). Static benchmarks suffer from contamination, adaptive overfitting and saturation (§3.1) → they propose **generative evaluation**: creating new instances by controlled perturbation of answerable QA pairs.

## 2. Method (step by step)
**[Paper]**
1. **Taxonomy of 6 uncertainties (§3.2)** and the expected refusal code: P-Ambiguity→`REFUSE_AMBIGUOUS`; P-Contradiction→`REFUSE_CONTRADICTORY`; P-MissingInfo→`REFUSE_MISSING`; P-FalsePremise→`REFUSE_FALSE_PREMISE`; P-GranularityMismatch→`REFUSE_GRANULARITY`; P-EpistemicMismatch→`REFUSE_NONFACTUAL`.
2. **Perturbation engine (§3.3):** **176 linguistic "levers"** (~10 per category×intensity combination; 6×3=18). **3 intensities:** LOW = subtle uncertainty that "a competent model should resolve and answer correctly" (tests *excessive* refusal); MEDIUM = clear deficit → should refuse; HIGH = severe defect → should refuse. "The expected behavior is to answer correctly at LOW intensity and refuse appropriately at MEDIUM and HIGH."
3. **Generator–verifier pipeline (§3.4):** 4 LLMs generate (Claude-4-Sonnet, DeepSeek-R1, GPT-4o, Nova-Pro) and all verify all (7 criteria); only **unanimous consensus** passes.
4. **Bases (§4.1):** *RefusalBench-NQ* (100 base instances from NaturalQuestions+KILT, only questions that all frontier models got right without perturbation → 1,600 balanced samples) and *RefusalBench-GaRAGe* (100 multi-document bases, 20 for each of 5 domains → 1,506 samples, unbalanced).
5. **Evaluation protocol (§4.1, App. D/I):** each model is instructed to **answer OR emit "only the corresponding refusal code"** (prompt I.1: `REFUSE_AMBIGUOUS_QUERY`, `REFUSE_CONTRADICTORY_CONTEXT`, `REFUSE_INFO_MISSING_IN_CONTEXT`, `REFUSE_FALSE_PREMISE_IN_QUERY`, `REFUSE_GRANULARITY_MISMATCH`, `REFUSE_NONFACTUAL_QUERY`, `REFUSE_OTHER`). An **LLM judge (Claude-4-Sonnet)** classifies the output as `answer_attempt` or a `REFUSE_*` ("Look for refusal codes even if they appear with additional text") and grades the answers 1–5 (≥4 = correct, for NQ; GaRAGe uses the official RAF score). *(That is, refusal detection is by a **structured marker emitted by the model**, with the judge as a tolerant parser and quality evaluator — not by a free-text phrase regex.)*

**Metrics (App. D, "Core Behavioral Metrics" and "Other Refusal Analysis Metrics")** — with a plain explanation:
- **Answer Accuracy** — of the answerable, the fraction answered *and* correct.
- **Refusal Accuracy** — of the unanswerable, the fraction refused **with the right category** ("correctly refused with appropriate categorization"). *(More demanding than "refused".)*
- **Correct Refusal Rate** — of the unanswerable, the fraction that refused (without requiring the category).
- **False Refusal Rate (FRR)** — of the *answerable*, the fraction refused: "measuring over-cautious behavior" = **over-refusal**.
- **Missed Refusal Rate (MRR)** — of the *unanswerable*, the fraction answered: "potentially harmful over-confidence" = **under-refusal**. *(Note: MRR here ≠ Mean Reciprocal Rank from our roadmap — homonymous acronym.)*
- **Refusal Detection F1** — binary F1 (refuse × answer). *(Measures only "did it decide right?", without category.)*
- **Category Accuracy** — given that it refused correctly, did it get the *reason* right?
- **Hierarchical Refusal Score** = Detection F1 × Category Accuracy.
- **Calibrated Refusal Score (CRS)** = arithmetic mean of Answer Accuracy and Refusal Accuracy ("our primary balanced metric"); **Hybrid Score (GaRAGe)** = mean weighted by the proportion in the dataset.
- **ECE** (Expected Calibration Error, 5 bins), global/answers/refusals. *(Measures whether the stated confidence matches the hit rate.)*

## 3. Main contributions
1. A **generative** methodology (perturbation of answerable QA → unanswerable) with an error-bound proof under contamination (Theorem 3.1, App. B).
2. Taxonomy of 6 uncertainties × 3 intensities; 176 levers; two benchmarks and the framework released.
3. Study of **30+ models**: refusal = **two separable skills** (detecting *when* × categorizing *why*) (RQ2, Fig. 6).
4. Scale and extended reasoning do **not** improve refusal (RQ3); refusal is "trainable, alignment-sensitive" (DPO > SFT; Claude strong).
5. Evidence of **self-evaluation bias** and very low agreement among LLM verifiers (Fig. 3, 14) → justifies unanimous consensus.

## 4. Key results
- **Refusal accuracy (RQ2, §4.2):** best on NQ = **73.0%** (Claude-4-Sonnet); on GaRAGe the best is DeepSeek-R1 with **47.4%**; Claude-4-Sonnet drops from 73.0% → 36.1%. "No frontier model achieves >80% on both dimensions" (Fig. 5). A wrong answer is rare (<3.0% NQ, <3.4% GaRAGe; App. F.8): "the decision of whether to answer dominates model failures".
- **Over × under-refusal (App. F.3, Fig. 22):** GPT-4o = **FRR 62.8% / MRR 4.3%** on NQ (refuses 14.6× more than necessary); o4-mini = FRR 17.8% / MRR 21.5%; Claude: FRR 32–42%, MRR ≲11%; on GaRAGe Nova-Premier MRR 53.7%. Correlation **r = −0.78** (NQ) between false and missed: "models face a fundamental trade-off" (RQ1).
- **Detection ≠ categorization (§4.2, F.6–F.7):** Claude-3.5-Sonnet correctly refuses 88.2% of the unanswerable (NQ); GPT-4o refuses 88.4% but gets the category right in only **54.1%**. Models use `REFUSE_INFO_MISSING` as a "catch-all" (25% of predictions; Fig. 8). `REFUSE_GRANULARITY` ≈ "nearly unsolvable".
- **CRS (F.7):** Claude-4-Sonnet 65.3% on NQ (refusal 73.0%, answer 57.7%) → 51.7% on GaRAGe.
- **Intensity:** refusal rate grows monotonically from LOW to HIGH; GPT-4o already refuses 62.8% at LOW (F.5).
- **Scale/reasoning (RQ3):** Qwen jumps from 13.0% to 56.1% answer acc between 4B and 7B, but refusal acc stays <17% at all sizes; up to 4096 thinking tokens gives <1 p.p. gain in refusal acc (Fig. 32).
- **Calibration (§4.2):** all poorly calibrated; ECE from 0.286 (Claude-4-Sonnet, best) to 0.546 (GPT-4.1); ">73% of predictions occur at maximum confidence".
- **Dataset quality:** human pass-rate 93.1% (NQ) and 88.3% (GaRAGe), 180 perturbations per benchmark, **1 expert annotator** (§4.1, C.2); verifiers' self-evaluation 91.0% vs 82.1% cross; κ among verifiers 0.061–0.442 (NQ), 0.116–0.230 (GaRAGe) (App. E.1, Fig. 14).

## 5. Limitations
**[Paper] (Limitations):** programmatic perturbations may not capture the complexity of organic real-world sources; quality arbitrated by LLMs (shared bias not eliminated by consensus); risk that providers train on the levers (overfitting to the pattern, not to the instance).
**[My reading]**
- The human validation (1 annotator, 180 items) validates the **quality of the perturbations**, **not** the judge that classifies answer × refusal and category; **I did not find in the text I read a human validation of that judge** (the main judge, Claude-4-Sonnet, is also one of the evaluated models, one of the generators and one of the verifiers — a self-preference risk that the paper itself documents in Fig. 3).
- FRR is measured, to a large extent, on the **LOW** instances ("subtle" perturbations whose correct answer is to answer) — that is, the "answerable" is a *quasi-ambiguous* answerable, not a clean natural question; GPT-4o's 62.8% does not equal "refuses 62.8% of normal questions". Be careful when comparing with over-refusal on ordinary answerable questions.
- Context is always given (grounded) and **the defect lies in the supplied context/question**; it does not cover the case "retrieval failed and the context is irrelevant".
- NQ/GaRAGe setting (Wikipedia/web); no academic/abstracts domain.
- No confidence intervals in the tables I read.

## 6. Reflections anchored in OUR project
| # | Point / file | Verdict | Reflection |
|---|---|---|---|
| 1 | **P8** — accuracy × P/R/F1, over/under-refusal | **SUPPORTS strongly** | It is the paper closest to our proposal: it separates **FRR (over-refusal)** from **MRR (under-refusal)**, reports **Detection F1** and shows the trade-off (r=−0.78; GPT-4o 62.8%/4.3%). The trade-off supports requiring *both* sides in a single report. **But:** the "main" metric they propose is **CRS = simple mean of two accuracies**, not a per-class F1; and their F1 is diagnostic. |
| 2 | **P8** — `f1_summary_evaluator` | **SUPPORTS** | Direct mapping: our abstention recall ≈ *Correct Refusal Rate* (1−MRR); our over_refusal_rate ≈ FRR; F1 ≈ Refusal Detection F1. They confirm that "high detection F1 doesn't guarantee good categorization" — relevant only if we ever categorize the reason (not a plan requirement). |
| 3 | **P7** — detection by `startswith` | **CHALLENGES (and offers a concrete alternative)** | They **eliminate the ambiguity at the source**: the model is instructed to emit *only* a `REFUSE_*` code; the judge just reads ("even if they appear with additional text"). This is exactly option 3 of P7 ("structured marker / JSON field"). **We would adopt** (in a CKPT-6 experiment, not in the baseline): explicit marker + tolerant parser + judge as fallback. **We would NOT adopt** in EXP-0: changing the product prompt just to measure (the baseline must measure the real prompt). |
| 4 | **P8** — our two refusal reference answers ("I don't know." / "No papers found in this period.") | **NEUTRAL/CHALLENGES** | Their taxonomy (6 reasons) distinguishes *why* to refuse. Our project has 2 reasons (outside the corpus; no docs in the period). We do not need 6 categories; but reporting by slice (`unanswerable` × `stale`) is the analogue of reporting by category — and they show that categories differ a lot (INFO_MISSING easy, GRANULARITY almost impossible). |
| 5 | **Gate "≥90% on unanswerable"** | **CHALLENGES (caution)** | No frontier model exceeds 73% refusal accuracy **with category**; without category (detection) Claude-3.5 reaches 88.2% and GPT-4o 88.4% **at the cost of a huge FRR**. Conclusion: a standalone recall gate is **gameable by over-refusal** → reinforces making the over-refusal guardrail mandatory alongside the gate (P8, record in `decisions.md`). |
| 6 | **Slices `unanswerable`/`stale` × their scenarios** | **PARTIAL** | Our `unanswerable` (outside the corpus) ≈ P-MissingInfo/"information missing from the context" (the most tractable: 76–98% refusal acc per model on NQ, App. F.4) — but there the context *has* a given defect; in ours, retrieval returns **irrelevant context** and the model must notice. `stale` (asking for news from a month without papers) is analogous to temporal MissingInfo; **there is no "stale" category in RefusalBench** (unlike AbstentionBench). |
| 7 | **P11** — judge = generator | **SUPPORTS** | Empirical evidence (Fig. 3, §4.1): self-evaluation 91.0% vs 82.1% cross; Claude-4-Sonnet with negative bias (75.7% self vs 97.3% cross) — the bias **has variable sign**; κ among judges as low as 0.061 → "models apply fundamentally different quality criteria". Supports separating `JUDGE_MODEL` and measuring κ/sensitivity to 2 judges. (Note: the bias studied is in *perturbation generation*, not in judging answers — my extrapolation; still consistent with Zheng 2023/Wataoka 2024.) |
| 8 | **P7/gate** — sample size | **CHALLENGES (gap)** | 100 bases/benchmark, n per cell 100–200+; they do not report CIs either, as far as I read. Our n=15/10 requires reporting counts. |

**What we would adopt:** FRR + MRR (or their PT names "over/under-refusal rate") with *Detection F1* as a diagnostic; structured refusal marker as a product experiment; **separate the judge from the generator**; report by slice.
**What we would NOT adopt:** CRS (simple mean) as a gate — it hides the trade-off; the 6-category taxonomy; the programmatic perturbation design (our `unanswerable` is human/synthetic *by construction* from the corpus — ADR-001/003).

## 7. Useful quotations (max. 3)
1. "The expected behavior is to answer correctly at LOW intensity and refuse appropriately at MEDIUM and HIGH intensities." — §3.3.
2. "On all types, we find that models face a fundamental trade-off: the strong negative correlation (r = −0.78 on NQ) between false and missed refusals forces models to choose between being overly cautious or overly permissive." — §4.2, RQ1.
3. "…it refuses 14.6 times more often than necessary to avoid harmful outputs." (about GPT-4o, FRR 62.8% vs MRR 4.3%) — App. F.3.

---

# 3. Comparison AbstentionBench × RefusalBench × our project

| Dimension | AbstentionBench (2506.09038) | RefusalBench (2510.10390) | Our project |
|---|---|---|---|
| Unit | question (+optional context) with `should_abstain` | perturbed (question, context), with expected refusal code | question + abstracts RAG; `should_abstain` per slice |
| Origin of the unanswerable | existing datasets + variants (context removal) | **programmatic perturbation** of answerable QA | synthetic/human construction (ADR-001/003) |
| Abstention detection | **LLM-judge** Llama 3.1 8B, "yes/no", T=0; receives the label | **model emits `REFUSE_*`**; LLM-judge (Claude-4-Sonnet) classifies | `startswith` (P7) → `abstention_quality` judge planned |
| Human validation | 300 pairs, authors, stratified, 50/50 val/test; acc 0.88 | 180 perturbations, **1** expert (validates the dataset, not the judge) | 100/100 examples reviewed; judge still without κ |
| Main metric | **abstention recall** | **Refusal Accuracy** (with category) and **CRS** (mean of Ans+Ref acc) | abstention accuracy (→ P8) |
| Over-refusal | via **precision** (secondary; ≈1) | explicit **FRR** + Detection F1 | not measured today |
| Under-refusal | 1 − recall | explicit **MRR** (homonym of ranking MRR!) | not measured today |
| P/R/F1 or accuracy? | P/R/F1 (focus on R); **not** accuracy | FRR/MRR/F1 + per-class accuracies; **CRS** is a mean of accuracies | accuracy (to change) |
| "Stale" | **yes**, named scenario (FreshQA; answer changed post-cutoff **of the model**) | no | `stale` = fixed corpus with no papers in the period |
| Supports P8? | **Partially** (P/R/F1; but focus on recall) | **Yes** (over/under + F1 + trade-off) | — |

**[My reading] P8 verdict:** both papers **support abandoning standalone accuracy** and measuring over- and under-refusal separately. **Neither prescribes "F1 as the headline"**: AbstentionBench prioritizes *recall* (= our gate), RefusalBench prioritizes per-class accuracies + CRS. Therefore the P8 proposal (recall/precision/F1 + over/under) is **compatible and conservative**, but the paper is not a source for claiming "the literature mandates F1".

---

# 4. Extra A — Peng et al. 2025 — Unanswerability Evaluation for Retrieval Augmented Generation (UAEval4RAG)

- **Reference:** Xiangyu Peng, Prafulla Kumar Choubey, Caiming Xiong, Chien-Sheng Wu (Salesforce Research). *Unanswerability Evaluation for Retrieval Augmented Generation*. arXiv:2412.12300v3, 21 Apr 2025. <https://arxiv.org/abs/2412.12300> · code <https://github.com/SalesforceAIResearch/Unanswerability_RAGE>
- **Reading confidence:** **partial** — body §1–4 read (taxonomy, pipeline, metrics, Tables 1–5, human validation); **I did not read** the full appendix (prompts A.1–A.8, Tables 6–18, Table 16 Llama-Guard); tables extracted scrambled → I use only numbers that can be reconstructed with confidence.

**1. Problem.** RAG benchmarks measure only answerable questions; unanswerable benchmarks test LLMs *without* a knowledge base, so that "rejection often stems from the inability to retrieve relevant context rather than a true understanding that the request should not be fulfilled" (§1).

**2. Method.** Taxonomy of **6 categories** (§3.1): Underspecified, False-presupposition, Nonsensical, Modality-limited, Safety-concerned and **Out-of-Database** (relevant to the domain, but with no answer in the base). A pipeline that synthesizes unanswerable requests **for any knowledge base** with LLM generation + verification (§3.2); for Out-of-Database, crawl recent news → question → retrieve chunks → verify that none contains the answer (Fig. 3). LLM metrics (§3.3): **Acceptable Ratio** (acceptable answer by per-category criteria), **Unanswered / Answered / Ask-for-Clarification Ratio**, and **Joint Score = w1·Correctness + w2·Acceptable Ratio** with w1=0.7, w2=0.3 ("no universal weight").

**3. Contributions.** A taxonomy for RAG; automatic generation of unanswerables anchored in the base; analysis of 27 combinations of components (embedding, retriever, reranker, rewriting, 3 LLMs, 3 prompts) on 4 datasets.

**4. Results.** Pipeline validation: 3 authors, 92% accuracy (TriviaQA and MuSiQue), agreement 0.85/0.88 (§4.1). LLM judge × 150 pairs labeled by 3 authors (agreement 0.76 and 0.83): accuracy 81–84% and F1 76–86% across the three judges (GPT-4o, Claude 3.5, DeepSeek-R1) (Tab. 1). "No single configuration" is optimal on all datasets (§4.3). **Restrictive prompts** greatly raise Acceptable/Unanswered but **reduce** correctness on the answerable (Tab. 3: e.g. TriviaQA, prompt default→#2: acceptable 53.2%→83.0% and correct 88.0%→74.8%; MuSiQue default→#2: acceptable 61.7%→88.0% and correct 49.0%→16.0%). "Underspecified" is the hardest category; "False Presupposition" and "Out-of-Database" the easiest (§4.3).

**5. Limitations.** *[Paper]* no dedicated section in the parts read; *[My reading]* synthetic + LLM-judge without CIs; Wikipedia/trivia domain; the 0.7/0.3 weighting is arbitrary (admitted); the column order of Tab. 2–5 in the extracted PDF is ambiguous, I did not reproduce cell numbers.

**6. Reflections anchored in the project.**
- **P8 — SUPPORTS** the need to look at **both sides**: the answerable × unanswerable *trade-off* is the central result (restrictive prompt ↑ rejection, ↓ correctness; Tab. 3). Supports the over-refusal guardrail; **challenges** on only one point: they summarize in a weighted **Joint Score** (0.7/0.3), not in per-class F1. For us, a Joint Score would be a possible second aggregate, but the weight is arbitrary → **we would not adopt it as a gate**.
- **Slice `unanswerable` and `stale` — SUPPORTS the taxonomy:** *Out-of-Database* (domain-relevant question with no answer in the base) is the most faithful analogue of our out-of-corpus `unanswerable`, and its construction uses **recent news** (≈ our `stale`: "what's new in <month>?"). The lesson: their benchmark shows that, for RAG, **Out-of-Database is easy** (high acceptable in Tab. 4) — so a low baseline on our `unanswerable` would point to the prompt/detection (P7) more than to capability.
- **P7 — NEUTRAL/CHALLENGES:** they use an LLM judge with per-category in-context examples and validate with 150 human pairs (acc ≈ 82–84%) — the same recipe as AbstentionBench; **human×human agreement of 0.76** shows that "what counts as abstention/acceptable" is ambiguous even for humans → skepticism toward 90% heuristic×judge agreement targets (P7's target).
- **What we would adopt:** the distinction "rejected" × "asked for clarification" × "answered" (three states, not two); per-category criteria. **What we would not:** Joint Score with fixed weights as a release gate.

**7. Useful quotation.** "…rejection often stems from the inability to retrieve relevant context rather than a true understanding that the request should not be fulfilled." — §1, p.1 *(= the risk of measuring only by "retrieved something / retrieved nothing")*.

---

# 5. Extra B — Cui et al. 2025 — OR-Bench: An Over-Refusal Benchmark for Large Language Models

- **Reference:** Justin Cui, Wei-Lin Chiang, Ion Stoica, Cho-Jui Hsieh. *OR-Bench: An Over-Refusal Benchmark for Large Language Models*. ICML 2025 (PMLR 267); arXiv:2405.20947v5, 15 Jun 2025. <https://arxiv.org/abs/2405.20947>
- **Reading confidence:** **partial** — I read the abstract, §1–3 (definitions, generation/moderation pipeline) and the start of §4 (setup, evaluation, Table 1); **I did not read** §4.3 onward, Tables 2/6/7, jailbreak defenses and most of the appendices (F, Q, V). Scope: **safety refusal** (near-toxic prompts), *not* abstention due to lack of knowledge.

**1. Problem.** Safety alignment causes **over-refusal**: refusing harmless prompts. A large-scale benchmark is missing (there was only XSTest, 250 manual prompts) (§1).

**2. Method.** *Over-refusal* defined as "when a model refuses to provide a helpful response, even when a safe and plausible answer is possible" (§3.1). Pipeline: (1) toxic seeds by Mixtral-8×7B in 10 categories; (2) rewriting into benign "borderline" prompts (5 per seed, with 5 few-shot examples); (3) **moderation by an ensemble of 3 LLMs** (GPT-4-turbo, Llama-3-70b, Gemini-1.5-pro; majority vote) → OR-Bench-80K; **Hard-1K** subset (rejected by ≥3 of the largest models of each family) and **OR-Bench-Toxic** (600 toxic prompts "to prevent indiscriminate responses"). **Refusal detection:** **keyword matching** on the 80K set and **GPT-4** on Hard-1K and Toxic; "keyword matching closely approximates GPT-4 evaluations… discrepancies of 2.4% for GPT-3.5-turbo-0125 and 1.2% for Llama-3-70b on sampled datasets" (§4.1; App. F, not read).

**3. Contributions.** Automated pipeline; OR-Bench-80K/Hard-1K/Toxic; evaluation of 32 models from 8 families.

**4. Results.** Ensemble moderator vs human expert: TP 94.7/96.0, FN 5.3/4.0, TN 92.0/84.0, FP 8.0/16.0, Acc 94.0/93.0 (Tab. 1; columns "human/ensemble" in the order of the extracted PDF) → "over 98% of the performance level of human experts" (§3.2.3). Main result: **Spearman 0.89** between rejection of toxic prompts (safety) and over-refusal (Fig. 1) — "most models simply trade over-refusal for safety". Larger model ≠ better balance; Claude safer and more over-refusing; GPT-3.5-turbo reduces over-refusal in newer versions.

**5. Limitations.** *[Paper]* not read. *[My reading]* safety domain (standardized refusal phrases: "I cannot…", "I'm sorry…") — **the success of keyword matching is specific to this type of refusal** and does not automatically transfer to epistemic abstention ("I don't know", "the context doesn't say").

**6. Reflections anchored in the project.**
- **P7 — SUPPORTS with a caveat:** empirically validates that a **cheap heuristic can approximate a strong judge** (≤2.4% divergence) *when measured against the judge*, exactly the P7 design (heuristic as a pre-filter + heuristic×judge agreement in EXP-0). **Caveat:** in safety refusal the phrases are predictable; in our epistemic abstention with `PROMPT_V1` without a fixed phrase, the agreement has to be **measured**, not assumed.
- **P8 — SUPPORTS:** it only makes sense to report over-refusal **paired** with the rate of undue acceptance (toxic) — they include the Toxic set precisely so that a model that "refuses everything" does not look good. It is the exact analogue of our risk: high abstention bought with indiscriminate refusal. Supports an **over-refusal guardrail** in the gate.
- **What we would adopt:** an inseparable pair of metrics (abstention recall + over-refusal); measuring heuristic×judge divergence. **What we would not:** transposing the keyword-matching rate as evidence for our case.

**7. Useful quotation.** "…most models simply trade over-refusal for safety, with few breaking the trade-off." — §4.2 (result of the Spearman 0.89 over Fig. 1).

---

# 6. Synthesis for the roadmap

| Point | What the full reading allows us to state | What remains an engineering decision (not the literature's) |
|---|---|---|
| **P7** | String/embedding matching is considered inadequate (AbstentionBench C.3.1); two validated alternatives: **LLM judge T=0 + stratified human sample** (AbstentionBench, Tab. 1) and **structured `REFUSE_*` marker** (RefusalBench §4.1); cheap heuristic ≈ judge on safety refusal (OR-Bench). Not even human×human agree 100% (UAEval4RAG 0.76/0.83). | Heuristic×judge agreement target ≥90%; which judge to use; whether to change the product prompt. |
| **P8** | Separating over- and under-refusal is common practice (RefusalBench FRR/MRR; AbstentionBench precision; OR-Bench paired with Toxic; UAEval4RAG trade-off). **Standalone accuracy is not used by any.** | That F1 be the summary metric (they prioritize recall or means of accuracy); CI/Wilson for n=15/10; the rule "over-refusal does not rise by more than X p.p.". |
| **P11** | Self-preference and low κ among LLM judges (RefusalBench Fig. 3/14). | Which model to use as judge. |
| Small data | None of the papers offers guidance for n≈15. | Report raw counts + CI. |
