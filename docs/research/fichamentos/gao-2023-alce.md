# Gao et al. 2023 — ALCE: generating text with citations and evaluating it automatically

- **Full reference:** Gao, T., Yen, H., Yu, J., Chen, D. (2023). *Enabling Large Language Models to Generate Text with Citations*. Proceedings of EMNLP 2023 (Princeton University). arXiv:2305.14627 [cs.CL]. Link: <https://arxiv.org/abs/2305.14627> (PDF: <https://arxiv.org/pdf/2305.14627>). Code and data: <https://github.com/princeton-nlp/ALCE> (footnote 1 of the paper; **I did not inspect the repository**).
- **Reading confidence:** **complete, from arXiv version v2** (2023-10-31), read from the text extracted from the PDF with `pdftotext` (nothing saved in the project). I read the body (§1–§8, Limitations) and Appendices A–I: ELI5 sub-claims (A), implementation (C), shortcuts (D), recall/precision discussion (E), human evaluation (F), extra experiments and Tables 12–21 (G), prompts 22–29 (H) and examples 30–31 (I). Tables 19–21 were cross-checked against Tables 4–6 on the main rows (this is where the small inconsistencies listed in §4 came from); **I did not check every standard deviation**. Figures 1–4 were read only through the caption and the extracted text (the images were not inspected; the recall@k numbers in Figure 4 were checked against Tables 12–14). I **did not read** TRUE (Honovich et al. 2022), Liu et al. 2023 or the ALCE code, so implementation details (handling of edge cases, number of annotators per item) remain **to verify**. The labels "(a) the paper states" and "(b) my interpretation" are marked in the text.

## 1. Problem the paper addresses

**(a) The paper states:** LLMs "are prone to hallucination" and it is hard for the user to trust them without evidence (§1). The proposal is to require the model to generate the text **with citations** to passages from a corpus, so that (1) the user can verify the claims and (2) the model faithfully follows the cited passages, which "has the promise to improve correctness and alleviate hallucination" (§1). Earlier work (WebGPT/Nakano 2021, Menick 2022) uses commercial search engines, closed models and human evaluation, "challenging to reproduce and compare different modeling approaches" (abstract; §1 also says "expensive and difficult to reproduce"). The stated contribution is **ALCE**, "the first reproducible benchmark for automatically evaluating LLMs’ generations with citations" (abstract, §1), with automatic metrics along three dimensions (fluency, correctness, citation quality) validated against humans (§6).

**(b) My interpretation:** what matters for us is not the benchmark itself, but the **decomposition of citation quality into two separate questions** (is the claim supported by what it cites? / was each citation necessary?) and the evidence that a citation metric **in isolation** can be fooled (§3.4, Appendix D). The paper comes from NLP, not from RAG engineering, and works with questions that are **always answerable** (see §5).

## 2. Method (step by step, with the exact definitions)

**Task (§2).** Given a question `q` and a corpus `D` of passages, the system returns an output `S` with `n` *statements* (claims) `s1…sn`; each `si` cites a list of passages `Ci = {ci,1, ci,2, …}` with `ci,j ∈ D`. The split into claims is **by sentence boundary** (footnote 5; in QAMPARI, each entity in the list is a claim). **At most 3 citations per claim** (footnote 4). The corpus is split into **100-word passages** (§2). The paper observes that "almost all" sentences that LLMs produce carry valuable information and require a citation (§2), so the benchmark does **not** exempt any sentence.

**Data (§2, Table 1).** 3 datasets, 1,000 random examples from the development set of each: **ASQA** (ambiguous factual question, Wikipedia 2018-12-20, 21M passages), **QAMPARI** (list of entities, Wikipedia) and **ELI5** (why/how, Sphere corpus, 899M passages). The benchmark ships no training data.

**Metrics — three dimensions (§3).**

- **Fluency = MAUVE** (§3.1; "how similar is the generated text to human text in distribution"): used only as a *sanity check*, because MAUVE "is sensitive to output length and text style". Applied to ASQA and ELI5; omitted for QAMPARI. In the implementation, output and reference are truncated to 100 words (App. C).

- **Correctness** (§3.2; "does the answer match a reference answer?"), one metric per dataset:
  - **ASQA — EM recall** (*exact match recall*; "of the short answers in the reference, what fraction appears as an exact span in the generation?"). Example from Figure 2: the reference has 3 dates, the generation gets 1 right, recall = 33.3%.
  - **QAMPARI — precision and recall** of entities, by *exact match* against the gold list, with a **recall-5** adjustment: recall counts as 100% if the prediction has at least 5 correct answers (the user only wants a few examples).
  - **ELI5 — claim recall** ("of the expected key claims, how many does the answer entail?"): `text-davinci-003` generates **3 sub-claims** from the human answer (with 3 hand-written examples, Table 22) and the **TRUE** model (T5-11B NLI) checks whether the output *entails* each one. The authors reject ROUGE-L because it "does not account for the different ways of expressing the same answer and it can be easily gamed" (App. A; Table 10: the "top-1 passage" has ROUGE-L 19.1 but claim recall 3.0).

- **Citation quality** (§3.3), the two metrics that interest us most. The verifier is the function `ϕ(premise, hypothesis)` = the TRUE NLI, which returns 1 if the premise entails the hypothesis and 0 otherwise (T5-11B fine-tuned on SNLI, MNLI, Fever, Scitail, PAWS and VitaminC; App. C). Each passage is fed as "Title: {TITLE}\n{TEXT}" and the cited passages are concatenated with "\n" (App. C). The idea follows the AIS framework (*attributable to identified sources*; "is this claim true based only on the cited sources?").
  - **Citation recall** (citation coverage; "is the claim supported by what it cites?"). Per claim, it is worth **1 if and only if** `Ci ≠ ∅` **and** `ϕ(concat(Ci), si) = 1`; otherwise 0. The final score is the **mean over the claims**. In other words: **a claim with no citation at all is worth 0**; the premise is the *concatenation* of all passages cited for that claim, not each one in isolation. Example from Figure 3: 3 claims, 2 supported → recall = 2/3 = 66%.
  - **Citation precision** (citation precision; "was each citation really necessary?"). Computed **per citation** (0 or 1) and averaged over **all citations**. A citation `ci,j` is "**irrelevant**" if and only if **(a)** `ϕ(ci,j, si) = 0` (on its own it does not support the claim) **AND (b)** `ϕ(concat(Ci∖{ci,j}), si) = 1` (the others, without it, already support it). Then `ci,j` has precision 1 if `si` has recall 1 **and** `ci,j` is not irrelevant. **Recall = 1 is a prerequisite** for precision = 1: if the claim's recall is 0, all of its citations have precision 0 (Figure 3). Example from Figure 3: 6 citations in total, 1 irrelevant ([2] in `s3`) and claim `s2` with recall 0 zeroing out its own → precision = 4/6 = 66%. It **does not require a minimal set**: the authors allow redundancy "because human writing often cites redundant sources to enhance credibility" (§3.3). The acknowledged gap: the algorithm **does not detect partial support** (a citation that supports only part of the claim may be wrongly marked "irrelevant") (§3.3, App. E).

- **Why all three together (§3.4, App. D).** Two shortcuts: (1) return the **top-1 passage** as the answer, citing it; (2) return the **first two sentences** of the top-1. Both have "almost perfect" citation, but (1) has low fluency (text too long) and (2) has low correctness (low coverage). The numbers are in Table 11 (see §4 below).

**Systems evaluated (§4).** Retrieval: GTR/DPR (Wikipedia) and BM25 (Sphere), top-100. Synthesis: **VANILLA** (top-k passages in the context + 2 demonstrations), **SUMM/SNIPPET** (summaries/snippets of 10 passages), **INTERACT** (check the full text on demand), **INLINESEARCH** (the model calls search during generation) and **CLOSEDBOOK** (no passages, no citing). Post-editing: **RERANK** (4 samples, picks the one with the highest automatic citation recall) and **POSTCITE** (cites afterwards, by similarity with GTR). Models: ChatGPT (gpt-3.5-turbo-0301, 4K), ChatGPT-16K, GPT-4 (8K), LLaMA and derivatives. Three runs with different seeds (except RERANK, 16K and GPT-4, one run) (App. C, G.6).

**Human validation (§6, App. F, G.5).** Surge AI, 20 USD/hour; **100 examples** from ASQA and from ELI5; outputs from **3 systems** (ChatGPT VANILLA, ChatGPT RERANK, Vicuna-13B VANILLA). Humans judge: utility (Likert 1–5); citation recall (do the citations together *fully* support the sentence?); citation precision (each citation *fully supports / partially supports / does not support*; precision 1 if the sentence has recall 1 and the citation supports it **at least partially**). **Note:** the human rule is more lenient than the NLI one (which only sees "entails or not"), which is why humans tend to give higher precision. The number of annotators per item and how κ was computed (over which labels) **do not appear in the text I read** (to verify).

## 3. Main contributions (list)

1. The ALCE benchmark: 3 datasets, 3 dimensions, open source (§1).
2. The operational definitions of **citation recall** and **citation precision** per claim/citation via NLI (§3.3).
3. The robustness-to-shortcuts argument, with the "top-1 passage" and "first 2 sents" cases (§3.4, App. D).
4. ELI5 correctness through **claim recall** with LLM-generated sub-claims verified by NLI (§3.2, App. A).
5. Comparison of synthesis and post-editing strategies (VANILLA, SUMM, SNIPPET, INTERACT, INLINESEARCH, RERANK, POSTCITE) and of models, with 5 findings (§1, §5).
6. Human validation that the ranking of the automatic metrics matches the human one (§6, Tables 8–9).
7. Three challenges pointed out: retriever quality, limited context window, and the difficulty of synthesizing multiple documents without being distracted by the irrelevant ones (§1, §8).

## 4. Key results (with numbers and where they are)

**(a) What the paper states:**

**Human validation of the metrics (Tables 8–9, §6).** Human and automatic rankings are "consistent"; absolute values are "very close", except for RERANK (which uses the automatic recall itself to choose, §6).

| System | ASQA human Rec./Prec. | ASQA ALCE Rec./Prec. | ELI5 human Rec./Prec. | ELI5 ALCE Rec./Prec. |
|---|---|---|---|---|
| ChatGPT VANILLA | 74.7 / 76.6 | 75.3 / 74.4 | 50.8 / 52.4 | 52.8 / 50.4 |
| ChatGPT + RERANK | 79.3 / 81.9 | 83.9 / 80.8 | 59.7 / 60.6 | 63.0 / 60.6 |
| Vicuna-13B VANILLA | 51.6 / 51.5 | 50.3 / 50.1 | 13.4 / 19.2 | 13.6 / 18.1 |

- **Cohen's κ human × ALCE: 0.698 for citation recall ("substantial agreement") and 0.525 for citation precision ("moderate")** (§6). Accuracy taking the human as the reference answer: **85.1% (recall) and 77.6% (precision)** (§6, App. G.5). Detection of insufficient citations: recall 82.3%, precision 84.2%; detection of "irrelevant" citations: **recall 75.6%, precision 66.1%** (effective, but with a relatively high false-positive rate because the NLI model cannot detect partial support; paraphrase, App. G.5).
- Utility (Likert) varies little across systems: 3.7–3.9 (ASQA) and 3.5–3.6 (ELI5) (§6).
- Quality of the ELI5 sub-claims: **112 out of 120 (93.33%)** good in 40 inspected answers; the NLI matched **80.0%** of the human labels on the same 40 outputs × 3 sub-claims (120 pairs) (App. A).

**Main results (Tables 4 and 6, §5.1).**

| Config. (ChatGPT unless noted) | ASQA: MAUVE / EM rec. / Cit. rec. / Cit. prec. | ELI5: MAUVE / claim rec. / Cit. rec. / Cit. prec. |
|---|---|---|
| VANILLA (5 psg) | 66.6 / 40.4 / 73.6 / 72.5 | 57.2 / 12.0 / 51.1 / 50.0 |
| VANILLA + RERANK | 77.0 / 40.2 / 84.8 / 81.6 | 56.1 / 11.4 / 69.3 / 67.8 |
| SUMM (10 psg) | 70.0 / 43.3 / 68.9 / 61.8 | 40.3 / 12.5 / 51.5 / 48.2 |
| CLOSEDBOOK + POSTCITE | 52.7 / 38.3 / 26.7 / 26.7 | 32.6 / 18.6 / 15.5 / 15.5 |
| GPT-4 VANILLA (5 psg) | 67.1 / 41.3 / 68.5 / 75.6 | 38.4 / 14.2 / 44.0 / 50.1 |

- **Abstract / §1:** on ELI5, "even the best models lack complete citation support 50% of the time" (ChatGPT VANILLA has recall 51.1 and GPT-4 44.0–49.5 in Tables 6 and 21, that is, **~49–56% of claims without full support**; the "50%" phrase is the authors' rounding).
- **VANILLA is strong** ("close-to-the-best performance among all prompting strategies", §5.1). SUMM/SNIPPET improve correctness and **worsen citation** on ASQA and ELI5. RERANK improves citation (ASQA: 73.6 → 84.8 recall; ELI5: 51.1 → 69.3). INLINESEARCH is worse than VANILLA on citation.
- **CLOSEDBOOK:** gives comparable correctness (ASQA 38.3 versus 40.4; on ELI5 even higher, 18.6 versus 12.0), but post-hoc citation is weak: recall 26.7 versus 73.6, "lower than VANILLA by 47% on ASQA" (a difference of 46.9 points; the arithmetic checks out) (§5.1). The authors' explanation: models with context "are easily distracted by irrelevant passages" and closed-book generates correct texts but "not similar to any retrieved passages", hard to cite after the fact.
- **Retrieval (Figure 4, Tables 12–14):** with 5 passages, GTR covers only **56.8%** of the ASQA answers (recall@5), and ChatGPT VANILLA's correctness (40.4) stays below that. The **oracle** version (5 gold passages, whose recall equals recall@100) reaches 48.9 on ASQA, 37.0 on QAMPARI and 21.3 on ELI5 (Tables 19–21), still below the retriever's recall (78.4; 65.6; 31.8, Tables 12–14). The authors conclude that, even though correct answers are in the context, LLMs struggle to use them (§5.3, my paraphrase). **Exception the text mentions:** on ELI5 top-5, VANILLA (12.0) is *above* BM25's recall@5 (9.6).
- More passages: for ChatGPT, correctness plateaus at top-1 and citation at top-3 (§5.3); GPT-4 improves with 20 passages, but the improvement is "not proportional to the retrieval performance".

**"Top-1 passage" shortcut (Table 11, App. D).**

| Variant (ASQA) | MAUVE | EM rec. | Cit. rec. | Cit. prec. |
|---|---|---|---|---|
| ChatGPT VANILLA (GTR top-5) | 66.6 | 40.4 | 73.6 | **63.0 — to verify** (Table 4 gives 72.5 for the same configuration) |
| Top-1 passage as the answer | **20.8** | 35.1 | **99.4** | **99.4** |
| First 2 sentences of the top-1 | 67.2 | **18.9** | 98.7 | 98.7 |

The shortcuts have almost perfect citation, but the first is brought down by **fluency** (MAUVE 20.8) and the second by **correctness** (EM 18.9, less than half of 40.4). Note: in case (1) correctness (35.1) drops only 5.3 points relative to the model; the App. D text says that "fluency and correctness are dramatically lower" for both, but only fluency supports this for the top-1 (my reading of the numbers).

**(b) Inconsistencies I found in the paper itself (all small, none changes the conclusions):**
1. **Table 11 × Table 4/7/19:** the "ChatGPT" row of Table 11 shows **precision 63.0**, whereas the same configuration (ChatGPT VANILLA, GTR top-5, ASQA) has **72.5** in Tables 4, 7, 15, 16, 17 and 19; MAUVE (66.6), EM (40.4) and recall (73.6) match. My hypotheses, not verifiable: a typo or a different run. **To verify** in the repository before citing the 63.0. If it were 72.5, the gap to the shortcut (99.4) would be 26.9 points.
2. **Table 4 × Table 19** (same experiment, different values due to rounding or error): VANILLA MAUVE 66.6 versus 66.8; SUMM recall 68.9 versus 68.8; CLOSEDBOOK EM 38.3 versus 38.2; INLINESEARCH: citation precision 58.2 (Table 4) versus 58.3 (Table 19), and Table 19 shows ROUGE-L = 58.2, a value out of line with the other rows (29–40), probably a typo (my hypothesis).
3. **Table 6 × Table 21:** CLOSEDBOOK citation recall/precision 15.5 versus 15.4; and a corrupted value in `SNIPPET` ("2.w") in Table 21.
4. **Instruction × metric:** the VANILLA prompt (Table 23) asks to "only cite a minimum sufficient subset of the documents", but the precision metric **does not require** a minimal set (§3.3). Not an error, it is a difference between what the model is asked and what is measured.

## 5. Limitations (the author's and the ones I identify)

**From the author (§Limitations, §3.3, App. E):**
- MAUVE is sensitive to length and can give unstable results.
- The ELI5 sub-claims may **not cover all possible answers**, because the question is open-ended.
- Citation quality is **limited by the NLI's accuracy**; the NLI does not detect "partial support", which yields **lower precision than the human one** (also App. G.5: 66.1% precision on detecting irrelevant ones). They tried prompting ChatGPT to judge partial support, which "yields poor results" (App. E).
- The datasets do not cover multi-hop reasoning, math or code.
- Focus on *prompting*, without training the model.

**Mine (marked as interpretation):**
- **No unanswerable questions and no abstention.** Every example has an answer. I searched the text for "unanswerable", "abstain", "refus", "don't know": **no occurrences**. An "I don't know" answer without a citation falls under the recall-0 rule (and the precision rule zeroes out the rest). So ALCE does not say how to handle refusal; any adaptation of ours is extrapolation.
- **The verifier is a 2022 NLI (T5-11B), not an LLM judge.** The κ of 0.698/0.525 holds for *that* verifier, with 100-word Wikipedia/Sphere premises and answerable questions. It does not automatically transfer to a gpt-4o-mini judge over arXiv abstracts.
- **Moderate κ (0.525) on precision**: under the acceptance criterion exemplified in our roadmap (P11, κ ≥ 0.6), ALCE's *own* precision metric **would not pass** (my interpretation; the roadmap says "e.g., κ ≥ 0.6", it is not a closed rule).
- **Small and narrow validation:** 100 examples, 3 systems, 2 datasets (no QAMPARI), number of annotators per item not found. No confidence intervals for κ in the text I read.
- **Recall is "all or nothing" per claim**: a claim with 2 facts and only 1 supported is worth 0, the same as a fully made-up one. It does not separate "partially supported" from "not supported".
- **Goodhart risk:** RERANK chooses by the automatic metric itself; the authors themselves see divergence from the human in this case (Table 8: 83.9 automatic versus 79.3 human) and confirm with humans that the gain is real, but smaller.
- **The citation points to an index** (`[1]…[5]` among the provided passages), so ALCE never has to handle a **nonexistent cited ID**. In our system the model cites a free-form `arxiv_id` and may invent it (our extension, see §6).
- **Attribution is not correctness:** a false claim can be "supported" by a wrong source (the NLI only sees premise and hypothesis). The paper handles this by reporting correctness separately.

## 6. Reflections anchored in OUR project

Reference for the points: P1–P11, D-1, D-2, L-3 in `docs/roadmap/ckpt-0.6.1b-metrics-roadmap.md`; metrics plan in `docs/roadmap/ckpt-0.6-plan.md` §3.2 (`citation_accuracy` **has no P number** in the roadmap; it is planned as "regex extracts `[2609.xxxxvN]` + judge per citation", `evaluators.py`, batch 0.6.3). About the data: 250 papers, 843 chunks of ~420 characters, abstracts only (roadmap §3).

**R1 — Split `citation_accuracy` into recall and precision (SUPPORTS the design, CHALLENGES the single-score version).**
Plan 0.6 defines a single score per citation: "does the chunk of that `arxiv_id` support the cited sentence?". This amounts to a *citation precision* without a recall precondition and **does not measure a claim with no citation** (the "recall" half of ALCE). ALCE says it prioritizes recall "as it entails a well-supported and truthful answer" (§3.3), and that precision serves the user experience (less review of superfluous sources). **Affects:** `citation_accuracy` (0.6.3, `evaluators.py`). **What we would adopt:** two outputs per answer, `citation_recall` (fraction of claims supported by their citations; a claim with no citation counts 0) and `citation_precision` (fraction of non-irrelevant citations), computed sentence by sentence, with the "precision requires recall = 1" rule from §3.3. Note: Magesh 2025 (another reading note) also asks to distinguish *a citation that does not support* from *a claim with no citation*; the two papers converge here (my interpretation).

**R2 — The premise of each citation is the concatenation of the chunks of that `arxiv_id` in the retrieved context (SUPPORTS; our adaptation).**
ALCE concatenates **all cited passages** of the claim (`concat(Ci)`) and uses that as the premise (§3.3). In our case, a citation is an `arxiv_id`, and the paper may have 2–5 chunks in the top-5. The premise should be the concatenation of the **retrieved** chunks of that `arxiv_id` (not the whole abstract from the corpus), since only what the model saw can support the claim. **Affects:** `citation_accuracy` and **D-1** (gold per chunk × per paper): the citation is at paper level, while the `gold_chunk_ids` of the `answerable`/`persona` slices is at chunk level. The paper-level premise is consistent with what the model cites, but it does **not** say whether the right chunk was there (that remains recall@k). **It does not decide D-1**; it only shows that the citation lives at the paper level (my interpretation). One advantage: ALCE shows that the unit of verification can be the **set** of passages of the claim, rather than each chunk in isolation, which frees us from the "chunk different from the gold but that also answers" problem.

**R3 — Citing an ID absent from the context = invalid citation (our extension; ALCE does not need this).**
In ALCE the model cites `[1]…[5]`, indices of provided passages, so the case "cited something it never received" does not exist. In our `[arxiv_id]` format the model can cite a real paper that was not retrieved, or invent an ID. **Proposed rule:** if the cited `arxiv_id` is not in `retrieved_context`, the citation is **invalid**: it does not count as support (it counts in the precision denominator as an error) and the claim is left without support for recall purposes. Report the number of invalid citations as a separate counter (Magesh's "fabricated"). **Affects:** step 1 (regex) of `citation_accuracy`. **Support from the paper:** none direct; it is a gap that ALCE's design does not cover (to verify whether the repository handles out-of-range indices).

**R4 — Abstention: exempt from recall (CHALLENGES direct use; link to P7 and P8).**
ALCE has no unanswerable questions (see §5), and a refusal without a citation gives **recall 0**. If we applied the metric raw, the 15 `unanswerable` + 10 `stale` (25 of 100 examples, roadmap P8) would penalize precisely the **correct** behavior. **Proposal:** when the abstention detector (P7, `abstained(answer)`, which today is fragile because of `startswith`) says there was a refusal, `citation_recall` and `citation_precision` become **`None`/skipped** (the same treatment as the "open-topic skip" in plan 0.6), never 0. Corollary: citation metrics should only be aggregated over the answers **that attempted to answer**; whether the refusal was appropriate is measured by `abstention_quality` (P8). **Caution:** the quality of the detector (P7) now contaminates citation (a false "did not abstain" on a polite refusal becomes recall 0). We also need to decide what to do with **sentences without facts** (preambles such as "Based on the context…"): ALCE assumes that almost every sentence requires a citation (§2); applying the same to the answers of our `PROMPT_V1` is an assumption to test.

**R5 — Calibrate the judge with real labeled answers, not only with the golden set (SUPPORTS P11; EXTENDS the plan).**
ALCE validates the verifier with **human judgment of outputs from real systems** (100 examples × 3 systems, §6), and not with the datasets' reference answers. Our 100-example golden set has a reviewed question and answer/gold, but **has no "this citation supports this sentence" labels** on generated answers. **Affects:** P11 and the calibration of 0.8. **What we would adopt:** hand-label a sample of **system answers** (claim + citations, with label "fully supports / partially / not") and measure judge × human κ **per metric** (recall and precision separately; in ALCE they came to 0.698 and 0.525). **Cautions:** (i) the `gpt-4o-mini` judge is the same as the generator (P11, self-preference bias), so calibration also needs to compare against a distinct judge; (ii) ALCE uses 3 levels in the human label (fully / partially / not, App. F.3) but only 2 in the NLI; labeling at 3 levels would give us the measure of "partial support", which ALCE's NLI cannot provide (§3.3, App. E), and the LLM judge could be asked to separate it. ALCE's κ is an **order-of-magnitude reference**, not a target, because it is a different verifier, a different corpus and a different type of question.

**R6 — Citation alone is gameable: report it together with correctness and a copy check (SUPPORTS P9; EXTENDS).**
Table 11 shows that citing the top-1 passage gives ~99% recall and precision. In our project, a model that **copies the chunk and cites it** would have high citation and depend only on another metric to be caught. ALCE uses **fluency (MAUVE)** and **correctness**; we do not have MAUVE (and the author himself calls it an unstable sanity check), and the plan's correctness is `token-F1` vs gold answer (P9, a weak metric, roadmap). **Affects:** P9 and the EXP-0 report. **What we would adopt:** (i) never report `citation_recall/precision` in isolation; always next to a correctness/completeness metric (`completeness_judge`, relevance, P9); (ii) as a cheap sanity check of our own (my interpretation, not from the paper), measure the **copy ratio** (fraction of the answer that is a substring of the cited chunks) and length; (iii) treat a drop in correctness with high citation as an alert. **We would not adopt** MAUVE nor claim recall via `text-davinci-003` (discontinued model, LLM-generated sub-claims with 93.3% quality and NLI with 80.0% hit rate: the ceiling of the automatic judge, App. A).

**R7 — What the paper says about retrieval reinforces the reading order of the metrics (SUPPORTS; link to D-2).**
Figure 4: even with the oracle, correctness stays below the retriever's recall@k (§5.3); hence **high recall@k does not guarantee a correct answer**, and a low `citation_recall` can be a generator failure even with the right chunk present. This suggests (my interpretation) conditioning the generation metrics on `hit@k = 1` in a separate report, to separate "retriever failed" from "generator failed". On **D-2** (hit@k ≡ recall@k when the gold has 1 document) and **L-3** (`ranx` as test oracle): the paper is **NEUTRAL**; it uses recall@k, not ranx nor MRR.

**R8 — Definition of relevance by answer coverage (App. G.1) (SUPPORTS the direction of D-1, without deciding).**
ALCE's oracle is built **by content**: it greedily chooses the 5 passages from the top-100 that maximize the answer's recall. It does not use the original dataset's "gold ID", because the datasets have no gold at the granularity of 100 words. This is a precedent for the "expanded gold chunk adjudicated by content" of D-1 (roadmap §3), instead of relying only on the source chunk's ID. **Neutral** as to deciding D-1.

**What we would adopt:**
(1) the recall/precision decomposition per claim, with premise = concatenation of the claim's citations (R1, R2); (2) the rule "recall = 1 as a prerequisite for precision = 1"; (3) human validation on real outputs before trusting the judge, with κ per metric (R5); (4) always reporting citation together with correctness (R6); (5) the idea of an oracle built by content (R8).

**What we would NOT adopt and why:**
(1) MAUVE and claim recall with `text-davinci-003` (discontinued; fluency is not our risk; MAUVE is unstable); (2) the T5-11B NLI as judge (our line is a structured LLM judge, 0.6.3; whether a local NLI would serve as a second judge for P11 is **to verify**, the paper does not test arXiv abstracts); (3) the paper's absolute values (73.6/72.5 etc.) as **targets**: they come from Wikipedia/Sphere with answerable questions and 100-word passages; (4) applying the metric without abstention exemption; (5) treating the κ of 0.698/0.525 as an acceptance target without redoing the measurement in our domain.

## 7. Useful quotations (short verbatim excerpts)

1. "its citation recall is 1 if and only if there is at least one citation" — §3.3 (Citation recall).
2. "this algorithm overlooks the scenario when one citation partially supports the statement" — §3.3 (Citation precision).
3. "Both cases have almost-perfect citation scores" — §3.4 ("top-1 passage" and "first 2 sents" shortcuts).
