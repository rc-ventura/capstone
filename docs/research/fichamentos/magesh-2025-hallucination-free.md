# Magesh et al. 2025 — "Hallucination-Free?" Reliability of AI legal research tools

- **Full reference:** Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C. D., Ho, D. E. (2025). *Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools*. Journal of Empirical Legal Studies (JELS) 2025. Preprint: arXiv:2405.20362 [cs.CL], v1 of 2024-05-30 (footnote 1 of the PDF says "This version of the manuscript (June 3, 2024) is updated to reflect an evaluation of Westlaw’s AI-Assisted Research"). Links: <https://arxiv.org/abs/2405.20362> (PDF: <https://arxiv.org/pdf/2405.20362>); published version: <https://onlinelibrary.wiley.com/doi/abs/10.1111/jels.12413>.
- **Reading confidence:** **full, but only of the arXiv v1 preprint** (24 pages of body + references + Appendices A–C), read from the text extracted with `pdftotext`. The arXiv listing shows only v1. **The published version (JELS 2025) was not read**: the Wiley site returned HTTP 403 (anti-bot block). Hence numbers and wording may differ in the published version. Figure 4 was checked visually (page rendered as an image); Figures 1 and 5 were not: Figure 1 came out only as axis labels in the text extraction, and in Figure 5 the values are not legible (marked "to verify" where this matters). The cause-attribution protocol of Table 6 is not described in the paper. The labels **(a) the paper states** and **(b) my interpretation** are marked in the text.

## 1. Problem the paper addresses

**(a) The paper states:** legal research vendors (LexisNexis, Thomson Reuters/Westlaw, Casetext) advertised RAG as "eliminating" or "avoid[ing]" hallucinations, or delivers legal citations that are "100% hallucination-free" (footnote 2 of §1, with the vendors' literal phrases), without empirical evidence and without defining "hallucination" (§1, §4.3). Since the systems are closed, evaluating these claims is hard (abstract). The authors propose "the first pre-registered empirical evaluation of AI-driven legal research tools" (abstract) and state four contributions: (1) the first evaluation of proprietary legal tools with RAG; (2) a pre-registered dataset of 202 queries; (3) a typology to separate hallucination from accurate response; (4) evidence for lawyers' responsibility to supervise and verify AI outputs (abstract; §1).

Implicit question: *has RAG solved hallucination in legal research?* The paper's answer: no. Hallucination drops relative to GPT-4 without RAG, but persists (§6.1).

**(b) My interpretation:** it is a paper on the **evaluation of closed products with expert human labeling**, not on an automatic metric. What is most useful to our project is not the number (17–33%), but (i) the **operational definition** of hallucination in two dimensions (correctness × groundedness) and (ii) the **labeling protocol with measured agreement**. The number, in turn, is not transferable (see §6, R6).

## 2. Method (step by step, with the exact definitions)

**Systems evaluated (§5.1).** Lexis+ AI (LexisNexis), Ask Practical Law AI (Thomson Reuters) and Westlaw AI-Assisted Research ("AI-AR"). As a reference, **GPT-4 in "closed-book" mode** (no external base; model `gpt-4-turbo-2024-04-09` via API, with a system prompt asking it to cite case law and not to hedge, §5.3). The three products are black boxes: the paper says there is no published technical detail and that it is unknown which retrieval parameters (similarity threshold, top-k) are used (footnote 26).

**Dataset (§5.2, Table 2).** 202 queries written by experts:
| Category | n | % of dataset | What it tests |
|---|---|---|---|
| General legal research (doctrine, bar exam, holdings) | 80 | 39.6% | paradigmatic use |
| Jurisdiction or time-specific (circuit splits, reversed cases, changes in the law) | 70 | 34.7% | variation in jurisdiction and time |
| False premise (user starts from a wrong legal premise) | 22 | 10.9% | bias toward accepting the premise ("contrafactual bias") |
| Factual recall (author of the opinion, year, citation) | 30 | 14.9% | facts that do not require interpretation |

Check: 80+70+22+30 = 202 ✓. Sources: 20 queries from LegalBench (Rule QA) and 20 from BARBRI (bar exam preparation) verbatim; the other 162 written or adapted by hand (§5.2; 202−40 = 162 ✓). **Pre-registration** on the Open Science Foundation on **2024-03-22**, before running the queries (§5.2–5.3; footnote 16: one exception, `changes-in-law-73`, and some minor rewordings, listed in Appendix B.1). Execution: Lexis+ AI, Practical Law and GPT-4 between 22/03 and 2024-04-22; Westlaw AI-AR between 23 and 2024-05-27 (§5.3). A new conversation per query (footnote 17).

**Label definitions (§4, Table 1, Appendix C), each in plain language:**

*Dimension 1 — Correctness (§4.1, Appendix C.2):*
- **Correct** (the answer is factually correct AND relevant to the question). **A partially correct answer counts as Correct** (one that contains no error, but does not cover the whole question, §4.1; Appendix C.1 says that "Partially Correct" was collapsed into "Correct" in the final analysis).
- **Incorrect** (the answer contains **any** factually false statement; Appendix C.2 adds "whether material to the response or not", that is, there is no severity threshold).
- **Refusal** (the model refuses to answer **or gives an irrelevant answer**, Table 1). In Appendix C.1: "Irrelevant/Unhelpful" and "Stock Refusal" were collapsed into Refusal.
- **Special rule for false-premise questions** (footnote 8 of §4.1; Appendix C.2): the desired behavior is to refute the premise. A refusal that **mentions that no pertinent source was found** ("I cannot find any information on this topic") is coded as **Correct**, whereas a standard refusal without that indication ("I cannot provide you with any information on this topic") is **Refusal**. This rule applies **only** to the false-premise category.

*Dimension 2 — Groundedness, only for Correct answers (§4.2, Appendix C.3):*
- **Grounded** (the key factual propositions cite valid and relevant legal sources; indirect support is accepted).
- **Ungrounded** (some material proposition **has no citation**; the statement is not false, just unsourced).
- **Misgrounded** (the proposition **is cited, but the source does not support** the statement, or is inapplicable, e.g. wrong jurisdiction or already-reversed case, §4.2).
- **Fabricated** (the answer cites a source **that does not exist**; appears **only in Appendix C.3**, not in Table 1 nor in §4).
- **Not Applicable** (only for answers without a factual proposition, that is, Irrelevant/Stock Refusal).
- Precedence rule (Appendix C.3): if an answer is both ungrounded and misgrounded, it "is labeled with the most serious offense: Misgrounded". Incorrect answers also receive groundedness N/A (C.1b).

*Higher-level labels (§4.3–4.4):*
- **Hallucination = Incorrect OR Misgrounded** (§4.3: "if a model makes a false statement or falsely asserts that a source supports a statement").
- **Accurate = Correct AND Grounded** (§4.4).
- **Incomplete = Refusal OR Ungrounded** (§4.4). The paper's rationale: an ungrounded answer "does not actually make any false assertions", but fails to provide the information (the authorities) the user needs.

The three labels (Hallucination / Accurate / Incomplete) form a **partition** of the 202 answers per system: each answer falls into exactly one (checked on the bars of Figure 4: Lexis 17+18+65, Westlaw 33+25+42, Practical Law 17+63+20, GPT-4 43+8+49, all sum to 100%).

**A note on the sense of "grounded" (§4.2) — (a) the paper states:** the usage is **legal**, not the computer-science one. In CS, "groundedness" is adherence to the provided documents, "regardless of the relevance or accuracy" of those documents (Agrawal et al., 2023). Here, if the retriever brings a document from the wrong jurisdiction and the model cites it, the answer is **Misgrounded**, "even though this might be a technically “grounded” response in the computer-science sense" (§4.2). This is what separates the paper from RAGAS-style faithfulness.

**Labeling protocol and reliability (§5.4, Appendix C):**
- Each answer was hand-coded by **one of three** law-trained expert labelers (authors), following the rubric in Appendix C. Answers were coded with two values (correctness and groundedness, C.1).
- A **fourth labeler** recoded **48 randomly sampled answers**, stratified by model and task type (with slight oversampling of the Bluebook citation task, "particularly technical"). They **did not talk to the other three nor see the initial labels**; they only read the written documentation of Appendix C (§5.4).
- Result: **Cohen's κ = 0.77** and **85.4% agreement** on the final 3-class label (correct / incomplete / hallucinated), "between the evaluation labeler and the initial labels" (§5.4). Check: 41/48 = 85.4% ✓ (consistent with n = 48).
- The paper rejects letting these legal AI tools "check themselves" (LLM checking LLM, citing Manakul 2023, Zheng 2023 etc.) as "not suitable for this application" (§5.4), because adherence to authority requires manual qualitative evaluation. In §7 it admits that "The trend toward LLM-based evaluations may address the latter obstacle" (the cost bottleneck), but that the legal AI product space "remains quite closed".
- Footnote 18: when including AI-AR, the authors ran **another round of validation of all hallucination labels**, and Practical Law's accuracy rose from 19% to 20% ("within the bounds of inter-rater reliability").
- **The paper reports neither a confidence interval for κ nor per-class κ.**

**Typology of causes (§6.3, Table 6).** The authors look at the retrieved documents (the products show the list, not the exact passages) and compare them with the answer, to propose likely causes: **naive retrieval** (failure to find the most relevant source), **inapplicable authority** (cites a document from the wrong jurisdiction, statute or court, or a reversed one), **sycophancy** (agrees with a false premise) and **reasoning error** (elementary reasoning error with the right text in hand). The categories are **not mutually exclusive** and the paper says that "Because these are closed systems" it cannot identify a single point of failure per hallucination (§6.3). **The attribution procedure (who labeled, decision rule, n per cause) is not described.**

## 3. Main contributions (list)

1. First **pre-registered** empirical evaluation of proprietary legal research tools with RAG (abstract; §1).
2. Dataset of 202 queries in 4 categories, with false premise and jurisdiction/time as explicit categories (§5.2, Table 2).
3. **Correctness × groundedness typology** and the definition hallucination = incorrect OR misgrounded, plus the definitions of *accurate* and *incomplete* (§4).
4. Labeling protocol with a written guide (Appendix C), blind fourth labeler and κ = 0.77 on 48 items (§5.4).
5. Catalog of failure modes with examples and Table 6 of contributing causes (§6.2–6.3), including "inapplicable authority" as a law-specific category that the paper says was not explored in prior literature (§6.3).
6. Argument that an error of a **real citation that does not support the statement** is worse than fabricating a case, because it is subtler and requires reading the source (§4.3).

## 4. Key results (with numbers and where they are)

**(a) The paper states — what "17–33%" exactly means:**

- The abstract says that Lexis+ AI and Westlaw/Practical Law "each hallucinate between 17% and 33% of the time". The number comes from **Figure 4 (left panel)**: "overall percentages of accurate, incomplete, and hallucinated responses". The **denominator is the 202 queries, including refusals**. That is, 17% is the "fraction of the 202 answers that were Incorrect or Misgrounded", **not** the "fraction of the answers given".
- Values from the left panel of Figure 4 (checked visually on the image of page 14):

| System | Accurate | Incomplete | Hallucination | Sources of the numbers |
|---|---|---|---|---|
| Lexis+ AI | 65% | 18% | 17% | Fig. 4; §6.1 (65%, 18%); §1 (65%) |
| Westlaw AI-AR | 42% | 25% | 33% | Fig. 4; §6.1 and footnote 18 (see inconsistencies below) |
| Ask Practical Law AI | 20% | 63% | 17% | Fig. 4; §6.1 says 19% and 62% |
| GPT-4 (no RAG) | 49% | 8% | 43% | Fig. 4 only |

- **In number of queries (my calculation, approximate):** 17% of 202 ≈ 34; 33% of 202 ≈ 67. The paper does **not** report the absolute counts in the text I read (Figure 4 shows only percentages).
- **Rate conditional on answering (Figure 4, right panel):** "the percentage of answers that are hallucinated when a direct response is given". The panel is a bar chart without numeric labels; from the height of the bars I see **approximately 0.20 (Lexis), 0.43 (Westlaw), 0.45 (Practical Law) and 0.47 (GPT-4)** — visual reading, **to verify**. **My calculation:** these values match Hallucination ÷ (Hallucination + Accurate), that is, the "responsive" denominator **excludes every Incomplete answer, not just the refusals** (e.g. Lexis: 17/(17+65) = 20.7%; Westlaw 33/75 = 44.0%; Practical Law 17/37 = 45.9%; GPT-4 43/92 = 46.7%). If this reading is right, "responsive" also excludes correct-but-uncited answers; the paper does not state this explicitly (to verify in the JELS version).
- **The paper's interpretation of the right panel:** Lexis+ AI has a "statistically significantly lower" rate than Westlaw and Practical Law even when conditioned on answering (§6.1); and Westlaw and Practical Law, which answer less than GPT-4, are **not significantly more reliable** in the answers they give (Fig. 4 caption). Error bars = 95% CI.
- **Why Practical Law's 17% does not mean "more reliable":** it refuses/was incomplete in ~63% (the reason the paper gives: its document universe is only Practical Law articles, §6.1). With few answers, the 17% over the total hides 17/37 ≈ 46% over those answered (my calculation, same reading as above).
- **Length confounder (§6.1):** excluding refusals, Westlaw writes on average 350 words (SD 120), versus 219 (SD 114) for Lexis+ AI and 175 (SD 67) for Practical Law. The paper argues that longer answers have "more falsifiable propositions" and therefore more chance of containing at least one hallucination. **(b) Interpretation:** combined with the rule "any false statement, whether material to the response or not" (C.2), the hallucination rate also depends on the **length** of the answer, not just on the quality of the system.
- **By query category (§6.1, Figure 5):** hallucination is "slightly higher" in jurisdiction/time, but "remain high" in general research (e.g. bar exam); accuracy is higher in the false-premise category and lower in the real-use categories. Figure 5 values are **illegible in the extraction: to verify**.
- **Causes (Table 6, proportion of each cause among the hallucinated answers of each system; they do not sum to 1):**

| Cause | Lexis | Westlaw | Pract. Law |
|---|---|---|---|
| Naive retrieval | 0.47 | 0.20 | 0.34 |
| Inapplicable authority | 0.38 | 0.23 | 0.34 |
| Reasoning error | 0.28 | 0.61 | 0.49 |
| Sycophancy | 0.06 | 0.00 | 0.03 |

  (Checked against the extracted text.) **GPT-4 does not appear in Table 6.** Sycophancy is rare because the systems, in general, corrected the false premise (§6.3).
- **Qualitative findings (§6.2, Tables 3–5):** examples of hallucination per system (10 in Westlaw, 6 in Lexis, 4 in Practical Law; footnote 20: proportional to the rate). Modes: understanding the *holding* backwards; confusing a litigant with a court; not respecting the hierarchy of authority; and fabricating a paragraph of a nonexistent law. There is a design finding: Westlaw sometimes **asserted a proposition based on a reversed case without citing it**, which the authors suspect is suppression of citations carrying a KeyCite "red flag" and that "impedes verification of the claims most likely to be false" (§6.2).
- **Example from the definition itself (§6.2/Fig. 2):** Lexis+ AI cites a real case (Reynolds, still valid) to describe Casey, which was reversed (Dobbs 2022). The citation is real, the answer is incorrect, and the citation "serves only to mislead the user about the reliability of its answer" (§4.3). This is what **an "does this citation exist?" check does not catch**.

**Internal inconsistencies of the v1 preprint (checked against the text):**
1. **Westlaw "accurate" 42% × 41%:** §1 says 42%, Fig. 4 shows 42%, §6.1 says "41% and 19%".
2. **Practical Law "accurate" 20% × 19%:** Fig. 4 shows 20%; §6.1 says 19%. Footnote 18 explains that the value in *Figure 1* rose from 19% to 20% after revalidation; therefore the 19% in the text of §6.1 seems to be a leftover from the earlier version, and Figure 1 and Figure 4 would already be at 20%. (I did not check Figure 1; Figure 4 yes.)
3. **Practical Law "incomplete" 62% × 63%:** §6.1 says 62% (§1 says "more than 60%"); Fig. 4 shows 63%. With 17+63+20 = 100, Figure 4 closes; with 17+62+19, it does not.
4. **"Fabricated" (label × usage):** Appendix C.3 defines *Fabricated* as "cites a source that does not exist", but (i) the label does not appear in Table 1 nor in §4, and (ii) §4.3 defines hallucination only as "incorrect or misgrounded", without saying where *Fabricated* fits (presumably under "incorrect" or "misgrounded"; the paper does not say). Moreover, the "Fabrications" subsection of §6.2 uses the term for **content invented from real documents** (e.g. the nonexistent FRBP paragraph, whose origin the authors attribute to a retrieved 1996 bankruptcy case), and not for a nonexistent source. The paper reports **no count of Fabricated answers**.
5. **"Incomplete" has two definitions:** Fig. 1 caption ("fail to either address the user’s query or provide proper citations for factual claims") × §4.4 ("refusals or ungrounded"). In essence they say the same, but the caption does not mention *Refusal*, and Refusal includes "irrelevant answer" (Table 1).
6. **Table 1 × Appendix C.1 on groundedness:** the Table 1 caption says "Groundedness is only applicable for correct responses"; C.1 b says that Irrelevant/Stock Refusal/Incorrect have groundedness N/A. Consistent, but it means that **an Incorrect answer is never classified as Misgrounded/Ungrounded**: the groundedness granularity exists only for Correct answers.
7. **False premise × Refusal:** footnote 8 makes a refusal that mentions not having found sources a Correct answer **only** for false premise; in Table 2 there are 22 queries of this type, but the paper does not report how many answers were recoded by this rule.
8. **"Over 1 in 6" × 17%:** consistent (17% ≈ 1/6); no problem.

**(b) My interpretation of the weight of the evidence:** the evidence is **strong for what it measures** (expert labeling, pre-registration, substantial κ, three products plus a baseline) and **weak for generalization**: n = 202 in a dataset that is adversarial by design, three products, one point in time (the authors say Lexis+ AI changed during the study, §7), no CI for κ.

## 5. Limitations (the author's and those I identify)

**From the author (§7):**
1. Only three products (others, such as Harvey, not evaluated for lack of access).
2. **Point in time**: the answers evolved during the study, especially in Lexis+ AI; test leakage (the dataset was sent to the providers) may mask general problems; the results "do not speak" to the "second generation" version of Lexis (footnote 13).
3. Evaluation limited to chat; more specified generation tasks (memos, contracts) are left out.
4. **n = 202 is small** (compared to Dahl et al., 2024), because of restricted access and manual work.
5. **Groundedness "may exist on a spectrum"**: a citation to a reversed case may still help the lawyer get started; it was coded as misgrounded, but usefulness depends on the use case.
6. **The dataset does not represent the natural distribution of queries**: "Our estimate of the hallucination rate is not meant to be an unbiased estimate of the (unknown) population-level rate of hallucinations in legal AI queries" (§7), but rather to test whether RAG did in fact solve hallucination, as claimed.
7. Measures only hallucination, accuracy and groundedness, not total "value"; the tools may be useful for starting the research.
8. Uncertainty about determinism (footnote 26): temperature/decoding may not be deterministic, and retrieval parameters are unknown.

**Mine (marked as interpretation):**
- **κ = 0.77 comes from n = 48, comparing only the 4th labeler with the initial labels** (one of three). There is no CI, no per-class κ, no agreement among the three initial labelers; with 48 items the uncertainty of κ is large.
- **The definition of "hallucination" is strict and asymmetric:** "any false statement, whether material to the response or not" (C.2) enlarges the hallucination rate; partially correct counts as Correct (§4.1) reduces it. The net effect is not measured.
- **The 17–33% mixes products with very different refusal rates** (Practical Law: ~63% incomplete). Comparing "hallucination rate" across systems without fixing the response rate is misleading; the paper shows the conditioned panel to mitigate this, but the abstract uses the range over the total.
- **The "17–33%" range in the abstract does not include GPT-4 (43%)** nor the 58–82% of Dahl et al. 2024 for general LLMs, which the paper cites as a reference (§2.2).
- **The cause attribution (Table 6) is not reproducible from the text**: no protocol, no n, no κ of the attribution.
- **GPT-4 as a baseline is not comparable in conditions**: it runs via API with a system prompt the author chose and without RAG; Westlaw AI-AR uses GPT-4 underneath (§5.1, "appears to"), but it is not known with what instructions.
- The conclusion that RAG reduces hallucination (§6.1, §9) is a comparison of **products** against a model without RAG, not a controlled experiment on the effect of RAG.
- **The paper does not measure citation *recall*** (how many material statements were left without a source) as a separate rate; *ungrounded* appears only inside "Incomplete" and is summed with Refusal.

## 6. Reflections anchored in OUR project

Reference for points: P1–P11, D-1, D-2, L-3 in `docs/roadmap/ckpt-0.6.1b-metrics-roadmap.md`; planned judges in `docs/roadmap/ckpt-0.6-plan.md` §3.2 (`faithfulness_judge`, `citation_accuracy`, `abstention_quality`, among others) and `docs/plan.md` (metrics table).

**R1 — The planned `citation_accuracy` covers only *misgrounded*; *ungrounded* (and, in part, *fabricated*) are missing. (SUPPORTS the idea of a dedicated metric; CHALLENGES the scope.)**
Plan 0.6 §3.2 describes: a regex extracts `[2609.xxxxvN]` from the answer (code); for each citation, the judge checks whether "the chunk of that `arxiv_id` in the retrieved context supports the cited sentence". `docs/plan.md` says "cited doc exists in the corpus AND supports the claim". Mapping to Magesh's categories (my translation):
| Magesh category | Equivalent in our project | Covered today in the plan? |
|---|---|---|
| Grounded | `[id]` exists, the chunk of `id` supports the sentence | yes (step 2, judge) |
| **Misgrounded** | `[id]` valid, but the chunk does not support the sentence | yes (step 2, judge) |
| **Ungrounded** | factual sentence **without** `[id]` | **no** (the regex only sees citations that are present; a sentence without a citation never becomes an item) |
| **Fabricated** | `[id]` that **does not exist in the corpus** | **partial**: `plan.md` mentions "exists in the corpus", but plan 0.6 §3.2 only says "in the retrieved context", and neither defines a separate output for this |

Observations: (i) Magesh treats *ungrounded* as **incomplete, not hallucination**, because the sentence is not false; for us, it is a failure of the product's promise ("Cited Q&A", `plan.md`) and **needs its own output**, for example a rate of "factual statements with citation" (a citation *recall*); (ii) my interpretation: the case "`[id]` exists in the corpus but **is not in the retrieved chunks**" is a third situation that Magesh does not separate (he defines *Fabricated* only as "nonexistent source"); in our system it indicates the model cited from memory, and would deserve its own label (e.g. `phantom`/`not-retrieved`); (iii) the `[id]` in our system is a **paper** ID, and the retriever returns **chunks** (D-1): "the chunk of that `arxiv_id`" can be more than one (2–5 per paper), so the judge's premise should be the concatenation of the retrieved chunks of that paper, not a single chunk. **Affects:** `citation_accuracy` (to implement in 0.6.3), `docs/plan.md` §3 and `ckpt-0.6-plan.md` §3.2. **What we would adopt:** an output with 4 states per citation/statement (`supported` / `unsupported-but-cited` = misgrounded / `uncited` = ungrounded / `phantom` = fabricated or outside the context) and reporting the rates separately; Magesh's precedence rule (misgrounded > ungrounded when they coexist, C.3) to aggregate per answer.

**R2 — Our `faithfulness_judge` is in the "CS" sense that Magesh distinguishes from his own; it does not capture the "applicability" of the source. (NEUTRAL as to usefulness; CHALLENGES the reading that Magesh supports `faithfulness`.)**
Magesh states explicitly (§4.2) that his "grounded" is **not** the sense of adherence to the provided document; it includes relevance, jurisdiction and current authority. Our `faithfulness_judge` (RAGAS-style, ref-free, against `retrieved_context`) is the CS sense. For the arXiv papers domain, the closest analogy to "inapplicable authority" is the **`stale`** slice (outdated content, `abstention_quality` pre-refresh) and attribution to the right paper. **Affects:** `faithfulness_judge` and the interpretation of `citation_accuracy`: Magesh supports **separating correctness from groundedness** (and separating citation from faithfulness), not faithfulness itself (the paper only cites Agrawal et al. 2023 for the CS sense of "groundedness", §4.2). **Recommendation:** do not attribute Magesh's authority to `faithfulness_judge`; cite him for the correctness × groundedness decomposition.

**R3 — Separate "informative refusal" from "standard refusal" (P7/P8, `abstention_quality`). (SUPPORTS with caution.)**
Magesh codes two refusals **differently**, **only for false premise**: "I cannot find any information on this topic" (says it found no source) = **Correct**; "I cannot provide you with any information on this topic" (standard refusal) = **Refusal** (footnote 8; C.2). For all other categories, any refusal is Refusal (→ *Incomplete*). This is a **category-dependent labeling rule**, not a general taxonomy of refusals. **Affects:** P7 (`abstained(answer)`, fragile `startswith` detection) and P8 (`abstention_summary`: over/under-refusal), as well as the `abstention_quality` judge. Our `unanswerable` (15) and `stale` (10) slices have `should_abstain=True`, where refusal **is** the right answer; in the answerable ones, any refusal is over-refusal. Two transferable lessons (my interpretation): (i) the abstention judge needs to distinguish "refusal **stating the reason** (not found in the papers)" from "generic refusal", and this could become a subfield (`informative_refusal`) for analysis, **without** changing the P/R/F1 computation of P8; (ii) treating false premise as its own category may be useful, but **we have no false-premise slice** (the closest is `unanswerable`); Magesh shows that the desired behavior is to *refute*, not just refuse, which our binary `abstention_quality` does not capture. **What we would adopt:** the `informative_refusal` field as a diagnostic. **We would not adopt:** the rule that an informative refusal counts as "correct" outside the `should_abstain` slices, since in our case this would mask over-refusal.

**R4 — κ = 0.77 on 48 items as a reference for the acceptance of judge×human calibration (P11). (SUPPORTS as a reference, CHALLENGES as a direct target.)**
Roadmap P11 proposes "judge×human κ per judge on the 100 labels; documented acceptance criterion (e.g.: κ ≥ 0.6) before freezing the baseline". Magesh gives a **human×human** comparison number of 0.77 (85.4% agreement), with protocol: written guide, blind 4th labeler, stratified sample. Cautions (my interpretation): (i) **0.77 is human×human on 3 classes in an expert legal task**, not a threshold for an LLM judge; in general a judge should not be required to exceed the human ceiling; (ii) κ **depends on prevalence** and on the number of classes, so the κ of a binary per-criterion judge (faithful/unfaithful, valid/invalid citation, correct/incorrect abstention) is not directly comparable to Magesh's 3-class κ; (iii) **n = 48 without CI**: with few items, the κ of each judge in our project, computed over small subsets (slices of 10–15), will have large uncertainty, so report a CI (bootstrap) or, at the very least, n; (iv) **point to verify in our project:** the "100 labels" mentioned in the plan are, as far as I read, the human review of the **golden set** (question/gold/provenance), not human labels of **answers generated** by the system; judge×human κ requires human labels of the outputs (as Magesh labeled the tools' answers). **What we would adopt:** (1) the protocol (written rubric guide + an independent, blind labeler, with a sample stratified by slice) as the standard for the ~100 labeled outputs; (2) 0.77 as a **ceiling reference** to cite, keeping 0.6 as the roadmap's acceptance floor (user's decision); (3) pre-registration as inspiration: freeze the golden set and the rubrics before EXP-0. **We would not adopt:** κ = 0.77 as a numeric target for the judge.

**R5 — Label with a human expert, not with an LLM (§5.4) × our use of LLM judges. (CHALLENGES the scope, not the decision.)**
Magesh rejects LLM self-verification for his task. Our plan uses LLM judges (`gpt-4o-mini`, same model as the generator, P11) and calibrates against humans. The paper's caveat serves as a reason to **not freeze the baseline before measuring κ per judge** and for P11's argument (separating `JUDGE_MODEL` from `GENERATION_MODEL`). But Magesh's argument is specific ("adherence to authority"), and our domain (arXiv abstracts, short claims) is far less demanding than law (my interpretation), so it does not extrapolate to "an LLM judge is invalid" in our case. **Affects:** P11, `config.py` (`JUDGE_MODEL`), calibration 0.8.

**R6 — The "17–33%" is NOT transferable to our domain. (CHALLENGES any use of the number as an expectation or baseline.)**
Reasons, separating what the paper says from what is my interpretation:
- *(a) The paper says:* the dataset is adversarial by design (bar exam, circuit splits, reversed cases) and the estimate "is not meant to be an unbiased estimate of the (unknown) population-level rate" (§7); it measures three **proprietary products** with unknown retrieval and parameters (footnote 26); at one point in time (March–May 2024); n = 202.
- *(b) My interpretation:*
  1. **Definition of hallucination**: "incorrect or misgrounded", with "any false statement" (C.2). Our `faithfulness_judge` measures adherence to the context, and `citation_accuracy` measures citation support; they do not measure "truth in the world". They are different constructs.
  2. **Denominator includes refusals**: 17% over the 202 answers depends on the response rate (Practical Law 17% with ~63% incomplete). Any rate of ours needs to declare the denominator (total × answered).
  3. **Domain**: law has a hierarchy of authority, jurisdiction and today-vs-yesterday; our corpus is **250 arXiv abstracts** (843 chunks of ~420 characters; `roadmap` §3). Misgrounding by "wrong jurisdiction" does not exist; short answers have fewer falsifiable propositions (the paper shows that length matters, §6.1).
  4. **Our generator** is `gpt-4o-mini` with top-5 chunks, over a closed corpus, not the paper's systems; and the paper's baseline (GPT-4 without RAG) does not exist in our project.
  5. **Labeling**: the paper uses human experts; we will use a calibrated LLM judge.
  Acceptable use of the number: **motivation** ("RAG with citations still hallucinates, even commercially") and a counterexample against "zero hallucination" claims, **never as a target or baseline**.

**R7 — Treatment of the number in project documents: corrections to consider. (NEUTRAL; action on text.)**
(i) `docs/research/rag_failure_modes_review.md` (line 62) writes "**17–34%** of queries"; the paper (abstract, v1) says **17% to 33%**. (ii) The same line says that the tools hallucinated "**despite retrieving real sources**"; the paper attributes part of the hallucinations to **naive retrieval** (Table 6: 47% of Lexis's hallucinations, 20% of Westlaw's, 34% of Practical Law's) and to **inapplicable authority**, not just to generation failure with the right source. The phrase "despite correct retrieval" in the bold label of the excerpt is, therefore, **forced**: the "17–33%" are not "despite correct retrieval". (iii) The roadmap (§ Grounding, item 5) already uses 17%–33% correctly, but cites only the summary; it is worth referencing the denominator (total of 202, with refusals). I **did not edit** those files (outside the scope of this task).

**R8 — Slices of 10–15 items and confidence intervals. (SUPPORTS the practice of reporting CIs.)**
Magesh shows 95% CIs in Figures 4–5 even with n = 202 and categories of 22–80 queries, and the differences between products are only "significant" where the bars do not overlap (§6.1). Our slices (`unanswerable` 15, `stale` 10, `multi-doc` 15, `format` 10, `persona` 10, `answerable` 40) are smaller. My interpretation: any comparison between variants per slice should come with a CI or explicit n. **Affects:** EXP-0 report, P8 (per slice).

**R9 — Neutral items in the roadmap.** D-1 (gold per chunk or paper): the paper does not address retrieval gold granularity (it judges per cited document, without chunk gold). D-2 (hit@k ≡ recall@k): unrelated. L-3 (`ranx` as oracle): unrelated (the paper does not use IR metrics). P9 (token-F1): the paper evaluates correctness by human rubric, not by token overlap; neutral. **NEUTRAL.**

**What we would adopt:** (1) the **correctness × groundedness** decomposition and the grounded / misgrounded / ungrounded / fabricated definitions as the vocabulary of `citation_accuracy` (with the 4 states of R1); (2) the labeling protocol (written rubric, blind labeler, stratified sample) for the outputs that will calibrate the judges (P11); (3) pre-registration/freezing of the golden set and rubric before EXP-0; (4) separating `Accurate / Incomplete / Hallucination` as an exclusive partition, reporting the rate over the total **and** over those answered; (5) CI per slice.
**What we would NOT adopt and why:** (1) the 17–33% as a baseline or target (R6); (2) the rule that "any false statement, whether material to the response or not" counts without adaptation (our generator produces short answers and a materiality threshold must be decided); (3) classifying *ungrounded* as mere "incomplete" without its own rate (R1); (4) the rule "informative refusal = correct" outside `should_abstain` (R3); (5) κ = 0.77 as a numeric target for a judge (R4); (6) the reading that RAG reduces hallucination as a causal effect, since the paper compares products, it does not isolate RAG.

## 7. Useful quotations (short literal excerpts)

1. "A response is considered hallucinated if it is either incorrect or misgrounded." — §4.3.
2. "an ungrounded response does not actually make any false assertions." — §4.4.
3. "Our estimate of the hallucination rate is not meant to be an unbiased estimate of the (unknown) population-level rate of hallucinations in legal AI queries" — §7 (Limitations, sixth limitation).
