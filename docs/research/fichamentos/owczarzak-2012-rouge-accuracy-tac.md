# Owczarzak, Conroy, Dang & Nenkova 2012 — An assessment of the accuracy of automatic evaluation in summarization

- **Full reference and link:** Owczarzak, K., Conroy, J. M., Dang, H. T., Nenkova, A. (2012). *An Assessment of the Accuracy of Automatic Evaluation in Summarization*. Proceedings of the Workshop on Evaluation Metrics and System Comparison for Automatic Summarization (NAACL-HLT 2012), Montréal, pp. 1–9 — <https://aclanthology.org/W12-2601/>.
- **Reading confidence:** **full** (all 9 pages, `pdftotext`; body, Tables 1–3, discussion; Figures 1–5 read through captions and the text that cites them). **Table 3 was reconstructed by me from a garbled text extraction** (columns assigned in reading order; the assignment agrees with the authors' sentences about which variant wins where), so cell-level numbers below carry that caveat.
- **Why it was added (2026-10-11):** the P9 normalization decision (stopword removal or not) had conflicting evidence. This is the best-powered study I found that compares ROUGE **with and without stopword removal** against human judgments. A secondary source (Loukina 2014) had cited it; this note replaces the second-hand reading.

> Convention: **(F)** = what the source states or measures; **(I)** = my interpretation or arithmetic.

## 1. Problem the source addresses

(F) ROUGE is the standard automatic metric for summarization, but "we have a poor understanding of the accuracy of automatic evaluation": how often does a reported ROUGE improvement contradict what a manual evaluation would have concluded? The paper also asks which ROUGE **variant** (n-gram size, stemming, stopwords) is most accurate.

## 2. Method

- (F) **Data:** the NIST TAC 2008–2011 query-focused multi-document news summarization tracks; per year 43–58 automatic summarizers and 8 human summarizers, 44–48 topics, 4 reference summaries per topic (Tab. 1). Manual gold: Pyramid and Responsiveness scores.
- (F) **Variants:** ROUGE-1, ROUGE-2, ROUGE-SU4, all with stemming, each "both with and without removing stopwords" (six variants). Recall is the reported score.
- (F) **Accuracy measure:** for every pair of systems, does ROUGE make the same call as the manual metric (A better, B better, or no significant difference, by a paired Wilcoxon signed-rank test, 95%)? Reported as percentage agreement and as "contradictions" (opposite significant calls). Pairwise accuracy is argued to be more interpretable than correlation.

## 3. Main contributions

- (F) Reference variants: ROUGE-2 with stemming is best at separating automatic systems; ROUGE-1 is best at separating humans from machines. The authors adopt both, "with stemming, without removing stopwords".
- (F) A significance-aware way to compare metrics, and an empirical link between the size of a ROUGE difference and the chance that a human evaluation disagrees.

## 4. Key results

- **Stopwords (Tab. 3, my reconstruction; "ms" = stemmed + stopwords removed, "m" = stemmed only).** Pairwise agreement with Pyramid / Responsiveness, in points, "stopwords removed" minus "kept":
  - automatic-only comparisons: ROUGE-1 Pyr 0 (sig) / +1 (all), Resp −1 / −2; ROUGE-2 Pyr 0 / 0, Resp −1 / −2; ROUGE-SU4 Pyr −1 / +1, Resp −2 / −1.
  - human-vs-automatic: ROUGE-1 Pyr/Resp 0 / −1; ROUGE-2 −3 / −1 (both manual metrics); ROUGE-SU4 +1 / 0.
  - (I) Of the 24 cells, 4 rose, 7 were unchanged and 13 fell; the range is −3 to +1 points.
- (F) Authors' reading: "removing stopwords often decreases performance of ROUGE, although not always."
- (F) Discussion: "ROUGE-2 recall with stemming and stopwords not removed, provides the best agreement with manual evaluations"; about 89% agreement with Pyramid and 83% with Responsiveness when significance is ignored, about 10 points lower when it is required.
- (F) Size of difference: for ROUGE-2 differences below 0.013, "about half of the times" ROUGE would be wrong about which system is better; above 0.04 it was always right on this dataset.

## 5. Limitations

- **From the authors:** news only; the accuracy figures hold "in a situation where many very different systems are compared"; the manual metrics are themselves the model to emulate.
- **Mine (I):**
  1. **System-level pairwise decisions** over 43–58 systems × 44–48 topics, not answer-level agreement. The unit is "which system is better", the same unit as our configuration gates, **not** the unit of C1 (answer by answer).
  2. **Different task and length:** multi-document news summaries of 100 words (query-focused), not 1–3 sentence factual answers.
  3. **Table 3 reconstructed** (see reading confidence). The sign pattern is robust to column errors only if my assignment is right; the authors' text is consistent with it.
  4. **Effects are small** (≤ 3 points) and no interval is given for the stopword contrast itself.

## 6. Reflections anchored in OUR project

Reference: P9 normalization (decision D-14, ADR-010), ADR-008 §2, roadmap §7.

| # | Point / file | Reflection | The source… |
|---|---|---|---|
| 1 | **Stopword removal in the correctness signals** | In the largest study of this contrast (TAC 2008–2011), removing stopwords did **not** help and mostly lowered agreement slightly (13 of 24 cells down, 4 up). It is no argument for adding a stopword list to our normalization. | **(i) SUPPORTS** keeping stopwords (SQuAD-style normalization) |
| 2 | **Configuration gates (≥ 10–15%)** | The unit studied (system-vs-system calls) is the unit of our gates. Differences of a few ROUGE points are noise-prone: below 0.013 about half the calls were wrong. With 65 correctness examples per run our noise is larger than theirs (44–48 topics per system but 4 references each). | **(iii) NEUTRAL**, a caution on small differences (links to P15) |
| 3 | **Conflict with Lin 2004** | Lin (read in full, see below) found stopword removal "usually improved" correlation on multi-document DUC data; the cause is plausibly his tiny system counts (6–18). Owczarzak has 43–58 systems per year over four years. | **(i) SUPPORTS** weighting the larger study |
| 4 | **Variant choice is part of the metric** | The authors fix and name the exact variant (ROUGE-2, stemmed, stopwords kept). Our D-13 logic (fix the definition before the data) is the same discipline. | **(i) SUPPORTS** |

**Contrasting source read in full (no own note):** Lin 2004, *ROUGE: A Package for Automatic Evaluation of Summaries* (<https://aclanthology.org/W04-1013.pdf>). (F) Correlation of **system-average** ROUGE with human coverage on DUC 2001–2003, 6–18 systems per task. Single-document tasks: "correlations were not affected by stemming or removal of stopwords". Multi-document 100-word tasks (Tab. 3): ROUGE-1 DUC 2001 (1 ref) 0.48 (case) → 0.86 (stopwords removed), DUC 2002 0.66 → 0.77, DUC 2003 0.58 → 0.71; ROUGE-2 mixed (DUC 2002 0.83 → 0.80). Conclusion (4): "exclusion of stopwords usually improved correlation". (F) Lin himself warns the multi-document correlations are unstable: about 30 judged summaries per system versus 149–624 in the single-document tasks. (I) It is the only source I found in favour of removing stopwords; its design (few systems, Pearson on averages) is the weakest of the three.

**What we would adopt:** the evidence does not justify a stopword list; keep the SQuAD-style normalization (lowercase, punctuation, `a/an/the`) and treat the stopword question as unresolved-but-not-favouring-removal. **What we would NOT adopt:** any of these numbers as expected agreement for our task; the 0.013 threshold (ROUGE-2 on news).

## 7. Useful quotations (verbatim from the text read)

- "removing stopwords often decreases performance of ROUGE, although not always."
- "we adopt both ROUGE-1 and ROUGE-2 (with stemming, without removing stopwords) as our reference automatic metrics for further comparisons."
- "ROUGE-2 recall with stemming and stopwords not removed, provides the best agreement with manual evaluations."
