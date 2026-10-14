# Ribeiro, Wu, Guestrin & Singh 2020 — Beyond accuracy: behavioral testing of NLP models with CheckList

- **Full reference and link:** Ribeiro, M. T., Wu, T., Guestrin, C., Singh, S. (2020). *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*. ACL 2020 (best paper). arXiv:2005.04118 — <https://aclanthology.org/2020.acl-main.442/>. Code: <https://github.com/marcotcr/checklist>.
- **Reading confidence:** **full except the second half of §4** (the user-study numbers and Table 4). Read with `pdftotext`: abstract, §1–3, Tables 1–3 (the tables came out as a linear dump, so column assignment of the per-model failure rates was **not** reconstructed and none of those per-model figures is used below), §4.1 first half, related work, conclusion. I did not run the code.
- **Why it was added (2026-10-12):** for the P9 decision on how to build the **canaries** of ADR-008 §4 (correct answers turned into their opposite, to test whether a correctness signal can notice a contradiction). CheckList is the established name and method for exactly that kind of test.

> Convention: **(F)** = what the source states or measures; **(I)** = my interpretation or arithmetic.

## 1. Problem the source addresses

(F) Accuracy on a held-out benchmark "is not sufficient": models that "solved" sentiment, duplicate-question and reading-comprehension benchmarks fail on basic linguistic capabilities such as negation, named entities and coreference. Evaluation should test **capabilities** one by one, in the way unit tests do in software, treating the model as a black box.

## 2. Method

- (F) A matrix of **capabilities** (Vocabulary+POS, Taxonomy, Robustness, NER, Fairness, Temporal, **Negation**, Coreference, SRL, Logic) × **test types**.
- (F) **MFT** (Minimum Functionality Test): small, simple labeled examples for one behavior. **INV** (Invariance): a label-preserving perturbation, the prediction must **not** change. **DIR** (Directional Expectation): a perturbation whose effect on the output has a **known direction** (e.g. adding "You are lame." must not make a sentiment score go up).
- (F) INV and DIR "test models on unlabeled data – they test behaviors that do not rely on ground truth labels, but rather on relationships between predictions after perturbations are applied".
- (F) Test cases come from scratch (few, high quality) or from **perturbing existing data** (many); **templates** with fill-in lexicons generate variety ("I {NEGATION} {POS_VERB} the {THING}").
- (F) Each test reports a **failure rate**; there is no pass mark in the paper.

## 3. Main contributions

- (F) The capability × test-type framework, the three test types, the templating tool, and a suite of tests for three tasks; a user study (a commercial team found new bugs in about 5 hours; practitioners with CheckList wrote about twice as many tests and found almost three times as many bugs).

## 4. Key results (only what bears on our decision)

- (F) **Negation is a case where accuracy misleads (§3):** fine-tuned BERT and RoBERTa, 91.5% and 93.9% accurate on the SST-2 validation subset that contains negation (18% of instances), "fail simple negation MFTs"; "performance on the original dataset can be misleading".
- (F) **Which form of negation matters (§3):** failure on commercial sentiment models "is near 100% … when the negation comes at the end of the sentence (e.g. 'I thought the plane would be awful, but it wasn't.'), or with neutral content between the negation and the sentiment-laden word", while the simple "The food is not poor." fails far less (the paper reports 54.2% and 29.4% for two of the models).
- (F) **Diversity of test cases (§2.3):** "A more diverse set of inputs is particularly helpful when a small set of test cases could miss a failure, e.g. if a model works for some forms of negation but not others."
- (F) **Absence of failure is weak evidence (§3, on fairness tests):** "absence of test failure does not imply that these models are fair – just that they are not unfair enough to fail these simple tests."

## 5. Limitations

- **From the authors:** CheckList "cannot be directly used for non-behavioral issues" (labeling errors, annotator biases, data versioning, worst-case security, interpretability); it complements, not replaces, benchmarks.
- **Mine (I):**
  1. **Classification models, 2019–2020.** The tests target sentiment, question-pair and span-extraction models, not the *score of an evaluator* applied to long generated answers.
  2. **No pass mark.** CheckList reports failure rates per test; our "≥ 80% of canaries below the cut line" (ADR-008 C2) has no source here, as before.
  3. **Template-made cases are synthetic** and mostly short sentences; our canaries would be edits of 1–3-sentence answers.
  4. **Failure-rate tables not reconstructed** (see reading confidence).

## 6. Reflections anchored in OUR project

Reference: ADR-008 §4 (canaries) and §5 (C2), roadmap P9 / D-13.

| # | Point / file | Reflection | The source… |
|---|---|---|---|
| 1 | **Canaries are a DIR test** | "Turn a correct answer into its opposite" is a perturbation with a known direction: the correctness score must go **down**. It needs no new human label, only the known direction. This is the method behind C2. | **(i) SUPPORTS** the design |
| 2 | **Vary the form of the contradiction** | A signal can catch one form (an explicit "not") and miss another (a swapped number, an antonym, a negation far from the verb). Canaries made only with "not" would measure one form. | **(i) SUPPORTS** stratifying canaries by perturbation type |
| 3 | **Report per type, not only pooled** | CheckList reports a failure rate per test. A pooled 80% over 10–20 canaries can hide a type with 0%. | **(iii) NEUTRAL**, a reporting practice for C2 |
| 4 | **A passing result is weak** | Passing the canaries shows the signal is "not wrong enough to fail these simple tests", not that it detects real contradictions (which the author's flag measures, ADR-008 §4.2). | **(i) SUPPORTS** keeping both mechanisms |
| 5 | **A paired form of the test** | Because DIR compares predictions before and after the perturbation, the score of the canary can also be compared with the score of the **original** answer ("did it go down?"), without a cut line. | **(iii) NEUTRAL**: a complementary diagnostic, not a replacement for C2 |

**What we would adopt:** the vocabulary (DIR test), canaries from several perturbation types, per-type reporting. **What we would NOT adopt:** the paper's failure-rate figures; any inference that passing the canaries proves the signal works.

## 7. Useful quotations (verbatim from the text read)

- "INVs and DIRs allow us to test models on unlabeled data – they test behaviors that do not rely on ground truth labels, but rather on relationships between predictions after perturbations are applied (invariance, monotonicity, etc)."
- "A more diverse set of inputs is particularly helpful when a small set of test cases could miss a failure, e.g. if a model works for some forms of negation but not others."
- "…absence of test failure does not imply that these models are fair – just that they are not unfair enough to fail these simple tests."
