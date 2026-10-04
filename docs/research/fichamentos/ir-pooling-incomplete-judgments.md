# Buckley & Voorhees 2004 · Buckley et al. 2007 (NIST) · Büttcher et al. 2007 — pooling, incomplete judgments and bias

- **Full references and links**
  1. Buckley, C. & Voorhees, E. M. (2004). *Retrieval Evaluation with Incomplete Information*. SIGIR '04, Sheffield, 25–29 Jul. 2004 (the PDF read carries no volume page numbering — I cite sections/tables/figures).
     PDF read: <https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=150469> · index: <https://www.semanticscholar.org/paper/Retrieval-evaluation-with-incomplete-information-Buckley-Voorhees/6878cdc5b632e018f827a9d1520e7353d8502d25>
  2. Buckley, C., Dimmick, D., Soboroff, I., Voorhees, E. (2007). *Bias and the Limits of Pooling for Large Collections*. NIST document dated "July 17, 2007" (16 pp.; **I did not confirm the publication venue** — treat as a NIST report/preprint). PDF read: <https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=51236> · page: <https://www.nist.gov/publications/bias-and-limits-pooling-large-collections>
  3. Büttcher, S., Clarke, C. L. A., Yeung, P. C. K., Soboroff, I. (2007). *Reliable Information Retrieval Evaluation with Incomplete and Biased Judgements*. SIGIR '07, Amsterdam. DOI <https://dl.acm.org/doi/10.1145/1277741.1277755>. **PDF read:** version hosted by the 1st author, <http://stefan.buettcher.org/papers/buettcher_2007_reliable_evaluation.pdf> (8 pp.; it carries the ACM copyright template text, so it may differ in details from the final version).
- **Reading confidence:**
  - B&V 2004: **full** (8 pp.). Charts (Figs. 1–5) read only through the captions and the text describing them.
  - NIST 2007: **full** text; Table 1 (collection characteristics) and the figures were not extracted — numbers from that table are **not** cited.
  - Büttcher 2007: **full text**; Tables 2, 3, 5, 7 and 8 lost their numbers in the PDF extraction, so I only use values that the text itself cites. Access to the ACM version was blocked (Cloudflare).

> Convention: **(F)** = what the source states; **(I)** = my interpretation. The three works each have a section (A, B, C) with items 1–5 of the format; the **Synthesis** carries items 6 and 7 and the answers to questions (a)–(e) of the request.

---

## A. Buckley & Voorhees 2004 — *Retrieval Evaluation with Incomplete Information*

### 1. Problem the source addresses
(F, §1) The Cranfield paradigm assumes that relevance judgments are **complete** (every document judged for every topic). In large collections built by *pooling* this assumption is only approximately true. The paper investigates the robustness of the measures to **gross** violations of this assumption, with two goals: (i) *incomplete* but unbiased judgments and (ii) *imperfect* judgments (judging documents that have since left the collection, §5). Motivation: TREC collections have ~800 thousand docs and pools of 1,000–2,000 docs/topic; the collection grows and the annotation effort does not.

### 2. Method
(F, §2.1) Measures compared:
- **P(10)** — precision at the top 10 (of every 10 retrieved, how many are relevant). Easy to interpret, but "the only thing that matters is a relevant document entering or leaving the top 10" and it has a large margin of error.
- **R-precision** — precision after R documents retrieved, where R is the number of relevant documents for the topic (the point where precision = recall).
- **MAP** — mean, over the relevant documents, of the precision at the moment each relevant appears (zero if not retrieved). (Measures the quality of the whole ranking; hard to interpret.)
- **bpref** (the paper's proposal, "binary preference") —
  `bpref = (1/R) · Σ_r [ 1 − |n above r| / R ]`, where `r` is a retrieved relevant document, `n` ranges over the **first R *judged* non-relevant documents** retrieved by the system, and R is the number of relevant documents. (In words: for each relevant document, it penalizes how many *explicitly judged* non-relevant documents came before it; documents without judgment are **ignored**.)
- **bpref-10** — variant used in the experiments: denominator `10 + R` and `n` among the first `10 + R` judged non-relevant documents; it guarantees ≥10 document pairs, because plain bpref is "excessively coarse" with 1–2 relevant documents (§2.1).
(F) Rationale for the design: N+R binary judgments generate N×R preferences; 12 naive bpref variants were tested and "average poorly".

Stability (§2.2): system ranking compared by **Kendall τ**; τ>0.9 ≈ equivalent rankings, τ<0.8 = notable differences.

Data (§2.3, Tab. 1): TREC-8 (528 thousand docs, 50 topics, 94.6 relevant/topic, 124 runs from 38 groups), TREC-10 (1.7 M web docs, 50 topics, 67.3, 77 runs/26 groups), TREC-12/Robust (528 thousand, 100 topics, 60.7, 73 runs/16 groups). Runs that retrieved <95% of the maximum were excluded.

Incompleteness experiment (§4): for each topic, random lists of judged relevant and judged non-relevant documents; 16 reduced qrels are created (90, 80, … 5, 4, 3, 2, 1% of the original), with at least 1 relevant and 10 non-relevant per topic. Each smaller qrels is a subset of the larger one. **Sampling is random ⇒ no bias against systems** (F, §4: "the reduced qrels are also unbiased with respect to systems").

Imperfection experiment (§5): full qrels, but the collection is reduced to 90–50% of the documents (random), evaluating only the top 400 retrieved.

### 3. Main contributions
- First systematic demonstration that MAP, P(10) and R-prec are **not robust** to massive incompleteness (§4, Fig. 3).
- Proposal of **bpref**, which evaluates only judged documents.
- Shows that with complete judgments bpref-10 and MAP agree (τ ≥ 0.9) — hence it is a reasonable substitute (§3, Tab. 3).
- Distinguishes *incomplete* from *imperfect* judgments and shows that the latter is less serious (§5).
- Makes explicit the **validity assumption of bpref** (§6): the chance of a document being in the pool must be independent of whether it is relevant or not.

### 4. Key results
- **Tab. 3 (complete judgments, τ against the MAP ranking, TREC-8/10/12):** bpref-10 (1000 docs) 0.934/0.895/0.942; R-prec 0.916/0.851/0.899; P(10) 0.813/0.729/0.721.
- **Tab. 2:** difference δ needed for 95% confidence (TREC-8): MAP 0.040 (9.7% of the best score), P(10) 0.093 (12.9%), R-prec 0.039 (8.9%), bpref-10 0.041 (9.3%). TREC-10 is the noisiest collection (relative δ 15.8–18.2%).
- **Few-relevant regime (§4):** with 5% of the qrels, the fraction of topics with a single relevant document is 26% / 34% / 40% (TREC-8/10/12); with 10%: 8/16/18%; with 1%: 78/92/90%. The author warns that, at this extreme, "absolute number effects" dominate incompleteness.
- **Fig. 2:** MAP, P(10) and R-prec fall monotonically as the qrels shrink; bpref-10 rises only slightly (except for tiny qrels) — consistent scores across topics with different degrees of incompleteness.
- **Fig. 3:** bpref-10's τ stays >0.9 down to the 50% qrels (TREC-10, the noisiest collection) and down to 25% (TREC-8). P(10) drops early and then flattens (unstable: dominated by topics with many relevant documents).
- **§5 (Fig. 5):** no measure is strongly affected by judged documents that left the collection; τ only goes from <0.9 when ~half of the collection disappears.
- **§6 (conclusion):** "Adding additional unjudged documents to a retrieved set can have no effect on that set's bpref score, but can have significant influence on the other measures' scores."

### 5. Limitations
- **From the author (§6):** bpref is only valid if belonging to the pool is independent of relevance; a "sufficiently novel" system may retrieve more non-relevant documents outside the pool. Indirect evidence (only 70 of the 124 TREC-8 runs contributed to the pool and bpref agreed with MAP on all of them) — but "more investigation needs to be done before concluding that bpref can fairly evaluate novel systems that did not contribute to the judgment pool".
- **Mine (I):**
  1. The simulated incompleteness is **random**, therefore it does not test the case that matters most to us (systematic bias against a type of retriever). That case is tested by Büttcher 2007 (section C) and discussed by NIST 2007 (section B).
  2. The success metric is the correlation between **system** rankings (Kendall τ), not the per-query score error; with few queries there may be order swaps that τ does not reveal.
  3. Topics have on average 60–95 relevant documents; this is **not** our regime (1 relevant chunk). The paper itself says that plain bpref is coarse with 1–2 relevant documents and that *all* measures are unstable with few relevant documents.
  4. bpref **requires judged non-relevant documents**; our golden set has only positives (see reflections).

---

## B. Buckley, Dimmick, Soboroff, Voorhees — *Bias and the Limits of Pooling for Large Collections* (NIST, 2007)

### 1. Problem the source addresses
(F, abstract/§1) *Pooling* assumes that, by merging the top-λ of several systems, the relevant documents found form an **unbiased** sample of the true relevant set. With constant λ and a growing collection, the hypothesis of approximately complete judgments fails — and the question is whether the **unbiasedness** hypothesis also fails. Answer: yes, in favor of relevant documents that contain **words from the topic title**.

### 2. Method
(F, §3–§5)
- **`titlestat`** — for a topic T and a set of docs C: `titlestat_T = (1/|T|) Σ_{t∈T} |C_t| / min(|C|, df_t)`, where `t` are title words, `C_t` = docs in C that contain `t`, `df_t` = term frequency in the collection. (Measures "what fraction of the documents contains the title words"; 1.0 = all contain all of them; 0 = none.) `titlestat_rel` = same restricted to the relevant documents; `titlestat_rank` = same restricted to the docs retrieved at a given rank by a set of runs.
- **Controlled comparison:** the same 50 topics in two collections — Disks4&5 (528,542 docs) and AQUAINT (1,033,461 docs; TREC 2005 HARD/Robust; 50 runs in the pool; pool depth = 55 vs ≥100 in Disks4&5).
- **LOU test ("leave out uniques", Zobel):** re-evaluates each run in the pool without the documents that only it contributed.
- **Case study `sab05ror1`:** a *routing* run (queries built from the relevant documents of Disks4&5, **without** using title words), which contributed an anomaly: many unique relevant documents.

### 3. Main contributions
- Empirical demonstration of **pooling bias that depends on collection size** (not on the number of relevant documents per topic).
- The `titlestat` measure to detect it.
- Counting argument: reasonable systems rank documents with title words first; with a large collection, these documents "fill the pool" and push out other types of relevant documents.
- Discussion of alternatives: deeper pools (move-to-front), random/stratified sampling, run diversity, "Engineering the judgment set" + bpref (§6).

### 4. Key results
- **§3.2:** `titlestat_rel` = **0.588** (Disks4&5) vs **0.719** (AQUAINT); higher in AQUAINT in **48 of 50 topics** (p = 6.25·10⁻¹⁰, paired t-test).
- **`sab05ror1` (§3.2):** 405 of the 2,750 documents it put in the pool were **unique** relevant documents — a TREC historical record in unique/contributed ratio. LOU: MAP drops from 0.266 to 0.202 (−23%) without its contributions; the second largest effect was −12%, the others ≤ 8%.
- **§4:** the run's `titlestat` = 0.388 vs 0.600 for the other runs; the unique relevant documents it brought have `titlestat` 0.530 (vs 0.719 for the set): relevant documents without title words **exist** and the pool missed them. Fig. 4: the bias is **not** related to the number of relevant documents of the topic.
- **§5 (Terabyte):** `titlestat_rank` ≈ 0.8 at rank 100 and ≈ 0.68 at rank 1000; `titlestat_rel` 0.889 (2004) and 0.898 (2005). Mean LOU 9.6% (max. 45.5%) in 2004 and 3.9% (max. 17.7%) in 2005 — small because all runs in the pool targeted title terms. In the TREC ad hoc collections "built after TREC-4" ~6% of the judged documents are relevant; in the Terabyte/AQUAINT collections it is much higher.
- **Control (§5):** in the "TREC-8 small web" collection (high `titlestat_rel`, 0.850) an active search for new relevant documents found **<1 new relevant document per topic** — that is, a high `titlestat_rel` alone does not prove bias; the indication is the *difference* between collections with the same topics.
- **§6:** move-to-front pooling recovers 79% of the relevant documents of the depth-50 pool while judging only 48% of the non-relevant ones.
- **§7:** the problem is "conservative" — an affected run is only penalized ("unjudged relevant documents will be counted as non-relevant"); comparisons between systems *within* the pool are not affected; the risk is for **different** systems (e.g. based on semantic concepts), which retrieve relevant documents without the title words.

### 5. Limitations
- **From the author:** the bias is demonstrated only for title words; for Terabyte the evidence is "circumstantial" (§5/§7); no proven solution ("it is currently unknown how to construct a fair sample", §6); binary textual relevance.
- **Mine (I):** (1) the paper is a 2007 report with dominant lexical systems — today the contrast that matters is lexical × dense, which BEIR (section D) measures; (2) `titlestat` measures the presence of words, not semantic equivalence; (3) the effect is studied in collections of millions of documents, whereas our corpus has 843 chunks (a relatively large pool — the factor that the paper identifies as critical, pool size vs collection size, is benign for us).

---

## C. Büttcher, Clarke, Yeung, Soboroff — *Reliable IR Evaluation with Incomplete and Biased Judgements* (SIGIR 2007)

### 1. Problem the source addresses
(F, §1–§2) Pooling is inherently biased against systems that did not contribute to the pool; the bpref/infAP literature studied only **unbiased** incompleteness. The author's example (§2): if only 50% of a system's top-10 was judged, its P@10 cannot exceed 0.5 even if more than half is relevant. The paper studies the **biased** case and tests whether an unbiased judgment set can be **reconstructed** from the biased one.

### 2. Method
(F, §2–§5)
- Measures: P@k; AP; nDCG@20; **bpref** (`1 − Σ_r |n above r| / (|R|·min{|R|,|N|})`, with R the known relevant documents and N the |R| best-ranked known non-relevant documents; judged docs absent from the ranking enter at rank ∞); **bpref-10**; **RankEff** (Grönqvist: `Σ_r |n above r| / (|R|·(|J|−|R|))`, with J = all judged — it uses *all* judged non-relevant documents); infAP (Yilmaz & Aslam); and the new **P@k(j)** = precision among the first k **judged** documents (the `-J` option of trec_eval).
- **Data (§4, Tab. 1):** TREC 2006 Terabyte/GOV2, 50 topics, 42 runs in the pool (11 manual, 31 automatic), depth 50, ~640 judged docs/topic (118 relevant, 522 non-relevant). Evaluation over the top 10,000 retrieved.
- **Bias experiments (§5.2):** (i) *leave-one-group-out*: remove from the qrels the documents that only one group contributed and re-evaluate that group's runs; (ii) *automatic-only*: remove all documents that only manual runs contributed (approximates a new, better and different method).
- **Proposed correction (§3, §5.3):** train a classifier on the judged documents and **predict the relevance of the unjudged ones** that the evaluated system retrieves; it uses two classifiers — one based on KL divergence between the document's language model and that of the relevant documents (threshold such that precision=recall on training) and an SVM (SVMlight, TF-IDF vectors of 10⁶ terms). Then evaluation proceeds normally over the "completed" qrels.
- Error evaluated with Kendall τ and RMS of the score (§4, Eq. 3).

### 3. Main contributions
- First study, along the lines of bpref/AP, of the impact of **bias** (and not only random incompleteness) in the judgments.
- Evidence that **bpref is not immune** to bias: where AP underestimates the system outside the pool, bpref overestimates it.
- The P@k(j) measure and the demonstration that ignoring unjudged documents is a poor approximation.
- A method for **completing judgments with a classifier**.
- Comparison with RankEff, which turns out to be the most robust among the "judged-only" measures.

### 4. Key results
- **§5.1 (random incompleteness):** reproduces the literature — AP, P@k and nDCG@k correlate poorly with the original ranking; bpref better; RankEff slightly better (Fig. 1b).
- **§5.2 leave-one-group-out:** removing a group's unique contributions takes away on average **22 judged documents per topic**; the group's run loses on average **2.1 positions** (up to 12 in P@20). AP changes ~1.5 positions; **bpref ~2 positions**, with 90.5% of the differences statistically significant (paired t, p<0.05). AP tends to **underestimate**, bpref to **overestimate**.
- **P@20(j):** after the removal, the run has on average **3.4 unjudged** in the top-20, of which only **0.3 are relevant** (according to the original qrels); hence ignoring unjudged documents "implicitly assumes that they exhibit the same proportion of relevant and non-relevant documents as the judged documents — an assumption that is simply wrong". P@20(j) overestimates by up to 22 positions.
- **RankEff:** the run's position changes on average only **0.857** (worst case 4).
- **Automatic-only (Tab. 4/5):** judged 31,984 → 23,099 (−28%), relevant 5,893 → 4,373 (−26%); 27.8% of the manual-only pool is relevant vs 18.9% of the automatic-only. Kendall τ <0.9 for all measures; the worst is P@20 with τ = 0.8281; bpref τ = 0.8676 (raises the score of all runs, even more for the manual ones, e.g. "zetaman" 0.398 → 0.475).
- **§5.3:** the classifiers alone have F1 ≈ 0.5 (Tab. 6) — "surprisingly poor" — but in leave-one-out the SVM reaches precision 0.7979 and recall 0.6872 (KLD: 0.6532/0.6642) and reduces the rank displacement to <1 position. In automatic-only, with SVM: τ of P@20 **0.8281 → 0.9512**; of bpref **0.8676 → 0.9164**; the mean displacement of a manual run in P@20 drops from 6.4 to 2.0.

### 5. Limitations
- **From the author (§6):** classifying documents *beyond the pool depth* may induce bias in the opposite direction; there is a risk that systems become optimized for the classifier and not for the human judgment ("only a short-term" solution); they used the inductive SVM (the transductive one would be better with small training).
- **Mine (I):** a single collection (GOV2/TB06) and textual relevance; the method depends on there being hundreds of judged documents per topic (here ~118 relevant) — with 1 relevant per query there is no data to train a classifier per query; tables with numbers missing in my extraction (the conclusions used above are in the text).

---

## Synthesis

**What the three sources say together (F + I):**
1. Treating unjudged as irrelevant gives a **lower bound** on performance (conservative); ignoring the unjudged (bpref, P@k(j), condensed lists) **is not neutral** — it overestimates when the unjudged have a lower proportion of relevant documents than the judged (Büttcher §5.2).
2. The bias comes from **how the pool was formed** (which systems contributed, which lexical characteristics the relevant documents had), not only from the size of the pool: NIST §3–§4.
3. The "clean" way out is to **judge the holes** of the evaluated system (which is what BEIR did for TREC-COVID — see `thakur-2021-beir.md`), and not a statistical correction.
4. All measures are unstable with **few relevant documents per query** (B&V §4); plain bpref is coarse with R = 1–2 (§2.1).

### 6. Reflections anchored in OUR project

| # | Point (roadmap) / file | Reflection | The source… |
|---|---|---|---|
| 1 | **P6**, `precision_at_k` (`evaluators.py`) | In `answerable`/`persona` the gold is **1 chunk** and the top-5 often brings **other chunks of the same paper** (`decisions.md`, 2026-09-27); counting them as "noise" is exactly treating unjudged as irrelevant. NIST §7 and Büttcher §5.2 show that this is a **conservative** bias (floor). This **supports** P6's decision not to use `precision_at_k` as a noise measure with 1-chunk gold (return `None`/replace with paper-level precision + `distinct_papers@k`). | **(i) SUPPORTS** |
| 2 | **"bpref-style tolerance" guardrail** (lesson `golden_dataset_construction_…`, "Design consequence" §1) and glossary | The lesson suggests "bpref-style tolerance" for the extra retrieved material. **Challenge:** (a) bpref requires **judged non-relevant documents** (`min{|R|,|N|}` in the denominator; B&V §2.1); our gold has only positives ⇒ undefined (ranx would divide by zero, see `bassani-2022-ranx.md`); (b) Büttcher §5.2 shows that bpref **overestimates** systems outside the pool (here: a retriever different from the one that generated the questions); (c) with R=1 B&V itself says it is coarse. Correct the lesson/roadmap text: the tolerance we adopt is "do not penalize unjudged", **not** bpref. | **(ii) CHALLENGES** the formulation |
| 3 | **D-1** (one level vs two levels) / `_ranked_and_gold` | The single-chunk gold is an **incomplete set of relevant documents** (siblings from the same paper may answer) ⇒ hit/recall/MRR at chunk level are **floors**. Paper level is a **ceiling** (any chunk of the paper counts). The sources support reporting the [floor, ceiling] range and **judging the holes** instead of choosing one level. Estimate (mine): ~50 examples with chunk gold × top-5, only siblings from the same paper (≤4 per paper) ⇒ ≤ ~200–250 judgments — feasible as an EXP-0 mini-pooling. | **(i) SUPPORTS** keeping 2 levels (as a range) and **CHALLENGES** choosing only one |
| 4 | **P5 / D-2**, `hit_rate` and `recall_at_k` | With |gold|=1 the two coincide; the sources do not discuss this, but measure here over qrels with many relevant documents. B&V §4 warns that with 1 relevant document "absolute number effects" dominate: the metric becomes a binary function per query, with a lot of sampling noise in 40–50 examples. | **(iii) NEUTRAL** |
| 5 | **Lexical-bias risk of the synthetic gold** (lesson L1; golden set ADR) | Questions generated **from the chunk itself** tend to share vocabulary with the chunk. NIST 2007 (bias toward title words) and BEIR §6 are **analogous** evidence (judgments derived from lexical search favor lexical retrievers); **none of the three sources studies synthetic questions** — it is my extrapolation. Practical consequence: when comparing dense retriever × BM25/hybrid (CKPT-1/4), the synthetic gold may favor the lexical one; be careful before accepting/rejecting an experiment on <10–15 p.p. | **(ii) CHALLENGES** the premise that synthetic gold is "neutral by construction" (only by analogy) |
| 6 | **EXP-0 mini-pooling** (lesson, CKPT-0.8) | The design "run retrieval, list unjudged candidates and adjudicate" is **what the literature recommends** (NIST §6/§7, Büttcher conclusion, BEIR §6). We would adopt: also reporting a "hole rate" (fraction of the top-k that is neither gold nor judged; BEIR "Hole@k"). For choosing which retriever supplies candidates: union the top-k of **more than one type** (dense + BM25), since NIST §6: "run diversity is not a complete solution" but it helps detect bias. | **(i) SUPPORTS** |
| 7 | **Dedup by paper** (L-1) | None of the three sources addresses duplicates per document in the ranking; the rule is an engineering decision of ours (already marked as such in roadmap §D). | **(iii) NEUTRAL** |
| 8 | **Gates ≥15%/≥10%** (roadmap §C) and `ranx.compare` | B&V Tab. 2 shows δ for 95% confidence of ~8–18% of the best score (7.7–18.2% depending on measure/collection) with **50–100 topics** and many relevant documents. With ~50 examples of single gold and a binary metric, the needed δ tends to be **larger** (I). It suggests that the 10–15% gates need a confidence interval/paired test (see `bassani-2022-ranx.md` for the options). | **(iii) NEUTRAL** (informs the decision) |

**What we would adopt:** (a) keep "unjudged ≠ irrelevant" and `precision_at_k` conditioned on complete gold (P6); (b) judge the siblings from the same paper in EXP-0 and report strict chunk-level as the floor and paper-level as the ceiling; (c) report a hole rate; (d) cite Büttcher 2007 to justify **not** using P@k(j)/condensed lists.
**What we would NOT adopt:** (a) **bpref** — it requires judged non-relevant documents that we do not have and is sensitive to bias; (b) Büttcher's **classifier** correction — it requires hundreds of judged documents per topic and would introduce a second model to calibrate; (c) magnitude conclusions (τ > 0.9 etc.) as thresholds for our case: they are calibrated on collections of 50–100 topics with dozens of relevant documents.

### Answers to the questions in the request (pooling part)
- **(a)** B&V/NIST/Büttcher do not discuss the hit≡recall equivalence for 1 relevant document; what they say is that P(10) and other "threshold" measures have greater noise. See `thakur-2021-beir.md` and `bassani-2022-ranx.md` for how BEIR and ranx define the metrics.
- **(b)** None of the three studies queries generated from the document; the evidence is analogous (lexical bias of the judgments — NIST §3–§5, `titlestat` table); see reflection 5.
- **(c)** Unjudged documents: **bpref** (ignores; B&V) is only valid with judged non-relevant documents and without pool bias; **condensed lists/P@k(j)** (Büttcher) are unstable and overestimate; **treating as irrelevant** is conservative (floor) and is safe for *comparing systems within the pool*; what the literature actually recommends is to **judge the holes** of the evaluated system. For a 1-chunk gold with possibly relevant siblings: floor (chunk) + ceiling (paper) + adjudicate the siblings.
- **(d)** The three sources operate at a single level (document); they do not discuss passage × document ⇒ **gap**. The reading I offer is (I): two levels, reported separately and as a range, without a combined average.

### 7. Useful quotations
1. Büttcher et al. 2007, §5.2: "Ignoring the unjudged documents in a system's ranking implicitly assumes that they exhibit the same proportion of relevant and non-relevant documents as the judged documents — an assumption that is simply wrong."
2. Buckley et al. 2007 (NIST), abstract: "This phenomenon is wholly dependent on the collection size and does not depend on the number of relevant documents for a given topic."
3. Buckley & Voorhees 2004, §6: "Adding additional unjudged documents to a retrieved set can have no effect on that set's bpref score, but can have significant influence on the other measures' scores."
