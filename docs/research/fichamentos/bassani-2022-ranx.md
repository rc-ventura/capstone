# Bassani 2022 — ranx: a fast Python library for ranking evaluation and comparison

- **Full reference and link:** Bassani, E. (2022). *ranx: A Blazing-Fast Python Library for Ranking Evaluation and Comparison*. ECIR 2022 (LNCS 13186, vol. 2, pp. 259–264), DOI 10.1007/978-3-030-99739-7_30 — <https://link.springer.com/chapter/10.1007/978-3-030-99739-7_30>. Repository: <https://github.com/AmenRa/ranx> · docs: <https://amenra.github.io/ranx/>. Related works by the same author (titles as in the README's BibTeX): *ranx.fuse: A Python Library for Metasearch* (CIKM 2022) and *ranxhub: An Online Repository for Information Retrieval Runs* (SIGIR 2023).
- **Reading confidence:** **PARTIAL — the full paper (6 pages, LNCS) was NOT read.** Springer, ACM and the institutional repository (boa.unimib.it) blocked access (login/Cloudflare); an attempt via OpenAlex pointed to an arXiv of a different paper and was discarded. What I actually read:
  1. the **official 1-page poster** from ECIR 2022 (<https://ecir2022.org/uploads/445.pdf>), including the efficiency table checked visually;
  2. the repository **README** and the docs pages (*metrics*, *stat_tests*, *faq*; these three via an automatic summary by a small model — not literal);
  3. the **source code of v0.3.21** (`ranx/metrics/{hit_rate,recall,precision,reciprocal_rank,bpref,common}.py`, `meta/compare.py`, `statistical_tests/fisher_randomization_test.py`, `setup.py`) and the PyPI metadata.
  **I did not run ranx** (session in read-only mode): the edge-case behavior below comes from *reading the code*, not from testing. Nothing here asserts sections, experiments or limitations of the paper beyond what appears in the poster.

> Convention: **(F)** = what the source (poster/docs/code) states or shows; **(I)** = my interpretation.

## 1. Problem the source addresses
(F, poster) Evaluating rankings in Python required slow or clumsy tools; `pytrec_eval` (the Python interface to trec_eval) is the reference, but the library proposes a *Plug & Play* philosophy: load `qrels` and `runs` (dictionaries, JSON, TREC files, DataFrames), compute several metrics in one line, **compare** several runs with a statistical test and **export LaTeX tables**. Under the hood, it uses **Numba** (JIT compiler) to vectorize and parallelize.

## 2. Method (what the library implements and how it validates)
**Metrics (F, poster + README + code):** Hits, Hit Rate, Precision, Recall, F1, r-Precision, MRR, MAP, nDCG (the nine in the poster); the README/code v0.3.21 adds **Bpref**, RBP and DCG. Each one accepts a "@k" cutoff and `rel_lvl` (minimum relevance level).

Definitions as per code and docs (each with what it measures in parentheses):
- **Hit Rate@k** — 1 if **any** relevant item appears in the top-k, else 0 (did the top-k bring at least one correct item?). Docstring: "it is equivalent to `success` from trec_eval". With empty `qrels` it returns 0.0.
- **Recall@k** — `r_k / R`: relevant items retrieved in the top-k ÷ total relevant items in the qrels (what fraction of the correct items appeared?). Empty `qrels` → 0.0.
- **Precision@k** — `r_k / k` (**the requested k**, or the run size if k=0 — *not* the number actually retrieved). (How many of the k retrieved are correct? If the run has fewer than k items, the denominator remains k.)
- **MRR / Reciprocal Rank@k** — `1/(i+1)` for the first relevant item within the top-k (0-based position `i`), else 0 (how close to the top is the first correct item?).
- **Bpref** — in the docstring: `(1/R) Σ_r [1 − |n above r| / R]` with `n` among the first R **judged** non-relevant items; in the implementation, the denominator is `min(n_rels, n_non_rels)`, so **qrels with no judged non-relevant item produce a division by zero** (reading of the code; not run).
- **nDCG, MAP, F1, r-Precision:** as per the docs (nDCG = DCG/IDCG; MAP = mean of Precision@r over the relevant items).
- **Data structure (F, README/code):** `Run` is a `doc → score` dictionary; ranx **sorts by score** (ties and repeated IDs cannot be represented; zero/negative scores are not filtered — FAQ).

**Correctness validation (F, poster):** the only statement is "All the available metrics were tested against trec_eval [4] for correctness." (README: "The metrics have been tested against TREC Eval for correctness."). The poster does **not** say how (datasets, numerical tolerance, number of cases). (I) For what we would actually need to know, the full paper must be consulted.

**Efficiency (F, poster, synthetic data: 1–10 relevant items/query, lists of 100):** comparison with `pytrec_eval` on NDCG, MAP and MRR over 1,000/10,000/100,000 queries, with 1–8 threads.

**Statistical tests (F, README + `compare.py`):** `compare(qrels, runs, metrics, stat_test="student", n_permutations=1000, max_p=0.01, random_seed=42, threads=0, make_comparable=False)`. Options: **two-sided paired t-test** (default in the code; the docstring wrongly says "Fisher's … is used by default"), **Fisher's randomization test** (approximate: 1,000 permutations; H0: systems identical on the mean of the metric) and **Tukey HSD**. The references given are Smucker et al. (CIKM'07), Carterette and Fuhr. The WebFetch summary of the docs page listed only two tests; README and code list three. I did not find in `compare.py` a multiple-comparison correction for the t/Fisher tests (the only "multiple" form is Tukey) — code reading, not tested.

## 3. Main contributions
- Implementation of ranking metrics in **Numba**, with automatic parallelism.
- Lean API for qrels/runs, evaluation and **comparison with a significance test and LaTeX report** in one call.
- Claim of correctness against trec_eval (poster).
- Ecosystem: `ranx.fuse` (fusion algorithms, CIKM 2022) and `ranxhub` (repository of runs, SIGIR 2023) — **these two other papers are not about validating the metrics** (I); the roadmap §C lists "ECIR 2022, CIKM 2022, SIGIR 2023" as "publications that validate" the lib, which is imprecise: only ECIR deals with evaluation.

## 4. Key results
- **Poster, efficiency table (100,000 queries, ms; speed-up vs `pytrec_eval`):**
  - **MRR:** pytrec 2,935 ms; ranx 1 thread 74 ms (**39.7×**); 8 threads 38 ms (**77.2×**).
  - **MAP:** 2,950 → 210 ms (14.0×) with 1 thread; 69 ms (42.8×) with 8.
  - **NDCG:** 2,991 → 347 ms (8.6×) with 1 thread; 152 ms (19.7×) with 8.
  - With 1,000 queries: ~27–28 ms (pytrec) vs 1–4 ms (ranx).
- **Poster, example Tab. 1:** models 1–5 with MAP@100, MRR@100, NDCG@10 and superscripts of significant difference by Fisher's randomization with p ≤ 0.01 (usage example, not the result of an experiment on real data).
- (I) The poster does not mention the cost of **JIT compilation** on the first call (the functions use `@njit(cache=True)`); for the test case (a few hundred queries), the speed advantage is irrelevant.

## 5. Limitations
- **From the author:** the poster declares no limitations; the README warns that ranx "is not suited for evaluating classifiers" (FAQ: run scores are not class labels).
- **Mine (I):**
  1. The correctness claim is **one sentence** in the poster, with no protocol; "tested against trec_eval" ≠ "validated against pytrec_eval in all edge cases".
  2. **Edge-case semantics differ from ours** and were read in the code: `precision@k` divides by k (our implementation divides by the number of items in the deduplicated ranking); empty `qrels` → 0.0 (our functions return `None`); `Run` does not accept repeated IDs — hence the oracle needs to deduplicate and use strictly decreasing scores (already the case in `tests/test_evaluators_oracle.py`).
  3. **Bpref without judged non-relevant items** divides by zero (code reading): not applicable to our gold, which has only positives.
  4. `compare` does not apply multiple-comparison correction in the t-test/Fisher; docstring/default inconsistent.
  5. **Dependency cost:** `ranx` 0.3.21 requires numpy, **numba≥0.54.1**, pandas, tabulate, tqdm, scipy≥1.8, **ir_datasets**, rich, orjson, lz4, cbor2, **seaborn**, fastparquet. In our tree, `uv.lock` gained **479 lines / 22 new packages**, including `numba 0.68.0`, `llvmlite 0.50.0`, `matplotlib 3.11.2`, `seaborn 0.13.2`, `ir-datasets 0.6.3`, `fastparquet 2026.9.0`, `pillow`, `rich`, `lz4`, `cbor2`, `fonttools`, `kiwisolver`, `contourpy`.
  6. All statements about the full paper remain **unverified** (see "Reading confidence").

## 6. Reflections anchored in OUR project

| # | Point / file | Reflection | The source… |
|---|---|---|---|
| 1 | **L-3** (ranx as test oracle), `tests/test_evaluators_oracle.py`, `pyproject.toml` (`dev` group) | **Validated, with caveats.** (i) ranx implements our four formulas (hit, recall, MRR, precision) with trec_eval semantics (docstring of `hit_rate`: "equivalent to `success`"), so it is a valid oracle; (ii) it is correctly in the **`dev` group** (does not go to runtime), consistent with L-3; (iii) **but** the only evidence of ranx's correctness that I read is one sentence in the poster; the *reference* tool is trec_eval. In terms of rigor, an oracle derived directly from trec_eval (`pytrec_eval`, via `ir-measures`) has the strongest provenance. | **(i) SUPPORTS** partially |
| 2 | **Dependency cost** (`uv.lock` +479 lines, numba/llvmlite/matplotlib/seaborn/ir-datasets) for four trivial formulas | Disproportionate weight **for what we use** (4 metrics, no `compare`). Leaner alternative (PyPI metadata consulted): **`ir-measures` 0.4.3**, whose only mandatory dependency is `pytrec-eval-terrier` (numpy+scipy; C extension over trec_eval; ranx is an optional *extra*). It exposes `Success@k`, `R@k`, `P@k`, `RR` and **`Judged@k`** (fraction of the top-k that has a judgment — useful for the "hole rate"), plus `Bpref`, and the `judged_only` option (docs: "TREC convention distinguishes [Success] from Recall@k"). **I did not test the installation**; I only saw macOS universal2/manylinux wheels on PyPI (compatibility with Python 3.12/3.13 **not verified**). Recommendation (I): keep ranx for now; if the lock/CI becomes heavy, swap to `ir-measures`+`pytrec-eval-terrier` or to hand-written truth tables (no dependency). | **(iii) NEUTRAL** (engineering decision) |
| 3 | **P1–P3**, `precision_at_k` with dedup (L-1) | The already documented divergence (ranx ÷k vs our ÷len(dedup)) is real and **the code confirms it**; the test works around it by using `k = len(ranked)`. This means that **the oracle does not validate the "fewer than k items retrieved" case** — the only one in which the two definitions diverge. Add an explicit unit test for `len(ranked) < k` or fix the semantics in an ADR (P6). | **(iii) NEUTRAL** / gap in the test |
| 4 | **P5 / D-2**, hit ≡ recall with a single gold | On both sides the semantics are the same: `recall.py` and `hit_rate.py` coincide when |qrels|=1. ranx **keeps both metrics** (and the docs distinguish hit rate from recall) — there is no conflict in having both; the decision to report only one per slice is ours. | **(iii) NEUTRAL** |
| 5 | **Roadmap §C / P12**: `ranx.compare` for the ≥15% / ≥10% gates | ranx offers paired t, Fisher and Tukey, and the paired t-test over **per-query** scores is the common procedure (Smucker et al. cited in the docs). But: (i) with ~50 single-gold examples and a binary metric (hit), the per-query scores are 0/1 — the t-test assumes approximate normality; Fisher randomization or bootstrap are more appropriate (I); (ii) no multiple-comparison correction in `compare`; (iii) comparing 3+ arms (EXP-1…8) requires care. We would adopt `compare` with `stat_test="fisher"` only **outside runtime** (EXP report), and we would report a bootstrap CI. | **(iii) NEUTRAL** (useful, but does not validate the gate) |
| 6 | **Bpref** and the lesson's guardrail ("bpref-style tolerance") | Since ranx ships bpref and the roadmap considered using it: the code divides by `min(R, N)` with N = **judged** non-relevant items → without N, undefined. Our gold is positive-only; therefore bpref is **not applicable** (see `ir-pooling-incomplete-judgments.md`). | **(ii) CHALLENGES** the "bpref" option |
| 7 | **Roadmap §C, "validated in publications (ECIR 2022, CIKM 2022, SIGIR 2023)"** | Imprecise: CIKM 2022 = fusion (`ranx.fuse`); SIGIR 2023 = repository of runs (`ranxhub`). Only ECIR 2022 deals with evaluation, and I only read its poster. Rewrite the sentence as "presented at ECIR 2022; claims correctness testing against trec_eval". | **(ii) CHALLENGES** the roadmap's wording |

**What we would adopt:** keep ranx **as a dev-only oracle** (L-3), with the test extended to `len(ranked) < k` and a duplicate-ID case handled explicitly; use `Judged@k` (via `ir-measures`) or our own calculation for the hole rate; use the paired test in the experiment report, never in LangSmith's execution path.
**What we would NOT adopt:** ranx at runtime (per-example overhead + numba); bpref; `compare` with the t-test as the sole gate criterion on binary metrics with n≈50; trusting ranx's validation as if it were equivalent to having tested against trec_eval **ourselves** (we can do it: compare against `pytrec_eval` in an optional test — not run).

### Answers to the questions in the request (ranx part)
- **(a)** Yes, ranx defines `hit_rate` (≡ trec_eval's `success`) and `recall` separately; with |gold|=1 the values coincide (code). It does not recommend a metric; it only implements.
- **(c)** It offers **bpref**, which ignores unjudged items, but requires judged non-relevant items; without them, undefined (code).
- **(e)** **What the paper validates:** per the poster, only "tested against trec_eval" (no protocol) + speed benchmark against `pytrec_eval` (synthetic data). **Metrics:** hit, precision, recall, F1, r-Precision, MRR, MAP, nDCG (+ bpref, RBP, DCG in the current code). **Paired tests:** paired t-test, Fisher randomization, Tukey HSD. **Suitability as an oracle:** suitable (same 4 formulas, derived from trec_eval), with edge-case semantics to control; **alternatives**: `ir-measures` (+`pytrec-eval-terrier`, no numba, with `Success@k`/`Judged@k`) or `pytrec_eval` directly — provenance closer to trec_eval. **Cost:** 22 new packages in the lock (numba, llvmlite, matplotlib, seaborn, ir-datasets…).

## 7. Useful quotations
1. ECIR 2022 poster: "All the available metrics were tested against trec_eval [4] for correctness."
2. Docstring of `ranx/metrics/hit_rate.py`: "Note: it is equivalent to `success` from trec_eval."
3. `ranx/meta/compare.py`: "`stat_test` … Use "fisher" for Fisher's Randomization Test, "student" for Two-sided Paired Student's t-Test, or "Tukey" for Tukey's HSD test. Defaults to "student"."
