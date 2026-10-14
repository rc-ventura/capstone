# ADR-010: Text normalization for the lexical correctness signals — SQuAD-style with punctuation as a space, no stopword list, arXiv ids removed everywhere

**Status**: Accepted
**Date**: 2026-10-11
**Related**: [Roadmap §5 decisions D-14, D-15](../roadmap/ckpt-0.6.1b-metrics-roadmap.md) · [ADR-008](./0008-answer-correctness-signals-and-calibration.md) (the three correctness signals; this record fixes the part ADR-008 left open: what "normalized" means) · Reading notes: [Owczarzak 2012](../research/fichamentos/owczarzak-2012-rouge-accuracy-tac.md) · [Loukina 2014](../research/fichamentos/loukina-2014-rouge-spoken-summaries.md) · [Adlakha 2024](../research/fichamentos/adlakha-2024-correctness-faithfulness-iqa.md) (addendum on the released code)
**Code**: `evaluators.py` (`normalize_text`, `token_f1`, `word_recall`, `token_f1_summary`; slice P9a, implemented 2026-10-12, uncommitted); tests in `tests/test_evaluators.py` (the cases T0–T4 and U1–U5 below are unit tests there). The embedding-cosine evaluator that applies D-15 (`strip_arxiv_ids`, `make_embedding_cosine_summary`, `embedding_cosine_summary`) was built on 2026-10-12 and checked against the real API ([`evidence/0010-embedding-cosine-live-check.py`](./evidence/0010-embedding-cosine-live-check.py), [`.out.txt`](./evidence/0010-embedding-cosine-live-check.out.txt)). **Evidence (reproducible)**: [`evidence/0010-normalization-synthetic-tests.py`](./evidence/0010-normalization-synthetic-tests.py) and its recorded output [`evidence/0010-normalization-synthetic-tests.out.txt`](./evidence/0010-normalization-synthetic-tests.out.txt).

---

> **This record is written for any reader, not only engineers.** The short version: two of our three cheap correctness signals count shared words between the answer and the reference answer. Before counting, the text must be cleaned the same way on both sides — otherwise "component-level" and "component level" count as different words, or a paper's id such as `2609.20804v1` counts as content. This record fixes the cleaning rule **before** any real answer exists, for the same reason ADR-008 fixed its pass marks early: a rule adjusted after seeing the labels would be tuned to the result. Every number below says how it was computed, and the script that produced it is stored next to this record.

## Context

### What "normalization" is and why it must be frozen early

Token-F1 and Recall (ADR-008 §2) compare the **words** of an answer with the words of the reference. A *token* here is simply a word after cleaning (lowercase, punctuation handled). *Recall* is the share of the reference's words that also appear in the answer; *token-F1* also penalizes extra words in the answer. The **cleaning rule is part of the definition of the signal**: two rules give two different numbers for the same answer. ADR-008 left it as "citations removed, text normalized" without saying what that means. D-13 forbids tuning the definition after looking at the labeled answers, so the rule has to be fixed now.

Today the code (`evaluators._tokens`) lowercases and splits with `\w+`; it does not remove citations or articles.

### Two separate questions

1. **Which words to drop or merge?** Case and punctuation are handled by every published variant; the open question was a **stopword list** (very frequent connecting words such as *the, and, of, to, with, not*; the standard NLTK English list has 198 entries) and the exact punctuation rule.
2. **What to do with arXiv ids?** The prompt asks the model to cite `[2609.xxxxx]` for every claim, and our gold answers contain ids too.

### What the code and the data show

- The 15 `multi-doc` gold answers average **54.3 tokens** under today's rule. All 15 contain at least one arXiv id; **12 of 15 contain ids outside brackets** (1.5 per answer on average), used as the sentence subject, e.g. "2609.20804v1 fixes the execution loop…". The model, by contrast, cites inside brackets.
- `PROMPT_V1` (`app.py:22`) shows the example `[2609.12345]` **without** a version, while the context given to the model prints ids **with** it (`[2609.20822v1]`; all 250 indexed abstracts carry a `v` suffix). The model may write either.

## Decision

**D-14 — normalization (applies to token-F1 and Recall).** Apply, in this order, to the answer **and** the reference:

1. Remove arXiv ids wherever they appear (D-15).
2. Lowercase.
3. Replace every character that is not a letter, digit, underscore or whitespace by a **space** (so `component-level` → `component level`, and the em dash `—` disappears).
4. Remove the articles `a`, `an`, `the`.
5. Split on whitespace. **No stopword list, no stemming.** The overlap stays a multiset (a word repeated in the answer counts at most as many times as in the reference), as in the current `_token_f1`.

This is the SQuAD / Adlakha normalization (lowercase, punctuation, `a/an/the`) with **one deliberate difference**: punctuation becomes a space instead of being deleted (called *N1-b* below; the literal version is *N1-a*).

**D-15 — arXiv ids.** Remove every id matching `\b\d{4}\.\d{4,5}(?:v\d+)?\b`, **case-insensitive, anywhere in the text, on both sides**, before any other step. The brackets left behind are punctuation and vanish in step 3. For the embedding cosine (ADR-008 §2), apply **only this step**: embeddings read natural text, so no other cleaning is applied. *(Clarified on 2026-10-12 by D-24: the brackets or parentheses left empty by the removal are dropped too.)*

```python
ARXIV_ID = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", re.I)

def normalize(text: str) -> list[str]:
    text = ARXIV_ID.sub(" ", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\b(a|an|the)\b", " ", text).split()
```

Which paper an answer cites is **not** judged here; that belongs to `citation_accuracy` (roadmap P12). The cases below become unit tests in slice P9a.

## Evidence

> Convention: **(F)** = measured or stated by a source; **(I)** = my interpretation. "Synthetic" tests build each answer mechanically from one of the 15 real `multi-doc` gold answers, one change at a time, so the correct score is known in advance. They show what a rule **can** do to a score; they do **not** show how often `gpt-4o-mini` writes such variants, and they do **not** measure agreement with human labels (that is the ADR-008 calibration, after EXP-0). Every number can be regenerated with the script (fixed seeds 0 and 1).

### 1. What the literature says about a stopword list (read in full)

| Source | Unit compared with humans | Effect of removing stopwords |
|---|---|---|
| Owczarzak et al. 2012 (TAC 2008–2011; 43–58 systems/year; 44–48 topics) | Which system is better, pair by pair | Of 24 cells (3 ROUGE variants × 4 agreement figures × 2 comparison types), 4 rose, 7 stayed equal, 13 fell, range −3 to +1 points; the authors chose "without removing stopwords". The table was reconstructed by me from a garbled extraction (see the note). |
| Loukina et al. 2014 (5,934 responses, ~72 words, abstractive; Spearman **per response**) | The answer itself | ρ 0.573 → 0.585 (ASR) and 0.606 → 0.600 (manual): +0.012 / −0.006, no consistent effect. |
| Lin 2004 (DUC; 6–18 systems; ~30 judged summaries per system in multi-document tasks) | Average per system | In favour of removal in *multi-document news summarization* (ROUGE-1, DUC 2001, one reference: 0.48 → 0.86); the author himself flags these correlations as unstable. **Unrelated to our `multi-doc` slice** (same word, different meaning). |
| Cohan & Goharian 2016 (9 systems, 20 topics) | Per system | Computed with stopwords removed; "similar results" with them kept, no numbers. Nothing new. |
| Adlakha et al. 2024, **released code** (`instruct_qa/evaluation/__init__.py`, branch `main`, read 2026-10-11) | — | Their F1 and Recall use lowercase + punctuation deletion + `a/an/the`, **no stopword list**. This is also the Recall whose agreement with humans (Spearman 60.0) gave ADR-008 its C1 reference level. |

**(I)** The only source in favour of removal is the weakest in design (few systems); the two best-powered ones show no benefit; and the setting closest to ours (the Adlakha code behind the C1 reference) keeps stopwords. There is no published support for adding a list.

### 2. Measurements of ours (script, Part A): how much each rule removes, and the "floor"

**Method.** For each of the 15 gold answers, count tokens under each rule; "removed" is `1 − tokens(rule) / tokens(current)`, averaged over the 15.

| Rule | Mean tokens per gold answer | Removed vs. today |
|---|---|---|
| A — today (`\w+`, lowercase) | 54.3 | 0.0% |
| N1-a — SQuAD literal (punctuation **deleted**) | 48.6 | 10.3% |
| N1-b — punctuation → space | 52.2 | 3.8% |
| N2 — N1-a + NLTK list | 37.9 | 30.1% |

`a/an/the` alone are **3.8%** of tokens (mean over the 15 of articles ÷ tokens). So the 10.3% of N1-a is **not** articles: deleting punctuation without a space merges hyphenated words (`component-level` → `componentlevel`) and glues the id (`2609.20804v1` → `260920804v1`). N1-b removes only the articles. (An earlier draft of this analysis wrongly attributed the 10% to articles.)

**The floor.** Take an "empty" answer made only of the reference's own function words (its tokens that are in the NLTK list or are `a/an/the`, in order). Worked example, gold answer 0: the empty answer is `the and and to are as by with and`; after N1-a it has 8 tokens (`the` is removed), all 8 appear in the reference, which has 53 tokens → **Recall = 8/53 = 0.151**. With the NLTK list every such word is removed and the Recall is **0.000**.

| | Recall of the empty answer |
|---|---|
| Gold answers 0, 1, 2 under N1-a | 0.151, 0.286, 0.262 |
| Mean of the 15, N1-a | 0.221 (min 0.11, max 0.29) |
| Mean of the 15, **adopted rule** (N1-b, ids removed) | **0.222** (min 0.11, max 0.29) |
| Mean of the 15, N1 + NLTK list | 0.000 |

**(I)** A stopword list would remove this floor, but the floor is a *level*, not a ranking problem: every answer of a given reference shares it. Owczarzak and Loukina (section 1) are the evidence on ranking, and they show no gain. A list also deletes `not`, `no`, `nor` (checked on the 198-entry file): `"the method does not improve accuracy"` becomes `method improve accuracy`, erasing the very signal ADR-008's canaries (C2) probe. Word metrics are negation-blind anyway (Adlakha); a list makes it worse.

### 3. Test 1 — N1-a vs. N1-b (script, Test 1)

**Rationale.** If an answer says the same thing as the reference with different punctuation, a good rule must still give 1.0. **Method.** Build the answer from the reference with one reformatting; ids are removed on both sides so only punctuation is tested; average over the 15.

| Reformatting of the reference | Recall N1-a | Recall N1-b | F1 N1-a | F1 N1-b |
|---|---|---|---|---|
| T0 identical copy | 1.000 | 1.000 | 1.000 | 1.000 |
| T1 hyphen → space (`component level`) | **0.946** | 1.000 | **0.922** | 1.000 |
| T2 em dash → comma | **0.989** | 1.000 | 0.994 | 1.000 |
| T3 UPPERCASE + loose punctuation | 1.000 | 1.000 | 1.000 | 1.000 |
| T4 T1 + T2 together | **0.935** | 1.000 | **0.916** | 1.000 |

Worst single reference in T4: N1-a **0.833**, N1-b 1.000. **Why:** under N1-a `component-level` becomes one word, `componentlevel`, that never matches `component` + `level`, and the em dash stays as a token (it is not an ASCII punctuation mark). The reference gold answers contain on average 3.2 hyphens/dashes.

**T5 — does the order of answers change?** **Method.** For each of the 15 references and each keep fraction *f* ∈ {0.2, 0.4, 0.6, 0.8, 1.0}, draw 40 answers by keeping each whitespace-separated word of the reference with probability *f* (seed 0): 15 × 5 × 40 = **3,000** answers. "Respell" additionally turns hyphens into spaces and dashes into commas. Score = Recall; the figure is the Spearman correlation between Recall and *f* (a perfect rule would give close to 1).

| Respell | Mean Recall at *f* = 1.0 (a / b) | Spearman (N1-a / N1-b) |
|---|---|---|
| No | 1.00 / 1.00 | 0.978 / 0.978 |
| Yes | **0.94** / 1.00 | 0.971 / 0.977 |

**(I)** The two rules are identical when spelling agrees. When it does not, N1-a lowers the **level** by up to 6 points on average (17 on one answer) while the **order** barely moves (0.971 vs. 0.977). Order is what C1 measures; level matters for the C2 cut line and the ±10–15% gates, where N1-a adds noise that N1-b does not.

### 4. Test 2 — what to do with ids (script, Test 2; N1-b underneath)

**Rationale.** The model and the gold answers write ids differently (model: always in brackets; gold: often bare). A good rule gives 1.0 to the same content however the id is written, gives 0 to an answer with no content, and does not mix "which paper was cited" into correctness. **Method.** Options: **O1** remove ids anywhere, both sides; **O2** remove only `[id]` in brackets, both sides; **O3** keep ids. Each case rewrites **only** how ids appear in the reference; cells are Recall | F1 averaged over the 15.

| Case | O1 | O2 | O3 |
|---|---|---|---|
| U1 same content, bare id written `[id]` (model style) | **1.00 \| 1.00** | 0.94 \| 0.97 | 1.00 \| 1.00 |
| U2 same content, citation without version `[2609.20804]` | **1.00 \| 1.00** | 0.94 \| 0.97 | 0.94 \| 0.94 |
| U3 same content, cites the **wrong** paper | 1.00 \| 1.00 | 0.94 \| 0.97 | 0.94 \| 0.94 |
| U4 same content, no citation at all | **1.00 \| 1.00** | 0.94 \| 0.97 | 0.87 \| 0.93 |
| U5 **only** ids, no content | **0.00 \| 0.00** | 0.00 \| 0.00 | **0.13 \| 0.22** |

- **O2 has a structural ceiling.** The bare ids of the reference have no counterpart once the answer's bracketed ids are stripped: a *perfect* model-style answer scores at most **0.945** on average (min 0.909, max 1.000).
- **O3 mixes two jobs.** Omitting the citation (U4, −13 points) costs more than citing the wrong paper (U3, −6), and an answer made only of ids earns 0.13 Recall / 0.22 F1 (gameable).
- **O1 is the only option with 1.0 / 0.0 where it should be.** Its blindness to U3 is by design (P12).
- With N1-a underneath the pattern is the same (U5 on O3: 0.07).

### 5. A correction recorded for transparency

The first run of T3 showed F1 0.93 because the test's own transformation (`"." → " ."`) split the dot inside ids (`2609 .20804V1`). I first blamed the uppercase `V1`. The transformation was fixed (only sentence-ending periods are spaced) and the script re-run; the figures above are from the fixed script. The regex is case-insensitive anyway.

## Alternatives considered

### Alternative A: N1-a — SQuAD literal (delete punctuation)

Identical to the Adlakha code. **Why not chosen:** merges hyphenated words and leaves `—` as a token (Test 1: up to −6.5 points of level, −17 on one answer, no order benefit). **Advantage not leveraged:** zero deviation from the code behind the C1 reference level; no published study tests the difference, so this is an engineering judgment without a source.

### Alternative B: N2 — N1 + stopword list (NLTK)

**Why not chosen:** no evidence of gain (section 1), it deletes `not`/`no`/`nor`, adds a list to maintain and version, and breaks the link with the Adlakha reference level. **Advantage not leveraged:** the empty-answer floor drops to 0.

### Alternative C: N3 — N1 plus a second, stopword-filtered signal reported from day one

**Why not chosen:** one more candidate signal against ~100–150 labels whose interval is already wide (a measured 0.60 could be 0.47–0.70, ADR-008) raises the chance that one looks good by luck and would call for a multiple-comparison correction. With sections 1–2 there is no hypothesis worth that cost.

### Alternative D: O2 — remove only bracketed ids

**Why not chosen:** a ceiling of 0.945 for a perfect answer (Test 2).

### Alternative E: O3 — keep ids

**Why not chosen:** mixes citation behavior into correctness (omitting a citation costs 13 points, citing the wrong paper only 6) and is gameable by an ids-only answer.

### Not adopted now: counting each word once ("types") instead of with repetitions

Loukina 2014 found +0.049 (ASR) / +0.028 (manual) Spearman from counting types in spoken, non-native responses scored for proficiency. It is **not shown for our regime** and we have no labels yet. It can be declared as a candidate **before** the calibration, never chosen after seeing the labels.

## Consequences

### Accepted

- One rule, frozen before the data, shared by token-F1 and Recall; the cosine only loses ids.
- The metric no longer depends on how an id or a hyphen happens to be written.
- A perfect model-style answer can reach 1.0 (no structural ceiling).
- The test cases double as regression tests for P9a.

### Trade-offs

- **The Recall of an empty answer is about 0.22** (range 0.11–0.29), because stopwords stay. Recall is a diagnostic until calibrated (ADR-008), but the level should not be read as "quality".
- **One detail departs from the Adlakha code** (punctuation as a space). No study evaluates it; the evidence is our synthetic Test 1.
- **The synthetic answers are mechanical.** How often `gpt-4o-mini` respells hyphens or drops a version is unknown until EXP-0; the calibration labels are the real test.
- **Blind to the cited paper** by design; P12 covers it. For a "compare paper A and B" question, an answer naming only one paper differs in content, not in ids, and is scored on content.
- **The id pattern is `YYMM.NNNN[N]` with an optional version** (the corpus is all `2609.*`). Old-style ids (`hep-th/9901001`) would not match. Ids written as `arXiv:2609.20822` or `(2609.20822)` are removed (the pattern is position-independent); the `arXiv:` word itself stays as an ordinary token.
- **English only**, like the rest of the evaluators.
- **For the cosine, a residual of about 0.004 remains** when an answer carries no citation at all and was written naturally (the space that used to precede the citation is gone: cosine 0.996 on average, worst 0.994, against its gold). It is negligible next to the spread of correct paraphrases (0.59–0.84 in the 2026-10-10 local test) and no further cleaning is applied for it (see D-24).
- **The embedding cache on disk is trusted as long as the model behind the pinned name does not change.** The key contains the model name, not a model version; if the provider ever changed the vectors behind `text-embedding-3-small`, the cache file must be deleted and the calibration redone (D-25).
- Word metrics stay negation-blind whatever the normalization (Adlakha); that is ADR-008's C2.

### Conditions that invalidate this decision

This decision should be **revisited** if:

1. The calibration shows (with the pass marks of D-13 unchanged) that the empty-answer floor distorts the labels' order — e.g. wrong-but-on-topic answers outscore correct short ones.
2. Real answers show that a variant we neutralized (hyphens, dashes) is rare, or that another surface variant is frequent (numbers written as words, British/American spelling) — then add a test and, if justified, a rule, **declared before** the labeled data.
3. The reference answers are regenerated or the id format changes (ADR-008 condition 4).
4. A study appears that measures stopword removal on answers and references like ours.
5. The generator changes in a way that changes citation style (e.g., no brackets).

### Migration path when needed

1. **Add a declared candidate** (types instead of tokens; a stopword-filtered sibling) *before* the labeling session, with its pass marks fixed beforehand.
2. **Tighten the id rule** (more id shapes) — pure regex change, no data change.
3. **Recalibrate** after any change of rule (the cut line of C2 depends on the score scale).

## Amendment (2026-10-12): two decisions on the embedding cosine

Approved by the author after the live check of the cosine evaluator. They concern only the embedding signal; the cleaning rule of the word metrics (D-14) and D-15 for the word metrics are unchanged.

**D-24 — Cleaning before embedding (clarifies D-15).** Besides the arXiv ids, the cosine drops the brackets or parentheses that held **only** ids (left empty, or with only commas/semicolons, after the removal) and collapses runs of spaces. Case, punctuation, hyphens and line breaks are left alone. Reason: with ids-only removal the brackets that held an id stayed (`[ ]`), and the real model scored a **same-content** answer at a cosine of **0.976 on average (worst 0.931)** against its gold when one text wrote the id `[id]` and the other bare, and 0.990 when the answer had no citation, instead of 1.0, so the cosine depended on how an id was written, which D-15 exists to avoid. With the clarification both cases give **1.0000** ([live check](./evidence/0010-embedding-cosine-live-check.py), parts 2–3; unit tests pin it). Cost: one more cleaning step than "ids only".

**D-25 — The embedding cache is also kept on disk.** `CachedEmbedder(…, cache_path=…)` persists every vector in `resources/answer_embeddings.jsonl` (git-ignored), an append-only JSON-lines file: key = sha256(model + text) so a different model never reuses a vector; the vector as base64 of its float32 bytes (about 8 KB per text; about 2.4 MB for the 300 texts of a calibration); **only the hash of the text is stored, never the text**; loaded lazily; a corrupt or truncated line is skipped; delete the file to invalidate it. The in-memory layer stays. Reason: the calibration is run once and then frozen; with the vectors on disk the ≈ 200 answers and 20 canaries can be re-scored offline while the cut lines are set, and the exact vectors used can be reproduced without calling the API again. Caching an embedding is standard practice because it is a deterministic function of (model, text); the corpus vectors of the retrieval index already live in `resources/*.parquet`. This is **not** a semantic cache (reusing an old answer for a similar question), which is not proposed. Checked live: a new process reading the file made **0** API requests and returned an identical score. Cost: an extra file to manage and the invalidation rule above.

## References

- Roadmap: [`ckpt-0.6.1b-metrics-roadmap.md`](../roadmap/ckpt-0.6.1b-metrics-roadmap.md) — §5 decisions D-14, D-15; §7 P9
- [ADR-008](./0008-answer-correctness-signals-and-calibration.md) — the signals, the pass marks and the calibration this rule feeds
- Reading notes (read in full): [`owczarzak-2012-rouge-accuracy-tac.md`](../research/fichamentos/owczarzak-2012-rouge-accuracy-tac.md) (NAACL-HLT workshop 2012; <https://aclanthology.org/W12-2601/>) · [`loukina-2014-rouge-spoken-summaries.md`](../research/fichamentos/loukina-2014-rouge-spoken-summaries.md) (<https://aclanthology.org/W14-1809/>) · [`adlakha-2024-correctness-faithfulness-iqa.md`](../research/fichamentos/adlakha-2024-correctness-faithfulness-iqa.md) (addendum on the code). Also read in full without own notes: Lin 2004 (<https://aclanthology.org/W04-1013.pdf>) and Cohan & Goharian 2016 (<https://arxiv.org/abs/1604.00400>), summarized in section 1
- SQuAD normalization: official evaluation script v1.1 (copy read: <https://github.com/aswalin/squad-1/blob/master/evaluate-v1.1.py>); Adlakha code: <https://github.com/McGill-NLP/instruct-qa/blob/main/instruct_qa/evaluation/__init__.py>
- NLTK English stopword list: <https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip> (198 entries in the file used)
- Evidence script and recorded output: [`evidence/0010-normalization-synthetic-tests.py`](./evidence/0010-normalization-synthetic-tests.py), [`.out.txt`](./evidence/0010-normalization-synthetic-tests.out.txt)
