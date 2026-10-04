# Zhou et al. 2023 — IFEval: instruction-following evaluation with "verifiable instructions"

- **Full reference:** Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S., Luan, Y., Zhou, D., Hou, L. (Google; Zhou also Yale) (2023). *Instruction-Following Evaluation for Large Language Models*. arXiv:2311.07911v1 [cs.CL], 2023-11-14 (PDF dated 2023-11-15). Link: <https://arxiv.org/abs/2311.07911> (PDF: <https://arxiv.org/pdf/2311.07911>). Dataset: <https://huggingface.co/datasets/google/IFEval>. Code: <https://github.com/google-research/google-research/tree/master/instruction_following_eval>.
- **Reading confidence:** **full for the paper (v1)**: text extracted from the PDF with `pdftotext` and read from the abstract to the appendix (Tables 1–3; Figure 1 and Figure 2 only through caption/text, without reading bar values; the appendix with the list of prompts was only skimmed, not read prompt by prompt). The paper is short (~7 pages + appendix). **I did not read later versions** (if any). **The `instruction_id_list` + `kwargs` schema is NOT in the paper:** it comes from the dataset card/data on HuggingFace; I checked this point in the card (`README.md`) and in the dataset's rows API on 2026-10-03 (541 rows, fields `key`, `prompt`, `instruction_id_list`, `kwargs`). **I did not read the source code of the checkers** in Google's repository, so any statement about how each checker is implemented is marked "to verify". Convention: **[Paper]** = stated in the text; **[HF/code]** = comes from the dataset/repository, outside the paper; **[My reading]** = my interpretation.

## 1. Problem the paper addresses

**[Paper]** Following natural-language instructions is a core capability of LLMs, but its evaluation "is not standardized" (abstract). The authors list three families of evaluation, each with a flaw (§1): (1) **human evaluation** is expensive, slow and poorly reproducible; (2) **model-based evaluation** (LLM-judge) depends on the correctness of the evaluator, which "is not guaranteed"; (3) existing quantitative benchmarks. Instructions such as "write with a funny tone" have a "greatly unclear" criterion (§1). The proposal: focus on **"verifiable instructions"**, defined as "instructions amenable to objective verification of compliance" (§1), for example "write 450 to 500 words", "your entire output should be in JSON output", "include a title, and put it into two square brackets such as [[ title ]]". This way the evaluation would be fully automatic and objective.

**[Paper]** The authors' own caveat (§1): "very few instructions are 100% verifiable objectively and automatically" (few instructions are 100% verifiable); there will always be edge cases. This is why the "loose" mode exists (§2.2).

**[My reading]** This is a **benchmark** paper, not a method paper: it does not test whether the verifier agrees with humans. The argument that verifying by code is "unbiased" (§4) is a claim, not a measured result.

## 2. Method (step by step)

### 2.1 What the paper states

**Verifiable instructions (Table 1).** 25 types, in 9 groups (I checked the count line by line: Keywords 4; Language 1; Length Constraints 4; Detectable Content 2; Detectable Format 6; Combination 2; Change Cases 3; Start/End 2; Punctuation 1 = 25). Examples: *Number Words* ("Answer with at least / around / at most {N} words."), *Number Bullets* ("Your answer must contain exactly {N} bullet points."), *JSON Format* ("Entire output should be wrapped in JSON format."), *No Commas*, *End Checker* ("Finish your response with this exact phrase {end phrase}."), *Title* (`<<title>>`). The authors say they chose these types because they are "either easy to verify or common in real-world applications" and that "The list can be expanded trivially", citing XML as an example of a future addition (Table 1 caption). Each instruction is "atomic" and verifiable by "a simple, interpretable, and deterministic program" (§1).

**Prompts (§1, §2.1).** **541 prompts**, each with 1 to 3 verifiable instructions. (The **abstract says "around 500"**; §1 says 541; the dataset on HF has 541 rows. I use 541.) Each instruction has **parameter** variants (450–500 vs 350–400 words) and **wording** variants (§1). Synthesis in 4 steps (§2.1): (1) generate base prompts with 1–3 randomly drawn instructions appended at the end; (2) *few-shot prompting* to identify and remove illogical prompts (conflicting instructions, e.g., 5 paragraphs vs fewer than 20 words); (3) another *few-shot* step to rewrite and increase wording diversity; (4) manual review and editing, one by one. Stated reason: avoid conflicts between instructions and avoid not knowing whether the model follows the instruction or only a particular wording of it (§2.1).

**Strict verification (§2.2, Eq. 1).** `is_followed(resp, inst)` returns true/false. The accuracy computed with this function is called **"strict"**.

**Loose verification (§2.2, Eq. 2).** It exists because the strict one produces **false negatives**. The paper's example: instruction "end your email with: P.S. I do like the cake"; the model writes "P.S. **I do like the cake**" (with markdown bold); exact string matching fails it. The loose version is `is_followed_loose = Any(is_followed(transform_t(resp), inst))`: the response passes if **any** of the transformations, applied before the check, makes the instruction followed. There are 3 base transformations: (1) remove markdown font markers, "especially `*` and `**`"; (2) remove the **first line** (skip introductions such as "Sure, here it is:"); (3) remove the **last line** (skip sign-offs such as "Hope it helps."). Combining "every two and all three", plus the identity transformation, gives **8 transformations in total** (3 single + 3 pairs + 1 triple + identity = 8). Cost, stated by the authors: the loose one "is likely to introduce **false positives**" (e.g., a response that violates the word count passes if removing the 1st line brings it within the limit); that is why loose is treated as a "complement" to the original criterion, not a substitute.

**The 4 metrics (§3), in plain language:**
- **Prompt-level strict** (accuracy per prompt, strict): *percentage of prompts in which ALL of the prompt's instructions were followed*, checked exactly. It is the hardest: a prompt with 3 instructions and 1 failure counts as a whole error.
- **Inst-level strict** (accuracy per instruction, strict): *percentage of individual instructions followed*, checked exactly. A prompt with 3 instructions and 1 failure counts as 2 hits out of 3.
- **Prompt-level loose**: same as the first, but the check of each instruction accepts any of the 8 transformations.
- **Inst-level loose**: same as the second, with the loose check.

**Models evaluated (§3).** GPT-4 (responses collected in November/2023) and PaLM 2 Small (August/2023), via API. The authors warn that **the two models are not directly comparable** because of the large difference in size (Table 3 caption).

### 2.2 What comes from the dataset/code on HuggingFace (OUTSIDE the paper)

**[HF/code]** Each row of the dataset (541) has: `key` (integer), `prompt` (text), **`instruction_id_list`** (list of ids such as `punctuation:no_comma`, `detectable_format:number_highlighted_sections`, `length_constraints:number_words`) and **`kwargs`** (a list, one entry per instruction, with the parameters). I checked in the card and in the API: the `kwargs` schema is a **"union" record with all possible fields** (`num_words`, `relation`, `num_bullets`, `num_sections`, `keywords`, `forbidden_words`, `end_phrase`, `language`, `letter`, `let_frequency`, `capital_relation`, `postscript_marker`, `first_word`, `nth_paragraph`, `prompt_to_repeat`, etc.), in which **most come as `None`** in each instruction (an instruction's entry only fills in the fields it uses). The code-based check (one checker per id, which reads the `kwargs`) is in Google's repository; **how each checker is implemented: to verify** (I did not read the code). The paper **does not mention** `instruction_id` or `kwargs` at any point.

## 3. Main contributions

1. Operational definition of "verifiable instruction" and a list of 25 types (Table 1, §1–§2).
2. A set of 541 prompts with 1–3 instructions each, with a synthesis method that tries to avoid conflict and single wording (§2.1).
3. A pair of criteria (strict and loose) and four metrics (prompt/inst × strict/loose) (§2.2, §3).
4. Baselines for two models (Table 3) and a breakdown by instruction category (Figure 2, chart only).
5. Open code and data (abstract, §1).

## 4. Key results

**[Paper] Table 3 (checked against the PDF text):**

| Model | Prompt strict (%) | Inst strict (%) | Prompt loose (%) | Inst loose (%) |
|---|---|---|---|---|
| GPT-4 | 76.89 | 83.57 | 79.30 | 85.37 |
| PaLM 2 S | 43.07 | 55.76 | 46.95 | 59.11 |

- **Inst-level ≥ prompt-level** in all cases, as expected from the definition (a prompt only counts if all instructions pass).
- **Loose ≥ strict** in all cases. **[My calculation, from Table 3]** The gain of loose over strict: GPT-4 +2.41 p.p. (prompt) and +1.80 p.p. (inst); PaLM 2 S +3.88 p.p. (prompt) and +3.35 p.p. (inst). The paper does **not** break down how much of this gain is recovered false negatives and how much is introduced false positives; there is **no human validation** of the loose criterion.
- Figure 2 shows strict accuracy per instruction category for the two models; I read only the caption (per-bar values **not** read, to verify if cited).
- The paper **does not report** confidence intervals, number of runs, temperature/decoding, nor the total number of individual instructions used at the "inst" level (to verify in the code/dataset).
- **[My calculation]** With 541 prompts, the 95% Wilson CI of a 76.89% hit rate (~416/541) is ~[73.2%; 80.2%]; of 43.07% (~233/541) it is ~[39.0%; 47.3%]. It is narrow enough to separate GPT-4 from PaLM, but the paper does not present it.

## 5. Limitations

**From the authors:**
- "very few instructions are 100% verifiable objectively and automatically" (§1); loose tries to mitigate false negatives, at the cost of false positives (§2.2).
- The implementation "can be improved across many fronts" (§4): increase the diversity and number of instructions; extend to multimodal; get closer to real applications (§4). The list of 25 is admitted to be incomplete (Table 1 caption).
- The two models are not directly comparable (Table 3 caption).

**Mine (interpretation):**
- Only **2 models**, each run once; no CI, no repetitions, no sensitivity analysis to wording.
- **Verifying ≠ agreeing with a human.** The paper does not measure whether the checker (strict or loose) coincides with a human judgment in edge cases; the premise "verifiable = correct" is not tested.
- The **8 transformations are blind**: they remove the first/last line *always*, without knowing whether that line is an "introduction" or content. The paper itself acknowledges the false positive (§2.2) but does not quantify it.
- The instructions are **synthetic and stylistic** (commas, capital letters, number of words); the only structured format is "JSON Format" (Table 1), described as "entire output should be wrapped in JSON format", with no key/schema check as far as the text says (to verify in the code).
- Free text in English; nothing about RAG, grounding in sources or citations.

## 6. Reflections anchored in OUR project

Reference: `docs/roadmap/ckpt-0.6.1b-metrics-roadmap.md` (P10, table of sources). Current state: `evaluators.py::format_validator(question, answer)` dispatches by substring of the question ("markdown table", "JSON array/object", "as CSV", "As YAML", "bullet point", "sentences"+"citation") and extracts columns/keys/counts by **regex over the question text** (`_check_markdown_table`, `_check_json`, `_check_csv`, `_check_yaml`, `_check_bullets`, `_check_sentences_citation`). There is only a strict criterion. An undetected format returns `{"score": 0, "reason": ...}` without `key`. The `format` slice has **N=10** (`build_golden_dataset.py::_FORMAT_EXAMPLES`), with 1–2 requirements per question (e.g., "exactly 3 bullets of at most 12 words").

**R1 — `instruction_id` + `kwargs` as a model for `metadata.format_spec` (P10). SUPPORTS.**
P10 proposes recording a structured spec (`{"type":"json_object","keys":[...]}`) instead of regex on the question. IFEval's pattern is exactly "**type + parameters**" verifiable by a deterministic checker, and the paper (§1) justifies the choice for the same reason as the roadmap: objectivity and automation. **Note: the `instruction_id_list` + `kwargs` format belongs to the dataset on HF, not to the paper [HF/code]**; the paper only states the principle "atomic instruction verifiable by a simple, deterministic program" (§1). **Affects:** `build_golden_dataset.format_examples` (recording the spec) and `evaluators.format_validator` (reading the spec, not the question).
**What we would adopt:** (a) a **list** of atomic requirements per example (like `instruction_id_list`), each with its checker registered by id (e.g., `table:columns`, `table:row_count`, `json:exact_keys`, `bullets:count`, `bullets:max_words`, `csv:header`, `sentences:count+final_citation`); (b) an **id→checker registry**, so that the question can be rewritten (including via the persona axis/ADR-004 and `regenerated: v2`) without breaking validation; (c) report **per requirement** in addition to the example total.
**What we would NOT adopt:** the `kwargs` "union" schema (all possible fields, almost all `None` [HF/code]). For us a type-specific `params` is better (`{"id":"bullets:count","n":3}`), more readable and validatable.

**R2 — Prompt-level vs instruction-level: report both, with prompt-level as the gate. SUPPORTS (my adaptation).**
Our questions have 1–2 requirements; today `format_validator` returns a single `format_valid` per example, which is equivalent to **prompt-level strict** (all or nothing). With the list of requirements from R1, **instruction-level** (fraction of requirements met) becomes possible, which gives fine-grained diagnosis (the table was right but with 2 rows instead of 3). **Affects:** `format_validator` (also return the per-requirement fraction) and the report for the `format` slice. **We would adopt:** prompt-level as the main number (it is what the user feels: the format is right or not) and inst-level as a diagnostic. With 1–2 requirements per question the difference between the two will be small; **to verify** whether it is worth the cost before implementing.

**R3 — strict vs loose: the paper CHALLENGES our "strict only" design.**
IFEval shows that strict produces false negatives attributed to the model (§2.2) and that this is a practical problem. Our validator has the same risk: `_check_json` uses `json.loads(answer)` on the whole answer, so "Here is the JSON: {...}" or a ```` ```json ```` fence fails; `_check_markdown_table` requires the first line to start with `|`, so an introductory sentence before the table fails; `_check_csv` likewise. In part this is **desirable** (the question says "No other text") and in part it is the false negative that IFEval describes. **Affects:** `format_validator` and the `format` slice. **We would adopt:** a **limited and explicit** loose variant per type (accept a code fence; accept 1 introduction line before the table when the question did **not** say "No other text"), **reporting strict and loose side by side** (IFEval does this and justifies it because of the false positive, §2.2). We would **not adopt** loose as a substitute for strict nor as the sole gate.

**R4 — The 8 blind transformations: we would NOT adopt them. CHALLENGES copying the mechanism.**
Removing **the first or the last line** of any answer (transformations 2 and 3, §2.2) would be destructive in our formats: the first line of a markdown table is the **header** (the requested columns), the first line of a CSV is the **header**, and the last line of the question with a citation ("ending with `[2609.xxxxxv1]`") is the **citation line**. Removing them could (i) make a wrong answer pass (false positive: without the header, a broken table can "look" correct) or (ii) make a right answer fail. The paper acknowledges this kind of false positive (§2.2) for word counts. **Affects:** any loose-mode design for `format_validator`. **Alternative:** **type-specific** normalization (strip the ```` ``` ```` fence, strip `**`, accept a 1-line preamble *only* if the remaining answer parses).

**R5 — N=10 in the `format` slice: the paper does NOT help and its dataset is ~54 times larger. NEUTRAL, with an important caveat.**
IFEval has 541 prompts; we have 10 examples in `format`. **[My calculation]** 95% Wilson CI: 10/10 → [72%; 100%]; 9/10 → [60%; 98%]; 8/10 → [49%; 94%]; 7/10 → [40%; 89%]. Therefore, even with 10/10 we cannot claim more than "≥ ~72%" with 95% confidence; and a difference of 1–2 hits between two versions of the system **is not distinguishable from noise**. The paper does not discuss small samples (it uses 541 without a CI). **Affects:** the gate of the `format` slice and `test_format_golds_dogfood_all_pass` (10/10 on gold, which validates the *checkers*, not the system). **We would adopt:** reporting `k/N` and the CI instead of just the percentage; treating the `format` slice as a **regression test** (any failure investigated by hand), not as a rate estimate. The gain from moving to atomic requirements (R1/R2) is also statistical: more items (e.g., ~15–20 requirements in 10 questions) for the inst-level version, **but these requirements are not independent** (they come from the same questions), so the CI does not shrink as much as N suggests (my interpretation).

**R6 — Deterministic code instead of LLM-judge for format. SUPPORTS.**
The paper argues for code-based checking against "model-based evaluation" because of the evaluator's bias/limits (§1). For format (Barnett's FP5) this matches our choice: zero LLM cost and a reproducible result. It is **neutral** for metrics that require semantic judgment (faithfulness, completeness, specificity), where there is no verifiable equivalent. **Affects:** `format_validator` (keep deterministic); P11 (judge≠generator) does not change.

**R7 — Fallback for unknown format (`score 0` without `key`). SUPPORTS the P10 fix.**
In IFEval, an instruction only enters the count if there is a checker for its id (by construction, Table 1). Our fallback scores **0** when it does not recognize the format, which contaminates the slice. IFEval does not handle this case (every prompt has a checker), so the support is only by analogy: with a structured spec, missing spec ⇒ `None` (skipped), as P10 proposes. **Affects:** `format_validator` (the `return {"score": 0, ...}` line).

**What we would adopt:** (1) `format_spec` = list of atomic requirements with id + parameters, and an id→checker registry; (2) report prompt-level (main) and inst-level (diagnostic); (3) strict as the gate, loose per type only as a diagnostic, always side by side; (4) `k/N` with CI in the `format` slice; (5) keep code-based checking.
**What we would NOT adopt:** (1) the generic 8 transformations (R4); (2) the `kwargs` "union" schema with `None` fields (R1); (3) IFEval's numbers (76.89 etc.) as a reference for our system: different task, domain and models; (4) treating loose as a substitute for strict.

## 7. Useful quotations (short literal excerpts)

1. "very few instructions are 100% verifiable objectively and automatically" — §1 (Introduction).
2. "Although this loose instruction-following verification process reduces false negatives, it is likely to introduce false positives." — §2.2 (IFEval Metrics).
3. "atomic instructions for which one can use a simple, interpretable, and deterministic program to verify" — §1 (Introduction).
