"""Offline unit tests for the retrieval evaluators (CKPT-0.6.1).

No network, no LLM — controlled example dicts. Contract under test:
docs/roadmap/ckpt-0.6-plan.md §3 (anti-pooling-bias semantics).
"""

import json
import math
import re

import pytest

import evaluators as ev


def ref(chunk_ids=(), arxiv_ids=()):
    return {"gold_chunk_ids": list(chunk_ids), "gold_arxiv_ids": list(arxiv_ids)}


# metadata under which precision is defined (roadmap P6 / D-8): the gold is complete.
ADJ = {"review": {"state": "adjudicated"}}


# --- recall_at_k ---------------------------------------------------------

def test_recall_full():
    assert ev.recall_at_k(["a", "b"], [], ref(chunk_ids=["a", "b"])) == 1.0


def test_recall_partial():
    assert ev.recall_at_k(["a"], [], ref(chunk_ids=["a", "b"])) == 0.5


def test_recall_none():
    assert ev.recall_at_k(["x"], [], ref(chunk_ids=["a"])) == 0.0


def test_recall_empty_gold_skips():
    assert ev.recall_at_k(["a"], [], ref()) is None


def test_recall_uses_arxiv_ids_for_multidoc():
    assert ev.recall_at_k([], ["p1", "p2"], ref(arxiv_ids=["p1", "p2"])) == 1.0


# --- hit_rate ------------------------------------------------------------

def test_hit_rate():
    assert ev.hit_rate(["x", "a"], [], ref(chunk_ids=["a"])) == 1.0
    assert ev.hit_rate(["x"], [], ref(chunk_ids=["a"])) == 0.0
    assert ev.hit_rate(["x"], [], ref()) is None


# --- mrr -----------------------------------------------------------------

def test_mrr_first_position():
    assert ev.mrr(["a", "x"], [], ref(chunk_ids=["a"])) == 1.0


def test_mrr_third_position():
    assert math.isclose(ev.mrr(["x", "y", "a"], [], ref(chunk_ids=["a"])), 1 / 3)


def test_mrr_absent():
    assert ev.mrr(["x"], [], ref(chunk_ids=["a"])) == 0.0


# --- precision_at_k ------------------------------------------------------

def test_precision_at_k():
    # 2 of 3 retrieved are gold
    r = ref(chunk_ids=["a", "b"])
    assert math.isclose(ev.precision_at_k(["a", "b", "x"], [], r, metadata=ADJ), 2 / 3)


def test_precision_empty_gold_skips():
    # unanswerable/stale: no gold labels -> precision is undefined by design
    assert ev.precision_at_k(["a"], [], ref(), metadata=ADJ) is None


def test_precision_none_for_single_paper_gold():
    # roadmap P6 / D-8: 1-paper gold of answerable/persona is not known to be complete, so a
    # perfect retrieval would score 1/k (0.2 at k=5) because the rest is unjudged, not wrong.
    meta = {"slice": "answerable", "review": {"state": "human_reviewed"}}
    assert ev.precision_at_k(CHUNKS, PAPERS, ref(arxiv_ids=["p1"]), metadata=meta) is None


def test_precision_none_without_metadata():
    # no evidence of completeness -> undefined (safe default)
    assert ev.precision_at_k(CHUNKS, PAPERS, ref(arxiv_ids=["p1", "p2"])) is None


def test_precision_defined_for_known_item_multidoc():
    meta = {"slice": "multi-doc", "review": {"state": "human_reviewed"}}
    got = ev.precision_at_k(CHUNKS, PAPERS, ref(arxiv_ids=["p1", "p2"]), metadata=meta)
    assert math.isclose(got, 2 / 5)


# --- is_gold_complete ----------------------------------------------------

def test_gold_complete_rules():
    two = ref(arxiv_ids=["p1", "p2"])
    one_chunk = ref(chunk_ids=["p1:aaa"])
    assert ev.is_gold_complete(two, {"slice": "multi-doc"})  # known-item: named papers
    assert not ev.is_gold_complete(ref(), {"slice": "multi-doc", "open_topic": True})
    assert not ev.is_gold_complete(two, {"slice": "multi-doc", "open_topic": True})  # pending
    assert not ev.is_gold_complete(one_chunk, {"slice": "answerable"})
    assert not ev.is_gold_complete(one_chunk, {"slice": "persona"})
    assert ev.is_gold_complete(one_chunk, ADJ)  # adjudicated: complete w.r.t. the judged pool
    assert not ev.is_gold_complete(ref(), ADJ)  # no gold at all: nothing to be complete about
    assert not ev.is_gold_complete(two, None)


# --- is_retrieval_evaluable (skip rules) --------------------------------

def test_open_topic_skipped():
    md = {"retrieval_gold": "pending-adjudication"}
    assert not ev.is_retrieval_evaluable(ref(chunk_ids=["shouldnt-matter"]), md)


def test_zero_gold_skipped():
    assert not ev.is_retrieval_evaluable(ref(), {"slice": "unanswerable"})


def test_known_item_evaluable():
    assert ev.is_retrieval_evaluable(ref(arxiv_ids=["p1"]), {"slice": "multi-doc"})


# --- format_validator (0.6.2) -------------------------------------------

import build_golden_dataset as gds


def test_format_golds_dogfood_all_pass():
    """The 10 curated format golds must pass the validator themselves."""
    for e in gds.format_examples():
        r = ev.format_validator(e["inputs"]["question"], e["outputs"]["answer"])
        assert r["score"] == 1, (e["inputs"]["question"][:60], r)


def test_format_json_wrong_key_fails():
    q = 'Return a JSON object with exactly the keys `paper_id`, `title`, `core_problem`, `approach` describing paper X. No other text.'
    r = ev.format_validator(q, json.dumps({"paper_id": 1, "title": "t"}))
    assert r["score"] == 0 and "keys" in r["reason"]


def test_format_table_missing_separator_fails():
    q = "In a markdown table with columns `Paper` and `Phenomenon`, compare A and B."
    r = ev.format_validator(q, "| Paper | Phenomenon |\n| a | b |")
    assert r["score"] == 0 and "separator" in r["reason"]


def test_format_bullets_over_words_fails():
    q = "In exactly 3 bullet points of at most 12 words each, summarize X."
    bad = "- one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    r = ev.format_validator(q, bad + "\n- ok\n- ok2")
    assert r["score"] == 0 and "words" in r["reason"]


def test_format_csv_order_fails():
    q = "Reply as CSV with header `arxiv_id,title,published` listing papers 2609.18471v1, 2609.18515v1 and 2609.18820v2, in that order. No other text."
    bad = "arxiv_id,title,published\n2609.18515v1,B,2026-09-16\n2609.18471v1,A,2026-09-16\n2609.18820v2,C,2026-09-16"
    r = ev.format_validator(q, bad)
    assert r["score"] == 0 and "order" in r["reason"]


def test_format_sentences_citation():
    q = "Answer in exactly 2 sentences, ending the second sentence with the citation [2609.18460v1]: how can a local deviation become collective loss of control?"
    good = "A deviation seeds contagion. Failure emerges when spread outpaces correction [2609.18460v1]."
    assert ev.format_validator(q, good)["score"] == 1
    assert ev.format_validator(q, "One sentence only [2609.18460v1].")["score"] == 0
    assert ev.format_validator(q, "Two sentences here. But no citation at end.")["score"] == 0


def test_format_unknown_instruction():
    r = ev.format_validator("Just answer plainly.", "ok")
    assert r["score"] == 0 and "no known" in r["reason"]


# --- lexical correctness signals (P9a; ADR-008 §2, ADR-010 / D-14, D-15) -------

def _scores(out):
    return {r["key"]: r["score"] for r in out["results"]}


MULTIDOC_GOLDS = [a for _, _, a in gds._MULTIDOC_EXAMPLES]  # the 15 real references of the slice


@pytest.mark.parametrize("text, expected", [
    ("The Sky, is BLUE!", ["sky", "is", "blue"]),                       # lowercase, punctuation, article
    ("component-level", ["component", "level"]),                        # hyphen is a space (N1-b), not deleted
    ("planning — action", ["planning", "action"]),                 # em dash disappears
    ("a data analysis of the an", ["data", "analysis", "of"]),          # articles only as whole words
    ("does not improve", ["does", "not", "improve"]),                   # NO stopword list: negation survives
    ("", []),
    (None, []),
])
def test_normalize_text_rule(text, expected):
    assert ev.normalize_text(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("[2609.20804v1] x", ["x"]),
    ("[2609.12345] x", ["x"]),                      # the PROMPT_V1 example has no version
    ("2609.20804v1 x", ["x"]),                      # bare id as the sentence subject (gold answers)
    ("[2609.20804V1] x", ["x"]),                    # case-insensitive
    ("[2609.20804v1, 2609.20474v1] x", ["x"]),      # several ids in one bracket
    ("(2609.20804) x", ["x"]),                      # any position, not only brackets
    ("arXiv:2609.20804v1 x", ["arxiv", "x"]),       # only the id goes; the word stays
    ("in 2026 we ran 12345 trials, v3.14", ["in", "2026", "we", "ran", "12345", "trials", "v3", "14"]),
])
def test_arxiv_ids_removed_in_every_spelling(text, expected):
    assert ev.normalize_text(text) == expected


def test_token_f1_and_recall_basics():
    assert ev.token_f1("the sky is blue", "the sky is blue") == 1.0
    assert ev.word_recall("the sky is blue", "the sky is blue") == 1.0
    assert ev.token_f1("grass green", "sky blue") == 0.0
    # a verbose answer that contains the reference: Recall 1.0 (Adlakha), F1 punished by length
    verbose = "sky blue and also many other unrelated things"
    assert ev.word_recall(verbose, "sky blue") == 1.0
    assert 0.0 < ev.token_f1(verbose, "sky blue") < 1.0
    # multiset: repeating a word never counts past its count in the reference
    assert ev.word_recall("cat cat cat", "cat dog") == 0.5
    assert math.isclose(ev.token_f1("cat cat cat", "cat dog"), 2 * (1 / 3) * 0.5 / (1 / 3 + 0.5))
    # empty inputs
    assert ev.token_f1("", "sky") == 0.0 and ev.word_recall("", "sky") == 0.0
    assert ev.token_f1("sky", "") == 0.0 and ev.word_recall("sky", "") == 0.0


# T0-T4 of the ADR-010 evidence: the answer is the reference reformatted in ONE way; the same
# content must score 1.0 (the literal SQuAD rule loses up to 17 points on T1/T4).
def _respell_hyphen(t):
    return re.sub(r"(?<=\w)-(?=\w)", " ", t)


def _dash_to_comma(t):
    return re.sub(r"\s*[—–]\s*", ", ", t)


def _shout(t):  # only sentence-ending periods are spaced: splitting the dot of an id is not a rule test
    return re.sub(r"\.(\s|$)", r" .\1", t.upper().replace(",", " ,"))


REFORMATS = {
    "T0 identical": lambda t: t,
    "T1 hyphen->space": _respell_hyphen,
    "T2 em dash->comma": _dash_to_comma,
    "T3 UPPERCASE + loose punctuation": _shout,
    "T4 T1+T2": lambda t: _dash_to_comma(_respell_hyphen(t)),
}


@pytest.mark.parametrize("name", list(REFORMATS))
def test_reformatting_the_reference_keeps_the_score_at_one(name):
    for gold in MULTIDOC_GOLDS:
        answer = REFORMATS[name](gold)
        assert math.isclose(ev.word_recall(answer, gold), 1.0), (name, gold[:50])
        assert math.isclose(ev.token_f1(answer, gold), 1.0), (name, gold[:50])


# U1-U5 of the ADR-010 evidence: option O1 (ids removed everywhere) is the only one that gives
# 1.0 to the same content however the id is written and 0.0 to an answer made only of ids.
ID_SPELLINGS = {
    "U1 bare id written [id] (model style)": lambda t: re.sub(r"(?<!\[)\b(\d{4}\.\d{4,5}v\d+)\b", r"[\1]", t),
    "U2 citation without version": lambda t: re.sub(r"\b(\d{4}\.\d{4,5})v\d+\b", r"\1", t),
    "U3 cites the wrong paper": lambda t: ev.ARXIV_ID.sub("2609.99999v1", t),
    "U4 no citation at all": lambda t: ev.ARXIV_ID.sub(" ", t),
}


@pytest.mark.parametrize("name", list(ID_SPELLINGS))
def test_id_spelling_does_not_change_the_score(name):
    for gold in MULTIDOC_GOLDS:
        answer = ID_SPELLINGS[name](gold)
        assert math.isclose(ev.word_recall(answer, gold), 1.0), (name, gold[:50])
        assert math.isclose(ev.token_f1(answer, gold), 1.0), (name, gold[:50])


def test_an_answer_made_only_of_ids_scores_zero():
    for gold in MULTIDOC_GOLDS:
        ids_only = " ".join(f"[{i}]" for i in ev.ARXIV_ID.findall(gold))
        assert ev.word_recall(ids_only, gold) == 0.0 and ev.token_f1(ids_only, gold) == 0.0


def test_empty_answer_floor_is_the_known_and_accepted_tradeoff():
    """ADR-010 trade-off: stopwords stay, so an answer made only of the reference's own function
    words scores above 0 (0.222 on average with the 198-word NLTK list; this smaller inline set
    gives a similar level). The test pins the *existence* of the floor so nobody mistakes the
    Recall level for quality — and so a stopword list is added deliberately, not by accident."""
    function_words = {
        "the", "and", "to", "are", "as", "by", "with", "of", "a", "an", "in", "is", "for", "from",
        "into", "than", "not", "at", "all", "most", "both", "which", "was", "its", "because", "or",
    }
    floors = []
    for gold in MULTIDOC_GOLDS:
        probe = " ".join(w for w in ev.normalize_text(gold) if w in function_words)
        floors.append(ev.word_recall(probe, gold))
    mean_floor = sum(floors) / len(floors)
    assert 0.05 < mean_floor < 0.35, mean_floor


# --- token_f1_summary (renamed from f1_summary_evaluator; P8/D-6 moved abstention out) -------

def _ex(slice_name):
    from types import SimpleNamespace

    return SimpleNamespace(inputs={}, outputs={}, metadata={"slice": slice_name})


def test_token_f1_summary_scope_and_per_slice_keys():
    outputs = [{"answer": "the sky is blue"}, {"answer": "sky blue"}, {"answer": "grass is green"},
               {"answer": "I don't know."}, {"answer": "ignored"}]
    refs = [{"answer": "the sky is blue"}, {"answer": "sky"}, {"answer": "grass is green"},
            {"answer": "I don't know."}, {"answer": ""}]
    examples = [_ex("answerable"), _ex("persona"), _ex("multi-doc"), _ex("unanswerable"), _ex("answerable")]
    scores = _scores(ev.token_f1_summary(outputs, refs, examples))
    # `unanswerable` is out of scope even though it has a gold answer; the empty gold is skipped
    assert scores["token_f1_summary"] == pytest.approx((1.0 + 2 / 3 + 1.0) / 3)
    assert scores["word_recall_summary"] == 1.0
    assert scores["token_f1_summary__answerable"] == 1.0
    assert scores["token_f1_summary__persona"] == pytest.approx(2 / 3)
    assert not any(k.endswith("__unanswerable") for k in scores)
    assert set(scores) == {
        f"{m}{suffix}" for m in ("token_f1_summary", "word_recall_summary")
        for suffix in ("", "__answerable", "__persona", "__multi-doc")
    }
    assert "f1_summary" not in scores  # the old key is gone


def test_token_f1_summary_never_guesses_the_slice():
    out = ev.token_f1_summary([{"answer": "x"}], [{"answer": "x"}])  # no `examples`
    assert out["results"] == []
    out = ev.token_f1_summary([{"answer": "x"}], [{"answer": "x"}], [_ex("format")])
    assert out["results"] == []


def test_f1_summary_evaluator_was_renamed():
    assert not hasattr(ev, "f1_summary_evaluator") and not hasattr(ev, "_token_f1")


def test_token_f1_summary_contract_with_langsmith():
    """Regression for P4: langsmith builds EvaluationResult(extra="forbid") per entry and only
    accepts summary evaluators whose args are runs/examples/inputs/outputs/reference_outputs."""
    from types import SimpleNamespace

    from langsmith.evaluation.evaluator import EvaluationResult, _normalize_summary_evaluator

    wrapped = _normalize_summary_evaluator(ev.token_f1_summary)  # raises on bad signature
    runs = [SimpleNamespace(outputs={"answer": "[2609.20804v1] fixes the loop"})]
    examples = [SimpleNamespace(inputs={}, outputs={"answer": "2609.20804v1 fixes the loop"},
                                metadata={"slice": "multi-doc"})]
    out = wrapped(runs, examples)
    results = out["results"] if isinstance(out, dict) else out.results
    parsed = [EvaluationResult(**r) for r in results]  # raises on any extra key
    by_key = {p.key: p for p in parsed}
    assert {"token_f1_summary", "word_recall_summary", "token_f1_summary__multi-doc"} <= set(by_key)
    assert by_key["token_f1_summary"].score == 1.0 and "n=1" in by_key["token_f1_summary"].comment


# --- abstained() (P7/D-5: detection by paraphrase, not two exact phrases) ----

# >= 15 refusal paraphrases: none of these starts with the two old fixed phrases
REFUSALS = [
    "I don't know.",
    "I do not know the answer to that.",
    "I'm sorry, the context doesn't say.",  # the reproduced P7 evidence: used to score 0.0
    "The context does not contain that information.",
    "No papers found in this period.",
    "I cannot find any abstract addressing that.",
    "I can't find the answer in the retrieved papers.",
    "I couldn't find relevant papers on this topic.",
    "I did not find any paper on this.",
    "The retrieved abstracts don't cover this question.",
    "None of the retrieved papers address this.",
    "That information isn't in the context.",
    "I have no information about that.",
    "The corpus doesn't include a paper on this.",
    "No relevant papers were found.",
    "This is beyond the scope of the corpus.",
]

# >= 10 real answers containing "know"/"knowledge" that must NOT be read as refusals
NOT_REFUSALS_WITH_KNOW = [
    "As far as we know, all three papers use the same benchmark.",
    "The authors don't know whether the effect generalizes to larger networks.",
    "It is known that transformers suffer from hallucination.",
    "The paper reviews what is currently known about multi-agent failure modes.",
    "Nobody knows the exact scaling law, but the paper estimates it empirically.",
    "We know from section 3 that the dataset contains 250 abstracts.",
    "To the best of our knowledge, this is the first survey of its kind.",
    "The model acknowledges it doesn't know the answer scale, and proposes a proxy.",
    "I know this sounds counterintuitive, but the paper's result holds.",
    "Knowing the limits of the corpus, the authors restrict claims to abstracts.",
]


def test_abstained_catches_paraphrases():
    for r in REFUSALS:
        assert ev.abstained(r), r


def test_abstented_keeps_real_answers_with_know():
    for a in NOT_REFUSALS_WITH_KNOW:
        assert not ev.abstained(a), a


def test_abstained_normalizes_case_quotes_and_apostrophes():
    assert ev.abstained("I DON'T KNOW.")
    assert ev.abstained("I don\u2019t know.")  # typographic apostrophe
    assert ev.abstained("  no papers   found.  ")


def test_abstained_empty_answer():
    assert not ev.abstained("")
    assert not ev.abstained(None)  # guarded by `answer or ""` in abstained()


def test_abstained_first_person_hedge_is_a_known_false_positive():
    # documented limitation (evaluators.abstained docstring): the judge (0.6.3) is
    # the ground truth; EXP-0 measures the heuristic x judge divergence.
    assert ev.abstained("I don't know whether the paper tests this, but the results suggest yes.")


def test_abstained_is_english_scoped():
    # The markers are English; PROMPT_V1 fixes no answer language but the golden set
    # is English. A refusal in another language reads as an answer — the documented
    # limitation (ADR-007 trade-offs). The judge (natively multilingual) is the
    # reference for that case; the fix is NOT translating the marker list.
    assert not ev.abstained("Desculpe, não encontrei nada sobre isso no contexto.")
    assert not ev.abstained("Lo siento, no tengo esa información.")
    assert not ev.abstained("Je ne sais pas.")


# --- wilson_ci (P8/D-6: counts + CI, honest for n ~ 10-25) ------------------

def test_wilson_anchors():
    lo, hi = ev.wilson_ci(10, 10)
    assert math.isclose(lo, 0.72, abs_tol=0.005) and math.isclose(hi, 1.0, abs_tol=1e-9)
    lo, hi = ev.wilson_ci(0, 10)
    assert lo == 0.0 and math.isclose(hi, 0.28, abs_tol=0.005)
    lo, hi = ev.wilson_ci(14, 15)
    assert math.isclose(lo, 0.70, abs_tol=0.005) and math.isclose(hi, 0.99, abs_tol=0.005)


def test_wilson_rejects_bad_counts():
    import pytest

    with pytest.raises(ValueError):
        ev.wilson_ci(0, 0)
    with pytest.raises(ValueError):
        ev.wilson_ci(16, 15)


# --- abstention_summary (P8/D-6: recall + over-refusal, never accuracy) ------

def _run_abstention(answers, shoulds, slices=None):
    outputs = [{"answer": a} for a in answers]
    refs = [{"should_abstain": s} for s in shoulds]
    examples = None
    if slices is not None:
        from types import SimpleNamespace

        examples = [SimpleNamespace(metadata={"slice": s}) for s in slices]
    return ev.abstention_summary(outputs, refs, examples)


def test_abstention_summary_never_refuses_dumb_system():
    # P8's evidence table: answers everything -> accuracy would LOOK 75%, gate 0%
    out = _scores(ev.abstention_summary(
        [{"answer": "confident answer"}] * 10,
        [{"should_abstain": True}] * 3 + [{"should_abstain": False}] * 7,
    ))
    assert out["abstention_recall"] == 0.0
    assert out["over_refusal_rate"] == 0.0


def test_abstention_summary_always_refuses_dumb_system():
    # P8's evidence table: refuses everything -> gate 100% (passes!) but is useless;
    # accuracy would be 25%. The guardrail exposes it.
    out = _scores(ev.abstention_summary(
        [{"answer": "I don't know."}] * 10,
        [{"should_abstain": True}] * 3 + [{"should_abstain": False}] * 7,
    ))
    assert out["abstention_recall"] == 1.0
    assert out["over_refusal_rate"] == 1.0


def test_abstention_summary_perfect_system():
    answers = ["I don't know."] * 3 + ["a real answer"] * 7
    shoulds = [True] * 3 + [False] * 7
    out = _scores(ev.abstention_summary(
        [{"answer": a} for a in answers], [{"should_abstain": s} for s in shoulds]
    ))
    assert out["abstention_recall"] == 1.0
    assert out["over_refusal_rate"] == 0.0
    assert out["abstention_f1"] == 1.0


def test_abstention_summary_reports_counts_and_wilson_ci_in_comment():
    results = ev.abstention_summary(
        [{"answer": "here is a confident answer"}] + [{"answer": "I don't know."}] * 14,
        [{"should_abstain": True}] * 15,
    )["results"]
    recall = next(r for r in results if r["key"] == "abstention_recall")
    assert "14/15" in recall["comment"] and "Wilson 95% CI" in recall["comment"]


def test_abstention_summary_per_slice():
    answers = ["I don't know.", "I can't find that.", "here is the answer anyway"]
    shoulds = [True, True, True]  # 2 correct refusals in unanswerable, 1 miss in stale
    slices = ["unanswerable", "unanswerable", "stale"]
    out = _scores(_run_abstention(answers, shoulds, slices))
    assert out["abstention_recall"] == 2 / 3
    assert out["abstention_recall__unanswerable"] == 1.0
    assert out["abstention_recall__stale"] == 0.0
    assert "over_refusal_rate" not in out  # no should-answer examples in this run


def test_abstention_summary_skips_examples_without_labels():
    out = ev.abstention_summary([{"answer": "x"}], [{"answer": "gold"}])
    assert out["results"] == []


def test_abstention_summary_contract_with_langsmith():
    """P4 regression, abstention flavor: the `examples` arg must be a supported
    langsmith name and every result must build an EvaluationResult (comment ok,
    nothing else)."""
    from types import SimpleNamespace

    from langsmith.evaluation.evaluator import EvaluationResult, _normalize_summary_evaluator

    wrapped = _normalize_summary_evaluator(ev.abstention_summary)
    runs = [
        SimpleNamespace(outputs={"answer": "I don't know."}),
        SimpleNamespace(outputs={"answer": "the paper uses 250 abstracts"}),
    ]
    examples = [
        SimpleNamespace(inputs={}, outputs={"should_abstain": True}, metadata={"slice": "unanswerable"}),
        SimpleNamespace(inputs={}, outputs={"should_abstain": False}, metadata={"slice": "answerable"}),
    ]
    out = wrapped(runs, examples)
    results = out["results"] if isinstance(out, dict) else out.results
    parsed = [EvaluationResult(**r) for r in results]
    by_key = {p.key: p for p in parsed}
    assert by_key["abstention_recall"].score == 1.0
    assert by_key["over_refusal_rate"].score == 0.0
    assert by_key["abstention_f1"].score == 1.0
    assert "1/1" in by_key["abstention_recall"].comment
    assert by_key["abstention_recall__unanswerable"].score == 1.0
    assert by_key["over_refusal_rate__answerable"].score == 0.0


# --- paper-level gold (CKPT-0.6.1b P1-P3) ---------------------------------

CHUNKS = ["p1:aaa", "p2:bbb", "p3:ccc", "p4:ddd", "p5:eee"]
PAPERS = ["p1", "p2", "p3", "p4", "p5"]


def test_mrr_paper_gold_not_shifted_by_chunk_list():
    # Regression: ranks used to be offset by len(chunk_ids) -> 1/6 instead of 1.
    assert ev.mrr(CHUNKS, PAPERS, ref(arxiv_ids=["p1"])) == 1.0
    assert math.isclose(ev.mrr(CHUNKS, PAPERS, ref(arxiv_ids=["p3"])), 1 / 3)


def test_precision_paper_gold_with_chunk_ids_present():
    # Regression: chunk ids were compared against paper gold -> always 0.0.
    assert math.isclose(ev.precision_at_k(CHUNKS, PAPERS, ref(arxiv_ids=["p1"]), metadata=ADJ), 1 / 5)
    two = ref(arxiv_ids=["p1", "p2"])
    assert math.isclose(ev.precision_at_k(CHUNKS, PAPERS, two, metadata=ADJ), 2 / 5)


def test_paper_level_rank_dedups_duplicate_chunks_keeping_order():
    # decisions.md 2026-09-27: top-5 can hold 3 chunks of one paper.
    dup_chunks = ["pA:1", "pA:2", "pA:3", "pB:4", "pC:5"]
    assert ev.mrr(dup_chunks, [], ref(arxiv_ids=["pB"])) == 1 / 2  # rank among distinct papers
    assert math.isclose(ev.precision_at_k(dup_chunks, [], ref(arxiv_ids=["pB"]), metadata=ADJ), 1 / 3)


def test_paper_ids_derived_from_chunk_ids_when_not_provided():
    assert ev.recall_at_k(CHUNKS, [], ref(arxiv_ids=["p2", "p9"])) == 0.5
    assert ev.hit_rate(CHUNKS, [], ref(arxiv_ids=["p9"])) == 0.0


def test_paper_level_default_prefers_explicit_paper_gold_over_chunk_gold():
    # roadmap D-1: the paper is the primary level; the chunk gold is diagnostic only.
    r = {"gold_chunk_ids": ["p2:bbb"], "gold_arxiv_ids": ["p1"]}
    assert ev.mrr(CHUNKS, PAPERS, r) == 1.0
    assert ev.mrr(CHUNKS, PAPERS, r, level="chunk") == 1 / 2


def test_chunk_level_ignores_arxiv_list():
    assert ev.mrr(["x:1", "a:1"], ["a", "x"], ref(chunk_ids=["a:1"]), level="chunk") == 1 / 2


def test_empty_retrieval_with_gold():
    assert ev.mrr([], [], ref(arxiv_ids=["p1"])) == 0.0
    assert ev.precision_at_k([], [], ref(arxiv_ids=["p1"]), metadata=ADJ) == 0.0
    assert ev.recall_at_k([], [], ref(arxiv_ids=["p1"])) == 0.0


def test_arxiv_id_derivation_matches_utils_chunk_id():
    from langchain_core.documents import Document

    import utils

    doc = Document(page_content="some chunk text", metadata={"arxiv_id": "2609.12345v1"})
    assert ev._arxiv_id_of_chunk(utils.chunk_id(doc)) == "2609.12345v1"


# --- D-1 / D-2 / P5: paper level by default, chunk level as diagnostic ----------

# Real case (golden set, "checkpoint handoff"): the gold chunk (2nd chunk of the paper)
# came 2nd; another chunk of the SAME paper came 1st.
HANDOFF_CHUNKS = [
    "2609.19636v1:7765a13e5f1b",  # same paper, sibling chunk
    "2609.19636v1:7e548892cc14",  # gold chunk (generated the question)
    "2609.18460v1:061c23fc26f2",
    "2609.19892v1:facb0f33496f",
    "2609.18864v1:1f1c23f8779a",
]
HANDOFF_REF = {"gold_chunk_ids": ["2609.19636v1:7e548892cc14"], "gold_arxiv_ids": []}


def test_chunk_gold_is_scored_at_paper_level_by_default():
    assert ev.mrr(HANDOFF_CHUNKS, [], HANDOFF_REF) == 1.0  # sibling in 1st = right paper
    assert ev.hit_rate(HANDOFF_CHUNKS, [], HANDOFF_REF, k=1) == 1.0


def test_exact_chunk_is_a_diagnostic_level():
    assert ev.mrr(HANDOFF_CHUNKS, [], HANDOFF_REF, level="chunk") == 1 / 2
    assert ev.hit_rate(HANDOFF_CHUNKS, [], HANDOFF_REF, level="chunk", k=1) == 0.0
    assert ev.hit_rate(HANDOFF_CHUNKS, [], HANDOFF_REF, level="chunk", k=2) == 1.0


def test_chunk_level_is_undefined_without_chunk_gold():
    assert ev.mrr(CHUNKS, PAPERS, ref(arxiv_ids=["p1"]), level="chunk") is None


def test_unknown_level_raises():
    import pytest

    with pytest.raises(ValueError):
        ev.hit_rate(CHUNKS, PAPERS, ref(arxiv_ids=["p1"]), level="sentence")


def test_k_cuts_the_raw_retrieved_list_before_dedup():
    dup = ["pA:1", "pA:2", "pB:3", "pC:4", "pD:5"]
    # top-2 raw = [pA, pA] -> 1 distinct paper; pB is not in the top-2
    assert ev.hit_rate(dup, [], ref(arxiv_ids=["pB"]), k=2) == 0.0
    assert ev.hit_rate(dup, [], ref(arxiv_ids=["pB"]), k=3) == 1.0
    assert ev.mrr(dup, [], ref(arxiv_ids=["pB"]), k=3) == 1 / 2  # rank among distinct papers


def test_hit_at_ks_curve():
    curve = ev.hit_at_ks(CHUNKS, PAPERS, ref(arxiv_ids=["p3"]), ks=(1, 3, 5))
    assert curve == {1: 0.0, 3: 1.0, 5: 1.0}
    assert ev.hit_at_ks(CHUNKS, PAPERS, ref(), ks=(1,)) == {1: None}


def test_single_gold_paper_makes_recall_equal_hit():
    # why D-2 reports hit+MRR (not recall) for 1 gold paper
    for retrieved in (PAPERS, ["x", "y"], ["p1", "x"]):
        r = ref(arxiv_ids=["p1"])
        assert ev.recall_at_k([], retrieved, r) == ev.hit_rate([], retrieved, r)


def test_multi_gold_makes_recall_differ_from_hit():
    r = ref(arxiv_ids=["p1", "p2"])
    assert ev.hit_rate([], ["p1", "x"], r) == 1.0
    assert ev.recall_at_k([], ["p1", "x"], r) == 0.5


def test_primary_retrieval_metrics_policy():
    assert ev.primary_retrieval_metrics(HANDOFF_REF) == ["hit", "mrr"]  # chunk gold -> 1 paper
    assert ev.primary_retrieval_metrics(ref(arxiv_ids=["p1"])) == ["hit", "mrr"]
    assert ev.primary_retrieval_metrics(ref(arxiv_ids=["p1", "p2"])) == ["recall", "hit", "mrr"]
    assert ev.primary_retrieval_metrics(ref()) == []  # unanswerable / stale / open-topic
