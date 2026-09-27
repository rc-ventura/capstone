"""Build the `arxiv-copilot-golden` dataset, core split (docs/plan.md §2).

Slices: hand-written `unanswerable`/`format`/`stale` (0.5.2a), hybrid `multi-doc`
(0.5.2b/c), synthetic `answerable`/`persona` (0.5.3) generated from sampled corpus
chunks — synthetic-by-construction, so the source chunk IS the retrieval gold label.
`deep-hit` lands in CKPT-0.8 (found empirically during EXP-0).

Sync is idempotent per slice: a slice at its target count is skipped, an empty
slice is created in bulk, and a partially-populated slice hard-fails — partial
state means a human changed the dataset by hand and should resolve it.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from typing import Any, Callable

from langchain_core.documents import Document
from langsmith import Client
from openai import OpenAI

import config
import utils

logger = logging.getLogger(__name__)

Example = dict[str, Any]

ABSTAIN_ANSWER = "I don't know."
STALE_ANSWER = "No papers found in this period."


def _example(
    question: str,
    *,
    answer: str,
    slice_name: str,
    method: str,
    gold_chunk_ids: list[str] | None = None,
    gold_arxiv_ids: list[str] | None = None,
    should_abstain: bool = False,
    **extra_metadata: Any,
) -> Example:
    """Uniform example schema (docs/plan/ckpt-0.5-review-plan.md §3.1): every field
    is always present so evaluators never branch on slice. Provenance is a
    two-axis model: authoring (origin × method) vs review state — never mixed."""
    return {
        "inputs": {"question": question},
        "outputs": {
            "answer": answer,
            "gold_chunk_ids": gold_chunk_ids or [],
            "gold_arxiv_ids": gold_arxiv_ids or [],
            "should_abstain": should_abstain,
        },
        "metadata": {
            "slice": slice_name,
            "authoring": {"origin": "llm", "method": method},
            "review": {"state": "draft"},
            **extra_metadata,
        },
    }


# --- unanswerable (n=15, FP1): topics with zero coverage in a cs.AI corpus ---
# Domains picked to be unambiguously outside the indexed papers; gold is the
# explicit abstention label (docs/plan.md §2).

_UNANSWERABLE_QUESTIONS = [
    "What did the latest LIGO observing run reveal about neutron-star equation-of-state constraints?",
    "How do lithium-metal battery dendrites nucleate at the anode interface during fast charging?",
    "What are the current first-line treatment guidelines for stage III HER2-positive breast cancer?",
    "How does the Atlantic Meridional Overturning Circulation respond to sustained Greenland meltwater pulses?",
    "What geological evidence supports the Late Heavy Bombardment hypothesis for the Moon?",
    "How do bar-tailed godwits navigate non-stop across the Pacific during migration?",
    "What measurable acoustic properties distinguish Stradivarius violins from modern instruments?",
    "How is radiocarbon dating calibrated for Late Bronze Age Mediterranean samples?",
    "What magnetospheric process produces auroral beads before substorm onset?",
    "How do deep-sea hydrothermal vent communities tolerate high hydrogen sulfide concentrations?",
    "How does the carbon-concentrating mechanism of marine diatoms work biochemically?",
    "What did the InSight mission conclude about the thickness of Mars' crust?",
    "How does the honeybee waggle dance encode distance and direction to a food source?",
    "Which theories best explain the Late Bronze Age collapse in the eastern Mediterranean?",
    "What is the accepted age and formation process of Saturn's rings?",
]


def unanswerable_examples() -> list[Example]:
    return [
        _example(
            q,
            answer=ABSTAIN_ANSWER,
            slice_name="unanswerable",
            method="topic-authored",
            should_abstain=True,
        )
        for q in _UNANSWERABLE_QUESTIONS
    ]


# --- format (n=10, FP5): the format instruction IS the test; gold answers are ---
# correctly-formatted, corpus-grounded exemplars. No gold retrieval label is
# needed (docs/plan.md §2 ground-truth map).

_FORMAT_EXAMPLES: list[tuple[str, str]] = [
    (
        "List three papers from the indexed corpus in a markdown table with columns "
        "`| arXiv ID | Title | Main Contribution |`.",
        "| arXiv ID | Title | Main Contribution |\n"
        "|---|---|---|\n"
        "| 2609.20130v1 | AdaRepair-Mem: Adaptive Experience Orchestration for Repository-Level Program Repair "
        "| Orchestrates episodic repair memory adaptively, fixing imbalance and redundancy in repository-level memory retrieval |\n"
        "| 2609.20754v1 | RAFT: A Stateful Retrieval-Augmented Framework for Troubleshooting Agents "
        "| Models closed support cases as directed chains of timeline entries for entry-level retrieval |\n"
        "| 2609.20822v1 | Coding Agents with an Obstacle-Aware Harness for Safe Robot Manipulation "
        "| Evaluates LLM coding agents on robot manipulation under an explicit no-touch obstacle safety constraint |",
    ),
    (
        "Return a JSON object with exactly the keys `paper_id`, `title`, `core_problem`, "
        "`approach` describing paper 2609.20754v1. No other text.",
        '{\n'
        '  "paper_id": "2609.20754v1",\n'
        '  "title": "RAFT: A Stateful Retrieval-Augmented Framework for Troubleshooting Agents",\n'
        '  "core_problem": "Existing RAG systems treat support cases as static documents and overlook '
        'their multi-stage, stateful nature.",\n'
        '  "approach": "Abstract each closed historical case into a directed chain of timeline entries '
        'and retrieve at the entry level."\n'
        '}',
    ),
    (
        "In exactly 3 bullet points, compare what TRACE (2609.19897v1) and RAFT (2609.20754v1) "
        "retrieve and for what purpose.",
        "- TRACE retrieves traceable sources from OCR-degraded historical archives; RAFT retrieves "
        "timeline entries from closed enterprise support cases.\n"
        "- Both replace flat-document RAG with retrieval over structured, heterogeneous sources.\n"
        "- TRACE optimizes scholarly accountability; RAFT optimizes actionable troubleshooting guidance.",
    ),
    (
        "Reply as CSV with header `arxiv_id,title,published` listing papers 2609.18471v1, "
        "2609.18515v1 and 2609.18820v2, in that order. No other text.",
        "arxiv_id,title,published\n"
        "2609.18471v1,First Token Matters: Understanding Safety Collapse in Large Reasoning Models,2026-09-16\n"
        "2609.18515v1,Beyond Routine Compliance: Cunning Data Cultivates Safety Vigilance in Large Language Models,2026-09-16\n"
        "2609.18820v2,Compositional Policy Violations: When Step-Level Compliance Fails In Agentic AI Workflows,2026-09-16",
    ),
    (
        "In a two-column markdown table (columns `Aspect` and `Finding`), summarize how "
        "2609.20822v1 operationalizes safety evaluation for robot coding agents.",
        "| Aspect | Finding |\n"
        "|---|---|\n"
        "| Task setup | Each manipulation task pairs a goal with an obstacle the robot must not touch |\n"
        "| Baseline behavior | Agents pursue the goal but collide with the obstacle in most cases |\n"
        "| Paradigm assessed | A language model writes the robot controller as a program, "
        "with no robot-specific training |",
    ),
    (
        "As YAML, output `paper_id`, `title` and `primary_category` for paper 2609.18460v1. "
        "No other text.",
        "paper_id: 2609.18460v1\n"
        "title: \"Collective Loss of Control in LLM Agent Systems: An Epidemic Account of "
        "Mutation, Contagion, and Recovery\"\n"
        "primary_category: cs.AI",
    ),
    (
        "Answer in exactly 2 sentences, ending the second sentence with the citation "
        "[2609.18460v1]: how can a local deviation become collective loss of control?",
        "A spontaneous deviation creates a seed that communication lets other agents adopt and "
        "retransmit as an unsafe strategy. Collective failure emerges when that propagation outpaces "
        "correction and containment [2609.18460v1].",
    ),
    (
        "Return a JSON array with two objects, each having keys `id` and `title`, for papers "
        "2609.19897v1 and 2609.20754v1. No other text.",
        '[\n'
        '  {"id": "2609.19897v1", "title": "TRACE: Accountable Agentic Retrieval for Source '
        'Discovery in Digital Archives"},\n'
        '  {"id": "2609.20754v1", "title": "RAFT: A Stateful Retrieval-Augmented Framework for '
        'Troubleshooting Agents"}\n'
        ']',
    ),
    (
        "In a markdown table with columns `Paper` and `Phenomenon`, compare the phenomenon "
        "studied in 2609.18460v1 and 2609.18736v1.",
        "| Paper | Phenomenon |\n"
        "|---|---|\n"
        "| 2609.18460v1 | How local deviations spread via contagion into collective failure "
        "in multi-agent LLM systems |\n"
        "| 2609.18736v1 | The difficulty of maintaining logically consistent deductive reasoning "
        "over extended multi-agent interactions |",
    ),
    (
        "In exactly 3 bullet points of at most 12 words each, summarize the contribution of "
        "2609.18736v1.",
        "- LLMs struggle with consistent deduction across extended multi-step interactions.\n"
        "- The paper builds a text-based multi-agent testbed for deductive reasoning.\n"
        "- Tool augmentation is used to support evidence integration and belief updates.",
    ),
]


def format_examples() -> list[Example]:
    return [
        _example(q, answer=a, slice_name="format", method="topic-authored")
        for q, a in _FORMAT_EXAMPLES
    ]


# --- stale (n=10, FP1'/R3): every question asks about papers published AFTER ---
# CORPUS_SNAPSHOT_DATE (2026-09-17) on topics the corpus covers, to tempt retrieval
# into answering with old material. Gold pre-refresh is abstention; the CKPT-8
# refresh drill re-writes gold after re-indexing (docs/plan/ckpt-0.5-plan.md §4.2).

_STALE_QUESTIONS = [
    "What new papers on tool hallucination in LLM agents were published after September 17, 2026?",
    "Summarize advances in memory mechanisms for repository-level coding agents published on or after September 18, 2026.",
    "Is there any work on accountable agentic retrieval for digital archives since September 18, 2026?",
    "What's new in modeling failure contagion in multi-agent LLM systems after September 17, 2026?",
    "Which papers after September 17, 2026 address stateful retrieval for enterprise troubleshooting agents?",
    "What new findings on safety collapse in large reasoning models appeared this past week, after September 17, 2026?",
    "What did arXiv cs.AI publish about obstacle-aware safety harnesses for robot coding agents after September 17, 2026?",
    "Are there benchmarks for LLM agent trustworthiness under pressure published since September 18, 2026?",
    "Anything new on self-evolving multi-agent systems for financial document QA after September 17, 2026?",
    "What recent work, published after September 17, 2026, improves federated learning efficiency at scale?",
]


def stale_examples() -> list[Example]:
    return [
        _example(
            q,
            answer=STALE_ANSWER,
            slice_name="stale",
            method="topic-authored",
            should_abstain=True,
            corpus_snapshot=config.CORPUS_SNAPSHOT_DATE.isoformat(),
        )
        for q in _STALE_QUESTIONS
    ]


# --- multi-doc (n=15, FP3/FP7): hybrid composition (decision 2026-09-26, ---
# docs/plan/ckpt-0.5-plan.md §8): 10 known-item questions name the required
# paper IDs (mirrors product Features B/D, mechanical retrieval gold) and 5
# open-topic questions name no IDs (realistic production queries; their retrieval
# gold stays PENDING until the EXP-0 mini-pooling adjudication, and their
# gold answer doubles as the completeness reference, §2b).
# ids=None marks an open-topic example.

_MULTIDOC_OPEN_META = {"open_topic": True, "retrieval_gold": "pending-adjudication"}

_MULTIDOC_EXAMPLES: list[tuple[str, list[str] | None, str]] = [
    (
        "How do the harness studies 2609.20804v1 and 2609.20474v1 each isolate which "
        "harness components actually matter, and what exactly does each vary?",
        ["2609.20804v1", "2609.20474v1"],
        "2609.20804v1 fixes the execution loop and varies three components — planning, action space, "
        "and context management — across four models to enable component-level comparison, since "
        "harnesses are usually evaluated as monoliths [2609.20804v1]. 2609.20474v1 isolates planning "
        "guidance content by pairing prewritten task-specific plans (Fixed) with word-count-matched "
        "shuffled policy text (Sham), measuring success, erroneous acceptance, and cost [2609.20474v1].",
    ),
    (
        "Taken together, what do 2609.20822v1 and 2609.20804v1 suggest that typical evaluations "
        "of coding agents miss?",
        ["2609.20822v1", "2609.20804v1"],
        "2609.20804v1 shows harnesses are evaluated as monolithic systems, leaving the effectiveness "
        "of individual components unclear [2609.20804v1]. 2609.20822v1 shows the safety question was "
        "not asked at all: agents pursuing a manipulation goal collide with a forbidden obstacle in "
        "most cases [2609.20822v1]. Together: component-blind outcome metrics miss both which design "
        "choice helps and whether behavior is safe.",
    ),
    (
        "Contrast the retrieval unit and motivation of TRACE (2609.19897v1) and RAFT (2609.20754v1).",
        ["2609.19897v1", "2609.20754v1"],
        "TRACE retrieves traceable sources from OCR-degraded, heterogeneous historical archives; its "
        "motivation is accountability for scholarly and institutional use [2609.19897v1]. RAFT retrieves "
        "individual timeline entries from closed support cases abstracted into directed chains, because "
        "troubleshooting cases are multi-stage and stateful rather than static documents [2609.20754v1].",
    ),
    (
        "Which indexed papers argue that flat retrieval over raw or document-level sources is "
        "insufficient, and what structured alternative does each propose?",
        None,
        "RAFT argues support cases are not static documents and structures each closed case as a "
        "directed chain of timeline entries for entry-level retrieval [2609.20754v1]. TRACE argues flat "
        "retrieval fails on OCR-degraded archives and proposes training-free agentic retrieval with "
        "strong source traceability [2609.19897v1]. 2609.19615v1 argues raw telemetry needs semantic "
        "reconciliation and induces a business semantic layer via hierarchical LLM + RAG abstraction "
        "[2609.19615v1].",
    ),
    (
        "Compare how AdaRepair-Mem (2609.20130v1) and the Adaptive Memory Module of 2609.19128v1 "
        "decide which memories are worth using.",
        ["2609.20130v1", "2609.19128v1"],
        "AdaRepair-Mem finds episodic repair memory imbalanced across repositories and that more memory "
        "does not monotonically help — relevance, quality, and redundancy matter — so it orchestrates "
        "experience adaptively [2609.20130v1]. The Adaptive Memory Module instead gates episodic storage "
        "by salience and retrieves on triggers within a dual-process SwiftSage agent [2609.19128v1].",
    ),
    (
        "How do 2609.18461v1 and 2609.19128v1 differ in how they structure memory for language agents?",
        ["2609.18461v1", "2609.19128v1"],
        "2609.18461v1 argues raw textual memories are entangled and noisy, flat retrieval scores "
        "fragments independently, and static query-agnostic graphs miss context-dependent relations — "
        "so it disentangles memory via latent neuro-symbolic reasoning [2609.18461v1]. 2609.19128v1 "
        "keeps memory modular: salience-gated episodic storage plus bounded self-reflection bolted onto "
        "a dual-process agent [2609.19128v1].",
    ),
    (
        "When agents condition on long interaction histories, what cost-versus-quality trade-offs "
        "do indexed papers report, and how do they resolve them?",
        None,
        "Both reject naive full-history conditioning: 2609.20130v1 shows more memory does not "
        "monotonically raise repair success, so retrieval quality beats volume [2609.20130v1]; "
        "2609.20820v1 shows full histories induce spurious correlations while in-the-loop VLM "
        "compression is expensive, so it moves VLM processing to train time and deploys a lightweight "
        "saliency-supervised memory [2609.20820v1].",
    ),
    (
        "According to 2609.18471v1 and 2609.18515v1, where do safety failures of aligned models "
        "come from?",
        ["2609.18471v1", "2609.18515v1"],
        "2609.18471v1 locates refusal collapse at the onset of generation: a token-level positional "
        "analysis identifies a localized vulnerability in the first tokens of large reasoning models' "
        "responses [2609.18471v1]. 2609.18515v1 locates failure in concealed intent: aligned models "
        "miss harmful premises hidden inside benign-looking contexts, so alignment needs vigilance — "
        "scrutinizing intent and assumptions beneath surface semantics [2609.18515v1].",
    ),
    (
        "At what stage do 2609.18820v2 and 2609.20412v1 each intervene on alignment, and what gap "
        "does each target?",
        ["2609.18820v2", "2609.20412v1"],
        "2609.18820v2 targets governance scope: controls are step-scoped (input-output classifiers, "
        "per-turn rails) while real policies like authority limits are properties of the whole "
        "execution, which admits compositional policy violations where every step complies "
        "[2609.18820v2]. 2609.20412v1 targets training stage: post-training cannot cover every "
        "deployment environment, so alignment midtraining continues pretraining on alignment-relevant "
        "documents and is stress-tested for generalization [2609.20412v1].",
    ),
    (
        "What mechanisms do 2609.18460v1 and 2609.20543v1 identify by which groups of LLM agents "
        "amplify individual deviations?",
        ["2609.18460v1", "2609.20543v1"],
        "2609.18460v1: a seeded deviation spreads when communication lets other agents adopt and "
        "retransmit the unsafe strategy; collective failure emerges when propagation outpaces "
        "correction and containment [2609.18460v1]. 2609.20543v1: LLM groups replaying human "
        "deliberation overstate consensus compared with matched human groups, and measured consensus "
        "is sensitive to how participation and final states are operationalized [2609.20543v1].",
    ),
    (
        "Some research claims that adding more LLM agents brings diminishing returns ('more is "
        "less'). Do any indexed papers offer a concrete mechanism for why extra agents can "
        "increase collective risk rather than capability?",
        None,
        "2609.19759v1 shows that as single-agent capability scales, adding agents yields diminishing "
        "returns plus growing context overhead, and it delineates the capability boundary where "
        "collaboration pays off [2609.19759v1]. 2609.18460v1 supplies a consistent mechanism: extra "
        "agents add communication channels through which seeded deviations contagiously spread, so "
        "more agents can mean more collective risk, not more capability [2609.18460v1].",
    ),
    (
        "How do PACT (2609.18605v1), 2609.18909v1, and Chronicle (2609.20625v1) each evaluate "
        "agents beyond final-task success?",
        ["2609.18605v1", "2609.18909v1", "2609.20625v1"],
        "PACT measures whether enterprise assistants violate compliance rules under pressure from "
        "persistent users, hurried managers, or convenience [2609.18605v1]. 2609.18909v1 uses six "
        "process signals from large-scale trajectories — not just final scores — to compress agent "
        "benchmarks [2609.18909v1]. Chronicle records agent runs at non-deterministic cut points so "
        "code changes can be regression-tested against replays of real trajectories [2609.20625v1].",
    ),
    (
        "2609.20812v1 defines overclaiming as a final response contradicting the agent's own "
        "context. Which other indexed paper supplies process-level evidence to audit such claims, "
        "and how?",
        ["2609.20812v1", "2609.18909v1"],
        "2609.20812v1 defines overclaiming without inferring intent: the final response contradicts "
        "information in the agent's context [2609.20812v1]. 2609.18909v1 supplies the audit signal: "
        "six trajectory-level process signals are systematically associated with final performance, "
        "so a claimed success can be checked against process evidence, not just the reported outcome "
        "[2609.18909v1].",
    ),
    (
        "What design principles for retrieval over agent interaction histories emerge from indexed "
        "work — for instance on enterprise troubleshooting cases and on repair-experience memory "
        "for coding agents?",
        None,
        "Both papers structure history before retrieval rather than retrieving over raw logs: RAFT "
        "abstracts closed cases into stateful directed chains of timeline entries [2609.20754v1], and "
        "AdaRepair-Mem orchestrates curated episodic memory because unfiltered memory volume hurts "
        "repair success [2609.20130v1]. Principle: retrieve over structured, curated, "
        "relevance-filtered representations of history.",
    ),
    (
        "Which indexed papers show risks that emerge from LLM agent collectives rather than single "
        "agents, and what distinct mechanism does each identify?",
        None,
        "2609.18460v1 identifies epidemic contagion: a local seeded deviation is adopted and "
        "retransmitted through agent communication until the collective loses control [2609.18460v1]. "
        "2609.19789v1 identifies adversarial poisoning of the communication fabric: black-box attacks "
        "enter only through admissible social-media feeds and spread across the GMATS multi-agent "
        "trading stack [2609.19789v1]. Both risks are properties of the interconnected system, not "
        "of any single agent.",
    ),
]


def multidoc_examples() -> list[Example]:
    """ids=None marks an open-topic question: no mechanical retrieval gold — the
    label is filled by human adjudication during the EXP-0 mini-pooling
    (docs/plan/ckpt-0.5-plan.md §8)."""
    return [
        _example(
            q,
            answer=a,
            slice_name="multi-doc",
            method="topic-authored",
            gold_arxiv_ids=ids,
            **(_MULTIDOC_OPEN_META if ids is None else {}),
        )
        for q, ids, a in _MULTIDOC_EXAMPLES
    ]


# --- answerable (n=40) & persona (n=10), synthetic-by-construction (§2) ------
# A question is generated FROM a sampled chunk, so that chunk is trivially the
# gold retrieval label. Sampling is seed-fixed, stratified (max 2 chunks/paper)
# to avoid topic collapse. Persona twins share the source chunk and differ only
# in the audience instruction — FP6 calibration, not persona theater (§1 C).
#
# Deviation note: persona bases are sampled from the corpus with their own seed
# instead of reusing answerable rows — the slices stay independently rebuildable
# (a failed answerable run never blocks persona, and vice-versa).

_QA_GEN_PROMPT = """A research-copilot answers user questions from arXiv cs.AI paper abstracts.
Given ONE excerpt of a paper, produce:
1. "question": a specific user question answerable ONLY from THIS excerpt and not
   from other papers in the area. It MUST target a concrete detail present in the
   excerpt: a reported number, a named mechanism, a stated limitation, an
   evaluation setup, or an explicit comparison. Include the distinctive technical
   terms from the excerpt so the question is uniquely identifiable.
   BANNED: definitional or topic-level questions such as "What is the main
   focus/purpose of ...?", "What is the X framework?", "What is the proposed
   method/measure ...?", "What does X do?". GOOD example: instead of "What is the
   proposed measure of robustness?", ask "How does the α-robustness definition
   extend standard social laws for stochastic multi-agent settings?".
   Do not mention "the paper", "this excerpt", or the arXiv ID.
2. "answer": a concise gold reference answer (max 3 sentences), fully supported by
   the excerpt, ending with the citation [{arxiv_id}].

Excerpt (arXiv {arxiv_id}):
\"\"\"{text}\"\"\"

Respond with JSON only: {{"question": "...", "answer": "..."}}"""

_PERSONA_GEN_PROMPT = """Given an excerpt of an arXiv paper and a question, write a reference
answer calibrated for a specific audience.

Excerpt (arXiv {arxiv_id}):
\"\"\"{text}\"\"\"

Question: {question}
Audience: {audience_label} — {audience_guideline}

Rules: max 3 sentences, fully supported by the excerpt, cite [{arxiv_id}] for claims.
Respond with JSON only: {{"answer": "..."}}"""


def _llm_json(client: OpenAI, prompt: str, temperature: float = 0.0) -> dict[str, str]:
    """One generation call parsed as JSON; one retry on parse failure."""
    for attempt in range(2):
        resp = client.chat.completions.create(
            model=config.GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError:
            if attempt == 1:
                raise
    raise AssertionError("unreachable")


def _sample_chunks(chunks: list[Document], n: int, seed: int) -> list[Document]:
    """Stratified sample: shuffle papers, take up to 2 chunks each until n."""
    by_paper: dict[str, list[Document]] = {}
    for c in chunks:
        by_paper.setdefault(c.metadata["arxiv_id"], []).append(c)
    rng = random.Random(seed)
    paper_ids = sorted(by_paper)
    rng.shuffle(paper_ids)
    picked: list[Document] = []
    taken = Counter()
    for take in (2, 1):  # pass 1 takes 2/paper; pass 2 tops up with 1/paper if short
        for pid in paper_ids:
            if len(picked) >= n:
                return picked
            room = min(take, len(by_paper[pid]) - taken[pid], n - len(picked))
            picked.extend(by_paper[pid][taken[pid] : taken[pid] + room])
            taken[pid] += room
    return picked


def answerable_examples(client: OpenAI | None = None) -> list[Example]:
    client = client or OpenAI()
    chunks = _sample_chunks(utils.load_cached_chunks(), config.GOLDEN_SLICE_TARGETS["answerable"], seed=42)
    examples = []
    for c in chunks:
        qa = _llm_json(client, _QA_GEN_PROMPT.format(arxiv_id=c.metadata["arxiv_id"], text=c.page_content))
        q, a = qa["question"].strip(), qa["answer"].strip()
        cid = c.metadata["arxiv_id"]
        if f"[{cid}]" not in a:
            a = f"{a} [{cid}]"
        examples.append(
            _example(
                q,
                answer=a,
                slice_name="answerable",
                method="from-chunk",
                gold_chunk_ids=[utils.chunk_id(c)],
                source_arxiv_id=cid,
            )
        )
    return examples


_AUDIENCES = {
    "pm": ("product manager", "focus on impact, use-cases and trade-offs; no formalism or math"),
    "phd": ("PhD researcher", "include the method/mechanism and its caveats; keep technical terms"),
}


def persona_examples(client: OpenAI | None = None) -> list[Example]:
    client = client or OpenAI()
    chunks = _sample_chunks(utils.load_cached_chunks(), config.GOLDEN_SLICE_TARGETS["persona"] // 2, seed=43)
    examples = []
    for i, c in enumerate(chunks):
        cid = c.metadata["arxiv_id"]
        # one base question per chunk, answered at both audience levels
        qa = _llm_json(client, _QA_GEN_PROMPT.format(arxiv_id=cid, text=c.page_content))
        base_q = qa["question"].strip()
        for aud in ("pm", "phd"):
            label, guideline = _AUDIENCES[aud]
            out = _llm_json(
                client,
                _PERSONA_GEN_PROMPT.format(
                    arxiv_id=cid, text=c.page_content, question=base_q,
                    audience_label=label, audience_guideline=guideline,
                ),
            )
            answer = out["answer"].strip()
            if f"[{cid}]" not in answer:
                answer = f"{answer} [{cid}]"
            examples.append(
                {
                    "inputs": {"question": f"{base_q} (Answer as if explaining to a {label}.)", "audience": aud},
                    "outputs": {
                        "answer": answer,
                        "gold_chunk_ids": [utils.chunk_id(c)],
                        "gold_arxiv_ids": [],
                        "should_abstain": False,
                    },
                    "metadata": {
                        "slice": "persona",
                        "authoring": {"origin": "llm", "method": "from-chunk"},
                        "review": {"state": "draft"},
                        "audience": aud,
                        "base_id": f"persona-{i:02d}",
                        "source_arxiv_id": cid,
                    },
                }
            )
    return examples


SLICE_BUILDERS: dict[str, Callable[[], list[Example]]] = {
    "answerable": answerable_examples,
    "unanswerable": unanswerable_examples,
    "multi-doc": multidoc_examples,
    "format": format_examples,
    "persona": persona_examples,
    "stale": stale_examples,
}


# Provenance honesty → two-axis data model (ADR-001, docs/adrs/): drafts are
# LLM-authored; humans REVIEW, they don't author. So origin=llm for everything
# today; what the human adds is tracked in review.state, not in authorship.
# The old single-field "provenance" mixed origin and review state and allowed
# impossible states ("curated" without review) — removed, not renamed.

_GENERIC_Q_PAT = re.compile(
    r"what is the (main|primary) (focus|purpose|goal|aim)|what is the .* framework|"
    r"^what (is|are|does|do)\b.{0,40}\?$|what (is|are) the proposed",
    re.IGNORECASE,
)


def regenerate_vague_answerable(ls_client: Client, llm_client: OpenAI | None = None) -> int:
    """Regenerate answerable examples whose question matches the generic-question
    heuristic (same source chunk — gold_chunk_id stays valid), updating in-place."""
    llm_client = llm_client or OpenAI()
    ds = ls_client.read_dataset(dataset_name=config.GOLDEN_DATASET_NAME)
    examples = [
        e
        for e in ls_client.list_examples(dataset_id=ds.id, splits=[config.CORE_SPLIT])
        if (e.metadata or {}).get("slice") == "answerable"
        and _GENERIC_Q_PAT.search(e.inputs["question"].lower())
    ]
    chunks = {utils.chunk_id(d): d for d in utils.load_cached_chunks()}
    for e in examples:
        src = chunks[e.outputs["gold_chunk_ids"][0]]
        cid = src.metadata["arxiv_id"]
        # temperature > 0 so repeated attempts actually vary (temp=0 repeats the
        # same banned phrasing); up to 3 attempts before giving up on the example
        for _ in range(3):
            qa = _llm_json(
                llm_client, _QA_GEN_PROMPT.format(arxiv_id=cid, text=src.page_content), temperature=0.8
            )
            q, a = qa["question"].strip(), qa["answer"].strip()
            if not _GENERIC_Q_PAT.search(q.lower()):
                break
        else:
            raise RuntimeError(f"3 regenerations still generic for {cid} — refine prompt")
        if f"[{cid}]" not in a:
            a = f"{a} [{cid}]"
        ls_client.update_example(
            e.id,
            inputs={"question": q},
            outputs=e.outputs | {"answer": a},
            metadata=e.metadata | {"regenerated": "v2-specific-question"},
        )
    return len(examples)


def build_golden_dataset(client: Client | None = None) -> dict[str, int]:
    """Create/sync the core split, one slice at a time. Returns per-slice counts.

    Idempotency contract (docs/plan.md §2): a slice already at its target count is
    skipped; an empty slice is bulk-created; a partially-populated slice raises —
    partial state implies manual edits in the UI that a human must reconcile.
    """
    client = client or Client()
    if client.has_dataset(dataset_name=config.GOLDEN_DATASET_NAME):
        dataset = client.read_dataset(dataset_name=config.GOLDEN_DATASET_NAME)
    else:
        dataset = client.create_dataset(
            config.GOLDEN_DATASET_NAME,
            description=(
                "Two-tier golden set for the arXiv copilot (docs/plan.md §2). "
                "Slices: answerable / unanswerable / deep-hit / multi-doc / format / persona / stale."
            ),
        )

    existing = Counter(
        (e.metadata or {}).get("slice", "<none>")
        for e in client.list_examples(dataset_id=dataset.id, splits=[config.CORE_SPLIT])
    )

    counts: dict[str, int] = {}
    for slice_name, target in config.GOLDEN_SLICE_TARGETS.items():
        n = existing.get(slice_name, 0)
        builder = SLICE_BUILDERS.get(slice_name)
        if n == target:
            logger.info("slice %-12s already at target (%d) — skipping", slice_name, n)
        elif builder is None:
            logger.info(
                "slice %-12s has %d/%d — builder lands in a later mini-checkpoint",
                slice_name, n, target,
            )
        elif 0 < n < target:
            raise RuntimeError(
                f"slice {slice_name!r} partially populated ({n}/{target}) — "
                "resolve in the LangSmith UI or delete the slice before re-running"
            )
        else:
            examples = builder()
            assert len(examples) == target, f"{slice_name}: builder returned {len(examples)} != {target}"
            client.create_examples(
                dataset_id=dataset.id,
                examples=[
                    {
                        "inputs": e["inputs"],
                        "outputs": e["outputs"],
                        "metadata": e["metadata"],
                        "split": [config.CORE_SPLIT],
                    }
                    for e in examples
                ],
            )
            n = len(examples)
            logger.info("slice %-12s created (%d examples)", slice_name, n)
        counts[slice_name] = n

    total = sum(counts.values())
    print(f"Golden dataset '{config.GOLDEN_DATASET_NAME}' (split={config.CORE_SPLIT}):")
    for slice_name, n in counts.items():
        print(f"  {slice_name:<13} {n:>3}/{config.GOLDEN_SLICE_TARGETS[slice_name]}")
    print(f"  {'TOTAL':<13} {total:>3}")
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_golden_dataset()
