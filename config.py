from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set — copy example.env to .env and fill it in")
    return value


# --- Environment ---
OPENAI_API_KEY = _require_env("OPENAI_API_KEY")
LANGSMITH_API_KEY = _require_env("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "capstone-arxiv-copilot")

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR / "resources"  # parquet caches per embedding/chunk config
RESULTS_DIR = BASE_DIR / "results"  # per-checkpoint result reports

# Frozen-corpus manifest (ADR-002/ADR-005): arxiv ids + a digest of each abstract, versioned in git.
# Unlike the git-ignored parquet caches it lets any clone rebuild the exact same corpus by id.
CORPUS_MANIFEST_PATH = RESOURCES_DIR / "corpus_manifest.json"

# --- Corpus (docs/plan.md §1) ---
ARXIV_CATEGORY = "cs.AI"
CORPUS_TARGET_SIZE = 250  # min papers; below this, retrieval recall ~100% and FP2 never manifests

# Snapshot pin (docs/plan/ckpt-0.5-plan.md §4.2): freeze the corpus at this date so the
# `stale` golden slice keeps meaning "papers published after the corpus" across every
# experiment's cache rebuilds. Equals the baseline cache's max published date, so the
# existing parquet stays valid. The CKPT-8 refresh drill bumps this date and re-indexes.
CORPUS_SNAPSHOT_DATE = datetime.date.fromisoformat(
    os.getenv("CORPUS_SNAPSHOT_DATE", "2026-09-17")
)
# Additive discard budget (not multiplicative): as of 2026-09-24 there are ~766
# cs.AI papers published after the snapshot pin, and ~128 more arrive per day. The
# filter needs target+budget raw results to fill the corpus; fetch_arxiv_corpus
# raises with a clear message if the budget ever runs short — bump it here.
CORPUS_SNAPSHOT_SKIP_BUDGET = 2000

# --- Golden dataset (docs/plan.md §2) ---
GOLDEN_DATASET_NAME = "arxiv-copilot-golden"
CORE_SPLIT = "core"
EXTENDED_SPLIT = "extended"
# Target counts per slice for the core split (docs/plan.md §2 table).
GOLDEN_SLICE_TARGETS = {
    "answerable": 40,
    "unanswerable": 15,
    "deep-hit": 15,  # identified empirically from EXP-0 (CKPT-0.8)
    "multi-doc": 15,
    "format": 10,
    "persona": 10,
    "stale": 10,
}

# --- Models ---
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4o-mini")  # CKPT-4 LLM-rerank scorer


@dataclass(frozen=True)
class RunConfig:
    """One arm of one experiment — every field doubles as trace run metadata."""

    app_version: str
    embedding_model: str = "text-embedding-3-small"
    k: int = 5
    chunk_size: int = 0  # 0 = no chunking: one record per whole abstract (ADR-005)
    chunk_overlap: int = 0
    prompt_version: str = "v1"
    query_rewrite: bool = False
    decompose: bool = False
    rerank: bool = False
    rerank_top_n: int = 20  # candidates fetched pre-rerank before cutting to k (CKPT-4)

    def __post_init__(self) -> None:
        if self.chunk_size == 0 and self.chunk_overlap != 0:
            raise ValueError("chunk_overlap requires chunk_size > 0 (chunk_size=0 means whole abstract)")

    @property
    def chunk_strategy(self) -> str:
        if self.chunk_size == 0:
            return "abstract"
        return f"{self.chunk_size}/{self.chunk_overlap}"


# EXP-0 baseline (docs/plan.md CKPT-0, ADR-005): k=5, text-embedding-3-small, ONE RECORD PER WHOLE
# ABSTRACT (chunk_size=0), prompt v1, no rewrite, no rerank. Every later checkpoint's gate compares
# against this. The unit was changed from 500-character chunks to the whole abstract BEFORE EXP-0 was run.
BASELINE = RunConfig(app_version="baseline-ckpt0")

# Experimental arm of CKPT-3 (chunking) and the unit the golden set's answerable/persona questions were
# generated from (docs/learning-lessons/retrieval_unit_and_gold_granularity.md). The golden-set builder
# pins this config so its source-chunk ids keep resolving.
CHUNKED_500_0 = RunConfig(app_version="arm-chunked-500-0", chunk_size=500, chunk_overlap=0)
