from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


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

# --- Corpus (docs/plan.md §1) ---
ARXIV_CATEGORY = "cs.AI"
CORPUS_TARGET_SIZE = 250  # min papers; below this, retrieval recall ~100% and FP2 never manifests

# --- Golden dataset (docs/plan.md §2) ---
GOLDEN_DATASET_NAME = "arxiv-copilot-golden"
CORE_SPLIT = "core"
EXTENDED_SPLIT = "extended"

# --- Models ---
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4o-mini")  # CKPT-4 LLM-rerank scorer


@dataclass(frozen=True)
class RunConfig:
    """One arm of one experiment — every field doubles as trace run metadata."""

    app_version: str
    embedding_model: str = "text-embedding-3-small"
    k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 0
    prompt_version: str = "v1"
    query_rewrite: bool = False
    decompose: bool = False
    rerank: bool = False
    rerank_top_n: int = 20  # candidates fetched pre-rerank before cutting to k (CKPT-4)

    @property
    def chunk_strategy(self) -> str:
        return f"{self.chunk_size}/{self.chunk_overlap}"


# EXP-0 baseline (docs/plan.md CKPT-0): k=5, text-embedding-3-small, chunk 500/0,
# prompt v1, no rewrite, no rerank. Every later checkpoint's gate compares against this.
BASELINE = RunConfig(app_version="baseline-ckpt0")
