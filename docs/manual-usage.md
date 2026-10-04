# Manual usage of the system (without the golden set)

How to inspect the copilot's behavior by hand — for auditing, debugging, and to
understand what the golden set will measure. Every command was tested in CKPT-0.5.

## 0. One-time setup

```bash
# .env with OPENAI_API_KEY, LANGSMITH_API_KEY (copy from example.env)
uv run python -c "import config; print('OK')"
```

The corpus is rebuilt on the first use **by arXiv id** from `resources/corpus_manifest.json` (verified against
the recorded abstract digests; ADR-002, ADR-005) and cached as parquet in `resources/` (git-ignored).

## 1. Full pipeline (corpus → retriever → traced answer)

```bash
uv run python main.py
```

Expected output: the corpus confirms N chunks, then `Q:` / `A:` of the built-in question.
The trace lands in the LangSmith project `capstone-arxiv-copilot`.

## 2. Idempotent sync of the golden set

```bash
uv run python build_golden_dataset.py
```

Creates missing slices, skips complete ones; the 2nd run is a no-op. Does not touch `app.py`.

## 3. Inspect the retriever in isolation (REPL)

```python
import utils, config

retriever = utils.get_retriever(config.BASELINE)   # k=5 baseline
for i, doc in enumerate(retriever.invoke("tool hallucination in LLM agents"), 1):
    print(i, doc.metadata["arxiv_id"], doc.metadata["published"], doc.metadata["title"][:70])
```

Swap in any free-form question. For a custom k: `config.RunConfig(k=10)`.

## 4. Run the pipeline with a free-form question

```python
import app, config

print(app.arxiv_copilot("What does TRACE retrieve and why?"))
```

## 5. Inspect the golden set via SDK

```python
from langsmith import Client
import config

c = Client()
ds = c.read_dataset(dataset_name=config.GOLDEN_DATASET_NAME)
exs = list(c.list_examples(dataset_id=ds.id, splits=[config.CORE_SPLIT]))
print("total core:", len(exs))
e = exs[0]
print("slice:", e.metadata["slice"], "| review:", e.metadata["review"])
print("Q:", e.inputs["question"])
print("G:", e.outputs["answer"])
```

## 6. Where to look in LangSmith (UI)

- Project `capstone-arxiv-copilot` → `arxiv_copilot` traces with run_type
  `chain`/`retriever` and metadata (embedding_model, k, chunk_strategy, prompt_version…).
- Dataset `arxiv-copilot-golden` → examples with `split=core`, metadata `slice` and `review`.

## Useful diagnostic shortcuts

| I want to know… | Command |
|---|---|
| which chunks a chunk_id points to | `utils.load_cached_chunks()` and filter by `utils.chunk_id(d)` |
| does the retriever dedup by paper? | run the §3 snippet and check whether `arxiv_id` repeats (finding in docs/decisions.md) |
| review state of the 100 | §5 snippet + `Counter(e.metadata["review"]["state"] for e in exs)` |
