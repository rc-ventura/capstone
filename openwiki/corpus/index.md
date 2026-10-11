# Files

- [Frozen Corpus & Manifest](frozen-corpus.md) - How the 250-paper arXiv cs.AI corpus is frozen at a snapshot date and reproduced exactly — the versioned corpus_manifest.json with per-abstract sha1 digests, verified rebuild-by-id, cache and date-scan fallbacks, and the failure behavior of each tier.
- [Indexing & Retrieval](indexing-and-retrieval.md) - How the corpus becomes a queryable index in utils.py — the whole-abstract vs 500-char retrieval units, deterministic chunk_id identity, per-config parquet caches backed by SKLearnVectorStore, and the retriever consumed by the pipeline.
