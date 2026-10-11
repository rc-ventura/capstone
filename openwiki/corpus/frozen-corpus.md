---
type: data provenance system
title: Frozen Corpus & Manifest
description: How the 250-paper arXiv cs.AI corpus is frozen at a snapshot date and reproduced exactly — the versioned corpus_manifest.json with per-abstract sha1 digests, verified rebuild-by-id, cache and date-scan fallbacks, and the failure behavior of each tier.
tags: [corpus, arxiv, manifest, reproducibility, snapshot, provenance]
verified:
  - by: openwiki/0.6.0
    at: 2026-10-11T10:08:11.054Z
sources:
  - id: openwiki-source-bbbb757758dc109f3a4477fc
    resource: repo://config.py
  - id: openwiki-source-5e8cfa6ef22542c9ea8aaf4a
    resource: repo://docs/decisions.md
  - id: openwiki-source-4b95b55c7ba39533fb7288db
    resource: repo://resources/corpus_manifest.json
  - id: openwiki-source-f5beecb1cfd369c30ed5f9f6
    resource: repo://tests/test_corpus_manifest.py
  - id: openwiki-source-53688c65a9b4548093faff14
    resource: repo://utils.py
generated: { by: "claude-code", at: "2026-10-11T10:08:11.054Z" }
---

# Frozen Corpus & Manifest

The corpus is 250 arXiv cs.AI abstracts frozen at **snapshot date 2026-09-17**. Freezing matters for evaluation semantics: the `stale` golden slice means "papers published after the corpus," and that meaning must survive every cache rebuild and every experiment. The mechanism lives in `utils.py`; the policy constants live in `config.py`.

## Three-tier load order

`load_frozen_corpus()` resolves the corpus in order of preference:

1. **Manifest rebuild** — if `resources/corpus_manifest.json` exists, `papers_from_manifest` fetches the 250 papers **by arXiv id** and verifies each abstract against its recorded digest.
2. **Existing cache fallback** — if no manifest, `_papers_from_existing_cache` reads the first `arxiv_*_*.parquet` in `resources/` and reassembles papers via `papers_from_chunks` (offline path; also bootstraps the manifest).
3. **Cold-start date scan** — `fetch_arxiv_corpus` scans newest-first arXiv results and keeps papers published on or before the pin. This tier also **writes** the manifest.

## The manifest is the source of truth

`resources/corpus_manifest.json` is **versioned in git** (unlike the git-ignored parquet caches) and records `snapshot_date`, `category`, `n_papers`, and per-paper `arxiv_id` / `published` / `abstract_sha1`. Because it stores ids, a fresh clone rebuilds the exact same corpus deterministically — `fetch_arxiv_papers_by_id` queries `arxiv.Search(id_list=...)` in batches of 100 rather than scanning by date.

### Digest semantics

`abstract_digest` is a sha1 over the abstract with **all whitespace collapsed** (`re.sub(r"\s+", " ", text).strip()`) — a line break versus a space does not change it, but any content change does. `papers_from_manifest` fails loudly twice: if any id is not returned by arXiv, and if any fetched abstract's digest differs from the recorded one. Rebuild order follows manifest order regardless of arXiv's response order.

## Why the date scan became a fallback only

The original design fetched the corpus by scanning newest-first and dropping papers newer than the pin (`CORPUS_SNAPSHOT_SKIP_BUDGET = 2000` bounds how many newer papers may be skipped). The decision log records the failure: by 2026-10-04, ~10 days after the budget was sized, all 2,250 scanned results were newer than the pin — cs.AI gains ~128 papers/day — so the scan returned 0 of 250 and raised. That incident produced the manifest: id-based rebuild takes ~8 s for 250 papers and does not degrade as the corpus ages. The date scan remains only as a cold-start path that writes the manifest for future runs.

## Reassembly from chunks and the whitespace caveat

`papers_from_chunks` groups cached chunks by `arxiv_id` and joins their `page_content` with a single space. `chunk_documents` drops the separator at each cut, so the join restores the abstract's words exactly — the only loss is whitespace: a line break at a chunk boundary becomes a space. This happened on 10 of 250 papers; re-chunking a reassembled abstract yields 840 chunks instead of 843, and because gold labels are paper ids it does not affect evaluation. (The digest's whitespace insensitivity is what makes a reassembled abstract still verify against the manifest.)

## Failure behavior summary

| Situation | Behavior |
|---|---|
| Abstract revised upstream after freeze | `RuntimeError`: "abstracts differ from the frozen corpus" |
| Manifest id not returned by arXiv | `RuntimeError`: "manifest papers not returned" |
| Date scan can't fill 250 within skip budget | `RuntimeError`: "raise CORPUS_SNAPSHOT_SKIP_BUDGET" |
| No manifest, no cache | date scan runs (network, increasingly expensive) then writes manifest |
| `load_corpus_manifest` on missing file | returns `None` (silent — triggers next tier) |

## Relationship to the index

`load_frozen_corpus` only produces `Document`s — chunking, embedding, and caching happen downstream in `get_vector_store`, and only when the requested arm's parquet cache is missing. See [Indexing & Retrieval](./indexing-and-retrieval.md).

## Representative tests

`tests/test_corpus_manifest.py` pins: whitespace-insensitive digests, manifest round-trip, manifest-order rebuild, and both fail-loud paths (missing id, revised abstract).
