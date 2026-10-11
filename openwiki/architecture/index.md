# Files

- [Experiment Configuration](experiment-configuration.md) - How config.py models one experiment arm as a frozen RunConfig, defines the BASELINE and CHUNKED_500_0 arms, loads environment variables, and turns every configuration field into LangSmith run metadata for metric slicing.
- [System Overview](overview.md) - What the arXiv Research Copilot is — a cited RAG Q&A assistant over 250 frozen arXiv cs.AI abstracts built as a vehicle for an evaluation system around Barnett et al.'s seven RAG failure points — and how its modules fit together.
- [RAG Pipeline](rag-pipeline.md) - The traced question-answering pipeline in app.py — rewrite, retrieve, rerank, generate — its prompt v1 contract for citations and abstention, multi-turn conversation support, and which stages are live versus reserved extension seams.
