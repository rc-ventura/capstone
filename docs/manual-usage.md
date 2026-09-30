# Uso manual do sistema (sem o golden set)

Como inspecionar o comportamento do copiloto na mão — para auditoria, debug e para
entender o que o golden set vai medir. Todo comando foi testado no CKPT-0.5.

## 0. Setup único

```bash
# .env com OPENAI_API_KEY, LANGSMITH_API_KEY (copie de example.env)
uv run python -c "import config; print('OK')"
```

O corpus é carregado do cache parquet em `resources/` (construído sob demanda na 1ª vez;
com o pin de snapshot não re-baixa papers novos — ADR-002).

## 1. Pipeline completo (corpus → retriever → resposta traceada)

```bash
uv run python main.py
```

Saída esperada: o corpus confirma N chunks, depois `Q:` / `A:` da pergunta embutida.
O trace cai no projeto LangSmith `capstone-arxiv-copilot`.

## 2. Sync idempotente do golden set

```bash
uv run python build_golden_dataset.py
```

Cria slices faltantes, pula as completas; 2ª rodada é noop. Não toca em `app.py`.

## 3. Inspecionar o retriever isolado (REPL)

```python
import utils, config

retriever = utils.get_retriever(config.BASELINE)   # k=5 baseline
for i, doc in enumerate(retriever.invoke("tool hallucination in LLM agents"), 1):
    print(i, doc.metadata["arxiv_id"], doc.metadata["published"], doc.metadata["title"][:70])
```

Troque a pergunta livre. Para k custom: `config.RunConfig(k=10)`.

## 4. Rodar o pipeline com uma pergunta livre

```python
import app, config

print(app.arxiv_copilot("What does TRACE retrieve and why?"))
```

## 5. Inspecionar o golden set via SDK

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

## 6. Onde olhar no LangSmith (UI)

- Projeto `capstone-arxiv-copilot` → traces de `arxiv_copilot` com run_type
  `chain`/`retriever` e metadata (embedding_model, k, chunk_strategy, prompt_version…).
- Dataset `arxiv-copilot-golden` → exemplos com `split=core`, metadata `slice` e `review`.

## Atalhos de diagnóstico úteis

| Quero saber… | Comando |
|---|---|
| quais chunks um chunk_id aponta | `utils.load_cached_chunks()` e filtrar por `utils.chunk_id(d)` |
| o retriever deduplica por paper? | rode o snippet §3 e veja se repete `arxiv_id` (achado em docs/decisions.md) |
| estado de revisão dos 100 | snippet §5 + `Counter(e.metadata["review"]["state"] for e in exs)` |
