# Plano — Interface web minimalista para o arXiv Research Copilot

## Contexto

Hoje o RAG (`app.arxiv_copilot`) só roda via `main.py` (uma pergunta fixa) ou pelo REPL
(`docs/manual-usage.md`). Falta uma forma rápida e agradável de **conversar com o copiloto,
ver as fontes citadas e gerar traces reais com thread** para inspeção manual. Isso vem antes
do `server.py` (FastAPI, CKPT-8) e não substitui ele: é uma ferramenta local de demo e debug.

Objetivo: um chat clean, de uma página, que:
- responde perguntas usando o pipeline baseline (`config.BASELINE`) sem alterar o comportamento;
- mostra as fontes recuperadas (arXiv ID, título, data, link) num bloco recolhível;
- mantém conversa multi-turn (Feature F) com `thread_id` por sessão no LangSmith.

**Framework: Gradio** (não Streamlit). `gr.ChatInterface(type="messages")` já entrega o
histórico no formato `[{"role", "content"}]` que `generate_response` espera, tem streaming
de UI, exemplos e botão de limpar prontos, e `gr.ChatMessage(metadata={"title": ...})`
renderiza um acordeão nativo — ideal para as fontes sem poluir a tela. Streamlit exigiria
gerenciar `session_state` e re-execução do script na mão para o mesmo resultado.

## Mudanças

### 1. `app.py` — expor as fontes sem mudar o contrato existente
`arxiv_copilot` devolve só a string; a UI precisa dos documentos. Para não duplicar a
orquestração nem recuperar duas vezes:

- Extrair o corpo de `arxiv_copilot` para um helper **não traceado**
  `_run_pipeline(question, run_config, conversation) -> tuple[str, list[Document]]`
  (rewrite → retrieve → rerank → generate, com os mesmos `langsmith_extra` de metadata).
- `arxiv_copilot` continua `@traceable(run_type="chain")`, mesma assinatura, chama o
  helper e retorna só `answer` → árvore de trace, `main.py` e futuros evaluators intactos.
- Nova `arxiv_copilot_with_sources(...)` com
  `@traceable(run_type="chain", name="arxiv_copilot", process_outputs=lambda o: {"output": o[0]})`
  → no LangSmith o run raiz fica idêntico (mesmo nome, mesmo output), mas o chamador
  recebe `(answer, documents)`. (Confirmar `process_outputs` na versão instalada do
  `langsmith` via context7 antes de implementar.)

### 2. `ui.py` (novo, raiz do projeto, ~80 linhas)
- `respond(message, history, request)`:
  - `conversation` = histórico filtrado: só `role`/`content`, **descartando** as mensagens
    de fontes (as que têm `metadata.title`) — senão o bloco de fontes vaza pro prompt.
  - chama `app.arxiv_copilot_with_sources(message, config.BASELINE, conversation,
    langsmith_extra={"metadata": app._run_metadata(config.BASELINE) | {"thread_id": tid}})`
    — mesmo padrão de `main.py:run_pipeline`.
  - retorna `[gr.ChatMessage(answer), gr.ChatMessage(sources_md, metadata={"title": "Fontes (N)"})]`.
- `thread_id`: um `uuid4` por sessão em `gr.State`; renovado quando o usuário limpa o chat
  (`chatbot.clear`).
- `format_sources(documents) -> str` (função pura, testável): dedup por `arxiv_id`
  preservando ordem de rank; cada linha `**[2609.12345]** Título · 2026-09-10 — [abs](source)`.
- Layout minimalista:
  - `gr.Blocks(theme=gr.themes.Base(primary_hue="slate", font=gr.themes.GoogleFont("Inter")),
    fill_height=True)`, coluna central `max-width ~760px` via `css` curto.
  - Cabeçalho de 1 linha: "arXiv Copilot" + subtítulo discreto "~250 papers cs.AI · snapshot
    `CORPUS_SNAPSHOT_DATE`".
  - `gr.ChatInterface` com 3 exemplos (um respondível, um multi-doc, um fora do corpus para
    mostrar a abstenção).
  - Rodapé em cinza pequeno: `app_version · k · GENERATION_MODEL` (lido de `config`).
  - Sem sidebar, sem controles de config — o baseline é fixo nesta versão.
- `if __name__ == "__main__": demo.launch()` (localhost apenas, sem `share=True`).
- Aquecer o vector store no import (`utils.get_vector_store(config.BASELINE)`) para a 1ª
  pergunta não pagar o load do parquet.

### 3. `pyproject.toml`
Adicionar `gradio>=5` às dependências (`uv add gradio`).

### 4. Testes — `tests/test_ui.py`
- `format_sources`: dedup por paper, ordem de rank preservada, lista vazia → mensagem neutra.
- filtro do histórico: mensagens com `metadata.title` são removidas; chaves extras somem.
Sem chamadas a OpenAI/LangSmith (funções puras, no estilo de `tests/test_evaluators.py`).

### 5. Docs
Nova seção em `docs/manual-usage.md`: "Interface web" com `uv run python ui.py` e o que
observar no LangSmith (runs agrupados por `thread_id`).

## Fora de escopo
- 👍/👎 e feedback URLs (CKPT-8, `feedback.py`) — `ChatInterface` aceita `.like()` depois.
- Trocar config/arm pela UI (k, rerank etc.) — entra quando houver mais de um arm funcional.
- Deploy/`share=True`.

## Observação de git
O working tree tem mudanças não commitadas do CKPT-0.6 (`evaluators.py`, testes,
`build_golden_dataset.py`). Implementar a UI em commit separado (ou branch própria) para
não misturar com esse trabalho.

## Verificação
1. `uv run pytest tests/test_ui.py tests/test_evaluators.py` — tudo verde.
2. `uv run python main.py` — `Q:`/`A:` continua igual (contrato de `arxiv_copilot` preservado).
3. `uv run python ui.py` → abrir `http://127.0.0.1:7860`:
   - pergunta de exemplo responde com citações `[arxiv_id]` e acordeão "Fontes (N)";
   - follow-up ("e quais as limitações?") usa o contexto da conversa;
   - pergunta fora do corpus → resposta de abstenção;
   - limpar chat → próxima pergunta gera novo `thread_id`.
4. LangSmith, projeto `capstone-arxiv-copilot`: run raiz `arxiv_copilot` com output string,
   filhos retriever/chain como antes, e as mensagens da sessão agrupadas na aba Threads.
5. Screenshot via Playwright em largura desktop e mobile (~390px) para checar o layout.
