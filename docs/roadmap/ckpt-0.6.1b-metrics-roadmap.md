# Roadmap vivo — CKPT-0.6.1b: correção e fortalecimento das métricas

> **Documento iterativo (versão 2, 2026-10-03).** Reescrito a cada rodada de discussão; a versão corrente é sempre esta.
> Regras de redação: toda métrica citada vem com o que mede entre parênteses quando isso não é óbvio; todo "erro de
> desenho" é explicado em linguagem simples, com exemplo numérico, **alternativas pesquisadas antes da recomendação**
> e a decisão registrada no log da §5. Nada abaixo de P5 foi implementado: são **propostas para discussão**.

## 1. Estado atual

| Ponto | Status | Observação |
|---|---|---|
| P1 MRR deslocado (gold por paper) | ✅ feito, **não commitado** | `evaluators.py` |
| P2 precision = 0 (gold por paper) | ✅ feito, **não commitado** | idem |
| P3 níveis chunk × paper misturados (causa de P1/P2) | ✅ feito, **não commitado** | helper `_ranked_and_gold`; oráculo `ranx` em teste |
| P4 `f1_summary_evaluator` com formato inválido | ✅ feito, **não commitado** | assinatura + retorno `{"results": [...]}` |
| P5 hit@k ≡ recall@k com 1 gold | 🟡 **proposta reescrita** (§6) | aguarda sua decisão D-2 |
| P6 `precision_at_k` com gold de 1 chunk | 🟡 **proposta reescrita** (§6) | aguarda D-1 e D-4 |
| P7 detecção de abstenção por `startswith` | 🟡 **proposta reescrita** (§6) | aguarda D-5 |
| P8 abstenção como acurácia | 🟡 **proposta reescrita** (§6) | aguarda D-6 |
| P9–P11 | 🟠 rascunho, **atualizado com as fichas** (§7) | discutir depois de P5–P8 |
| P12–P17 (novos, surgidos das fichas) | 🆕 listados na §7 | triagem pendente |
| Fichamentos | ✅ **15/15 gravados** | §4 |

Verificação do lote já feito: `uv run pytest` → 35 testes verdes; MRR (Mean Reciprocal Rank: 1 ÷ posição do primeiro
documento relevante) de 0,167 → 1,0 e precision@k (fração do top-k que é relevante) de 0,0 → 0,2 nos casos que
reproduziam o bug; 600 casos aleatórios concordam com o `ranx`.

## 2. Glossário de métricas (o que cada uma mede)

| Métrica | O que mede, em uma frase |
|---|---|
| **hit@k** (acerto) | O top-k trouxe **pelo menos um** documento correto? Vale 1 (sim) ou 0 (não). Nos padrões de IR chama-se *success@k*. |
| **recall@k** (cobertura) | De **todos** os documentos corretos, que fração apareceu no top-k? (2 corretos, achou 1 → 0,5) |
| **MRR** (Mean Reciprocal Rank — "quão no topo está o primeiro correto") | 1 ÷ posição do primeiro correto. 1.º → 1,0; 2.º → 0,5; 3.º → 0,33; 5.º → 0,2; ausente → 0. |
| **precision@k** (pureza) | Do que foi recuperado, que fração é correta? (5 recuperados, 1 correto → 0,2) |
| **nDCG** (normalized Discounted Cumulative Gain — qualidade do ranking inteiro) | Premia correto no topo e aceita graus de relevância. Com 1 relevante binário vale 1/log₂(posição+1): 1.º → 1,0; 2.º → 0,63; 3.º → 0,5; 5.º → 0,39. |
| **Hole@k** (taxa de buracos) | Fração do top-k que **ninguém julgou** (nem é gold, nem foi marcado irrelevante). Mede quanto do que o sistema recuperou está "no escuro". |
| **distinct_papers@k** (diversidade) | Papers distintos ÷ k. 5 chunks do mesmo paper → 0,2. Mede redundância, **não** relevância. |
| **abstention recall** (abstenção correta) | Dos exemplos em que o sistema **deveria** se abster, quantos se abstiveram. É o gate do plano (≥ 90% em `unanswerable`). |
| **over-refusal rate** (recusa indevida) | Dos exemplos **respondíveis**, quantos o sistema recusou. Mede inutilidade. |
| **under-refusal rate** (resposta indevida) | Dos exemplos **irrespondíveis**, quantos o sistema respondeu mesmo assim (= 1 − abstention recall). Mede risco de alucinação. |
| **F1 de classificação** (equilíbrio entre acertos e falsos alarmes) | Média harmônica de precision e recall de uma classe. |
| **token-F1** (sobreposição de palavras) | F1 sobre as palavras da resposta gerada × resposta-gabarito. |
| **faithfulness** (fidelidade ao contexto) | A resposta só afirma o que o contexto recuperado sustenta? |
| **κ de Cohen** (concordância além do acaso) | Quanto dois avaliadores (ex.: juiz-LLM e humano) concordam descontando a concordância por sorte. |
| **IC de Wilson** (intervalo de confiança para proporções) | Faixa plausível da taxa verdadeira dado o n pequeno. 10 acertos em 10 → [72%; 100%]. |

**Oráculo de teste:** implementação de referência usada só para conferir se a nossa dá o mesmo resultado. Não faz parte do produto.

## 3. Por que o projeto usa IDs de chunk **e** IDs de paper? (pergunta 3 do usuário)

**Fatos do corpus (medidos em `resources/*.parquet`):** 250 papers, 843 chunks, de 2 a 5 chunks por paper
(média 3,4), ~420 caracteres por chunk. Cada paper é só o *abstract*, cortado em pedaços de ~500 caracteres.

**O retriever devolve chunks**, não papers. O que muda é **como o gabarito (gold) de cada exemplo foi construído**
(ADR-001, ADR-003):

| Slice | Como a pergunta nasceu | O que sabemos com certeza | Gold que dá para gravar |
|---|---|---|---|
| `answerable` (40), `persona` (10) | Um LLM leu **um chunk** e gerou a pergunta (*synthetic-by-construction*) | Qual **trecho exato** responde | `gold_chunk_ids` (1 chunk) |
| `multi-doc` known-item (10) | "Compare o paper A e o paper B" | Quais **papers** — mas não qual chunk de cada | `gold_arxiv_ids` (2–3 papers) |
| `multi-doc` open-topic (5) | Pergunta temática, sem nomear papers | Nada ainda | vazio até a adjudicação humana (EXP-0) |
| `unanswerable` (15), `stale` (10), `format` (10) | Fora do corpus / sem retrieval | Nada a recuperar | vazio |

Não é escolha de arquitetura: é consequência de dois jeitos de construir o dataset. Os avaliadores precisam
comparar **no nível em que o gold existe**, e misturar os dois níveis causou os bugs P1–P3.

**Nova leitura depois das fichas (decisão D-1):** o gold de 1 chunk é, na verdade, um **conjunto de relevantes
incompleto** — os outros 2–4 chunks do mesmo paper podem também responder. Logo:
- métricas em **nível de chunk** são um **piso** (só contam o trecho exato);
- métricas em **nível de paper** são um **teto** (qualquer chunk do paper conta);
- a literatura de IR (Büttcher 2007, NIST 2007, BEIR §6) recomenda **julgar os "buracos"** em vez de escolher um nível.
Proposta: manter os dois níveis, **reportados lado a lado como faixa [piso, teto]** (sem média combinada), e adjudicar
os chunks-irmãos no mini-pooling do EXP-0 (~200–250 julgamentos, **estimativa do agente de IR, ainda não conferida
contra o golden set**). Ver `ir-pooling-incomplete-judgments.md` (reflexão 3).

## 4. Fichamentos (leitura integral dos artigos) — 15/15 gravados

Pasta: `docs/research/fichamentos/`. Todas seguem: referência · problema · método · contribuições · números com
seção/tabela · limitações · **reflexões ancoradas no nosso plano** · confiança da leitura.

| Ficha | Confiança | Usada em |
|---|---|---|
| [`yu-2024-rag-eval-survey.md`](../research/fichamentos/yu-2024-rag-eval-survey.md) | integral (HTML; sem Apêndice 0.A) | catálogo de métricas |
| [`es-2023-ragas.md`](../research/fichamentos/es-2023-ragas.md) | integral (HTML) | P9, juízes |
| [`saad-falcon-2023-ares.md`](../research/fichamentos/saad-falcon-2023-ares.md) | integral (HTML) | P11, calibração |
| [`ru-2024-ragchecker.md`](../research/fichamentos/ru-2024-ragchecker.md) | integral (HTML; Tab. 16 e figuras não vistas) | D-1, P6, P9 |
| [`barnett-2024-seven-failure-points.md`](../research/fichamentos/barnett-2024-seven-failure-points.md) | integral (v1) | mapeamento FP→métrica (P13) |
| [`magesh-2025-hallucination-free.md`](../research/fichamentos/magesh-2025-hallucination-free.md) | integral (**preprint v1**; versão publicada não lida) | P12, P7/P8, P11 |
| [`gao-2023-alce.md`](../research/fichamentos/gao-2023-alce.md) | integral (v2) | P12 |
| [`zheng-2023-llm-as-judge.md`](../research/fichamentos/zheng-2023-llm-as-judge.md) | integral (v4) | P11, P14 |
| [`wataoka-2024-self-preference.md`](../research/fichamentos/wataoka-2024-self-preference.md) | integral (v2); Fig. 1b só GPT-4 | P11 |
| [`qa-eval-judge-vs-f1.md`](../research/fichamentos/qa-eval-judge-vs-f1.md) | integral (corpo); apêndices C–E do 2305.12421 não lidos | P9 |
| [`ir-pooling-incomplete-judgments.md`](../research/fichamentos/ir-pooling-incomplete-judgments.md) | integral; tabelas do Büttcher parciais | D-1, P5, P6 |
| [`thakur-2021-beir.md`](../research/fichamentos/thakur-2021-beir.md) | integral (v4) | P5, P6 |
| [`bassani-2022-ranx.md`](../research/fichamentos/bassani-2022-ranx.md) | **PARCIAL** (paper completo bloqueado: pôster, docs e código) | L-3 |
| [`abstention-benchmarks.md`](../research/fichamentos/abstention-benchmarks.md) | integral (AbstentionBench, RefusalBench) + extras parciais | P7, P8 |
| [`zhou-2023-ifeval.md`](../research/fichamentos/zhou-2023-ifeval.md) | integral (v1); código dos checadores não lido | P10 |

## 5. Log de decisões

| ID | Data | Decisão / pergunta | Base | Status |
|---|---|---|---|---|
| L-1 | 2026-10-03 | Comparar sempre **no nível do gold**; deduplicar papers preservando a ordem | Bugs P1–P3; `decisions.md` (mesmo paper 3× no top-5) | Implementado |
| L-2 | 2026-10-03 | Summary evaluator segue o contrato do LangSmith: `(outputs, reference_outputs)` → `{"results":[{key,score}]}` | `langsmith` 0.13.0 (`extra="forbid"`) + curso | Implementado |
| L-3 | 2026-10-03 | `ranx` como **oráculo de teste (dev)** | `bassani-2022-ranx.md` (parcial) + teste empírico abaixo | **Aguarda você** (ver nota) |
| D-1 | — | Dois níveis (chunk=piso, paper=teto) lado a lado + adjudicar irmãos no EXP-0? | §3, `ir-pooling…`, `thakur…` | Proposta |
| D-2 | — | Em gold único reportar hit@k + MRR (+ curva hit@1/3/5) e recall só em multi-gold? | P5 | Proposta |
| D-3 | — | Rodar BM25 em paralelo a todo retriever novo para detectar viés lexical do gold sintético? | `thakur…` reflexão 4 (analogia) | Proposta |
| D-4 | — | `precision_at_k` só com gold completo + `distinct_papers@k` à parte? | P6 | Proposta |
| D-5 | — | Abstenção: juiz LLM como verdade + heurística como baseline barato; marcador estruturado só no CKPT-6? | P7 | Proposta |
| D-6 | — | Abstenção: gate = recall **com** guarda-corpo obrigatório de over-refusal, por slice, com IC? | P8 | Proposta |

**Nota L-3 (teste empírico feito em ambiente descartável, sem tocar no projeto):**

| | `ranx` (instalado hoje) | `ir-measures` + `pytrec-eval-terrier` |
|---|---|---|
| Pacotes novos | **22** (numba, llvmlite, matplotlib, seaborn, ir-datasets…) | **4** (ir-measures, numpy, pytrec-eval-terrier, scipy) |
| Instala no Python 3.13 do projeto | sim | **sim** (testado: resolve e roda) |
| `success@k` vs `recall@k` com 1 gold | `hit_rate` ≡ recall | `Success@5 = R@5 = 1.0` nos dois casos testados (confirma P5 na prática) |
| `precision@k` | ÷ k | ÷ k (`P@5 = 0.2` com 1 acerto) |
| Base | pôster: "testado contra trec_eval"; paper não lido | wrapper do próprio `trec_eval` (referência da comunidade) |
Ambos concordam com as nossas fórmulas nos casos testados. Recomendação: **trocar para `ir-measures`** (mesma garantia
com ~5× menos dependências) e manter o `ranx` fora do lock; o teste de propriedade continua igual. Só o `ranx` dá os
testes pareados (t, Fisher, Tukey) — se você quiser significância estatística (P15), reavalie.

### Correções a documentos existentes (propostas; **nenhuma aplicada**)

| # | Onde | O que está lá | Correção | Fonte |
|---|---|---|---|---|
| C-1 | `docs/learning-lessons/golden_dataset_…` (§Design consequence) | "bpref-style tolerance" | bpref exige não-relevantes **julgados** (nosso gold só tem positivos) e superestima sistemas fora do pool; a tolerância correta é "não penalizar não-julgado" | `ir-pooling…` reflexão 2 |
| C-2 | mesma lesson, L3 (linha 161) | a revisão humana do golden set vira o "conjunto de calibração de juízes" | A revisão julgou **perguntas/gold**, não **respostas geradas**; calibrar juízes exige rotular respostas reais (ver P11) | `results/ckpt-0.5-review.md`; ARES; Zheng |
| C-3 | `docs/plan.md` / 0.6 §3.1 | FP3 = "ruído"; `faithfulness`/`citation_accuracy` = FP4 | No Barnett, FP3 = limite de consolidação (recuperado mas não coube no contexto); FP4 = omissão (resposta no contexto, não extraída). `precision_at_k` não mede FP3 | `barnett-2024…` |
| C-4 | `docs/research/rag_failure_modes_review.md` | "15k docs"; "17–34%" | Barnett: 4.017 docs no §4.3 (o §1 diz 15.000: inconsistência do paper); Magesh v1: 17–33% | `barnett…`, `magesh…` |
| C-5 | roadmap §6 (rascunho antigo) e plano 0.6 | `ranx` "validado em ECIR/CIKM/SIGIR" | Só o ECIR 2022 trata de avaliação; CIKM 2022 = fusão (`ranx.fuse`), SIGIR 2023 = repositório de runs (`ranxhub`) | `bassani…` |
| C-6 | rascunho P9 / fundamentação | "token-F1 pune correta e premia errada" | Os artigos mostram só que **subestima respostas corretas**; 0,22/0,40/0,85 são médias de Pearson sobre 1.288 julgamentos binários de QA **extrativo** | `qa-eval-judge-vs-f1.md` |
| C-7 | rascunho "ARES: 59,3 p.p." | 59,3 | O paper usa 59,3 (§1) e 59,9 (§5.1); a média da Tab. 1 dá 59,9 | `saad-falcon-2023-ares.md` |
| C-8 | plano 0.6 §4 (formato do juiz) | JSON `{"score","reason"}` | A nota vem **antes** da justificativa, anulando o efeito de explicar primeiro (Zheng); mas "dar razões" piorou o GPT-3.5 em Wang → testar localmente | `zheng…`, `qa-eval…` |

**Achados do curso `intro-to-langsmith` (módulo 2):** o summary evaluator do curso é um **F1 de classificação** (TP/FP/FN),
com assinatura `(outputs: list[dict], reference_outputs: list[dict])`; o nosso calculava token-F1. O juiz do curso é testado
com um par de sentido **oposto** ("Yes… integrated" × "No… NOT integrated"). O curso lê `outputs["output"]` e o nosso
avaliador lê `outputs["answer"]` → o runner do 0.7 precisa devolver `answer`. O curso não tem métricas de retrieval.

---

## 6. Propostas reescritas: P5, P6, P7, P8

> Formato de cada ponto: **(1) o que está implementado · (2) o problema em linguagem simples · (3) evidência ·
> (4) o que a literatura diz (com ficha) · (5) alternativas · (6) recomendação · (7) passos · (8) o que preciso de você.**

---

### P5 — hit@k e recall@k são o mesmo número quando o gold tem 1 documento

**(1) Implementado.** `hit_rate` e `recall_at_k` em `evaluators.py`, calculadas separadamente e destinadas a sair como duas colunas.

**(2) Em linguagem simples.** hit@k pergunta "o top-k trouxe algum documento correto?". recall@k pergunta "que fração dos
documentos corretos apareceu?". Se só existe **um** correto, a "fração" só pode ser 0/1 ou 1/1 — exatamente a resposta do hit.
Isso **não é bug de conta**; é redundância que pode enganar a leitura do relatório (duas colunas "concordando" que na verdade
são uma).

| Situação (gold = 1 chunk) | hit@5 | recall@5 |
|---|---|---|
| Chunk correto no top-5 | 1 | 1 |
| Chunk correto fora do top-5 | 0 | 0 |
| (multi-doc, 2 papers gold, achou 1) | 1 | **0,5** ← só aqui diferem |

**(3) Evidência.** `answerable` (40) + `persona` (10) = **50 dos 100 exemplos** têm gold de 1 chunk. Confirmado também empiricamente
com `ir-measures`: `Success@5 = R@5 = 1.0` e ambos `0.0` com gold ausente. Efeito sutil: como o gold é "este chunk exato", o hit
vira 0 mesmo se o sistema trouxer **outro chunk do mesmo paper** que responde — a métrica mede "achou o trecho exato", não
"achou evidência útil" (isso é o D-1).

**(4) Literatura.**
- `thakur-2021-beir.md`: o BEIR **não define hit rate**; descarta precision/recall por serem "rank unaware" (não consideram a posição) e adota nDCG@10. Com 1 relevante binário, nDCG@k = 1/log₂(posição+1), isto é, um MRR com desconto mais suave (derivação do agente, conferida na tabela do glossário).
- `bassani-2022-ranx.md` (código) e `ir-measures`: nomeiam `success` e `recall` separadamente (trec_eval convention), embora coincidam com 1 relevante.
- `ir-pooling…` (B&V §4): com poucos relevantes por consulta todas as medidas ficam instáveis; com 40–50 exemplos de resultado binário o ruído amostral é grande. **Nenhuma fonte recomenda MRR para gold único** — isso é convenção do campo, não resultado.

**(5) Alternativas.**
| | O que faria | Prós | Contras |
|---|---|---|---|
| A | hit@k + MRR em gold único; recall só em multi-gold | sem duplicação; MRR traz a posição | exige regra "qual métrica é primária por slice" |
| B | curva **hit@1 / hit@3 / hit@5** | informa o efeito de k (experimento CKPT-1) quase de graça | 3 números por exemplo |
| C | nDCG@k no lugar do MRR | padrão do BEIR | redundante com MRR em gold binário de 1 item; só vale com graus de relevância |
| D | ampliar o gold (irmãos adjudicados, D-1) | recall volta a ser informativo e o "piso" sobe | custo humano (~200–250 julgamentos, estimativa) |
| E | manter as duas e só documentar | zero esforço | leitura enganosa |

**(6) Recomendação (D-2): A + B agora; D em paralelo via D-1.** nDCG fica para quando existir relevância graduada (CKPT-4).

**(7) Passos (se aprovado).**
1. `hit_rate(..., k: int | None = None)`: corta `ranked[:k]` antes de comparar (hoje usa o top-k inteiro recebido). Idem MRR/recall para consistência.
2. Helper `primary_retrieval_metrics(reference_outputs) -> list[str]`: `["hit", "mrr"]` se `|gold| == 1`; `["recall", "hit", "mrr"]` se `|gold| ≥ 2`. O runner (0.7) usa isso para decidir o que reporta por slice.
3. Docstrings e `docs/plan.md` §3: "recall e hit só divergem com |gold| ≥ 2".
4. Testes: identidade recall≡hit com gold único; divergência com 2 golds; `hit@1/3/5` em listas conhecidas. Oráculo (L-3) cobre `Success@k`.

**(8) Preciso de você:** concorda com A+B? Quer a curva hit@1/3/5 ou só hit@5?

---

### P6 — `precision_at_k` com gold de 1 chunk: teto de 1/k e pune quem não foi julgado

**(1) Implementado.** `precision_at_k` roda em qualquer exemplo com gold não vazio e conta tudo fora do gold como erro. O
docstring admite a restrição ("meaningful only where the gold is the complete relevant set"), mas o código não a impõe.
Após P1–P3 o denominador em nível de paper é o nº de papers **distintos** recuperados.

**(2) Em linguagem simples.** Precision pergunta "do que o sistema trouxe, quanto prestava?". Mas só sabemos que **1 chunk**
prestava — os outros 4 do top-5 podem prestar também (3,4 chunks por paper em média; o `decisions.md` já registrou o
mesmo paper 3× no top-5). Contá-los como "erro" é tratar **"não julgado" como "irrelevante"** — e com gold de 1 chunk a nota
máxima possível é 1/5 = 0,2, mesmo com retrieval perfeito.

**(3) Evidência.** Reproduzido: gold de 1 chunk com acerto em 1.º → `precision = 0.2` (máximo possível). Hoje o plano 0.6 §3.1 ainda diz que
precision mede "ruído (FP3)"; o Barnett define FP3 de outra forma (ver C-3).

**(4) Literatura.**
- `ir-pooling…` (NIST 2007 §7; Büttcher 2007 §5.2): tratar não-julgado como irrelevante dá uma **cota inferior** (conservadora); **ignorar** não-julgados (bpref, P@k(j), listas condensadas) **não é neutro** e superestima. Citação: "Ignoring the unjudged documents… assumes… the same proportion of relevant… — an assumption that is simply wrong."
- `thakur-2021-beir.md` (§6, Tab. 4): quando o sistema recupera algo **sem julgamento**, o score cai sem erro dele, e o efeito é desigual (BM25 +0,012 vs ANCE +0,081 ao anotar os buracos). Solução do BEIR: **anotar os buracos** e medir Hole@10.
- `barnett-2024…`: FP3 = "recuperado mas não entrou no contexto" (limite de consolidação) — com k=5 chunks de ~420 caracteres é praticamente inexistente no nosso sistema. **`precision_at_k` não mede FP3.**
- `ru-2024-ragchecker.md`: *context precision* por **conteúdo** (um chunk é relevante se sustenta alguma claim do gabarito), não por ID — contorna chunk-irmão e quebra de ID por re-chunking, **mas as métricas diagnósticas do RAGChecker não foram validadas com humanos** no paper.
- `saad-falcon-2023-ares.md`: trata passagens do **mesmo documento** como negativos difíceis — exatamente a premissa que o nosso corpus de abstracts fragmentados desmente.

**(5) Alternativas.**
| | O que faria | Prós | Contras |
|---|---|---|---|
| A | `precision_at_k` só com **gold completo** (flag `gold_complete`); `None` caso contrário | honesto; segue NIST/Büttcher | some a métrica em `answerable` |
| B | `distinct_papers@k` como métrica separada (diversidade) | mede o colapso observado (mesmo paper 3×) sem fingir relevância | não diz se o conteúdo presta |
| C | **julgar os buracos** (Hole@k) no EXP-0 e só então calcular precision de chunk | é o que a literatura recomenda; sobe o piso do gold | custo humano |
| D | context precision por conteúdo (RAGChecker/RAGAS) | independe de ID e de chunking | custo de LLM; não validado em humanos; usaria gpt-4o-mini = gerador |
| E | manter como está | zero esforço | métrica enganosa |

**(6) Recomendação (D-4): A + B agora; C no EXP-0; D só como diagnóstico para os 5 `open_topic`** (que não têm gold até a adjudicação).
Além disso: **renomear o papel da métrica** no plano de "ruído/FP3" para "pureza do top-k (só com gold completo)".

**(7) Passos.**
1. `build_golden_dataset.py` grava `reference.gold_complete` (`True` só em multi-doc known-item); migração idempotente dos 100 exemplos.
2. `precision_at_k` retorna `None` se `gold_complete` for falso/ausente.
3. Nova `distinct_papers_at_k(retrieved_chunk_ids, retrieved_arxiv_ids)` (pura, sem gold).
4. `hole_rate_at_k(ranked, judged_ids)` (preparada; usada no EXP-0).
5. ADR-005 curto: "semântica de precision" + correção C-3 no plano. Testes: gold de 1 chunk ⇒ `None`; multi-doc known ⇒ valor; mesmo paper 3× ⇒ `distinct_papers@5 = 0,6`.

**(8) Preciso de você:** (i) concorda em renomear o papel de precision? (ii) aceita julgar os buracos no EXP-0 (D-1/C)? (iii) a diversidade (B) entra como métrica oficial ou só diagnóstico?

---

### P7 — Detectar abstenção por `startswith`

**(1) Implementado.** Dentro de `f1_summary_evaluator`: `answer.strip().startswith(("I don't know", "No papers found"))`.

**(2) Em linguagem simples.** O avaliador só reconhece a recusa se a resposta **começar** com uma de duas frases exatas. Mas o prompt
(`app.py`, `PROMPT_V1`) só diz "say you don't know", sem fixar frase, e **nunca** instrui "No papers found" (essa frase só existe no gabarito de `stale`).
Qualquer paráfrase ("I'm sorry, the context doesn't say…") conta como **não** abstenção.

**(3) Evidência.** Reproduzido: `"I'm sorry, the context doesn't say."` → `abstention_accuracy = 0.0` quando devia ser 1.0.

**(4) Literatura.**
- `abstention-benchmarks.md` (AbstentionBench, Meta FAIR 2025): define abstenção de forma **ampla** (não responder diretamente, expressar incerteza, caveats, resposta parcial); gastou um apêndice justificando **não** usar match de string e adotou **juiz LLM (Llama 3.1 8B, T=0)** validado em **300 pares** anotados e estratificados (88% de acurácia).
- Mesma ficha (RefusalBench): elimina a ambiguidade na origem — o modelo emite **só** um código `REFUSE_*`; um juiz apenas lê.
- Mesma ficha (OR-Bench, leitura parcial): keyword matching diverge ≤ 2,4% de um juiz GPT-4 — **mas em recusas de segurança**, frases padronizadas; não vale para "I don't know" aberto.
- `magesh-2025…`: distingue **recusa informativa** (explica por quê) de **recusa padrão**; rotulagem humana especializada.

**(5) Alternativas.**
| | O que faria | Prós | Contras |
|---|---|---|---|
| A | heurística robusta (conjunto de paráfrases + normalização) | grátis, determinística, testável offline | nunca cobre tudo |
| B | **juiz LLM** (`abstention_quality`, 0.6.3) como fonte de verdade, validado em amostra humana estratificada | é o padrão dos benchmarks | custo; precisa de validação própria (88% deles não se transfere) |
| C | marcador estruturado no prompt (`REFUSE`/JSON) | elimina ambiguidade | **muda o produto**: o baseline deve medir o prompt real |
| D | híbrido A + B | A como baseline barato e como medida de divergência | duas implementações |

**(6) Recomendação (D-5): D — B como fonte de verdade, A como baseline/guarda barata; C só como experimento do CKPT-6.** O juiz deve ver só a resposta
(testar antes se passar `should_abstain` ao juiz enviesa); T=0.

**(7) Passos.**
1. `abstained(answer) -> bool` (função pura): normalização (caixa, aspas tipográficas) + padrões ("don't know", "do not know", "cannot find", "no papers found", "not in the context"…); tabela de testes com ≥ 15 paráfrases de recusa e ≥ 10 respostas não-recusa que contêm "know".
2. `f1_summary_evaluator` passa a usar `abstained` (mudança mínima).
3. No 0.6.3, juiz `abstention_quality` com saída estruturada; **sua** rotulagem de ~50 respostas reais estratificadas (`should_abstain × predição`) valida o juiz (junto da calibração de P11).
4. Relatório do EXP-0 mostra a **divergência heurística × juiz** (alvo ≥ 90% de concordância).

**(8) Preciso de você:** aceita D? Quantas respostas reais você topa rotular (a literatura usa 300 pares; propus ~50 só para abstenção, dentro de uma amostra maior em P11)?

---

### P8 — Abstenção medida como acurácia (e não como recall + over/under-refusal)

**(1) Implementado.** `abstention_accuracy = acertos / total` sobre `should_abstain`, junto do token-F1 em `f1_summary_evaluator`.

**(2) Em linguagem simples.** Há **dois erros opostos**, e a acurácia os mistura num número só:
- **under-refusal** (resposta indevida): responder algo que **não** estava no corpus → risco de alucinação;
- **over-refusal** (recusa indevida): recusar algo **respondível** → o produto fica inútil.
O gate do plano ("abstenção correta ≥ 90% em `unanswerable`") é, na prática, o **recall de abstenção**. Só que recall sozinho é **gameável**.

**(3) Evidência (com os números do nosso golden set core, 100 exemplos).** 25 deveriam abster (`unanswerable` 15 + `stale` 10); 75 não.
| Sistema "bobo" | Acurácia | Abstention recall (gate) | Over-refusal |
|---|---|---|---|
| Nunca se abstém | **75%** (parece bom) | **0%** | 0% |
| Sempre se abstém | 25% | **100%** (passa o gate!) | **100%** |
Com n=15 no gate, 14/15 = 93,3% passa e 13/15 = 86,7% reprova: **um único exemplo** decide.

**(4) Literatura.**
- `abstention-benchmarks.md` (AbstentionBench): reporta **recall (principal)**, precision e F1; foca em recall porque precision ≈ 1 nos modelos deles. Sustenta o recall como gate, **não** o F1 como métrica principal.
- Mesma ficha (RefusalBench): separa **False Refusal Rate** (= over-refusal) e **Missed Refusal Rate** (= under-refusal; sigla "MRR" no paper — **não confundir** com Mean Reciprocal Rank), mais **Refusal Detection F1**; mostra o trade-off (correlação −0,78) e que o GPT-4o recusa 62,8% do respondível e deixa passar 4,3% do irrespondível. Nenhum modelo de fronteira passa de 73% de acerto de recusa **com** a categoria certa. Eles propõem o **CRS** (média simples de duas acurácias) — a ficha recomenda **não** adotá-lo como gate, porque esconde o trade-off.
- `yu-2024-rag-eval-survey.md`: o "Rejection Rate" do survey é **unilateral** (só um lado).
- `magesh-2025…`: recusa conta como "incompleta", não como alucinação.
- Nenhuma fonte discute **amostras pequenas**: decisão de IC (Wilson) é **de engenharia**, não da literatura.

**(5) Alternativas.**
| | Métrica | Prós | Contras |
|---|---|---|---|
| A | acurácia (atual) | simples | mistura os dois erros; enviesada pelo desbalanceamento 75/25 |
| B | **recall de abstenção (gate) + over-refusal como guarda-corpo + F1 diagnóstico, por slice** | alinhado a AbstentionBench/RefusalBench; separa riscos | 3 números |
| C | RefusalBench completo (FRR, Missed Refusal Rate, Detection F1, CRS, categoria) | mais rico | categorização não é requisito; CRS esconde trade-off |
| D | número único (CRS) | fácil de comparar | esconde o trade-off (a ficha desaconselha) |

**(6) Recomendação (D-6): B**, usando os nomes do RefusalBench em português (`abstention_recall`, `over_refusal_rate`, `abstention_f1` diagnóstico),
**por slice** (`unanswerable` × `stale`; o conceito de "stale" do AbstentionBench é outro — ver `abstention-benchmarks.md` reflexão 6) e **sempre com contagem bruta + IC de Wilson**.
Gate proposto: `abstention_recall ≥ 90%` em `unanswerable` **e** `over_refusal_rate` sem piorar além do IC do baseline (limiar numérico: a definir com você).

**(7) Passos.**
1. Novo summary evaluator `abstention_summary(outputs, reference_outputs, examples)` (o argumento `examples` dá acesso a `metadata.slice`), retornando `{"results": [...]}` (contrato L-2): `abstention_recall`, `over_refusal_rate`, `abstention_f1` — total e por slice.
2. Usa `abstained()` do P7; `f1_summary_evaluator` perde a chave `abstention_accuracy` (fica só o token-F1, que o P9 reavalia).
3. Contagens (`n`, `k`) e IC de Wilson no `comment` de cada resultado.
4. Testes com matrizes de confusão sintéticas (TP/FP/TN/FN), incluindo os dois sistemas "bobos" da tabela.
5. Registro no `decisions.md`: regra de regressão "ganho de recall não pode vir de aumento de over-refusal".

**(8) Preciso de você:** (i) aceita B como desenho? (ii) qual tolerância de piora do over-refusal (ex.: ≤ 1 exemplo? ≤ 5 p.p.?) (iii) concorda em manter F1 só como diagnóstico?

---

## 7. O que as fichas mudaram em P9–P11, e pontos novos

### Atualização dos rascunhos P9–P11 (discussão depois de P5–P8)
- **P9 (token-F1):** a evidência é **mais fraca** do que eu dizia (C-6). `ru-2024-ragchecker.md` (Tab. 2/5, Pearson com humanos): métrica claim-level 61,93; RAGAS answer-similarity 48,31; ROUGE-L 43,10; BLEU 35,14; BERTScore 33,51; humano×humano 70,09 (juiz-base Llama3-70B). `qa-eval-judge-vs-f1.md` (Wang 2023): o juiz GPT-3.5 **não** superou a comparação lexical em respostas longas (69,5% vs 82,3% no BingChat). Direção provisória: juiz de correção reference-based **calibrado localmente** + token-F1 só diagnóstico.
- **P10 (format_spec):** `zhou-2023-ifeval.md` apoia spec estruturada, mas o par `instruction_id`+`kwargs` é do **dataset**, não do artigo; os `kwargs` são um registro "união" quase todo `None` → preferir requisitos atômicos com parâmetros por tipo. As 8 transformações do modo *loose* **não** servem (remover 1.ª/última linha derruba cabeçalho de tabela/CSV/citação). Com n=10, IC de Wilson de 10/10 = [72%; 100%]: o slice `format` é **teste de regressão**, não estimativa de taxa.
- **P11 (juiz ≠ gerador):** `wataoka-2024…`: viés do GPT-4 = 0,520, ligado à **perplexidade** do texto, não à autoria; GPT-4/3.5 ficaram fora da análise de perplexidade; **ninguém testou gpt-4o-mini**. `abstention-benchmarks.md` (RefusalBench): auto-avaliação 91,0% vs 82,1% cross; κ entre juízes tão baixo quanto 0,061. `zheng-2023…`: auto-enhancement observado (GPT-4 +10%, Claude-v1 +25%) mas os autores **não conseguem concluir**. → separar `JUDGE_MODEL`, de **outra família**.
  **Descoberta nova (C-2):** as "100 labels humanas" revisaram o golden set, **não respostas geradas**. O ARES usa ≥150 pontos rotulando **saídas de sistema** e mostra que com 100–150 o poder de discriminação é limitado (Tab. 3). Para calibrar juízes (κ) é preciso **você rotular respostas reais** depois do baseline (~100–150, faithfulness + abstenção + citação). Magesh dá um marco: κ 0,77 em 48 itens (domínio jurídico).

### Pontos novos surgidos das fichas (triagem pendente)
| ID | Ponto | Fonte | Resumo |
|---|---|---|---|
| P12 | `citation_accuracy` incompleto | `magesh…`, `gao-2023-alce.md` | Cobre só *misgrounded* (citação não sustenta). Faltam *ungrounded* (afirmação sem citação) e *fabricated* (ID não recuperado). ALCE: citation **recall** (a afirmação é sustentada pelas citações?) e **precision** (cada citação é necessária?) por sentença; κ humano×ALCE 0,698 / 0,525. Abstenção sem citação daria recall 0 → isentar. Citação isolada é gameável (no atalho "top-1 passage" a correção cai só 5 pts; quem denuncia é a fluência) → reportar junto de correção. |
| P13 | Mapeamento FP→métrica | `barnett-2024…` | C-3. Evidência do Barnett é fraca (3 estudos de caso, sem contagem por FP): usar como **taxonomia de cobertura**, não prova de prevalência. |
| P14 | Formato e ordem do prompt dos juízes | `zheng…`, `qa-eval…` | C-8. Testar explicar-antes-de-pontuar, incluir gold answer (Zheng Tab. 4: 70% → 15% de falhas em matemática com reference-guided). |
| P15 | Significância estatística dos gates (≥15%/≥10%) | `ir-pooling…`, `bassani…` | B&V: δ de ~8–18% para 95% de confiança com 50–100 tópicos e muitos relevantes; com ~50 exemplos binários o δ tende a ser maior. `ranx.compare` oferece t/Fisher/Tukey mas **sem correção para comparações múltiplas**. Propor IC/bootstrap pareado. |
| P16 | Viés lexical do gold sintético | `thakur…` (analogia), `ir-pooling…` | Perguntas geradas do chunk compartilham vocabulário com ele; pode favorecer BM25/híbrido contra denso em CKPT-1/4. **Nenhuma fonte estuda gold sintético** — extrapolação. Mitigação: D-3 (BM25 em paralelo) + perguntas parafraseadas. |
| P17 | Diagnósticos novos de RAGChecker | `ru-2024-ragchecker.md` | *Context utilization* (achou mas não usou) ≈ acurácia condicionada a hit@k=1 vs 0 (barata); *noise sensitivity*; *hallucination vs self-knowledge*; trilema utilização/ruído/fidelidade ao otimizar prompt (CKPT-6). Métricas diagnósticas **não validadas** com humanos no paper. |

---

## 8. Registro histórico: planos originais P1–P4 (executados) e rascunhos P9–P11

> Mantidos como estavam na versão 1 para rastreabilidade. P5–P8 foram **substituídos** pela §6; a antiga seção
> "Fundamentação científica (resumos)" foi **substituída pelas fichas** (§4).

## P1 + P2 + P3 — Normalizar o nível de comparação (chunk × paper) — ✅ EXECUTADO

### O que está implementado
`evaluators.py` recebe `retrieved_chunk_ids` e `retrieved_arxiv_ids` e o gold pode ser `gold_chunk_ids`
(slice `answerable`) **ou** `gold_arxiv_ids` (multi-doc, persona). As quatro funções fazem
`gold = chunk_gold or arxiv_gold` e depois tratam as duas listas de recuperados como um só universo:
- `recall_at_k` / `hit_rate`: `set(chunk) | set(arxiv)` — funciona por acaso (IDs de formatos distintos não colidem).
- `mrr`: `enumerate([*chunk_ids, *arxiv_ids], 1)` — **concatena** as duas listas.
- `precision_at_k`: `retrieved = chunk_ids if chunk_ids else arxiv_ids` — escolhe a lista pela presença, não pelo tipo do gold.

### Evidência do erro (reproduzido offline)
```
ids=[a..e]; chunks=["a:h",...]; ref={"gold_arxiv_ids":["a"]}   # paper gold em 1º lugar
mrr(chunks, ids, ref)            -> 0.1667   (esperado 1.0 — rank 6 em vez de 1)
precision_at_k(chunks, ids, ref) -> 0.0      (esperado 0.2 — compara chunk IDs com gold de paper)
```
Cobertura: `tests/test_evaluators.py` tem `test_recall_uses_arxiv_ids_for_multidoc`, mas
**nenhum teste de MRR/precision com gold por `arxiv_id`** — por isso passou.
Agravante (decisions.md, 2026-09-27): o top-5 baseline devolve o mesmo paper até 3×, então o rank
em nível de paper precisa de **dedup preservando ordem** (caso contrário MRR/precision em paper ficam inflados/deflacionados).

### Solução (passo a passo)
1. Escrever o helper `_ranked_at_gold_level(retrieved_chunk_ids, retrieved_arxiv_ids, reference_outputs) -> tuple[list[str], set[str]] | None`:
   - gold de chunk não vazio → `(retrieved_chunk_ids, gold_chunks)` (nível chunk);
   - senão gold de paper não vazio → `(dedup_ordenado(retrieved_arxiv_ids), gold_papers)` (nível paper, `dict.fromkeys` p/ preservar rank);
   - senão `None`.
2. Se o runner (0.7) passar só `retrieved_chunk_ids`, derivar `arxiv_id` com `chunk_id.split(":")[0]` quando `retrieved_arxiv_ids` vier vazio (contrato do 0.6 §2 documentado no docstring).
3. Reescrever as quatro métricas sobre o helper: recall = `|gold ∩ ranked| / |gold|`; hit = `any`; MRR = `1/rank` do 1º gold em `ranked`; precision = `|gold ∩ ranked| / len(ranked)` (P6 refina o denominador).
4. Testes novos (offline, em `tests/test_evaluators.py`): MRR e precision com gold de paper em 1º/3º/ausente; top-k com o mesmo paper 3× (dedup); fallback de `arxiv_id` derivado do chunk ID; gold misto (chunk tem precedência).
5. **Oráculo independente**: adicionar `ranx` (ou `ir_measures`) como dependência **dev** e um teste de propriedade que gera runs/qrels aleatórios e confere `recall/mrr/hit/precision` contra a lib. Não usar a lib em runtime: os avaliadores do LangSmith chamam por exemplo (1 query), e montar `Qrels/Run` por chamada é overhead sem ganho; a lib entra como prova de correção.

### Por que melhora
- Remove a causa (granularidade misturada), não só os dois sintomas — qualquer métrica nova (nDCG no CKPT-4) herda o comportamento certo.
- O baseline de multi-doc/persona deixa de ser subestimado ~6× em MRR; o gate do rerank (CKPT-4) mede o efeito real.
- O oráculo `ranx` torna a defesa simples: "implementação conferida contra a lib de referência".

### Verificação
`uv run pytest tests/test_evaluators.py` verde; os dois comandos da evidência retornam `1.0` e `0.2`; teste de propriedade com ≥200 casos aleatórios concorda com `ranx`.

---

## P4 — `f1_summary_evaluator` com formato de retorno inválido para o LangSmith — ✅ EXECUTADO

> **Correção (2026-10-03):** o texto original abaixo diz que a chave extra é "ignorada". Verificado no `langsmith` 0.13.0: ela é **rejeitada** (`EvaluationResult` tem `extra="forbid"`), o runner captura a exceção e **descarta o evaluator inteiro**; além disso a assinatura `(results)` nem passava na validação de `summary_evaluators=`.

### O que está implementado
Retorna `{"key": "f1_summary", "score": ..., "abstention_accuracy": ...}` (uma chave extra num dict).

### Evidência do erro
Saída real do teste: `{'key': 'f1_summary', 'score': 0.33, 'abstention_accuracy': 0.0}`. Summary evaluators do LangSmith
consomem `key`/`score` (ou `{"results": [...]}`); a chave extra **não vira métrica** no experimento — a abstenção seria
calculada e descartada silenciosamente. (Confirmar o formato exato na versão instalada do `langsmith` via context7 antes de codar.)

### Solução
1. Ler a doc da versão instalada (`uv pip show langsmith`; context7) para o formato de retorno de summary evaluators com múltiplas métricas.
2. Quebrar em **evaluators de resumo separados e de responsabilidade única**: `token_f1_summary` (P9), `abstention_summary` (P8) — ou um único retornando `{"results": [{"key":..., "score":...}, ...]}` se for o formato suportado.
3. Teste que valida o *shape* (lista/dict com `key` e `score` numéricos) e, de preferência, um teste de integração mínimo com `langsmith.evaluate` sobre dataset em memória (sem rede, via `upload_results=False` se disponível).

### Por que melhora
Cada métrica vira uma coluna visível/comparável no LangSmith — requisito do `decisions.md` ("metric before → after"). Hoje a abstenção, release-blocking (gate 90%), não apareceria.

### Verificação
Rodar o evaluator num experimento local e ver as duas métricas listadas; teste de shape verde.

---


## P9 — Token-F1 vs gold answer como métrica de qualidade

### O que está implementado
`_token_f1(pred, ref)` estilo SQuAD (bag-of-words) comparando resposta gerada com o gold answer, agregada em `f1_summary`. O plano (`ckpt-0.6-plan.md` §3.3) descreve outra coisa ("F1 entre substantivo × alucinado"), então **código e plano divergem**.

### Evidência
"Agents hallucinate tools [2609.12345]." vs "LLM agents often hallucinate tool calls." (resposta correta) → **0.36**. Paráfrase é punida; os tokens da citação `[2609.12345]` inflam o denominador; o prompt limita a 3 frases, o que torna o F1 sensível a comprimento.

### Solução
1. Decidir e corrigir a divergência: o plano passa a dizer "token-F1 = métrica **diagnóstica de cobertura lexical**", não de qualidade.
2. Pré-processar antes de comparar: remover citações (`\[\d{4}\.\d{4,5}v\d+\]`) e stopwords; renomear `f1_summary` → `token_f1_summary`.
3. Qualidade passa a vir de **dois sinais semânticos**: (a) `correctness_judge` reference-based (LLM, resposta vs gold answer; via `openevals.create_llm_as_judge` ou prompt próprio no 0.6.3) e (b) similaridade de embedding (`compare_semantic_similarity`, já citado no plano §2b) como alternativa barata.
4. Calibrar contra as labels humanas do golden set (100/100 revisados): correlação (Spearman) token-F1 × julgamento humano vs correctness_judge × humano — o resultado decide se o token-F1 sai do relatório principal.

### Por que melhora
Troca uma métrica que pune respostas corretas por uma que mede o que o produto promete; a calibração humana transforma a escolha numa decisão empírica defensável.

### Verificação
Testes de normalização (citação removida, caixa, pontuação); correlação reportada em `results/`.

---

## P10 — `format_validator` acoplado ao texto da pergunta

### O que está implementado
`format_validator(question, answer)` despacha por substring ("markdown table", "JSON array"…) e extrai colunas/chaves/contagens com regex sobre a pergunta (`_check_*`). Formato desconhecido retorna `{"score": 0, "reason": ...}` **sem `key`**; há código morto (`if "No other text" ...: pass` em `_check_json`).

### Evidência
Qualquer reescrita de pergunta (ex.: `regenerated: v2` em `build_golden_dataset.py:635`, ou o `stale`/persona do eixo ADR-004) pode quebrar o parse e virar `score 0` — **falso negativo atribuído ao sistema**. O fallback de "formato não detectado" pontua 0 em vez de ser excluído, contaminando o slice `format`.

### Solução
1. Em `build_golden_dataset.format_examples`, gravar uma **spec estruturada** em `metadata.format_spec`, ex.: `{"type":"json_object","keys":["title","year"],"exact_keys":true}` / `{"type":"markdown_table","columns":[...],"rows":3}` / `{"type":"bullets","count":3,"max_words":12}`.
2. `format_validator(answer, format_spec)` valida só contra a spec (sem regex de pergunta); a pergunta fica apenas para exibição.
3. Spec ausente/desconhecida ⇒ `score=None` (pulado, como `is_retrieval_evaluable`) e sempre com `key="format_valid"`.
4. Migração idempotente do dataset (o builder já é idempotente — adicionar atualização de metadata nos 10 exemplos existentes) + remover o código morto.
5. Manter `test_format_golds_dogfood_all_pass` (10/10) e adicionar caso "spec ausente ⇒ None".

### Por que melhora
Valida o contrato que o dataset declara, não uma interpretação de texto livre; elimina falsos negativos por reescrita e alinha com o padrão IFEval ("instrução verificável programaticamente").

### Verificação
Dogfood 10/10 com specs; teste de reescrita de pergunta não altera o resultado.

---

## P11 — Juiz e gerador usam o mesmo modelo (gpt-4o-mini)

### O que está implementado
`config.GENERATION_MODEL` (`gpt-4o-mini`) gera as respostas; o plano (`ckpt-0.6-plan.md` §4) define o **mesmo modelo** como juiz, `temperature=0`. `config.RERANK_MODEL` também é o mesmo.

### Evidência
Não há erro numérico a reproduzir — é um risco metodológico conhecido (autopreferência/self-enhancement bias de LLM-judges). O plano já prevê medir variância entre rodadas e calibrar com kappa no 0.8, mas **não** separa o modelo do juiz.

### Solução
1. Nova variável `JUDGE_MODEL = os.getenv("JUDGE_MODEL", ...)` em `config.py`, **distinta** de `GENERATION_MODEL`; default: modelo mais forte ou de outra família (decisão do usuário, ver pergunta aberta).
2. Todos os juízes (0.6.3) leem `JUDGE_MODEL`; registrar `judge_model` na metadata do experimento.
3. Calibração (0.8): kappa juiz×humano **por juiz** nas 100 labels; critério de aceite documentado (ex.: κ ≥ 0.6) antes de congelar o baseline.
4. Rodada de sensibilidade: reavaliar o mesmo conjunto de respostas com 2 juízes e reportar a diferença no relatório do EXP-0.

### Por que melhora
Remove a objeção mais comum a evals com LLM-judge ("o modelo avalia a si mesmo") e torna as deltas dos gates atribuíveis ao sistema, não ao viés do juiz.

### Verificação
`config.JUDGE_MODEL` aparece na metadata das runs; relatório de calibração com κ por juiz.

---

## Critérios de pronto do lote
1. `uv run pytest` verde, incluindo o oráculo `ranx` (P3) e os testes de P4/P7/P8/P10.
2. Reprodução dos 5 comandos de evidência retorna os valores esperados (1.0, 0.2, shape válido, abstenção por paráfrase, token-F1 normalizado).
3. `docs/plan.md` §3/§4 e `ckpt-0.6-plan.md` §3 atualizados para refletir as novas semânticas (recall×hit, precision, F1/abstenção, token-F1).
4. ADR-005 (semântica de precision) e uma entrada em `docs/decisions.md` → Observações descrevendo a correção dos bugs de IR **antes** do EXP-0.
5. Um commit por ponto (mensagens `fix(evaluators): ...`), sem misturar com o trabalho não commitado do CKPT-0.6.

---

