# Bassani 2022 — ranx: biblioteca Python rápida de avaliação e comparação de rankings

- **Referência completa e link:** Bassani, E. (2022). *ranx: A Blazing-Fast Python Library for Ranking Evaluation and Comparison*. ECIR 2022 (LNCS 13186, vol. 2, pp. 259–264), DOI 10.1007/978-3-030-99739-7_30 — <https://link.springer.com/chapter/10.1007/978-3-030-99739-7_30>. Repositório: <https://github.com/AmenRa/ranx> · docs: <https://amenra.github.io/ranx/>. Trabalhos relacionados do mesmo autor (títulos conforme os BibTeX do README): *ranx.fuse: A Python Library for Metasearch* (CIKM 2022) e *ranxhub: An Online Repository for Information Retrieval Runs* (SIGIR 2023).
- **Confiança da leitura:** **PARCIAL — o paper completo (6 págs., LNCS) NÃO foi lido.** Springer, ACM e o repositório institucional (boa.unimib.it) bloquearam o acesso (login/Cloudflare); uma tentativa via OpenAlex apontou para um arXiv de outro artigo e foi descartada. O que li de fato:
  1. o **pôster oficial de 1 página** do ECIR 2022 (<https://ecir2022.org/uploads/445.pdf>), incluindo a tabela de eficiência conferida visualmente;
  2. o **README** do repositório e as páginas de docs (*metrics*, *stat_tests*, *faq*; estas três via resumo automático por modelo pequeno — não literal);
  3. o **código-fonte da v0.3.21** (`ranx/metrics/{hit_rate,recall,precision,reciprocal_rank,bpref,common}.py`, `meta/compare.py`, `statistical_tests/fisher_randomization_test.py`, `setup.py`) e os metadados do PyPI.
  **Não executei o ranx** (sessão em modo somente leitura): o comportamento em casos de borda abaixo vem da *leitura do código*, não de teste. Nada aqui afirma seções, experimentos ou limitações do paper além do que consta no pôster.

> Convenção: **(F)** = o que a fonte (pôster/docs/código) afirma ou mostra; **(I)** = minha interpretação.

## 1. Problema que a fonte ataca
(F, pôster) Avaliar rankings em Python exigia ferramentas lentas ou desajeitadas; o `pytrec_eval` (interface Python do trec_eval) é a referência, mas a biblioteca propõe uma filosofia *Plug & Play*: carregar `qrels` e `runs` (dicionários, JSON, arquivos TREC, DataFrames), calcular várias métricas numa linha, **comparar** várias runs com teste estatístico e **exportar tabelas LaTeX**. Por baixo, usa **Numba** (compilador JIT) para vetorizar e paralelizar.

## 2. Método (o que a biblioteca implementa e como valida)
**Métricas (F, pôster + README + código):** Hits, Hit Rate, Precision, Recall, F1, r-Precision, MRR, MAP, nDCG (as nove do pôster); o README/código v0.3.21 acrescenta **Bpref**, RBP e DCG. Cada uma aceita cutoff "@k" e `rel_lvl` (relevância mínima).

Definições conforme código e docs (cada uma com o que mede entre parênteses):
- **Hit Rate@k** — 1 se **algum** relevante aparece no top-k, senão 0 (o top-k trouxe pelo menos um correto?). Docstring: "it is equivalent to `success` from trec_eval". Com `qrels` vazio devolve 0.0.
- **Recall@k** — `r_k / R`: relevantes recuperados no top-k ÷ total de relevantes no qrels (que fração dos corretos apareceu?). `qrels` vazio → 0.0.
- **Precision@k** — `r_k / k` (**o k pedido**, ou o tamanho da run se k=0 — *não* o número realmente recuperado). (Quantos dos k recuperados são corretos? Se a run tem menos de k itens, o denominador continua k.)
- **MRR / Reciprocal Rank@k** — `1/(i+1)` para o primeiro relevante dentro do top-k (posição 0-based `i`), senão 0 (quão no topo está o primeiro correto?).
- **Bpref** — na docstring: `(1/R) Σ_r [1 − |n acima de r| / R]` com `n` entre os R primeiros não-relevantes **julgados**; na implementação, o denominador é `min(n_rels, n_non_rels)`, de modo que **qrels sem nenhum não-relevante julgado produz divisão por zero** (leitura do código; não executado).
- **nDCG, MAP, F1, r-Precision:** conforme docs (nDCG = DCG/IDCG; MAP = média de Precision@r nos relevantes).
- **Estrutura de dados (F, README/código):** `Run` é um dicionário `doc → score`; o ranx **ordena por score** (empates e IDs repetidos não são representáveis; scores zero/negativos não são filtrados — FAQ).

**Validação de correção (F, pôster):** a única afirmação é "All the available metrics were tested against trec_eval [4] for correctness." (README: "The metrics have been tested against TREC Eval for correctness."). O pôster **não** diz como (conjuntos de dados, tolerância numérica, número de casos). (I) Para o que realmente precisaríamos saber, o paper completo precisa ser consultado.

**Eficiência (F, pôster, dados sintéticos: 1–10 relevantes/consulta, listas de 100):** comparação com `pytrec_eval` em NDCG, MAP e MRR sobre 1.000/10.000/100.000 consultas, com 1–8 threads.

**Testes estatísticos (F, README + `compare.py`):** `compare(qrels, runs, metrics, stat_test="student", n_permutations=1000, max_p=0.01, random_seed=42, threads=0, make_comparable=False)`. Opções: **t-test pareado bilateral** (padrão no código; a docstring diz erradamente "Fisher's … is used by default"), **Fisher's randomization test** (aproximado: 1.000 permutações; H0: sistemas idênticos na média da métrica) e **Tukey HSD**. As referências indicadas são Smucker et al. (CIKM'07), Carterette e Fuhr. O resumo do WebFetch da página de docs listou só dois testes; README e código listam três. Não encontrei no `compare.py` correção de múltiplas comparações para o t/Fisher (a única forma "múltipla" é o Tukey) — leitura de código, não testado.

## 3. Principais contribuições
- Implementação de métricas de ranking em **Numba**, com paralelismo automático.
- API enxuta para qrels/runs, avaliação e **comparação com teste de significância e relatório LaTeX** em uma chamada.
- Alegação de correção contra o trec_eval (pôster).
- Ecossistema: `ranx.fuse` (algoritmos de fusão, CIKM 2022) e `ranxhub` (repositório de runs, SIGIR 2023) — **esses dois outros artigos não são sobre validar as métricas** (I); o roadmap §C lista "ECIR 2022, CIKM 2022, SIGIR 2023" como "publicações que validam" a lib, o que é impreciso: só o ECIR trata de avaliação.

## 4. Resultados-chave
- **Pôster, tabela de eficiência (100.000 consultas, ms; speed-up vs `pytrec_eval`):**
  - **MRR:** pytrec 2.935 ms; ranx 1 thread 74 ms (**39,7×**); 8 threads 38 ms (**77,2×**).
  - **MAP:** 2.950 → 210 ms (14,0×) com 1 thread; 69 ms (42,8×) com 8.
  - **NDCG:** 2.991 → 347 ms (8,6×) com 1 thread; 152 ms (19,7×) com 8.
  - Com 1.000 consultas: ~27–28 ms (pytrec) vs 1–4 ms (ranx).
- **Pôster, Tab. 1 de exemplo:** modelos 1–5 com MAP@100, MRR@100, NDCG@10 e sobrescritos de diferença significativa por Fisher's randomization com p ≤ 0,01 (exemplo de uso, não resultado de experimento sobre dados reais).
- (I) O pôster não menciona o custo de **compilação JIT** na primeira chamada (as funções usam `@njit(cache=True)`); para o caso de teste (poucas centenas de consultas), a vantagem de velocidade é irrelevante.

## 5. Limitações
- **Do autor:** o pôster não declara limitações; o README avisa que o ranx "is not suited for evaluating classifiers" (FAQ: scores de run não são labels de classe).
- **Minhas (I):**
  1. A alegação de correção é **uma frase** no pôster, sem protocolo; "tested against trec_eval" ≠ "valida contra pytrec_eval em todos os casos de borda".
  2. **Semânticas de borda diferem da nossa** e foram lidas no código: `precision@k` divide por k (nossa implementação divide pelo nº de itens do ranking deduplicado); `qrels` vazio → 0.0 (nossas funções retornam `None`); `Run` não aceita IDs repetidos — por isso o oráculo precisa deduplicar e usar scores estritamente decrescentes (já está assim em `tests/test_evaluators_oracle.py`).
  3. **Bpref sem não-relevantes julgados** divide por zero (leitura do código): não aplicável ao nosso gold, que só tem positivos.
  4. `compare` não aplica correção de múltiplas comparações no t-test/Fisher; docstring/padrão inconsistentes.
  5. **Custo de dependências:** o `ranx` 0.3.21 exige numpy, **numba≥0.54.1**, pandas, tabulate, tqdm, scipy≥1.8, **ir_datasets**, rich, orjson, lz4, cbor2, **seaborn**, fastparquet. Na nossa árvore, o `uv.lock` ganhou **479 linhas / 22 pacotes novos**, incluindo `numba 0.68.0`, `llvmlite 0.50.0`, `matplotlib 3.11.2`, `seaborn 0.13.2`, `ir-datasets 0.6.3`, `fastparquet 2026.9.0`, `pillow`, `rich`, `lz4`, `cbor2`, `fonttools`, `kiwisolver`, `contourpy`.
  6. Todas as afirmações sobre o paper completo ficam **sem verificação** (ver "Confiança da leitura").

## 6. Reflexões ancoradas no NOSSO projeto

| # | Ponto / arquivo | Reflexão | A fonte… |
|---|---|---|---|
| 1 | **L-3** (ranx como oráculo de teste), `tests/test_evaluators_oracle.py`, `pyproject.toml` (grupo `dev`) | **Validada, com ressalvas.** (i) O ranx implementa as nossas quatro fórmulas (hit, recall, MRR, precision) com semântica de trec_eval (docstring de `hit_rate`: "equivalent to `success`"), então é um oráculo válido; (ii) está corretamente no **grupo `dev`** (não vai ao runtime), coerente com L-3; (iii) **mas** a única evidência de correção do ranx que li é uma frase no pôster; a ferramenta *de referência* é o trec_eval. Em termos de rigor, um oráculo derivado direto do trec_eval (`pytrec_eval`, via `ir-measures`) tem a proveniência mais forte. | **(i) APOIA** parcialmente |
| 2 | **Custo de dependências** (`uv.lock` +479 linhas, numba/llvmlite/matplotlib/seaborn/ir-datasets) para quatro fórmulas triviais | Peso desproporcional **para o que usamos** (4 métricas, sem `compare`). Alternativa mais enxuta (metadados do PyPI consultados): **`ir-measures` 0.4.3**, cuja única dependência obrigatória é `pytrec-eval-terrier` (numpy+scipy; extensão em C sobre o trec_eval; ranx é *extra* opcional). Ele expõe `Success@k`, `R@k`, `P@k`, `RR` e **`Judged@k`** (fração do top-k que tem julgamento — útil para a "taxa de buracos"), mais `Bpref`, e a opção `judged_only` (docs: "TREC convention distinguishes [Success] from Recall@k"). **Não testei a instalação**; só vi wheels macOS universal2/manylinux no PyPI (compatibilidade com Python 3.12/3.13 **não verificada**). Recomendação (I): manter ranx por ora; se o lock/CI pesar, trocar por `ir-measures`+`pytrec-eval-terrier` ou por tabelas-verdade escritas à mão (sem dependência). | **(iii) NEUTRA** (decisão de engenharia) |
| 3 | **P1–P3**, `precision_at_k` com dedup (L-1) | A divergência já documentada (ranx ÷k vs nosso ÷len(dedup)) é real e **o código confirma**; o teste a contorna usando `k = len(ranked)`. Isso significa que **o oráculo não valida o caso "menos de k itens recuperados"** — o único em que as duas definições divergem. Adicionar um teste unitário explícito para `len(ranked) < k` ou fixar a semântica em ADR (P6). | **(iii) NEUTRA** / lacuna no teste |
| 4 | **P5 / D-2**, hit ≡ recall com gold único | Nos dois lados a semântica é a mesma: `recall.py` e `hit_rate.py` coincidem quando |qrels|=1. O ranx **mantém as duas métricas** (e as docs distinguem hit rate de recall) — não há conflito em ter ambas; a decisão de reportar só uma por slice é nossa. | **(iii) NEUTRA** |
| 5 | **Roadmap §C / P12**: `ranx.compare` para os gates ≥15% / ≥10% | O ranx oferece t pareado, Fisher e Tukey, e o t-test pareado sobre scores **por consulta** é o procedimento comum (Smucker et al. citado nas docs). Mas: (i) com ~50 exemplos de gold único e métrica binária (hit), os scores por consulta são 0/1 — o t-test assume aproximadamente normalidade; Fisher randomization ou bootstrap são mais adequados (I); (ii) sem correção de múltiplas comparações no `compare`; (iii) comparar 3+ braços (EXP-1…8) exige cuidado. Adotaríamos `compare` com `stat_test="fisher"` apenas **fora do runtime** (relatório do EXP), e relataríamos IC por bootstrap. | **(iii) NEUTRA** (útil, mas não valida o gate) |
| 6 | **Bpref** e o guardrail da lesson ("bpref-style tolerance") | Como o ranx traz bpref e o roadmap cogitou usá-lo: o código divide por `min(R, N)` com N = não-relevantes **julgados** → sem N, indefinido. Nosso gold é só positivo; portanto bpref **não é aplicável** (ver `ir-pooling-incomplete-judgments.md`). | **(ii) DESAFIA** a opção "bpref" |
| 7 | **Roadmap §C, "validada em publicações (ECIR 2022, CIKM 2022, SIGIR 2023)"** | Impreciso: CIKM 2022 = fusão (`ranx.fuse`); SIGIR 2023 = repositório de runs (`ranxhub`). Só o ECIR 2022 trata de avaliação, e eu só li o pôster dele. Reescrever a frase como "apresentada no ECIR 2022; afirma teste de correção contra trec_eval". | **(ii) DESAFIA** a redação do roadmap |

**O que adotaríamos:** manter ranx **como oráculo dev-only** (L-3), com o teste ampliado para `len(ranked) < k` e um caso de ID duplicado tratado explicitamente; usar `Judged@k` (via `ir-measures`) ou cálculo próprio para a taxa de buracos; usar o teste pareado em relatório de experimento, nunca no caminho de execução do LangSmith.
**O que NÃO adotaríamos:** ranx em runtime (overhead por exemplo + numba); bpref; o `compare` com t-test como único critério de gate sobre métricas binárias com n≈50; confiar na validação do ranx como se fosse equivalente a ter testado contra trec_eval **nós mesmos** (podemos fazer: comparar contra `pytrec_eval` num teste opcional — não executado).

### Respostas às perguntas do pedido (parte ranx)
- **(a)** Sim, o ranx define `hit_rate` (≡ `success` do trec_eval) e `recall` separadamente; com |gold|=1 os valores coincidem (código). Não recomenda métrica; só implementa.
- **(c)** Oferece **bpref**, que ignora não-julgados, mas exige não-relevantes julgados; sem eles, indefinido (código).
- **(e)** **O que o paper valida:** pelo pôster, apenas "testado contra trec_eval" (sem protocolo) + benchmark de velocidade contra `pytrec_eval` (dados sintéticos). **Métricas:** hit, precision, recall, F1, r-Precision, MRR, MAP, nDCG (+ bpref, RBP, DCG no código atual). **Testes pareados:** t-test pareado, Fisher randomization, Tukey HSD. **Adequação como oráculo:** adequado (mesmas 4 fórmulas, derivado do trec_eval), com semânticas de borda a controlar; **alternativas**: `ir-measures` (+`pytrec-eval-terrier`, sem numba, com `Success@k`/`Judged@k`) ou `pytrec_eval` direto — proveniência mais próxima do trec_eval. **Custo:** 22 pacotes novos no lock (numba, llvmlite, matplotlib, seaborn, ir-datasets…).

## 7. Citações úteis
1. Pôster ECIR 2022: "All the available metrics were tested against trec_eval [4] for correctness."
2. Docstring de `ranx/metrics/hit_rate.py`: "Note: it is equivalent to `success` from trec_eval."
3. `ranx/meta/compare.py`: "`stat_test` … Use "fisher" for Fisher's Randomization Test, "student" for Two-sided Paired Student's t-Test, or "Tukey" for Tukey's HSD test. Defaults to "student"."
