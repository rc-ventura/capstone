# Barnett et al. 2024 — Sete pontos de falha ao construir um sistema RAG

- **Referência completa:** Barnett, S., Kurniawan, S., Thudumu, S., Brannelly, Z., Abdelrazek, M. (2024). *Seven Failure Points When Engineering a Retrieval Augmented Generation System*. 3rd International Conference on AI Engineering — Software Engineering for AI (CAIN 2024), Lisboa. arXiv:2401.05856 [cs.SE]. Link: <https://arxiv.org/abs/2401.05856> (PDF: <https://arxiv.org/pdf/2401.05856>).
- **Confiança da leitura:** **integral, mas só da versão arXiv v1** (11/01/2024, 6 páginas), lida a partir do texto extraído do PDF (WebFetch devolveu o PDF binário; usei `pdftotext`). Li o artigo todo (resumo, §1–§7, Tabelas 1–2, Figura 1 só pela legenda; a imagem da figura não foi inspecionada). Não conferi versões posteriores nem a versão publicada pela ACM, então números e redação podem diferir nelas. Os rótulos "(a) o artigo afirma" e "(b) minha interpretação" estão marcados no texto.

## 1. Problema que o artigo ataca

**(a) O artigo afirma:** engenheiros de software estão adicionando busca semântica às aplicações com RAG (recuperar documentos e passá-los a um LLM), mas não havia relato de "o que quebra" na prática. O artigo se declara "um relato de experiência" (abstract) sobre falhas de RAG em 3 estudos de caso (pesquisa, educação, biomédico) e diz ser, "até onde sabemos", o primeiro insight empírico sobre os desafios de criar RAGs robustos (§1). Entrega: (1) um catálogo de pontos de falha (FP), (2) um relato de 3 casos, (3) uma agenda de pesquisa (§1, "Contributions").

Duas perguntas de pesquisa (§1): **RQ1** "Quais pontos de falha ocorrem ao construir um RAG?" (respondida na §5, com o experimento BioASQ) e **RQ2** "Quais as considerações-chave ao construir um RAG?" (respondida na §6, com as lições dos 3 casos).

**(b) Minha interpretação:** é um artigo de engenharia de software (CAIN), não de avaliação. Ele cataloga *onde* o RAG falha, mas não propõe métricas nem mede prevalência de cada falha. Isso importa para o nosso uso (ver §6 desta ficha).

## 2. Método (passo a passo, com as definições exatas)

**Arquitetura de referência (§3, Figura 1).** Dois processos:
- **Index** (tempo de desenvolvimento): cada documento é dividido em *chunks*, cada chunk vira um *embedding* (vetor numérico que representa o sentido do texto) e é gravado numa base. Decisões: tamanho do chunk ("se são pequenos, certas perguntas não podem ser respondidas; se são longos, as respostas incluem ruído gerado", §3.1) e modelo de embedding (trocar exige reindexar tudo).
- **Query** (tempo de execução): pergunta → reescrita em consulta geral (*rewriter*) → embedding → top-k por similaridade (ex.: cosseno) → *re-ranker* → **Consolidator** (reduz o que cabe no prompt por causa de limite de tokens e de taxa) → *Reader* (o LLM que filtra ruído, obedece o formato e produz a resposta) (§3.2).

**Estudos de caso (§4, Tabela 1).** Três sistemas:
| Caso | Domínio | Tipo de doc | Tamanho do dataset | Estágios | Em uso? |
|---|---|---|---|---|---|
| Cognitive Reviewer | Pesquisa | PDFs | "(Any size)" | Chunker, Rewriter, Retriever, Reader | sim (*) |
| AI Tutor | Educação | vídeos, HTML, PDF | 38 | Chunker, Rewriter, Retriever, Reader | sim (*) (piloto com 200 alunos, §4.2) |
| BioASQ | Biomédico | PDFs científicos | 4017 | Chunker, Retriever, Reader | experimento |

Só o BioASQ tem dados abertos (figshare, nota 5); os outros dois foram omitidos "por confidencialidade" (§4).

**Único experimento quantitativo (§4.3).** BioASQ: baixaram 4017 documentos de acesso aberto e 1000 perguntas, indexaram tudo e geraram as respostas com GPT-4. Avaliação: a técnica "OpenEvals" da OpenAI (avaliador automático por LLM); depois, **inspeção manual de 40 casos e de "todos os casos que o OpenEvals marcou como inexatos"**. Conclusão do §4.3: a avaliação automática foi "mais pessimista que um avaliador humano" nesse domínio. Ameaça à validade declarada pelos autores: os revisores não eram especialistas biomédicos, então o LLM pode saber mais que eles.

**Os sete pontos de falha (§5, textuais, tradução minha entre aspas simples, original em inglês logo depois):**
- **FP1 Missing Content** — a pergunta não pode ser respondida a partir dos documentos. No caso feliz o sistema diz "Sorry, I don't know"; mas "para perguntas relacionadas ao conteúdo e sem resposta, o sistema pode ser enganado a dar uma resposta".
- **FP2 Missed the Top Ranked Documents** — a resposta está no documento mas ele não ficou bem ranqueado para ser devolvido; na prática devolve-se o top-K, com K escolhido "com base em desempenho".
- **FP3 Not in Context — Consolidation strategy Limitations** — documentos com a resposta *foram recuperados* da base mas "não entraram no contexto" para gerar a resposta; ocorre quando muitos documentos voltam e um processo de consolidação seleciona.
- **FP4 Not Extracted** — a resposta *está no contexto*, mas o LLM não a extraiu; "tipicamente" por excesso de ruído ou informação contraditória no contexto.
- **FP5 Wrong Format** — a pergunta pedia um formato (tabela, lista) e o LLM ignorou a instrução.
- **FP6 Incorrect Specificity** — a resposta vem, mas não é específica o bastante ou é específica demais; ocorre quando os projetistas têm um resultado desejado (ex.: professores para alunos, em que se espera conteúdo educacional específico "não só a resposta") e também quando o usuário não sabe perguntar e é geral demais.
- **FP7 Incomplete** — respostas "não incorretas", mas que omitem parte da informação que estava no contexto. Exemplo: "Quais os pontos-chave dos documentos A, B e C?"; os autores sugerem perguntar separadamente.

**Como (não) se mede cada FP.** O artigo **não define métrica, limiar nem procedimento de detecção para nenhum dos 7 FP**. O mais próximo: no §3 diz que RAGs são difíceis de testar porque não há dados e eles precisam ser obtidos por geração sintética ou piloto com pouco teste; no §6.3, que são necessários pares pergunta–resposta específicos da aplicação e métricas de qualidade, que usar LLMs é caro, introduz latência e muda a cada versão; cita G-Eval (Liu et al. 2023) como técnica de avaliação offline "promissora", mas "condicionada a ter pares Q&A rotulados" (Tabela 2, última linha).

## 3. Principais contribuições (lista)

1. Catálogo de 7 pontos de falha organizados pelo pipeline Index/Query (§5, Figura 1).
2. Relato de experiência de 3 casos (2 em operação na Deakin University) (§4).
3. Tabela de 9 lições (Tabela 2) ligando cada lição a FPs e a casos.
4. Agenda de pesquisa: chunking e embeddings (§6.1), RAG vs. fine-tuning (§6.2), testes e monitoramento (§6.3).
5. Duas conclusões-chave (abstract): "validar um sistema RAG só é viável durante a operação" e "a robustez evolui, não é projetada no início".

## 4. Resultados-chave (com números e onde estão)

**(a) O que o artigo afirma, com o n de cada evidência:**
- **Evidência quantitativa:** só o BioASQ (§4.3): 4017 documentos, 1000 pares Q&A, 40 casos inspecionados manualmente mais todos os marcados como incorretos pelo avaliador automático. **Não há contagem por FP, nem taxa de acerto geral, nem tabela de resultados.** O único resultado reportado é qualitativo: avaliador automático mais pessimista que humano.
- **Inconsistência interna:** o §1 (RQ1) diz que o experimento envolveu "15,000 documents and 1000 question and answer pairs"; o §4.3 e a Tabela 1 dizem 4017 documentos. O `docs/research/rag_failure_modes_review.md` do projeto repete "15k docs". Não consegui resolver qual está certo só com a v1.
- **Lições (Tabela 2) — o que é "lição" e não evidência medida:**
  | Lição | FP | Caso | Natureza |
  |---|---|---|---|
  | Contexto maior dá melhores resultados ("8K vs 4K", contrário a trabalho prévio com GPT-3.5) | FP4 | AI Tutor | observação em 1 sistema, sem número reportado |
  | Cache semântico reduz custo e latência | FP1 | AI Tutor | recomendação |
  | Jailbreaks contornam o RAG | FP5–7 | AI Tutor | recomendação apoiada em literatura externa |
  | Metadados (nome do arquivo, nº do chunk) melhoram a recuperação | FP2, FP4 | AI Tutor | observação qualitativa |
  | Embeddings open-source foram tão bons quanto os fechados em texto curto | FP2, FP4–7 | BioASQ, AI Tutor | observação, sem número |
  | RAG exige calibração contínua | FP2–7 | AI Tutor, BioASQ | opinião/lição |
  | Implementar um pipeline configurável | FP1, FP2 | os 3 | lição |
  | Pipelines montados sob medida são subótimos (treino ponta a ponta ajuda) | FP2, FP4 | BioASQ, AI Tutor | apoiada em Siriwardhana et al. 2023 |
  | Teste de desempenho só é possível em runtime | FP2–7 | Cognitive Reviewer, AI Tutor | lição |

  **Minha leitura:** quase todas as linhas da Tabela 2 são relatos qualitativos; nenhuma vem com tamanho de efeito.

**(b) Minha interpretação do peso da evidência:** os FPs são uma **taxonomia qualitativa** derivada de 3 sistemas, útil para cobertura de testes (cobrir cada modo de falha), mas não é estimativa de prevalência nem prova de causa.

## 5. Limitações (as do autor e as que eu identifico)

**Do autor:**
- Ameaça à validade no BioASQ: revisores não especialistas (§4.3).
- Dois dos três casos não podem ser abertos por confidencialidade (§4), logo não são reproduzíveis.
- O §6.3 reconhece que a questão de gerar perguntas e respostas realistas do domínio "permanece um problema aberto" e que testar RAG exige dados que normalmente não existem.

**Minhas (marcadas como interpretação):**
- Sem contagem por FP, não dá para dizer quais FPs dominam. As afirmações sobre FP1–FP7 como "mais comuns" nos nossos documentos de revisão vêm de outras fontes (RaftLabs, Suthar), não deste artigo.
- Inconsistência 15.000 × 4017 documentos (ver §4).
- O catálogo mistura estágios do pipeline com sintomas na resposta: FP1 é propriedade do corpus em relação à pergunta; FP2 e FP3 são perdas de recuperação; FP4–FP7 são falhas do leitor (LLM). Não há tratamento de **alucinação com a evidência certa no contexto** (um LLM que afirma algo que o contexto não diz), apenas de omissão (FP4/FP7). Isso é uma lacuna em relação a Magesh et al. 2025, que o próprio Magesh nota ao dizer que a tipologia dele "colapsa" alguns FPs do Barnett e introduz novos.
- Sem comparação com uma baseline sem RAG, sem análise estatística, sem intervalos de confiança.
- A lição "contexto maior é melhor" vem de um sistema e contradiz Liu et al. 2023 (*Lost in the middle*); o artigo apenas observa o contraste, sem testar a causa.

## 6. Reflexões ancoradas no NOSSO projeto

Referência de pontos: P1–P11, D-1, D-2, L-3 em `docs/roadmap/ckpt-0.6.1b-metrics-roadmap.md`; plano em `docs/roadmap/ckpt-0.6-plan.md` (§3) e `docs/plan.md`.

**R1 — O mapeamento "FP3 = ruído no contexto" não bate com a definição do Barnett. (DESAFIA o que está escrito.)**
O plano 0.6 §3.1 (`retrieval_precision_at_k`) diz "FP3 — retorno irrelevante/noise" e o roadmap P6 diz que `distinct_papers@k` e a precision "respondem FP3 ('ruído no contexto')". No Barnett (§5), **FP3 é "Not in Context — consolidation strategy limitations"**: o documento certo *foi recuperado* mas *não entrou no contexto* por causa de consolidação, reranking ou limite de tokens. O ruído aparece no artigo como **causa de FP4**. **Afeta:** `evaluators.py::precision_at_k`, `docs/plan.md` §3, `ckpt-0.6-plan.md` §3.1, roadmap P6.
Para o nosso sistema (top-5 de chunks de ~420 caracteres, sem passo de consolidação), FP3 provavelmente quase não ocorre; só voltaria a existir com um reranker que corte documentos (CKPT-4) ou com truncamento de contexto. **Recomendação:** renomear o rótulo de `precision_at_k` (e de `distinct_papers@k`) para "ruído no contexto — causa de FP4" e tratar FP3 como "recuperado mas descartado", mensurável só quando existir passo de seleção entre retrieval e prompt (ex.: `gold_chunk ∈ retrieved` mas `gold_chunk ∉ prompt`).

**R2 — `faithfulness_judge` e `citation_accuracy` rotulados "FP4" estão mal ancorados. (DESAFIA.)**
O plano 0.6 §3.2 coloca os dois sob FP4. FP4 do Barnett é **omissão** (a resposta estava no contexto e o LLM não a extraiu). Fidelidade e acurácia de citação medem o contrário: **afirmação sem suporte no contexto** (comissão/alucinação). **Afeta:** `faithfulness_judge`, `citation_accuracy`, e o gap que FP4 realmente pede: uma métrica de **"resposta correta dado que o chunk-gold foi recuperado"** (por exemplo, correção condicionada a `hit@k=1`) e/ou completeness no nível do chunk. **O que adotaríamos:** manter os dois juízes (pelo Magesh e pelo ALCE), mas corrigir o rótulo para "alucinação com contexto presente (fora do catálogo do Barnett)" e acrescentar a medida condicional para FP4. **Neutro** quanto à utilidade dos juízes; só desafia o rótulo.

**R3 — Os demais mapeamentos estão coerentes. (APOIA.)**
FP1→`abstention_quality` + slices `unanswerable`/`stale` (P7, P8); FP2→`recall@k`/`hit@k`/`MRR` e slice `deep-hit`; FP5→`format_validator` (P10); FP6→`specificity_judge` e slice `persona`; FP7→`completeness_judge` e slice `multi-doc`. O exemplo de FP7 do artigo ("pontos-chave dos documentos A, B e C") é exatamente o nosso `multi-doc`. A recomendação do artigo de perguntar separadamente sustenta o experimento de decomposição (CKPT-5).

**R4 — "Validação só em operação": o artigo APOIA o monitoramento, mas é um argumento, não um resultado.**
Isso sustenta o CKPT-8 (avaliadores online) e o princípio do `rag_failure_modes_review.md`. Mas a base é anedótica (AI Tutor em piloto). **Cuidado:** não citar como "achado empírico" em relatório; é uma lição de engenharia.

**R5 — Estimativa de prevalência por FP não existe: o artigo não ajuda a dimensionar os slices do golden set. (NEUTRO.)**
Os 100 exemplos (40 answerable, 15 unanswerable, 10 stale, 15 multi-doc, 10 format, 10 persona; ver roadmap §3) foram dimensionados por decisão nossa. Nada no Barnett diz quantos casos de cada FP são necessários. Com slices de 10–15 itens, os intervalos de confiança são largos (minha interpretação), então relatar IC por slice (cf. P11 e calibração).

**R6 — "Avaliador automático mais pessimista que humano" (§4.3): APOIA a calibração humana do juiz (P11, 0.8).**
O único dado de avaliação do artigo é que um LLM-judge divergiu do humano. É n pequeno (40 casos), com revisor não especialista. **Afeta:** calibração κ juiz×humano com as 100 labels revisadas. Reforça a decisão de não congelar o baseline antes de medir κ por juiz.

**R7 — Chunking: o artigo diz que chunks pequenos impedem responder e chunks grandes trazem ruído (§3.1) e pede avaliação sistemática (§6.1). (APOIA a pergunta, não dá resposta.)**
Nosso corpus tem 843 chunks de ~420 caracteres (2–5 por paper). O artigo não fornece número nem regra. Relaciona-se a D-1 (gold por chunk ou por paper): como um abstract é cortado em 2–5 pedaços, o "chunk certo" pode ser diferente do chunk que responde parcialmente. Nada no Barnett decide D-1.

**O que adotaríamos:** (1) FP1–FP7 como **lista de verificação de cobertura** do golden set e como vocabulário do relatório; (2) corrigir os rótulos FP3/FP4 (R1, R2); (3) a ideia de medir retrieval e geração separadamente (estrutura em 2 processos).
**O que NÃO adotaríamos e por quê:** (1) tratar o catálogo como completo, pois ele não cobre alucinação com contexto presente nem citação falsa (ver Magesh); (2) citar "15k documentos" (ver §4); (3) usar a Tabela 2 como evidência quantitativa, pois ela é qualitativa; (4) herdar a ordem FP1–FP7 como prioridade, pois o artigo não ordena por frequência.

## 7. Citações úteis (trechos literais curtos)

1. "validation of a RAG system is only feasible during operation" — Abstract.
2. "Documents with the answer were retrieved from the database but did not make it into the context for generating an answer." — §5, FP3.
3. "Here the answer is present in the context, but the large language model failed to extract out the correct answer." — §5, FP4.
