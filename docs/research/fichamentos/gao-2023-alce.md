# Gao et al. 2023 — ALCE: gerar texto com citações e avaliá-lo automaticamente

- **Referência completa:** Gao, T., Yen, H., Yu, J., Chen, D. (2023). *Enabling Large Language Models to Generate Text with Citations*. Proceedings of EMNLP 2023 (Princeton University). arXiv:2305.14627 [cs.CL]. Link: <https://arxiv.org/abs/2305.14627> (PDF: <https://arxiv.org/pdf/2305.14627>). Código e dados: <https://github.com/princeton-nlp/ALCE> (nota 1 do artigo; **não inspecionei o repositório**).
- **Confiança da leitura:** **integral, da versão arXiv v2** (31/10/2023), lida a partir do texto extraído do PDF com `pdftotext` (nada gravado no projeto). Li o corpo (§1–§8, Limitações) e os Apêndices A–I: sub-claims do ELI5 (A), implementação (C), atalhos (D), discussão de recall/precision (E), avaliação humana (F), experimentos extras e Tabelas 12–21 (G), prompts 22–29 (H) e os exemplos 30–31 (I). As Tabelas 19–21 foram cruzadas com as Tabelas 4–6 nas linhas principais (daí saíram as pequenas inconsistências listadas no §4); **não conferi cada desvio-padrão**. As Figuras 1–4 foram lidas só pela legenda e pelo texto extraído (as imagens não foram inspecionadas; os números de recall@k da Figura 4 foram conferidos nas Tabelas 12–14). **Não li** TRUE (Honovich et al. 2022), Liu et al. 2023 nem o código do ALCE, então detalhes de implementação (tratamento de casos de borda, número de anotadores por item) ficam **a verificar**. Os rótulos "(a) o artigo afirma" e "(b) minha interpretação" estão marcados no texto.

## 1. Problema que o artigo ataca

**(a) O artigo afirma:** LLMs "são propensos a alucinação" e é difícil para o usuário confiar neles sem evidência (§1). A proposta é exigir que o modelo gere o texto **com citações** a passagens de um corpus, para (1) o usuário verificar as afirmações e (2) o modelo seguir fielmente os trechos citados, o que "tem a promessa de melhorar a correção e aliviar a alucinação" (§1). Os trabalhos anteriores (WebGPT/Nakano 2021, Menick 2022) usam buscadores comerciais, modelos fechados e avaliação humana, "difíceis de reproduzir e comparar" (abstract, §1). A contribuição declarada é **ALCE**, "o primeiro benchmark reproduzível para avaliar automaticamente gerações de LLMs com citações" (§1), com métricas automáticas em três dimensões (fluência, correção, qualidade da citação) validadas contra humanos (§6).

**(b) Minha interpretação:** o ponto que importa para nós não é o benchmark em si, e sim a **decomposição da qualidade da citação em duas perguntas separadas** (a afirmação é sustentada pelo que ela cita? / cada citação era necessária?) e a evidência de que uma métrica de citação **isolada** pode ser enganada (§3.4, Apêndice D). O artigo é de PLN, não de engenharia de RAG, e trabalha com perguntas **sempre respondíveis** (ver §5).

## 2. Método (passo a passo, com as definições exatas)

**Tarefa (§2).** Dada uma pergunta `q` e um corpus `D` de passagens, o sistema devolve uma saída `S` com `n` *statements* (afirmações) `s1…sn`; cada `si` cita uma lista de passagens `Ci = {ci,1, ci,2, …}` com `ci,j ∈ D`. O corte em afirmações é **por fronteira de sentença** (nota 5; no QAMPARI, cada entidade da lista é uma afirmação). **No máximo 3 citações por afirmação** (nota 4). O corpus é dividido em **passagens de 100 palavras** (§2). O artigo observa que "quase todas" as sentenças que os LLMs produzem trazem informação valiosa e exigem citação (§2), então o benchmark **não** isenta nenhuma sentença.

**Dados (§2, Tabela 1).** 3 datasets, 1.000 exemplos aleatórios do conjunto de desenvolvimento de cada um: **ASQA** (pergunta factual ambígua, Wikipedia 2018-12-20, 21M passagens), **QAMPARI** (lista de entidades, Wikipedia) e **ELI5** (por quê/como, corpus Sphere, 899M passagens). O benchmark não traz dados de treino.

**Métricas — três dimensões (§3).**

- **Fluência = MAUVE** (§3.1; "quão parecido é o texto gerado com texto humano em distribuição"): usado só como *sanity check* (verificação de sanidade), porque "é sensível ao tamanho e ao estilo". Aplicado a ASQA e ELI5; omitido no QAMPARI. Na implementação, saída e referência são truncadas a 100 palavras (App. C).

- **Correção** (§3.2; "a resposta bate com um gabarito?"), uma métrica por dataset:
  - **ASQA — EM recall** (*exact match recall*; "das respostas curtas do gabarito, que fração aparece como trecho exato na geração?"). Ex. da Figura 2: gabarito com 3 datas, a geração acerta 1, recall = 33,3%.
  - **QAMPARI — precision e recall** de entidades, por *exact match* com a lista gold, com ajuste **recall-5**: recall conta 100% se a predição tem pelo menos 5 respostas corretas (o usuário quer só alguns exemplos).
  - **ELI5 — claim recall** ("das afirmações-chave esperadas, quantas a resposta implica?"): `text-davinci-003` gera **3 sub-claims** a partir da resposta humana (com 3 exemplos escritos à mão, Tabela 22) e o modelo **TRUE** (NLI T5-11B) verifica se a saída *entails* (implica) cada uma. Os autores rejeitam ROUGE-L porque "não considera as diferentes formas de expressar a mesma resposta e pode ser facilmente enganado" (App. A; Tabela 10: o "top-1 passage" tem ROUGE-L 19,1 mas claim recall 3,0).

- **Qualidade da citação** (§3.3), as duas métricas que mais nos interessam. O verificador é a função `ϕ(premissa, hipótese)` = o NLI TRUE, que devolve 1 se a premissa implica a hipótese e 0 caso contrário (T5-11B ajustado em SNLI, MNLI, Fever, Scitail, PAWS e VitaminC; App. C). Cada passagem entra como "Title: {TITLE}\n{TEXT}" e as passagens citadas são concatenadas com "\n" (App. C). A ideia segue o arcabouço AIS (*attributable to identified sources*; "esta afirmação é verdadeira com base só nas fontes citadas?").
  - **Citation recall** (cobertura de citação; "a afirmação é sustentada pelo que ela cita?"). Por afirmação, vale **1 se e somente se** `Ci ≠ ∅` **e** `ϕ(concat(Ci), si) = 1`; senão 0. A nota final é a **média sobre as afirmações**. Ou seja: **afirmação sem nenhuma citação vale 0**; a premissa é a *concatenação* de todas as passagens citadas naquela afirmação, não cada uma isolada. Exemplo da Figura 3: 3 afirmações, 2 sustentadas → recall = 2/3 = 66%.
  - **Citation precision** (precisão de citação; "cada citação era realmente necessária?"). Calculada **por citação** (0 ou 1) e promediada sobre **todas as citações**. Uma citação `ci,j` é "**irrelevante**" se e somente se **(a)** `ϕ(ci,j, si) = 0` (sozinha não sustenta a afirmação) **E (b)** `ϕ(concat(Ci∖{ci,j}), si) = 1` (as demais, sem ela, já sustentam). Então `ci,j` tem precision 1 se `si` tem recall 1 **e** `ci,j` não é irrelevante. O **recall = 1 é pré-requisito** para precision = 1: se o recall da afirmação é 0, todas as suas citações têm precision 0 (Figura 3). Exemplo da Figura 3: 6 citações no total, 1 irrelevante ([2] em `s3`) e a afirmação `s2` com recall 0 zerando a sua → precision = 4/6 = 66%. **Não exige conjunto mínimo**: os autores permitem redundância "porque a escrita humana frequentemente cita fontes redundantes para aumentar a credibilidade" (§3.3). A lacuna reconhecida: o algoritmo **não detecta suporte parcial** (uma citação que sustenta só parte da afirmação pode ser marcada como "irrelevante" por engano) (§3.3, App. E).

- **Por que as três juntas (§3.4, App. D).** Dois atalhos: (1) devolver o **top-1 passage** como resposta, citando-o; (2) devolver as **duas primeiras sentenças** do top-1. Ambos têm citação "quase perfeita", mas (1) tem fluência baixa (texto longo demais) e (2) tem correção baixa (cobertura baixa). Os números estão na Tabela 11 (ver §4 abaixo).

**Sistemas avaliados (§4).** Recuperação: GTR/DPR (Wikipedia) e BM25 (Sphere), top-100. Síntese: **VANILLA** (top-k passagens no contexto + 2 demonstrações), **SUMM/SNIPPET** (resumos/trechos de 10 passagens), **INTERACT** (checar o texto completo sob demanda), **INLINESEARCH** (o modelo chama busca durante a geração) e **CLOSEDBOOK** (sem passagens, sem citar). Pós-edição: **RERANK** (4 amostras, escolhe a de maior citation recall automático) e **POSTCITE** (cita depois, por similaridade com GTR). Modelos: ChatGPT (gpt-3.5-turbo-0301, 4K), ChatGPT-16K, GPT-4 (8K), LLaMA e derivados. Três rodadas com sementes diferentes (exceto RERANK, 16K e GPT-4, uma rodada) (App. C, G.6).

**Validação humana (§6, App. F, G.5).** Surge AI, 20 USD/hora; **100 exemplos** de ASQA e de ELI5; saídas de **3 sistemas** (ChatGPT VANILLA, ChatGPT RERANK, Vicuna-13B VANILLA). Humanos julgam: utilidade (Likert 1–5); citation recall (as citações juntas sustentam *totalmente* a sentença?); citation precision (cada citação *sustenta totalmente / parcialmente / não sustenta*; precision 1 se a sentença tem recall 1 e a citação sustenta **ao menos parcialmente**). **Nota:** a regra humana é mais branda que a do NLI (que só vê "implica ou não"), por isso o humano tende a dar precision maior. O número de anotadores por item e como o κ foi calculado (sobre quais rótulos) **não aparecem no texto que li** (a verificar).

## 3. Principais contribuições (lista)

1. O benchmark ALCE: 3 datasets, 3 dimensões, código aberto (§1).
2. As definições operacionais de **citation recall** e **citation precision** por afirmação/citação via NLI (§3.3).
3. O argumento de robustez a atalhos, com os casos "top-1 passage" e "first 2 sents" (§3.4, App. D).
4. Correção do ELI5 por **claim recall** com sub-claims gerados por LLM e verificados por NLI (§3.2, App. A).
5. Comparação de estratégias de síntese e pós-edição (VANILLA, SUMM, SNIPPET, INTERACT, INLINESEARCH, RERANK, POSTCITE) e de modelos, com 5 achados (§1, §5).
6. Validação humana de que o ranking das métricas automáticas coincide com o humano (§6, Tabelas 8–9).
7. Três desafios apontados: qualidade do retriever, janela de contexto limitada e dificuldade de sintetizar vários documentos sem se distrair com os irrelevantes (§1, §8).

## 4. Resultados-chave (com números e onde estão)

**(a) O que o artigo afirma:**

**Validação humana das métricas (Tabelas 8–9, §6).** Rankings humano e automático "consistentes"; valores absolutos "muito próximos", exceto no RERANK (que usa o próprio recall automático para escolher, §6).

| Sistema | ASQA humano Rec./Prec. | ASQA ALCE Rec./Prec. | ELI5 humano Rec./Prec. | ELI5 ALCE Rec./Prec. |
|---|---|---|---|---|
| ChatGPT VANILLA | 74,7 / 76,6 | 75,3 / 74,4 | 50,8 / 52,4 | 52,8 / 50,4 |
| ChatGPT + RERANK | 79,3 / 81,9 | 83,9 / 80,8 | 59,7 / 60,6 | 63,0 / 60,6 |
| Vicuna-13B VANILLA | 51,6 / 51,5 | 50,3 / 50,1 | 13,4 / 19,2 | 13,6 / 18,1 |

- **κ de Cohen humano × ALCE: 0,698 para citation recall ("concordância substancial") e 0,525 para citation precision ("moderada")** (§6). Acurácia tomando o humano como gabarito: **85,1% (recall) e 77,6% (precision)** (§6, App. G.5). Detecção de citações insuficientes: recall 82,3%, precision 84,2%; detecção de citações "irrelevantes": **recall 75,6%, precision 66,1%** ("eficaz, mas com taxa relativamente alta de falsos positivos" por não detectar suporte parcial, App. G.5).
- Utilidade (Likert) varia pouco entre sistemas: 3,7–3,9 (ASQA) e 3,5–3,6 (ELI5) (§6).
- Qualidade dos sub-claims do ELI5: **112 de 120 (93,33%)** boas em 40 respostas inspecionadas; o NLI acertou **80,0%** dos rótulos humanos nos mesmos 40 outputs × 3 sub-claims (120 pares) (App. A).

**Resultados principais (Tabelas 4 e 6, §5.1).**

| Config. (ChatGPT, salvo indicação) | ASQA: MAUVE / EM rec. / Cit. rec. / Cit. prec. | ELI5: MAUVE / claim rec. / Cit. rec. / Cit. prec. |
|---|---|---|
| VANILLA (5 psg) | 66,6 / 40,4 / 73,6 / 72,5 | 57,2 / 12,0 / 51,1 / 50,0 |
| VANILLA + RERANK | 77,0 / 40,2 / 84,8 / 81,6 | 56,1 / 11,4 / 69,3 / 67,8 |
| SUMM (10 psg) | 70,0 / 43,3 / 68,9 / 61,8 | 40,3 / 12,5 / 51,5 / 48,2 |
| CLOSEDBOOK + POSTCITE | 52,7 / 38,3 / 26,7 / 26,7 | 32,6 / 18,6 / 15,5 / 15,5 |
| GPT-4 VANILLA (5 psg) | 67,1 / 41,3 / 68,5 / 75,6 | 38,4 / 14,2 / 44,0 / 50,1 |

- **Abstract / §1:** no ELI5, "mesmo os melhores modelos carecem de suporte completo de citação em 50% das vezes" (ChatGPT VANILLA tem recall 51,1 e GPT-4 44,0–49,5 nas Tabelas 6 e 21, ou seja, **~49–56% das afirmações sem suporte completo**; a frase "50%" é arredondamento do autor).
- **VANILLA é forte** ("desempenho próximo ao melhor entre todas as estratégias", §5.1). SUMM/SNIPPET melhoram correção e **pioram citação** no ASQA e ELI5. RERANK melhora citação (ASQA: 73,6 → 84,8 de recall; ELI5: 51,1 → 69,3). INLINESEARCH é pior que VANILLA em citação.
- **CLOSEDBOOK:** dá correção comparável (ASQA 38,3 contra 40,4; no ELI5 até maior, 18,6 contra 12,0), mas a citação pós-hoc é fraca: recall 26,7 contra 73,6, "menor em 47%" no ASQA (diferença de 46,9 pontos; conta confere) (§5.1). Explicação dos autores: modelos com contexto "são facilmente distraídos por passagens irrelevantes" e o closed-book gera textos corretos mas "não similares a nenhuma passagem recuperada", difíceis de citar a posteriori.
- **Retrieval (Figura 4, Tabelas 12–14):** com 5 passagens, GTR cobre só **56,8%** das respostas do ASQA (recall@5), e a correção do ChatGPT VANILLA (40,4) fica abaixo disso. A versão **oráculo** (5 passagens gold, cujo recall iguala o recall@100) chega a 48,9 no ASQA, 37,0 no QAMPARI e 21,3 no ELI5 (Tabelas 19–21), ainda abaixo do recall do retriever (78,4; 65,6; 31,8, Tabelas 12–14). Os autores concluem que, apesar de haver respostas corretas no contexto, os LLMs têm dificuldade de utilizá-las (§5.3, paráfrase minha). **Exceção que o texto menciona:** no ELI5 top-5, o VANILLA (12,0) fica *acima* do recall@5 do BM25 (9,6).
- Mais passagens: no ChatGPT, correção estaciona no top-1 e citação no top-3 (§5.3); o GPT-4 melhora com 20 passagens, "mas não proporcionalmente" ao retrieval.

**Atalho "top-1 passage" (Tabela 11, App. D).**

| Variante (ASQA) | MAUVE | EM rec. | Cit. rec. | Cit. prec. |
|---|---|---|---|---|
| ChatGPT VANILLA (GTR top-5) | 66,6 | 40,4 | 73,6 | **63,0 — a verificar** (Tabela 4 dá 72,5 para a mesma configuração) |
| Top-1 passage como resposta | **20,8** | 35,1 | **99,4** | **99,4** |
| Primeiras 2 sentenças do top-1 | 67,2 | **18,9** | 98,7 | 98,7 |

Os atalhos têm citação quase perfeita, mas o primeiro é derrubado pela **fluência** (MAUVE 20,8) e o segundo pela **correção** (EM 18,9, menos da metade de 40,4). Atenção: no caso (1) a correção (35,1) cai apenas 5,3 pontos em relação ao modelo; o texto do App. D diz que "fluência e correção são dramaticamente menores" para os dois, mas só fluência sustenta isso no top-1 (leitura minha dos números).

**(b) Inconsistências que encontrei no próprio artigo (todas pequenas, nenhuma muda as conclusões):**
1. **Tabela 11 × Tabela 4/7/19:** a linha "ChatGPT" da Tabela 11 mostra **precision 63,0**, enquanto a mesma configuração (ChatGPT VANILLA, GTR top-5, ASQA) tem **72,5** nas Tabelas 4, 7, 15, 16, 17 e 19; MAUVE (66,6), EM (40,4) e recall (73,6) coincidem. Hipóteses minhas, não verificáveis: erro de digitação ou outra execução. **A verificar** no repositório antes de citar o 63,0. Se fosse 72,5, a diferença para o atalho (99,4) seria de 26,9 pontos.
2. **Tabela 4 × Tabela 19** (mesmo experimento, valores diferentes por arredondamento ou erro): MAUVE do VANILLA 66,6 contra 66,8; SUMM recall 68,9 contra 68,8; CLOSEDBOOK EM 38,3 contra 38,2; INLINESEARCH: citation precision 58,2 (Tabela 4) contra 58,3 (Tabela 19), e a Tabela 19 traz ROUGE-L = 58,2, valor fora do padrão das demais linhas (29–40), provável erro de digitação (hipótese minha).
3. **Tabela 6 × Tabela 21:** CLOSEDBOOK citation recall/precision 15,5 contra 15,4; e um valor corrompido em `SNIPPET` ("2.w") na Tabela 21.
4. **Instrução × métrica:** o prompt do VANILLA (Tabela 23) pede "cite apenas um subconjunto mínimo suficiente", mas a métrica de precision **não exige** conjunto mínimo (§3.3). Não é erro, é uma diferença entre o que se pede ao modelo e o que se mede.

## 5. Limitações (as do autor e as que eu identifico)

**Do autor (§Limitações, §3.3, App. E):**
- MAUVE é sensível a tamanho e pode dar resultados instáveis.
- As sub-claims do ELI5 podem **não cobrir todas as respostas possíveis**, por a pergunta ser aberta.
- A qualidade da citação é **limitada pela acurácia do NLI**; o NLI não detecta "suporte parcial", o que gera **precision menor que a humana** (também App. G.5: 66,1% de precision na detecção de irrelevantes). Tentaram usar ChatGPT para julgar suporte parcial e o resultado foi "ruim" (App. E).
- Os datasets não cobrem raciocínio multi-hop, matemática nem código.
- Foco em *prompting*, sem treinar o modelo.

**Minhas (marcadas como interpretação):**
- **Sem perguntas irrespondíveis e sem abstenção.** Todo exemplo tem resposta. Pesquisei o texto por "unanswerable", "abstain", "refus", "don't know": **nenhuma ocorrência**. Uma resposta "não sei" sem citação cai na regra de recall 0 (e a regra de precision zera as demais). Logo o ALCE não diz como tratar recusa; qualquer adaptação nossa é extrapolação.
- **O verificador é um NLI de 2022 (T5-11B), não um LLM-juiz.** O κ de 0,698/0,525 vale para *esse* verificador, com premissas de 100 palavras de Wikipedia/Sphere e perguntas respondíveis. Não se transfere automaticamente para um juiz gpt-4o-mini sobre abstracts de arXiv.
- **κ moderado (0,525) na precision**: pelo critério de aceite exemplificado no nosso roadmap (P11, κ ≥ 0,6), a *própria* métrica de precision do ALCE **não passaria** (interpretação minha; o roadmap diz "ex.: κ ≥ 0.6", não é regra fechada).
- **Validação pequena e estreita:** 100 exemplos, 3 sistemas, 2 datasets (sem QAMPARI), número de anotadores por item não encontrado. Sem intervalos de confiança para o κ no texto que li.
- **Recall é "tudo ou nada" por afirmação**: uma afirmação com 2 fatos e só 1 sustentado vale 0, igual a uma totalmente inventada. Não separa "parcialmente sustentada" de "não sustentada".
- **Risco de Goodhart:** o RERANK escolhe pela própria métrica automática; os próprios autores veem divergência com o humano nesse caso (Tabela 8: 83,9 automático contra 79,3 humano) e confirmam em humanos que o ganho é real, mas menor.
- **A citação aponta para um índice** (`[1]…[5]` entre as passagens fornecidas), então o ALCE nunca precisa tratar um **ID citado inexistente**. No nosso sistema o modelo cita um `arxiv_id` livre e pode inventá-lo (extensão nossa, ver §6).
- **Atribuição não é correção:** uma afirmação falsa pode ser "sustentada" por uma fonte errada (o NLI só vê premissa e hipótese). O artigo trata isso reportando correção em separado.

## 6. Reflexões ancoradas no NOSSO projeto

Referência de pontos: P1–P11, D-1, D-2, L-3 em `docs/roadmap/ckpt-0.6.1b-metrics-roadmap.md`; plano das métricas em `docs/roadmap/ckpt-0.6-plan.md` §3.2 (o `citation_accuracy` **não tem número P** no roadmap; está planejado como "regex extrai `[2609.xxxxvN]` + juiz por citação", `evaluators.py`, lote 0.6.3). Sobre os dados: 250 papers, 843 chunks de ~420 caracteres, só abstracts (roadmap §3).

**R1 — Separar `citation_accuracy` em recall e precision (APOIA o desenho, DESAFIA a versão de uma nota só).**
O plano 0.6 define uma única nota por citação: "o chunk daquele `arxiv_id` suporta a frase citada?". Isso equivale a um *citation precision* sem pré-condição de recall e **não mede afirmação sem citação** (a metade "recall" do ALCE). O ALCE diz que prioriza o recall "porque implica uma resposta bem sustentada e verdadeira" (§3.3, tradução minha), e que a precision serve à experiência do usuário (menos revisão de fontes supérfluas). **Afeta:** `citation_accuracy` (0.6.3, `evaluators.py`). **Adotaríamos:** duas saídas por resposta, `citation_recall` (fração de afirmações sustentadas pelas suas citações; afirmação sem citação conta 0) e `citation_precision` (fração de citações não irrelevantes), calculadas sentença a sentença, com a regra "precision exige recall = 1" do §3.3. Nota: o Magesh 2025 (outra ficha) também pede distinguir *citação que não sustenta* de *afirmação sem citação*; os dois artigos convergem aqui (interpretação minha).

**R2 — A premissa de cada citação é a concatenação dos chunks daquele `arxiv_id` no contexto recuperado (APOIA; adaptação nossa).**
O ALCE concatena **todas as passagens citadas** da afirmação (`concat(Ci)`) e usa isso como premissa (§3.3). No nosso caso, uma citação é um `arxiv_id`, e o paper pode ter 2–5 chunks no top-5. A premissa deve ser a concatenação dos chunks **recuperados** daquele `arxiv_id` (não o abstract inteiro do corpus), pois só o que o modelo viu pode sustentar a afirmação. **Afeta:** `citation_accuracy` e **D-1** (gold por chunk × por paper): a citação é em nível de paper, enquanto o `gold_chunk_ids` das slices `answerable`/`persona` é em nível de chunk. A premissa em nível de paper é coerente com o que o modelo cita, mas **não** diz se o chunk certo estava lá (isso continua sendo recall@k). **Não decide D-1**; só mostra que a citação mora no nível de paper (interpretação minha). Uma vantagem: o ALCE mostra que a unidade de verificação pode ser o **conjunto** de passagens da afirmação, e não cada chunk isolado, o que nos livra do problema "chunk diferente do gold mas que também responde".

**R3 — Citação de ID ausente do contexto = citação inválida (extensão nossa; o ALCE não precisa disso).**
No ALCE o modelo cita `[1]…[5]`, índices de passagens fornecidas, então o caso "citou algo que nunca recebeu" não existe. No nosso formato `[arxiv_id]` o modelo pode citar um paper real que não foi recuperado, ou inventar um ID. **Regra proposta:** se o `arxiv_id` citado não está em `retrieved_context`, a citação é **inválida**: não entra como suporte (conta no denominador de precision como erro) e a afirmação fica sem suporte para efeito de recall. Reportar o número de citações inválidas como contador separado (o "fabricated" do Magesh). **Afeta:** passo 1 (regex) de `citation_accuracy`. **Apoio do artigo:** nenhum direto; é lacuna que o desenho do ALCE não cobre (a verificar se o repositório trata índices fora do intervalo).

**R4 — Abstenção: isentar do recall (DESAFIA o uso direto; ligação com P7 e P8).**
O ALCE não tem perguntas irrespondíveis (ver §5), e uma recusa sem citação dá **recall 0**. Se aplicássemos a métrica crua, os 15 `unanswerable` + 10 `stale` (25 de 100 exemplos, roadmap P8) penalizariam justamente o comportamento **correto**. **Proposta:** quando o detector de abstenção (P7, `abstained(answer)`, que hoje é frágil por `startswith`) disser que houve recusa, `citation_recall` e `citation_precision` ficam **`None`/pulados** (o mesmo tratamento do "open-topic skip" do plano 0.6), nunca 0. Corolário: as métricas de citação só devem ser agregadas nas respostas **que tentaram responder**; quem mede se a recusa foi apropriada é `abstention_quality` (P8). **Atenção:** a qualidade do detector (P7) passa a contaminar a citação (um falso "não abstém" numa recusa polida vira recall 0). Também é preciso decidir o que fazer com **frases sem fato** (preâmbulos como "Com base no contexto…"): o ALCE assume que quase toda sentença exige citação (§2); aplicar o mesmo às respostas do nosso `PROMPT_V1` é suposição a testar.

**R5 — Calibrar o juiz com respostas reais rotuladas, não só com o golden set (APOIA P11; ESTENDE o plano).**
O ALCE valida o verificador com **julgamento humano de saídas de sistemas reais** (100 exemplos × 3 sistemas, §6), e não com os gabaritos dos datasets. O nosso golden set de 100 exemplos tem pergunta e resposta/gold revisados, mas **não tem rótulos de "esta citação sustenta esta frase"** em respostas geradas. **Afeta:** P11 e a calibração do 0.8. **Adotaríamos:** rotular à mão uma amostra de **respostas do sistema** (afirmação + citações, com rótulo "sustenta totalmente / parcialmente / não") e medir κ juiz × humano **por métrica** (recall e precision separados; no ALCE deram 0,698 e 0,525). **Cuidados:** (i) o juiz `gpt-4o-mini` é o mesmo gerador (P11, viés de autopreferência), então a calibração precisa também comparar contra um juiz distinto; (ii) o ALCE usa 3 níveis no rótulo humano (totalmente / parcialmente / não, App. F.3) mas só 2 no NLI; rotular em 3 níveis nos daria a medida de "suporte parcial", que o NLI do ALCE não consegue (§3.3, App. E), e o juiz LLM poderia ser pedido a separá-lo. O κ do ALCE é **referência de ordem de grandeza**, não meta, por ser outro verificador, outro corpus e outro tipo de pergunta.

**R6 — Citação sozinha é gameável: reportar junto de correção e de uma checagem de cópia (APOIA P9; ESTENDE).**
A Tabela 11 mostra que citar o top-1 passage dá ~99% de recall e precision. No nosso projeto, um modelo que **copia o chunk e o cita** teria citação alta e dependeria só de outra métrica para ser pego. O ALCE usa **fluência (MAUVE)** e **correção**; nós não temos MAUVE (e o próprio autor o chama de sanity check instável), e a correção do plano é `token-F1` vs gold answer (P9, métrica fraca, roadmap). **Afeta:** P9 e o relatório do EXP-0. **Adotaríamos:** (i) nunca reportar `citation_recall/precision` isoladas; sempre ao lado de uma métrica de correção/completude (`completeness_judge`, relevância, P9); (ii) como *sanity check* barato nosso (interpretação minha, não vem do artigo), medir **razão de cópia** (fração da resposta que é substring dos chunks citados) e comprimento; (iii) tratar queda de correção com citação alta como alerta. **Não adotaríamos** MAUVE nem o claim recall via `text-davinci-003` (modelo descontinuado, sub-claims gerados por LLM com 93,3% de qualidade e NLI com 80,0% de acerto: teto do juiz automático, App. A).

**R7 — O que o artigo diz sobre retrieval reforça a ordem de leitura das métricas (APOIA; ligação com D-2).**
Figura 4: mesmo com o oráculo, a correção fica abaixo do recall@k do retriever (§5.3); logo **recall@k alto não garante resposta correta**, e uma `citation_recall` baixa pode ser falha do gerador mesmo com o chunk certo presente. Isso sugere (interpretação minha) condicionar as métricas de geração a `hit@k = 1` em um relatório à parte, para separar "retriever falhou" de "gerador falhou". Sobre **D-2** (hit@k ≡ recall@k quando o gold tem 1 documento) e **L-3** (`ranx` como oráculo de teste): o artigo é **NEUTRO**; usa recall@k, não ranx nem MRR.

**R8 — Definição de relevância por cobertura da resposta (App. G.1) (APOIA a direção de D-1, sem decidir).**
O oráculo do ALCE é construído **por conteúdo**: escolhe-se, de forma gulosa, as 5 passagens do top-100 que maximizam o recall da resposta. Não usa "ID gold" do dataset original, porque os datasets não têm gold na granularidade de 100 palavras. Isso é um precedente para o "gold de chunk ampliado e adjudicado por conteúdo" de D-1 (roadmap §3), em vez de depender só do ID do chunk de origem. **Neutro** quanto a decidir D-1.

**O que adotaríamos:**
(1) a decomposição recall/precision por afirmação, com premissa = concatenação das citações da afirmação (R1, R2); (2) a regra "recall = 1 como pré-requisito de precision = 1"; (3) a validação humana em saídas reais antes de confiar no juiz, com κ por métrica (R5); (4) reportar citação sempre junto de correção (R6); (5) a ideia de oráculo construído por conteúdo (R8).

**O que NÃO adotaríamos e por quê:**
(1) MAUVE e o claim recall com `text-davinci-003` (descontinuado; fluência não é nosso risco; MAUVE é instável); (2) o NLI T5-11B como juiz (nossa linha é LLM-juiz estruturado, 0.6.3; se um NLI local serviria como segundo juiz para o P11 é **a verificar**, o artigo não testa abstracts de arXiv); (3) os valores absolutos do artigo (73,6/72,5 etc.) como **metas**: são de Wikipedia/Sphere com perguntas respondíveis e passagens de 100 palavras; (4) aplicar a métrica sem isenção de abstenção; (5) tratar o κ de 0,698/0,525 como meta de aceite sem refazer a medida no nosso domínio.

## 7. Citações úteis (trechos literais curtos)

1. "its citation recall is 1 if and only if there is at least one citation" — §3.3 (Citation recall).
2. "this algorithm overlooks the scenario when one citation partially supports the statement" — §3.3 (Citation precision).
3. "Both cases have almost-perfect citation scores" — §3.4 (atalhos "top-1 passage" e "first 2 sents").
