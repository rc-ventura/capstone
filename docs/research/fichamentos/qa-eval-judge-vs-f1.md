# QA: juiz LLM vs. EM/F1 — Ho et al. (2504.11972) e Wang et al. 2023 (2305.12421)

Ficha dupla: uma parte por artigo (A e B) + comparação final (C) com a recomendação de métrica para o nosso projeto.

- **Referências completas e links:**
  - **[A]** Ho, X.; Huang, J.; Boudin, F.; Aizawa, A. *Reassessing Extractive QA Datasets at Scale: LLM-as-a-Judge and In-Depth Analyses* (título da v1/abstract original, usado no roadmap: *LLM-as-a-Judge: Reassessing the Performance of LLMs in Extractive QA*). arXiv:2504.11972 (**v3, 29/05/2026**; texto lido = v3). <https://arxiv.org/abs/2504.11972> · dados/código: <https://github.com/Alab-NII/llm-judge-extract-qa>. Venue: não consta no texto lido.
  - **[B]** Wang, C.; Cheng, S.; Guo, Q.; Yue, Y.; Ding, B.; Xu, Z.; Wang, Y.; Hu, X.; Zhang, Z.; Zhang, Y. *Evaluating Open-QA Evaluation*. NeurIPS 2023, Datasets and Benchmarks Track. arXiv:2305.12421 (v4, 23/10/2023). <https://arxiv.org/abs/2305.12421> · <https://github.com/wangcunxiang/QA-Eval>. (Nota: não é "Li et al."; o primeiro autor é Cunxiang Wang.)
- **Confiança da leitura:** **integral** para ambos (corpo completo + tabelas; do [B] o corpo e as Tabs. 1–7 foram lidas; **apêndices C–E do [B]** — Tab. 8 precisão/recall, Tab. 11 de erros por categoria, Tab. 12 de prompts, definições do E.4 — **não foram lidos em detalhe**; do [A] não li os apêndices A–C além do que o corpo referencia: guideline de anotação, limiares de F1 e exemplos de auto-preferência). Figuras: só legendas/texto. **Versão:** os números de [A] são os da v3; como o título mudou, os números da v1 podem diferir (o abstract da v3 repete 0,22/0,40/0,85).
- **Convenção:** **[Artigo]** = afirmado/medido; **[Interpretação]** = leitura minha.

---

# PARTE A — Ho et al. (2504.11972): juiz LLM vs EM/F1 em QA extrativo

## A1. Problema que o artigo ataca

EM (Exact Match: a resposta é *idêntica* a uma resposta de referência, após normalização) e F1 de tokens (sobreposição de palavras entre predição e referência) "frequentemente falham em refletir o desempenho real" em QA extrativo, subestimando respostas corretas escritas de outra forma (Fig. 1: gold "EPA", predição "Environmental Protection Agency (EPA)" → EM=falso, F1=0,4, juiz=verdadeiro). Estudos prévios com LLM-as-a-judge em QA (Kamalloo 2023; Verga 2024; Adlakha 2024) não cobrem bem: **tipos de resposta**, **auto-preferência** e **robustez a variações de prompt** (§1). O artigo faz esse estudo em **quatro datasets** e **várias famílias** de LLM.

## A2. Método

- **Datasets (§3, Tab. 1):** Quoref, DROP, HotpotQA, 2WikiMultiHopQA; ~1.000 amostras de cada (**4.193 no total**: Quoref 1.051; DROP 1.018; HotpotQA 1.124; 2Wiki 1.000), com 7 tipos de resposta obtidos por regras (place, name, job, date, number, year, string). **Exclui** perguntas booleanas (EM/F1 já funcionam). Critérios: usam EM/F1, não resolvidos (<95%), têm contexto, respostas diversas.
- **Modelos de QA (§4.1):** 8 modelos instruídos — Mistral-7B v0.1, Mixtral 8x7B, Qwen 2 (7B, 72B), Gemma 2 (9B, 27B), Llama 3.1 (8B, 70B); prompt zero-shot CoT (Apêndice B.1).
- **Juízes:** Mistral-7B-Instruct-v0.3, **Llama 3.3 70B**, **Qwen 2.5 72B**; **6-shot** (4 exemplos escolhidos à mão + 2 de Verga et al. 2024); entrada: pergunta, gold, predição e contexto; saída: **CORRECT** ou **INCORRECT** (binário); greedy decoding (§4.2).
- **Rótulo humano (§5.1):** 200 instâncias (50/dataset), cada uma com as 8 respostas dos 8 modelos; 2 autores como anotadores, **1 anotador por amostra**, com discussão de casos ambíguos; excluídas as com gold incorreto → **161 amostras × 8 = 1.288 respostas** rotuladas Correct/Incorrect.
- **Métrica de concordância:** **correlação de Pearson** entre o rótulo humano (0/1) e a saída do juiz (0/1) (mede o quanto o juiz acompanha o humano; 1 = perfeito). Para EM/F1 a mesma correlação é calculada com F1 **binarizado** em limiar **0,5** (testados 0,3/0,4/0,6/0,7/0,8).
- **Auto-preferência (§6.2):** 4 modelos QA (Llama 3.1 8B/70B, Qwen 2 7B/72B) × 7 juízes. **Definição própria:** há viés quando **todos os demais juízes** dizem INCORRETO e **só o juiz que é o próprio modelo-QA** diz CORRETO (limiar 100%; também 83% e 67%); reportado como % dentro das respostas com EM falso. **Sibling-preference:** igual, com juiz da mesma família (Llama 3.3 70B para Llama 3.1; Qwen 2.5 para Qwen 2).
- **Robustez do prompt (§6.3, Tab. 7):** 6-shot (inicial), zero-shot, 2-shot, sem contexto, troca de palavras.
- **Uso prático:** o juiz é aplicado **somente às respostas com EM falso** (as com EM=1 recebem nota 1); a comparação com "todas as amostras" tem gaps mínimos (§5.2, Apêndice C.2).

## A3. Principais contribuições

1. Estudo em larga escala (4 datasets, 8 modelos de QA, 3 juízes) mostrando que o juiz LLM correlaciona muito mais com humanos que EM/F1.
2. Primeira análise **por tipo de resposta**.
3. Teste de **auto-preferência e sibling-preference**: nenhuma evidência no regime extrativo.
4. Robustez a variações de prompt; zero-shot sem contexto frequentemente é o melhor.
5. Dados e código abertos.

## A4. Resultados-chave

- **Os números "0,22 / 0,40 / 0,85" (Resumo; Tab. 2; §5.1):** são **correlações de Pearson médias** com o rótulo humano (1.288 respostas), **sobre 8 modelos de QA**: **EM = 0,220** (média de 7: o Llama 3.1 70B dá NaN porque nenhum EM=1, e a coluna fica constante), **F1 = 0,404** (binarizado em 0,5), e juiz: **Mistral-7B v0.3 = 0,653; Llama 3.3 70B = 0,750; Qwen 2.5 72B = 0,847 (≈0,85)**. Por modelo de QA, a correlação do Qwen 2.5 chega a **0,945** (Llama 3.1 8B) e **0,908** (Gemma 2 27B), mas só **0,723** para o Mixtral (Tab. 2). "Até 0,85" é, portanto, a **média do melhor juiz**, não um máximo.
- **O limiar de F1 muda o resultado:** correlações médias do F1 binarizado: 0,3 → 0,451; 0,4 → 0,432; 0,5 → 0,404 (base); 0,6 → 0,379; 0,7 → 0,242; 0,8 → 0,237 (§5.1). Mesmo o mais indulgente (0,451) fica abaixo do pior juiz (0,653).
- **Gap EM/F1 vs juiz nas respostas com EM=0 (Tab. 3):** o juiz dá nota muito mais alta; **maior gap = 81,9 pontos** (EM vs Llama-70B-juiz em HotpotQA, para Mixtral 8x7B); menor = 15,9 (2Wiki, Gemma 2 9B). Exemplo (Tab. 3): HotpotQA, Mixtral 8x7B: **EM 7,1 / F1 26,1 / juiz 84,6–89,0** (Mistral 85,9; Llama 89,0; Qwen 84,6).
- **Por tipo de resposta (Tab. 4, juiz Qwen):** date 1,000 (n=16), number 0,899 (336), name 0,862 (464), string 0,862 (160), place 0,771 (240), **job 0,352 (72)**. O juiz é "menos rigoroso que o humano" em respostas com várias profissões (Tab. 5, exemplo 1: gold "actor" vs predição com várias profissões → humano F, modelo T).
- **Auto-preferência (Tab. 6):** limiar 100%: Llama 3.1 8B **5,77%**; Llama 3.1 70B 0,26%; Qwen 2 7B 0,63%; Qwen 2 72B 0,14% → os autores concluem "sem viés"; com limiares 83%/67% o Llama 3.1 8B sobe a 12,04%/14,77%. **Sibling-preference:** menos pronunciada que a auto-preferência (Apêndice C.3).
- **Prompts (Tab. 7, correlação média):** Qwen 2.5: 6-shot 0,847; zero-shot 0,877; 2-shot 0,850; **sem contexto 0,909**; troca de palavras 0,854; variância 0,0007 (Mistral 0,0037; Llama 0,0014).

## A5. Limitações

**Dos autores (seção Limitations):** só LLMs **open-source**; ~1.000 amostras/dataset (não os conjuntos completos); **avaliação humana pequena**: 161 amostras (1.288 respostas), **um anotador por resposta**.

**Que eu identifico [Interpretação]:**
- **Regime extrativo/curto, com gold verificável**: os próprios autores restringem as conclusões a QA "extrativo ou de resposta curta" (§1). **Nada** sobre respostas **geradas/longas**, embora afirmem que o problema "fica ainda mais pronunciado com IA generativa" (§1), sem medir.
- **Anotadores = autores**, sem κ inter-anotador (cada amostra tem 1 rótulo).
- **Correlação de Pearson entre dois vetores binários** (equivale ao coeficiente phi), com F1 **dicotomizado** a 0,5: o resultado de EM/F1 (0,22/0,40) depende dessa binarização arbitrária; um F1 *contínuo* contra a nota humana contínua não foi analisado.
- **Definição de auto-preferência fraca**: exige unanimidade dos demais juízes (conservadora), não compara com um **baseline esperado** e usa **apenas 4 modelos QA** (todos de 2 famílias). Llama 3.1 8B chega a 5,77% (12–15% com limiares mais folgados) e o texto o chama de "relativamente pequeno". "Não há viés" é uma leitura generosa; **"não detectado por este critério"** seria mais correto.
- Os juízes da análise principal são **mais novos/maiores** que os modelos de QA (Llama 3.3 70B e Qwen 2.5 72B vs Llama 3.1 e Qwen 2); o teste "mesmo modelo" só cobre 4 modelos (Llama 3.1 8B/70B, Qwen 2 7B/72B) julgando as próprias respostas.
- Zero-shot sem contexto ser o melhor **sugere que a tarefa é fácil** (comparar duas respostas curtas); não transfere para juízos com muita informação (faithfulness).

## A6. Reflexões ancoradas no NOSSO projeto

**R-A1 — Token-F1 como métrica de qualidade (P9; `_token_f1` e `f1_summary_evaluator` em `evaluators.py`; ckpt-0.6 §3.3).** Posição: **(i) APOIA** a desconfiança no token-F1, **(com ressalva de escopo)**. Os 0,40 (F1) vs 0,85 (juiz) são **reais**, mas são de QA **extrativo curto** com 8 modelos pequenos/medianos e **1.288** julgamentos; **a tarefa deles é o oposto da nossa**: respostas de 3 frases geradas por `gpt-4o-mini`, com citações, e gabarito *synthetic-by-construction*. A ressalva do roadmap (§D) está correta; eu acrescento: **(a)** "até 0,85" é a média do melhor juiz (0,847), não um piso garantido; **(b)** o F1 deles foi binarizado em 0,5 (e vai de 0,24 a 0,45 conforme o limiar); **(c)** o artigo mostra **subestimação** (falso negativo) de EM/F1, **não** que o F1 "premia respostas incorretas por sobreposição" — **essa segunda afirmação do roadmap §B/P9 não é demonstrada por este artigo** (nem por [B], que observa FP raros do lexical). **Adotaríamos** a evidência para justificar o *rebaixamento* do token-F1 a "diagnóstico lexical" e o teste local (Spearman/κ nas labels). **Não adotaríamos** "0,85" como meta nossa.

**R-A2 — Juiz de qualidade reference-based (P9; `correctness_judge`).** Posição: **(i) APOIA.** O desenho do artigo é praticamente o que planejamos: juiz recebe **pergunta + gold + predição**, saída **binária**. Lições operacionais: (a) **zero-shot ≥ 6-shot** e **sem contexto ≥ com contexto** (Tab. 7) — para um juiz de correção vs gabarito, o contexto recuperado é dispensável (e evita misturar com faithfulness); (b) a variância entre prompts é baixa (0,0007–0,0037) para juízes fortes, **alta** para o 7B (0,0037): confirma que o juiz **não deve ser pequeno demais**; (c) **tipos "job"/multi-valor** (corr. 0,352) são o ponto fraco: no nosso corpus, perguntas `multi-doc` e `completeness` (vários itens) são o análogo — **exigir rubrica "todos os pontos do gold presentes"**, não só "bate com o gold".

**R-A3 — Juiz = gerador (P11; `JUDGE_MODEL`).** Posição: **(ii) DESAFIA** parcialmente a urgência, **mas sem poder de refutação** (ver A5): "sem auto-preferência" em QA extrativo com gabarito curto **não contradiz** Wataoka (preferência de estilo em pairwise aberto). **Adotaríamos** o *desenho de medida* deles como barato: comparar, **para o mesmo conjunto de respostas**, o veredito do juiz-do-gerador vs. juízes de outra família, e contar discordâncias do tipo "só o próprio juiz acha correto" (versão simples da Tab. 6). **Não adotaríamos** a conclusão "sem viés".

**R-A4 — Uso híbrido barato→caro (custo).** Posição: **(iii) NEUTRO.** O artigo aplica o juiz **só nos EM falsos**. Para nós, o custo de 100 exemplos é irrelevante, então **não adotaríamos** a economia; **mas** a ideia de *filtro determinístico primeiro* é útil para a **abstenção** (P7: regra de abstenção + juiz), já planejado.

---

# PARTE B — Wang et al. 2023 (2305.12421): "Evaluating Open-QA Evaluation" (EVOUNA)

## B1. Problema que o artigo ataca

Open-QA é usado para estimar a **factualidade** de LLMs, mas é avaliado por **Exact Match / contenção**, que "não considera a variação de expressão" (ex.: "Lionel Messi" vs "Messi") e é "inaplicável" a respostas detalhadas de LLMs (§1). Os autores criam a tarefa **QA-Eval** (avaliar os *avaliadores* de QA) e o dataset **EVOUNA** para medir quais avaliadores concordam com o humano.

## B2. Método

- **Dados (§3.1, §4, Tab. 1):** Natural Questions (dev; **3.610 → 3.020** após filtro de perguntas temporais/gold ruim) e TriviaQA (**2.000 → 1.938**). **5 sistemas** de Open-QA: DPR+FiD (retriever-reader com T5-large), GPT-3.5 (text-davinci-003, T=0), ChatGPT-3.5 (gpt-3.5-turbo, T=0), ChatGPT-4 e BingChat (via web, abr/2023). Cada resposta é rotulada pelo humano como correta/incorreta (dados: 5 × (3.020+1.938) respostas).
- **Anotação humana (§4):** feita **pelos autores** com guidelines (Apêndice C.2); **κ de Cohen** entre anotadores em 500 amostras de cada subset: **86,4–100** (Tab. 2; NQ-BingChat 86,4 o menor; TQ-FiD 100).
- **Avaliadores comparados (§3.3):**
  - **Lexical Matching:** para o DPR+FiD, **EM** (a saída é idêntica a um gold); para as respostas longas dos LLMs, **contenção** — correta se **algum gold aparece dentro** da resposta. **[Interpretação importante]**: **não é token-F1**; não há F1 de tokens neste artigo. O "F1" das tabelas é o **Macro-F1 de classificação** do avaliador contra o rótulo humano (média do F1 das classes "correta" e "incorreta").
  - **BERTScore** (similaridade por embeddings contextuais; ref = pergunta+gold; hip = pergunta+resposta; **limiar τ = 0,5**). BART-Score e GPT-Score foram descartados por darem nota contínua sem separar correto/incorreto.
  - **GPT-3.5 (text-davinci-003)** como juiz, T=0, prompt Yes/No com pergunta + lista de golds + resposta.
  - **"Another Human"** como referência (teto).
- **Métricas dos avaliadores:** **acurácia** e **Macro-F1** vs. rótulo humano (tarefa binária); mais o **ranking** dos 5 sistemas (Tab. 6).
- **Engenharia de prompt (§6.4, Tab. 7):** ignorar informações de fundo, dar razões, **CoT**, **In-Context Learning** (4 exemplos escolhidos por k-means, entre domínios).
- **Normalização:** remove símbolos/fontes das respostas do BingChat e perguntas finais ("Quer saber mais?") para não induzir o juiz a responder em vez de julgar (§4).

## B3. Principais contribuições

1. Tarefa **QA-Eval** e dataset **EVOUNA** (humano em 5 sistemas × NQ/TQ).
2. Evidência de que **EM/contenção subestima** e **troca o ranking** dos sistemas.
3. Comparação de **3 famílias** de avaliador (lexical, neural, LLM) com **taxonomia de erros** (categorizados à mão).
4. Análise do efeito do tipo de prompt no juiz.

## B4. Resultados-chave

- **EM/contenção subestima (Tab. 4):** NQ — DPR+FiD **humano 68,9 vs lexical 59,2**; GPT-3.5 65,5 vs 50,7; ChatGPT-3.5 73,0 vs 57,9; ChatGPT-4 78,8 vs 61,8; BingChat 79,9 vs 65,4. TriviaQA: 81,5 vs 73,5; 78,4 vs 71,0; 84,45 vs 76,7; 90,2 vs 82,1; 89,6 vs 81,6. **Gap** humano − lexical (aritmética minha, Tab. 4): de **7,4 a 17,0 pontos** (maior: ChatGPT-4 em NQ, 17,0).
- **Acerto do avaliador vs humano (Tab. 5, acc/Macro-F1; NQ):**

| Subset NQ | Lexical | BERTScore | GPT-3.5 (juiz) | Outro humano |
|---|---|---|---|---|
| FiD | 89,7/92,0 | 75,1/83,5 | **93,6/95,3** | 96,3/97,4 |
| GPT-3.5 | **84,8/86,9** | 69,5/77,6 | 84,1/87,2 | 96,8/97,8 |
| ChatGPT-3.5 | 80,3/84,9 | 72,8/81,2 | **82,2/86,9** | 95,6/96,5 |
| ChatGPT-4 | **82,5/87,6** | 76,0/84,3 | 80,9/86,9 | 96,6/97,9 |
| BingChat | **82,3/87,8** | 67,5/77,6 | 69,5/77,2 | 95,5/97,2 |

  → **Acurácia do juiz GPT-3.5 menos a do lexical (NQ, aritmética minha):** FiD +3,9; GPT-3.5 −0,7; ChatGPT-3.5 +1,9; ChatGPT-4 −1,6; **BingChat −12,8**. Ou seja, o juiz ganha pouco nas respostas curtas/medianas e **perde** nas respostas longas e carregadas de informação extra (ChatGPT-4, BingChat). Em TriviaQA (acurácia): juiz melhor em FiD (95,7 vs 91,8), ChatGPT-3.5 (92,5 vs 92,3) e ChatGPT-4 (92,4 vs 91,1); lexical melhor em GPT-3.5 (92,3 vs 91,2) e **BingChat (89,8 vs 80,9)**.
- **BERTScore é o pior** na maioria (§6.3: "poorest performance among the three"): sensível a **limiar** ("muitos datasets são altamente sensíveis ao limiar") e a **respostas estendidas** com informação extra; distribui erros igualmente entre FP e FN (§6.1, Fig. 1).
- **Perfil de erro (§6.1, §6.3; Fig. 1–2):** lexical e GPT-3.5 têm **poucos falsos positivos** — lexical "marca como incorretas respostas que humanos consideram corretas, mas raramente o contrário"; erros típicos do lexical: paráfrase, sinônimo, **variação estrutural** ("8 September 2010" × "September 8, 2010"). O juiz GPT-3.5 sofre com **informação extra e formatação do BingChat** ("excluir BingChat melhora significativamente o desempenho"), erra por **paráfrase** (o erro mais comum do juiz, §6.3), **overgeneralization**, e às vezes **usa conhecimento próprio e ignora o gold** (aceitou "Bidhan Chandra Roy" para gold "Prafulla Chandra Ghosh").
- **Ranking (Tab. 6):** **nenhum avaliador reproduz o ranking humano** dos 5 sistemas. NQ: humano → GPT-4 1º, BingChat 2º, GPT-3.5 3º, FiD 4º, ChatGPT-3.5 5º; lexical → BingChat 1º, GPT-4 2º, FiD 3º, ChatGPT-3.5 4º, GPT-3.5 5º; GPT-3.5-juiz coloca BingChat **por último** (§5.2).
- **Prompts (Tab. 7, NQ, acc/F1):** BingChat: original 69,5/77,2; **CoT 80,4/87,1**; ICL 75,3/82,5; ignorar contexto 65,7/73,4; **dar razões 55,6/62,2**. ChatGPT-4: original 80,9/86,9; CoT 86,0/91,2; razões 64,3/71,2. Mas **CoT piora no NQ-FiD** (93,6 → 90,2): efeito depende da distribuição (§6.4).

## B5. Limitações

**Dos autores (§7):** dados via API/web mudam (não reprodutível); **sem GPT-4 como avaliador** (limite de API); só parte de NQ e TriviaQA rotulada; **gold com erros** (mantidos).

**Que eu identifico [Interpretação]:**
- **Juiz fraco e antigo** (text-davinci-003): os resultados **não** representam juízes modernos; [A], com juízes bem maiores, mostra um quadro bem mais favorável ao LLM.
- **Rótulo humano dos próprios autores**, concordância κ alta, mas não é um teto independente.
- **Respostas "longas" aqui são ainda factoides** (uma entidade/data cercada de texto); não são respostas *sintéticas de 3 frases* com várias afirmações como as do nosso RAG. Nota de rodapé 3: o estudo "concentra-se em QA objetivo com respostas relativamente curtas".
- **"Lexical matching" ≠ token-F1:** a conclusão de que lexical "raramente dá FP" vale para **contenção do gold**, não para F1 de tokens, que mistura precisão e cobertura de palavras e **dá crédito parcial** a respostas erradas com muita sobreposição.
- **Inconsistência interna:** as notas humanas da **Tab. 6** não batem com as da **Tab. 4** (ex.: NQ GPT-3.5: 70,3 vs 65,5; ChatGPT-3.5: 63,0 vs 73,0; DPR+FiD: 69,7 vs 68,9). As afirmações sobre *ranking* (§5.2) dependem da Tab. 6; **tratar com cautela** (não encontrei explicação no texto lido).
- Prompt original do juiz pede só "Yes/No" (sem raciocínio); a Tab. 7 testa 4 variantes numa só família.

## B6. Reflexões ancoradas no NOSSO projeto

**R-B1 — "Juiz LLM é melhor que token-F1" é condicional (P9).** Posição: **(ii) DESAFIA** a leitura simplificada do roadmap §B/P9 ("0,22/0,40 vs 0,85"). Em respostas **longas/informativas**, o juiz de 2023 **não** bateu a contenção lexical (Tab. 5). **O que isso exige:** o juiz precisa de **prompt e modelo bons e validados** (a escolha do juiz importa tanto quanto "usar juiz"), e **a calibração local não é opcional** — reforça a decisão do roadmap de medir Spearman/κ nas labels antes de aposentar o token-F1.

**R-B2 — Falsos negativos do lexical vs falsos positivos (P9; `f1_summary_evaluator`).** Posição: **(i) APOIA** o diagnóstico do roadmap (paráfrase punida), **(ii) DESAFIA** a premissa de que lexical "premia respostas incorretas": aqui o lexical quase não dá FP. **[Interpretação]:** para o nosso caso, onde o gold é uma frase e a resposta tem 3 frases, o problema dominante do token-F1 é **sensibilidade ao comprimento/citações** (exemplo do roadmap: 0,36 para resposta correta) — **falso negativo**, consistente com os dois artigos. O risco de *falso positivo* aparece com **abstenções vs gold de abstenção** e com respostas longas que contêm as palavras do gold; ambos testáveis localmente.

**R-B3 — BERTScore/embeddings como alternativa barata (P9 §3(b): `compare_semantic_similarity` / similaridade de embeddings).** Posição: **(ii) DESAFIA.** Wang et al.: BERTScore foi o **pior avaliador**, sensível a limiar e a "resposta estendida com informação extra" — exatamente o formato de resposta do nosso RAG (3 frases + citações). O curso LangSmith, ao contrário, usa um **juiz** de similaridade semântica (nota 1–10) e não embedding puro. **Adotaríamos** embeddings **só como sinal secundário** (diagnóstico), **nunca** como substituto do juiz de correção; **não adotaríamos** um limiar fixo (0,5) sem calibração.

**R-B4 — Prompt do juiz de correção (P9/`correctness_judge`; 0.6.3).** Posição: **(i) APOIA** parcialmente. Wang: **CoT e ICL melhoram nas respostas longas** (BingChat 69,5 → 80,4 com CoT); **"dar razões" piora** (55,6) — mas Zheng et al. recomendam explicação antes do veredito. **[Interpretação]:** o efeito é instável e depende do modelo; **testar 2–3 variantes** (veredito direto; CoT antes do veredito; razões depois do veredito) **nas 100 labels** e escolher por κ. **Não adotaríamos** "dar razões" por padrão.

**R-B5 — Ranking de sistemas (gates do plano §2 de ≥10–15% de melhora; EXP-0).** Posição: **(iii) NEUTRO com alerta.** Nenhum avaliador reproduziu o ranking humano em Tab. 6 (que tem inconsistência com a Tab. 4; ver B5). **Consequência [Interpretação]:** ao decidir gates entre configurações, o que importa é a **correlação do ranking** do avaliador com o humano, não só o acerto por exemplo; reportar também **concordância de ranking/delta** na calibração do 0.8.

---

# PARTE C — Comparação final e recomendação

## C1. Quadro comparativo

| Aspecto | **[A] Ho et al. (2504.11972, v3)** | **[B] Wang et al. (2305.12421, NeurIPS'23)** |
|---|---|---|
| Tarefa | QA **extrativo** com contexto (Quoref, DROP, HotpotQA, 2Wiki) | **Open-QA sem contexto** (NQ, TriviaQA) — respostas de sistemas, algumas **longas** |
| Gold | span/curto, 1 gold | lista de golds; respostas dos LLMs **mais longas e verbosas** que o gold |
| Nº de amostras avaliadas por humanos | 1.288 respostas (161 itens × 8 modelos) | NQ 3.020 + TQ 1.938, ×5 sistemas (cada resposta rotulada); κ medido em 500/subset |
| Anotador | 2 autores, **1 por item**; sem κ | autores; **κ 86,4–100** (Tab. 2) |
| "Baseline lexical" | **EM** e **F1 de tokens** (binarizado em 0,5) | **EM** (FiD) e **contenção do gold** (LLMs); **não** há token-F1 |
| Juízes | Mistral 7B v0.3, Llama 3.3 70B, Qwen 2.5 72B (**open, 2024–25**) | GPT-3.5 text-davinci-003 (**2022–23**), BERTScore |
| Métrica de concordância | **Pearson** juiz×humano (média por modelo de QA) | **Acurácia e Macro-F1** vs humano; **ranking** dos sistemas |
| Resultado principal | EM 0,22; F1 0,40; juiz **0,65–0,85** | lexical subestima (até −17 pts); juiz ≈ lexical nas respostas longas; BERTScore pior |
| Respostas longas/generativas | **Não cobre**; só diz que "é pior" | Cobre parcialmente (respostas longas de LLMs), mas factoides |
| Auto-preferência | Testada, **"não há"** (definição frágil) | **Não** testada |
| Alternativas à token-F1 | juiz LLM; cita **BEM** (Bulian et al. 2022) só como trabalho relacionado (§2) | **BERTScore** (pior), contenção lexical, juiz LLM; BART/GPT-Score descartados |
| Embeddings/NLI | **Não testados** | BERTScore (embedding contextual) testado; **NLI não testado** |

**Concordam em:** (1) EM/lexical **subestima** respostas corretas parafraseadas; (2) o lexical quase não produz falso positivo; (3) juiz LLM captura paráfrase/sinônimo; (4) tipos de resposta ou formato (job/multi-valor em [A]; resposta com informação extra em [B]) são os pontos fracos do juiz. **Divergem em:** o **quão melhor** é o juiz — por causa do **juiz usado** (moderno e grande em [A]; antigo em [B]) e da **métrica** (Pearson sobre 0/1 vs acurácia/Macro-F1 vs ranking). **Nenhum dos dois** avalia **respostas generativas longas com várias afirmações e citações** nem **abstenção**.

## C2. Em que condições token-F1/EM falha e o juiz LLM ajuda (respostas às perguntas do team-lead)

- **Falha do EM/F1:** paráfrase/sinônimo/forma ("EPA" × "Environmental Protection Agency"; "8 September 2010" × "September 8, 2010"), **resposta mais longa que o gold** (informação extra), resposta correta com **contexto adicional** ("93,5" × "93,5%"; [A] Tab. 2, nota do NaN) — **falsos negativos** (Ho §1, Fig. 1; Wang §6.3).
- **Onde o juiz ajuda mais:** gold curto e verificável (número, data, nome) — [A] Tab. 4; juiz **forte**.
- **Onde o juiz é fraco:** respostas multi-valor/ambíguas (job 0,352, [A]); respostas longas com informação extra e formatação ([B], BingChat 69,5%); quando usa conhecimento próprio e ignora o gold ([B] §6.3).
- **Os números 0,22/0,40/0,85:** dataset = **4 datasets extrativos** (Quoref, DROP, HotpotQA, 2Wiki); **8 modelos de QA**; juízes **Mistral-7B/Llama-3.3-70B/Qwen-2.5-72B**; **1.288 julgamentos** (161 itens × 8 respostas), **Pearson**, F1 binarizado em 0,5. Cobrem **extrativo**, **não generativo longo**.

## C3. Melhor métrica de "qualidade da resposta vs gabarito" para o nosso caso (reflexão ancorada; P9, D-1 não afetado)

**[Interpretação, com base nas evidências acima; não é recomendação do artigo.]**

1. **Métrica principal: `correctness_judge` reference-based** — juiz LLM recebe **pergunta + resposta-gabarito + resposta gerada** (sem contexto recuperado, como em [A] Tab. 7), com **rubrica explícita**: *(a)* os fatos-chave do gabarito estão presentes e sem contradição; *(b)* informação extra **correta** é permitida; *(c)* resposta de abstenção é tratada à parte (P7/P8). Saída com escala curta (pass/partial/fail ou 0–1) — para respostas de 3 frases com vários pontos, **partial** evita a falha de [A] em multi-valor. **Modelo juiz de outra família** que o gerador (Wataoka; P11) e **forte** (a evidência de [A] vs [B] mostra que juiz fraco perde para lexical).
2. **Calibrar antes de confiar:** κ e acordo por juiz contra humano, com baseline trivial e teto humano (Zheng); comparar **com o token-F1 normalizado** (Spearman/AUC contra o rótulo humano) como P9 já prevê — essa comparação é o que **nenhum dos dois artigos faz para respostas generativas**.
3. **Token-F1 → diagnóstico** (cobertura lexical), com pré-processamento (remover citações `[2609.xxxxx]`; P9).
4. **Embeddings/BERTScore:** só diagnóstico (Wang: pior, sensível a limiar e a respostas estendidas). **NLI** (ex.: checagem de *entailment* claim a claim): **não coberto** por estes artigos; fica como alternativa a investigar nas fichas de RAGAS/RAGChecker, não aqui.
5. **Filtro determinístico opcional** (contenção do gold-entidade) como pré-triagem barata, como em [A] (juiz só nos EM falsos) — útil para `unanswerable` e para o relatório, **não** para a nota principal.

## C4. O que adotaríamos / não adotaríamos (resumo)

- **Adotaríamos:** juiz reference-based, forte e de outra família; teste local nas 100 labels; tipos de resposta/slices no relatório (como a Tab. 4 de [A]: `answerable` vs `multi-doc`); teste de 2–3 formatos de prompt; reportar correlação de ranking dos braços.
- **Não adotaríamos:** "0,85" como meta; BERTScore com limiar fixo; conclusão "sem auto-preferência" de [A]; dar razões após o veredito por padrão; generalizar [B] (juiz de 2022) para juízes modernos.

---

## 7. Citações úteis (máx. 3 no total)

1. "As shown in Table 2, EM and F1 scores correlate less with human judgments than any LLM-as-a-judge model." — [A] Ho et al., §5.1.
2. "It often marks answers that humans consider correct as incorrect, but rarely does the opposite." — [B] Wang et al., §6.3 (Lexical Matching).
3. "the LLM-evaluators tends to perform much worse on long answers with much additional information." — [B] Wang et al., §1 (Introdução).
