# Magesh et al. 2025 — "Hallucination-Free?" Confiabilidade de ferramentas de pesquisa jurídica com IA

- **Referência completa:** Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C. D., Ho, D. E. (2025). *Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools*. Journal of Empirical Legal Studies (JELS) 2025. Preprint: arXiv:2405.20362 [cs.CL], v1 de 30/05/2024 (a nota 1 do PDF diz "versão de 3 de junho de 2024", atualizada para incluir o Westlaw AI-AR). Links: <https://arxiv.org/abs/2405.20362> (PDF: <https://arxiv.org/pdf/2405.20362>); versão publicada: <https://onlinelibrary.wiley.com/doi/abs/10.1111/jels.12413>.
- **Confiança da leitura:** **integral, mas só do preprint arXiv v1** (24 páginas de corpo + referências + Apêndices A–C), lido a partir do texto extraído com `pdftotext`. A listagem do arXiv mostra só a v1. **A versão publicada (JELS 2025) não foi lida**: o site da Wiley devolveu HTTP 403 (bloqueio anti-bot). Logo, números e redação podem diferir na versão publicada. A Figura 4 foi conferida visualmente (página renderizada como imagem); as Figuras 1 e 5 não: a Figura 1 saiu só como rótulos de eixo na extração de texto, e na Figura 5 os valores não são legíveis (marcado "a verificar" onde isso importa). O protocolo de atribuição de causas da Tabela 6 não está descrito no artigo. Os rótulos **(a) o artigo afirma** e **(b) minha interpretação** estão marcados no texto.

## 1. Problema que o artigo ataca

**(a) O artigo afirma:** fornecedores de pesquisa jurídica (LexisNexis, Thomson Reuters/Westlaw, Casetext) anunciaram que RAG "elimina" ou "evita" alucinações, ou entrega citações jurídicas "100% hallucination-free" (nota 2 da §1, com as frases literais dos fornecedores), sem evidência empírica e sem definir "alucinação" (§1, §4.3). Como os sistemas são fechados, avaliar essas alegações é difícil (abstract). Os autores propõem "a primeira avaliação empírica **pré-registrada**" dessas ferramentas (abstract) e declaram quatro contribuições: (1) a primeira avaliação de ferramentas jurídicas proprietárias com RAG; (2) um dataset pré-registrado de 202 consultas; (3) uma tipologia para separar alucinação de resposta acurada; (4) evidência para a responsabilidade dos advogados de supervisionar e verificar saídas de IA (abstract; §1).

Pergunta implícita: *RAG resolveu a alucinação em pesquisa jurídica?* Resposta do artigo: não. A alucinação cai em relação ao GPT-4 sem RAG, mas persiste (§6.1).

**(b) Minha interpretação:** é um artigo de **avaliação de produtos fechados com rotulagem humana especializada**, não de métrica automática. O que mais serve ao nosso projeto não é o número (17–33%), e sim (i) a **definição operacional** de alucinação em duas dimensões (correção × fundamentação) e (ii) o **protocolo de rotulagem com concordância medida**. O número, por sua vez, não é transferível (ver §6, R6).

## 2. Método (passo a passo, com as definições exatas)

**Sistemas avaliados (§5.1).** Lexis+ AI (LexisNexis), Ask Practical Law AI (Thomson Reuters) e Westlaw AI-Assisted Research ("AI-AR"). Como referência, **GPT-4 em modo "closed-book"** (sem base externa; modelo `gpt-4-turbo-2024-04-09` via API, com um prompt de sistema pedindo para citar jurisprudência e não hedgear, §5.3). Os três produtos são caixas-pretas: o artigo diz não haver detalhe técnico publicado e que não se sabe quais parâmetros de recuperação (limiar de similaridade, top-k) são usados (nota 26).

**Dataset (§5.2, Tabela 2).** 202 consultas escritas por especialistas:
| Categoria | n | % do dataset | O que testa |
|---|---|---|---|
| General legal research (doutrina, bar exam, holdings) | 80 | 39,6% | uso paradigmático |
| Jurisdiction or time-specific (circuit splits, casos revertidos, mudanças na lei) | 70 | 34,7% | variação de jurisdição e tempo |
| False premise (usuário parte de uma premissa jurídica errada) | 22 | 10,9% | viés de aceitar a premissa ("contrafactual bias") |
| Factual recall (autor da opinião, ano, citação) | 30 | 14,9% | fatos que não exigem interpretação |

Conferência: 80+70+22+30 = 202 ✓. Fontes: 20 consultas do LegalBench (Rule QA) e 20 do BARBRI (preparação para o exame da ordem) verbatim; as outras 162 escritas ou adaptadas à mão (§5.2; 202−40 = 162 ✓). **Pré-registro** no Open Science Foundation em **22/03/2024**, antes de rodar as consultas (§5.2–5.3; nota 16: uma exceção, `changes-in-law-73`, e algumas reformulações menores, listadas no Apêndice B.1). Execução: Lexis+ AI, Practical Law e GPT-4 entre 22/03 e 22/04/2024; Westlaw AI-AR entre 23 e 27/05/2024 (§5.3). Uma conversa nova por consulta (nota 17).

**Definições de rótulo (§4, Tabela 1, Apêndice C), cada uma em linguagem simples:**

*Dimensão 1 — Correção (§4.1, Apêndice C.2):*
- **Correct** (a resposta é factualmente correta E relevante à pergunta). **Resposta parcialmente correta conta como Correct** (uma que não contém erro, mas não cobre toda a pergunta, §4.1; Apêndice C.1 diz que "Partially Correct" foi colapsado em "Correct" na análise final).
- **Incorrect** (a resposta contém **qualquer** afirmação factualmente falsa; o Apêndice C.2 acrescenta "material ou não para a resposta", ou seja, não há limiar de gravidade).
- **Refusal** (o modelo se recusa a responder **ou dá uma resposta irrelevante**, Tabela 1). No Apêndice C.1: "Irrelevant/Unhelpful" e "Stock Refusal" foram colapsados em Refusal.
- **Regra especial de pergunta com premissa falsa** (nota 8 da §4.1; Apêndice C.2): o comportamento desejado é refutar a premissa. Uma recusa que **menciona que nenhuma fonte pertinente foi encontrada** ("I cannot find any information on this topic") é codificada como **Correct**, enquanto uma recusa padrão sem essa indicação ("I cannot provide you with any information on this topic") é **Refusal**. Esta regra vale **só** para a categoria de premissa falsa.

*Dimensão 2 — Fundamentação ("groundedness"), só para respostas Correct (§4.2, Apêndice C.3):*
- **Grounded** (as proposições factuais-chave citam fontes jurídicas válidas e relevantes; suporte indireto é aceito).
- **Ungrounded** (alguma proposição material **não tem citação**; a afirmação não é falsa, só está sem fonte).
- **Misgrounded** (a proposição **é citada, mas a fonte não sustenta** a afirmação, ou é inaplicável, p.ex. jurisdição errada ou caso já revertido, §4.2).
- **Fabricated** (a resposta cita uma fonte **que não existe**; aparece **só no Apêndice C.3**, não na Tabela 1 nem no §4).
- **Not Applicable** (só para respostas sem proposição factual, isto é, Irrelevant/Stock Refusal).
- Regra de precedência (Apêndice C.3): se uma resposta é ao mesmo tempo ungrounded e misgrounded, "é rotulada com a ofensa mais grave: Misgrounded". Respostas Incorrect também recebem groundedness N/A (C.1b).

*Rótulos de nível mais alto (§4.3–4.4):*
- **Hallucination = Incorrect OU Misgrounded** (§4.3: "se um modelo faz uma afirmação falsa ou afirma falsamente que uma fonte sustenta uma afirmação").
- **Accurate = Correct E Grounded** (§4.4).
- **Incomplete = Refusal OU Ungrounded** (§4.4). Justificativa do artigo: uma resposta ungrounded "não faz nenhuma afirmação falsa", mas deixa de fornecer a informação (as autoridades) de que o usuário precisa.

Os três rótulos (Hallucination / Accurate / Incomplete) formam uma **partição** das 202 respostas por sistema: cada resposta cai em exatamente um (conferido nas barras da Figura 4: Lexis 17+18+65, Westlaw 33+25+42, Practical Law 17+63+20, GPT-4 43+8+49, todas somam 100%).

**Uma observação sobre o sentido de "grounded" (§4.2) — (a) o artigo afirma:** o uso é **jurídico**, não o de ciência da computação. Em CS, "groundedness" é aderência aos documentos fornecidos, "independentemente de relevância ou acurácia" desses documentos (Agrawal et al., 2023). Aqui, se o retriever traz um documento da jurisdição errada e o modelo o cita, a resposta é **Misgrounded**, "ainda que fosse tecnicamente grounded no sentido CS" (§4.2). Isso é o que separa o artigo da faithfulness estilo RAGAS.

**Protocolo de rotulagem e confiabilidade (§5.4, Apêndice C):**
- Cada resposta foi codificada à mão por **um de três** rotuladores especialistas em direito (autores), seguindo a rubrica do Apêndice C. Respostas foram codificadas com dois valores (correção e groundedness, C.1).
- Um **quarto rotulador** recodificou **48 respostas** amostradas aleatoriamente, estratificadas por modelo e tipo de tarefa (com leve sobreamostragem da tarefa de citação Bluebook, "tecnicamente difícil"). Ele **não conversou com os outros três nem viu os rótulos iniciais**; só leu a documentação escrita do Apêndice C (§5.4).
- Resultado: **κ de Cohen = 0,77** e **concordância de 85,4%** no rótulo final de 3 classes (correct / incomplete / hallucinated), "entre o rotulador de avaliação e os rótulos iniciais" (§5.4). Conferência: 41/48 = 85,4% ✓ (consistente com n = 48).
- O artigo rejeita "deixar as ferramentas se checarem" (LLM verificando LLM, citando Manakul 2023, Zheng 2023 etc.) como "inadequado para esta aplicação" (§5.4), porque aderência à autoridade exige avaliação qualitativa manual. Na §7 admite que "a tendência de avaliações baseadas em LLM pode resolver" o gargalo de custo, mas que o campo jurídico ainda é fechado.
- Nota 18: ao incluir o AI-AR, os autores fizeram **outra rodada de validação de todos os rótulos de alucinação**, e a acurácia do Practical Law subiu de 19% para 20% ("dentro dos limites da confiabilidade entre rotuladores").
- **O artigo não reporta intervalo de confiança do κ, nem o κ por classe.**

**Tipologia de causas (§6.3, Tabela 6).** Os autores observam os documentos recuperados (os produtos mostram a lista, não os trechos exatos) e comparam com a resposta, para propor causas prováveis: **naive retrieval** (falha em achar a fonte mais relevante), **inapplicable authority** (cita documento da jurisdição, estatuto ou tribunal errado, ou revertido), **sycophancy** (concorda com premissa falsa) e **reasoning error** (erro elementar de raciocínio com o texto certo na mão). As categorias **não são mutuamente exclusivas** e o artigo diz que "como são sistemas fechados" não consegue identificar um único ponto de falha por alucinação (§6.3). **O procedimento de atribuição (quem rotulou, regra de decisão, n por causa) não está descrito.**

## 3. Principais contribuições (lista)

1. Primeira avaliação empírica **pré-registrada** de ferramentas de pesquisa jurídica proprietárias com RAG (abstract; §1).
2. Dataset de 202 consultas em 4 categorias, com premissa falsa e jurisdição/tempo como categorias explícitas (§5.2, Tabela 2).
3. **Tipologia correção × fundamentação** e a definição "alucinação = incorreta OU misgrounded", mais as definições de *accurate* e *incomplete* (§4).
4. Protocolo de rotulagem com guia escrito (Apêndice C), quarto rotulador cego e κ = 0,77 em 48 itens (§5.4).
5. Catálogo de modos de falha com exemplos e Tabela 6 de causas contribuintes (§6.2–6.3), incluindo "inapplicable authority" como categoria específica do direito, que o artigo diz não ter sido explorada na literatura anterior (§6.3).
6. Argumento de que um erro de **citação real que não sustenta a afirmação** é pior do que fabricar um caso, porque é mais sutil e exige ler a fonte (§4.3).

## 4. Resultados-chave (com números e onde estão)

**(a) O artigo afirma — o que significa exatamente "17–33%":**

- O abstract diz que Lexis+ AI e Westlaw/Practical Law "cada um alucina entre 17% e 33% das vezes". O número vem da **Figura 4 (painel esquerdo)**: "porcentagens gerais de respostas accurate, incomplete e hallucinated". O **denominador são as 202 consultas, incluindo as recusas**. Ou seja, 17% é "fração das 202 respostas que foram Incorrect ou Misgrounded", **não** "fração das respostas dadas".
- Valores do painel esquerdo da Figura 4 (conferidos visualmente na imagem da página 14):

| Sistema | Accurate | Incomplete | Hallucination | Fontes dos números |
|---|---|---|---|---|
| Lexis+ AI | 65% | 18% | 17% | Fig. 4; §6.1 (65%, 18%); §1 (65%) |
| Westlaw AI-AR | 42% | 25% | 33% | Fig. 4; §6.1 e nota 18 (ver inconsistências abaixo) |
| Ask Practical Law AI | 20% | 63% | 17% | Fig. 4; §6.1 diz 19% e 62% |
| GPT-4 (sem RAG) | 49% | 8% | 43% | Fig. 4 apenas |

- **Em número de consultas (cálculo meu, aproximado):** 17% de 202 ≈ 34; 33% de 202 ≈ 67. O artigo **não** informa as contagens absolutas no texto que li (a Figura 4 só mostra percentuais).
- **Taxa condicionada a responder (Figura 4, painel direito):** "a porcentagem de respostas que são alucinadas quando há resposta direta". O painel é um gráfico de barras sem rótulos numéricos; pela altura das barras vejo **aproximadamente 0,20 (Lexis), 0,43 (Westlaw), 0,45 (Practical Law) e 0,47 (GPT-4)** — leitura visual, **a verificar**. **Cálculo meu:** esses valores batem com Hallucination ÷ (Hallucination + Accurate), isto é, o denominador "responsivo" **exclui toda resposta Incomplete, não só as recusas** (p. ex. Lexis: 17/(17+65) = 20,7%; Westlaw 33/75 = 44,0%; Practical Law 17/37 = 45,9%; GPT-4 43/92 = 46,7%). Se essa leitura estiver certa, "responsivo" inclui também a exclusão das respostas corretas porém sem citação; o artigo não explicita isso (a verificar na versão JELS).
- **Interpretação do artigo para o painel direito:** o Lexis+ AI tem taxa "estatisticamente significativamente menor" que Westlaw e Practical Law mesmo condicionada a responder (§6.1); e Westlaw e Practical Law, que respondem menos que o GPT-4, **não são significativamente mais confiáveis** nas respostas que dão (legenda da Fig. 4). Barras de erro = IC 95%.
- **Por que 17% do Practical Law não significa "mais confiável":** ele recusa/ficou incompleto em ~63% (a razão que o artigo dá: seu universo de documentos é só artigos de Practical Law, §6.1). Com poucas respostas, o 17% sobre o total esconde 17/37 ≈ 46% sobre as respondidas (cálculo meu, mesma leitura acima).
- **Confundidor de tamanho (§6.1):** excluindo recusas, o Westlaw escreve em média 350 palavras (DP 120), contra 219 (DP 114) do Lexis+ AI e 175 (DP 67) do Practical Law. O artigo argumenta que respostas mais longas têm "mais proposições falsificáveis" e portanto mais chance de conter ao menos uma alucinação. **(b) Interpretação:** combinado com a regra "qualquer afirmação falsa, material ou não" (C.2), a taxa de alucinação depende também do **comprimento** da resposta, não só da qualidade do sistema.
- **Por categoria de consulta (§6.1, Figura 5):** a alucinação é "levemente maior" em jurisdição/tempo, mas "continua alta" em pesquisa geral (p. ex. bar exam); a acurácia é maior na categoria de premissa falsa e menor nas categorias de uso real. Valores da Figura 5 **ilegíveis na extração: a verificar**.
- **Causas (Tabela 6, proporção de cada causa entre as respostas alucinadas de cada sistema; não somam 1):**

| Causa | Lexis | Westlaw | Pract. Law |
|---|---|---|---|
| Naive retrieval | 0,47 | 0,20 | 0,34 |
| Inapplicable authority | 0,38 | 0,23 | 0,34 |
| Reasoning error | 0,28 | 0,61 | 0,49 |
| Sycophancy | 0,06 | 0,00 | 0,03 |

  (Conferido contra o texto extraído.) **GPT-4 não aparece na Tabela 6.** Sicofância é rara porque os sistemas, em geral, corrigiram a premissa falsa (§6.3).
- **Achados qualitativos (§6.2, Tabelas 3–5):** exemplos de alucinação por sistema (10 no Westlaw, 6 no Lexis, 4 no Practical Law; nota 20: proporcional à taxa). Modos: entender o *holding* ao contrário; confundir litigante com tribunal; não respeitar a hierarquia de autoridade; e fabricar um parágrafo de lei inexistente. Há um achado de design: o Westlaw às vezes **afirmava uma proposição baseada em caso revertido sem citá-lo**, o que os autores suspeitam ser supressão de citações com "red flag" do KeyCite e que "impede a verificação das afirmações mais prováveis de serem falsas" (§6.2).
- **Exemplo da própria definição (§6.2/Fig. 2):** o Lexis+ AI cita um caso real (Reynolds, ainda válido) para descrever Casey, que foi revertido (Dobbs 2022). A citação é real, a resposta é incorreta, e a citação "serve só para enganar o usuário sobre a confiabilidade" (§4.3). Isso é o que **uma checagem "essa citação existe?" não pega**.

**Inconsistências internas do preprint v1 (verificadas contra o texto):**
1. **Westlaw "accurate" 42% × 41%:** o §1 diz 42%, a Fig. 4 mostra 42%, o §6.1 diz "41% e 19%".
2. **Practical Law "accurate" 20% × 19%:** a Fig. 4 mostra 20%; o §6.1 diz 19%. A nota 18 explica que o valor da *Figura 1* subiu de 19% para 20% após a revalidação; portanto o 19% do texto do §6.1 parece resíduo da versão anterior, e a Figura 1 e a Figura 4 já estariam em 20%. (Não conferi a Figura 1; a Figura 4 sim.)
3. **Practical Law "incomplete" 62% × 63%:** o §6.1 diz 62% (o §1 diz "mais de 60%"); a Fig. 4 mostra 63%. Com 17+63+20 = 100, a Figura 4 fecha; com 17+62+19, não.
4. **"Fabricated" (rótulo × uso):** o Apêndice C.3 define *Fabricated* como "cita uma fonte que não existe", mas (i) o rótulo não aparece na Tabela 1 nem no §4, e (ii) o §4.3 define alucinação só como "incorreta ou misgrounded", sem dizer onde o *Fabricated* entra (presumivelmente em "incorreta" ou "misgrounded"; o artigo não diz). Além disso, a subseção "Fabrications" do §6.2 usa o termo para **conteúdo inventado a partir de documentos reais** (p. ex. o parágrafo inexistente das FRBP, cuja origem os autores atribuem a um caso de falência de 1996 recuperado), e não para fonte inexistente. O artigo não reporta **nenhuma contagem de respostas Fabricated**.
5. **"Incomplete" tem duas definições:** legenda da Fig. 1 ("falham em endereçar a pergunta **ou** em fornecer citações adequadas para afirmações factuais") × §4.4 ("recusas ou ungrounded"). Em essência dizem o mesmo, mas a legenda não menciona *Refusal*, e Refusal inclui "resposta irrelevante" (Tabela 1).
6. **Tabela 1 × Apêndice C.1 sobre groundedness:** a legenda da Tabela 1 diz "groundedness só se aplica a respostas corretas"; o C.1 b diz que Irrelevant/Stock Refusal/Incorrect têm groundedness N/A. Consistente, mas significa que **uma resposta Incorrect nunca é classificada como Misgrounded/Ungrounded**: a granularidade de fundamentação só existe para respostas Correct.
7. **Premissa falsa × Refusal:** a nota 8 faz da "recusa que menciona não ter achado fontes" uma resposta Correct **só** para premissa falsa; na Tabela 2 há 22 consultas desse tipo, mas o artigo não informa quantas respostas foram recodificadas por essa regra.
8. **"Over 1 in 6" × 17%:** consistente (17% ≈ 1/6); sem problema.

**(b) Minha interpretação do peso da evidência:** a evidência é **forte para o que mede** (rotulagem especializada, pré-registro, κ substancial, três produtos mais um baseline) e **fraca para generalização**: n = 202 em um dataset adversarial por desenho, três produtos, um instante no tempo (os autores dizem que o Lexis+ AI mudou durante o estudo, §7), sem IC do κ.

## 5. Limitações (as do autor e as que eu identifico)

**Do autor (§7):**
1. Só três produtos (outros, como Harvey, não avaliados por falta de acesso).
2. **Ponto no tempo**: as respostas evoluíram durante o estudo, sobretudo no Lexis+ AI; vazamento de teste (o dataset foi enviado aos provedores) pode mascarar problemas gerais; os resultados "não falam" da versão de "segunda geração" do Lexis (nota 13).
3. Avaliação limitada ao chat; tarefas de geração mais especificadas (memorandos, contratos) ficam de fora.
4. **n = 202 é pequeno** (comparado a Dahl et al., 2024), por causa do acesso restrito e do trabalho manual.
5. **A groundedness "pode existir num espectro"**: uma citação a um caso revertido ainda pode ajudar o advogado a começar; foi codificada como misgrounded, mas a utilidade depende do caso de uso.
6. **O dataset não representa a distribuição natural de consultas**: "Nossa estimativa da taxa de alucinação não pretende ser uma estimativa não enviesada da taxa populacional" (§7), e sim testar se o RAG de fato resolveu a alucinação, como alegado.
7. Mede só alucinação, acurácia e groundedness, não "valor" total; as ferramentas podem ser úteis para começar a pesquisa.
8. Incerteza sobre determinismo (nota 26): a temperatura/decodificação pode não ser determinística, e parâmetros de recuperação são desconhecidos.

**Minhas (marcadas como interpretação):**
- **κ = 0,77 vem de n = 48, comparando só o 4º rotulador com os rótulos iniciais** (um de três). Não há IC, nem κ por classe, nem concordância entre os três rotuladores iniciais; com 48 itens a incerteza do κ é grande.
- **A definição de "alucinação" é estrita e assimétrica:** "qualquer afirmação falsa, material ou não" (C.2) amplia a taxa de alucinação; "parcialmente correto conta como Correct" (§4.1) a reduz. O efeito líquido não é medido.
- **Os 17–33% misturam produtos com taxas de recusa muito diferentes** (Practical Law: ~63% incomplete). Comparar "taxa de alucinação" entre sistemas sem fixar a taxa de resposta engana; o artigo mostra o painel condicionado para mitigar, mas o abstract usa a faixa sobre o total.
- **A faixa "17–33%" do abstract não inclui o GPT-4 (43%)** nem os 58–82% de Dahl et al. 2024 para LLMs gerais, que o artigo cita como referência (§2.2).
- **A atribuição de causas (Tabela 6) não é reproduzível a partir do texto**: sem protocolo, sem n, sem κ da atribuição.
- **GPT-4 como baseline não é comparável em condições**: roda via API com um prompt de sistema que o autor escolheu e sem RAG; o Westlaw AI-AR usa GPT-4 por baixo (§5.1, "parece"), mas não se sabe com quais instruções.
- A conclusão "RAG reduz alucinação" (§6.1, §9) é uma comparação de **produtos** contra um modelo sem RAG, não um experimento controlado do efeito do RAG.
- **O artigo não mede *recall* de citação** (quantas afirmações materiais ficaram sem fonte) como taxa separada; o *ungrounded* aparece só dentro de "Incomplete" e é somado a Refusal.

## 6. Reflexões ancoradas no NOSSO projeto

Referência de pontos: P1–P11, D-1, D-2, L-3 em `docs/roadmap/ckpt-0.6.1b-metrics-roadmap.md`; juízes planejados em `docs/roadmap/ckpt-0.6-plan.md` §3.2 (`faithfulness_judge`, `citation_accuracy`, `abstention_quality`, entre outros) e `docs/plan.md` (tabela de métricas).

**R1 — O `citation_accuracy` planejado cobre só *misgrounded*; faltam *ungrounded* (e, em parte, *fabricated*). (APOIA a ideia de métrica própria; DESAFIA o escopo.)**
O plano 0.6 §3.2 descreve: regex extrai `[2609.xxxxvN]` da resposta (code); para cada citação, o juiz checa se "o chunk daquele `arxiv_id` no contexto recuperado sustenta a frase citada". `docs/plan.md` diz "cited doc exists in the corpus AND supports the claim". Mapeando para as categorias do Magesh (minha tradução):
| Categoria Magesh | Equivalente no nosso projeto | Coberto hoje no plano? |
|---|---|---|
| Grounded | `[id]` existe, o chunk do `id` sustenta a frase | sim (passo 2, juiz) |
| **Misgrounded** | `[id]` válido, mas o chunk não sustenta a frase | sim (passo 2, juiz) |
| **Ungrounded** | frase factual **sem** `[id]` | **não** (o regex só vê citações presentes; frase sem citação nunca vira item) |
| **Fabricated** | `[id]` que **não existe no corpus** | **parcial**: `plan.md` menciona "exists in the corpus", mas o plano 0.6 §3.2 só fala "no contexto recuperado", e nenhum dos dois define uma saída separada para isso |

Observações: (i) o Magesh trata *ungrounded* como **incompleta, não alucinação**, porque a frase não é falsa; para nós, é uma falha da promessa do produto ("Cited Q&A", `plan.md`) e **precisa de uma saída própria**, por exemplo uma taxa de "afirmações factuais com citação" (um *recall* de citação); (ii) minha interpretação: o caso "`[id]` existe no corpus mas **não está nos chunks recuperados**" é uma terceira situação que o Magesh não separa (ele define *Fabricated* só como "fonte inexistente"); no nosso sistema ela indica que o modelo citou de memória, e mereceria um rótulo próprio (p. ex. `phantom`/`not-retrieved`); (iii) o `[id]` no nosso sistema é um ID de **paper**, e o retriever devolve **chunks** (D-1): "o chunk daquele `arxiv_id`" pode ser mais de um (2–5 por paper), então a premissa do juiz deve ser a concatenação dos chunks recuperados daquele paper, não um chunk único. **Afeta:** `citation_accuracy` (a implementar no 0.6.3), `docs/plan.md` §3 e `ckpt-0.6-plan.md` §3.2. **O que adotaríamos:** saída com 4 estados por citação/afirmação (`supported` / `unsupported-but-cited` = misgrounded / `uncited` = ungrounded / `phantom` = fabricated ou fora do contexto) e relatar as taxas separadamente; a regra de precedência do Magesh (misgrounded > ungrounded quando coexistem, C.3) para agregar por resposta.

**R2 — Nossa `faithfulness_judge` está no sentido "CS" que o Magesh distingue do seu; ela não captura a "aplicabilidade" da fonte. (NEUTRO quanto à utilidade; DESAFIA a leitura de que o Magesh sustenta a `faithfulness`.)**
O Magesh diz explicitamente (§4.2) que seu "grounded" **não** é o sentido de aderência ao documento fornecido; inclui relevância, jurisdição e autoridade vigente. A nossa `faithfulness_judge` (estilo RAGAS, ref-free, contra `retrieved_context`) é o sentido CS. Para o domínio de papers de arXiv, a analogia mais próxima de "autoridade inaplicável" é o slice **`stale`** (conteúdo desatualizado, `abstention_quality` pré-refresh) e a atribuição ao paper certo. **Afeta:** `faithfulness_judge` e a interpretação da `citation_accuracy`: o Magesh sustenta **separar correção de fundamentação** (e separar citação de fidelidade), não a faithfulness em si (o artigo só cita Agrawal et al. 2023 para o sentido CS de "groundedness", §4.2). **Recomendação:** não atribuir à `faithfulness_judge` a autoridade do Magesh; citá-lo para a decomposição correção × fundamentação.

**R3 — Separar "recusa informativa" de "recusa padrão" (P7/P8, `abstention_quality`). (APOIA com cautela.)**
O Magesh codifica **diferente** duas recusas **só para premissa falsa**: "I cannot find any information on this topic" (diz que não achou fonte) = **Correct**; "I cannot provide you with any information on this topic" (recusa padrão) = **Refusal** (nota 8; C.2). Para todas as outras categorias, qualquer recusa é Refusal (→ *Incomplete*). Isso é uma **regra de rotulagem dependente da categoria**, não uma taxonomia geral de recusas. **Afeta:** P7 (`abstained(answer)`, detecção por `startswith` frágil) e P8 (`abstention_summary`: over/under-refusal), além do juiz `abstention_quality`. Nossos slices `unanswerable` (15) e `stale` (10) têm `should_abstain=True`, onde a recusa **é** a resposta certa; já nas respondíveis, qualquer recusa é over-refusal. Duas lições transferíveis (interpretação minha): (i) o juiz de abstenção precisa distinguir "recusa **com indicação do motivo** (não achei nos papers)" de "recusa genérica", e isso pode virar um subcampo (`informative_refusal`) para análise, **sem** alterar o cálculo de P/R/F1 da P8; (ii) tratar a premissa falsa como categoria própria pode ser útil, mas **não temos slice de premissa falsa** (o mais próximo é `unanswerable`); o Magesh mostra que o comportamento desejado é *refutar*, não apenas recusar, o que o nosso `abstention_quality` binário não captura. **O que adotaríamos:** o campo `informative_refusal` como diagnóstico. **Não adotaríamos:** a regra de que recusa informativa conta como "correct" fora dos slices `should_abstain`, pois no nosso caso isso mascararia over-refusal.

**R4 — κ = 0,77 em 48 itens como referência de aceite da calibração juiz×humano (P11). (APOIA como referência, DESAFIA como alvo direto.)**
O roadmap P11 propõe "κ juiz×humano por juiz nas 100 labels; critério de aceite documentado (ex.: κ ≥ 0,6) antes de congelar o baseline". O Magesh dá um número de comparação **humano×humano** de 0,77 (85,4% de concordância), com protocolo: guia escrito, 4º rotulador cego, amostra estratificada. Cautelas (interpretação minha): (i) **0,77 é humano×humano em 3 classes numa tarefa jurídica de especialistas**, não um limiar para juiz-LLM; em geral um juiz não deve ser exigido a passar do teto humano; (ii) o κ **depende da prevalência** e do número de classes, então κ de um juiz binário por critério (fiel/infiel, citação válida/inválida, abstenção correta/incorreta) não é diretamente comparável ao κ de 3 classes do Magesh; (iii) **n = 48 sem IC**: com poucos itens, o κ de cada juiz do nosso projeto, calculado sobre subconjuntos pequenos (slices de 10–15), terá incerteza grande, então relatar IC (bootstrap) ou, no mínimo, o n; (iv) **ponto a verificar no nosso projeto:** as "100 labels" mencionadas no plano são, até onde li, a revisão humana do **golden set** (pergunta/gold/proveniência), não rótulos humanos de **respostas geradas** pelo sistema; o κ juiz×humano exige rótulos humanos das saídas (como o Magesh rotulou as respostas das ferramentas). **O que adotaríamos:** (1) o protocolo (guia escrito de rubrica + um rotulador independente e cego, com amostra estratificada por slice) como padrão para as ~100 saídas rotuladas; (2) 0,77 como **referência de teto** a citar, mantendo 0,6 como piso de aceite do roadmap (decisão do usuário); (3) o pré-registro como inspiração: congelar o golden set e as rubricas antes do EXP-0. **Não adotaríamos:** κ = 0,77 como meta numérica do juiz.

**R5 — "Rotular com especialista humano, não com LLM" (§5.4) × nosso uso de juízes LLM. (DESAFIA o escopo, não a decisão.)**
O Magesh rejeita auto-verificação por LLM para a sua tarefa. O nosso plano usa juízes LLM (`gpt-4o-mini`, mesmo modelo do gerador, P11) e calibra com humano. A ressalva do artigo vale como motivo para **não congelar o baseline antes de medir κ por juiz** e para o argumento de P11 (separar `JUDGE_MODEL` de `GENERATION_MODEL`). Mas o argumento do Magesh é específico ("adherence to authority"), e o nosso domínio (abstracts de arXiv, claims curtas) é bem menos exigente do que direito (minha interpretação), então não extrapola para "juiz LLM é inválido" no nosso caso. **Afeta:** P11, `config.py` (`JUDGE_MODEL`), calibração 0.8.

**R6 — O "17–33%" NÃO é transferível ao nosso domínio. (DESAFIA qualquer uso do número como expectativa ou baseline.)**
Razões, separando o que o artigo diz e o que é minha interpretação:
- *(a) O artigo diz:* o dataset é adversarial por desenho (bar exam, circuit splits, casos revertidos) e a estimativa "não pretende ser uma estimativa não enviesada da taxa populacional" (§7); mede três **produtos proprietários** com recuperação e parâmetros desconhecidos (nota 26); em um instante (março–maio de 2024); n = 202.
- *(b) Minha interpretação:*
  1. **Definição de alucinação**: "incorreta ou misgrounded", com "qualquer afirmação falsa" (C.2). O nosso `faithfulness_judge` mede aderência ao contexto, e o `citation_accuracy` mede suporte da citação; não medem "verdade no mundo". São construtos diferentes.
  2. **Denominador inclui recusas**: 17% sobre as 202 respostas depende da taxa de resposta (Practical Law 17% com ~63% incompleta). Qualquer taxa nossa precisa declarar o denominador (total × respondidas).
  3. **Domínio**: o direito tem hierarquia de autoridade, jurisdição e hoje-vs-ontem; nosso corpus é de **250 abstracts de arXiv** (843 chunks de ~420 caracteres; `roadmap` §3). Misgrounding por "jurisdição errada" não existe; respostas curtas têm menos proposições falsificáveis (o artigo mostra que o comprimento importa, §6.1).
  4. **Nosso gerador** é `gpt-4o-mini` com top-5 de chunks, num corpus fechado, não os sistemas do artigo; e o baseline do artigo (GPT-4 sem RAG) não existe no nosso projeto.
  5. **Rotulagem**: o artigo usa especialistas humanos; nós usaremos juiz LLM calibrado.
  Uso aceitável do número: **motivação** ("RAG com citações ainda alucina, mesmo comercialmente") e contraexemplo contra alegações de "zero alucinação", **nunca como meta ou baseline**.

**R7 — Tratamento do número em documentos do projeto: correções a considerar. (NEUTRO; ação sobre texto.)**
(i) `docs/research/rag_failure_modes_review.md` (linha 62) escreve "**17–34%** of queries"; o artigo (abstract, v1) diz **17% a 33%**. (ii) A mesma linha diz que as ferramentas alucinaram "**despite retrieving real sources**"; o artigo atribui parte das alucinações a **recuperação ingênua** (Tabela 6: 47% das alucinações do Lexis, 20% do Westlaw, 34% do Practical Law) e a **autoridade inaplicável**, não só a falha na geração com a fonte certa. A frase "despite correct retrieval" no rótulo em negrito do trecho está, portanto, **forçada**: os "17–33%" não são "apesar da recuperação correta". (iii) O roadmap (§ Fundamentação, item 5) já usa 17%–33% corretamente, mas cita só o resumo; vale referenciar o denominador (total de 202, com recusas). Eu **não editei** esses arquivos (fora do escopo desta tarefa).

**R8 — Slices de 10–15 itens e intervalos de confiança. (APOIA a prática de relatar IC.)**
O Magesh mostra IC 95% nas Figuras 4–5 mesmo com n = 202 e categorias de 22–80 consultas, e as diferenças entre produtos só são "significativas" onde as barras não se sobrepõem (§6.1). Nossos slices (`unanswerable` 15, `stale` 10, `multi-doc` 15, `format` 10, `persona` 10, `answerable` 40) são menores. Interpretação minha: qualquer comparação entre variantes por slice deve vir com IC ou n explícito. **Afeta:** relatório EXP-0, P8 (por slice).

**R9 — Itens neutros no roadmap.** D-1 (gold por chunk ou paper): o artigo não trata granularidade de gold de recuperação (julga por documento citado, sem gold de chunk). D-2 (hit@k ≡ recall@k): sem relação. L-3 (`ranx` como oráculo): sem relação (artigo não usa métricas de IR). P9 (token-F1): o artigo avalia correção por rubrica humana, não por sobreposição de tokens; neutro. **NEUTRO.**

**O que adotaríamos:** (1) a decomposição **correção × fundamentação** e as definições grounded / misgrounded / ungrounded / fabricated como vocabulário do `citation_accuracy` (com os 4 estados da R1); (2) o protocolo de rotulagem (rubrica escrita, rotulador cego, amostra estratificada) para as saídas que calibrarão os juízes (P11); (3) o pré-registro/congelamento do golden set e da rubrica antes do EXP-0; (4) separar `Accurate / Incomplete / Hallucination` como partição exclusiva, relatando taxa sobre o total **e** sobre as respondidas; (5) IC por slice.
**O que NÃO adotaríamos e por quê:** (1) o 17–33% como baseline ou meta (R6); (2) a regra "qualquer afirmação falsa conta, material ou não" sem adaptação (o nosso gerador produz respostas curtas e um limiar de materialidade deve ser decidido); (3) classificar *ungrounded* como mera "incompleta" sem uma taxa própria (R1); (4) a regra de "recusa informativa = correta" fora de `should_abstain` (R3); (5) κ = 0,77 como alvo numérico para um juiz (R4); (6) a leitura "RAG reduz alucinação" como efeito causal, pois o artigo compara produtos, não isola o RAG.

## 7. Citações úteis (trechos literais curtos)

1. "A response is considered hallucinated if it is either incorrect or misgrounded." — §4.3.
2. "an ungrounded response does not actually make any false assertions." — §4.4.
3. "Our estimate of the hallucination rate is not meant to be an unbiased estimate of the (unknown) population-level rate of hallucinations in legal AI queries" — §7 (Limitations, sexta limitação).
