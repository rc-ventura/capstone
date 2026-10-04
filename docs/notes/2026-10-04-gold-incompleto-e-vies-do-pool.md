# Nota do autor

## 04/10/2026

**Tipo:** decisão

**Registro objetivo:** Aprovei a opção B, pelo que entendi do problema: nosso gold foi criado de forma sintética por chunk, então a pergunta foi criada em cima do chunk. Hoje o chunk significa o abstract completo, e não 3,4 chunks por paper. Solucionamos o problema dos chunks irmãos, mas sobrou outro problema: um possível gold incompleto e único. Alguns dos fatos que o agente mencionou (A, B, C) explicam algumas observações. Trabalhos citados (lidos por agente, não por mim): Thakur et al. 2021, BEIR (NeurIPS Datasets & Benchmarks, arXiv:2104.08663), sobre Hole@k e BM25 em paralelo; Buckley & Voorhees 2004, Retrieval Evaluation with Incomplete Information (SIGIR '04); Buckley, Dimmick, Soboroff & Voorhees 2007, Bias and the Limits of Pooling for Large Collections (relatório NIST); Büttcher, Clarke, Yeung & Soboroff 2007, Reliable Information Retrieval Evaluation with Incomplete and Biased Judgements (SIGIR '07).

**Reflexão:** O fato B traz a preocupação do viés do retriever: por mais confiável que nosso gold esteja, depois do mini-pooling ele ainda será um julgamento em cima do baseline. Será que não existe outro paper que responda? Não sabemos, porque um possível paper X nunca apareceu no baseline. Já documentamos algumas soluções: rodar o BM25 (lexical) em paralelo, ou montar o pool com mais de um retriever, o que já está documentado em trabalhos. O outro caminho é adotar a métrica Hole@k (já existe no BEIR), que mede quantos papers do top 5 de um retriever ninguém julgou; assim podemos saber a grandeza da incerteza do score (o viés em si só aparece ao julgar esses papers) e reaproveitar esses papers não julgados.

**Tags:** gold-incompleto, vies-do-retriever, bm25, hole-at-k
