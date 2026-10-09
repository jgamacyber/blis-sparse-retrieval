# DPR: recuperação densa por codificador duplo

O artigo "Dense Passage Retrieval for Open-Domain Question Answering", de Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen e Wen-tau Yih, foi publicado no EMNLP 2020.

## A proposta

Dois codificadores BERT, um para a pergunta e um para a passagem, treinados para que o produto interno entre os dois vetores seja alto quando a passagem responde à pergunta.

```
sim(q, p) = E_q(q)ᵀ · E_p(p)
```

Cada passagem do corpus é codificada uma vez, offline, e indexada em uma estrutura de busca de vizinhos próximos. Na consulta, só a pergunta é codificada.

## O treino com negativos no lote

A parte que mais importa na prática é como vêm os exemplos negativos. O artigo usa negativos **dentro do lote**: para cada pergunta, as passagens positivas das outras perguntas do mesmo lote servem de negativas. Isso dá muitos negativos de graça, porque as representações já foram calculadas para o lote.

Os autores também mostram o ganho de incluir negativos difíceis, isto é, passagens que o BM25 traz no topo mas que não contêm a resposta. Negativo fácil ensina pouco: o modelo precisa de exemplos que ele erraria.

## O resultado e o que veio depois

O DPR superou o BM25 por margem larga nos conjuntos de perguntas e respostas de domínio aberto em que foi treinado, e foi um dos trabalhos que estabeleceu a recuperação densa como padrão.

O que veio depois complicou a conclusão. Avaliações fora do domínio de treino, em especial o BEIR, mostraram que esse ganho não transfere de graça: em vários corpora novos, modelos densos treinados em um domínio ficam abaixo do BM25.

A lição não é que recuperação densa não funciona. É que o ganho dela é medido dentro de uma distribuição, e que generalização para corpus novo é uma propriedade separada, que precisa ser medida separadamente.
