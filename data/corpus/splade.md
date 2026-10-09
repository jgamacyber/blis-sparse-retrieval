# SPLADE: esparsidade aprendida sobre o vocabulário

O artigo "SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval", de Thibault Formal, Carlos Lassance, Benjamin Piwowarski e Stéphane Clinchant, da Naver Labs Europe, foi publicado em 2021.

## A ideia

Em vez de comprimir o texto em um vetor denso de algumas centenas de dimensões, o SPLADE produz um vetor **esparso** com uma dimensão por token do vocabulário do BERT, isto é, 30522 posições, das quais apenas algumas centenas ficam diferentes de zero.

A chave é de onde vêm esses pesos. O SPLADE usa a cabeça de modelagem de linguagem mascarada do BERT, a mesma que foi treinada para prever token faltante. Para cada token de entrada `i` e cada token `j` do vocabulário:

```
w_ij = transform(h_i)ᵀ E_j + b_j
```

onde `h_i` é a representação contextual do token de entrada, `E_j` é o embedding de entrada do token `j` do vocabulário, `b_j` é um viés por token, e `transform` é uma camada linear com GeLU e normalização de camada.

Isso dá uma matriz de logits sobre o vocabulário inteiro. O vetor final sai de uma agregação sobre os tokens de entrada.

## A mudança do v2: max em vez de soma

A primeira versão somava sobre os tokens de entrada:

```
w_j = Σ  log(1 + ReLU(w_ij))
     i∈t
```

A versão 2 troca a soma por um máximo:

```
w_j = max  log(1 + ReLU(w_ij))
      i∈t
```

Os autores chamam o modelo com soma de SPLADE e o com máximo de SPLADE-max. A troca parece pequena e muda bastante o comportamento: com soma, um termo que aparece muitas vezes acumula peso, o que reintroduz exatamente o problema que o `k1` do BM25 existe para conter. Com máximo, o peso é o da ocorrência mais informativa, e a repetição não infla nada.

O `log(1 + ReLU(·))` tem duas funções. O ReLU zera as contribuições negativas, e é dele que vem a esparsidade. O logaritmo satura o crescimento, pelo mesmo motivo que o BM25 satura frequência.

## O score

O score é o produto interno entre os vetores esparsos da consulta e do documento. Como os dois são esparsos sobre o mesmo vocabulário, isso é exatamente o que um índice invertido sabe fazer rápido.

Existe uma variante, o SPLADE-doc, que não expande a consulta, apenas o documento. Nela o score fica:

```
s(q, d) = Σ  w_j^d
         j∈q
```

Essa variante é mais rápida na consulta, porque não precisa rodar o modelo sobre a consulta em tempo de busca, e perde algum ganho.

## O regularizador FLOPS

Esparsidade não aparece sozinha: é preciso pressionar por ela na função de perda. O SPLADE usa o regularizador FLOPS, que é uma relaxação suave do número médio de operações de ponto flutuante de uma busca no índice invertido:

```
ℓ_FLOPS = Σ  ā_j²,  com  ā_j = (1/N) Σ  w_j^(d_i)
         j∈V                          i=1..N
```

`ā_j` é a ativação média do token `j` em um lote de `N` documentos. Elevar ao quadrado e somar penaliza mais fortemente o token que fica ativo em muitos documentos, que é justamente o que faz uma lista de postagem crescer e a busca ficar lenta.

A perda total separa consulta e documento:

```
L = L_rank-IBN + λ_q · L_reg^q + λ_d · L_reg^d
```

Os dois coeficientes existem porque a pressão desejada é diferente nos dois lados. Os autores colocam mais pressão na consulta, porque consulta esparsa é o que mantém a busca rápida: cada termo ativo na consulta é uma lista de postagem a percorrer.

## Por que isso importa para quem vem do BM25

O SPLADE não troca o índice invertido por outra coisa. Ele continua produzindo pesos por termo do vocabulário, continua pontuando por produto interno e continua rodando em índice invertido.

O que ele acrescenta é a **expansão aprendida**: um documento que fala de "insuficiência cardíaca" ganha peso também em tokens como "coração" e "falha", porque a cabeça de linguagem mascarada prevê esses tokens naquele contexto. A incompatibilidade de vocabulário é atacada sem abandonar o casamento exato de termos, porque os termos literais continuam lá, com peso alto.

Em troca, o SPLADE paga o que todo modelo neural paga: precisa de treino, precisa de inferência por documento na indexação, e as listas de postagem ficam mais longas, porque cada documento agora ativa mais termos do que os que literalmente contém.

## Variantes publicadas

O repositório da Naver lista, com MRR@10 no conjunto de desenvolvimento do MS MARCO: `splade_v2_max` com 34,0, `splade_v2_distil` com 36,8, `splade-cocondenser-selfdistil` com 37,6, `splade-cocondenser-ensembledistil` com 38,3, e as variantes eficientes com até 38,8 usando modelos separados para consulta e documento.
