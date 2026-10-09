# BEIR: avaliação fora do domínio de treino

O artigo "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models", de Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava e Iryna Gurevych, foi publicado no NeurIPS 2021, na trilha de conjuntos de dados e benchmarks.

## Por que ele existe

Modelos de recuperação eram comparados quase sempre no MS MARCO, que é grande, em inglês e de um domínio só. Um modelo treinado ali e avaliado ali mede uma coisa; o que a maioria das pessoas precisa saber é outra: como esse modelo se comporta no **meu** corpus, sobre o qual ele não foi treinado.

O BEIR reúne cerca de dezoito conjuntos de domínios e formatos diferentes, de artigos científicos a verificação de fatos e perguntas de fórum, e avalia tudo em regime de tiro zero, sem nenhum treino no conjunto alvo.

## O achado incômodo

O BM25 se mostrou uma linha de base difícil. Vários modelos densos que batiam o BM25 com folga no MS MARCO ficaram **abaixo** dele em boa parte dos conjuntos do BEIR.

A explicação é que recuperação densa aprende uma noção de similaridade a partir da distribuição de treino. Em corpus com vocabulário, tamanho de documento e tipo de consulta diferentes, essa noção deixa de valer. O BM25, por não aprender nada, não tem nada para desaprender.

## O que foi melhor

As configurações mais fortes no agregado foram as de dois estágios, com recuperação seguida de reordenação por codificador cruzado. O artigo registra, junto, o custo: reordenar é ordens de magnitude mais caro por consulta que buscar.

## A lição de método

Daqui sai uma regra prática que vale para este módulo: **nenhum número de recuperação significa algo sem dizer em que corpus e em que tipo de consulta foi medido**. Uma troca de recuperador que ganha 10% em um corpus pode perder no seguinte, e as duas medidas estão certas.
