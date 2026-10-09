# ColBERT: interação tardia entre tokens

O artigo "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT", de Omar Khattab e Matei Zaharia, de Stanford, foi publicado no SIGIR 2020.

## O problema de arquitetura

Havia duas opções, e as duas eram ruins por motivos opostos.

O codificador cruzado coloca consulta e documento na mesma passada e vê tudo, mas não permite pré-calcular nada, então é inviável para varrer um corpus.

O codificador duplo pré-calcula tudo, mas comprime o documento inteiro em um único vetor antes de saber qual será a consulta. Informação que a consulta precisaria já foi descartada.

## A interação tardia

A proposta é adiar a interação, sem adiá-la até o fim. Cada texto é codificado separadamente, o que preserva o pré-cálculo, mas a representação guardada é **um vetor por token**, não um por texto.

O score soma, para cada token da consulta, o melhor casamento dele com algum token do documento:

```
S(q, d) = Σ      max      E_qi · E_djᵀ
         i∈q    j∈d
```

Essa operação é chamada de MaxSim. Vale notar a diferença em relação ao score multivetor do BGE-M3, que usa a **média** dos máximos em vez da soma: com soma, consulta mais longa produz score maior, o que é irrelevante para ordenar documentos de uma mesma consulta mas impede comparar scores entre consultas.

## O custo

O ganho de qualidade vem com uma conta de armazenamento que não é pequena: guardar um vetor por token custa, por documento, tantas vezes mais que a abordagem de vetor único quantos tokens o documento tiver.

O artigo trata isso com redução de dimensão por projeção linear e quantização dos vetores. Trabalhos posteriores da mesma linha reduziram bastante esse custo.

## Por que isso aparece aqui

A interação tardia é a terceira das três funções que o BGE-M3 reúne. Entender o MaxSim é o que torna aquela terceira parcela do score integrado algo mais que uma linha de fórmula.
