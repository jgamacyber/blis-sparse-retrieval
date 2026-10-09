# Fusão recíproca de postos

O artigo "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods", de Gordon Cormack, Charles Clarke e Stefan Buettcher, foi publicado no SIGIR 2009.

## O problema

Dois sistemas de busca produzem listas ordenadas para a mesma consulta, e você quer uma lista só. O caminho óbvio é somar os scores, e ele não funciona: os scores não estão na mesma escala. Um BM25 devolve algo entre zero e trinta, um cosseno devolve algo entre menos um e um, e um codificador cruzado devolve logits sem limite. Somar isso é somar unidades diferentes.

## A fórmula

A fusão recíproca de postos descarta o score e usa apenas a **posição**:

```
RRF(d) = Σ  1 / (k + rank_s(d))
         s
```

onde a soma é sobre os sistemas, `rank_s(d)` é a posição do documento no sistema `s`, e `k` é uma constante. O artigo usa `k = 60`.

O `k` controla quanto as primeiras posições dominam. Com `k` pequeno, a primeira posição vale muito mais que a terceira. Com `k` grande, as posições se achatam e a fusão fica mais parecida com uma votação.

## A vantagem e o preço

A vantagem é que não há nada para calibrar. Qualquer par de sistemas pode ser fundido no dia em que você os tem, sem normalizar escala, sem conjunto de validação.

O preço é que **a magnitude do score é jogada fora**. Se um sistema está muito confiante no primeiro lugar e pouco confiante do terceiro para baixo, a fusão recíproca não tem como saber: para ela, a diferença entre o primeiro e o terceiro é sempre a mesma, qualquer que seja o score.

É por isso que a combinação convexa, que normaliza os scores e os soma com pesos, às vezes ganha. O benchmark de documentos com tabelas mediu exatamente isso e encontrou a combinação convexa com pesos iguais acima da fusão recíproca com `k = 60`.

E o `k = 60` não é constante da natureza: é o valor de um artigo de 2009, em um conjunto de dados daquela época. Vale medir o seu.
