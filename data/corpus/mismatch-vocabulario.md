# Incompatibilidade de vocabulário

## O problema

Quem escreve o documento e quem escreve a consulta escolhem palavras diferentes para a mesma coisa. O documento diz "insuficiência cardíaca", a consulta diz "falha do coração". O documento diz "rescisão unilateral", a consulta diz "cancelar contrato sem acordo".

Para um modelo que casa termos literalmente, esses pares não têm nada em comum. O documento certo existe no corpus, está indexado, e não é recuperado.

O inverso também acontece, e é mais difícil de notar: a mesma palavra com sentidos diferentes. "Prazo" em um contrato e "prazo" em um relatório de entrega não são a mesma coisa, e o casamento literal não distingue.

## As soluções clássicas

Antes dos modelos neurais, a expansão da consulta era o caminho.

**Realimentação de relevância** parte dos documentos que o usuário marcou como relevantes e acrescenta à consulta os termos frequentes neles. A versão sem usuário, a realimentação cega, assume que os primeiros resultados são relevantes e expande a partir deles, com o risco óbvio de reforçar um erro inicial.

**Tesauro** mapeia sinônimos à mão. Funciona bem em domínio fechado e com manutenção, e não escala para a língua inteira.

**Análise de contexto local** expande com termos que coocorrem com os da consulta nos documentos melhor ranqueados, que é uma forma de tesauro automático e dependente do corpus.

Todas funcionam em parte e todas são heurísticas, no sentido de que ninguém aprendeu os pesos: eles foram escolhidos.

## Os dois caminhos modernos

**Representação densa** abandona o casamento de termos. Projeta texto em um espaço contínuo onde proximidade significa similaridade semântica, e aí "falha do coração" e "insuficiência cardíaca" caem perto. O preço é perder o casamento exato: termo raro e preciso, como um código ou um nome próprio, se dilui no vetor.

**Esparsidade aprendida** mantém o casamento de termos e aprende a expansão. O documento continua representado por pesos sobre o vocabulário, só que agora alguns termos que ele não contém literalmente têm peso diferente de zero, porque o modelo aprendeu que eles pertencem àquele contexto.

A segunda é a aposta do SPLADE, e a razão de ela ser interessante é que ela não troca um defeito por outro: o termo literal continua lá, com peso alto.
