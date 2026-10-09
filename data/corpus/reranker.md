# Reordenação com codificador cruzado

## A diferença de arquitetura

Um recuperador trabalha com **codificação dupla**: consulta e documento passam pelo modelo separadamente, e o score sai de uma operação simples entre as duas representações, geralmente produto interno. Isso é o que permite pré-calcular a representação de todos os documentos e indexá-los: na hora da busca, só a consulta precisa ser processada.

Um reordenador com **codificador cruzado** faz o oposto. Consulta e documento entram juntos na mesma passada, e o modelo vê os dois ao mesmo tempo, com atenção cruzada entre todos os tokens dos dois textos. A saída é um único número de relevância.

A consequência é dura e inevitável: não há nada para pré-calcular. Cada par consulta e documento custa uma passada completa. Por isso codificador cruzado nunca é usado para varrer o corpus, e sim para reordenar um conjunto pequeno de candidatos que o recuperador já trouxe.

Essa é a arquitetura de dois estágios: um recuperador rápido e aproximado abre o funil, um reordenador caro e preciso fecha.

## O bge-reranker-v2-m3

O BAAI distribui o `bge-reranker-v2-m3` dentro do FlagEmbedding, construído sobre o BGE-M3, sob licença Apache 2.0. A ficha do modelo o descreve como leve, multilíngue, fácil de implantar e de inferência rápida, com cerca de 0,6 bilhão de parâmetros.

A interface recebe pares de consulta e passagem e devolve um score de relevância, em que maior é mais relevante. Os valores brutos são logits e não estão em escala limitada: a própria ficha do modelo mostra exemplos como -5,65 e 5,26. Com a normalização ligada, uma sigmoide mapeia o valor para o intervalo de zero a um, e aquele -5,65 vira aproximadamente 0,0035.

Isso tem uma consequência prática que passa batido com facilidade. O score bruto **não é probabilidade e não é comparável entre consultas diferentes**. Serve para ordenar candidatos de uma mesma consulta. Usar um limiar fixo sobre o valor bruto, do tipo descartar tudo abaixo de zero, é uma decisão que precisa ser calibrada no seu conjunto, não herdada de outro.

A família tem outras opções, com a troca usual. O `v2-m3` é a escolha de eficiência e multilinguismo. O `v2-gemma` vai melhor em inglês e em multilíngue, custando mais. O `v2-minicpm-layerwise` permite escolher a camada de saída, trocando qualidade por velocidade de forma controlada.

## Por que o reordenador costuma dar o maior ganho

Um recuperador precisa comparar a consulta com milhões de documentos, então sua representação é obrigada a ser barata e independente. Um reordenador compara a consulta com cinquenta documentos, e por isso pode olhar os dois textos juntos.

O resultado é que o reordenador enxerga coisas que nenhum produto interno enxerga: negação, condição, qual número se refere a qual entidade, se a passagem responde à pergunta ou apenas fala do mesmo assunto.

É por isso que, em comparações de ponta a ponta, acrescentar reordenação a um recuperador mediano costuma render mais que trocar o recuperador por um melhor.

## O tamanho do conjunto de candidatos

O reordenador só pode reordenar o que recebeu. Se o documento certo não estiver entre os candidatos, nenhuma qualidade de reordenação o traz de volta.

Daí a variável mais importante e mais esquecida da arquitetura de dois estágios: quantos candidatos o primeiro estágio entrega. Conjunto pequeno demais limita o teto do sistema ao recall do recuperador naquele corte. Conjunto grande demais multiplica o custo linearmente, porque cada candidato é uma passada do codificador cruzado.

Essa é uma escolha a ser medida, não escolhida por hábito, e é um dos experimentos deste módulo.
