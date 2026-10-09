# Como medir recuperação

## As quatro métricas usuais

**Hit@k** pergunta se pelo menos um documento relevante apareceu nas `k` primeiras posições. É a métrica mais fácil de explicar e a mais grosseira: ela não distingue o sistema que coloca o relevante em primeiro do que o coloca em décimo.

**Recall@k** é a fração dos documentos relevantes que apareceu nas `k` primeiras posições. É a métrica certa para avaliar o **primeiro estágio** de um pipeline de dois estágios, porque o que importa ali é não perder o relevante, e não ordená-lo bem: a ordenação é trabalho do reordenador.

**MRR**, posto recíproco médio, é a média de `1 / posição do primeiro relevante`. Ela se importa muito com o topo: sair da segunda para a primeira posição vale 0,5, e sair da décima para a nona vale 0,011.

**nDCG@k**, ganho cumulativo descontado e normalizado, acumula o ganho de cada relevante com desconto logarítmico pela posição, e divide pelo valor do ordenamento ideal. É a mais completa das quatro, porque usa todas as posições e aceita graus de relevância, e a menos intuitiva de ler.

## A escolha depende do estágio

Esta é a parte que se erra com frequência.

Para o **primeiro estágio**, meça Recall@k com `k` igual ao tamanho do conjunto que você vai entregar ao reordenador. O documento que não estiver nesse conjunto está perdido, e nenhuma qualidade de reordenação o recupera. Otimizar MRR no primeiro estágio é otimizar a coisa errada.

Para o **estágio final**, meça nDCG ou MRR, porque aí a posição é o produto entregue.

## O risco de medir só o agregado

Uma média sobre o conjunto de consultas esconde comportamento oposto entre tipos de consulta. Um recuperador pode ganhar 20% nas consultas de paráfrase e perder 15% nas de termo exato, e a média mostrar um ganho modesto de 5%, que não descreve nenhum dos dois casos.

Por isso vale anotar o **tipo** de cada consulta no conjunto de avaliação e reportar por tipo. É mais trabalho uma vez, e passa a responder a pergunta que interessa: não "qual recuperador é melhor", mas "melhor para qual consulta".
