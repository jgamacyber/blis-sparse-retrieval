# Expansão de documento por geração de consulta

Os artigos "Document Expansion by Query Prediction", de Rodrigo Nogueira, Wei Yang, Jimmy Lin e Kyunghyun Cho, de 2019, e a versão seguinte com o T5, conhecida como docT5query, atacam a incompatibilidade de vocabulário por um caminho que hoje parece quase ingênuo, e que funciona.

## A ideia

Em vez de melhorar o modelo de busca, melhore o documento. Para cada documento do corpus, um modelo generativo prevê **quais consultas aquele documento responderia**. As consultas previstas são anexadas ao texto do documento, e o índice é construído sobre o texto aumentado.

Depois disso, a busca é BM25 comum. Nenhuma mudança no recuperador.

## Por que funciona

O vocabulário do documento e o vocabulário de quem pergunta são diferentes. Um texto técnico diz "insuficiência cardíaca"; o usuário digita "falha do coração". Gerar consultas plausíveis para o documento traz o vocabulário do usuário para dentro do documento, antes da busca.

O efeito tem duas partes: termos novos entram no documento, o que resolve casamento que não existia, e termos que já estavam lá aparecem repetidos, o que aumenta o peso deles no BM25.

## Onde está o custo

Todo o custo vai para a **indexação**, e nenhum para a consulta. É preciso rodar um modelo generativo uma vez por documento do corpus, o que é caro e offline, mas depois a busca é tão barata quanto BM25 sempre foi.

Essa distribuição de custo é o oposto da do codificador cruzado, que não custa nada na indexação e custa em toda consulta.

## O contraste com o SPLADE

As duas técnicas resolvem o mesmo problema por dois níveis diferentes.

A expansão por geração de consulta opera no **nível do texto**: escreve palavras novas no documento, e deixa o BM25 pesá-las com a fórmula dele.

O SPLADE opera no **nível do peso**: não escreve texto nenhum, produz diretamente o vetor de pesos sobre o vocabulário, e aprende quanto cada token do vocabulário deve valer naquele documento.

A segunda é mais direta, porque não passa pelo intermediário de gerar texto e depois contá-lo. A primeira é mais simples de implantar, porque o índice continua sendo um índice de texto comum.
