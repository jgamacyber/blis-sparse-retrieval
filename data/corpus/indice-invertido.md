# Índice invertido e o que a esparsidade custa

## A estrutura

Um índice invertido mapeia cada termo do vocabulário para a lista dos documentos que o contêm, com a frequência e, às vezes, as posições. Essa lista é a lista de postagem do termo.

Buscar é percorrer as listas de postagem apenas dos termos presentes na consulta, acumulando score por documento. O corpus inteiro nunca é tocado: só as listas dos termos pedidos.

É essa propriedade que faz uma busca léxica responder em milissegundos sobre centenas de milhões de documentos, com décadas de otimização acumulada em compressão de listas, travessia e poda.

## Onde a esparsidade entra na conta

O custo de uma consulta é, grosso modo, a soma dos comprimentos das listas de postagem dos termos da consulta.

Isso dá duas alavancas, e as duas aparecem no regularizador do SPLADE.

**Esparsidade da consulta**: cada termo ativo na consulta é uma lista a percorrer. Consulta com trinta termos ativos custa muito mais que consulta com cinco. É por isso que o SPLADE coloca um coeficiente de regularização separado e maior para a consulta.

**Esparsidade do documento**: cada termo ativo em um documento o insere em uma lista de postagem. Se a expansão aprendida ativa duzentos termos por documento em vez dos cinquenta que ele literalmente contém, as listas crescem e a busca desacelera, mesmo que o índice ainda seja invertido.

## O termo que estraga tudo

O caso ruim não é o documento com muitos termos: é o **termo ativo em muitos documentos**. A lista de postagem dele fica enorme, e qualquer consulta que o inclua paga por isso.

O regularizador FLOPS do SPLADE é construído exatamente contra esse caso: ele soma o quadrado da ativação média de cada token sobre o lote, e o quadrado é o que penaliza desproporcionalmente o token que está ativo em todo lugar.

É um detalhe que mostra o cuidado do desenho: a penalidade não é por usar muitos termos, e sim por usar termos que destroem a estrutura que torna a busca léxica rápida.
