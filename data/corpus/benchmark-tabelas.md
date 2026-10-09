# Um benchmark que contraria a expectativa

O artigo "From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents", de Meftun Akarsu, Recep Kaan Karaman e Christopher Mierbach, foi publicado como preprint em abril de 2026. Ele compara estratégias de recuperação em perguntas e respostas sobre documentos financeiros que misturam texto corrido e tabelas.

## O resultado que incomoda

Em quase toda métrica, o **BM25 ganhou da recuperação densa**. Recall@5 de 0,644 para o BM25 contra 0,587 para o denso, usando um modelo de embedding grande e comercial. Só em Recall@20 os dois praticamente empatam.

A explicação que os autores oferecem é a terminologia financeira exata. Nomes de linha de balanço, rubricas contábeis e códigos são termos raros e precisos, e representação densa tende a diluir exatamente esse tipo de termo.

Ou seja, não é um resultado contra recuperação neural em geral. É um resultado sobre o tipo de consulta: quando a consulta depende de termo exato, o casamento léxico é difícil de bater.

## O que realmente deu ganho

A configuração melhor foi o **híbrido seguido de reordenação**, com Recall@5 de 0,816. Para comparar: híbrido com fusão recíproca de postos sozinho dá 0,695, BM25 sozinho 0,644 e denso sozinho 0,587.

O salto de 0,695 para 0,816 vem do reordenador. É o maior ganho de uma única peça em toda a comparação, e é coerente com o que se vê em outros domínios: o segundo estágio rende mais que trocar o primeiro.

## Fusão: a combinação convexa bateu a fusão de postos

Na ablação de fusão, a combinação convexa com pesos iguais deu 0,726, acima da fusão recíproca de postos com `k = 60`, que deu 0,695. Baixar o `k` da fusão para 10 melhorou para 0,716.

Isso é um recado contra o automatismo. A fusão recíproca de postos virou padrão porque dispensa calibrar escala entre sistemas de score incomparáveis, mas ela joga fora a magnitude do score, e nem sempre esse descarte é de graça. O `k = 60` é herança de um artigo de 2009, não uma constante da natureza.

## O tamanho do conjunto de candidatos domina o reordenador

A ablação mais instrutiva: com 20 candidatos, o Recall@5 depois da reordenação é 0,458. Com 50, sobe para 0,826. Com 100, para 0,888.

Vinte candidatos é pior que não reordenar, porque o teto do sistema fica preso ao recall do primeiro estágio naquele corte. O reordenador não inventa documento que não recebeu.

## HyDE e multiconsulta não ajudaram

O HyDE ficou **abaixo** do denso puro, e a multiconsulta deu ganho desprezível sobre o BM25.

As duas técnicas funcionam gerando texto para enriquecer a consulta. Em consulta numérica e precisa, essa geração introduz mais ruído que sinal: o documento hipotético inventa números plausíveis que não são os do corpus.

## Onde o sistema erra

De cem falhas amostradas, 73% vêm de **incompatibilidade de estrutura de tabela**, e 20% de raciocínio numérico.

Isso coloca a discussão em outro lugar. Se três quartos dos erros são de como a tabela foi convertida em texto para indexação, trocar o recuperador melhora pouco. O conserto está antes, na ingestão.

## A lição para este módulo

Três coisas para levar.

Primeiro: BM25 não é linha de base fraca. Em corpus com terminologia exata, ele é adversário difícil, e qualquer afirmação de que foi superado precisa dizer em que tipo de consulta.

Segundo: o maior ganho por unidade de esforço está no segundo estágio, não no primeiro.

Terceiro: antes de escolher recuperador, vale olhar de onde vêm os erros. Um pipeline pode estar limitado por chunking e conversão de tabela, e nesse caso a escolha entre BM25, SPLADE e denso decide pouco.
