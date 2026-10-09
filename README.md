# BLIS Sparse Retrieval --> do BM25 à esparsidade aprendida

Módulo bônus da trilha de RAG do BLIS. A pergunta que organiza o repositório é a que todo mundo faz ao ver SPLADE e BGE-M3 pela primeira vez: **vale a pena substituir o BM25?**

A resposta curta, e que o código mede: **não, porque a pergunta está errada.** O SPLADE não substitui o BM25. Ele mantém tudo o que o BM25 tem de bom e troca apenas a função que produz o peso de cada termo.

> ⚠️ **Aviso de uso.** Este material é **educacional**, voltado ao estudo de arquiteturas de recuperação. As implementações são reproduções didáticas e simplificadas dos artigos originais, não substituem os métodos publicados nem os resultados neles reportados.

> 📄 **Sobre os artigos.** Os artigos que fundamentam cada peça são **citados ao longo de todo o trabalho**, nos docstrings dos módulos e nos comentários das decisões de projeto. Os PDFs **não** estão incluídos no repositório por questão de direitos autorais. Use as referências ao final para localizá-los nas fontes originais.

## A tese

BM25, SPLADE e a parcela léxica do BGE-M3 são o **mesmo objeto matemático**: um vetor esparso por documento, pontuado por produto interno, servido por um índice invertido. A diferença entre eles é de onde vem o peso:

| | De onde vem o peso | Expande? | Treina? |
|---|---|---|---|
| **BM25** | fórmula fechada, com IDF, saturação e normalização por tamanho | não | não |
| **SPLADE** | cabeça de linguagem mascarada de um transformador | sim, para termos ausentes | sim |
| **BGE-M3 léxico** | projeção linear aprendida, restrita à interseção | não | sim |

Isso está escrito em código, e não só em prosa: `bm25.py` produz o documento como `VetorEsparso` e um teste verifica que o produto interno no índice devolve **exatamente** o mesmo score da fórmula, até 1e-9. Se essa igualdade caísse, a tese cairia com ela.

O BM25 tem um defeito, e é só um: ele não casa o que não coincide literalmente. O SPLADE conserta esse defeito sem abandonar o índice invertido, o casamento exato nem a interpretabilidade. É isso, e não a troca de uma arquitetura por outra.

## O que o experimento mede

O conjunto de avaliação tem 24 consultas divididas em duas classes, e **a divisão é medida, não declarada**. A medida é a cobertura lexical: que fração dos termos da consulta aparece no trecho relevante.

| Classe | Consultas | Cobertura lexical média |
|---|---|---|
| `exato` | 12 | 0,75 |
| `parafrase` | 12 | 0,24 |

Rode `python main.py consultas` para ver a medida por consulta. Sem essa diferença, chamar uma consulta de "paráfrase" seria afirmação sobre a intenção de quem escreveu, e não sobre o dado.

## O resultado

Recall@5, backend `local`, 13 documentos e 68 trechos, relevância anotada por seção:

| Configuração | geral | `exato` | `parafrase` |
|---|---|---|---|
| BM25 (linha de base) | 0,667 | **0,958** | 0,375 |
| SPLADE (expande os dois) | **0,688** | **0,958** | 0,417 |
| SPLADE-doc (expande só o doc) | 0,646 | 0,875 | 0,417 |
| BGE-M3 denso | 0,542 | 0,792 | 0,292 |
| BGE-M3 léxico | 0,646 | **0,958** | 0,333 |
| BGE-M3 integrado | 0,667 | **0,958** | 0,375 |
| híbrido BM25+denso, RRF | 0,667 | 0,875 | 0,458 |
| híbrido BM25+denso, convexo | **0,688** | 0,875 | **0,500** |
| híbrido SPLADE+denso, RRF | 0,667 | **0,958** | 0,375 |

Três leituras, e a terceira é a que importa.

**Nenhuma configuração superou o BM25 em consultas de termo exato.** Quando a consulta traz o termo literal e raro, casamento exato é difícil de bater. Isso reproduz, em miniatura, o achado do benchmark de documentos financeiros que motivou este módulo.

**Todas ganharam dele em paráfrase**, que é onde o defeito do BM25 vive. O melhor foi o híbrido com combinação convexa, com 0,500 contra 0,375.

**E a coluna "geral" esconde as duas coisas.** O híbrido convexo ganha 0,125 em paráfrase e perde 0,083 em termo exato, e a média mostra um ganho modesto de 0,021 que não descreve nenhum dos dois casos. É por isso que toda tabela aqui é quebrada por tipo, e por isso o relatório recusa imprimir uma linha chamada "o melhor recuperador".

## O preço da expansão

O número que falta em quase toda comparação de recuperadores:

| | BM25 | SPLADE |
|---|---|---|
| termos ativos por trecho | 46,1 | 87,3 |
| postagens no índice | 3.134 | 5.939 |
| maior lista de postagem | 33 | 44 |
| FLOPS normalizado | 0,05 | 0,13 |

A busca continua em índice invertido, e continua cerca de duas vezes mais cara. No artigo, essa pressão é controlada pelo regularizador FLOPS durante o treino; aqui, pela poda em `SPLADE_TOP_TERMOS`.

Uma observação de método: o FLOPS **cru** não serve para comparar dois esquemas de peso, porque depende da escala dos pesos, e o IDF do BM25 vive em outra ordem de grandeza. Comparado cru, ele chega a inverter a conclusão. O repositório imprime as duas linhas e explica a diferença, em vez de escolher a que soa melhor.

## Os dois backends

Os modelos deste módulo têm peso e rodam localmente. A OpenRouter não serve nenhum dos três, então **não há chave de API e não há custo por consulta** em nenhum comando.

**`local`** (padrão): substitutos determinísticos, sem dependência além da biblioteca padrão. Mesma forma de saída e mesmo caminho de pontuação dos modelos reais, o que torna toda a mecânica testável de graça.

**`hf`**: SPLADE, BGE-M3 e bge-reranker-v2-m3 de verdade, via `transformers` e `FlagEmbedding`.

Trocar de backend não muda uma linha do índice, da fusão, da reordenação nem das métricas.

### Onde o backend local é honesto e onde não é

A mecânica é a mesma, então um teste que passa ali garante que o encanamento está certo. A qualidade não é a mesma, e a diferença tem direção conhecida:

- O expansor por coocorrência só conhece relações que aparecem **neste** corpus. O SPLADE traz o que o modelo aprendeu em pré-treino, e liga termos que nunca apareceram juntos aqui.
- O embedder por hashing não tem semântica: "falha do coração" e "insuficiência cardíaca" ficam longe. Ele mede sobreposição de n-gramas. Isso **subestima a parcela densa** e, por consequência, superestima a importância relativa do léxico na tabela acima.
- O substituto do reordenador pontua por cobertura e proximidade de termos. Não vê negação, condição nem a que entidade um número se refere, que é exatamente o que um codificador cruzado enxerga.

Essa última limitação aparece nos números, e o relatório a interpreta a partir do dado em vez de esconder: alargando o conjunto de candidatos de 5 para 40, o **teto** do primeiro estágio sobe de 0,688 para 0,979, e o Recall@5 final não acompanha. A leitura correta é que o gargalo passou para o segundo estágio, e nesse backend isso era esperado. Rode com `BACKEND=hf` para medir o reordenador de verdade.

## Início rápido

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt       # não instala nada: o backend local não precisa

python testes_offline.py              # 201 verificações determinísticas
python testes_mock.py                 # 48 verificações do caminho dos modelos reais

python main.py consultas              # a medida que separa os dois tipos
python main.py expansao "a incompatibilidade de vocabulario atrapalha o casamento literal"
python main.py avaliar                # a ablação completa
```

Com os modelos de verdade:

```bash
pip install -r requirements-hf.txt    # torch, transformers, FlagEmbedding
BACKEND=hf python main.py avaliar
```

O passo a passo completo, com o que esperar de cada comando, está em **[REPRODUTIBILIDADE.md](./REPRODUTIBILIDADE.md)**.

## Comandos

Todos rodam sem instalar nada no backend padrão:

| Comando | O que mostra |
|---|---|
| `config` | configuração ativa e backend |
| `corpus` | documentos, chunking por seção e vocabulário |
| `consultas` | o conjunto e a cobertura lexical por tipo |
| `bm25 "..."` | BM25 com a contribuição de cada termo ao score |
| `expansao "..."` | o que a expansão acrescentou, separado dos termos literais |
| `comparar "..."` | a mesma consulta nos sete recuperadores |
| `indices` | o preço da expansão, e os termos de lista mais longa |
| `fusao "..."` | RRF com dois valores de k contra três combinações convexas |
| `pipeline "..."` | dois estágios, com o teto do primeiro e os logits do reordenador |
| `avaliar` | a ablação completa, com as três tabelas e a leitura dos números |

## Estrutura

```
blis-sparse-retrieval/
├── corpus.py            chunking por seção e cobertura lexical
├── esparso.py           vetor esparso e índice invertido (a estrutura comum)
├── bm25.py              BM25 escrito como produtor de vetores esparsos
├── splade.py            log(1+ReLU), max pooling, FLOPS, dois backends
├── m3.py                as três funções do BGE-M3 e o score integrado
├── reranker.py          codificador cruzado, sigmoide e teto do 1º estágio
├── backends.py          os substitutos determinísticos, e o que eles não fazem
├── fusao.py             RRF contra combinação convexa
├── metricas.py          Hit, Recall, MRR, nDCG, e qual usar em cada estágio
├── sistema.py           indexa uma vez, responde por qualquer configuração
├── avaliacao.py         ablação e relatório, com leitura dirigida pelo dado
├── main.py              CLI
├── testes_offline.py    201 verificações determinísticas
├── testes_mock.py       48 verificações do caminho dos modelos reais
└── data/
    ├── corpus/          13 resumos autorais dos estudos
    └── qa_eval.json     24 consultas anotadas por seção e por tipo
```

## Corpus

Os 13 documentos em `data/corpus/` são **resumos autorais em português**, escritos a partir da leitura dos artigos. Não são traduções nem trechos reproduzidos.

Eles cobrem BM25, SPLADE, BGE-M3, reordenação com codificador cruzado, o benchmark de documentos com tabelas, ColBERT, doc2query, DPR, RRF, BEIR, índice invertido, incompatibilidade de vocabulário e avaliação de recuperação. O corpus é, de propósito, sobre o próprio assunto do módulo: assim as consultas de avaliação são perguntas que alguém faria de verdade.

## O que este módulo não faz

Para não prometer o que não entrega:

- **Não treina nada.** O regularizador FLOPS é medido, não minimizado. Treinar um SPLADE exige dado anotado, GPU e as decisões de destilação do artigo.
- **Não usa índice aproximado.** As parcelas densa e multivetor varrem o corpus. Com 68 trechos é irrelevante; com milhões, exigiria busca aproximada, e a parcela multivetor exigiria muito mais memória.
- **Não mede latência.** As contas de custo são estruturais: termos ativos, postagens, pares avaliados. Tempo de parede depende de máquina e não diria nada reproduzível.
- **Não tem conjunto grande.** São 24 consultas. Diferença de duas ou três consultas muda a terceira casa decimal, então leia as tabelas como demonstração de mecanismo e como geradora de hipótese, não como evidência de que um recuperador é melhor que outro.

## Licença

MIT, ver [`LICENSE`](./LICENSE).

A licença cobre o **código deste repositório**. Os artigos citados pertencem a seus respectivos autores e editoras. Os modelos SPLADE, BGE-M3 e bge-reranker-v2-m3 têm suas próprias licenças, nos respectivos repositórios.

## Referências

1. Robertson, S. e Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval.
2. Formal, T., Lassance, C., Piwowarski, B. e Clinchant, S. (2021). *SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval.* arXiv:2109.10086.
3. Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D. e Liu, Z. (2024). *M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.* arXiv:2402.03216.
4. Akarsu, M., Karaman, R. K. e Mierbach, C. (2026). *From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents.* arXiv:2604.01733.
5. Khattab, O. e Zaharia, M. (2020). *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* SIGIR.
6. Nogueira, R., Yang, W., Lin, J. e Cho, K. (2019). *Document Expansion by Query Prediction.* arXiv.
7. Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP.
8. Cormack, G., Clarke, C. e Buettcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR.
9. Thakur, N., Reimers, N., Rücklé, A., Srivastava, A. e Gurevych, I. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets & Benchmarks.

Repositórios e modelos usados como referência de implementação: [naver/splade](https://github.com/naver/splade), [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding), [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3).

## Sobre

Bônus da trilha prática do **BLIS, Brazilian Laboratory for Intelligent Systems**, 2026, complementando os módulos de RAG.

Outros repositórios da trilha: [BLIS_LLM_SEC](https://github.com/jgamacyber/BLIS_LLM_SEC), [blis-rag](https://github.com/jgamacyber/blis-rag), [blis-prompt-engineering](https://github.com/jgamacyber/blis-prompt-engineering), [blis-evals](https://github.com/jgamacyber/blis-evals) e [blis-agents](https://github.com/jgamacyber/blis-agents).
