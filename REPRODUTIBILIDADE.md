# Documento de Reprodutibilidade

Roteiro para reproduzir todos os resultados deste repositório na sua máquina. Cada comando traz o que esperar de saída.

Uma diferença em relação aos outros módulos da trilha: **este repositório não usa API**. Não há chave, não há custo por consulta e não há limite de requisição. O custo do backend `hf` é download de pesos e tempo de máquina.

---

## 1. Ambiente

### Requisitos

- Python **3.10 ou superior** (o código usa `X | None`, sintaxe de união de tipos do 3.10)
- Para o backend `local`: nada além disso. Sem internet, sem GPU, sem dependência.
- Para o backend `hf`: internet para o download inicial dos pesos, e cerca de 6 GB livres em disco.

```bash
python --version      # deve mostrar 3.10+
```

Se a sua distribuição tiver `python3` em vez de `python`, use `python3` em todos os comandos.

### Instalação

```bash
cd blis-sparse-retrieval

python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows (PowerShell)

pip install -r requirements.txt
```

O `requirements.txt` é propositalmente vazio de pacotes: o backend `local` roda com a biblioteca padrão. BM25, índice invertido, SPLADE, as três funções do BGE-M3, fusão, métricas e as duas suítes de teste não precisam de nada instalado.

### Configuração

Opcional, porque todos os padrões funcionam:

```bash
cp .env.example .env              # Windows: copy .env.example .env
```

O `.env` aqui não guarda segredo nenhum, só escolhas de backend e de hiperparâmetro. Ele está no `.gitignore` por hábito e porque é o arquivo local de cada um.

**Importante rodar de dentro da pasta do repositório.** Os caminhos de dados são relativos (`data/corpus`, `data/qa_eval.json`), então chamar o `main.py` de fora não encontra o corpus. O erro diz isso.

---

## 2. Validação sem custo

As duas suítes não fazem uma única chamada de rede, não baixam modelo e levam **menos de um segundo cada**.

```bash
python testes_offline.py
python testes_mock.py
```

Esperado ao final de cada uma:

```
======================================================================
  Todos os testes passaram.
======================================================================
```

São **201 verificações offline** e **48 do caminho dos modelos reais**. Se alguma falhar, a saída lista o nome de cada verificação que falhou antes do resumo.

### O que a suíte offline cobre

Vale saber o que ela garante, porque é o que dá confiança no resto.

**BM25 contra a fórmula, com números feitos à mão.** Se a saturação ou a normalização por tamanho estiverem erradas, todo número do repositório fica errado em silêncio e a linha de base deixa de ser uma linha de base. Há verificações específicas para `b=0` (normalização desligada), `b=1` (normalização completa) e para o fato de três ocorrências não valerem três vezes uma.

**A equivalência entre o índice e a fórmula.** O produto interno no índice invertido devolve exatamente o mesmo score do cálculo direto, até 1e-9, em cinco consultas diferentes. Essa igualdade é a tese do repositório escrita em código: se o BM25 não puder ser expresso como produto interno de vetores esparsos, a afirmação de que ele e o SPLADE são o mesmo objeto cai.

**As transformações do SPLADE.** `log(1+ReLU(x))` zera logit negativo e satura o positivo; a agregação por máximo e por soma dão resultados diferentes para termo repetido, que é a mudança do v2; o FLOPS penaliza mais o termo concentrado que o espalhado.

**As três funções do BGE-M3**, com valores calculados à mão, inclusive a diferença entre média dos máximos (M3) e soma dos máximos (ColBERT): repetir a consulta não muda o score multivetor, por ser média.

**A degeneração do minmax com listas curtas**, que é um argumento concreto a favor da fusão por posto e está documentado em um teste em vez de em um comentário.

### O que a suíte de modelos simulados cobre

Esta existe por um motivo específico: o backend `hf` é o que produz números comparáveis com os artigos, e é justamente o que não roda em máquina sem GPU nem download. Sem esses testes, o caminho real seria publicado sem nunca ter sido executado.

O erro mais provável nesse caminho é silencioso: um eixo trocado no agrupamento, a máscara de atenção esquecida, o mapeamento de id para token invertido. Nada disso levanta exceção, e tudo isso estraga o vetor.

A estratégia é substituir o tensor e o modelo, não a aritmética: o `TensorFalso` implementa exatamente as operações que o código usa, sobre listas aninhadas, e os valores esperados são calculados à mão no teste. O modelo simulado tem uma posição de preenchimento com logit 9 em todo o vocabulário, então **se a máscara de atenção for esquecida, o teste quebra**.

---

## 3. Comandos, um por um

Todos no backend `local`, sem custo.

### 3.1 Configuração

```bash
python main.py config
```

Esperado:

```
  backend=local | chunk=900/150 | bm25 k1=1.2 b=0.75 | rrf k=60 | candidatos=20 | top_k=5

  backend 'local': substitutos determinísticos, não os modelos dos artigos. ...

  Nenhum comando deste repositório chama API, nos dois backends.
```

### 3.2 Corpus e chunking

```bash
python main.py corpus
```

Esperado: **13 documentos, 68 trechos**, vocabulário de **1336 termos distintos**, e a lista de quantos trechos e seções cada documento produziu.

Se você mudar `TAMANHO_CHUNK`, o número de trechos muda. A anotação de relevância sobrevive a isso, porque é por seção e não por trecho.

### 3.3 O conjunto de avaliação

```bash
python main.py consultas
```

Este é o comando mais importante para entender o experimento. Ele imprime, por consulta, a **cobertura lexical**: a fração dos termos da consulta que aparece no trecho relevante.

Esperado ao final:

```
  exato      12 consultas, cobertura média 0.75 (mín 0.33, máx 1.00)
  parafrase  12 consultas, cobertura média 0.24 (mín 0.10, máx 0.57)
```

A diferença entre as duas médias é o que torna o experimento possível. Sem ela, a divisão em tipos seria afirmação sobre a intenção de quem escreveu as consultas, e não uma propriedade medida do dado.

O comando também valida a anotação: se algum alvo anotado não existir no corpus, ele avisa e devolve código de saída 1.

### 3.4 BM25 com explicação

```bash
python main.py bm25 "o que e MaxSim"
```

Esperado:

```
  termos no vocabulário: 1 (cobertura 100%)

  [  4.23] colbert#Por que isso aparece aqui
           contribuição: maxsim=4.23
```

A linha de contribuição é interpretabilidade, que é uma propriedade da recuperação léxica que a densa não tem, e que só existe se alguém a expuser.

Vale tentar também uma consulta cujo vocabulário quase não existe no corpus:

```bash
python main.py bm25 "qual a dosagem recomendada do medicamento"
```

A linha de cobertura mostra 25%, ou seja, três dos quatro termos não existem no índice e não contribuem com nada. O que sobra é um casamento fraco em um termo genérico. É assim que o modo de falha do casamento literal aparece na prática: não com uma lista vazia, e sim com uma lista plausível e errada.

Para contraste, `python main.py bm25 "insuficiencia cardiaca em pacientes idosos"` **funciona bem**, porque esse par de termos é o exemplo usado nos documentos do corpus sobre incompatibilidade de vocabulário. Serve de lembrete de que "termo fora do vocabulário" é propriedade do corpus, não da consulta.

### 3.5 A expansão, visível

```bash
python main.py expansao "a incompatibilidade de vocabulario atrapalha o casamento literal"
```

Esperado, separando o vetor em duas partes:

```
  literais (5):
    incompatibilidade=1.25, vocabulario=1.25, atrapalha=1.25, casamento=1.25, literal=1.25

  acrescentados pela expansão (18):
    cardiaca=0.61, coracao=0.61, falha=0.61, insuficiencia=0.61, ...
```

Repare em duas coisas. Primeiro, o exemplo canônico do problema ("insuficiência cardíaca" e "falha do coração") aparece como expansão, porque é isso que coocorre com "vocabulário" neste corpus. Segundo, e mais importante: **os literais pesam o dobro dos acrescentados**. É isso que mantém o casamento exato dominando, e é a razão de a esparsidade aprendida não trocar um defeito por outro.

### 3.6 Comparar recuperadores

```bash
python main.py comparar "quantos documentos mandar para a segunda etapa"
```

Mostra a mesma consulta nos sete recuperadores. Repare que as escalas de score não são comparáveis entre blocos: BM25 na casa das unidades, denso entre menos um e um, fusão recíproca perto de zero. É exatamente por isso que fundir somando exige normalizar.

### 3.7 O preço da expansão

```bash
python main.py indices
```

Esperado:

```
                                       BM25       SPLADE
  termos ativos por trecho            46.09        87.34
  postagens no índice                  3134         5939
  maior lista de postagem                33           44
  FLOPS cru (escalas diferentes)      19.09        10.81
  FLOPS normalizado                    0.05         0.13
```

As duas linhas de FLOPS merecem atenção, e são um bom exemplo de artefato de medição. A cru depende da escala dos pesos: o IDF do BM25 leva os valores dele a outra ordem de grandeza, e comparado cru o BM25 parece mais caro, **o que é falso**, já que ele tem metade das postagens. A normalizada compara estrutura e dá a conclusão certa.

No artigo isso não aparece porque o regularizador é aplicado ao próprio modelo durante o treino, sempre na mesma escala.

### 3.8 Fusão

```bash
python main.py fusao "por que nao da para somar as notas de dois buscadores"
```

Mostra as duas listas de entrada e cinco fusões: RRF com `k=60` e `k=10`, e combinação convexa com minmax, zscore e pesos 2:1.

### 3.9 Pipeline de dois estágios

```bash
python main.py pipeline "quantos documentos devo mandar para a segunda etapa"
```

Esperado: a ordem antes e depois da reordenação, com os logits e a posição anterior de cada candidato, mais a informação de se o primeiro colocado mudou.

Os logits do substituto local ficam próximos entre si, na casa de -1,8. Isso é sintoma de que as características dele (cobertura, proximidade, densidade) quase não diferenciam esses candidatos, e é a limitação declarada do backend `local`.

### 3.10 A ablação completa

```bash
python main.py avaliar
python main.py avaliar --erros              # mostra onde cada configuração erra
python main.py avaliar --candidatos 5,10,20,40,60
python main.py avaliar --sem-reordenacao    # mais rápido
```

Leva poucos segundos no backend `local`. Imprime quatro blocos e salva o JSON em `saidas/relatorio_local.json`.

**Tabela principal**, Recall@5 por tipo de consulta. Esperado:

```
  configuração                      geral       exato   parafrase
  BM25 (linha de base)              0.667       0.958       0.375
  SPLADE (expande os dois)          0.688       0.958       0.417
  BGE-M3 denso                      0.542       0.792       0.292
  híbrido BM25+denso, convexo       0.688       0.875       0.500
```

**Leitura dos números**, construída a partir do que foi medido e não de texto fixo. Se a ordem mudar com outro backend ou outro corpus, o texto muda com ela.

**Tamanho do conjunto de candidatos.** Esperado:

```
     n   Recall@k      MRR     teto    pares
     5      0.688    0.538    0.688      116
    10      0.646    0.524    0.729      231
    20      0.646    0.531    0.917      461
    40      0.646    0.527    0.979      903
```

O `teto` sobe de 0,688 para 0,979 e o Recall@5 não acompanha. A leitura que o comando imprime é que o gargalo passou para o segundo estágio: o relevante passou a chegar ao reordenador e ele não o colocou no topo. No backend `local` isso era esperado, e o próprio relatório diz que esse número é evidência sobre o substituto, não sobre reordenação em geral.

**Fusão.** Esperado: RRF `k=10` acima de `k=60`, e a combinação convexa acima das duas, o que reproduz a direção do achado do benchmark de documentos com tabelas.

---

## 4. Rodando com os modelos de verdade

```bash
pip install -r requirements-hf.txt
BACKEND=hf python main.py avaliar
```

### O que vai acontecer

A primeira execução baixa os pesos. Ordem de grandeza aproximada:

| Modelo | Tamanho aproximado |
|---|---|
| `naver/splade-cocondenser-ensembledistil` | 440 MB |
| `BAAI/bge-m3` | 2,2 GB |
| `BAAI/bge-reranker-v2-m3` | 2,2 GB |

Confira os tamanhos atuais nas páginas dos modelos, porque eles mudam. Reserve cerca de 6 GB livres, contando o cache do HuggingFace.

Em **CPU** funciona e é lento. A indexação roda o SPLADE e o BGE-M3 uma vez por trecho, são 68 trechos, e a reordenação roda o codificador cruzado uma vez por par. Com `--candidatos 5,10` em vez do padrão, a parte mais cara encurta bastante.

Em **GPU**, defina `DISPOSITIVO=cuda`. O `use_fp16` é ligado automaticamente quando o dispositivo não é `cpu`.

### Comandos úteis no backend hf

```bash
BACKEND=hf python main.py expansao "a incompatibilidade de vocabulario atrapalha o casamento literal"
```

Esta é a comparação mais instrutiva do repositório. Rode o mesmo comando nos dois backends e compare a lista de termos acrescentados. O backend `local` só acha o que coocorre neste corpus; o SPLADE traz o que o modelo aprendeu em pré-treino, e liga termos que nunca apareceram juntos aqui. A diferença é grande e é o ponto todo da esparsidade aprendida.

```bash
BACKEND=hf python main.py indices
```

No backend `hf` os tokens são do vocabulário do BERT, com marcação de subpalavra, e o número de termos ativos por trecho costuma ser bem maior que no substituto local. A linha de FLOPS normalizado é a que compara.

```bash
BACKEND=hf python main.py avaliar --candidatos 5,10,20,40
```

A ablação de candidatos no backend `hf` é onde a comparação com o benchmark de documentos financeiros faz sentido. Lá, o reordenador levou o Recall@5 de 0,695 para 0,816, e o tamanho do conjunto de candidatos dominou o resultado: 20 candidatos deram 0,458, 50 deram 0,826 e 100 deram 0,888.

Com 24 consultas e 68 trechos, não espere reproduzir esses valores. O que vale comparar é a **direção**: se o teto subir e o Recall acompanhar, o reordenador está fazendo trabalho de verdade, ao contrário do que acontece no backend `local`.

---

## 5. Solução de problemas

**`FileNotFoundError: corpus não encontrado`**
Você rodou o comando de fora da pasta do repositório. Entre nela primeiro.

**`backend 'hf' precisa de torch e transformers`**
É a mensagem esperada quando o extra não está instalado. Ela diz o arquivo a instalar e lembra que `BACKEND=local` funciona sem nada. Esse caminho é coberto por teste, justamente para a mensagem não degenerar em um `ImportError` cru.

**O download trava ou cai**
Os pesos vêm do HuggingFace Hub. Em rede com proxy, configure `HF_ENDPOINT` ou as variáveis de proxy do `huggingface_hub`. O backend `local` continua rodando tudo enquanto isso.

**Falta de memória no backend hf em CPU**
Baixe `MAX_TOKENS_MODELO` de 512 para 256 e rode `--sem-reordenacao` primeiro, para separar o custo da indexação do custo da reordenação.

**Os números não batem com os meus**
No backend `local` eles devem bater exatamente: tudo ali é determinístico, sem semente aleatória e sem rede. Se não baterem, algo mudou no corpus, no chunking ou nos hiperparâmetros, e vale comparar a saída de `python main.py config` e `python main.py corpus`.

No backend `hf` não vão bater exatamente, porque versão de modelo, de biblioteca e aritmética de ponto flutuante em GPU variam. O que deve se manter é o comportamento estrutural: literais pesando mais que acrescentados, BM25 forte em consulta de termo exato, expansão ajudando em paráfrase, e o teto do primeiro estágio crescendo com o conjunto de candidatos.

**A tabela mudou de ordem depois de eu mexer em um parâmetro**
Era o esperado, e é a razão de a leitura do relatório ser calculada a partir dos números em vez de escrita à mão. Com 24 consultas, duas ou três mudando de lado já alteram a ordem das configurações. Leia as tabelas como demonstração de mecanismo, não como evidência de que um recuperador é melhor que outro.
