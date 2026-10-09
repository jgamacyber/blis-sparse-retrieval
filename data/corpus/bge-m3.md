# BGE-M3: três funções de recuperação em um modelo

O artigo "M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation", de Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian e Zheng Liu, da BAAI, foi publicado em 2024. O modelo é distribuído como BGE-M3, dentro do projeto FlagEmbedding.

## Os três M

**Multilinguismo**: recuperação semântica em mais de cem línguas, inclusive entre línguas diferentes.

**Multifuncionalidade**: o mesmo modelo produz, de uma só passada, as três formas de recuperação que até então exigiam modelos separados.

**Multigranularidade**: entrada de uma frase curta até documento de 8192 tokens.

## As três funções

**Recuperação densa.** O vetor é a representação normalizada do token especial de início:

```
e_q = norm(H_q[0])
s_dense = ⟨e_p, e_q⟩
```

É o vetor único por texto, com busca por produto interno.

**Recuperação léxica, ou esparsa.** Para cada token da entrada, um peso de importância sai de uma projeção linear com ReLU sobre a representação contextual:

```
w_qt = ReLU(W_lexᵀ · H_q[i])
```

Termo repetido mantém apenas o maior peso, que é a mesma decisão de agregação por máximo que o SPLADE v2 adotou. O score soma os produtos dos pesos nos termos que aparecem nos dois textos:

```
s_lex = Σ  (w_qt · w_pt)
       t ∈ q ∩ p
```

Repare na diferença em relação ao SPLADE: aqui a soma é restrita à **interseção** dos termos presentes nos dois textos, sem expansão para tokens ausentes. É casamento léxico com peso aprendido, não expansão aprendida.

**Recuperação multivetor, com interação tardia.** Em vez de um vetor por texto, um vetor por token, projetado e normalizado:

```
E_q = norm(W_mulᵀ · H_q)
E_p = norm(W_mulᵀ · H_p)
```

E o score é a média, sobre os tokens da consulta, do melhor casamento de cada um com algum token do documento:

```
s_mul = (1/N) Σ      max      E_q[i] · E_p[j]ᵀ
             i=1..N  j=1..M
```

Isso é a interação tardia no estilo do ColBERT. A estrutura "média dos máximos" é o que permite que um termo da consulta encontre seu correspondente em qualquer posição do documento, sem que a representação precise comprimir o texto todo em um ponto.

O custo é de armazenamento: guardar um vetor por token custa muito mais que um por texto. Por isso a função multivetor é usada normalmente só para reordenar os candidatos que as outras duas trouxeram.

## O score integrado

O artigo soma as três, sem pesos:

```
s_inter = s_dense + s_lex + s_mul
```

Vale notar que o artigo não define pesos padrão: cada parcela entra com peso um. Implementações costumam expor pesos configuráveis, e aí a escolha passa a ser do usuário, não do artigo.

## Autodestilação

O ponto de treino do artigo é usar o score integrado como professor dos três individuais.

Cada função é treinada com uma perda contrastiva do tipo InfoNCE:

```
L = -log  exp(s(q, p*)/τ) / Σ exp(s(q, p)/τ)
```

E cada função recebe também uma perda de destilação em que o alvo é a distribuição do score integrado:

```
L'_* = -p(s_inter) · log p(s_*)
```

As perdas de destilação são promediadas sobre as três funções, e o objetivo final soma as duas partes:

```
L_final = L + L'
```

A intuição é direta: as três funções erram de maneiras diferentes, então a combinação delas é um sinal melhor do que qualquer uma isolada, e pode ensinar cada uma a errar menos. É destilação sem modelo professor externo, só com o conjunto das próprias saídas.

## O que isso muda na prática

Antes do M3, montar um sistema híbrido significava manter dois ou três modelos, com dois ou três pipelines de indexação, e calibrar a fusão entre saídas que nunca foram treinadas juntas.

Com um modelo que produz as três representações na mesma passada, o híbrido deixa de ser uma montagem e passa a ser uma configuração. E as três saídas vêm de um treino em que elas se ensinaram mutuamente, o que é um argumento a favor de combiná-las.
