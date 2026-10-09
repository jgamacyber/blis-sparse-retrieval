"""
Substitutos determinísticos do backend `local`.

Nada aqui é um modelo treinado. São três peças construídas para ter a **mesma
forma de saída** dos modelos reais, de modo que o índice, a fusão, a
reordenação, as métricas e os testes exercitem o caminho verdadeiro sem GPU e
sem download:

1. `ExpansorCoocorrencia` ocupa o lugar da cabeça de linguagem mascarada do
   SPLADE. Em vez de prever token no contexto, ele expande um termo para os
   termos que mais coocorrem com ele no próprio corpus.
2. `EmbedderHash` ocupa o lugar do vetor denso do BGE-M3. É um saco de
   n-gramas projetado por hashing em dimensão fixa.
3. `pontuar_par` ocupa o lugar do codificador cruzado da reordenação. É um
   escore por características de casamento entre consulta e passagem.

**Onde isso é honesto e onde não é.** A forma é a mesma e a mecânica é a mesma,
então um teste que passa aqui garante que o encanamento está certo. A
qualidade não é a mesma e a diferença é grande:

- O expansor por coocorrência só conhece relações que aparecem **neste**
  corpus. O SPLADE traz o que o modelo aprendeu em pré-treino, e por isso
  liga termos que nunca apareceram juntos aqui. A diferença fica maior quanto
  menor o corpus.
- O embedder por hashing não tem semântica nenhuma: dois textos com palavras
  diferentes e sentido igual ficam longe. Ele mede sobreposição de n-gramas
  com outra aparência, não significado.
- O escore por características não vê negação, condição nem a quem um número
  se refere, que é justamente o que o codificador cruzado enxerga e é a razão
  de ele existir.

Ou seja: os números do backend `local` dizem respeito a este corpus e a estes
substitutos. Para comparar com os artigos, é `BACKEND=hf`. Essa frase está
repetida no relatório de propósito, porque é o tipo de ressalva que se perde
quando alguém copia uma tabela para um slide.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict

from corpus import tokenizar


# --------------------------------------------------------------------------- #
# 1. Expansão por coocorrência (substituto da cabeça MLM)
# --------------------------------------------------------------------------- #


class ExpansorCoocorrencia:
    """
    Para cada termo, os termos que mais coocorrem com ele no corpus.

    A força da associação usa informação mútua pontual positiva:

        pmi(a, b) = log( p(a, b) / (p(a) · p(b)) )

    O que a PMI corrige em relação à contagem crua é o viés de frequência: dois
    termos comuns coocorrem muito só por serem comuns. Dividir pelo produto das
    marginais mede o excesso de coocorrência em relação ao esperado por acaso,
    que é o que significa "estes termos andam juntos".

    Mantém-se apenas a parte positiva, porque associação negativa não serve
    para expandir: saber que dois termos se evitam não ajuda a recuperar nada.
    """

    def __init__(self, textos: list[str], top_por_termo: int = 6,
                 minimo_coocorrencia: int = 2,
                 fracao_maxima_df: float = 0.25) -> None:
        self.top_por_termo = top_por_termo
        self.minimo = minimo_coocorrencia
        # Termo presente em boa parte do corpus não é expandido. Expandir
        # "documento" ou "consulta" neste corpus acrescenta ruído a toda
        # consulta, e o ganho da expansão está nos termos discriminativos.
        self.fracao_maxima_df = fracao_maxima_df

        contagem_termo: Counter = Counter()
        contagem_par: Counter = Counter()
        n_docs = 0

        for texto in textos:
            termos = set(tokenizar(texto))
            if not termos:
                continue
            n_docs += 1
            contagem_termo.update(termos)
            ordenados = sorted(termos)
            for i, a in enumerate(ordenados):
                for b in ordenados[i + 1:]:
                    contagem_par[(a, b)] += 1

        self.n_docs = max(1, n_docs)
        self.contagem_termo = contagem_termo

        # Piso de 2: com a fração sozinha, um corpus pequeno zera o limite e
        # nada é expandido, em silêncio. Com 5 documentos e fração 0,25 o
        # limite daria 1, e um termo presente em 2 documentos já seria tratado
        # como comum. O piso evita que a expansão desapareça sem aviso.
        self.df_maximo = max(2, int(self.n_docs * self.fracao_maxima_df))

        bruto: dict[str, list[tuple[str, float]]] = defaultdict(list)
        maior_pmi = 0.0

        for (a, b), n_ab in contagem_par.items():
            if n_ab < self.minimo:
                continue
            p_ab = n_ab / self.n_docs
            p_a = contagem_termo[a] / self.n_docs
            p_b = contagem_termo[b] / self.n_docs
            pmi = math.log(p_ab / (p_a * p_b)) if p_a and p_b else 0.0
            if pmi <= 0:
                continue

            maior_pmi = max(maior_pmi, pmi)
            if contagem_termo[a] <= self.df_maximo:
                bruto[a].append((b, pmi))
            if contagem_termo[b] <= self.df_maximo:
                bruto[b].append((a, pmi))

        # Normalização GLOBAL, não por termo. Normalizar por termo faria o
        # vizinho mais forte de cada termo valer sempre 1,0, achatando a
        # diferença entre uma associação forte e uma fraca: todo termo passaria
        # a ter uma expansão de peso máximo, inclusive os que não têm nenhuma
        # associação boa. Com escala global, força de associação vira peso.
        self.maior_pmi = maior_pmi or 1.0

        self.expansoes: dict[str, list[tuple[str, float]]] = {}
        for termo, vizinhos in bruto.items():
            vizinhos.sort(key=lambda kv: (-kv[1], kv[0]))
            cortados = vizinhos[:top_por_termo]
            self.expansoes[termo] = [
                (t, p / self.maior_pmi) for t, p in cortados
            ]

    def expandir(self, termo: str) -> list[tuple[str, float]]:
        return self.expansoes.get(termo, [])

    def __len__(self) -> int:
        return len(self.expansoes)

    def estatisticas(self) -> dict:
        tamanhos = [len(v) for v in self.expansoes.values()]
        return {
            "termos_com_expansao": len(self.expansoes),
            "expansoes_por_termo": (sum(tamanhos) / len(tamanhos)) if tamanhos else 0.0,
            "documentos_vistos": self.n_docs,
            "df_maximo_para_expandir": self.df_maximo,
            "maior_pmi": round(self.maior_pmi, 3),
        }


# --------------------------------------------------------------------------- #
# 2. Embedder por hashing (substituto do vetor denso)
# --------------------------------------------------------------------------- #


class EmbedderHash:
    """
    Saco de n-gramas de caracteres projetado por hashing, normalizado em L2.

    Determinístico, sem treino e sem rede. Mede sobreposição de n-gramas, o
    que o torna tolerante a plural e a pequena variação de grafia, e cego a
    sinônimo: "falha do coração" e "insuficiência cardíaca" não compartilham
    n-gramas e caem longe.

    Essa cegueira é exatamente o ponto cego que a recuperação densa de verdade
    resolve, então convém lembrar que, no backend `local`, a parcela densa do
    BGE-M3 está sendo representada por algo que não faz o trabalho dela.
    """

    def __init__(self, dim: int = 256, n: int = 4) -> None:
        self.dim = dim
        self.n = n

    def _ngramas(self, texto: str) -> list[str]:
        base = " ".join(tokenizar(texto, manter_stopwords=True))
        if not base:
            return []
        preenchido = f" {base} "
        ngramas = [preenchido[i:i + self.n]
                   for i in range(max(1, len(preenchido) - self.n + 1))]
        # Palavras inteiras também entram, para que termo raro e exato pese.
        return ngramas + base.split()

    def codificar(self, texto: str) -> list[float]:
        vetor = [0.0] * self.dim
        for grama in self._ngramas(texto):
            digest = hashlib.blake2b(grama.encode("utf-8"), digest_size=8).digest()
            valor = int.from_bytes(digest, "big")
            indice = valor % self.dim
            sinal = 1.0 if (valor >> 63) & 1 else -1.0
            vetor[indice] += sinal

        norma = math.sqrt(sum(v * v for v in vetor))
        if norma == 0:
            return vetor
        return [v / norma for v in vetor]

    def codificar_lote(self, textos: list[str]) -> list[list[float]]:
        return [self.codificar(t) for t in textos]


def produto_interno(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
# 3. Escore por características (substituto do codificador cruzado)
# --------------------------------------------------------------------------- #


def pontuar_par(consulta: str, passagem: str) -> float:
    """
    Escore de relevância por características de casamento.

    Três sinais, todos calculáveis sem modelo:

    - **cobertura**: fração dos termos da consulta presentes na passagem. É o
      sinal mais forte e o mais óbvio.
    - **proximidade**: quão juntos os termos da consulta aparecem na passagem.
      Dois termos pedidos juntos e encontrados na mesma frase significam mais
      que os mesmos dois em pontos opostos do texto. Esse é o único sinal aqui
      que o produto interno do primeiro estágio não tem.
    - **densidade**: a cobertura descontada pelo tamanho da passagem, para que
      passagem longa não ganhe por acumular termos sem foco.

    A saída é um logit, no mesmo formato do reranker real: valor sem escala
    limitada, positivo para casamento bom e negativo para ruim, para que o
    caminho da sigmoide seja o mesmo nos dois backends.
    """
    termos_q = tokenizar(consulta)
    if not termos_q:
        return -10.0

    tokens_p = tokenizar(passagem)
    if not tokens_p:
        return -10.0

    posicoes: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(tokens_p):
        posicoes[t].append(i)

    presentes = [t for t in set(termos_q) if t in posicoes]
    cobertura = len(presentes) / len(set(termos_q))

    # Proximidade: menor janela que contém o máximo de termos distintos.
    proximidade = 0.0
    if len(presentes) >= 2:
        todas = sorted((p, t) for t in presentes for p in posicoes[t])
        melhor = None
        distintos = len(presentes)
        for i in range(len(todas)):
            vistos = set()
            for j in range(i, len(todas)):
                vistos.add(todas[j][1])
                if len(vistos) == distintos:
                    janela = todas[j][0] - todas[i][0]
                    melhor = janela if melhor is None else min(melhor, janela)
                    break
        if melhor is not None:
            proximidade = 1.0 / (1.0 + melhor / max(1, distintos))
    elif presentes:
        proximidade = 0.3

    densidade = cobertura / (1.0 + math.log(1 + len(tokens_p) / 50))

    bruto = 3.2 * cobertura + 1.4 * proximidade + 1.0 * densidade
    # Centra em zero para que a sigmoide dê algo próximo de 0,5 no caso médio.
    return 6.0 * bruto / 5.6 - 3.0


def sigmoide(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)
