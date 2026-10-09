"""
BM25, escrito como um produtor de vetores esparsos.

A implementação usual do BM25 calcula o score dentro de um laço, na hora da
consulta. Aqui ele é escrito de outro jeito, equivalente e mais revelador:
todo o peso que depende apenas do documento é **pré-calculado** em um vetor
esparso, e a consulta vira um vetor binário sobre os seus termos.

    w(t, d) = IDF(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1 - b + b·|d|/avgdl) )

    score(q, d) = Σ  1 · w(t, d)        = ⟨vetor_consulta, vetor_documento⟩
                 t∈q

Isso é exatamente o BM25, sem aproximação: cada fator da fórmula só depende do
termo e do documento, nunca da consulta. Escrevê-lo assim deixa à vista que o
BM25 **já é** um modelo de recuperação esparsa, e que a diferença entre ele e o
SPLADE é a função que produz o peso, não a arquitetura de busca.

Uma nota sobre a frequência do termo na consulta: a formulação completa tem um
terceiro parâmetro, `k3`, que satura a frequência dentro da consulta. Como
consulta repetir termo é raro e o efeito é desprezível em consultas curtas, as
implementações correntes o omitem, e esta também. Fica registrado para que a
omissão seja escolha e não esquecimento.

Referência: Robertson, S. e Zaragoza, H. (2009). *The Probabilistic Relevance
Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval.
"""

from __future__ import annotations

import math
from collections import Counter

from corpus import Chunk, tokenizar
from esparso import VetorEsparso


def idf_robertson(n_docs: int, df: int) -> float:
    """
    IDF da família Robertson, com o 0,5 de suavização.

        idf(t) = ln( (N - df + 0.5) / (df + 0.5) + 1 )

    O `+1` dentro do logaritmo é o que mantém o valor não negativo. Sem ele, um
    termo presente em mais da metade do corpus receberia peso negativo, e
    conter o termo pedido pioraria o score do documento, o que é absurdo como
    comportamento de busca. Várias implementações antigas tinham esse defeito.
    """
    return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


class BM25:
    """Índice BM25 que expõe o documento como vetor esparso."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks = {c.cid: c for c in chunks}

        self.frequencias: dict[str, Counter] = {}
        self.comprimentos: dict[str, int] = {}

        for chunk in chunks:
            tokens = tokenizar(chunk.para_indexar())
            self.frequencias[chunk.cid] = Counter(tokens)
            self.comprimentos[chunk.cid] = len(tokens)

        n = len(chunks)
        self.n_docs = n
        self.avgdl = (sum(self.comprimentos.values()) / n) if n else 1.0

        df: Counter = Counter()
        for contagem in self.frequencias.values():
            df.update(contagem.keys())
        self.df = df
        self.idf = {t: idf_robertson(n, k) for t, k in df.items()}

    # ------------------------------ vetores -------------------------------- #

    def vetor_documento(self, cid: str) -> VetorEsparso:
        """Pesos BM25 do documento, independentes de qualquer consulta."""
        contagem = self.frequencias[cid]
        comprimento = self.comprimentos[cid]
        norma = self.k1 * (1 - self.b + self.b * comprimento / self.avgdl)

        pesos = {}
        for termo, f in contagem.items():
            pesos[termo] = self.idf[termo] * (f * (self.k1 + 1)) / (f + norma)
        return VetorEsparso(pesos)

    def vetores_documentos(self) -> dict[str, VetorEsparso]:
        return {cid: self.vetor_documento(cid) for cid in self.chunks}

    def vetor_consulta(self, consulta: str) -> VetorEsparso:
        """
        Vetor binário sobre os termos da consulta.

        Termos fora do vocabulário do corpus são descartados aqui, porque a
        lista de postagem deles é vazia e eles não contribuiriam de todo modo.
        """
        termos = set(tokenizar(consulta))
        return VetorEsparso({t: 1.0 for t in termos if t in self.idf})

    # --------------------------- caminho direto ---------------------------- #

    def pontuar(self, consulta: str, cid: str) -> float:
        """Score pela fórmula, sem passar pelo índice. Serve de conferência."""
        contagem = self.frequencias[cid]
        comprimento = self.comprimentos[cid]
        norma = self.k1 * (1 - self.b + self.b * comprimento / self.avgdl)

        score = 0.0
        for termo in set(tokenizar(consulta)):
            f = contagem.get(termo, 0)
            if f:
                score += self.idf[termo] * (f * (self.k1 + 1)) / (f + norma)
        return score

    def cobertura(self, consulta: str) -> float:
        """Fração dos termos da consulta que existe no vocabulário do corpus."""
        termos = set(tokenizar(consulta))
        if not termos:
            return 0.0
        return sum(1 for t in termos if t in self.idf) / len(termos)
