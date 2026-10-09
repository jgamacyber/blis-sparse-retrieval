"""
Métricas de recuperação, e qual usar em cada estágio.

As quatro usuais, com a definição e a razão de existir.

**Hit@k**: apareceu algum relevante no topo `k`? A mais fácil de explicar e a
mais grosseira: não distingue o relevante em primeiro do relevante em décimo.

**Recall@k**: que fração dos relevantes apareceu no topo `k`? É a métrica certa
para o **primeiro estágio** de um pipeline de dois, porque ali o que importa é
não perder o relevante. Ordenar bem é trabalho do reordenador.

**MRR**: média de 1 / posição do primeiro relevante. Pesa muito o topo: sair da
segunda para a primeira posição vale 0,5; da décima para a nona, 0,011.

**nDCG@k**: ganho acumulado com desconto logarítmico, dividido pelo ideal. A
mais completa e a menos intuitiva.

A escolha errada de métrica é um erro comum e silencioso: otimizar MRR no
primeiro estágio é otimizar a coisa errada, porque o documento que não entrar
no conjunto de candidatos está perdido de qualquer forma.

Nota de implementação: a relevância é anotada por **seção**, no formato
`documento#seção`, e não por documento nem por trecho.

Por documento seria fácil demais: com treze documentos, qualquer recuperador
acerta, todas as configurações empatam perto do teto e a tabela não distingue
nada. Foi o primeiro desenho deste conjunto, e ele foi descartado justamente
por isso.

Por trecho seria frágil: a anotação quebraria a cada mudança de TAMANHO_CHUNK.

Por seção fica no meio: é a granularidade que um RAG precisa acertar, porque é
o trecho que vai para o prompt, e sobrevive a mudanças de chunking enquanto os
títulos de seção existirem. Quando uma seção longa gera vários trechos, acertar
qualquer um deles conta, o que é o comportamento desejado.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def alvos_recuperados(cids: list[str], mapa: dict[str, str]) -> list[str]:
    """
    Alvos na ordem de primeira aparição, sem repetir.

    `mapa` leva o identificador do trecho ao alvo anotado, que aqui é
    `documento#seção`. Remover repetição importa: duas janelas da mesma seção
    não são dois acertos, e contá-las duas vezes infla Recall@k sem que o
    sistema tenha encontrado nada novo.
    """
    vistos: list[str] = []
    for cid in cids:
        alvo = mapa.get(cid)
        if alvo and alvo not in vistos:
            vistos.append(alvo)
    return vistos


# Nome antigo, mantido para não quebrar quem já escreveu código contra ele.
docs_recuperados = alvos_recuperados


def hit_at_k(recuperados: list[str], relevantes: set[str], k: int) -> float:
    return 1.0 if set(recuperados[:k]) & relevantes else 0.0


def recall_at_k(recuperados: list[str], relevantes: set[str], k: int) -> float:
    if not relevantes:
        return 1.0
    return len(set(recuperados[:k]) & relevantes) / len(relevantes)


def precision_at_k(recuperados: list[str], relevantes: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(recuperados[:k]) & relevantes) / k


def mrr(recuperados: list[str], relevantes: set[str]) -> float:
    for i, doc in enumerate(recuperados, start=1):
        if doc in relevantes:
            return 1.0 / i
    return 0.0


def ndcg_at_k(recuperados: list[str], relevantes: set[str], k: int) -> float:
    """
    nDCG com ganho binário e desconto log2(i + 1).

    O ideal é a lista que colocaria todos os relevantes nas primeiras posições,
    limitado a `k`. Dividir por ele é o que torna o valor comparável entre
    consultas com números diferentes de relevantes.
    """
    if not relevantes:
        return 1.0

    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, doc in enumerate(recuperados[:k], start=1)
        if doc in relevantes
    )
    ideal = sum(
        1.0 / math.log2(i + 1)
        for i in range(1, min(len(relevantes), k) + 1)
    )
    return dcg / ideal if ideal else 0.0


# --------------------------------------------------------------------------- #
# Agregação
# --------------------------------------------------------------------------- #


@dataclass
class Medidas:
    hit: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    n: int = 0

    def como_dict(self) -> dict:
        return {
            "hit": self.hit, "recall": self.recall, "precision": self.precision,
            "mrr": self.mrr, "ndcg": self.ndcg, "n": self.n,
        }

    def linha(self, rotulo: str, largura: int = 26) -> str:
        return (f"  {rotulo:<{largura}} {self.hit:>7.3f} {self.recall:>9.3f} "
                f"{self.mrr:>7.3f} {self.ndcg:>8.3f}  {self.n:>3}")

    @staticmethod
    def cabecalho(largura: int = 26) -> str:
        return (f"  {'configuração':<{largura}} {'Hit@k':>7} {'Recall@k':>9} "
                f"{'MRR':>7} {'nDCG@k':>8}  {'n':>3}\n"
                f"  {'-' * (largura + 38)}")


def avaliar_uma(recuperados: list[str], relevantes: set[str], k: int) -> Medidas:
    return Medidas(
        hit=hit_at_k(recuperados, relevantes, k),
        recall=recall_at_k(recuperados, relevantes, k),
        precision=precision_at_k(recuperados, relevantes, k),
        mrr=mrr(recuperados, relevantes),
        ndcg=ndcg_at_k(recuperados, relevantes, k),
        n=1,
    )


def media(medidas: list[Medidas]) -> Medidas:
    if not medidas:
        return Medidas()
    n = len(medidas)
    return Medidas(
        hit=sum(m.hit for m in medidas) / n,
        recall=sum(m.recall for m in medidas) / n,
        precision=sum(m.precision for m in medidas) / n,
        mrr=sum(m.mrr for m in medidas) / n,
        ndcg=sum(m.ndcg for m in medidas) / n,
        n=n,
    )
