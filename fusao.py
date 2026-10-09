"""
Fusão de listas: posto contra score.

Duas famílias, e a escolha entre elas não é detalhe de implementação.

**Fusão recíproca de postos.** Descarta o score e usa só a posição:

    RRF(d) = Σ  1 / (k + rank_s(d))
             s

Referência: Cormack, G., Clarke, C. e Buettcher, S. (2009). *Reciprocal Rank
Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR.

A vantagem é dispensar calibração: BM25 devolve algo entre zero e trinta, um
cosseno algo entre menos um e um, e um codificador cruzado um logit sem limite.
Somar isso seria somar unidades diferentes.

O preço é jogar a magnitude fora. Se um sistema está muito confiante no
primeiro lugar e nada confiante do terceiro para baixo, a fusão recíproca não
tem como saber: para ela a distância entre o primeiro e o terceiro é sempre a
mesma.

**Combinação convexa.** Normaliza os scores e soma com pesos:

    conv(d) = Σ  w_s · norm(score_s(d))
              s

Preserva a magnitude e exige escolher a normalização e os pesos.

O benchmark de documentos com tabelas mediu as duas no mesmo conjunto e
encontrou a combinação convexa com pesos iguais em 0,726 contra 0,695 da fusão
recíproca com `k = 60`, e 0,716 com `k = 10`. Ou seja: o `k = 60` é o valor de
um artigo de 2009, não uma constante da natureza, e descartar a magnitude não
é de graça. As duas estão implementadas aqui para que a comparação seja um
comando e não uma opinião.
"""

from __future__ import annotations

from dataclasses import dataclass

from esparso import Resultado

NORMALIZACOES = ("minmax", "zscore", "soma", "nenhuma")


@dataclass
class Lista:
    """Uma lista ordenada de um sistema, com nome para aparecer no relatório."""

    nome: str
    resultados: list[Resultado]

    def postos(self) -> dict[str, int]:
        return {r.cid: i for i, r in enumerate(self.resultados, start=1)}

    def scores(self) -> dict[str, float]:
        return {r.cid: r.score for r in self.resultados}


# --------------------------------------------------------------------------- #
# Fusão recíproca de postos
# --------------------------------------------------------------------------- #


def rrf(listas: list[Lista], k: int = 60,
        top_k: int | None = None) -> list[Resultado]:
    """
    RRF(d) = Σ_s 1 / (k + rank_s(d)).

    Documento ausente de uma lista simplesmente não contribui por ela, o que é
    o comportamento do artigo: não existe penalidade por ausência, apenas falta
    de contribuição.
    """
    acumulado: dict[str, float] = {}

    for lista in listas:
        for posicao, resultado in enumerate(lista.resultados, start=1):
            acumulado[resultado.cid] = acumulado.get(resultado.cid, 0.0) + 1.0 / (
                k + posicao
            )

    ordenado = sorted(acumulado.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k:
        ordenado = ordenado[:top_k]
    return [
        Resultado(cid=cid, score=score, posicao=i)
        for i, (cid, score) in enumerate(ordenado, start=1)
    ]


# --------------------------------------------------------------------------- #
# Combinação convexa
# --------------------------------------------------------------------------- #


def normalizar(scores: dict[str, float], modo: str = "minmax") -> dict[str, float]:
    """
    Põe os scores em escala comparável.

    `minmax` é o mais usado e o mais frágil: um único valor extremo comprime
    todo o resto para perto de zero. `zscore` é robusto a isso e pode devolver
    valores negativos, o que muda o significado de somar com peso. `soma`
    transforma em distribuição e é adequada quando os scores já são positivos.

    Não existe escolha neutra aqui, e é por isso que a fusão recíproca de
    postos é popular: ela não obriga ninguém a escolher.
    """
    if modo not in NORMALIZACOES:
        raise ValueError(f"normalização inválida: '{modo}'. Use {NORMALIZACOES}.")
    if not scores or modo == "nenhuma":
        return dict(scores)

    valores = list(scores.values())

    if modo == "minmax":
        menor, maior = min(valores), max(valores)
        if maior == menor:
            return {c: 1.0 for c in scores}
        return {c: (v - menor) / (maior - menor) for c, v in scores.items()}

    if modo == "zscore":
        n = len(valores)
        media = sum(valores) / n
        variancia = sum((v - media) ** 2 for v in valores) / n
        desvio = variancia ** 0.5
        if desvio == 0:
            return {c: 0.0 for c in scores}
        return {c: (v - media) / desvio for c, v in scores.items()}

    total = sum(valores)
    if total == 0:
        return {c: 0.0 for c in scores}
    return {c: v / total for c, v in scores.items()}


def combinacao_convexa(listas: list[Lista], pesos: list[float] | None = None,
                       normalizacao: str = "minmax",
                       top_k: int | None = None) -> list[Resultado]:
    """
    conv(d) = Σ_s w_s · norm(score_s(d)).

    Documento ausente de uma lista entra com zero naquela parcela. Essa decisão
    merece atenção: com normalização `zscore`, zero não é o valor mínimo, é a
    média, então um documento ausente recebe tratamento de documento mediano.
    Para evitar essa armadilha silenciosa, `zscore` desloca a escala para que o
    mínimo observado seja zero antes de somar.
    """
    if pesos is None:
        pesos = [1.0] * len(listas)
    if len(pesos) != len(listas):
        raise ValueError("um peso por lista")

    acumulado: dict[str, float] = {}

    for lista, peso in zip(listas, pesos):
        normalizados = normalizar(lista.scores(), normalizacao)
        if normalizacao == "zscore" and normalizados:
            deslocamento = min(normalizados.values())
            normalizados = {c: v - deslocamento for c, v in normalizados.items()}
        for cid, valor in normalizados.items():
            acumulado[cid] = acumulado.get(cid, 0.0) + peso * valor

    ordenado = sorted(acumulado.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k:
        ordenado = ordenado[:top_k]
    return [
        Resultado(cid=cid, score=score, posicao=i)
        for i, (cid, score) in enumerate(ordenado, start=1)
    ]
