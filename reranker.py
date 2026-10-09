"""
Reordenação por codificador cruzado, segundo estágio do pipeline.

Modelo de referência: `BAAI/bge-reranker-v2-m3`, construído sobre o BGE-M3.
Ele recebe pares de consulta e passagem e devolve um logit de relevância;
com normalização ligada, uma sigmoide mapeia para o intervalo de zero a um.

Três cuidados que este arquivo trata como parte da implementação, e não como
nota de rodapé.

**O score bruto não é probabilidade e não é comparável entre consultas.** A
própria ficha do modelo mostra valores como -5,65 e 5,26. Serve para ordenar
candidatos de uma mesma consulta. Um limiar fixo do tipo "descartar abaixo de
zero" é decisão que precisa ser calibrada no seu conjunto, e a função
`aplicar_limiar` existe para deixar essa decisão visível em vez de embutida.

**O reordenador não inventa documento que não recebeu.** O teto do sistema é o
recall do primeiro estágio no corte de candidatos. A função
`teto_do_primeiro_estagio` calcula esse teto, porque é o número que explica a
maior parte das surpresas de um pipeline de dois estágios.

**O custo é linear no número de candidatos.** Cada par é uma passada completa,
sem nada pré-calculado. Por isso `reordenar` contabiliza quantos pares avaliou:
é essa contagem, e não o tempo de parede, que diz quanto o estágio custa.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backends import pontuar_par, sigmoide
from config import SETTINGS, Settings
from corpus import Chunk


@dataclass
class Candidato:
    cid: str
    texto: str
    score_recuperacao: float = 0.0
    posicao_recuperacao: int = 0
    score_reordenacao: float = 0.0
    score_normalizado: float = 0.0


@dataclass
class ResultadoReordenacao:
    candidatos: list[Candidato] = field(default_factory=list)
    pares_avaliados: int = 0

    def ordenados(self) -> list[Candidato]:
        return sorted(self.candidatos,
                      key=lambda c: (-c.score_reordenacao, c.cid))

    def cids(self, top_k: int | None = None) -> list[str]:
        ordem = [c.cid for c in self.ordenados()]
        return ordem[:top_k] if top_k else ordem

    def mudou_o_topo(self) -> bool:
        """O reordenador trocou o primeiro colocado do primeiro estágio?"""
        if not self.candidatos:
            return False
        antes = min(self.candidatos, key=lambda c: c.posicao_recuperacao)
        return self.ordenados()[0].cid != antes.cid


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class RerankerLocal:
    """
    Substituto determinístico. **Não é um codificador cruzado.**

    Pontua por cobertura de termos, proximidade entre eles e densidade. A
    proximidade é o único sinal aqui que o produto interno do primeiro estágio
    não tem, e por isso é de onde vem qualquer ganho que este substituto
    produza.

    O que ele não faz, e que é a razão de existir um codificador cruzado de
    verdade: não vê negação, não vê condição e não sabe a que entidade um
    número se refere. Um ganho pequeno aqui não é evidência de que reordenar
    rende pouco; é evidência de que este substituto é raso.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or SETTINGS
        self.nome = "reranker-local"

    def pontuar(self, consulta: str, passagens: list[str]) -> list[float]:
        return [pontuar_par(consulta, p) for p in passagens]


class RerankerHF:
    """O bge-reranker-v2-m3 de verdade, via FlagEmbedding."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or SETTINGS
        self.nome = f"reranker-hf:{self.settings.modelo_reranker}"
        self._modelo = None

    def _carregar(self) -> None:
        if self._modelo is not None:
            return
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as erro:  # pragma: no cover - depende do ambiente
            raise RuntimeError(
                "backend 'hf' precisa de FlagEmbedding. Instale com:\n"
                "  pip install -r requirements-hf.txt\n"
                "Ou rode com BACKEND=local."
            ) from erro

        self._modelo = FlagReranker(
            self.settings.modelo_reranker,
            use_fp16=self.settings.dispositivo != "cpu",
            devices=self.settings.dispositivo,
        )

    def pontuar(self, consulta: str, passagens: list[str]) -> list[float]:
        self._carregar()
        if not passagens:
            return []

        # normalize=False de propósito: o logit bruto é o que o modelo produz,
        # e a sigmoide é aplicada depois, no mesmo ponto nos dois backends.
        scores = self._modelo.compute_score(
            [[consulta, p] for p in passagens], normalize=False
        )
        if isinstance(scores, (int, float)):
            return [float(scores)]
        return [float(s) for s in scores]


def obter_reranker(settings: Settings | None = None):
    settings = settings or SETTINGS
    settings.validar()
    if settings.backend == "hf":
        return RerankerHF(settings)
    return RerankerLocal(settings)


# --------------------------------------------------------------------------- #
# O segundo estágio
# --------------------------------------------------------------------------- #


def reordenar(modelo, consulta: str, candidatos: list[Candidato]) -> ResultadoReordenacao:
    """Pontua todos os candidatos e devolve o resultado com a contagem de pares."""
    if not candidatos:
        return ResultadoReordenacao()

    scores = modelo.pontuar(consulta, [c.texto for c in candidatos])
    for candidato, score in zip(candidatos, scores):
        candidato.score_reordenacao = score
        candidato.score_normalizado = sigmoide(score)

    return ResultadoReordenacao(candidatos=list(candidatos),
                                pares_avaliados=len(candidatos))


def montar_candidatos(resultados, chunks: dict[str, Chunk],
                      n: int) -> list[Candidato]:
    """Converte a saída do primeiro estágio em candidatos para o segundo."""
    return [
        Candidato(
            cid=r.cid,
            texto=chunks[r.cid].para_indexar(),
            score_recuperacao=r.score,
            posicao_recuperacao=r.posicao or (i + 1),
        )
        for i, r in enumerate(resultados[:n])
    ]


def aplicar_limiar(resultado: ResultadoReordenacao,
                   limiar: float = 0.5) -> list[Candidato]:
    """
    Corta pelo score normalizado.

    Existe para deixar a decisão explícita. O limiar certo depende do seu
    conjunto: 0,5 na sigmoide corresponde a logit zero, que não tem significado
    especial no modelo. Herdar esse valor de outro projeto é o erro comum.
    """
    return [c for c in resultado.ordenados() if c.score_normalizado >= limiar]


def teto_do_primeiro_estagio(cids_candidatos: list[str],
                             relevantes: set[str],
                             mapa: dict[str, str]) -> float:
    """
    Fração dos relevantes que chegou ao conjunto de candidatos.

    Esse é o teto do pipeline: o reordenador não recupera o que não recebeu.
    Medir isso separadamente é o que distingue "o reordenador é ruim" de "o
    primeiro estágio não entregou o documento", que pedem consertos opostos.
    """
    if not relevantes:
        return 1.0
    alvos = {mapa[c] for c in cids_candidatos if c in mapa}
    return len(alvos & relevantes) / len(relevantes)
