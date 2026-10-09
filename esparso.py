"""
Vetor esparso e índice invertido: a estrutura comum a tudo aqui.

A observação que organiza este repositório: **BM25, SPLADE e a parcela léxica
do BGE-M3 são o mesmo objeto matemático**. Os três representam cada documento
como um vetor esparso sobre um vocabulário, pontuam por produto interno e
rodam no mesmo índice invertido.

A diferença entre eles é só uma: de onde vem o peso de cada termo.

    BM25        peso vem de uma fórmula fechada, com IDF, saturação e
                normalização por tamanho. Nenhum parâmetro é aprendido.
    SPLADE      peso vem da cabeça de linguagem mascarada de um transformador,
                e inclui termos que o documento não contém.
    M3 léxico   peso vem de uma projeção linear aprendida, restrita aos termos
                que o documento contém.

Enxergar isso muda a pergunta. Não se trata de "substituir o BM25 por um
modelo neural", como se fossem arquiteturas rivais. Trata-se de **trocar a
função de peso** e manter toda a estrutura: o índice, o produto interno, a
travessia de listas de postagem e as décadas de engenharia em cima disso.

É também o que explica por que o SPLADE é interessante e a recuperação densa
é outra coisa: a densa abandona a estrutura, a esparsa aprendida fica nela.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class VetorEsparso:
    """Pesos por termo. Ausente significa zero, não desconhecido."""

    pesos: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.pesos)

    def __contains__(self, termo: str) -> bool:
        return termo in self.pesos

    def get(self, termo: str, padrao: float = 0.0) -> float:
        return self.pesos.get(termo, padrao)

    @property
    def l0(self) -> int:
        """Número de termos ativos. É o que determina o custo no índice."""
        return len(self.pesos)

    @property
    def l1(self) -> float:
        return sum(abs(v) for v in self.pesos.values())

    def produto_interno(self, outro: "VetorEsparso") -> float:
        """
        Itera sobre o vetor menor: numa busca, a consulta.

        Essa escolha não é detalhe. Iterar sobre o documento custaria o número
        de termos do documento por documento; iterar sobre a consulta custa o
        número de termos da consulta, que é o que o índice invertido explora.
        """
        if len(self.pesos) > len(outro.pesos):
            self, outro = outro, self
        return sum(p * outro.pesos.get(t, 0.0) for t, p in self.pesos.items())

    def podar(self, top: int) -> "VetorEsparso":
        """
        Mantém os `top` maiores pesos.

        No artigo do SPLADE a esparsidade vem do regularizador durante o
        treino, não de poda depois. A poda aqui é o controle equivalente para
        quem não vai treinar nada: ela limita o comprimento das listas de
        postagem, que é o recurso que a esparsidade existe para proteger.
        """
        if top <= 0 or len(self.pesos) <= top:
            return VetorEsparso(dict(self.pesos))
        maiores = sorted(self.pesos.items(), key=lambda kv: kv[1], reverse=True)[:top]
        return VetorEsparso(dict(maiores))

    def normalizar_l2(self) -> "VetorEsparso":
        norma = sum(v * v for v in self.pesos.values()) ** 0.5
        if norma == 0:
            return VetorEsparso(dict(self.pesos))
        return VetorEsparso({t: v / norma for t, v in self.pesos.items()})

    def top(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.pesos.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def formatar(self, n: int = 10) -> str:
        partes = [f"{t}={p:.3f}" for t, p in self.top(n)]
        resto = f" (+{len(self.pesos) - n})" if len(self.pesos) > n else ""
        return ", ".join(partes) + resto


# --------------------------------------------------------------------------- #
# Índice invertido
# --------------------------------------------------------------------------- #


@dataclass
class Resultado:
    cid: str
    score: float
    posicao: int = 0


class IndiceInvertido:
    """
    Termo para lista de postagem. A busca só toca os termos da consulta.

    Guarda também o suficiente para explicar o custo: comprimento das listas,
    termos ativos por documento e uma estimativa do mesmo custo que o
    regularizador FLOPS do SPLADE tenta minimizar durante o treino.
    """

    def __init__(self) -> None:
        self.postagens: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self.vetores: dict[str, VetorEsparso] = {}
        self.ordem: list[str] = []

    def indexar(self, cid: str, vetor: VetorEsparso) -> None:
        if cid in self.vetores:
            raise ValueError(f"documento já indexado: {cid}")
        self.vetores[cid] = vetor
        self.ordem.append(cid)
        for termo, peso in vetor.pesos.items():
            if peso != 0.0:
                self.postagens[termo].append((cid, peso))

    def indexar_todos(self, vetores: dict[str, VetorEsparso]) -> None:
        for cid, vetor in vetores.items():
            self.indexar(cid, vetor)

    def buscar(self, consulta: VetorEsparso, top_k: int = 10) -> list[Resultado]:
        """
        Acumula score percorrendo só as listas dos termos da consulta.

        Documento que não compartilha nenhum termo com a consulta nunca é
        tocado, e é por isso que essa busca não cresce com o tamanho do corpus,
        e sim com o comprimento das listas dos termos pedidos.
        """
        acumulado: dict[str, float] = defaultdict(float)

        for termo, peso_q in consulta.pesos.items():
            if peso_q == 0.0:
                continue
            for cid, peso_d in self.postagens.get(termo, ()):
                acumulado[cid] += peso_q * peso_d

        ordenado = sorted(acumulado.items(), key=lambda kv: (-kv[1], kv[0]))
        return [
            Resultado(cid=cid, score=score, posicao=i)
            for i, (cid, score) in enumerate(ordenado[:top_k], start=1)
        ]

    def explicar(self, consulta: VetorEsparso, cid: str,
                 n: int = 8) -> list[tuple[str, float]]:
        """
        Contribuição de cada termo ao score deste documento.

        Interpretabilidade é uma das propriedades que a recuperação léxica tem
        e a densa não, e ela não sobrevive sozinha: precisa ser exposta. Esta
        função é a que permite responder "por que este documento veio".
        """
        vetor = self.vetores.get(cid)
        if vetor is None:
            return []
        contribuicoes = [
            (t, p * vetor.get(t)) for t, p in consulta.pesos.items() if t in vetor
        ]
        return sorted(contribuicoes, key=lambda kv: kv[1], reverse=True)[:n]

    # ------------------------------ custo --------------------------------- #

    def estatisticas(self) -> dict:
        if not self.vetores:
            return {}

        comprimentos = [len(p) for p in self.postagens.values()]
        ativos = [v.l0 for v in self.vetores.values()]
        n = len(self.vetores)

        # Mesma forma do regularizador FLOPS do SPLADE: soma do quadrado da
        # ativação média por termo. Aqui é medida, não minimizada: serve para
        # comparar o custo de dois esquemas de peso no mesmo corpus.
        flops = 0.0
        for termo, postagem in self.postagens.items():
            media = sum(p for _, p in postagem) / n
            flops += media * media

        return {
            "documentos": n,
            "termos_no_vocabulario": len(self.postagens),
            "termos_ativos_por_doc": sum(ativos) / n,
            "termos_ativos_max": max(ativos),
            "postagem_media": sum(comprimentos) / len(comprimentos),
            "postagem_max": max(comprimentos),
            "postagens_totais": sum(comprimentos),
            "flops_estimado": flops,
        }

    def termos_mais_frequentes(self, n: int = 10) -> list[tuple[str, int]]:
        """Os termos com lista de postagem mais longa, que são o gargalo."""
        contagens = [(t, len(p)) for t, p in self.postagens.items()]
        return sorted(contagens, key=lambda kv: kv[1], reverse=True)[:n]

    def __len__(self) -> int:
        return len(self.vetores)
