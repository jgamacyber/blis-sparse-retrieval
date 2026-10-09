"""
O sistema de recuperação: indexa uma vez, responde por qualquer configuração.

Cada configuração aqui é uma linha da tabela de ablação, e a razão de todas
viverem no mesmo objeto é que elas precisam compartilhar exatamente o mesmo
corpus, o mesmo chunking e a mesma tokenização. Comparar recuperadores
indexados de formas diferentes mede a diferença de indexação, não de modelo, e
é um erro fácil de cometer sem perceber.

Três notas de custo, para que as linhas da tabela não pareçam equivalentes:

- **BM25 e SPLADE** rodam em índice invertido: a busca toca só as listas de
  postagem dos termos da consulta.
- **A parcela densa e a multivetor** não têm índice aqui: elas varrem o corpus
  inteiro. Com 68 trechos isso é irrelevante; com milhões, exigiria um índice
  de vizinhos aproximados, e a parcela multivetor exigiria muito mais memória.
  Por isso o padrão é usar o multivetor apenas para reordenar candidatos, que é
  o uso recomendado no próprio artigo do BGE-M3.
- **O reordenador** custa uma passada por candidato, sem nada pré-calculado.
"""

from __future__ import annotations

from dataclasses import dataclass

from bm25 import BM25
from config import SETTINGS, Settings
from corpus import Chunk
from esparso import IndiceInvertido, Resultado
from fusao import Lista, combinacao_convexa, rrf
from m3 import (
    M3Local,
    RepresentacaoM3,
    obter_m3,
    score_denso,
    score_integrado,
    score_lexical,
    score_multivetor,
)
from reranker import Candidato, montar_candidatos, obter_reranker, reordenar
from splade import flops, obter_splade

# Configurações de primeiro estágio disponíveis.
RECUPERADORES = (
    "bm25",
    "splade",
    "splade-doc",
    "m3-denso",
    "m3-lexical",
    "m3-integrado",
    "hibrido-rrf",
    "hibrido-convexo",
    "splade-denso-rrf",
)


@dataclass
class Custo:
    """O que uma configuração gasta, separado do que ela acerta."""

    termos_ativos_por_doc: float = 0.0
    postagens_totais: int = 0
    postagem_max: int = 0
    flops_estimado: float = 0.0
    varre_corpus: bool = False
    pares_reordenados: int = 0


class Sistema:
    def __init__(self, chunks: list[Chunk], settings: Settings | None = None,
                 agregacao_splade: str = "max",
                 com_multivetor: bool = True) -> None:
        self.settings = settings or SETTINGS
        self.settings.validar()
        self.chunks = {c.cid: c for c in chunks}
        self.lista_chunks = list(chunks)
        self.mapa_doc = {c.cid: c.doc for c in chunks}
        # A relevância é anotada por seção; este é o mapa usado na avaliação.
        self.mapa_fonte = {c.cid: c.fonte for c in chunks}
        self.com_multivetor = com_multivetor

        # --- BM25 ---
        self.bm25 = BM25(chunks, self.settings.bm25_k1, self.settings.bm25_b)
        self.indice_bm25 = IndiceInvertido()
        self.indice_bm25.indexar_todos(self.bm25.vetores_documentos())

        # --- SPLADE ---
        self.splade = obter_splade(chunks, self.settings, agregacao_splade)
        vetores_splade = {
            c.cid: self.splade.codificar_documento(c) for c in chunks
        }
        self.indice_splade = IndiceInvertido()
        self.indice_splade.indexar_todos(vetores_splade)
        vetores_bm25 = self.bm25.vetores_documentos()
        self.flops_splade = flops(list(vetores_splade.values()))
        self.flops_bm25 = flops(list(vetores_bm25.values()))
        # Normalizado é o único dos dois que compara estrutura entre esquemas
        # de peso com escalas diferentes. Ver a docstring de `flops`.
        self.flops_splade_norm = flops(list(vetores_splade.values()), True)
        self.flops_bm25_norm = flops(list(vetores_bm25.values()), True)

        # --- BGE-M3 ---
        self.m3 = obter_m3(chunks, self.settings)
        self.reps_m3: dict[str, RepresentacaoM3] = {
            c.cid: self.m3.codificar_documento(c, multivetor=com_multivetor)
            for c in chunks
        }

        # --- reordenador ---
        self.reranker = obter_reranker(self.settings)

    # --------------------------- primeiro estágio -------------------------- #

    def buscar_bm25(self, consulta: str, top_k: int = 10) -> list[Resultado]:
        return self.indice_bm25.buscar(self.bm25.vetor_consulta(consulta), top_k)

    def buscar_splade(self, consulta: str, top_k: int = 10,
                      expandir_consulta: bool = True) -> list[Resultado]:
        vetor = self.splade.codificar_consulta(consulta, expandir=expandir_consulta)
        return self.indice_splade.buscar(vetor, top_k)

    def _buscar_m3(self, consulta: str, top_k: int, funcao: str) -> list[Resultado]:
        """
        Varredura completa com uma das funções do M3.

        Sem índice de propósito: o ponto do módulo é comparar funções de score,
        e introduzir busca aproximada aqui misturaria o erro da aproximação com
        a diferença entre as funções.
        """
        usar_multivetor = funcao in ("multivetor", "integrado") and self.com_multivetor
        rep_q = self.m3.codificar(
            consulta,
            denso=funcao in ("denso", "integrado"),
            lexical=funcao in ("lexical", "integrado"),
            multivetor=usar_multivetor,
        )

        scores: list[tuple[str, float]] = []
        for cid, rep_d in self.reps_m3.items():
            if funcao == "denso":
                score = score_denso(rep_q, rep_d)
            elif funcao == "lexical":
                score = score_lexical(rep_q, rep_d)
            elif funcao == "multivetor":
                score = score_multivetor(rep_q, rep_d)
            else:
                score = score_integrado(
                    rep_q, rep_d,
                    self.settings.m3_peso_denso,
                    self.settings.m3_peso_lexical,
                    self.settings.m3_peso_multivetor if usar_multivetor else 0.0,
                )
            if score != 0.0:
                scores.append((cid, score))

        scores.sort(key=lambda kv: (-kv[1], kv[0]))
        return [
            Resultado(cid=cid, score=s, posicao=i)
            for i, (cid, s) in enumerate(scores[:top_k], start=1)
        ]

    def buscar_m3(self, consulta: str, top_k: int = 10,
                  funcao: str = "integrado") -> list[Resultado]:
        return self._buscar_m3(consulta, top_k, funcao)

    def buscar_hibrido(self, consulta: str, top_k: int = 10, modo: str = "rrf",
                       primeiro: str = "bm25", segundo: str = "m3-denso",
                       k_rrf: int | None = None, pesos: list[float] | None = None,
                       normalizacao: str = "minmax",
                       profundidade: int = 50) -> list[Resultado]:
        """
        Funde duas listas.

        `profundidade` é quantos resultados cada sistema contribui para a fusão.
        Fundir só o top 5 de cada um limita a fusão a reordenar o que os dois já
        concordavam; é preciso dar profundidade para que a fusão possa resgatar
        um documento que só um dos dois encontrou, que é a razão de fundir.
        """
        listas = [
            Lista(primeiro, self._buscar_por_nome(primeiro, consulta, profundidade)),
            Lista(segundo, self._buscar_por_nome(segundo, consulta, profundidade)),
        ]

        if modo == "rrf":
            return rrf(listas, k_rrf or self.settings.rrf_k, top_k)
        return combinacao_convexa(listas, pesos, normalizacao, top_k)

    def _buscar_por_nome(self, nome: str, consulta: str,
                         top_k: int) -> list[Resultado]:
        if nome == "bm25":
            return self.buscar_bm25(consulta, top_k)
        if nome == "splade":
            return self.buscar_splade(consulta, top_k)
        if nome == "splade-doc":
            return self.buscar_splade(consulta, top_k, expandir_consulta=False)
        if nome.startswith("m3-"):
            return self.buscar_m3(consulta, top_k, nome[3:])
        raise ValueError(f"recuperador desconhecido: '{nome}'")

    def buscar(self, configuracao: str, consulta: str,
               top_k: int = 10, **kwargs) -> list[Resultado]:
        """Ponto único de entrada, para que a ablação seja uma lista de nomes."""
        if configuracao == "hibrido-rrf":
            return self.buscar_hibrido(consulta, top_k, "rrf", **kwargs)
        if configuracao == "hibrido-convexo":
            return self.buscar_hibrido(consulta, top_k, "convexo", **kwargs)
        if configuracao == "splade-denso-rrf":
            return self.buscar_hibrido(
                consulta, top_k, "rrf", primeiro="splade", segundo="m3-denso", **kwargs
            )
        return self._buscar_por_nome(configuracao, consulta, top_k)

    # --------------------------- segundo estágio --------------------------- #

    def reordenar(self, consulta: str, resultados: list[Resultado],
                  n_candidatos: int | None = None, top_k: int = 10):
        n = n_candidatos or self.settings.n_candidatos
        candidatos: list[Candidato] = montar_candidatos(resultados, self.chunks, n)
        return reordenar(self.reranker, consulta, candidatos)

    def pipeline(self, consulta: str, configuracao: str = "hibrido-rrf",
                 n_candidatos: int | None = None, top_k: int = 5):
        """Primeiro estágio com profundidade de candidatos, depois reordenação."""
        n = n_candidatos or self.settings.n_candidatos
        primeiros = self.buscar(configuracao, consulta, top_k=n)
        reordenado = self.reordenar(consulta, primeiros, n, top_k)
        return primeiros, reordenado

    # ------------------------------- custo --------------------------------- #

    def custo(self, configuracao: str) -> Custo:
        if configuracao == "bm25":
            st = self.indice_bm25.estatisticas()
            return Custo(st["termos_ativos_por_doc"], st["postagens_totais"],
                         st["postagem_max"], self.flops_bm25, False)
        if configuracao in ("splade", "splade-doc"):
            st = self.indice_splade.estatisticas()
            return Custo(st["termos_ativos_por_doc"], st["postagens_totais"],
                         st["postagem_max"], self.flops_splade, False)
        if configuracao.startswith("m3-"):
            return Custo(varre_corpus=True)
        return Custo(varre_corpus=True)

    def comparar_indices(self) -> dict:
        """
        O preço da expansão, em estrutura de índice.

        É o número que falta em quase toda comparação de recuperadores: o
        SPLADE acerta mais e também ativa mais termos por documento, o que
        alonga as listas de postagem. Sem esta tabela, a comparação mede
        qualidade e esconde custo.
        """
        bm25 = self.indice_bm25.estatisticas()
        splade = self.indice_splade.estatisticas()
        return {
            "bm25": {**bm25, "flops": self.flops_bm25,
                     "flops_normalizado": self.flops_bm25_norm},
            "splade": {**splade, "flops": self.flops_splade,
                       "flops_normalizado": self.flops_splade_norm},
            "razao_termos_ativos": (
                splade["termos_ativos_por_doc"] / bm25["termos_ativos_por_doc"]
                if bm25["termos_ativos_por_doc"] else 0.0
            ),
            "razao_postagens": (
                splade["postagens_totais"] / bm25["postagens_totais"]
                if bm25["postagens_totais"] else 0.0
            ),
        }
