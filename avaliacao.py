"""
Ablação e relatório.

O relatório é organizado em torno de uma recusa: **não existe a linha "o melhor
recuperador"**. Toda tabela aqui é quebrada por tipo de consulta, porque a
média sobre o conjunto inteiro esconde comportamento oposto entre os tipos, e
nesse caso esconder a diferença é esconder o resultado.

O conjunto tem duas classes anotadas:

- `exato`: a consulta usa o termo literal e raro que está no trecho relevante.
- `parafrase`: a consulta descreve a mesma coisa com outras palavras.

A divisão não é declarada, é medida: `cobertura_lexical` calcula que fração dos
termos da consulta aparece no trecho relevante, e a diferença entre as duas
classes aparece no comando `python main.py consultas`. Sem essa medida, chamar
uma consulta de "paráfrase" seria afirmação sobre a intenção de quem escreveu,
não sobre o dado.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from config import AVISO_LOCAL, SETTINGS, Settings
from corpus import Chunk, Consulta
from metricas import Medidas, alvos_recuperados, avaliar_uma, media
from reranker import teto_do_primeiro_estagio
from sistema import Sistema

# As configurações que entram na tabela principal, em ordem de leitura.
CONFIGURACOES = [
    ("bm25", "BM25 (linha de base)"),
    ("splade", "SPLADE (expande os dois)"),
    ("splade-doc", "SPLADE-doc (expande só o doc)"),
    ("m3-denso", "BGE-M3 denso"),
    ("m3-lexical", "BGE-M3 léxico"),
    ("m3-integrado", "BGE-M3 integrado"),
    ("hibrido-rrf", "híbrido BM25+denso, RRF"),
    ("hibrido-convexo", "híbrido BM25+denso, convexo"),
    ("splade-denso-rrf", "híbrido SPLADE+denso, RRF"),
]


@dataclass
class ResultadoConsulta:
    consulta_id: str
    tipo: str
    medidas: Medidas
    recuperados: list[str] = field(default_factory=list)
    relevantes: list[str] = field(default_factory=list)
    teto_candidatos: float = 1.0


@dataclass
class Avaliacao:
    """Uma configuração avaliada no conjunto inteiro."""

    nome: str
    rotulo: str
    resultados: list[ResultadoConsulta] = field(default_factory=list)
    pares_reordenados: int = 0

    def por_tipo(self, tipo: str | None = None) -> Medidas:
        escolhidos = [
            r.medidas for r in self.resultados
            if tipo is None or r.tipo == tipo
        ]
        return media(escolhidos)

    @property
    def tipos(self) -> list[str]:
        vistos: list[str] = []
        for r in self.resultados:
            if r.tipo not in vistos:
                vistos.append(r.tipo)
        return vistos

    def piores(self, n: int = 3) -> list[ResultadoConsulta]:
        return sorted(self.resultados, key=lambda r: r.medidas.recall)[:n]


def avaliar_configuracao(sistema: Sistema, consultas: list[Consulta],
                         configuracao: str, rotulo: str, top_k: int = 5,
                         profundidade: int = 20) -> Avaliacao:
    avaliacao = Avaliacao(nome=configuracao, rotulo=rotulo)

    for consulta in consultas:
        resultados = sistema.buscar(configuracao, consulta.consulta,
                                    top_k=profundidade)
        recuperados = alvos_recuperados([r.cid for r in resultados],
                                        sistema.mapa_fonte)
        relevantes = set(consulta.relevantes)

        avaliacao.resultados.append(ResultadoConsulta(
            consulta_id=consulta.id,
            tipo=consulta.tipo,
            medidas=avaliar_uma(recuperados, relevantes, top_k),
            recuperados=recuperados[:top_k],
            relevantes=sorted(relevantes),
        ))

    return avaliacao


def avaliar_com_reordenacao(sistema: Sistema, consultas: list[Consulta],
                            configuracao: str, n_candidatos: int,
                            top_k: int = 5) -> Avaliacao:
    """
    Pipeline de dois estágios.

    Registra também o **teto do primeiro estágio**: a fração dos relevantes que
    chegou ao conjunto de candidatos. Esse número é o que separa "o reordenador
    é ruim" de "o primeiro estágio não entregou o documento", e os dois pedem
    consertos opostos.
    """
    rotulo = f"{configuracao} + rerank (n={n_candidatos})"
    avaliacao = Avaliacao(nome=f"{configuracao}+rerank-{n_candidatos}",
                          rotulo=rotulo)

    for consulta in consultas:
        primeiros = sistema.buscar(configuracao, consulta.consulta,
                                   top_k=n_candidatos)
        relevantes = set(consulta.relevantes)
        teto = teto_do_primeiro_estagio(
            [r.cid for r in primeiros], relevantes, sistema.mapa_fonte
        )

        reordenado = sistema.reordenar(consulta.consulta, primeiros,
                                       n_candidatos, top_k)
        avaliacao.pares_reordenados += reordenado.pares_avaliados

        recuperados = alvos_recuperados(reordenado.cids(), sistema.mapa_fonte)
        avaliacao.resultados.append(ResultadoConsulta(
            consulta_id=consulta.id,
            tipo=consulta.tipo,
            medidas=avaliar_uma(recuperados, relevantes, top_k),
            recuperados=recuperados[:top_k],
            relevantes=sorted(relevantes),
            teto_candidatos=teto,
        ))

    return avaliacao


# --------------------------------------------------------------------------- #
# Relatório
# --------------------------------------------------------------------------- #


@dataclass
class Relatorio:
    avaliacoes: list[Avaliacao] = field(default_factory=list)
    backend: str = "local"
    top_k: int = 5
    comparacao_indices: dict = field(default_factory=dict)
    ablacao_candidatos: list[tuple[int, Medidas, float, int]] = field(
        default_factory=list
    )
    ablacao_fusao: list[tuple[str, Medidas]] = field(default_factory=list)

    # --------------------------- tabela principal -------------------------- #

    def imprimir_principal(self) -> None:
        print(f"\n{'=' * 78}")
        print(f"  TABELA PRINCIPAL — top_k={self.top_k}, backend={self.backend}")
        print("=" * 78)

        if self.backend == "local":
            print(f"\n  {AVISO_LOCAL}")

        tipos = self.avaliacoes[0].tipos if self.avaliacoes else []
        largura = max(len(a.rotulo) for a in self.avaliacoes) + 1

        print(f"\n  Recall@{self.top_k} por tipo de consulta "
              f"(o número que interessa no primeiro estágio)\n")
        cabecalho = f"  {'configuração':<{largura}} {'geral':>8}"
        for tipo in tipos:
            cabecalho += f" {tipo:>11}"
        print(cabecalho)
        print("  " + "-" * (largura + 8 + 12 * len(tipos)))

        for a in self.avaliacoes:
            linha = f"  {a.rotulo:<{largura}} {a.por_tipo().recall:>8.3f}"
            for tipo in tipos:
                linha += f" {a.por_tipo(tipo).recall:>11.3f}"
            print(linha)

        print(f"\n  Todas as métricas, no conjunto inteiro\n")
        print(Medidas.cabecalho(largura))
        for a in self.avaliacoes:
            print(a.por_tipo().linha(a.rotulo, largura))

    def destaque(self) -> str:
        """
        A leitura que a tabela sustenta, construída a partir dos números.

        Esta função não tem texto fixo sobre quem ganhou: ela compara as
        configurações medidas e descreve o que encontrou. Se a ordem mudar com
        outro backend ou outro corpus, o texto muda com ela.
        """
        if not self.avaliacoes:
            return ""

        def pegar(nome: str) -> Avaliacao | None:
            return next((a for a in self.avaliacoes if a.nome == nome), None)

        linhas = ["", "  Leitura dos números acima", "  " + "-" * 60]

        bm25 = pegar("bm25")
        if bm25 is None:
            return "\n".join(linhas)

        exato_bm25 = bm25.por_tipo("exato").recall
        paraf_bm25 = bm25.por_tipo("parafrase").recall

        melhor_exato = max(self.avaliacoes, key=lambda a: a.por_tipo("exato").recall)
        melhor_paraf = max(self.avaliacoes,
                           key=lambda a: a.por_tipo("parafrase").recall)

        linhas.append(
            f"  Em consultas de termo exato, o BM25 fez {exato_bm25:.3f} e o melhor "
            f"foi {melhor_exato.rotulo} com {melhor_exato.por_tipo('exato').recall:.3f}."
        )
        if melhor_exato.por_tipo("exato").recall <= exato_bm25 + 1e-9:
            linhas.append(
                "  Nenhuma configuração superou o BM25 nesse tipo: quando a consulta "
                "traz o termo literal, casamento exato é difícil de bater."
            )

        linhas.append(
            f"  Em consultas de paráfrase, o BM25 caiu para {paraf_bm25:.3f} e o "
            f"melhor foi {melhor_paraf.rotulo} com "
            f"{melhor_paraf.por_tipo('parafrase').recall:.3f}."
        )

        ganho = melhor_paraf.por_tipo("parafrase").recall - paraf_bm25
        perda = exato_bm25 - melhor_paraf.por_tipo("exato").recall
        if ganho > 0 and perda > 0:
            linhas.append(
                f"  Essa configuração ganha {ganho:+.3f} em paráfrase e perde "
                f"{-perda:+.3f} em termo exato. A média esconde as duas coisas: é "
                f"por isso que a tabela é quebrada por tipo."
            )
        elif ganho > 0 and perda <= 0:
            linhas.append(
                f"  Essa configuração ganha {ganho:+.3f} em paráfrase sem perder "
                f"em termo exato, que é o desfecho desejado e não o usual."
            )

        # O reordenador entrou na tabela? Se sim, vale comparar com a entrada
        # dele, porque reordenação que piora é resultado, não erro de código.
        com_rerank = next((a for a in self.avaliacoes if "+rerank" in a.nome), None)
        if com_rerank is not None:
            base_nome = com_rerank.nome.split("+rerank")[0]
            base = pegar(base_nome)
            if base is not None:
                delta = com_rerank.por_tipo().recall - base.por_tipo().recall
                linhas.append(
                    f"  A reordenação sobre {base.rotulo} mudou o Recall@"
                    f"{self.top_k} em {delta:+.3f}."
                )
                if delta < 0:
                    linhas.append(
                        "  Reordenar piorou. Isso é um resultado, não um defeito "
                        "de código: o segundo estágio só ajuda se o seu critério "
                        "de relevância for melhor que a ordem que ele recebeu."
                    )

        return "\n".join(linhas)

    # ------------------------------ índices -------------------------------- #

    def imprimir_indices(self) -> None:
        if not self.comparacao_indices:
            return

        c = self.comparacao_indices
        print(f"\n{'=' * 78}")
        print("  O PREÇO DA EXPANSÃO, NO ÍNDICE")
        print("=" * 78)
        print("\n  O que a expansão aprendida cobra em estrutura de índice, que é o")
        print("  número que falta em quase toda comparação de recuperadores.\n")

        print(f"  {'':<28} {'BM25':>12} {'SPLADE':>12}")
        print("  " + "-" * 54)
        for rotulo, chave in [
            ("termos ativos por trecho", "termos_ativos_por_doc"),
            ("postagens no índice", "postagens_totais"),
            ("maior lista de postagem", "postagem_max"),
            ("FLOPS cru (escalas diferentes)", "flops"),
            ("FLOPS normalizado", "flops_normalizado"),
        ]:
            a, b = c["bm25"][chave], c["splade"][chave]
            fmt = ".2f" if isinstance(a, float) else "d"
            print(f"  {rotulo:<28} {a:>12{fmt}} {b:>12{fmt}}")

        print(f"\n  O SPLADE ativa {c['razao_termos_ativos']:.1f}x mais termos por "
              f"trecho e tem {c['razao_postagens']:.1f}x mais postagens.")
        print("  A busca continua em índice invertido, e continua mais cara. No")
        print("  artigo, essa pressão é controlada pelo regularizador durante o")
        print("  treino; aqui, pela poda em SPLADE_TOP_TERMOS.")
        print("\n  As duas linhas de FLOPS merecem cuidado. A cru depende da escala")
        print("  dos pesos: o IDF do BM25 leva os valores dele a outra ordem de")
        print("  grandeza, então comparar o valor cru mede escala, não estrutura, e")
        print("  chega a inverter a conclusão. A normalizada compara estrutura.")

    # --------------------------- ablação de candidatos --------------------- #

    def imprimir_candidatos(self) -> None:
        if not self.ablacao_candidatos:
            return

        print(f"\n{'=' * 78}")
        print("  TAMANHO DO CONJUNTO DE CANDIDATOS")
        print("=" * 78)
        print("\n  O reordenador não recupera o que não recebeu. O teto é a fração")
        print("  dos relevantes que chegou ao conjunto de candidatos.\n")

        print(f"  {'n':>4} {'Recall@k':>10} {'MRR':>8} {'teto':>8} {'pares':>8}")
        print("  " + "-" * 42)
        for n, m, teto, pares in self.ablacao_candidatos:
            print(f"  {n:>4} {m.recall:>10.3f} {m.mrr:>8.3f} {teto:>8.3f} {pares:>8}")

        print("\n  A coluna `pares` é o custo: uma passada de codificador cruzado")
        print("  por par, sem nada pré-calculado, e ela cresce linearmente com n.")
        print(self.leitura_candidatos())

    def leitura_candidatos(self) -> str:
        """
        Interpreta a ablação a partir dos números medidos, não de texto fixo.

        Há dois desfechos possíveis e eles pedem consertos opostos, então a
        conclusão não pode ser escrita de antemão:

        - teto cresce **e** recall cresce: o primeiro estágio era o gargalo, e
          alargar o conjunto de candidatos foi o conserto certo.
        - teto cresce **e** recall não cresce: o relevante passou a chegar ao
          reordenador e ele não o colocou no topo. O gargalo é o segundo
          estágio, e alargar mais só aumenta a conta.
        """
        if len(self.ablacao_candidatos) < 2:
            return ""

        primeiro, ultimo = self.ablacao_candidatos[0], self.ablacao_candidatos[-1]
        n0, m0, teto0, pares0 = primeiro
        n1, m1, teto1, pares1 = ultimo

        d_teto = teto1 - teto0
        d_recall = m1.recall - m0.recall
        fator_custo = pares1 / pares0 if pares0 else 0.0

        linhas = ["", "  Leitura desta ablação", "  " + "-" * 60]
        linhas.append(
            f"  De n={n0} para n={n1}, o teto subiu {d_teto:+.3f} "
            f"({teto0:.3f} para {teto1:.3f}) e o Recall@{self.top_k} variou "
            f"{d_recall:+.3f}, com {fator_custo:.1f}x mais pares avaliados."
        )

        if d_teto > 0.05 and d_recall > 0.02:
            linhas.append(
                "  O gargalo era o primeiro estágio: alargar o conjunto de "
                "candidatos trouxe o relevante e o reordenador aproveitou."
            )
        elif d_teto > 0.05 and d_recall <= 0.02:
            linhas.append(
                "  O gargalo é o segundo estágio: o relevante passou a chegar ao "
                "reordenador e ele não o colocou no topo. Alargar mais só aumenta "
                "a conta."
            )
            if self.backend == "local":
                linhas.append(
                    "  Era o resultado esperado neste backend: o substituto do "
                    "codificador cruzado pontua por cobertura e proximidade de "
                    "termos, e não vê o que um codificador cruzado vê. Este "
                    "número não é evidência sobre reordenação em geral, é "
                    "evidência sobre o substituto. Rode com BACKEND=hf para medir "
                    "o bge-reranker-v2-m3."
                )
        else:
            linhas.append(
                "  O teto quase não se moveu no intervalo testado: o primeiro "
                "estágio já entregava o que tinha para entregar."
            )

        return "\n".join(linhas)

    # ----------------------------- ablação de fusão ------------------------ #

    def imprimir_fusao(self) -> None:
        if not self.ablacao_fusao:
            return

        print(f"\n{'=' * 78}")
        print("  FUSÃO: POSTO CONTRA SCORE")
        print("=" * 78)
        print()
        largura = max(len(r) for r, _ in self.ablacao_fusao) + 1
        print(Medidas.cabecalho(largura))
        for rotulo, m in self.ablacao_fusao:
            print(m.linha(rotulo, largura))
        print("\n  A fusão recíproca de postos descarta a magnitude do score, e")
        print("  essa troca não é de graça. O k=60 é o valor de um artigo de 2009.")

    # ------------------------------- erros --------------------------------- #

    def imprimir_erros(self, n: int = 3) -> None:
        print(f"\n{'=' * 78}")
        print("  ONDE CADA CONFIGURAÇÃO ERRA")
        print("=" * 78)
        for a in self.avaliacoes:
            piores = [p for p in a.piores(n) if p.medidas.recall < 1.0]
            if not piores:
                print(f"\n  {a.rotulo}: nenhuma consulta com recall abaixo de 1.")
                continue
            print(f"\n  {a.rotulo}")
            for p in piores:
                print(f"    {p.consulta_id} [{p.tipo}] recall {p.medidas.recall:.2f}")
                print(f"      esperado:   {', '.join(p.relevantes)}")
                print(f"      recuperado: {', '.join(p.recuperados[:3]) or 'nada'}")

    # ------------------------------ serialização --------------------------- #

    def como_dict(self) -> dict:
        return {
            "backend": self.backend,
            "top_k": self.top_k,
            "aviso": AVISO_LOCAL if self.backend == "local" else "",
            "comparacao_indices": self.comparacao_indices,
            "configuracoes": [
                {
                    "nome": a.nome,
                    "rotulo": a.rotulo,
                    "geral": a.por_tipo().como_dict(),
                    "por_tipo": {t: a.por_tipo(t).como_dict() for t in a.tipos},
                    "pares_reordenados": a.pares_reordenados,
                    "consultas": [
                        {
                            "id": r.consulta_id, "tipo": r.tipo,
                            "recall": r.medidas.recall, "mrr": r.medidas.mrr,
                            "ndcg": r.medidas.ndcg,
                            "teto_candidatos": r.teto_candidatos,
                            "relevantes": r.relevantes,
                            "recuperados": r.recuperados,
                        }
                        for r in a.resultados
                    ],
                }
                for a in self.avaliacoes
            ],
            "ablacao_candidatos": [
                {"n": n, "recall": m.recall, "mrr": m.mrr, "teto": teto,
                 "pares": pares}
                for n, m, teto, pares in self.ablacao_candidatos
            ],
            "ablacao_fusao": [
                {"configuracao": r, **m.como_dict()} for r, m in self.ablacao_fusao
            ],
        }

    def salvar(self, caminho: str | Path) -> Path:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(
            json.dumps(self.como_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return caminho


# --------------------------------------------------------------------------- #
# Execução completa
# --------------------------------------------------------------------------- #


def rodar_ablacao(chunks: list[Chunk], consultas: list[Consulta],
                  settings: Settings | None = None,
                  cortes_candidatos: tuple[int, ...] = (5, 10, 20, 40),
                  com_reordenacao: bool = True,
                  verboso: bool = True) -> Relatorio:
    settings = settings or SETTINGS
    sistema = Sistema(chunks, settings)

    relatorio = Relatorio(
        backend=settings.backend,
        top_k=settings.top_k,
        comparacao_indices=sistema.comparar_indices(),
    )

    for nome, rotulo in CONFIGURACOES:
        if verboso:
            print(f"  avaliando {nome}...")
        relatorio.avaliacoes.append(avaliar_configuracao(
            sistema, consultas, nome, rotulo, settings.top_k
        ))

    melhor = max(relatorio.avaliacoes, key=lambda a: a.por_tipo().recall)

    if com_reordenacao:
        if verboso:
            print(f"  reordenando sobre {melhor.nome}...")
        for n in cortes_candidatos:
            avaliacao = avaliar_com_reordenacao(
                sistema, consultas, melhor.nome, n, settings.top_k
            )
            teto = sum(r.teto_candidatos for r in avaliacao.resultados) / len(
                avaliacao.resultados
            )
            relatorio.ablacao_candidatos.append((
                n, avaliacao.por_tipo(), teto, avaliacao.pares_reordenados
            ))
            if n == settings.n_candidatos:
                relatorio.avaliacoes.append(avaliacao)

    # Ablação de fusão: o mesmo par de listas, fundido de quatro maneiras.
    if verboso:
        print("  ablação de fusão...")
    for rotulo, kwargs in [
        ("RRF k=60", {"modo": "rrf", "k_rrf": 60}),
        ("RRF k=10", {"modo": "rrf", "k_rrf": 10}),
        ("convexa minmax 1:1", {"modo": "convexo", "normalizacao": "minmax"}),
        ("convexa zscore 1:1", {"modo": "convexo", "normalizacao": "zscore"}),
        ("convexa minmax 2:1", {"modo": "convexo", "normalizacao": "minmax",
                                "pesos": [2.0, 1.0]}),
    ]:
        medidas = []
        for consulta in consultas:
            resultados = sistema.buscar_hibrido(
                consulta.consulta, top_k=20, **kwargs
            )
            recuperados = alvos_recuperados([r.cid for r in resultados],
                                            sistema.mapa_fonte)
            medidas.append(avaliar_uma(recuperados, set(consulta.relevantes),
                                       settings.top_k))
        relatorio.ablacao_fusao.append((rotulo, media(medidas)))

    return relatorio
