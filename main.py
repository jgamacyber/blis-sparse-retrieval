"""
CLI do módulo bônus: do BM25 à esparsidade aprendida.

Com o backend padrão (`local`), **todos** os comandos rodam sem chave de API,
sem download e sem GPU:

    python main.py config          configuração ativa
    python main.py corpus          corpus, chunking e vocabulário
    python main.py consultas       o conjunto e a medida que separa os tipos
    python main.py bm25 "..."      BM25 com explicação termo a termo
    python main.py expansao "..."  o que a expansão acrescenta ao vetor
    python main.py comparar "..."  a mesma consulta nos cinco recuperadores
    python main.py indices         o preço da expansão no índice
    python main.py fusao "..."     RRF contra combinação convexa
    python main.py pipeline "..."  dois estágios, com o teto do primeiro
    python main.py avaliar         a ablação completa
    python testes_offline.py       verificações determinísticas
    python testes_mock.py          verificações do caminho dos modelos reais

Para rodar com SPLADE, BGE-M3 e bge-reranker-v2-m3 de verdade:

    pip install -r requirements-hf.txt
    BACKEND=hf python main.py avaliar

Não há custo por consulta em nenhum dos dois: não há API. O custo do backend
`hf` é download de pesos e tempo de CPU ou GPU. Os detalhes estão em
REPRODUTIBILIDADE.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from avaliacao import CONFIGURACOES, rodar_ablacao
from config import AVISO_LOCAL, SETTINGS, resumo_config
from corpus import (
    carregar_consultas,
    carregar_corpus,
    cobertura_lexical,
    documentos,
    tokenizar,
)
from esparso import IndiceInvertido
from fusao import Lista, combinacao_convexa, rrf
from metricas import alvos_recuperados, avaliar_uma, media
from sistema import Sistema
from splade import diagnosticar

DIR_SAIDA = Path("saidas")


def titulo(texto: str) -> None:
    print(f"\n{'=' * 78}\n  {texto}\n{'=' * 78}")


def aviso_backend() -> None:
    if SETTINGS.backend == "local":
        print(f"\n  {AVISO_LOCAL}")


def montar_sistema(**kwargs) -> tuple[Sistema, list]:
    chunks = carregar_corpus(
        tamanho=SETTINGS.tamanho_chunk, sobreposicao=SETTINGS.sobreposicao
    )
    return Sistema(chunks, SETTINGS, **kwargs), chunks


# --------------------------------------------------------------------------- #
# config, corpus, consultas
# --------------------------------------------------------------------------- #


def cmd_config(_args: argparse.Namespace) -> int:
    titulo("Configuração")
    SETTINGS.validar()
    print(f"\n  {resumo_config()}")

    if SETTINGS.usa_modelos_reais:
        print(f"\n  SPLADE:    {SETTINGS.modelo_splade}")
        print(f"  BGE-M3:    {SETTINGS.modelo_m3}")
        print(f"  reranker:  {SETTINGS.modelo_reranker}")
        print(f"  dispositivo: {SETTINGS.dispositivo}")
    else:
        print(f"\n  {AVISO_LOCAL}")
        print("\n  Nenhum comando deste repositório chama API, nos dois backends.")
        print("  Não existe custo por consulta aqui, existe custo de download e")
        print("  de tempo de máquina no backend hf.")

    print(f"\n  pesos do score integrado do M3: denso={SETTINGS.m3_peso_denso}, "
          f"léxico={SETTINGS.m3_peso_lexical}, "
          f"multivetor={SETTINGS.m3_peso_multivetor}")
    print("  (1/1/1 reproduz a soma sem pesos do artigo)")
    return 0


def cmd_corpus(_args: argparse.Namespace) -> int:
    titulo("Corpus e chunking")
    chunks = carregar_corpus(
        tamanho=SETTINGS.tamanho_chunk, sobreposicao=SETTINGS.sobreposicao
    )
    docs = documentos(chunks)

    print(f"\n  {len(docs)} documentos, {len(chunks)} trechos "
          f"(chunk={SETTINGS.tamanho_chunk}, sobreposição={SETTINGS.sobreposicao})\n")
    for doc in docs:
        do_doc = [c for c in chunks if c.doc == doc]
        secoes = len({c.secao for c in do_doc})
        print(f"    {doc:<24} {len(do_doc):>2} trechos, {secoes:>2} seções — "
              f"{do_doc[0].titulo}")

    vocabulario = set()
    for c in chunks:
        vocabulario.update(tokenizar(c.para_indexar()))
    print(f"\n  vocabulário: {len(vocabulario)} termos distintos")

    print("\n  Os documentos são resumos autorais em português, escritos a partir")
    print("  da leitura dos artigos. Não são traduções nem trechos reproduzidos.")
    return 0


def cmd_consultas(_args: argparse.Namespace) -> int:
    titulo("Conjunto de avaliação")
    chunks = carregar_corpus(
        tamanho=SETTINGS.tamanho_chunk, sobreposicao=SETTINGS.sobreposicao
    )
    por_fonte: dict[str, list[str]] = {}
    for c in chunks:
        por_fonte.setdefault(c.fonte, []).append(c.para_indexar())

    consultas = carregar_consultas()
    por_tipo: dict[str, list[float]] = {}

    print("\n  A coluna `cobertura` é a fração dos termos da consulta que aparece")
    print("  no trecho relevante. É ela que separa os dois tipos, medindo em vez")
    print("  de declarar.\n")
    print(f"  {'id':<5} {'tipo':<10} {'cob.':>6}  consulta")
    print("  " + "-" * 74)

    faltando = []
    for consulta in consultas:
        textos = []
        for alvo in consulta.relevantes:
            textos.extend(por_fonte.get(alvo, []))
            if alvo not in por_fonte:
                faltando.append((consulta.id, alvo))

        cobertura = cobertura_lexical(consulta.consulta, textos)
        por_tipo.setdefault(consulta.tipo, []).append(cobertura)
        print(f"  {consulta.id:<5} {consulta.tipo:<10} {cobertura:>6.2f}  "
              f"{consulta.consulta[:52]}")

    print()
    for tipo, valores in por_tipo.items():
        m = sum(valores) / len(valores)
        print(f"  {tipo:<10} {len(valores):>2} consultas, cobertura média "
              f"{m:.2f} (mín {min(valores):.2f}, máx {max(valores):.2f})")

    if faltando:
        print(f"\n  ATENÇÃO: alvos anotados que não existem no corpus: {faltando}")
        return 1

    print("\n  A diferença entre as duas médias é o que torna o experimento")
    print("  possível: sem ela, a divisão em tipos seria afirmação, não medida.")
    return 0


# --------------------------------------------------------------------------- #
# bm25, expansao
# --------------------------------------------------------------------------- #


def cmd_bm25(args: argparse.Namespace) -> int:
    titulo("BM25, com explicação termo a termo")
    sistema, _ = montar_sistema()

    vetor = sistema.bm25.vetor_consulta(args.consulta)
    print(f"\n  consulta: {args.consulta}")
    print(f"  termos no vocabulário: {vetor.l0} "
          f"(cobertura {sistema.bm25.cobertura(args.consulta):.0%})")

    resultados = sistema.buscar_bm25(args.consulta, args.top_k)
    if not resultados:
        print("\n  Nenhum resultado: nenhum termo da consulta existe no corpus.")
        print("  Esse é o modo de falha do casamento literal, e é o problema que")
        print("  a expansão aprendida existe para resolver.")
        return 0

    for r in resultados:
        chunk = sistema.chunks[r.cid]
        print(f"\n  [{r.score:6.2f}] {chunk.fonte}")
        contribuicoes = sistema.indice_bm25.explicar(vetor, r.cid)
        detalhe = ", ".join(f"{t}={v:.2f}" for t, v in contribuicoes)
        print(f"           contribuição: {detalhe}")

    print("\n  Interpretabilidade é uma propriedade da recuperação léxica que a")
    print("  densa não tem, e que só existe se alguém a expuser. É esta linha de")
    print("  contribuição que permite responder por que um trecho veio.")
    return 0


def cmd_expansao(args: argparse.Namespace) -> int:
    titulo("O que a expansão acrescenta")
    sistema, _ = montar_sistema()
    aviso_backend()

    diag = diagnosticar(sistema.splade, args.consulta)
    print(f"\n  texto: {args.consulta}")
    print(f"  termos ativos no vetor: {diag.l0}")

    print(f"\n  literais ({len(diag.literais)}):")
    print("   ", ", ".join(f"{t}={p:.2f}" for t, p in diag.literais[:12]))

    print(f"\n  acrescentados pela expansão ({len(diag.expandidos)}):")
    if diag.expandidos:
        print("   ", ", ".join(f"{t}={p:.2f}" for t, p in diag.expandidos[:14]))
    else:
        print("    nenhum")

    print("\n  Repare na ordem de grandeza: os literais pesam mais que os")
    print("  acrescentados. É isso que mantém o casamento exato dominando, e é a")
    print("  razão de a esparsidade aprendida não trocar um defeito por outro.")
    return 0


# --------------------------------------------------------------------------- #
# comparar, indices, fusao, pipeline
# --------------------------------------------------------------------------- #


def cmd_comparar(args: argparse.Namespace) -> int:
    titulo("A mesma consulta em cada recuperador")
    sistema, _ = montar_sistema()
    aviso_backend()
    print(f"\n  consulta: {args.consulta}")

    for nome in ("bm25", "splade", "m3-denso", "m3-lexical", "m3-integrado",
                 "hibrido-rrf", "hibrido-convexo"):
        resultados = sistema.buscar(nome, args.consulta, top_k=args.top_k)
        print(f"\n  {nome}")
        if not resultados:
            print("    (nenhum resultado)")
            continue
        for r in resultados:
            print(f"    [{r.score:7.4f}] {sistema.chunks[r.cid].fonte}")

    print("\n  As escalas de score não são comparáveis entre blocos: BM25 vive na")
    print("  casa das unidades, o denso entre menos um e um, a fusão recíproca")
    print("  perto de zero. É exatamente por isso que fundir somando exige")
    print("  normalizar, e que fundir por posto dispensa.")
    return 0


def cmd_indices(_args: argparse.Namespace) -> int:
    sistema, _ = montar_sistema()
    from avaliacao import Relatorio

    relatorio = Relatorio(backend=SETTINGS.backend, top_k=SETTINGS.top_k,
                          comparacao_indices=sistema.comparar_indices())
    relatorio.imprimir_indices()

    print("\n  Termos com a maior lista de postagem")
    print(f"\n  {'BM25':<34} {'SPLADE':<34}")
    print("  " + "-" * 70)
    a = sistema.indice_bm25.termos_mais_frequentes(8)
    b = sistema.indice_splade.termos_mais_frequentes(8)
    for (ta, na), (tb, nb) in zip(a, b):
        print(f"  {ta:<22}{na:>6}      {tb:<22}{nb:>6}")

    print("\n  O caso ruim não é o trecho com muitos termos: é o termo ativo em")
    print("  muitos trechos. É contra ele que o regularizador FLOPS é desenhado,")
    print("  e é por isso que ele soma o quadrado da ativação média.")
    return 0


def cmd_fusao(args: argparse.Namespace) -> int:
    titulo("Fusão: posto contra score")
    sistema, _ = montar_sistema()
    aviso_backend()

    lex = sistema.buscar_bm25(args.consulta, 20)
    den = sistema.buscar_m3(args.consulta, 20, "denso")
    listas = [Lista("bm25", lex), Lista("m3-denso", den)]

    print(f"\n  consulta: {args.consulta}")
    print("\n  entradas (top 5 de cada)")
    for lista in listas:
        print(f"\n    {lista.nome}")
        for r in lista.resultados[:5]:
            print(f"      [{r.score:7.4f}] {sistema.chunks[r.cid].fonte}")

    saidas = [
        ("RRF k=60", rrf(listas, 60, args.top_k)),
        ("RRF k=10", rrf(listas, 10, args.top_k)),
        ("convexa minmax 1:1", combinacao_convexa(listas, None, "minmax", args.top_k)),
        ("convexa zscore 1:1", combinacao_convexa(listas, None, "zscore", args.top_k)),
        ("convexa minmax 2:1", combinacao_convexa(listas, [2.0, 1.0], "minmax",
                                                  args.top_k)),
    ]

    print("\n  saídas")
    for rotulo, resultados in saidas:
        print(f"\n    {rotulo}")
        for r in resultados:
            print(f"      [{r.score:7.4f}] {sistema.chunks[r.cid].fonte}")

    print("\n  O k da fusão recíproca controla quanto as primeiras posições")
    print("  dominam. Com k pequeno, a primeira vale muito mais que a terceira.")
    print("  O k=60 vem de um artigo de 2009, e vale medir o seu.")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    titulo("Pipeline de dois estágios")
    sistema, _ = montar_sistema()
    aviso_backend()

    primeiros, reordenado = sistema.pipeline(
        args.consulta, args.configuracao, args.candidatos, args.top_k
    )

    print(f"\n  consulta: {args.consulta}")
    print(f"  primeiro estágio: {args.configuracao}, "
          f"{len(primeiros)} candidatos")

    print("\n  antes da reordenação")
    for r in primeiros[:args.top_k]:
        print(f"    [{r.score:7.4f}] {sistema.chunks[r.cid].fonte}")

    print(f"\n  depois da reordenação ({reordenado.pares_avaliados} pares avaliados)")
    for c in reordenado.ordenados()[:args.top_k]:
        print(f"    [logit {c.score_reordenacao:7.3f} -> "
              f"{c.score_normalizado:.3f}] "
              f"{sistema.chunks[c.cid].fonte}  (era {c.posicao_recuperacao}º)")

    print(f"\n  o reordenador trocou o primeiro colocado: "
          f"{'sim' if reordenado.mudou_o_topo() else 'não'}")
    print("\n  O score bruto é um logit: não é probabilidade e não é comparável")
    print("  entre consultas. A sigmoide serve para ler, não para comparar, e um")
    print("  limiar fixo sobre ele precisa ser calibrado no seu conjunto.")
    return 0


# --------------------------------------------------------------------------- #
# avaliar
# --------------------------------------------------------------------------- #


def cmd_avaliar(args: argparse.Namespace) -> int:
    titulo(f"Ablação completa — backend {SETTINGS.backend}")
    chunks = carregar_corpus(
        tamanho=SETTINGS.tamanho_chunk, sobreposicao=SETTINGS.sobreposicao
    )
    consultas = carregar_consultas()
    print(f"\n  {len(chunks)} trechos, {len(consultas)} consultas, "
          f"top_k={SETTINGS.top_k}\n")

    relatorio = rodar_ablacao(
        chunks, consultas, SETTINGS,
        cortes_candidatos=tuple(int(n) for n in args.candidatos.split(",")),
        com_reordenacao=not args.sem_reordenacao,
        verboso=True,
    )

    relatorio.imprimir_principal()
    print(relatorio.destaque())
    relatorio.imprimir_indices()
    relatorio.imprimir_candidatos()
    relatorio.imprimir_fusao()
    if args.erros:
        relatorio.imprimir_erros()

    DIR_SAIDA.mkdir(exist_ok=True)
    caminho = relatorio.salvar(DIR_SAIDA / f"relatorio_{SETTINGS.backend}.json")
    print(f"\n  Relatório salvo em {caminho}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Do BM25 à esparsidade aprendida — BLIS, bônus da trilha de RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="comando")

    sub.add_parser("config", help="configuração ativa")
    sub.add_parser("corpus", help="corpus, chunking e vocabulário")
    sub.add_parser("consultas", help="o conjunto e a cobertura lexical por tipo")
    sub.add_parser("indices", help="o preço da expansão no índice")

    p = sub.add_parser("bm25", help="BM25 com explicação termo a termo")
    p.add_argument("consulta")
    p.add_argument("--top-k", type=int, default=5, dest="top_k")

    p = sub.add_parser("expansao", help="o que a expansão acrescenta ao vetor")
    p.add_argument("consulta")

    p = sub.add_parser("comparar", help="a mesma consulta em cada recuperador")
    p.add_argument("consulta")
    p.add_argument("--top-k", type=int, default=3, dest="top_k")

    p = sub.add_parser("fusao", help="RRF contra combinação convexa")
    p.add_argument("consulta")
    p.add_argument("--top-k", type=int, default=5, dest="top_k")

    p = sub.add_parser("pipeline", help="dois estágios, com o teto do primeiro")
    p.add_argument("consulta")
    p.add_argument("--configuracao", default="hibrido-convexo",
                   choices=[n for n, _ in CONFIGURACOES])
    p.add_argument("--candidatos", type=int, default=None)
    p.add_argument("--top-k", type=int, default=5, dest="top_k")

    p = sub.add_parser("avaliar", help="a ablação completa")
    p.add_argument("--candidatos", default="5,10,20,40",
                   help="cortes do conjunto de candidatos, separados por vírgula")
    p.add_argument("--sem-reordenacao", action="store_true",
                   dest="sem_reordenacao")
    p.add_argument("--erros", action="store_true",
                   help="mostra onde cada configuração erra")

    return parser


COMANDOS = {
    "config": cmd_config,
    "corpus": cmd_corpus,
    "consultas": cmd_consultas,
    "indices": cmd_indices,
    "bm25": cmd_bm25,
    "expansao": cmd_expansao,
    "comparar": cmd_comparar,
    "fusao": cmd_fusao,
    "pipeline": cmd_pipeline,
    "avaliar": cmd_avaliar,
}


def main(argv: list[str] | None = None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)

    if not args.comando:
        parser.print_help()
        print("\nComece por: python main.py consultas")
        return 0

    try:
        return COMANDOS[args.comando](args)
    except (FileNotFoundError, ValueError) as erro:
        print(f"\n  erro: {erro}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
