"""
Testes offline — determinísticos, sem rede, sem modelo baixado.

Cobrem a tokenização, o chunking, o BM25 contra valores calculados à mão, o
índice invertido, as transformações do SPLADE, as três funções do BGE-M3, as
duas famílias de fusão, as métricas e a consistência do conjunto de avaliação.

Dois grupos merecem destaque.

O primeiro verifica o **BM25 contra a fórmula**, com números feitos à mão. Se a
saturação ou a normalização por tamanho estiverem erradas, todo número deste
repositório fica errado em silêncio, e a linha de base deixa de ser uma linha
de base.

O segundo verifica que o **índice invertido devolve exatamente o mesmo score**
que o cálculo direto. Essa equivalência é a tese do repositório escrita em
código: se BM25 não puder ser expresso como produto interno de vetores
esparsos, a afirmação de que ele e o SPLADE são o mesmo objeto cai.

    python testes_offline.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

from backends import EmbedderHash, ExpansorCoocorrencia, pontuar_par, sigmoide
from bm25 import BM25, idf_robertson
from config import Settings
from corpus import (
    Chunk,
    carregar_consultas,
    carregar_corpus,
    cobertura_lexical,
    dividir_por_secao,
    documentos,
    tokenizar,
)
from esparso import IndiceInvertido, Resultado, VetorEsparso
from fusao import Lista, combinacao_convexa, normalizar, rrf
from m3 import (
    M3Local,
    RepresentacaoM3,
    score_denso,
    score_integrado,
    score_lexical,
    score_multivetor,
)
from metricas import (
    Medidas,
    alvos_recuperados,
    avaliar_uma,
    hit_at_k,
    media,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from reranker import (
    Candidato,
    RerankerLocal,
    aplicar_limiar,
    reordenar,
    teto_do_primeiro_estagio,
)
from sistema import Sistema
from splade import SpladeLocal, agregar, flops, transformar

FALHAS: list[str] = []


def checar(condicao: bool, descricao: str) -> None:
    if condicao:
        print(f"  [ok]   {descricao}")
    else:
        print(f"  [FALHA] {descricao}")
        FALHAS.append(descricao)


def secao(titulo: str) -> None:
    print(f"\n{titulo}")
    print("-" * 70)


def perto(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


# --------------------------------------------------------------------------- #
# 1. Tokenização e chunking
# --------------------------------------------------------------------------- #

DOC = """# Título

Abertura antes de qualquer seção.

## Primeira

Conteúdo curto da primeira.

## Segunda

Conteúdo curto da segunda.
"""


def testar_tokenizacao() -> None:
    secao("1. Tokenização")

    checar(tokenizar("A memória do MemGPT!") == ["memoria", "memgpt"],
           "remove acento, pontuação e stopword")

    # Termo exato é metade do assunto deste repositório: picar "Recall@5" na
    # tokenização sabotaria o experimento antes de ele começar.
    checar("recall@5" in tokenizar("qual o Recall@5 do híbrido"),
           "mantém Recall@5 como um token só")
    checar("ndcg@10" in tokenizar("nDCG@10 no TREC"),
           "mantém nDCG@10 como um token só")
    checar("normalize" in tokenizar("o que o normalize=True faz"),
           "quebra normalize=True sem perder o nome do parâmetro")

    checar(tokenizar("") == [], "texto vazio não gera token")
    checar(tokenizar("a o de em") == [], "só stopwords não gera token")
    # O filtro de um caractere continua valendo nos dois modos: "a" e "o"
    # sozinhos são ruído em qualquer esquema de peso.
    checar(tokenizar("a o de em", manter_stopwords=True) == ["de", "em"],
           "com manter_stopwords, as stopwords de dois caracteres ficam")


def testar_chunking() -> None:
    secao("2. Chunking por seção")

    chunks = dividir_por_secao(DOC, "exemplo")
    checar(len(chunks) == 3, f"abertura mais duas seções dão 3 trechos ({len(chunks)})")
    checar([c.secao for c in chunks] == ["", "Primeira", "Segunda"],
           "as seções são reconhecidas na ordem")
    checar(chunks[0].fonte == "exemplo" and chunks[1].fonte == "exemplo#Primeira",
           "a fonte junta documento e seção")
    checar(all(c.titulo == "Título" for c in chunks),
           "todo trecho carrega o título do documento")
    checar("Título" in chunks[1].para_indexar()
           and "Primeira" in chunks[1].para_indexar(),
           "título e seção entram no texto indexado")

    longo = "# T\n\n## Longa\n\n" + ("palavra " * 400)
    janelas = dividir_por_secao(longo, "longo", tamanho=300, sobreposicao=60)
    checar(len(janelas) > 1, f"seção longa vira várias janelas ({len(janelas)})")
    checar(all(len(j.texto) <= 300 for j in janelas),
           "nenhuma janela passa do tamanho pedido")

    cids = [c.cid for c in janelas]
    checar(len(set(cids)) == len(cids), "os identificadores de trecho são únicos")

    checar(dividir_por_secao("", "vazio") == [], "documento vazio não gera trecho")

    with tempfile.TemporaryDirectory() as tmp:
        try:
            carregar_corpus(Path(tmp) / "nao-existe")
            checar(False, "corpus inexistente levanta erro")
        except FileNotFoundError as erro:
            checar("de dentro da pasta" in str(erro),
                   "o erro de corpus ausente explica a causa provável")


# --------------------------------------------------------------------------- #
# 3. BM25 contra a fórmula
# --------------------------------------------------------------------------- #


def testar_bm25() -> None:
    secao("3. BM25 contra valores calculados à mão")

    # IDF: termo em 1 de 2 documentos.
    esperado = math.log((2 - 1 + 0.5) / (1 + 0.5) + 1.0)
    checar(perto(idf_robertson(2, 1), esperado), "IDF da família Robertson")

    # O +1 dentro do log é o que mantém o IDF não negativo. Sem ele, termo
    # presente em quase todo documento teria peso negativo, e conter o termo
    # pedido pioraria o score, que é absurdo como comportamento de busca.
    checar(idf_robertson(10, 9) > 0, "IDF não fica negativo com df alto")
    checar(idf_robertson(10, 10) > 0, "IDF não fica negativo com df igual a N")
    checar(idf_robertson(100, 1) > idf_robertson(100, 50),
           "termo raro tem IDF maior que termo comum")

    chunks = [
        Chunk("a:0", "a", "T", "S", "gato gato gato cachorro"),
        Chunk("b:0", "b", "T", "S", "cachorro passaro"),
    ]
    indice = BM25(chunks, k1=1.2, b=0.75)

    # Conta à mão para "gato" no documento a.
    # tokens de a: titulo T, secao S e o corpo -> ['gato','gato','gato','cachorro']
    comprimento_a = len(tokenizar(chunks[0].para_indexar()))
    comprimento_b = len(tokenizar(chunks[1].para_indexar()))
    avgdl = (comprimento_a + comprimento_b) / 2
    checar(perto(indice.avgdl, avgdl), "avgdl é a média dos comprimentos")

    norma = 1.2 * (1 - 0.75 + 0.75 * comprimento_a / avgdl)
    esperado_gato = indice.idf["gato"] * (3 * (1.2 + 1)) / (3 + norma)
    checar(perto(indice.pontuar("gato", "a:0"), esperado_gato),
           "score de um termo bate com a fórmula")

    # Saturação: três ocorrências não valem três vezes uma.
    uma = Chunk("c:0", "c", "T", "S", "gato cachorro cachorro cachorro")
    indice2 = BM25([chunks[0], uma], k1=1.2, b=0.75)
    s3 = indice2.pontuar("gato", "a:0")
    s1 = indice2.pontuar("gato", "c:0")
    checar(s3 < 3 * s1, f"a saturação impede score linear em f ({s3:.3f} < {3*s1:.3f})")
    checar(s3 > s1, "mais ocorrências ainda valem mais que menos")

    # b = 0 desliga a normalização por tamanho.
    sem_norma = BM25(chunks, k1=1.2, b=0.0)
    curto = sem_norma.pontuar("cachorro", "b:0")
    longo = sem_norma.pontuar("cachorro", "a:0")
    checar(perto(curto, longo),
           "com b=0, documentos de tamanhos diferentes e mesma f empatam")

    com_norma = BM25(chunks, k1=1.2, b=1.0)
    checar(com_norma.pontuar("cachorro", "b:0")
           > com_norma.pontuar("cachorro", "a:0"),
           "com b=1, o documento mais curto ganha")

    checar(indice.pontuar("inexistente", "a:0") == 0.0,
           "termo fora do corpus não pontua")
    checar(indice.cobertura("gato inexistente") == 0.5,
           "cobertura mede quanto da consulta existe no vocabulário")


def testar_equivalencia_indice() -> None:
    secao("4. Índice invertido devolve o mesmo score da fórmula")

    chunks = carregar_corpus()
    indice_bm25 = BM25(chunks)
    indice = IndiceInvertido()
    indice.indexar_todos(indice_bm25.vetores_documentos())

    checar(len(indice) == len(chunks), "todos os trechos foram indexados")

    consultas = ["o que e MaxSim", "regularizador FLOPS", "fusao reciproca de postos",
                 "lista de postagem", "parametros k1 e b do BM25"]
    todas_iguais = True
    for consulta in consultas:
        vetor = indice_bm25.vetor_consulta(consulta)
        for resultado in indice.buscar(vetor, 999):
            direto = indice_bm25.pontuar(consulta, resultado.cid)
            if not perto(resultado.score, direto, 1e-9):
                todas_iguais = False

    checar(todas_iguais,
           "o produto interno no índice é idêntico à fórmula, em todas as consultas")

    # É esta equivalência que sustenta a tese do repositório.
    vetor = indice_bm25.vetor_consulta("o que e MaxSim")
    resultados = indice.buscar(vetor, 3)
    checar(bool(resultados), "a busca por um termo raro devolve resultado")
    checar(resultados[0].cid.startswith("colbert"),
           "MaxSim cai no documento do ColBERT")
    contribuicoes = indice.explicar(vetor, resultados[0].cid)
    checar(bool(contribuicoes) and contribuicoes[0][0] == "maxsim",
           "a explicação aponta o termo que gerou o score")
    checar(perto(sum(v for _, v in contribuicoes), resultados[0].score, 1e-9),
           "as contribuições somam exatamente o score")


def testar_vetor_esparso() -> None:
    secao("5. Vetor esparso")

    a = VetorEsparso({"x": 2.0, "y": 3.0})
    b = VetorEsparso({"y": 4.0, "z": 5.0})
    checar(perto(a.produto_interno(b), 12.0), "produto interno usa só a interseção")
    checar(perto(a.produto_interno(b), b.produto_interno(a)),
           "o produto interno é simétrico, qual seja o lado iterado")

    checar(a.l0 == 2 and perto(a.l1, 5.0), "l0 conta termos e l1 soma módulos")
    checar(VetorEsparso().produto_interno(a) == 0.0,
           "vetor vazio dá produto interno zero")

    grande = VetorEsparso({f"t{i}": float(i) for i in range(10)})
    podado = grande.podar(3)
    checar(podado.l0 == 3, "a poda mantém o número pedido de termos")
    checar(set(podado.pesos) == {"t9", "t8", "t7"}, "a poda mantém os maiores pesos")
    checar(grande.podar(0).l0 == 10, "poda com zero não poda")
    checar(grande.podar(99).l0 == 10, "poda maior que o vetor não inventa termos")

    normalizado = a.normalizar_l2()
    norma = sum(v * v for v in normalizado.pesos.values()) ** 0.5
    checar(perto(norma, 1.0), "a normalização L2 dá norma um")
    checar(VetorEsparso().normalizar_l2().l0 == 0,
           "normalizar vetor vazio não quebra")

    indice = IndiceInvertido()
    indice.indexar("d1", a)
    try:
        indice.indexar("d1", b)
        checar(False, "indexar o mesmo id duas vezes levanta erro")
    except ValueError:
        checar(True, "indexar o mesmo id duas vezes levanta erro")

    vazio = IndiceInvertido()
    checar(vazio.estatisticas() == {}, "índice vazio devolve estatísticas vazias")
    checar(vazio.buscar(a, 5) == [], "busca em índice vazio devolve lista vazia")


# --------------------------------------------------------------------------- #
# 6. SPLADE
# --------------------------------------------------------------------------- #


def testar_transformacoes_splade() -> None:
    secao("6. Transformações do SPLADE")

    # log(1 + ReLU(x)): não negativo, saturado, e zero para entrada negativa.
    checar(transformar(-5.0) == 0.0, "ReLU zera logit negativo")
    checar(transformar(0.0) == 0.0, "logit zero dá contribuição zero")
    checar(perto(transformar(1.0), math.log(2.0)), "log1p na parte positiva")
    checar(transformar(10.0) < 10.0, "o logaritmo satura o crescimento")
    checar(transformar(100.0) - transformar(50.0) < transformar(2.0),
           "a saturação achata a diferença entre logits grandes")

    contribuicoes = {"a": [1.0, 3.0, 2.0], "b": [0.5]}
    checar(agregar(contribuicoes, "max") == {"a": 3.0, "b": 0.5},
           "agregação por máximo pega a ocorrência mais informativa")
    checar(agregar(contribuicoes, "soma") == {"a": 6.0, "b": 0.5},
           "agregação por soma acumula, como no v1")

    # A diferença entre as duas é o ponto da mudança do v2.
    checar(agregar(contribuicoes, "soma")["a"] > agregar(contribuicoes, "max")["a"],
           "com soma, termo repetido acumula peso; com máximo, não")

    try:
        agregar(contribuicoes, "mediana")
        checar(False, "agregação desconhecida levanta erro")
    except ValueError as erro:
        checar("max" in str(erro), "o erro de agregação lista os modos válidos")

    checar(agregar({"a": []}, "max") == {}, "termo sem contribuição é descartado")

    # FLOPS: soma do quadrado da ativação média.
    v1 = VetorEsparso({"x": 2.0})
    v2 = VetorEsparso({"x": 2.0})
    checar(perto(flops([v1, v2]), 4.0), "FLOPS de termo com ativação média 2 dá 4")

    espalhado = [VetorEsparso({"a": 2.0}), VetorEsparso({"b": 2.0})]
    concentrado = [VetorEsparso({"a": 2.0}), VetorEsparso({"a": 2.0})]
    checar(flops(concentrado) > flops(espalhado),
           "o quadrado penaliza o termo ativo em muitos documentos")

    checar(flops([]) == 0.0, "FLOPS de conjunto vazio é zero")

    # Comparar esquemas de peso com escalas diferentes exige normalizar.
    grande = [VetorEsparso({"a": 100.0, "b": 100.0})]
    pequeno = [VetorEsparso({"a": 0.1, "b": 0.1})]
    checar(flops(grande) > flops(pequeno),
           "o FLOPS cru depende da escala dos pesos")
    checar(perto(flops(grande, normalizar=True), flops(pequeno, normalizar=True)),
           "normalizado, os dois têm a mesma estrutura e o mesmo FLOPS")


def testar_splade_local() -> None:
    secao("7. SPLADE local: expansão e poda")

    chunks = carregar_corpus()
    modelo = SpladeLocal(chunks)

    texto = "a incompatibilidade de vocabulario atrapalha o casamento literal"
    vetor = modelo.codificar(texto)
    presentes = set(tokenizar(texto))

    checar(all(p > 0 for p in vetor.pesos.values()), "todos os pesos são positivos")
    checar(all(t in vetor for t in presentes),
           "nenhum termo literal é perdido pela expansão")

    acrescentados = set(vetor.pesos) - presentes
    checar(bool(acrescentados), f"a expansão acrescentou termos ({len(acrescentados)})")

    maior_literal = max(vetor.get(t) for t in presentes)
    maior_acrescentado = max(vetor.get(t) for t in acrescentados)
    checar(maior_literal > maior_acrescentado,
           "termo literal pesa mais que termo acrescentado")

    # É isso que impede a expansão de atropelar o casamento exato, que é o
    # argumento central a favor da esparsidade aprendida.
    sem_expansao = modelo.codificar(texto, expandir=False)
    checar(set(sem_expansao.pesos) == presentes,
           "sem expansão, o vetor tem exatamente os termos do texto")
    checar(sem_expansao.l0 < vetor.l0, "a expansão aumenta o número de termos ativos")

    configuracao = Settings(splade_top_termos=5)
    podado = SpladeLocal(chunks, configuracao).codificar(texto)
    checar(podado.l0 <= 5, "a poda respeita SPLADE_TOP_TERMOS")

    consulta = modelo.codificar_consulta("vocabulario")
    consulta_sem = modelo.codificar_consulta("vocabulario", expandir=False)
    checar(consulta.l0 >= consulta_sem.l0,
           "a consulta expandida tem pelo menos tantos termos quanto a literal")
    checar(set(consulta_sem.pesos) == {"vocabulario"},
           "SPLADE-doc deixa a consulta no casamento literal")


def testar_expansor() -> None:
    secao("8. Expansor por coocorrência")

    textos = [
        "insuficiencia cardiaca tratamento",
        "insuficiencia cardiaca diagnostico",
        "insuficiencia cardiaca prognostico",
        "futebol campeonato brasileiro",
        "futebol campeonato europeu",
    ]
    # Fração alta de propósito: com cinco documentos, a fração padrão de 0,25
    # trataria como comum qualquer termo em dois documentos.
    expansor = ExpansorCoocorrencia(textos, minimo_coocorrencia=2,
                                    fracao_maxima_df=0.8)

    vizinhos = dict(expansor.expandir("insuficiencia"))
    checar("cardiaca" in vizinhos, "termo que sempre coocorre entra na expansão")
    checar("futebol" not in vizinhos, "termo de outro assunto não entra")

    checar(all(0.0 < f <= 1.0 for f in vizinhos.values()),
           "as forças ficam entre zero e um")

    # Normalização global: se fosse por termo, o vizinho mais forte de cada
    # termo valeria sempre 1,0 e a diferença entre associação forte e fraca
    # desapareceria.
    forcas = [f for t in expansor.expansoes for _, f in expansor.expandir(t)]
    checar(len(set(round(f, 6) for f in forcas)) > 1,
           "as forças não são todas iguais, ou seja, a escala é global")

    checar(expansor.expandir("inexistente") == [],
           "termo desconhecido devolve expansão vazia")

    st = expansor.estatisticas()
    checar(st["documentos_vistos"] == 5, "contou os documentos")
    checar(st["df_maximo_para_expandir"] >= 1,
           "existe um limite de frequência para expandir")

    # Termo frequente não é expandido: expandi-lo acrescenta ruído a tudo.
    muitos = [f"comum termo{i}" for i in range(20)]
    expansor2 = ExpansorCoocorrencia(muitos, minimo_coocorrencia=1)
    checar("comum" not in expansor2.expansoes,
           "termo presente em todo o corpus não ganha expansão")

    checar(len(ExpansorCoocorrencia([]).expansoes) == 0,
           "corpus vazio não quebra o expansor")


# --------------------------------------------------------------------------- #
# 9. BGE-M3
# --------------------------------------------------------------------------- #


def testar_m3() -> None:
    secao("9. As três funções do BGE-M3")

    q = RepresentacaoM3(
        denso=[1.0, 0.0],
        lexical=VetorEsparso({"a": 1.0, "b": 2.0}),
        multivetor=[[1.0, 0.0], [0.0, 1.0]],
    )
    p = RepresentacaoM3(
        denso=[0.6, 0.8],
        lexical=VetorEsparso({"b": 3.0, "c": 4.0}),
        multivetor=[[1.0, 0.0], [0.5, 0.5]],
    )

    checar(perto(score_denso(q, p), 0.6), "o score denso é o produto interno")
    checar(perto(score_lexical(q, p), 6.0),
           "o score léxico soma só a interseção dos termos")

    # (1/2) * (max(1, 0.5) + max(0, 0.5)) = (1 + 0.5) / 2 = 0.75
    checar(perto(score_multivetor(q, p), 0.75),
           "o multivetor é a média dos máximos, não a soma")

    checar(perto(score_integrado(q, p), 0.6 + 6.0 + 0.75),
           "o score integrado soma as três sem peso, como no artigo")
    checar(perto(score_integrado(q, p, 1.0, 0.0, 0.0), 0.6),
           "zerar pesos isola uma parcela, para a ablação")

    # A diferença entre média dos máximos e soma dos máximos: com média, o
    # score não cresce com o comprimento da consulta.
    q_longa = RepresentacaoM3(multivetor=q.multivetor * 3)
    checar(perto(score_multivetor(q_longa, p), score_multivetor(q, p)),
           "repetir a consulta não muda o score multivetor, por ser média")

    vazio = RepresentacaoM3()
    checar(score_denso(vazio, p) == 0.0, "sem vetor denso o score é zero")
    checar(score_lexical(vazio, p) == 0.0, "sem vetor léxico o score é zero")
    checar(score_multivetor(vazio, p) == 0.0, "sem multivetor o score é zero")

    chunks = carregar_corpus()
    modelo = M3Local(chunks)
    rep = modelo.codificar("o regularizador FLOPS do SPLADE")

    checar(len(rep.denso) == modelo.settings.dim_densa,
           "o vetor denso tem a dimensão configurada")
    checar(perto(sum(v * v for v in rep.denso) ** 0.5, 1.0, 1e-6),
           "o vetor denso é normalizado em L2")
    checar(rep.lexical.l0 > 0, "o vetor léxico tem termos")
    checar(len(rep.multivetor) > 0, "o multivetor tem um vetor por token")

    # O léxico do M3 não expande: só termos presentes no texto.
    presentes = set(tokenizar("o regularizador FLOPS do SPLADE"))
    checar(set(rep.lexical.pesos) == presentes,
           "o léxico do M3 fica restrito aos termos do texto, sem expansão")

    rep_parcial = modelo.codificar("texto", denso=False, multivetor=False)
    checar(not rep_parcial.denso and not rep_parcial.multivetor,
           "é possível pedir só uma das três funções")

    iguais = modelo.codificar("mesmo texto").denso == modelo.codificar(
        "mesmo texto").denso
    checar(iguais, "o embedder local é determinístico")


def testar_embedder() -> None:
    secao("10. Embedder por hashing")

    embedder = EmbedderHash(dim=64)
    a = embedder.codificar("insuficiencia cardiaca")
    b = embedder.codificar("insuficiencia cardiaca")
    c = embedder.codificar("falha do coracao")

    checar(a == b, "textos iguais dão vetores iguais")

    from backends import produto_interno

    checar(perto(produto_interno(a, b), 1.0, 1e-6),
           "o produto interno de um texto com ele mesmo é um")

    # Esta é a limitação declarada: o hashing é cego a sinônimo, e é justamente
    # o trabalho que a parcela densa de verdade faz.
    checar(produto_interno(a, c) < 0.5,
           "sinônimos sem n-grama em comum ficam longe, como esperado do hashing")

    parecido = embedder.codificar("insuficiencia cardiacas")
    checar(produto_interno(a, parecido) > produto_interno(a, c),
           "variação de grafia fica mais perto que sinônimo, por ser n-grama")

    checar(len(embedder.codificar("")) == 64, "texto vazio devolve vetor da dimensão")


# --------------------------------------------------------------------------- #
# 11. Fusão
# --------------------------------------------------------------------------- #


def testar_fusao() -> None:
    secao("11. Fusão")

    lista_a = Lista("a", [Resultado("d1", 10.0, 1), Resultado("d2", 5.0, 2)])
    lista_b = Lista("b", [Resultado("d2", 0.9, 1), Resultado("d3", 0.1, 2)])

    fundido = rrf([lista_a, lista_b], k=60)
    pontos = {r.cid: r.score for r in fundido}

    checar(perto(pontos["d1"], 1 / 61), "d1 aparece só na primeira lista")
    checar(perto(pontos["d2"], 1 / 62 + 1 / 61), "d2 soma as duas contribuições")
    checar(fundido[0].cid == "d2", "o documento nas duas listas vence")
    checar(all(r.posicao == i for i, r in enumerate(fundido, start=1)),
           "a saída vem com as posições renumeradas")

    # O k controla quanto o topo domina.
    k_pequeno = {r.cid: r.score for r in rrf([lista_a, lista_b], k=1)}
    checar(k_pequeno["d1"] / k_pequeno["d2"] > pontos["d1"] / pontos["d2"],
           "com k menor, a primeira posição domina mais")

    checar(rrf([], k=60) == [], "fundir nada devolve nada")
    checar(len(rrf([lista_a, lista_b], 60, top_k=2)) == 2, "o corte top_k é aplicado")

    # Normalizações.
    scores = {"a": 0.0, "b": 5.0, "c": 10.0}
    minmax = normalizar(scores, "minmax")
    checar(minmax == {"a": 0.0, "b": 0.5, "c": 1.0}, "minmax leva ao intervalo 0-1")

    iguais = normalizar({"a": 3.0, "b": 3.0}, "minmax")
    checar(iguais == {"a": 1.0, "b": 1.0},
           "minmax com todos iguais não divide por zero")

    z = normalizar(scores, "zscore")
    checar(perto(sum(z.values()), 0.0, 1e-9), "zscore tem média zero")
    checar(normalizar({"a": 2.0, "b": 2.0}, "zscore") == {"a": 0.0, "b": 0.0},
           "zscore com desvio zero não divide por zero")

    soma = normalizar(scores, "soma")
    checar(perto(sum(soma.values()), 1.0), "soma vira distribuição")
    checar(normalizar(scores, "nenhuma") == scores, "nenhuma não altera nada")

    try:
        normalizar(scores, "inventada")
        checar(False, "normalização desconhecida levanta erro")
    except ValueError:
        checar(True, "normalização desconhecida levanta erro")

    convexa = combinacao_convexa([lista_a, lista_b], [1.0, 1.0], "minmax")
    checar(bool(convexa), "a combinação convexa devolve resultado")

    # Aqui a combinação convexa empata d1 e d2, e isso é a degeneração do
    # minmax com lista de dois itens: o primeiro de cada lista vira 1,0 e o
    # último vira 0,0, então d1 (1,0 + ausente) e d2 (0,0 + 1,0) empatam. A
    # fusão por posto não tem esse problema, e é um argumento concreto a favor
    # dela quando as listas são curtas.
    scores_convexa = {r.cid: r.score for r in convexa}
    checar(perto(scores_convexa["d1"], scores_convexa["d2"]),
           "com listas de dois itens, o minmax empata o topo de cada lista")

    lista_c = Lista("c", [Resultado("d1", 10.0, 1), Resultado("d2", 8.0, 2),
                          Resultado("d4", 1.0, 3)])
    lista_d = Lista("d", [Resultado("d2", 0.9, 1), Resultado("d3", 0.5, 2),
                          Resultado("d5", 0.1, 3)])
    convexa2 = combinacao_convexa([lista_c, lista_d], [1.0, 1.0], "minmax")
    checar(convexa2[0].cid == "d2",
           "com listas maiores, o documento bem posto nas duas vence")

    com_peso = combinacao_convexa([lista_a, lista_b], [10.0, 1.0], "minmax")
    checar(com_peso[0].cid == "d1",
           "peso alto na primeira lista muda o vencedor")

    try:
        combinacao_convexa([lista_a], [1.0, 2.0])
        checar(False, "número de pesos diferente do de listas levanta erro")
    except ValueError:
        checar(True, "número de pesos diferente do de listas levanta erro")

    # A armadilha do zscore: ausente entra com zero, que na escala do zscore é
    # a média, não o mínimo. A implementação desloca a escala antes de somar.
    z_convexa = combinacao_convexa([lista_a, lista_b], None, "zscore")
    checar(all(r.score >= 0 for r in z_convexa),
           "com zscore, a escala é deslocada e nenhum score fica negativo")


# --------------------------------------------------------------------------- #
# 12. Métricas
# --------------------------------------------------------------------------- #


def testar_metricas() -> None:
    secao("12. Métricas")

    recuperados = ["a", "b", "c", "d"]
    relevantes = {"c", "e"}

    checar(hit_at_k(recuperados, relevantes, 3) == 1.0, "Hit@3 encontra o relevante")
    checar(hit_at_k(recuperados, relevantes, 2) == 0.0, "Hit@2 não encontra")
    checar(perto(recall_at_k(recuperados, relevantes, 3), 0.5),
           "Recall@3 é 1 de 2 relevantes")
    checar(perto(precision_at_k(recuperados, relevantes, 4), 0.25), "Precision@4")
    checar(perto(mrr(recuperados, relevantes), 1 / 3), "MRR é 1 sobre a posição")
    checar(mrr(["x", "y"], relevantes) == 0.0, "MRR zero quando nada é relevante")

    # nDCG: relevante na primeira posição dá 1 quando há um só relevante.
    checar(perto(ndcg_at_k(["c"], {"c"}, 1), 1.0), "nDCG de ordenação perfeita é 1")
    checar(ndcg_at_k(["a", "c"], {"c"}, 2) < 1.0,
           "nDCG cai quando o relevante não está em primeiro")
    checar(perto(ndcg_at_k(["a", "c"], {"c"}, 2), (1 / math.log2(3)) / 1.0),
           "nDCG com desconto log2(i+1)")
    checar(ndcg_at_k(recuperados, set(), 3) == 1.0,
           "sem relevantes, nDCG é 1 por convenção")

    # O conjunto vazio de relevantes também não deve quebrar Recall.
    checar(recall_at_k(recuperados, set(), 3) == 1.0, "Recall sem relevantes é 1")

    mapa = {"c1": "doc#A", "c2": "doc#A", "c3": "doc#B"}
    checar(alvos_recuperados(["c1", "c2", "c3"], mapa) == ["doc#A", "doc#B"],
           "duas janelas da mesma seção contam como um alvo só")
    checar(alvos_recuperados(["x"], mapa) == [],
           "identificador fora do mapa é ignorado")

    m = avaliar_uma(recuperados, relevantes, 3)
    checar(m.n == 1, "avaliar uma consulta conta n igual a 1")

    agregado = media([avaliar_uma(["c"], {"c"}, 1), avaliar_uma(["x"], {"c"}, 1)])
    checar(perto(agregado.recall, 0.5) and agregado.n == 2, "a média agrega as duas")
    checar(media([]).n == 0, "média de lista vazia não quebra")

    checar("Recall@k" in Medidas.cabecalho(), "o cabeçalho nomeia as métricas")


# --------------------------------------------------------------------------- #
# 13. Reordenação
# --------------------------------------------------------------------------- #


def testar_reordenacao() -> None:
    secao("13. Reordenação")

    checar(perto(sigmoide(0.0), 0.5), "sigmoide de zero é meio")
    checar(sigmoide(10.0) > 0.99 and sigmoide(-10.0) < 0.01,
           "a sigmoide satura nos extremos")
    checar(0.0 < sigmoide(-800.0) < 1e-300 or sigmoide(-800.0) == 0.0,
           "a sigmoide não estoura com entrada muito negativa")

    consulta = "qual o total do pedido com desconto"
    bom = "o total do pedido com desconto aplicado fica em 765 reais"
    ruim = "futebol campeonato brasileiro tabela de classificacao"
    checar(pontuar_par(consulta, bom) > pontuar_par(consulta, ruim),
           "o escore separa passagem relevante de irrelevante")
    checar(pontuar_par("", bom) < -5, "consulta vazia recebe escore muito baixo")
    checar(pontuar_par(consulta, "") < -5, "passagem vazia recebe escore muito baixo")

    # Proximidade é o único sinal aqui que o produto interno não tem.
    juntos = "o total do pedido com desconto"
    espalhados = "o total " + ("palavra " * 40) + " do pedido " + ("outra " * 40) + " com desconto"
    checar(pontuar_par(consulta, juntos) > pontuar_par(consulta, espalhados),
           "termos próximos pontuam mais que termos espalhados")

    modelo = RerankerLocal()
    candidatos = [
        Candidato("c1", ruim, 9.0, 1),
        Candidato("c2", bom, 1.0, 2),
    ]
    resultado = reordenar(modelo, consulta, candidatos)

    checar(resultado.pares_avaliados == 2, "contou os pares avaliados")
    checar(resultado.ordenados()[0].cid == "c2",
           "a reordenação promoveu o candidato melhor")
    checar(resultado.mudou_o_topo(), "detectou a troca do primeiro colocado")
    checar(all(0.0 <= c.score_normalizado <= 1.0 for c in resultado.candidatos),
           "o score normalizado fica entre zero e um")

    checar(reordenar(modelo, consulta, []).pares_avaliados == 0,
           "reordenar lista vazia não quebra")

    acima = aplicar_limiar(resultado, limiar=0.0)
    checar(len(acima) == 2, "limiar zero mantém tudo")
    checar(aplicar_limiar(resultado, limiar=1.1) == [], "limiar acima de um corta tudo")

    mapa = {"c1": "doc#A", "c2": "doc#B"}
    checar(perto(teto_do_primeiro_estagio(["c1", "c2"], {"doc#B"}, mapa), 1.0),
           "o teto é 1 quando o relevante está entre os candidatos")
    checar(perto(teto_do_primeiro_estagio(["c1"], {"doc#B"}, mapa), 0.0),
           "o teto é 0 quando o relevante não chegou")
    checar(perto(teto_do_primeiro_estagio(["c1"], set(), mapa), 1.0),
           "sem relevantes, o teto é 1")


# --------------------------------------------------------------------------- #
# 14. Conjunto de avaliação e sistema
# --------------------------------------------------------------------------- #


def testar_conjunto() -> None:
    secao("14. Conjunto de avaliação")

    chunks = carregar_corpus()
    consultas = carregar_consultas()
    alvos = {c.fonte for c in chunks}

    checar(len(consultas) == 24, f"vinte e quatro consultas ({len(consultas)})")
    checar(len({c.id for c in consultas}) == len(consultas), "identificadores únicos")

    faltando = [(c.id, r) for c in consultas for r in c.relevantes if r not in alvos]
    checar(not faltando, f"todo alvo anotado existe no corpus ({faltando})")

    tipos = {c.tipo for c in consultas}
    checar(tipos == {"exato", "parafrase"}, f"dois tipos anotados: {sorted(tipos)}")

    por_tipo: dict[str, list[float]] = {}
    por_fonte: dict[str, list[str]] = {}
    for c in chunks:
        por_fonte.setdefault(c.fonte, []).append(c.para_indexar())

    for consulta in consultas:
        textos = [t for alvo in consulta.relevantes for t in por_fonte[alvo]]
        por_tipo.setdefault(consulta.tipo, []).append(
            cobertura_lexical(consulta.consulta, textos)
        )

    media_exato = sum(por_tipo["exato"]) / len(por_tipo["exato"])
    media_paraf = sum(por_tipo["parafrase"]) / len(por_tipo["parafrase"])

    # Sem esta diferença, a divisão em tipos seria afirmação e não medida, e o
    # experimento central do repositório não existiria.
    checar(media_exato > media_paraf + 0.25,
           f"as de termo exato têm cobertura bem maior ({media_exato:.2f} contra "
           f"{media_paraf:.2f})")
    checar(len(por_tipo["exato"]) == len(por_tipo["parafrase"]),
           "os dois tipos têm o mesmo número de consultas")

    checar(perto(cobertura_lexical("", ["texto"]), 0.0),
           "consulta vazia tem cobertura zero")
    checar(perto(cobertura_lexical("termo", []), 0.0),
           "sem texto relevante, a cobertura é zero")


def testar_sistema() -> None:
    secao("15. Sistema completo")

    chunks = carregar_corpus()
    sistema = Sistema(chunks, Settings())

    checar(len(sistema.chunks) == len(chunks), "indexou todos os trechos")
    checar(len(sistema.indice_bm25) == len(chunks), "o índice BM25 está completo")
    checar(len(sistema.indice_splade) == len(chunks), "o índice SPLADE está completo")
    checar(len(sistema.reps_m3) == len(chunks), "as representações M3 estão completas")

    consulta = "o regularizador FLOPS"
    for nome in ("bm25", "splade", "splade-doc", "m3-denso", "m3-lexical",
                 "m3-integrado", "hibrido-rrf", "hibrido-convexo",
                 "splade-denso-rrf"):
        resultados = sistema.buscar(nome, consulta, top_k=5)
        checar(bool(resultados), f"{nome} devolve resultado")
        checar(all(r.cid in sistema.chunks for r in resultados),
               f"{nome} devolve identificadores válidos")
        scores = [r.score for r in resultados]
        checar(scores == sorted(scores, reverse=True),
               f"{nome} devolve em ordem decrescente de score")

    try:
        sistema.buscar("inexistente", consulta)
        checar(False, "configuração desconhecida levanta erro")
    except ValueError as erro:
        checar("desconhecido" in str(erro),
               "o erro de configuração desconhecida é explicativo")

    # A expansão custa índice: é o número que falta em quase toda comparação.
    comparacao = sistema.comparar_indices()
    checar(comparacao["razao_termos_ativos"] > 1.0,
           "o SPLADE ativa mais termos por trecho que o BM25")
    checar(comparacao["splade"]["flops_normalizado"]
           > comparacao["bm25"]["flops_normalizado"],
           "normalizado, o FLOPS do SPLADE é maior, coerente com mais postagens")

    primeiros, reordenado = sistema.pipeline(consulta, "bm25", 10, 5)
    checar(len(primeiros) <= 10, "o primeiro estágio respeita o corte de candidatos")
    checar(reordenado.pares_avaliados == len(primeiros),
           "o segundo estágio avaliou um par por candidato")

    configuracao = Settings(backend="inventado")
    try:
        configuracao.validar()
        checar(False, "backend inválido levanta erro")
    except ValueError as erro:
        checar("local" in str(erro), "o erro de backend lista os válidos")


# --------------------------------------------------------------------------- #


def main() -> int:
    print("=" * 70)
    print("  Testes offline — do BM25 à esparsidade aprendida")
    print("  Determinísticos, sem rede e sem modelo baixado.")
    print("=" * 70)

    testar_tokenizacao()
    testar_chunking()
    testar_bm25()
    testar_equivalencia_indice()
    testar_vetor_esparso()
    testar_transformacoes_splade()
    testar_splade_local()
    testar_expansor()
    testar_m3()
    testar_embedder()
    testar_fusao()
    testar_metricas()
    testar_reordenacao()
    testar_conjunto()
    testar_sistema()

    print("\n" + "=" * 70)
    if FALHAS:
        print(f"  {len(FALHAS)} FALHA(S):")
        for f in FALHAS:
            print(f"    - {f}")
        print("=" * 70)
        return 1

    print("  Todos os testes passaram.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
