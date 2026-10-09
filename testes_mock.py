"""
Testes do caminho dos modelos reais, com modelos simulados.

Por que isto existe: o backend `hf` é o que produz números comparáveis com os
artigos, e é justamente o que não roda em ambiente sem GPU nem download. Sem
estes testes, o código do caminho real seria enviado sem nunca ter sido
executado, e o erro mais provável nele é silencioso: um eixo trocado no
agrupamento, a máscara de atenção esquecida, o mapeamento de id para token
invertido. Nada disso levanta exceção, tudo isso estraga o vetor.

A estratégia é substituir o tensor e o modelo, não a aritmética. O `TensorFalso`
implementa exatamente as operações que `SpladeHF` usa, sobre listas aninhadas,
de modo que a conta verificada aqui é a mesma conta que roda com torch:

    w_j = max  log(1 + ReLU(w_ij))        sobre as posições válidas
          i∈t

Os valores esperados são calculados à mão no próprio teste.

    python testes_mock.py
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

from config import Settings
from corpus import Chunk
from esparso import VetorEsparso

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
# Tensor e torch simulados
# --------------------------------------------------------------------------- #


class TensorFalso:
    """
    Tensor mínimo sobre listas aninhadas.

    Implementa só o que o caminho do SPLADE usa. Qualquer operação a mais que
    o código real passe a usar vai falhar aqui de forma ruidosa, o que é o
    comportamento desejado: o teste quebra quando o código muda de forma, em
    vez de passar sobre uma simulação desatualizada.
    """

    def __init__(self, dados) -> None:
        self.dados = dados

    # --- forma ---

    @property
    def shape(self) -> tuple[int, ...]:
        forma, atual = [], self.dados
        while isinstance(atual, list):
            forma.append(len(atual))
            atual = atual[0] if atual else None
        return tuple(forma)

    def unsqueeze(self, dim: int) -> "TensorFalso":
        if dim != -1:
            raise NotImplementedError("o código real só usa unsqueeze(-1)")
        return TensorFalso([[[v] for v in linha] for linha in self.dados])

    def squeeze(self, dim: int) -> "TensorFalso":
        if dim == 0:
            return TensorFalso(self.dados[0])
        if dim == -1:
            return TensorFalso([
                v[0] if isinstance(v, list) else v for v in self.dados
            ])
        raise NotImplementedError(f"squeeze({dim}) não é usado pelo código real")

    # --- aritmética ---

    def __mul__(self, outro: "TensorFalso") -> "TensorFalso":
        """Multiplicação com difusão na última dimensão, que é o caso da máscara."""
        return TensorFalso([
            [
                [v * mascara[0] for v in posicao]
                for posicao, mascara in zip(lote_a, lote_b)
            ]
            for lote_a, lote_b in zip(self.dados, outro.dados)
        ])

    def max(self, dim: int):
        if dim != 1:
            raise NotImplementedError("o código real agrega sobre dim=1")
        valores = [
            [max(posicao[j] for posicao in lote) for j in range(len(lote[0]))]
            for lote in self.dados
        ]
        return SimpleNamespace(values=TensorFalso(valores))

    def sum(self, dim: int) -> "TensorFalso":
        if dim != 1:
            raise NotImplementedError("o código real agrega sobre dim=1")
        return TensorFalso([
            [sum(posicao[j] for posicao in lote) for j in range(len(lote[0]))]
            for lote in self.dados
        ])

    # --- indexação ---

    def __getitem__(self, indices):
        if isinstance(indices, TensorFalso):
            return TensorFalso([self.dados[i] for i in indices.dados])
        return TensorFalso(self.dados[indices])

    def tolist(self):
        return self.dados


class TorchFalso:
    @staticmethod
    def no_grad():
        class Contexto:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, *_):
                return False

        return Contexto()

    @staticmethod
    def relu(t: TensorFalso) -> TensorFalso:
        def aplicar(x):
            if isinstance(x, list):
                return [aplicar(v) for v in x]
            return x if x > 0 else 0.0

        return TensorFalso(aplicar(t.dados))

    @staticmethod
    def log1p(t: TensorFalso) -> TensorFalso:
        def aplicar(x):
            if isinstance(x, list):
                return [aplicar(v) for v in x]
            return math.log1p(x)

        return TensorFalso(aplicar(t.dados))

    @staticmethod
    def nonzero(t: TensorFalso) -> TensorFalso:
        return TensorFalso([[i] for i, v in enumerate(t.dados) if v != 0.0])


class EntradasFalsas(dict):
    """Saída do tokenizer: dicionário com `.to(dispositivo)`."""

    def to(self, _dispositivo):
        return self


class TokenizerFalso:
    def __init__(self, vocabulario: list[str]) -> None:
        self.vocabulario = vocabulario
        self.chamadas: list[dict] = []

    def __call__(self, texto, **kwargs):
        self.chamadas.append({"texto": texto, **kwargs})
        # Duas posições válidas e uma de preenchimento: a máscara é o ponto.
        return EntradasFalsas({
            "input_ids": TensorFalso([[1, 2, 0]]),
            "attention_mask": TensorFalso([[1, 1, 0]]),
        })

    def convert_ids_to_tokens(self, indices):
        return [self.vocabulario[i] for i in indices]


class ModeloMLMFalso:
    """
    Devolve logits fixos, escolhidos para que a conta seja verificável à mão.

    Vocabulário de quatro tokens e três posições, sendo a terceira de
    preenchimento com valores altos: se a máscara for esquecida, ela domina o
    máximo e o teste quebra.
    """

    LOGITS = [[
        [2.0, -1.0, 0.0, 0.5],     # posição 1, válida
        [1.0, 3.0, -5.0, 0.25],    # posição 2, válida
        [9.0, 9.0, 9.0, 9.0],      # posição 3, PREENCHIMENTO
    ]]

    def __init__(self) -> None:
        self.chamadas = 0

    def __call__(self, **_kwargs):
        self.chamadas += 1
        return SimpleNamespace(logits=TensorFalso(self.LOGITS))

    def to(self, _dispositivo):
        return self

    def eval(self):
        return self


# --------------------------------------------------------------------------- #
# 1. SPLADE no caminho real
# --------------------------------------------------------------------------- #

VOCABULARIO = ["alfa", "beta", "gama", "delta"]

# As classes reais, capturadas na importação. O teste do sistema inteiro troca
# os atributos do módulo por dublês, e sem guardar a referência original aqui a
# fábrica abaixo chamaria o próprio dublê, em recursão infinita.
import m3 as _mod_m3                      # noqa: E402
import reranker as _mod_reranker          # noqa: E402
import splade as _mod_splade              # noqa: E402

SPLADE_HF = _mod_splade.SpladeHF
M3_HF = _mod_m3.M3HF
RERANKER_HF = _mod_reranker.RerankerHF


def montar_splade_hf(agregacao: str = "max"):
    modelo = SPLADE_HF(Settings(backend="hf", splade_top_termos=96), agregacao)
    modelo._torch = TorchFalso()
    modelo._tokenizer = TokenizerFalso(VOCABULARIO)
    modelo._modelo = ModeloMLMFalso()
    modelo._carregado = True
    return modelo


def testar_splade_hf() -> None:
    secao("1. SPLADE no caminho HuggingFace")

    modelo = montar_splade_hf("max")
    vetor = modelo.codificar("qualquer texto")

    # Conta à mão, só sobre as duas posições válidas:
    #   alfa : max(log1p(2.0), log1p(1.0))   = log(3)
    #   beta : max(0,          log1p(3.0))   = log(4)
    #   gama : max(0,          0)            = 0, descartado
    #   delta: max(log1p(0.5), log1p(0.25))  = log(1.5)
    checar(perto(vetor.get("alfa"), math.log(3.0)),
           "alfa: máximo de log1p sobre as posições válidas")
    checar(perto(vetor.get("beta"), math.log(4.0)),
           "beta: o ReLU zera o logit negativo da primeira posição")
    checar("gama" not in vetor,
           "gama sai do vetor: ReLU zerou tudo, e daí vem a esparsidade")
    checar(perto(vetor.get("delta"), math.log(1.5)), "delta: máximo entre os dois")

    # O teste que importa: a posição de preenchimento tem logit 9 em tudo. Se a
    # máscara de atenção não fosse aplicada, todo peso seria log(10).
    checar(not perto(vetor.get("alfa"), math.log(10.0)),
           "a máscara de atenção descarta a posição de preenchimento")
    checar(all(p < math.log(10.0) for p in vetor.pesos.values()),
           "nenhum peso vem do preenchimento")

    checar(vetor.l0 == 3, f"três termos ativos de quatro no vocabulário ({vetor.l0})")

    # Agregação por soma: o v1 do artigo.
    soma = montar_splade_hf("soma").codificar("qualquer texto")
    checar(perto(soma.get("alfa"), math.log(3.0) + math.log(2.0)),
           "com soma, as duas posições acumulam, como no v1")
    checar(soma.get("alfa") > vetor.get("alfa"),
           "a soma dá peso maior que o máximo para termo em duas posições")

    # Truncamento e parâmetros enviados ao tokenizer.
    chamada = modelo._tokenizer.chamadas[0]
    checar(chamada["truncation"] is True, "pede truncamento ao tokenizer")
    checar(chamada["max_length"] == 512,
           "envia o limite de tokens configurado")
    checar(chamada["return_tensors"] == "pt", "pede tensores do torch")


def testar_splade_hf_consulta() -> None:
    secao("2. SPLADE: consulta, poda e variante doc")

    modelo = montar_splade_hf("max")

    consulta = modelo.codificar_consulta("qualquer texto")
    checar(consulta.l0 <= max(8, 96 // 3),
           "a consulta é podada com mais pressão que o documento")

    # SPLADE-doc: a consulta não passa pelo modelo.
    antes = modelo._modelo.chamadas
    literal = modelo.codificar_consulta("alfa beta inexistente", expandir=False)
    checar(modelo._modelo.chamadas == antes,
           "na variante doc, a consulta não gasta uma passada no modelo")
    checar(set(literal.pesos) == {"alfa", "beta", "inexistente"},
           "na variante doc, a consulta é casamento literal com peso 1")
    checar(all(perto(p, 1.0) for p in literal.pesos.values()),
           "os pesos da consulta literal são todos 1, como em Σ_{j∈q} w_j^d")

    # Poda configurada.
    modelo_podado = montar_splade_hf("max")
    modelo_podado.settings = Settings(backend="hf", splade_top_termos=2)
    podado = modelo_podado.codificar("qualquer texto")
    checar(podado.l0 == 2, "a poda do documento respeita SPLADE_TOP_TERMOS")
    checar(set(podado.pesos) == {"beta", "alfa"},
           "a poda mantém os dois maiores, beta e alfa")

    chunk = Chunk("x:0", "x", "T", "S", "corpo do trecho")
    vetor = modelo.codificar_documento(chunk)
    checar(vetor.l0 > 0, "codificar_documento usa o texto indexado do trecho")


def testar_splade_hf_sem_dependencia() -> None:
    secao("3. Mensagem de erro quando falta a dependência")

    # Simula transformers ausente, que é o estado da maioria das máquinas.
    salvos = {}
    for nome in ("torch", "transformers"):
        salvos[nome] = sys.modules.get(nome)
        sys.modules[nome] = None

    try:
        modelo = SPLADE_HF(Settings(backend="hf"))
        try:
            modelo.codificar("texto")
            checar(False, "faltando a dependência, levanta erro")
        except RuntimeError as erro:
            mensagem = str(erro)
            checar("requirements-hf.txt" in mensagem,
                   "o erro diz exatamente o que instalar")
            checar("BACKEND=local" in mensagem,
                   "o erro oferece a alternativa que funciona sem instalar nada")
        except (ImportError, TypeError):
            checar(False, "o erro de dependência é convertido em RuntimeError claro")
    finally:
        for nome, valor in salvos.items():
            if valor is None:
                sys.modules.pop(nome, None)
            else:
                sys.modules[nome] = valor


# --------------------------------------------------------------------------- #
# 4. BGE-M3 no caminho real
# --------------------------------------------------------------------------- #


class M3FlagFalso:
    """Dublê do BGEM3FlagModel, com a forma de saída que o FlagEmbedding usa."""

    def __init__(self, *_args, **kwargs) -> None:
        self.kwargs = kwargs
        self.chamadas: list[dict] = []

    def encode(self, textos, **kwargs):
        self.chamadas.append({"textos": textos, **kwargs})
        saida = {}
        if kwargs.get("return_dense"):
            saida["dense_vecs"] = [[0.6, 0.8]]
        if kwargs.get("return_sparse"):
            # Chaves por id de token, como o FlagEmbedding devolve, e um peso
            # zero que precisa ser descartado.
            saida["lexical_weights"] = [{101: 0.5, 202: 0.25, 303: 0.0}]
        if kwargs.get("return_colbert_vecs"):
            saida["colbert_vecs"] = [[[1.0, 0.0], [0.0, 1.0]]]
        return saida


def testar_m3_hf() -> None:
    secao("4. BGE-M3 no caminho HuggingFace")

    from m3 import score_denso, score_integrado, score_lexical

    modelo = M3_HF(Settings(backend="hf"))
    modelo._modelo = M3FlagFalso()

    rep = modelo.codificar("texto qualquer")

    checar(rep.denso == [0.6, 0.8], "lê dense_vecs")
    checar(rep.lexical.l0 == 2,
           "lê lexical_weights e descarta peso zero, que não entra no índice")
    checar(set(rep.lexical.pesos) == {"101", "202"},
           "a chave por id de token é convertida para texto")
    checar(len(rep.multivetor) == 2, "lê colbert_vecs, um vetor por token")

    chamada = modelo._modelo.chamadas[0]
    checar(chamada["textos"] == ["texto qualquer"], "envia o texto em lote de um")
    checar(chamada["return_dense"] and chamada["return_sparse"]
           and chamada["return_colbert_vecs"],
           "pede as três representações na mesma passada, que é a multifuncionalidade")
    checar(chamada["max_length"] == 512, "envia o limite de tokens configurado")

    # Pedir só uma função não deve pedir as outras ao modelo.
    modelo._modelo.chamadas.clear()
    rep_denso = modelo.codificar("texto", lexical=False, multivetor=False)
    chamada = modelo._modelo.chamadas[0]
    checar(not chamada["return_sparse"] and not chamada["return_colbert_vecs"],
           "pedir só o denso não gasta as outras duas representações")
    checar(rep_denso.lexical.l0 == 0 and not rep_denso.multivetor,
           "as representações não pedidas voltam vazias")

    # O score integrado sobre as representações reais.
    rep_a = modelo.codificar("a")
    rep_b = modelo.codificar("b")
    esperado = (score_denso(rep_a, rep_b) + score_lexical(rep_a, rep_b)
                + 1.0)   # multivetor: média dos máximos de vetores idênticos
    checar(perto(score_integrado(rep_a, rep_b), esperado),
           "o score integrado soma as três parcelas das saídas do modelo")


# --------------------------------------------------------------------------- #
# 5. Reordenador no caminho real
# --------------------------------------------------------------------------- #


class RerankerFlagFalso:
    """Dublê do FlagReranker. Devolve logits, não probabilidades."""

    def __init__(self, *_args, **kwargs) -> None:
        self.kwargs = kwargs
        self.chamadas: list[dict] = []
        self.scores = [-5.65, 5.26]

    def compute_score(self, pares, **kwargs):
        self.chamadas.append({"pares": pares, **kwargs})
        if len(pares) == 1:
            return self.scores[0]        # a interface devolve float, não lista
        return self.scores[:len(pares)]


def testar_reranker_hf() -> None:
    secao("5. Reordenador no caminho HuggingFace")

    from reranker import Candidato, reordenar

    modelo = RERANKER_HF(Settings(backend="hf"))
    modelo._modelo = RerankerFlagFalso()

    scores = modelo.pontuar("consulta", ["passagem ruim", "passagem boa"])
    checar(scores == [-5.65, 5.26], "lê os dois logits")

    chamada = modelo._modelo.chamadas[0]
    checar(chamada["pares"] == [["consulta", "passagem ruim"],
                                ["consulta", "passagem boa"]],
           "monta os pares no formato consulta e passagem")
    checar(chamada["normalize"] is False,
           "pede logit cru: a sigmoide é aplicada no mesmo ponto nos dois backends")

    # A interface devolve float quando há um par só, e isso quebraria o zip.
    um = modelo.pontuar("consulta", ["uma passagem"])
    checar(um == [-5.65], "um par só também volta como lista")

    checar(modelo.pontuar("consulta", []) == [],
           "lista de passagens vazia não chama o modelo")

    candidatos = [Candidato("c1", "passagem ruim", 9.0, 1),
                  Candidato("c2", "passagem boa", 1.0, 2)]
    resultado = reordenar(modelo, "consulta", candidatos)

    checar(resultado.ordenados()[0].cid == "c2",
           "a reordenação promove o candidato de logit maior")
    checar(resultado.mudou_o_topo(), "detecta a troca do primeiro colocado")

    # O valor da ficha do modelo: -5,65 normalizado dá cerca de 0,0035.
    pior = next(c for c in resultado.candidatos if c.cid == "c1")
    checar(abs(pior.score_normalizado - 0.0035) < 0.0005,
           "a sigmoide de -5,65 dá cerca de 0,0035, como na ficha do modelo")

    melhor = next(c for c in resultado.candidatos if c.cid == "c2")
    checar(melhor.score_normalizado > 0.99,
           "a sigmoide de 5,26 fica perto de um")


def testar_reranker_sem_dependencia() -> None:
    secao("6. Reordenador sem a dependência instalada")

    salvo = sys.modules.get("FlagEmbedding")
    sys.modules["FlagEmbedding"] = None
    try:
        modelo = RERANKER_HF(Settings(backend="hf"))
        try:
            modelo.pontuar("consulta", ["passagem"])
            checar(False, "faltando FlagEmbedding, levanta erro")
        except RuntimeError as erro:
            checar("requirements-hf.txt" in str(erro),
                   "o erro diz o que instalar")
            checar("BACKEND=local" in str(erro),
                   "o erro oferece a alternativa sem instalação")
        except (ImportError, TypeError):
            checar(False, "o erro de dependência é convertido em RuntimeError claro")
    finally:
        if salvo is None:
            sys.modules.pop("FlagEmbedding", None)
        else:
            sys.modules["FlagEmbedding"] = salvo


# --------------------------------------------------------------------------- #
# 7. O sistema com os dublês
# --------------------------------------------------------------------------- #


def testar_sistema_hf() -> None:
    secao("7. O sistema inteiro com os modelos simulados")

    import m3 as mod_m3
    import reranker as mod_reranker
    import splade as mod_splade
    from sistema import Sistema

    chunks = [
        Chunk("a:0", "a", "Titulo A", "Secao", "alfa beta conteudo do trecho a"),
        Chunk("b:0", "b", "Titulo B", "Secao", "gama delta conteudo do trecho b"),
        Chunk("c:0", "c", "Titulo C", "Secao", "alfa gama conteudo do trecho c"),
    ]

    splade_original = mod_splade.SpladeHF
    m3_original = mod_m3.M3HF
    reranker_original = mod_reranker.RerankerHF

    def splade_falso(settings=None, agregacao="max"):
        return montar_splade_hf(agregacao)

    def m3_falso(settings=None):
        modelo = M3_HF(settings)
        modelo._modelo = M3FlagFalso()
        return modelo

    def reranker_falso(settings=None):
        modelo = RERANKER_HF(settings)
        modelo._modelo = RerankerFlagFalso()
        return modelo

    mod_splade.SpladeHF = splade_falso
    mod_m3.M3HF = m3_falso
    mod_reranker.RerankerHF = reranker_falso
    try:
        sistema = Sistema(chunks, Settings(backend="hf"))

        checar(len(sistema.indice_splade) == 3,
               "o índice SPLADE foi montado com os vetores do modelo simulado")
        checar(len(sistema.reps_m3) == 3, "as três representações M3 foram montadas")

        resultados = sistema.buscar("splade", "alfa beta", top_k=3)
        checar(bool(resultados), "a busca SPLADE funciona no backend hf")

        resultados = sistema.buscar("bm25", "alfa", top_k=3)
        checar(resultados[0].cid in ("a:0", "c:0"),
               "o BM25 não depende de backend e continua funcionando")

        primeiros, reordenado = sistema.pipeline("alfa", "bm25", 2, 2)
        checar(reordenado.pares_avaliados == len(primeiros),
               "o pipeline de dois estágios roda ponta a ponta no backend hf")

        comparacao = sistema.comparar_indices()
        checar("flops_normalizado" in comparacao["splade"],
               "a comparação de índices também é calculada no backend hf")
    finally:
        mod_splade.SpladeHF = splade_original
        mod_m3.M3HF = m3_original
        mod_reranker.RerankerHF = reranker_original


# --------------------------------------------------------------------------- #


def main() -> int:
    print("=" * 70)
    print("  Testes do caminho dos modelos reais, com modelos simulados")
    print("  Nenhum download, nenhuma GPU, nenhuma chamada de rede.")
    print("=" * 70)

    testar_splade_hf()
    testar_splade_hf_consulta()
    testar_splade_hf_sem_dependencia()
    testar_m3_hf()
    testar_reranker_hf()
    testar_reranker_sem_dependencia()
    testar_sistema_hf()

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
