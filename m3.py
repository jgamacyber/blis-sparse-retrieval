"""
BGE-M3: as três funções de recuperação e o score integrado.

Referência: Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D. e Liu, Z. (2024).
*M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text
Embeddings Through Self-Knowledge Distillation.*

As três funções, como o artigo as define:

    denso        e_q = norm(H_q[0]);          s_dense = ⟨e_p, e_q⟩
    léxico       w_qt = ReLU(W_lexᵀ H_q[i]);  s_lex = Σ_{t ∈ q∩p} w_qt · w_pt
    multivetor   E = norm(W_mulᵀ H);          s_mul = (1/N) Σ_i max_j E_q[i]·E_p[j]ᵀ

    integrado    s_inter = s_dense + s_lex + s_mul

Três observações que mudam a leitura das fórmulas.

**O léxico do M3 não é o SPLADE.** A soma é restrita à interseção `t ∈ q ∩ p`,
isto é, aos termos que aparecem nos dois textos. Não há expansão para termo
ausente. É casamento léxico com peso aprendido, e não expansão aprendida. Quem
lê os dois artigos rápido junta as duas coisas, e elas resolvem problemas
diferentes: o M3 aprende *quanto* cada termo vale, o SPLADE aprende *quais*
termos deveriam estar lá.

**O multivetor usa média dos máximos, o ColBERT usa soma dos máximos.** Com
média, o score não cresce com o comprimento da consulta, o que o torna
comparável entre consultas. Com soma, não.

**O artigo soma as três parcelas sem peso.** Cada uma entra com peso um. Os
pesos configuráveis aqui existem para a ablação, e `1/1/1` reproduz o artigo.
Vale lembrar que somar sem peso só faz sentido porque as três saídas foram
treinadas juntas, com autodestilação: em sistemas treinados separadamente, as
escalas não são comparáveis, e é daí que vem a necessidade de fusão por posto.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from backends import EmbedderHash, produto_interno
from config import SETTINGS, Settings
from corpus import Chunk, tokenizar
from esparso import VetorEsparso


@dataclass
class RepresentacaoM3:
    """As três saídas de uma passada, como o artigo as entrega."""

    denso: list[float] = field(default_factory=list)
    lexical: VetorEsparso = field(default_factory=VetorEsparso)
    multivetor: list[list[float]] = field(default_factory=list)

    @property
    def tem_multivetor(self) -> bool:
        return bool(self.multivetor)


# --------------------------------------------------------------------------- #
# As três funções de score
# --------------------------------------------------------------------------- #


def score_denso(q: RepresentacaoM3, p: RepresentacaoM3) -> float:
    """⟨e_p, e_q⟩ com os vetores já normalizados."""
    if not q.denso or not p.denso:
        return 0.0
    return produto_interno(q.denso, p.denso)


def score_lexical(q: RepresentacaoM3, p: RepresentacaoM3) -> float:
    """
    Σ_{t ∈ q ∩ p} w_qt · w_pt, restrito à interseção.

    O produto interno de VetorEsparso já faz exatamente isso: termo ausente
    conta como zero, e o laço corre sobre o vetor menor. A restrição à
    interseção não é uma otimização, é a definição da função no artigo.
    """
    return q.lexical.produto_interno(p.lexical)


def score_multivetor(q: RepresentacaoM3, p: RepresentacaoM3) -> float:
    """
    (1/N) Σ_i max_j E_q[i] · E_p[j]ᵀ, a interação tardia.

    Para cada token da consulta, o melhor casamento com algum token da
    passagem; a média sobre os tokens da consulta. É o que permite que um termo
    da consulta encontre seu correspondente em qualquer posição da passagem,
    sem que a representação precise comprimir o texto todo em um ponto.
    """
    if not q.multivetor or not p.multivetor:
        return 0.0

    total = 0.0
    for vetor_q in q.multivetor:
        total += max(produto_interno(vetor_q, vetor_p) for vetor_p in p.multivetor)
    return total / len(q.multivetor)


def score_integrado(q: RepresentacaoM3, p: RepresentacaoM3,
                    peso_denso: float = 1.0, peso_lexical: float = 1.0,
                    peso_multivetor: float = 1.0) -> float:
    """s_inter. Com os três pesos em 1,0, é a soma do artigo."""
    return (
        peso_denso * score_denso(q, p)
        + peso_lexical * score_lexical(q, p)
        + peso_multivetor * score_multivetor(q, p)
    )


def scores_separados(q: RepresentacaoM3, p: RepresentacaoM3) -> dict[str, float]:
    """As três parcelas, para inspecionar quem contribuiu com o quê."""
    return {
        "denso": score_denso(q, p),
        "lexical": score_lexical(q, p),
        "multivetor": score_multivetor(q, p),
    }


# --------------------------------------------------------------------------- #
# Backend local
# --------------------------------------------------------------------------- #


class M3Local:
    """
    Substituto determinístico das três funções. **Não é o BGE-M3.**

    O que reproduz: a forma das três saídas, as três fórmulas de score, a
    agregação por máximo em termo repetido no léxico, a normalização L2 do
    multivetor e o score integrado.

    O que não reproduz: a semântica. A parcela densa aqui é hashing de
    n-gramas, cega a sinônimo, que é justamente o trabalho da parcela densa de
    verdade. Então o resultado do backend `local` subestima o denso e, por
    consequência, superestima a importância relativa do léxico.

    Essa é a distorção mais importante a ter em mente ao ler a tabela: ela não
    é aleatória, ela empurra a conclusão em uma direção conhecida.
    """

    def __init__(self, chunks: list[Chunk], settings: Settings | None = None) -> None:
        self.settings = settings or SETTINGS
        self.nome = "m3-local"
        self.embedder = EmbedderHash(dim=self.settings.dim_densa)
        # Dimensão menor no multivetor: um vetor por token, e o custo de
        # armazenamento é o motivo pelo qual o artigo usa a função multivetor
        # só para reordenar candidatos.
        self.embedder_token = EmbedderHash(dim=max(32, self.settings.dim_densa // 8))

        # IDF do corpus, usada como substituto do peso aprendido do léxico.
        from collections import Counter

        df: Counter = Counter()
        for chunk in chunks:
            df.update(set(tokenizar(chunk.para_indexar())))
        n = max(1, len(chunks))
        self.idf = {t: math.log(1 + n / (1 + k)) for t, k in df.items()}
        self.idf_padrao = math.log(1 + n)

    def _lexical(self, texto: str) -> VetorEsparso:
        """
        ReLU de uma projeção aprendida, substituída por IDF saturada.

        Termo repetido mantém o maior peso, como o artigo determina. Aqui o
        peso de um termo não depende da posição, então "manter o máximo" é o
        mesmo que contar uma vez, e a implementação deixa isso explícito em
        vez de esconder a equivalência.
        """
        pesos: dict[str, float] = {}
        for token in tokenizar(texto):
            peso = math.log1p(self.idf.get(token, self.idf_padrao))
            pesos[token] = max(pesos.get(token, 0.0), peso)
        return VetorEsparso(pesos).normalizar_l2()

    def _multivetor(self, texto: str, max_tokens: int = 48) -> list[list[float]]:
        tokens = tokenizar(texto)[:max_tokens]
        return [self.embedder_token.codificar(t) for t in tokens]

    def codificar(self, texto: str, denso: bool = True, lexical: bool = True,
                  multivetor: bool = True) -> RepresentacaoM3:
        return RepresentacaoM3(
            denso=self.embedder.codificar(texto) if denso else [],
            lexical=self._lexical(texto) if lexical else VetorEsparso(),
            multivetor=self._multivetor(texto) if multivetor else [],
        )

    def codificar_documento(self, chunk: Chunk, **kwargs) -> RepresentacaoM3:
        return self.codificar(chunk.para_indexar(), **kwargs)


# --------------------------------------------------------------------------- #
# Backend HuggingFace
# --------------------------------------------------------------------------- #


class M3HF:
    """
    O BGE-M3 de verdade, via FlagEmbedding.

    O `BGEM3FlagModel` devolve as três representações na mesma passada, que é
    a multifuncionalidade do artigo: `dense_vecs`, `lexical_weights` e
    `colbert_vecs`. Os pesos léxicos vêm indexados por id de token, e a chave
    é convertida para string para caber no mesmo `VetorEsparso` que o resto do
    repositório usa.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or SETTINGS
        self.nome = f"m3-hf:{self.settings.modelo_m3}"
        self._modelo = None

    def _carregar(self) -> None:
        if self._modelo is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as erro:  # pragma: no cover - depende do ambiente
            raise RuntimeError(
                "backend 'hf' precisa de FlagEmbedding. Instale com:\n"
                "  pip install -r requirements-hf.txt\n"
                "Ou rode com BACKEND=local."
            ) from erro

        self._modelo = BGEM3FlagModel(
            self.settings.modelo_m3,
            use_fp16=self.settings.dispositivo != "cpu",
            devices=self.settings.dispositivo,
        )

    def codificar(self, texto: str, denso: bool = True, lexical: bool = True,
                  multivetor: bool = True) -> RepresentacaoM3:
        self._carregar()
        saida = self._modelo.encode(
            [texto],
            max_length=self.settings.max_tokens_modelo,
            return_dense=denso,
            return_sparse=lexical,
            return_colbert_vecs=multivetor,
        )

        vetor_denso = []
        if denso and saida.get("dense_vecs") is not None:
            vetor_denso = [float(x) for x in saida["dense_vecs"][0]]

        vetor_lexical = VetorEsparso()
        if lexical and saida.get("lexical_weights"):
            vetor_lexical = VetorEsparso({
                str(token): float(peso)
                for token, peso in saida["lexical_weights"][0].items()
                if float(peso) > 0.0
            })

        vetores_token = []
        if multivetor and saida.get("colbert_vecs") is not None:
            vetores_token = [
                [float(x) for x in linha] for linha in saida["colbert_vecs"][0]
            ]

        return RepresentacaoM3(
            denso=vetor_denso, lexical=vetor_lexical, multivetor=vetores_token
        )

    def codificar_documento(self, chunk: Chunk, **kwargs) -> RepresentacaoM3:
        return self.codificar(chunk.para_indexar(), **kwargs)


def obter_m3(chunks: list[Chunk], settings: Settings | None = None):
    settings = settings or SETTINGS
    settings.validar()
    if settings.backend == "hf":
        return M3HF(settings)
    return M3Local(chunks, settings)
