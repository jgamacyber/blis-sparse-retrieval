"""
SPLADE: esparsidade aprendida, com os dois backends.

Referência: Formal, T., Lassance, C., Piwowarski, B. e Clinchant, S. (2021).
*SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval.*

As três peças do artigo que este arquivo implementa:

1. **A transformação por termo.** Dos logits da cabeça de linguagem mascarada,
   um peso não negativo e saturado:

       w_ij = transform(h_i)ᵀ E_j + b_j
       contribuição = log(1 + ReLU(w_ij))

   O ReLU zera o que é negativo, e daí vem a esparsidade. O logaritmo satura,
   pelo mesmo motivo que o `k1` do BM25 satura frequência.

2. **A agregação por máximo**, que é a mudança do v2 em relação ao v1:

       w_j = max  log(1 + ReLU(w_ij))        em vez de    Σ  log(1 + ReLU(w_ij))
             i∈t                                         i∈t

   Com soma, um termo repetido acumula peso e a repetição volta a inflar o
   score. Com máximo, o peso é o da ocorrência mais informativa. A função
   `agregar` abaixo faz as duas, porque comparar as duas é um experimento do
   módulo, não uma escolha fixa.

3. **O regularizador FLOPS**, que é como o artigo compra esparsidade durante o
   treino:

       ℓ_FLOPS = Σ ā_j²,   com   ā_j = (1/N) Σ w_j^(d_i)
                 j∈V                        i

   Aqui ele é **medido**, não minimizado: este repositório não treina nada.
   Medi-lo serve para comparar o custo de dois esquemas de peso no mesmo
   corpus, e para mostrar o que a poda compra.

Sobre a variante SPLADE-doc, que expande só o documento: ela está disponível
por flag. A troca é real: não rodar o modelo sobre a consulta economiza a
latência de inferência em tempo de busca e perde parte do ganho.
"""

from __future__ import annotations

from dataclasses import dataclass

from backends import ExpansorCoocorrencia
from config import SETTINGS, Settings
from corpus import Chunk, tokenizar
from esparso import VetorEsparso

AGREGACOES = ("max", "soma")


def transformar(logit: float) -> float:
    """log(1 + ReLU(x)): não negativo e saturado."""
    import math

    return math.log1p(logit) if logit > 0.0 else 0.0


def agregar(contribuicoes: dict[str, list[float]],
            modo: str = "max") -> dict[str, float]:
    """
    Agrega as contribuições de cada posição de entrada em um peso por termo.

    `max` é o SPLADE v2, `soma` é o v1. A função existe nos dois modos porque a
    diferença entre eles é um dos experimentos deste módulo, e porque a troca
    de soma por máximo é o tipo de detalhe que soa irrelevante e não é.
    """
    if modo not in AGREGACOES:
        raise ValueError(f"agregação inválida: '{modo}'. Use uma de {AGREGACOES}.")

    if modo == "max":
        return {t: max(v) for t, v in contribuicoes.items() if v}
    return {t: sum(v) for t, v in contribuicoes.items() if v}


def flops(vetores: list[VetorEsparso], normalizar: bool = False) -> float:
    """
    ℓ_FLOPS = Σ_j ā_j², com ā_j a ativação média do termo j no conjunto.

    O quadrado é a parte que importa: ele penaliza desproporcionalmente o termo
    que fica ativo em muitos documentos, que é o que faz uma lista de postagem
    crescer e a busca desacelerar. A penalidade não é por usar muitos termos,
    é por usar termos que destroem a estrutura do índice invertido.

    **Sobre `normalizar`, e por que ele existe.** O valor cru depende da escala
    dos pesos, então comparar o FLOPS de dois esquemas de peso diferentes mede
    a escala, não a estrutura. Os pesos BM25 deste corpus chegam à casa das
    unidades por causa do IDF; os do substituto do SPLADE ficam abaixo de 1,3.
    Comparado cru, o BM25 parece mais caro, o que é falso: ele tem metade das
    postagens.

    No artigo isso não aparece porque o regularizador é aplicado ao próprio
    modelo durante o treino, sempre na mesma escala. Para comparação entre
    esquemas, `normalizar=True` normaliza cada vetor em L2 antes de agregar, e
    aí o número fala da estrutura do índice.
    """
    if not vetores:
        return 0.0

    if normalizar:
        vetores = [v.normalizar_l2() for v in vetores]

    n = len(vetores)
    soma_por_termo: dict[str, float] = {}
    for vetor in vetores:
        for termo, peso in vetor.pesos.items():
            soma_por_termo[termo] = soma_por_termo.get(termo, 0.0) + peso

    return sum((s / n) ** 2 for s in soma_por_termo.values())


# --------------------------------------------------------------------------- #
# Backend local
# --------------------------------------------------------------------------- #


class SpladeLocal:
    """
    Substituto determinístico. **Não é o SPLADE.**

    O que ele reproduz: a forma do vetor (pesos não negativos sobre termos,
    incluindo termos ausentes do texto), a transformação saturada, a agregação
    por máximo, a poda e o caminho de pontuação por produto interno.

    O que ele não reproduz: de onde vem a expansão. Aqui ela vem da
    coocorrência **neste** corpus; no artigo vem do que o transformador
    aprendeu em pré-treino. A consequência prática é que este substituto não
    liga termos que nunca apareceram juntos no corpus, e o SPLADE liga.
    """

    def __init__(self, chunks: list[Chunk], settings: Settings | None = None,
                 agregacao: str = "max") -> None:
        self.settings = settings or SETTINGS
        self.agregacao = agregacao
        self.expansor = ExpansorCoocorrencia(
            [c.para_indexar() for c in chunks]
        )
        self.nome = "splade-local"

    def _contribuicoes(self, texto: str, expandir: bool) -> dict[str, list[float]]:
        tokens = tokenizar(texto)
        contribuicoes: dict[str, list[float]] = {}

        for token in tokens:
            # Termo literal: o logit alto é o que mantém o casamento exato
            # dominando a expansão. Essa ordem de grandeza é o que impede o
            # modelo de "esquecer" o termo que o usuário realmente pediu.
            contribuicoes.setdefault(token, []).append(transformar(2.5))

            if not expandir:
                continue

            for vizinho, forca in self.expansor.expandir(token):
                logit = 2.5 * self.settings.splade_peso_expansao * forca
                contribuicoes.setdefault(vizinho, []).append(transformar(logit))

        return contribuicoes

    def codificar(self, texto: str, expandir: bool = True,
                  podar: bool | None = None) -> VetorEsparso:
        pesos = agregar(self._contribuicoes(texto, expandir), self.agregacao)
        vetor = VetorEsparso(pesos)
        if podar is not False:
            vetor = vetor.podar(self.settings.splade_top_termos)
        return vetor

    def codificar_documento(self, chunk: Chunk) -> VetorEsparso:
        return self.codificar(chunk.para_indexar(), expandir=True)

    def codificar_consulta(self, consulta: str,
                           expandir: bool = True) -> VetorEsparso:
        """
        Consulta expandida ou não.

        `expandir=False` é a variante SPLADE-doc: a expansão fica só no
        documento, e a consulta vira casamento literal. Mais barato na busca,
        porque não há inferência em tempo de consulta, e menos eficaz.
        """
        vetor = self.codificar(consulta, expandir=expandir, podar=False)
        # Consulta sempre podada com mais pressão: cada termo ativo na consulta
        # é uma lista de postagem a percorrer. É o motivo de o artigo usar um
        # coeficiente de regularização maior para a consulta.
        return vetor.podar(max(8, self.settings.splade_top_termos // 3))


# --------------------------------------------------------------------------- #
# Backend HuggingFace
# --------------------------------------------------------------------------- #


class SpladeHF:
    """
    O SPLADE de verdade, via `transformers`.

    Caminho: tokenizar, passar pelo modelo de linguagem mascarada, pegar os
    logits sobre o vocabulário, aplicar log(1 + ReLU), agregar por máximo sobre
    as posições válidas e mapear os índices de vocabulário de volta para
    tokens.

    O mascaramento de atenção importa: sem ele, as posições de preenchimento
    entram no máximo e contaminam o vetor com termos que não vieram do texto.
    """

    def __init__(self, settings: Settings | None = None,
                 agregacao: str = "max") -> None:
        self.settings = settings or SETTINGS
        self.agregacao = agregacao
        self.nome = f"splade-hf:{self.settings.modelo_splade}"
        self._carregado = False
        self._tokenizer = None
        self._modelo = None
        self._torch = None

    def _carregar(self) -> None:
        if self._carregado:
            return
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as erro:  # pragma: no cover - depende do ambiente
            raise RuntimeError(
                "backend 'hf' precisa de torch e transformers. Instale com:\n"
                "  pip install -r requirements-hf.txt\n"
                "Ou rode com BACKEND=local, que não precisa de nada disso."
            ) from erro

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.settings.modelo_splade)
        self._modelo = AutoModelForMaskedLM.from_pretrained(
            self.settings.modelo_splade
        ).to(self.settings.dispositivo).eval()
        self._carregado = True

    def codificar(self, texto: str, expandir: bool = True,
                  podar: bool | None = None) -> VetorEsparso:
        self._carregar()
        torch = self._torch

        entradas = self._tokenizer(
            texto, return_tensors="pt", truncation=True,
            max_length=self.settings.max_tokens_modelo,
        ).to(self.settings.dispositivo)

        with torch.no_grad():
            logits = self._modelo(**entradas).logits        # (1, seq, |V|)

        ativado = torch.log1p(torch.relu(logits))

        # Mascara as posições de preenchimento antes de agregar.
        mascara = entradas["attention_mask"].unsqueeze(-1)   # (1, seq, 1)
        ativado = ativado * mascara

        if self.agregacao == "max":
            agregado = ativado.max(dim=1).values.squeeze(0)
        else:
            agregado = ativado.sum(dim=1).squeeze(0)

        indices = torch.nonzero(agregado).squeeze(-1)
        pesos = agregado[indices]

        vocabulario = self._tokenizer.convert_ids_to_tokens(indices.tolist())
        vetor = VetorEsparso({
            token: float(peso)
            for token, peso in zip(vocabulario, pesos.tolist())
            if peso > 0.0
        })

        if podar is not False:
            vetor = vetor.podar(self.settings.splade_top_termos)
        return vetor

    def codificar_documento(self, chunk: Chunk) -> VetorEsparso:
        return self.codificar(chunk.para_indexar())

    def codificar_consulta(self, consulta: str,
                           expandir: bool = True) -> VetorEsparso:
        if not expandir:
            # SPLADE-doc: a consulta não passa pelo modelo. Em vez disso, os
            # termos literais entram com peso 1, e todo o trabalho fica no
            # vetor do documento. Score = Σ_{j∈q} w_j^d, como no artigo.
            return VetorEsparso({t: 1.0 for t in set(tokenizar(consulta))})

        vetor = self.codificar(consulta, podar=False)
        return vetor.podar(max(8, self.settings.splade_top_termos // 3))


# --------------------------------------------------------------------------- #
# Fábrica
# --------------------------------------------------------------------------- #


@dataclass
class Diagnostico:
    """Para mostrar o que a expansão acrescentou, termo a termo."""

    literais: list[tuple[str, float]]
    expandidos: list[tuple[str, float]]
    l0: int
    flops: float


def obter_splade(chunks: list[Chunk], settings: Settings | None = None,
                 agregacao: str = "max"):
    settings = settings or SETTINGS
    settings.validar()
    if settings.backend == "hf":
        return SpladeHF(settings, agregacao)
    return SpladeLocal(chunks, settings, agregacao)


def diagnosticar(modelo, texto: str) -> Diagnostico:
    """
    Separa, no vetor produzido, o que é termo literal do que é expansão.

    É o diagnóstico que torna a expansão visível: sem ele, o vetor é só uma
    lista de números, e não se vê que o modelo acrescentou "coração" a um
    documento que só dizia "cardíaca".
    """
    vetor = modelo.codificar(texto)
    presentes = set(tokenizar(texto))

    literais, expandidos = [], []
    for termo, peso in vetor.top(len(vetor)):
        # No backend hf os tokens podem vir com marcação de subpalavra.
        limpo = termo.lstrip("#").lstrip("Ġ").lower()
        if limpo in presentes or termo.lower() in presentes:
            literais.append((termo, peso))
        else:
            expandidos.append((termo, peso))

    return Diagnostico(
        literais=literais,
        expandidos=expandidos,
        l0=vetor.l0,
        flops=flops([vetor]),
    )
