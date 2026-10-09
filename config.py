"""
Configuração central do módulo bônus.

A diferença em relação aos módulos anteriores da trilha está no backend. Aqui os
modelos são SPLADE, BGE-M3 e bge-reranker-v2-m3, que têm peso e rodam
localmente: a OpenRouter não serve nenhum dos três, então não há chave de API
nem custo por consulta neste repositório. O que existe é download de modelo e
tempo de GPU ou de CPU.

Dois backends:

- `local`: substitutos determinísticos, sem dependência pesada, sem rede. Eles
  têm a MESMA forma de saída e passam pelo MESMO caminho de pontuação dos
  modelos reais, o que torna toda a mecânica testável de graça. Não são os
  modelos dos artigos e a diferença de qualidade é grande, como está dito em
  cada arquivo e repetido no relatório.
- `hf`: os modelos de verdade, via `transformers`. Exige instalar o extra e
  baixar os pesos.

Trocar de backend não muda uma linha do índice, da fusão, da reordenação nem
das métricas. Essa separação é o ponto do desenho: se ela não se sustentasse, os
testes offline não diriam nada sobre o caminho real.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

BACKENDS = ("local", "hf")


@dataclass(frozen=True)
class Settings:
    # --- Backend ---
    backend: str = field(default_factory=lambda: os.getenv("BACKEND", "local"))

    # --- Modelos do backend hf ---
    modelo_splade: str = field(
        default_factory=lambda: os.getenv(
            "MODELO_SPLADE", "naver/splade-cocondenser-ensembledistil"
        )
    )
    modelo_m3: str = field(
        default_factory=lambda: os.getenv("MODELO_M3", "BAAI/bge-m3")
    )
    modelo_reranker: str = field(
        default_factory=lambda: os.getenv("MODELO_RERANKER", "BAAI/bge-reranker-v2-m3")
    )
    dispositivo: str = field(default_factory=lambda: os.getenv("DISPOSITIVO", "cpu"))
    max_tokens_modelo: int = field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS_MODELO", "512"))
    )

    # --- Chunking ---
    tamanho_chunk: int = field(
        default_factory=lambda: int(os.getenv("TAMANHO_CHUNK", "900"))
    )
    sobreposicao: int = field(
        default_factory=lambda: int(os.getenv("SOBREPOSICAO", "150"))
    )

    # --- BM25 ---
    # 1.2 e 0.75 são os valores usuais do arcabouço probabilístico.
    bm25_k1: float = field(default_factory=lambda: float(os.getenv("BM25_K1", "1.2")))
    bm25_b: float = field(default_factory=lambda: float(os.getenv("BM25_B", "0.75")))

    # --- SPLADE ---
    # Quantos termos manter por vetor. É a poda que mantém a lista de postagem
    # curta; no artigo esse controle vem do regularizador durante o treino.
    splade_top_termos: int = field(
        default_factory=lambda: int(os.getenv("SPLADE_TOP_TERMOS", "96"))
    )
    # Peso da expansão em relação aos termos literais, no backend local.
    splade_peso_expansao: float = field(
        default_factory=lambda: float(os.getenv("SPLADE_PESO_EXPANSAO", "0.45"))
    )

    # --- BGE-M3 ---
    # O artigo soma as três parcelas sem peso. Os pesos aqui existem para a
    # ablação: 1/1/1 reproduz o artigo.
    m3_peso_denso: float = field(
        default_factory=lambda: float(os.getenv("M3_PESO_DENSO", "1.0"))
    )
    m3_peso_lexical: float = field(
        default_factory=lambda: float(os.getenv("M3_PESO_LEXICAL", "1.0"))
    )
    m3_peso_multivetor: float = field(
        default_factory=lambda: float(os.getenv("M3_PESO_MULTIVETOR", "1.0"))
    )
    dim_densa: int = field(default_factory=lambda: int(os.getenv("DIM_DENSA", "256")))

    # --- Fusão ---
    # k = 60 é o valor do artigo de 2009, não uma constante da natureza.
    rrf_k: int = field(default_factory=lambda: int(os.getenv("RRF_K", "60")))

    # --- Pipeline de dois estágios ---
    # Candidatos do primeiro estágio. O benchmark de tabelas mediu 20, 50 e 100
    # e a diferença foi enorme: é a variável mais subestimada da arquitetura.
    n_candidatos: int = field(
        default_factory=lambda: int(os.getenv("N_CANDIDATOS", "20"))
    )
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))

    def validar(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(
                f"BACKEND inválido: '{self.backend}'. Use um de {BACKENDS}."
            )

    @property
    def usa_modelos_reais(self) -> bool:
        return self.backend == "hf"


SETTINGS = Settings()


def resumo_config(settings: Settings | None = None) -> str:
    s = settings or SETTINGS
    return (
        f"backend={s.backend} | chunk={s.tamanho_chunk}/{s.sobreposicao} | "
        f"bm25 k1={s.bm25_k1} b={s.bm25_b} | rrf k={s.rrf_k} | "
        f"candidatos={s.n_candidatos} | top_k={s.top_k}"
    )


AVISO_LOCAL = (
    "backend 'local': substitutos determinísticos, não os modelos dos artigos. "
    "A mecânica é a mesma; a qualidade não. Para números comparáveis com os "
    "artigos, rode com BACKEND=hf."
)
