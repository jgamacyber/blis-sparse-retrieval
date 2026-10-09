"""
Corpus, chunking e tokenização.

Chunking por seção do Markdown, com janelas sobrepostas quando a seção é longa.
É o mesmo critério do módulo 06: cortar no meio de uma seção devolve contexto
mutilado, e a sobreposição existe porque uma definição pode cair exatamente na
emenda entre duas janelas.

A tokenização aqui serve ao BM25 e aos substitutos locais. No backend `hf`, a
tokenização é a do modelo, e esta não é usada no caminho de pontuação. Ela
continua servindo para o diagnóstico de cobertura lexical, que é o que separa
as consultas de termo exato das de paráfrase.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

DIR_CORPUS = Path("data/corpus")
CAMINHO_QA = Path("data/qa_eval.json")

# Stopwords do português, mais algumas do inglês que aparecem em nome de artigo.
STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos",
    "das", "em", "no", "na", "nos", "nas", "por", "pelo", "pela", "para", "com",
    "sem", "que", "e", "ou", "se", "ao", "aos", "ser", "sao", "eh", "ja", "la",
    "isso", "isto", "esse", "essa", "este", "esta", "meu", "minha", "seu", "sua",
    "mais", "menos", "muito", "pouco", "nao", "sim", "the", "of", "and", "to",
    "in", "for", "on", "is", "as", "at", "by", "it",
}

_TOKEN = re.compile(r"[a-z0-9@+]+")


def sem_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def tokenizar(texto: str, manter_stopwords: bool = False) -> list[str]:
    """
    Tokenização simples: minúsculas, sem acento, sem pontuação.

    Mantém dígitos e os sinais `@` e `+`, para que "Recall@5", "nDCG@10" e
    "normalize=True" não sejam picados em pedaços sem sentido. Termo exato é
    metade do assunto deste repositório, e picar o termo exato na tokenização
    seria sabotar o experimento antes de ele começar.
    """
    bruto = _TOKEN.findall(sem_acentos((texto or "").lower()))
    if manter_stopwords:
        return [t for t in bruto if len(t) > 1]
    return [t for t in bruto if t not in STOPWORDS and len(t) > 1]


# --------------------------------------------------------------------------- #
# Chunks
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Chunk:
    """Um trecho recuperável, com endereço suficiente para virar citação."""

    cid: str
    doc: str
    titulo: str
    secao: str
    texto: str

    @property
    def fonte(self) -> str:
        return f"{self.doc}#{self.secao}" if self.secao else self.doc

    def para_indexar(self) -> str:
        """Título e seção entram no texto indexado: eles carregam sinal."""
        return f"{self.titulo}\n{self.secao}\n{self.texto}"


def dividir_por_secao(texto: str, doc: str, tamanho: int = 900,
                      sobreposicao: int = 150) -> list[Chunk]:
    titulo = ""
    secao_atual = ""
    buffer: list[str] = []
    blocos: list[tuple[str, str]] = []

    for linha in (texto or "").splitlines():
        if linha.startswith("# "):
            titulo = linha[2:].strip()
            continue
        if linha.startswith("## "):
            if buffer:
                blocos.append((secao_atual, "\n".join(buffer).strip()))
                buffer = []
            secao_atual = linha[3:].strip()
            continue
        buffer.append(linha)

    if buffer:
        blocos.append((secao_atual, "\n".join(buffer).strip()))

    chunks: list[Chunk] = []
    for secao, corpo in blocos:
        if not corpo:
            continue

        if len(corpo) <= tamanho:
            janelas = [corpo]
        else:
            janelas = []
            passo = max(1, tamanho - sobreposicao)
            for inicio in range(0, len(corpo), passo):
                janela = corpo[inicio:inicio + tamanho].strip()
                if janela:
                    janelas.append(janela)
                if inicio + tamanho >= len(corpo):
                    break

        for janela in janelas:
            chunks.append(Chunk(
                cid=f"{doc}:{len(chunks):02d}",
                doc=doc,
                titulo=titulo,
                secao=secao,
                texto=janela,
            ))

    return chunks


def carregar_corpus(diretorio: str | Path = DIR_CORPUS, tamanho: int = 900,
                    sobreposicao: int = 150) -> list[Chunk]:
    diretorio = Path(diretorio)
    if not diretorio.exists():
        raise FileNotFoundError(
            f"corpus não encontrado em '{diretorio}'. Rode o comando de dentro "
            f"da pasta do repositório."
        )

    chunks: list[Chunk] = []
    for caminho in sorted(diretorio.glob("*.md")):
        chunks.extend(dividir_por_secao(
            caminho.read_text(encoding="utf-8"), caminho.stem, tamanho, sobreposicao
        ))

    if not chunks:
        raise ValueError(f"nenhum documento .md em '{diretorio}'")
    return chunks


def documentos(chunks: list[Chunk]) -> list[str]:
    vistos: list[str] = []
    for c in chunks:
        if c.doc not in vistos:
            vistos.append(c.doc)
    return vistos


# --------------------------------------------------------------------------- #
# Conjunto de avaliação
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Consulta:
    id: str
    consulta: str
    relevantes: tuple[str, ...]
    tipo: str = "exato"
    nota: str = ""


def carregar_consultas(caminho: str | Path = CAMINHO_QA) -> list[Consulta]:
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    return [
        Consulta(
            id=d["id"],
            consulta=d["consulta"],
            relevantes=tuple(d["relevantes"]),
            tipo=d.get("tipo", "exato"),
            nota=d.get("nota", ""),
        )
        for d in dados
    ]


def cobertura_lexical(consulta: str, textos: list[str]) -> float:
    """
    Fração dos termos da consulta que aparece no melhor dos textos relevantes.

    É o diagnóstico que sustenta a divisão do conjunto em dois tipos. Sem
    medir isso, dizer que uma consulta é "de paráfrase" é afirmação sobre a
    intenção de quem escreveu, não sobre o dado. Com a medida, a divisão passa
    a ser verificável, e o comando `python main.py consultas` a imprime.
    """
    termos = set(tokenizar(consulta))
    if not termos:
        return 0.0

    melhor = 0.0
    for texto in textos:
        presentes = set(tokenizar(texto))
        melhor = max(melhor, len(termos & presentes) / len(termos))
    return melhor
