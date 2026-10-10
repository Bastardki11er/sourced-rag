"""Wires chunking, indexing, retrieval and generation together."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_markdown
from .config import Settings
from .embeddings import Embedder
from .generation import Answer, Answerer
from .retrieval import Mode, Retriever, SearchResult
from .store import Index

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    chunks: int = 0


@dataclass
class AskResult:
    answer: Answer
    results: list[SearchResult]
    timings_ms: dict[str, float]


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        index: Index,
        embedder: Embedder,
        answerer: Answerer,
        retriever: Retriever | None = None,
    ) -> None:
        self.settings = settings
        self.index = index
        self.embedder = embedder
        self.answerer = answerer
        self.retriever = retriever or Retriever(index, embedder)

    def ingest(self, root: Path) -> IngestReport:
        """Indexes every Markdown/text file under ``root``. Unchanged files are skipped by content hash."""
        report = IngestReport()
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix in SUPPORTED_SUFFIXES)
        base = root.parent if root.is_file() else root
        for path in files:
            source = path.relative_to(base).as_posix()
            text = path.read_text(encoding="utf-8")
            sha = hashlib.sha256(text.encode()).hexdigest()
            previous = self.index.document_hash(source)
            if previous == sha:
                report.unchanged.append(source)
                continue
            chunks = chunk_markdown(source, text, self.settings.max_chunk_chars)
            embeddings = self.embedder.embed_documents([c.indexed_text for c in chunks])
            self.index.replace_document(source, sha, chunks, embeddings)
            (report.added if previous is None else report.updated).append(source)
        report.chunks = self.index.chunk_count()
        return report

    def search(
        self, query: str, top_k: int | None = None, mode: Mode = "hybrid", rerank: bool = False
    ) -> list[SearchResult]:
        return self.retriever.search(query, top_k or self.settings.top_k, mode, rerank)

    def ask(self, question: str, top_k: int | None = None, rerank: bool | None = None) -> AskResult:
        started = time.perf_counter()
        results = self.search(question, top_k, "hybrid", self.settings.rerank if rerank is None else rerank)
        retrieved = time.perf_counter()
        answer = self.answerer.answer(question, results)
        finished = time.perf_counter()
        return AskResult(
            answer,
            results,
            {
                "retrieval": round((retrieved - started) * 1000, 1),
                "generation": round((finished - retrieved) * 1000, 1),
            },
        )


def build_pipeline(settings: Settings, with_reranker: bool | None = None) -> Pipeline:
    """Production wiring: local embedding model, plus Claude when credentials are configured."""
    import anthropic

    from .embeddings import FastEmbedEmbedder
    from .generation import ClaudeAnswerer, ExtractiveAnswerer
    from .retrieval import CrossEncoderReranker

    index = Index(settings.index_path)
    embedder = FastEmbedEmbedder(settings.embedding_model)
    use_reranker = settings.rerank if with_reranker is None else with_reranker
    reranker = CrossEncoderReranker(settings.rerank_model) if use_reranker else None
    answerer: Answerer = (
        ClaudeAnswerer(anthropic.Anthropic(), settings.claude_model, settings.claude_effort)
        if settings.answerer == "claude"
        else ExtractiveAnswerer()
    )
    return Pipeline(settings, index, embedder, answerer, Retriever(index, embedder, reranker))
