"""Wires chunking, indexing, retrieval and generation together."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import CHUNKER_VERSION, chunk_markdown
from .config import Settings
from .embeddings import Embedder
from .generation import Answer, Answerer
from .retrieval import Mode, Retriever, SearchResult
from .store import REBUILD, Index, IndexBusyError, IndexConfigError

__all__ = [
    "DocumentLinkError",
    "IndexBindingError",
    "IndexBusyError",
    "IndexConfigError",
    "Pipeline",
    "build_pipeline",
]

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}


class IndexBindingError(ValueError):
    """The path to ingest doesn't belong to this index's folder."""


class DocumentLinkError(ValueError):
    """The documents to ingest include a symbolic link, which ingest doesn't follow."""


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
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

    def ingest(self, path: Path) -> IngestReport:
        """Indexes a folder of Markdown/text files, or one file in it. Unchanged files are skipped by hash.

        An index holds one folder (see ``_folder``), and documents are named by their path
        inside it, so ingesting one file updates the same document a folder ingest would.
        A folder ingest also removes the documents whose files are gone, but only after
        every file in the folder was read and indexed: a missing path, an unreadable file
        or a failure part-way leaves them all in place. A single file never removes others.

        The whole run holds the index's ingest lock, so a second ingest (any process) fails
        with IndexBusyError instead of interleaving with this one. A failed run can leave
        some documents written: only a run that returns completes the index.

        Symbolic links inside the folder (files, directories, broken ones), or a single file
        that is a link, stop the run with DocumentLinkError before the index is touched, so
        a link can't pull in text from outside the folder. The folder itself may be reached
        through a link. Checked once up front: not a guard against files swapped mid-run.
        """
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        with self.index.ingest_lock():
            root = self.index.meta().get("root")
            _refuse_links(path, Path(root) if root else path.parent.resolve())
            return self._ingest(path, self._folder(path))

    def _ingest(self, path: Path, folder: Path) -> IngestReport:
        if path.is_file():
            files = [path.resolve()]
        else:
            files = sorted(p for p in folder.rglob("*") if p.suffix in SUPPORTED_SUFFIXES and p.is_file())
        report = IngestReport()
        for file in files:
            source = file.relative_to(folder).as_posix()
            text = file.read_text(encoding="utf-8")
            sha = hashlib.sha256(text.encode()).hexdigest()
            previous = self.index.document_hash(source)
            if previous == sha:
                report.unchanged.append(source)
                continue
            chunks = chunk_markdown(source, text, self.settings.max_chunk_chars)
            embeddings = self.embedder.embed_documents([c.indexed_text for c in chunks])
            self.index.replace_document(source, sha, chunks, embeddings)
            (report.added if previous is None else report.updated).append(source)
        if path.is_dir():
            seen = {*report.added, *report.updated, *report.unchanged}
            report.removed = [source for source in self.index.sources() if source not in seen]
            self.index.remove_documents(report.removed)
        report.chunks = self.index.chunk_count()
        return report

    def _config(self) -> dict[str, str]:
        """What the index's chunks and vectors depend on; ingest refuses an index built with anything else."""
        return {
            "embedding_model": self.embedder.model_id,
            "dimensions": str(self.embedder.dimensions),
            "max_chunk_chars": str(self.settings.max_chunk_chars),
            "chunker_version": CHUNKER_VERSION,
        }

    def _folder(self, path: Path) -> Path:
        """The folder this index belongs to; on a new index, creates it with this folder and configuration.

        Raises before anything is written. Without a binding, two folders that both have a
        README.md would overwrite each other's document, and removing "files that are gone"
        would delete the other folder's documents. Paths are compared after resolving them
        (absolute, no ``..``, symlinks followed), so a moved folder needs a new index.
        Runs under the ingest lock, so two first ingests can't both see an empty index.
        """
        wanted = (path if path.is_dir() else path.parent).resolve()
        meta = self.index.meta()
        if "root" not in meta:
            if self.index.sources():
                # Built by a version that didn't record its folder and model: refuse rather than
                # guess which documents belong where (and remove the wrong ones).
                raise IndexBindingError(
                    f"{self.settings.index_path} was built by an older version of sourced that didn't record which "
                    f"folder and embedding model it uses, so it can't be updated. Keyword search still works. {REBUILD}"
                )
            self.index.initialize(wanted, self._config())
            return wanted
        changed = [
            f"{key} {meta.get(key)} -> {value}" for key, value in self._config().items() if meta.get(key) != value
        ]
        if changed:
            raise IndexConfigError(
                f"{self.settings.index_path} was built with a different configuration ({'; '.join(changed)}). {REBUILD}"
            )
        bound = Path(meta["root"])
        inside = wanted.is_relative_to(bound) if path.is_file() else wanted == bound
        if not inside:
            raise IndexBindingError(
                f"{self.settings.index_path} holds the documents of {bound}. To index {wanted}, "
                f"use a separate index: SOURCED_INDEX=<another file> sourced ingest {path}"
            )
        return bound

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


def _refuse_links(path: Path, root: Path) -> None:
    """Raises DocumentLinkError for a link between ``root`` (the folder, however it was reached) and the documents."""
    if path.is_dir():
        links = sorted(p for p in path.resolve().rglob("*") if p.is_symlink())
    elif path.is_symlink():
        links = [path]
    else:
        # A single file: also every directory between it and the folder ("docs/shortcut/a.md").
        # A file outside the folder is left to the folder check, whatever links lie above it.
        links, between = [], []
        for parent in path.absolute().parents:  # absolute() keeps the links that resolve() would follow
            if parent.resolve() == root:
                links = [p for p in between if p.is_symlink()][-1:]
                break
            between.append(parent)
    if links:
        more = f" (and {len(links) - 1} more)" if len(links) > 1 else ""
        raise DocumentLinkError(
            f"{links[0]} is a symbolic link{more}. sourced doesn't follow links in the document folder; "
            "copy the file in or remove the link, then ingest again. Nothing was changed."
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
