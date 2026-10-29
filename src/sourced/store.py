"""SQLite-backed index: chunk text, an FTS5 full-text index for BM25, and embeddings.

One file, no server. Dense search loads every vector into a NumPy matrix and
does a single matrix-vector product, which takes well under a millisecond for
tens of thousands of chunks. Past a few hundred thousand chunks this is where
a real vector index (HNSW via pgvector, FAISS, ...) would take over.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows: ingest refuses to run rather than run unlocked (see Index.ingest_lock)
    fcntl = None

import numpy as np

from .chunking import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    source TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id        TEXT PRIMARY KEY,
    source    TEXT NOT NULL REFERENCES documents(source) ON DELETE CASCADE,
    title     TEXT NOT NULL,
    section   TEXT NOT NULL,
    text      TEXT NOT NULL,
    position  INTEGER NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED, heading, text, tokenize = 'porter unicode61'
);
-- 'root': the folder this index's documents come from; plus the embedding model, vector
-- dimensions and chunking that built it (see Pipeline.ingest).
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

REBUILD = (
    "Build a new index instead: SOURCED_INDEX=<new file> sourced ingest <folder>, check that it finished "
    "without errors and that searching it works, keep the old file as a backup, then point SOURCED_INDEX at "
    "the new file and restart the server."
)


class IndexConfigError(ValueError):
    """The index was built with another embedding model or chunking than this process uses."""


class IndexBusyError(ValueError):
    """Another ingest holds this index."""


# Words that match nearly every chunk and only add noise to an OR query.
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it my of on or should the "
    "there this to was we what when where which who why will with you your".split()
)


def to_fts_query(text: str) -> str:
    """Turns free text into a safe FTS5 query: quoted terms joined with OR.

    Quoting stops user input such as ``KR-4021`` or ``AND`` from being parsed as
    FTS5 syntax, and OR (rather than the default AND) means one rare matching
    term is enough to surface a chunk; BM25 then ranks by how many terms match.
    """
    # Letters and digits in any script ("café", "naïve"), after NFC so a base letter plus a
    # combining accent becomes one character. Not \w: a lone "_" would be an empty phrase.
    terms = [t for t in re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text)) if t.lower() not in _STOPWORDS]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


class Index:
    """Thread-safe: the API serves requests from a thread pool, so every access to the
    shared SQLite connection and the cached vector matrix goes through one lock.

    The vector cache also follows changes made by other processes (``sourced ingest``
    while ``sourced serve`` runs): SQLite's ``PRAGMA data_version`` changes on this
    connection whenever another connection commits, and the cache is rebuilt then.
    """

    def __init__(self, path: Path, readonly: bool = False) -> None:
        """Opens the index without changing it: the tables are created by ``initialize``, which
        only ingest calls. Until then every read returns nothing. ``readonly`` makes SQLite
        itself refuse writes on this connection."""
        self.path = path
        if readonly:
            self._db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        self._vectors: tuple[np.ndarray, list[str]] | None = None
        self._vectors_version: int | None = None  # data_version the cache was read at

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _query(self, sql: str, params: tuple | list = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, params).fetchall() if self._has("chunks") else []

    def _has(self, table: str) -> bool:
        return self._db.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone() is not None

    @contextmanager
    def ingest_lock(self) -> Iterator[None]:
        """Lets one ingest at a time write this index, across processes.

        An OS lock (flock) on ``<index>.lock`` next to the resolved index path, so every
        writer contends for the same file, also via symlinks or relative paths. Each call
        opens the file anew, so two Pipelines in one process exclude each other too. The OS
        drops the lock when the file is closed, on an exception, or when the process dies;
        the file itself is left in place (deleting it could let two writers lock two files).
        """
        if fcntl is None:
            raise RuntimeError("sourced ingest needs OS file locks (fcntl), which this platform doesn't provide")
        lock_path = self.path.resolve().with_name(self.path.resolve().name + ".lock")
        with open(lock_path, "a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise IndexBusyError(
                    f"another sourced ingest is running on {self.path.resolve()}; run this one after it has finished"
                ) from None
            yield

    # --- writes (only ingest calls these, holding ingest_lock) ------------------

    def initialize(self, root: Path, config: dict[str, str]) -> None:
        """Creates the tables and records the folder and configuration this index is built with."""
        with self._lock:
            self._db.executescript(_SCHEMA)
            with self._db:
                self._db.executemany(
                    "INSERT INTO meta (key, value) VALUES (?, ?)", [("root", str(root)), *config.items()]
                )

    def replace_document(self, source: str, sha256: str, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Atomically swaps all chunks of one document, so re-ingesting is idempotent.

        Refuses vectors that don't fit the index (count, recorded dimensions, NaN/inf).
        """
        vectors = np.asarray(embeddings, dtype=np.float32)
        with self._lock, self._db:
            row = self._db.execute("SELECT value FROM meta WHERE key = 'dimensions'").fetchone()
            if row is None:
                raise IndexConfigError(f"{self.path} records no vector dimensions. {REBUILD}")
            if chunks and (vectors.shape != (len(chunks), int(row[0])) or not np.isfinite(vectors).all()):
                raise ValueError(
                    f"{source}: expected {len(chunks)} finite vectors of {row[0]} dimensions, got shape {vectors.shape}"
                )
            self._delete(source)
            self._db.execute("INSERT INTO documents (source, sha256) VALUES (?, ?)", (source, sha256))
            self._db.executemany(
                "INSERT INTO chunks (id, source, title, section, text, position, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (c.id, c.source, c.title, c.section, c.text, c.position, e.tobytes())
                    for c, e in zip(chunks, vectors, strict=True)
                ],
            )
            self._db.executemany(
                "INSERT INTO chunks_fts (chunk_id, heading, text) VALUES (?, ?, ?)",
                [(c.id, c.heading, c.text) for c in chunks],
            )
            self._vectors = None

    def remove_documents(self, sources: list[str]) -> None:
        """Deletes documents with their chunks and FTS rows, in one transaction."""
        with self._lock, self._db:
            for source in sources:
                self._delete(source)
            self._vectors = None

    def _delete(self, source: str) -> None:
        self._db.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE source = ?)", (source,))
        self._db.execute("DELETE FROM documents WHERE source = ?", (source,))

    # --- reads ----------------------------------------------------------------

    def document_hash(self, source: str) -> str | None:
        rows = self._query("SELECT sha256 FROM documents WHERE source = ?", (source,))
        return rows[0][0] if rows else None

    def meta(self) -> dict[str, str]:
        """The recorded folder and configuration; empty for a new index or one built before they were recorded."""
        with self._lock:
            return dict(self._db.execute("SELECT key, value FROM meta").fetchall()) if self._has("meta") else {}

    def root(self) -> Path | None:
        root = self.meta().get("root")
        return Path(root) if root else None

    def vector_space_problem(self, model_id: str, dimensions: int) -> str | None:
        """Why this index's vectors can't be compared with ``model_id`` vectors, or None if they can."""
        meta = self.meta()
        if (meta.get("embedding_model"), meta.get("dimensions")) == (model_id, str(dimensions)):
            return None
        if self.chunk_count() == 0:
            return None  # nothing to compare against yet
        if "embedding_model" not in meta:
            return (
                f"{self.path} doesn't record which embedding model wrote its vectors (it was built by an older "
                f"version of sourced), so dense and hybrid search can't be trusted; keyword search (bm25) still "
                f"works. {REBUILD}"
            )
        return (
            f"{self.path} holds vectors from {meta['embedding_model']} ({meta.get('dimensions')} dimensions), "
            f"but queries are embedded with {model_id} ({dimensions} dimensions). {REBUILD}"
        )

    def sources(self) -> list[str]:
        return [row[0] for row in self._query("SELECT source FROM documents ORDER BY source")]

    def chunk_count(self) -> int:
        rows = self._query("SELECT COUNT(*) FROM chunks")
        return rows[0][0] if rows else 0

    def bm25(self, query: str, limit: int) -> list[str]:
        fts_query = to_fts_query(query)
        if not fts_query:
            return []
        # bm25() weights: chunk_id (unindexed), heading 2x, body 1x. Lower is better.
        rows = self._query(
            "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts, 0.0, 2.0, 1.0) LIMIT ?",
            (fts_query, limit),
        )
        return [row[0] for row in rows]

    def dense(self, query_vector: np.ndarray, limit: int) -> list[str]:
        matrix, ids = self._load_vectors()
        if not ids:
            return []
        scores = matrix @ query_vector
        top = np.argsort(-scores)[:limit]
        return [ids[i] for i in top]

    def get(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._query(
            f"SELECT id, source, title, section, text, position FROM chunks WHERE id IN ({placeholders})", ids
        )
        by_id = {row[0]: Chunk(*row) for row in rows}
        return [by_id[i] for i in ids if i in by_id]

    def _load_vectors(self) -> tuple[np.ndarray, list[str]]:
        with self._lock:
            # Version first, vectors second. A commit landing in between leaves the cache
            # labelled with the older version, so the next call reloads; the other order
            # could label old vectors with the new version and keep them for good.
            # This connection's own writes don't change data_version; they reset the cache.
            version = self._db.execute("PRAGMA data_version").fetchone()[0]
            if self._vectors is None or version != self._vectors_version:
                rows = self._read_vectors()
                matrix = (
                    np.stack([np.frombuffer(row[1], dtype=np.float32) for row in rows])
                    if rows
                    else np.zeros((0, 0), dtype=np.float32)
                )
                self._vectors, self._vectors_version = (matrix, [row[0] for row in rows]), version
            return self._vectors

    def _read_vectors(self) -> list[tuple]:
        if not self._has("chunks"):
            return []
        return self._db.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
