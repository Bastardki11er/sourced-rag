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
from pathlib import Path

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
"""

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
    terms = [t for t in re.findall(r"[A-Za-z0-9]+", text) if t.lower() not in _STOPWORDS]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


class Index:
    """Thread-safe: the API serves requests from a thread pool, so every access to the
    shared SQLite connection and the cached vector matrix goes through one lock."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()
        self._vectors: tuple[np.ndarray, list[str]] | None = None

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _query(self, sql: str, params: tuple | list = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    # --- writes ---------------------------------------------------------------

    def replace_document(self, source: str, sha256: str, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Atomically swaps all chunks of one document, so re-ingesting is idempotent."""
        with self._lock, self._db:
            self._delete(source)
            self._db.execute("INSERT INTO documents (source, sha256) VALUES (?, ?)", (source, sha256))
            self._db.executemany(
                "INSERT INTO chunks (id, source, title, section, text, position, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (c.id, c.source, c.title, c.section, c.text, c.position, e.astype(np.float32).tobytes())
                    for c, e in zip(chunks, embeddings, strict=True)
                ],
            )
            self._db.executemany(
                "INSERT INTO chunks_fts (chunk_id, heading, text) VALUES (?, ?, ?)",
                [(c.id, c.heading, c.text) for c in chunks],
            )
            self._vectors = None

    def remove_document(self, source: str) -> None:
        with self._lock, self._db:
            self._delete(source)
            self._vectors = None

    def _delete(self, source: str) -> None:
        self._db.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE source = ?)", (source,))
        self._db.execute("DELETE FROM documents WHERE source = ?", (source,))

    # --- reads ----------------------------------------------------------------

    def document_hash(self, source: str) -> str | None:
        rows = self._query("SELECT sha256 FROM documents WHERE source = ?", (source,))
        return rows[0][0] if rows else None

    def sources(self) -> list[str]:
        return [row[0] for row in self._query("SELECT source FROM documents ORDER BY source")]

    def chunk_count(self) -> int:
        return self._query("SELECT COUNT(*) FROM chunks")[0][0]

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
            if self._vectors is None:
                rows = self._db.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
                matrix = (
                    np.stack([np.frombuffer(row[1], dtype=np.float32) for row in rows])
                    if rows
                    else np.zeros((0, 0), dtype=np.float32)
                )
                self._vectors = (matrix, [row[0] for row in rows])
            return self._vectors
