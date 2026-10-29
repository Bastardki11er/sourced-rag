import sqlite3
from pathlib import Path

import pytest

from sourced.config import Settings
from sourced.embeddings import HashingEmbedder
from sourced.generation import ExtractiveAnswerer
from sourced.pipeline import Pipeline
from sourced.store import Index

HANDBOOK = Path(__file__).parent.parent / "data" / "handbook"


@pytest.fixture
def make_pipeline(tmp_path):
    """Builds a pipeline with a deterministic embedder, so tests never download a model."""
    created = []

    def make(answerer=None, reranker=None, index_path=None, embedder=None, max_chunk_chars=1200):
        """A new pipeline; pass another pipeline's index_path to open the same index file again."""
        from sourced.retrieval import Retriever

        settings = Settings(
            index_path=index_path or tmp_path / f"index-{len(created)}.db",
            rerank=False,
            max_chunk_chars=max_chunk_chars,
        )
        index = Index(settings.index_path)
        embedder = embedder or HashingEmbedder()
        pipeline = Pipeline(
            settings, index, embedder, answerer or ExtractiveAnswerer(), Retriever(index, embedder, reranker)
        )
        created.append(pipeline)
        return pipeline

    yield make
    for pipeline in created:
        pipeline.index.close()


def index_state(path):
    """Everything retrieval reads, to check that a refused operation changed nothing."""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        return [
            db.execute(sql).fetchall()
            for sql in (
                "SELECT * FROM documents ORDER BY source",
                "SELECT * FROM chunks ORDER BY id",
                "SELECT chunk_id, heading, text FROM chunks_fts ORDER BY chunk_id",
            )
        ]


@pytest.fixture
def handbook_pipeline(make_pipeline):
    pipeline = make_pipeline()
    pipeline.ingest(HANDBOOK)
    return pipeline
