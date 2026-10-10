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

    def make(answerer=None, reranker=None):
        from sourced.retrieval import Retriever

        settings = Settings(index_path=tmp_path / f"index-{len(created)}.db", rerank=False)
        index = Index(settings.index_path)
        embedder = HashingEmbedder()
        pipeline = Pipeline(
            settings, index, embedder, answerer or ExtractiveAnswerer(), Retriever(index, embedder, reranker)
        )
        created.append(pipeline)
        return pipeline

    yield make
    for pipeline in created:
        pipeline.index.close()


@pytest.fixture
def handbook_pipeline(make_pipeline):
    pipeline = make_pipeline()
    pipeline.ingest(HANDBOOK)
    return pipeline
