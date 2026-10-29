"""An index is tied to the model and chunking that built it; anything else must rebuild into a new index."""

import hashlib
import sqlite3

import numpy as np
import pytest

from conftest import index_state
from sourced.chunking import chunk_markdown
from sourced.embeddings import HashingEmbedder
from sourced.store import Index


class OtherModel(HashingEmbedder):
    """Same dimensions as the default test embedder, different vector space."""

    model_id = "other-test-model"

    def _vector(self, text):
        return np.roll(super()._vector(text), 7)


@pytest.fixture
def docs(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "deploy.md").write_text("# Deploy\n\n## Rollback\n\nRun kctl rollback to undo a release.\n")
    (folder / "laptops.md").write_text("# IT\n\n## Lost laptops\n\nReport a lost laptop to IT within 24 hours.\n")
    return folder


@pytest.fixture
def built(make_pipeline, docs):
    pipeline = make_pipeline()
    pipeline.ingest(docs)
    return pipeline


def reopen(make_pipeline, built, **kwargs):
    return make_pipeline(index_path=built.settings.index_path, **kwargs)


@pytest.mark.parametrize(
    "change",
    [
        {"embedder": OtherModel()},  # same dimensions, other model: can't be told apart by shape
        {"embedder": HashingEmbedder(128)},
        {"max_chunk_chars": 300},
    ],
    ids=["same-dimension-model", "other-dimension-model", "chunk-size"],
)
def test_ingest_with_another_configuration_is_refused_before_anything_changes(make_pipeline, built, docs, change):
    (docs / "deploy.md").write_text("# Deploy\n\n## Rollback\n\nRun kctl rollback --to <id>.\n")  # would be re-embedded
    before = index_state(built.settings.index_path)

    with pytest.raises(ValueError, match="SOURCED_INDEX"):
        reopen(make_pipeline, built, **change).ingest(docs)

    assert index_state(built.settings.index_path) == before


@pytest.mark.parametrize("embedder", [OtherModel(), HashingEmbedder(128)], ids=["same-dimension", "other-dimension"])
def test_dense_search_with_another_model_is_refused_but_keyword_search_works(make_pipeline, built, embedder):
    other = reopen(make_pipeline, built, embedder=embedder)

    for mode in ("dense", "hybrid"):
        with pytest.raises(ValueError, match="SOURCED_INDEX"):
            other.search("lost laptop", mode=mode)
    assert other.search("lost laptop", top_k=1, mode="bm25")[0].chunk.source == "laptops.md"


def test_an_index_from_an_older_chunker_is_refused(make_pipeline, built, docs):
    with sqlite3.connect(built.settings.index_path) as db:
        db.execute("UPDATE meta SET value = '1' WHERE key = 'chunker_version'")
    before = index_state(built.settings.index_path)

    with pytest.raises(ValueError, match="SOURCED_INDEX"):
        reopen(make_pipeline, built).ingest(docs)
    assert index_state(built.settings.index_path) == before


def test_a_failed_first_ingest_can_be_retried_with_the_same_configuration_only(make_pipeline, docs):
    (docs / "broken.md").write_bytes(b"\xff\xfe not UTF-8")
    pipeline = make_pipeline()
    with pytest.raises(UnicodeDecodeError):
        pipeline.ingest(docs)
    (docs / "broken.md").unlink()

    with pytest.raises(ValueError, match="SOURCED_INDEX"):
        reopen(make_pipeline, pipeline, embedder=OtherModel()).ingest(docs)
    assert pipeline.ingest(docs).added == ["deploy.md", "laptops.md"]


def test_vectors_that_dont_fit_the_index_are_never_written(built):
    chunks = chunk_markdown("new.md", "# New\n\n## One\n\nfirst\n\n## Two\n\nsecond\n")
    good = HashingEmbedder().embed_documents([c.indexed_text for c in chunks])
    nan = good.copy()
    nan[1, 3] = np.nan
    before = index_state(built.settings.index_path)

    for bad in (nan, good[:1], HashingEmbedder(128).embed_documents([c.indexed_text for c in chunks])):
        with pytest.raises(ValueError):
            built.index.replace_document("new.md", "sha", chunks, bad)

    assert index_state(built.settings.index_path) == before


# --- indexes without configuration (built before it was recorded) ---------------------------------


@pytest.fixture
def legacy(built):
    """What an index written by an older version looks like: documents, no meta table."""
    path = built.settings.index_path
    built.index.close()
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE meta")
    return path


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_opening_a_legacy_index_for_queries_writes_nothing(make_pipeline, legacy):
    before = file_hash(legacy)

    pipeline = make_pipeline(index_path=legacy)
    assert pipeline.search("lost laptop", top_k=1, mode="bm25")[0].chunk.source == "laptops.md"
    pipeline.index.close()

    assert file_hash(legacy) == before


def test_a_legacy_index_serves_keyword_search_only_and_refuses_ingest(make_pipeline, legacy, docs):
    pipeline = make_pipeline(index_path=legacy)
    before = index_state(legacy)

    for mode in ("dense", "hybrid"):  # nothing says which model wrote its vectors
        with pytest.raises(ValueError, match="SOURCED_INDEX"):
            pipeline.search("lost laptop", mode=mode)
    with pytest.raises(ValueError, match="older version"):
        pipeline.ingest(docs)

    assert index_state(legacy) == before


def test_a_legacy_index_opened_read_only_by_sqlite_still_answers_keyword_queries(legacy):
    from sourced.retrieval import Retriever

    index = Index(legacy, readonly=True)  # SQLite refuses every write on this connection
    retriever = Retriever(index, HashingEmbedder())

    assert retriever.search("lost laptop", top_k=1, mode="bm25")[0].chunk.source == "laptops.md"
    assert index.sources() == ["deploy.md", "laptops.md"]
    with pytest.raises(ValueError, match="SOURCED_INDEX"):
        retriever.search("lost laptop", mode="dense")
    index.close()


def test_a_new_index_is_only_created_by_ingest(make_pipeline, docs):
    pipeline = make_pipeline()

    assert pipeline.search("rollback", mode="hybrid") == []
    assert pipeline.index.sources() == [] and pipeline.index.chunk_count() == 0
    with sqlite3.connect(pipeline.settings.index_path) as db:
        assert db.execute("SELECT name FROM sqlite_master").fetchall() == []

    pipeline.ingest(docs)
    assert pipeline.search("rollback", top_k=1, mode="hybrid")[0].chunk.source == "deploy.md"
