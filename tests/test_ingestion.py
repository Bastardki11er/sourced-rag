import sqlite3
from dataclasses import replace

import numpy as np
import pytest

from sourced.chunking import chunk_markdown
from sourced.store import Index


def test_missing_input_does_not_succeed(make_pipeline, tmp_path):
    with pytest.raises(FileNotFoundError):
        make_pipeline().ingest(tmp_path / "missing")


def test_unsupported_file_is_rejected(make_pipeline, tmp_path):
    path = tmp_path / "image.png"
    path.write_text("not documentation")
    with pytest.raises(ValueError, match="Unsupported"):
        make_pipeline().ingest(path)


def test_uppercase_extensions_and_directories(make_pipeline, tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "nested.md").mkdir()
    (root / "POLICY.MD").write_text("# Policy\nUse approved tools.")
    report = make_pipeline().ingest(root)
    assert report.added == ["POLICY.MD"]


def test_chunk_configuration_change_reindexes(make_pipeline, tmp_path):
    path = tmp_path / "policy.md"
    path.write_text("# Policy\n" + "alpha beta gamma " * 100)
    pipeline = make_pipeline()
    pipeline.ingest(path)
    pipeline.settings = replace(pipeline.settings, max_chunk_chars=100)
    report = pipeline.ingest(path)
    assert report.updated == ["policy.md"]
    assert report.chunks > 1
    assert pipeline.ingest(path).unchanged == ["policy.md"]


def test_same_name_from_other_corpus_cannot_overwrite(make_pipeline, tmp_path):
    pipeline = make_pipeline()
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        (root / "policy.md").write_text(f"# Policy\n{name}")
    pipeline.ingest(tmp_path / "first")
    with pytest.raises(ValueError, match="corpus"):
        pipeline.ingest(tmp_path / "second")
    assert pipeline.search("first", mode="bm25")[0].chunk.text == "first"


@pytest.mark.parametrize("vectors", [np.ones((1, 3)), np.full((1, 256), np.nan), np.ones((2, 256))])
def test_bad_vectors_do_not_destroy_existing_document(make_pipeline, tmp_path, vectors):
    pipeline = make_pipeline()
    path = tmp_path / "policy.md"
    path.write_text("# Policy\nOriginal policy.")
    pipeline.ingest(path)
    pipeline.search("policy")  # populate the vector cache
    original_hash = pipeline.index.document_hash("policy.md")
    chunks = chunk_markdown("policy.md", "# Policy\nReplacement.")
    with pytest.raises(ValueError):
        pipeline.index.replace_document("policy.md", "new", chunks, vectors)
    assert pipeline.index.document_hash("policy.md") == original_hash
    assert pipeline.search("policy")[0].chunk.text == "Original policy."


def test_empty_document_is_indexed_without_embedding_call(make_pipeline, tmp_path, monkeypatch):
    pipeline = make_pipeline()
    path = tmp_path / "empty.md"
    path.write_text("")

    def unexpected(texts):
        raise AssertionError("An empty document must not call the embedder")

    monkeypatch.setattr(pipeline.embedder, "embed_documents", unexpected)
    assert pipeline.ingest(path).added == ["empty.md"]
    assert pipeline.index.chunk_count() == 0


def test_model_and_corpus_bindings_survive_restart(tmp_path):
    path = tmp_path / "index.db"
    index = Index(path)
    index.bind("embedding_model", "model-a")
    index.bind("corpus_root", "/docs/a")
    index.close()
    index = Index(path)
    try:
        index.bind("embedding_model", "model-a")
        with pytest.raises(ValueError, match="embedding_model"):
            index.bind("embedding_model", "model-b")
        with pytest.raises(ValueError, match="corpus_root"):
            index.bind("corpus_root", "/docs/b")
    finally:
        index.close()


def test_changed_document_removes_old_fts_and_vectors(make_pipeline, tmp_path):
    pipeline = make_pipeline()
    path = tmp_path / "policy.md"
    path.write_text("# Policy\nOldunique content.")
    pipeline.ingest(path)
    pipeline.search("Oldunique")
    path.write_text("# Policy\nNewunique content.")
    pipeline.ingest(path)
    assert pipeline.search("Oldunique", mode="bm25") == []
    assert pipeline.search("Newunique")[0].chunk.text == "Newunique content."


def test_failed_replacement_rolls_back_fts_and_vectors(make_pipeline, tmp_path):
    pipeline = make_pipeline()
    path = tmp_path / "policy.md"
    path.write_text("# Policy\nOriginalunique policy.")
    pipeline.ingest(path)
    pipeline.search("Originalunique")
    chunks = chunk_markdown("policy.md", "# Policy\nReplacement.")
    # Duplicate IDs fail after deletion has begun, exercising SQLite rollback.
    with pytest.raises(sqlite3.IntegrityError):
        pipeline.index.replace_document("policy.md", "new", chunks * 2, np.ones((2, 256)))
    assert pipeline.search("Originalunique", mode="bm25")[0].chunk.text == "Originalunique policy."
    assert pipeline.search("Originalunique")[0].chunk.text == "Originalunique policy."


def test_document_symlink_cannot_import_outside_corpus(make_pipeline, tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("# Outside\nPrivate content.")
    (root / "link.md").symlink_to(outside)
    pipeline = make_pipeline()
    with pytest.raises(ValueError, match="symlink"):
        pipeline.ingest(root)
    assert pipeline.index.sources() == []


def test_legacy_index_requires_explicit_rebuild(tmp_path):
    path = tmp_path / "legacy.db"
    index = Index(path)
    chunks = chunk_markdown("policy.md", "# Policy\nExisting data.")
    index.replace_document("policy.md", "oldhash", chunks, np.ones((1, 256)))
    index.close()
    index = Index(path)
    try:
        with pytest.raises(ValueError, match="Legacy index"):
            index.bind("embedding_model", "new-model")
        assert index.sources() == ["policy.md"]
        assert index.document_hash("policy.md") == "oldhash"
    finally:
        index.close()
