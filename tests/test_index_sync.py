"""Keeping the index in step with the documents: other processes, deleted files, one folder per index."""

import sqlite3

import pytest

from sourced.pipeline import IndexBindingError


def write(folder, name, section, body):
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Doc\n\n## {section}\n\n{body}\n")
    return path


def texts(results):
    return [r.chunk.text for r in results]


@pytest.fixture
def docs(tmp_path):
    folder = tmp_path / "docs"
    write(folder, "deploy.md", "Rollback", "Run kctl rollback to undo a release.")
    write(folder, "laptops.md", "Lost laptops", "Report a lost laptop to IT within 24 hours.")
    return folder


@pytest.fixture
def server_and_cli(make_pipeline):
    """Two pipelines on one index file, like `sourced serve` and a later `sourced ingest`."""
    server = make_pipeline()
    cli = make_pipeline(index_path=server.settings.index_path)
    return server, cli


# --- 1. another process changes the index -------------------------------------------


def test_the_server_sees_a_document_reindexed_by_another_process(server_and_cli, docs):
    server, cli = server_and_cli
    cli.ingest(docs)
    assert server.search("rollback release", mode="hybrid")  # loads the server's vector cache

    write(docs, "deploy.md", "Rollback", "Run kctl rollback --to <id> to undo a release.")
    cli.ingest(docs)

    for mode in ("dense", "hybrid"):
        found = texts(server.search("rollback release", mode=mode))
        assert "Run kctl rollback --to <id> to undo a release." in found
        assert "Run kctl rollback to undo a release." not in found


def test_a_document_removed_by_another_process_leaves_search(server_and_cli, docs):
    server, cli = server_and_cli
    cli.ingest(docs)
    assert server.search("lost laptop", mode="dense")

    cli.index.remove_documents(["laptops.md"])

    assert all(r.chunk.source != "laptops.md" for r in server.search("lost laptop", mode="hybrid"))


def test_a_commit_while_the_cache_loads_is_not_mistaken_for_up_to_date(server_and_cli, docs):
    # The other process commits right after the server has read the vectors. The cache
    # then holds the old vectors; it must not be labelled with the new data version.
    server, cli = server_and_cli
    cli.ingest(docs)
    read_vectors = server.index._read_vectors

    def read_then_someone_commits():
        rows = read_vectors()
        write(docs, "deploy.md", "Rollback", "Rollback is now done with kctl revert.")
        cli.ingest(docs)
        return rows

    server.index._read_vectors = read_then_someone_commits
    server.search("rollback", mode="dense")
    server.index._read_vectors = read_vectors

    assert "Rollback is now done with kctl revert." in texts(server.search("rollback", mode="dense"))


def test_a_chunk_vanishing_mid_query_keeps_scores_with_their_own_chunks(make_pipeline, docs):
    class ScoresByLength:
        def score(self, query, passages):
            return [float(len(p)) for p in passages]

    pipeline = make_pipeline(reranker=ScoresByLength())
    pipeline.ingest(docs)
    get = pipeline.index.get
    gone = pipeline.search("laptop rollback", mode="hybrid")[0].chunk.id

    def get_after_a_delete(ids):  # as if another process deleted it between ranking and fetching
        return [c for c in get(ids) if c.id != gone]

    pipeline.index.get = get_after_a_delete

    for rerank in (False, True):
        results = pipeline.search("laptop rollback", mode="hybrid", rerank=rerank)
        assert results and all(r.chunk.id != gone for r in results)
    for r in pipeline.search("laptop rollback", mode="hybrid", rerank=True):
        assert r.score == len(r.chunk.indexed_text)  # each score belongs to the chunk it's attached to


# --- 2. deleted files ---------------------------------------------------------------


def fts_rows(pipeline, source):
    return pipeline.index._query(
        "SELECT COUNT(*) FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE source = ?)", (source,)
    )[0][0]


def test_a_deleted_file_leaves_the_index_on_the_next_ingest(make_pipeline, docs):
    pipeline = make_pipeline()
    pipeline.ingest(docs)
    assert pipeline.search("lost laptop", top_k=1, mode="dense")[0].chunk.source == "laptops.md"  # cache loaded

    (docs / "laptops.md").unlink()
    report = pipeline.ingest(docs)

    assert report.removed == ["laptops.md"]
    assert pipeline.index.sources() == ["deploy.md"]
    total_fts = pipeline.index._query("SELECT COUNT(*) FROM chunks_fts")[0][0]
    assert total_fts == pipeline.index.chunk_count() == 1
    for mode in ("bm25", "dense", "hybrid"):
        assert all(r.chunk.source != "laptops.md" for r in pipeline.search("lost laptop", mode=mode))


def test_ingesting_one_file_never_removes_the_others(make_pipeline, docs):
    pipeline = make_pipeline()
    pipeline.ingest(docs)

    report = pipeline.ingest(docs / "deploy.md")

    assert report.removed == []
    assert pipeline.index.sources() == ["deploy.md", "laptops.md"]


def test_a_failed_ingest_removes_nothing(make_pipeline, docs):
    pipeline = make_pipeline()
    pipeline.ingest(docs)
    (docs / "laptops.md").unlink()
    (docs / "broken.md").write_bytes(b"# Broken\n\n\xff\xfe not UTF-8\n")

    with pytest.raises(UnicodeDecodeError):
        pipeline.ingest(docs)

    assert pipeline.index.sources() == ["deploy.md", "laptops.md"]


def test_a_missing_folder_is_an_error_not_an_empty_folder(make_pipeline, docs):
    pipeline = make_pipeline()
    pipeline.ingest(docs)

    with pytest.raises(FileNotFoundError):
        pipeline.ingest(docs / "no-such-folder")

    assert pipeline.index.sources() == ["deploy.md", "laptops.md"]


# --- 3. one folder per index --------------------------------------------------------


def test_an_index_refuses_a_second_folder_before_touching_anything(make_pipeline, tmp_path):
    team_a, team_b = tmp_path / "team-a", tmp_path / "team-b"
    write(team_a, "README.md", "Oncall", "Team A pages Alice.")
    write(team_b, "README.md", "Oncall", "Team B pages Bob.")
    pipeline = make_pipeline()
    pipeline.ingest(team_a)

    with pytest.raises(IndexBindingError, match="SOURCED_INDEX"):
        pipeline.ingest(team_b)
    with pytest.raises(IndexBindingError, match="SOURCED_INDEX"):
        pipeline.ingest(team_b / "README.md")

    assert "Alice" in pipeline.search("pages", top_k=1, mode="bm25")[0].chunk.text
    assert pipeline.index.root() == team_a.resolve()


def test_the_folder_is_compared_after_normalizing_the_path(make_pipeline, docs):
    pipeline = make_pipeline()
    pipeline.ingest(docs)

    report = pipeline.ingest(docs.parent / "docs" / ".." / "docs")

    assert report.unchanged == ["deploy.md", "laptops.md"]


def test_a_single_file_belongs_to_the_folder_and_keeps_its_folder_relative_name(make_pipeline, docs):
    nested = write(docs, "team/oncall.md", "Pager", "The pager rotates weekly.")
    pipeline = make_pipeline()
    pipeline.ingest(docs)

    report = pipeline.ingest(nested)

    assert report.unchanged == ["team/oncall.md"]  # same document, not a second copy named oncall.md
    assert pipeline.index.sources() == ["deploy.md", "laptops.md", "team/oncall.md"]


def test_a_single_file_on_a_new_index_binds_its_folder(make_pipeline, docs):
    pipeline = make_pipeline()

    pipeline.ingest(docs / "deploy.md")

    assert pipeline.index.root() == docs.resolve()
    assert pipeline.ingest(docs).added == ["laptops.md"]


def test_an_index_from_before_folder_binding_is_refused_not_guessed(make_pipeline, docs):
    pipeline = make_pipeline()
    pipeline.ingest(docs)
    # What an index written by an older version looks like: documents, no folder recorded.
    with sqlite3.connect(pipeline.settings.index_path) as db:
        db.execute("DELETE FROM meta")
    legacy = make_pipeline(index_path=pipeline.settings.index_path)
    (docs / "laptops.md").unlink()

    with pytest.raises(IndexBindingError, match="older version"):
        legacy.ingest(docs)

    assert legacy.index.sources() == ["deploy.md", "laptops.md"]  # nothing removed on a guess
    assert legacy.search("lost laptop", top_k=1, mode="bm25")  # still readable


def test_a_heading_renamed_mid_query_never_lends_its_old_rerank_score(make_pipeline, docs):
    # Another process renames a heading (same body) after the reranker scored the old
    # chunk and before the results are fetched. The returned chunk's score must still be
    # the score of exactly the text it shows.
    class ScoresByLength:
        def score(self, query, passages):
            return [float(len(p)) for p in passages]

    server = make_pipeline(reranker=ScoresByLength())
    server.ingest(docs)
    writer = make_pipeline(index_path=server.settings.index_path)
    get, calls = server.index.get, []

    def get_then_rename(ids):
        calls.append(ids)
        if len(calls) == 2:  # the final fetch, after reranking
            write(docs, "deploy.md", "Undoing a release", "Run kctl rollback to undo a release.")
            writer.ingest(docs)
        return get(ids)

    server.index.get = get_then_rename
    results = server.search("rollback release", mode="hybrid", rerank=True)

    assert len(calls) == 2 and results
    for r in results:
        assert r.score == len(r.chunk.indexed_text)
