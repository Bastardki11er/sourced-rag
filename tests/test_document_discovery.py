"""Which files an ingest reads: supported suffixes in any case, nothing hidden inside the folder."""

import os
import sys

import numpy as np
import pytest

from conftest import index_state
from sourced.chunking import chunk_markdown


def write(path, text="# T\n\n## S\n\nBody text.\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def docs(tmp_path):
    folder = tmp_path / "docs"
    for name in ("a.md", "B.MD", "c.Markdown", "d.TXT", "sub/visible.md"):
        write(folder / name, f"# {name}\n\n## Part\n\nText of {name}.\n")
    for name in ("notes.py", ".hidden.md", ".git/config.md", ".venv/lib/pkg/README.md", "sub/.secret/x.md"):
        write(folder / name, f"# {name}\n\n## Part\n\nPRIVATE {name}.\n")
    return folder


VISIBLE = ["B.MD", "a.md", "c.Markdown", "d.TXT", "sub/visible.md"]


def test_a_folder_ingest_reads_supported_suffixes_in_any_case_and_nothing_hidden(make_pipeline, docs):
    pipeline = make_pipeline()

    report = pipeline.ingest(docs)

    assert sorted(report.added) == VISIBLE
    assert "PRIVATE" not in repr(index_state(pipeline.settings.index_path))


def test_hidden_folders_are_not_entered_at_all(make_pipeline, docs, tmp_path):
    (docs / ".venv" / "link").symlink_to(tmp_path, target_is_directory=True)  # would be refused if scanned
    locked = docs / ".cache"
    locked.mkdir()
    write(locked / "x.md")
    locked.chmod(0)
    try:
        report = make_pipeline().ingest(docs)
    finally:
        locked.chmod(0o755)

    assert sorted(report.added) == VISIBLE


def test_a_link_in_a_visible_path_is_still_refused(make_pipeline, docs, tmp_path):
    (docs / "sub" / "alias.md").symlink_to(docs / "a.md")

    with pytest.raises(ValueError, match="is a symbolic link"):
        make_pipeline().ingest(docs)


def test_the_folder_itself_may_be_hidden(make_pipeline, tmp_path):
    folder = tmp_path / ".notes"
    write(folder / "a.md")
    write(folder / ".draft.md")

    assert make_pipeline().ingest(folder).added == ["a.md"]


@pytest.mark.parametrize("name", ["notes.py", ".hidden.md", ".git/config.md", "sub/.secret/x.md"])
def test_a_single_file_must_follow_the_same_rules(make_pipeline, docs, name):
    pipeline = make_pipeline()
    pipeline.ingest(docs)
    before = index_state(pipeline.settings.index_path)

    with pytest.raises(ValueError, match="not a supported document|hidden"):
        pipeline.ingest(docs / name)

    assert index_state(pipeline.settings.index_path) == before


def test_a_single_file_with_an_uppercase_suffix_is_accepted(make_pipeline, docs):
    pipeline = make_pipeline()
    pipeline.ingest(docs)
    write(docs / "B.MD", "# B\n\n## Part\n\nChanged.\n")

    assert pipeline.ingest(docs / "B.MD").updated == ["B.MD"]


def test_hidden_documents_indexed_by_an_older_version_are_removed_by_a_full_ingest_only(make_pipeline, docs):
    pipeline = make_pipeline()
    pipeline.ingest(docs)
    chunks = chunk_markdown(".git/config.md", "# Old\n\n## Part\n\nOld hidden text.\n")
    vectors = np.asarray(pipeline.embedder.embed_documents([c.indexed_text for c in chunks]))
    pipeline.index.replace_document(".git/config.md", "old", chunks, vectors)  # as an older version indexed it

    assert pipeline.ingest(docs / "a.md").removed == []  # a single file never removes others
    assert ".git/config.md" in pipeline.index.sources()
    assert pipeline.ingest(docs).removed == [".git/config.md"]
    assert pipeline.index.sources() == sorted(VISIBLE)


def test_an_unreadable_document_names_the_file_and_removes_nothing(make_pipeline, docs):
    pipeline = make_pipeline()
    pipeline.ingest(docs)
    (docs / "a.md").unlink()  # would be removed by a successful run
    (docs / "sub" / "broken.md").write_bytes(b"# B\n\n\xff\xfe not UTF-8\n")
    before = index_state(pipeline.settings.index_path)

    with pytest.raises(ValueError, match=r"sub/broken\.md.*UTF-8"):
        pipeline.ingest(docs)

    assert index_state(pipeline.settings.index_path) == before


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions, not root")
def test_a_document_that_cannot_be_opened_names_the_file(make_pipeline, docs):
    pipeline = make_pipeline()
    (docs / "a.md").chmod(0)
    try:
        with pytest.raises(ValueError, match=r"a\.md.*could not be read"):
            pipeline.ingest(docs)
    finally:
        (docs / "a.md").chmod(0o644)
