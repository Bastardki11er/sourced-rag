"""One ingest at a time per index file, across processes. Sync points: the child prints
"paused" from inside its ingest and waits for a line on stdin; no sleeps."""

import subprocess
import sys
from pathlib import Path

import pytest

from conftest import index_state

CHILD = Path(__file__).parent / "ingest_child.py"


def write(folder, name, body):
    (folder / name).write_text(f"# {name}\n\n## Section\n\n{body}\n")


@pytest.fixture
def docs(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    write(folder, "a.md", "alpha")
    write(folder, "b.md", "beta")
    return folder


@pytest.fixture
def paused_child(make_pipeline, docs):
    """An index with a.md and b.md, and another process stopped in the middle of re-ingesting b.md."""
    pipeline = make_pipeline()
    pipeline.ingest(docs)
    write(docs, "b.md", "beta, second version")
    child = subprocess.Popen(
        [sys.executable, str(CHILD), str(pipeline.settings.index_path), str(docs)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout.readline() == "paused\n"  # it holds the lock now, mid-ingest
    yield pipeline, child
    if child.poll() is None:
        child.kill()
        child.wait()


def finish(child):
    out, _ = child.communicate("go\n", timeout=60)
    assert child.returncode == 0, out
    return out


def test_a_second_ingest_is_refused_while_another_process_ingests(paused_child, docs, make_pipeline):
    pipeline, child = paused_child
    write(docs, "c.md", "gamma")  # appeared after the child listed the folder
    other = make_pipeline(index_path=pipeline.settings.index_path)  # also a fresh connection and lock handle
    before = index_state(pipeline.settings.index_path)

    with pytest.raises(ValueError, match="another sourced ingest"):
        other.ingest(docs)
    assert index_state(pipeline.settings.index_path) == before

    assert "['b.md']" in finish(child)  # the child completes normally and releases the lock
    assert other.ingest(docs).added == ["c.md"]
    assert other.index.sources() == ["a.md", "b.md", "c.md"]  # nothing removed by the interleaving


def test_the_lock_is_released_when_the_holder_is_killed(paused_child, docs):
    pipeline, child = paused_child
    child.kill()
    child.wait()

    assert pipeline.ingest(docs).updated == ["b.md"]


def test_the_lock_is_released_when_ingest_fails(make_pipeline, docs):
    pipeline = make_pipeline()
    (docs / "0-broken.md").write_bytes(b"\xff\xfe")  # sorts first: fails before writing anything
    with pytest.raises(UnicodeDecodeError):
        pipeline.ingest(docs)
    (docs / "0-broken.md").unlink()

    assert make_pipeline(index_path=pipeline.settings.index_path).ingest(docs).added == ["a.md", "b.md"]


def test_two_pipelines_in_one_process_also_take_turns(make_pipeline, docs):
    first = make_pipeline()
    second = make_pipeline(index_path=first.settings.index_path)
    embed, seen = first.embedder.embed_documents, []

    def embed_while_the_other_tries(texts):
        with pytest.raises(ValueError, match="another sourced ingest"):
            second.ingest(docs)
        seen.append(True)
        return embed(texts)

    first.embedder.embed_documents = embed_while_the_other_tries
    first.ingest(docs)

    assert seen
    assert second.ingest(docs).unchanged == ["a.md", "b.md"]


def test_the_lock_follows_the_resolved_index_path(make_pipeline, docs, tmp_path):
    pipeline = make_pipeline()
    pipeline.ingest(docs)
    link = tmp_path / "link.db"
    link.symlink_to(pipeline.settings.index_path)
    via_link = make_pipeline(index_path=link)
    embed = pipeline.embedder.embed_documents
    write(docs, "b.md", "beta again")

    def embed_while_the_link_tries(texts):
        with pytest.raises(ValueError, match="another sourced ingest"):
            via_link.ingest(docs)
        return embed(texts)

    pipeline.embedder.embed_documents = embed_while_the_link_tries
    assert pipeline.ingest(docs).updated == ["b.md"]
