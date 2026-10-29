from pathlib import Path

from sourced import cli, pipeline


def test_ingest_reports_removed_files_and_refuses_a_second_folder(make_pipeline, monkeypatch, tmp_path, capsys):
    shared = make_pipeline()
    monkeypatch.setattr(pipeline, "build_pipeline", lambda settings, with_reranker=None: shared)  # no model download
    docs, other = tmp_path / "docs", tmp_path / "other"
    for folder in (docs, other):
        folder.mkdir()
        (folder / "a.md").write_text("# A\n\n## One\n\nFirst.\n")
    (docs / "b.md").write_text("# B\n\n## Two\n\nSecond.\n")
    assert cli.main(["ingest", str(docs)]) == 0
    (docs / "b.md").unlink()

    assert cli.main(["ingest", str(docs)]) == 0
    assert "removed 1" in capsys.readouterr().out
    assert cli.main(["ingest", str(other)]) == 2
    assert "SOURCED_INDEX" in capsys.readouterr().err
    assert cli.main(["ingest", str(tmp_path / "missing")]) == 2


def test_eval_ranks_each_question_once_per_retriever_and_gates_on_those_ranks(handbook_pipeline, monkeypatch, capsys):
    from sourced.evaluation import load_questions

    monkeypatch.setattr(pipeline, "build_pipeline", lambda settings, with_reranker=None: handbook_pipeline)
    calls = []
    search = handbook_pipeline.search
    monkeypatch.setattr(handbook_pipeline, "search", lambda *a, **kw: calls.append(kw["mode"]) or search(*a, **kw))
    questions = Path(__file__).parent.parent / "eval" / "questions.jsonl"
    answerable = sum(q.answerable for q in load_questions(questions))

    assert cli.main(["eval", str(questions), "--fail-under", "0.0"]) == 0

    assert sorted(calls) == sorted(["bm25", "dense", "hybrid"] * answerable)
    out = capsys.readouterr().out
    hybrid_hit5 = out.split("| Hybrid (RRF) | ")[1].split(" | ")[2]
    assert f"OK: hit@5 {hybrid_hit5} >= 0.00" in out  # the gate uses the number the table shows
    assert cli.main(["eval", str(questions), "--fail-under", "1.01"]) == 1


def test_ingest_exits_non_zero_while_another_ingest_runs(make_pipeline, monkeypatch, tmp_path, capsys):
    import subprocess
    import sys

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\n## One\n\nFirst.\n")
    shared = make_pipeline()
    child = subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "ingest_child.py"), str(shared.settings.index_path), str(docs)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline() == "paused\n"
        monkeypatch.setattr(pipeline, "build_pipeline", lambda settings, with_reranker=None: shared)

        assert cli.main(["ingest", str(docs)]) == 2
        assert "another sourced ingest" in capsys.readouterr().err
    finally:
        child.communicate("go\n", timeout=60)


def test_a_blank_search_query_exits_with_an_error(handbook_pipeline, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "build_pipeline", lambda settings, with_reranker=None: handbook_pipeline)

    assert cli.main(["search", "  \t "]) == 2
    captured = capsys.readouterr()
    assert "empty" in captured.err and captured.out == ""


def test_ingest_with_a_symlink_in_the_folder_exits_with_an_error(make_pipeline, monkeypatch, tmp_path, capsys):
    shared = make_pipeline()
    monkeypatch.setattr(pipeline, "build_pipeline", lambda settings, with_reranker=None: shared)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\n## One\n\nFirst.\n")
    (docs / "elsewhere.md").symlink_to(tmp_path / "missing.md")

    assert cli.main(["ingest", str(docs)]) == 2
    assert "is a symbolic link" in capsys.readouterr().err
    assert shared.index.sources() == []
