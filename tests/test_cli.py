from pathlib import Path

import pytest

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


def test_eval_answers_follow_the_rerank_flag_the_pipeline_was_built_with(make_pipeline, monkeypatch, tmp_path, capsys):
    # Same wiring as the real CLI: a reranker is loaded only when the command asks for one,
    # while SOURCED_RERANK keeps its default (true).
    from conftest import HANDBOOK
    from sourced.config import Settings

    class ScoresByLength:
        def score(self, query, passages):
            return [float(len(p)) for p in passages]

    built = {}

    def build(settings, with_reranker=None):
        built["pipeline"] = make_pipeline(reranker=ScoresByLength() if with_reranker else None)
        built["pipeline"].settings = Settings(index_path=built["pipeline"].settings.index_path)  # rerank=True
        built["pipeline"].ingest(HANDBOOK)
        return built["pipeline"]

    monkeypatch.setattr(pipeline, "build_pipeline", build)
    questions = Path(__file__).parent.parent / "eval" / "questions.jsonl"

    assert cli.main(["eval", str(questions), "--answers"]) == 0
    assert "Contains the expected facts" in capsys.readouterr().out
    assert cli.main(["eval", str(questions), "--answers", "--rerank"]) == 0


def _claude_pipeline(handbook_pipeline, status=None, raise_timeout=False):
    import anthropic
    import httpx2

    from sourced.generation import ClaudeAnswerer

    def handler(request):
        if raise_timeout:
            raise httpx2.ReadTimeout("timed out", request=request)
        body = {"type": "error", "error": {"type": "x", "message": "SECRET-UPSTREAM-BODY"}}
        return httpx2.Response(status, json=body)

    sdk = anthropic.Anthropic(
        api_key="test-key-not-real", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    handbook_pipeline.answerer = ClaudeAnswerer(sdk, "claude-opus-5-5", "medium")
    return handbook_pipeline


@pytest.mark.parametrize(
    "status, timeout, words",
    [
        (429, False, "rate limit"),
        (529, False, "overloaded"),
        (None, True, "timed out"),
        (500, False, "500"),
        (401, False, "authentication"),
    ],
    ids=["429", "529", "timeout", "500", "401"],
)
def test_claude_errors_in_ask_are_short_clear_and_exit_2(
    handbook_pipeline, monkeypatch, capsys, status, timeout, words
):
    pipeline_ = _claude_pipeline(handbook_pipeline, status, timeout)
    monkeypatch.setattr(pipeline, "build_pipeline", lambda settings, with_reranker=None: pipeline_)

    assert cli.main(["ask", "How do I roll back a release?"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""  # no answer, extractive or otherwise
    assert words in captured.err.lower() and captured.err.startswith("sourced: error:")
    for leak in ("Traceback", "SECRET-UPSTREAM-BODY", "test-key-not-real", "x-api-key"):
        assert leak not in captured.err


def test_an_undecodable_document_names_the_file_and_exits_2(make_pipeline, monkeypatch, tmp_path, capsys):
    shared = make_pipeline()
    monkeypatch.setattr(pipeline, "build_pipeline", lambda settings, with_reranker=None: shared)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\n## One\n\nFirst.\n")
    assert cli.main(["ingest", str(docs)]) == 0
    (docs / "a.md").unlink()
    (docs / "b.md").write_bytes(b"# B\n\n\xff\xfe\n")
    capsys.readouterr()

    assert cli.main(["ingest", str(docs)]) == 2
    err = capsys.readouterr().err
    assert "b.md" in err and "UTF-8" in err and "Traceback" not in err
    assert shared.index.sources() == ["a.md"]  # nothing removed


def test_an_unsupported_single_file_exits_2(make_pipeline, monkeypatch, tmp_path, capsys):
    shared = make_pipeline()
    monkeypatch.setattr(pipeline, "build_pipeline", lambda settings, with_reranker=None: shared)
    (tmp_path / "script.py").write_text("print('hi')\n")

    assert cli.main(["ingest", str(tmp_path / "script.py")]) == 2
    assert "not a supported document" in capsys.readouterr().err
    assert shared.index.sources() == []
