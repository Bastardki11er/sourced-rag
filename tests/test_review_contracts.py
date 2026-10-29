"""Regression contracts from the hands-on review; no model downloads or paid calls."""

from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from sourced import cli
from sourced import pipeline as wiring
from sourced.api import create_app
from sourced.config import Settings
from sourced.generation import ClaudeAnswerer


@pytest.mark.parametrize(
    "stop, content",
    [
        ("max_tokens", [{"type": "text", "text": "To roll back, first"}]),
        ("max_tokens", []),
        ("end_turn", []),
        ("end_turn", [{"type": "text", "text": " \n "}]),
        ("end_turn", [{"type": "thinking", "thinking": "not an answer", "signature": "sig"}]),
        ("pause_turn", [{"type": "text", "text": "Partial answer"}]),
    ],
)
def test_unfinished_or_empty_sdk_replies_are_502_without_retry(handbook_pipeline, stop, content):
    sent = []

    def handler(request):
        sent.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "stop_reason": stop,
                "stop_sequence": None,
                "usage": {"input_tokens": 20, "output_tokens": 5},
                "content": content,
            },
        )

    with anthropic.Anthropic(
        api_key="test-only", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    ) as sdk:
        handbook_pipeline.answerer = ClaudeAnswerer(sdk, "claude-opus-5-5", "medium")
        with TestClient(create_app(handbook_pipeline)) as client:
            response = client.post("/ask", json={"question": "How do I roll back a release?"})
    assert response.status_code == 502
    assert "answer" not in response.json()
    assert "language model" in response.json()["detail"].lower()
    assert len(sent) == 1


def test_unavailable_reranking_is_rejected_before_any_search(handbook_pipeline, monkeypatch):
    def no_search(*a, **kw):
        raise AssertionError("retrieval ran despite unavailable reranking")

    monkeypatch.setattr(handbook_pipeline.index, "bm25", no_search)
    with TestClient(create_app(handbook_pipeline)) as client:
        for endpoint, body in (("/search", {"query": "rollback"}), ("/ask", {"question": "rollback"})):
            response = client.post(endpoint, json={**body, "rerank": True})
            assert response.status_code == 409
            assert "rerank" in response.json()["detail"].lower()
        assert client.get("/health").json()["reranking"] == "disabled"


@pytest.mark.parametrize("bad", [0, -1, 21, True, 2.5, "5"])
def test_python_entry_points_reject_invalid_top_k_before_retrieval(handbook_pipeline, monkeypatch, bad):
    monkeypatch.setattr(handbook_pipeline.index, "bm25", lambda *a: pytest.fail("retrieval ran"))
    for search in (handbook_pipeline.search, handbook_pipeline.ask, handbook_pipeline.retriever.search):
        with pytest.raises(ValueError, match="top_k"):
            search("rollback", top_k=bad)


@pytest.mark.parametrize("bad", [0, -1, 21, True, 2.5, "5"])
def test_settings_reject_invalid_top_k(bad):
    with pytest.raises(ValueError, match="top_k"):
        Settings(top_k=bad)


@pytest.mark.parametrize(
    "name, bad",
    [
        ("SOURCED_TOP_K", "0"),
        ("SOURCED_TOP_K", "21"),
        ("SOURCED_TOP_K", "oops"),
        ("SOURCED_ANSWERER", "claud"),
        ("SOURCED_RERANK", "treu"),
    ],
)
def test_invalid_environment_exits_before_loading_models(monkeypatch, capsys, name, bad):
    monkeypatch.setenv(name, bad)
    monkeypatch.setattr(wiring, "build_pipeline", lambda *a, **kw: pytest.fail("pipeline loaded"))
    assert cli.main(["search", "rollback"]) == 2
    assert name in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1", "21", "2.5", "five"])
def test_invalid_cli_top_k_exits_before_loading_models(monkeypatch, capsys, value):
    monkeypatch.setattr(wiring, "build_pipeline", lambda *a, **kw: pytest.fail("pipeline loaded"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["search", "rollback", f"--top-k={value}"])
    assert exc.value.code == 2
    assert "--top-k must be an integer between 1 and 20" in capsys.readouterr().err


def test_cli_search_uses_configured_top_k_unless_overridden(handbook_pipeline, monkeypatch):
    monkeypatch.setenv("SOURCED_TOP_K", "2")
    monkeypatch.setattr(wiring, "build_pipeline", lambda *a, **kw: handbook_pipeline)
    search = MagicMock(return_value=[])
    monkeypatch.setattr(handbook_pipeline, "search", search)
    assert cli.main(["search", "rollback"]) == 0
    assert search.call_args.args[1] is None
    assert cli.main(["search", "rollback", "--top-k", "3"]) == 0
    assert search.call_args.args[1] == 3


def test_unavailable_reranking_is_a_clean_cli_error(handbook_pipeline, monkeypatch, capsys):
    monkeypatch.setattr(wiring, "build_pipeline", lambda *a, **kw: handbook_pipeline)
    assert cli.main(["search", "rollback", "--rerank"]) == 2
    assert "rerank" in capsys.readouterr().err.lower()


def test_generation_failures_are_a_clean_cli_error(handbook_pipeline, monkeypatch, capsys):
    client = MagicMock()
    client.beta.messages.create.return_value = NS(stop_reason="max_tokens", content=[])
    handbook_pipeline.answerer = ClaudeAnswerer(client, "m", "medium")
    monkeypatch.setattr(wiring, "build_pipeline", lambda *a, **kw: handbook_pipeline)
    assert cli.main(["ask", "rollback"]) == 2
    assert capsys.readouterr().out == ""
    client.beta.messages.create.assert_called_once()


@pytest.mark.parametrize("bad", [True, 2.5, "5"])
def test_api_top_k_requires_an_integer(handbook_pipeline, bad):
    with TestClient(create_app(handbook_pipeline)) as client:
        assert client.post("/search", json={"query": "rollback", "top_k": bad}).status_code == 422
        assert client.post("/ask", json={"question": "rollback", "top_k": bad}).status_code == 422


def test_python_default_top_k_and_valid_boundaries(handbook_pipeline):
    from dataclasses import replace

    handbook_pipeline.settings = replace(handbook_pipeline.settings, top_k=2)
    assert len(handbook_pipeline.search("policy")) == 2
    assert len(handbook_pipeline.ask("policy").results) == 2
    for top_k in (1, 20):
        assert len(handbook_pipeline.search("policy", top_k=top_k)) == top_k


@pytest.mark.parametrize(
    "raw, expected", [("true", True), ("yes", True), ("1", True), ("false", False), ("no", False), ("0", False)]
)
def test_boolean_settings_keep_supported_spellings(monkeypatch, raw, expected):
    monkeypatch.setenv("SOURCED_RERANK", raw)
    assert Settings.from_env().rerank is expected


def test_invalid_answerer_never_silently_selects_extractive():
    with pytest.raises(ValueError, match="answerer"):
        Settings(answerer="claud")


def test_available_reranking_is_reported_and_executed(make_pipeline, tmp_path):
    reranker = MagicMock()
    reranker.score.side_effect = lambda query, passages: [1.0] * len(passages)
    p = make_pipeline(reranker=reranker)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# Releases\n\nUse kctl rollback to roll back a release.")
    p.ingest(docs)
    with TestClient(create_app(p)) as client:
        assert client.get("/health").json()["reranking"] == "ok"
        assert client.post("/search", json={"query": "rollback", "rerank": True}).status_code == 200
        assert client.post("/ask", json={"question": "rollback", "rerank": True}).status_code == 200
    assert reranker.score.call_count == 2


def _sdk_answerer(reply, sent):
    def handler(request):
        sent.append(request)
        return httpx2.Response(200, json=reply)

    sdk = anthropic.Anthropic(
        api_key="test-only", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    return ClaudeAnswerer(sdk, "claude-opus-5-5", "medium")


def _reply(stop, content):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 5},
        "content": content,
    }


def test_complete_answers_and_refusals_keep_their_behaviour_through_the_real_sdk(handbook_pipeline):
    cited = {
        "type": "text",
        "text": "Run kctl rollback.",
        "citations": [
            {
                "type": "search_result_location",
                "source": "deployment.md",
                "title": "Deploying to Production > Rolling back",
                "cited_text": "kctl rollback --to <release-id>",
                "search_result_index": 0,
                "start_block_index": 0,
                "end_block_index": 0,
            }
        ],
    }
    for reply, check in (
        (
            _reply("end_turn", [cited]),
            lambda b: b["refused"] is False and b["citations"][0]["source"] == "deployment.md",
        ),
        (_reply("refusal", []), lambda b: b["refused"] is True and b["citations"] == []),
    ):
        sent = []
        handbook_pipeline.answerer = _sdk_answerer(reply, sent)
        with TestClient(create_app(handbook_pipeline)) as client:
            response = client.post("/ask", json={"question": "How do I roll back a release?"})
        assert response.status_code == 200 and check(response.json())
        assert len(sent) == 1


def test_no_retrieved_passages_answers_not_found_without_calling_claude(make_pipeline):
    sent = []
    empty = make_pipeline(answerer=_sdk_answerer(_reply("end_turn", []), sent))  # nothing ingested: no results

    with TestClient(create_app(empty)) as client:
        body = client.post("/ask", json={"question": "kctl"}).json()

    assert body["citations"] == [] and "couldn't find" in body["answer"] and sent == []
