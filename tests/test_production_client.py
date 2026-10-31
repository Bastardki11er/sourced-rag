"""The Claude client that build_pipeline really builds: retries, timeout and error mapping. No network."""

import json

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from conftest import HANDBOOK
from sourced import embeddings, retrieval
from sourced.api import create_app
from sourced.config import Settings
from sourced.embeddings import HashingEmbedder
from sourced.pipeline import build_pipeline


class LengthReranker:
    def __init__(self, model_name):
        pass

    def score(self, query, passages):
        return [float(len(p)) for p in passages]


def error_reply(status):
    body = {"type": "error", "error": {"type": "overloaded_error" if status == 529 else "api_error", "message": "x"}}
    return lambda request: httpx2.Response(status, json=body)


def timeout_reply(request):
    raise httpx2.ReadTimeout("timed out", request=request)


@pytest.fixture
def production(tmp_path, monkeypatch):
    """build_pipeline with test models and a mocked HTTP transport under the real Anthropic SDK."""
    sent, made = [], {}
    real_client = anthropic.Anthropic

    def client_with_mock_transport(**kwargs):
        made["kwargs"] = kwargs

        def handler(request):
            sent.append(request)
            return made["reply"](request)

        return real_client(http_client=httpx2.Client(transport=httpx2.MockTransport(handler)), **kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", client_with_mock_transport)
    monkeypatch.setattr(embeddings, "FastEmbedEmbedder", lambda name: HashingEmbedder())
    monkeypatch.setattr(retrieval, "CrossEncoderReranker", LengthReranker)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    def build(reply):
        made["reply"] = reply
        pipeline = build_pipeline(Settings(index_path=tmp_path / "index.db", answerer="claude"))
        pipeline.ingest(HANDBOOK)
        return pipeline

    return build, sent, made


def test_the_production_client_never_retries_and_has_an_explicit_timeout(production):
    build, sent, made = production
    pipeline = build(error_reply(500))

    assert made["kwargs"] == {"max_retries": 0, "timeout": 60.0}
    assert pipeline.answerer.client.max_retries == 0
    assert pipeline.answerer.client.timeout == 60.0


@pytest.mark.parametrize(
    "reply, status",
    [(error_reply(429), 503), (error_reply(529), 503), (timeout_reply, 503), (error_reply(500), 502)],
    ids=["rate-limited", "overloaded", "timeout", "server-error"],
)
def test_one_failed_ask_sends_one_request_and_maps_to_a_clear_status(production, reply, status):
    build, sent, _ = production
    pipeline = build(reply)

    with TestClient(create_app(pipeline)) as client:
        response = client.post("/ask", json={"question": "How do I roll back a release?"})

    assert len(sent) == 1  # no automatic SDK retry, whatever the error
    assert response.status_code == status
    detail = response.json()["detail"]
    assert "language model" in detail and "test-key" not in detail and "api_error" not in detail


def test_a_successful_reply_still_goes_through(production):
    build, sent, _ = production
    reply = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "content": [{"type": "text", "text": "Run kctl rollback."}],
    }
    pipeline = build(lambda request: httpx2.Response(200, json=reply))

    with TestClient(create_app(pipeline)) as client:
        body = client.post("/ask", json={"question": "How do I roll back a release?"}).json()

    assert body["answer"] == "Run kctl rollback." and body["answerer"] == "claude" and len(sent) == 1
    assert json.loads(sent[0].content)["model"] == "claude-opus-5-5"
