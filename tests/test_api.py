import anthropic
import httpx2
from fastapi.testclient import TestClient

from sourced.api import create_app
from sourced.generation import Answer


def client_for(pipeline) -> TestClient:
    return TestClient(create_app(pipeline))


def test_health_reports_index_size(handbook_pipeline):
    with client_for(handbook_pipeline) as client:
        body = client.get("/health").json()

    assert body == {
        "status": "ok",
        "documents": 14,
        "chunks": handbook_pipeline.index.chunk_count(),
        "answerer": "extractive",
        "reranking": "disabled",
        "dense_search": "ok",
    }


def test_search_returns_ranked_passages(handbook_pipeline):
    with client_for(handbook_pipeline) as client:
        response = client.post("/search", json={"query": "KR-3000", "top_k": 2, "mode": "bm25"})

    assert response.status_code == 200
    passages = response.json()
    assert len(passages) == 2
    assert passages[0]["source"] == "error-codes.md"
    assert passages[0]["ranks"] == {"bm25": 1}


def test_ask_returns_answer_citations_and_timings(handbook_pipeline):
    with client_for(handbook_pipeline) as client:
        body = client.post("/ask", json={"question": "What command rolls back a release?"}).json()

    assert "kctl rollback" in body["answer"]
    assert body["citations"][0]["source"] == "deployment.md"
    assert body["answerer"] == "extractive"
    assert set(body["timings_ms"]) == {"retrieval", "generation"}


def test_invalid_requests_are_rejected(handbook_pipeline):
    with client_for(handbook_pipeline) as client:
        assert client.post("/ask", json={"question": ""}).status_code == 422
        assert client.post("/search", json={"query": "x", "top_k": 500}).status_code == 422
        assert client.post("/search", json={"query": "x", "mode": "magic"}).status_code == 422


def test_llm_outages_map_to_503(make_pipeline):
    class DownAnswerer:
        name = "claude"

        def answer(self, question, results) -> Answer:
            raise anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com"))

    with client_for(make_pipeline(answerer=DownAnswerer())) as client:
        response = client.post("/ask", json={"question": "anything"})

    assert response.status_code == 503


def test_an_index_without_a_recorded_model_serves_keyword_search_and_says_why_dense_is_off(handbook_pipeline):
    import sqlite3

    with sqlite3.connect(handbook_pipeline.settings.index_path) as db:
        db.execute("DROP TABLE meta")  # what an index built by an older version looks like

    with client_for(handbook_pipeline) as client:
        assert client.post("/search", json={"query": "KR-3000", "mode": "bm25"}).status_code == 200
        dense = client.post("/search", json={"query": "KR-3000", "mode": "dense"})
        ask = client.post("/ask", json={"question": "What command rolls back a release?"})
        health = client.get("/health").json()

    for response in (dense, ask):
        assert response.status_code == 409
        assert "SOURCED_INDEX" in response.json()["detail"]
    assert "older version" in health["dense_search"]


def test_openapi_lists_the_errors_each_endpoint_can_return(handbook_pipeline):
    with client_for(handbook_pipeline) as client:
        paths = client.get("/openapi.json").json()["paths"]

    assert set(paths["/search"]["post"]["responses"]) == {"200", "409", "422"}
    assert set(paths["/ask"]["post"]["responses"]) == {"200", "409", "422", "502", "503"}
    assert set(paths["/health"]["get"]["responses"]) == {"200"}


def test_blank_queries_are_rejected_before_retrieval(handbook_pipeline, monkeypatch):
    calls = []
    search = handbook_pipeline.search
    monkeypatch.setattr(handbook_pipeline, "search", lambda *a, **kw: calls.append(a) or search(*a, **kw))

    with client_for(handbook_pipeline) as client:
        for blank in (" ", "\n\t ", "   \n"):
            assert client.post("/search", json={"query": blank}).status_code == 422
            assert client.post("/ask", json={"question": blank}).status_code == 422
        assert calls == []
        assert (
            client.post("/search", json={"query": "  KR-3000  ", "mode": "bm25"}).json()[0]["source"]
            == "error-codes.md"
        )
