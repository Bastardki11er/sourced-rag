import pytest
from fastapi.testclient import TestClient

from sourced.api import create_app
from sourced.config import Settings


@pytest.mark.parametrize("endpoint,field", [("/ask", "question"), ("/search", "query")])
def test_whitespace_requests_rejected(handbook_pipeline, endpoint, field):
    with TestClient(create_app(handbook_pipeline)) as client:
        assert client.post(endpoint, json={field: " \n\t "}).status_code == 422


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"max_chunk_chars": 0}, {"answerer": "typo"}])
def test_invalid_settings_fail_early(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"top_k": -1}, {"mode": "typo"}])
def test_python_search_rejects_invalid_options(handbook_pipeline, kwargs):
    with pytest.raises(ValueError):
        handbook_pipeline.search("policy", **kwargs)
