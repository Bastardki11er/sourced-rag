from types import SimpleNamespace as NS
from unittest.mock import MagicMock

from sourced.chunking import Chunk
from sourced.generation import NOT_FOUND, ClaudeAnswerer, ExtractiveAnswerer
from sourced.retrieval import SearchResult

ROLLBACK = Chunk(
    "c1",
    "deployment.md",
    "Deploying to Production",
    "Rolling back",
    "To roll back, run kctl rollback --to <release-id>. Rollbacks do not need approval.",
    3,
)
FREEZE = Chunk(
    "c2",
    "deployment.md",
    "Deploying to Production",
    "Deploy freezes",
    "There is a deploy freeze from December 15 to January 2.",
    2,
)
RESULTS = [SearchResult(ROLLBACK, 0.03), SearchResult(FREEZE, 0.02)]


def citation(chunk: Chunk, quote: str):
    return NS(
        type="search_result_location",
        source=chunk.source,
        title=chunk.heading,
        cited_text=quote,
        search_result_index=0,
        start_block_index=0,
        end_block_index=0,
    )


def fake_client(content, stop_reason="end_turn"):
    client = MagicMock()
    client.beta.messages.create.return_value = NS(content=content, stop_reason=stop_reason)
    return client


def test_sends_chunks_as_citable_search_results_with_fallback_enabled():
    client = fake_client([NS(type="text", text="Run kctl rollback.", citations=None)])

    ClaudeAnswerer(client, "claude-opus-5-5", "medium").answer("How do I roll back?", RESULTS)

    request = client.beta.messages.create.call_args.kwargs
    first, second, question = request["messages"][0]["content"]
    assert first == {
        "type": "search_result",
        "source": "deployment.md",
        "title": "Deploying to Production > Rolling back",
        "content": [{"type": "text", "text": ROLLBACK.text}],
        "citations": {"enabled": True},
    }
    assert second["title"].endswith("Deploy freezes")
    assert question == {"type": "text", "text": "How do I roll back?"}
    assert request["model"] == "claude-opus-5-5"
    assert request["output_config"] == {"effort": "medium"}
    assert request["fallbacks"] == "default"
    assert request["betas"] == ["server-side-fallback-2026-07-01"]


def test_citations_become_numbered_markers_with_quotes():
    client = fake_client(
        [
            NS(type="thinking", thinking=""),
            NS(
                type="text",
                text="Run `kctl rollback --to <release-id>`",
                citations=[
                    citation(ROLLBACK, "To roll back, run kctl rollback --to <release-id>."),
                ],
            ),
            NS(
                type="text",
                text=", no approval needed",
                citations=[
                    citation(ROLLBACK, "Rollbacks do not need approval."),
                ],
            ),
            NS(
                type="text",
                text=". Note the freeze from December 15.",
                citations=[
                    citation(FREEZE, "There is a deploy freeze from December 15 to January 2."),
                ],
            ),
        ]
    )

    answer = ClaudeAnswerer(client, "m", "medium").answer("How do I roll back?", RESULTS)

    assert answer.text == (
        "Run `kctl rollback --to <release-id>`[1], no approval needed[1]. Note the freeze from December 15.[2]"
    )
    assert [(c.number, c.heading) for c in answer.citations] == [
        (1, "Deploying to Production > Rolling back"),
        (2, "Deploying to Production > Deploy freezes"),
    ]
    assert answer.citations[0].quotes == [
        "To roll back, run kctl rollback --to <release-id>.",
        "Rollbacks do not need approval.",
    ]


def test_refusals_are_reported_instead_of_returning_partial_content():
    client = fake_client([], stop_reason="refusal")

    answer = ClaudeAnswerer(client, "m", "medium").answer("question", RESULTS)

    assert answer.refused
    assert answer.citations == []


def test_no_retrieved_passages_means_no_llm_call():
    client = fake_client([])

    answer = ClaudeAnswerer(client, "m", "medium").answer("anything", [])

    assert answer.text == NOT_FOUND
    client.beta.messages.create.assert_not_called()


def test_extractive_answerer_quotes_the_best_passage():
    answer = ExtractiveAnswerer().answer("How do I roll back?", RESULTS)

    assert answer.text == ROLLBACK.text + " [1]"
    assert answer.citations[0].source == "deployment.md"


def test_the_real_sdk_serializes_the_request_and_parses_a_cited_reply():
    # The real Anthropic client over a mocked HTTP transport: checks what goes on the wire
    # (beta header, body fields) and that a reply in the documented shape parses. It says
    # nothing about whether the live API accepts the request; that needs a real key.
    import json

    import anthropic
    import httpx2

    sent = []
    reply = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "content": [
            {
                "type": "text",
                "text": "Run kctl rollback.",
                "citations": [
                    {
                        "type": "search_result_location",
                        "source": "deployment.md",
                        "title": "Deploying to Production > Rolling back",
                        "cited_text": "To roll back, run kctl rollback --to <release-id>.",
                        "search_result_index": 0,
                        "start_block_index": 0,
                        "end_block_index": 0,
                    }
                ],
            }
        ],
    }

    def handler(request):
        sent.append(request)
        return httpx2.Response(200, json=reply)

    client = anthropic.Anthropic(
        api_key="test-key", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )

    answer = ClaudeAnswerer(client, "claude-opus-5-5", "medium").answer("How do I roll back?", RESULTS)

    (request,) = sent
    assert request.method == "POST" and request.url.path == "/v1/messages"
    assert "server-side-fallback-2026-07-01" in request.headers["anthropic-beta"]
    body = json.loads(request.content)
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"] == {"effort": "medium"}
    assert "betas" not in body  # a header, not a body field
    first, _, question = body["messages"][0]["content"]
    assert first["type"] == "search_result" and first["citations"] == {"enabled": True}
    assert question == {"type": "text", "text": "How do I roll back?"}
    assert answer.text == "Run kctl rollback.[1]"
    assert [(c.source, c.heading) for c in answer.citations] == [
        ("deployment.md", "Deploying to Production > Rolling back")
    ]


def test_the_system_prompt_is_about_the_provided_documents_not_one_company():
    import json

    import anthropic
    import httpx2

    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        reply = {
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "content": [{"type": "text", "text": "Answer."}],
        }
        return httpx2.Response(200, json=reply)

    client = anthropic.Anthropic(
        api_key="k", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    ClaudeAnswerer(client, "claude-opus-5-5", "medium").answer("How do I roll back?", RESULTS)

    system = sent[0]["system"]
    system = system if isinstance(system, str) else " ".join(block["text"] for block in system)
    for word in ("handbook", "employee", "company"):
        assert word not in system.lower()
    assert "provided documents" in system and "only" in system and "not find" in system
    assert "handbook" not in NOT_FOUND.lower()
