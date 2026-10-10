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
