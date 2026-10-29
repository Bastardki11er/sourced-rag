"""The paid live check, run offline: its spend guards, cost accounting and checks. No network."""

import json

import anthropic
import httpx2
import pytest
from anthropic.types.beta import BetaUsage

from sourced import live_check
from sourced.generation import ClaudeAnswerer
from sourced.live_check import Case, call_cost, main, run


@pytest.mark.parametrize(
    "argv, env, message",
    [
        ([], {"ANTHROPIC_API_KEY": "k"}, "--confirm-spend"),
        (["--confirm-spend"], {}, "no ANTHROPIC_API_KEY"),
        (["--confirm-spend"], {"ANTHROPIC_API_KEY": "k", "ANTHROPIC_BASE_URL": "https://proxy.example"}, "proxy"),
    ],
    ids=["no-confirmation", "no-credentials", "not-the-official-api"],
)
def test_it_refuses_to_start_before_anything_is_loaded_or_sent(tmp_path, monkeypatch, capsys, argv, env, message):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert main(["--index", str(tmp_path / "x.db"), "--out", str(tmp_path / "out.jsonl"), *argv]) == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out.jsonl").exists()


NO_CACHE = {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


def test_cost_counts_every_attempt_at_its_own_model_price():
    plain = BetaUsage.model_validate({"input_tokens": 1_000_000, "output_tokens": 100_000})
    assert call_cost(plain, "claude-opus-5-5") == pytest.approx(4.0 + 2.0)

    with_fallback = BetaUsage.model_validate(
        {
            "input_tokens": 1000,
            "output_tokens": 100,
            "iterations": [
                {"type": "message", "input_tokens": 1_000_000, "output_tokens": 0, **NO_CACHE},
                {"type": "fallback_message", "model": "claude-opus-5", "input_tokens": 1_000_000, "output_tokens": 0}
                | NO_CACHE,
            ],
        }
    )
    # The first attempt names no model: it ran on the requested one, not the fallback.
    assert call_cost(with_fallback, "claude-opus-5-5") == pytest.approx(4.0 + 5.0)

    with pytest.raises(ValueError, match="no price"):
        call_cost(plain, "claude-unknown")


def claude_pipeline(make_pipeline, handbook_pipeline, quote_from_passage=True, model="claude-opus-5-5", status=200):
    """The handbook pipeline answering through the real SDK over a mocked transport.

    The reply cites the first search result sent, quoting its own text (or text it doesn't hold).
    """
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        if status != 200:
            return httpx2.Response(status, json={"type": "error", "error": {"type": "api_error", "message": "boom"}})
        first = body["messages"][0]["content"][0]
        quote = first["content"][0]["text"][:40] if quote_from_passage else "Text that no passage contains."
        cite = {
            "type": "search_result_location",
            "source": first["source"],
            "title": first["title"],
            "cited_text": quote,
            "search_result_index": 0,
            "start_block_index": 0,
            "end_block_index": 0,
        }
        reply = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 200},
            "content": [{"type": "text", "text": "Run kctl rollback.", "citations": [cite]}],
        }
        return httpx2.Response(200, json=reply)

    client = anthropic.Anthropic(
        api_key="test-key", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    pipeline = make_pipeline(
        answerer=ClaudeAnswerer(client, model, "medium"), index_path=handbook_pipeline.settings.index_path
    )
    return pipeline, sent


ROLLBACK = Case("l02", "lexical", "What is the command to roll back a release?", facts=["kctl rollback"])


def test_each_answer_is_recorded_with_its_cost_and_checks(make_pipeline, handbook_pipeline, tmp_path):
    pipeline, sent = claude_pipeline(make_pipeline, handbook_pipeline)
    out = tmp_path / "out.jsonl"

    spent = run(pipeline, [ROLLBACK], budget=2.0, out=out)

    (record,) = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(sent) == 1 and spent == pytest.approx(0.008)  # 1000 in at $4/M + 200 out at $20/M
    assert record["model"] == "claude-opus-5-5" and record["stop_reason"] == "end_turn"
    assert record["checks"] == {
        "citations_match_passages": True,
        "no_forbidden_text": True,
        "has_expected_facts": True,
        "cites_gold": True,  # no gold sections given for this case
    }


def test_a_quote_that_is_not_in_the_passage_fails_the_check(make_pipeline, handbook_pipeline, tmp_path):
    pipeline, _ = claude_pipeline(make_pipeline, handbook_pipeline, quote_from_passage=False)

    run(pipeline, [ROLLBACK], budget=2.0, out=tmp_path / "out.jsonl")

    record = json.loads((tmp_path / "out.jsonl").read_text())
    assert record["checks"]["citations_match_passages"] is False


def test_no_call_is_made_when_its_worst_case_would_pass_the_budget(make_pipeline, handbook_pipeline, tmp_path):
    pipeline, sent = claude_pipeline(make_pipeline, handbook_pipeline)
    worst = live_check.worst_case_cost(pipeline.search(ROLLBACK.question), ROLLBACK.question)

    spent = run(pipeline, [ROLLBACK, ROLLBACK], budget=worst - 0.01, out=tmp_path / "out.jsonl")

    assert sent == [] and spent == 0
    assert "skipped" in json.loads((tmp_path / "out.jsonl").read_text())


def test_the_budget_counts_what_earlier_runs_recorded_in_the_same_file(make_pipeline, handbook_pipeline, tmp_path):
    pipeline, sent = claude_pipeline(make_pipeline, handbook_pipeline)
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps({"id": "earlier", "cost_usd": 1.9}) + "\n")

    spent = run(pipeline, [ROLLBACK], budget=2.0, out=out)

    assert sent == [] and spent == pytest.approx(1.9)  # $0.10 left is below one call's worst case
    assert "skipped" in json.loads(out.read_text().splitlines()[-1])


@pytest.mark.parametrize(
    "kwargs",
    [{"status": 500}, {"model": "claude-without-a-price"}],
    ids=["server-error-may-be-billed", "reply-priced-at-no-known-rate"],
)
def test_a_call_whose_cost_cant_be_counted_stops_the_run_and_blocks_resuming(
    make_pipeline, handbook_pipeline, tmp_path, kwargs
):
    pipeline, sent = claude_pipeline(make_pipeline, handbook_pipeline, **kwargs)
    out = tmp_path / "out.jsonl"

    assert run(pipeline, [ROLLBACK, ROLLBACK], budget=2.0, out=out) is None
    assert len(sent) == 1  # stopped after the first call
    assert json.loads(out.read_text())["cost_unknown"] is True

    assert run(pipeline, [ROLLBACK], budget=2.0, out=out) is None  # not counted as $0 and resumed
    assert len(sent) == 1


def _injection_answer(case, extra=""):
    from sourced.chunking import Chunk
    from sourced.generation import Answer, Citation
    from sourced.retrieval import SearchResult

    source, section = case.gold[0]
    title = live_check.INJECTION_DOCS[source].splitlines()[0].lstrip("# ")
    body = live_check.INJECTION_DOCS[source].split("\n\n")[2].strip()
    passage = SearchResult(Chunk("c1", source, title, section, body, 0), 1.0)
    cite = Citation(1, source, passage.chunk.heading, [body.split(". ")[0]])
    return Answer(f"{body.split('. ')[0]}. {case.facts[0]}.[1] {extra}", [cite], "claude"), [passage]


@pytest.mark.parametrize(
    "case, planted",
    [(case, text) for case in live_check.INJECTIONS for text in case.forbidden] or [(None, None)],
)
def test_an_answer_that_repeats_planted_text_fails_even_with_right_facts_and_citations(case, planted):
    assert case is not None, "the injection cases list no forbidden text"
    assert not case.cite_every_gold

    good, passages = _injection_answer(case)
    obeyed, _ = _injection_answer(case, extra=planted)

    assert live_check.checks(case, good, passages) == {
        "citations_match_passages": True,
        "no_forbidden_text": True,
        "has_expected_facts": True,
        "cites_gold": True,
    }
    result = live_check.checks(case, obeyed, passages)
    assert result["no_forbidden_text"] is False
    assert result["has_expected_facts"] and result["citations_match_passages"]  # only the planted text fails it


@pytest.mark.parametrize("budget", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_run_refuses_a_budget_that_is_not_a_finite_positive_amount(make_pipeline, handbook_pipeline, tmp_path, budget):
    pipeline, sent = claude_pipeline(make_pipeline, handbook_pipeline)
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps({"id": "earlier", "cost_usd": 0.01}) + "\n")
    before = out.read_text()

    with pytest.raises(ValueError, match="budget"):
        run(pipeline, [ROLLBACK], budget=budget, out=out)

    assert sent == [] and out.read_text() == before


@pytest.mark.parametrize("budget", ["nan", "inf", "-inf", "0", "-2"])
def test_main_refuses_a_bad_budget_before_loading_models_or_opening_the_output(tmp_path, monkeypatch, capsys, budget):
    import sourced.embeddings

    def no_model(*args, **kwargs):
        raise AssertionError("a model was loaded")

    monkeypatch.setattr(sourced.embeddings, "FastEmbedEmbedder", no_model)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    out = tmp_path / "out.jsonl"

    argv = ["--index", str(tmp_path / "x.db"), "--out", str(out), "--confirm-spend", f"--budget-usd={budget}"]
    assert main(argv) == 2
    assert "budget" in capsys.readouterr().err and not out.exists()


@pytest.mark.parametrize("stop, content", [("max_tokens", "unfinished"), ("end_turn", " ")])
def test_incomplete_generation_is_billed_recorded_and_stops_without_retry(
    make_pipeline, handbook_pipeline, tmp_path, stop, content
):
    from sourced.generation import GenerationError

    pipeline, sent = claude_pipeline(make_pipeline, handbook_pipeline)
    messages = pipeline.answerer.client.beta.messages
    create = messages.create

    def incomplete(**kwargs):
        response = create(**kwargs)
        response.stop_reason = stop
        response.content[0].text = content
        return response

    messages.create = incomplete
    out = tmp_path / "out.jsonl"
    with pytest.raises(GenerationError):
        run(pipeline, [ROLLBACK, ROLLBACK], budget=2.0, out=out)

    record = json.loads(out.read_text())
    assert len(sent) == 1
    assert record["stop_reason"] == stop
    assert record["error"]["type"] == "GenerationError"
    assert "answer" not in record and "checks" not in record
    assert record["cost_usd"] == pytest.approx(0.008)
    assert live_check.spent_so_far(out) == pytest.approx(0.008)
    # A later invocation cannot forget the charged failed attempt.
    run(pipeline, [ROLLBACK], budget=0.008, out=out)
    assert len(sent) == 1


def test_main_stops_cleanly_after_charged_incomplete_generation(
    make_pipeline, handbook_pipeline, tmp_path, monkeypatch, capsys
):
    import sourced.embeddings

    pipeline, sent = claude_pipeline(make_pipeline, handbook_pipeline)
    sdk = pipeline.answerer.client
    create = sdk.beta.messages.create

    def incomplete(**kwargs):
        response = create(**kwargs)
        response.stop_reason = "max_tokens"
        return response

    sdk.beta.messages.create = incomplete
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: sdk)
    monkeypatch.setattr(sourced.embeddings, "FastEmbedEmbedder", lambda *a: pipeline.embedder)
    monkeypatch.setenv("SOURCED_RERANK", "false")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    out = tmp_path / "out.jsonl"
    assert (
        main(["--index", str(pipeline.settings.index_path), "--out", str(out), "--confirm-spend", "--only", "l02"]) == 1
    )
    assert len(sent) == 1
    assert "stopped:" in capsys.readouterr().out
    assert json.loads(out.read_text())["cost_usd"] == pytest.approx(0.008)


def test_incomplete_generation_with_missing_usage_blocks_resuming(make_pipeline, handbook_pipeline, tmp_path):
    pipeline, sent = claude_pipeline(make_pipeline, handbook_pipeline)
    messages = pipeline.answerer.client.beta.messages
    create = messages.create

    def incomplete(**kwargs):
        response = create(**kwargs)
        response.stop_reason = "max_tokens"
        response.usage = None
        return response

    messages.create = incomplete
    out = tmp_path / "out.jsonl"
    assert run(pipeline, [ROLLBACK, ROLLBACK], budget=2.0, out=out) is None
    assert json.loads(out.read_text())["cost_unknown"] is True
    assert run(pipeline, [ROLLBACK], budget=2.0, out=out) is None
    assert len(sent) == 1
