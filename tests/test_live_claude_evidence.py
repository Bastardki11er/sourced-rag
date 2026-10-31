"""The published evidence of the live Claude checks, and the tool that builds and checks it (offline)."""

import importlib.util
import json
from pathlib import Path

import pytest

EVIDENCE = Path(__file__).parent.parent / "eval" / "live-claude"
spec = importlib.util.spec_from_file_location("extract_public", EVIDENCE / "extract_public.py")
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)

RUNS = {
    "2026-10-29": {"calls": 13, "distinct_questions": 12, "repeated": ["l02"], "total_cost_usd": 0.14146, "quotes": 15},
    "2026-10-31": {"calls": 12, "distinct_questions": 12, "repeated": [], "total_cost_usd": 0.13156, "quotes": 14},
}


@pytest.mark.parametrize("run", RUNS)
def test_each_published_run_recounts_and_its_quotes_are_in_their_sources(run):
    summary = extract.check(EVIDENCE / run / "calls.jsonl")

    assert summary["problems"] == []
    assert {k: summary[k] for k in RUNS[run]} == RUNS[run]
    assert summary["quotes_equal_to_whole_passage"] == summary["quotes"]
    assert summary["fallback_attempts"] == 0 and summary["stop_reasons"] == ["end_turn"]


@pytest.mark.parametrize("run", RUNS)
def test_only_the_listed_fields_are_published(run):
    path = EVIDENCE / run / "calls.jsonl"
    allowed = {"call", "citations", "retrieved", "usage", "usage_iterations", "cost_usd", "cost_usd_as_recorded"}
    allowed |= set(extract.KEEP) | {"checks_recorded_at_run_time", "checks_rerun_offline"}
    for row in extract.read_jsonl(path):
        assert set(row) <= allowed
        assert set(row["usage"]) <= set(extract.USAGE_KEEP) | {"thinking_tokens"}
        assert all(set(i) <= set(extract.ITERATION_KEEP) for i in row["usage_iterations"])
    text = path.read_text()
    for private in ("/Users/", "passage_texts", "inference_geo", "x-api-key", "sk-ant-", "authorization"):
        assert private not in text


def test_build_never_overwrites_published_evidence(tmp_path, capsys):
    existing = EVIDENCE / "2026-10-29" / "calls.jsonl"
    before = existing.read_bytes()

    with pytest.raises(SystemExit, match="already exists"):
        extract.main(["build", str(tmp_path / "any.jsonl"), "--out", str(existing)])
    assert existing.read_bytes() == before


@pytest.mark.parametrize(
    "content, message",
    [(None, "no such file"), ("not json\n", "not valid JSON"), ('{"id": "l02"}\n', "not a live_check result")],
    ids=["missing", "invalid-json", "not-a-record"],
)
def test_build_explains_bad_input_and_writes_nothing(tmp_path, content, message):
    records = tmp_path / "records.jsonl"
    if content is not None:
        records.write_text(content)
    out = tmp_path / "out" / "calls.jsonl"

    with pytest.raises(SystemExit, match=message):
        extract.main(["build", str(records), "--out", str(out)])
    assert not out.exists()


def test_check_reports_a_tampered_quote_and_cost(tmp_path):
    rows = extract.read_jsonl(EVIDENCE / "2026-10-31" / "calls.jsonl")
    rows[0]["citations"][0]["quotes"][0] += " (added)"
    rows[1]["cost_usd"] += 1
    tampered = tmp_path / "calls.jsonl"
    tampered.write_text("".join(json.dumps(r) + "\n" for r in rows))

    problems = extract.check(tampered)["problems"]

    assert any("quote not inside" in p for p in problems) and any("cost_usd" in p for p in problems)
    assert extract.main(["check", str(tampered)]) == 1
