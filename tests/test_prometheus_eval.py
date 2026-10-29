"""The external-corpus evaluation in eval/prometheus: conversion, corpus hashes and label checks (offline)."""

import importlib.util
import json
from pathlib import Path

import pytest

EVAL = Path(__file__).parent.parent / "eval" / "prometheus"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EVAL / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare, run_eval = load("prepare_corpus"), load("run_eval")


def test_only_the_leading_front_matter_is_removed():
    page = "---\ntitle: Storage\nsort_rank: 5\n---\n\n# Storage\n\nText\n\n---\n\nmore\n"

    assert prepare.strip_front_matter(page) == "\n# Storage\n\nText\n\n---\n\nmore\n"
    assert prepare.strip_front_matter("# No front matter\n") == "# No front matter\n"
    assert prepare.structure(page) == prepare.structure(prepare.strip_front_matter(page))


def test_the_committed_corpus_matches_its_manifest():
    assert prepare.check() == 0
    manifest = json.loads((EVAL / "manifest.json").read_text())
    assert manifest["commit"] == prepare.COMMIT and len(manifest["files"]) == len(prepare.FILES)


def test_the_committed_labels_hold_and_match_the_frozen_baseline():
    assert run_eval.check_labels() == []
    baseline = json.loads((EVAL / "baseline.json").read_text())
    hashes = run_eval.file_hashes()
    assert hashes["questions.jsonl"] == baseline["questions.jsonl"]
    assert hashes["evidence.jsonl"] == baseline["evidence.jsonl"]


@pytest.mark.parametrize(
    "change, problem",
    [
        (lambda q, e: q[0]["gold"].__setitem__(0, {"source": "storage.md", "section": "Nope"}), "no chunk"),
        (lambda q, e: e[0]["support"][0].__setitem__("quote", "words that are not in the page"), "quote not found"),
        (lambda q, e: q.append(dict(q[0])), "duplicate id"),
        (lambda q, e: q[0].__setitem__("kind", "unanswerable"), "doesn't match its gold"),
    ],
    ids=["missing-section", "wrong-quote", "duplicate-id", "kind-vs-gold"],
)
def test_a_broken_label_is_reported(tmp_path, monkeypatch, change, problem):
    questions = [json.loads(line) for line in (EVAL / "questions.jsonl").read_text().splitlines()]
    evidence = [json.loads(line) for line in (EVAL / "evidence.jsonl").read_text().splitlines()]
    change(questions, evidence)
    for name, rows in (("questions.jsonl", questions), ("evidence.jsonl", evidence)):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(run_eval, "QUESTIONS", tmp_path / "questions.jsonl")
    monkeypatch.setattr(run_eval, "EVIDENCE", tmp_path / "evidence.jsonl")

    assert any(problem in p for p in run_eval.check_labels())
