"""Offline evaluation.

Retrieval is scored against hand-labelled questions: for each question we know
which handbook section holds the answer, and measure how high the retriever
ranks it.

  hit@k  share of questions whose answer section appears in the top k results
  MRR    mean of 1/rank of the first correct result (0 if not in the top 10)

Answer quality (``--answers``) calls the configured answerer and checks that
the answer contains the expected facts, cites the right section, and declines
when the handbook has no answer.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunk
from .generation import NOT_FOUND
from .pipeline import Pipeline

K_VALUES = (1, 3, 5)
MRR_DEPTH = 10


@dataclass
class Question:
    id: str
    question: str
    kind: str
    gold: list[dict] = field(default_factory=list)
    answer_contains: list[str] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return bool(self.gold)

    def is_gold(self, chunk: Chunk) -> bool:
        return any(chunk.source == g["source"] and g["section"] in chunk.section for g in self.gold)


def load_questions(path: Path) -> list[Question]:
    with path.open() as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]


@dataclass
class RetrievalScores:
    questions: int
    hits: dict[int, float]
    mrr: float


def first_gold_rank(question: Question, chunks: Iterable[Chunk]) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if question.is_gold(chunk):
            return rank
    return None


def score_retrieval(pipeline: Pipeline, questions: list[Question], mode: str, rerank: bool = False) -> RetrievalScores:
    ranks: list[int | None] = []
    for q in questions:
        results = pipeline.search(q.question, top_k=MRR_DEPTH, mode=mode, rerank=rerank)
        ranks.append(first_gold_rank(q, (r.chunk for r in results)))
    n = len(ranks)
    hits = {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in K_VALUES}
    mrr = sum(1 / r for r in ranks if r is not None) / n
    return RetrievalScores(n, hits, mrr)


def retrieval_report(pipeline: Pipeline, questions: list[Question], with_rerank: bool) -> str:
    answerable = [q for q in questions if q.answerable]
    kinds = sorted({q.kind for q in answerable})
    configs = [("BM25", "bm25", False), ("Dense", "dense", False), ("Hybrid (RRF)", "hybrid", False)]
    if with_rerank:
        configs.append(("Hybrid + rerank", "hybrid", True))

    header = (
        "| Retriever | "
        + " | ".join(f"hit@{k}" for k in K_VALUES)
        + " | MRR | "
        + " | ".join(f"hit@3 ({kind})" for kind in kinds)
        + " |"
    )
    breakdown = ", ".join(f"{sum(q.kind == k for q in answerable)} {k}" for k in kinds)
    lines = [
        f"{len(answerable)} answerable questions ({breakdown})",
        "",
        header,
        "|" + "---|" * (len(K_VALUES) + 2 + len(kinds)),
    ]
    for label, mode, rerank in configs:
        overall = score_retrieval(pipeline, answerable, mode, rerank)
        by_kind = [score_retrieval(pipeline, [q for q in answerable if q.kind == k], mode, rerank) for k in kinds]
        lines.append(
            f"| {label} | "
            + " | ".join(f"{overall.hits[k]:.2f}" for k in K_VALUES)
            + f" | {overall.mrr:.2f} | "
            + " | ".join(f"{s.hits[3]:.2f}" for s in by_kind)
            + " |"
        )
    return "\n".join(lines)


def answer_report(pipeline: Pipeline, questions: list[Question]) -> str:
    correct = cited_right = declined = 0
    answerable = [q for q in questions if q.answerable]
    unanswerable = [q for q in questions if not q.answerable]
    failures: list[str] = []

    for q in questions:
        result = pipeline.ask(q.question)
        text = result.answer.text.lower()
        if q.answerable:
            ok = all(fact.lower() in text for fact in q.answer_contains)
            correct += ok
            cited = {(c.source, c.heading) for c in result.answer.citations}
            cited_right += any(src == g["source"] and g["section"] in heading for src, heading in cited for g in q.gold)
            if not ok:
                failures.append(f"- {q.id}: expected {q.answer_contains}, got: {result.answer.text[:160]!r}")
        else:
            said_not_found = (
                NOT_FOUND.lower().rstrip(".") in text or "couldn't find" in text or "could not find" in text
            )
            declined += said_not_found and not result.answer.citations
            if not said_not_found:
                failures.append(f"- {q.id}: should have declined, got: {result.answer.text[:160]!r}")

    lines = [
        f"Answerer: {pipeline.answerer.name}",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Contains the expected facts | {correct}/{len(answerable)} |",
        f"| Cites the correct section | {cited_right}/{len(answerable)} |",
        f"| Declines when the handbook has no answer | {declined}/{len(unanswerable)} |",
    ]
    if failures:
        lines += ["", "Failures:", *failures]
    return "\n".join(lines)
