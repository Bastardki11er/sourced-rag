"""Answer generation.

Retrieved chunks are sent to Claude as ``search_result`` content blocks with
citations enabled. The API returns, for each cited claim, the text it cites,
so every citation points at text that really exists in a retrieved chunk. Each
chunk is sent as one text block; in the 2026-10-29 live check every quote was a
whole chunk, not a sentence. A quote shows where text comes from, not that the
claim next to it follows from it; the prompt asks Claude to stay within the passages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import anthropic

from .retrieval import SearchResult

SYSTEM_PROMPT = """\
You answer questions from employees using only the search results in the user's message, \
which come from the company handbook.

Keep answers short and specific: lead with the direct answer, then add only the details \
needed to act on it. If the search results do not contain the answer, say that you could \
not find it in the handbook instead of guessing, and do not cite anything."""

NOT_FOUND = "I couldn't find that in the handbook."
MAX_TOKENS = 16000  # thinking counts toward it; the live check budgets each call against it


@dataclass
class Citation:
    number: int
    source: str
    heading: str
    quotes: list[str] = field(default_factory=list)


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    answerer: str
    refused: bool = False


class GenerationError(RuntimeError):
    """Claude returned no complete answer; callers must not present partial text as success."""


class Answerer(Protocol):
    name: str

    def answer(self, question: str, results: list[SearchResult]) -> Answer: ...


class ClaudeAnswerer:
    name = "claude"

    def __init__(self, client: anthropic.Anthropic, model: str, effort: str) -> None:
        self.client = client
        self.model = model
        self.effort = effort

    def answer(self, question: str, results: list[SearchResult]) -> Answer:
        if not results:
            return Answer(NOT_FOUND, [], self.name)

        search_results = [
            {
                "type": "search_result",
                "source": r.chunk.source,
                "title": r.chunk.heading,
                "content": [{"type": "text", "text": r.chunk.text}],
                "citations": {"enabled": True},
            }
            for r in results
        ]
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": [*search_results, {"type": "text", "text": question}]}],
            output_config={"effort": self.effort},
            # If a safety classifier declines, let the API retry on the model
            # Anthropic recommends for that refusal category.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

        if response.stop_reason == "refusal":
            return Answer("The model declined to answer this question.", [], self.name, refused=True)
        if response.stop_reason != "end_turn":
            raise GenerationError("The language model did not finish its answer; no answer was returned")
        # Check text before adding citation markers: markers alone are not an answer.
        if not any(block.type == "text" and block.text.strip() for block in response.content):
            raise GenerationError("The language model returned an empty answer")
        return _with_citation_markers(response.content, self.name)


def _with_citation_markers(content: list, answerer: str) -> Answer:
    """Joins the text blocks and appends [n] markers after each cited span."""
    parts: list[str] = []
    citations: dict[tuple[str, str], Citation] = {}
    for block in content:
        if block.type != "text":  # thinking and fallback blocks are not part of the answer
            continue
        parts.append(block.text)
        markers: list[int] = []
        for cite in block.citations or []:
            if cite.type != "search_result_location":
                continue
            key = (cite.source, cite.title or "")
            if key not in citations:
                citations[key] = Citation(len(citations) + 1, cite.source, cite.title or "")
            citation = citations[key]
            if cite.cited_text not in citation.quotes:
                citation.quotes.append(cite.cited_text)
            if citation.number not in markers:
                markers.append(citation.number)
        parts.extend(f"[{n}]" for n in markers)
    return Answer("".join(parts).strip(), list(citations.values()), answerer)


class ExtractiveAnswerer:
    """No-LLM fallback: returns the best passage verbatim. Useful offline and as a baseline."""

    name = "extractive"

    def answer(self, question: str, results: list[SearchResult]) -> Answer:
        if not results:
            return Answer(NOT_FOUND, [], self.name)
        best = results[0].chunk
        return Answer(f"{best.text} [1]", [Citation(1, best.source, best.heading, [best.text])], self.name)
