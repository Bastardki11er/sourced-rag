"""Markdown-aware chunking.

Chunks follow the document's own structure: each heading starts a new chunk, and
only sections longer than ``max_chars`` are split further, on paragraph
boundaries. Every chunk remembers its heading path ("On-Call Handbook > Severity
levels"), which is prepended when indexing so that a chunk saying "acknowledge
within 5 minutes" is still findable by a query about "SEV1".
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _next_fence(line: str, fence: str | None) -> str | None:
    match = _FENCE.match(line)
    if match is None:
        return fence
    marker, tail = match.groups()
    if fence is None:
        return marker
    if marker[0] == fence[0] and len(marker) >= len(fence) and not tail.strip():
        return None
    return fence


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    title: str
    section: str
    text: str
    position: int

    @property
    def heading(self) -> str:
        """Full heading path, e.g. "On-Call Handbook > Severity levels"."""
        return f"{self.title} > {self.section}" if self.section else self.title

    @property
    def indexed_text(self) -> str:
        """What retrieval sees: the heading path followed by the body."""
        return f"{self.heading}\n\n{self.text}"


def chunk_markdown(source: str, markdown: str, max_chars: int = 1200) -> list[Chunk]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    title = source
    headings: list[tuple[int, str]] = []
    sections: list[tuple[str, list[str]]] = []
    current: list[str] = []
    fence: str | None = None

    def flush() -> None:
        path = " > ".join(text for level, text in headings if level > 1)
        if any(line.strip() for line in current):
            sections.append((path, current.copy()))
        current.clear()

    for line in markdown.splitlines():
        previous_fence = fence
        fence = _next_fence(line, fence)
        match = None if previous_fence or fence else _HEADING.match(line)
        if match:
            flush()
            level, text = len(match.group(1)), match.group(2)
            if level == 1 and title == source:
                title = text
            headings = [(lvl, txt) for lvl, txt in headings if lvl < level] + [(level, text)]
        else:
            current.append(line)
    flush()

    chunks: list[Chunk] = []
    for section, lines in sections:
        for body in _split_long(_paragraphs(lines), max_chars):
            position = len(chunks)
            digest = hashlib.sha1(f"{source}\0{position}\0{body}".encode()).hexdigest()[:16]
            chunks.append(Chunk(digest, source, title, section, body, position))
    return chunks


def _paragraphs(lines: list[str]) -> list[str]:
    """Groups lines into paragraphs, keeping fenced code blocks and lists intact."""
    paragraphs: list[str] = []
    buffer: list[str] = []
    fence: str | None = None
    for line in lines:
        fence = _next_fence(line, fence)
        if not line.strip() and not fence:
            if buffer:
                paragraphs.append("\n".join(buffer).strip())
                buffer = []
        else:
            buffer.append(line)
    if buffer:
        paragraphs.append("\n".join(buffer).strip())
    return [p for p in paragraphs if p]


def _split_long(paragraphs: list[str], max_chars: int) -> list[str]:
    """Packs paragraphs into chunks of at most ``max_chars``.

    Consecutive chunks of the same section share one paragraph of overlap, so a
    fact that straddles the boundary appears whole in at least one chunk.
    """
    chunks: list[str] = []
    current: list[str] = []
    bounded: list[str] = []
    for paragraph in paragraphs:
        # A single paragraph (or code block) can itself exceed the model budget.
        # Prefer a whitespace boundary, but also handle long unbroken tokens.
        while len(paragraph) > max_chars:
            boundary = max(paragraph.rfind(" ", 0, max_chars + 1), paragraph.rfind("\n", 0, max_chars + 1))
            end = boundary if boundary > 0 else max_chars
            bounded.append(paragraph[:end])
            paragraph = paragraph[end:].lstrip()
        if paragraph:
            bounded.append(paragraph)
    for paragraph in bounded:
        candidate = "\n\n".join([*current, paragraph])
        if current and len(candidate) > max_chars:
            chunks.append("\n\n".join(current))
            current = [current[-1], paragraph] if len(current[-1]) + len(paragraph) + 2 <= max_chars else [paragraph]
        else:
            current.append(paragraph)
    if current:
        chunks.append("\n\n".join(current))
    return chunks
