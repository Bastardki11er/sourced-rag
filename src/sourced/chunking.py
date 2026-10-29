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

# Recorded in each index. Bump it whenever chunk boundaries or ids change, so an index
# built with the old rules is rebuilt instead of mixing old and new chunks.
CHUNKER_VERSION = "2"

# Closing #s count only after whitespace, so "## Notes on C#" keeps its "#".
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


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
    title = source
    headings: list[tuple[int, str]] = []
    sections: list[tuple[str, list[str]]] = []
    current: list[str] = []
    in_fence = False

    def flush() -> None:
        path = " > ".join(text for level, text in headings if level > 1)
        if any(line.strip() for line in current):
            sections.append((path, current.copy()))
        current.clear()

    for line in markdown.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
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
            # Everything retrieval reads goes into the id, so a renamed heading is a new chunk.
            digest = hashlib.sha1(f"{source}\0{position}\0{title}\0{section}\0{body}".encode()).hexdigest()[:16]
            chunks.append(Chunk(digest, source, title, section, body, position))
    return chunks


def _paragraphs(lines: list[str]) -> list[str]:
    """Groups lines into paragraphs, keeping fenced code blocks and lists intact."""
    paragraphs: list[str] = []
    buffer: list[str] = []
    in_fence = False
    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
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
    for paragraph in (piece for p in paragraphs for piece in _fit(p, max_chars)):
        candidate = "\n\n".join([*current, paragraph])
        if current and len(candidate) > max_chars:
            chunks.append("\n\n".join(current))
            # Repeat the previous paragraph only if both fit, the blank line between them included.
            overlap = len(current[-1]) + 2 + len(paragraph) <= max_chars
            current = [current[-1], paragraph] if overlap else [paragraph]
        else:
            current.append(paragraph)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


_SENTENCE_END = re.compile(r"([.!?][\"')\]]*)\s")


def _fit(paragraph: str, max_chars: int) -> list[str]:
    """Cuts a paragraph longer than ``max_chars`` into pieces that fit.

    Each cut goes after the last sentence end in the window, else at the last line
    break, else at the last space (sentence ends and line breaks only count in the
    second half, to avoid tiny pieces). A string with no break at all is cut at
    ``max_chars``. Only the whitespace at a cut is dropped.
    """
    pieces: list[str] = []
    rest = paragraph
    while len(rest) > max_chars:
        window = rest[: max_chars + 1]  # whitespace right after max_chars characters still allows a cut there
        half = max_chars // 2
        sentence = [m.end(1) for m in _SENTENCE_END.finditer(window) if m.end(1) > half]
        line = [i for i, ch in enumerate(window) if ch == "\n" and i > half]
        space = [i for i, ch in enumerate(window) if ch.isspace() and i > 0]
        cut = (sentence or line or space or [max_chars])[-1]
        pieces.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        pieces.append(rest)
    return pieces
