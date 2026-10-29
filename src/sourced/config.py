"""Runtime settings, read from environment variables so the same code runs locally, in Docker and in CI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .chunking import check_chunk_size


class ConfigError(ValueError):
    """A setting has a value the service can't run with."""


@dataclass(frozen=True)
class Settings:
    index_path: Path = Path(".sourced/index.db")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "medium"
    # "claude" generates answers with citations; "extractive" returns the best passages
    # verbatim, so the service still works without an API key.
    answerer: str = "extractive"
    # Cross-encoder reranking for /ask: best measured retrieval quality, ~110 ms per query on CPU.
    rerank: bool = True
    max_chunk_chars: int = 1200
    top_k: int = 5

    def __post_init__(self) -> None:
        try:
            check_chunk_size(self.max_chunk_chars, "max_chunk_chars")
        except ValueError as exc:
            raise ConfigError(str(exc)) from None

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        has_key = bool(env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN"))
        return cls(
            index_path=Path(env.get("SOURCED_INDEX", cls.index_path)),
            embedding_model=env.get("SOURCED_EMBEDDING_MODEL", cls.embedding_model),
            rerank_model=env.get("SOURCED_RERANK_MODEL", cls.rerank_model),
            claude_model=env.get("SOURCED_CLAUDE_MODEL", cls.claude_model),
            claude_effort=env.get("SOURCED_CLAUDE_EFFORT", cls.claude_effort),
            answerer=env.get("SOURCED_ANSWERER", "claude" if has_key else "extractive"),
            rerank=env.get("SOURCED_RERANK", "true").lower() in ("1", "true", "yes"),
            max_chunk_chars=_chunk_chars(env.get("SOURCED_MAX_CHUNK_CHARS")),
            top_k=int(env.get("SOURCED_TOP_K", cls.top_k)),
        )


def _chunk_chars(raw: str | None) -> int:
    if raw is None:
        return Settings.max_chunk_chars
    try:
        return check_chunk_size(int(raw.strip()), "SOURCED_MAX_CHUNK_CHARS")
    except ValueError:
        raise ConfigError(
            f"SOURCED_MAX_CHUNK_CHARS must be a whole number of characters above 0, got {raw!r}"
        ) from None
