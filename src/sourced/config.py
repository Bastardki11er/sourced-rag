"""Runtime settings, read from environment variables so the same code runs locally, in Docker and in CI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .chunking import check_chunk_size


class ConfigError(ValueError):
    """A setting has a value the service can't run with."""


MAX_TOP_K = 20


def check_top_k(value: int, name: str = "top_k") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_TOP_K:
        raise ConfigError(f"{name} must be an integer between 1 and {MAX_TOP_K}, got {value!r}")
    return value


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
        check_top_k(self.top_k)
        if self.answerer not in ("claude", "extractive"):
            raise ConfigError("SOURCED_ANSWERER (answerer) must be 'claude' or 'extractive'")
        if not isinstance(self.rerank, bool):
            raise ConfigError("SOURCED_RERANK (rerank) must be a boolean")
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
            rerank=_rerank(env.get("SOURCED_RERANK", "true")),
            max_chunk_chars=_chunk_chars(env.get("SOURCED_MAX_CHUNK_CHARS")),
            top_k=_top_k(env.get("SOURCED_TOP_K", str(cls.top_k))),
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


def _top_k(raw: str) -> int:
    try:
        return check_top_k(int(raw.strip()), "SOURCED_TOP_K")
    except ValueError:
        raise ConfigError(f"SOURCED_TOP_K must be an integer between 1 and {MAX_TOP_K}, got {raw!r}") from None


def _rerank(raw: str) -> bool:
    value = raw.strip().lower()
    if value in ("1", "true", "yes"):
        return True
    if value in ("0", "false", "no"):
        return False
    raise ConfigError(f"SOURCED_RERANK must be true/false, yes/no or 1/0, got {raw!r}")
