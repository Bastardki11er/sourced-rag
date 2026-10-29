"""Hybrid retrieval: BM25 and dense search, fused with Reciprocal Rank Fusion.

The two retrievers fail in opposite ways. BM25 nails exact tokens ("KR-4021",
"kctl rollback") but misses paraphrases; embeddings understand that "file a
receipt" means "submit expenses" but blur rare identifiers. RRF combines their
rankings without having to calibrate their incomparable scores:

    score(chunk) = sum over retrievers of 1 / (k + rank)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from .chunking import Chunk
from .embeddings import Embedder
from .store import Index, IndexConfigError

Mode = Literal["bm25", "dense", "hybrid"]

RRF_K = 60


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Re-scores the fused candidates by reading query and passage together."""

    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        return list(self._model.rerank(query, passages))


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)


def reciprocal_rank_fusion(rankings: dict[str, list[str]], k: int = RRF_K) -> list[tuple[str, float, dict]]:
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ids in rankings.items():
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(chunk_id, {})[name] = rank
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [(cid, scores[cid], ranks[cid]) for cid in ordered]


class Retriever:
    def __init__(
        self, index: Index, embedder: Embedder, reranker: Reranker | None = None, candidates: int = 20
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates

    def search(self, query: str, top_k: int = 5, mode: Mode = "hybrid", rerank: bool = False) -> list[SearchResult]:
        rankings: dict[str, list[str]] = {}
        if mode in ("bm25", "hybrid"):
            rankings["bm25"] = self.index.bm25(query, self.candidates)
        if mode in ("dense", "hybrid"):
            query_vector = self.embedder.embed_query(query)
            # Checked against what the embedder actually produced, before any matrix math.
            if problem := self.index.vector_space_problem(self.embedder.model_id, len(query_vector)):
                raise IndexConfigError(problem)
            rankings["dense"] = self.index.dense(query_vector, self.candidates)

        fused = reciprocal_rank_fusion(rankings)
        if rerank and self.reranker is not None:
            fused = self._rerank(query, fused)

        # Fetch every candidate and keep the first top_k that still exist. Another process
        # re-indexing can delete a chunk between ranking and fetching; matching by id keeps
        # each score with its own chunk, and the next candidate fills the gap.
        chunks = {c.id: c for c in self.index.get([cid for cid, _, _ in fused])}
        results = [SearchResult(chunks[cid], score, ranks) for cid, score, ranks in fused if cid in chunks]
        return results[:top_k]

    def _rerank(self, query: str, fused: list[tuple[str, float, dict]]) -> list[tuple[str, float, dict]]:
        candidates = fused[: self.candidates]
        ranks = {cid: r for cid, _, r in candidates}
        chunks = self.index.get(list(ranks))  # candidates that vanished are simply not returned
        scores = self.reranker.score(query, [c.indexed_text for c in chunks])
        # The scores were computed from exactly these chunks: one each, same order.
        rescored = [(c.id, float(s), ranks[c.id]) for c, s in zip(chunks, scores, strict=True)]
        return sorted(rescored, key=lambda item: item[1], reverse=True)
