"""HTTP API.

uvicorn sourced.api:app            # or: sourced serve
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .config import Settings
from .pipeline import Pipeline, build_pipeline
from .retrieval import SearchResult


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    rerank: bool = False


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool | None = None  # None: use the server default (SOURCED_RERANK)


class Passage(BaseModel):
    source: str
    heading: str
    text: str
    score: float
    ranks: dict[str, int]


class CitationOut(BaseModel):
    number: int
    source: str
    heading: str
    quotes: list[str]


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    answerer: str
    refused: bool
    passages: list[Passage]
    timings_ms: dict[str, float]


def _passage(result: SearchResult) -> Passage:
    return Passage(
        source=result.chunk.source,
        heading=result.chunk.heading,
        text=result.chunk.text,
        score=result.score,
        ranks=result.ranks,
    )


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or build_pipeline(Settings.from_env())
        yield
        if pipeline is None:  # an injected pipeline belongs to whoever created it
            app.state.pipeline.index.close()

    app = FastAPI(title="Sourced", summary="Answers from your docs, with citations.", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict:
        p: Pipeline = request.app.state.pipeline
        return {
            "status": "ok",
            "documents": len(p.index.sources()),
            "chunks": p.index.chunk_count(),
            "answerer": p.answerer.name,
        }

    @app.post("/search", response_model=list[Passage])
    async def search(body: SearchRequest, request: Request) -> list[Passage]:
        p: Pipeline = request.app.state.pipeline
        results = await run_in_threadpool(p.search, body.query, body.top_k, body.mode, body.rerank)
        return [_passage(r) for r in results]

    @app.post("/ask", response_model=AskResponse)
    async def ask(body: AskRequest, request: Request) -> AskResponse:
        p: Pipeline = request.app.state.pipeline
        try:
            result = await run_in_threadpool(p.ask, body.question, body.top_k, body.rerank)
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as exc:
            raise HTTPException(503, "The language model is temporarily unavailable, try again shortly") from exc
        except anthropic.APIStatusError as exc:
            raise HTTPException(502, f"The language model returned an error ({exc.status_code})") from exc
        return AskResponse(
            answer=result.answer.text,
            citations=[CitationOut(**vars(c)) for c in result.answer.citations],
            answerer=result.answer.answerer,
            refused=result.answer.refused,
            passages=[_passage(r) for r in result.results],
            timings_ms=result.timings_ms,
        )

    return app


app = create_app()
