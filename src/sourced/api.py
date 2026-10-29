"""HTTP API.

uvicorn sourced.api:app            # or: sourced serve
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Literal

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, StringConstraints

from .config import Settings
from .pipeline import IndexConfigError, Pipeline, build_pipeline
from .retrieval import SearchResult

# Leading and trailing whitespace is dropped before the length check, so a blank query is a 422.
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class SearchRequest(BaseModel):
    query: Text
    top_k: int = Field(default=5, ge=1, le=20)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    rerank: bool = False


class AskRequest(BaseModel):
    question: Text
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


class ErrorOut(BaseModel):
    detail: str


# Documents the errors the handlers below raise, so /docs matches them (422 is FastAPI's own).
_INDEX_MISMATCH = {
    409: {
        "model": ErrorOut,
        "description": "Dense or hybrid search on an index built with another model or by an older version",
    }
}
_LLM_ERRORS = {
    502: {"model": ErrorOut, "description": "The language model returned an error"},
    503: {"model": ErrorOut, "description": "The language model is rate limited or unreachable"},
}


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
            "dense_search": p.index.vector_space_problem(p.embedder.model_id, p.embedder.dimensions) or "ok",
        }

    @app.post("/search", response_model=list[Passage], responses=_INDEX_MISMATCH)
    async def search(body: SearchRequest, request: Request) -> list[Passage]:
        p: Pipeline = request.app.state.pipeline
        try:
            results = await run_in_threadpool(p.search, body.query, body.top_k, body.mode, body.rerank)
        except IndexConfigError as exc:
            raise HTTPException(409, str(exc)) from exc
        return [_passage(r) for r in results]

    @app.post("/ask", response_model=AskResponse, responses={**_INDEX_MISMATCH, **_LLM_ERRORS})
    async def ask(body: AskRequest, request: Request) -> AskResponse:
        p: Pipeline = request.app.state.pipeline
        try:
            result = await run_in_threadpool(p.ask, body.question, body.top_k, body.rerank)
        except IndexConfigError as exc:  # the index's vectors don't match this server's model
            raise HTTPException(409, str(exc)) from exc
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
