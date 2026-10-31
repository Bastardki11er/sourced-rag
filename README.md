# Sourced

[![CI](https://github.com/Lovio-Huang/sourced-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Lovio-Huang/sourced-rag/actions/workflows/ci.yml)
![Tested on Python 3.13](https://img.shields.io/badge/tested%20on-Python%203.13-blue)

Sourced answers questions about a folder of Markdown documents and shows, for every answer, the exact
passages it came from. It is a small retrieval-augmented generation (RAG) service built to be inspected:
each quality claim below links to how it was measured.

[Example](#example) · [Demo and replay](docs/demo.md) · [What has been checked](#what-has-been-checked) ·
[Performance](docs/performance.md) · [Try it](#try-it) · [Limitations](#limitations-and-next-steps)

**Answers and citations.** Sourced first retrieves the passages most likely to hold the answer. With
Claude credentials, Claude writes the answer from those passages only, and the API returns, for each cited
claim, the passage text it cites. A citation shows where a claim came from; it doesn't prove the claim is
right. Without credentials, an `extractive` mode skips the language model and returns the best-matching
passage verbatim: that is a retrieved passage, not a judged answer.

### Example

A stored Claude answer from the 2026-10-29 live check (`claude-opus-5-5`; not re-run).
The handbook belongs to a fictional company, so `kctl` is a command in its text, not one to run:

> **Question:** What is the command to roll back a release?
>
> **Answer** (as returned):
>
> The rollback command is `kctl rollback --to <release-id>`[1]
>
> - **No approval needed:** Rollbacks do not need approval and should be your first move when a release causes an incident.[1]
> - **Timing:** A rollback takes about 8 minutes to reach every site.[1]
> - **Follow-up:** After rolling back, open a ticket so the change can be fixed before the next train.[1]
>
> **[1]** `deployment.md`, *Deploying to Production > Rolling back*. Quoted text: "To roll back, run:
> `kctl rollback --to <release-id>` Rollbacks do not need approval and should be your first move when a
> release causes an incident. A rollback takes about 8 minutes to reach every site. After rolling back, open
> a ticket so the change can be fixed before the next train."

Every claim in the answer appears in the quote, and the quote is the
[source section](data/handbook/deployment.md#rolling-back) word for word (its code block shown inline here).
[docs/demo.md](docs/demo.md) has the exact text, how to check it, a no-answer example, an extractive run you
can reproduce and a short terminal replay.

## What has been checked

Unless the row says otherwise, on the bundled sample handbook: 14 short documents about a fictional company
(87 chunks).

| Claim | Result | Scope |
|---|---|---|
| The right passage is retrieved | hit@5 **0.98** (hybrid + rerank), 43 answerable questions; CI fails below 0.95 | retrieval only, not answer accuracy |
| …also on real documentation it wasn't written for | **Prometheus docs** (23 pages, 417 chunks, 24 answerable questions): hybrid + rerank hit@1 0.92, hit@5 1.00; BM25 alone hit@5 0.96 | one external corpus, questions written by the developer's assistant; not in CI ([details](docs/evaluation.md#external-corpus-prometheus-documentation)) |
| Retrieval is fast enough for interactive use | Prometheus docs, 417 chunks: hybrid 3.1–3.2 ms p50, hybrid + rerank 296–332 ms p50 (3 rounds) | one laptop, one query at a time; not a load test ([performance](docs/performance.md)) |
| Claude answers from the retrieved passages, with checkable citations | Two runs of 12 questions (2026-10-29 with the earlier prompt, 2026-10-31 with the current one): main facts right on all 10 answerable ones in both, both unanswerable ones declined, planted prompt injections not followed (though one citation quotes the planted text); $0.14 and $0.13 from response usage ([records](eval/live-claude/README.md)) | small samples, about one call per question; not an accuracy rate |
| The service behaves as documented | 236 offline tests; CI builds the Docker image and queries it | the CI smoke test uses the extractive mode |

**Costs** on an Apple M5 laptop (2026-10-29), measured one query or one ingest at a time: no concurrency, not
a load test.

| Chunks | Full ingest | Hybrid query p50, no reranking (3 rounds) | Index file |
|---|---|---|---|
| 417 (the 23 Prometheus pages) | 16.7 s | 3.1–3.2 ms | 1.7 MiB |
| 10,537 (the same 23 pages repeated 24 times) | 421 s | 6.2–6.5 ms | 40.7 MiB |

- The second row is **repeated text**: it shows how costs grow with the number of chunks, not how 10,000
  different documents would behave.
- **Reranking** costs 296–332 ms p50 per query at 417 chunks. It is not reported as a trend: in the repeated
  corpus its candidates are near-duplicates, so its timing there says nothing about larger real corpora.
- Method, memory, every round and the raw samples: [docs/performance.md](docs/performance.md); rerun with
  `python bench/run_benchmark.py`.

Methods, definitions, both retrieval evaluations and the full live-check record:
[docs/evaluation.md](docs/evaluation.md).

## Try it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install --no-deps -e .   # exact pins, the same install CI uses

sourced ingest data/handbook            # index the sample handbook (or your own docs)
sourced search "how do I roll back?"    # retrieval only, no language model
export ANTHROPIC_API_KEY=...            # optional: without it, ask returns the best passage verbatim
sourced ask "Who do I tell if my laptop is stolen?"
sourced serve                           # HTTP API on 127.0.0.1:8000, docs at /docs
sourced eval eval/questions.jsonl --rerank   # reproduce the retrieval table
```

Or with Docker (models and the sample index are baked into the image):

```bash
docker build -t sourced . && docker run -p 127.0.0.1:8000:8000 -e ANTHROPIC_API_KEY sourced
```

The service has no authentication: keep it on your own machine or a trusted network. Serving it to anyone
else needs authentication, access control and limits on Claude spending first.

## How it works

```mermaid
flowchart LR
    subgraph Ingest
        D[Markdown docs] --> C["Heading-aware chunker<br/>(adds 'Doc > Section' context)"]
        C --> F[(SQLite FTS5<br/>BM25 index)]
        C --> E[bge-small embeddings] --> V[(vectors in SQLite)]
    end
    subgraph Query
        Q([Question]) --> B[BM25 top 20]
        Q --> S[Dense top 20]
        B --> R[Reciprocal Rank Fusion]
        S --> R
        R --> X[Cross-encoder rerank] --> T[Top 5 passages]
        T --> L["Claude<br/>search_result blocks + citations"]
        L --> A([Answer + quoted sources])
    end
    F -.-> B
    V -.-> S
```

1. **Chunking follows the document's structure.** Each heading starts a chunk, and only long sections are split further, on paragraph boundaries with one paragraph of overlap. Each chunk is indexed with its heading path ("On-Call Handbook > Severity levels"), so a chunk that only says "acknowledge within 5 minutes" still matches a question about SEV1.
2. **Two retrievers, fused.** BM25 catches exact tokens like `KR-4021` or `kctl rollback`. Embeddings catch paraphrases ("file a receipt" means "submit expenses"). [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) merges the two rankings without trying to compare their incomparable scores.
3. **Reranking.** A cross-encoder reads the question and each candidate together and re-scores the top 20.
4. **Grounded generation.** The passages go to Claude as `search_result` content blocks with citations enabled, so the quoted text comes back as structured data taken from the passages, not as free text the model typed. If the passages don't contain the answer, the prompt tells Claude to say so instead of guessing.

## Engineering choices

| Choice | Why |
|---|---|
| Hybrid retrieval plus a reranker | On the handbook, BM25 alone missed paraphrases (0.70 hit@3); reranking gave the best ranking overall ([numbers](docs/evaluation.md#retrieval-quality)) |
| Claude `search_result` blocks with citations, not "please cite your sources" in the prompt | Citations come back as structured data (`cited_text`, `source`, `title`) taken from the provided text, so they can be shown and checked |
| Retrieval quality gate in CI | A chunking or retrieval change that drops hit@5 below 0.95 fails the build |
| The index records the model and chunking that built it | Mixing vectors from two models, or chunks from two chunkers, is refused instead of silently degrading ([details](docs/index-management.md#what-an-index-is-tied-to)) |
| One folder per index, one ingest at a time | Re-ingesting removes deleted files safely, and two ingests can't interleave ([details](docs/index-management.md#what-ingest-does)) |
| Explicit failures instead of silent fallbacks | An unfinished Claude reply is a 502, a reranking request without a reranker is a 409, an invalid `top_k` or setting is refused |
| SQLite for everything (FTS5 + vectors) | One file, no extra services, transactional re-indexing. Dense search scans every vector with NumPy and stayed under 3.5 ms p50 up to 10,537 chunks ([performance](docs/performance.md)); at what size an approximate vector index becomes necessary was not measured |
| Local ONNX embedding and reranking models | No per-query embedding cost or third-party data sharing, deterministic for evals, and the Docker image runs offline |
| Server-side refusal fallback (`fallbacks: "default"`) | If a safety classifier declines a benign question, the API can retry on a recommended fallback model (accepted by the API, not yet seen to trigger) |

## API

| Method | Path | Body | Returns | Errors |
|---|---|---|---|---|
| `POST` | `/ask` | `{"question": "...", "top_k": 5}` | `answer` with `[n]` markers, `citations` (source, heading, quoted text), retrieved `passages`, `timings_ms` | 409, 422, 502, 503 |
| `POST` | `/search` | `{"query": "...", "mode": "hybrid", "rerank": false}` | Ranked passages, with where each retriever ranked them | 409, 422 |
| `GET` | `/health` | | Document and chunk counts, active `answerer`, `reranking` (`"ok"` or `"disabled"`), `dense_search` (`"ok"` or why dense/hybrid search is refused) | |

Errors come back as `{"detail": "..."}`:

- **409**: dense or hybrid search (and so `/ask`) on an index whose vectors come from another embedding model,
  or from an old index that doesn't record its model ([details](docs/index-management.md#what-an-index-is-tied-to));
  or `"rerank": true` when the server has no reranker loaded.
- **422**: the body failed validation: `query` / `question` blank or over 1000 characters once surrounding
  whitespace is removed, `top_k` not an integer from 1 to 20, or an unknown `mode`.
- **502**: Claude returned an error, stopped before finishing (for example at `max_tokens`), or returned no
  answer text. Partial text is never returned as an answer.
- **503**: Claude is rate limited (429), overloaded (529), unreachable, or the request timed out; try again.

502 and 503 only come from `/ask` with the Claude answerer. `/docs` lists the same responses.

**Retries and timeouts.** The Claude client makes no automatic retries (`max_retries=0`), so a failed
`/ask` sends one HTTP request; asking again is up to the caller. The client's `timeout=60.0` is passed to its
HTTP library, which applies it to each step of the request (connecting, sending, waiting for the next data);
so it is not a total limit on the request, and even less on the whole `/ask` call, which also includes
retrieval and reranking.
Separately, the request enables the API's server-side refusal fallback: if Claude declines, the API itself may
answer with a fallback model inside that same request (not yet seen to trigger).

**CLI errors.** `sourced` prints one line, `sourced: error: ...`, and exits with code 2 for: an invalid `SOURCED_*` setting or a missing path; an index that
doesn't fit (another folder, model, chunking, or an ingest already running); a blank query or invalid
`--top-k`; requested reranking that isn't loaded; a document that is a link, hidden, unsupported or not valid
UTF-8 (named in the message); and a failed Claude call, described by kind (rate limit, overload, timeout,
connection, authentication or the HTTP status) without the upstream response. An unfinished or empty Claude
answer is an error too, never shown as an answer.

## Documentation

- [docs/demo.md](docs/demo.md): complete examples (historical Claude answers and a reproducible extractive run), with a terminal replay.
- [docs/evaluation.md](docs/evaluation.md): how retrieval (on the handbook and on the Prometheus docs) and answers were evaluated, the live Claude check, what CI runs, and what hasn't been measured.
- [docs/index-management.md](docs/index-management.md): ingesting, what an index is tied to, rebuilding and switching indexes.
- [docs/performance.md](docs/performance.md): ingest time, query latency, memory and index size at 417, 1,297 and 10,537 chunks, with the raw data.
- [docs/configuration.md](docs/configuration.md): every setting, `top_k` and reranking in the CLI vs the API, and where Sourced has been run.

## Tests

```bash
pytest -q        # 236 tests, ~3 s, no network: a hashing embedder stands in for the model
ruff check src tests bench eval docs
```

CI runs three jobs on every push to `main` and every pull request: lint and tests, the retrieval quality gate
with the real models, and a Docker build plus smoke test. What the tests cover:
[docs/evaluation.md](docs/evaluation.md#offline-tests).

`requirements.txt` (runtime) and `requirements-dev.txt` (runtime + test tools) pin every package to an exact
version; CI and the Docker image install from them, then add the package with `--no-deps`. `pyproject.toml`
keeps the looser ranges. To move the pins, edit `pyproject.toml` if needed and rerun the `uv pip compile`
command at the top of each file; Dependabot opens weekly PRs for the pins, the base image and the CI actions
(which are pinned to commit SHAs).

## Project layout

```
src/sourced/
├── chunking.py     heading-aware Markdown chunker
├── store.py        SQLite: documents, chunks, FTS5, embeddings
├── embeddings.py   fastembed (ONNX) embedder + deterministic test embedder
├── retrieval.py    BM25 / dense / RRF fusion / cross-encoder rerank
├── generation.py   Claude answerer with citations, extractive fallback
├── pipeline.py     ingest / search / ask
├── evaluation.py   hit@k, MRR, answer checks
├── live_check.py   opt-in paid check of real Claude answers
├── api.py          FastAPI app
└── cli.py          sourced ingest | search | ask | eval | serve
data/handbook/      sample corpus: a fictional robotics company's engineering handbook
eval/               labelled questions for the handbook
eval/prometheus/    external corpus (Prometheus docs, Apache-2.0), its questions, scripts and results
eval/live-claude/   public records of the live Claude checks (one folder per run), and the script that checks them
docs/               demo, evaluation, performance, index management, configuration
bench/              performance benchmark script and its published results
```

## Limitations and next steps

- **Answer quality is checked on 12 questions only.** `sourced eval --answers` runs keyword checks over all 48, but those numbers haven't been published. Next would be an LLM-as-judge for faithfulness on open-ended answers.
- **The corpus is small and synthetic** (14 documents, 87 chunks), so these results show the method, not production performance. The real test is a large, messy corpus with many near-duplicate passages.
- **Extractive mode has no abstention:** it always returns the top passage, even for an unrelated question.
- **English only, as far as tested:** the bundled models target English; other languages haven't been measured.
- **Citations quote whole chunks**, not sentences.
- **No authentication or per-user document permissions.** Answers and citations can quote anything in the index, so it only fits callers who may read every indexed document.
- **Ingest doesn't run on Windows** (it needs `fcntl` locks). **Streaming** answers and **PDF ingestion** are not implemented.

The sample handbook describes a fictional company and was written for this project.

## License

MIT
