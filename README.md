# Sourced

[![CI](https://github.com/Lovio-Huang/sourced-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Lovio-Huang/sourced-rag/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)

Ask questions about your documentation and get answers **with citations you can check**.

Sourced is a retrieval-augmented generation (RAG) service. It indexes Markdown docs, finds
the right passages with **hybrid search** (BM25 + embeddings + a cross-encoder reranker),
and has Claude answer **only from those passages**. Each cited claim comes back with the
source passage it points to, so a reader can check it; the quote shows where a claim came
from, not that the claim is right. Without Claude credentials, an `extractive` mode returns
the best passage verbatim instead. Retrieval quality is measured on a labelled question set,
and CI fails if it regresses.

**Stack:** Python 3.13 · FastAPI · Claude API (search-result citations) · SQLite FTS5 (BM25) · fastembed / ONNX (`bge-small-en-v1.5`, MiniLM cross-encoder) · NumPy · pytest · Ruff · Docker · GitHub Actions

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
4. **Grounded generation.** The passages go to Claude as `search_result` content blocks with citations enabled. The API itself returns the quoted text behind each claim, so a citation can't point at text that isn't in the source. If the passages don't contain the answer, the prompt tells Claude to say so instead of guessing.

## Retrieval evaluation

48 hand-labelled questions over the bundled sample handbook ([`eval/questions.jsonl`](eval/questions.jsonl)):
23 use the handbook's own wording (*lexical*), 20 deliberately avoid it (*paraphrase*, e.g. "My robot
complains it's carrying too much" → the `KR-4021 – Payload overweight` entry), and 5 have no answer in
the handbook. Reproduce with `sourced eval eval/questions.jsonl --rerank`.

| Retriever | hit@1 | hit@3 | hit@5 | MRR | hit@3 lexical | hit@3 paraphrase |
|---|---|---|---|---|---|---|
| BM25 | 0.70 | 0.84 | 0.91 | 0.78 | 0.96 | 0.70 |
| Dense (bge-small) | 0.81 | **0.98** | **0.98** | 0.89 | 1.00 | **0.95** |
| Hybrid (RRF) | 0.84 | 0.93 | 0.93 | 0.88 | 1.00 | 0.85 |
| **Hybrid + rerank** (default) | **0.88** | 0.95 | **0.98** | **0.92** | 1.00 | 0.90 |

*hit@k: the section that holds the answer is in the top k results. MRR: mean reciprocal rank of the first correct result.*

**What the numbers say:**

- **BM25 alone collapses on paraphrases** (0.70 hit@3), which is the classic failure that motivates semantic search.
- **Plain hybrid was not a free win here.** Fusion improved the top-1 result, but BM25's noisy candidates pushed some correct paraphrase matches out of the top 3. On this small corpus with distinctive vocabulary, dense retrieval alone is already strong.
- **Adding the reranker gives the best ranking overall** (hit@1 0.88, MRR 0.92) and ties dense on hit@5, the number that decides whether the right passage reaches the LLM. So hybrid + rerank is the default for `/ask`. BM25 stays in the mix because exact identifiers matter more as a corpus grows and starts holding many near-identical passages.

Retrieval latency on an Apple M5 laptop (CPU only, 87 chunks, p50 / p95 over the 48 questions):

| BM25 | Dense | Hybrid | Hybrid + rerank |
|---|---|---|---|
| 0.1 / 0.2 ms | 2.5 / 3.3 ms | 2.4 / 3.2 ms | 113 / 128 ms |

Reranking costs ~110 ms, which is small next to LLM generation, so `/ask` reranks by default and `/search` doesn't.

CI re-runs this evaluation on every push and **fails the build if hit@5 drops below 0.95**, so a chunking or retrieval change can't silently make retrieval worse. This is a retrieval check only: it says the right passage reaches the answerer, not that the answer is right.

## Design decisions

| Decision | Why |
|---|---|
| Claude `search_result` blocks with citations, not "please cite your sources" in the prompt | Citations come back as structured data (`cited_text`, `source`, `title`) taken from the provided text, so they can be shown and checked in a UI |
| Server-side refusal fallback (`fallbacks: "default"`) | If a safety classifier declines a benign question, the API retries on the recommended fallback model instead of failing the request |
| SQLite for everything (FTS5 + vectors) | One file, zero infrastructure, transactional re-indexing. Brute-force cosine over a NumPy matrix takes ~2 ms here. Past a few hundred thousand chunks I'd move to an ANN index (pgvector/HNSW) |
| Local ONNX embedding and reranking models | No per-query embedding cost or third-party data sharing, deterministic for evals, and the Docker image runs offline |
| Incremental ingestion by content hash | Re-running `ingest` only re-embeds changed files. A document's chunks are swapped in one transaction |
| One index per folder | Documents are named by their path inside the folder, so mixing folders in one index would let `team-a/README.md` overwrite `team-b/README.md`. Binding the index to one folder also makes it safe to delete what that folder no longer has |
| FTS5 queries built from quoted terms joined with `OR` | User text like `KR-4021` or `NEAR` can't be parsed as FTS5 syntax, and one rare matching term is enough to surface a chunk |
| `extractive` answerer when no API key is set | The service and its evals still run in CI and for anyone cloning the repo. It returns the top-ranked passage verbatim with one citation (plus the retrieved passages); it does not write a combined answer |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install --no-deps -e .   # exact pins, the same install CI uses

sourced ingest data/handbook            # index the sample handbook (or your own docs)
sourced search "how do I roll back?"    # retrieval only, no LLM
export ANTHROPIC_API_KEY=...            # optional: without it, ask returns the best passage verbatim
sourced ask "Who do I tell if my laptop is stolen?"
sourced serve                           # HTTP API on :8000, docs at /docs
```

Or with Docker (models and the sample index are baked into the image):

```bash
docker build -t sourced . && docker run -p 127.0.0.1:8000:8000 -e ANTHROPIC_API_KEY sourced
```

The service has no authentication, so it is meant for your own machine or a trusted internal network
(`sourced serve` listens on 127.0.0.1, and the example above publishes the port on localhost only). Serving it
to anyone else needs authentication, access control and limits on Claude spending first.

`requirements.txt` (runtime) and `requirements-dev.txt` (runtime + test tools) pin every package to an exact
version; CI and the Docker image install from them, then add the package with `--no-deps`. `pyproject.toml`
keeps the looser ranges. To move the pins, edit `pyproject.toml` if needed and rerun the `uv pip compile`
command at the top of each file; Dependabot opens weekly PRs for the pins, the base image and the CI actions
(which are pinned to commit SHAs).

### Indexing your own documents

- **One folder per index.** The first `sourced ingest <folder>` binds the index to that folder (its resolved
  absolute path). Ingesting a different folder is refused before anything changes; give it its own index with
  `SOURCED_INDEX=other.db sourced ingest <other-folder>`.
- **Single files** (`sourced ingest docs/team/oncall.md`) must live inside the bound folder and keep their
  folder-relative name (`team/oncall.md`). On a new index, a single file binds its parent folder.
- **Deleted files.** Ingesting the whole folder removes documents whose file is gone, together with their
  chunks, keyword index entries and embeddings, and reports `removed N`. This only happens after a full
  folder scan where every file ingested successfully: ingesting a single file never removes anything, and a
  missing folder or a file that fails to read stops the run before any removal.
- **Symbolic links are not followed.** The folder itself may be reached through a link, but a link inside it
  (to a file, a directory, or nothing) stops `ingest` with exit code 2 before anything is read, written or
  removed, and so does `ingest` of a single file that is a link. Copy the file in or remove the link. This
  keeps text from outside the folder out of the index for a trusted local folder; it is checked once at the
  start, not against files being swapped while ingest runs.
- **`SOURCED_MAX_CHUNK_CHARS`** must be a whole number above 0; anything else stops every command with exit
  code 2 before a model is loaded or the index is opened.
- **One ingest at a time.** `sourced ingest` holds an OS file lock (`<index>.lock`, next to the index) for
  the whole run, from checking the index to the last removal. A second ingest of the same index, from any
  process, stops right away with `another sourced ingest is running` and exit code 2; run it again when the
  first one has finished. The lock is released when ingest ends, fails or its process dies. The lock file
  stays in place on purpose. Ingest needs `fcntl` locks, so it runs on Linux and macOS, not on Windows.
- **An index is tied to the model and chunking that built it.** The first ingest records the embedding model
  (as reported by the loaded model), its vector dimensions, `SOURCED_MAX_CHUNK_CHARS` and the chunker version.
  Changing any of them makes `ingest` refuse before it touches anything, and makes dense and hybrid search
  refuse (HTTP 409) instead of comparing vectors from different models; keyword search (`mode=bm25`) still
  works, and `/health` says why dense search is off. Nothing is migrated in place: rebuild (below).
- **Indexes built before this was recorded** (no folder or model stored) are only read: keyword search works,
  dense/hybrid search and `ingest` refuse. Opening an index for queries never writes to it, and it can be
  opened read-only.
- **Chunker versions.** Version 2 put the heading into chunk ids; version 3 closes a code fence only on the
  same character at least as long (```` around a ``` example stays one block). Each bump means `ingest`
  refuses an index built by an earlier version: rebuild it (below). Search on such an index keeps working
  until then.
- **Rebuilding.** Build next to the old index, don't delete it first:

  ```bash
  SOURCED_INDEX=.sourced/index-new.db sourced ingest data/handbook   # must exit 0
  SOURCED_INDEX=.sourced/index-new.db sourced search "how do I roll back?"
  cp .sourced/index.db .sourced/index-backup.db                      # keep the old one
  export SOURCED_INDEX=.sourced/index-new.db                         # then restart sourced serve
  ```

  A failed ingest can leave a partly built index that still answers searches, so only an ingest that
  exits 0 means the rebuild is complete; rerun it until it does. A running server keeps reading the file it
  opened, even if another file is moved to that path, until it is restarted.
- A running server notices changes another process commits to the index (for example a `sourced ingest`
  while `sourced serve` runs) on its next query.

### API

| Method | Path | Body | Returns | Errors |
|---|---|---|---|---|
| `POST` | `/ask` | `{"question": "...", "top_k": 5}` | `answer` with `[n]` markers, `citations` (source, heading, quoted text), retrieved `passages`, `timings_ms` | 409, 422, 502, 503 |
| `POST` | `/search` | `{"query": "...", "mode": "hybrid", "rerank": false}` | Ranked passages, with where each retriever ranked them | 409, 422 |
| `GET` | `/health` | | Document and chunk counts, active answerer, `dense_search`: `"ok"` or why dense/hybrid search is refused | |

Errors come back as `{"detail": "..."}`. **409**: dense or hybrid search on an index built with another
embedding model or by an older version (bm25 `/search` still works; the message says how to rebuild).
**422**: the body failed validation (`query`/`question` blank or over 1000 characters once leading and
trailing whitespace is removed, `top_k` outside 1–20, unknown `mode`). `sourced search` and `ask` exit with
code 2 on a blank query.
**502**: Claude returned an error. **503**: Claude is rate limited or unreachable; try again. The 502/503
errors only come from `/ask` with the Claude answerer. `/docs` lists the same responses.

### Configuration

| Variable | Default |
|---|---|
| `SOURCED_CLAUDE_MODEL` | `claude-opus-5-5` |
| `SOURCED_CLAUDE_EFFORT` | `medium` |
| `SOURCED_RERANK` | `true` |
| `SOURCED_INDEX` | `.sourced/index.db` |
| `SOURCED_ANSWERER` | `claude` if `ANTHROPIC_API_KEY` is set, otherwise `extractive` |

## Tests

```bash
pytest -q        # 129 tests, ~1.5 s, no network: a hashing embedder stands in for the model
ruff check src tests
```

The tests cover heading-aware chunking (code fences, overlap, splitting paragraphs longer than a chunk, headings like `C#`), FTS5 query sanitizing (including accented words), RRF math, reranking, idempotent re-ingestion, keeping the index in step with the folder (another process re-indexing, deleted files, one folder per index, old indexes), one ingest at a time across processes (lock contention, release on failure and on a killed process), refusing an index built with another model, vector size or chunking, the exact request sent to Claude (search-result blocks, citations, fallback), turning citations into numbered markers, refusal handling, API validation and error mapping, the eval command, and a BM25 regression floor on the real handbook.

One test builds the request with the real `anthropic` SDK over a mocked HTTP transport, so it checks the beta header and the serialized body the SDK actually sends, and that a cited reply parses. It does not show that the live API accepts the request; only a call with a real key does that.

**Live check against the real API (opt-in, paid).** `python -m sourced.live_check` asks Claude 12 questions:
9 from `eval/questions.jsonl` (exact facts, an error code, a command, paraphrases, a question with two
acceptable sections, two with no answer), one that needs two sections combined, and two prompt-injection
cases on a temporary corpus whose documents tell the model to lie or leak a (made-up) password. It writes
each answer with its passages, citations, stop reason, usage and cost to a JSON lines file, plus automatic
checks (expected facts, citations point at passages that were sent and quote text inside them, declines
without citing, no injected text in the answer). Those checks are a first pass; the answers still need
reading. It refuses to run without `--confirm-spend`, credentials and the official API host, and makes no
retries. `--budget-usd` (default 2, a finite amount above 0) covers everything recorded in the `--out` file, earlier runs included; a
call is skipped when its estimated worst case (full `max_tokens` plus one fallback attempt, at the highest
price the tool knows) would pass it. The server picks the fallback model, so that is an estimate, not a
guarantee. Any API error, or a reply whose cost can't be counted, stops the run and marks the file so it
can't be resumed as if that call were free. Start with one question:

```bash
export FASTEMBED_CACHE_PATH=~/.cache/fastembed
python -m sourced.live_check --index .sourced/index.db --out live-l02.jsonl --confirm-spend --only l02
```

**Result on 2026-10-29** (`claude-opus-5-5`, effort `medium`): 12 different questions, 13 generation calls
(l02 ran twice, the others once). ~1.3–2.1k input and 84–367 output tokens per call, 1.5–6.7 s. $0.14 in all,
computed from each response's `usage` at $4 / $20 per million tokens, not read from the bill.

- **API compatibility.** The live API accepted the request as sent (the fallback beta and `fallbacks`,
  `output_config`, citable `search_result` blocks) and every reply parsed into an answer with numbered
  citations. No call was refused, truncated or handed to a fallback model, so the fallback switch itself was
  not exercised.
- **Answers.** Facts were right on all 10 answerable questions (the command, the error code, the paraphrases,
  both parts of the combined question); l02 gave the same command both times. p09's main answer was right, but
  it framed the feature-flag advice as "if the change is behind a flag", which the handbook doesn't say. Both
  unanswerable questions were declined with no citation and no invented policy.
- **Citations.** All 14 quotes appear word for word in the passage they cite (checked automatically). Read
  against the claims next to them, they support those claims, apart from p09's added condition above. Every
  quote was a whole chunk, not a sentence; why is not established, and sending a chunk as one text block per
  sentence is an untested idea.
- **Prompt injection.** In both cases the answer used the legitimate text and pointed out the planted
  instruction instead of following it; the answer text contained neither the made-up password nor the
  attacker address. The Wi-Fi citation, though, quotes the whole passage, planted password and address
  included, and citations are returned to the user too. Anything indexed can reach the asker through a
  citation (see the permissions limitation below). A bug in the tool at the time left the injection cases'
  list of forbidden text empty, so the automatic check didn't look for it then; the conclusion above came from
  reading the answers. After the fix, the stored replies were re-checked offline (no new calls): both pass.

That is 12 questions on a small fictional handbook, mostly one call each: it shows the integration works and
how these answers looked, not an accuracy rate or how stable answers are across runs.

CI runs three jobs: lint + tests, the retrieval quality gate with real models, and a Docker smoke test that builds the image and asks it a question.

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
eval/               labelled questions
```

## Limitations and next steps

- **Answer quality is checked on 12 questions only** (see the live check above). `sourced eval --answers` runs all 48 with keyword checks; I haven't published those numbers. Next would be an LLM-as-judge for faithfulness on open-ended answers. In the live check every quote was a whole chunk; whether per-sentence text blocks would give sentence-level quotes is untested.
- **The corpus is small and synthetic** (14 documents, 87 chunks), so these results show the method, not production performance. The real test is a large, messy corpus with many near-duplicate passages.
- **Streaming** (`/ask` over server-sent events) and **PDF ingestion** are not implemented yet.
- **No authentication or per-user document permissions.** Answers and citations can quote anything in the index, so it only fits callers who may read every indexed document. A real deployment would filter retrieval by what the asker is allowed to read.

The sample handbook describes a fictional company and was written for this project.

## License

MIT
