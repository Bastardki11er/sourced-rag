# Configuration

All settings come from environment variables, read once when a command or the server starts
(`Settings.from_env` in [`src/sourced/config.py`](../src/sourced/config.py)). Sourced does **not** read a
`.env` file; set variables in the shell, in the command line (`SOURCED_INDEX=... sourced serve`) or with
`docker run -e`. Invalid values of `SOURCED_MAX_CHUNK_CHARS`, `SOURCED_RERANK`, `SOURCED_TOP_K` and
`SOURCED_ANSWERER` stop the command (exit 2) or the server start before any model is loaded. Model names are
checked by fastembed when the model loads; the Claude model and effort are only checked by the API.

## Variables

| Variable | Default | Accepted values | Index compatibility |
|---|---|---|---|
| `SOURCED_INDEX` | `.sourced/index.db` | a file path; its folder is created if needed | which index is used |
| `SOURCED_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | a fastembed text embedding model name | **yes**: recorded in the index |
| `SOURCED_MAX_CHUNK_CHARS` | `1200` | a whole number above 0 | **yes**: recorded in the index |
| `SOURCED_RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | a fastembed cross-encoder model name | no |
| `SOURCED_RERANK` | `true` | `true`/`false`, `yes`/`no`, `1`/`0` (any case) | no |
| `SOURCED_TOP_K` | `5` | an integer from 1 to 20 | no |
| `SOURCED_ANSWERER` | `claude` if `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` is set, otherwise `extractive` | `claude` or `extractive` only | no |
| `SOURCED_CLAUDE_MODEL` | `claude-opus-5-5` | passed to the Claude API as is | no |
| `SOURCED_CLAUDE_EFFORT` | `medium` | passed to the Claude API as is | no |

Other variables that matter:

- `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN`: read by the Anthropic SDK; their presence also picks the
  default answerer. Sourced never prints or stores them.
- `FASTEMBED_CACHE_PATH`: fastembed's own setting for where downloaded models are kept. Set it to a lasting
  folder (the Docker image uses `/app/models`) so models aren't downloaded again.

The embedding model and chunk size are recorded by the first ingest; with other values, `ingest` refuses
the index, and a different embedding model also stops dense and hybrid search. The chunker version is part
of the code, not a setting. Details: [index-management.md](index-management.md#what-an-index-is-tied-to).

## `top_k`: CLI and API differ

- **API** (`/search`, `/ask`): `top_k` in the request body, default **5** when omitted. The API default does
  not read `SOURCED_TOP_K`.
- **CLI and Python** (`sourced search`, `sourced ask`, `Pipeline.search` / `ask`): an omitted `top_k` uses
  `SOURCED_TOP_K` (default 5); an explicit `--top-k` / `top_k=` overrides it.
- Everywhere the value must be an integer from 1 to 20; `0`, negative numbers, booleans and decimals are
  refused (API: 422; CLI: exit 2), never replaced by the default.

## Reranking: server default and explicit requests

- **Server** (`sourced serve`, `uvicorn sourced.api:app`): `SOURCED_RERANK=true` loads the cross-encoder
  and makes `/ask` rerank unless the request sends `"rerank": false`. `/search` reranks only when the request
  sends `"rerank": true`. With `SOURCED_RERANK=false` no reranker is loaded, and a request with
  `"rerank": true` gets HTTP 409 instead of unreranked results. `/health` reports `reranking: "ok"` or
  `"disabled"`.
- **CLI**: `sourced search --rerank` and `sourced eval --rerank` load the reranker for that command, whatever
  `SOURCED_RERANK` says. `sourced ask` follows `SOURCED_RERANK`; `--no-rerank` skips it. `sourced eval
  --answers` reranks only with `--rerank`.

## Answerer

- `claude`: Claude writes the answer from the retrieved passages, with structured citations.
- `extractive`: no language model. `/ask` returns the top-ranked passage verbatim with one citation. It does
  not judge whether the passage answers the question, so even an unrelated question gets a passage with
  `refused: false`; a citation is not evidence that the answer is right.

## Where it has been run

- **Python**: 3.13 is what CI, the Docker image and local development use. `pyproject.toml` allows 3.11+,
  but 3.11 and 3.12 are not tested.
- **Operating systems**: Linux (CI, Docker) and macOS (local development). On Windows, `ingest` refuses to
  run because the ingest lock needs `fcntl`; other commands have not been tried there.
- **Languages**: the bundled embedding and reranking models target English. Retrieval quality for other
  languages has not been measured.
- **Deployment**: the server has no authentication. `sourced serve` listens on 127.0.0.1 by default; keep it
  on your own machine or a trusted network.
