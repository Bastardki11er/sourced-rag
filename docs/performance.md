# Performance

How long ingest and retrieval take, how much memory the process uses and how large the index gets, at three
corpus sizes. One machine, one process at a time, one query at a time: there is no HTTP layer, no
concurrency and no load test, so none of this says how many users or requests per second the service can
handle.

Raw data, including every timed sample: [`bench/results/20261029T230115Z/raw.json`](../bench/results/20261029T230115Z/raw.json).
Summary computed from it: [`summary.json`](../bench/results/20261029T230115Z/summary.json).

## Setup

- **Machine:** Apple M5 (10 logical CPUs), 24 GiB RAM, macOS 27.0.1, arm64.
- **Software:** Python 3.13.15, SQLite 3.50.4, NumPy 2.5.3, fastembed 0.9.0, onnxruntime 1.31.0. Thread counts
  are the library defaults; no thread environment variables were set.
- **Models:** `BAAI/bge-small-en-v1.5` embeddings (384 dimensions) and the `Xenova/ms-marco-MiniLM-L-6-v2`
  cross-encoder, both from the local cache. Chunker version 3, `max_chunk_chars` 1200.
- **Code:** commit `49b0871` with no uncommitted changes under `src/`; the benchmark script's SHA-256 and that
  of every source file are in `raw.json`.
- **Run:** 2026-10-29, 23:01–23:11 UTC, about 10 minutes in all.

## Data

| Scale | Built from | Documents | Chunks (after chunking) |
|---|---|---|---|
| base | the 23 Prometheus pages of the [external evaluation](../eval/prometheus/README.md) | 23 | 417 |
| 1k | 3 copies of those 23 pages | 69 | 1,297 |
| 10k | 24 copies of those 23 pages | 552 | 10,537 |

Copies are made by [`bench/run_benchmark.py`](../bench/run_benchmark.py): copy 0 is the original, and every
other copy differs only in its H1 title, which gets " (copy k)". So the 1k and 10k corpora are 100% repeated
real documentation (two thirds and 23/24 of their chunks repeat another chunk's body), not 1,000 or 10,000
independent documents. That is fine for costs that grow with the number of chunks (embedding, index size,
scanning vectors and keyword postings), and wrong for anything that depends on what the top results look
like; see the reranking caveat below. No quality metric is reported on these corpora.

## How it was measured

[`bench/run_benchmark.py`](../bench/run_benchmark.py) runs every step in a new process (spawned, so memory
from one step can't carry into the next), with the corpus and index in a temporary folder:

1. **Ingest process**, once per scale: load the models; ingest the corpus into a new index (full ingest, with
   real embeddings); ingest again with nothing changed; edit some files and ingest again. The edited files
   are ceil(files × 5%), spread evenly over the sorted file list: 2 of 23 (8.7%), 4 of 69 (5.8%) and 28 of 552
   (5.1%); each gets one paragraph appended at the end. Sourced re-chunks and re-embeds every chunk of a
   changed file, not only the edited part. Each timing covers the whole `Pipeline.ingest` call: scanning and
   hashing every file, then chunking, embedding and writing the changed ones; no file was deleted, so nothing
   was removed. The counts confirm what happened, e.g. at 10k: `added 552`, then `unchanged 552`, then
   `updated 28, unchanged 524`.
2. **Three query processes** per scale, all opening the index built in step 1 (the index is reused, not
   rebuilt per round). Each loads the models, then for each retriever: one first query (timed and reported
   apart), 3 untimed warm-up queries, then the 31 questions of `eval/prometheus/questions.jsonl`, each timed
   once: 31 samples per retriever per round. The order of the four retrievers rotates from round to round.
   Every call is a complete `Pipeline.search` with `top_k` 5 and 20 candidates per retriever: query
   embedding, SQLite access, fusion, optional reranking and fetching the chunks.
3. **Memory:** current RSS read from `ps` (macOS; `/proc` on Linux) at fixed points, and the process's peak
   RSS from `getrusage` (bytes on macOS, KiB on Linux, both converted to MiB). These show when memory moved,
   not which object or library holds it.
4. **Index size:** the index file and every file next to it. Only `index.db` and the empty `index.db.lock`
   existed after each ingest; SQLite's rollback journal is removed when a transaction commits.

Percentiles use NumPy's default (linear interpolation) over one round's 31 samples; each round is reported
on its own, never averaged into one figure. The OS file cache was not cleared, so nothing here is a cold-disk
measurement.

## Results

### Query latency (ms, p50 / p95, one value per round, 31 samples each)

| Scale | BM25 | Dense | Hybrid | Hybrid + rerank |
|---|---|---|---|---|
| base (417 chunks) | 0.2 / 0.3, 0.3 / 0.4, 0.3 / 0.4 | 3.0 / 4.6, 2.7 / 3.9, 3.0 / 3.9 | 3.1 / 4.2, 3.2 / 4.2, 3.1 / 4.7 | 296 / 376, 313 / 429, 332 / 403 |
| 1k (1,297) | 0.4 / 0.6, 0.6 / 0.7, 0.5 / 0.7 | 2.8 / 3.8, 3.0 / 3.7, 3.0 / 4.2 | 4.4 / 17.1, 4.0 / 4.7, 3.7 / 6.2 | 274 / 429, 285 / 378, 281 / 407 † |
| 10k (10,537) | 2.0 / 2.9, 2.6 / 3.5, 2.3 / 3.3 | 3.4 / 4.6, 3.4 / 5.2, 3.3 / 5.0 | 6.5 / 8.8, 6.2 / 8.9, 6.5 / 9.3 | 169 / 305, 168 / 306, 170 / 290 † |

**† Reranked latency at 1k and 10k should not be read as a size trend.** What was checked: BM25's own 20
candidates (its ranking doesn't depend on the embedding model) are a median of 20 distinct passages at the
base scale, 7 at 1k and 1 at 10k (20 copies of one passage), and the longest of them shrinks from a median of
about 1,230 to 710 characters at 10k. What was not checked: the reranker reads the top 20 of the *fused*
BM25 + dense list, and those actual inputs were not inspected. Shorter, more repetitive input to the reranker
is a plausible explanation for the drop from ~300 to ~170 ms, not a verified one. Either way, the repeated
corpus is not evidence of how reranking behaves on a larger natural corpus; the base-scale figure is the
one to use.

**First query in a new process** (after the models are loaded): BM25 0.5–1.9 ms; the first dense or hybrid
query of the process is 4.1–4.5 ms at base, 5.5–6.2 ms at 1k and 22.4–24.0 ms at 10k, because it loads every
vector from SQLite into memory; the first reranked query is 341–357 ms at base (69–72 ms at 10k, with the
same † caveat).

**Model loading** (constructing both models from the local cache): 0.20–0.27 s per process.

### Ingest

| Scale | Full ingest (with embeddings) | Unchanged corpus | ceil(5%) of files edited | Index file |
|---|---|---|---|---|
| base | 16.7 s (417 chunks) | 0.002 s | 0.56 s (2 of 23 files) | 1.7 MiB |
| 1k | 52.9 s (1,297) | 0.005 s | 1.07 s (4 of 69) | 5.1 MiB |
| 10k | 421.3 s (10,537) | 0.037 s | 12.8 s (28 of 552) | 40.7 MiB |

Each ingest step ran once per scale (not repeated across rounds).

### Memory (RSS, MiB)

| Scale | After loading models | Query process, after the vector cache loaded | Query process, after reranked queries | Ingest process, after full ingest |
|---|---|---|---|---|
| base | 351–352 | 395–396 | 1,761–1,763 | 1,691 |
| 1k | 351–375 | 397–422 | 1,690–1,712 | 2,519 |
| 10k | 351–352 | 372 | 1,269–1,271 | 2,522 |

Current RSS (from `ps`, in KiB) and peak RSS (from `getrusage`, in bytes) are separate measurements of the
same process. At the end of every process the current sample was within 0.13 MiB of the process peak, so the
table shows the current readings only; both are in `raw.json`. Two readings don't show whether memory fell
and rose again in between. RSS rises by over
1 GiB during the first reranked queries and during ingest's embedding runs; this sampling shows when that
happens, not how much of it belongs to the models, onnxruntime's buffers or anything else. Why the 10k query
processes end lower after reranking was not investigated.

## What changes with size

- **Full ingest grows linearly and is dominated by computing embeddings:** about 25 chunks per second at
  every scale (24.5–25.0). A 10,000-chunk corpus took 7 minutes on this laptop.
- **Re-ingesting unchanged files is cheap** (37 ms for 552 files: hashing only), and an incremental ingest
  costs about the full-ingest time of the files that changed: changed files are re-chunked and every chunk of
  them re-embedded.
- **Keyword and vector search stay in single-digit milliseconds up to 10k chunks.** BM25 grew about tenfold
  (0.2–0.3 to 2.0–2.6 ms p50), partly because every term now matches 24 copies; dense search barely moved
  (2.7–3.0 to 3.3–3.4 ms), since embedding the query costs more than scanning 10,537 vectors.
- **Reranking dominates `/ask`-style retrieval:** ~300 ms at base against ~3 ms for hybrid without it.
- **The index is about 4 KiB per chunk** (40.7 MiB for 10,537 chunks), vectors, text and the keyword index
  together.

## What this doesn't show

- Throughput, concurrency, or the latency of the HTTP server under load.
- Anything about corpora larger than 10,537 chunks, or about corpora of 10k *different* passages; when an
  approximate vector index would be needed was not measured.
- Cold starts with an empty OS file cache, other CPUs, Linux, or other thread settings.
- Answer generation: no Claude call was made, and answer latency is not included anywhere.
- The variation between rounds comes from three processes on one machine, not from repeated full experiments.
  Round-to-round p50 spread was small for the cheap retrievers and up to 12% for reranking at base
  (296–332 ms); single slow samples (for example a 44 ms hybrid query at 1k, behind that round's 17 ms p95)
  are kept.

## Reproduce

```bash
export FASTEMBED_CACHE_PATH=~/.cache/fastembed     # where the models are cached; downloads are disabled
python bench/run_benchmark.py                       # all scales, 3 rounds, ~10 minutes on the machine above
python bench/run_benchmark.py --scales base --rounds 1 --out /tmp/bench-trial   # quick trial
```

Results go to a new `bench/results/<UTC time>` folder; an existing folder is only overwritten with `--force`.
The temporary corpora and indexes are deleted when the run ends.

## Earlier figures

The README previously gave retrieval latency on the 87-chunk sample handbook (BM25 0.1 / 0.2 ms, dense 2.5 /
3.3 ms, hybrid 2.4 / 3.2 ms, hybrid + rerank 113 / 128 ms, p50 / p95 over 48 questions). That was a different
corpus with much shorter chunks (median 200 characters, against 433 for the Prometheus pages), measured with an
earlier, unpublished method. It is kept as a historical figure and is not directly comparable with the
tables above. The longer chunks fit the higher reranking time here, but that cause was not tested.
