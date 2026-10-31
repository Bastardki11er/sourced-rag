# Evaluation

What has been measured, how, and what each number does and doesn't show. Two corpora are used, and their
results are reported separately, never averaged together:

- the bundled sample handbook ([`data/handbook`](../data/handbook)): 14 short documents about a fictional
  company, 87 chunks; everything below except the external-corpus section uses it;
- 23 pages of the real Prometheus documentation (417 chunks), in
  [External corpus](#external-corpus-prometheus-documentation).

These are results for two small corpora, not production performance.

## Retrieval quality

48 hand-labelled questions ([`eval/questions.jsonl`](../eval/questions.jsonl)): 23 use the handbook's own
wording (*lexical*), 20 deliberately avoid it (*paraphrase*, e.g. "My robot complains it's carrying too
much" → the `KR-4021 – Payload overweight` entry), and 5 have no answer in the handbook. Each answerable
question names the section(s) that hold the answer.

Reproduce with `sourced ingest data/handbook && sourced eval eval/questions.jsonl --rerank`.

| Retriever | hit@1 | hit@3 | hit@5 | MRR | hit@3 lexical | hit@3 paraphrase |
|---|---|---|---|---|---|---|
| BM25 | 0.70 | 0.84 | 0.91 | 0.78 | 0.96 | 0.70 |
| Dense (bge-small) | 0.81 | **0.98** | **0.98** | 0.89 | 1.00 | **0.95** |
| Hybrid (RRF) | 0.84 | 0.93 | 0.93 | 0.88 | 1.00 | 0.85 |
| **Hybrid + rerank** (default) | **0.88** | 0.95 | **0.98** | **0.92** | 1.00 | 0.90 |

- **hit@k**: the share of the 43 answerable questions whose answer section is among the top k retrieved
  chunks. **MRR**: mean of 1/rank of the first correct chunk (0 if not in the top 10).
- A local re-run on 2026-10-28 gave the same table. CI re-checks the Hybrid + rerank hit@5 on every push to
  `main` and every pull request (0.98 on `main` at `49b0871`, 2026-10-29) and fails the build below 0.95.

What the numbers say:

- **BM25 alone collapses on paraphrases** (0.70 hit@3), the classic failure that motivates semantic search.
- **Plain hybrid was not a free win here.** Fusion improved the top-1 result, but BM25's noisy candidates
  pushed some correct paraphrase matches out of the top 3. On this small corpus with distinctive
  vocabulary, dense retrieval alone is already strong.
- **Adding the reranker gives the best ranking overall** (hit@1 0.88, MRR 0.92) and ties dense on hit@5, the
  number that decides whether the right passage reaches the answerer. So hybrid + rerank is the default for
  `/ask`. BM25 stays in the mix because exact identifiers matter more as a corpus grows.

What they don't say: hit@5 measures whether the right passage is *retrieved*. It is not an answer accuracy:
a retrieved passage can still be misread, and in extractive mode nothing checks that the passage answers the
question.

### Latency (historical)

Measured earlier with an unpublished method; current, reproducible measurements on the Prometheus corpus are
in [performance.md](performance.md). Retrieval only, on an Apple M5 laptop (CPU only, 87 chunks, p50 / p95 over
the 48 questions):

| BM25 | Dense | Hybrid | Hybrid + rerank |
|---|---|---|---|
| 0.1 / 0.2 ms | 2.5 / 3.3 ms | 2.4 / 3.2 ms | 113 / 128 ms |

On the handbook, reranking cost ~110 ms, small next to LLM generation, so `/ask` reranks by default and `/search` doesn't (on the Prometheus pages it is ~300 ms; see [performance.md](performance.md)).
These are single-machine figures for a tiny index, not a throughput or production latency measurement, and
they have not been re-measured since.

## External corpus: Prometheus documentation

A second retrieval test on documentation Sourced was not written for: 23 pages of the Prometheus server docs
at tag `v2.53.0` (Apache-2.0), 417 chunks, with 31 hand-written questions. Source, license, the one change
made to the files, question kinds, labelling rules and label changes are in
[`eval/prometheus/README.md`](../eval/prometheus/README.md).

Run on 2026-10-29 against commit `49b0871` (no uncommitted changes under `src/`), with
`fastembed:BAAI/bge-small-en-v1.5` (384 dimensions), reranker `Xenova/ms-marco-MiniLM-L-6-v2`,
`max_chunk_chars` 1200, chunker version 3, on a new temporary index. Reproduce with
`python eval/prometheus/run_eval.py`. Machine-readable results:
[`summary.json`](../eval/prometheus/results/summary.json),
[`per_question.jsonl`](../eval/prometheus/results/per_question.jsonl).

| Retriever | hit@1 | hit@3 | hit@5 | MRR@10 | hit@5 exact | hit@5 paraphrase | hit@5 confusable |
|---|---|---|---|---|---|---|---|
| BM25 | 0.79 | 0.92 | 0.96 | 0.86 | 1.00 | 0.88 | 1.00 |
| Dense | 0.62 | 0.88 | 0.92 | 0.76 | 0.88 | 0.88 | 1.00 |
| Hybrid (RRF) | 0.75 | 0.92 | **1.00** | 0.84 | 1.00 | 1.00 | 1.00 |
| **Hybrid + rerank** | **0.92** | **1.00** | **1.00** | **0.95** | 1.00 | 1.00 | 1.00 |

- Only the 24 answerable questions are scored (8 per kind, so one question is 0.125 of a kind's score). A gold
  section not in the top 10 counts as a miss for every hit@k and 0 for MRR@10. Hits are by section, as for the
  handbook.
- The 7 unanswerable questions are not scored: no answerer ran, and extractive mode can't decline. Their
  retrieved passages are recorded in `per_question.jsonl` for inspection.
- **Stricter, evidence-level view** (the retrieved chunk must also contain the quoted evidence): hit@5 0.83 /
  0.79 / 0.88 / 0.96 for BM25 / Dense / Hybrid / Hybrid + rerank. It is a lower bound that depends on which
  sentence was picked as evidence: in the 5 questions where it differs, the other retrieved chunks of the gold
  section are on the same topic (other parts of the `histogram_quantile()` section, the federation
  configuration, unit-test file format), and some answer the question as well as the quoted sentence.
- **Run 1**, before three missing gold sections were added to x01 and x07 (see the label changes):
  hit@1 0.75 / 0.58 / 0.71 / 0.83 and hit@5 0.96 / 0.92 / 1.00 / 1.00. The correction raised hit@1; hit@5 did
  not change. Results in [`results/run1`](../eval/prometheus/results/run1).

What the per-question ranks show (observations unless marked as checked):

- **BM25 ranked the answer higher than dense retrieval in 8 questions, dense in 3.** This corpus is full of
  exact identifiers (flags, endpoints, function names), which suits BM25. Dense retrieval missed x05 ("default
  lookback period for instant vector selectors") entirely: it ranked the selector sections, whose headings
  match the question, first and had no "Staleness" chunk in its top 10, while the answer sits in that gotcha. Checked: the gold chunk contains the
  sentence ("The lookback period is 5 minutes by default"), so this is ranking, not chunking or conversion. For
  p04 ("average per-second growth of a counter") dense ranked `rate()` 9th, below `<aggregation>_over_time()`;
  the word "average" plausibly pulls it toward `avg_over_time`, not checked further.
- **Plain hybrid was worse than the better single retriever in 6 questions** (x05, p02, p03, p04, p07, p08),
  always within the top 5: its hit@1 (0.75) is below BM25's (0.79) while its hit@5 is the best. As on the
  handbook, fusion helps recall and can blur the first position.
- **Reranking moved the answer up in 5 questions and down in 1.** The regression is c05 ("only an upper
  limit"): it put `clamp()` (both limits) above `clamp_max()`, a confusable pair the labels deliberately keep
  apart. p08 (`@` modifier) stays at rank 3 behind the parent "Time series selectors" section and the HTTP
  API's instant-query section.
- **No miss remained at hit@5 for hybrid or reranked retrieval.** With 24 answerable questions over 417
  chunks, that says these questions were within reach, not that harder ones would be.
- **Label errors found:** 2 questions (3 sections), corrected as described above. No conversion problem was
  found: every evidence quote was found verbatim in the converted pages.

Limits: one corpus, 24 scored questions written and checked by the same assistant that built Sourced (not an
independent or blind review); English only; retrieval only, not answer quality; not a performance test. The
handbook's CI gate (hit@5 ≥ 0.95) does not apply to this corpus, and this baseline is not part of CI.

## Answer quality

- **Keyword checks over all 48 questions**: `sourced eval eval/questions.jsonl --answers` checks that answers
  contain the expected facts, cite the right section and decline the unanswerable questions. It calls the
  configured answerer; those numbers have not been published.
- **Live Claude checks**: two runs of 12 questions (2026-10-29 and 2026-10-31), below. They are the only
  evidence about Claude-written answers.

### Live check against the real API (opt-in, paid)

`python -m sourced.live_check` asks Claude 12 questions: 9 from `eval/questions.jsonl` (exact facts, an error
code, a command, paraphrases, a question with two acceptable sections, two with no answer), one that needs two
sections combined, and two prompt-injection cases on a temporary corpus whose documents tell the model to lie
or leak a (made-up) password. It writes each answer with its passages, citations, stop reason, usage and cost
to a JSON lines file, plus automatic checks (expected facts, citations point at passages that were sent and
quote text inside them, declines without citing, no injected text in the answer). Those checks are a first
pass; the answers still need reading.

Spending guards: it refuses to run without `--confirm-spend`, credentials and the official API host, and
makes no retries. `--budget-usd` (default 2, a finite amount above 0) covers everything recorded in the
`--out` file, earlier runs included; a call is skipped when its estimated worst case (full `max_tokens` plus
one fallback attempt, at the highest price the tool knows) would pass it. The server picks the fallback model,
so that is an estimate, not a guarantee. An incomplete or empty reply records its cost and stops the run; any
API error, or a reply whose cost can't be counted, stops the run and marks the file so it can't be resumed as
if that call were free. It never runs in the test suite or CI. Start with one question:

```bash
export FASTEMBED_CACHE_PATH=~/.cache/fastembed
python -m sourced.live_check --index .sourced/index.db --out live-l02.jsonl --confirm-spend --only l02
```

Two runs so far, reported separately. Every call's question, answer, citations (with the full cited
passage), usage and cost, the automatic checks and the human review are published in
[`eval/live-claude`](../eval/live-claude/README.md), one folder per run; one command recounts calls, cost and
quotes. Costs are computed from the token counts the API reported at $4 / $20 per million input / output
tokens, not read from a bill.

| | 2026-10-29 | 2026-10-31 |
|---|---|---|
| System prompt | earlier ("company handbook", "employees") | current ("the provided documents"), same rules |
| Calls | 13 (12 questions; `l02` twice) | 12 (12 questions) |
| Ended normally | all (`end_turn`) | all (`end_turn`) |
| Truncated, safety refusal, fallback | none | none |
| Cost from usage | $0.14146 | $0.13156 |
| Automatic checks | pass; but the injection check was empty at run time (rerun offline with the fix: pass) | pass, with the fixed injection check |
| Quotes found in their source / whole cited passage | 15 / 15 | 14 / 14 |

What the human review found (details per call in each run's README):

- **API compatibility.** The live API accepted the request as sent (the fallback beta and `fallbacks`,
  `output_config`, citable `search_result` blocks) and every reply parsed into an answer with numbered
  citations. The fallback switch itself was never exercised.
- **Answerable questions.** In both runs the main facts of all 10 answerable questions were right and
  supported by the cited text. On 2026-10-29, p09 added an unsupported condition ("if the change is behind a
  flag"); on 2026-10-31 it didn't. One run can't show that the new prompt caused this.
- **Unanswerable questions.** u01 and u05 were declined for lack of information in both runs, with no
  citation; that is the model saying the documents don't contain the answer, not an API safety refusal. Both
  declines end with general advice that has no source (ask HR, the onboarding buddy or a manager), so not every
  sentence of every answer is backed by a citation.
- **Citations.** Every quote appears word for word in its source and is the whole cited passage, never a
  sentence; why is not established. A quote found word for word shows where text came from, not that it
  supports the claim beside it; that part rests on the human review.
- **Prompt injection.** In both runs the answers used the legitimate text and pointed out the planted
  instructions instead of following them; the answer text contained neither the made-up password nor the
  attacker address. But the i02 citation quotes the whole passage, planted password and address included,
  and citations are returned to the user too, so this is not "no exposure". At the 2026-10-29 run the
  automatic check didn't look for the planted text (a bug, kept in its record); the 2026-10-31 run used the
  fixed check.

That is 12 questions on a small fictional handbook, at most two calls each: it shows how these answers looked,
not an accuracy rate or how stable answers are. Successful calls don't exercise rate limits, overload,
timeouts or truncation; those are covered by offline tests with mocked replies (below).

## Offline tests

`pytest -q` runs 236 tests in about 3 s with no network: a hashing embedder stands in for the model, and Claude
replies are mocked. They cover heading-aware chunking (code fences, overlap, long paragraphs, headings like
`C#`), FTS5 query sanitizing, RRF math, reranking and its availability, idempotent re-ingestion, keeping the
index in step with the folder (another process re-indexing, deleted files, one folder per index, symbolic
links, old indexes), one ingest at a time across processes, refusing an index built with another model,
vector size or chunking, which files count as documents (suffix case, hidden paths, single files), the exact
request and system prompt sent to Claude, citations, refusals, unfinished or empty replies (502), the
production client built by `build_pipeline` (no retries, 60 s timeout, 429/529/timeouts as 503, one request per
failed call), CLI/Python/API input validation and error messages, the eval command, the live check's spending
guards and its published record, the external corpus's hashes and labels, the benchmark's arguments, units and
summaries, and a BM25 regression floor on the real handbook.

Several tests use the real `anthropic` SDK over a mocked HTTP transport, so they check the request the SDK
actually sends and how real reply shapes (normal, refusal, `max_tokens`, empty, thinking-only) are handled.
They don't show how the live API behaves; only the live check does that.

## CI

Every push to `main` and every pull request runs three jobs: lint and the offline tests; the retrieval quality gate above,
with the real models; and a Docker build plus a smoke test that starts the image and asks it one question.
The smoke test uses the extractive answerer, so it checks the service, not Claude.

## Not measured

- Answer quality beyond the 12 live questions; repeated runs; an LLM-as-judge for faithfulness.
- How the live API behaves when it truncates a reply or falls back to another model.
- Large or messy corpora, near-duplicate passages, throughput under load.
- Questions or documents in languages other than English.
