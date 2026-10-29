# External corpus evaluation: Prometheus documentation

A retrieval test on real, public technical documentation that Sourced was not written for, alongside the
fictional handbook in [`data/handbook`](../../data/handbook). Results: [`results/summary.json`](results/summary.json)
and [`results/per_question.jsonl`](results/per_question.jsonl); the analysis is in the repository's
`docs/evaluation.md`, section "External corpus: Prometheus documentation".

## Source and license

- **Documents:** 23 pages of the Prometheus server documentation (`docs/` in
  [prometheus/prometheus](https://github.com/prometheus/prometheus)), tag `v2.53.0`, commit
  `4c35b9250afefede41c5f5acd76191f90f625898`. Source URLs and SHA-256 of every source and converted file:
  [`manifest.json`](manifest.json).
- **Selection** (made before any question was written): every `docs/*.md` page at that commit except the four
  `index.md` navigation stubs and `configuration/configuration.md`, a single 160 KB reference page that would be
  most of the corpus on its own.
- **License:** Apache License 2.0, copyright The Prometheus Authors. [`LICENSE`](LICENSE) and
  [`NOTICE`](NOTICE) are copied unchanged from the same commit.
- **Changes made to the files in [`corpus/`](corpus):** the YAML front matter at the top of each page (`title`,
  `sort_rank`, used by the website) was removed. Nothing else was changed; `prepare_corpus.py` checks that the
  number of headings, code fence lines and table rows is the same before and after.

## Questions and labels

[`questions.jsonl`](questions.jsonl) uses the same format as the handbook's `eval/questions.jsonl`: 31
questions written by the same assistant that develops Sourced (not an independent reviewer):

| Kind | Count | What it tests |
|---|---|---|
| `exact` | 8 | exact names: flags, endpoints, functions, commands |
| `paraphrase` | 8 | the same needs in words the docs don't use |
| `confusable` | 8 | neighbouring concepts: `rate`/`irate`, `increase`/`delta`, `clamp`/`clamp_max`, `offset`/`@`, two metadata endpoints |
| `unanswerable` | 7 | in-domain questions whose answer is not in these 23 pages (it may exist elsewhere) |

- A `gold` entry is a source file plus a section; a retrieved chunk counts when its source matches and its
  section path contains that section name. A section is gold when it states the asked fact about the asked
  thing; when several sections do, all are listed. Two questions accept any chunk of a one-topic page
  (`"section": ""`): p05 (`federation.md`) and p07 (`configuration/unit_testing_rules.md`).
- [`evidence.jsonl`](evidence.jsonl) gives, for every gold section, a quote from the page that supports the
  answer, plus notes on deliberate exclusions (for example why `rate()` is not gold for c01).
- `run_eval.py --check` verifies that IDs are unique, every gold source and section exists after chunking, and
  every quote is inside a chunk of its gold section. Unanswerable questions were checked by searching the
  corpus for their key terms (Slack, Grafana, downsampling, high availability, Windows, LDAP, Consul): none
  appear.

### Label changes

The labels were frozen in [`baseline.json`](baseline.json) before the first run. After run 1, three gold
sections were found missing and added; `baseline.json` keeps the old hashes and the reason in its history,
and run 1 is kept in [`results/run1`](results/run1).

| Question | Added gold section | Why |
|---|---|---|
| x01 (`--web.enable-lifecycle`) | `management_api.md` › Quit; `migration.md` › Prometheus lifecycle | Both say the flag enables these endpoints, exactly like the Reload section already labelled gold |
| x07 (`clean_tombstones`) | `querying/api.md` › Delete Series | It states that the Clean Tombstones endpoint cleans up deleted data still on disk |

These changes raised scores: x01 moved from rank 2–3 to rank 1 in all four retrievers, and x07 from 2 to 1
with reranking. No other label was changed; the failures listed in the analysis were left as failures.

- run 1 labels: `questions.jsonl` `96cdaf1f…`, `evidence.jsonl` `180ce936…`
- current labels: `questions.jsonl` `40572cad…`, `evidence.jsonl` `0eef8d45…` (full hashes in `baseline.json`)

## Reproduce

From the repository root, after the normal install:

```bash
python eval/prometheus/prepare_corpus.py          # check corpus/ against manifest.json (offline)
python eval/prometheus/run_eval.py --check        # check the labels (offline)
python eval/prometheus/run_eval.py                # evaluate with the real models (downloads them once)
```

`run_eval.py` builds a new index in a temporary folder, removed afterwards, and writes
[`results/summary.json`](results/summary.json) (corpus size, index settings, code commit, file hashes,
metrics) and [`results/per_question.jsonl`](results/per_question.jsonl) (for every question and retriever:
the rank of the first gold chunk, the rank of the first gold chunk that contains an evidence quote, and the
top 5 sources and sections). It refuses to run if the label files differ from `baseline.json`.
`prepare_corpus.py --fetch` downloads the 23 pages and the license files again from the pinned commit and
rebuilds `corpus/`.
