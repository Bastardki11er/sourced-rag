# Live Claude checks: public evidence

The real Claude API calls behind Sourced's claims about Claude-written answers, one folder per run. Each run
was made with `python -m sourced.live_check` ([docs/evaluation.md](../../docs/evaluation.md#live-check-against-the-real-api-opt-in-paid))
on the fictional sample handbook and two injection documents. The runs are reported separately; they are not
combined into one accuracy figure.

| Run | Prompt | Calls | Cost from usage | Details |
|---|---|---|---|---|
| 2026-10-29 | earlier prompt ("company handbook", "employees") | 13 (12 questions, `l02` twice) | $0.14146 | [2026-10-29/README.md](2026-10-29/README.md) |
| 2026-10-31 | current prompt ("the provided documents") | 12 (12 questions) | $0.13156 | [2026-10-31/README.md](2026-10-31/README.md) |

Each folder has `calls.jsonl` (one line per call, in run order), `run.json` (model, settings, prompt hash, what
is known about the code that ran) and a README with the human review. The calls were not repeated to build
these files, and nothing was re-run when they were published.

## What is published

[`extract_public.py`](extract_public.py) builds `calls.jsonl` from the private run records and never writes
over an existing file:

- **Kept:** question id, kind and text; model; `stop_reason`; `refused`; whether a fallback model ran; the
  answer exactly as returned; each citation (source, heading, quoted text) with the full text of the passage it
  cites; the source and heading of the 5 passages sent to Claude; token counts (input, output, thinking,
  cache) and the per-attempt counts from `usage.iterations`; the cost computed from them and the cost the run
  printed; seconds; the automatic checks recorded at run time, and the same checks rerun offline today.
- **Removed:** the text of retrieved passages that weren't cited, and usage fields that don't bear on the
  result (`inference_geo`, `service_tier`, `speed`). The records contain no request headers, API keys, account
  or request identifiers, or local paths.

## Recount it

```bash
python eval/live-claude/extract_public.py check eval/live-claude/2026-10-29/calls.jsonl eval/live-claude/2026-10-31/calls.jsonl
```

Read-only: it recounts calls and distinct questions, the number of fallback attempts, the cost from each
call's usage (and the sum of the per-call costs the run printed, rounded to 5 decimals each), and checks that
every quote is inside its cited passage and every cited passage is word for word in its public source.

What that proves and what it doesn't:

- A quote found word for word shows where the text came from, **not** that it supports the sentence next to
  it; that judgement is the human review in each run's README.
- Costs are **computed from the token counts the API reported**, at $4 / $20 per million input / output tokens
  for `claude-opus-5-5` (no cache tokens were used). They were not checked against a bill.
- Without the uncited passages you can't re-read everything Claude was shown, only what it cited.
- Successful calls say nothing about rate limits, overload, timeouts or truncation; those paths are covered
  by offline tests with mocked replies.
