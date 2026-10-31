"""Public evidence of the live Claude checks: build it from private run records, or check it offline.

    python eval/live-claude/extract_public.py check eval/live-claude/2026-10-29/calls.jsonl
    python eval/live-claude/extract_public.py build RECORD.jsonl [RECORD.jsonl ...] --out DIR/calls.jsonl

`build` reads the private files written by `python -m sourced.live_check --out ...` (in run order), keeps only
the fields listed in KEEP, USAGE_KEEP and ITERATION_KEEP plus the text of every cited passage, recomputes each
call's cost from its usage, and reruns today's automatic checks next to the ones recorded at run time. It
refuses to write over an existing file.

`check` only reads: it recounts calls and distinct questions, recomputes the cost, and confirms that every
quote is inside its cited passage and every cited passage is word for word in its public source (the sample
handbook, or the live check's injection documents). It reports how many quotes are the whole passage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

PRICES = {"claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_write": 5.0, "cache_read": 0.20}}  # $ per MTok
KEEP = ["id", "kind", "question", "model", "stop_reason", "refused", "fallback_ran", "answer", "seconds"]
USAGE_KEEP = ["input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]
ITERATION_KEEP = ["type", "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]
RECORD_KEYS = {"id", "question", "model", "stop_reason", "answer", "citations", "passages", "passage_texts", "usage"}


def cost(model: str, usage: dict) -> float:
    """USD from token counts at PRICES (the published per-token rates; not a bill)."""
    if model not in PRICES:
        raise ValueError(f"no price for {model}")
    p = PRICES[model]
    return (
        usage["input_tokens"] * p["input"]
        + usage["output_tokens"] * p["output"]
        + (usage.get("cache_creation_input_tokens") or 0) * p["cache_write"]
        + (usage.get("cache_read_input_tokens") or 0) * p["cache_read"]
    ) / 1e6


def squash(text: str) -> str:
    return " ".join(text.split())


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"{path}: no such file")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{number}: not valid JSON ({exc.msg})") from None
    return rows


def public_row(record: dict, call: int, cases: dict) -> dict:
    from sourced import live_check
    from sourced.chunking import Chunk
    from sourced.generation import Answer, Citation
    from sourced.retrieval import SearchResult

    missing = RECORD_KEYS - set(record)
    if missing:
        raise SystemExit(f"call {call}: record lacks {sorted(missing)}; not a live_check result")
    texts = {(p["source"], p["heading"]): t for p, t in zip(record["passages"], record["passage_texts"], strict=True)}
    results = [
        SearchResult(Chunk("x", source, *heading.partition(" > ")[::2], text, 0), 0.0)
        for (source, heading), text in texts.items()
    ]
    answer = Answer(record["answer"], [Citation(**c) for c in record["citations"]], "claude")
    usage = {k: record["usage"].get(k) or 0 for k in USAGE_KEEP}
    usage["thinking_tokens"] = (record["usage"].get("output_tokens_details") or {}).get("thinking_tokens")
    return (
        {"call": call}
        | {k: record.get(k) for k in KEEP}
        | {
            "citations": [c | {"cited_passage": texts[c["source"], c["heading"]]} for c in record["citations"]],
            "retrieved": [f"{p['source']} > {p['heading']}" for p in record["passages"]],
            "usage": usage,
            "usage_iterations": [
                {k: i.get(k) for k in ITERATION_KEEP} for i in record["usage"].get("iterations") or []
            ],
            "cost_usd": round(cost(record["model"], usage), 6),
            "cost_usd_as_recorded": record.get("cost_usd"),
            "checks_recorded_at_run_time": record.get("checks"),
            "checks_rerun_offline": live_check.checks(cases[record["id"]], answer, results),
        }
    )


def build(records: list[Path], out: Path) -> None:
    from sourced import live_check

    if out.exists():
        raise SystemExit(f"{out} already exists; published evidence is never overwritten, choose a new --out")
    cases = {c.id: c for c in live_check.handbook_cases(ROOT / "eval" / "questions.jsonl") + live_check.INJECTIONS}
    rows = []
    for path in records:
        for record in read_jsonl(path):
            rows.append(public_row(record, len(rows) + 1, cases))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {out}: {len(rows)} calls")


def source_text(source: str) -> str:
    from sourced.live_check import INJECTION_DOCS

    path = ROOT / "data" / "handbook" / source
    return path.read_text(encoding="utf-8") if path.is_file() else INJECTION_DOCS.get(source, "")


def check(evidence: Path) -> dict:
    """Recounts one evidence file; returns the summary (problems listed under "problems")."""
    rows = read_jsonl(evidence)
    problems, quotes, whole = [], 0, 0
    for r in rows:
        if abs(cost(r["model"], r["usage"]) - r["cost_usd"]) > 1e-6:
            problems.append(f"call {r['call']}: cost_usd doesn't match its usage")
        if (
            r["usage_iterations"]
            and sum(i["output_tokens"] for i in r["usage_iterations"]) != r["usage"]["output_tokens"]
        ):
            problems.append(f"call {r['call']}: iterations don't add up to the usage")
        for c in r["citations"]:
            passage = squash(c["cited_passage"])
            if passage not in squash(source_text(c["source"])):
                problems.append(f"call {r['call']}: cited passage not found in {c['source']}")
            for q in c["quotes"]:
                quotes += 1
                whole += squash(q) == passage
                if squash(q) not in passage:
                    problems.append(f"call {r['call']}: quote not inside its cited passage")
    ids = [r["id"] for r in rows]
    return {
        "calls": len(rows),
        "distinct_questions": len(set(ids)),
        "repeated": sorted({i for i in ids if ids.count(i) > 1}),
        "models": sorted({r["model"] for r in rows}),
        "stop_reasons": sorted({r["stop_reason"] for r in rows}),
        "fallback_attempts": sum(i["type"] == "fallback_message" for r in rows for i in r["usage_iterations"]),
        "total_cost_usd": round(sum(cost(r["model"], r["usage"]) for r in rows), 5),
        "total_of_rounded_costs_usd": round(sum(r["cost_usd_as_recorded"] or 0 for r in rows), 5),
        "quotes": quotes,
        "quotes_equal_to_whole_passage": whole,
        "problems": problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build", help="build public evidence from private run records")
    b.add_argument("records", nargs="+", type=Path, help="live_check output files, in run order")
    b.add_argument("--out", type=Path, required=True, help="new evidence file (must not exist)")
    c = commands.add_parser("check", help="recount public evidence files (read-only)")
    c.add_argument("evidence", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.records, args.out)
        return 0
    failed = False
    for path in args.evidence:
        summary = check(path)
        print(f"{path}: {json.dumps(summary)}")
        failed |= bool(summary["problems"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
