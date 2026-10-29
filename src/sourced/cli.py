"""Command line interface.

sourced ingest data/handbook
sourced search "how do I roll back a release?"
sourced ask "who do I tell if I lose my laptop?"
sourced eval eval/questions.jsonl --rerank
sourced serve
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from .config import ConfigError, Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sourced", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="index a folder of Markdown/text files")
    ingest.add_argument("path", type=Path)

    search = commands.add_parser("search", help="show the passages retrieval finds, without an LLM")
    search.add_argument("query")
    search.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--rerank", action="store_true")

    ask = commands.add_parser("ask", help="answer a question with citations")
    ask.add_argument("question")
    ask.add_argument(
        "--no-rerank", dest="rerank", action="store_false", default=None, help="skip cross-encoder reranking"
    )

    evaluate = commands.add_parser("eval", help="score retrieval (and optionally answers) on a labelled set")
    evaluate.add_argument("questions", type=Path)
    evaluate.add_argument("--rerank", action="store_true", help="also score hybrid retrieval + cross-encoder")
    evaluate.add_argument("--answers", action="store_true", help="also score generated answers (calls the LLM)")
    evaluate.add_argument(
        "--fail-under",
        type=float,
        metavar="HIT5",
        help="exit 1 if hit@5 of the default retriever (hybrid, reranked with --rerank) is below this",
    )

    serve = commands.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env()
    except ConfigError as error:  # before any model is loaded or file opened
        print(f"sourced: error: {error}", file=sys.stderr)
        return 2

    if args.command == "serve":
        import uvicorn

        uvicorn.run("sourced.api:app", host=args.host, port=args.port)
        return 0

    from .pipeline import build_pipeline

    # Only load the reranker model for commands that will use it.
    wants_rerank = getattr(args, "rerank", None)
    if args.command == "ask":
        wants_rerank = settings.rerank if wants_rerank is None else wants_rerank
    pipeline = build_pipeline(settings, with_reranker=bool(wants_rerank))

    from .pipeline import DocumentLinkError, IndexBindingError, IndexBusyError, IndexConfigError
    from .retrieval import EmptyQueryError

    try:
        return _run(args, settings, pipeline)
    except (
        IndexBindingError,
        IndexBusyError,
        IndexConfigError,
        DocumentLinkError,
        EmptyQueryError,
        FileNotFoundError,
    ) as error:
        print(f"sourced: error: {error}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace, settings: Settings, pipeline) -> int:
    if args.command == "ingest":
        report = pipeline.ingest(args.path)
        print(
            f"added {len(report.added)}, updated {len(report.updated)}, unchanged {len(report.unchanged)}, "
            f"removed {len(report.removed)} -> {report.chunks} chunks in {settings.index_path}"
        )

    elif args.command == "search":
        for i, r in enumerate(pipeline.search(args.query, args.top_k, args.mode, args.rerank), start=1):
            ranks = ", ".join(f"{k} #{v}" for k, v in r.ranks.items())
            print(f"[{i}] {r.chunk.source} - {r.chunk.heading}  ({ranks})")
            print(textwrap.indent(textwrap.shorten(r.chunk.text, 300), "    "))

    elif args.command == "ask":
        result = pipeline.ask(args.question, rerank=args.rerank)
        print(result.answer.text, end="\n\n")
        for c in result.answer.citations:
            print(f"[{c.number}] {c.source} - {c.heading}")
            for quote in c.quotes:
                print(textwrap.indent(textwrap.shorten(quote, 200), "    > "))
        print(
            f"\n({result.answer.answerer}; retrieval {result.timings_ms['retrieval']} ms, "
            f"generation {result.timings_ms['generation']} ms)"
        )

    elif args.command == "eval":
        from .evaluation import answer_report, load_questions, retrieval_report

        questions = load_questions(args.questions)
        report, scores = retrieval_report(pipeline, questions, with_rerank=args.rerank)
        print(report)
        if args.answers:
            print()
            print(answer_report(pipeline, questions))
        if args.fail_under is not None:
            hit5 = scores["hybrid", args.rerank].hits[5]  # the row the table shows, not a second run
            if hit5 < args.fail_under:
                print(f"\nFAIL: hit@5 {hit5:.2f} is below the required {args.fail_under:.2f}")
                return 1
            print(f"\nOK: hit@5 {hit5:.2f} >= {args.fail_under:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
