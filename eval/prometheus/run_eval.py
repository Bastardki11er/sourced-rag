"""Retrieval evaluation on the Prometheus docs corpus (eval/prometheus/corpus).

    python eval/prometheus/run_eval.py --check    # offline: corpus hashes, question and evidence labels
    python eval/prometheus/run_eval.py --freeze   # record the question/evidence hashes in baseline.json (once)
    python eval/prometheus/run_eval.py            # build a temporary index with the real models and evaluate

The evaluation refuses to run if questions.jsonl or evidence.jsonl differ from baseline.json, so labels
can't change silently; a deliberate change means a new baseline, recorded in README.md. Scoring reuses
sourced.evaluation (first_gold_rank, summarize): only answerable questions count, a gold section that is
not in the top 10 has no rank (0 for MRR, a miss for every hit@k).
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from sourced.chunking import CHUNKER_VERSION, chunk_markdown
from sourced.config import Settings
from sourced.evaluation import MRR_DEPTH, first_gold_rank, load_questions, summarize

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
QUESTIONS, EVIDENCE, BASELINE = HERE / "questions.jsonl", HERE / "evidence.jsonl", HERE / "baseline.json"
RESULTS = HERE / "results"
KINDS = ("exact", "paraphrase", "confusable", "unanswerable")
RETRIEVERS = [
    ("BM25", "bm25", False),
    ("Dense", "dense", False),
    ("Hybrid", "hybrid", False),
    ("Hybrid + rerank", "hybrid", True),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def squash(text: str) -> str:
    return " ".join(text.split())


def corpus_chunks(max_chars: int = Settings.max_chunk_chars) -> list:
    return [
        chunk
        for path in sorted(CORPUS.rglob("*.md"))
        for chunk in chunk_markdown(path.relative_to(CORPUS).as_posix(), path.read_text(), max_chars)
    ]


def check_labels() -> list[str]:
    """Problems with the questions and their evidence; empty when every label holds."""
    questions = load_questions(QUESTIONS)
    evidence = {e["id"]: e for e in map(json.loads, EVIDENCE.read_text().splitlines())}
    chunks = corpus_chunks()
    sources = {c.source for c in chunks}
    problems = []
    ids = [q.id for q in questions]
    problems += [f"duplicate id {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    problems += [f"evidence for unknown id {i}" for i in sorted(set(evidence) - set(ids))]
    for q in questions:
        if q.kind not in KINDS:
            problems.append(f"{q.id}: unknown kind {q.kind}")
        if (q.kind == "unanswerable") == bool(q.gold):
            problems.append(f"{q.id}: kind {q.kind} doesn't match its gold list")
        support = evidence.get(q.id, {}).get("support", [])
        for g in q.gold:
            if g["source"] not in sources:
                problems.append(f"{q.id}: gold source {g['source']} is not in the corpus")
            elif not any(c.source == g["source"] and g["section"] in c.section for c in chunks):
                problems.append(f"{q.id}: no chunk of {g['source']} has a section matching {g['section']!r}")
            quotes = [s["quote"] for s in support if (s["source"], s["section"]) == (g["source"], g["section"])]
            if not quotes:
                problems.append(f"{q.id}: no evidence quote for gold {g['source']} / {g['section']!r}")
            for quote in quotes:
                if not any(
                    c.source == g["source"] and g["section"] in c.section and squash(quote) in squash(c.text)
                    for c in chunks
                ):
                    problems.append(f"{q.id}: quote not found in {g['source']} / {g['section']!r}: {quote[:60]!r}")
        golds = {(g["source"], g["section"]) for g in q.gold}
        problems += [
            f"{q.id}: evidence for a non-gold section {s['section']!r}"
            for s in support
            if (s["source"], s["section"]) not in golds
        ]
    return problems


def file_hashes() -> dict[str, str]:
    return {p.name: sha256(p) for p in (QUESTIONS, EVIDENCE, HERE / "manifest.json")}


def evaluate() -> None:
    from sourced.pipeline import build_pipeline

    baseline = json.loads(BASELINE.read_text())
    hashes = file_hashes()
    for name in ("questions.jsonl", "evidence.jsonl"):
        if hashes[name] != baseline[name]:
            raise SystemExit(f"{name} differs from baseline.json; record the change and refreeze before running")
    questions = load_questions(QUESTIONS)
    answerable = [q for q in questions if q.answerable]
    quotes = {
        e["id"]: [squash(s["quote"]) for s in e["support"]] for e in map(json.loads, EVIDENCE.read_text().splitlines())
    }

    def evidence_rank(q, chunks) -> int | None:
        """Rank of the first retrieved chunk that is gold and contains one of the question's evidence quotes."""
        for rank, chunk in enumerate(chunks, start=1):
            if q.is_gold(chunk) and any(quote in squash(chunk.text) for quote in quotes[q.id]):
                return rank
        return None

    with tempfile.TemporaryDirectory() as tmp:
        settings = replace(Settings.from_env(), index_path=Path(tmp) / "index.db", answerer="extractive")
        pipeline = build_pipeline(settings, with_reranker=True)
        report = pipeline.ingest(CORPUS)
        meta = pipeline.index.meta()
        rows, strict_rows, per_question = (
            {},
            {},
            {q.id: {"id": q.id, "kind": q.kind, "question": q.question, "gold": q.gold} for q in questions},
        )
        for label, mode, rerank in RETRIEVERS:
            ranks, strict = [], []
            for q in questions:
                results = pipeline.search(q.question, top_k=MRR_DEPTH, mode=mode, rerank=rerank)
                rank = first_gold_rank(q, (r.chunk for r in results)) if q.answerable else None
                per_question[q.id][label] = {
                    "first_gold_rank": rank,
                    "first_evidence_rank": evidence_rank(q, (r.chunk for r in results)) if q.answerable else None,
                    "top5": [f"{r.chunk.source} > {r.chunk.section}" for r in results[:5]],
                }
                if q.answerable:
                    ranks.append(rank)
                    strict.append(per_question[q.id][label]["first_evidence_rank"])
            strict_rows[label] = {"all": summarize(strict)}
            by_kind = {
                kind: summarize([r for q, r in zip(answerable, ranks, strict=True) if q.kind == kind])
                for kind in KINDS[:3]
            }
            rows[label] = {"all": summarize(ranks), **by_kind}
        pipeline.index.close()

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src"], capture_output=True, text=True, cwd=HERE.parent.parent
    ).stdout.strip()
    summary = {
        "corpus": {
            "documents": len(report.added),
            "chunks": report.chunks,
            "source": "Prometheus v2.53.0 docs, see manifest.json",
        },
        "index": {key: meta[key] for key in ("embedding_model", "dimensions", "max_chunk_chars", "chunker_version")},
        "reranker": settings.rerank_model,
        "code": {"commit": git, "src_uncommitted_changes": dirty or "none"},
        "files_sha256": hashes,
        "questions": {kind: sum(q.kind == kind for q in questions) for kind in KINDS},
        "metric_notes": f"answerable questions only; ranks from the top {MRR_DEPTH}; no gold in the top {MRR_DEPTH} "
        "counts as a miss for hit@k and 0 for MRR@10",
        "results": {
            label: {
                part: {
                    "questions": s.questions,
                    **{f"hit@{k}": round(v, 3) for k, v in s.hits.items()},
                    "mrr@10": round(s.mrr, 3),
                }
                for part, s in parts.items()
            }
            for label, parts in rows.items()
        },
        "results_evidence_level": {
            label: {f"hit@{k}": round(v, 3) for k, v in parts["all"].hits.items()}
            | {"mrr@10": round(parts["all"].mrr, 3)}
            for label, parts in strict_rows.items()
        },
        "evidence_level_notes": "stricter: the chunk must be in a gold section and contain one of the evidence quotes",
    }
    assert summary["index"]["chunker_version"] == CHUNKER_VERSION
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (RESULTS / "per_question.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_question.values()))
    print(json.dumps({k: summary[k] for k in ("corpus", "index", "questions")}, indent=1))
    print("| Retriever | hit@1 | hit@3 | hit@5 | MRR@10 | hit@5 exact | hit@5 paraphrase | hit@5 confusable |")
    print("|---|---|---|---|---|---|---|---|")
    for label, parts in summary["results"].items():
        a = parts["all"]
        kinds = " | ".join(f"{parts[k]['hit@5']:.2f}" for k in KINDS[:3])
        print(f"| {label} | {a['hit@1']:.2f} | {a['hit@3']:.2f} | {a['hit@5']:.2f} | {a['mrr@10']:.2f} | {kinds} |")
    print("evidence-level:", json.dumps(summary["results_evidence_level"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="validate corpus and labels (offline)")
    group.add_argument("--freeze", action="store_true", help="write baseline.json from the current label files")
    parser.add_argument("--reason", help="with --freeze: why an existing baseline is replaced (kept in its history)")
    args = parser.parse_args()
    if args.check or args.freeze:
        problems = check_labels()
        for problem in problems:
            print(problem)
        if problems:
            sys.exit(1)
        hashes = file_hashes()
        print(f"labels ok: {len(load_questions(QUESTIONS))} questions, {len(corpus_chunks())} chunks")
        if args.freeze:
            current = {k: hashes[k] for k in ("questions.jsonl", "evidence.jsonl")}
            history = []
            if BASELINE.exists():
                if not args.reason:
                    sys.exit("baseline.json exists; replacing it needs --reason (kept in its history)")
                old = json.loads(BASELINE.read_text())
                history = old.pop("history", []) + [old | {"replaced_because": args.reason}]
            BASELINE.write_text(json.dumps(current | ({"history": history} if history else {}), indent=2) + "\n")
            print("froze", BASELINE.name)
    else:
        evaluate()
