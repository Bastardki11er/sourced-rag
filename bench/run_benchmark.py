"""Single-machine, sequential performance benchmark for ingest and retrieval.

    python bench/run_benchmark.py                          # scales base,1k,10k; 3 query rounds
    python bench/run_benchmark.py --scales base --rounds 1 --out /tmp/bench-trial

Measures, per corpus scale, in fresh subprocesses: model loading, a full ingest into a new index, re-ingesting
an unchanged corpus, an incremental ingest after editing 5% of the files, index file sizes, process memory,
and the latency of complete `Pipeline.search` calls for BM25, dense, hybrid and hybrid + rerank.

Corpora are built in a temporary folder from eval/prometheus/corpus (23 real pages). "base" is that corpus.
Larger scales repeat all 23 pages K times (copy-000 ... copy-K-1); copies 1 and up differ only in their H1
title, which gets " (copy k)". So beyond "base", the text is duplicated real documentation, not new documents.

Uses the locally cached fastembed models (FASTEMBED_CACHE_PATH) with downloads disabled, no Claude, and no
credentials. Indexes and corpora live in a temporary folder that is deleted at the end. Results go to a new
folder (bench/results/<UTC time> by default); an existing folder is only overwritten with --force.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "prometheus" / "corpus"
QUESTIONS = ROOT / "eval" / "prometheus" / "questions.jsonl"
SCALES = {"base": None, "1k": 1000, "10k": 10000}  # target chunk counts; None: the corpus as it is
CONFIGS = {
    "bm25": ("bm25", False),
    "dense": ("dense", False),
    "hybrid": ("hybrid", False),
    "hybrid+rerank": ("hybrid", True),
}
TOP_K = 5
EDIT_SHARE = 0.05
EDIT_TEXT = "\n\nBenchmark edit: this paragraph was appended to measure an incremental ingest.\n"


# --- corpus generation -----------------------------------------------------------------


def copies_for(target: int | None, base_chunks: int) -> int:
    """How many copies of the base corpus reach at least ``target`` chunks (1 for the base scale)."""
    return 1 if target is None or target <= base_chunks else math.ceil(target / base_chunks)


def build_corpus(dest: Path, copies: int) -> None:
    pages = sorted(CORPUS.rglob("*.md"))
    for k in range(copies):
        for page in pages:
            text = page.read_text(encoding="utf-8")
            if k:  # copy 0 is the original; others only change the H1 title
                first, rest = text.split("\n", 1)
                text = f"{first} (copy {k})\n{rest}"
            out = (dest / f"copy-{k:03d}" if copies > 1 else dest) / page.relative_to(CORPUS)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")


def files_to_edit(folder: Path, share: float = EDIT_SHARE) -> list[Path]:
    """A fixed, evenly spread ``share`` of the corpus files (at least one), in sorted order."""
    files = sorted(folder.rglob("*.md"))
    count = max(1, math.ceil(len(files) * share))
    step = len(files) / count
    return [files[int(i * step)] for i in range(count)]


# --- measurement helpers ---------------------------------------------------------------


def to_mib(value: float, unit: str) -> float:
    return value * {"B": 1, "KiB": 1024, "MiB": 1024**2}[unit] / 1024**2


def peak_rss_mib() -> float:
    """Peak resident set size of this process (getrusage: bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return to_mib(peak, "B" if sys.platform == "darwin" else "KiB")


def rss_mib() -> float:
    """Current resident set size of this process."""
    if sys.platform.startswith("linux"):
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return to_mib(int(line.split()[1]), "KiB")
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True)
    return to_mib(int(out.stdout.strip()), "KiB")  # ps reports KiB on macOS and Linux


def index_files(index: Path) -> dict[str, int]:
    """Size in bytes of the index file and every side file next to it (journal, lock)."""
    return {p.name: p.stat().st_size for p in sorted(index.parent.glob(index.name + "*"))}


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q))  # numpy's default linear interpolation


def load_models():
    from sourced.config import Settings
    from sourced.embeddings import FastEmbedEmbedder
    from sourced.retrieval import CrossEncoderReranker

    settings = Settings()
    started = time.perf_counter()
    embedder = FastEmbedEmbedder(settings.embedding_model)
    loaded_embedder = time.perf_counter()
    reranker = CrossEncoderReranker(settings.rerank_model)
    timings = {"embedder_s": loaded_embedder - started, "reranker_s": time.perf_counter() - loaded_embedder}
    return settings, embedder, reranker, timings


def pipeline_for(index_path: Path, settings, embedder, reranker):
    from dataclasses import replace

    from sourced.generation import ExtractiveAnswerer
    from sourced.pipeline import Pipeline
    from sourced.retrieval import Retriever
    from sourced.store import Index

    index = Index(index_path)
    return Pipeline(
        replace(settings, index_path=index_path),
        index,
        embedder,
        ExtractiveAnswerer(),
        Retriever(index, embedder, reranker),
    )


# --- child processes ---------------------------------------------------------------------


def ingest_job(corpus: str, index: str, out: str) -> None:
    corpus_dir, index_path = Path(corpus), Path(index)
    record = {"rss_mib_start": rss_mib()}
    settings, embedder, reranker, record["model_load"] = load_models()
    record["rss_mib_after_model_load"] = rss_mib()
    pipeline = pipeline_for(index_path, settings, embedder, reranker)

    def timed_ingest():
        started = time.perf_counter()
        report = pipeline.ingest(corpus_dir)
        return {
            "seconds": time.perf_counter() - started,
            **{k: len(getattr(report, k)) for k in ("added", "updated", "unchanged", "removed")},
            "chunks": report.chunks,
        }

    record["full"] = timed_ingest()
    record["rss_mib_after_full_ingest"] = rss_mib()
    record["index_files_after_full"] = index_files(index_path)
    record["unchanged"] = timed_ingest()
    edited = files_to_edit(corpus_dir)
    for path in edited:
        path.write_text(path.read_text(encoding="utf-8") + EDIT_TEXT, encoding="utf-8")
    record["incremental"] = timed_ingest() | {"edited_files": len(edited)}
    record["index_files_after_incremental"] = index_files(index_path)
    meta = pipeline.index.meta()
    record["index_meta"] = {k: meta[k] for k in ("embedding_model", "dimensions", "max_chunk_chars", "chunker_version")}
    record["documents"] = len(pipeline.index.sources())
    record["rss_mib_end"] = rss_mib()
    record["peak_rss_mib"] = peak_rss_mib()
    pipeline.index.close()
    Path(out).write_text(json.dumps(record))


def query_job(index: str, order: list[str], warmup: int, out: str) -> None:
    questions = [json.loads(line)["question"] for line in QUESTIONS.read_text().splitlines()]
    record = {"order": order, "rss_mib_start": rss_mib()}
    settings, embedder, reranker, record["model_load"] = load_models()
    record["rss_mib_after_model_load"] = rss_mib()
    pipeline = pipeline_for(Path(index), settings, embedder, reranker)
    record["first_query_s"], record["samples_s"], record["rss_mib_after_config"] = {}, {}, {}
    for name in order:
        mode, rerank = CONFIGS[name]

        def search(question, mode=mode, rerank=rerank):
            started = time.perf_counter()
            results = pipeline.search(question, top_k=TOP_K, mode=mode, rerank=rerank)
            elapsed = time.perf_counter() - started
            assert len(results) == TOP_K
            return elapsed

        record["first_query_s"][name] = search(questions[0])
        if mode != "bm25" and "rss_mib_after_vector_cache" not in record:
            record["rss_mib_after_vector_cache"] = rss_mib()  # the first dense query loads every vector
        for question in questions[1 : 1 + warmup]:
            search(question)
        record["samples_s"][name] = [search(q) for q in questions]
        record["rss_mib_after_config"][name] = rss_mib()
    record["rss_mib_end"] = rss_mib()
    record["peak_rss_mib"] = peak_rss_mib()
    pipeline.index.close()
    Path(out).write_text(json.dumps(record))


def run_child(target, *args) -> dict:
    out = Path(args[-1])
    process = multiprocessing.get_context("spawn").Process(target=target, args=args)
    process.start()
    process.join()
    if process.exitcode != 0:
        raise SystemExit(f"{target.__name__} failed with exit code {process.exitcode}")
    return json.loads(out.read_text())


# --- summary ---------------------------------------------------------------------------


def summarize(raw: dict) -> dict:
    """Everything reported, computed only from the raw results (so it can be recomputed and checked)."""
    summary = {}
    for scale, data in raw["scales"].items():
        ingest = data["ingest"]
        rounds = data["query_rounds"]
        summary[scale] = {
            "documents": ingest["documents"],
            "chunks": ingest["full"]["chunks"],
            "ingest_s": {k: round(ingest[k]["seconds"], 2) for k in ("full", "unchanged", "incremental")},
            "incremental_edited_files": ingest["incremental"]["edited_files"],
            "index_bytes": ingest["index_files_after_full"],
            "model_load_s": [round(r["model_load"]["embedder_s"] + r["model_load"]["reranker_s"], 2) for r in rounds],
            "rss_mib": {
                "ingest_after_model_load": round(ingest["rss_mib_after_model_load"], 1),
                "ingest_after_full_ingest": round(ingest["rss_mib_after_full_ingest"], 1),
                "ingest_peak": round(ingest["peak_rss_mib"], 1),
                "query_after_model_load": [round(r["rss_mib_after_model_load"], 1) for r in rounds],
                "query_after_vector_cache": [round(r["rss_mib_after_vector_cache"], 1) for r in rounds],
                "query_after_each_config": [
                    {k: round(v, 1) for k, v in r["rss_mib_after_config"].items()} for r in rounds
                ],
                "query_peak": [round(r["peak_rss_mib"], 1) for r in rounds],
            },
            "latency_ms": {
                name: {
                    "per_round": [
                        {
                            "n": len(r["samples_s"][name]),
                            "p50": round(percentile(r["samples_s"][name], 50) * 1000, 2),
                            "p95": round(percentile(r["samples_s"][name], 95) * 1000, 2),
                        }
                        for r in rounds
                    ],
                    "first_query_in_new_process": [round(r["first_query_s"][name] * 1000, 1) for r in rounds],
                }
                for name in CONFIGS
            },
        }
    return summary


def markdown(summary: dict) -> str:
    lines = [
        "| Scale | Docs | Chunks | Full ingest | Unchanged | Incremental | Index | "
        + " | ".join(f"{n} p50 / p95 (ms)" for n in CONFIGS)
        + " |",
        "|" + "---|" * (7 + len(CONFIGS)),
    ]
    for scale, s in summary.items():
        lat = " | ".join(
            ", ".join(f"{r['p50']:.1f} / {r['p95']:.1f}" for r in s["latency_ms"][n]["per_round"]) for n in CONFIGS
        )
        size = sum(s["index_bytes"].values()) / 1024**2
        lines.append(
            f"| {scale} | {s['documents']} | {s['chunks']} | {s['ingest_s']['full']:.1f} s | "
            f"{s['ingest_s']['unchanged']:.1f} s | {s['ingest_s']['incremental']:.1f} s | {size:.1f} MiB | {lat} |"
        )
    return "\n".join(lines) + "\n"


# --- environment ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment() -> dict:
    import sqlite3

    import fastembed
    import onnxruntime

    def sysctl(name):
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True)
        return out.stdout.strip() or None

    if sys.platform == "darwin":
        cpu, memory = sysctl("machdep.cpu.brand_string"), int(sysctl("hw.memsize") or 0)
    else:
        info = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
        cpu = next((line.split(":", 1)[1].strip() for line in info.splitlines() if line.startswith("model name")), None)
        mem = Path("/proc/meminfo").read_text().split()[1] if Path("/proc/meminfo").exists() else 0
        memory = int(mem) * 1024
    return {
        "os": platform.platform(terse=True),
        "machine": platform.machine(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "memory_gib": round(memory / 1024**3, 1),
        "threads": "library defaults (onnxruntime, NumPy); thread environment variables: "
        + (", ".join(f"{k}={v}" for k, v in os.environ.items() if k.endswith("_NUM_THREADS")) or "none set"),
        "python": platform.python_version(),
        "versions": {
            "sqlite": sqlite3.sqlite_version,
            "numpy": np.__version__,
            "fastembed": fastembed.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
    }


def code_identity() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "src_uncommitted_changes": git("status", "--porcelain", "--", "src") or "none",
        "src_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src" / "sourced").glob("*.py"))},
        "benchmark_sha256": sha256(Path(__file__)),
        "questions_sha256": sha256(QUESTIONS),
        "corpus_manifest_sha256": sha256(ROOT / "eval" / "prometheus" / "manifest.json"),
    }


# --- main ------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scales", default="base,1k,10k", help=f"comma-separated, from {', '.join(SCALES)}")
    parser.add_argument("--rounds", type=int, default=3, help="query rounds, each in a new process (1-10)")
    parser.add_argument(
        "--warmup", type=int, default=3, help="untimed queries per configuration before sampling (0-10)"
    )
    parser.add_argument("--out", type=Path, help="results folder (default: bench/results/<UTC time>)")
    parser.add_argument("--force", action="store_true", help="write into an existing results folder")
    args = parser.parse_args(argv)
    args.scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    unknown = [s for s in args.scales if s not in SCALES]
    if unknown or not args.scales or len(set(args.scales)) != len(args.scales):
        parser.error(f"--scales takes distinct values from {', '.join(SCALES)}")
    if not 1 <= args.rounds <= 10:
        parser.error("--rounds must be between 1 and 10")
    if not 0 <= args.warmup <= 10:
        parser.error("--warmup must be between 0 and 10")
    started = datetime.now(UTC)
    args.started = started
    args.out = args.out or ROOT / "bench" / "results" / started.strftime("%Y%m%dT%H%M%SZ")
    if args.out.exists() and any(args.out.iterdir()) and not args.force:
        parser.error(f"{args.out} already has results; choose another --out or pass --force")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use cached models only; children inherit this
    raw = {
        "started_utc": args.started.isoformat(timespec="seconds"),
        "scope": "single machine, one process at a time, sequential calls; no HTTP, no concurrency",
        "settings": {
            "top_k": TOP_K,
            "candidates_per_retriever": 20,
            "warmup_queries": args.warmup,
            "rounds": args.rounds,
            "edit_share": EDIT_SHARE,
            "questions": "eval/prometheus/questions.jsonl",
        },
        "corpus_rule": "base = eval/prometheus/corpus (23 pages); larger scales = K copies of it, copies 1+ only "
        "get ' (copy k)' appended to their H1 title",
        "environment": environment(),
        "code": code_identity(),
        "scales": {},
    }
    base_chunks = None
    with tempfile.TemporaryDirectory(prefix="sourced-bench-") as tmp:
        for scale in args.scales:
            work = Path(tmp) / scale
            if base_chunks is None:
                from sourced.chunking import chunk_markdown

                base_chunks = sum(len(chunk_markdown(p.name, p.read_text())) for p in CORPUS.rglob("*.md"))
            copies = copies_for(SCALES[scale], base_chunks)
            build_corpus(work / "corpus", copies)
            print(f"[{scale}] {copies} cop{'y' if copies == 1 else 'ies'}: ingesting...", flush=True)
            started = time.perf_counter()
            ingest = run_child(ingest_job, str(work / "corpus"), str(work / "index.db"), str(work / "ingest.json"))
            rounds = []
            for r in range(args.rounds):
                names = list(CONFIGS)
                order = names[r % len(names) :] + names[: r % len(names)]  # rotate which configuration runs first
                print(f"[{scale}] query round {r + 1}: {', '.join(order)}", flush=True)
                rounds.append(
                    run_child(query_job, str(work / "index.db"), order, args.warmup, str(work / f"q{r}.json"))
                )
            raw["scales"][scale] = {
                "copies": copies,
                "ingest": ingest,
                "query_rounds": rounds,
                "wall_s": time.perf_counter() - started,
            }
    raw["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    summary = summarize(raw)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "raw.json").write_text(json.dumps(raw, indent=1) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "summary.md").write_text(markdown(summary))
    print(markdown(summary))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
