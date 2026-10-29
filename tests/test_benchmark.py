"""bench/run_benchmark.py: arguments, units, corpus scaling and the summary computed from raw results (offline)."""

import importlib.util
import json
from pathlib import Path

import pytest

BENCH = Path(__file__).parent.parent / "bench"
spec = importlib.util.spec_from_file_location("run_benchmark", BENCH / "run_benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.mark.parametrize(
    "argv",
    [
        ["--scales", "huge"],
        ["--scales", "base,base"],
        ["--scales", ""],
        ["--rounds", "0"],
        ["--rounds", "11"],
        ["--warmup", "-1"],
        ["--warmup", "11"],
    ],
)
def test_bad_arguments_are_refused(argv, tmp_path):
    with pytest.raises(SystemExit):
        bench.parse_args([*argv, "--out", str(tmp_path / "out")])


def test_existing_results_are_not_overwritten_without_force(tmp_path):
    (tmp_path / "raw.json").write_text("{}")
    with pytest.raises(SystemExit):
        bench.parse_args(["--out", str(tmp_path)])
    assert bench.parse_args(["--out", str(tmp_path), "--force"]).out == tmp_path


def test_defaults_and_scales():
    args = bench.parse_args(["--scales", "base, 10k"])
    assert args.scales == ["base", "10k"] and args.rounds == 3 and args.warmup == 3
    assert args.out.parent == BENCH / "results"


def test_units_are_converted_to_mib():
    assert bench.to_mib(1024**2, "B") == 1
    assert bench.to_mib(2048, "KiB") == 2
    assert bench.to_mib(3, "MiB") == 3


def test_copies_reach_the_target_chunk_count():
    assert bench.copies_for(None, 417) == 1
    assert bench.copies_for(1000, 417) == 3
    assert bench.copies_for(10000, 417) == 24
    assert bench.copies_for(100, 417) == 1


def test_scaled_corpus_only_changes_titles_and_edits_an_even_share(tmp_path, monkeypatch):
    source = tmp_path / "src"
    for name in ("a.md", "b/c.md"):
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_text(f"# {name}\n\nBody of {name}.\n")
    monkeypatch.setattr(bench, "CORPUS", source)
    bench.build_corpus(tmp_path / "out", 3)

    files = sorted((tmp_path / "out").rglob("*.md"))
    assert len(files) == 6
    assert (tmp_path / "out/copy-000/a.md").read_text() == "# a.md\n\nBody of a.md.\n"
    assert (tmp_path / "out/copy-002/b/c.md").read_text() == "# b/c.md (copy 2)\n\nBody of b/c.md.\n"
    assert len(bench.files_to_edit(tmp_path / "out")) == 1  # 5% of 6, at least one
    assert len(bench.files_to_edit(tmp_path / "out", share=0.5)) == 3


def _round(samples, first=0.002):
    return {
        "model_load": {"embedder_s": 0.5, "reranker_s": 0.25},
        "rss_mib_after_model_load": 300.0,
        "rss_mib_after_vector_cache": 320.0,
        "rss_mib_after_config": {name: 400.0 for name in bench.CONFIGS},
        "peak_rss_mib": 500.0,
        "first_query_s": {name: first for name in bench.CONFIGS},
        "samples_s": {name: samples for name in bench.CONFIGS},
    }


def test_the_summary_is_computed_from_the_raw_samples():
    samples = [i / 1000 for i in range(1, 101)]  # 1..100 ms
    ingest_run = {"seconds": 10.0, "added": 2, "updated": 0, "unchanged": 0, "removed": 0, "chunks": 40}
    raw = {
        "scales": {
            "base": {
                "ingest": {
                    "documents": 2,
                    "full": ingest_run,
                    "unchanged": ingest_run | {"seconds": 0.01},
                    "incremental": ingest_run | {"seconds": 1.0, "edited_files": 1},
                    "index_files_after_full": {"index.db": 2 * 1024**2, "index.db.lock": 0},
                    "rss_mib_after_model_load": 300.0,
                    "rss_mib_after_full_ingest": 900.0,
                    "peak_rss_mib": 1000.0,
                },
                "query_rounds": [_round(samples), _round(samples[::-1])],
            }
        }
    }
    s = bench.summarize(raw)["base"]

    assert s["chunks"] == 40 and s["ingest_s"] == {"full": 10.0, "unchanged": 0.01, "incremental": 1.0}
    assert s["model_load_s"] == [0.75, 0.75]
    rounds = s["latency_ms"]["hybrid+rerank"]["per_round"]
    assert rounds == [{"n": 100, "p50": 50.5, "p95": 95.05}] * 2  # per round, numpy linear interpolation
    assert s["latency_ms"]["bm25"]["first_query_in_new_process"] == [2.0, 2.0]
    assert "| base | 2 | 40 | 10.0 s | 0.0 s | 1.0 s | 2.0 MiB |" in bench.markdown(bench.summarize(raw))


def test_published_summaries_match_their_raw_results():
    published = sorted(p for p in (BENCH / "results").glob("*/raw.json"))
    for raw_path in published:
        raw = json.loads(raw_path.read_text())
        assert json.loads((raw_path.parent / "summary.json").read_text()) == bench.summarize(raw), raw_path
