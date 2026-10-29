from pathlib import Path

import pytest

from sourced.retrieval import reciprocal_rank_fusion
from sourced.store import to_fts_query

HANDBOOK = Path(__file__).parent.parent / "data" / "handbook"


def test_fts_query_quotes_terms_so_user_input_cannot_inject_syntax():
    assert to_fts_query('What is KR-4021 NEAR "(drop"?') == '"KR" OR "4021" OR "NEAR" OR "drop"'
    assert to_fts_query("what is the") == ""


def test_rrf_rewards_documents_both_retrievers_agree_on():
    fused = reciprocal_rank_fusion({"bm25": ["a", "b", "c"], "dense": ["c", "a", "d"]}, k=60)
    order = [chunk_id for chunk_id, _, _ in fused]

    assert order[:2] == ["a", "c"]  # in both lists
    assert set(order[2:]) == {"b", "d"}
    assert fused[0][2] == {"bm25": 1, "dense": 2}
    assert fused[0][1] == 1 / 61 + 1 / 62


def test_bm25_finds_exact_error_codes(handbook_pipeline):
    top = handbook_pipeline.search("KR-4021", top_k=1, mode="bm25")[0]

    assert top.chunk.source == "error-codes.md"
    assert "Payload errors" in top.chunk.section


def test_hybrid_results_report_where_each_retriever_ranked_them(handbook_pipeline):
    results = handbook_pipeline.search("how do I roll back a release", top_k=3, mode="hybrid")

    assert results[0].chunk.section == "Rolling back"
    assert set(results[0].ranks) <= {"bm25", "dense"}
    assert results[0].ranks


def test_reranker_reorders_the_fused_candidates(make_pipeline):
    class PrefersSabbatical:
        def score(self, query, passages):
            return [1.0 if "sabbatical" in p.lower() else 0.0 for p in passages]

    pipeline = make_pipeline(reranker=PrefersSabbatical())
    pipeline.ingest(HANDBOOK)

    top = pipeline.search("time off", top_k=1, mode="hybrid", rerank=True)[0]

    assert top.chunk.section == "Sabbatical"


def test_reingesting_unchanged_files_is_a_no_op(make_pipeline, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\n## One\n\nFirst version.")
    pipeline = make_pipeline()

    first = pipeline.ingest(docs)
    second = pipeline.ingest(docs)
    (docs / "a.md").write_text("# A\n\n## One\n\nSecond version.\n\n## Two\n\nNew section.")
    third = pipeline.ingest(docs)

    assert first.added == ["a.md"]
    assert second.unchanged == ["a.md"]
    assert third.updated == ["a.md"]
    assert third.chunks == 2  # old chunks were replaced, not duplicated
    assert "Second version." in pipeline.search("version", top_k=1, mode="bm25")[0].chunk.text


def test_fts_query_keeps_accented_words_whole():
    assert to_fts_query("café résumé naïve") == '"café" OR "résumé" OR "naïve"'
    # The same words typed as base letter + combining accent come out identical.
    assert to_fts_query("café naïve") == '"café" OR "naïve"'


def test_bm25_finds_accented_words(make_pipeline, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "office.md").write_text("# Office\n\n## Café\n\nThe café on floor 2 opens at 8.\n")
    (docs / "hr.md").write_text("# HR\n\n## Hiring\n\nEvery résumé gets two reviewers.\n")
    pipeline = make_pipeline()
    pipeline.ingest(docs)

    assert pipeline.search("café hours", top_k=1, mode="bm25")[0].chunk.source == "office.md"
    assert pipeline.search("résumé", top_k=1, mode="bm25")[0].chunk.source == "hr.md"


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_a_blank_query_is_refused(handbook_pipeline, blank):
    with pytest.raises(ValueError, match="empty"):
        handbook_pipeline.search(blank)
