from pathlib import Path

from sourced.chunking import Chunk
from sourced.evaluation import Question, first_gold_rank, load_questions, score_retrieval

QUESTIONS = Path(__file__).parent.parent / "eval" / "questions.jsonl"


def chunk(source, section):
    return Chunk("id", source, "Title", section, "text", 0)


def test_gold_rank_matches_on_source_and_section():
    q = Question("q", "?", "lexical", gold=[{"source": "a.md", "section": "Rollback"}])
    ranked = [chunk("b.md", "Rollback"), chunk("a.md", "Freeze"), chunk("a.md", "Deploy > Rollback")]

    assert first_gold_rank(q, ranked) == 3
    assert first_gold_rank(q, ranked[:2]) is None


def test_question_set_is_well_formed():
    questions = load_questions(QUESTIONS)

    assert len({q.id for q in questions}) == len(questions)
    assert {q.kind for q in questions} == {"lexical", "paraphrase", "unanswerable"}
    assert all(q.answerable == (q.kind != "unanswerable") for q in questions)


def test_bm25_baseline_on_the_handbook_does_not_regress(handbook_pipeline):
    answerable = [q for q in load_questions(QUESTIONS) if q.answerable]

    scores = score_retrieval(handbook_pipeline, answerable, mode="bm25")

    # The real numbers are in the README; this guards against chunking/indexing regressions.
    assert scores.hits[5] >= 0.85
    assert scores.mrr >= 0.7
