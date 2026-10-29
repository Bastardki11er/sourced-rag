import pytest

from sourced.chunking import chunk_markdown

DOC = """\
# Deploying

Intro paragraph.

## Release trains

Trains leave on Tuesday.

### Hotfixes

Hotfixes can skip the train.

## Rolling back

Run this:

```
# not a heading, just a shell comment
kctl rollback --to <release-id>
```
"""


def test_each_heading_starts_a_chunk_with_its_full_path():
    chunks = chunk_markdown("deploy.md", DOC)

    assert [c.heading for c in chunks] == [
        "Deploying",
        "Deploying > Release trains",
        "Deploying > Release trains > Hotfixes",
        "Deploying > Rolling back",
    ]
    assert all(c.title == "Deploying" for c in chunks)


def test_headings_inside_code_fences_are_ignored():
    rollback = chunk_markdown("deploy.md", DOC)[-1]

    assert "# not a heading" in rollback.text
    assert "kctl rollback" in rollback.text


def test_indexed_text_carries_the_heading_so_retrieval_sees_context():
    hotfixes = chunk_markdown("deploy.md", DOC)[2]

    assert hotfixes.indexed_text.startswith("Deploying > Release trains > Hotfixes\n\n")


def test_long_sections_split_on_paragraphs_with_overlap():
    paragraphs = [f"Paragraph {i} " + "word " * 40 for i in range(6)]
    doc = "# Big\n\n## Section\n\n" + "\n\n".join(paragraphs)

    chunks = chunk_markdown("big.md", doc, max_chars=500)

    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)
    # The last paragraph of one chunk is repeated at the start of the next.
    assert chunks[1].text.startswith(chunks[0].text.split("\n\n")[-1])
    assert all(c.section == "Section" for c in chunks)


def test_chunk_ids_are_stable_across_runs():
    first, second = chunk_markdown("deploy.md", DOC), chunk_markdown("deploy.md", DOC)
    assert [c.id for c in first] == [c.id for c in second]


@pytest.mark.parametrize("body", ["word " * 200, "x" * 1000])
def test_single_long_paragraph_respects_budget(body):
    chunks = chunk_markdown("long.md", "# Long\n" + body, max_chars=100)
    assert len(chunks) > 1
    assert all(0 < len(c.text) <= 100 for c in chunks)
    assert "".join("".join(c.text.split()) for c in chunks) == "".join(body.split())


def test_mixed_fences_do_not_end_code_block():
    doc = "# Real\n````markdown\n~~~\n# Still code\n```\n# Also code\n````\n## After\nBody"
    chunks = chunk_markdown("fences.md", doc)
    assert len(chunks) == 2
    assert "# Still code" in chunks[0].text
    assert "# Also code" in chunks[0].text
    assert chunks[1].section == "After"
