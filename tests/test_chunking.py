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


def test_a_long_paragraph_is_split_at_sentence_ends_without_losing_text():
    sentences = [f"Step {i}: check the gauge and log the reading." for i in range(150)]
    paragraph = " ".join(sentences)  # 6,939 characters, no blank line anywhere

    chunks = chunk_markdown("big.md", f"# Big\n\n## Procedure\n\n{paragraph}\n", max_chars=1200)

    assert len(chunks) > 1
    assert all(len(c.text) <= 1200 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)  # cut between sentences, not inside one
    covered = " ".join(c.text for c in chunks)
    assert all(sentence in covered for sentence in sentences)


def test_text_without_any_break_is_cut_by_length_and_nothing_is_lost():
    token = "".join(f"{i:04d}" for i in range(1000))  # 4,000 characters, no space

    chunks = chunk_markdown("blob.md", f"# Blob\n\n{token}\n", max_chars=1200)

    assert [len(c.text) for c in chunks] == [1200, 1200, 1200, 400]
    assert "".join(c.text for c in chunks) == token


def test_the_overlap_paragraph_never_pushes_a_chunk_over_the_limit():
    # 600 + 599 characters fit under 1,200 on their own, but not with the blank line between them.
    paragraphs = ["a" * 1000, "b" * 600, "c" * 599]

    chunks = chunk_markdown("edge.md", "# Edge\n\n" + "\n\n".join(paragraphs), max_chars=1200)

    assert all(len(c.text) <= 1200 for c in chunks)
    assert "".join(c.text for c in chunks).count("c") == 599


def test_headings_ending_in_a_hash_keep_it_and_closing_hashes_are_dropped():
    doc = (
        "# C#\n\n## Notes on C#\n\ntext\n\n## F# tips\n\ntext\n\n"
        "## Closing hashes ##\n\ntext\n\n### Trailing #\n\ntext\n"
    )

    chunks = chunk_markdown("langs.md", doc)

    assert chunks[0].title == "C#"
    assert [c.section for c in chunks] == ["Notes on C#", "F# tips", "Closing hashes", "Closing hashes > Trailing"]


def test_chunk_ids_follow_everything_retrieval_reads():
    doc = "# Deploy\n\n## Rollback\n\nRun kctl rollback.\n"
    (chunk,) = chunk_markdown("deploy.md", doc)

    assert chunk_markdown("deploy.md", doc)[0].id == chunk.id  # same input, same id
    renamed_section = chunk_markdown("deploy.md", doc.replace("## Rollback", "## Undoing a release"))[0]
    renamed_title = chunk_markdown("deploy.md", doc.replace("# Deploy", "# Releases"))[0]
    assert len({chunk.id, renamed_section.id, renamed_title.id}) == 3
