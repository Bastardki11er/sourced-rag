# Index management

How `sourced ingest` builds and updates an index, what an index is tied to, and how to rebuild one
safely. Settings are described in [configuration.md](configuration.md).

## What ingest does

- **One folder per index.** The first `sourced ingest <folder>` binds the index to that folder (its resolved
  absolute path). Ingesting a different folder is refused before anything changes; give it its own index with
  `SOURCED_INDEX=other.db sourced ingest <other-folder>`.
- **Which files are documents.** Files ending in `.md`, `.markdown` or `.txt`, in any case (`README.MD`
  counts). Anything hidden inside the folder is skipped: a file or directory whose name starts with `.`
  (`.git`, `.venv`, `.draft.md`, ...) is not read, and a hidden directory is not even entered, so nothing in it
  is listed or checked for links. The folder you ingest may itself be hidden (`sourced ingest ~/.notes`); the
  rule applies to the paths inside it. Other files are ignored by a folder ingest.
- **Single files** (`sourced ingest docs/team/oncall.md`) must live inside the bound folder and keep their
  folder-relative name (`team/oncall.md`). On a new index, a single file binds its parent folder. A single
  file must be a document by the rule above: an unsupported suffix, or a hidden file or directory anywhere in
  its path inside the folder, stops `ingest` with exit code 2 and changes nothing.
- **Changed files only.** A file whose content hash is unchanged is skipped; a changed file has all of its
  chunks replaced in one transaction.
- **Deleted files.** Ingesting the whole folder removes documents whose file is gone, together with their
  chunks, keyword index entries and embeddings, and reports `removed N`. This only happens after a full
  folder scan where every file ingested successfully: ingesting a single file never removes anything, and a
  missing folder, a directory that can't be listed, or a file that can't be read (named in the error, e.g. not
  valid UTF-8) stops the run before any removal. Documents that an older version indexed from hidden paths
  are removed by the next successful folder ingest, like any other file that is no longer a document.
- **Symbolic links are not followed.** The folder itself may be reached through a link, but a visible link
  inside it (to a file, a directory, or nothing), a single file that is a link, or a single file reached through a
  linked directory inside the folder stops `ingest` with exit code 2 before anything is read, written or
  removed. Copy the file in or remove the link. This keeps text from outside the folder out of the index for
  a trusted local folder; it is checked once at the start, not against files being swapped while ingest runs.
- **One ingest at a time.** `sourced ingest` holds an OS file lock (`<index>.lock`, next to the index) for the
  whole run. A second ingest of the same index, from any process, stops right away with
  `another sourced ingest is running` and exit code 2. The lock is released when ingest ends, fails or its
  process dies; the lock file stays in place on purpose. The lock needs `fcntl`, so ingest runs on Linux and
  macOS and refuses to run on Windows.
- **A failed ingest can leave a partly updated index** that still answers searches. Only an ingest that exits
  0 has finished.
- A running server notices changes another process commits to the index (for example a `sourced ingest`
  while `sourced serve` runs) on its next query.

## What an index is tied to

The first ingest records five things in the index: the folder, the embedding model (as reported by the
loaded model), its vector dimensions, `SOURCED_MAX_CHUNK_CHARS` and the chunker version. Later commands check
them as follows:

| Change since the index was built | `ingest` | `bm25` search | `dense` / `hybrid` search, `/ask` |
|---|---|---|---|
| Different folder | refused (exit 2) | works | works |
| Different embedding model or vector size (`SOURCED_EMBEDDING_MODEL`) | refused (exit 2) | works | refused: HTTP 409, CLI exit 2 |
| Different `SOURCED_MAX_CHUNK_CHARS` | refused (exit 2) | works | works |
| Index built by an earlier chunker version | refused (exit 2) | works | works |
| Index built before any of this was recorded (no folder or model stored) | refused (exit 2) | works | refused: HTTP 409, CLI exit 2 |

Queries only compare what they need: dense and hybrid search embed the query, so they refuse an index whose
vectors come from another model. Chunk size and chunker version only decide how documents are cut, so an
index built with other values keeps answering searches as built; `ingest` refuses it so that old and new
chunks are never mixed. `/health` reports `dense_search: "ok"`, or the reason dense search is refused.
`/ask` always uses hybrid search, so it is refused in the same cases.

Nothing is migrated in place, and opening an index for queries never writes to it. To change any of these
settings, rebuild.

Chunker versions so far: version 2 put the heading into chunk ids; version 3 closes a code fence only on the
same character, at least as long (```` around a ``` example stays one block).

## Rebuilding

Build next to the old index; don't delete or overwrite it first:

```bash
SOURCED_INDEX=.sourced/index-new.db sourced ingest data/handbook   # must exit 0
SOURCED_INDEX=.sourced/index-new.db sourced search "how do I roll back?"
cp .sourced/index.db .sourced/index-backup.db                      # keep the old one
export SOURCED_INDEX=.sourced/index-new.db                         # then restart sourced serve
```

- Only an ingest that exits 0 means the rebuild is complete; rerun it until it does.
- A running server keeps reading the file it opened, even if another file is moved to that path, until it is
  restarted. Switching indexes always means restarting the server with the new `SOURCED_INDEX`.
- To go back, restart the server with `SOURCED_INDEX` pointing at the old file. It stays searchable as
  described in the table above.
