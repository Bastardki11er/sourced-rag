"""Runs `Pipeline.ingest` in its own process and stops inside it until told to go on.

Used by test_ingest_lock.py: python ingest_child.py INDEX FOLDER. Prints "paused" once
it is embedding (so after binding, the config check and listing the folder), then waits
for a line on stdin before finishing.
"""

import sys
from pathlib import Path

from sourced.config import Settings
from sourced.embeddings import HashingEmbedder
from sourced.generation import ExtractiveAnswerer
from sourced.pipeline import Pipeline
from sourced.store import Index


class PausingEmbedder(HashingEmbedder):
    def embed_documents(self, texts):
        print("paused", flush=True)
        sys.stdin.readline()
        return super().embed_documents(texts)


index_path, folder = Path(sys.argv[1]), Path(sys.argv[2])
embedder = PausingEmbedder()
pipeline = Pipeline(Settings(index_path=index_path, rerank=False), Index(index_path), embedder, ExtractiveAnswerer())
report = pipeline.ingest(folder)
print("done", report.added, report.updated, report.removed, flush=True)
