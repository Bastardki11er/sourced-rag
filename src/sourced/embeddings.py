"""Text embedding models.

The default runs ``BAAI/bge-small-en-v1.5`` locally through ONNX Runtime
(fastembed): 384 dimensions, ~67 MB, no GPU and no API key needed.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class FastEmbedEmbedder:
    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding  # heavy import, only when actually used

        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self._model.passage_embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        # bge models expect queries and passages to be embedded differently;
        # query_embed adds the query instruction for us.
        return _normalize(np.array(next(iter(self._model.query_embed(text))), dtype=np.float32))


class HashingEmbedder:
    """Deterministic bag-of-words vectors. Used in tests so they need no model download."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimensions] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return _normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vector(text))
