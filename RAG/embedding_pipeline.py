"""Embedding helpers for GraphRAG."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(slots=True)
class EmbeddingRecord:
    chunk_id: str
    vector: np.ndarray


class EmbeddingService:
    """Wraps model inference for text embeddings."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - handled at runtime
            raise RuntimeError(
                "sentence-transformers is required for EmbeddingService. Install with 'pip install sentence-transformers'."
            ) from exc
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        embeddings = self.model.encode(list(texts), convert_to_numpy=True)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            embeddings = embeddings / norms
        return embeddings


class EmbeddingIndex:
    """Stores embeddings in memory and runs cosine similarity search."""

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._ids: List[str] = []

    def add(self, records: Iterable[EmbeddingRecord]) -> None:
        vectors: List[np.ndarray] = []
        ids: List[str] = []
        for record in records:
            if not isinstance(record.vector, np.ndarray):
                raise TypeError("Embedding vector must be a numpy.ndarray")
            vectors.append(record.vector.astype(np.float32))
            ids.append(record.chunk_id)
        if not vectors:
            return
        matrix = np.vstack(vectors)
        if self._vectors is None:
            self._vectors = matrix
            self._ids = ids
        else:
            self._vectors = np.vstack([self._vectors, matrix])
            self._ids.extend(ids)

    def is_ready(self) -> bool:
        return self._vectors is not None and len(self._ids) > 0

    def search(self, query_vector: np.ndarray, top_k: int) -> List[tuple[str, float]]:
        if self._vectors is None:
            raise RuntimeError("Embedding index is empty")
        if query_vector.ndim != 1:
            raise ValueError("query_vector must be a 1-D numpy array")
        query_norm = np.linalg.norm(query_vector)
        if math.isclose(query_norm, 0.0):
            raise ValueError("query_vector norm is zero")
        normalized_query = query_vector / query_norm
        scores = self._vectors @ normalized_query
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self._ids[idx], float(scores[idx])) for idx in top_indices]
