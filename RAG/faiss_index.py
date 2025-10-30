"""FAISS-based vector index for large-scale retrieval."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import faiss  # type: ignore
import numpy as np

from embedding_pipeline import EmbeddingRecord


class FaissIndex:
    """Wraps a FAISS index with optional disk persistence."""

    def __init__(self, dimension: int, store_dir: Path | None = None, use_gpu: bool = False) -> None:
        self.dimension = dimension
        self.store_dir = store_dir
        self.use_gpu = use_gpu
        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatIP(dimension)
            self.index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            self.index = faiss.IndexFlatIP(dimension)
        self.ids: List[str] = []

    def add(self, records: Iterable[EmbeddingRecord]) -> None:
        vectors: List[np.ndarray] = []
        ids: List[str] = []
        for record in records:
            vectors.append(record.vector.astype(np.float32))
            ids.append(record.chunk_id)
        if not vectors:
            return
        matrix = np.vstack(vectors)
        self.index.add(matrix)
        self.ids.extend(ids)

    def search(self, vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        vector = vector.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(vector, top_k)
        top_hits: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            top_hits.append((self.ids[idx], float(score)))
        return top_hits

    def save(self) -> None:
        if self.store_dir is None:
            return
        self.store_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index, str(self.store_dir / "index.faiss"))
        (self.store_dir / "ids.txt").write_text("\n".join(self.ids), encoding="utf-8")

    def load(self) -> None:
        if self.store_dir is None:
            raise RuntimeError("store_dir not set for persistent index")
        index_path = self.store_dir / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        cpu_index = faiss.read_index(str(index_path))
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.index = cpu_index
        ids_path = self.store_dir / "ids.txt"
        self.ids = ids_path.read_text(encoding="utf-8").splitlines()

    def is_trained(self) -> bool:
        return self.index.is_trained and len(self.ids) > 0


__all__ = ["FaissIndex"]
