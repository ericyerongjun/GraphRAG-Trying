"""Persistent storage helpers for large-scale GraphRAG datasets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
import pandas as pd

from data_pipeline import Chunk


@dataclass(slots=True)
class DatasetPaths:
    """Locations of persisted artifacts."""

    dataset_parquet: Path
    embeddings_npy: Path
    metadata_json: Path


class DatasetStorage:
    """Writes and loads chunk datasets at scale using Parquet/NumPy."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def paths_for(self, name: str) -> DatasetPaths:
        dataset_dir = self.base_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return DatasetPaths(
            dataset_parquet=dataset_dir / "chunks.parquet",
            embeddings_npy=dataset_dir / "embeddings.npy",
            metadata_json=dataset_dir / "metadata.json",
        )

    def save_chunks(self, chunks: Sequence[Chunk], paths: DatasetPaths) -> None:
        records = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "local_index": chunk.local_index,
            }
            for chunk in chunks
        ]
        frame = pd.DataFrame.from_records(records)
        frame.to_parquet(paths.dataset_parquet, engine="pyarrow", index=False)
        metadata = {
            "count": len(records),
            "columns": list(frame.columns),
        }
        paths.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def load_chunks(self, paths: DatasetPaths) -> List[Chunk]:
        frame = pd.read_parquet(paths.dataset_parquet)
        chunks: List[Chunk] = []
        for row in frame.itertuples(index=False):
            chunks.append(
                Chunk(
                    chunk_id=str(row.chunk_id),
                    doc_id=str(row.doc_id),
                    text=str(row.text),
                    local_index=int(row.local_index),
                )
            )
        return chunks

    def save_embeddings(self, embeddings: np.ndarray, chunk_ids: Sequence[str], paths: DatasetPaths) -> None:
        np.save(paths.embeddings_npy, embeddings.astype(np.float32))
        meta = paths.metadata_json
        existing = {}
        if meta.exists():
            existing = json.loads(meta.read_text(encoding="utf-8"))
        existing.update({"embedding_dim": int(embeddings.shape[1]), "chunk_ids": list(chunk_ids)})
        meta.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def load_embeddings(self, paths: DatasetPaths) -> tuple[np.ndarray, List[str]]:
        embeddings = np.load(paths.embeddings_npy)
        meta = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
        chunk_ids: List[str] = meta.get("chunk_ids", [])
        if embeddings.shape[0] != len(chunk_ids):
            raise RuntimeError("Mismatch between embeddings and chunk metadata")
        return embeddings, chunk_ids

    def iter_chunk_batches(self, paths: DatasetPaths, batch_size: int = 1024) -> Iterator[Sequence[Chunk]]:
        frame = pd.read_parquet(paths.dataset_parquet)
        total = len(frame)
        for start in range(0, total, batch_size):
            batch = frame.iloc[start : start + batch_size]
            yield [
                Chunk(
                    chunk_id=str(row.chunk_id),
                    doc_id=str(row.doc_id),
                    text=str(row.text),
                    local_index=int(row.local_index),
                )
                for row in batch.itertuples(index=False)
            ]