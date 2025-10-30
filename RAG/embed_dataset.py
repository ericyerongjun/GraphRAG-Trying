"""Embed an existing chunked dataset and persist embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:  # Support relative imports when executed as module
    from .data_pipeline import DatasetBuilder
    from .embedding_pipeline import EmbeddingService
except ImportError:  # pragma: no cover - fallback for direct script execution
    from data_pipeline import DatasetBuilder
    from embedding_pipeline import EmbeddingService


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed chunks produced by dataset_maker")
    parser.add_argument("dataset", type=Path, help="Path to JSONL dataset")
    parser.add_argument("output", type=Path, help="Path to store embeddings (npz)")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", type=str, default=None, help="Optional device hint for sentence-transformers")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()
    builder = DatasetBuilder(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = builder.load_dataset(args.dataset)
    if not chunks:
        raise SystemExit(f"Dataset {args.dataset} is empty")
    embedder = EmbeddingService(args.embedding_model, device=args.device)
    embeddings = embedder.embed([chunk.text for chunk in chunks])
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, embeddings=embeddings, chunk_ids=np.array(chunk_ids))
    print(f"Stored embeddings for {len(chunks)} chunks at {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI execution path
    main()
