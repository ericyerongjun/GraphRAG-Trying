"""Command line helper to create chunked datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

try:  # Allow execution as module or script
    from .config import GraphRAGConfig
    from .data_pipeline import DatasetBuilder
except ImportError:  # pragma: no cover - fallback when run as script
    from config import GraphRAGConfig
    from data_pipeline import DatasetBuilder


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a chunked dataset for GraphRAG")
    parser.add_argument("input_dir", type=Path, help="Directory containing source .txt files")
    parser.add_argument("output", type=Path, help="JSONL path to write dataset chunks")
    parser.add_argument("--chunk-size", type=int, default=500, help="Number of characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Overlap between successive chunks")
    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()
    config = GraphRAGConfig(data_dir=args.output.parent, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    config.resolve_paths()
    builder = DatasetBuilder(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    documents = builder.load_documents(args.input_dir)
    chunks = builder.build_chunks(documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    builder.save_dataset(chunks, args.output)
    print(f"Created dataset with {len(chunks)} chunks at {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI execution path
    main()
