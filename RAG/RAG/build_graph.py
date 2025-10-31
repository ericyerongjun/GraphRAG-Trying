"""Build and export a knowledge graph from chunked data."""
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx

try:  # Support relative imports when executed as module
    from .config import GraphRAGConfig
    from .data_pipeline import DatasetBuilder
    from .graph_builder import KnowledgeGraphBuilder
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import GraphRAGConfig
    from data_pipeline import DatasetBuilder
    from graph_builder import KnowledgeGraphBuilder


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construct a knowledge graph from dataset chunks")
    parser.add_argument("dataset", type=Path, help="Path to JSONL dataset")
    parser.add_argument("output", type=Path, help="GraphML file to write")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--min-token-length", type=int, default=3)
    parser.add_argument("--max-keywords", type=int, default=12)
    parser.add_argument("--weight-threshold", type=int, default=2)
    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()
    config = GraphRAGConfig(
        data_dir=args.dataset.parent,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_token_length=args.min_token_length,
        max_keywords=args.max_keywords,
    )
    builder = DatasetBuilder(config.chunk_size, config.chunk_overlap)
    chunks = builder.load_dataset(args.dataset)
    if not chunks:
        raise SystemExit(f"Dataset {args.dataset} is empty")
    graph_builder = KnowledgeGraphBuilder(
        stop_words=config.stop_words,
        min_token_length=config.min_token_length,
        max_keywords=config.max_keywords,
    )
    graph = graph_builder.build(chunks, weight_threshold=args.weight_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, args.output)
    print(f"Wrote graph with {graph.number_of_nodes()} nodes to {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI execution path
    main()
