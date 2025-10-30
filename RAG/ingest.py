"""End-to-end ingestion pipeline for GraphRAG assets."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:  # Support package-relative imports and direct execution
    from .config import GraphRAGConfig
    from .data_pipeline import DatasetBuilder
    from .embedding_pipeline import EmbeddingService, EmbeddingRecord, EmbeddingIndex
    from .graph_builder import KnowledgeGraphBuilder
except ImportError:  # pragma: no cover - fallback for script execution
    from config import GraphRAGConfig
    from data_pipeline import DatasetBuilder
    from embedding_pipeline import EmbeddingService, EmbeddingRecord, EmbeddingIndex
    from graph_builder import KnowledgeGraphBuilder


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full GraphRAG ingestion pipeline")
    parser.add_argument("input_dir", type=Path, help="Directory containing source .txt files")
    parser.add_argument("--artifacts", type=Path, default=Path("./artifacts"), help="Directory to store outputs")
    parser.add_argument("--dataset-name", type=str, default="chunks.jsonl")
    parser.add_argument("--embeddings-name", type=str, default="embeddings.npz")
    parser.add_argument("--graph-name", type=str, default="graph.graphml")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--min-token-length", type=int, default=3)
    parser.add_argument("--max-keywords", type=int, default=12)
    parser.add_argument("--weight-threshold", type=int, default=2)
    parser.add_argument("--device", type=str, default=None, help="Optional device override for embedding model")
    return parser


def run_pipeline(args: argparse.Namespace) -> None:
    artifacts_dir = args.artifacts
    config = GraphRAGConfig(
        data_dir=artifacts_dir,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_token_length=args.min_token_length,
        max_keywords=args.max_keywords,
        device=args.device,
    )
    config.resolve_paths()

    dataset_path = artifacts_dir / args.dataset_name
    embeddings_path = artifacts_dir / args.embeddings_name
    graph_path = artifacts_dir / args.graph_name

    builder = DatasetBuilder(config.chunk_size, config.chunk_overlap)
    documents = builder.load_documents(args.input_dir)
    if not documents:
        raise SystemExit(f"No documents found in {args.input_dir}")
    chunks = builder.build_chunks(documents)
    builder.save_dataset(chunks, dataset_path)

    embedder = EmbeddingService(config.embedding_model, device=config.device)
    embeddings_matrix = embedder.embed([chunk.text for chunk in chunks])
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    np.savez_compressed(embeddings_path, embeddings=embeddings_matrix, chunk_ids=np.array(chunk_ids))

    graph_builder = KnowledgeGraphBuilder(
        stop_words=config.stop_words,
        min_token_length=config.min_token_length,
        max_keywords=config.max_keywords,
    )
    graph = graph_builder.build(chunks, weight_threshold=args.weight_threshold)
    import networkx as nx  # Local import to keep dependency optional for other modules

    nx.write_graphml(graph, graph_path)

    index = EmbeddingIndex()
    index.add(EmbeddingRecord(chunk_id, embeddings_matrix[idx]) for idx, chunk_id in enumerate(chunk_ids))

    print("Ingestion complete")
    print(f"  Documents: {len(documents)}")
    print(f"  Chunks: {len(chunks)} -> {dataset_path}")
    print(f"  Embeddings: {embeddings_matrix.shape} -> {embeddings_path}")
    print(f"  Graph nodes: {graph.number_of_nodes()} edges: {graph.number_of_edges()} -> {graph_path}")


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":  # pragma: no cover - CLI execution path
    main()
