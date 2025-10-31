"""End-to-end ingestion pipeline for GraphRAG assets."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

try:  # Support package-relative imports and direct execution
    from .GraphRAG import GraphRAG
    from .analytics import corpus_statistics, graph_statistics
    from .config import GraphRAGConfig
except ImportError:  # pragma: no cover - fallback for script execution
    from GraphRAG import GraphRAG
    from analytics import corpus_statistics, graph_statistics
    from config import GraphRAGConfig


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full GraphRAG ingestion pipeline")
    parser.add_argument("input_dir", type=Path, help="Directory containing source .txt files")
    parser.add_argument("--artifacts", type=Path, default=Path("./artifacts"), help="Directory to store outputs")
    parser.add_argument("--namespace", type=str, default="default", help="Artifact namespace for storage")
    parser.add_argument("--incremental", action="store_true", help="Append to existing namespace if present")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--min-token-length", type=int, default=3)
    parser.add_argument("--max-keywords", type=int, default=12)
    parser.add_argument("--weight-threshold", type=int, default=2)
    parser.add_argument("--device", type=str, default=None, help="Optional device override for embedding model")
    parser.add_argument(
        "--embedding-backend",
        type=str,
        default="sentence-transformers",
        choices=["sentence-transformers", "huggingface"],
        help="Embedding model backend",
    )
    parser.add_argument("--huggingface-token", type=str, default=None, help="Hugging Face token for private models")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow Hugging Face models to execute remote code")
    parser.add_argument("--index-backend", type=str, default="memory", choices=["memory", "faiss"], help="Vector index backend")
    parser.add_argument("--processes", type=int, default=1, help="Number of worker processes for chunking")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for embedding calls")
    return parser


def run_pipeline(args: argparse.Namespace) -> None:
    artifacts_dir = args.artifacts
    hf_token = args.huggingface_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    config = GraphRAGConfig(
        data_dir=artifacts_dir,
        embedding_model=args.embedding_model,
        embedding_backend=args.embedding_backend,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_token_length=args.min_token_length,
        max_keywords=args.max_keywords,
        device=args.device,
        index_backend=args.index_backend,
        artifacts_namespace=args.namespace,
        processes=args.processes,
        batch_size=args.batch_size,
        huggingface_token=hf_token,
        huggingface_trust_remote_code=args.trust_remote_code,
    )
    config.resolve_paths()
    rag = GraphRAG(config)
    rag.build(args.input_dir, incremental=args.incremental)
    chunks = list(rag.chunks.values())
    stats = corpus_statistics(chunks)
    graph_stats = graph_statistics(rag.graph) if rag.graph is not None else {}

    print("Ingestion complete")
    print(f"  Namespace: {args.namespace}")
    print(f"  Documents: {stats.document_count}")
    print(f"  Chunks: {stats.chunk_count}")
    print(f"  Avg chunk length: {stats.average_chunk_length:.2f} tokens")
    if graph_stats:
        print(f"  Graph nodes: {int(graph_stats['nodes'])} edges: {int(graph_stats['edges'])}")
        print(f"  Graph density: {graph_stats['density']:.4f}")
    print("  Top terms:")
    for term, count in stats.top_terms[:10]:
        print(f"    {term}: {count}")


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":  # pragma: no cover - CLI execution path
    main()
