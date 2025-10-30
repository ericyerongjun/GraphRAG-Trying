"""Command line entry point for GraphRAG question answering."""
from __future__ import annotations

import argparse
from pathlib import Path

try:  # Support running inside package or as standalone script
    from .config import GraphRAGConfig
    from .GraphRAG import GraphRAG
    from .qa_pipeline import AnswerGenerator, OpenAIClient
    from .retrieval import GraphRetriever
except ImportError:  # pragma: no cover - fallback for direct execution
    from config import GraphRAGConfig
    from GraphRAG import GraphRAG
    from qa_pipeline import AnswerGenerator, OpenAIClient
    from retrieval import GraphRetriever


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask a question using GraphRAG + OpenAI")
    parser.add_argument("input_dir", type=Path, help="Directory containing raw text files")
    parser.add_argument("question", type=str, help="Question to answer")
    parser.add_argument("--data-dir", type=Path, default=Path("./artifacts"))
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--neighbor-k", type=int, default=3)
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openai-temperature", type=float, default=0.2)
    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    config = GraphRAGConfig(
        data_dir=args.data_dir,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    rag = GraphRAG(config)
    rag.build(args.input_dir)
    retriever = GraphRetriever(rag)
    llm = OpenAIClient(model=args.openai_model, temperature=args.openai_temperature)
    generator = AnswerGenerator(retriever, llm)
    response = generator.answer(args.question, top_k=args.top_k, neighbor_k=args.neighbor_k)
    print("Answer:\n" + response.answer)


if __name__ == "__main__":  # pragma: no cover - CLI execution path
    main()
