"""Command line entry point for GraphRAG question answering."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

try:  # Support running inside package or as standalone script
    from .config import GraphRAGConfig
    from .GraphRAG import GraphRAG
    from .qa_pipeline import (
        AnswerGenerator,
        AnthropicClient,
        GeminiClient,
        HuggingFaceClient,
        OpenAIClient,
    )
    from .reranker import CrossEncoderReranker
    from .retrieval import GraphRetriever
except ImportError:  # pragma: no cover - fallback for direct execution
    from config import GraphRAGConfig
    from GraphRAG import GraphRAG
    from qa_pipeline import (
        AnswerGenerator,
        AnthropicClient,
        GeminiClient,
        HuggingFaceClient,
        OpenAIClient,
    )
    from reranker import CrossEncoderReranker
    from retrieval import GraphRetriever


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask a question using GraphRAG with configurable LLMs")
    parser.add_argument("input_dir", type=Path, help="Directory containing raw text files")
    parser.add_argument("question", type=str, help="Question to answer")
    parser.add_argument("--data-dir", type=Path, default=Path("./artifacts"))
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--embedding-backend",
        type=str,
        default="sentence-transformers",
        choices=["sentence-transformers", "huggingface"],
        help="Embedding model backend",
    )
    parser.add_argument("--huggingface-token", type=str, default=None, help="Hugging Face token for private models")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow Hugging Face models to execute remote code",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--neighbor-k", type=int, default=3)
    parser.add_argument("--device", type=str, default=None, help="Embedding/reranker device override")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "huggingface", "gemini", "anthropic"],
        help="LLM provider",
    )
    parser.add_argument("--model", type=str, default=None, help="LLM model identifier")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--api-key", type=str, default=None, help="API key for OpenAI, Gemini, or Anthropic")
    parser.add_argument("--hf-llm-token", type=str, default=None, help="Hugging Face token for LLMs")
    parser.add_argument("--hf-llm-device", type=str, default=None, help="Device override for Hugging Face LLMs")
    parser.add_argument("--hf-llm-max-new-tokens", type=int, default=512)
    parser.add_argument("--hf-llm-trust-remote-code", action="store_true")
    parser.add_argument("--gemini-max-output-tokens", type=int, default=1024)
    parser.add_argument("--anthropic-max-tokens", type=int, default=1024)
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable cross-encoder reranking",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        help="Reranker checkpoint",
    )
    parser.add_argument(
        "--reranker-backend",
        type=str,
        default="sentence-transformers",
        choices=["sentence-transformers", "huggingface"],
    )
    parser.add_argument("--reranker-max-length", type=int, default=512)
    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    embedding_backend = args.embedding_backend
    hf_token = args.huggingface_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    config = GraphRAGConfig(
        data_dir=args.data_dir,
        embedding_model=args.embedding_model,
        embedding_backend=embedding_backend,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        device=args.device,
        huggingface_token=hf_token,
        huggingface_trust_remote_code=args.trust_remote_code,
        reranker_model=args.reranker_model if args.use_reranker else None,
        reranker_backend=args.reranker_backend,
    )
    rag = GraphRAG(config)
    rag.build(args.input_dir)
    reranker = None
    if args.use_reranker:
        reranker_token = hf_token if args.reranker_backend == "huggingface" else None
        reranker = CrossEncoderReranker(
            args.reranker_model,
            device=args.device,
            backend=args.reranker_backend,
            hf_token=reranker_token,
            trust_remote_code=args.trust_remote_code,
            max_length=args.reranker_max_length,
        )
    retriever = GraphRetriever(rag, reranker=reranker)
    provider = args.provider.lower()
    model_name = args.model
    if model_name is None:
        if provider == "openai":
            model_name = "gpt-4o-mini"
        elif provider == "huggingface":
            model_name = "meta-llama/Llama-3.1-8B-Instruct"
        elif provider == "gemini":
            model_name = "gemini-1.5-flash"
        else:
            model_name = "claude-3-5-sonnet-20240620"
    if provider == "openai":
        llm = OpenAIClient(model=model_name, temperature=args.temperature, api_key=args.api_key)
    elif provider == "huggingface":
        token = args.hf_llm_token or hf_token
        llm = HuggingFaceClient(
            model=model_name,
            temperature=args.temperature,
            device=args.hf_llm_device or args.device,
            token=token,
            max_new_tokens=args.hf_llm_max_new_tokens,
            trust_remote_code=args.hf_llm_trust_remote_code or args.trust_remote_code,
        )
    elif provider == "gemini":
        llm = GeminiClient(
            model=model_name,
            temperature=args.temperature,
            api_key=args.api_key,
            max_output_tokens=args.gemini_max_output_tokens,
        )
    else:
        llm = AnthropicClient(
            model=model_name,
            temperature=args.temperature,
            api_key=args.api_key,
            max_tokens=args.anthropic_max_tokens,
        )
    generator = AnswerGenerator(retriever, llm)
    response = generator.answer(args.question, top_k=args.top_k, neighbor_k=args.neighbor_k)
    print("Answer:\n" + response.answer)


if __name__ == "__main__":  # pragma: no cover - CLI execution path
    main()
