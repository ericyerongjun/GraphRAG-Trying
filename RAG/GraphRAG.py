"""End-to-end orchestration for the GraphRAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:  # Support running as module or script
	from .config import GraphRAGConfig
	from .data_pipeline import Chunk, DatasetBuilder
	from .embedding_pipeline import EmbeddingIndex, EmbeddingRecord, EmbeddingService
	from .graph_builder import KnowledgeGraphBuilder
except ImportError:  # pragma: no cover - fallback for direct execution
	from config import GraphRAGConfig
	from data_pipeline import Chunk, DatasetBuilder
	from embedding_pipeline import EmbeddingIndex, EmbeddingRecord, EmbeddingService
	from graph_builder import KnowledgeGraphBuilder


@dataclass(slots=True)
class QueryResult:
	chunk_id: str
	score: float
	text: str
	neighbor_ids: List[str]


class GraphRAG:
	"""High-level interface that combines dataset, embeddings, and graph search."""

	def __init__(self, config: GraphRAGConfig) -> None:
		self.config = config
		self.config.resolve_paths()
		self.dataset_builder = DatasetBuilder(config.chunk_size, config.chunk_overlap)
		self.embedding_service = EmbeddingService(config.embedding_model, device=config.device)
		self.graph_builder = KnowledgeGraphBuilder(
			stop_words=config.stop_words,
			min_token_length=config.min_token_length,
			max_keywords=config.max_keywords,
		)
		self.index = EmbeddingIndex()
		self.chunks: Dict[str, Chunk] = {}
		self.graph = None

	def build(self, input_dir: Path, patterns: Sequence[str] = ("*.txt",)) -> None:
		documents = self.dataset_builder.load_documents(input_dir, patterns=patterns)
		if not documents:
			raise ValueError(f"No documents found in {input_dir}")
		chunks = self.dataset_builder.build_chunks(documents)
		if not chunks:
			raise ValueError("Chunking produced no results; adjust chunk size or overlap")
		dataset_path = self.config.data_dir / "chunks.jsonl"
		self.dataset_builder.save_dataset(chunks, dataset_path)
		embeddings = self.embedding_service.embed([chunk.text for chunk in chunks])
		records = [EmbeddingRecord(chunk.chunk_id, embeddings[idx]) for idx, chunk in enumerate(chunks)]
		self.index.add(records)
		self.graph = self.graph_builder.build(chunks)
		self.chunks = {chunk.chunk_id: chunk for chunk in chunks}

	def query(self, question: str, top_k: int | None = None, neighbor_k: int = 3) -> Dict[str, Iterable[QueryResult]]:
		if not self.index.is_ready():
			raise RuntimeError("Pipeline is not built yet; call build() first")
		top_k = top_k or self.config.max_keywords
		query_vector = self.embedding_service.embed([question])[0]
		similar = self.index.search(query_vector, top_k)
		if self.graph is None:
			raise RuntimeError("Knowledge graph is unavailable; ensure build() ran successfully")
		results: List[QueryResult] = []
		for chunk_id, score in similar:
			neighbors = self.graph_builder.neighbors_with_weights(self.graph, chunk_id, neighbor_k)
			neighbor_ids = [neighbor_id for neighbor_id, _ in neighbors]
			text = self.chunks[chunk_id].text
			results.append(QueryResult(chunk_id=chunk_id, score=score, text=text, neighbor_ids=neighbor_ids))
		return {"results": results}

	def build_context(self, result: QueryResult) -> str:
		parts = [self.chunks[result.chunk_id].text]
		for neighbor_id in result.neighbor_ids:
			if neighbor_id in self.chunks:
				parts.append(self.chunks[neighbor_id].text)
		return "\n".join(parts)


def _format_output(results: Dict[str, Iterable[QueryResult]]) -> str:
	lines: List[str] = []
	for idx, result in enumerate(results["results"], start=1):
		lines.append(f"Result {idx}: {result.chunk_id} (score={result.score:.4f})")
		for neighbor in result.neighbor_ids:
			lines.append(f"  neighbor: {neighbor}")
	return "\n".join(lines)


def main() -> None:
	import argparse

	parser = argparse.ArgumentParser(description="Run GraphRAG over a directory of text files")
	parser.add_argument("input_dir", type=Path, help="Directory with source .txt files")
	parser.add_argument("question", type=str, help="Natural language question to ask")
	parser.add_argument("--data-dir", type=Path, default=Path("./artifacts"), help="Directory to store intermediate datasets")
	parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
	parser.add_argument("--chunk-size", type=int, default=500)
	parser.add_argument("--chunk-overlap", type=int, default=100)
	parser.add_argument("--top-k", type=int, default=5)
	parser.add_argument("--neighbor-k", type=int, default=3)
	args = parser.parse_args()

	config = GraphRAGConfig(
		data_dir=args.data_dir,
		embedding_model=args.embedding_model,
		chunk_size=args.chunk_size,
		chunk_overlap=args.chunk_overlap,
	)
	rag = GraphRAG(config)
	rag.build(args.input_dir)
	results = rag.query(args.question, top_k=args.top_k, neighbor_k=args.neighbor_k)
	print(_format_output(results))


if __name__ == "__main__":  # pragma: no cover - CLI execution path
	main()
