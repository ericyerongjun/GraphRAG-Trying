"""End-to-end orchestration for the GraphRAG pipeline."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import os
import numpy as np

try:  # Support running as module or script
	from .config import GraphRAGConfig
	from .data_pipeline import Chunk, DatasetBuilder, Document
	from .embedding_pipeline import EmbeddingIndex, EmbeddingRecord, EmbeddingService
	from .graph_builder import KnowledgeGraphBuilder
	from .mindmap import MindMapBuilder
	from .storage import DatasetPaths, DatasetStorage
	try:
		from .faiss_index import FaissIndex
	except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
		FaissIndex = None  # type: ignore[misc,assignment]
except ImportError:  # pragma: no cover - fallback for direct execution
	from config import GraphRAGConfig
	from data_pipeline import Chunk, DatasetBuilder, Document
	from embedding_pipeline import EmbeddingIndex, EmbeddingRecord, EmbeddingService
	from graph_builder import KnowledgeGraphBuilder
	from mindmap import MindMapBuilder
	from storage import DatasetPaths, DatasetStorage
	try:
		from faiss_index import FaissIndex
	except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
		FaissIndex = None  # type: ignore[misc,assignment]


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
		self.embedding_service = EmbeddingService(
			config.embedding_model,
			device=config.device,
			backend=config.embedding_backend,
			hf_token=config.huggingface_token,
			trust_remote_code=config.huggingface_trust_remote_code,
		)
		self.graph_builder = KnowledgeGraphBuilder(
			stop_words=config.stop_words,
			min_token_length=config.min_token_length,
			max_keywords=config.max_keywords,
		)
		self.storage = DatasetStorage(self.config.data_dir / "datasets")
		self.namespace = self.config.artifacts_namespace
		self.index_backend = self.config.index_backend.lower()
		self.index: EmbeddingIndex | object | None = None
		self.index_ready = False
		self.chunks: Dict[str, Chunk] = {}
		self.graph = None

	def build(
		self,
		input_dir: Path,
		patterns: Sequence[str] = ("*.txt",),
		incremental: bool = False,
	) -> None:
		documents = self.dataset_builder.load_documents(input_dir, patterns=patterns)
		if not documents:
			raise ValueError(f"No documents found in {input_dir}")
		chunks = self._build_chunks(documents)
		if not chunks:
			raise ValueError("Chunking produced no results; adjust chunk size or overlap")
		paths = self.storage.paths_for(self.namespace)
		existing_chunks: List[Chunk] = []
		if incremental and paths.dataset_parquet.exists():
			existing_chunks = self.storage.load_chunks(paths)
			existing_ids = {chunk.chunk_id for chunk in existing_chunks}
			new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
			all_chunks = existing_chunks + new_chunks
		else:
			all_chunks = chunks
		self.storage.save_chunks(all_chunks, paths)
		embeddings = self.embedding_service.embed([chunk.text for chunk in all_chunks])
		chunk_ids = [chunk.chunk_id for chunk in all_chunks]
		self.storage.save_embeddings(embeddings, chunk_ids, paths)
		self.index = self._build_index(embeddings, all_chunks, persist=True)
		self.index_ready = True
		self.graph = self.graph_builder.build(all_chunks, weight_threshold=self.config.edge_weight_threshold)
		self.chunks = {chunk.chunk_id: chunk for chunk in all_chunks}
		if isinstance(self.index, FaissIndex):
			self.index.save()

	def query(self, question: str, top_k: int | None = None, neighbor_k: int = 3) -> Dict[str, Iterable[QueryResult]]:
		if not self.index_ready or self.index is None:
			raise RuntimeError("Pipeline is not built yet; call build() or load() first")
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


	def mindmap_builder(self) -> MindMapBuilder:
		if self.graph is None:
			raise RuntimeError("Knowledge graph is unavailable; ensure build() ran successfully")
		return MindMapBuilder(self.chunks, self.graph)

	def load(self, namespace: str | None = None) -> None:
		self.namespace = namespace or self.namespace
		paths = self.storage.paths_for(self.namespace)
		if not paths.dataset_parquet.exists():
			raise FileNotFoundError(f"No dataset found for namespace '{self.namespace}'")
		chunks = self.storage.load_chunks(paths)
		embeddings, _ = self.storage.load_embeddings(paths)
		self.index = self._build_index(embeddings, chunks, persist=False)
		self.index_ready = True
		self.graph = self.graph_builder.build(chunks)
		self.chunks = {chunk.chunk_id: chunk for chunk in chunks}

	def _build_chunks(self, documents: Sequence[Document]) -> List[Chunk]:
		if self.config.processes > 1:
			with ProcessPoolExecutor(max_workers=self.config.processes) as executor:
				args = ((doc, self.config.chunk_size, self.config.chunk_overlap) for doc in documents)
				batches = executor.map(_chunk_document, args)
				chunks = [chunk for batch in batches for chunk in batch]
		else:
			chunks = self.dataset_builder.build_chunks(documents)
		return chunks

	def _build_index(
		self,
		embeddings: np.ndarray,
		chunks: Sequence[Chunk],
		*,
		persist: bool,
	) -> EmbeddingIndex | object:
		if self.index_backend == "faiss":
			if FaissIndex is None:
				raise RuntimeError(
					"Faiss backend requested but the 'faiss' package is not installed."
				)
			index_dir = self.config.data_dir / "indices" / self.namespace
			index = FaissIndex(embeddings.shape[1], store_dir=index_dir)
			records = [EmbeddingRecord(chunk.chunk_id, embeddings[idx]) for idx, chunk in enumerate(chunks)]
			index.add(records)
			if persist:
				index.save()
			return index
		index = EmbeddingIndex()
		records = [EmbeddingRecord(chunk.chunk_id, embeddings[idx]) for idx, chunk in enumerate(chunks)]
		index.add(records)
		return index


def _format_output(results: Dict[str, Iterable[QueryResult]]) -> str:
	lines: List[str] = []
	for idx, result in enumerate(results["results"], start=1):
		lines.append(f"Result {idx}: {result.chunk_id} (score={result.score:.4f})")
		for neighbor in result.neighbor_ids:
			lines.append(f"  neighbor: {neighbor}")
	return "\n".join(lines)


def _chunk_document(payload: Tuple[Document, int, int]) -> List[Chunk]:
	document, chunk_size, chunk_overlap = payload
	builder = DatasetBuilder(chunk_size, chunk_overlap)
	return list(builder.iter_chunks(document))


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
	parser.add_argument("--namespace", type=str, default="default")
	parser.add_argument("--index-backend", type=str, default="memory", choices=["memory", "faiss"])
	parser.add_argument("--incremental", action="store_true")
	parser.add_argument("--processes", type=int, default=1)
	parser.add_argument("--batch-size", type=int, default=1024)
	parser.add_argument("--edge-weight-threshold", type=int, default=2)
	parser.add_argument("--embedding-backend", type=str, default="sentence-transformers", choices=["sentence-transformers", "huggingface"])
	parser.add_argument("--device", type=str, default=None)
	parser.add_argument("--huggingface-token", type=str, default=None)
	parser.add_argument("--trust-remote-code", action="store_true")
	args = parser.parse_args()

	config = GraphRAGConfig(
		data_dir=args.data_dir,
		embedding_model=args.embedding_model,
		embedding_backend=args.embedding_backend,
		chunk_size=args.chunk_size,
		chunk_overlap=args.chunk_overlap,
		artifacts_namespace=args.namespace,
		index_backend=args.index_backend,
		processes=args.processes,
		batch_size=args.batch_size,
		edge_weight_threshold=args.edge_weight_threshold,
		device=args.device,
		huggingface_token=args.huggingface_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
		huggingface_trust_remote_code=args.trust_remote_code,
	)
	rag = GraphRAG(config)
	rag.build(args.input_dir, incremental=args.incremental)
	results = rag.query(args.question, top_k=args.top_k, neighbor_k=args.neighbor_k)
	print(_format_output(results))


if __name__ == "__main__":  # pragma: no cover - CLI execution path
	main()
