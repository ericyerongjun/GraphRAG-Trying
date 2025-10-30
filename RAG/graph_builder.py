"""Knowledge graph construction utilities."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:  # Support relative imports when executed as module
    from .config import GraphRAGConfig
    from .data_pipeline import Chunk, DatasetBuilder, Document
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import GraphRAGConfig
    from data_pipeline import Chunk, DatasetBuilder, Document

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError("networkx is required for graph construction. Install with 'pip install networkx'.") from exc


class KnowledgeGraphBuilder:
    """Builds a lightweight graph linking related chunks by shared keywords."""

    def __init__(self, stop_words: Sequence[str], min_token_length: int, max_keywords: int) -> None:
        self.stop_words = {word.lower() for word in stop_words}
        self.min_token_length = min_token_length
        self.max_keywords = max_keywords

    def _tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        token = []
        for char in text:
            if char.isalnum():
                token.append(char.lower())
            else:
                if token:
                    tokens.append("".join(token))
                    token.clear()
        if token:
            tokens.append("".join(token))
        filtered = [tok for tok in tokens if len(tok) >= self.min_token_length and tok not in self.stop_words]
        return filtered

    def _top_keywords(self, text: str) -> List[str]:
        counter = Counter(self._tokenize(text))
        return [word for word, _ in counter.most_common(self.max_keywords)]

    def build(self, chunks: Iterable[Chunk], weight_threshold: int = 2) -> nx.Graph:
        graph = nx.Graph()
        keyword_to_chunks: Dict[str, List[str]] = defaultdict(list)
        for chunk in chunks:
            graph.add_node(chunk.chunk_id, doc_id=chunk.doc_id, text=chunk.text)
            keywords = self._top_keywords(chunk.text)
            for keyword in keywords:
                keyword_to_chunks[keyword].append(chunk.chunk_id)
        for keyword, chunk_ids in keyword_to_chunks.items():
            for i, source in enumerate(chunk_ids):
                for target in chunk_ids[i + 1 :]:
                    current = graph.get_edge_data(source, target, default={}).get("weight", 0)
                    weight = current + 1
                    # Accumulate evidence so heavily related chunks surface sooner.
                    graph.add_edge(source, target, weight=weight)
                    if weight >= weight_threshold:
                        graph[source][target]["keyword"] = keyword
        return graph

    def neighbors_with_weights(self, graph: nx.Graph, chunk_id: str, top_k: int) -> List[Tuple[str, float]]:
        neighbors: List[Tuple[str, float]] = []
        for neighbor in graph.neighbors(chunk_id):
            edge_data = graph.get_edge_data(chunk_id, neighbor, default={})
            weight = float(edge_data.get("weight", 0.0))
            neighbors.append((neighbor, weight))
        neighbors.sort(key=lambda item: item[1], reverse=True)
        return neighbors[:top_k]


def generate_knowledge_graph(
    input_dir: Path,
    config: GraphRAGConfig,
    patterns: Sequence[str] = ("*.txt",),
    weight_threshold: int = 2,
) -> nx.Graph:
    """Create a knowledge graph directly from documents stored in ``input_dir``.

    Parameters
    ----------
    input_dir:
        Directory that contains the uploaded source documents.
    config:
        GraphRAG configuration controlling chunking and keyword extraction.
    patterns:
        Glob patterns to filter which files to ingest (defaults to ``*.txt``).
    weight_threshold:
        Minimum edge weight before tagging the edge with the shared keyword.

    Returns
    -------
    networkx.Graph
        An undirected graph where nodes represent chunks and edges capture
        strong keyword overlap between chunks.
    """

    dataset_builder = DatasetBuilder(config.chunk_size, config.chunk_overlap)
    documents: List[Document] = dataset_builder.load_documents(input_dir, patterns=patterns)
    if not documents:
        raise ValueError(f"No documents found for patterns {patterns} in {input_dir}")
    chunks = dataset_builder.build_chunks(documents)
    if not chunks:
        raise ValueError("Chunking produced no results; consider adjusting chunk parameters")
    graph_builder = KnowledgeGraphBuilder(
        stop_words=config.stop_words,
        min_token_length=config.min_token_length,
        max_keywords=config.max_keywords,
    )
    return graph_builder.build(chunks, weight_threshold=weight_threshold)
