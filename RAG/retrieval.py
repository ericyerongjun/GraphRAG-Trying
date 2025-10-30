"""High-level retrieval helpers built on top of the GraphRAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

try:  # Support module or script imports
    from .GraphRAG import GraphRAG, QueryResult
except ImportError:  # pragma: no cover - fallback for direct execution
    from GraphRAG import GraphRAG, QueryResult


@dataclass(slots=True)
class RetrievedContext:
    """Container for GraphRAG search output and stitched context."""

    query: str
    result: QueryResult
    context: str


class GraphRetriever:
    """Runs GraphRAG queries and prepares contexts for downstream LLMs."""

    def __init__(self, rag: GraphRAG) -> None:
        self.rag = rag

    def search(self, question: str, top_k: int = 5, neighbor_k: int = 3) -> List[RetrievedContext]:
        payload = self.rag.query(question, top_k=top_k, neighbor_k=neighbor_k)
        results: Iterable[QueryResult] = payload["results"]
        collected: List[RetrievedContext] = []
        for result in results:
            context = self.rag.build_context(result)
            collected.append(RetrievedContext(query=question, result=result, context=context))
        return collected

    def best_context(self, question: str, top_k: int = 5, neighbor_k: int = 3) -> RetrievedContext | None:
        contexts = self.search(question, top_k=top_k, neighbor_k=neighbor_k)
        if not contexts:
            return None
        return contexts[0]

    def batch_search(self, questions: Sequence[str], top_k: int = 5, neighbor_k: int = 3) -> List[RetrievedContext]:
        results: List[RetrievedContext] = []
        for question in questions:
            results.extend(self.search(question, top_k=top_k, neighbor_k=neighbor_k))
        return results
