"""High-level retrieval helpers built on top of the GraphRAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

try:  # Support module or script imports
    from .GraphRAG import GraphRAG, QueryResult
    from .mindmap import MindMapBuilder
    from .reranker import CrossEncoderReranker
except ImportError:  # pragma: no cover - fallback for direct execution
    from GraphRAG import GraphRAG, QueryResult
    from mindmap import MindMapBuilder
    from reranker import CrossEncoderReranker


@dataclass(slots=True)
class RetrievedContext:
    """Container for GraphRAG search output and stitched context."""

    query: str
    result: QueryResult
    context: str
    mindmap: str | None = None


class GraphRetriever:
    """Runs GraphRAG queries and prepares contexts for downstream LLMs."""

    def __init__(self, rag: GraphRAG, reranker: Optional[CrossEncoderReranker] = None) -> None:
        self.rag = rag
        self.reranker = reranker

    def search(self, question: str, top_k: int = 5, neighbor_k: int = 3) -> List[RetrievedContext]:
        payload = self.rag.query(question, top_k=top_k, neighbor_k=neighbor_k)
        results: Iterable[QueryResult] = payload["results"]
        reranked_results = self._maybe_rerank(question, results, top_k)
        mindmap_builder = MindMapBuilder(self.rag.chunks, self.rag.graph)
        collected: List[RetrievedContext] = []
        for result in reranked_results:
            context = self.rag.build_context(result)
            mindmap_text: str | None = None
            try:
                mindmap_text = mindmap_builder.mindmap_text_for_chunk(result.chunk_id, neighbor_k=neighbor_k)
            except Exception:  # pragma: no cover - best-effort mind map
                mindmap_text = None
            collected.append(
                RetrievedContext(
                    query=question,
                    result=result,
                    context=context,
                    mindmap=mindmap_text,
                )
            )
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

    def _maybe_rerank(
        self,
        question: str,
        results: Iterable[QueryResult],
        top_k: int,
    ) -> List[QueryResult]:
        results = list(results)
        if not self.reranker or not results:
            return results
        candidates = [(result.chunk_id, self.rag.chunks[result.chunk_id].text) for result in results]
        reranked = self.reranker.rerank(question, candidates, top_k)
        score_map = {item.chunk_id: item.score for item in reranked}
        results.sort(key=lambda res: score_map.get(res.chunk_id, res.score), reverse=True)
        return results
