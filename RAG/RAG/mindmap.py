"""Mind map construction utilities for GraphRAG."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence

import networkx as nx

try:  # Support module or package imports
    from .data_pipeline import Chunk
except ImportError:  # pragma: no cover - fallback for direct execution
    from data_pipeline import Chunk


@dataclass(slots=True)
class MindMapNode:
    """Represents a node in the hierarchical mind map."""

    id: str
    label: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: list["MindMapNode"] = field(default_factory=list)


class MindMapBuilder:
    """Creates hierarchical views of the knowledge graph to aid navigation."""

    def __init__(self, chunks: Mapping[str, Chunk], graph: nx.Graph | None = None) -> None:
        self._chunks = chunks
        self._graph = graph if graph is not None else nx.Graph()
        self._stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "but",
            "by",
            "for",
            "if",
            "in",
            "into",
            "is",
            "it",
            "no",
            "not",
            "of",
            "on",
            "or",
            "such",
            "that",
            "the",
            "their",
            "then",
            "there",
            "these",
            "they",
            "this",
            "to",
            "was",
            "will",
            "with",
        }

    def build_full_map(
        self,
        doc_ids: Sequence[str] | None = None,
        max_chunks_per_doc: int = 5,
        neighbor_k: int = 3,
        use_communities: bool = True,
    ) -> MindMapNode:
        """Create a document-centric mind map of the entire knowledge graph."""

        root = MindMapNode(id="root", label="Knowledge Graph", metadata={})
        filtered_chunks = [chunk for chunk in self._chunks.values() if not doc_ids or chunk.doc_id in doc_ids]
        if not filtered_chunks:
            return root

        community_map = self._community_assignments() if use_communities else {}
        if community_map:
            grouped: Dict[int | None, Dict[str, list[Chunk]]] = defaultdict(lambda: defaultdict(list))
            for chunk in filtered_chunks:
                community_id = community_map.get(chunk.chunk_id)
                grouped[community_id][chunk.doc_id].append(chunk)
            keywords = self._cluster_keywords(grouped)
            for community_id, doc_map in sorted(grouped.items(), key=lambda item: (item[0] is None, item[0])):
                label = "Ungrouped" if community_id is None else f"Topic {community_id}"
                metadata = {"keywords": keywords.get(community_id, [])}
                topic_node = MindMapNode(
                    id=f"topic::{community_id}",
                    label=label,
                    metadata=metadata,
                    children=[],
                )
                for doc_id, doc_chunks in sorted(doc_map.items(), key=lambda item: item[0]):
                    topic_node.children.append(
                        self._document_node(doc_id, doc_chunks, max_chunks_per_doc, neighbor_k)
                    )
                root.children.append(topic_node)
        else:
            doc_map: Dict[str, list[Chunk]] = defaultdict(list)
            for chunk in filtered_chunks:
                doc_map[chunk.doc_id].append(chunk)
            for doc_id, doc_chunks in sorted(doc_map.items(), key=lambda item: item[0]):
                root.children.append(self._document_node(doc_id, doc_chunks, max_chunks_per_doc, neighbor_k))
        return root

    def focus_on_chunk(self, chunk_id: str, neighbor_k: int = 3) -> MindMapNode:
        """Create a small mind map focused on a specific chunk and its neighbors."""

        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            raise ValueError(f"Unknown chunk_id: {chunk_id}")
        doc_chunks = [c for c in self._chunks.values() if c.doc_id == chunk.doc_id]
        doc_node = self._document_node(chunk.doc_id, doc_chunks, max_chunks_per_doc=1, neighbor_k=neighbor_k)
        return doc_node

    def _chunk_node(self, chunk: Chunk, neighbor_k: int) -> MindMapNode:
        preview = self._preview_text(chunk.text)
        neighbors = self._top_neighbors(chunk.chunk_id, neighbor_k)
        metadata = {
            "chunk_index": chunk.local_index,
            "preview": preview,
            "neighbor_count": len(neighbors),
        }
        chunk_node = MindMapNode(
            id=chunk.chunk_id,
            label=f"Chunk {chunk.local_index}",
            metadata=metadata,
            children=[
                MindMapNode(
                    id=f"neighbor::{chunk.chunk_id}::{neighbor_id}",
                    label=f"Neighbor {idx + 1}: {neighbor_id}",
                    metadata={
                        "weight": weight,
                        "doc_id": self._chunks.get(neighbor_id).doc_id if neighbor_id in self._chunks else None,
                        "preview": self._preview_text(self._chunks[neighbor_id].text)
                        if neighbor_id in self._chunks
                        else None,
                    },
                )
                for idx, (neighbor_id, weight) in enumerate(neighbors)
            ],
        )
        return chunk_node

    def _top_neighbors(self, chunk_id: str, neighbor_k: int) -> list[tuple[str, float]]:
        if chunk_id not in self._graph:
            return []
        neighbors = []
        for neighbor_id, edge_data in self._graph[chunk_id].items():
            weight = float(edge_data.get("weight", 0.0))
            neighbors.append((neighbor_id, weight))
        neighbors.sort(key=lambda item: item[1], reverse=True)
        return neighbors[:neighbor_k]

    @staticmethod
    def _preview_text(text: str, limit: int = 180) -> str:
        sanitized = text.replace("\n", " ").strip()
        if len(sanitized) <= limit:
            return sanitized
        return sanitized[:limit].rstrip() + "…"

    def _community_assignments(self) -> Dict[str, int]:
        if self._graph.number_of_nodes() == 0:
            return {}
        try:
            communities = nx.algorithms.community.louvain_communities(self._graph, weight="weight")  # type: ignore[attr-defined]
        except AttributeError:
            communities = nx.algorithms.community.greedy_modularity_communities(self._graph, weight="weight")
        mapping: Dict[str, int] = {}
        for idx, community in enumerate(communities):
            for node in community:
                mapping[str(node)] = idx
        return mapping

    def _cluster_keywords(self, grouped: Mapping[int | None, Mapping[str, Sequence[Chunk]]], top_k: int = 8) -> Dict[int | None, list[str]]:
        keywords: Dict[int | None, list[str]] = {}
        for community_id, doc_map in grouped.items():
            counter: Counter[str] = Counter()
            for chunks in doc_map.values():
                for chunk in chunks:
                    counter.update(self._extract_terms(chunk.text))
            if counter:
                keywords[community_id] = [term for term, _ in counter.most_common(top_k)]
        return keywords

    def _extract_terms(self, text: str) -> Iterable[str]:
        for token in text.split():
            cleaned = "".join(ch for ch in token.lower() if ch.isalnum())
            if len(cleaned) < 3 or cleaned in self._stop_words:
                continue
            yield cleaned

    def _document_node(
        self,
        doc_id: str,
        doc_chunks: Sequence[Chunk],
        max_chunks_per_doc: int,
        neighbor_k: int,
    ) -> MindMapNode:
        node = MindMapNode(
            id=f"doc::{doc_id}",
            label=f"Document: {doc_id}",
            metadata={"chunk_count": len(doc_chunks)},
            children=[],
        )
        sorted_chunks = sorted(doc_chunks, key=lambda c: c.local_index)
        for chunk in sorted_chunks[:max_chunks_per_doc]:
            node.children.append(self._chunk_node(chunk, neighbor_k))
        if len(sorted_chunks) > max_chunks_per_doc:
            node.children.append(
                MindMapNode(
                    id=f"doc::{doc_id}::more",
                    label="… more chunks",
                    metadata={"omitted": len(sorted_chunks) - max_chunks_per_doc},
                )
            )
        return node

    def to_dict(self, node: MindMapNode) -> Dict[str, Any]:
        """Convert a mind map node into a serializable dictionary."""

        return {
            "id": node.id,
            "label": node.label,
            "metadata": node.metadata,
            "children": [self.to_dict(child) for child in node.children],
        }

    def to_markdown(self, node: MindMapNode, depth: int = 0) -> str:
        """Render the mind map into Markdown bullet points."""

        indent = "  " * depth
        metadata_snippets = []
        if "chunk_index" in node.metadata:
            metadata_snippets.append(f"index={node.metadata['chunk_index']}")
        if "neighbor_count" in node.metadata:
            metadata_snippets.append(f"neighbors={node.metadata['neighbor_count']}")
        if "weight" in node.metadata:
            metadata_snippets.append(f"weight={node.metadata['weight']:.2f}")
        meta = f" ({', '.join(metadata_snippets)})" if metadata_snippets else ""
        preview = node.metadata.get("preview")
        preview_text = f" — {preview}" if preview else ""
        line = f"{indent}- {node.label}{meta}{preview_text}".rstrip()
        child_lines = [self.to_markdown(child, depth + 1) for child in node.children]
        return "\n".join([line, *[child for child in child_lines if child]]) if child_lines else line

    def mindmap_text_for_chunk(self, chunk_id: str, neighbor_k: int = 3) -> str:
        """Convenience helper returning markdown text for a focused chunk map."""

        node = self.focus_on_chunk(chunk_id, neighbor_k)
        return self.to_markdown(node)

    def mindmap_text_for_docs(
        self,
        doc_ids: Sequence[str] | None = None,
        max_chunks_per_doc: int = 5,
        neighbor_k: int = 3,
    ) -> str:
        root = self.build_full_map(doc_ids=doc_ids, max_chunks_per_doc=max_chunks_per_doc, neighbor_k=neighbor_k)
        return self.to_markdown(root)


__all__ = ["MindMapBuilder", "MindMapNode"]
