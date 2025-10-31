"""Analytics helpers for large GraphRAG corpora."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List, Sequence

import networkx as nx
import pandas as pd

from data_pipeline import Chunk


@dataclass(slots=True)
class CorpusStats:
    document_count: int
    chunk_count: int
    average_chunk_length: float
    median_chunk_length: float
    chunk_length_std: float
    vocabulary_size: int
    type_token_ratio: float
    average_chunks_per_document: float
    top_terms: List[tuple[str, int]]


def corpus_statistics(chunks: Sequence[Chunk], top_terms: int = 20) -> CorpusStats:
    doc_ids = {chunk.doc_id for chunk in chunks}
    lengths = [len(chunk.text.split()) for chunk in chunks]
    term_counter: Counter[str] = Counter()
    total_tokens = 0
    for chunk in chunks:
        tokens = [token.lower() for token in chunk.text.split() if token.isalpha()]
        total_tokens += len(tokens)
        term_counter.update(tokens)
    avg_length = mean(lengths) if lengths else 0.0
    median_length = median(lengths) if lengths else 0.0
    std_length = pstdev(lengths) if len(lengths) > 1 else 0.0
    ttr = (len(term_counter) / total_tokens) if total_tokens else 0.0
    avg_chunks_per_doc = (len(chunks) / len(doc_ids)) if doc_ids else 0.0
    return CorpusStats(
        document_count=len(doc_ids),
        chunk_count=len(chunks),
        average_chunk_length=avg_length,
        median_chunk_length=median_length,
        chunk_length_std=std_length,
        vocabulary_size=len(term_counter),
        type_token_ratio=ttr,
        average_chunks_per_document=avg_chunks_per_doc,
        top_terms=term_counter.most_common(top_terms),
    )


@dataclass(slots=True)
class DocumentStats:
    doc_id: str
    chunk_count: int
    total_tokens: int
    average_chunk_tokens: float


def document_statistics(chunks: Sequence[Chunk]) -> List[DocumentStats]:
    doc_map: Dict[str, List[Chunk]] = defaultdict(list)
    for chunk in chunks:
        doc_map[chunk.doc_id].append(chunk)
    summary: List[DocumentStats] = []
    for doc_id, doc_chunks in doc_map.items():
        token_counts = [len(chunk.text.split()) for chunk in doc_chunks]
        total_tokens = sum(token_counts)
        avg_tokens = mean(token_counts) if token_counts else 0.0
        summary.append(
            DocumentStats(
                doc_id=doc_id,
                chunk_count=len(doc_chunks),
                total_tokens=total_tokens,
                average_chunk_tokens=avg_tokens,
            )
        )
    summary.sort(key=lambda item: (item.chunk_count, item.total_tokens), reverse=True)
    return summary


def graph_statistics(graph: nx.Graph, centrality_top_k: int = 5) -> Dict[str, object]:
    if graph.number_of_nodes() == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "density": 0.0,
            "average_degree": 0.0,
            "average_clustering": 0.0,
            "component_count": 0,
            "largest_component_size": 0,
            "isolated_nodes": 0,
            "avg_shortest_path_lcc": None,
            "top_degree_nodes": [],
            "top_degree_centrality": [],
            "top_betweenness_centrality": [],
        }
    degrees = [degree for _, degree in graph.degree()]
    components = list(nx.connected_components(graph))
    largest_component_size = max((len(component) for component in components), default=0)
    isolated_nodes = sum(1 for degree in degrees if degree == 0)
    avg_shortest_path = None
    if largest_component_size > 1:
        largest_component = max(components, key=len)
        lcc = graph.subgraph(largest_component)
        if lcc.number_of_nodes() <= 500:
            avg_shortest_path = float(nx.average_shortest_path_length(lcc))
    top_degree_nodes = [
        {"node": node, "degree": int(degree)}
        for node, degree in sorted(graph.degree(), key=lambda item: item[1], reverse=True)[:centrality_top_k]
    ]
    degree_centrality: List[Dict[str, float]] = []
    betweenness_centrality: List[Dict[str, float]] = []
    if graph.number_of_nodes() <= 500:
        deg_cent = nx.degree_centrality(graph)
        degree_centrality = [
            {"node": node, "centrality": float(value)}
            for node, value in sorted(deg_cent.items(), key=lambda item: item[1], reverse=True)[:centrality_top_k]
        ]
        bet_cent = nx.betweenness_centrality(graph, weight="weight")
        betweenness_centrality = [
            {"node": node, "centrality": float(value)}
            for node, value in sorted(bet_cent.items(), key=lambda item: item[1], reverse=True)[:centrality_top_k]
        ]
    return {
        "nodes": float(graph.number_of_nodes()),
        "edges": float(graph.number_of_edges()),
        "density": float(nx.density(graph)),
        "average_degree": float(sum(degrees) / len(degrees)),
        "average_clustering": float(nx.average_clustering(graph, weight="weight")),
        "component_count": len(components),
        "largest_component_size": largest_component_size,
        "isolated_nodes": isolated_nodes,
        "avg_shortest_path_lcc": avg_shortest_path,
        "top_degree_nodes": top_degree_nodes,
        "top_degree_centrality": degree_centrality,
        "top_betweenness_centrality": betweenness_centrality,
    }


def export_degree_distribution(graph: nx.Graph, output: Path) -> None:
    frame = pd.DataFrame(
        [(node, degree) for node, degree in graph.degree()],
        columns=["node", "degree"],
    )
    frame.to_csv(output, index=False)