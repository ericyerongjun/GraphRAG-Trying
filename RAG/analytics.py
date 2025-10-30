"""Analytics helpers for large GraphRAG corpora."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import networkx as nx
import pandas as pd

from data_pipeline import Chunk


@dataclass(slots=True)
class CorpusStats:
    document_count: int
    chunk_count: int
    average_chunk_length: float
    vocabulary_size: int
    top_terms: List[tuple[str, int]]


def corpus_statistics(chunks: Sequence[Chunk], top_terms: int = 20) -> CorpusStats:
    doc_ids = {chunk.doc_id for chunk in chunks}
    lengths = [len(chunk.text.split()) for chunk in chunks]
    term_counter: Counter[str] = Counter()
    for chunk in chunks:
        tokens = [token.lower() for token in chunk.text.split() if token.isalpha()]
        term_counter.update(tokens)
    return CorpusStats(
        document_count=len(doc_ids),
        chunk_count=len(chunks),
        average_chunk_length=sum(lengths) / len(lengths) if lengths else 0.0,
        vocabulary_size=len(term_counter),
        top_terms=term_counter.most_common(top_terms),
    )


def graph_statistics(graph: nx.Graph) -> Dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "density": 0.0,
            "average_degree": 0.0,
            "average_clustering": 0.0,
        }
    degrees = [degree for _, degree in graph.degree()]
    return {
        "nodes": float(graph.number_of_nodes()),
        "edges": float(graph.number_of_edges()),
        "density": float(nx.density(graph)),
        "average_degree": float(sum(degrees) / len(degrees)),
        "average_clustering": float(nx.average_clustering(graph, weight="weight")),
    }


def export_degree_distribution(graph: nx.Graph, output: Path) -> None:
    frame = pd.DataFrame(
        [(node, degree) for node, degree in graph.degree()],
        columns=["node", "degree"],
    )
    frame.to_csv(output, index=False)