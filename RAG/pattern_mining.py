"""Lightweight pattern mining utilities for data mining experiments."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from data_pipeline import Chunk


@dataclass(slots=True)
class TermPattern:
    term: str
    frequency: int
    document_frequency: int


def frequent_terms(chunks: Sequence[Chunk], min_doc_frequency: int = 5, top_k: int = 50) -> List[TermPattern]:
    term_counts: Counter[str] = Counter()
    doc_frequency: Dict[str, int] = defaultdict(int)
    for chunk in chunks:
        tokens = [token.lower() for token in chunk.text.split() if token.isalpha()]
        term_counts.update(tokens)
        for token in set(tokens):
            doc_frequency[token] += 1
    patterns: List[TermPattern] = []
    for term, frequency in term_counts.most_common():
        df = doc_frequency[term]
        if df < min_doc_frequency:
            continue
        patterns.append(TermPattern(term=term, frequency=frequency, document_frequency=df))
        if len(patterns) >= top_k:
            break
    return patterns


def co_occurrence(chunk: Chunk, window: int = 5) -> Dict[tuple[str, str], int]:
    tokens = [token.lower() for token in chunk.text.split() if token.isalpha()]
    counts: Dict[tuple[str, str], int] = defaultdict(int)
    for idx, token in enumerate(tokens):
        for other in tokens[idx + 1 : idx + 1 + window]:
            if token == other:
                continue
            pair = tuple(sorted((token, other)))
            counts[pair] += 1
    return counts