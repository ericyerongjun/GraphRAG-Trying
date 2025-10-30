"""Configuration objects for GraphRAG pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


@dataclass(slots=True)
class GraphRAGConfig:
    """Top level configuration options for the GraphRAG pipeline."""

    data_dir: Path
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 100
    min_token_length: int = 3
    max_keywords: int = 12
    device: Optional[str] = None
    stop_words: Sequence[str] = field(
        default_factory=lambda: (
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
        )
    )

    def resolve_paths(self) -> None:
        """Ensure important directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
