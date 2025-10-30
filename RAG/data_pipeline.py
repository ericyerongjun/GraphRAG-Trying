"""Utilities to turn raw documents into clean RAG-ready chunks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence


@dataclass(slots=True)
class Document:
    """Represents a loaded source document."""

    doc_id: str
    text: str
    source_path: Path


@dataclass(slots=True)
class Chunk:
    """Represents a chunked segment of a document."""

    chunk_id: str
    doc_id: str
    text: str
    local_index: int


class DatasetBuilder:
    """Loads raw text assets and prepares serialized datasets."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self, input_dir: Path, patterns: Sequence[str] = ("*.txt",)) -> List[Document]:
        documents: List[Document] = []
        for pattern in patterns:
            for path in sorted(input_dir.glob(pattern)):
                if path.is_file():
                    text = path.read_text(encoding="utf-8")
                    documents.append(Document(doc_id=path.stem, text=text, source_path=path))
        return documents

    def iter_chunks(self, doc: Document) -> Iterator[Chunk]:
        text = doc.text
        start = 0
        end = len(text)
        index = 0
        while start < end:
            chunk_text = text[start : start + self.chunk_size]
            if not chunk_text.strip():
                break
            chunk_id = f"{doc.doc_id}_{index}"
            yield Chunk(chunk_id=chunk_id, doc_id=doc.doc_id, text=chunk_text, local_index=index)
            start += self.chunk_size - self.chunk_overlap
            index += 1

    def build_chunks(self, documents: Iterable[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in documents:
            chunks.extend(list(self.iter_chunks(doc)))
        return chunks

    def save_dataset(self, chunks: Sequence[Chunk], output_path: Path) -> None:
        """Persist chunk metadata and text into JSON lines."""
        with output_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                entry = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "local_index": chunk.local_index,
                }
                f.write(json.dumps(entry, ensure_ascii=False))
                f.write("\n")

    def load_dataset(self, dataset_path: Path) -> List[Chunk]:
        chunks: List[Chunk] = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                payload = json.loads(line)
                chunks.append(
                    Chunk(
                        chunk_id=payload["chunk_id"],
                        doc_id=payload["doc_id"],
                        text=payload["text"],
                        local_index=int(payload["local_index"]),
                    )
                )
        return chunks
