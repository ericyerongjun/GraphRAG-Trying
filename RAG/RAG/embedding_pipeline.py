"""Embedding helpers for GraphRAG."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(slots=True)
class EmbeddingRecord:
    chunk_id: str
    vector: np.ndarray


def _resolve_torch_device(device: str | None, torch_module):
    if device:
        return torch_module.device(device)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


class EmbeddingService:
    """Wraps model inference for text embeddings."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        *,
        backend: str = "sentence-transformers",
        hf_token: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.backend = backend.lower()
        self.model_name = model_name
        self._hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self._trust_remote_code = trust_remote_code
        self._torch = None
        self._torch_device = None

        if self.backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - handled at runtime
                raise RuntimeError(
                    "sentence-transformers is required for the 'sentence-transformers' embedding backend."
                    " Install with 'pip install sentence-transformers'."
                ) from exc
            self.model = SentenceTransformer(model_name, device=device)
        elif self.backend in {"huggingface", "transformers", "hf"}:
            try:
                import torch  # type: ignore
                from transformers import AutoModel, AutoTokenizer  # type: ignore
            except ImportError as exc:  # pragma: no cover - surface dependency issues
                raise RuntimeError(
                    "transformers is required for the 'huggingface' embedding backend."
                    " Install with 'pip install transformers'."
                ) from exc
            auth_token = self._hf_token
            self._torch = torch
            self._torch_device = _resolve_torch_device(device, torch)
            tokenizer_kwargs: dict[str, str] = {}
            if auth_token:
                tokenizer_kwargs["token"] = auth_token
                tokenizer_kwargs["use_auth_token"] = auth_token
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                **tokenizer_kwargs,
            )
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            model_kwargs: dict[str, str] = {}
            if auth_token:
                model_kwargs["token"] = auth_token
                model_kwargs["use_auth_token"] = auth_token
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
            model.to(self._torch_device)
            model.eval()
            self.tokenizer = tokenizer
            self.model = model
        else:
            raise ValueError(f"Unsupported embedding backend: {backend}")

    def embed(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if self.backend == "sentence-transformers":
            embeddings = self.model.encode(list(texts), convert_to_numpy=True)
        elif self.backend in {"huggingface", "transformers", "hf"}:
            embeddings = self._embed_with_transformers(texts)
        else:  # pragma: no cover - guarded in __init__
            raise RuntimeError(f"Unsupported embedding backend: {self.backend}")

        if normalize and embeddings.size:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            embeddings = embeddings / norms
        return embeddings.astype(np.float32)

    def _embed_with_transformers(self, texts: Sequence[str]) -> np.ndarray:
        if self._torch is None or self._torch_device is None:
            raise RuntimeError("Hugging Face backend not initialised correctly")
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {key: value.to(self._torch_device) for key, value in batch.items()}
        with self._torch.inference_mode():
            outputs = self.model(**batch)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            last_hidden = outputs.last_hidden_state
            attention_mask = batch.get("attention_mask")
            if attention_mask is None:
                embeddings = last_hidden.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1.0)
                embeddings = summed / counts
        else:  # pragma: no cover - safety net for unusual models
            raise RuntimeError("Transformer model does not provide pooler_output or last_hidden_state")
        return embeddings.detach().cpu().numpy()


class EmbeddingIndex:
    """Stores embeddings in memory and runs cosine similarity search."""

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._ids: List[str] = []

    def add(self, records: Iterable[EmbeddingRecord]) -> None:
        vectors: List[np.ndarray] = []
        ids: List[str] = []
        for record in records:
            if not isinstance(record.vector, np.ndarray):
                raise TypeError("Embedding vector must be a numpy.ndarray")
            vectors.append(record.vector.astype(np.float32))
            ids.append(record.chunk_id)
        if not vectors:
            return
        matrix = np.vstack(vectors)
        if self._vectors is None:
            self._vectors = matrix
            self._ids = ids
        else:
            self._vectors = np.vstack([self._vectors, matrix])
            self._ids.extend(ids)

    def is_ready(self) -> bool:
        return self._vectors is not None and len(self._ids) > 0

    def search(self, query_vector: np.ndarray, top_k: int) -> List[tuple[str, float]]:
        if self._vectors is None:
            raise RuntimeError("Embedding index is empty")
        if query_vector.ndim != 1:
            raise ValueError("query_vector must be a 1-D numpy array")
        query_norm = np.linalg.norm(query_vector)
        if math.isclose(query_norm, 0.0):
            raise ValueError("query_vector norm is zero")
        normalized_query = query_vector / query_norm
        scores = self._vectors @ normalized_query
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self._ids[idx], float(scores[idx])) for idx in top_indices]
