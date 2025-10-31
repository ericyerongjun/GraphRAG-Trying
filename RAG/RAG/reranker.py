"""Optional rerankers to improve retrieval quality."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover - fallback when dependency missing
    CrossEncoder = None  # type: ignore


@dataclass(slots=True)
class RerankResult:
    chunk_id: str
    score: float


def _resolve_torch_device(device: str | None, torch_module):
    if device:
        return torch_module.device(device)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


class CrossEncoderReranker:
    """Uses a cross-encoder model to rerank candidate chunks."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        *,
        backend: str = "sentence-transformers",
        hf_token: str | None = None,
        trust_remote_code: bool = False,
        max_length: int = 512,
    ) -> None:
        self.backend = backend.lower()
        self.model_name = model_name
        self.max_length = max_length
        self._hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self._torch = None
        self._torch_device = None

        if self.backend == "sentence-transformers":
            if CrossEncoder is None:
                raise RuntimeError(
                    "sentence-transformers must be installed to use the cross-encoder reranker backend."
                )
            self.model = CrossEncoder(model_name, device=device)
        elif self.backend in {"huggingface", "transformers", "hf"}:
            try:
                import torch  # type: ignore
                from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
            except ImportError as exc:  # pragma: no cover - dependency not installed
                raise RuntimeError(
                    "transformers is required for the 'huggingface' reranker backend."
                    " Install with 'pip install transformers'."
                ) from exc
            self._torch = torch
            self._torch_device = _resolve_torch_device(device, torch)
            auth_token = self._hf_token
            tokenizer_kwargs: dict[str, str] = {}
            if auth_token:
                tokenizer_kwargs["token"] = auth_token
                tokenizer_kwargs["use_auth_token"] = auth_token
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                **tokenizer_kwargs,
            )
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            model_kwargs: dict[str, str] = {}
            if auth_token:
                model_kwargs["token"] = auth_token
                model_kwargs["use_auth_token"] = auth_token
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
            self.model.to(self._torch_device)
            self.model.eval()
        else:
            raise ValueError(f"Unsupported reranker backend: {backend}")

    def rerank(self, query: str, candidates: Sequence[Tuple[str, str]], top_k: int) -> List[RerankResult]:
        if not candidates:
            return []
        if self.backend == "sentence-transformers":
            pairs = [(query, text) for _, text in candidates]
            scores = self.model.predict(pairs)
            scores = np.asarray(scores)
        elif self.backend in {"huggingface", "transformers", "hf"}:
            scores = self._rerank_with_transformers(query, candidates)
        else:  # pragma: no cover - guarded in __init__
            raise RuntimeError(f"Unsupported reranker backend: {self.backend}")

        indices = np.argsort(scores)[::-1][:top_k]
        return [
            RerankResult(chunk_id=candidates[idx][0], score=float(scores[idx]))
            for idx in indices
        ]

    def _rerank_with_transformers(self, query: str, candidates: Sequence[Tuple[str, str]]) -> np.ndarray:
        if self._torch is None or self._torch_device is None:
            raise RuntimeError("Hugging Face reranker backend not initialised correctly")
        texts = [text for _, text in candidates]
        batch = self.tokenizer(
            [query] * len(texts),
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(self._torch_device) for key, value in batch.items()}
        with self._torch.inference_mode():
            outputs = self.model(**batch)
        logits = outputs.logits
        if logits.ndim == 2 and logits.shape[-1] > 1:
            scores = self._torch.softmax(logits, dim=-1)[:, -1]
        else:
            scores = logits.squeeze(-1)
        return scores.detach().cpu().numpy().astype(np.float32)