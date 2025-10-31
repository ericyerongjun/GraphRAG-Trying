"""Question-answering pipeline built on GraphRAG retrieval."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:  # Support running inside package
    from .retrieval import GraphRetriever, RetrievedContext
except ImportError:  # pragma: no cover - fallback for scripts
    from retrieval import GraphRetriever, RetrievedContext


@dataclass(slots=True)
class LLMResponse:
    question: str
    answer: str
    used_context: str


class LLMClient:
    """Abstract interface for language model completion."""

    def complete(self, prompt: str) -> str:  # pragma: no cover - override in subclass
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """Thin wrapper around the OpenAI Chat Completions API."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2, api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - handled at runtime
            raise RuntimeError("openai package is required for OpenAIClient. Install with 'pip install openai'.") from exc
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("An OpenAI API key is required. Set OPENAI_API_KEY or provide one explicitly.")
        self._client = OpenAI(api_key=key)
        self.model = model
        self.temperature = temperature

    def complete(self, prompt: str) -> str:
        response = self._client.responses.create(
            model=self.model,
            input=prompt,
            temperature=self.temperature,
        )
        content = response.output_text
        return content.strip()


class HuggingFaceClient(LLMClient):
    """Runs local or remote Hugging Face causal language models."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        *,
        device: str | None = None,
        token: str | None = None,
        max_new_tokens: int = 512,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency not installed
            raise RuntimeError(
                "transformers is required for HuggingFaceClient. Install with 'pip install transformers'."
            ) from exc

        self._torch = torch
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        tokenizer_kwargs: dict[str, str] = {}
        if self._token:
            tokenizer_kwargs["token"] = self._token
            tokenizer_kwargs["use_auth_token"] = self._token
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            **tokenizer_kwargs,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs: dict[str, str] = {}
        if self._token:
            model_kwargs["token"] = self._token
            model_kwargs["use_auth_token"] = self._token
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
        self._device = self._resolve_device(device)
        self.model.to(self._device)
        self.model.eval()

    def _resolve_device(self, device: Optional[str]):
        if device:
            return self._torch.device(device)
        if self._torch.cuda.is_available():
            return self._torch.device("cuda")
        if hasattr(self._torch.backends, "mps") and self._torch.backends.mps.is_available():
            return self._torch.device("mps")
        return self._torch.device("cpu")

    def complete(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()


class GeminiClient(LLMClient):
    """Client for Google Gemini models."""

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.2,
        *,
        api_key: str | None = None,
        max_output_tokens: int = 1024,
    ) -> None:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency not installed
            raise RuntimeError(
                "google-generativeai is required for GeminiClient. Install with 'pip install google-generativeai'."
            ) from exc
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("A Gemini API key is required. Set GEMINI_API_KEY or provide one explicitly.")
        genai.configure(api_key=key)
        self._genai = genai
        self.model = genai.GenerativeModel(model)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def complete(self, prompt: str) -> str:
        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            },
        )
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        # Fallback: concatenate candidate parts
        candidates = getattr(response, "candidates", []) or []
        for candidate in candidates:
            parts = getattr(candidate, "content", None)
            if parts and getattr(parts, "parts", None):
                fragments = [getattr(part, "text", "") for part in parts.parts]
                text = "".join(fragments).strip()
                if text:
                    return text
        raise RuntimeError("Gemini response did not contain any text content")


class AnthropicClient(LLMClient):
    """Client for Anthropic Claude models."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20240620",
        temperature: float = 0.2,
        *,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency not installed
            raise RuntimeError("anthropic package is required for AnthropicClient. Install with 'pip install anthropic'.") from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("An Anthropic API key is required. Set ANTHROPIC_API_KEY or provide one explicitly.")
        self._client = Anthropic(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        for item in response.content:
            if getattr(item, "type", "") == "text" and getattr(item, "text", ""):
                return item.text.strip()
        raise RuntimeError("Anthropic response did not contain textual content")


class AnswerGenerator:
    """Combines retrieval output and an LLM to synthesize answers."""

    def __init__(self, retriever: GraphRetriever, llm: LLMClient, prompt_template: str | None = None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template or self._default_template()

    def _format_prompt(self, context: RetrievedContext, material: str | None = None) -> str:
        payload = material if material is not None else context.context
        return self.prompt_template.format(query=context.query, context=payload)

    def answer(self, question: str, top_k: int = 5, neighbor_k: int = 3) -> LLMResponse:
        context = self.retriever.best_context(question, top_k=top_k, neighbor_k=neighbor_k)
        if context is None:
            raise RuntimeError("No context found for the question")
        combined_context = context.context
        if context.mindmap:
            combined_context = f"Mind map\n{context.mindmap}\n\nChunk context\n{context.context}"
        prompt = self._format_prompt(context, combined_context)
        answer = self.llm.complete(prompt)
        return LLMResponse(question=question, answer=answer, used_context=combined_context)

    @staticmethod
    def _default_template() -> str:
        return (
            "You are a helpful assistant. Answer the question using only the provided context.\n"
            "Context:\n{context}\n\n"
            "Question: {query}\n"
            "Answer:"
        )
