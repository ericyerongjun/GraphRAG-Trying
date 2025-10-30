"""Question-answering pipeline built on GraphRAG retrieval."""
from __future__ import annotations

import os
from dataclasses import dataclass

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

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - handled at runtime
            raise RuntimeError("openai package is required for OpenAIClient. Install with 'pip install openai'.") from exc
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        self._client = OpenAI(api_key=api_key)
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


class AnswerGenerator:
    """Combines retrieval output and an LLM to synthesize answers."""

    def __init__(self, retriever: GraphRetriever, llm: LLMClient, prompt_template: str | None = None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template or self._default_template()

    def _format_prompt(self, context: RetrievedContext) -> str:
        return self.prompt_template.format(query=context.query, context=context.context)

    def answer(self, question: str, top_k: int = 5, neighbor_k: int = 3) -> LLMResponse:
        context = self.retriever.best_context(question, top_k=top_k, neighbor_k=neighbor_k)
        if context is None:
            raise RuntimeError("No context found for the question")
        prompt = self._format_prompt(context)
        answer = self.llm.complete(prompt)
        return LLMResponse(question=question, answer=answer, used_context=context.context)

    @staticmethod
    def _default_template() -> str:
        return (
            "You are a helpful assistant. Answer the question using only the provided context.\n"
            "Context:\n{context}\n\n"
            "Question: {query}\n"
            "Answer:"
        )
