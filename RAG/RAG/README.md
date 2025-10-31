# GraphRAG (extended)

This directory contains a lightweight GraphRAG implementation that now supports multiple embedding backends and large language model providers. Use it to ingest plain text documents into graph-aware retrieval structures and answer questions with your preferred model stack.

## Key Features

- **Flexible embeddings** – choose between `sentence-transformers` and Hugging Face transformer checkpoints (including DeepSeek, Qwen, Llama, etc.).
- **Optional reranking** – enable a cross-encoder reranker with either backend for short-list refinement.
- **Multi-provider LLMs** – run QA via OpenAI, Hugging Face text-generation models, Google Gemini, or Anthropic Claude.
- **Streamlit UI** – explore analytics, graph visualisations, and retrieval results interactively with backend selection controls.

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -e .
```

You may also need provider-specific SDKs (e.g. `google-generativeai`, `anthropic`) depending on the models you plan to use.

## Environment Variables

CLI commands accept tokens via flags, but the following environment variables are also respected:

- `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN` – used for embedding or Hugging Face LLM checkpoints when the `--huggingface-token` or `--hf-llm-token` flags are omitted.
- `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` – defaults for the corresponding providers if `--api-key` is not supplied.

## Ingestion

Convert raw `.txt` files into graph-aware artifacts:

```bash
python ingest.py ./data --artifacts ./artifacts --embedding-backend huggingface \
  --embedding-model meta-llama/Meta-Llama-3-8B-Instruct --huggingface-token $HF_TOKEN
```

Useful flags:

- `--trust-remote-code` when using custom HF models that require it.
- `--index-backend faiss` if you have FAISS available for larger corpora.

## Asking Questions (CLI)

```bash
python ask.py ./data "What is the mission?" --provider huggingface \
  --model meta-llama/Meta-Llama-3-8B-Instruct --hf-llm-token $HF_TOKEN --use-reranker
```

Important options:

- `--provider {openai,huggingface,gemini,anthropic}` selects the LLM stack.
- `--model` chooses the specific checkpoint (provider-specific defaults are applied when omitted).
- `--api-key` supplies provider keys; otherwise environment variables are used.
- `--use-reranker` enables cross-encoder reranking with `--reranker-model` and `--reranker-backend` controlling the checkpoint and framework.

## Streamlit App

Launch the UI to inspect graphs and test retrieval interactively:

```bash
streamlit run streamlit_app.py
```

Select providers and enter tokens in the sidebar. The graph tab now exposes node inspection tools and enhanced visualisations.

## Embedding Existing Datasets

If you already have chunked datasets:

```bash
python embed_dataset.py ./artifacts/dataset.jsonl ./artifacts/embeddings.npz \
  --embedding-backend huggingface --huggingface-token $HF_TOKEN
```

## Notes

- Hugging Face remote code execution is disabled by default; opt in with `--trust-remote-code` only for trusted models.
- Large open-source models may require GPU support; pass `--device cuda` where appropriate.
- Ensure reranker checkpoints are compatible with their chosen backend.
