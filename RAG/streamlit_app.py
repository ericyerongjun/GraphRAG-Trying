"""Streamlit user interface for the GraphRAG toolkit."""
from __future__ import annotations

import os
import shutil
import tempfile
from io import BytesIO, StringIO
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List, Sequence

import json

import plotly.graph_objects as go
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - allow runtime warning when PDF ingested
    PdfReader = None  # type: ignore[assignment]

import networkx as nx
from networkx.readwrite import json_graph

try:  # Support package-relative imports when deployed as a module
    from .GraphRAG import GraphRAG
    from .analytics import corpus_statistics, graph_statistics
    from .config import GraphRAGConfig
    from .mindmap import MindMapBuilder
    from .pattern_mining import frequent_terms
    from .qa_pipeline import (
        AnswerGenerator,
        AnthropicClient,
        GeminiClient,
        HuggingFaceClient,
        OpenAIClient,
    )
    from .reranker import CrossEncoderReranker
    from .retrieval import GraphRetriever
except ImportError:  # pragma: no cover - fallback for direct execution
    from GraphRAG import GraphRAG
    from analytics import corpus_statistics, graph_statistics
    from config import GraphRAGConfig
    from mindmap import MindMapBuilder
    from pattern_mining import frequent_terms
    from qa_pipeline import (
        AnswerGenerator,
        AnthropicClient,
        GeminiClient,
        HuggingFaceClient,
        OpenAIClient,
    )
    from reranker import CrossEncoderReranker
    from retrieval import GraphRetriever


def init_session_state() -> None:
    """Ensure session state keys exist."""
    defaults = {
        "rag": None,
        "temp_dir": None,
        "graph_stats": None,
        "response": None,
        "retrieval_results": None,
        "analytics": None,
        "reranker_cache": {},
        "last_retriever": None,
        "retriever_settings": None,
        "hf_token": "",
        "huggingface_trust_remote_code": False,
        "llm_settings": None,
        "reranker_max_length": 512,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def cleanup_temp_dir() -> None:
    temp_dir = st.session_state.get("temp_dir")
    if temp_dir and Path(temp_dir).exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    st.session_state["temp_dir"] = None


def _text_from_pdf(upload: UploadedFile) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf package is required to parse PDF files. Install with 'pip install pypdf'.")
    reader = PdfReader(BytesIO(upload.getvalue()))
    text_parts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text_parts.append(text)
    return "\n".join(text_parts)


def persist_uploads(files: List[UploadedFile]) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="graphrag_"))
    for uploaded in files:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix == ".pdf":
            text = _text_from_pdf(uploaded)
            target = temp_dir / f"{Path(uploaded.name).stem}.txt"
            target.write_text(text, encoding="utf-8")
        else:
            target = temp_dir / uploaded.name
            target.write_bytes(uploaded.getvalue())
    st.session_state["temp_dir"] = str(temp_dir)
    return temp_dir


def build_pipeline(upload_dir: Path, config: GraphRAGConfig) -> GraphRAG:
    rag = GraphRAG(config)
    rag.build(upload_dir)
    return rag


def describe_graph(rag: GraphRAG) -> dict[str, int]:
    graph = rag.graph
    if graph is None:
        return {"nodes": 0, "edges": 0}
    return {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}


def compute_analytics(rag: GraphRAG, min_doc_frequency: int, top_k_terms: int) -> dict[str, Any]:
    chunks = list(rag.chunks.values())
    corpus_stats = asdict(corpus_statistics(chunks)) if chunks else {}
    graph_stats_summary = graph_statistics(rag.graph) if rag.graph is not None else {}
    frequent = (
        frequent_terms(chunks, min_doc_frequency=min_doc_frequency, top_k=top_k_terms) if chunks else []
    )
    return {
        "corpus": corpus_stats,
        "graph": graph_stats_summary,
        "frequent_terms": frequent,
    }


def get_reranker(
    model_name: str,
    *,
    backend: str,
    device: str | None = None,
    hf_token: str | None = None,
    trust_remote_code: bool = False,
    max_length: int = 512,
) -> CrossEncoderReranker | None:
    if "reranker_cache" not in st.session_state:
        st.session_state["reranker_cache"] = {}
    cache: Dict[tuple, CrossEncoderReranker | None] = st.session_state["reranker_cache"]
    if not model_name:
        return None
    cache_key = (backend, model_name, device, trust_remote_code, max_length, hf_token)
    if cache_key in cache:
        return cache[cache_key]
    try:
        reranker = CrossEncoderReranker(
            model_name,
            device=device,
            backend=backend,
            hf_token=hf_token,
            trust_remote_code=trust_remote_code,
            max_length=max_length,
        )
    except Exception as exc:  # pragma: no cover - surface runtime errors
        st.error(f"Failed to load reranker '{model_name}': {exc}")
        cache[cache_key] = None
        return None
    cache[cache_key] = reranker
    return reranker


def build_graph_retriever(
    rag: GraphRAG,
    use_reranker: bool,
    reranker_model: str,
    reranker_backend: str,
    device: str | None = None,
    hf_token: str | None = None,
    trust_remote_code: bool = False,
    max_length: int = 512,
) -> GraphRetriever:
    reranker = None
    if use_reranker and reranker_model:
        reranker = get_reranker(
            reranker_model,
            backend=reranker_backend,
            device=device,
            hf_token=hf_token,
            trust_remote_code=trust_remote_code,
            max_length=max_length,
        )
    return GraphRetriever(rag, reranker=reranker)


def build_llm_client(settings: Dict[str, Any]):
    provider = (settings.get("provider") or "openai").lower()
    temperature = float(settings.get("temperature", 0.2))
    model_name = settings.get("model")
    if not model_name:
        raise RuntimeError("LLM model name is required")

    if provider == "openai":
        return OpenAIClient(model=model_name, temperature=temperature, api_key=settings.get("api_key"))
    if provider == "huggingface":
        return HuggingFaceClient(
            model=model_name,
            temperature=temperature,
            device=settings.get("device"),
            token=settings.get("token"),
            max_new_tokens=int(settings.get("max_new_tokens", 512)),
            trust_remote_code=bool(settings.get("trust_remote_code", False)),
        )
    if provider == "gemini":
        return GeminiClient(
            model=model_name,
            temperature=temperature,
            api_key=settings.get("api_key"),
            max_output_tokens=int(settings.get("max_output_tokens", 1024)),
        )
    if provider == "anthropic":
        return AnthropicClient(
            model=model_name,
            temperature=temperature,
            api_key=settings.get("api_key"),
            max_tokens=int(settings.get("max_tokens", 1024)),
        )
    raise RuntimeError(f"Unsupported LLM provider: {provider}")


def build_mindmap_payload(rag: GraphRAG, doc_ids: Sequence[str] | None = None, max_chunks: int = 5, neighbor_k: int = 3):
    if rag.graph is None:
        raise RuntimeError("Knowledge graph not built yet")
    builder = MindMapBuilder(rag.chunks, rag.graph)
    node = builder.build_full_map(doc_ids=doc_ids, max_chunks_per_doc=max_chunks, neighbor_k=neighbor_k)
    return builder.to_dict(node), builder.to_markdown(node)


def _build_graph_figure(graph: nx.Graph, max_nodes: int = 200) -> go.Figure:
    if graph.number_of_nodes() > max_nodes:
        nodes = list(graph.nodes())[:max_nodes]
        graph = graph.subgraph(nodes).copy()
    layout = nx.spring_layout(graph, seed=42)
    edge_x: List[float] = []
    edge_y: List[float] = []
    for source, target in graph.edges():
        x0, y0 = layout[source]
        x1, y1 = layout[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.8, color="#888"),
        hoverinfo="none",
        mode="lines",
    )
    node_x = []
    node_y = []
    text = []
    degrees = dict(graph.degree())
    for node in graph.nodes():
        x, y = layout[node]
        node_x.append(x)
        node_y.append(y)
        doc_id = graph.nodes[node].get("doc_id", "unknown")
        text.append(f"Chunk: {node}<br>Doc: {doc_id}<br>Degree: {degrees.get(node, 0)}")
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=text,
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            reversescale=True,
            color=[degrees.get(node, 0) for node in graph.nodes()],
            size=10,
            colorbar=dict(title="Degree"),
            line_width=1,
        ),
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=20, r=20, t=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def _mindmap_treemap_data(node: dict[str, Any], parent: str | None = None):
    labels: List[str] = []
    parents: List[str] = []
    values: List[int] = []

    def _walk(current: dict[str, Any], parent_name: str | None) -> None:
        name = current.get("name", "(unnamed)")
        chunks = current.get("chunks", []) or []
        value = max(len(chunks), 1)
        labels.append(name)
        parents.append(parent_name or "")
        values.append(value)
        for child in current.get("children", []) or []:
            _walk(child, name)

    _walk(node, parent)
    if parents:
        parents[0] = ""  # Root node needs empty parent for treemap
    return labels, parents, values


def build_mindmap_figure(node: dict[str, Any]) -> go.Figure:
    labels, parents, values = _mindmap_treemap_data(node)
    treemap = go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br>Chunks: %{value}<extra></extra>",
    )
    fig = go.Figure(treemap)
    fig.update_layout(margin=dict(t=20, l=10, r=10, b=10))
    return fig


def _top_degree_table(graph: nx.Graph, limit: int = 20) -> List[dict[str, Any]]:
    ranked = sorted(graph.degree(), key=lambda item: item[1], reverse=True)[:limit]
    table: List[dict[str, Any]] = []
    for node, degree in ranked:
        attrs = graph.nodes[node]
        table.append(
            {
                "Node": node,
                "Degree": degree,
                "Document": attrs.get("doc_id", "-"),
            }
        )
    return table


def _graph_download_payloads(graph: nx.Graph) -> dict[str, tuple[str, str]]:
    node_link_json = json.dumps(json_graph.node_link_data(graph), indent=2)
    buffer = StringIO()
    nx.write_gexf(graph, buffer)
    gexf_data = buffer.getvalue()
    return {
        "knowledge_graph.json": (node_link_json, "application/json"),
        "knowledge_graph.gexf": (gexf_data, "application/xml"),
    }


def run_retrieval(
    rag: GraphRAG,
    question: str,
    top_k: int,
    neighbor_k: int,
    use_reranker: bool,
    reranker_model: str,
    reranker_backend: str,
    device: str | None,
    hf_token: str | None,
    trust_remote_code: bool,
    max_length: int,
) -> list:
    retriever = build_graph_retriever(
        rag,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
        reranker_backend=reranker_backend,
        device=device,
        hf_token=hf_token,
        trust_remote_code=trust_remote_code,
        max_length=max_length,
    )
    results = retriever.search(question, top_k=top_k, neighbor_k=neighbor_k)
    st.session_state["retrieval_results"] = results
    st.session_state["last_retriever"] = retriever
    st.session_state["retriever_settings"] = {
        "use_reranker": use_reranker,
        "reranker_model": reranker_model,
        "reranker_backend": reranker_backend,
        "device": device,
        "top_k": top_k,
        "neighbor_k": neighbor_k,
        "hf_token": hf_token,
        "trust_remote_code": trust_remote_code,
        "max_length": max_length,
    }
    return results


def maybe_answer_question(
    question: str,
    rag: GraphRAG,
    use_reranker: bool,
    reranker_model: str,
    reranker_backend: str,
    device: str | None,
    hf_token: str | None,
    trust_remote_code: bool,
    max_length: int,
    llm_settings: Dict[str, Any] | None,
):
    results = st.session_state.get("retrieval_results")
    if not results:
        st.warning("Run retrieval before generating an answer.")
        return None
    retriever = st.session_state.get("last_retriever")
    settings = st.session_state.get("retriever_settings") or {}
    if (
        retriever is None
        or settings.get("use_reranker") != use_reranker
        or settings.get("reranker_model") != reranker_model
        or settings.get("reranker_backend") != reranker_backend
        or settings.get("device") != device
        or settings.get("hf_token") != hf_token
        or settings.get("trust_remote_code") != trust_remote_code
        or settings.get("max_length") != max_length
    ):
        retriever = build_graph_retriever(
            rag,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
            reranker_backend=reranker_backend,
            device=device,
            hf_token=hf_token,
            trust_remote_code=trust_remote_code,
            max_length=max_length,
        )
        st.session_state["last_retriever"] = retriever
        st.session_state["retriever_settings"] = {
            "use_reranker": use_reranker,
            "reranker_model": reranker_model,
            "reranker_backend": reranker_backend,
            "device": device,
            "hf_token": hf_token,
            "trust_remote_code": trust_remote_code,
            "max_length": max_length,
        }
    if not llm_settings:
        st.warning("Configure an LLM provider before generating an answer.")
        return None
    try:
        llm = build_llm_client(llm_settings)
    except RuntimeError as exc:
        st.error(str(exc))
        return None
    generator = AnswerGenerator(retriever, llm)
    try:
        response = generator.answer(question)
    except Exception as exc:  # pragma: no cover - surface runtime errors
        st.error(f"Failed to generate answer: {exc}")
        return None
    st.session_state["response"] = response
    st.session_state["llm_settings"] = llm_settings
    return response


def main() -> None:
    st.set_page_config(page_title="GraphRAG UI", layout="wide")
    st.title("GraphRAG Interactive Explorer")
    init_session_state()

    with st.sidebar:
        st.header("Configuration")
        chunk_size = st.number_input("Chunk size", min_value=100, max_value=2000, value=500, step=50)
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=100, step=25)
        max_keywords = st.number_input("Max keywords", min_value=3, max_value=50, value=12, step=1)
        min_token_length = st.number_input("Min token length", min_value=2, max_value=10, value=3, step=1)
        embedding_backend_options = ["sentence-transformers", "huggingface"]
        default_embedding_backend = st.session_state.get("embedding_backend", embedding_backend_options[0])
        if default_embedding_backend not in embedding_backend_options:
            default_embedding_backend = embedding_backend_options[0]
        embedding_backend = st.selectbox(
            "Embedding backend",
            options=embedding_backend_options,
            index=embedding_backend_options.index(default_embedding_backend),
        )
        st.session_state["embedding_backend"] = embedding_backend
        default_embedding_model = st.session_state.get(
            "embedding_model_input",
            "sentence-transformers/all-MiniLM-L6-v2"
            if embedding_backend == "sentence-transformers"
            else "BAAI/bge-base-en-v1.5",
        )
        embedding_model = st.text_input(
            "Embedding model",
            value=default_embedding_model,
            key="embedding_model_input",
            help="Hugging Face repo ID or sentence-transformers checkpoint.",
        )
        hf_token_input = st.text_input(
            "Hugging Face token (optional)",
            value=st.session_state.get("hf_token", ""),
            type="password",
            help="Used for private Hugging Face models.",
        )
        trust_remote_code = st.checkbox(
            "Trust remote code for Hugging Face models",
            value=st.session_state.get("huggingface_trust_remote_code", False),
        )
        st.session_state["hf_token"] = hf_token_input
        st.session_state["huggingface_trust_remote_code"] = trust_remote_code
        neighbor_k = st.slider("Neighbor context", min_value=1, max_value=10, value=3)
        top_k = st.slider("Top K results", min_value=1, max_value=10, value=5)
        artifacts_dir = st.text_input("Artifacts directory", value="artifacts")
        namespace = st.text_input("Artifacts namespace", value="default")
        index_backend = st.selectbox("Vector index backend", options=["memory", "faiss"], index=0)
        device = st.text_input("Torch device override", value="", help="Leave blank for auto-detection.")
        use_reranker = st.checkbox("Enable cross-encoder reranker", value=False)
        reranker_backend_options = ["sentence-transformers", "huggingface"]
        default_reranker_backend = st.session_state.get("reranker_backend_select", reranker_backend_options[0])
        if default_reranker_backend not in reranker_backend_options:
            default_reranker_backend = reranker_backend_options[0]
        reranker_backend = st.selectbox(
            "Reranker backend",
            options=reranker_backend_options,
            index=reranker_backend_options.index(default_reranker_backend),
            disabled=not use_reranker,
            key="reranker_backend_select",
        )
        reranker_model_input = st.text_input(
            "Reranker model",
            value=st.session_state.get("reranker_model_input", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
            disabled=not use_reranker,
            key="reranker_model_input",
            help="Sentence-transformers cross encoder or Hugging Face sequence classification model.",
        )
        reranker_model = reranker_model_input if use_reranker else ""
        reranker_max_length = st.slider(
            "Reranker max sequence length",
            min_value=128,
            max_value=2048,
            value=int(st.session_state.get("reranker_max_length", 512)),
            step=64,
            disabled=not use_reranker,
        )
        st.session_state["reranker_max_length"] = reranker_max_length
        min_doc_frequency = st.slider("Min doc frequency", min_value=1, max_value=20, value=5)
        top_terms = st.slider("Top frequent terms", min_value=5, max_value=100, value=25, step=5)
        huggingface_token = hf_token_input.strip() or None

    uploaded_files = st.file_uploader(
        "Upload documents", type=["txt", "md", "pdf"], accept_multiple_files=True
    )
    build_col, reset_col = st.columns(2)
    with build_col:
        if st.button("Build knowledge graph", disabled=not uploaded_files):
            cleanup_temp_dir()
            upload_dir = persist_uploads(uploaded_files)
            config = GraphRAGConfig(
                data_dir=Path(artifacts_dir),
                embedding_model=embedding_model,
                embedding_backend=embedding_backend,
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                max_keywords=int(max_keywords),
                min_token_length=int(min_token_length),
                artifacts_namespace=namespace,
                index_backend=index_backend,
                device=device or None,
                reranker_model=reranker_model or None,
                reranker_backend=reranker_backend,
                huggingface_token=huggingface_token,
                huggingface_trust_remote_code=trust_remote_code,
            )
            try:
                rag = build_pipeline(upload_dir, config)
            except Exception as exc:  # pragma: no cover - surface runtime errors
                st.error(f"Failed to build GraphRAG pipeline: {exc}")
            else:
                st.session_state["rag"] = rag
                stats = describe_graph(rag)
                st.session_state["graph_stats"] = stats
                st.session_state["embedding_backend"] = embedding_backend
                st.session_state["hf_token"] = hf_token_input
                st.session_state["huggingface_trust_remote_code"] = trust_remote_code
                st.session_state["reranker_max_length"] = reranker_max_length
                try:
                    analytics_summary = compute_analytics(rag, int(min_doc_frequency), int(top_terms))
                except Exception as exc:  # pragma: no cover - analytics best-effort
                    st.warning(f"Analytics computation failed: {exc}")
                    st.session_state["analytics"] = None
                else:
                    st.session_state["analytics"] = analytics_summary
                st.session_state["retrieval_results"] = None
                st.session_state["response"] = None
                st.session_state["last_retriever"] = None
                st.session_state["retriever_settings"] = None
                st.success(
                    f"Graph built with {stats['nodes']} nodes and {stats['edges']} edges from {len(uploaded_files)} files."
                )
    with reset_col:
        if st.button("Reset session"):
            cleanup_temp_dir()
            for key in [
                "rag",
                "graph_stats",
                "response",
                "retrieval_results",
                "analytics",
                "last_retriever",
                "retriever_settings",
            ]:
                st.session_state[key] = None
            st.session_state["hf_token"] = ""
            st.session_state["huggingface_trust_remote_code"] = False
            st.session_state["llm_settings"] = None
            st.session_state["reranker_max_length"] = 512
            st.session_state["embedding_backend"] = "sentence-transformers"
            st.experimental_rerun()

    rag = st.session_state.get("rag")
    if rag is None:
        st.info("Upload documents and build the graph to begin exploring.")
        return

    stats = st.session_state.get("graph_stats") or describe_graph(rag)
    rag_config = getattr(rag, "config", None)
    config_device = getattr(rag_config, "device", None) if rag_config else None
    config_reranker_model = getattr(rag_config, "reranker_model", None) if rag_config else None
    config_reranker_backend = getattr(rag_config, "reranker_backend", "sentence-transformers") if rag_config else "sentence-transformers"
    config_hf_token = getattr(rag_config, "huggingface_token", None) if rag_config else None
    config_trust_remote_code = getattr(rag_config, "huggingface_trust_remote_code", False) if rag_config else False
    active_device = (device or None) or config_device
    active_reranker_model = reranker_model or (config_reranker_model or "")
    active_reranker_backend = reranker_backend or config_reranker_backend
    effective_hf_token = huggingface_token or config_hf_token
    active_trust_remote_code = trust_remote_code if trust_remote_code is not None else config_trust_remote_code
    reranker_length_setting = int(st.session_state.get("reranker_max_length", reranker_max_length))
    overview_tab, chunks_tab, graph_tab, mindmap_tab, analytics_tab = st.tabs(
        ["Overview", "Chunks", "Graph", "Mind Map", "Analytics"]
    )

    with overview_tab:
        st.subheader("Graph overview")
        st.json(stats)

    with chunks_tab:
        st.subheader("Chunk preview")
        chunk_total = len(rag.chunks)
        if chunk_total == 0:
            st.info("No chunks available to preview.")
        else:
            max_rows = min(100, chunk_total)
            min_rows = 1 if max_rows < 5 else 5
            default_rows = max_rows if max_rows < 10 else 10
            step = 1 if max_rows < 5 else 5
            chunk_limit = st.slider(
                "Rows",
                min_value=min_rows,
                max_value=max_rows,
                value=default_rows,
                step=step,
                key="chunk_preview_limit",
            )
            preview = [
                {
                    "Chunk ID": chunk.chunk_id,
                    "Document": chunk.doc_id,
                    "Preview": (chunk.text.replace("\n", " ")[:200] + ("…" if len(chunk.text) > 200 else "")),
                }
                for chunk in list(rag.chunks.values())[:chunk_limit]
            ]
            st.dataframe(preview, use_container_width=True)

    with graph_tab:
        st.subheader("Knowledge graph")
        graph = rag.graph
        if graph is None or graph.number_of_nodes() == 0:
            st.info("Graph is empty; upload more content or adjust chunking parameters.")
        else:
            if graph.number_of_nodes() > 200:
                st.caption("Showing the first 200 nodes for readability.")
            fig = _build_graph_figure(graph)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Top-degree nodes**")
            st.dataframe(_top_degree_table(graph), use_container_width=True)
            selected_node = st.text_input(
                "Inspect node",
                value=st.session_state.get("selected_graph_node", ""),
                placeholder="Enter chunk ID",
                key="graph_node_input",
                help="Type a chunk ID to view its neighbors.",
            )
            st.session_state["selected_graph_node"] = selected_node
            if selected_node:
                if graph.has_node(selected_node):
                    neighbors = list(graph.neighbors(selected_node))
                    neighbor_rows = []
                    for neighbor in neighbors:
                        attrs = graph.nodes[neighbor]
                        neighbor_rows.append(
                            {
                                "Neighbor": neighbor,
                                "Document": attrs.get("doc_id", "-"),
                                "Preview": attrs.get("text", "")[:200],
                                "Degree": graph.degree(neighbor),
                            }
                        )
                    if neighbor_rows:
                        st.markdown(f"**Neighbors of {selected_node}**")
                        st.dataframe(neighbor_rows, use_container_width=True)
                    else:
                        st.info("Selected node has no neighbors.")
                else:
                    st.warning("Node not found in graph.")
            downloads = _graph_download_payloads(graph)
            for filename, (payload, mime) in downloads.items():
                st.download_button(
                    label=f"Download {filename}",
                    data=payload,
                    file_name=filename,
                    mime=mime,
                    use_container_width=True,
                )

    with mindmap_tab:
        st.subheader("Mind map overview")
        if rag.graph is None or rag.graph.number_of_nodes() == 0:
            st.info("Build the graph to view the mind map.")
        else:
            slider_max = min(len(rag.chunks), 20)
            max_chunks = st.slider(
                "Chunks per document",
                min_value=1,
                max_value=max(1, slider_max),
                value=min(5, max(1, slider_max)),
                step=1,
                key="mindmap_chunk_limit",
            )
            try:
                mindmap_dict, mindmap_text = build_mindmap_payload(
                    rag,
                    max_chunks=max_chunks,
                    neighbor_k=int(neighbor_k),
                )
            except Exception as exc:  # pragma: no cover - surface runtime errors
                st.error(f"Unable to build mind map: {exc}")
            else:
                map_cols = st.columns(2)
                with map_cols[0]:
                    st.markdown("### Mind map outline")
                    st.code(mindmap_text, language="markdown")
                    st.download_button(
                        label="Download mind map markdown",
                        data=mindmap_text,
                        file_name="mindmap.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with map_cols[1]:
                    st.markdown("### Hierarchy view")
                    mindmap_fig = build_mindmap_figure(mindmap_dict)
                    st.plotly_chart(mindmap_fig, use_container_width=True)
                    st.download_button(
                        label="Download mind map JSON",
                        data=json.dumps(mindmap_dict, indent=2),
                        file_name="mindmap.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                with st.expander("Mind map as JSON", expanded=False):
                    st.json(mindmap_dict)

    with analytics_tab:
        st.subheader("Analytics and pattern mining")
        analytics = st.session_state.get("analytics")
        if st.button("Recompute analytics"):
            try:
                analytics = compute_analytics(rag, int(min_doc_frequency), int(top_terms))
            except Exception as exc:  # pragma: no cover - analytics best-effort
                st.error(f"Analytics computation failed: {exc}")
            else:
                st.session_state["analytics"] = analytics
        if not analytics:
            st.info("Analytics not available. Rebuild the graph to compute statistics.")
        else:
            corpus_stats = analytics.get("corpus") or {}
            graph_stats_details = analytics.get("graph") or {}
            frequent = analytics.get("frequent_terms") or []
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Corpus statistics**")
                if corpus_stats:
                    st.json(corpus_stats)
                else:
                    st.caption("No corpus statistics available.")
                st.markdown("**Graph statistics**")
                if graph_stats_details:
                    st.json(graph_stats_details)
                else:
                    st.caption("Graph statistics unavailable.")
            with col_right:
                st.markdown("**Frequent terms**")
                if frequent:
                    freq_table = [
                        {
                            "Term": pattern.term,
                            "Frequency": pattern.frequency,
                            "Documents": pattern.document_frequency,
                        }
                        for pattern in frequent
                    ]
                    st.dataframe(freq_table, use_container_width=True)
                else:
                    st.caption("No frequent terms met the document frequency threshold.")

    question = st.text_input("Ask a question about your documents")
    if st.button("Run retrieval", disabled=not question):
        try:
            results = run_retrieval(
                rag,
                question,
                top_k=int(top_k),
                neighbor_k=int(neighbor_k),
                use_reranker=use_reranker,
                reranker_model=active_reranker_model,
                reranker_backend=active_reranker_backend,
                device=active_device,
                hf_token=effective_hf_token,
                trust_remote_code=active_trust_remote_code,
                max_length=reranker_length_setting,
            )
        except Exception as exc:  # pragma: no cover - surface runtime errors
            st.error(f"Retrieval failed: {exc}")
        else:
            for idx, context in enumerate(results, start=1):
                st.markdown(f"**Result {idx} — score {context.result.score:.3f}**")
                st.text_area(
                    label=f"Chunk {context.result.chunk_id}",
                    value=context.context,
                    height=200,
                    key=f"chunk_{context.result.chunk_id}",
                )
                if context.mindmap:
                    with st.expander("Mind map focus", expanded=False):
                        st.code(context.mindmap, language="markdown")

    if question:
        with st.expander("Optional: Generate an answer with an LLM"):
            llm_options = ["openai", "huggingface", "gemini", "anthropic"]
            stored_llm_settings = st.session_state.get("llm_settings") or {}
            default_provider = stored_llm_settings.get("provider", "openai")
            if default_provider not in llm_options:
                default_provider = "openai"
            llm_provider = st.selectbox(
                "LLM provider",
                options=llm_options,
                index=llm_options.index(default_provider),
            )
            temperature_default = float(stored_llm_settings.get("temperature", 0.2))
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=temperature_default,
                step=0.05,
                key="llm_temperature_slider",
            )
            llm_settings: Dict[str, Any] = {
                "provider": llm_provider,
                "temperature": temperature,
            }
            generate_disabled = False

            if llm_provider == "openai":
                openai_model = st.text_input(
                    "OpenAI model",
                    value=stored_llm_settings.get("model", "gpt-4o-mini"),
                    key="openai_model_input",
                )
                openai_key_default = stored_llm_settings.get("api_key") or os.getenv("OPENAI_API_KEY", "")
                openai_api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    value=openai_key_default,
                    key="openai_api_key_input",
                )
                llm_settings.update({"model": openai_model, "api_key": openai_api_key})
                generate_disabled = not (openai_api_key or os.getenv("OPENAI_API_KEY"))
                if generate_disabled:
                    st.info("Provide an OpenAI API key to generate answers.")
            elif llm_provider == "huggingface":
                hf_model = st.text_input(
                    "Hugging Face model",
                    value=stored_llm_settings.get("model", "meta-llama/Llama-3.1-8B-Instruct"),
                    key="hf_llm_model_input",
                )
                hf_token_default = stored_llm_settings.get("token") or (effective_hf_token or "")
                hf_llm_token = st.text_input(
                    "Hugging Face token (optional)",
                    value=hf_token_default,
                    type="password",
                    key="hf_llm_token_input",
                )
                hf_device = st.text_input(
                    "Model device override",
                    value=stored_llm_settings.get("device") or (device or ""),
                    key="hf_llm_device_input",
                    help="Leave blank for auto-detection.",
                )
                hf_max_new_tokens = st.slider(
                    "Max new tokens",
                    min_value=32,
                    max_value=2048,
                    value=int(stored_llm_settings.get("max_new_tokens", 512)),
                    step=32,
                    key="hf_llm_max_tokens_slider",
                )
                hf_trust_remote_code = st.checkbox(
                    "Trust remote code for LLM",
                    value=bool(stored_llm_settings.get("trust_remote_code", active_trust_remote_code)),
                    key="hf_llm_trust_remote_code_checkbox",
                )
                llm_settings.update(
                    {
                        "model": hf_model,
                        "token": hf_llm_token or None,
                        "device": hf_device or None,
                        "max_new_tokens": hf_max_new_tokens,
                        "trust_remote_code": hf_trust_remote_code,
                    }
                )
            elif llm_provider == "gemini":
                gemini_model = st.text_input(
                    "Gemini model",
                    value=stored_llm_settings.get("model", "gemini-1.5-flash"),
                    key="gemini_model_input",
                )
                gemini_key_default = stored_llm_settings.get("api_key") or os.getenv("GEMINI_API_KEY", "")
                gemini_api_key = st.text_input(
                    "Gemini API Key",
                    type="password",
                    value=gemini_key_default,
                    key="gemini_api_key_input",
                )
                gemini_max_tokens = st.slider(
                    "Gemini max output tokens",
                    min_value=256,
                    max_value=4096,
                    value=int(stored_llm_settings.get("max_output_tokens", 1024)),
                    step=256,
                    key="gemini_max_tokens_slider",
                )
                llm_settings.update(
                    {
                        "model": gemini_model,
                        "api_key": gemini_api_key,
                        "max_output_tokens": gemini_max_tokens,
                    }
                )
                generate_disabled = not (gemini_api_key or os.getenv("GEMINI_API_KEY"))
                if generate_disabled:
                    st.info("Provide a Gemini API key to generate answers.")
            else:  # anthropic
                anthropic_model = st.text_input(
                    "Anthropic model",
                    value=stored_llm_settings.get("model", "claude-3-5-sonnet-20240620"),
                    key="anthropic_model_input",
                )
                anthropic_key_default = stored_llm_settings.get("api_key") or os.getenv("ANTHROPIC_API_KEY", "")
                anthropic_api_key = st.text_input(
                    "Anthropic API Key",
                    type="password",
                    value=anthropic_key_default,
                    key="anthropic_api_key_input",
                )
                anthropic_max_tokens = st.slider(
                    "Anthropic max tokens",
                    min_value=256,
                    max_value=4096,
                    value=int(stored_llm_settings.get("max_tokens", 1024)),
                    step=256,
                    key="anthropic_max_tokens_slider",
                )
                llm_settings.update(
                    {
                        "model": anthropic_model,
                        "api_key": anthropic_api_key,
                        "max_tokens": anthropic_max_tokens,
                    }
                )
                generate_disabled = not (anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"))
                if generate_disabled:
                    st.info("Provide an Anthropic API key to generate answers.")

            if st.button("Generate answer", key="generate_answer_button", disabled=generate_disabled):
                response = maybe_answer_question(
                    question,
                    rag,
                    use_reranker,
                    active_reranker_model,
                    active_reranker_backend,
                    active_device,
                    effective_hf_token,
                    active_trust_remote_code,
                    reranker_length_setting,
                    llm_settings,
                )
                if response:
                    st.subheader("Answer")
                    st.write(response.answer)
                    st.subheader("Context used")
                    st.text_area("Answer context", value=response.used_context, height=200)


if __name__ == "__main__":  # pragma: no cover - manual execution path
    main()
