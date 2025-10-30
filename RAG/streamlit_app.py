"""Streamlit user interface for the GraphRAG toolkit."""
from __future__ import annotations

import os
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List

import plotly.graph_objects as go
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - allow runtime warning when PDF ingested
    PdfReader = None  # type: ignore[assignment]

import networkx as nx

try:  # Support package-relative imports when deployed as a module
    from .GraphRAG import GraphRAG
    from .config import GraphRAGConfig
    from .qa_pipeline import AnswerGenerator, OpenAIClient
    from .retrieval import GraphRetriever
except ImportError:  # pragma: no cover - fallback for direct execution
    from GraphRAG import GraphRAG
    from config import GraphRAGConfig
    from qa_pipeline import AnswerGenerator, OpenAIClient
    from retrieval import GraphRetriever


def init_session_state() -> None:
    """Ensure session state keys exist."""
    defaults = {
        "rag": None,
        "temp_dir": None,
        "graph_stats": None,
        "response": None,
        "retrieval_results": None,
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


def run_retrieval(rag: GraphRAG, question: str, top_k: int, neighbor_k: int) -> list:
    retriever = GraphRetriever(rag)
    results = retriever.search(question, top_k=top_k, neighbor_k=neighbor_k)
    st.session_state["retrieval_results"] = results
    return results


def maybe_answer_question(question: str, api_key: str | None, model: str, temperature: float):
    results = st.session_state.get("retrieval_results")
    rag = st.session_state.get("rag")
    if not results or rag is None:
        st.warning("Run retrieval before generating an answer.")
        return None
    retriever = GraphRetriever(rag)
    if not api_key:
        st.info("No OpenAI API key provided; showing retrieved context only.")
        return None
    os.environ["OPENAI_API_KEY"] = api_key
    try:
        llm = OpenAIClient(model=model, temperature=temperature)
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
        embedding_model = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2")
        neighbor_k = st.slider("Neighbor context", min_value=1, max_value=10, value=3)
        top_k = st.slider("Top K results", min_value=1, max_value=10, value=5)
        artifacts_dir = st.text_input("Artifacts directory", value="artifacts")

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
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                max_keywords=int(max_keywords),
                min_token_length=int(min_token_length),
            )
            try:
                rag = build_pipeline(upload_dir, config)
            except Exception as exc:  # pragma: no cover - surface runtime errors
                st.error(f"Failed to build GraphRAG pipeline: {exc}")
            else:
                st.session_state["rag"] = rag
                stats = describe_graph(rag)
                st.session_state["graph_stats"] = stats
                st.success(
                    f"Graph built with {stats['nodes']} nodes and {stats['edges']} edges from {len(uploaded_files)} files."
                )
    with reset_col:
        if st.button("Reset session"):
            cleanup_temp_dir()
            for key in ["rag", "graph_stats", "response", "retrieval_results"]:
                st.session_state[key] = None
            st.experimental_rerun()

    rag = st.session_state.get("rag")
    if rag is None:
        st.info("Upload documents and build the graph to begin exploring.")
        return

    stats = st.session_state.get("graph_stats") or describe_graph(rag)
    overview_tab, chunks_tab, graph_tab = st.tabs(["Overview", "Chunks", "Graph"])

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

    question = st.text_input("Ask a question about your documents")
    if st.button("Run retrieval", disabled=not question):
        try:
            results = run_retrieval(rag, question, top_k=int(top_k), neighbor_k=int(neighbor_k))
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

    if question:
        with st.expander("Optional: Generate an answer with OpenAI"):
            api_key = st.text_input("OpenAI API Key", type="password")
            model = st.text_input("OpenAI model", value="gpt-4o-mini")
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
            if st.button("Generate answer", disabled=api_key is None or api_key == ""):
                response = maybe_answer_question(question, api_key, model, temperature)
                if response:
                    st.subheader("Answer")
                    st.write(response.answer)
                    st.subheader("Context used")
                    st.text_area("Answer context", value=response.used_context, height=200)


if __name__ == "__main__":  # pragma: no cover - manual execution path
    main()
