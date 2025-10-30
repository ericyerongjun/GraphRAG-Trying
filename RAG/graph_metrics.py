"""Compute basic metrics for a GraphRAG knowledge graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import networkx as nx


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report metrics for a GraphRAG GraphML file")
    parser.add_argument("graph", type=Path, help="Path to the GraphML graph")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top nodes to report by degree")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file to write metrics")
    return parser


def compute_metrics(graph: nx.Graph, top_n: int) -> Dict[str, Any]:
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    degrees = sorted(graph.degree, key=lambda x: x[1], reverse=True)
    top_degree = degrees[:top_n]
    density = nx.density(graph)
    components = list(nx.connected_components(graph))
    metrics: Dict[str, Any] = {
        "nodes": node_count,
        "edges": edge_count,
        "density": density,
        "components": len(components),
        "largest_component": max((len(c) for c in components), default=0),
        "top_degree": [
            {
                "node": node,
                "degree": degree,
                "doc_id": graph.nodes[node].get("doc_id"),
            }
            for node, degree in top_degree
        ],
    }
    return metrics


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()
    graph = nx.read_graphml(args.graph)
    metrics = compute_metrics(graph, args.top_n)
    report = json.dumps(metrics, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Metrics written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":  # pragma: no cover - CLI execution path
    main()
