"""Validated PM2.5 domain knowledge used to ground forecast explanations."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "knowledge" / "pm25_knowledge_graph.json"
ALLOWED_RELATIONS = {"EMITS", "INFLUENCED_BY", "MITIGATED_BY"}


class KnowledgeGraphError(ValueError):
    """The PM2.5 graph is missing or violates its schema constraints."""


def _require_text(payload: dict[str, Any], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeGraphError(f"Missing {key} in {context}")
    return value.strip()


def _validate_graph(graph: dict[str, Any]) -> None:
    _require_text(graph, "graph_id", context="graph")
    focus_node = _require_text(graph, "focus_node", context="graph")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    sources = graph.get("sources")
    if not isinstance(nodes, list) or not nodes:
        raise KnowledgeGraphError("Graph nodes must be a non-empty list")
    if not isinstance(edges, list) or not edges:
        raise KnowledgeGraphError("Graph edges must be a non-empty list")
    if not isinstance(sources, list):
        raise KnowledgeGraphError("Graph sources must be a list")

    node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise KnowledgeGraphError(f"Invalid node at index {index}")
        node_id = _require_text(node, "id", context=f"node[{index}]")
        _require_text(node, "type", context=f"node[{index}]")
        _require_text(node, "label_vi", context=f"node[{index}]")
        if node_id in node_ids:
            raise KnowledgeGraphError(f"Duplicate node id: {node_id}")
        node_ids.add(node_id)
    if focus_node not in node_ids:
        raise KnowledgeGraphError(f"Unknown focus node: {focus_node}")

    source_ids: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise KnowledgeGraphError(f"Invalid source at index {index}")
        source_id = _require_text(source, "id", context=f"source[{index}]")
        _require_text(source, "organization", context=f"source[{index}]")
        _require_text(source, "title", context=f"source[{index}]")
        _require_text(source, "url", context=f"source[{index}]")
        if source_id in source_ids:
            raise KnowledgeGraphError(f"Duplicate source id: {source_id}")
        source_ids.add(source_id)

    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise KnowledgeGraphError(f"Invalid edge at index {index}")
        source = _require_text(edge, "source", context=f"edge[{index}]")
        target = _require_text(edge, "target", context=f"edge[{index}]")
        relation = _require_text(edge, "relation", context=f"edge[{index}]")
        _require_text(edge, "statement_vi", context=f"edge[{index}]")
        if source not in node_ids or target not in node_ids:
            raise KnowledgeGraphError(f"Unknown node reference in edge[{index}]")
        if relation not in ALLOWED_RELATIONS:
            raise KnowledgeGraphError(f"Unsupported relation: {relation}")
        refs = edge.get("source_refs", [])
        if not isinstance(refs, list) or any(ref not in source_ids for ref in refs):
            raise KnowledgeGraphError(f"Unknown source reference in edge[{index}]")
        if relation in {"EMITS", "INFLUENCED_BY"} and edge.get("event_claim_allowed") is not False:
            raise KnowledgeGraphError(
                f"{relation} edges must explicitly disable event-level causal claims"
            )


@lru_cache(maxsize=8)
def _load_graph_cached(path: str) -> dict[str, Any]:
    graph_path = Path(path)
    if not graph_path.is_file():
        raise KnowledgeGraphError(f"Knowledge graph not found: {graph_path}")
    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KnowledgeGraphError(f"Unable to read knowledge graph: {graph_path}") from error
    if not isinstance(graph, dict):
        raise KnowledgeGraphError("Knowledge graph root must be an object")
    _validate_graph(graph)
    return graph


def load_pm25_knowledge_graph(
    path: str | Path = DEFAULT_GRAPH_PATH,
) -> dict[str, Any]:
    """Return a defensive copy of the validated PM2.5 graph."""

    resolved = str(Path(path).resolve())
    return copy.deepcopy(_load_graph_cached(resolved))


def query_pm25_knowledge_graph(
    relation: str | None = None,
    *,
    path: str | Path = DEFAULT_GRAPH_PATH,
) -> dict[str, Any]:
    """Return the full graph or one relation family with its referenced nodes."""

    graph = load_pm25_knowledge_graph(path)
    selected_relation = relation.upper() if relation else None
    if selected_relation and selected_relation not in ALLOWED_RELATIONS:
        raise KnowledgeGraphError(f"Unsupported relation: {relation}")
    edges = [
        edge
        for edge in graph["edges"]
        if selected_relation is None or edge["relation"] == selected_relation
    ]
    referenced_nodes = {graph["focus_node"]}
    referenced_sources: set[str] = set()
    for edge in edges:
        referenced_nodes.update((edge["source"], edge["target"]))
        referenced_sources.update(edge.get("source_refs", []))
    graph["nodes"] = [node for node in graph["nodes"] if node["id"] in referenced_nodes]
    graph["edges"] = edges
    graph["sources"] = [
        source for source in graph["sources"] if source["id"] in referenced_sources
    ]
    graph["relation_filter"] = selected_relation
    return graph


def build_pm25_knowledge_context(
    observed_condition_codes: Iterable[str],
    *,
    screening_level_code: int | None,
    path: str | Path = DEFAULT_GRAPH_PATH,
) -> dict[str, Any]:
    """Bind general graph knowledge to current evidence without asserting causality."""

    graph = load_pm25_knowledge_graph(path)
    observed = {str(code) for code in observed_condition_codes}
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    relations: list[dict[str, Any]] = []
    emission_sources: dict[str, dict[str, Any]] = {}
    meteorological_factors: dict[str, dict[str, Any]] = {}
    mitigations: dict[str, dict[str, Any]] = {}

    for edge in graph["edges"]:
        source = node_by_id[edge["source"]]
        target = node_by_id[edge["target"]]
        required_codes = set(edge.get("activation_evidence_codes", []))
        matched_codes = sorted(observed & required_codes)
        minimum_level = edge.get("relevant_from_screening_level")
        screening_relevant = bool(
            isinstance(minimum_level, int)
            and isinstance(screening_level_code, int)
            and screening_level_code >= minimum_level
        )
        evidence_supported = bool(required_codes and matched_codes)
        relation = {
            "source": {
                "id": source["id"],
                "label_vi": source["label_vi"],
                "type": source["type"],
            },
            "relation": edge["relation"],
            "target": {
                "id": target["id"],
                "label_vi": target["label_vi"],
                "type": target["type"],
            },
            "statement_vi": edge["statement_vi"],
            "claim_scope": edge.get("claim_scope") or edge.get("mitigation_scope"),
            "matched_evidence_codes": matched_codes,
            "supported_by_current_evidence": evidence_supported,
            "screening_relevant": screening_relevant,
            "event_claim_allowed": bool(edge.get("event_claim_allowed", False)),
        }
        relations.append(relation)

        if edge["relation"] == "EMITS":
            item = emission_sources.setdefault(
                source["id"],
                {
                    "id": source["id"],
                    "label_vi": source["label_vi"],
                    "label_en": source.get("label_en"),
                    "aliases_vi": source.get("aliases_vi", []),
                    "emitted_pollutants": [],
                    "matched_evidence_codes": [],
                    "supported_by_current_evidence": False,
                    "event_claim_allowed": False,
                },
            )
            item["emitted_pollutants"].append(target["id"])
            item["matched_evidence_codes"] = sorted(
                set(item["matched_evidence_codes"]) | set(matched_codes)
            )
            item["supported_by_current_evidence"] = bool(
                item["supported_by_current_evidence"] or evidence_supported
            )
        elif edge["relation"] == "INFLUENCED_BY":
            item = meteorological_factors.setdefault(
                target["id"],
                {
                    "id": target["id"],
                    "label_vi": target["label_vi"],
                    "label_en": target.get("label_en"),
                    "matched_evidence_codes": [],
                    "currently_observed": False,
                    "event_claim_allowed": False,
                    "statements_vi": [],
                },
            )
            item["matched_evidence_codes"] = sorted(
                set(item["matched_evidence_codes"]) | set(matched_codes)
            )
            item["currently_observed"] = bool(
                item["currently_observed"] or evidence_supported
            )
            item["statements_vi"].append(edge["statement_vi"])
        elif edge["relation"] == "MITIGATED_BY":
            item = mitigations.setdefault(
                target["id"],
                {
                    "id": target["id"],
                    "label_vi": target["label_vi"],
                    "label_en": target.get("label_en"),
                    "scope": edge.get("mitigation_scope"),
                    "applies_to": [],
                    "statements_vi": [],
                    "automatic_action_allowed": False,
                },
            )
            item["applies_to"].append(source["id"])
            item["statements_vi"].append(edge["statement_vi"])

    supported_sources = [
        item for item in emission_sources.values() if item["supported_by_current_evidence"]
    ]
    unverified_sources = [
        item for item in emission_sources.values() if not item["supported_by_current_evidence"]
    ]

    return {
        "graph_id": graph["graph_id"],
        "focus_node": graph["focus_node"],
        "disclaimer_vi": graph["disclaimer_vi"],
        "observed_condition_codes": sorted(observed),
        "supported_emission_sources": supported_sources,
        "unverified_emission_sources": unverified_sources,
        "meteorological_factors": list(meteorological_factors.values()),
        "mitigations": list(mitigations.values()),
        "relations": relations,
        "sources": graph["sources"],
        "causal_claim_allowed": False,
        "diagnostic_claim_allowed": False,
    }


def unsupported_emission_source_terms(context: dict[str, Any]) -> list[str]:
    """Return source aliases that cannot be mentioned without current evidence."""

    terms: list[str] = []
    for item in context.get("unverified_emission_sources", []):
        terms.extend(
            str(value)
            for value in [
                item.get("label_vi"),
                item.get("label_en"),
                *(item.get("aliases_vi") or []),
            ]
            if value
        )
    return sorted(set(terms), key=str.casefold)


__all__ = [
    "ALLOWED_RELATIONS",
    "DEFAULT_GRAPH_PATH",
    "KnowledgeGraphError",
    "build_pm25_knowledge_context",
    "load_pm25_knowledge_graph",
    "query_pm25_knowledge_graph",
    "unsupported_emission_source_terms",
]
