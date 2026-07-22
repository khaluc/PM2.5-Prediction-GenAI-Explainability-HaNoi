"""Tests for the validated and evidence-bound PM2.5 knowledge graph."""

from __future__ import annotations

from src.genai.knowledge_graph import (
    build_pm25_knowledge_context,
    load_pm25_knowledge_graph,
    query_pm25_knowledge_graph,
)


def test_graph_contains_requested_nodes_and_relations() -> None:
    graph = load_pm25_knowledge_graph()
    node_ids = {node["id"] for node in graph["nodes"]}
    assert node_ids == {
        "pm25",
        "pm10",
        "no2",
        "so2",
        "co",
        "vehicle_emission",
        "factory",
        "construction",
        "wind",
        "humidity",
        "rain",
        "temperature",
        "tree_planting",
        "low_emission_zone",
        "public_transport",
    }
    relation_counts = {
        relation: sum(edge["relation"] == relation for edge in graph["edges"])
        for relation in {"EMITS", "INFLUENCED_BY", "MITIGATED_BY"}
    }
    assert relation_counts == {"EMITS": 11, "INFLUENCED_BY": 4, "MITIGATED_BY": 7}
    assert all(edge["source_refs"] for edge in graph["edges"])
    assert all(
        edge["event_claim_allowed"] is False
        for edge in graph["edges"]
        if edge["relation"] in {"EMITS", "INFLUENCED_BY"}
    )


def test_relation_query_returns_only_referenced_nodes_and_sources() -> None:
    graph = query_pm25_knowledge_graph("INFLUENCED_BY")
    assert graph["relation_filter"] == "INFLUENCED_BY"
    assert {edge["relation"] for edge in graph["edges"]} == {"INFLUENCED_BY"}
    assert {node["id"] for node in graph["nodes"]} == {
        "pm25",
        "wind",
        "humidity",
        "rain",
        "temperature",
    }
    assert "Atmospheric Chemistry and Physics" in {
        source["organization"] for source in graph["sources"]
    }


def test_current_evidence_activates_only_supported_domain_relations() -> None:
    context = build_pm25_knowledge_context(
        [
            "traffic_congestion",
            "rain_observed",
            "wind_observed",
            "humidity_observed",
            "temperature_observed",
        ],
        screening_level_code=3,
    )
    assert [item["id"] for item in context["supported_emission_sources"]] == [
        "vehicle_emission"
    ]
    assert {item["id"] for item in context["unverified_emission_sources"]} == {
        "factory",
        "construction",
    }
    weather_by_id = {item["id"]: item for item in context["meteorological_factors"]}
    assert all(item["currently_observed"] for item in weather_by_id.values())
    mitigation_by_id = {item["id"]: item for item in context["mitigations"]}
    assert set(mitigation_by_id) == {
        "tree_planting",
        "low_emission_zone",
        "public_transport",
    }
    assert context["causal_claim_allowed"] is False
    assert context["diagnostic_claim_allowed"] is False
