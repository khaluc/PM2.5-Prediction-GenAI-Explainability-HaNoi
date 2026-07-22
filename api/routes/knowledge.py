"""Read-only endpoints for the validated PM2.5 domain knowledge graph."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException

from src.genai.knowledge_graph import KnowledgeGraphError, query_pm25_knowledge_graph


router = APIRouter(prefix="/knowledge-graph", tags=["knowledge-graph"])


@router.get("/pm25")
def pm25_knowledge_graph(
    relation: Literal["EMITS", "INFLUENCED_BY", "MITIGATED_BY"] | None = None,
) -> dict:
    """Return the complete PM2.5 graph or one validated relation family."""

    try:
        graph = query_pm25_knowledge_graph(relation)
    except KnowledgeGraphError as error:
        raise HTTPException(
            status_code=500,
            detail="PM2.5 knowledge graph is unavailable",
        ) from error
    return {"status": "ok", "graph": graph}
