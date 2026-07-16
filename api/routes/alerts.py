"""Alert query, evaluation and acknowledgement endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_alert_store
from api.schemas import AlertAcknowledgeRequest, AlertEvaluationRequest
from src.alerts.alert_service import (
    AlertNotFoundError,
    FileAlertStore,
    create_alert,
)
from src.database.alert_store import DatabaseAlertStore


router = APIRouter(tags=["alerts"])


@router.get("/alerts")
def list_alerts(
    station_id: str | None = None,
    status: Literal["active", "acknowledged"] | None = None,
    severity: Literal["warning", "critical"] | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    store: FileAlertStore | DatabaseAlertStore = Depends(get_alert_store),
) -> dict:
    return store.list(
        station_id=station_id,
        status=status,
        severity=severity,
        limit=limit,
        offset=offset,
    )


@router.post("/alerts/evaluate")
def evaluate_alert(
    request: AlertEvaluationRequest,
    store: FileAlertStore | DatabaseAlertStore = Depends(get_alert_store),
) -> dict:
    alert = create_alert(request.environment_result, store=store)
    return {"created": alert is not None, "alert": alert}


@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(
    alert_id: str,
    request: AlertAcknowledgeRequest,
    store: FileAlertStore | DatabaseAlertStore = Depends(get_alert_store),
) -> dict:
    try:
        return store.acknowledge(alert_id, acknowledged_by=request.acknowledged_by)
    except AlertNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"Alert not found: {error}") from error
