"""Current and historical monitoring endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_monitoring_repository
from api.schemas import StationSummary
from src.services.monitoring_repository import (
    DataSourceUnavailableError,
    MonitoringRepository,
    StationNotFoundError,
    normalise_timestamp,
)
from src.database.monitoring_repository import DatabaseMonitoringRepository


router = APIRouter(tags=["monitoring"])


def _translate_error(error: Exception) -> HTTPException:
    if isinstance(error, StationNotFoundError):
        return HTTPException(status_code=404, detail=f"Station not found: {error}")
    return HTTPException(status_code=503, detail=str(error))


@router.get("/stations", response_model=list[StationSummary])
def list_stations(
    repository: MonitoringRepository | DatabaseMonitoringRepository = Depends(get_monitoring_repository),
) -> list[dict]:
    try:
        return repository.list_stations()
    except DataSourceUnavailableError as error:
        raise _translate_error(error) from error


@router.get("/stations/{station_id}/latest")
def station_latest(
    station_id: str,
    repository: MonitoringRepository | DatabaseMonitoringRepository = Depends(get_monitoring_repository),
) -> dict:
    try:
        return repository.latest(station_id)
    except (StationNotFoundError, DataSourceUnavailableError) as error:
        raise _translate_error(error) from error


@router.get("/stations/{station_id}/history")
def station_history(
    station_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = Query(default=168, ge=1, le=2_000),
    offset: int = Query(default=0, ge=0),
    order: Literal["asc", "desc"] = "asc",
    repository: MonitoringRepository | DatabaseMonitoringRepository = Depends(get_monitoring_repository),
) -> dict:
    if start and end and normalise_timestamp(start) > normalise_timestamp(end):
        raise HTTPException(status_code=422, detail="start must be earlier than or equal to end")
    try:
        return repository.history(
            station_id,
            start=start,
            end=end,
            limit=limit,
            offset=offset,
            order=order,
        )
    except (StationNotFoundError, DataSourceUnavailableError) as error:
        raise _translate_error(error) from error
