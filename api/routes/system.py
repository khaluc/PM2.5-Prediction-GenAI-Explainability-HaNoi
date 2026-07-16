"""Runtime status and controlled operations for hourly environment updates."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_hourly_update_service, get_monitoring_repository
from src.database.monitoring_repository import DatabaseMonitoringRepository
from src.services.hourly_update_service import HourlyUpdateService
from src.services.monitoring_repository import MonitoringRepository


router = APIRouter(prefix="/system", tags=["system"])


@router.get("/database")
def database_status(
    repository: MonitoringRepository | DatabaseMonitoringRepository = Depends(
        get_monitoring_repository
    ),
) -> dict:
    """Return active storage backend, readiness and row counts."""

    return repository.health()


@router.get("/hourly-update")
def hourly_update_status(
    service: HourlyUpdateService = Depends(get_hourly_update_service),
) -> dict:
    return service.status()


@router.post("/hourly-update/run", status_code=status.HTTP_202_ACCEPTED)
def run_hourly_update(
    service: HourlyUpdateService = Depends(get_hourly_update_service),
) -> dict:
    if not service.enabled:
        raise HTTPException(status_code=409, detail="Hourly updates are disabled")
    accepted = service.trigger_now()
    return {"accepted": accepted, "status": service.status()}
