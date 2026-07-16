"""Forecasting and anomaly inference endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_alert_store, get_monitoring_repository
from api.schemas import AnomalyRequest, PredictRequest
from src.alerts.alert_service import FileAlertStore, create_alert
from src.services.inference_service import detect_anomaly_from_features, predict_from_features
from src.services.live_feature_service import build_latest_feature_row
from src.services.monitoring_repository import (
    DataSourceUnavailableError,
    MonitoringRepository,
    StationNotFoundError,
)


router = APIRouter(tags=["inference"])


def _features(
    request: PredictRequest | AnomalyRequest,
    repository: MonitoringRepository,
) -> tuple[dict[str, Any], str]:
    if request.features:
        features = dict(request.features)
        if request.station_id:
            features.setdefault("station_id", request.station_id)
        return features, "request"
    return (
        build_latest_feature_row(repository.period_frame(str(request.station_id))),
        "latest_observation_features",
    )


def _input_error(error: Exception) -> HTTPException:
    if isinstance(error, StationNotFoundError):
        return HTTPException(status_code=404, detail=f"Station not found: {error}")
    if isinstance(error, DataSourceUnavailableError):
        return HTTPException(status_code=503, detail=str(error))
    if isinstance(error, FileNotFoundError):
        return HTTPException(status_code=503, detail=str(error))
    return HTTPException(status_code=422, detail=str(error))


@router.post("/predict")
def predict(
    request: PredictRequest,
    repository: MonitoringRepository = Depends(get_monitoring_repository),
    alert_store: FileAlertStore = Depends(get_alert_store),
) -> dict:
    try:
        features, input_source = _features(request, repository)
        result = predict_from_features(features)
        alert = create_alert(result, store=alert_store) if request.create_alert else None
        return {"status": "ok", "input_source": input_source, "result": result, "alert": alert}
    except (
        ValueError,
        FileNotFoundError,
        StationNotFoundError,
        DataSourceUnavailableError,
    ) as error:
        raise _input_error(error) from error


@router.post("/detect-anomaly")
def detect_anomaly(
    request: AnomalyRequest,
    repository: MonitoringRepository = Depends(get_monitoring_repository),
    alert_store: FileAlertStore = Depends(get_alert_store),
) -> dict:
    try:
        features, input_source = _features(request, repository)
        result = detect_anomaly_from_features(features)
        alert = create_alert(result, store=alert_store) if request.create_alert else None
        return {"status": "ok", "input_source": input_source, "result": result, "alert": alert}
    except (
        ValueError,
        FileNotFoundError,
        StationNotFoundError,
        DataSourceUnavailableError,
    ) as error:
        raise _input_error(error) from error
