"""Controlled GenAI explanation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_monitoring_repository, get_traffic_repository
from api.routes.predictions import _input_error
from api.schemas import ForecastExplanationRequest
from src.genai.forecast_explainer import explain_forecast
from src.services.inference_service import predict_from_features
from src.services.live_feature_service import build_latest_feature_row
from src.services.monitoring_repository import (
    DataSourceUnavailableError,
    MonitoringRepository,
    StationNotFoundError,
)
from src.services.traffic_repository import TrafficRepository


router = APIRouter(tags=["genai"])


@router.post("/forecast-explanation")
def forecast_explanation(
    request: ForecastExplanationRequest,
    repository: MonitoringRepository = Depends(get_monitoring_repository),
    traffic_repository: TrafficRepository = Depends(get_traffic_repository),
) -> dict:
    try:
        features = build_latest_feature_row(repository.period_frame(request.station_id))
        prediction = predict_from_features(features)
        traffic = traffic_repository.latest_near(
            request.station_id,
            prediction.get("timestamp"),
        )
        result = explain_forecast(
            prediction,
            request.horizon_hours,
            use_llm=request.use_llm,
            traffic=traffic,
        )
        return {
            "status": "ok",
            "input_source": "latest_observation_features",
            "result": result,
        }
    except (
        ValueError,
        FileNotFoundError,
        StationNotFoundError,
        DataSourceUnavailableError,
    ) as error:
        raise _input_error(error) from error
