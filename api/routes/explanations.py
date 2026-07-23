"""Controlled GenAI explanation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import (
    get_explanation_cache,
    get_monitoring_repository,
    get_traffic_repository,
)
from api.routes.predictions import _input_error
from api.schemas import ForecastExplanationRequest
from src.genai.explanation_cache import ForecastExplanationCache
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
    explanation_cache: ForecastExplanationCache = Depends(get_explanation_cache),
) -> dict:
    try:
        features = build_latest_feature_row(repository.period_frame(request.station_id))
        prediction = predict_from_features(features)
        traffic = traffic_repository.latest_near(
            request.station_id,
            prediction.get("timestamp"),
        )
        result, cache = explanation_cache.resolve(
            station_id=request.station_id,
            forecast_issued_at=prediction.get("timestamp"),
            horizon_hours=request.horizon_hours,
            use_llm=request.use_llm,
            generator=lambda: explain_forecast(
                prediction,
                request.horizon_hours,
                use_llm=request.use_llm,
                traffic=traffic,
            ),
        )
        generation = result.get("generation")
        if isinstance(generation, dict):
            generation["cache_status"] = cache["status"]
            generation["cache_backend"] = cache["backend"]
        return {
            "status": "ok",
            "input_source": "latest_observation_features",
            "cache": cache,
            "result": result,
        }
    except (
        ValueError,
        FileNotFoundError,
        StationNotFoundError,
        DataSourceUnavailableError,
    ) as error:
        raise _input_error(error) from error
