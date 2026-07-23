"""Reusable, override-friendly FastAPI dependencies."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

from src.alerts.alert_service import DEFAULT_ALERT_PATH, FileAlertStore
from src.collection.news_collector import DEFAULT_NEWS_CACHE_PATH, NewsCrawler
from src.database.alert_store import DatabaseAlertStore
from src.database.connection import environment_flag, get_database_url
from src.database.monitoring_repository import DatabaseMonitoringRepository
from src.database.writer import DatabaseWriter
from src.genai.explanation_cache import ForecastExplanationCache
from src.services.monitoring_repository import (
    DEFAULT_FEATURE_PATH,
    DEFAULT_LIVE_AIR_PATH,
    DEFAULT_LIVE_WEATHER_PATH,
    DEFAULT_MONITORING_PATH,
    FeatureRepository,
    MonitoringRepository,
)
from src.services.traffic_repository import DEFAULT_TRAFFIC_PATH, TrafficRepository
from src.services.hourly_update_service import HourlyUpdateService
from src.services.inference_service import predict_from_features
from src.services.live_feature_service import build_latest_feature_row


load_dotenv(override=False)


@lru_cache(maxsize=1)
def get_monitoring_repository() -> MonitoringRepository | DatabaseMonitoringRepository:
    fallback = MonitoringRepository(
        os.getenv("MONITORING_DATA_PATH", str(DEFAULT_MONITORING_PATH)),
        live_air_path=os.getenv("LIVE_AIR_DATA_PATH", str(DEFAULT_LIVE_AIR_PATH)),
        live_weather_path=os.getenv("LIVE_WEATHER_DATA_PATH", str(DEFAULT_LIVE_WEATHER_PATH)),
    )
    if environment_flag("DATABASE_READ_ENABLED", False) and "PYTEST_CURRENT_TEST" not in os.environ:
        return DatabaseMonitoringRepository(fallback, get_database_url())
    return fallback


@lru_cache(maxsize=1)
def get_database_writer() -> DatabaseWriter:
    return DatabaseWriter(
        get_database_url(),
        batch_size=int(os.getenv("DATABASE_BATCH_SIZE", "2000")),
    )


@lru_cache(maxsize=1)
def get_feature_repository() -> FeatureRepository:
    return FeatureRepository(os.getenv("FEATURE_DATA_PATH", str(DEFAULT_FEATURE_PATH)))


@lru_cache(maxsize=1)
def get_alert_store() -> FileAlertStore | DatabaseAlertStore:
    fallback = FileAlertStore(os.getenv("ALERT_STORE_PATH", str(DEFAULT_ALERT_PATH)))
    if environment_flag("DATABASE_READ_ENABLED", False) and "PYTEST_CURRENT_TEST" not in os.environ:
        return DatabaseAlertStore(fallback, get_database_url())
    return fallback


@lru_cache(maxsize=1)
def get_traffic_repository() -> TrafficRepository:
    return TrafficRepository(
        os.getenv("LIVE_TRAFFIC_DATA_PATH", str(DEFAULT_TRAFFIC_PATH)),
        max_age_minutes=float(os.getenv("TRAFFIC_MAX_AGE_MINUTES", "120")),
    )


@lru_cache(maxsize=1)
def get_explanation_cache() -> ForecastExplanationCache:
    database_enabled = (
        environment_flag("DATABASE_READ_ENABLED", False)
        and environment_flag("DATABASE_WRITE_ENABLED", False)
        and "PYTEST_CURRENT_TEST" not in os.environ
    )
    return ForecastExplanationCache(
        get_database_url(),
        database_enabled=database_enabled,
        success_ttl_seconds=int(
            os.getenv("GENAI_SUCCESS_CACHE_TTL_SECONDS", "7200")
        ),
        fallback_ttl_seconds=int(
            os.getenv("GENAI_FALLBACK_CACHE_TTL_SECONDS", "300")
        ),
        cache_version=os.getenv(
            "GENAI_EXPLANATION_CACHE_VERSION",
            "hourly-explanation-v1",
        ),
    )


@lru_cache(maxsize=1)
def get_news_crawler() -> NewsCrawler:
    return NewsCrawler(
        os.getenv("NEWS_CACHE_PATH", str(DEFAULT_NEWS_CACHE_PATH)),
        cache_ttl_seconds=int(os.getenv("NEWS_CACHE_TTL_SECONDS", "1800")),
        timeout_seconds=float(os.getenv("NEWS_TIMEOUT_SECONDS", "20")),
    )


def _refresh_hourly_forecasts() -> dict:
    """Run the trained model once per station after new observations arrive."""

    repository = get_monitoring_repository()
    invalidate = getattr(repository, "invalidate_cache", None)
    if callable(invalidate):
        invalidate()
    stations = repository.list_stations()
    refreshed: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for station in stations:
        station_id = str(station["station_id"])
        try:
            features = build_latest_feature_row(repository.period_frame(station_id))
            result = predict_from_features(features)
            database_persisted = None
            if environment_flag("DATABASE_WRITE_ENABLED", False):
                database_persisted = get_database_writer().persist_prediction(
                    result, features=features
                )
            refreshed[station_id] = {
                "timestamp": result.get("timestamp"),
                "model": result.get("model"),
                "forecast_pm25": result.get("forecast_pm25"),
                "database": database_persisted,
            }
        except Exception as error:  # Keep refreshing other stations on a partial failure.
            failures[station_id] = str(error)[:400]
    return {
        "attempted": len(stations),
        "succeeded": len(refreshed),
        "failed": len(failures),
        "stations": refreshed,
        "errors": failures,
    }


@lru_cache(maxsize=1)
def get_hourly_update_service() -> HourlyUpdateService:
    return HourlyUpdateService.from_environment(
        forecast_refresher=_refresh_hourly_forecasts,
    )
