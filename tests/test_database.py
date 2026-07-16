"""Database schema, idempotent writer and CSV fallback integration tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import func, select

from src.database.connection import get_database_url, get_engine, get_session_factory
from src.alerts.alert_service import FileAlertStore
from src.database.alert_store import DatabaseAlertStore
from src.database.models import (
    AirQualityObservation,
    Base,
    Forecast,
    Report,
    WeatherObservation,
)
from src.database.monitoring_repository import DatabaseMonitoringRepository
from src.database.writer import DatabaseWriter
from src.services.monitoring_repository import MONITORING_COLUMNS, MonitoringRepository


def _database_url(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'environment.db').as_posix()}"


def _frame(pm25: float = 42.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": "2026-07-15T19:00:00+07:00",
                "station_id": "HN_TEST",
                "location_name": "Hanoi Test",
                "latitude": 21.0285,
                "longitude": 105.8542,
                "is_forecast": False,
                "pm25": pm25,
                "pm10": 51.0,
                "co": 400.0,
                "no2": 25.0,
                "so2": 8.0,
                "o3": 60.0,
                "us_aqi": 117.0,
                "temperature": 31.0,
                "humidity": 75.0,
                "wind_speed": 8.0,
                "wind_direction": 180.0,
                "precipitation": 0.0,
                "rain": 0.0,
                "surface_pressure": 1001.0,
                "cloud_cover": 45.0,
                "air_source": "test_air",
                "weather_source": "test_weather",
                "quality_flags": None,
                "is_imputed": False,
                "is_possible_outlier": False,
                "data_quality_score": 1.0,
            }
        ]
    )


def _fallback_csv(path: Path) -> Path:
    row = _frame(pm25=5.0).iloc[0].to_dict()
    row.setdefault("weather_source", "fallback_weather")
    pd.DataFrame([row], columns=MONITORING_COLUMNS).to_csv(path, index=False)
    return path


def test_schema_and_writer_are_idempotent(tmp_path: Path) -> None:
    database_url = _database_url(tmp_path)
    Base.metadata.create_all(get_engine(database_url))
    writer = DatabaseWriter(database_url)

    assert writer.upsert_monitoring(_frame()) == {"air_quality": 1, "weather": 1}
    writer.upsert_monitoring(_frame(pm25=47.5))

    with get_session_factory(database_url)() as session:
        assert session.scalar(select(func.count()).select_from(AirQualityObservation)) == 1
        assert session.scalar(select(func.count()).select_from(WeatherObservation)) == 1
        assert session.scalar(select(AirQualityObservation.pm25)) == 47.5


def test_database_url_is_safely_built_from_container_environment(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_HOST", "db")
    monkeypatch.setenv("DATABASE_USER", "environment")
    monkeypatch.setenv("DATABASE_PASSWORD", "secret@with:characters")
    monkeypatch.setenv("DATABASE_NAME", "environment")
    url = get_database_url()
    assert "db:5432/environment" in url
    assert "secret%40with%3Acharacters" in url
    assert "secret@with:characters" not in url


def test_database_repository_uses_database_after_completed_import(tmp_path: Path) -> None:
    database_url = _database_url(tmp_path)
    Base.metadata.create_all(get_engine(database_url))
    writer = DatabaseWriter(database_url)
    writer.upsert_monitoring(_frame(pm25=47.5))
    writer.set_state("initial_import", {"completed": True})

    repository = DatabaseMonitoringRepository(
        MonitoringRepository(_fallback_csv(tmp_path / "fallback.csv")),
        database_url,
    )
    assert repository.latest("HN_TEST")["pm25"] == 47.5
    assert repository.list_stations()[0]["station_id"] == "HN_TEST"
    assert repository.health()["backend"] == "postgresql"


def test_database_repository_falls_back_before_import(tmp_path: Path) -> None:
    database_url = _database_url(tmp_path)
    Base.metadata.create_all(get_engine(database_url))
    repository = DatabaseMonitoringRepository(
        MonitoringRepository(_fallback_csv(tmp_path / "fallback.csv")),
        database_url,
    )
    assert repository.latest("HN_TEST")["pm25"] == 5.0
    assert repository.health()["backend"] == "csv_fallback"


def test_prediction_upsert_does_not_duplicate(tmp_path: Path) -> None:
    database_url = _database_url(tmp_path)
    Base.metadata.create_all(get_engine(database_url))
    writer = DatabaseWriter(database_url)
    writer.upsert_monitoring(_frame())
    result = {
        "station_id": "HN_TEST",
        "timestamp": "2026-07-15T19:00:00+07:00",
        "model": "LightGBM",
        "forecast_pm25": {"1h": 50.0, "3h": 55.0, "6h": 58.0},
        "anomaly_detection": {"available": False},
    }
    writer.persist_prediction(result, features={"timestamp": pd.Timestamp("2026-07-15")})
    writer.persist_prediction(result, features={"timestamp": pd.Timestamp("2026-07-15")})
    with get_session_factory(database_url)() as session:
        assert session.scalar(select(func.count()).select_from(Forecast)) == 3


def test_database_alert_store_upserts_lists_and_acknowledges(tmp_path: Path) -> None:
    database_url = _database_url(tmp_path)
    Base.metadata.create_all(get_engine(database_url))
    writer = DatabaseWriter(database_url)
    writer.upsert_monitoring(_frame())
    writer.set_state("initial_import", {"completed": True})
    store = DatabaseAlertStore(FileAlertStore(tmp_path / "alerts.json"), database_url)
    alert = {
        "alert_id": "ALT-TEST",
        "station_id": "HN_TEST",
        "created_at_utc": "2026-07-15T12:00:00+00:00",
        "type": "pollution_episode",
        "severity": "warning",
        "status": "active",
        "title_vi": "Test alert",
    }
    store.upsert(alert)
    store.upsert(alert)
    assert store.list(station_id="HN_TEST")["total"] == 1
    acknowledged = store.acknowledge("ALT-TEST", acknowledged_by="operator")
    assert acknowledged["status"] == "acknowledged"


def test_report_metadata_is_idempotently_persisted(tmp_path: Path) -> None:
    database_url = _database_url(tmp_path)
    Base.metadata.create_all(get_engine(database_url))
    writer = DatabaseWriter(database_url)
    writer.upsert_monitoring(_frame())
    result = {"report_id": "RPT-TEST", "format": "pdf", "output_path": None}
    kwargs = {
        "station_id": "HN_TEST",
        "period_start": "2026-07-15T18:00:00+07:00",
        "period_end": "2026-07-15T19:00:00+07:00",
        "metadata": {"metrics": {"pm25_mean": 42.0}},
    }
    writer.persist_report(result, **kwargs)
    writer.persist_report(result, **kwargs)
    with get_session_factory(database_url)() as session:
        assert session.scalar(select(func.count()).select_from(Report)) == 1
