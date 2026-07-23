"""Integration tests for the FastAPI backend."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.dependencies import (
    get_alert_store,
    get_explanation_cache,
    get_hourly_update_service,
    get_monitoring_repository,
    get_news_crawler,
    get_traffic_repository,
)
from api.main import app
from src.alerts.alert_service import FileAlertStore
from src.genai.explanation_cache import ForecastExplanationCache
from src.services.monitoring_repository import MONITORING_COLUMNS, MonitoringRepository


class FakeTrafficRepository:
    def latest_near(self, station_id, reference_timestamp):
        return {
            "available": True,
            "station_id": station_id,
            "reference_timestamp": reference_timestamp,
            "current_speed_kmh": 20.0,
            "free_flow_speed_kmh": 40.0,
            "congestion_ratio": 0.5,
            "confidence": 0.95,
            "source": "tomtom_flow_segment",
        }


class FakeNewsCrawler:
    def fetch(self, category, *, page, limit, force_refresh):
        return {
            "source": {"name": "Môi Trường Thủ Đô", "url": "https://moitruongthudo.vn/thong-tin"},
            "category": category,
            "category_label": "Tin mới",
            "page": page,
            "limit": limit,
            "total": 1,
            "has_next": True,
            "fetched_at": "2026-07-14T01:00:00Z",
            "cache_status": "live",
            "stale": False,
            "items": [
                {
                    "id": "news-1",
                    "title": "Tin môi trường thử nghiệm",
                    "excerpt": "Mô tả ngắn.",
                    "published_date": "2026-07-14",
                    "date_display": "14/07/2026",
                    "url": "https://moitruongthudo.vn/thong-tin/1/test",
                    "image_url": None,
                    "source": "Môi Trường Thủ Đô",
                }
            ],
        }


class FakeHourlyUpdateService:
    enabled = True

    def status(self):
        return {
            "enabled": True,
            "running": False,
            "last_success_at": "2026-07-14T00:01:00+07:00",
            "next_run_at": "2026-07-14T01:00:45+07:00",
            "last_result": {"forecast_refresh": {"succeeded": 2}},
            "last_error": None,
        }

    def trigger_now(self):
        return True


def _monitoring_csv(path: Path) -> Path:
    defaults = {
        "location_name": "Test Station",
        "latitude": 21.0,
        "longitude": 105.8,
        "is_forecast": False,
        "pm25": 20.0,
        "pm10": 30.0,
        "co": 300.0,
        "no2": 20.0,
        "so2": 5.0,
        "o3": 40.0,
        "us_aqi": 50.0,
        "temperature": 25.0,
        "humidity": 70.0,
        "wind_speed": 5.0,
        "wind_direction": 180.0,
        "precipitation": 0.0,
        "surface_pressure": 1005.0,
        "cloud_cover": 50.0,
        "air_source": "test_air",
        "weather_source": "test_weather",
        "quality_flags": None,
        "is_imputed": False,
        "is_possible_outlier": False,
        "data_quality_score": 1.0,
    }
    rows = []
    for station_id, timestamp, pm25 in [
        ("ST_A", "2026-05-01T00:00:00+07:00", 20.0),
        ("ST_A", "2026-05-01T01:00:00+07:00", 40.0),
        ("ST_A", "2026-05-01T02:00:00+07:00", 60.0),
        ("ST_B", "2026-05-01T00:00:00+07:00", 30.0),
    ]:
        rows.append({**defaults, "station_id": station_id, "timestamp": timestamp, "pm25": pm25})
    pd.DataFrame(rows, columns=MONITORING_COLUMNS).to_csv(path, index=False)
    return path


def _live_csvs(tmp_path: Path) -> tuple[Path, Path]:
    air_path = tmp_path / "live_air.csv"
    weather_path = tmp_path / "live_weather.csv"
    common = {
        "station_id": "ST_A",
        "location_name": "Test Station",
        "latitude": 21.0,
        "longitude": 105.8,
        "source": "live_test",
        "collected_at": "2026-07-13T04:15:00+00:00",
    }
    pd.DataFrame(
        [
            {
                **common,
                "timestamp": "2026-07-13T10:00:00+07:00",
                "pm25": 71.0,
                "is_forecast": False,
            },
            {
                **common,
                "timestamp": "2026-07-13T11:00:00+07:00",
                "pm25": 76.0,
                "is_forecast": True,
            },
        ]
    ).to_csv(air_path, index=False)
    pd.DataFrame(
        [
            {
                **common,
                "timestamp": "2026-07-13T10:00:00+07:00",
                "humidity": 80.0,
                "is_forecast": False,
            },
            {
                **common,
                "timestamp": "2026-07-13T11:00:00+07:00",
                "humidity": 82.0,
                "is_forecast": True,
            },
        ]
    ).to_csv(weather_path, index=False)
    return air_path, weather_path


@pytest.fixture
def api_client(tmp_path: Path):
    repository = MonitoringRepository(_monitoring_csv(tmp_path / "monitoring.csv"))
    alert_store = FileAlertStore(tmp_path / "alerts.json")
    app.dependency_overrides[get_monitoring_repository] = lambda: repository
    app.dependency_overrides[get_alert_store] = lambda: alert_store
    app.dependency_overrides[get_traffic_repository] = lambda: FakeTrafficRepository()
    app.dependency_overrides[get_news_crawler] = lambda: FakeNewsCrawler()
    app.dependency_overrides[get_hourly_update_service] = lambda: FakeHourlyUpdateService()
    explanation_cache = ForecastExplanationCache(database_enabled=False)
    app.dependency_overrides[get_explanation_cache] = lambda: explanation_cache
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_root_redirects_to_swagger_docs() -> None:
    response = TestClient(app).get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_openapi_contains_required_routes() -> None:
    paths = TestClient(app).get("/openapi.json").json()["paths"]
    required = {
        "/stations",
        "/stations/{station_id}/latest",
        "/stations/{station_id}/history",
        "/predict",
        "/detect-anomaly",
        "/forecast-explanation",
        "/knowledge-graph/pm25",
        "/alerts",
        "/reports/generate",
        "/news",
        "/system/hourly-update",
        "/system/hourly-update/run",
        "/system/database",
    }
    assert required <= set(paths)


def test_pm25_knowledge_graph_route_can_filter_relations() -> None:
    client = TestClient(app)
    response = client.get("/knowledge-graph/pm25", params={"relation": "EMITS"})
    assert response.status_code == 200
    graph = response.json()["graph"]
    assert graph["relation_filter"] == "EMITS"
    assert len(graph["edges"]) == 11
    assert {edge["relation"] for edge in graph["edges"]} == {"EMITS"}
    assert client.get(
        "/knowledge-graph/pm25", params={"relation": "UNKNOWN"}
    ).status_code == 422


def test_hourly_update_status_and_manual_trigger(api_client: TestClient) -> None:
    status_response = api_client.get("/system/hourly-update")
    assert status_response.status_code == 200
    assert status_response.json()["enabled"] is True
    assert status_response.json()["last_result"]["forecast_refresh"]["succeeded"] == 2
    trigger = api_client.post("/system/hourly-update/run")
    assert trigger.status_code == 202
    assert trigger.json()["accepted"] is True


def test_news_route_validates_filters_and_returns_attributed_items(api_client: TestClient) -> None:
    response = api_client.get(
        "/news",
        params={"category": "domestic", "page": 2, "limit": 8, "refresh": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["category"] == "domestic"
    assert payload["page"] == 2
    assert payload["source"]["name"] == "Môi Trường Thủ Đô"
    assert payload["items"][0]["url"].startswith("https://moitruongthudo.vn/")
    assert api_client.get("/news", params={"category": "unsupported"}).status_code == 422
    assert api_client.get("/news", params={"page": 11}).status_code == 422


def test_station_list_latest_and_paginated_history(api_client: TestClient) -> None:
    stations = api_client.get("/stations")
    assert stations.status_code == 200
    assert [item["station_id"] for item in stations.json()] == ["ST_A", "ST_B"]
    latest = api_client.get("/stations/ST_A/latest")
    assert latest.status_code == 200
    assert latest.json()["pm25"] == 60.0
    history = api_client.get(
        "/stations/ST_A/history",
        params={"limit": 1, "offset": 1, "order": "desc"},
    )
    assert history.status_code == 200
    assert history.json()["total"] == 3
    assert history.json()["items"][0]["pm25"] == 40.0
    assert api_client.get("/stations/UNKNOWN/latest").status_code == 404


def test_monitoring_repository_overlays_live_actual_and_excludes_future_rows(
    tmp_path: Path,
) -> None:
    air_path, weather_path = _live_csvs(tmp_path)
    repository = MonitoringRepository(
        _monitoring_csv(tmp_path / "monitoring.csv"),
        live_air_path=air_path,
        live_weather_path=weather_path,
    )
    latest = repository.latest("ST_A")
    assert latest["timestamp"] == "2026-07-13T10:00:00+07:00"
    assert latest["pm25"] == 71.0
    assert latest["humidity"] == 80.0
    history = repository.history("ST_A", order="desc")
    assert history["items"][0]["is_forecast"] is False
    assert all(item["timestamp"] != "2026-07-13T11:00:00+07:00" for item in history["items"])


def test_predict_and_anomaly_routes_accept_explicit_features(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "api.routes.predictions.predict_from_features",
        lambda features: {"station_id": features["station_id"], "forecast_pm25": {"1h": 55.0}},
    )
    monkeypatch.setattr(
        "api.routes.predictions.detect_anomaly_from_features",
        lambda features: {
            "station_id": features["station_id"],
            "is_anomaly": False,
            "requires_attention": False,
        },
    )
    prediction = api_client.post("/predict", json={"features": {"station_id": "ST_A", "pm25": 50}})
    assert prediction.status_code == 200
    assert prediction.json()["result"]["forecast_pm25"]["1h"] == 55.0
    monkeypatch.setattr(
        "api.routes.predictions.build_latest_feature_row",
        lambda frame: {"station_id": str(frame["station_id"].iloc[-1]), "pm25": 60},
    )
    live_prediction = api_client.post("/predict", json={"station_id": "ST_A"})
    assert live_prediction.status_code == 200
    assert live_prediction.json()["input_source"] == "latest_observation_features"
    anomaly = api_client.post(
        "/detect-anomaly", json={"features": {"station_id": "ST_A", "pm25": 50}}
    )
    assert anomaly.status_code == 200
    assert anomaly.json()["result"]["is_anomaly"] is False
    assert api_client.post("/predict", json={}).status_code == 422


def test_forecast_explanation_uses_latest_ml_prediction(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "api.routes.explanations.build_latest_feature_row",
        lambda frame: {"station_id": "ST_A", "pm25": 60.0},
    )
    monkeypatch.setattr(
        "api.routes.explanations.predict_from_features",
        lambda features: {"station_id": "ST_A", "forecast_pm25": {"3h": 72.0}},
    )
    monkeypatch.setattr(
        "api.routes.explanations.explain_forecast",
        lambda prediction, horizon_hours, use_llm, traffic: {
            "station_id": prediction["station_id"],
            "forecast": {"horizon_hours": horizon_hours, "predicted_pm25": 72.0},
            "grounding": {"traffic": traffic},
            "generation": {"mode": "deterministic_fallback"},
        },
    )
    response = api_client.post(
        "/forecast-explanation",
        json={"station_id": "ST_A", "horizon_hours": 3, "use_llm": False},
    )
    assert response.status_code == 200
    assert response.json()["result"]["forecast"]["predicted_pm25"] == 72.0
    assert response.json()["result"]["grounding"]["traffic"]["source"] == "tomtom_flow_segment"
    assert response.json()["cache"]["status"] == "bypass"
    assert response.json()["result"]["generation"]["cache_status"] == "bypass"
    assert api_client.post(
        "/forecast-explanation", json={"station_id": "ST_A", "horizon_hours": 2}
    ).status_code == 422


def test_forecast_explanation_reuses_one_hour_result(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    monkeypatch.setattr(
        "api.routes.explanations.build_latest_feature_row",
        lambda frame: {"station_id": "ST_A", "pm25": 60.0},
    )
    monkeypatch.setattr(
        "api.routes.explanations.predict_from_features",
        lambda features: {
            "station_id": "ST_A",
            "timestamp": "2026-07-23T10:00:00+07:00",
            "forecast_pm25": {"1h": 61.0},
        },
    )

    def explain(prediction, horizon_hours, use_llm, traffic):
        nonlocal calls
        calls += 1
        return {
            "station_id": prediction["station_id"],
            "forecast": {
                "horizon_hours": horizon_hours,
                "predicted_pm25": 61.0,
            },
            "generation": {
                "mode": "dashscope",
                "provider_model": "dashscope:deepseek-v4-flash",
                "fallback_reason": None,
            },
        }

    monkeypatch.setattr("api.routes.explanations.explain_forecast", explain)
    payload = {"station_id": "ST_A", "horizon_hours": 1, "use_llm": True}
    first = api_client.post("/forecast-explanation", json=payload)
    second = api_client.post("/forecast-explanation", json=payload)

    assert first.status_code == second.status_code == 200
    assert calls == 1
    assert first.json()["cache"]["status"] == "miss"
    assert second.json()["cache"]["status"] == "hit"
    assert second.json()["result"]["generation"]["mode"] == "dashscope"


def test_alert_create_deduplicate_and_acknowledge(api_client: TestClient) -> None:
    payload = {
        "environment_result": {
            "station_id": "ST_A",
            "timestamp": "2026-05-01T02:00:00+07:00",
            "current_measurements": {"pm25": 92},
            "anomaly_detection": {"requires_attention": True, "reason": "test"},
        }
    }
    first = api_client.post("/alerts/evaluate", json=payload).json()["alert"]
    second = api_client.post("/alerts/evaluate", json=payload).json()["alert"]
    assert first["alert_id"] == second["alert_id"]
    listed = api_client.get("/alerts", params={"status": "active"}).json()
    assert listed["total"] == 1
    acknowledged = api_client.post(
        f"/alerts/{first['alert_id']}/acknowledge",
        json={"acknowledged_by": "operator-01"},
    )
    assert acknowledged.status_code == 200
    assert acknowledged.json()["status"] == "acknowledged"


def test_generate_deterministic_report_without_persisting(api_client: TestClient) -> None:
    response = api_client.post(
        "/reports/generate",
        json={"station_id": "ST_A", "format": "json", "persist": False},
    )
    assert response.status_code == 200
    report = response.json()
    assert report["persisted"] is False
    assert report["content"]["metrics"]["pm25_mean"] == 40.0
    assert report["content"]["period"]["observation_count"] == 3


def test_generate_and_download_pdf_report(
    api_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_dir = tmp_path / "pdf"
    monkeypatch.setattr("src.services.report_service.DEFAULT_PDF_REPORT_DIR", pdf_dir)
    monkeypatch.setattr("api.routes.reports.DEFAULT_PDF_REPORT_DIR", pdf_dir)
    generated = api_client.post(
        "/reports/generate",
        json={"station_id": "ST_A", "format": "pdf", "persist": True},
    )
    assert generated.status_code == 200
    payload = generated.json()
    assert payload["format"] == "pdf"
    assert payload["output_path"].endswith(".pdf")
    assert payload["download_url"].endswith("/download")
    target = pdf_dir / f"{payload['report_id']}.pdf"
    assert target.read_bytes().startswith(b"%PDF-")

    downloaded = api_client.get(payload["download_url"])
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "application/pdf"
    assert "attachment" in downloaded.headers["content-disposition"]
    assert downloaded.content.startswith(b"%PDF-")
    assert api_client.get("/reports/not-safe/download").status_code == 404
