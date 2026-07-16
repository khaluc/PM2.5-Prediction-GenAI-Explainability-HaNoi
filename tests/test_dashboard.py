"""Tests for the Flask dashboard and its FastAPI proxy."""

from __future__ import annotations

import httpx

from dashboard.api_client import DashboardApiClient, DashboardApiError
from dashboard.app import create_app


class FakeEnvironmentApi:
    def __init__(self) -> None:
        self.calls = []

    def request(self, method, path, *, params=None, json=None):
        self.calls.append({"method": method, "path": path, "params": params, "json": json})
        if path == "/health":
            return {"status": "ok"}
        if path == "/stations":
            return [
                {
                    "station_id": "ST_A",
                    "name": "Ba Dinh",
                    "latitude": 21.03,
                    "longitude": 105.83,
                    "latest_pm25": 40,
                }
            ]
        if path == "/system/hourly-update":
            return {
                "enabled": True,
                "running": False,
                "last_success_at": "2026-07-14T00:01:00+07:00",
                "next_run_at": "2026-07-14T01:00:45+07:00",
                "last_error": None,
            }
        if path == "/stations/ST_A/latest":
            return {"station_id": "ST_A", "timestamp": "2026-05-30T23:00:00+07:00", "pm25": 40}
        if path == "/stations/ST_A/history":
            return {"items": [{"timestamp": "2026-05-30T23:00:00+07:00", "pm25": 40}], "total": 1}
        if path == "/predict":
            return {"result": {"model": "LightGBM", "forecast_pm25": {"1h": 42, "3h": 45, "6h": 48}, "anomaly_detection": {"is_anomaly": False, "detection_source": "none"}}}
        if path == "/detect-anomaly":
            return {"result": {"is_anomaly": False, "detection_source": "none"}}
        if path == "/alerts":
            return {"items": [], "total": 0}
        if path == "/reports/generate":
            return {"report_id": "RPT-1", "content": "# Report", "output_path": "artifacts/reports/RPT-1.md"}
        if path == "/forecast-explanation":
            return {
                "status": "ok",
                "result": {
                    "forecast": {"model": "LightGBM", "horizon_hours": 3, "predicted_pm25": 45},
                    "explanation": {"headline": "PM2.5 +3 giờ"},
                    "generation": {"mode": "deterministic_fallback"},
                },
            }
        if path == "/news":
            return {
                "source": {"name": "Môi Trường Thủ Đô", "url": "https://moitruongthudo.vn/thong-tin"},
                "category": params["category"],
                "category_label": "Tin mới",
                "page": params["page"],
                "limit": params["limit"],
                "total": 1,
                "has_next": True,
                "fetched_at": "2026-07-14T01:00:00Z",
                "cache_status": "live",
                "stale": False,
                "items": [{"title": "Tin môi trường", "url": "https://moitruongthudo.vn/thong-tin/1/test"}],
            }
        raise AssertionError(f"Unexpected request: {method} {path}")

    def download(self, path):
        self.calls.append({"method": "GET", "path": path, "params": None, "json": None})
        if path == "/reports/RPT-0000000000000001/download":
            return b"%PDF-1.7\n% test", "application/pdf", "environment-report.pdf"
        raise AssertionError(f"Unexpected download: {path}")


def test_dashboard_page_has_required_sections() -> None:
    app = create_app(api_client=FakeEnvironmentApi(), testing=True)
    response = app.test_client().get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    for text in [
        "Chỉ số hiện tại",
        "Diễn biến PM2.5",
        "Dự báo PM2.5",
        "Phát hiện bất thường",
        "Bản đồ khu vực Hà Nội",
        "Cảnh báo gần đây",
        "Tạo báo cáo",
        "Giải thích dự báo ML",
    ]:
        assert text.lower() in html.lower()
    assert "vendor/leaflet/leaflet.css" in html
    assert "vendor/leaflet/leaflet.js" in html
    assert 'id="station-map"' in html
    assert 'class="pm25-hero pm-unknown"' in html
    assert 'class="particle-layer"' in html
    assert 'class="lung-illustration"' in html
    assert 'class="pm25-scale"' in html
    assert 'id="chart-minimum"' in html
    assert 'id="chart-maximum"' in html
    assert "Mốc 15 µg/m³" in html
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    assert "https://tile.openstreetmap.org" in response.headers["Content-Security-Policy"]
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    css = app.test_client().get("/static/css/dashboard.css").get_data(as_text=True)
    script = app.test_client().get("/static/js/dashboard.js").get_data(as_text=True)
    assert "@keyframes breathe" in css
    assert "prefers-reduced-motion: reduce" in css
    assert "function pm25ScalePosition" in script
    assert "function animateMetric" in script
    assert "pm-chart-bar" in script
    assert "chart-threshold" in script
    assert "chart-tooltip" in script
    assert "syncHourlyUpdateStatus" in script
    assert "30_000" in script
    assert "nextBrowserHourlyRefresh" in script


def test_dashboard_snapshot_aggregates_backend_calls() -> None:
    fake = FakeEnvironmentApi()
    client = create_app(api_client=fake, testing=True).test_client()
    assert client.get("/api/stations").status_code == 200
    response = client.get("/api/stations/ST_A/snapshot")
    assert response.status_code == 200
    body = response.get_json()
    assert body["latest"]["pm25"] == 40
    assert body["prediction"]["forecast_pm25"]["6h"] == 48
    assert body["anomaly"]["is_anomaly"] is False
    assert any(call["path"] == "/stations/ST_A/history" and call["params"]["limit"] == 24 for call in fake.calls)
    assert not any(call["path"].endswith("/forecast") for call in fake.calls)
    assert not any(call["path"] == "/detect-anomaly" for call in fake.calls)
    assert client.get("/api/stations/not.valid/snapshot").status_code == 400


def test_dashboard_proxies_hourly_update_status() -> None:
    fake = FakeEnvironmentApi()
    client = create_app(api_client=fake, testing=True).test_client()
    response = client.get("/api/system/hourly-update")
    assert response.status_code == 200
    assert response.get_json()["enabled"] is True
    assert response.get_json()["next_run_at"].endswith("+07:00")


def test_news_page_and_proxy_are_available_and_source_attributed() -> None:
    fake = FakeEnvironmentApi()
    client = create_app(api_client=fake, testing=True).test_client()
    page = client.get("/news")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "Tin môi trường" in html
    assert "Trong nước" in html
    assert "Quốc tế" in html
    assert "Môi Trường Thủ Đô" in html
    assert 'id="featured-news"' in html
    assert "https://moitruongthudo.vn" in page.headers["Content-Security-Policy"]

    feed = client.get("/api/news?category=domestic&page=2&limit=8&refresh=true")
    assert feed.status_code == 200
    assert feed.get_json()["category"] == "domestic"
    call = fake.calls[-1]
    assert call["path"] == "/news"
    assert call["params"] == {
        "category": "domestic",
        "page": 2,
        "limit": 8,
        "refresh": True,
    }
    assert client.get("/api/news?category=unknown").status_code == 400
    assert client.get("/api/news?page=zero").status_code == 400


def test_dashboard_report_validates_and_proxies() -> None:
    fake = FakeEnvironmentApi()
    client = create_app(api_client=fake, testing=True).test_client()
    report = client.post(
        "/api/reports",
        json={
            "station_id": "ST_A",
            "start": "2026-05-30T00:00:00+07:00",
            "end": "2026-05-30T23:00:00+07:00",
            "format": "markdown",
        },
    )
    assert report.status_code == 200
    assert report.get_json()["report_id"] == "RPT-1"


def test_dashboard_downloads_valid_pdf_report() -> None:
    fake = FakeEnvironmentApi()
    client = create_app(api_client=fake, testing=True).test_client()
    response = client.get("/api/reports/RPT-0000000000000001/download")
    assert response.status_code == 200
    assert response.content_type == "application/pdf"
    assert response.data.startswith(b"%PDF-")
    assert "attachment" in response.headers["Content-Disposition"]
    assert client.get("/api/reports/unsafe/download").status_code == 400

    page = client.get("/").get_data(as_text=True)
    assert "Tạo báo cáo PDF" in page
    assert 'id="download-report"' in page
    script = client.get("/static/js/dashboard.js").get_data(as_text=True)
    assert 'format: "pdf"' in script
    assert "downloadLink.click()" in script


def test_dashboard_forecast_explanation_validates_and_proxies() -> None:
    fake = FakeEnvironmentApi()
    client = create_app(api_client=fake, testing=True).test_client()
    response = client.post(
        "/api/forecast-explanation",
        json={"station_id": "ST_A", "horizon_hours": 3, "use_llm": True},
    )
    assert response.status_code == 200
    assert response.get_json()["result"]["forecast"]["predicted_pm25"] == 45
    call = fake.calls[-1]
    assert call["path"] == "/forecast-explanation"
    assert call["json"] == {"station_id": "ST_A", "horizon_hours": 3, "use_llm": True}
    assert client.post(
        "/api/forecast-explanation", json={"station_id": "ST_A", "horizon_hours": 2}
    ).status_code == 400


def test_dashboard_api_client_sanitizes_connection_and_backend_errors() -> None:
    denied = DashboardApiClient(
        "http://api",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(502, json={"detail": "Provider unavailable"})
            )
        ),
    )
    try:
        denied.request("GET", "/test")
    except DashboardApiError as error:
        assert error.status_code == 502
        assert error.detail == "Provider unavailable"
    else:
        raise AssertionError("Expected DashboardApiError")


def test_dashboard_api_client_downloads_binary_and_sanitizes_filename() -> None:
    client = DashboardApiClient(
        "http://api",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    content=b"%PDF-1.7\n",
                    headers={
                        "content-type": "application/pdf",
                        "content-disposition": 'attachment; filename="report.pdf"',
                    },
                )
            )
        ),
    )
    content, content_type, filename = client.download("/reports/RPT/download")
    assert content.startswith(b"%PDF-")
    assert content_type == "application/pdf"
    assert filename == "report.pdf"
