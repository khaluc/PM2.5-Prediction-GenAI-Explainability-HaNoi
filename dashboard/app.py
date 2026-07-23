"""Flask dashboard for the Environment AI FastAPI backend."""

from __future__ import annotations

import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from flask import Flask, current_app, jsonify, render_template, request, send_file

from dashboard.api_client import DashboardApiClient, DashboardApiError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{2,64}$")
REPORT_ID_PATTERN = re.compile(r"^RPT-[A-F0-9]{16}$")
NEWS_CATEGORIES = {"latest", "domestic", "international"}
KNOWLEDGE_RELATIONS = {"EMITS", "INFLUENCED_BY", "MITIGATED_BY"}


def _client() -> DashboardApiClient:
    return current_app.extensions["environment_api_client"]


def _valid_station_id(station_id: str) -> bool:
    return bool(STATION_ID_PATTERN.fullmatch(station_id))


def create_app(*, api_client: DashboardApiClient | Any | None = None, testing: bool = False) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(
        TESTING=testing,
        JSON_SORT_KEYS=False,
        ENVIRONMENT_API_URL=os.getenv("ENVIRONMENT_API_URL", "http://127.0.0.1:8000"),
    )
    app.extensions["environment_api_client"] = api_client or DashboardApiClient(
        app.config["ENVIRONMENT_API_URL"]
    )

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "style-src-elem 'self'; style-src-attr 'unsafe-inline'; "
            "img-src 'self' data: blob: https://tile.openstreetmap.org "
            "https://moitruongthudo.vn https://www.moitruongthudo.vn; "
            "connect-src 'self'; font-src 'self'"
        )
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(DashboardApiError)
    def backend_error(error: DashboardApiError):
        return jsonify({"error": "backend_error", "detail": error.detail}), error.status_code

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/news")
    def news_page():
        return render_template("news.html")

    @app.get("/knowledge-graph")
    def knowledge_graph_page():
        return render_template("knowledge_graph.html")

    @app.get("/health")
    def health():
        backend = _client().request("GET", "/health")
        return jsonify({"status": "ok", "backend": backend})

    @app.get("/api/stations")
    def stations():
        return jsonify(_client().request("GET", "/stations"))

    @app.get("/api/system/hourly-update")
    def hourly_update_status():
        return jsonify(_client().request("GET", "/system/hourly-update"))

    @app.get("/api/news")
    def news_feed():
        category = str(request.args.get("category") or "latest").strip().lower()
        if category not in NEWS_CATEGORIES:
            return jsonify({"error": "invalid_news_category"}), 400
        try:
            page = int(request.args.get("page", "1"))
            limit = int(request.args.get("limit", "12"))
        except (TypeError, ValueError):
            return jsonify({"error": "invalid_news_pagination"}), 400
        if not 1 <= page <= 10 or not 1 <= limit <= 30:
            return jsonify({"error": "invalid_news_pagination"}), 400
        refresh = str(request.args.get("refresh") or "false").lower() in {
            "1",
            "true",
            "yes",
        }
        return jsonify(
            _client().request(
                "GET",
                "/news",
                params={
                    "category": category,
                    "page": page,
                    "limit": limit,
                    "refresh": refresh,
                },
            )
        )

    @app.get("/api/stations/<station_id>/snapshot")
    def station_snapshot(station_id: str):
        if not _valid_station_id(station_id):
            return jsonify({"error": "invalid_station_id"}), 400
        client = _client()
        latest = client.request("GET", f"/stations/{station_id}/latest")
        history = client.request(
            "GET",
            f"/stations/{station_id}/history",
            params={"limit": 24, "order": "desc"},
        )
        try:
            prediction = client.request("POST", "/predict", json={"station_id": station_id})
            prediction = prediction.get("result", prediction)
        except DashboardApiError as error:
            prediction = {"available": False, "reason": error.detail}
        anomaly = prediction.get("anomaly_detection")
        if not isinstance(anomaly, dict):
            try:
                anomaly = client.request("POST", "/detect-anomaly", json={"station_id": station_id})
                anomaly = anomaly.get("result", anomaly)
            except DashboardApiError as error:
                anomaly = {"available": False, "reason": error.detail}
        alerts = client.request(
            "GET",
            "/alerts",
            params={"station_id": station_id, "limit": 20},
        )
        return jsonify(
            {
                "station_id": station_id,
                "latest": latest,
                "history": history,
                "prediction": prediction,
                "anomaly": anomaly,
                "alerts": alerts,
            }
        )

    @app.post("/api/reports")
    def reports():
        payload = request.get_json(silent=True) or {}
        station_id = str(payload.get("station_id") or "").strip()
        if not _valid_station_id(station_id):
            return jsonify({"error": "invalid_station_id"}), 400
        output_format = str(payload.get("format") or "markdown")
        if output_format not in {"json", "markdown", "pdf"}:
            return jsonify({"error": "invalid_format"}), 400
        backend_payload = {
            "station_id": station_id,
            "start": payload.get("start"),
            "end": payload.get("end"),
            "format": output_format,
            "persist": bool(payload.get("persist", True)),
        }
        return jsonify(_client().request("POST", "/reports/generate", json=backend_payload))

    @app.get("/api/reports/<report_id>/download")
    def report_download(report_id: str):
        if not REPORT_ID_PATTERN.fullmatch(report_id):
            return jsonify({"error": "invalid_report_id"}), 400
        content, content_type, filename = _client().download(
            f"/reports/{report_id}/download"
        )
        if content_type != "application/pdf" or not content.startswith(b"%PDF-"):
            return jsonify({"error": "invalid_report_file"}), 502
        return send_file(
            BytesIO(content),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename or f"environment-ai-{report_id}.pdf",
            max_age=0,
        )

    @app.post("/api/forecast-explanation")
    def forecast_explanation():
        payload = request.get_json(silent=True) or {}
        station_id = str(payload.get("station_id") or "").strip()
        if not _valid_station_id(station_id):
            return jsonify({"error": "invalid_station_id"}), 400
        try:
            horizon_hours = int(payload.get("horizon_hours", 1))
        except (TypeError, ValueError):
            return jsonify({"error": "invalid_horizon_hours"}), 400
        if horizon_hours != 1:
            return jsonify({"error": "invalid_horizon_hours"}), 400
        backend_payload = {
            "station_id": station_id,
            "horizon_hours": horizon_hours,
            "use_llm": bool(payload.get("use_llm", True)),
        }
        return jsonify(
            _client().request("POST", "/forecast-explanation", json=backend_payload)
        )

    @app.get("/api/knowledge-graph/pm25")
    def pm25_knowledge_graph():
        relation = str(request.args.get("relation") or "").strip().upper()
        if relation and relation not in KNOWLEDGE_RELATIONS:
            return jsonify({"error": "invalid_knowledge_relation"}), 400
        params = {"relation": relation} if relation else None
        return jsonify(
            _client().request("GET", "/knowledge-graph/pm25", params=params)
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("DASHBOARD_PORT", "8501")), debug=False)
