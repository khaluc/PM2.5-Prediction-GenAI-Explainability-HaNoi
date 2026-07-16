"""Create and persist deduplicated environmental alerts."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALERT_PATH = PROJECT_ROOT / "artifacts" / "alerts" / "alerts.json"


class AlertNotFoundError(LookupError):
    pass


class FileAlertStore:
    def __init__(self, path: str | Path = DEFAULT_ALERT_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Alert store must contain a JSON array: {self.path}")
        return payload

    def _write(self, alerts: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(alerts, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def upsert(self, alert: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            alerts = self._read()
            for existing in alerts:
                if existing.get("deduplication_key") == alert.get("deduplication_key") and existing.get("status") == "active":
                    return existing
            alerts.append(alert)
            self._write(alerts)
        return alert

    def list(
        self,
        *,
        station_id: str | None = None,
        status: str | None = None,
        severity: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        alerts = self._read()
        if station_id:
            alerts = [item for item in alerts if item.get("station_id") == station_id]
        if status:
            alerts = [item for item in alerts if item.get("status") == status]
        if severity:
            alerts = [item for item in alerts if item.get("severity") == severity]
        alerts.sort(key=lambda item: item.get("created_at_utc", ""), reverse=True)
        return {
            "total": len(alerts),
            "limit": limit,
            "offset": offset,
            "items": alerts[offset : offset + limit],
        }

    def acknowledge(self, alert_id: str, *, acknowledged_by: str) -> dict[str, Any]:
        with self._lock:
            alerts = self._read()
            for alert in alerts:
                if alert.get("alert_id") == alert_id:
                    alert["status"] = "acknowledged"
                    alert["acknowledged_by"] = acknowledged_by
                    alert["acknowledged_at_utc"] = datetime.now(timezone.utc).isoformat()
                    self._write(alerts)
                    return alert
        raise AlertNotFoundError(alert_id)


def _forecast_max(result: dict[str, Any]) -> float | None:
    forecast = result.get("forecast_pm25") or result.get("forecast") or {}
    values = []
    if isinstance(forecast, dict):
        for value in forecast.values():
            if isinstance(value, (int, float)):
                values.append(float(value))
            elif isinstance(value, dict) and isinstance(value.get("pm25"), (int, float)):
                values.append(float(value["pm25"]))
    return max(values) if values else None


def create_alert(
    environment_result: dict[str, Any],
    *,
    store: FileAlertStore | None = None,
) -> dict[str, Any] | None:
    """Create an alert only when evidence says the event requires attention."""
    anomaly = environment_result.get("anomaly_detection") or environment_result.get("anomaly") or environment_result
    requires_attention = bool(anomaly.get("requires_attention") or anomaly.get("is_anomaly"))
    current = environment_result.get("current") or environment_result.get("current_measurements") or {}
    current_pm25 = current.get("pm25")
    forecast_pm25 = _forecast_max(environment_result)
    high_pollution = any(
        value is not None and float(value) >= 75.0 for value in (current_pm25, forecast_pm25)
    )
    if not requires_attention and not high_pollution:
        return None

    station_id = str(environment_result.get("station_id") or current.get("station_id") or "UNKNOWN")
    event_timestamp = str(environment_result.get("timestamp") or current.get("timestamp") or "unknown")
    event_hour = event_timestamp[:13]
    alert_type = "pollution_episode" if high_pollution else "data_anomaly"
    peak = max(
        [float(value) for value in (current_pm25, forecast_pm25) if value is not None],
        default=None,
    )
    severity = "critical" if peak is not None and peak >= 150 else "warning"
    deduplication_key = f"{station_id}|{alert_type}|{event_hour}"
    created_at = datetime.now(timezone.utc).isoformat()
    alert = {
        "alert_id": "ALT-" + hashlib.sha256(deduplication_key.encode("utf-8")).hexdigest()[:16].upper(),
        "deduplication_key": deduplication_key,
        "station_id": station_id,
        "event_timestamp": event_timestamp,
        "created_at_utc": created_at,
        "type": alert_type,
        "severity": severity,
        "status": "active",
        "title_vi": "Cảnh báo ô nhiễm cần xác minh" if high_pollution else "Dữ liệu bất thường cần kiểm tra",
        "evidence": {
            "current_pm25": current_pm25,
            "maximum_forecast_pm25": forecast_pm25,
            "requires_attention": requires_attention,
            "detection_source": anomaly.get("detection_source"),
            "reason": anomaly.get("reason"),
        },
        "automatic_operational_action_executed": False,
    }
    return (store or FileAlertStore()).upsert(alert)
