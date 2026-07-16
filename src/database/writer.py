"""Idempotent PostgreSQL/SQLite writers for pandas frames and ML outputs."""

from __future__ import annotations

import math
import hashlib
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from src.database.connection import get_database_url, session_scope
from src.database.models import (
    AirQualityObservation,
    Alert,
    AnomalyEvent,
    Forecast,
    Report,
    Station,
    SystemState,
    TrafficObservation,
    WeatherObservation,
)


DEFAULT_TIMEZONE = "Asia/Ho_Chi_Minh"


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _timestamp(value: Any, timezone_name: str = DEFAULT_TIMEZONE):
    value = _clean(value)
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone_name)
    return timestamp.to_pydatetime()


def _boolean(value: Any, default: bool = False) -> bool:
    value = _clean(value)
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _json_safe(value: Any) -> Any:
    value = _clean(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return value


def _source(row: dict[str, Any], *columns: str) -> str:
    for column in columns:
        value = _clean(row.get(column))
        if value:
            return str(value)
    return "unknown"


def _batches(records: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(records), size):
        yield records[start : start + size]


class DatabaseWriter:
    """Persist normalized records with ON CONFLICT updates, safe for hourly reruns."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        timezone_name: str = DEFAULT_TIMEZONE,
        batch_size: int = 2_000,
    ) -> None:
        self.database_url = database_url or get_database_url()
        self.timezone_name = timezone_name
        self.batch_size = max(100, int(batch_size))

    def _insert(self, session: Session, model, records: list[dict[str, Any]], unique: list[str]) -> None:
        if not records:
            return
        table = model.__table__
        dialect = session.bind.dialect.name if session.bind is not None else ""
        for batch in _batches(records, self.batch_size):
            if dialect == "postgresql":
                statement = postgresql_insert(table)
            elif dialect == "sqlite":
                statement = sqlite_insert(table)
            else:
                session.execute(table.insert(), batch)
                continue
            immutable = set(unique) | {"id", "created_at"}
            update = {
                column.name: getattr(statement.excluded, column.name)
                for column in table.columns
                if column.name not in immutable and column.name in batch[0]
            }
            upsert = statement.on_conflict_do_update(
                index_elements=[table.c[column] for column in unique],
                set_=update,
            )
            # Parameter sets use the driver's executemany/insertmanyvalues path,
            # avoiding PostgreSQL's 65,535-bind limit for large CSV chunks.
            session.execute(upsert, batch)

    def _station_records(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        results: list[dict[str, Any]] = []
        for station_id, selected in frame.groupby("station_id", sort=False, observed=True):
            timestamps = pd.to_datetime(selected["timestamp"], errors="coerce", utc=True)
            latest = selected.iloc[-1].to_dict()
            results.append(
                {
                    "station_id": str(station_id),
                    "name": _clean(latest.get("location_name")),
                    "latitude": _clean(latest.get("latitude")),
                    "longitude": _clean(latest.get("longitude")),
                    "timezone": self.timezone_name,
                    "first_seen_at": timestamps.min().to_pydatetime() if timestamps.notna().any() else None,
                    "last_seen_at": timestamps.max().to_pydatetime() if timestamps.notna().any() else None,
                }
            )
        return results

    def _upsert_stations(self, session: Session, frame: pd.DataFrame) -> None:
        records = self._station_records(frame)
        if not records:
            return
        table = Station.__table__
        dialect = session.bind.dialect.name if session.bind is not None else ""
        if dialect not in {"postgresql", "sqlite"}:
            self._insert(session, Station, records, ["station_id"])
            return
        statement = (
            postgresql_insert(table).values(records)
            if dialect == "postgresql"
            else sqlite_insert(table).values(records)
        )
        excluded = statement.excluded
        least = func.least if dialect == "postgresql" else func.min
        greatest = func.greatest if dialect == "postgresql" else func.max
        session.execute(
            statement.on_conflict_do_update(
                index_elements=[table.c.station_id],
                set_={
                    "name": func.coalesce(excluded.name, table.c.name),
                    "latitude": func.coalesce(excluded.latitude, table.c.latitude),
                    "longitude": func.coalesce(excluded.longitude, table.c.longitude),
                    "timezone": excluded.timezone,
                    "first_seen_at": least(
                        func.coalesce(table.c.first_seen_at, excluded.first_seen_at),
                        func.coalesce(excluded.first_seen_at, table.c.first_seen_at),
                    ),
                    "last_seen_at": greatest(
                        func.coalesce(table.c.last_seen_at, excluded.last_seen_at),
                        func.coalesce(excluded.last_seen_at, table.c.last_seen_at),
                    ),
                    "updated_at": func.now(),
                },
            )
        )

    def upsert_air_quality(self, frame: pd.DataFrame, *, historical: bool = False) -> int:
        if frame.empty:
            return 0
        records = []
        for row in frame.to_dict(orient="records"):
            observed_at = _timestamp(row.get("timestamp"), self.timezone_name)
            station_id = _clean(row.get("station_id"))
            if station_id is None or observed_at is None:
                continue
            records.append(
                {
                    "station_id": str(station_id),
                    "observed_at": observed_at,
                    "pm25": _clean(row.get("pm25")),
                    "pm10": _clean(row.get("pm10")),
                    "co": _clean(row.get("co")),
                    "no2": _clean(row.get("no2")),
                    "so2": _clean(row.get("so2")),
                    "o3": _clean(row.get("o3")),
                    "us_aqi": _clean(row.get("us_aqi")),
                    "is_forecast": _boolean(row.get("is_forecast")),
                    "source": _source(row, "air_source" if historical else "source", "source"),
                    "collected_at": _timestamp(
                        row.get("air_collected_at" if historical else "collected_at"),
                        self.timezone_name,
                    ),
                    "quality_flags": _clean(
                        row.get("air_quality_flags" if historical else "quality_flags")
                    ),
                    "is_imputed": _boolean(
                        row.get("air_is_imputed" if historical else "is_imputed")
                    ),
                    "is_possible_outlier": _boolean(
                        row.get(
                            "air_is_possible_outlier" if historical else "is_possible_outlier"
                        )
                    ),
                    "data_quality_score": _clean(
                        row.get("air_quality_score" if historical else "data_quality_score")
                    ),
                }
            )
        with session_scope(self.database_url) as session:
            self._upsert_stations(session, frame)
            self._insert(
                session,
                AirQualityObservation,
                records,
                ["station_id", "observed_at", "source"],
            )
        return len(records)

    def upsert_weather(self, frame: pd.DataFrame, *, historical: bool = False) -> int:
        if frame.empty:
            return 0
        records = []
        for row in frame.to_dict(orient="records"):
            observed_at = _timestamp(row.get("timestamp"), self.timezone_name)
            station_id = _clean(row.get("station_id"))
            if station_id is None or observed_at is None:
                continue
            records.append(
                {
                    "station_id": str(station_id),
                    "observed_at": observed_at,
                    "temperature": _clean(row.get("temperature")),
                    "humidity": _clean(row.get("humidity")),
                    "wind_speed": _clean(row.get("wind_speed")),
                    "wind_direction": _clean(row.get("wind_direction")),
                    "precipitation": _clean(row.get("precipitation")),
                    "rain": _clean(row.get("rain")),
                    "surface_pressure": _clean(row.get("surface_pressure")),
                    "cloud_cover": _clean(row.get("cloud_cover")),
                    "is_forecast": _boolean(row.get("is_forecast")),
                    "source": _source(
                        row, "weather_source" if historical else "source", "source"
                    ),
                    "collected_at": _timestamp(
                        row.get("weather_collected_at" if historical else "collected_at"),
                        self.timezone_name,
                    ),
                    "quality_flags": _clean(
                        row.get("weather_quality_flags" if historical else "quality_flags")
                    ),
                    "is_imputed": _boolean(
                        row.get("weather_is_imputed" if historical else "is_imputed")
                    ),
                    "is_possible_outlier": _boolean(
                        row.get(
                            "weather_is_possible_outlier" if historical else "is_possible_outlier"
                        )
                    ),
                    "data_quality_score": _clean(
                        row.get("weather_quality_score" if historical else "data_quality_score")
                    ),
                }
            )
        with session_scope(self.database_url) as session:
            self._upsert_stations(session, frame)
            self._insert(
                session,
                WeatherObservation,
                records,
                ["station_id", "observed_at", "source"],
            )
        return len(records)

    def upsert_monitoring(self, frame: pd.DataFrame) -> dict[str, int]:
        return {
            "air_quality": self.upsert_air_quality(frame, historical=True),
            "weather": self.upsert_weather(frame, historical=True),
        }

    def upsert_traffic(self, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        records = []
        for row in frame.to_dict(orient="records"):
            observed_at = _timestamp(row.get("timestamp"), self.timezone_name)
            station_id = _clean(row.get("station_id"))
            if station_id is None or observed_at is None:
                continue
            records.append(
                {
                    "station_id": str(station_id),
                    "observed_at": observed_at,
                    "current_speed": _clean(row.get("current_speed")),
                    "free_flow_speed": _clean(row.get("free_flow_speed")),
                    "current_travel_time": _clean(row.get("current_travel_time")),
                    "free_flow_travel_time": _clean(row.get("free_flow_travel_time")),
                    "traffic_congestion": _clean(row.get("traffic_congestion")),
                    "confidence": _clean(row.get("confidence")),
                    "road_closure": (
                        None if _clean(row.get("road_closure")) is None else _boolean(row.get("road_closure"))
                    ),
                    "road_class": _clean(row.get("road_class")),
                    "source": _source(row, "source"),
                    "collected_at": _timestamp(row.get("collected_at"), self.timezone_name),
                }
            )
        with session_scope(self.database_url) as session:
            self._upsert_stations(session, frame)
            self._insert(
                session,
                TrafficObservation,
                records,
                ["station_id", "observed_at", "source"],
            )
        return len(records)

    def persist_prediction(
        self, result: dict[str, Any], *, features: dict[str, Any] | None = None
    ) -> dict[str, int]:
        station_id = str(result["station_id"])
        issued_at = _timestamp(result["timestamp"], self.timezone_name)
        model_name = str(result.get("model") or "unknown")
        model_version = str(result.get("model_version") or "current")
        forecasts = []
        for key, value in (result.get("forecast_pm25") or {}).items():
            horizon = int(str(key).lower().replace("h", ""))
            forecasts.append(
                {
                    "station_id": station_id,
                    "issued_at": issued_at,
                    "target_at": issued_at + timedelta(hours=horizon),
                    "horizon_hours": horizon,
                    "predicted_pm25": float(value),
                    "model_name": model_name,
                    "model_version": model_version,
                    "features": _json_safe(features),
                }
            )
        anomaly = result.get("anomaly_detection") or {}
        anomalies = []
        if anomaly.get("available") and issued_at is not None:
            anomalies.append(
                {
                    "station_id": station_id,
                    "observed_at": issued_at,
                    "detector_name": "threshold_and_isolation_forest",
                    "model_version": model_version,
                    "is_anomaly": bool(anomaly.get("is_anomaly")),
                    "is_rule_anomaly": anomaly.get("is_rule_anomaly"),
                    "is_isolation_anomaly": anomaly.get("is_isolation_anomaly"),
                    "score": anomaly.get("isolation_forest_score"),
                    "threshold": anomaly.get("score_threshold"),
                    "reason": anomaly.get("reason"),
                    "details": _json_safe(anomaly),
                }
            )
        with session_scope(self.database_url) as session:
            self._insert(
                session,
                Forecast,
                forecasts,
                ["station_id", "issued_at", "horizon_hours", "model_name", "model_version"],
            )
            self._insert(
                session,
                AnomalyEvent,
                anomalies,
                ["station_id", "observed_at", "detector_name", "model_version"],
            )
        return {"forecasts": len(forecasts), "anomalies": len(anomalies)}

    def set_state(self, key: str, value: dict[str, Any]) -> None:
        with session_scope(self.database_url) as session:
            self._insert(session, SystemState, [{"key": key, "value": value}], ["key"])

    def upsert_alert(self, alert: dict[str, Any]) -> int:
        station_id = str(alert.get("station_id") or "UNKNOWN")
        created_at = _timestamp(
            alert.get("created_at_utc") or alert.get("created_at"), self.timezone_name
        )
        station_frame = pd.DataFrame(
            [
                {
                    "station_id": station_id,
                    "location_name": station_id,
                    "timestamp": created_at,
                }
            ]
        )
        record = {
            "alert_id": str(alert["alert_id"]),
            "station_id": station_id,
            "alert_type": str(alert.get("type") or alert.get("alert_type") or "environment"),
            "severity": str(alert.get("severity") or "warning"),
            "status": str(alert.get("status") or "active"),
            "message": str(alert.get("title_vi") or alert.get("message") or "Environmental alert"),
            "payload": _json_safe(alert),
            "created_at": created_at,
            "acknowledged_at": _timestamp(
                alert.get("acknowledged_at_utc") or alert.get("acknowledged_at"),
                self.timezone_name,
            ),
            "acknowledged_by": _clean(alert.get("acknowledged_by")),
        }
        with session_scope(self.database_url) as session:
            self._upsert_stations(session, station_frame)
            self._insert(session, Alert, [record], ["alert_id"])
        return 1

    def persist_report(
        self,
        result: dict[str, Any],
        *,
        station_id: str,
        period_start: Any,
        period_end: Any,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        output_path = result.get("output_path")
        checksum = None
        if output_path:
            path = Path(str(output_path))
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[2] / path
            if path.is_file():
                checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        record = {
            "report_id": str(result["report_id"]),
            "station_id": station_id,
            "period_start": _timestamp(period_start, self.timezone_name),
            "period_end": _timestamp(period_end, self.timezone_name),
            "output_format": str(result.get("format") or "json"),
            "file_path": str(output_path) if output_path else None,
            "checksum": checksum,
            "metadata": _json_safe(metadata),
        }
        with session_scope(self.database_url) as session:
            self._insert(session, Report, [record], ["report_id"])
        return 1

    def get_state(self, key: str) -> dict[str, Any] | None:
        with session_scope(self.database_url) as session:
            value = session.scalar(select(SystemState.value).where(SystemState.key == key))
            return value if isinstance(value, dict) else None

    def observation_count(self) -> int:
        with session_scope(self.database_url) as session:
            return int(session.scalar(select(func.count()).select_from(AirQualityObservation)) or 0)
