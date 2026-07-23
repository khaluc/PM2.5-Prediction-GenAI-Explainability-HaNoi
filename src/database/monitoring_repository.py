"""Monitoring repository backed by PostgreSQL with a CSV safety fallback."""

from __future__ import annotations

from datetime import datetime, timezone
import threading
import time
from typing import Any, Callable

import pandas as pd
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError

from src.database.connection import get_database_url, get_session_factory
from src.database.models import (
    AirQualityObservation,
    Alert,
    AnomalyEvent,
    Forecast,
    GenAIExplanation,
    Report,
    Station,
    SystemState,
    TrafficObservation,
    WeatherObservation,
)
from src.services.monitoring_repository import (
    MONITORING_COLUMNS,
    MonitoringRepository,
    StationNotFoundError,
    _python_value,
    normalise_timestamp,
    records_from_frame,
)


class DatabaseMonitoringRepository:
    """Expose the existing monitoring interface while preferring PostgreSQL."""

    def __init__(
        self,
        fallback: MonitoringRepository,
        database_url: str | None = None,
        *,
        timezone_name: str = "Asia/Ho_Chi_Minh",
    ) -> None:
        self.fallback = fallback
        # Keep compatibility with report provenance code that exposes the
        # historical source path even when PostgreSQL is the active backend.
        self.path = fallback.path
        self.database_url = database_url or get_database_url()
        self.timezone_name = timezone_name
        self._last_error: str | None = None
        self._frame_cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl_seconds = 30.0

    def _session(self):
        return get_session_factory(self.database_url)()

    def _database_ready(self) -> bool:
        with self._session() as session:
            value = session.scalar(
                select(SystemState.value).where(SystemState.key == "initial_import")
            )
            return bool(isinstance(value, dict) and value.get("completed"))

    def _with_fallback(self, database_call: Callable[[], Any], fallback_call: Callable[[], Any]):
        try:
            if self._database_ready():
                result = database_call()
                self._last_error = None
                return result
        except SQLAlchemyError as error:
            self._last_error = f"{type(error).__name__}: {error}"[:800]
        return fallback_call()

    def _frame(
        self,
        station_id: str,
        *,
        start: Any | None = None,
        end: Any | None = None,
    ) -> pd.DataFrame:
        use_cache = start is None and end is None
        if use_cache:
            with self._cache_lock:
                cached = self._frame_cache.get(station_id)
                if cached and cached[0] > time.monotonic():
                    return cached[1].copy()
        with self._session() as session:
            air_query = (
                select(
                    AirQualityObservation.station_id,
                    Station.name.label("location_name"),
                    Station.latitude,
                    Station.longitude,
                    AirQualityObservation.observed_at.label("timestamp"),
                    AirQualityObservation.is_forecast,
                    AirQualityObservation.pm25,
                    AirQualityObservation.pm10,
                    AirQualityObservation.co,
                    AirQualityObservation.no2,
                    AirQualityObservation.so2,
                    AirQualityObservation.o3,
                    AirQualityObservation.us_aqi,
                    AirQualityObservation.source.label("air_source"),
                    AirQualityObservation.collected_at.label("air_collected_at"),
                    AirQualityObservation.quality_flags,
                    AirQualityObservation.is_imputed,
                    AirQualityObservation.is_possible_outlier,
                    AirQualityObservation.data_quality_score,
                )
                .join(Station, Station.station_id == AirQualityObservation.station_id)
                .where(
                    AirQualityObservation.station_id == station_id,
                    AirQualityObservation.is_forecast.is_(False),
                )
            )
            weather_query = select(
                WeatherObservation.station_id,
                WeatherObservation.observed_at.label("timestamp"),
                WeatherObservation.temperature,
                WeatherObservation.humidity,
                WeatherObservation.wind_speed,
                WeatherObservation.wind_direction,
                WeatherObservation.precipitation,
                WeatherObservation.rain,
                WeatherObservation.surface_pressure,
                WeatherObservation.cloud_cover,
                WeatherObservation.source.label("weather_source"),
                WeatherObservation.collected_at.label("weather_collected_at"),
            ).where(
                WeatherObservation.station_id == station_id,
                WeatherObservation.is_forecast.is_(False),
            )
            if start is not None:
                lower = normalise_timestamp(start, self.timezone_name).to_pydatetime()
                air_query = air_query.where(AirQualityObservation.observed_at >= lower)
                weather_query = weather_query.where(WeatherObservation.observed_at >= lower)
            if end is not None:
                upper = normalise_timestamp(end, self.timezone_name).to_pydatetime()
                air_query = air_query.where(AirQualityObservation.observed_at <= upper)
                weather_query = weather_query.where(WeatherObservation.observed_at <= upper)
            air = pd.DataFrame(session.execute(air_query).mappings().all())
            if air.empty:
                raise StationNotFoundError(station_id)
            weather = pd.DataFrame(session.execute(weather_query).mappings().all())

        air["timestamp"] = pd.to_datetime(air["timestamp"], utc=True).dt.tz_convert(
            self.timezone_name
        )
        air["air_collected_at"] = pd.to_datetime(
            air["air_collected_at"], errors="coerce", utc=True
        )
        air = air.sort_values(["timestamp", "air_collected_at"], kind="stable").drop_duplicates(
            ["station_id", "timestamp"], keep="last"
        )
        if not weather.empty:
            weather["timestamp"] = pd.to_datetime(
                weather["timestamp"], utc=True
            ).dt.tz_convert(self.timezone_name)
            weather["weather_collected_at"] = pd.to_datetime(
                weather["weather_collected_at"], errors="coerce", utc=True
            )
            weather = weather.sort_values(
                ["timestamp", "weather_collected_at"], kind="stable"
            ).drop_duplicates(["station_id", "timestamp"], keep="last")
            frame = air.merge(weather, on=["station_id", "timestamp"], how="left")
        else:
            frame = air.copy()
            frame["weather_source"] = None
            for column in (
                "temperature", "humidity", "wind_speed", "wind_direction", "precipitation",
                "rain", "surface_pressure", "cloud_cover",
            ):
                frame[column] = None
        for column in MONITORING_COLUMNS:
            if column not in frame:
                frame[column] = None
        result = frame[MONITORING_COLUMNS].sort_values(
            "timestamp", kind="stable"
        ).reset_index(drop=True)
        if use_cache:
            with self._cache_lock:
                self._frame_cache[station_id] = (
                    time.monotonic() + self._cache_ttl_seconds,
                    result,
                )
        return result.copy()

    def invalidate_cache(self) -> None:
        with self._cache_lock:
            self._frame_cache.clear()

    def list_stations(self) -> list[dict[str, Any]]:
        def database() -> list[dict[str, Any]]:
            with self._session() as session:
                aggregates = session.execute(
                    select(
                        AirQualityObservation.station_id,
                        func.min(AirQualityObservation.observed_at).label("first_timestamp"),
                        func.max(AirQualityObservation.observed_at).label("latest_timestamp"),
                        func.count(AirQualityObservation.id).label("observation_count"),
                    )
                    .where(AirQualityObservation.is_forecast.is_(False))
                    .group_by(AirQualityObservation.station_id)
                    .order_by(AirQualityObservation.station_id)
                ).mappings().all()
                stations = {
                    item.station_id: item
                    for item in session.scalars(select(Station)).all()
                }
                results = []
                for aggregate in aggregates:
                    station_id = aggregate["station_id"]
                    latest = session.execute(
                        select(AirQualityObservation.pm25, AirQualityObservation.source)
                        .where(
                            AirQualityObservation.station_id == station_id,
                            AirQualityObservation.is_forecast.is_(False),
                        )
                        .order_by(
                            AirQualityObservation.observed_at.desc(),
                            AirQualityObservation.collected_at.desc().nullslast(),
                        )
                        .limit(1)
                    ).first()
                    station = stations.get(station_id)
                    results.append(
                        {
                            "station_id": station_id,
                            "name": station.name if station else None,
                            "latitude": station.latitude if station else None,
                            "longitude": station.longitude if station else None,
                            "first_timestamp": _python_value(aggregate["first_timestamp"]),
                            "latest_timestamp": _python_value(aggregate["latest_timestamp"]),
                            "observation_count": int(aggregate["observation_count"]),
                            "latest_pm25": _python_value(latest.pm25 if latest else None),
                            "data_source": latest.source if latest else None,
                        }
                    )
                return results

        return self._with_fallback(database, self.fallback.list_stations)

    def latest(self, station_id: str) -> dict[str, Any]:
        return self._with_fallback(
            lambda: records_from_frame(self._frame(station_id).tail(1))[0],
            lambda: self.fallback.latest(station_id),
        )

    def period_frame(
        self,
        station_id: str,
        *,
        start: Any | None = None,
        end: Any | None = None,
    ) -> pd.DataFrame:
        return self._with_fallback(
            lambda: self._frame(station_id, start=start, end=end),
            lambda: self.fallback.period_frame(station_id, start=start, end=end),
        )

    def history(
        self,
        station_id: str,
        *,
        start: Any | None = None,
        end: Any | None = None,
        limit: int = 168,
        offset: int = 0,
        order: str = "asc",
    ) -> dict[str, Any]:
        def database() -> dict[str, Any]:
            selected = self._frame(station_id, start=start, end=end)
            selected = selected.sort_values("timestamp", ascending=order == "asc", kind="stable")
            total = len(selected)
            return {
                "station_id": station_id,
                "start": _python_value(selected["timestamp"].min()) if total else None,
                "end": _python_value(selected["timestamp"].max()) if total else None,
                "total": total,
                "limit": limit,
                "offset": offset,
                "order": order,
                "items": records_from_frame(selected.iloc[offset : offset + limit]),
            }

        return self._with_fallback(
            database,
            lambda: self.fallback.history(
                station_id, start=start, end=end, limit=limit, offset=offset, order=order
            ),
        )

    def health(self) -> dict[str, Any]:
        database_available = False
        database_ready = False
        counts: dict[str, int] = {}
        import_state = None
        try:
            with self._session() as session:
                session.execute(text("SELECT 1"))
                import_state = session.scalar(
                    select(SystemState.value).where(SystemState.key == "initial_import")
                )
                database_ready = bool(
                    isinstance(import_state, dict) and import_state.get("completed")
                )
                for name, model in (
                    ("stations", Station),
                    ("air_quality", AirQualityObservation),
                    ("weather", WeatherObservation),
                    ("traffic", TrafficObservation),
                    ("forecasts", Forecast),
                    ("genai_explanations", GenAIExplanation),
                    ("anomalies", AnomalyEvent),
                    ("alerts", Alert),
                    ("reports", Report),
                ):
                    counts[name] = int(session.scalar(select(func.count()).select_from(model)) or 0)
            database_available = True
            self._last_error = None
        except SQLAlchemyError as error:
            self._last_error = f"{type(error).__name__}: {error}"[:800]
        return {
            "backend": "postgresql" if database_ready else "csv_fallback",
            "available": database_ready or self.fallback.health().get("available", False),
            "database_available": database_available,
            "database_ready": database_ready,
            "database_url": self.database_url.split("@")[-1],
            "database_error": self._last_error,
            "counts": counts,
            "initial_import": import_state,
            "fallback": self.fallback.health(),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
