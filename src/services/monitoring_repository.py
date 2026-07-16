"""Cached, read-only access to processed monitoring and feature data."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MONITORING_PATH = PROJECT_ROOT / "data" / "processed" / "air_quality_clean.csv"
DEFAULT_FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "ml_features.csv.gz"
DEFAULT_LIVE_AIR_PATH = PROJECT_ROOT / "data" / "raw" / "air_quality.csv"
DEFAULT_LIVE_WEATHER_PATH = PROJECT_ROOT / "data" / "raw" / "weather.csv"

MONITORING_COLUMNS = [
    "timestamp",
    "station_id",
    "location_name",
    "latitude",
    "longitude",
    "is_forecast",
    "pm25",
    "pm10",
    "co",
    "no2",
    "so2",
    "o3",
    "us_aqi",
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "rain",
    "surface_pressure",
    "cloud_cover",
    "air_source",
    "weather_source",
    "quality_flags",
    "is_imputed",
    "is_possible_outlier",
    "data_quality_score",
]


class DataSourceUnavailableError(RuntimeError):
    pass


class StationNotFoundError(LookupError):
    pass


class StaleFeatureError(RuntimeError):
    def __init__(self, station_id: str, timestamp: Any, max_age_hours: float) -> None:
        self.station_id = station_id
        self.timestamp = timestamp
        self.max_age_hours = max_age_hours
        super().__init__(
            f"Latest engineered features for {station_id} are stale "
            f"({timestamp}); maximum age is {max_age_hours:g} hours. "
            "Rebuild live features or provide an explicit fresh feature row."
        )


def _python_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def records_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {column: _python_value(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def normalise_timestamp(value: Any, timezone_name: str = "Asia/Ho_Chi_Minh") -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone_name)
    return timestamp.tz_convert(timezone_name)


class CachedCsvRepository:
    """Reload a CSV only when its modification time changes."""

    def __init__(
        self,
        path: str | Path,
        *,
        usecols: Iterable[str] | None = None,
        timezone_name: str = "Asia/Ho_Chi_Minh",
    ) -> None:
        self.path = Path(path)
        self.usecols = list(usecols) if usecols is not None else None
        self.timezone_name = timezone_name
        self._frame: pd.DataFrame | None = None
        self._mtime_ns: int | None = None
        self._lock = threading.Lock()

    def _load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise DataSourceUnavailableError(f"Data file not found: {self.path}")
        modified = self.path.stat().st_mtime_ns
        if self._frame is not None and modified == self._mtime_ns:
            return self._frame
        with self._lock:
            if self._frame is not None and modified == self._mtime_ns:
                return self._frame
            frame = pd.read_csv(self.path, usecols=self.usecols, low_memory=False)
            if "timestamp" not in frame or "station_id" not in frame:
                raise DataSourceUnavailableError(
                    f"Data file must contain timestamp and station_id: {self.path}"
                )
            timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
            if timestamps.isna().any():
                raise DataSourceUnavailableError(f"Invalid timestamps in {self.path}")
            frame["timestamp"] = timestamps.dt.tz_convert(self.timezone_name)
            frame["station_id"] = frame["station_id"].astype(str)
            frame = frame.sort_values(["station_id", "timestamp"], kind="stable").reset_index(drop=True)
            self._frame = frame
            self._mtime_ns = modified
        return self._frame

    def health(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "available": self.path.exists(),
            "loaded": self._frame is not None,
            "modified_at_ns": self.path.stat().st_mtime_ns if self.path.exists() else None,
        }


class MonitoringRepository(CachedCsvRepository):
    """Combine immutable training history with current provider observations.

    The cleaned historical file remains untouched. Only rows marked as actual are
    overlaid at read time, so provider forecasts cannot leak into monitoring or ML
    inference.
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_MONITORING_PATH,
        *,
        live_air_path: str | Path | None = None,
        live_weather_path: str | Path | None = None,
    ) -> None:
        super().__init__(path, usecols=MONITORING_COLUMNS)
        self.live_air_path = Path(live_air_path) if live_air_path else None
        self.live_weather_path = Path(live_weather_path) if live_weather_path else None
        self._combined_frame: pd.DataFrame | None = None
        self._combined_signature: tuple[int | None, tuple[int | None, int | None] | None] | None = None
        self._live_frame: pd.DataFrame | None = None
        self._live_signature: tuple[int | None, int | None] | None = None
        self._live_lock = threading.Lock()

    @staticmethod
    def _boolean_series(values: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(values):
            return values.fillna(False)
        return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

    def _file_mtime(self, path: Path | None) -> int | None:
        return path.stat().st_mtime_ns if path is not None and path.exists() else None

    def _read_live(self) -> pd.DataFrame:
        signature = (
            self._file_mtime(self.live_air_path),
            self._file_mtime(self.live_weather_path),
        )
        if self._live_frame is not None and signature == self._live_signature:
            return self._live_frame
        with self._live_lock:
            if self._live_frame is not None and signature == self._live_signature:
                return self._live_frame
            if self.live_air_path is None or not self.live_air_path.exists():
                frame = pd.DataFrame(columns=MONITORING_COLUMNS)
            else:
                air = pd.read_csv(self.live_air_path, low_memory=False)
                required = {"timestamp", "station_id", "is_forecast"}
                if not required.issubset(air.columns):
                    raise DataSourceUnavailableError(
                        f"Live air file is missing required columns: {self.live_air_path}"
                    )
                air["timestamp"] = pd.to_datetime(air["timestamp"], errors="coerce", utc=True)
                if air["timestamp"].isna().any():
                    raise DataSourceUnavailableError(
                        f"Invalid timestamps in {self.live_air_path}"
                    )
                air["timestamp"] = air["timestamp"].dt.tz_convert(self.timezone_name)
                air["station_id"] = air["station_id"].astype(str)
                air["is_forecast"] = self._boolean_series(air["is_forecast"])
                if "collected_at" in air:
                    air = air.sort_values("collected_at", kind="stable")
                air = air.drop_duplicates(["station_id", "timestamp"], keep="last")

                weather_columns = [
                    "timestamp",
                    "station_id",
                    "temperature",
                    "humidity",
                    "wind_speed",
                    "wind_direction",
                    "precipitation",
                    "rain",
                    "surface_pressure",
                    "cloud_cover",
                    "source",
                ]
                if self.live_weather_path is not None and self.live_weather_path.exists():
                    weather = pd.read_csv(self.live_weather_path, low_memory=False)
                    missing = {"timestamp", "station_id"} - set(weather.columns)
                    if missing:
                        raise DataSourceUnavailableError(
                            f"Live weather file is missing required columns: {self.live_weather_path}"
                        )
                    weather["timestamp"] = pd.to_datetime(
                        weather["timestamp"], errors="coerce", utc=True
                    )
                    if weather["timestamp"].isna().any():
                        raise DataSourceUnavailableError(
                            f"Invalid timestamps in {self.live_weather_path}"
                        )
                    weather["timestamp"] = weather["timestamp"].dt.tz_convert(
                        self.timezone_name
                    )
                    weather["station_id"] = weather["station_id"].astype(str)
                    if "collected_at" in weather:
                        weather = weather.sort_values("collected_at", kind="stable")
                    weather = weather.drop_duplicates(["station_id", "timestamp"], keep="last")
                    weather = weather[
                        [column for column in weather_columns if column in weather.columns]
                    ].rename(columns={"source": "weather_source"})
                    frame = air.merge(
                        weather,
                        on=["timestamp", "station_id"],
                        how="left",
                        validate="one_to_one",
                    )
                else:
                    frame = air.copy()
                    frame["weather_source"] = None

                frame = frame.rename(columns={"source": "air_source"})
                defaults: dict[str, Any] = {
                    "quality_flags": None,
                    "is_imputed": False,
                    "is_possible_outlier": False,
                    "data_quality_score": None,
                }
                for column in MONITORING_COLUMNS:
                    if column not in frame:
                        frame[column] = defaults.get(column)
                frame = frame[MONITORING_COLUMNS].sort_values(
                    ["station_id", "timestamp"], kind="stable"
                ).reset_index(drop=True)
            self._live_frame = frame
            self._live_signature = signature
            self._combined_frame = None
            self._combined_signature = None
        return self._live_frame

    def _load(self) -> pd.DataFrame:
        historical = super()._load()
        live = self._read_live()
        actual = live[~live["is_forecast"]] if not live.empty else live
        if actual.empty:
            return historical
        signature = (self._mtime_ns, self._live_signature)
        if self._combined_frame is None or signature != self._combined_signature:
            combined = pd.concat([historical, actual], ignore_index=True)
            combined = combined.drop_duplicates(
                ["station_id", "timestamp"], keep="last"
            ).sort_values(["station_id", "timestamp"], kind="stable")
            self._combined_frame = combined.reset_index(drop=True)
            self._combined_signature = signature
        return self._combined_frame

    def list_stations(self) -> list[dict[str, Any]]:
        frame = self._load()
        grouped = frame.groupby("station_id", sort=True, observed=True)
        results = []
        for station_id, station_frame in grouped:
            latest = station_frame.iloc[-1]
            results.append(
                {
                    "station_id": station_id,
                    "name": _python_value(latest.get("location_name")),
                    "latitude": _python_value(latest.get("latitude")),
                    "longitude": _python_value(latest.get("longitude")),
                    "first_timestamp": _python_value(station_frame["timestamp"].iloc[0]),
                    "latest_timestamp": _python_value(latest["timestamp"]),
                    "observation_count": int(len(station_frame)),
                    "latest_pm25": _python_value(latest.get("pm25")),
                    "data_source": _python_value(latest.get("air_source")),
                }
            )
        return results

    def _station_frame(self, station_id: str) -> pd.DataFrame:
        frame = self._load()
        selected = frame[frame["station_id"] == station_id]
        if selected.empty:
            raise StationNotFoundError(station_id)
        return selected

    def latest(self, station_id: str) -> dict[str, Any]:
        return records_from_frame(self._station_frame(station_id).tail(1))[0]

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
        selected = self.period_frame(station_id, start=start, end=end)
        selected = selected.sort_values("timestamp", ascending=order == "asc", kind="stable")
        total = int(len(selected))
        page = selected.iloc[offset : offset + limit]
        return {
            "station_id": station_id,
            "start": _python_value(selected["timestamp"].min()) if total else None,
            "end": _python_value(selected["timestamp"].max()) if total else None,
            "total": total,
            "limit": limit,
            "offset": offset,
            "order": order,
            "items": records_from_frame(page),
        }

    def period_frame(
        self,
        station_id: str,
        *,
        start: Any | None = None,
        end: Any | None = None,
    ) -> pd.DataFrame:
        selected = self._station_frame(station_id)
        if start is not None:
            selected = selected[selected["timestamp"] >= normalise_timestamp(start, self.timezone_name)]
        if end is not None:
            selected = selected[selected["timestamp"] <= normalise_timestamp(end, self.timezone_name)]
        return selected.copy()

    def health(self) -> dict[str, Any]:
        result = super().health()
        result["live_air_path"] = str(self.live_air_path) if self.live_air_path else None
        result["live_air_available"] = bool(
            self.live_air_path is not None and self.live_air_path.exists()
        )
        result["live_weather_path"] = (
            str(self.live_weather_path) if self.live_weather_path else None
        )
        result["live_weather_available"] = bool(
            self.live_weather_path is not None and self.live_weather_path.exists()
        )
        return result


class FeatureRepository(CachedCsvRepository):
    def __init__(self, path: str | Path = DEFAULT_FEATURE_PATH) -> None:
        super().__init__(path)

    def latest(
        self, station_id: str, *, max_age_hours: float | None = None
    ) -> dict[str, Any]:
        frame = self._load()
        selected = frame[frame["station_id"] == station_id]
        if selected.empty:
            raise StationNotFoundError(station_id)
        latest = selected.tail(1)
        if max_age_hours is not None:
            timestamp = latest["timestamp"].iloc[0]
            now = pd.Timestamp.now(tz=self.timezone_name)
            if now - timestamp > pd.Timedelta(hours=max_age_hours):
                raise StaleFeatureError(station_id, timestamp.isoformat(), max_age_hours)
        return records_from_frame(latest)[0]
