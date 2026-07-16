"""Collect hourly weather values for configured Hanoi sampling points."""

from __future__ import annotations

import logging
from datetime import date
from typing import Iterable

import httpx
import pandas as pd

from src.collection.common import (
    CollectionError,
    HanoiLocation,
    collected_at_utc,
    get_json,
    is_future_hour,
    local_iso_timestamp,
)

LOGGER = logging.getLogger(__name__)
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FIELDS = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "precipitation": "precipitation",
    "rain": "rain",
    "surface_pressure": "surface_pressure",
    "cloud_cover": "cloud_cover",
}


def _rows_from_payload(
    payload: dict,
    location: HanoiLocation,
    timezone_name: str,
    collected_at: str,
) -> list[dict]:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    rows: list[dict] = []
    for index, provider_time in enumerate(times):
        row = {
            "timestamp": local_iso_timestamp(provider_time, timezone_name),
            "station_id": location.station_id,
            "location_name": location.name,
            "latitude": location.latitude,
            "longitude": location.longitude,
            "is_forecast": is_future_hour(provider_time, timezone_name, collected_at),
            "source": "open_meteo_forecast",
            "collected_at": collected_at,
        }
        for provider_field, output_field in WEATHER_FIELDS.items():
            values = hourly.get(provider_field) or []
            row[output_field] = values[index] if index < len(values) else None
        rows.append(row)
    return rows


def collect_weather_data(
    locations: Iterable[HanoiLocation],
    *,
    past_hours: int = 24,
    forecast_hours: int = 24,
    timezone_name: str = "Asia/Ho_Chi_Minh",
    timeout_seconds: float = 30,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Collect normalized hourly weather observations and forecasts."""
    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout_seconds)
    collected_at = collected_at_utc()
    rows: list[dict] = []
    errors: list[str] = []
    try:
        for location in locations:
            params = {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "hourly": ",".join(WEATHER_FIELDS),
                "timezone": timezone_name,
                "past_hours": past_hours,
                "forecast_hours": forecast_hours,
                "wind_speed_unit": "kmh",
            }
            try:
                payload = get_json(http_client, WEATHER_URL, params)
                rows.extend(_rows_from_payload(payload, location, timezone_name, collected_at))
            except CollectionError as exc:
                errors.append(f"{location.station_id}: {exc}")
                LOGGER.warning("Weather collection failed for %s: %s", location.name, exc)
    finally:
        if owns_client:
            http_client.close()

    if not rows:
        raise CollectionError("No weather data collected. " + "; ".join(errors))
    return pd.DataFrame(rows)


def collect_historical_weather(
    locations: Iterable[HanoiLocation],
    start_date: date,
    end_date: date,
    *,
    model: str = "era5",
    timezone_name: str = "Asia/Ho_Chi_Minh",
    timeout_seconds: float = 60,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Collect consistent hourly reanalysis weather for a historical range."""
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout_seconds)
    collected_at = collected_at_utc()
    rows: list[dict] = []
    errors: list[str] = []
    try:
        for location in locations:
            params = {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "hourly": ",".join(WEATHER_FIELDS),
                "timezone": timezone_name,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "wind_speed_unit": "kmh",
                "models": model,
            }
            try:
                payload = get_json(http_client, WEATHER_ARCHIVE_URL, params)
                rows.extend(_rows_from_payload(payload, location, timezone_name, collected_at))
            except CollectionError as exc:
                errors.append(f"{location.station_id}: {exc}")
                LOGGER.warning("Historical weather collection failed for %s: %s", location.name, exc)
    finally:
        if owns_client:
            http_client.close()

    if not rows:
        raise CollectionError("No historical weather data collected. " + "; ".join(errors))
    frame = pd.DataFrame(rows)
    frame["is_forecast"] = False
    frame["source"] = f"open_meteo_{model}_reanalysis"
    return frame
