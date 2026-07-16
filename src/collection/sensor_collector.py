"""Collect modelled air-quality values for configured Hanoi sampling points."""

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
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
AIR_FIELDS = {
    "pm2_5": "pm25",
    "pm10": "pm10",
    "carbon_monoxide": "co",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "ozone": "o3",
    "us_aqi": "us_aqi",
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
            "source": "open_meteo_cams",
            "collected_at": collected_at,
        }
        for provider_field, output_field in AIR_FIELDS.items():
            values = hourly.get(provider_field) or []
            row[output_field] = values[index] if index < len(values) else None
        rows.append(row)
    return rows


def collect_sensor_data(
    locations: Iterable[HanoiLocation],
    *,
    past_hours: int = 24,
    forecast_hours: int = 24,
    timezone_name: str = "Asia/Ho_Chi_Minh",
    timeout_seconds: float = 30,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Collect hourly CAMS air-quality values through Open-Meteo.

    These values are model estimates at grid cells, not measurements from official
    Hanoi ground sensor stations. The source column preserves that distinction.
    """
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
                "hourly": ",".join(AIR_FIELDS),
                "timezone": timezone_name,
                "past_hours": past_hours,
                "forecast_hours": forecast_hours,
                "domains": "cams_global",
            }
            try:
                payload = get_json(http_client, AIR_QUALITY_URL, params)
                rows.extend(_rows_from_payload(payload, location, timezone_name, collected_at))
            except CollectionError as exc:
                errors.append(f"{location.station_id}: {exc}")
                LOGGER.warning("Air-quality collection failed for %s: %s", location.name, exc)
    finally:
        if owns_client:
            http_client.close()

    if not rows:
        raise CollectionError("No air-quality data collected. " + "; ".join(errors))
    return pd.DataFrame(rows)


def collect_historical_air_quality(
    locations: Iterable[HanoiLocation],
    start_date: date,
    end_date: date,
    *,
    timezone_name: str = "Asia/Ho_Chi_Minh",
    timeout_seconds: float = 60,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Collect a bounded historical CAMS Global date range for Hanoi."""
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
                "hourly": ",".join(AIR_FIELDS),
                "timezone": timezone_name,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "domains": "cams_global",
            }
            try:
                payload = get_json(http_client, AIR_QUALITY_URL, params)
                rows.extend(_rows_from_payload(payload, location, timezone_name, collected_at))
            except CollectionError as exc:
                errors.append(f"{location.station_id}: {exc}")
                LOGGER.warning("Historical air collection failed for %s: %s", location.name, exc)
    finally:
        if owns_client:
            http_client.close()

    if not rows:
        raise CollectionError("No historical air-quality data collected. " + "; ".join(errors))
    frame = pd.DataFrame(rows)
    frame["is_forecast"] = False
    return frame
