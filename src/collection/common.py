"""Shared collection types, HTTP handling and CSV persistence."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

LOGGER = logging.getLogger(__name__)


class CollectionError(RuntimeError):
    """Raised when a provider cannot return usable collection data."""


@dataclass(frozen=True)
class HanoiLocation:
    """A configured sampling point, not necessarily a physical sensor station."""

    station_id: str
    name: str
    latitude: float
    longitude: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HanoiLocation":
        return cls(
            station_id=str(value["station_id"]),
            name=str(value["name"]),
            latitude=float(value["latitude"]),
            longitude=float(value["longitude"]),
        )


def locations_from_config(values: Iterable[dict[str, Any]]) -> list[HanoiLocation]:
    """Validate and convert configured sampling point dictionaries."""
    locations = [HanoiLocation.from_dict(value) for value in values]
    if not locations:
        raise ValueError("At least one Hanoi collection location is required.")
    station_ids = [location.station_id for location in locations]
    if len(station_ids) != len(set(station_ids)):
        raise ValueError("Collection station_id values must be unique.")
    return locations


def get_json(
    client: httpx.Client,
    url: str,
    params: dict[str, Any],
    *,
    attempts: int = 3,
) -> dict[str, Any]:
    """GET JSON with short retries for provider and network failures."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise CollectionError(f"Unexpected JSON response from {url}")
            if payload.get("error"):
                raise CollectionError(str(payload.get("reason", "Provider error")))
            return payload
        except httpx.HTTPStatusError as exc:
            last_error = CollectionError(
                f"Provider returned HTTP {exc.response.status_code}"
            )
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
        except httpx.RequestError as exc:
            # Do not stringify the request because its URL may contain an API key.
            last_error = CollectionError(type(exc).__name__)
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
        except (ValueError, CollectionError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise CollectionError(f"Request failed after {attempts} attempts: {last_error}")


def local_iso_timestamp(value: str, timezone_name: str) -> str:
    """Attach the configured timezone when a provider returns local naive time."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
    return parsed.isoformat()


def collected_at_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_future_hour(value: str, timezone_name: str, collected_at: str) -> bool:
    """Mark hourly provider values later than the collection hour as forecasts."""
    timestamp = datetime.fromisoformat(local_iso_timestamp(value, timezone_name))
    collected = datetime.fromisoformat(collected_at).astimezone(ZoneInfo(timezone_name))
    return timestamp > collected.replace(minute=0, second=0, microsecond=0)


def append_deduplicated_csv(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    unique_columns: list[str],
    exclude_forecasts: bool = False,
) -> int:
    """Append rows to CSV atomically while replacing duplicate provider records."""
    if frame.empty:
        return 0

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    if exclude_forecasts and not existing.empty and "is_forecast" in existing:
        forecast_mask = (
            existing["is_forecast"]
            if pd.api.types.is_bool_dtype(existing["is_forecast"])
            else existing["is_forecast"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        )
        existing = existing[~forecast_mask].copy()
    combined = pd.concat([existing, frame], ignore_index=True)
    before = len(existing)
    if exclude_forecasts and "is_forecast" in combined:
        forecast_mask = (
            combined["is_forecast"]
            if pd.api.types.is_bool_dtype(combined["is_forecast"])
            else combined["is_forecast"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        )
        combined = combined[~forecast_mask].copy()
    combined = combined.drop_duplicates(subset=unique_columns, keep="last")

    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)
    written = max(0, len(combined) - before)
    LOGGER.info("Saved %s rows to %s (%s new)", len(combined), path, written)
    return written


def split_date_range(start_date: date, end_date: date, months: int = 12):
    """Yield inclusive date chunks without requiring calendar extensions."""
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    if months < 1:
        raise ValueError("months must be at least 1")

    current = start_date
    while current <= end_date:
        month_index = current.year * 12 + current.month - 1 + months
        next_year, next_month_index = divmod(month_index, 12)
        next_month = next_month_index + 1
        boundary = date(next_year, next_month, 1)
        chunk_end = min(end_date, boundary - timedelta(days=1))
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def write_csv_atomic(frame: pd.DataFrame, output_path: str | Path) -> None:
    """Write a complete CSV through a temporary file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
