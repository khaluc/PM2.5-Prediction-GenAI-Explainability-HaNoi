"""Download the configured Hanoi historical dataset and its provenance manifest."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collection.common import (  # noqa: E402
    locations_from_config,
    split_date_range,
    write_csv_atomic,
)
from src.collection.sensor_collector import collect_historical_air_quality  # noqa: E402
from src.collection.weather_collector import collect_historical_weather  # noqa: E402

LOGGER = logging.getLogger("hanoi_history")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start-date", help="Override requested start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Override end date (YYYY-MM-DD)")
    return parser.parse_args()


def load_config(path: str) -> tuple[dict, dict]:
    with Path(path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    return config["collection"], config["historical_collection"]


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.drop_duplicates(subset=["timestamp", "station_id", "source"], keep="last")
        .sort_values(["timestamp", "station_id"])
        .reset_index(drop=True)
    )


def dataset_stats(frame: pd.DataFrame, value_columns: list[str]) -> dict:
    return {
        "rows": len(frame),
        "stations": int(frame["station_id"].nunique()),
        "start": str(frame["timestamp"].min()),
        "end": str(frame["timestamp"].max()),
        "duplicates": int(
            frame.duplicated(subset=["timestamp", "station_id", "source"]).sum()
        ),
        "missing": {column: int(frame[column].isna().sum()) for column in value_columns},
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    collection, history = load_config(args.config)
    locations = locations_from_config(collection["locations"])
    requested_start = date.fromisoformat(
        args.start_date or str(history["requested_start_date"])
    )
    end_date = date.fromisoformat(args.end_date or str(history["end_date"]))
    air_available = date.fromisoformat(str(history["air_quality_available_from"]))
    air_start = max(requested_start, air_available)
    chunk_months = int(history.get("chunk_months", 12))
    timezone_name = str(collection.get("timezone", "Asia/Ho_Chi_Minh"))
    timeout = float(collection.get("timeout_seconds", 60))

    if requested_start > end_date:
        raise ValueError("Requested start date must not be after end date.")

    weather_parts: list[pd.DataFrame] = []
    for chunk_start, chunk_end in split_date_range(requested_start, end_date, chunk_months):
        LOGGER.info("Weather %s to %s", chunk_start, chunk_end)
        weather_parts.append(
            collect_historical_weather(
                locations,
                chunk_start,
                chunk_end,
                model=str(history.get("weather_model", "era5")),
                timezone_name=timezone_name,
                timeout_seconds=timeout,
            )
        )
    weather = normalize(pd.concat(weather_parts, ignore_index=True))
    write_csv_atomic(weather, history["weather_output"])

    air_parts: list[pd.DataFrame] = []
    if air_start <= end_date:
        for chunk_start, chunk_end in split_date_range(air_start, end_date, chunk_months):
            LOGGER.info("Air quality %s to %s", chunk_start, chunk_end)
            air_parts.append(
                collect_historical_air_quality(
                    locations,
                    chunk_start,
                    chunk_end,
                    timezone_name=timezone_name,
                    timeout_seconds=timeout,
                )
            )
    if not air_parts:
        raise ValueError("Requested end date predates CAMS Global availability.")
    air_quality = normalize(pd.concat(air_parts, ignore_index=True))
    pollutant_columns = ["pm25", "pm10", "co", "no2", "so2", "o3"]
    unavailable_mask = air_quality[pollutant_columns].isna().all(axis=1)
    unavailable_rows = int(unavailable_mask.sum())
    air_quality = air_quality.loc[~unavailable_mask].reset_index(drop=True)
    write_csv_atomic(air_quality, history["air_quality_output"])

    actual_air_start = pd.to_datetime(air_quality["timestamp"].min())
    provider_gap_end = actual_air_start - pd.Timedelta(hours=1)
    coverage_gaps = []
    if requested_start < air_available:
        coverage_gaps.append(
            {
                "start": requested_start.isoformat(),
                "end": (air_available.fromordinal(air_available.toordinal() - 1)).isoformat(),
                "reason": "CAMS Global data is unavailable before August 2022",
            }
        )
    if unavailable_rows:
        coverage_gaps.append(
            {
                "start": air_available.isoformat(),
                "end": provider_gap_end.isoformat(),
                "reason": "Provider returned no pollutant values during archive initialization",
                "removed_rows": unavailable_rows,
            }
        )

    manifest = {
        "area": "Ha Noi",
        "timezone": timezone_name,
        "requested_frequency": "hourly",
        "requested_start_date": requested_start.isoformat(),
        "requested_end_date": end_date.isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locations": [location.__dict__ for location in locations],
        "weather": {
            "provider": "Open-Meteo Historical Weather API",
            "model": str(history.get("weather_model", "era5")),
            "native_frequency": "hourly",
            **dataset_stats(
                weather,
                ["temperature", "humidity", "wind_speed", "precipitation"],
            ),
        },
        "air_quality": {
            "provider": "Open-Meteo Air Quality API / CAMS Global",
            "official_available_start_date": air_start.isoformat(),
            "actual_data_start": actual_air_start.isoformat(),
            "native_frequency": "3-hourly",
            "api_output_frequency": "hourly",
            "coverage_gaps": coverage_gaps,
            **dataset_stats(
                air_quality,
                ["pm25", "pm10", "co", "no2", "so2", "o3", "us_aqi"],
            ),
        },
    }
    manifest_path = Path(history["manifest_output"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Saved %s weather rows", len(weather))
    LOGGER.info("Saved %s air-quality rows", len(air_quality))
    LOGGER.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
