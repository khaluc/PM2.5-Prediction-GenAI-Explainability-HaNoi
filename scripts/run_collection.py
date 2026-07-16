"""Collect Hanoi air quality, weather and optional traffic data."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collection.common import (
    CollectionError,
    append_deduplicated_csv,
    locations_from_config,
)
from src.collection.sensor_collector import collect_sensor_data
from src.collection.traffic_collector import collect_traffic_data
from src.collection.weather_collector import collect_weather_data
from src.database.connection import environment_flag

LOGGER = logging.getLogger("hanoi_collection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--past-hours", type=int, help="Override configured past hours")
    parser.add_argument("--forecast-hours", type=int, help="Override configured forecast hours")
    parser.add_argument(
        "--include-traffic",
        action="store_true",
        help="Collect TomTom traffic data (requires TOMTOM_API_KEY)",
    )
    return parser.parse_args()


def load_collection_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    collection = config.get("collection")
    if not isinstance(collection, dict):
        raise ValueError("config.yaml must contain a collection section.")
    return collection


def run_collection(config: dict, *, include_traffic: bool = False) -> dict[str, int]:
    """Run providers independently and persist all successful results."""
    locations = locations_from_config(config.get("locations", []))
    timezone_name = str(config.get("timezone", "Asia/Ho_Chi_Minh"))
    timeout = float(config.get("timeout_seconds", 30))
    past_hours = int(config.get("past_hours", 24))
    forecast_hours = int(config.get("forecast_hours", 1))
    include_provider_forecast = bool(config.get("include_provider_forecast", False))
    result: dict[str, int] = {}
    failures: list[str] = []
    database_writer = None
    if environment_flag("DATABASE_WRITE_ENABLED", False):
        from src.database.writer import DatabaseWriter

        database_writer = DatabaseWriter(
            batch_size=int(os.getenv("DATABASE_BATCH_SIZE", "2000"))
        )

    def persist_database(name: str, operation) -> None:
        if database_writer is None:
            return
        try:
            result[f"database_{name}"] = int(operation())
        except Exception as exc:
            message = f"database_{name}: {type(exc).__name__}: {exc}"
            if environment_flag("DATABASE_REQUIRED", False):
                raise CollectionError(message) from exc
            failures.append(message)
            LOGGER.exception("Database persistence failed for %s", name)

    try:
        air_quality = collect_sensor_data(
            locations,
            past_hours=past_hours,
            forecast_hours=forecast_hours,
            timezone_name=timezone_name,
            timeout_seconds=timeout,
        )
        if not include_provider_forecast:
            air_quality = air_quality[~air_quality["is_forecast"]].copy()
        result["air_quality"] = append_deduplicated_csv(
            air_quality,
            config["air_quality_output"],
            unique_columns=["timestamp", "station_id", "source"],
            exclude_forecasts=not include_provider_forecast,
        )
        persist_database("air_quality", lambda: database_writer.upsert_air_quality(air_quality))
    except (CollectionError, KeyError) as exc:
        failures.append(f"air_quality: {exc}")
        LOGGER.error("Air-quality pipeline failed: %s", exc)

    try:
        weather = collect_weather_data(
            locations,
            past_hours=past_hours,
            forecast_hours=forecast_hours,
            timezone_name=timezone_name,
            timeout_seconds=timeout,
        )
        if not include_provider_forecast:
            weather = weather[~weather["is_forecast"]].copy()
        result["weather"] = append_deduplicated_csv(
            weather,
            config["weather_output"],
            unique_columns=["timestamp", "station_id", "source"],
            exclude_forecasts=not include_provider_forecast,
        )
        persist_database("weather", lambda: database_writer.upsert_weather(weather))
    except (CollectionError, KeyError) as exc:
        failures.append(f"weather: {exc}")
        LOGGER.error("Weather pipeline failed: %s", exc)

    traffic_config = config.get("traffic", {})
    traffic_enabled = include_traffic or bool(traffic_config.get("enabled", False))
    if traffic_enabled:
        max_locations = int(traffic_config.get("max_locations", len(locations)))
        try:
            traffic = collect_traffic_data(
                locations[:max_locations],
                zoom=int(traffic_config.get("zoom", 10)),
                timezone_name=timezone_name,
                timeout_seconds=timeout,
            )
            result["traffic"] = append_deduplicated_csv(
                traffic,
                config["traffic_output"],
                unique_columns=["timestamp", "station_id", "source"],
            )
            persist_database("traffic", lambda: database_writer.upsert_traffic(traffic))
        except (CollectionError, KeyError) as exc:
            failures.append(f"traffic: {exc}")
            LOGGER.error("Traffic pipeline failed: %s", exc)

    if not result:
        raise CollectionError("All configured collectors failed. " + "; ".join(failures))
    if failures:
        LOGGER.warning("Collection completed with partial failures: %s", "; ".join(failures))
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv()
    config = load_collection_config(args.config)
    if args.past_hours is not None:
        config["past_hours"] = args.past_hours
    if args.forecast_hours is not None:
        config["forecast_hours"] = args.forecast_hours
    summary = run_collection(config, include_traffic=args.include_traffic)
    for dataset, new_rows in summary.items():
        LOGGER.info("%s: %s new rows", dataset, new_rows)


if __name__ == "__main__":
    main()
