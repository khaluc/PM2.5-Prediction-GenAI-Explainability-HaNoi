"""Clean Hanoi historical data and build the ML-ready merged table."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collection.common import write_csv_atomic  # noqa: E402
from src.preprocessing.clean_data import clean_environmental_data  # noqa: E402

LOGGER = logging.getLogger("environment_cleaning")

AIR_COLUMNS = ["pm25", "pm10", "co", "no2", "so2", "o3", "us_aqi"]
WEATHER_COLUMNS = [
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "rain",
    "surface_pressure",
    "cloud_cover",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if "cleaning" not in config:
        raise ValueError("config.yaml must contain a cleaning section")
    return config["cleaning"]


def clean_table(
    path: str,
    value_columns: list[str],
    dataset_name: str,
    config: dict,
):
    LOGGER.info("Reading %s", path)
    raw = pd.read_csv(path)
    return clean_environmental_data(
        raw,
        value_columns=value_columns,
        valid_ranges=config["valid_ranges"][dataset_name],
        declared_units=config.get("units", {}).get(dataset_name, {}),
        timezone_name=str(config.get("timezone", "Asia/Ho_Chi_Minh")),
        frequency=str(config.get("frequency", "1h")),
        interpolation_limit=int(config.get("interpolation_limit_hours", 3)),
        spike_window=int(config.get("spike_window_hours", 24)),
        spike_min_periods=int(config.get("spike_min_periods", 12)),
        spike_threshold=float(config.get("spike_z_threshold", 6.0)),
        spike_columns=config.get("spike_columns", {}).get(dataset_name, value_columns),
        spike_min_absolute_change=config.get("spike_min_absolute_change", {}).get(
            dataset_name, {}
        ),
        replace_spikes=bool(config.get("replace_spikes", False)),
    )


def build_merged_table(air: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join clean provider tables without hiding their independent quality flags."""
    air = air.rename(
        columns={
            "source": "air_source",
            "collected_at": "air_collected_at",
            "quality_flags": "air_quality_flags",
            "is_imputed": "air_is_imputed",
            "is_possible_outlier": "air_is_possible_outlier",
            "data_quality_score": "air_quality_score",
        }
    )
    weather = weather.rename(
        columns={
            "source": "weather_source",
            "collected_at": "weather_collected_at",
            "quality_flags": "weather_quality_flags",
            "is_imputed": "weather_is_imputed",
            "is_possible_outlier": "weather_is_possible_outlier",
            "data_quality_score": "weather_quality_score",
        }
    )
    weather_keep = [
        "timestamp",
        "station_id",
        *WEATHER_COLUMNS,
        "weather_source",
        "weather_collected_at",
        "weather_quality_flags",
        "weather_is_imputed",
        "weather_is_possible_outlier",
        "weather_quality_score",
    ]
    merged = air.merge(
        weather[weather_keep],
        on=["timestamp", "station_id"],
        how="left",
        validate="one_to_one",
    )
    air_flags = merged["air_quality_flags"].fillna("")
    weather_flags = merged["weather_quality_flags"].fillna("")
    merged["quality_flags"] = np.where(
        air_flags.eq(""),
        weather_flags,
        np.where(weather_flags.eq(""), air_flags, air_flags + ";" + weather_flags),
    )
    merged["is_imputed"] = (
        merged["air_is_imputed"].fillna(False)
        | merged["weather_is_imputed"].fillna(False)
    )
    merged["is_possible_outlier"] = (
        merged["air_is_possible_outlier"].fillna(False)
        | merged["weather_is_possible_outlier"].fillna(False)
    )
    merged["data_quality_score"] = merged[
        ["air_quality_score", "weather_quality_score"]
    ].min(axis=1)
    return merged.sort_values(["timestamp", "station_id"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)

    air_result = clean_table(config["air_input"], AIR_COLUMNS, "air", config)
    weather_result = clean_table(
        config["weather_input"], WEATHER_COLUMNS, "weather", config
    )
    write_csv_atomic(air_result.data, config["air_output"])
    write_csv_atomic(weather_result.data, config["weather_output"])

    merged = build_merged_table(air_result.data, weather_result.data)
    write_csv_atomic(merged, config["merged_output"])

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "air": air_result.report,
        "weather": weather_result.report,
        "merged": {
            "rows": len(merged),
            "start": str(merged["timestamp"].min()),
            "end": str(merged["timestamp"].max()),
            "stations": int(merged["station_id"].nunique()),
            "duplicate_keys": int(
                merged.duplicated(subset=["timestamp", "station_id"]).sum()
            ),
            "rows_with_quality_flags": int(merged["quality_flags"].ne("").sum()),
            "possible_outlier_rows": int(merged["is_possible_outlier"].sum()),
        },
    }
    report_path = Path(config["report_output"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Air rows: %s", len(air_result.data))
    LOGGER.info("Weather rows: %s", len(weather_result.data))
    LOGGER.info("Merged rows: %s", len(merged))
    LOGGER.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
