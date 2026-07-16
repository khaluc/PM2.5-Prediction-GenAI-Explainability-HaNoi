"""Create the leakage-safe supervised PM2.5 feature table."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.feature_engineering import build_features  # noqa: E402

LOGGER = logging.getLogger("feature_engineering")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with Path(args.config).open("r", encoding="utf-8") as config_file:
        config = (yaml.safe_load(config_file) or {})["feature_engineering"]

    LOGGER.info("Reading %s", config["input"])
    data = pd.read_csv(config["input"], low_memory=False)
    result = build_features(
        data,
        group_column=str(config.get("group_column", "station_id")),
        target_column=str(config.get("target_column", "pm25")),
        timezone_name=str(config.get("timezone", "Asia/Ho_Chi_Minh")),
        target_horizons=list(config.get("target_horizons", [1, 3, 6])),
        pm25_lags=list(config.get("pm25_lags", [1, 3, 6, 24, 168])),
        pollutant_lags=list(config.get("pollutant_lags", [1, 3, 24])),
        weather_lags=list(config.get("weather_lags", [1, 3, 6, 24])),
        rolling_windows=list(config.get("rolling_windows", [3, 6, 24, 168])),
        weather_rolling_windows=list(
            config.get("weather_rolling_windows", [6, 24])
        ),
        ewm_halflives=list(config.get("ewm_halflives", [6, 24])),
        drop_incomplete_rows=bool(config.get("drop_incomplete_rows", True)),
    )

    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_csv(output, index=False, compression="gzip")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(config["input"]),
        "output": str(output),
        "feature_columns": result.feature_columns,
        "target_columns": result.target_columns,
        "feature_groups": result.feature_groups,
        "report": result.report,
    }
    metadata_path = Path(config["metadata_output"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Rows: %s", len(result.data))
    LOGGER.info("Features: %s", len(result.feature_columns))
    LOGGER.info("Targets: %s", ", ".join(result.target_columns))
    LOGGER.info("Output: %s", output)


if __name__ == "__main__":
    main()
