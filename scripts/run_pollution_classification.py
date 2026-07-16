"""Classify rolling 24-hour PM2.5 using WHO 2021 target bands."""

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

from src.assessment.who_pm25 import (  # noqa: E402
    LEVELS,
    WHO_PM25_24H_THRESHOLDS,
    WHO_REFERENCE_URL,
    add_rolling_24h_assessment,
)

LOGGER = logging.getLogger("pollution_classification")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with Path(args.config).open("r", encoding="utf-8") as config_file:
        config = (yaml.safe_load(config_file) or {})["pollution_classification"]

    LOGGER.info("Reading %s", config["input"])
    data = pd.read_csv(config["input"], low_memory=False)
    assessed = add_rolling_24h_assessment(
        data,
        pm25_column=str(config.get("pm25_column", "pm25")),
        timestamp_column=str(config.get("timestamp_column", "timestamp")),
        station_column=str(config.get("station_column", "station_id")),
        min_valid_hours=int(config.get("min_valid_hours", 18)),
        timezone_name=str(config.get("timezone", "Asia/Ho_Chi_Minh")),
    )

    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    assessed.to_csv(output, index=False, compression="gzip")

    valid = assessed[assessed["who_assessment_valid"]].copy()
    distribution = (
        valid.groupby(
            ["who_level_code", "who_band", "project_label_vi"],
            observed=True,
            dropna=False,
        )
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("who_level_code")
    )
    distribution["percent"] = distribution["rows"] / len(valid) * 100
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "standard": "WHO Global Air Quality Guidelines 2021",
        "important_note": (
            "WHO defines AQG and interim targets, not official good/moderate/bad "
            "AQI categories. Project labels are presentation labels. A rolling "
            "window does not by itself determine annual WHO compliance."
        ),
        "pollutant": "PM2.5",
        "unit": "ug/m3",
        "averaging_period": "rolling 24h",
        "aqg_statistical_form": "annual 99th percentile of 24-hour means",
        "annual_compliance_determined": False,
        "timezone": str(config.get("timezone", "Asia/Ho_Chi_Minh")),
        "min_valid_hours": int(config.get("min_valid_hours", 18)),
        "who_thresholds": WHO_PM25_24H_THRESHOLDS,
        "levels": list(LEVELS),
        "reference_url": WHO_REFERENCE_URL,
        "rows": {
            "input": len(data),
            "valid_assessment": len(valid),
            "insufficient_coverage": int((~assessed["who_assessment_valid"]).sum()),
        },
        "exceeds_aqg_percent": float(valid["exceeds_who_aqg"].mean() * 100),
        "distribution": distribution.to_dict(orient="records"),
        "output": str(output),
    }
    report_path = Path(config["report_output"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Valid assessments: %s/%s", f"{len(valid):,}", f"{len(data):,}")
    LOGGER.info("WHO AQG exceedance: %.2f%%", report["exceeds_aqg_percent"])
    LOGGER.info("Output: %s", output)
    LOGGER.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
