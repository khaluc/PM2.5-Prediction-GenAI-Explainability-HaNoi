"""Create chronological PM2.5 train, validation and test files."""

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

from src.models.time_split import split_time_series  # noqa: E402

LOGGER = logging.getLogger("time_split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with Path(args.config).open("r", encoding="utf-8") as config_file:
        config = (yaml.safe_load(config_file) or {})["time_split"]

    LOGGER.info("Reading %s", config["input"])
    data = pd.read_csv(config["input"], low_memory=False)
    result = split_time_series(
        data,
        validation_start=config["validation_start"],
        test_start=config["test_start"],
        max_target_horizon_hours=int(config["max_target_horizon_hours"]),
        timestamp_column=str(config.get("timestamp_column", "timestamp")),
        group_column=str(config.get("group_column", "station_id")),
        timezone_name=str(config.get("timezone", "Asia/Ho_Chi_Minh")),
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": output_dir / "train.csv.gz",
        "validation": output_dir / "validation.csv.gz",
        "test": output_dir / "test.csv.gz",
    }
    result.train.to_csv(outputs["train"], index=False, compression="gzip")
    result.validation.to_csv(outputs["validation"], index=False, compression="gzip")
    result.test.to_csv(outputs["test"], index=False, compression="gzip")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(config["input"]),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "report": result.report,
    }
    manifest_path = Path(config["manifest_output"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name in ["train", "validation", "test"]:
        LOGGER.info("%s: %s rows", name, result.report[name]["rows"])
    LOGGER.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
