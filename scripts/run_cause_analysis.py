"""Fit historical baselines and rank pollution cause hypotheses."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.cause_analyzer import (  # noqa: E402
    BASELINE_FEATURES,
    CAUSE_LABELS,
    LIMITATIONS,
    analyze_pollution_causes,
    fit_cause_profile,
)

LOGGER = logging.getLogger("cause_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with Path(args.config).open("r", encoding="utf-8") as config_file:
        config = (yaml.safe_load(config_file) or {})["cause_analysis"]

    feature_columns = list(config.get("baseline_features", BASELINE_FEATURES))
    base_columns = [
        "timestamp",
        "station_id",
        "location_name",
        "pm25_rolling_mean_24h",
        *feature_columns,
    ]
    base_columns = list(dict.fromkeys(base_columns))
    split_dir = Path(config["split_dir"])
    splits = {}
    for name in ["train", "validation", "test"]:
        LOGGER.info("Reading %s", name)
        frame = pd.read_csv(
            split_dir / f"{name}.csv.gz", usecols=base_columns, low_memory=False
        )
        frame["split"] = name
        splits[name] = frame

    profile = fit_cause_profile(
        splits["train"],
        feature_columns=feature_columns,
        group_columns=list(config.get("baseline_groups", ["station_id", "month"])),
        timezone_name=str(config.get("timezone", "Asia/Ho_Chi_Minh")),
        minimum_score=float(config.get("minimum_hypothesis_score", 0.35)),
    )
    model_output = Path(config["profile_output"])
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(profile, model_output, compress=3)

    all_data = pd.concat(splits.values(), ignore_index=True)
    all_data["timestamp"] = pd.to_datetime(
        all_data["timestamp"], errors="raise", utc=True
    ).dt.tz_convert(str(config.get("timezone", "Asia/Ho_Chi_Minh")))
    current_threshold = float(config.get("event_pm25_current", 100.0))
    rolling_threshold = float(config.get("event_pm25_24h", 75.0))

    city_context = all_data.groupby("timestamp", observed=True).agg(
        city_pm25_median=("pm25", "median"),
        city_pm25_mean=("pm25", "mean"),
        city_pm25_std=("pm25", "std"),
        city_station_count=("station_id", "nunique"),
        city_affected_count=("pm25", lambda values: values.ge(rolling_threshold).sum()),
    ).reset_index()
    city_context["city_affected_fraction"] = (
        city_context["city_affected_count"] / city_context["city_station_count"]
    )
    city_context["city_spatial_cv"] = (
        city_context["city_pm25_std"] / city_context["city_pm25_mean"].clip(lower=1e-6)
    ).fillna(0.0)

    event_mask = all_data["pm25"].ge(current_threshold) | all_data[
        "pm25_rolling_mean_24h"
    ].ge(rolling_threshold)
    events = all_data.loc[event_mask].copy()
    events = events.merge(city_context, on="timestamp", how="left", validate="many_to_one")

    anomaly_path = Path(config.get("anomaly_results", ""))
    if anomaly_path.exists():
        anomaly = pd.read_csv(
            anomaly_path,
            usecols=[
                "split",
                "timestamp",
                "station_id",
                "is_anomaly",
                "is_rule_anomaly",
                "is_pollution_episode",
                "anomaly_reason",
            ],
        )
        anomaly["timestamp"] = pd.to_datetime(
            anomaly["timestamp"], errors="raise", utc=True
        ).dt.tz_convert(str(config.get("timezone", "Asia/Ho_Chi_Minh")))
        events = events.merge(
            anomaly,
            on=["split", "timestamp", "station_id"],
            how="left",
            validate="one_to_one",
        )

    LOGGER.info("Analyzing %s pollution events", f"{len(events):,}")
    analyzed = analyze_pollution_causes(events, profile)
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    analyzed.to_csv(output, index=False, compression="gzip")

    hypothesis_counts = (
        analyzed.groupby(["top_hypothesis", "top_hypothesis_vi"], observed=True)
        .size()
        .rename("events")
        .reset_index()
        .sort_values("events", ascending=False)
    )
    hypothesis_counts["percent"] = hypothesis_counts["events"] / len(analyzed) * 100
    by_split = (
        analyzed.groupby(["split", "top_hypothesis"], observed=True)
        .size()
        .rename("events")
        .reset_index()
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "rule-based ranking against station/month train baselines",
        "interpretation": (
            "Hypotheses are plausible contributing conditions, not formal source "
            "apportionment or proven causality."
        ),
        "training_split": "train only",
        "event_definition": {
            "pm25_current_gte": current_threshold,
            "pm25_rolling_24h_gte": rolling_threshold,
        },
        "baseline": {
            "groups": profile.group_columns,
            "features": profile.feature_columns,
            "minimum_hypothesis_score": profile.minimum_score,
            "rows": len(profile.baseline),
        },
        "data_availability": {
            "historical_traffic": False,
            "chemical_speciation": False,
            "emission_inventory": False,
            "back_trajectory": False,
            "pollutants_and_weather": True,
        },
        "limitations": LIMITATIONS,
        "events": {
            "total": len(analyzed),
            "by_split": {str(k): int(v) for k, v in analyzed["split"].value_counts().items()},
            "evidence_strength": {
                str(k): int(v)
                for k, v in analyzed["evidence_strength"].value_counts().items()
            },
        },
        "hypothesis_distribution": hypothesis_counts.to_dict(orient="records"),
        "hypothesis_by_split": by_split.to_dict(orient="records"),
        "cause_labels": CAUSE_LABELS,
        "references": [
            "https://digitalcommons.unl.edu/usepapapers/109/",
            "https://www.sciencedirect.com/science/article/abs/pii/S1352231002002959",
            "https://www.sciencedirect.com/science/article/abs/pii/S0021850220302019",
        ],
        "artifacts": {"profile": str(model_output), "analysis": str(output)},
    }
    report_output = Path(config["report_output"])
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Profile: %s", model_output)
    LOGGER.info("Analysis: %s", output)
    LOGGER.info("Report: %s", report_output)
    LOGGER.info("Top hypotheses:\n%s", hypothesis_counts.to_string(index=False))


if __name__ == "__main__":
    main()
