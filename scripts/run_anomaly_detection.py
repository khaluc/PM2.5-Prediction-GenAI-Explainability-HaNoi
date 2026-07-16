"""Train and run hybrid threshold-rule plus Isolation Forest detection."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_anomaly import fit_anomaly_model, score_anomalies  # noqa: E402

LOGGER = logging.getLogger("anomaly_detection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def _split_summary(scored: pd.DataFrame) -> dict:
    rows = len(scored)
    return {
        "rows": rows,
        "rule_anomalies": int(scored["is_rule_anomaly"].sum()),
        "rule_rate_percent": float(scored["is_rule_anomaly"].mean() * 100),
        "isolation_anomalies": int(scored["is_isolation_anomaly"].sum()),
        "isolation_rate_percent": float(scored["is_isolation_anomaly"].mean() * 100),
        "combined_anomalies": int(scored["is_anomaly"].sum()),
        "combined_rate_percent": float(scored["is_anomaly"].mean() * 100),
        "pollution_episodes": int(scored["is_pollution_episode"].sum()),
        "pollution_episode_rate_percent": float(
            scored["is_pollution_episode"].mean() * 100
        ),
        "requires_attention": int(scored["requires_attention"].sum()),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with Path(args.config).open("r", encoding="utf-8") as config_file:
        config = (yaml.safe_load(config_file) or {})["anomaly"]

    feature_columns = list(config["features"])
    base_columns = [
        "timestamp",
        "station_id",
        "location_name",
        "pm25",
        "pm10",
        "pm25_change_1h",
        "pm25_change_3h",
        "pm25_rolling_mean_24h",
        "pm25_rolling_std_24h",
        "is_possible_outlier",
    ]
    required_columns = list(dict.fromkeys([*base_columns, *feature_columns]))
    split_dir = Path(config["split_dir"])
    splits: dict[str, pd.DataFrame] = {}
    for name in ["train", "validation", "test"]:
        path = split_dir / f"{name}.csv.gz"
        LOGGER.info("Reading %s", path)
        splits[name] = pd.read_csv(path, usecols=required_columns, low_memory=False)

    LOGGER.info("Fitting Isolation Forest on %s train rows", f"{len(splits['train']):,}")
    bundle = fit_anomaly_model(
        splits["train"],
        feature_columns=feature_columns,
        rules=dict(config["rules"]),
        contamination=float(config.get("contamination", 0.03)),
        random_state=int(config.get("random_state", 42)),
        n_estimators=int(config.get("n_estimators", 300)),
        max_samples=int(config.get("max_samples", 4096)),
    )

    model_output = Path(config["model_output"])
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_output, compress=3)

    scored_parts = []
    summaries = {}
    rule_counts = {}
    for name, frame in splits.items():
        LOGGER.info("Scoring %s", name)
        scores = score_anomalies(frame, bundle)
        result = pd.concat(
            [frame[base_columns].reset_index(drop=True), scores.reset_index(drop=True)],
            axis=1,
        )
        result.insert(0, "split", name)
        scored_parts.append(result)
        summaries[name] = _split_summary(result)
        rule_counts[name] = {
            column: int(result[column].sum())
            for column in result.columns
            if column.startswith("rule_") and column != "rule_reason"
        }

    scored_all = pd.concat(scored_parts, ignore_index=True)
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    scored_all.to_csv(output, index=False, compression="gzip")

    top_anomalies = (
        scored_all[scored_all["is_anomaly"]]
        .nsmallest(20, "isolation_forest_score")[
            [
                "split",
                "timestamp",
                "station_id",
                "pm25",
                "pm10",
                "isolation_forest_score",
                "detection_source",
                "anomaly_reason",
            ]
        ]
        .to_dict(orient="records")
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "threshold rules OR Isolation Forest",
        "important_note": (
            "No ground-truth anomaly labels are available; reported rates are "
            "detection diagnostics, not accuracy. Pollution episodes are kept "
            "separate from sensor/data anomalies."
        ),
        "training_split": "train only",
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "isolation_forest": {
            "n_estimators": int(config.get("n_estimators", 300)),
            "max_samples": int(config.get("max_samples", 4096)),
            "contamination": float(config.get("contamination", 0.03)),
            "train_score_threshold": bundle.score_threshold,
            "random_state": int(config.get("random_state", 42)),
        },
        "rules": config["rules"],
        "split_summary": summaries,
        "rule_counts": rule_counts,
        "detection_source_counts": {
            str(key): int(value)
            for key, value in scored_all["detection_source"].value_counts().items()
        },
        "top_anomalies": top_anomalies,
        "artifacts": {"model": str(model_output), "scores": str(output)},
    }
    report_output = Path(config["report_output"])
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, summary in summaries.items():
        LOGGER.info(
            "%s: combined %.2f%%, rules %.2f%%, IF %.2f%%, episodes %.2f%%",
            name,
            summary["combined_rate_percent"],
            summary["rule_rate_percent"],
            summary["isolation_rate_percent"],
            summary["pollution_episode_rate_percent"],
        )
    LOGGER.info("Model: %s", model_output)
    LOGGER.info("Scores: %s", output)
    LOGGER.info("Report: %s", report_output)


if __name__ == "__main__":
    main()
