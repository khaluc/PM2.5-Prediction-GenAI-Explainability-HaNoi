"""Application-level inference wrappers with stable JSON results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.models.predict import (
    DEFAULT_ANOMALY_MODEL_PATH,
    DEFAULT_CAUSE_PROFILE_PATH,
    DEFAULT_MODEL_PATH,
    load_anomaly_artifact,
    predict_environment,
)
from src.models.train_anomaly import score_anomalies


def predict_from_features(
    features: dict[str, Any],
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    anomaly_model_path: str | Path | None = DEFAULT_ANOMALY_MODEL_PATH,
    cause_profile_path: str | Path | None = DEFAULT_CAUSE_PROFILE_PATH,
) -> dict[str, Any]:
    return predict_environment(
        features,
        model_path=model_path,
        anomaly_model_path=anomaly_model_path,
        cause_profile_path=cause_profile_path,
    )


def detect_anomaly_from_features(
    features: dict[str, Any],
    *,
    model_path: str | Path = DEFAULT_ANOMALY_MODEL_PATH,
) -> dict[str, Any]:
    artifact = load_anomaly_artifact(str(Path(model_path).resolve()))
    missing = sorted(set(artifact.feature_columns) - set(features))
    if missing:
        raise ValueError(f"Missing anomaly features: {missing}")
    scored = score_anomalies(pd.DataFrame([features]), artifact).iloc[0]
    return {
        "station_id": features.get("station_id"),
        "timestamp": str(features.get("timestamp")) if features.get("timestamp") is not None else None,
        "is_anomaly": bool(scored["is_anomaly"]),
        "is_rule_anomaly": bool(scored["is_rule_anomaly"]),
        "is_isolation_anomaly": bool(scored["is_isolation_anomaly"]),
        "is_pollution_episode": bool(scored["is_pollution_episode"]),
        "requires_attention": bool(scored["requires_attention"]),
        "detection_source": str(scored["detection_source"]),
        "reason": str(scored["anomaly_reason"]),
        "isolation_forest_score": float(scored["isolation_forest_score"]),
        "score_threshold": float(artifact.score_threshold),
    }
