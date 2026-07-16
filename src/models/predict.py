"""Stable PM2.5 forecasting interface consumed by the API."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.analysis.cause_analyzer import (
    CauseProfile,
    analyze_pollution_causes,
)
from src.assessment.who_pm25 import (
    classify_hourly_forecast_proxy,
    classify_pm25_24h,
)
from src.models.train_anomaly import AnomalyModelBundle, score_anomalies
from src.models.train_forecast import predict_horizon_models, prepare_tree_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "pm25_forecast.joblib"
DEFAULT_ANOMALY_MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "anomaly_detector.joblib"
)
DEFAULT_CAUSE_PROFILE_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "cause_analyzer.joblib"
)


@lru_cache(maxsize=4)
def load_forecast_artifact(model_path: str = str(DEFAULT_MODEL_PATH)) -> dict[str, Any]:
    """Load and cache the selected forecast model bundle."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Forecast artifact not found: {path}. Run scripts/run_training.py first."
        )
    artifact = joblib.load(path)
    required = {"model_name", "horizons", "feature_columns"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Invalid forecast artifact; missing keys: {sorted(missing)}")
    return artifact


@lru_cache(maxsize=4)
def load_anomaly_artifact(
    model_path: str = str(DEFAULT_ANOMALY_MODEL_PATH),
) -> AnomalyModelBundle:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Anomaly artifact not found: {path}. Run scripts/run_anomaly_detection.py first."
        )
    artifact = joblib.load(path)
    if not isinstance(artifact, AnomalyModelBundle):
        raise ValueError("Invalid anomaly model artifact")
    return artifact


@lru_cache(maxsize=4)
def load_cause_profile(
    model_path: str = str(DEFAULT_CAUSE_PROFILE_PATH),
) -> CauseProfile:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cause profile not found: {path}. Run scripts/run_cause_analysis.py first."
        )
    artifact = joblib.load(path)
    if not isinstance(artifact, CauseProfile):
        raise ValueError("Invalid pollution cause profile")
    return artifact


def predict_pm25(
    frame: pd.DataFrame,
    artifact: dict[str, Any],
) -> pd.DataFrame:
    """Predict all configured horizons for one or more feature rows."""
    if frame.empty:
        raise ValueError("At least one feature row is required")
    model_name = str(artifact["model_name"])
    horizons = [int(value) for value in artifact["horizons"]]

    if model_name == "Baseline":
        if "pm25" not in frame:
            raise ValueError("Baseline prediction requires the pm25 column")
        predicted = np.repeat(
            frame[["pm25"]].to_numpy(dtype=np.float32), len(horizons), axis=1
        )
    elif model_name in {"RandomForest", "XGBoost", "LightGBM"}:
        required_features = list(artifact["feature_columns"])
        station_column = str(artifact.get("station_column", "station_id"))
        missing = sorted(set([*required_features, station_column]) - set(frame.columns))
        if missing:
            preview = missing[:10]
            suffix = "..." if len(missing) > 10 else ""
            raise ValueError(f"Missing forecast features: {preview}{suffix}")
        matrix, _ = prepare_tree_matrix(
            frame,
            required_features,
            list(artifact["station_categories"]),
            station_column,
        )
        model = artifact["model"]
        if model_name == "RandomForest":
            predicted = model.predict(matrix)
        else:
            predicted = predict_horizon_models(model, matrix)
    elif model_name == "LSTM":
        raise ValueError(
            "LSTM inference requires a continuous sequence; use the sequence "
            "metadata stored in the artifact instead of predict_pm25()."
        )
    else:
        raise ValueError(f"Unsupported forecast model: {model_name}")

    predicted = np.asarray(predicted, dtype=float)
    if predicted.ndim == 1:
        predicted = predicted.reshape(-1, 1)
    if predicted.shape != (len(frame), len(horizons)):
        raise ValueError(f"Unexpected prediction shape: {predicted.shape}")
    predicted = np.maximum(predicted, 0.0)
    return pd.DataFrame(
        {
            f"predicted_pm25_t_plus_{horizon}h": predicted[:, index]
            for index, horizon in enumerate(horizons)
        },
        index=frame.index,
    )


def predict_environment(
    input_data: dict[str, Any],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    anomaly_model_path: str | Path | None = DEFAULT_ANOMALY_MODEL_PATH,
    cause_profile_path: str | Path | None = DEFAULT_CAUSE_PROFILE_PATH,
) -> dict[str, Any]:
    """Return PM2.5 forecasts for one feature-engineered observation."""
    artifact = load_forecast_artifact(str(Path(model_path).resolve()))
    prediction = predict_pm25(pd.DataFrame([input_data]), artifact).iloc[0]
    horizons = [int(value) for value in artifact["horizons"]]
    forecast_values = {
        f"{horizon}h": float(prediction[f"predicted_pm25_t_plus_{horizon}h"])
        for horizon in horizons
    }
    result = {
        "model": artifact["model_name"],
        "station_id": input_data.get(artifact.get("station_column", "station_id")),
        "timestamp": input_data.get(artifact.get("timestamp_column", "timestamp")),
        "forecast_pm25": forecast_values,
        "forecast_screening_levels": {
            horizon: classify_hourly_forecast_proxy(value)
            for horizon, value in forecast_values.items()
        },
        "unit": "ug/m3",
    }
    current_columns = [
        "station_id",
        "timestamp",
        "pm25",
        "pm10",
        "co",
        "no2",
        "so2",
        "o3",
        "temperature",
        "humidity",
        "wind_speed",
        "precipitation",
    ]
    result["current_measurements"] = {
        column: input_data[column]
        for column in current_columns
        if column in input_data and pd.notna(input_data[column])
    }
    result["measurement_units"] = {
        "pm25": "ug/m3",
        "pm10": "ug/m3",
        "co": "ug/m3",
        "no2": "ug/m3",
        "so2": "ug/m3",
        "o3": "ug/m3",
        "temperature": "celsius",
        "humidity": "percent",
        "wind_speed": "km/h",
        "precipitation": "mm",
    }
    recent_mean = input_data.get("pm25_24h_mean")
    recent_source = "rolling_24h_including_current"
    if recent_mean is None:
        recent_mean = input_data.get("pm25_rolling_mean_24h")
        recent_source = "feature_window_t_minus_24h_to_t_minus_1h"
    if recent_mean is not None and pd.notna(recent_mean):
        result["recent_24h_assessment"] = classify_pm25_24h(float(recent_mean))
        result["recent_24h_assessment"]["value_source"] = recent_source
    if anomaly_model_path is not None and Path(anomaly_model_path).exists():
        anomaly_artifact = load_anomaly_artifact(
            str(Path(anomaly_model_path).resolve())
        )
        missing = sorted(set(anomaly_artifact.feature_columns) - set(input_data))
        if missing:
            result["anomaly_detection"] = {
                "available": False,
                "reason": "missing_features",
                "missing_features": missing,
            }
        else:
            anomaly = score_anomalies(pd.DataFrame([input_data]), anomaly_artifact).iloc[0]
            result["anomaly_detection"] = {
                "available": True,
                "is_anomaly": bool(anomaly["is_anomaly"]),
                "is_rule_anomaly": bool(anomaly["is_rule_anomaly"]),
                "is_isolation_anomaly": bool(anomaly["is_isolation_anomaly"]),
                "is_pollution_episode": bool(anomaly["is_pollution_episode"]),
                "requires_attention": bool(anomaly["requires_attention"]),
                "detection_source": str(anomaly["detection_source"]),
                "reason": str(anomaly["anomaly_reason"]),
                "isolation_forest_score": float(anomaly["isolation_forest_score"]),
                "score_threshold": float(anomaly_artifact.score_threshold),
            }
    current_pm25 = float(input_data.get("pm25", 0.0) or 0.0)
    recent_pm25 = float(recent_mean) if recent_mean is not None and pd.notna(recent_mean) else 0.0
    is_pollution_event = current_pm25 >= 100.0 or recent_pm25 >= 75.0
    if is_pollution_event and cause_profile_path is not None and Path(cause_profile_path).exists():
        cause_profile = load_cause_profile(str(Path(cause_profile_path).resolve()))
        cause_required = set(cause_profile.feature_columns + ["timestamp", "station_id"])
        cause_missing = sorted(cause_required - set(input_data))
        if cause_missing:
            result["cause_analysis"] = {
                "available": False,
                "reason": "missing_features",
                "missing_features": cause_missing,
            }
        else:
            analyzed = analyze_pollution_causes(
                pd.DataFrame([input_data]), cause_profile
            ).iloc[0]
            result["cause_analysis"] = {
                "available": True,
                "top_hypothesis": str(analyzed["top_hypothesis"]),
                "top_hypothesis_vi": str(analyzed["top_hypothesis_vi"]),
                "score": float(analyzed["top_hypothesis_score"]),
                "evidence_strength": str(analyzed["evidence_strength"]),
                "causal_claim_allowed": False,
                "evidence": json.loads(str(analyzed["evidence_json"])),
                "limitations": json.loads(str(analyzed["limitations_json"])),
            }
    return result
