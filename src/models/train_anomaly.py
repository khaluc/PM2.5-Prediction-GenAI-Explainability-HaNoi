"""Hybrid threshold-rule and Isolation Forest anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


@dataclass
class AnomalyModelBundle:
    feature_columns: list[str]
    imputer: SimpleImputer
    scaler: RobustScaler
    model: IsolationForest
    score_threshold: float
    contamination: float
    rules: dict[str, Any]


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def apply_threshold_rules(
    frame: pd.DataFrame,
    rules: dict[str, Any],
) -> pd.DataFrame:
    """Evaluate deterministic sensor-quality and pollution-episode rules."""
    pm25 = _numeric(frame, "pm25")
    pm10 = _numeric(frame, "pm10")
    co = _numeric(frame, "co")
    no2 = _numeric(frame, "no2")
    so2 = _numeric(frame, "so2")
    o3 = _numeric(frame, "o3")
    change_1h = _numeric(frame, "pm25_change_1h")
    change_3h = _numeric(frame, "pm25_change_3h")
    rolling_std = _numeric(frame, "pm25_rolling_std_24h")
    rolling_mean = _numeric(frame, "pm25_rolling_mean_24h")

    hard_ranges = rules.get("hard_ranges", {})
    invalid = pd.Series(False, index=frame.index)
    for column, values in {
        "pm25": pm25,
        "pm10": pm10,
        "co": co,
        "no2": no2,
        "so2": so2,
        "o3": o3,
    }.items():
        if column not in hard_ranges:
            continue
        lower, upper = hard_ranges[column]
        invalid |= values.notna() & (values.lt(float(lower)) | values.gt(float(upper)))

    cleaning_spike = (
        frame.get("is_possible_outlier", pd.Series(False, index=frame.index))
        .fillna(False)
        .astype(bool)
    )
    rapid_change = change_1h.abs().ge(float(rules.get("pm25_jump_abs_1h", 75.0)))
    rapid_change |= change_3h.abs().ge(float(rules.get("pm25_jump_abs_3h", 100.0)))
    flatline = rolling_std.le(float(rules.get("flatline_std_24h", 0.2)))
    flatline &= rolling_mean.ge(float(rules.get("flatline_min_mean", 5.0)))
    pm_inconsistency = pm25.gt(
        pm10 * float(rules.get("pm25_pm10_ratio_max", 1.1))
        + float(rules.get("pm25_pm10_tolerance", 5.0))
    )
    extreme_pm25 = pm25.ge(float(rules.get("pm25_extreme", 200.0)))
    pollution_episode = rolling_mean.gt(
        float(rules.get("pollution_episode_24h", 75.0))
    )

    result = pd.DataFrame(
        {
            "rule_invalid_range": invalid,
            "rule_cleaning_spike": cleaning_spike,
            "rule_rapid_pm25_change": rapid_change,
            "rule_flatline": flatline,
            "rule_pm_inconsistency": pm_inconsistency,
            "rule_extreme_pm25": extreme_pm25,
            "is_pollution_episode": pollution_episode,
        },
        index=frame.index,
    )
    rule_columns = [column for column in result if column.startswith("rule_")]
    result["is_rule_anomaly"] = result[rule_columns].any(axis=1)

    reasons = np.full(len(frame), "", dtype=object)
    for column in rule_columns:
        label = column.removeprefix("rule_")
        mask = result[column].to_numpy(dtype=bool)
        reasons[mask] = np.where(
            reasons[mask] == "", label, reasons[mask] + ";" + label
        )
    result["rule_reason"] = reasons
    return result


def fit_anomaly_model(
    train: pd.DataFrame,
    *,
    feature_columns: list[str],
    rules: dict[str, Any],
    contamination: float = 0.03,
    random_state: int = 42,
    n_estimators: int = 300,
    max_samples: int = 4096,
) -> AnomalyModelBundle:
    """Fit all preprocessing and Isolation Forest using train only."""
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5")
    missing = sorted(set(feature_columns) - set(train.columns))
    if missing:
        raise ValueError(f"Missing anomaly features: {missing}")
    matrix = train[feature_columns].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(matrix).astype(np.float32)
    scaler = RobustScaler(quantile_range=(10, 90))
    scaled = scaler.fit_transform(imputed).astype(np.float32)
    model = IsolationForest(
        n_estimators=int(n_estimators),
        max_samples=min(int(max_samples), len(train)),
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=int(random_state),
    )
    model.fit(scaled)
    train_scores = model.score_samples(scaled)
    score_threshold = float(np.quantile(train_scores, contamination))
    return AnomalyModelBundle(
        feature_columns=list(feature_columns),
        imputer=imputer,
        scaler=scaler,
        model=model,
        score_threshold=score_threshold,
        contamination=float(contamination),
        rules=dict(rules),
    )


def score_anomalies(
    frame: pd.DataFrame,
    bundle: AnomalyModelBundle,
) -> pd.DataFrame:
    """Return rule, Isolation Forest and combined anomaly decisions."""
    missing = sorted(set(bundle.feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing anomaly features: {missing}")
    matrix = frame[bundle.feature_columns].replace([np.inf, -np.inf], np.nan)
    imputed = bundle.imputer.transform(matrix).astype(np.float32)
    scaled = bundle.scaler.transform(imputed).astype(np.float32)
    scores = bundle.model.score_samples(scaled)
    isolation_anomaly = scores <= bundle.score_threshold
    rule_result = apply_threshold_rules(frame, bundle.rules)
    result = rule_result.copy()
    result["isolation_forest_score"] = scores
    result["isolation_forest_margin"] = bundle.score_threshold - scores
    result["is_isolation_anomaly"] = isolation_anomaly
    result["is_anomaly"] = result["is_rule_anomaly"] | isolation_anomaly
    result["requires_attention"] = result["is_anomaly"] | result["is_pollution_episode"]
    result["detection_source"] = np.select(
        [
            result["is_rule_anomaly"] & result["is_isolation_anomaly"],
            result["is_rule_anomaly"],
            result["is_isolation_anomaly"],
        ],
        ["both", "rule", "isolation_forest"],
        default="none",
    )
    result["anomaly_reason"] = result["rule_reason"]
    isolation_only = result["is_isolation_anomaly"] & result["rule_reason"].eq("")
    both = result["is_isolation_anomaly"] & result["rule_reason"].ne("")
    result.loc[isolation_only, "anomaly_reason"] = "multivariate_pattern"
    result.loc[both, "anomaly_reason"] += ";multivariate_pattern"
    return result
