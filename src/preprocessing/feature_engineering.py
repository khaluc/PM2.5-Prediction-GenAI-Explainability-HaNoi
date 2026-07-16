"""Leakage-safe time-series features for PM2.5 forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


POLLUTANT_COLUMNS = ["pm10", "co", "no2", "so2", "o3"]
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
BASE_FEATURE_COLUMNS = [
    "pm25",
    *POLLUTANT_COLUMNS,
    "us_aqi",
    *WEATHER_COLUMNS,
    "data_quality_score",
    "is_possible_outlier",
]


@dataclass
class FeatureResult:
    data: pd.DataFrame
    feature_columns: list[str]
    target_columns: list[str]
    feature_groups: dict[str, list[str]]
    report: dict[str, Any]


def _group_shift(
    frame: pd.DataFrame, column: str, hours: int, group_column: str
) -> pd.Series:
    return frame.groupby(group_column, sort=False)[column].shift(hours)


def _past_rolling(
    frame: pd.DataFrame,
    column: str,
    window: int,
    group_column: str,
    statistic: str,
) -> pd.Series:
    """Calculate rolling values from t-1 backwards, never including row t."""
    shifted = _group_shift(frame, column, 1, group_column)
    grouped = shifted.groupby(frame[group_column], sort=False)
    if statistic == "mean":
        return grouped.transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
    if statistic == "std":
        return grouped.transform(
            lambda values: values.rolling(window, min_periods=window).std(ddof=0)
        )
    if statistic == "min":
        return grouped.transform(
            lambda values: values.rolling(window, min_periods=window).min()
        )
    if statistic == "max":
        return grouped.transform(
            lambda values: values.rolling(window, min_periods=window).max()
        )
    if statistic == "sum":
        return grouped.transform(
            lambda values: values.rolling(window, min_periods=window).sum()
        )
    raise ValueError(f"Unsupported rolling statistic: {statistic}")


def _past_ewm(
    frame: pd.DataFrame,
    column: str,
    halflife: int,
    group_column: str,
) -> pd.Series:
    shifted = _group_shift(frame, column, 1, group_column)
    return shifted.groupby(frame[group_column], sort=False).transform(
        lambda values: values.ewm(halflife=halflife, adjust=False).mean()
    )


def build_features(
    data: pd.DataFrame,
    *,
    group_column: str = "station_id",
    timestamp_column: str = "timestamp",
    target_column: str = "pm25",
    timezone_name: str = "Asia/Ho_Chi_Minh",
    target_horizons: list[int] | None = None,
    pm25_lags: list[int] | None = None,
    pollutant_lags: list[int] | None = None,
    weather_lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    weather_rolling_windows: list[int] | None = None,
    ewm_halflives: list[int] | None = None,
    drop_incomplete_rows: bool = True,
) -> FeatureResult:
    """Build supervised hourly features independently within each station."""
    target_horizons = target_horizons or [1, 3, 6]
    pm25_lags = sorted(
        set(pm25_lags or [1, 2, 3, 6, 12, 24, 48, 72, 168]) | {1, 3, 6}
    )
    pollutant_lags = pollutant_lags or [1, 3, 24]
    weather_lags = weather_lags or [1, 3, 6, 24]
    rolling_windows = rolling_windows or [3, 6, 12, 24, 72, 168]
    weather_rolling_windows = weather_rolling_windows or [6, 24]
    ewm_halflives = ewm_halflives or [6, 24]

    required = {group_column, timestamp_column, target_column, *BASE_FEATURE_COLUMNS}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")

    frame = data.copy()
    input_rows = len(frame)
    frame[timestamp_column] = pd.to_datetime(
        frame[timestamp_column], errors="coerce", utc=True
    ).dt.tz_convert(timezone_name)
    invalid_timestamps = int(frame[timestamp_column].isna().sum())
    frame = frame.dropna(subset=[timestamp_column, group_column]).copy()
    duplicate_rows = int(
        frame.duplicated(subset=[group_column, timestamp_column]).sum()
    )
    if duplicate_rows:
        raise ValueError(
            f"Feature input contains {duplicate_rows} duplicate station/timestamp keys"
        )
    frame = frame.sort_values([group_column, timestamp_column]).reset_index(drop=True)

    numeric_columns = [
        column for column in BASE_FEATURE_COLUMNS if column != "is_possible_outlier"
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["is_possible_outlier"] = (
        frame["is_possible_outlier"].fillna(False).astype(bool).astype("int8")
    )

    feature_groups: dict[str, list[str]] = {
        "current": list(BASE_FEATURE_COLUMNS),
        "calendar": [],
        "cyclical": [],
        "pm25_lags": [],
        "pollutant_lags": [],
        "weather_lags": [],
        "rolling": [],
        "ewm": [],
        "change_and_interaction": [],
    }

    timestamp = frame[timestamp_column]
    calendar_values = {
        "hour": timestamp.dt.hour,
        "day_of_week": timestamp.dt.dayofweek,
        "day_of_month": timestamp.dt.day,
        "day_of_year": timestamp.dt.dayofyear,
        "week_of_year": timestamp.dt.isocalendar().week.astype(int),
        "month": timestamp.dt.month,
        "quarter": timestamp.dt.quarter,
        "is_weekend": timestamp.dt.dayofweek.ge(5).astype("int8"),
    }
    for column, values in calendar_values.items():
        frame[column] = values
        feature_groups["calendar"].append(column)

    cyclical_values = {
        "hour_sin": np.sin(2 * np.pi * frame["hour"] / 24),
        "hour_cos": np.cos(2 * np.pi * frame["hour"] / 24),
        "dow_sin": np.sin(2 * np.pi * frame["day_of_week"] / 7),
        "dow_cos": np.cos(2 * np.pi * frame["day_of_week"] / 7),
        "month_sin": np.sin(2 * np.pi * (frame["month"] - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (frame["month"] - 1) / 12),
    }
    for column, values in cyclical_values.items():
        frame[column] = values
        feature_groups["cyclical"].append(column)

    for lag in pm25_lags:
        column = f"pm25_lag_{lag}h"
        frame[column] = _group_shift(frame, target_column, lag, group_column)
        feature_groups["pm25_lags"].append(column)

    for pollutant in POLLUTANT_COLUMNS:
        for lag in pollutant_lags:
            column = f"{pollutant}_lag_{lag}h"
            frame[column] = _group_shift(frame, pollutant, lag, group_column)
            feature_groups["pollutant_lags"].append(column)

    for weather in ["temperature", "humidity", "wind_speed", "precipitation"]:
        for lag in weather_lags:
            column = f"{weather}_lag_{lag}h"
            frame[column] = _group_shift(frame, weather, lag, group_column)
            feature_groups["weather_lags"].append(column)

    rolling_features: dict[str, pd.Series] = {}
    for window in rolling_windows:
        for statistic in ["mean", "std", "min", "max"]:
            column = f"pm25_rolling_{statistic}_{window}h"
            rolling_features[column] = _past_rolling(
                frame, target_column, window, group_column, statistic
            )
            feature_groups["rolling"].append(column)

    for weather in ["humidity", "wind_speed", "temperature"]:
        for window in weather_rolling_windows:
            column = f"{weather}_rolling_mean_{window}h"
            rolling_features[column] = _past_rolling(
                frame, weather, window, group_column, "mean"
            )
            feature_groups["rolling"].append(column)
    for window in weather_rolling_windows:
        column = f"precipitation_rolling_sum_{window}h"
        rolling_features[column] = _past_rolling(
            frame, "precipitation", window, group_column, "sum"
        )
        feature_groups["rolling"].append(column)

    for halflife in ewm_halflives:
        column = f"pm25_ewm_halflife_{halflife}h"
        rolling_features[column] = _past_ewm(
            frame, target_column, halflife, group_column
        )
        feature_groups["ewm"].append(column)
    frame = pd.concat(
        [frame, pd.DataFrame(rolling_features, index=frame.index)], axis=1
    )

    interaction_values = {
        "pm25_change_1h": frame[target_column] - frame["pm25_lag_1h"],
        "pm25_change_3h": frame[target_column] - frame["pm25_lag_3h"],
        "pm25_trend_6h": (frame["pm25_lag_1h"] - frame["pm25_lag_6h"]) / 5,
        "pm25_to_pm10_ratio": frame["pm25"] / frame["pm10"].replace(0, np.nan),
        "wind_u": -frame["wind_speed"] * np.sin(np.deg2rad(frame["wind_direction"])),
        "wind_v": -frame["wind_speed"] * np.cos(np.deg2rad(frame["wind_direction"])),
        "humidity_temperature_interaction": frame["humidity"] * frame["temperature"],
        "stagnation_proxy": frame["humidity"] / (frame["wind_speed"] + 1),
    }
    for column, values in interaction_values.items():
        feature_groups["change_and_interaction"].append(column)
    frame = pd.concat(
        [frame, pd.DataFrame(interaction_values, index=frame.index)], axis=1
    )

    target_columns: list[str] = []
    target_values: dict[str, pd.Series] = {}
    for horizon in target_horizons:
        column = f"target_pm25_t_plus_{horizon}h"
        target_values[column] = frame.groupby(group_column, sort=False)[target_column].shift(
            -horizon
        )
        target_columns.append(column)
    frame = pd.concat(
        [frame, pd.DataFrame(target_values, index=frame.index)], axis=1
    )

    feature_columns = [
        column for columns in feature_groups.values() for column in columns
    ]
    required_model_columns = feature_columns + target_columns
    missing_before_drop = {
        column: int(frame[column].isna().sum()) for column in required_model_columns
    }
    if drop_incomplete_rows:
        frame = frame.dropna(subset=required_model_columns).reset_index(drop=True)

    for column in feature_columns + target_columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].astype("float32")

    report = {
        "input_rows": input_rows,
        "output_rows": len(frame),
        "rows_removed": input_rows - len(frame),
        "stations": int(frame[group_column].nunique()),
        "start": frame[timestamp_column].min().isoformat(),
        "end": frame[timestamp_column].max().isoformat(),
        "invalid_timestamps_removed": invalid_timestamps,
        "feature_count": len(feature_columns),
        "target_count": len(target_columns),
        "missing_before_drop": missing_before_drop,
        "remaining_missing_model_values": int(
            frame[required_model_columns].isna().sum().sum()
        ),
        "leakage_rule": "All lag, rolling and EWM features use only t-1 or earlier values",
    }
    frame[timestamp_column] = frame[timestamp_column].map(lambda value: value.isoformat())
    return FeatureResult(
        data=frame,
        feature_columns=feature_columns,
        target_columns=target_columns,
        feature_groups=feature_groups,
        report=report,
    )
