"""PM2.5 assessment bands derived from the WHO 2021 24-hour targets.

WHO publishes an Air Quality Guideline (AQG) and interim targets, not an AQI
category system. The project labels in this module are presentation labels; the
``who_band`` field preserves the actual WHO target-band terminology.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

WHO_PM25_24H_THRESHOLDS = {
    "AQG": 15.0,
    "IT-4": 25.0,
    "IT-3": 37.5,
    "IT-2": 50.0,
    "IT-1": 75.0,
}

WHO_REFERENCE_URL = "https://www.who.int/publications/i/item/9789240034228/"

LEVELS: tuple[dict[str, Any], ...] = (
    {
        "level_code": 0,
        "project_level": "guideline_met",
        "project_label_vi": "Đạt khuyến nghị",
        "who_band": "≤ AQG",
        "who_band_vi": "Đạt mức AQG",
        "upper_bound": 15.0,
        "color": "#2e7d32",
    },
    {
        "level_code": 1,
        "project_level": "watch",
        "project_label_vi": "Cần theo dõi",
        "who_band": "AQG–IT-4",
        "who_band_vi": "Vượt AQG, chưa vượt IT-4",
        "upper_bound": 25.0,
        "color": "#f9a825",
    },
    {
        "level_code": 2,
        "project_level": "elevated",
        "project_label_vi": "Ô nhiễm mức 1",
        "who_band": "IT-4–IT-3",
        "who_band_vi": "Vượt IT-4, chưa vượt IT-3",
        "upper_bound": 37.5,
        "color": "#ef6c00",
    },
    {
        "level_code": 3,
        "project_level": "high",
        "project_label_vi": "Ô nhiễm mức 2",
        "who_band": "IT-3–IT-2",
        "who_band_vi": "Vượt IT-3, chưa vượt IT-2",
        "upper_bound": 50.0,
        "color": "#d84315",
    },
    {
        "level_code": 4,
        "project_level": "very_high",
        "project_label_vi": "Ô nhiễm mức 3",
        "who_band": "IT-2–IT-1",
        "who_band_vi": "Vượt IT-2, chưa vượt IT-1",
        "upper_bound": 75.0,
        "color": "#8e24aa",
    },
    {
        "level_code": 5,
        "project_level": "severe",
        "project_label_vi": "Ô nhiễm nghiêm trọng",
        "who_band": "> IT-1",
        "who_band_vi": "Vượt IT-1",
        "upper_bound": None,
        "color": "#6d1b1b",
    },
)


def _level_for_value(value: float) -> dict[str, Any]:
    if not np.isfinite(value) or value < 0:
        raise ValueError("PM2.5 must be a finite, non-negative concentration")
    for level in LEVELS:
        upper_bound = level["upper_bound"]
        if upper_bound is None or value <= upper_bound:
            return level
    raise AssertionError("Unreachable PM2.5 level")


def classify_pm25_24h(value: float) -> dict[str, Any]:
    """Classify a measured rolling/daily 24-hour PM2.5 mean."""
    numeric_value = float(value)
    level = _level_for_value(numeric_value)
    return {
        "pm25_24h_mean": numeric_value,
        "unit": "ug/m3",
        "averaging_period": "24h",
        "standard": "WHO Global Air Quality Guidelines 2021",
        "who_category_system": False,
        "assessment_type": "who_24h_target_band",
        "aqg_statistical_form": "annual 99th percentile of 24-hour means",
        "annual_compliance_determined": False,
        "exceeds_aqg": numeric_value > WHO_PM25_24H_THRESHOLDS["AQG"],
        **level,
        "reference_url": WHO_REFERENCE_URL,
    }


def classify_hourly_forecast_proxy(value: float) -> dict[str, Any]:
    """Screen one hourly forecast against WHO bands without claiming compliance."""
    numeric_value = float(value)
    level = _level_for_value(numeric_value)
    return {
        "pm25_hourly_forecast": numeric_value,
        "unit": "ug/m3",
        "assessment_type": "hourly_screening_proxy",
        "who_comparable": False,
        "screening_only": True,
        **level,
        "note_vi": (
            "Giá trị theo giờ chỉ dùng sàng lọc; cần trung bình PM2.5 24 giờ "
            "để đánh giá theo WHO 2021."
        ),
    }


def classify_pm25_series(values: pd.Series) -> pd.DataFrame:
    """Vectorized classification for 24-hour PM2.5 mean values."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.ge(0) & np.isfinite(numeric)
    codes = pd.cut(
        numeric.where(valid),
        bins=[-np.inf, 15.0, 25.0, 37.5, 50.0, 75.0, np.inf],
        labels=False,
        right=True,
    ).astype("Int64")
    lookup = pd.DataFrame(LEVELS).set_index("level_code")
    result = pd.DataFrame(index=values.index)
    result["who_level_code"] = codes
    for column in [
        "project_level",
        "project_label_vi",
        "who_band",
        "who_band_vi",
        "color",
    ]:
        result[column] = codes.map(lookup[column])
    result["exceeds_who_aqg"] = numeric.gt(WHO_PM25_24H_THRESHOLDS["AQG"]).where(valid)
    result["who_assessment_valid"] = valid
    return result


def add_rolling_24h_assessment(
    data: pd.DataFrame,
    *,
    pm25_column: str = "pm25",
    timestamp_column: str = "timestamp",
    station_column: str = "station_id",
    min_valid_hours: int = 18,
    timezone_name: str = "Asia/Ho_Chi_Minh",
) -> pd.DataFrame:
    """Add a station-local rolling 24-hour mean and WHO target-band fields."""
    if not 1 <= min_valid_hours <= 24:
        raise ValueError("min_valid_hours must be between 1 and 24")
    required = {pm25_column, timestamp_column, station_column}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing assessment columns: {missing}")

    work = data.copy()
    work[timestamp_column] = pd.to_datetime(
        work[timestamp_column], errors="raise", utc=True
    ).dt.tz_convert(timezone_name)
    work[pm25_column] = pd.to_numeric(work[pm25_column], errors="coerce")
    work = work.sort_values([station_column, timestamp_column]).reset_index(drop=True)
    rolling_mean = pd.Series(np.nan, index=work.index, dtype=float)
    valid_hours = pd.Series(0, index=work.index, dtype="int64")

    for _, station_index in work.groupby(station_column, sort=False).groups.items():
        station = work.loc[station_index].sort_values(timestamp_column)
        indexed = station.set_index(timestamp_column)[pm25_column]
        rolling = indexed.rolling("24h", min_periods=min_valid_hours, closed="right")
        rolling_mean.loc[station.index] = rolling.mean().to_numpy()
        coverage = indexed.rolling("24h", min_periods=1, closed="right").count()
        valid_hours.loc[station.index] = coverage.to_numpy(dtype="int64")

    work["pm25_24h_mean"] = rolling_mean.round(3)
    work["pm25_24h_valid_hours"] = valid_hours
    assessment = classify_pm25_series(work["pm25_24h_mean"])
    return pd.concat([work, assessment], axis=1)
