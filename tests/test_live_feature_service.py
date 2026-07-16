"""Tests for live feature generation used by ML inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.services.live_feature_service import build_latest_feature_row
from src.services.monitoring_repository import DataSourceUnavailableError


def _observations(periods: int = 169) -> pd.DataFrame:
    timestamp = pd.date_range("2026-07-01", periods=periods, freq="h", tz="Asia/Ho_Chi_Minh")
    values = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "station_id": "HN_BA_DINH",
            "pm25": 10 + values,
            "pm10": 20 + values,
            "co": 500 + values,
            "no2": 20 + values / 10,
            "so2": 5 + values / 20,
            "o3": 40 + values / 5,
            "us_aqi": 80 + values / 10,
            "temperature": 30 + values / 100,
            "humidity": 70 - values / 100,
            "wind_speed": 8 + values / 100,
            "wind_direction": 120,
            "precipitation": 0.0,
            "rain": 0.0,
            "surface_pressure": 1000,
            "cloud_cover": 50,
            "data_quality_score": 1.0,
            "is_possible_outlier": False,
        }
    )


def test_build_latest_feature_row_uses_contiguous_observations() -> None:
    features = build_latest_feature_row(_observations())
    assert features["station_id"] == "HN_BA_DINH"
    assert features["pm25"] == pytest.approx(178.0)
    assert features["pm25_lag_1h"] == pytest.approx(177.0)
    assert features["pm25_lag_168h"] == pytest.approx(10.0)
    assert features["pm25_rolling_mean_168h"] == pytest.approx(93.5)


def test_build_latest_feature_row_rejects_latest_segment_after_gap() -> None:
    frame = _observations()
    frame.loc[100:, "timestamp"] = frame.loc[100:, "timestamp"] + pd.Timedelta(hours=4)
    with pytest.raises(DataSourceUnavailableError, match="consecutive hourly observations"):
        build_latest_feature_row(frame)
