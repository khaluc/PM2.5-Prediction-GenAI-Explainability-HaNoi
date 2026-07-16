"""Tests that time-series features are station-safe and leakage-safe."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.preprocessing.feature_engineering import (
    BASE_FEATURE_COLUMNS,
    build_features,
)


def sample_station(station_id: str, offset: float = 0) -> pd.DataFrame:
    size = 24
    timestamps = pd.date_range(
        "2026-01-01", periods=size, freq="h", tz="Asia/Ho_Chi_Minh"
    )
    frame = pd.DataFrame(
        {
            "station_id": station_id,
            "timestamp": timestamps.astype(str),
            "pm25": np.arange(size, dtype=float) + offset,
            "pm10": np.arange(size, dtype=float) + offset + 10,
            "co": 100.0,
            "no2": 20.0,
            "so2": 5.0,
            "o3": 30.0,
            "us_aqi": 80.0,
            "temperature": 25.0,
            "humidity": 70.0,
            "wind_speed": 10.0,
            "wind_direction": 90.0,
            "precipitation": 0.0,
            "rain": 0.0,
            "surface_pressure": 1000.0,
            "cloud_cover": 40.0,
            "data_quality_score": 1.0,
            "is_possible_outlier": False,
        }
    )
    assert set(BASE_FEATURE_COLUMNS).issubset(frame.columns)
    return frame


class FeatureEngineeringTests(unittest.TestCase):
    def test_lags_rolling_and_targets_do_not_leak(self) -> None:
        result = build_features(
            sample_station("A"),
            target_horizons=[1, 3],
            pm25_lags=[1, 3],
            pollutant_lags=[1],
            weather_lags=[1],
            rolling_windows=[3],
            weather_rolling_windows=[3],
            ewm_halflives=[3],
            drop_incomplete_rows=False,
        )
        row = result.data.iloc[6]
        self.assertEqual(row["pm25"], 6)
        self.assertEqual(row["pm25_lag_1h"], 5)
        self.assertEqual(row["pm25_lag_3h"], 3)
        self.assertEqual(row["pm25_rolling_mean_3h"], 4)
        self.assertEqual(row["target_pm25_t_plus_1h"], 7)
        self.assertEqual(row["target_pm25_t_plus_3h"], 9)

    def test_station_boundaries_are_not_crossed(self) -> None:
        data = pd.concat(
            [sample_station("A", 0), sample_station("B", 1000)], ignore_index=True
        )
        result = build_features(
            data,
            target_horizons=[1],
            pm25_lags=[1],
            pollutant_lags=[1],
            weather_lags=[1],
            rolling_windows=[3],
            weather_rolling_windows=[3],
            ewm_halflives=[3],
            drop_incomplete_rows=False,
        )
        first_b = result.data[result.data["station_id"] == "B"].iloc[0]
        self.assertTrue(pd.isna(first_b["pm25_lag_1h"]))

    def test_drop_incomplete_rows_and_wind_components(self) -> None:
        result = build_features(
            sample_station("A"),
            target_horizons=[1],
            pm25_lags=[1],
            pollutant_lags=[1],
            weather_lags=[1],
            rolling_windows=[3],
            weather_rolling_windows=[3],
            ewm_halflives=[3],
            drop_incomplete_rows=True,
        )
        self.assertEqual(result.report["remaining_missing_model_values"], 0)
        self.assertAlmostEqual(result.data.iloc[0]["wind_u"], -10.0, places=5)
        self.assertAlmostEqual(result.data.iloc[0]["wind_v"], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
