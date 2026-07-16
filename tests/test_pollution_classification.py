"""Tests for WHO-derived PM2.5 target-band assessment."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.assessment.who_pm25 import (
    add_rolling_24h_assessment,
    classify_hourly_forecast_proxy,
    classify_pm25_24h,
    classify_pm25_series,
)


class WHOClassificationTests(unittest.TestCase):
    def test_boundaries_follow_who_interim_targets(self) -> None:
        cases = [
            (0, 0),
            (15, 0),
            (15.01, 1),
            (25, 1),
            (37.5, 2),
            (50, 3),
            (75, 4),
            (75.01, 5),
        ]
        for value, expected_code in cases:
            with self.subTest(value=value):
                self.assertEqual(classify_pm25_24h(value)["level_code"], expected_code)

    def test_invalid_concentration_is_rejected(self) -> None:
        for value in [-1, np.nan, np.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                classify_pm25_24h(value)

    def test_hourly_value_is_explicitly_screening_only(self) -> None:
        result = classify_hourly_forecast_proxy(80)
        self.assertTrue(result["screening_only"])
        self.assertFalse(result["who_comparable"])
        self.assertEqual(result["level_code"], 5)

    def test_vectorized_invalid_values_have_no_level(self) -> None:
        result = classify_pm25_series(pd.Series([10, -1, np.nan, 80]))
        self.assertEqual(result["who_assessment_valid"].tolist(), [True, False, False, True])
        self.assertTrue(pd.isna(result.loc[1, "who_level_code"]))

    def test_rolling_mean_is_station_local(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
        frame = pd.concat(
            [
                pd.DataFrame(
                    {"timestamp": timestamps, "station_id": "A", "pm25": 10.0}
                ),
                pd.DataFrame(
                    {"timestamp": timestamps, "station_id": "B", "pm25": 100.0}
                ),
            ],
            ignore_index=True,
        )
        assessed = add_rolling_24h_assessment(frame, min_valid_hours=24)
        last = assessed.groupby("station_id", sort=False).tail(1).set_index("station_id")
        self.assertEqual(last.loc["A", "who_level_code"], 0)
        self.assertEqual(last.loc["B", "who_level_code"], 5)
        self.assertEqual(last.loc["A", "pm25_24h_valid_hours"], 24)


if __name__ == "__main__":
    unittest.main()
