"""Tests for auditable environmental data cleaning."""

from __future__ import annotations

import unittest

import pandas as pd

from src.preprocessing.clean_data import clean_environmental_data


class CleanDataTests(unittest.TestCase):
    def test_invalid_duplicate_negative_and_missing_hour(self) -> None:
        raw = pd.DataFrame(
            [
                {"station_id": "A", "timestamp": "2026-01-01T00:00:00+07:00", "pm25": 10},
                {"station_id": "A", "timestamp": "2026-01-01T01:00:00+07:00", "pm25": 20},
                {"station_id": "A", "timestamp": "2026-01-01T01:00:00+07:00", "pm25": 30},
                {"station_id": "A", "timestamp": "2026-01-01T03:00:00+07:00", "pm25": 40},
                {"station_id": "A", "timestamp": "bad-time", "pm25": 50},
                {"station_id": "A", "timestamp": "2026-01-01T04:00:00+07:00", "pm25": -1},
            ]
        )
        result = clean_environmental_data(
            raw,
            value_columns=["pm25"],
            valid_ranges={"pm25": [0, 1000]},
            declared_units={"pm25": "ug/m3"},
            spike_min_periods=20,
        )
        self.assertEqual(result.report["invalid_timestamp_rows_removed"], 1)
        self.assertEqual(result.report["duplicates_removed"], 1)
        self.assertEqual(result.report["missing_timestamp_rows_added"], 1)
        hour_two = result.data[result.data["timestamp"].str.contains("T02:00")].iloc[0]
        self.assertEqual(hour_two["pm25"], 35)
        self.assertTrue(hour_two["is_imputed"])
        self.assertEqual(result.report["negative_values"]["pm25"], 1)

    def test_row_level_unit_conversion(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "station_id": "A",
                    "timestamp": "2026-01-01T00:00:00+07:00",
                    "temperature": 68,
                    "temperature_unit": "fahrenheit",
                }
            ]
        )
        result = clean_environmental_data(
            raw,
            value_columns=["temperature"],
            valid_ranges={"temperature": [-20, 60]},
            declared_units={"temperature": "celsius"},
            spike_min_periods=20,
        )
        self.assertAlmostEqual(result.data.iloc[0]["temperature"], 20)
        self.assertIn("unit_converted:temperature", result.data.iloc[0]["quality_flags"])

    def test_possible_spike_is_flagged_not_deleted(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=30, freq="h", tz="Asia/Ho_Chi_Minh")
        values = [10, 11, 9, 10, 12, 9, 11, 10] * 3 + [10, 11, 9, 10, 10, 200]
        raw = pd.DataFrame(
            {"station_id": "A", "timestamp": timestamps.astype(str), "pm25": values}
        )
        result = clean_environmental_data(
            raw,
            value_columns=["pm25"],
            valid_ranges={"pm25": [0, 1000]},
            declared_units={"pm25": "ug/m3"},
            spike_window=12,
            spike_min_periods=6,
            spike_threshold=5,
            replace_spikes=False,
        )
        last = result.data.iloc[-1]
        self.assertEqual(last["pm25"], 200)
        self.assertTrue(last["is_possible_outlier"])
        self.assertIn("possible_spike:pm25", last["quality_flags"])


if __name__ == "__main__":
    unittest.main()
