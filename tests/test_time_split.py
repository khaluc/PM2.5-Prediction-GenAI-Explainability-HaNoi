"""Tests for chronological splitting and boundary purging."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.time_split import split_time_series


class TimeSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        timestamps = pd.date_range(
            "2024-12-31T12:00:00+07:00",
            "2026-01-01T12:00:00+07:00",
            freq="h",
        )
        self.data = pd.concat(
            [
                pd.DataFrame(
                    {
                        "timestamp": timestamps.astype(str),
                        "station_id": station,
                        "value": range(len(timestamps)),
                    }
                )
                for station in ["A", "B"]
            ],
            ignore_index=True,
        )

    def test_split_is_chronological_and_purged(self) -> None:
        result = split_time_series(
            self.data,
            validation_start="2025-01-01",
            test_start="2026-01-01",
            max_target_horizon_hours=6,
        )
        train_end = pd.to_datetime(result.train["timestamp"], utc=True).max()
        validation_end = pd.to_datetime(result.validation["timestamp"], utc=True).max()
        self.assertLess(
            train_end + pd.Timedelta(hours=6),
            pd.Timestamp("2025-01-01", tz="Asia/Ho_Chi_Minh").tz_convert("UTC"),
        )
        self.assertLess(
            validation_end + pd.Timedelta(hours=6),
            pd.Timestamp("2026-01-01", tz="Asia/Ho_Chi_Minh").tz_convert("UTC"),
        )
        self.assertEqual(result.report["train_boundary_rows_purged"], 12)
        self.assertEqual(result.report["validation_boundary_rows_purged"], 12)
        self.assertTrue(all(result.report["chronology_checks"].values()))

    def test_invalid_boundary_order_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            split_time_series(
                self.data,
                validation_start="2026-01-01",
                test_start="2025-01-01",
                max_target_horizon_hours=6,
            )

    def test_duplicate_keys_are_rejected(self) -> None:
        duplicated = pd.concat([self.data, self.data.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            split_time_series(
                duplicated,
                validation_start="2025-01-01",
                test_start="2026-01-01",
                max_target_horizon_hours=6,
            )


if __name__ == "__main__":
    unittest.main()
