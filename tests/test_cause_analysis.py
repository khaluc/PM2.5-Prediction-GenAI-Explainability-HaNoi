"""Tests for evidence-backed pollution cause hypotheses."""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.analysis.cause_analyzer import (
    BASELINE_FEATURES,
    analyze_pollution_causes,
    fit_cause_profile,
)


def normal_training_data() -> pd.DataFrame:
    random = np.random.default_rng(42)
    size = 400
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=size, freq="h", tz="Asia/Ho_Chi_Minh"),
            "station_id": "A",
            "pm25": random.normal(40, 4, size),
            "pm10": random.normal(60, 6, size),
            "co": random.normal(500, 40, size),
            "no2": random.normal(25, 3, size),
            "so2": random.normal(8, 1, size),
            "o3": random.normal(45, 5, size),
            "temperature": random.normal(25, 2, size),
            "humidity": random.normal(70, 5, size),
            "wind_speed": random.normal(10, 2, size),
            "precipitation": 0.2,
            "pm25_to_pm10_ratio": random.normal(0.67, 0.03, size),
            "pm25_rolling_mean_24h": random.normal(40, 2, size),
        }
    )


class CauseAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = normal_training_data()
        self.profile = fit_cause_profile(self.train, feature_columns=BASELINE_FEATURES)

    def _event(self, **updates) -> pd.DataFrame:
        event = self.train.iloc[[0]].copy()
        event["timestamp"] = pd.Timestamp("2025-01-15T07:00:00+07:00")
        event["pm25"] = 120.0
        event["pm25_rolling_mean_24h"] = 90.0
        event["city_affected_fraction"] = 0.25
        event["city_spatial_cv"] = 0.5
        for key, value in updates.items():
            event[key] = value
        return event

    def test_stagnant_citywide_event_has_supported_hypothesis(self) -> None:
        event = self._event(
            wind_speed=1.0,
            precipitation=0.0,
            humidity=95.0,
            city_affected_fraction=1.0,
            city_spatial_cv=0.05,
        )
        result = analyze_pollution_causes(event, self.profile).iloc[0]
        self.assertIn(
            result["top_hypothesis"],
            {"regional_accumulation", "atmospheric_stagnation"},
        )
        self.assertGreaterEqual(len(json.loads(result["evidence_json"])), 3)

    def test_dry_windy_coarse_event_points_to_dust(self) -> None:
        event = self._event(
            pm10=250.0,
            pm25_to_pm10_ratio=0.35,
            wind_speed=25.0,
            humidity=35.0,
            precipitation=0.0,
        )
        result = analyze_pollution_causes(event, self.profile).iloc[0]
        self.assertEqual(result["top_hypothesis"], "coarse_dust_resuspension")


if __name__ == "__main__":
    unittest.main()
