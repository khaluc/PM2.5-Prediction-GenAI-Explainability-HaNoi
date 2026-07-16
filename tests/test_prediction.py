"""Tests for forecasting inference contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.models.predict import predict_environment, predict_pm25


class PredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline_artifact = {
            "model_name": "Baseline",
            "horizons": [1, 3, 6],
            "feature_columns": ["pm25"],
            "station_column": "station_id",
            "timestamp_column": "timestamp",
        }

    def test_baseline_repeats_current_pm25(self) -> None:
        frame = pd.DataFrame({"pm25": [12.5, 30.0]})
        result = predict_pm25(frame, self.baseline_artifact)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.iloc[0].tolist(), [12.5, 12.5, 12.5])

    def test_single_row_api_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forecast.joblib"
            joblib.dump(self.baseline_artifact, path)
            result = predict_environment(
                {
                    "timestamp": "2026-05-30T12:00:00+07:00",
                    "station_id": "HN_HOAN_KIEM",
                    "pm25": 42.0,
                },
                path,
                anomaly_model_path=None,
                cause_profile_path=None,
            )
        self.assertEqual(result["model"], "Baseline")
        self.assertEqual(result["forecast_pm25"]["6h"], 42.0)
        self.assertTrue(result["forecast_screening_levels"]["6h"]["screening_only"])
        self.assertEqual(result["unit"], "ug/m3")

    def test_recent_24h_mean_gets_who_assessment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forecast.joblib"
            joblib.dump(self.baseline_artifact, path)
            result = predict_environment(
                {"station_id": "A", "pm25": 40.0, "pm25_24h_mean": 20.0},
                path,
                anomaly_model_path=None,
                cause_profile_path=None,
            )
        self.assertEqual(result["recent_24h_assessment"]["level_code"], 1)
        self.assertEqual(
            result["recent_24h_assessment"]["assessment_type"],
            "who_24h_target_band",
        )

    def test_negative_predictions_are_clipped_for_baseline(self) -> None:
        result = predict_pm25(pd.DataFrame({"pm25": [-2.0]}), self.baseline_artifact)
        self.assertEqual(result.iloc[0].tolist(), [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
