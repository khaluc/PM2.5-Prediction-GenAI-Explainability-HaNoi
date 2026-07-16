"""Tests for PM2.5 forecast metrics and model input preparation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.evaluate import evaluate_forecast, evaluate_multi_horizon, safe_mape
from src.models.train_forecast import build_lstm_sequences, prepare_tree_matrix


class ForecastMetricTests(unittest.TestCase):
    def test_regression_metrics(self) -> None:
        metrics = evaluate_forecast([10, 20, 30], [12, 18, 33])
        self.assertAlmostEqual(metrics["MAE"], 7 / 3)
        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(17 / 3))
        self.assertLess(metrics["R2"], 1.0)
        self.assertGreater(metrics["MAPE"], 0.0)

    def test_safe_mape_handles_zero(self) -> None:
        self.assertAlmostEqual(safe_mape([0, 10], [1, 8]), 60.0)

    def test_multi_horizon_shape_must_match(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_multi_horizon(
                np.ones((4, 2)),
                np.ones((4, 1)),
                model_name="test",
                horizons=[1, 3],
                split_name="validation",
            )


class ForecastInputTests(unittest.TestCase):
    def setUp(self) -> None:
        first = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        second = pd.date_range("2026-01-02", periods=4, freq="h", tz="UTC")
        timestamps = first.append(second)
        self.frame = pd.DataFrame(
            {
                "timestamp": timestamps.astype(str),
                "station_id": ["A"] * 5 + ["B"] * 4,
                "pm25": np.arange(9, dtype=float) + 10,
                "humidity": np.arange(9, dtype=float) + 60,
                "target_1h": np.arange(9, dtype=float) + 11,
                "target_3h": np.arange(9, dtype=float) + 13,
            }
        )

    def test_tree_matrix_has_deterministic_station_one_hot(self) -> None:
        matrix, columns = prepare_tree_matrix(
            self.frame, ["pm25", "humidity"], ["A", "B"]
        )
        self.assertEqual(matrix.shape, (9, 4))
        self.assertEqual(columns[-2:], ["station__A", "station__B"])
        np.testing.assert_array_equal(matrix[0, -2:], [1, 0])
        np.testing.assert_array_equal(matrix[-1, -2:], [0, 1])

    def test_lstm_windows_stay_inside_station(self) -> None:
        sequences = build_lstm_sequences(
            self.frame,
            feature_columns=["pm25", "humidity"],
            target_columns=["target_1h", "target_3h"],
            station_categories=["A", "B"],
            sequence_length=3,
            fit_scalers=True,
        )
        self.assertEqual(sequences.x.shape, (5, 3, 3))
        self.assertEqual(sequences.effective_features[-1], "station__code")
        np.testing.assert_array_equal(
            sequences.y_original[:, 0], [13, 14, 15, 18, 19]
        )

    def test_lstm_stride_reduces_only_training_windows(self) -> None:
        sequences = build_lstm_sequences(
            self.frame,
            feature_columns=["pm25"],
            target_columns=["target_1h"],
            station_categories=["A", "B"],
            sequence_length=3,
            fit_scalers=True,
            stride=2,
        )
        self.assertEqual(len(sequences.x), 3)


if __name__ == "__main__":
    unittest.main()
