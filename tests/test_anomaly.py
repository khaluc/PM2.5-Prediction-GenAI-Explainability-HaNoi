"""Tests for hybrid rule and Isolation Forest anomaly detection."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.train_anomaly import (
    apply_threshold_rules,
    fit_anomaly_model,
    score_anomalies,
)


RULES = {
    "hard_ranges": {"pm25": [0, 1000], "pm10": [0, 1500]},
    "pm25_jump_abs_1h": 75,
    "pm25_jump_abs_3h": 100,
    "flatline_std_24h": 0.2,
    "flatline_min_mean": 5,
    "pm25_pm10_ratio_max": 1.1,
    "pm25_pm10_tolerance": 5,
    "pm25_extreme": 200,
    "pollution_episode_24h": 75,
}


class AnomalyRuleTests(unittest.TestCase):
    def test_each_rule_and_episode_are_distinct(self) -> None:
        frame = pd.DataFrame(
            {
                "pm25": [-1, 40, 40, 50, 210, 90],
                "pm10": [20, 60, 60, 20, 250, 120],
                "pm25_change_1h": [0, 80, 0, 0, 0, 0],
                "pm25_change_3h": [0, 0, 0, 0, 0, 0],
                "pm25_rolling_std_24h": [5, 5, 0.1, 5, 5, 5],
                "pm25_rolling_mean_24h": [20, 20, 20, 20, 20, 80],
                "is_possible_outlier": [False] * 6,
            }
        )
        result = apply_threshold_rules(frame, RULES)
        self.assertTrue(result.loc[0, "rule_invalid_range"])
        self.assertTrue(result.loc[1, "rule_rapid_pm25_change"])
        self.assertTrue(result.loc[2, "rule_flatline"])
        self.assertTrue(result.loc[3, "rule_pm_inconsistency"])
        self.assertTrue(result.loc[4, "rule_extreme_pm25"])
        self.assertTrue(result.loc[5, "is_pollution_episode"])
        self.assertFalse(result.loc[5, "is_rule_anomaly"])


class IsolationForestTests(unittest.TestCase):
    def test_training_and_scoring_contract(self) -> None:
        random = np.random.default_rng(42)
        train = pd.DataFrame(
            {
                "pm25": random.normal(40, 5, 300),
                "pm10": random.normal(60, 7, 300),
                "pm25_change_1h": random.normal(0, 2, 300),
                "pm25_change_3h": random.normal(0, 4, 300),
                "pm25_rolling_std_24h": random.uniform(4, 10, 300),
                "pm25_rolling_mean_24h": random.normal(40, 3, 300),
                "is_possible_outlier": False,
            }
        )
        features = ["pm25", "pm10", "pm25_change_1h"]
        bundle = fit_anomaly_model(
            train,
            feature_columns=features,
            rules=RULES,
            contamination=0.05,
            n_estimators=50,
            max_samples=128,
        )
        test = train.iloc[:5].copy()
        test.loc[0, ["pm25", "pm10", "pm25_change_1h"]] = [500, 600, 300]
        result = score_anomalies(test, bundle)
        self.assertEqual(len(result), 5)
        self.assertIn("is_anomaly", result)
        self.assertTrue(result.loc[0, "is_anomaly"])
        self.assertTrue(result.loc[0, "is_rule_anomaly"])


if __name__ == "__main__":
    unittest.main()
