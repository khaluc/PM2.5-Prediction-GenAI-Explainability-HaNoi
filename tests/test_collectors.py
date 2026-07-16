"""Unit tests for Hanoi provider normalization without network access."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from src.collection.common import (
    HanoiLocation,
    append_deduplicated_csv,
    split_date_range,
)
from src.collection.sensor_collector import (
    collect_historical_air_quality,
    collect_sensor_data,
)
from src.collection.traffic_collector import collect_traffic_data
from src.collection.weather_collector import (
    collect_historical_weather,
    collect_weather_data,
)


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class FakeClient:
    def __init__(self, payload: dict):
        self.payload = payload

    def get(self, _url: str, params: dict):
        self.params = params
        return FakeResponse(self.payload)


LOCATION = HanoiLocation("HN_TEST", "Hanoi Test", 21.0285, 105.8542)


class CollectorTests(unittest.TestCase):
    def test_air_quality_normalization(self) -> None:
        payload = {
            "hourly": {
                "time": ["2026-07-13T10:00"],
                "pm2_5": [31.2],
                "pm10": [47.1],
                "carbon_monoxide": [120.0],
                "nitrogen_dioxide": [18.0],
                "sulphur_dioxide": [4.0],
                "ozone": [55.0],
                "us_aqi": [92],
            }
        }
        frame = collect_sensor_data([LOCATION], client=FakeClient(payload))
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["pm25"], 31.2)
        self.assertEqual(frame.iloc[0]["source"], "open_meteo_cams")
        self.assertTrue(frame.iloc[0]["timestamp"].endswith("+07:00"))

    def test_weather_normalization(self) -> None:
        payload = {
            "hourly": {
                "time": ["2026-07-13T10:00"],
                "temperature_2m": [33.0],
                "relative_humidity_2m": [70],
                "wind_speed_10m": [8.5],
                "wind_direction_10m": [120],
                "precipitation": [0.0],
                "rain": [0.0],
                "surface_pressure": [1001.2],
                "cloud_cover": [40],
            }
        }
        frame = collect_weather_data([LOCATION], client=FakeClient(payload))
        self.assertEqual(frame.iloc[0]["temperature"], 33.0)
        self.assertEqual(frame.iloc[0]["humidity"], 70)

    def test_traffic_congestion_ratio(self) -> None:
        payload = {
            "flowSegmentData": {
                "currentSpeed": 20,
                "freeFlowSpeed": 40,
                "currentTravelTime": 120,
                "freeFlowTravelTime": 60,
                "confidence": 0.9,
                "roadClosure": False,
                "frc": "FRC2",
            }
        }
        frame = collect_traffic_data(
            [LOCATION], api_key="test", client=FakeClient(payload)
        )
        self.assertEqual(frame.iloc[0]["traffic_congestion"], 0.5)

    def test_csv_append_replaces_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "values.csv"
            first = pd.DataFrame([{"station_id": "A", "timestamp": "T", "value": 1}])
            second = pd.DataFrame([{"station_id": "A", "timestamp": "T", "value": 2}])
            append_deduplicated_csv(
                first, path, unique_columns=["station_id", "timestamp"]
            )
            append_deduplicated_csv(
                second, path, unique_columns=["station_id", "timestamp"]
            )
            saved = pd.read_csv(path)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved.iloc[0]["value"], 2)

    def test_csv_append_can_remove_existing_provider_forecasts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "values.csv"
            pd.DataFrame(
                [
                    {"station_id": "A", "timestamp": "T1", "value": 1, "is_forecast": False},
                    {"station_id": "A", "timestamp": "T2", "value": 2, "is_forecast": True},
                ]
            ).to_csv(path, index=False)
            append_deduplicated_csv(
                pd.DataFrame(
                    [{"station_id": "A", "timestamp": "T3", "value": 3, "is_forecast": False}]
                ),
                path,
                unique_columns=["station_id", "timestamp"],
                exclude_forecasts=True,
            )
            saved = pd.read_csv(path)
            self.assertEqual(saved["timestamp"].tolist(), ["T1", "T3"])
            self.assertFalse(saved["is_forecast"].astype(bool).any())

    def test_historical_collectors_send_bounded_dates(self) -> None:
        air_payload = {
            "hourly": {
                "time": ["2022-08-01T00:00"],
                **{field: [1.0] for field in (
                    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
                    "sulphur_dioxide", "ozone", "us_aqi"
                )},
            }
        }
        air_client = FakeClient(air_payload)
        air = collect_historical_air_quality(
            [LOCATION], date(2022, 8, 1), date(2022, 8, 2), client=air_client
        )
        self.assertEqual(air_client.params["start_date"], "2022-08-01")
        self.assertFalse(bool(air.iloc[0]["is_forecast"]))

        weather_payload = {
            "hourly": {
                "time": ["2022-01-01T00:00"],
                **{field: [1.0] for field in (
                    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                    "wind_direction_10m", "precipitation", "rain",
                    "surface_pressure", "cloud_cover"
                )},
            }
        }
        weather_client = FakeClient(weather_payload)
        weather = collect_historical_weather(
            [LOCATION], date(2022, 1, 1), date(2022, 1, 2), client=weather_client
        )
        self.assertEqual(weather_client.params["models"], "era5")
        self.assertEqual(weather.iloc[0]["source"], "open_meteo_era5_reanalysis")

    def test_date_range_chunks_are_inclusive(self) -> None:
        chunks = list(split_date_range(date(2022, 1, 1), date(2023, 2, 2), 12))
        self.assertEqual(chunks[0], (date(2022, 1, 1), date(2022, 12, 31)))
        self.assertEqual(chunks[1], (date(2023, 1, 1), date(2023, 2, 2)))


if __name__ == "__main__":
    unittest.main()
