"""Tests for time-aligned TomTom traffic grounding."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.services.traffic_repository import TrafficRepository


def _traffic_csv(path: Path) -> Path:
    pd.DataFrame(
        [
            {
                "timestamp": "2026-07-14T00:10:00+07:00",
                "station_id": "ST_A",
                "current_speed": 20,
                "free_flow_speed": 40,
                "current_travel_time": 120,
                "free_flow_travel_time": 60,
                "traffic_congestion": 0.5,
                "confidence": 0.95,
                "road_closure": False,
                "road_class": "FRC2",
                "source": "tomtom_flow_segment",
            },
            {
                "timestamp": "2026-07-14T00:50:00+07:00",
                "station_id": "ST_A",
                "current_speed": 36,
                "free_flow_speed": 40,
                "traffic_congestion": 0.1,
                "confidence": 0.9,
                "road_closure": False,
                "source": "tomtom_flow_segment",
            },
        ]
    ).to_csv(path, index=False)
    return path


def test_latest_near_selects_closest_record_and_derives_metadata(tmp_path: Path) -> None:
    repository = TrafficRepository(_traffic_csv(tmp_path / "traffic.csv"), max_age_minutes=60)
    result = repository.latest_near("ST_A", "2026-07-14T00:00:00+07:00")
    assert result["available"] is True
    assert result["current_speed_kmh"] == 20.0
    assert result["congestion_percent"] == 50.0
    assert result["age_minutes"] == 10.0
    assert result["causal_claim_allowed"] is False


def test_latest_near_rejects_stale_or_missing_traffic(tmp_path: Path) -> None:
    repository = TrafficRepository(_traffic_csv(tmp_path / "traffic.csv"), max_age_minutes=30)
    stale = repository.latest_near("ST_A", "2026-07-14T04:00:00+07:00")
    assert stale["available"] is False
    assert stale["reason"] == "traffic_data_stale"
    missing = repository.latest_near("ST_UNKNOWN", "2026-07-14T00:00:00+07:00")
    assert missing["available"] is False
    assert missing["reason"] == "traffic_station_not_found"
