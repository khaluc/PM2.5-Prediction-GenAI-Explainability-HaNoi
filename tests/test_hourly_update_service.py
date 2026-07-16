"""Tests for exact-hour data collection and ML refresh scheduling."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from src.services.hourly_update_service import HourlyUpdateService, next_hourly_run


HANOI = ZoneInfo("Asia/Ho_Chi_Minh")


def test_next_hourly_run_uses_current_hour_when_provider_delay_is_ahead() -> None:
    before_delay = datetime(2026, 7, 15, 19, 0, 20, tzinfo=HANOI)
    after_delay = datetime(2026, 7, 15, 19, 0, 50, tzinfo=HANOI)
    assert next_hourly_run(before_delay, delay_seconds=45) == datetime(
        2026, 7, 15, 19, 0, 45, tzinfo=HANOI
    )
    assert next_hourly_run(after_delay, delay_seconds=45) == datetime(
        2026, 7, 15, 20, 0, 45, tzinfo=HANOI
    )


def test_run_once_collects_then_refreshes_forecasts(tmp_path) -> None:
    calls: list[str] = []

    def collect():
        calls.append("collection")
        return {"air_quality": 8, "weather": 8, "traffic": 8}

    def forecast():
        calls.append("forecast")
        return {"attempted": 8, "succeeded": 8, "failed": 0}

    service = HourlyUpdateService(
        collect,
        forecast_refresher=forecast,
        status_path=tmp_path / "status.json",
        now=lambda: datetime(2026, 7, 15, 19, 1, tzinfo=HANOI),
    )
    assert service.run_once("test") is True
    assert calls == ["collection", "forecast"]
    status = service.status()
    assert status["running"] is False
    assert status["last_result"]["collection"]["air_quality"] == 8
    assert status["last_result"]["forecast_refresh"]["succeeded"] == 8
    assert status["consecutive_failures"] == 0
    assert (tmp_path / "status.json").is_file()


def test_provider_failure_is_recorded_without_crashing_worker(tmp_path) -> None:
    def fail():
        raise RuntimeError("provider unavailable")

    service = HourlyUpdateService(
        fail,
        status_path=tmp_path / "status.json",
        now=lambda: datetime(2026, 7, 15, 19, 1, tzinfo=HANOI),
    )
    assert service.run_once("test") is False
    status = service.status()
    assert status["running"] is False
    assert status["consecutive_failures"] == 1
    assert "provider unavailable" in status["last_error"]
