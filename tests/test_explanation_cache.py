"""Persistent cache behavior for hourly GenAI forecast explanations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import func, select

from src.database.connection import get_engine, get_session_factory
from src.database.models import Base, GenAIExplanation, Station
from src.genai.explanation_cache import ForecastExplanationCache


ISSUED_AT = datetime(2026, 7, 23, 10, 0, tzinfo=timezone.utc)


def _database_url(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'explanations.db').as_posix()}"


def _prepare_database(tmp_path: Path) -> str:
    database_url = _database_url(tmp_path)
    Base.metadata.create_all(get_engine(database_url))
    with get_session_factory(database_url)() as session:
        session.add(Station(station_id="HN_BA_DINH", name="Ba Đình"))
        session.commit()
    return database_url


def _result(mode: str, marker: str) -> dict:
    return {
        "station_id": "HN_BA_DINH",
        "forecast": {"horizon_hours": 1, "predicted_pm25": 36.3},
        "explanation": {"headline": marker},
        "generation": {
            "mode": mode,
            "provider_model": (
                "dashscope:deepseek-v4-flash"
                if mode == "dashscope"
                else "deterministic"
            ),
            "fallback_reason": (
                None if mode == "dashscope" else "provider_or_guardrail_failure"
            ),
        },
    }


def test_successful_explanation_is_reused_from_database(tmp_path: Path) -> None:
    database_url = _prepare_database(tmp_path)
    calls = 0

    def generate() -> dict:
        nonlocal calls
        calls += 1
        return _result("dashscope", f"deepseek-{calls}")

    first_cache = ForecastExplanationCache(
        database_url,
        database_enabled=True,
        success_ttl_seconds=7_200,
    )
    first, first_meta = first_cache.resolve(
        station_id="HN_BA_DINH",
        forecast_issued_at=ISSUED_AT,
        horizon_hours=1,
        use_llm=True,
        generator=generate,
        now=ISSUED_AT,
    )
    second, second_meta = first_cache.resolve(
        station_id="HN_BA_DINH",
        forecast_issued_at=ISSUED_AT,
        horizon_hours=1,
        use_llm=True,
        generator=generate,
        now=ISSUED_AT + timedelta(minutes=5),
    )
    restarted_cache = ForecastExplanationCache(
        database_url,
        database_enabled=True,
        success_ttl_seconds=7_200,
    )
    third, third_meta = restarted_cache.resolve(
        station_id="HN_BA_DINH",
        forecast_issued_at=ISSUED_AT,
        horizon_hours=1,
        use_llm=True,
        generator=generate,
        now=ISSUED_AT + timedelta(minutes=10),
    )

    assert calls == 1
    assert first == second == third
    assert first_meta["status"] == "miss"
    assert first_meta["backend"] == "sqlite"
    assert second_meta["status"] == "hit"
    assert third_meta["status"] == "hit"
    assert third_meta["backend"] == "sqlite"
    with get_session_factory(database_url)() as session:
        assert session.scalar(
            select(func.count()).select_from(GenAIExplanation)
        ) == 1


def test_fallback_expires_quickly_then_deepseek_replaces_it(tmp_path: Path) -> None:
    database_url = _prepare_database(tmp_path)
    calls = 0

    def generate() -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _result("deterministic_fallback", "fallback")
        return _result("dashscope", "deepseek")

    cache = ForecastExplanationCache(
        database_url,
        database_enabled=True,
        success_ttl_seconds=7_200,
        fallback_ttl_seconds=300,
    )
    first, first_meta = cache.resolve(
        station_id="HN_BA_DINH",
        forecast_issued_at=ISSUED_AT,
        horizon_hours=1,
        use_llm=True,
        generator=generate,
        now=ISSUED_AT,
    )
    cached_fallback, cached_meta = cache.resolve(
        station_id="HN_BA_DINH",
        forecast_issued_at=ISSUED_AT,
        horizon_hours=1,
        use_llm=True,
        generator=generate,
        now=ISSUED_AT + timedelta(minutes=4),
    )
    refreshed, refreshed_meta = cache.resolve(
        station_id="HN_BA_DINH",
        forecast_issued_at=ISSUED_AT,
        horizon_hours=1,
        use_llm=True,
        generator=generate,
        now=ISSUED_AT + timedelta(minutes=6),
    )
    stable, stable_meta = cache.resolve(
        station_id="HN_BA_DINH",
        forecast_issued_at=ISSUED_AT,
        horizon_hours=1,
        use_llm=True,
        generator=generate,
        now=ISSUED_AT + timedelta(minutes=7),
    )

    assert calls == 2
    assert first["explanation"]["headline"] == "fallback"
    assert cached_fallback == first
    assert first_meta["retry_after"] is not None
    assert cached_meta["status"] == "hit"
    assert refreshed["explanation"]["headline"] == "deepseek"
    assert refreshed_meta["status"] == "refresh"
    assert refreshed_meta["retry_after"] is None
    assert stable == refreshed
    assert stable_meta["status"] == "hit"


def test_cache_is_scoped_to_forecast_timestamp_and_bypasses_non_llm() -> None:
    cache = ForecastExplanationCache(database_enabled=False)
    calls = 0

    def generate() -> dict:
        nonlocal calls
        calls += 1
        return _result("dashscope", f"result-{calls}")

    for issued_at in (ISSUED_AT, ISSUED_AT + timedelta(hours=1)):
        cache.resolve(
            station_id="HN_BA_DINH",
            forecast_issued_at=issued_at,
            horizon_hours=1,
            use_llm=True,
            generator=generate,
            now=issued_at,
        )
    _, bypass_meta = cache.resolve(
        station_id="HN_BA_DINH",
        forecast_issued_at=ISSUED_AT,
        horizon_hours=1,
        use_llm=False,
        generator=generate,
        now=ISSUED_AT,
    )

    assert calls == 3
    assert bypass_meta["status"] == "bypass"
    assert bypass_meta["reason"] == "llm_disabled"
