"""Persistent, bounded cache for controlled hourly GenAI explanations."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import threading
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import SQLAlchemyError

from src.database.connection import get_database_url, session_scope
from src.database.models import GenAIExplanation


CACHEABLE_HORIZON_HOURS = 1


def _utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class ForecastExplanationCache:
    """Reuse one explanation for one station/forecast hour across page loads."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        database_enabled: bool = False,
        success_ttl_seconds: int = 7_200,
        fallback_ttl_seconds: int = 300,
        cache_version: str = "hourly-explanation-v1",
    ) -> None:
        self.database_url = database_url or get_database_url()
        self.database_enabled = bool(database_enabled)
        self.success_ttl_seconds = max(300, int(success_ttl_seconds))
        self.fallback_ttl_seconds = max(30, int(fallback_ttl_seconds))
        self.cache_version = str(cache_version).strip() or "hourly-explanation-v1"
        self._memory: dict[tuple[str, datetime, int, str], dict[str, Any]] = {}
        self._key_locks: dict[tuple[str, datetime, int, str], threading.Lock] = {}
        self._guard = threading.RLock()
        self._last_database_error: str | None = None

    def _key(
        self,
        station_id: str,
        forecast_issued_at: datetime,
        horizon_hours: int,
    ) -> tuple[str, datetime, int, str]:
        return (
            str(station_id),
            forecast_issued_at,
            int(horizon_hours),
            self.cache_version,
        )

    def _lock_for(
        self,
        key: tuple[str, datetime, int, str],
    ) -> threading.Lock:
        with self._guard:
            return self._key_locks.setdefault(key, threading.Lock())

    @staticmethod
    def _entry_from_row(
        row: GenAIExplanation,
        backend: str,
    ) -> dict[str, Any]:
        return {
            "result": copy.deepcopy(row.result),
            "expires_at": _utc_datetime(row.expires_at),
            "generation_mode": row.generation_mode,
            "provider_model": row.provider_model,
            "fallback_reason": row.fallback_reason,
            "backend": backend,
        }

    def _database_get(
        self,
        key: tuple[str, datetime, int, str],
    ) -> dict[str, Any] | None:
        station_id, issued_at, horizon_hours, cache_version = key
        try:
            with session_scope(self.database_url) as session:
                backend = (
                    session.bind.dialect.name
                    if session.bind is not None
                    else "database"
                )
                row = session.scalar(
                    select(GenAIExplanation).where(
                        GenAIExplanation.station_id == station_id,
                        GenAIExplanation.forecast_issued_at == issued_at,
                        GenAIExplanation.horizon_hours == horizon_hours,
                        GenAIExplanation.cache_version == cache_version,
                    )
                )
                result = (
                    self._entry_from_row(row, backend)
                    if row is not None
                    else None
                )
            self._last_database_error = None
            return result
        except SQLAlchemyError as error:
            self._last_database_error = type(error).__name__
            return None

    def _get(
        self,
        key: tuple[str, datetime, int, str],
    ) -> dict[str, Any] | None:
        with self._guard:
            cached = self._memory.get(key)
            if cached is not None:
                return copy.deepcopy(cached)
        if not self.database_enabled:
            return None
        cached = self._database_get(key)
        if cached is not None:
            with self._guard:
                self._memory[key] = copy.deepcopy(cached)
        return cached

    def _database_save(
        self,
        key: tuple[str, datetime, int, str],
        entry: dict[str, Any],
        now: datetime,
    ) -> str | None:
        station_id, issued_at, horizon_hours, cache_version = key
        values = {
            "station_id": station_id,
            "forecast_issued_at": issued_at,
            "horizon_hours": horizon_hours,
            "cache_version": cache_version,
            "generation_mode": entry["generation_mode"],
            "provider_model": entry["provider_model"],
            "fallback_reason": entry["fallback_reason"],
            "result": copy.deepcopy(entry["result"]),
            "expires_at": entry["expires_at"],
            "updated_at": now,
        }
        unique_columns = [
            "station_id",
            "forecast_issued_at",
            "horizon_hours",
            "cache_version",
        ]
        try:
            with session_scope(self.database_url) as session:
                table = GenAIExplanation.__table__
                dialect = session.bind.dialect.name if session.bind is not None else ""
                if dialect == "postgresql":
                    statement = postgresql_insert(table).values(**values)
                elif dialect == "sqlite":
                    statement = sqlite_insert(table).values(**values)
                else:
                    existing = session.scalar(
                        select(GenAIExplanation).where(
                            GenAIExplanation.station_id == station_id,
                            GenAIExplanation.forecast_issued_at == issued_at,
                            GenAIExplanation.horizon_hours == horizon_hours,
                            GenAIExplanation.cache_version == cache_version,
                        )
                    )
                    if existing is None:
                        session.add(GenAIExplanation(**values))
                    else:
                        for field, value in values.items():
                            setattr(existing, field, value)
                    self._last_database_error = None
                    return dialect or "database"
                statement = statement.on_conflict_do_update(
                    index_elements=[table.c[column] for column in unique_columns],
                    set_={
                        "generation_mode": statement.excluded.generation_mode,
                        "provider_model": statement.excluded.provider_model,
                        "fallback_reason": statement.excluded.fallback_reason,
                        "result": statement.excluded.result,
                        "expires_at": statement.excluded.expires_at,
                        "updated_at": statement.excluded.updated_at,
                    },
                )
                session.execute(statement)
            self._last_database_error = None
            return dialect or "database"
        except SQLAlchemyError as error:
            self._last_database_error = type(error).__name__
            return None

    def _save(
        self,
        key: tuple[str, datetime, int, str],
        entry: dict[str, Any],
        now: datetime,
    ) -> str:
        backend = "memory"
        if self.database_enabled:
            database_backend = self._database_save(key, entry, now)
            if database_backend:
                backend = database_backend
        stored = {**entry, "backend": backend}
        with self._guard:
            self._memory[key] = copy.deepcopy(stored)
        return backend

    def resolve(
        self,
        *,
        station_id: str,
        forecast_issued_at: Any,
        horizon_hours: int,
        use_llm: bool,
        generator: Callable[[], dict[str, Any]],
        now: datetime | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        issued_at = _utc_datetime(forecast_issued_at)
        if (
            not use_llm
            or int(horizon_hours) != CACHEABLE_HORIZON_HOURS
            or issued_at is None
        ):
            return generator(), {
                "status": "bypass",
                "backend": "none",
                "cache_version": self.cache_version,
                "reason": (
                    "llm_disabled"
                    if not use_llm
                    else "unsupported_horizon_or_timestamp"
                ),
            }

        current_time = _utc_datetime(now) or datetime.now(timezone.utc)
        key = self._key(station_id, issued_at, horizon_hours)
        with self._lock_for(key):
            cached = self._get(key)
            expires_at = _utc_datetime(cached.get("expires_at")) if cached else None
            if cached is not None and expires_at is not None and expires_at > current_time:
                return copy.deepcopy(cached["result"]), {
                    "status": "hit",
                    "backend": cached.get("backend") or "memory",
                    "cache_version": self.cache_version,
                    "forecast_issued_at": issued_at.isoformat(),
                    "expires_at": expires_at.isoformat(),
                    "generation_mode": cached.get("generation_mode"),
                }

            result = generator()
            generation = result.get("generation") if isinstance(result, dict) else {}
            generation = generation if isinstance(generation, dict) else {}
            generation_mode = str(
                generation.get("mode") or "deterministic_fallback"
            )
            successful = generation_mode == "dashscope"
            ttl_seconds = (
                self.success_ttl_seconds
                if successful
                else self.fallback_ttl_seconds
            )
            expires_at = current_time + timedelta(seconds=ttl_seconds)
            entry = {
                "result": copy.deepcopy(result),
                "expires_at": expires_at,
                "generation_mode": generation_mode,
                "provider_model": generation.get("provider_model"),
                "fallback_reason": generation.get("fallback_reason"),
            }
            backend = self._save(key, entry, current_time)
            return result, {
                "status": "refresh" if cached is not None else "miss",
                "backend": backend,
                "cache_version": self.cache_version,
                "forecast_issued_at": issued_at.isoformat(),
                "expires_at": expires_at.isoformat(),
                "generation_mode": generation_mode,
                "retry_after": (
                    expires_at.isoformat() if not successful else None
                ),
            }


__all__ = ["CACHEABLE_HORIZON_HOURS", "ForecastExplanationCache"]
