"""PostgreSQL alert store with the existing JSON file as a safety fallback."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from src.alerts.alert_service import AlertNotFoundError, FileAlertStore
from src.database.connection import get_database_url, get_session_factory
from src.database.models import Alert, SystemState
from src.database.writer import DatabaseWriter


class DatabaseAlertStore:
    def __init__(
        self,
        fallback: FileAlertStore,
        database_url: str | None = None,
    ) -> None:
        self.fallback = fallback
        self.database_url = database_url or get_database_url()
        self.writer = DatabaseWriter(self.database_url)

    def _session(self):
        return get_session_factory(self.database_url)()

    def _ready(self) -> bool:
        with self._session() as session:
            value = session.scalar(
                select(SystemState.value).where(SystemState.key == "initial_import")
            )
        return bool(isinstance(value, dict) and value.get("completed"))

    def _fallback_on_error(self, database: Callable[[], Any], fallback: Callable[[], Any]):
        try:
            if self._ready():
                return database()
        except SQLAlchemyError:
            pass
        return fallback()

    @staticmethod
    def _payload(record: Alert) -> dict[str, Any]:
        payload = dict(record.payload or {})
        payload.update(
            {
                "alert_id": record.alert_id,
                "station_id": record.station_id,
                "type": record.alert_type,
                "severity": record.severity,
                "status": record.status,
                "created_at_utc": record.created_at.isoformat(),
            }
        )
        if record.acknowledged_at:
            payload["acknowledged_at_utc"] = record.acknowledged_at.isoformat()
        if record.acknowledged_by:
            payload["acknowledged_by"] = record.acknowledged_by
        return payload

    def upsert(self, alert: dict[str, Any]) -> dict[str, Any]:
        def database() -> dict[str, Any]:
            with self._session() as session:
                existing = session.get(Alert, str(alert["alert_id"]))
                if existing is not None and existing.status == "active":
                    return self._payload(existing)
            self.writer.upsert_alert(alert)
            return alert

        return self._fallback_on_error(database, lambda: self.fallback.upsert(alert))

    def list(
        self,
        *,
        station_id: str | None = None,
        status: str | None = None,
        severity: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        def database() -> dict[str, Any]:
            filters = []
            if station_id:
                filters.append(Alert.station_id == station_id)
            if status:
                filters.append(Alert.status == status)
            if severity:
                filters.append(Alert.severity == severity)
            with self._session() as session:
                total = int(
                    session.scalar(select(func.count()).select_from(Alert).where(*filters)) or 0
                )
                rows = session.scalars(
                    select(Alert)
                    .where(*filters)
                    .order_by(Alert.created_at.desc())
                    .offset(offset)
                    .limit(limit)
                ).all()
                return {
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "items": [self._payload(row) for row in rows],
                }

        return self._fallback_on_error(
            database,
            lambda: self.fallback.list(
                station_id=station_id,
                status=status,
                severity=severity,
                limit=limit,
                offset=offset,
            ),
        )

    def acknowledge(self, alert_id: str, *, acknowledged_by: str) -> dict[str, Any]:
        def database() -> dict[str, Any]:
            with self._session() as session:
                record = session.get(Alert, alert_id)
                if record is None:
                    raise AlertNotFoundError(alert_id)
                record.status = "acknowledged"
                record.acknowledged_by = acknowledged_by
                record.acknowledged_at = datetime.now(timezone.utc)
                payload = dict(record.payload or {})
                payload.update(
                    {
                        "status": record.status,
                        "acknowledged_by": acknowledged_by,
                        "acknowledged_at_utc": record.acknowledged_at.isoformat(),
                    }
                )
                record.payload = payload
                session.commit()
                return self._payload(record)

        return self._fallback_on_error(
            database,
            lambda: self.fallback.acknowledge(alert_id, acknowledged_by=acknowledged_by),
        )
