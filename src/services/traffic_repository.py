"""Read and time-align live TomTom flow data for GenAI grounding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.services.monitoring_repository import (
    CachedCsvRepository,
    DataSourceUnavailableError,
    _python_value,
    normalise_timestamp,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAFFIC_PATH = PROJECT_ROOT / "data" / "raw" / "traffic.csv"


class TrafficRepository(CachedCsvRepository):
    """Return the TomTom record nearest to an air-quality observation."""

    def __init__(
        self,
        path: str | Path = DEFAULT_TRAFFIC_PATH,
        *,
        timezone_name: str = "Asia/Ho_Chi_Minh",
        max_age_minutes: float = 120.0,
    ) -> None:
        super().__init__(path, timezone_name=timezone_name)
        self.max_age_minutes = float(max_age_minutes)

    def latest_near(
        self,
        station_id: str,
        reference_timestamp: Any | None,
    ) -> dict[str, Any]:
        """Return a bounded-age record without making traffic mandatory for inference."""
        try:
            frame = self._load()
        except DataSourceUnavailableError as error:
            return {
                "available": False,
                "reason": "traffic_data_unavailable",
                "detail": str(error),
                "source": "tomtom_flow_segment",
            }
        required = {"timestamp", "station_id", "current_speed", "free_flow_speed"}
        missing = sorted(required - set(frame.columns))
        if missing:
            return {
                "available": False,
                "reason": "traffic_columns_missing",
                "missing_columns": missing,
                "source": "tomtom_flow_segment",
            }
        selected = frame[frame["station_id"] == str(station_id)].copy()
        if selected.empty:
            return {
                "available": False,
                "reason": "traffic_station_not_found",
                "source": "tomtom_flow_segment",
            }

        reference = (
            normalise_timestamp(reference_timestamp, self.timezone_name)
            if reference_timestamp
            else pd.Timestamp.now(tz=self.timezone_name)
        )
        selected["_offset_minutes"] = (
            selected["timestamp"] - reference
        ).dt.total_seconds() / 60.0
        selected["_absolute_offset_minutes"] = selected["_offset_minutes"].abs()
        row = selected.sort_values(
            ["_absolute_offset_minutes", "timestamp"],
            ascending=[True, False],
            kind="stable",
        ).iloc[0]
        age_minutes = float(row["_absolute_offset_minutes"])
        if age_minutes > self.max_age_minutes:
            return {
                "available": False,
                "reason": "traffic_data_stale",
                "observed_at": _python_value(row["timestamp"]),
                "reference_timestamp": reference.isoformat(),
                "age_minutes": round(age_minutes, 1),
                "max_age_minutes": self.max_age_minutes,
                "source": _python_value(row.get("source")) or "tomtom_flow_segment",
            }

        def numeric(column: str) -> float | None:
            value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            return None if pd.isna(value) else float(value)

        current_speed = numeric("current_speed")
        free_flow_speed = numeric("free_flow_speed")
        congestion = numeric("traffic_congestion")
        if congestion is None and current_speed is not None and free_flow_speed:
            congestion = max(0.0, min(1.0, 1 - current_speed / free_flow_speed))
        confidence = numeric("confidence")
        road_closure_value = row.get("road_closure", False)
        road_closure = str(road_closure_value).strip().lower() in {"true", "1", "yes"}
        return {
            "available": True,
            "station_id": str(station_id),
            "observed_at": _python_value(row["timestamp"]),
            "reference_timestamp": reference.isoformat(),
            "time_offset_minutes": round(float(row["_offset_minutes"]), 1),
            "age_minutes": round(age_minutes, 1),
            "current_speed_kmh": current_speed,
            "free_flow_speed_kmh": free_flow_speed,
            "current_travel_time_seconds": numeric("current_travel_time"),
            "free_flow_travel_time_seconds": numeric("free_flow_travel_time"),
            "congestion_ratio": round(congestion, 4) if congestion is not None else None,
            "congestion_percent": round(congestion * 100, 1) if congestion is not None else None,
            "confidence": confidence,
            "road_closure": road_closure,
            "road_class": _python_value(row.get("road_class")),
            "source": _python_value(row.get("source")) or "tomtom_flow_segment",
            "spatial_scope": "nearest_road_segment_to_sampling_point",
            "causal_claim_allowed": False,
        }
