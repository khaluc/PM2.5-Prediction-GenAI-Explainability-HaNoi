"""Collect optional TomTom traffic features for Hanoi sampling points."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from src.collection.common import CollectionError, HanoiLocation, collected_at_utc, get_json

TOMTOM_FLOW_URL = (
    "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/{zoom}/json"
)


def collect_traffic_data(
    locations: Iterable[HanoiLocation],
    *,
    api_key: str | None = None,
    zoom: int = 10,
    timezone_name: str = "Asia/Ho_Chi_Minh",
    timeout_seconds: float = 30,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Collect current road-flow indicators and derive a congestion ratio."""
    key = api_key or os.getenv("TOMTOM_API_KEY") or os.getenv("TRAFFIC_API_KEY")
    if not key:
        raise CollectionError("TOMTOM_API_KEY is required for traffic collection.")

    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout_seconds)
    collected_at = collected_at_utc()
    timestamp = datetime.now(ZoneInfo(timezone_name)).replace(second=0, microsecond=0).isoformat()
    rows: list[dict] = []
    errors: list[str] = []
    try:
        for location in locations:
            params = {
                "key": key,
                "point": f"{location.latitude},{location.longitude}",
                "unit": "KMPH",
            }
            try:
                payload = get_json(
                    http_client,
                    TOMTOM_FLOW_URL.format(zoom=zoom),
                    params,
                )
                flow = payload.get("flowSegmentData") or {}
                current_speed = flow.get("currentSpeed")
                free_flow_speed = flow.get("freeFlowSpeed")
                congestion = None
                if current_speed is not None and free_flow_speed:
                    congestion = max(0.0, min(1.0, 1 - current_speed / free_flow_speed))
                rows.append(
                    {
                        "timestamp": timestamp,
                        "station_id": location.station_id,
                        "location_name": location.name,
                        "latitude": location.latitude,
                        "longitude": location.longitude,
                        "current_speed": current_speed,
                        "free_flow_speed": free_flow_speed,
                        "current_travel_time": flow.get("currentTravelTime"),
                        "free_flow_travel_time": flow.get("freeFlowTravelTime"),
                        "traffic_congestion": congestion,
                        "confidence": flow.get("confidence"),
                        "road_closure": flow.get("roadClosure"),
                        "road_class": flow.get("frc"),
                        "source": "tomtom_flow_segment",
                        "collected_at": collected_at,
                    }
                )
            except CollectionError as exc:
                errors.append(f"{location.station_id}: {exc}")
    finally:
        if owns_client:
            http_client.close()

    if not rows:
        raise CollectionError("No traffic data collected. " + "; ".join(errors))
    return pd.DataFrame(rows)
