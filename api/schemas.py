"""Validated API request and response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnvironmentalObservation(StrictModel):
    station_id: str
    timestamp: datetime
    pm25: float = Field(ge=0)
    pm10: float | None = Field(default=None, ge=0)
    temperature: float | None = None
    humidity: float | None = Field(default=None, ge=0, le=100)
    wind_speed: float | None = Field(default=None, ge=0)


class StationSummary(StrictModel):
    station_id: str
    name: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    first_timestamp: datetime
    latest_timestamp: datetime
    observation_count: int
    latest_pm25: float | None = None
    data_source: str | None = None


class FeatureInputRequest(StrictModel):
    station_id: str | None = Field(default=None, description="Build features from the latest continuous observation window")
    features: dict[str, Any] | None = Field(default=None, description="Explicit feature-engineered row")

    @model_validator(mode="after")
    def require_input(self) -> "FeatureInputRequest":
        if not self.station_id and not self.features:
            raise ValueError("Provide station_id or features")
        return self


class PredictRequest(FeatureInputRequest):
    create_alert: bool = False


class AnomalyRequest(FeatureInputRequest):
    create_alert: bool = False


class ForecastExplanationRequest(StrictModel):
    station_id: str = Field(min_length=2, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    horizon_hours: Literal[1, 3, 6] = 3
    use_llm: bool = True


class AlertEvaluationRequest(StrictModel):
    environment_result: dict[str, Any]


class AlertAcknowledgeRequest(StrictModel):
    acknowledged_by: str = Field(min_length=2, max_length=100)


class ReportGenerateRequest(StrictModel):
    station_id: str
    start: datetime | None = None
    end: datetime | None = None
    format: Literal["json", "markdown", "pdf"] = "json"
    persist: bool = True

    @model_validator(mode="after")
    def validate_period(self) -> "ReportGenerateRequest":
        if self.start and self.end:
            start = self.start if self.start.tzinfo else self.start.replace(tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
            end = self.end if self.end.tzinfo else self.end.replace(tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
            if start > end:
                raise ValueError("start must be earlier than or equal to end")
        return self
