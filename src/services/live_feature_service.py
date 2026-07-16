"""Build leakage-safe inference features from the latest observed time segment."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from src.preprocessing.feature_engineering import BASE_FEATURE_COLUMNS, build_features
from src.services.monitoring_repository import DataSourceUnavailableError


MINIMUM_HISTORY_HOURS = 168


def _latest_contiguous_segment(
    observations: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    minimum_history_hours: int = MINIMUM_HISTORY_HOURS,
) -> pd.DataFrame:
    """Return the latest hourly segment and reject gaps that invalidate lag features."""
    if observations.empty:
        raise DataSourceUnavailableError("No observations are available for live ML inference.")
    work = observations.copy()
    work[timestamp_column] = pd.to_datetime(
        work[timestamp_column], errors="coerce", utc=True
    )
    if work[timestamp_column].isna().any():
        raise DataSourceUnavailableError("Live observations contain invalid timestamps.")
    work = (
        work.sort_values(timestamp_column, kind="stable")
        .drop_duplicates(timestamp_column, keep="last")
        .reset_index(drop=True)
    )
    gaps = work[timestamp_column].diff().ne(pd.Timedelta(hours=1))
    gaps.iloc[0] = False
    if gaps.any():
        work = work.iloc[int(gaps[gaps].index[-1]) :].reset_index(drop=True)
    required_rows = minimum_history_hours + 1
    if len(work) < required_rows:
        latest = work[timestamp_column].iloc[-1].isoformat()
        raise DataSourceUnavailableError(
            "Live ML forecast needs at least "
            f"{required_rows} consecutive hourly observations ending at {latest}; "
            f"only {len(work)} are available. Run the observation collector with "
            f"past_hours >= {required_rows}."
        )
    return work


def build_latest_feature_row(
    observations: pd.DataFrame,
    *,
    required_feature_columns: Iterable[str] | None = None,
    timezone_name: str = "Asia/Ho_Chi_Minh",
) -> dict[str, Any]:
    """Create the newest model row using observations only, never provider forecasts."""
    work = _latest_contiguous_segment(observations)
    missing_columns = sorted(set(BASE_FEATURE_COLUMNS) - set(work.columns))
    if missing_columns:
        raise DataSourceUnavailableError(
            f"Live observations are missing feature inputs: {missing_columns}"
        )

    # Live provider rows do not pass through the offline cleaning report. Preserve
    # the training contract with deterministic, auditable defaults.
    work["data_quality_score"] = pd.to_numeric(
        work["data_quality_score"], errors="coerce"
    ).fillna(1.0)
    work["is_possible_outlier"] = work["is_possible_outlier"].fillna(False)
    if work["rain"].isna().any():
        work["rain"] = pd.to_numeric(work["rain"], errors="coerce").fillna(
            pd.to_numeric(work["precipitation"], errors="coerce")
        )

    current_missing = [
        column
        for column in BASE_FEATURE_COLUMNS
        if pd.isna(work[column].iloc[-1])
    ]
    if current_missing:
        raise DataSourceUnavailableError(
            f"Latest observation is missing model inputs: {current_missing}"
        )

    result = build_features(
        work,
        timezone_name=timezone_name,
        drop_incomplete_rows=False,
    )
    latest = result.data.iloc[-1]
    required = list(required_feature_columns or result.feature_columns)
    missing_features = sorted(set(required) - set(latest.index))
    if missing_features:
        raise DataSourceUnavailableError(
            f"Live feature builder does not provide model features: {missing_features}"
        )
    null_features = [column for column in required if pd.isna(latest[column])]
    if null_features:
        raise DataSourceUnavailableError(
            f"Latest live feature row is incomplete: {null_features}"
        )
    return latest.to_dict()
