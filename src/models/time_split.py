"""Chronological train/validation/test splitting with purged boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd


@dataclass
class TimeSplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    report: dict[str, Any]


def _local_boundary(value: str | date | datetime, timezone_name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone_name)
    return timestamp.tz_convert(timezone_name)


def _summary(frame: pd.DataFrame, timestamp_column: str, group_column: str) -> dict:
    if frame.empty:
        return {"rows": 0, "stations": 0, "start": None, "end": None}
    return {
        "rows": len(frame),
        "stations": int(frame[group_column].nunique()),
        "start": frame[timestamp_column].min().isoformat(),
        "end": frame[timestamp_column].max().isoformat(),
    }


def split_time_series(
    data: pd.DataFrame,
    *,
    validation_start: str | date | datetime,
    test_start: str | date | datetime,
    max_target_horizon_hours: int,
    timestamp_column: str = "timestamp",
    group_column: str = "station_id",
    timezone_name: str = "Asia/Ho_Chi_Minh",
) -> TimeSplitResult:
    """Split chronologically and purge rows whose labels cross a boundary.

    For a maximum target horizon H, a row at time t belongs to the earlier set
    only when t + H is strictly before the next set's start timestamp.
    """
    required = {timestamp_column, group_column}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing split columns: {', '.join(missing)}")
    if max_target_horizon_hours < 0:
        raise ValueError("max_target_horizon_hours must be non-negative")

    frame = data.copy()
    frame[timestamp_column] = pd.to_datetime(
        frame[timestamp_column], errors="coerce", utc=True
    ).dt.tz_convert(timezone_name)
    invalid_timestamps = int(frame[timestamp_column].isna().sum())
    if invalid_timestamps:
        raise ValueError(f"Input contains {invalid_timestamps} invalid timestamps")
    duplicate_keys = int(
        frame.duplicated(subset=[timestamp_column, group_column]).sum()
    )
    if duplicate_keys:
        raise ValueError(f"Input contains {duplicate_keys} duplicate time/station keys")

    validation_boundary = _local_boundary(validation_start, timezone_name)
    test_boundary = _local_boundary(test_start, timezone_name)
    if validation_boundary >= test_boundary:
        raise ValueError("validation_start must be earlier than test_start")

    frame = frame.sort_values([timestamp_column, group_column]).reset_index(drop=True)
    horizon = pd.Timedelta(hours=max_target_horizon_hours)
    label_time = frame[timestamp_column] + horizon

    before_validation = frame[timestamp_column] < validation_boundary
    train_purge = before_validation & label_time.ge(validation_boundary)
    train_mask = before_validation & ~train_purge

    in_validation = frame[timestamp_column].ge(validation_boundary) & frame[
        timestamp_column
    ].lt(test_boundary)
    validation_purge = in_validation & label_time.ge(test_boundary)
    validation_mask = in_validation & ~validation_purge

    test_mask = frame[timestamp_column].ge(test_boundary)

    train = frame.loc[train_mask].copy()
    validation = frame.loc[validation_mask].copy()
    test = frame.loc[test_mask].copy()
    for part in (train, validation, test):
        part[timestamp_column] = part[timestamp_column].map(lambda value: value.isoformat())

    assigned = train_mask | validation_mask | test_mask
    report = {
        "input_rows": len(frame),
        "validation_start": validation_boundary.isoformat(),
        "test_start": test_boundary.isoformat(),
        "max_target_horizon_hours": max_target_horizon_hours,
        "train_boundary_rows_purged": int(train_purge.sum()),
        "validation_boundary_rows_purged": int(validation_purge.sum()),
        "unassigned_rows": int((~assigned).sum()),
        "train": _summary(
            frame.loc[train_mask], timestamp_column, group_column
        ),
        "validation": _summary(
            frame.loc[validation_mask], timestamp_column, group_column
        ),
        "test": _summary(frame.loc[test_mask], timestamp_column, group_column),
        "chronology_checks": {
            "train_labels_before_validation": bool(
                train.empty
                or (
                    pd.to_datetime(train[timestamp_column], utc=True).max()
                    + horizon
                    < validation_boundary.tz_convert("UTC")
                )
            ),
            "validation_labels_before_test": bool(
                validation.empty
                or (
                    pd.to_datetime(validation[timestamp_column], utc=True).max()
                    + horizon
                    < test_boundary.tz_convert("UTC")
                )
            ),
            "test_starts_on_boundary": bool(
                test.empty
                or pd.to_datetime(test[timestamp_column], utc=True).min()
                >= test_boundary.tz_convert("UTC")
            ),
        },
    }
    return TimeSplitResult(
        train=train,
        validation=validation,
        test=test,
        report=report,
    )
