"""Data-quality pipeline for hourly environmental observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


CANONICAL_UNITS = {
    "pm25": "ug/m3",
    "pm10": "ug/m3",
    "co": "ug/m3",
    "no2": "ug/m3",
    "so2": "ug/m3",
    "o3": "ug/m3",
    "temperature": "celsius",
    "humidity": "percent",
    "wind_speed": "km/h",
    "wind_direction": "degree",
    "precipitation": "mm",
    "rain": "mm",
    "surface_pressure": "hPa",
    "cloud_cover": "percent",
    "traffic_congestion": "ratio",
}

UNIT_ALIASES = {
    "µg/m³": "ug/m3",
    "μg/m³": "ug/m3",
    "ug/m³": "ug/m3",
    "mg/m³": "mg/m3",
    "°c": "celsius",
    "c": "celsius",
    "°f": "fahrenheit",
    "f": "fahrenheit",
    "kph": "km/h",
    "kmh": "km/h",
    "m/s": "m/s",
    "pa": "Pa",
    "hpa": "hPa",
    "%": "percent",
}


@dataclass
class CleaningResult:
    data: pd.DataFrame
    report: dict[str, Any]


def _append_flag(flags: pd.Series, mask: pd.Series, label: str) -> pd.Series:
    result = flags.copy()
    mask = mask.fillna(False)
    if not bool(mask.any()):
        return result
    current = result.loc[mask]
    result.loc[mask] = np.where(current.eq(""), label, current + ";" + label)
    return result


def _normalize_unit_name(value: Any) -> str:
    text = str(value).strip()
    return UNIT_ALIASES.get(text.lower(), UNIT_ALIASES.get(text, text))


def _convert_values(values: pd.Series, source_unit: str, target_unit: str) -> pd.Series:
    source = _normalize_unit_name(source_unit)
    target = _normalize_unit_name(target_unit)
    if source == target:
        return values
    if source == "mg/m3" and target == "ug/m3":
        return values * 1000
    if source == "ug/m3" and target == "mg/m3":
        return values / 1000
    if source == "fahrenheit" and target == "celsius":
        return (values - 32) * 5 / 9
    if source == "kelvin" and target == "celsius":
        return values - 273.15
    if source == "m/s" and target == "km/h":
        return values * 3.6
    if source == "km/h" and target == "m/s":
        return values / 3.6
    if source == "Pa" and target == "hPa":
        return values / 100
    raise ValueError(f"Unsupported unit conversion: {source_unit} -> {target_unit}")


def normalize_units(
    frame: pd.DataFrame,
    value_columns: list[str],
    declared_units: dict[str, str] | None,
    flags: pd.Series,
) -> tuple[dict[str, str], pd.Series]:
    """Convert supported file-level or row-level units to canonical units."""
    output_units: dict[str, str] = {}
    declared_units = declared_units or {}
    for column in value_columns:
        target = CANONICAL_UNITS.get(column, declared_units.get(column, "unknown"))
        output_units[column] = target
        unit_column = f"{column}_unit"
        if unit_column in frame.columns:
            for source_unit in frame[unit_column].dropna().unique():
                mask = frame[unit_column].eq(source_unit)
                try:
                    frame.loc[mask, column] = _convert_values(
                        frame.loc[mask, column], str(source_unit), target
                    )
                    if _normalize_unit_name(source_unit) != target:
                        flags = _append_flag(flags, mask, f"unit_converted:{column}")
                except ValueError:
                    frame.loc[mask, column] = np.nan
                    flags = _append_flag(flags, mask, f"unsupported_unit:{column}")
            frame[unit_column] = target
            continue

        source = declared_units.get(column, target)
        try:
            frame[column] = _convert_values(frame[column], source, target)
            if _normalize_unit_name(source) != target:
                flags = _append_flag(
                    flags,
                    pd.Series(True, index=frame.index),
                    f"unit_converted:{column}",
                )
        except ValueError:
            frame[column] = np.nan
            flags = _append_flag(
                flags,
                pd.Series(True, index=frame.index),
                f"unsupported_unit:{column}",
            )
    return output_units, flags


def _parse_timestamps(
    values: pd.Series, timezone_name: str
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Parse aware and naive timestamps without interpreting local time as UTC."""
    text = values.astype("string").str.strip()
    aware_mask = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True, na=False)
    result = pd.Series(pd.NaT, index=values.index, dtype=f"datetime64[ns, {timezone_name}]")

    aware = pd.to_datetime(text.loc[aware_mask], errors="coerce", utc=True)
    result.loc[aware_mask] = aware.dt.tz_convert(timezone_name)

    naive = pd.to_datetime(text.loc[~aware_mask], errors="coerce")
    if not naive.empty:
        naive = naive.dt.tz_localize(timezone_name, ambiguous="NaT", nonexistent="NaT")
        result.loc[~aware_mask] = naive

    invalid = result.isna()
    rounded = result.notna() & (
        (result.dt.minute != 0) | (result.dt.second != 0) | (result.dt.microsecond != 0)
    )
    return result.dt.floor("h"), invalid, rounded


def _add_missing_hour_rows(
    frame: pd.DataFrame,
    station_column: str,
    timestamp_column: str,
    frequency: str,
    value_columns: list[str],
) -> tuple[pd.DataFrame, int]:
    parts: list[pd.DataFrame] = []
    added = 0
    metadata_columns = [
        column
        for column in frame.columns
        if column not in {timestamp_column, "quality_flags", *value_columns}
    ]
    for station_id, group in frame.groupby(station_column, sort=False):
        group = group.sort_values(timestamp_column).set_index(timestamp_column)
        full_index = pd.date_range(group.index.min(), group.index.max(), freq=frequency)
        expanded = group.reindex(full_index)
        missing_rows = expanded[station_column].isna()
        added += int(missing_rows.sum())
        expanded[station_column] = station_id
        for column in metadata_columns:
            if column == station_column:
                continue
            expanded[column] = expanded[column].ffill().bfill()
        expanded["quality_flags"] = expanded["quality_flags"].fillna("")
        expanded["quality_flags"] = _append_flag(
            expanded["quality_flags"], missing_rows, "missing_timestamp"
        )
        expanded.index.name = timestamp_column
        parts.append(expanded.reset_index())
    return pd.concat(parts, ignore_index=True), added


def _spike_mask(
    frame: pd.DataFrame,
    column: str,
    station_column: str,
    window: int,
    min_periods: int,
    threshold: float,
    min_absolute_change: float,
) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for _, indices in frame.groupby(station_column, sort=False).groups.items():
        values = frame.loc[indices, column]
        past = values.shift(1)
        baseline = past.rolling(window, min_periods=min_periods).median()
        deviation = (past - baseline).abs()
        mad = deviation.rolling(window, min_periods=min_periods).median()
        robust_z = (values - baseline).abs() / (1.4826 * mad.replace(0, np.nan))
        absolute_change = (values - past).abs()
        mask.loc[indices] = (
            robust_z.gt(threshold)
            & absolute_change.ge(min_absolute_change)
        ).fillna(False)
    return mask


def clean_environmental_data(
    data: pd.DataFrame,
    *,
    value_columns: list[str],
    valid_ranges: dict[str, list[float]],
    declared_units: dict[str, str] | None = None,
    station_column: str = "station_id",
    timestamp_column: str = "timestamp",
    timezone_name: str = "Asia/Ho_Chi_Minh",
    frequency: str = "1h",
    interpolation_limit: int = 3,
    spike_window: int = 24,
    spike_min_periods: int = 12,
    spike_threshold: float = 6.0,
    spike_columns: list[str] | None = None,
    spike_min_absolute_change: dict[str, float] | None = None,
    replace_spikes: bool = False,
) -> CleaningResult:
    """Clean one environmental table and retain auditable quality flags."""
    required = {station_column, timestamp_column, *value_columns}
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    frame = data.copy()
    input_rows = len(frame)
    frame["quality_flags"] = ""
    parsed, invalid_time, rounded_time = _parse_timestamps(
        frame[timestamp_column], timezone_name
    )
    invalid_timestamp_rows = int(invalid_time.sum())
    frame[timestamp_column] = parsed
    frame = frame.loc[~invalid_time].copy()
    rounded_time = rounded_time.loc[frame.index]
    frame["quality_flags"] = _append_flag(
        frame["quality_flags"], rounded_time, "timestamp_rounded"
    )

    for column in value_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    units, frame["quality_flags"] = normalize_units(
        frame,
        value_columns,
        declared_units,
        frame["quality_flags"],
    )

    negative_counts: dict[str, int] = {}
    range_counts: dict[str, int] = {}
    for column in value_columns:
        negative = frame[column].lt(0)
        negative_counts[column] = int(negative.sum())
        frame["quality_flags"] = _append_flag(
            frame["quality_flags"], negative, f"negative:{column}"
        )
        frame.loc[negative, column] = np.nan

        if column in valid_ranges:
            lower, upper = valid_ranges[column]
            outside = frame[column].notna() & ~frame[column].between(lower, upper)
            range_counts[column] = int(outside.sum())
            frame["quality_flags"] = _append_flag(
                frame["quality_flags"], outside, f"out_of_range:{column}"
            )
            frame.loc[outside, column] = np.nan

    sort_columns = [station_column, timestamp_column]
    if "collected_at" in frame.columns:
        sort_columns.append("collected_at")
    frame = frame.sort_values(sort_columns)
    duplicate_mask = frame.duplicated(
        subset=[station_column, timestamp_column], keep="last"
    )
    duplicates_removed = int(duplicate_mask.sum())
    frame = frame.loc[~duplicate_mask].copy()

    frame, missing_rows_added = _add_missing_hour_rows(
        frame,
        station_column,
        timestamp_column,
        frequency,
        value_columns,
    )
    frame = frame.sort_values([station_column, timestamp_column]).reset_index(drop=True)

    missing_before = {column: int(frame[column].isna().sum()) for column in value_columns}
    imputed_any = pd.Series(False, index=frame.index)
    for column in value_columns:
        before = frame[column].isna()
        frame[column] = frame.groupby(station_column, sort=False)[column].transform(
            lambda values: values.interpolate(
                method="linear",
                limit=interpolation_limit,
                limit_area="inside",
            )
        )
        imputed = before & frame[column].notna()
        imputed_any |= imputed
        frame["quality_flags"] = _append_flag(
            frame["quality_flags"], imputed, f"interpolated:{column}"
        )

    spike_counts: dict[str, int] = {}
    outlier_any = pd.Series(False, index=frame.index)
    selected_spike_columns = set(spike_columns or value_columns)
    minimum_changes = spike_min_absolute_change or {}
    for column in value_columns:
        if column not in selected_spike_columns:
            spike_counts[column] = 0
            continue
        spikes = _spike_mask(
            frame,
            column,
            station_column,
            spike_window,
            spike_min_periods,
            spike_threshold,
            float(minimum_changes.get(column, 0)),
        )
        spike_counts[column] = int(spikes.sum())
        outlier_any |= spikes
        frame["quality_flags"] = _append_flag(
            frame["quality_flags"], spikes, f"possible_spike:{column}"
        )
        if replace_spikes:
            frame.loc[spikes, column] = np.nan

    frame["is_imputed"] = imputed_any
    frame["is_possible_outlier"] = outlier_any
    remaining_missing_count = frame[value_columns].isna().sum(axis=1)
    frame["data_quality_score"] = (
        1.0
        - imputed_any.astype(float) * 0.1
        - outlier_any.astype(float) * 0.2
        - (remaining_missing_count / max(1, len(value_columns))) * 0.5
    ).clip(0, 1)
    frame[timestamp_column] = frame[timestamp_column].map(lambda value: value.isoformat())

    report = {
        "input_rows": input_rows,
        "output_rows": len(frame),
        "invalid_timestamp_rows_removed": invalid_timestamp_rows,
        "duplicates_removed": duplicates_removed,
        "missing_timestamp_rows_added": missing_rows_added,
        "negative_values": negative_counts,
        "out_of_range_values": range_counts,
        "missing_before_interpolation": missing_before,
        "missing_after_interpolation": {
            column: int(frame[column].isna().sum()) for column in value_columns
        },
        "imputed_rows": int(imputed_any.sum()),
        "possible_spikes": spike_counts,
        "possible_outlier_rows": int(outlier_any.sum()),
        "units": units,
    }
    return CleaningResult(data=frame, report=report)
