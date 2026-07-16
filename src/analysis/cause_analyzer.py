"""Rank evidence-backed pollution cause hypotheses.

This module does not perform formal source apportionment. It ranks plausible
contributing conditions from pollutant and meteorological signals relative to
station/month historical baselines, and emits structured evidence for operators.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


CAUSE_LABELS = {
    "regional_accumulation": "Tích tụ trên diện rộng",
    "atmospheric_stagnation": "Khí quyển ít khuếch tán",
    "combustion_traffic_proxy": "Tín hiệu đốt cháy/giao thông",
    "secondary_aerosol_conditions": "Điều kiện hình thành aerosol thứ cấp",
    "coarse_dust_resuspension": "Bụi thô hoặc tái huyền phù",
    "photochemical_conditions": "Điều kiện quang hóa",
    "unresolved": "Chưa đủ bằng chứng",
}

BASELINE_FEATURES = [
    "pm25",
    "pm10",
    "co",
    "no2",
    "so2",
    "o3",
    "temperature",
    "humidity",
    "wind_speed",
    "precipitation",
    "pm25_to_pm10_ratio",
]

LIMITATIONS = [
    "Các điểm Hà Nội dùng ước lượng CAMS, không phải phép đo hóa học tại trạm mặt đất.",
    "Không có dữ liệu thành phần ion, carbon hay kim loại để làm source apportionment.",
    "Không có kiểm kê phát thải hoặc quỹ đạo khối khí/back-trajectory trong pipeline.",
    "TomTom chỉ có snapshot hiện tại, không đủ để quy nguồn giao thông cho lịch sử.",
    "Điểm giả thuyết biểu thị độ mạnh bằng chứng, không phải xác suất nhân quả.",
]


@dataclass
class CauseProfile:
    feature_columns: list[str]
    group_columns: list[str]
    baseline: pd.DataFrame
    global_baseline: dict[str, dict[str, float]]
    timezone_name: str
    minimum_score: float


def _prepare_time(
    frame: pd.DataFrame,
    timestamp_column: str,
    timezone_name: str,
) -> pd.DataFrame:
    work = frame.copy()
    work[timestamp_column] = pd.to_datetime(
        work[timestamp_column], errors="raise", utc=True
    ).dt.tz_convert(timezone_name)
    work["month"] = work[timestamp_column].dt.month.astype("int8")
    work["hour"] = work[timestamp_column].dt.hour.astype("int8")
    return work


def fit_cause_profile(
    train: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    group_columns: list[str] | None = None,
    timestamp_column: str = "timestamp",
    timezone_name: str = "Asia/Ho_Chi_Minh",
    minimum_score: float = 0.35,
) -> CauseProfile:
    """Fit robust station/season baselines using train only."""
    features = list(feature_columns or BASELINE_FEATURES)
    groups = list(group_columns or ["station_id", "month"])
    required = set(features + [timestamp_column, "station_id"])
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"Missing cause baseline columns: {missing}")
    work = _prepare_time(train, timestamp_column, timezone_name)
    grouped = work.groupby(groups, observed=True)[features]
    median = grouped.median().add_suffix("__median")
    q25 = grouped.quantile(0.25).add_suffix("__q25")
    q75 = grouped.quantile(0.75).add_suffix("__q75")
    count = grouped.size().rename("baseline_rows")
    baseline = pd.concat([median, q25, q75, count], axis=1).reset_index()
    global_baseline = {
        feature: {
            "median": float(pd.to_numeric(work[feature], errors="coerce").median()),
            "q25": float(pd.to_numeric(work[feature], errors="coerce").quantile(0.25)),
            "q75": float(pd.to_numeric(work[feature], errors="coerce").quantile(0.75)),
        }
        for feature in features
    }
    return CauseProfile(
        feature_columns=features,
        group_columns=groups,
        baseline=baseline,
        global_baseline=global_baseline,
        timezone_name=timezone_name,
        minimum_score=float(minimum_score),
    )


def _clip01(values) -> pd.Series:
    return pd.Series(values).clip(lower=0.0, upper=1.0)


def _attach_baseline(
    frame: pd.DataFrame,
    profile: CauseProfile,
    timestamp_column: str,
) -> pd.DataFrame:
    work = _prepare_time(frame, timestamp_column, profile.timezone_name)
    work = work.merge(
        profile.baseline,
        on=profile.group_columns,
        how="left",
        validate="many_to_one",
    )
    for feature in profile.feature_columns:
        for statistic in ["median", "q25", "q75"]:
            column = f"{feature}__{statistic}"
            work[column] = work[column].fillna(
                profile.global_baseline[feature][statistic]
            )
        iqr = (work[f"{feature}__q75"] - work[f"{feature}__q25"]).abs()
        fallback_iqr = max(
            profile.global_baseline[feature]["q75"]
            - profile.global_baseline[feature]["q25"],
            1e-3,
        )
        iqr = iqr.where(iqr > 1e-6, fallback_iqr)
        work[f"{feature}__robust_z"] = (
            pd.to_numeric(work[feature], errors="coerce")
            - work[f"{feature}__median"]
        ) / iqr
    return work


def _confidence(score: float, evidence_count: int) -> str:
    if score >= 0.7 and evidence_count >= 3:
        return "high"
    if score >= 0.5 and evidence_count >= 2:
        return "moderate"
    return "low"


def _evidence_for_row(row: pd.Series, cause: str) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []

    def add(code: str, condition: bool, value: Any, description: str) -> None:
        if bool(condition):
            evidence.append(
                {"code": code, "value": None if pd.isna(value) else float(value), "description_vi": description}
            )

    if cause in {"regional_accumulation", "atmospheric_stagnation"}:
        add(
            "low_wind",
            row["signal_low_wind"] >= 0.5,
            row["wind_speed"],
            "Tốc độ gió thấp so với nền cùng trạm và tháng.",
        )
        add(
            "no_rain",
            row["signal_no_rain"] >= 0.5,
            row["precipitation"],
            "Không có mưa đáng kể để rửa trôi hạt bụi.",
        )
        add(
            "citywide",
            row["city_affected_fraction"] >= 0.75,
            row["city_affected_fraction"],
            "Phần lớn khu vực cùng ghi nhận PM2.5 cao tại thời điểm này.",
        )
        add(
            "high_humidity",
            row["signal_high_humidity"] >= 0.5,
            row["humidity"],
            "Độ ẩm cao hơn nền, thuận lợi cho tăng trưởng hút ẩm của aerosol.",
        )
    elif cause == "combustion_traffic_proxy":
        add("high_co", row["signal_high_co"] >= 0.5, row["co"], "CO cao hơn nền mùa tại trạm.")
        add("high_no2", row["signal_high_no2"] >= 0.5, row["no2"], "NO₂ cao hơn nền mùa tại trạm.")
        add("rush_hour", row["signal_rush_hour"] >= 0.5, row["hour"], "Thời điểm thuộc khung giờ giao thông cao điểm.")
        add("fine_fraction", row["signal_fine_ratio"] >= 0.5, row["pm25_to_pm10_ratio"], "Tỷ lệ PM2.5/PM10 thiên về hạt mịn.")
    elif cause == "secondary_aerosol_conditions":
        add("fine_fraction", row["signal_fine_ratio"] >= 0.5, row["pm25_to_pm10_ratio"], "Tỷ lệ hạt mịn cao.")
        add("high_humidity", row["signal_high_humidity"] >= 0.5, row["humidity"], "Độ ẩm cao hơn nền.")
        add("high_no2", row["signal_high_no2"] >= 0.5, row["no2"], "NO₂ tiền chất cao hơn nền.")
        add("high_so2", row["signal_high_so2"] >= 0.5, row["so2"], "SO₂ tiền chất cao hơn nền.")
    elif cause == "coarse_dust_resuspension":
        add("high_pm10", row["signal_high_pm10"] >= 0.5, row["pm10"], "PM10 cao hơn nền mùa tại trạm.")
        add("coarse_fraction", row["signal_coarse_ratio"] >= 0.5, row["pm25_to_pm10_ratio"], "Tỷ lệ PM2.5/PM10 thấp, thiên về hạt thô.")
        add("high_wind", row["signal_high_wind"] >= 0.5, row["wind_speed"], "Gió cao hơn nền có thể làm tái huyền phù bụi.")
        add("low_humidity", row["signal_low_humidity"] >= 0.5, row["humidity"], "Không khí khô hơn nền.")
    elif cause == "photochemical_conditions":
        add("high_o3", row["signal_high_o3"] >= 0.5, row["o3"], "O₃ cao hơn nền mùa tại trạm.")
        add("high_temperature", row["signal_high_temperature"] >= 0.5, row["temperature"], "Nhiệt độ cao hơn nền mùa.")
        add("daylight", row["signal_daylight"] >= 0.5, row["hour"], "Thời điểm ban ngày thuận lợi cho phản ứng quang hóa.")
    return evidence


def analyze_pollution_causes(
    events: pd.DataFrame,
    profile: CauseProfile,
    *,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Score, rank and explain plausible contributing-condition hypotheses."""
    required = set(profile.feature_columns + [timestamp_column, "station_id"])
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Missing cause analysis columns: {missing}")
    work = _attach_baseline(events, profile, timestamp_column)

    high = lambda feature: _clip01((work[f"{feature}__robust_z"] - 0.25) / 1.5)
    low = lambda feature: _clip01((-work[f"{feature}__robust_z"] - 0.25) / 1.5)
    work["signal_low_wind"] = low("wind_speed")
    work["signal_high_wind"] = high("wind_speed")
    work["signal_no_rain"] = work["precipitation"].le(0.1).astype(float)
    work["signal_high_humidity"] = high("humidity")
    work["signal_low_humidity"] = low("humidity")
    work["signal_high_temperature"] = high("temperature")
    work["signal_high_pm10"] = high("pm10")
    work["signal_high_co"] = high("co")
    work["signal_high_no2"] = high("no2")
    work["signal_high_so2"] = high("so2")
    work["signal_high_o3"] = high("o3")
    work["signal_fine_ratio"] = _clip01(
        (work["pm25_to_pm10_ratio"] - 0.65) / 0.20
    )
    work["signal_coarse_ratio"] = _clip01(
        (0.70 - work["pm25_to_pm10_ratio"]) / 0.20
    )
    work["signal_rush_hour"] = work["hour"].isin([6, 7, 8, 9, 16, 17, 18, 19, 20]).astype(float)
    work["signal_daylight"] = work["hour"].between(10, 16).astype(float)
    work["signal_pollution_strength"] = _clip01(
        (work["pm25_rolling_mean_24h"] - 50.0) / 75.0
    )
    if "city_affected_fraction" not in work:
        work["city_affected_fraction"] = 0.0
    if "city_spatial_cv" not in work:
        work["city_spatial_cv"] = 1.0
    work["signal_spatial_uniformity"] = _clip01(
        1.0 - work["city_spatial_cv"].fillna(1.0) / 0.40
    )

    scores = pd.DataFrame(index=work.index)
    scores["regional_accumulation"] = (
        0.40 * work["city_affected_fraction"]
        + 0.20 * work["signal_spatial_uniformity"]
        + 0.20 * work["signal_low_wind"]
        + 0.10 * work["signal_no_rain"]
        + 0.10 * work["signal_pollution_strength"]
    )
    scores["atmospheric_stagnation"] = (
        0.35 * work["signal_low_wind"]
        + 0.20 * work["signal_no_rain"]
        + 0.15 * work["signal_high_humidity"]
        + 0.20 * work["signal_pollution_strength"]
        + 0.10 * work["city_affected_fraction"]
    )
    scores["combustion_traffic_proxy"] = (
        0.27 * work["signal_high_co"]
        + 0.27 * work["signal_high_no2"]
        + 0.16 * work["signal_rush_hour"]
        + 0.14 * work["signal_fine_ratio"]
        + 0.08 * work["signal_low_wind"]
        + 0.08 * work["signal_pollution_strength"]
    )
    scores["secondary_aerosol_conditions"] = (
        0.25 * work["signal_fine_ratio"]
        + 0.20 * work["signal_high_humidity"]
        + 0.15 * work["signal_high_no2"]
        + 0.15 * work["signal_high_so2"]
        + 0.15 * work["signal_low_wind"]
        + 0.10 * work["signal_pollution_strength"]
    )
    scores["coarse_dust_resuspension"] = (
        0.30 * work["signal_high_pm10"]
        + 0.25 * work["signal_coarse_ratio"]
        + 0.15 * work["signal_low_humidity"]
        + 0.15 * work["signal_high_wind"]
        + 0.10 * work["signal_no_rain"]
        + 0.05 * work["signal_pollution_strength"]
    )
    scores["photochemical_conditions"] = (
        0.35 * work["signal_high_o3"]
        + 0.25 * work["signal_high_temperature"]
        + 0.15 * work["signal_daylight"]
        + 0.15 * work["signal_low_wind"]
        + 0.10 * work["signal_pollution_strength"]
    )
    scores = scores.clip(0.0, 1.0)

    ranked_causes = np.argsort(-scores.to_numpy(), axis=1)
    cause_names = np.asarray(scores.columns)
    top_causes = cause_names[ranked_causes[:, 0]]
    top_scores = scores.to_numpy()[np.arange(len(scores)), ranked_causes[:, 0]]
    second_causes = cause_names[ranked_causes[:, 1]]
    second_scores = scores.to_numpy()[np.arange(len(scores)), ranked_causes[:, 1]]
    unresolved = top_scores < profile.minimum_score
    top_causes = np.where(unresolved, "unresolved", top_causes)

    output = events.reset_index(drop=True).copy()
    work = work.reset_index(drop=True)
    scores = scores.reset_index(drop=True)
    output["top_hypothesis"] = top_causes
    output["top_hypothesis_vi"] = [CAUSE_LABELS[value] for value in top_causes]
    output["top_hypothesis_score"] = top_scores
    output["second_hypothesis"] = second_causes
    output["second_hypothesis_vi"] = [CAUSE_LABELS[value] for value in second_causes]
    output["second_hypothesis_score"] = second_scores

    evidence_json = []
    confidence = []
    score_json = []
    for index, cause in enumerate(top_causes):
        evidence = [] if cause == "unresolved" else _evidence_for_row(work.iloc[index], cause)
        evidence_json.append(json.dumps(evidence, ensure_ascii=False))
        confidence.append(_confidence(float(top_scores[index]), len(evidence)))
        score_json.append(
            json.dumps(
                {
                    name: round(float(scores.loc[index, name]), 4)
                    for name in scores.columns
                },
                ensure_ascii=False,
            )
        )
    output["evidence_strength"] = confidence
    output["evidence_json"] = evidence_json
    output["hypothesis_scores_json"] = score_json
    output["limitations_json"] = json.dumps(LIMITATIONS, ensure_ascii=False)
    output["causal_claim_allowed"] = False
    return output
