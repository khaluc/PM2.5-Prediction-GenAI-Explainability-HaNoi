"""Deterministic environmental reports generated from monitoring data."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.services.monitoring_repository import MonitoringRepository, _python_value
from src.services.pdf_report import render_environment_report_pdf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"
DEFAULT_PDF_REPORT_DIR = PROJECT_ROOT / "output" / "pdf"


def _metric(series: pd.Series, operation: str) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return round(float(getattr(numeric, operation)()), 3)


def _markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    quality = report["data_quality"]
    return "\n".join(
        [
            f"# Báo cáo chất lượng không khí – {report['station']['name']}",
            "",
            f"- Mã trạm: `{report['station']['station_id']}`",
            f"- Thời gian dữ liệu: {report['period']['start']} – {report['period']['end']}",
            f"- Số bản ghi: {report['period']['observation_count']}",
            f"- Nguồn: {report['provenance']['air_source']}",
            "",
            "## PM2.5",
            "",
            f"- Trung bình giờ: {metrics['pm25_mean']} µg/m³",
            f"- Lớn nhất: {metrics['pm25_max']} µg/m³ tại {metrics['pm25_peak_timestamp']}",
            f"- Nhỏ nhất: {metrics['pm25_min']} µg/m³",
            "",
            "## Khí tượng và chất lượng dữ liệu",
            "",
            f"- Độ ẩm trung bình: {metrics['humidity_mean']}%",
            f"- Tốc độ gió trung bình: {metrics['wind_speed_mean']} km/h",
            f"- Tỷ lệ dữ liệu nội suy: {quality['imputed_fraction']}",
            f"- Tỷ lệ cờ ngoại lệ: {quality['possible_outlier_fraction']}",
            "",
            "> Báo cáo mô tả dữ liệu theo giờ; không phải kết luận pháp lý, y tế hoặc quy nguồn phát thải.",
        ]
    )


def generate_report(
    repository: MonitoringRepository,
    *,
    station_id: str,
    start: Any | None = None,
    end: Any | None = None,
    output_format: str = "json",
    persist: bool = True,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    frame = repository.period_frame(station_id, start=start, end=end)
    if frame.empty:
        raise ValueError("No observations in the requested reporting period")
    period_days = (frame["timestamp"].max() - frame["timestamp"].min()).total_seconds() / 86_400
    if period_days > 366:
        raise ValueError("Report period cannot exceed 366 days")
    pm25 = pd.to_numeric(frame["pm25"], errors="coerce")
    peak_index = pm25.idxmax() if pm25.notna().any() else None
    latest = frame.iloc[-1]
    identity = f"{station_id}|{frame['timestamp'].min()}|{frame['timestamp'].max()}|{output_format}"
    report_id = "RPT-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16].upper()
    report = {
        "report_id": report_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "station": {
            "station_id": station_id,
            "name": _python_value(latest.get("location_name")),
            "latitude": _python_value(latest.get("latitude")),
            "longitude": _python_value(latest.get("longitude")),
        },
        "period": {
            "start": _python_value(frame["timestamp"].min()),
            "end": _python_value(frame["timestamp"].max()),
            "observation_count": int(len(frame)),
        },
        "metrics": {
            "pm25_mean": _metric(frame["pm25"], "mean"),
            "pm25_min": _metric(frame["pm25"], "min"),
            "pm25_max": _metric(frame["pm25"], "max"),
            "pm25_peak_timestamp": _python_value(frame.loc[peak_index, "timestamp"]) if peak_index is not None else None,
            "pm10_mean": _metric(frame["pm10"], "mean"),
            "humidity_mean": _metric(frame["humidity"], "mean"),
            "wind_speed_mean": _metric(frame["wind_speed"], "mean"),
        },
        "data_quality": {
            "mean_score": _metric(frame["data_quality_score"], "mean"),
            "imputed_fraction": round(float(frame["is_imputed"].fillna(False).astype(bool).mean()), 4),
            "possible_outlier_fraction": round(
                float(frame["is_possible_outlier"].fillna(False).astype(bool).mean()), 4
            ),
        },
        "provenance": {
            "air_source": _python_value(latest.get("air_source")),
            "weather_source": _python_value(latest.get("weather_source")),
            "source_file": str(repository.path),
        },
        "limitations": [
            "Các điểm Hà Nội hiện dùng ước lượng CAMS, không phải trạm quan trắc mặt đất chính thức.",
            "Thống kê PM2.5 theo giờ không tự động tạo thành đánh giá tuân thủ pháp lý hoặc khuyến nghị y tế.",
            "Báo cáo không xác định nguồn phát thải hoặc tổ chức vi phạm.",
        ],
    }
    if output_format not in {"json", "markdown", "pdf"}:
        raise ValueError("Unsupported report format")
    content: Any = report if output_format in {"json", "pdf"} else _markdown(report)
    output_path = None
    if persist:
        directory = Path(
            output_dir
            if output_dir is not None
            else (DEFAULT_PDF_REPORT_DIR if output_format == "pdf" else DEFAULT_REPORT_DIR)
        )
        directory.mkdir(parents=True, exist_ok=True)
        suffix = {"json": ".json", "markdown": ".md", "pdf": ".pdf"}[output_format]
        target = directory / f"{report_id}{suffix}"
        if output_format == "json":
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        elif output_format == "markdown":
            target.write_text(str(content), encoding="utf-8")
        else:
            render_environment_report_pdf(report, frame, target)
        try:
            output_path = str(target.resolve().relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            output_path = target.name
    return {
        "report_id": report_id,
        "format": output_format,
        "persisted": persist,
        "output_path": output_path,
        "content": content,
        "download_url": f"/reports/{report_id}/download" if output_format == "pdf" and persist else None,
        "_database_metadata": report,
    }
