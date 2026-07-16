"""Grounded, deterministic environmental report endpoints."""

from __future__ import annotations

import os
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from api.dependencies import get_database_writer, get_monitoring_repository
from src.database.connection import environment_flag
from api.schemas import ReportGenerateRequest
from src.services.monitoring_repository import (
    DataSourceUnavailableError,
    MonitoringRepository,
    StationNotFoundError,
)
from src.services.report_service import DEFAULT_PDF_REPORT_DIR, generate_report


router = APIRouter(tags=["reports"])
REPORT_ID_PATTERN = re.compile(r"^RPT-[A-F0-9]{16}$")


@router.post("/reports/generate")
def reports_generate(
    request: ReportGenerateRequest,
    repository: MonitoringRepository = Depends(get_monitoring_repository),
) -> dict:
    try:
        result = generate_report(
            repository,
            station_id=request.station_id,
            start=request.start,
            end=request.end,
            output_format=request.format,
            persist=request.persist,
        )
        metadata = result.pop("_database_metadata", None)
        if (
            environment_flag("DATABASE_WRITE_ENABLED", False)
            and "PYTEST_CURRENT_TEST" not in os.environ
            and isinstance(metadata, dict)
        ):
            try:
                period = metadata["period"]
                get_database_writer().persist_report(
                    result,
                    station_id=request.station_id,
                    period_start=period["start"],
                    period_end=period["end"],
                    metadata=metadata,
                )
                result["database_persisted"] = True
            except Exception as error:
                result["database_persisted"] = False
                result["database_error"] = f"{type(error).__name__}: {error}"[:400]
        return result
    except StationNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"Station not found: {error}") from error
    except DataSourceUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/reports/{report_id}/download", response_class=FileResponse)
def report_download(report_id: str) -> FileResponse:
    """Download a previously generated PDF report by its safe report ID."""

    if not REPORT_ID_PATTERN.fullmatch(report_id):
        raise HTTPException(status_code=404, detail="Report not found")
    target = DEFAULT_PDF_REPORT_DIR / f"{report_id}.pdf"
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(
        target,
        media_type="application/pdf",
        filename=f"environment-ai-{report_id}.pdf",
    )
