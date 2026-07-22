"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from api.dependencies import (
    get_feature_repository,
    get_hourly_update_service,
    get_monitoring_repository,
)
from api.routes import (
    alerts,
    explanations,
    knowledge,
    monitoring,
    news,
    predictions,
    reports,
    system,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    hourly_updates = get_hourly_update_service()
    hourly_updates.start()
    try:
        yield
    finally:
        hourly_updates.stop()

app = FastAPI(
    title="Environment AI API",
    version="0.2.0",
    description=(
        "API giám sát chất lượng không khí Hà Nội, dự báo PM2.5, phát hiện bất thường, "
        "cảnh báo và báo cáo."
    ),
    lifespan=lifespan,
)

app.include_router(monitoring.router)
app.include_router(predictions.router)
app.include_router(explanations.router)
app.include_router(knowledge.router)
app.include_router(alerts.router)
app.include_router(reports.router)
app.include_router(news.router)
app.include_router(system.router)


@app.get("/", include_in_schema=False)
def api_home() -> RedirectResponse:
    """Open the interactive API documentation from the base URL."""

    return RedirectResponse(url="/docs", status_code=307)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready", tags=["system"])
def readiness() -> dict[str, Any]:
    monitoring_status = get_monitoring_repository().health()
    feature_status = get_feature_repository().health()
    components = {
        "monitoring_data": monitoring_status,
        "feature_data": feature_status,
        "hourly_update": get_hourly_update_service().status(),
    }
    ready = monitoring_status.get("available", False) and feature_status.get("available", False)
    return {"status": "ready" if ready else "not_ready", "components": components}
