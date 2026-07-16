"""Public environmental news feed endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_news_crawler
from src.collection.news_collector import NewsCrawler, NewsSourceUnavailableError


router = APIRouter(tags=["news"])


@router.get("/news")
def list_news(
    category: Literal["latest", "domestic", "international"] = "latest",
    page: int = Query(default=1, ge=1, le=10),
    limit: int = Query(default=12, ge=1, le=30),
    refresh: bool = False,
    crawler: NewsCrawler = Depends(get_news_crawler),
) -> dict:
    """Return source-attributed summaries from the public listing page."""

    try:
        return crawler.fetch(
            category,
            page=page,
            limit=limit,
            force_refresh=refresh,
        )
    except NewsSourceUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
