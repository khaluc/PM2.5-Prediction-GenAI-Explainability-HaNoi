"""Collect public news summaries from Môi Trường Thủ Đô.

Only metadata already present on the public listing page is collected. Full
article bodies are deliberately not downloaded or stored.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup


NEWS_SOURCE_URL = "https://moitruongthudo.vn/thong-tin"
NEWS_SOURCE_NAME = "Môi Trường Thủ Đô"
NEWS_ALLOWED_HOST = "moitruongthudo.vn"
NEWS_CATEGORIES = {
    "latest": {"label": "Tin mới", "mode": None},
    "domestic": {"label": "Tin trong nước", "mode": "1"},
    "international": {"label": "Tin quốc tế", "mode": "2"},
}
DEFAULT_NEWS_CACHE_PATH = Path("artifacts/news/news_cache.json")

NewsCategory = Literal["latest", "domestic", "international"]


class NewsSourceUnavailableError(RuntimeError):
    """Raised when the source is unavailable and no cached copy exists."""


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _safe_source_url(value: str, *, article: bool = False) -> str | None:
    absolute = urljoin(NEWS_SOURCE_URL, value.strip())
    parsed = urlparse(absolute)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if parsed.scheme != "https" or host != NEWS_ALLOWED_HOST:
        return None
    if article and not parsed.path.startswith("/thong-tin/"):
        return None
    return absolute


def _image_from_article(article: Any) -> str | None:
    image_box = article.select_one(".imgbg")
    if image_box:
        style = str(image_box.get("style") or "")
        match = re.search(r"background-image\s*:\s*url\(\s*['\"]?([^)'\"]+)", style, re.I)
        if match:
            return _safe_source_url(match.group(1))
    image = article.select_one("img[src]")
    return _safe_source_url(str(image.get("src"))) if image else None


def _parse_date(raw_date: str) -> tuple[str, str | None]:
    display = _normalise_text(raw_date)
    try:
        return display, datetime.strptime(display, "%d/%m/%Y").date().isoformat()
    except ValueError:
        return display, None


def parse_news_html(html_text: str) -> list[dict[str, Any]]:
    """Parse listing-page HTML into safe, source-attributed news metadata."""

    soup = BeautifulSoup(html_text, "html.parser")
    items: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for article in soup.select(".news-item article"):
        heading = article.select_one("h3")
        link = heading.find_parent("a", href=True) if heading else None
        if not heading or not link:
            continue
        url = _safe_source_url(str(link.get("href")), article=True)
        title = _normalise_text(heading.get_text(" ", strip=True))
        if not url or not title or url in seen_urls:
            continue

        date_node = article.select_one(".byline span")
        date_display, published_date = _parse_date(
            date_node.get_text(" ", strip=True) if date_node else ""
        )
        excerpt_node = article.select_one("p")
        excerpt = _normalise_text(
            excerpt_node.get_text(" ", strip=True) if excerpt_node else ""
        )
        if len(excerpt) > 360:
            excerpt = f"{excerpt[:357].rstrip()}…"

        seen_urls.add(url)
        items.append(
            {
                "id": hashlib.sha256(url.encode("utf-8")).hexdigest()[:16],
                "title": title,
                "excerpt": excerpt,
                "published_date": published_date,
                "date_display": date_display,
                "url": url,
                "image_url": _image_from_article(article),
                "source": NEWS_SOURCE_NAME,
            }
        )
    return items


class NewsCrawler:
    """Rate-limited listing crawler with an on-disk stale-cache fallback."""

    def __init__(
        self,
        cache_path: str | Path = DEFAULT_NEWS_CACHE_PATH,
        *,
        cache_ttl_seconds: int = 1_800,
        timeout_seconds: float = 20.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.cache_ttl_seconds = max(60, int(cache_ttl_seconds))
        self._owns_client = http_client is None
        self.http_client = http_client or httpx.Client(
            timeout=max(3.0, float(timeout_seconds)),
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "EnvironmentAI-NewsReader/1.0 "
                    "(+local environmental dashboard; source attribution)"
                ),
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        self._lock = threading.RLock()
        self._cache = self._read_cache()

    def close(self) -> None:
        if self._owns_client:
            self.http_client.close()

    def _read_cache(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {"version": 1, "entries": {}}
        if not isinstance(payload, dict) or not isinstance(payload.get("entries"), dict):
            return {"version": 1, "entries": {}}
        return payload

    def _write_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.cache_path)

    @staticmethod
    def _cache_key(category: NewsCategory, page: int) -> str:
        return f"{category}:{page}"

    def _cached_entry(self, key: str) -> dict[str, Any] | None:
        entry = self._cache.get("entries", {}).get(key)
        return entry if isinstance(entry, dict) and isinstance(entry.get("items"), list) else None

    def _is_fresh(self, entry: dict[str, Any]) -> bool:
        try:
            fetched_at = datetime.fromisoformat(str(entry["fetched_at"]).replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError):
            return False
        age = (datetime.now(timezone.utc) - fetched_at.astimezone(timezone.utc)).total_seconds()
        return 0 <= age < self.cache_ttl_seconds

    @staticmethod
    def _source_url(category: NewsCategory, page: int) -> str:
        params: dict[str, str | int] = {}
        mode = NEWS_CATEGORIES[category]["mode"]
        if mode:
            params["mode"] = str(mode)
        if page > 1:
            params["page"] = page
        return f"{NEWS_SOURCE_URL}?{urlencode(params)}" if params else NEWS_SOURCE_URL

    def _response_payload(
        self,
        entry: dict[str, Any],
        *,
        category: NewsCategory,
        page: int,
        limit: int,
        cache_status: str,
        stale: bool,
    ) -> dict[str, Any]:
        all_items = list(entry.get("items", []))
        return {
            "source": {
                "name": NEWS_SOURCE_NAME,
                "url": NEWS_SOURCE_URL,
                "attribution": "Tiêu đề và tóm tắt từ trang tin công khai; đọc toàn văn tại nguồn.",
            },
            "category": category,
            "category_label": NEWS_CATEGORIES[category]["label"],
            "page": page,
            "limit": limit,
            "total": min(len(all_items), limit),
            "has_next": bool(all_items),
            "fetched_at": entry.get("fetched_at"),
            "cache_status": cache_status,
            "stale": stale,
            "items": all_items[:limit],
        }

    def fetch(
        self,
        category: NewsCategory = "latest",
        *,
        page: int = 1,
        limit: int = 12,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        if category not in NEWS_CATEGORIES:
            raise ValueError("Unsupported news category")
        if not 1 <= page <= 10:
            raise ValueError("page must be between 1 and 10")
        if not 1 <= limit <= 30:
            raise ValueError("limit must be between 1 and 30")

        key = self._cache_key(category, page)
        with self._lock:
            cached = self._cached_entry(key)
            if cached and not force_refresh and self._is_fresh(cached):
                return self._response_payload(
                    cached,
                    category=category,
                    page=page,
                    limit=limit,
                    cache_status="fresh_cache",
                    stale=False,
                )

        try:
            response = self.http_client.get(self._source_url(category, page))
            response.raise_for_status()
            if not _safe_source_url(str(response.url)):
                raise NewsSourceUnavailableError("Trang nguồn chuyển hướng tới địa chỉ không được phép.")
            content_type = response.headers.get("content-type", "text/html").lower()
            if "html" not in content_type:
                raise NewsSourceUnavailableError("Trang nguồn không trả về nội dung HTML.")
            if len(response.content) > 2_500_000:
                raise NewsSourceUnavailableError("Trang nguồn vượt quá giới hạn dữ liệu cho phép.")
            items = parse_news_html(response.text)
            if not items:
                raise NewsSourceUnavailableError("Trang nguồn không có tin phù hợp để hiển thị.")
        except (httpx.HTTPError, NewsSourceUnavailableError) as error:
            with self._lock:
                cached = self._cached_entry(key)
                if cached:
                    payload = self._response_payload(
                        cached,
                        category=category,
                        page=page,
                        limit=limit,
                        cache_status="stale_cache",
                        stale=True,
                    )
                    payload["warning"] = "Nguồn tin tạm thời không phản hồi; đang hiển thị bản lưu gần nhất."
                    return payload
            raise NewsSourceUnavailableError(
                "Không thể tải tin từ Môi Trường Thủ Đô và chưa có bản lưu cục bộ."
            ) from error

        fetched_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        entry = {"fetched_at": fetched_at, "items": items}
        with self._lock:
            self._cache.setdefault("entries", {})[key] = entry
            try:
                self._write_cache()
            except OSError:
                # A read-only cache directory must not make live news unavailable.
                pass
        return self._response_payload(
            entry,
            category=category,
            page=page,
            limit=limit,
            cache_status="live",
            stale=False,
        )
