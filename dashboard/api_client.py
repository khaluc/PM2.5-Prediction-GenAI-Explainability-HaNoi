"""Small, sanitized HTTP client used by the Flask dashboard."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx


class DashboardApiError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class DashboardApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 45.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.http_client = http_client or httpx.Client(timeout=timeout_seconds)

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        try:
            response = self.http_client.request(
                method,
                f"{self.base_url}{path}",
                params=params,
                json=json,
            )
        except httpx.HTTPError as error:
            raise DashboardApiError(503, "Không thể kết nối tới Environment AI API.") from error
        if response.is_error:
            try:
                payload = response.json()
                detail = payload.get("detail") or payload.get("error") or "Backend request failed"
            except ValueError:
                detail = "Backend request failed"
            raise DashboardApiError(response.status_code, str(detail))
        try:
            return response.json()
        except ValueError as error:
            raise DashboardApiError(502, "Backend trả về dữ liệu không hợp lệ.") from error

    def download(self, path: str) -> tuple[bytes, str, str | None]:
        """Download a binary artifact while preserving sanitized backend errors."""

        try:
            response = self.http_client.get(f"{self.base_url}{path}")
        except httpx.HTTPError as error:
            raise DashboardApiError(503, "Không thể kết nối tới Environment AI API.") from error
        if response.is_error:
            try:
                payload = response.json()
                detail = payload.get("detail") or payload.get("error") or "Backend request failed"
            except ValueError:
                detail = "Backend request failed"
            raise DashboardApiError(response.status_code, str(detail))
        content_type = response.headers.get("content-type", "application/octet-stream").split(";", 1)[0]
        disposition = response.headers.get("content-disposition", "")
        filename = None
        match = re.search(r"filename\*?=(?:UTF-8'')?[\"']?([^\"';]+)", disposition, re.I)
        if match:
            filename = Path(unquote(match.group(1))).name
        return response.content, content_type, filename
