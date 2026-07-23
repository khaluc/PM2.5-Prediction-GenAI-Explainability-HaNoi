"""DashScope OpenAI-compatible client for the DeepSeek V4 model."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv


DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class DashScopeClientError(RuntimeError):
    """A sanitized DashScope provider error safe to expose to application logs."""


@dataclass(frozen=True)
class DashScopeConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = "deepseek-v4-flash"
    timeout_seconds: float = 90.0
    max_output_tokens: int = 1400
    temperature: float = 0.1
    thinking_enabled: bool = False

    @classmethod
    def from_environment(cls) -> "DashScopeConfig":
        load_dotenv(override=False)
        # DEEPSEEK_API_KEY/MODEL are accepted only as a local migration alias.
        # The legacy DEEPSEEK_BASE_URL is intentionally ignored because this
        # client must always target DashScope unless DASHSCOPE_BASE_URL is set.
        return cls(
            api_key=(
                os.getenv("DASHSCOPE_API_KEY", "").strip()
                or os.getenv("DEEPSEEK_API_KEY", "").strip()
            ),
            base_url=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL).strip()
            or DEFAULT_BASE_URL,
            model=(
                os.getenv("DASHSCOPE_MODEL", "").strip()
                or os.getenv("DEEPSEEK_MODEL", "").strip()
                or "deepseek-v4-flash"
            ),
            timeout_seconds=float(
                os.getenv("DASHSCOPE_TIMEOUT_SECONDS")
                or os.getenv("DEEPSEEK_TIMEOUT_SECONDS")
                or "90"
            ),
            max_output_tokens=int(
                os.getenv("DASHSCOPE_MAX_OUTPUT_TOKENS")
                or os.getenv("DEEPSEEK_MAX_OUTPUT_TOKENS")
                or "1400"
            ),
            temperature=float(
                os.getenv("DASHSCOPE_TEMPERATURE")
                or os.getenv("DEEPSEEK_TEMPERATURE")
                or "0.1"
            ),
            thinking_enabled=(
                os.getenv("DASHSCOPE_THINKING_ENABLED")
                or os.getenv("DEEPSEEK_THINKING_ENABLED")
                or "false"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
        )


class DashScopeClient:
    def __init__(
        self,
        config: DashScopeConfig | None = None,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config or DashScopeConfig.from_environment()
        self.http_client = http_client or httpx.Client(
            timeout=self.config.timeout_seconds
        )

    @property
    def available(self) -> bool:
        return bool(self.config.api_key)

    @property
    def endpoint(self) -> str:
        base_url = self.config.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if not self.available:
            raise DashScopeClientError("DashScope is not configured")
        if self.config.thinking_enabled:
            raise DashScopeClientError(
                "DashScope thinking mode is incompatible with JSON output"
            )
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "enable_thinking": False,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            response = self.http_client.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except httpx.HTTPError as error:
            raise DashScopeClientError("DashScope network request failed") from error
        if response.is_error:
            raise DashScopeClientError(
                f"DashScope returned HTTP {response.status_code}"
            )
        try:
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message content is not text")
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.removeprefix("```json").removeprefix("```")
                cleaned = cleaned.removesuffix("```").strip()
            result = json.loads(cleaned)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise DashScopeClientError(
                "DashScope returned an invalid JSON response"
            ) from error
        if not isinstance(result, dict):
            raise DashScopeClientError("DashScope JSON response must be an object")
        return result
