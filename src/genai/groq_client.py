"""Minimal OpenAI-compatible client for Groq Chat Completions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv


DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


class GroqClientError(RuntimeError):
    """A sanitized Groq provider error safe to expose to application logs."""


@dataclass(frozen=True)
class GroqConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = "openai/gpt-oss-120b"
    timeout_seconds: float = 60.0
    max_output_tokens: int = 1400
    temperature: float = 0.1

    @classmethod
    def from_environment(cls) -> "GroqConfig":
        load_dotenv(override=False)
        return cls(
            api_key=os.getenv("GROQ_API_KEY", "").strip(),
            base_url=os.getenv("GROQ_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL,
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
            or "openai/gpt-oss-120b",
            timeout_seconds=float(os.getenv("GROQ_TIMEOUT_SECONDS", "60")),
            max_output_tokens=int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "1400")),
            temperature=float(os.getenv("GROQ_TEMPERATURE", "0.1")),
        )


class GroqClient:
    def __init__(
        self,
        config: GroqConfig | None = None,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config or GroqConfig.from_environment()
        self.http_client = http_client or httpx.Client(timeout=self.config.timeout_seconds)

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
            raise GroqClientError("Groq is not configured")
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_output_tokens,
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
            raise GroqClientError("Groq network request failed") from error
        if response.is_error:
            raise GroqClientError(f"Groq returned HTTP {response.status_code}")
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
            raise GroqClientError("Groq returned an invalid JSON response") from error
        if not isinstance(result, dict):
            raise GroqClientError("Groq JSON response must be an object")
        return result
