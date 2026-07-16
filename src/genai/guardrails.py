"""Input/output guardrails for forecast explanations."""

from __future__ import annotations

import re
from typing import Any, Iterable


REQUIRED_TEXT_FIELDS = ("headline", "summary", "sensitive_group_advice", "uncertainty")
REQUIRED_LIST_FIELDS = ("contributing_conditions", "recommended_actions")
FORBIDDEN_CLAIMS = (
    re.compile(r"\bnguyên nhân (?:chính |trực tiếp |duy nhất )?là\b", re.IGNORECASE),
    re.compile(r"\bchắc chắn (?:là|do|gây)\b", re.IGNORECASE),
    re.compile(r"\b(?:công ty|nhà máy|doanh nghiệp|tổ chức)\b.{0,80}\b(?:gây|xả)\b", re.IGNORECASE),
    re.compile(r"\b(?:tự động )?(?:tắt trạm|dừng cảm biến|sơ tán|phong tỏa)\b", re.IGNORECASE),
)
NUMBER_PATTERN = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?")


class GuardrailViolation(ValueError):
    """Generated content is not grounded in the supplied forecast context."""


def _all_text(payload: dict[str, Any]) -> str:
    values: list[str] = []
    for value in payload.values():
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value)
    return "\n".join(values)


def _normalise_number(value: str | int | float) -> float:
    return round(float(str(value).replace(",", ".")), 2)


def validate_generated_explanation(
    payload: dict[str, Any],
    *,
    allowed_numbers: Iterable[int | float],
) -> dict[str, Any]:
    """Validate shape, claims and numerical grounding of an LLM response."""
    for field in REQUIRED_TEXT_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > 900:
            raise GuardrailViolation(f"Invalid generated field: {field}")
    for field in REQUIRED_LIST_FIELDS:
        value = payload.get(field)
        if not isinstance(value, list) or not 1 <= len(value) <= 6:
            raise GuardrailViolation(f"Invalid generated list: {field}")
        if any(not isinstance(item, str) or not item.strip() or len(item) > 400 for item in value):
            raise GuardrailViolation(f"Invalid generated list item: {field}")

    text = _all_text(payload)
    if any(pattern.search(text) for pattern in FORBIDDEN_CLAIMS):
        raise GuardrailViolation("Generated text contains an unsupported or unsafe claim")

    allowed = {_normalise_number(value) for value in allowed_numbers}
    # Pollutant notation is a name, not a generated measurement. Remove it before
    # checking that every remaining number is grounded in the supplied context.
    number_text = re.sub(r"\bPM\s*2[.,]5\b", "PM", text, flags=re.IGNORECASE)
    generated = {_normalise_number(value) for value in NUMBER_PATTERN.findall(number_text)}
    unexpected = sorted(generated - allowed)
    if unexpected:
        raise GuardrailViolation(f"Generated text contains unsupported numbers: {unexpected}")

    return {
        "headline": payload["headline"].strip(),
        "summary": payload["summary"].strip(),
        "contributing_conditions": [item.strip() for item in payload["contributing_conditions"]],
        "sensitive_group_advice": payload["sensitive_group_advice"].strip(),
        "recommended_actions": [item.strip() for item in payload["recommended_actions"]],
        "uncertainty": payload["uncertainty"].strip(),
    }
