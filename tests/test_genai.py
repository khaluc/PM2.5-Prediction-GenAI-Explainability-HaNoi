"""Tests for controlled PM2.5 forecast explanations."""

from __future__ import annotations

import json

import httpx

from src.genai.forecast_explainer import build_forecast_context, explain_forecast
from src.genai.groq_client import GroqClient, GroqConfig


def _prediction() -> dict:
    return {
        "model": "LightGBM",
        "station_id": "HN_HOAN_KIEM",
        "timestamp": "2026-07-14T00:00:00+07:00",
        "forecast_pm25": {"1h": 66.0, "3h": 72.0, "6h": 75.0},
        "current_measurements": {
            "pm25": 56.6,
            "temperature": 28.0,
            "humidity": 89.0,
            "wind_speed": 8.0,
            "precipitation": 0.0,
        },
    }


def _traffic(*, confidence: float = 0.95) -> dict:
    return {
        "available": True,
        "observed_at": "2026-07-14T00:10:00+07:00",
        "current_speed_kmh": 20.0,
        "free_flow_speed_kmh": 40.0,
        "congestion_ratio": 0.5,
        "congestion_percent": 50.0,
        "confidence": confidence,
        "road_closure": False,
        "source": "tomtom_flow_segment",
        "spatial_scope": "nearest_road_segment_to_sampling_point",
        "causal_claim_allowed": False,
    }


def test_context_distinguishes_observed_conditions_from_causes() -> None:
    context = build_forecast_context(_prediction(), 3, traffic=_traffic())
    assert context["predicted_pm25"] == 72.0
    assert context["change_pm25"] == 15.4
    assert context["hourly_screening"]["screening_only"] is True
    assert context["constraints"]["causal_claim_allowed"] is False
    codes = {item["code"] for item in context["observed_conditions"]}
    assert {
        "forecast_rising",
        "low_wind",
        "high_humidity",
        "little_rain",
        "traffic_congestion",
    } <= codes
    assert context["traffic"]["source"] == "tomtom_flow_segment"


def test_low_confidence_traffic_is_not_used_as_contributing_condition() -> None:
    context = build_forecast_context(_prediction(), 3, traffic=_traffic(confidence=0.5))
    codes = {item["code"] for item in context["observed_conditions"]}
    assert "traffic_congestion" not in codes
    assert context["traffic"]["available"] is True


def test_explanation_falls_back_safely_without_groq_key() -> None:
    client = GroqClient(GroqConfig(api_key=""))
    result = explain_forecast(_prediction(), 3, client=client, traffic=_traffic())
    assert result["generation"]["mode"] == "deterministic_fallback"
    assert result["forecast"]["predicted_pm25"] == 72.0
    assert "không phải AQI chính thức" in result["explanation"]["uncertainty"]
    assert "có thể" in " ".join(result["explanation"]["contributing_conditions"])
    assert result["grounding"]["traffic"]["congestion_percent"] == 50.0


def test_valid_grounded_groq_json_is_used() -> None:
    generated = {
        "headline": "PM2.5 +3 giờ: 72.0 µg/m³",
        "summary": "Mô hình dự báo PM2.5 tăng từ 56.6 lên 72.0 µg/m³ sau 3 giờ.",
        "contributing_conditions": [
            "Gió yếu và độ ẩm cao có thể góp phần, nhưng chưa chứng minh nguyên nhân."
        ],
        "sensitive_group_advice": "Nhóm nhạy cảm nên giảm gắng sức kéo dài ngoài trời.",
        "recommended_actions": ["Theo dõi quan trắc tiếp theo và kiểm tra cảm biến."],
        "uncertainty": "Đây là mức sàng lọc theo giờ, không phải AQI chính thức.",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "openai/gpt-oss-120b"
        assert body["max_completion_tokens"] == 1400
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(generated, ensure_ascii=False)}}]},
        )

    client = GroqClient(
        GroqConfig(api_key="test-key"),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = explain_forecast(_prediction(), 3, client=client)
    assert result["generation"] == {
        "mode": "groq",
        "provider_model": "openai/gpt-oss-120b",
        "fallback_reason": None,
    }
    assert result["explanation"]["headline"] == generated["headline"]


def test_unsupported_groq_claim_or_number_triggers_fallback() -> None:
    generated = {
        "headline": "Dự báo PM2.5",
        "summary": "Nguyên nhân chính là nhà máy gây ô nhiễm 999 µg/m³.",
        "contributing_conditions": ["Chắc chắn do nguồn phát thải."],
        "sensitive_group_advice": "Ở trong nhà.",
        "recommended_actions": ["Theo dõi dữ liệu."],
        "uncertainty": "Không có.",
    }
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(generated, ensure_ascii=False)}}]},
        )
    )
    client = GroqClient(
        GroqConfig(api_key="test-key"),
        http_client=httpx.Client(transport=transport),
    )
    result = explain_forecast(_prediction(), 3, client=client)
    assert result["generation"]["mode"] == "deterministic_fallback"
    assert result["generation"]["fallback_reason"] == "provider_or_guardrail_failure"
    assert "nhà máy" not in result["explanation"]["summary"].lower()
