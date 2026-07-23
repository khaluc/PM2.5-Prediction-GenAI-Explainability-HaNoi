"""Tests for controlled PM2.5 forecast explanations."""

from __future__ import annotations

import json

import httpx

from src.genai.forecast_explainer import build_forecast_context, explain_forecast
from src.genai.dashscope_client import DashScopeClient, DashScopeConfig


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
    assert context["trend_assessment"]["code"] == "clear_rise"
    assert context["health_guidance"]["scope_vi"]
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
    knowledge = context["knowledge_graph"]
    assert [item["id"] for item in knowledge["supported_emission_sources"]] == [
        "vehicle_emission"
    ]
    assert {item["id"] for item in knowledge["unverified_emission_sources"]} == {
        "factory",
        "construction",
    }
    assert all(
        item["currently_observed"] for item in knowledge["meteorological_factors"]
    )


def test_low_confidence_traffic_is_not_used_as_contributing_condition() -> None:
    context = build_forecast_context(_prediction(), 3, traffic=_traffic(confidence=0.5))
    codes = {item["code"] for item in context["observed_conditions"]}
    assert "traffic_congestion" not in codes
    assert context["traffic"]["available"] is True


def test_explanation_falls_back_safely_without_dashscope_key() -> None:
    client = DashScopeClient(DashScopeConfig(api_key=""))
    result = explain_forecast(_prediction(), 3, client=client, traffic=_traffic())
    assert result["generation"]["mode"] == "deterministic_fallback"
    assert result["forecast"]["predicted_pm25"] == 72.0
    assert "không phải AQI chính thức" in result["explanation"]["uncertainty"]
    assert "có thể" in " ".join(result["explanation"]["contributing_conditions"])
    assert result["grounding"]["traffic"]["congestion_percent"] == 50.0
    assert result["grounding"]["knowledge_graph"]["graph_id"] == "urban-air-quality-domain-v2"
    assert "Nhóm nhạy cảm" in result["explanation"]["sensitive_group_advice"]
    actions = " ".join(result["explanation"]["recommended_actions"]).lower()
    assert "máy lọc không khí" not in actions
    assert "khẩu trang" not in actions


def test_valid_grounded_dashscope_json_is_used() -> None:
    generated = {
        "headline": "PM2.5 +3 giờ: 72.0 µg/m³",
        "summary": "Mô hình dự báo PM2.5 tăng từ 56.6 lên 72.0 µg/m³ sau 3 giờ.",
        "overall_interpretation": (
            "Mức tăng cần được theo dõi cùng gió, độ ẩm và số đo quan trắc tiếp theo; "
            "các điều kiện hiện tại chưa chứng minh nguyên nhân."
        ),
        "contributing_conditions": [
            "Gió yếu và độ ẩm cao có thể góp phần, nhưng chưa chứng minh nguyên nhân."
        ],
        "sensitive_group_advice": "Nhóm nhạy cảm nên giảm gắng sức kéo dài ngoài trời.",
        "recommended_actions": [
            "Ưu tiên ngay: theo dõi quan trắc tiếp theo.",
            "Xác minh khi cần: kiểm tra cảm biến nếu số đo lệch dự báo.",
            "Dài hạn: đánh giá biện pháp quy hoạch phù hợp địa điểm.",
        ],
        "uncertainty": "Đây là mức sàng lọc theo giờ, không phải AQI chính thức.",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "deepseek-v4-flash"
        assert body["max_tokens"] == 1400
        assert body["enable_thinking"] is False
        assert body["response_format"] == {"type": "json_object"}
        assert "knowledge_graph" in body["messages"][1]["content"]
        assert "unverified_emission_sources" in body["messages"][1]["content"]
        assert "overall_interpretation" in body["messages"][1]["content"]
        assert "recommended_actions phải có đúng ba mục" in body["messages"][1]["content"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(generated, ensure_ascii=False)}}]},
        )

    client = DashScopeClient(
        DashScopeConfig(api_key="test-key"),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = explain_forecast(_prediction(), 3, client=client)
    assert result["generation"] == {
        "mode": "dashscope",
        "provider_model": "deepseek-v4-flash",
        "fallback_reason": None,
    }
    assert result["explanation"]["headline"] == generated["headline"]
    assert result["explanation"]["overall_interpretation"] == generated[
        "overall_interpretation"
    ]


def test_guardrail_retries_once_with_a_repair_prompt() -> None:
    invalid = {
        "headline": "PM2.5 forecast",
        "summary": "Unsupported measurement: 999.",
        "overall_interpretation": "Observed conditions do not prove causality.",
        "contributing_conditions": ["Observed conditions require verification."],
        "sensitive_group_advice": "Sensitive groups should monitor updates.",
        "recommended_actions": [
            "Monitor the next observation.",
            "Verify the sensor when necessary.",
            "Assess long-term planning separately.",
        ],
        "uncertainty": "This is an hourly screening forecast.",
    }
    valid = {
        "headline": "PM2.5 forecast",
        "summary": "The forecast changes from 56.6 to 72.0 after 3 hours.",
        "overall_interpretation": (
            "Observed conditions may act in different directions, so the dominant "
            "factor cannot be identified from current evidence."
        ),
        "contributing_conditions": ["Observed conditions may contribute."],
        "sensitive_group_advice": "Sensitive groups should monitor updates.",
        "recommended_actions": [
            "Monitor the next observation.",
            "Verify the sensor when necessary.",
            "Assess long-term planning separately.",
        ],
        "uncertainty": "This is an hourly screening forecast.",
    }
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        payload = invalid if len(calls) == 1 else valid
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(payload, ensure_ascii=False)}}
                ]
            },
        )

    client = DashScopeClient(
        DashScopeConfig(api_key="test-key"),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = explain_forecast(_prediction(), 3, client=client)

    assert len(calls) == 2
    assert result["generation"]["mode"] == "dashscope"
    assert result["explanation"]["summary"] == valid["summary"]


def test_dashscope_may_use_one_decimal_rounding_of_grounded_values() -> None:
    prediction = _prediction()
    prediction["forecast_pm25"]["3h"] = 72.04
    generated = {
        "headline": "PM2.5 forecast: 72.0",
        "summary": "The forecast changes from 56.6 to 72.0 after 3 hours.",
        "overall_interpretation": (
            "Observed conditions may act in different directions and require "
            "confirmation from the next observation."
        ),
        "contributing_conditions": ["Observed conditions may contribute."],
        "sensitive_group_advice": "Sensitive groups should monitor updates.",
        "recommended_actions": [
            "Monitor the next observation.",
            "Verify the sensor when necessary.",
            "Assess long-term planning separately.",
        ],
        "uncertainty": "This is an hourly screening forecast.",
    }
    client = DashScopeClient(
        DashScopeConfig(api_key="test-key"),
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        generated,
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                )
            )
        ),
    )

    result = explain_forecast(prediction, 3, client=client)

    assert result["generation"]["mode"] == "dashscope"
    assert result["explanation"]["headline"] == generated["headline"]


def test_unsupported_dashscope_claim_or_number_triggers_fallback() -> None:
    generated = {
        "headline": "Dự báo PM2.5",
        "summary": "Nguyên nhân chính là nhà máy gây ô nhiễm 999 µg/m³.",
        "overall_interpretation": "Nhà máy chắc chắn gây ra mức ô nhiễm hiện tại.",
        "contributing_conditions": ["Chắc chắn do nguồn phát thải."],
        "sensitive_group_advice": "Ở trong nhà.",
        "recommended_actions": [
            "Theo dõi dữ liệu.",
            "Kiểm tra cảm biến.",
            "Đánh giá biện pháp dài hạn.",
        ],
        "uncertainty": "Không có.",
    }
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(generated, ensure_ascii=False)}}]},
        )
    )
    client = DashScopeClient(
        DashScopeConfig(api_key="test-key"),
        http_client=httpx.Client(transport=transport),
    )
    result = explain_forecast(_prediction(), 3, client=client)
    assert result["generation"]["mode"] == "deterministic_fallback"
    assert result["generation"]["fallback_reason"] == "provider_or_guardrail_failure"
    assert "nhà máy" not in result["explanation"]["summary"].lower()


def test_unverified_graph_source_mention_triggers_fallback() -> None:
    generated = {
        "headline": "Dự báo PM2.5",
        "summary": "Cần kiểm tra công trường gần khu vực quan trắc.",
        "overall_interpretation": (
            "Công trường được xem là nguồn cần xác minh dù chưa có bằng chứng hiện tại."
        ),
        "contributing_conditions": ["Gió yếu có thể làm giảm khuếch tán."],
        "sensitive_group_advice": "Nhóm nhạy cảm nên theo dõi cập nhật.",
        "recommended_actions": [
            "Theo dõi dữ liệu tiếp theo.",
            "Kiểm tra cảm biến khi cần.",
            "Đánh giá biện pháp dài hạn.",
        ],
        "uncertainty": "Đây là dự báo có sai số.",
    }
    client = DashScopeClient(
        DashScopeConfig(api_key="test-key"),
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "choices": [
                            {"message": {"content": json.dumps(generated, ensure_ascii=False)}}
                        ]
                    },
                )
            )
        ),
    )
    result = explain_forecast(_prediction(), 3, client=client)
    assert result["generation"]["mode"] == "deterministic_fallback"
    assert "công trường" not in result["explanation"]["summary"].lower()
