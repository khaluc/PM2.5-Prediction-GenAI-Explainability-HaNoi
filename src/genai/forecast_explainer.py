"""Grounded, controlled explanations of internal PM2.5 ML forecasts."""

from __future__ import annotations

import json
import re
from typing import Any

from src.assessment.who_pm25 import classify_hourly_forecast_proxy
from src.genai.groq_client import GroqClient, GroqClientError
from src.genai.guardrails import GuardrailViolation, validate_generated_explanation
from src.genai.knowledge_graph import (
    build_pm25_knowledge_context,
    unsupported_emission_source_terms,
)


SYSTEM_PROMPT = """Bạn là trợ lý giải thích dự báo PM2.5 cho hệ thống hỗ trợ quyết định.
Chỉ sử dụng JSON dữ liệu được cung cấp. Không tự tạo số liệu, nguồn phát thải hoặc tổ chức vi phạm.
Phân biệt dữ kiện với suy luận. Không khẳng định quan hệ nhân quả; chỉ nói điều kiện "có thể góp phần".
Chỉ sử dụng giao thông TomTom khi traffic.available=true và confidence đủ cao. Dữ liệu này chỉ đại diện đoạn đường gần điểm lấy mẫu, không đại diện toàn quận hoặc toàn thành phố.
Knowledge Graph chỉ cung cấp kiến thức miền chung. Cạnh EMITS mô tả loại chất có thể được nguồn phát thải phát ra, không chứng minh nguồn gây ra sự kiện hiện tại.
Chỉ nhắc nguồn phát thải khi nguồn đó nằm trong supported_emission_sources; không nhắc các nguồn trong unverified_emission_sources.
Cạnh INFLUENCED_BY mô tả ảnh hưởng khí tượng, không phải kết luận nhân quả cho một giờ cụ thể. MITIGATED_BY là biện pháp quy hoạch hoặc chính sách, không phải hành động tự động và không bảo đảm hiệu quả tức thời.
Không gọi dự báo PM2.5 theo giờ là AQI chính thức hoặc kết luận tuân thủ WHO.
Nếu thiếu bằng chứng, phải nói rõ và đề xuất xác minh dữ liệu/cảm biến/hiện trường.
Không tự động đề xuất hành động nguy hiểm. Khuyến nghị sức khỏe chỉ mang tính thận trọng chung.
Trả về duy nhất một JSON object hợp lệ, không Markdown."""

CONTEXT_NUMBER_PATTERN = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?")


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _condition_facts(measurements: dict[str, Any], delta: float) -> list[dict[str, str]]:
    facts: list[dict[str, str]] = []
    if delta >= 5:
        facts.append(
            {
                "code": "forecast_rising",
                "evidence": f"PM2.5 dự báo tăng {delta:.1f} µg/m³ so với hiện tại.",
                "interpretation": "Xu hướng tăng cần được theo dõi và đối chiếu ở lần quan trắc tiếp theo.",
            }
        )
    elif delta <= -5:
        facts.append(
            {
                "code": "forecast_falling",
                "evidence": f"PM2.5 dự báo giảm {abs(delta):.1f} µg/m³ so với hiện tại.",
                "interpretation": "Mô hình cho thấy xu hướng giảm, nhưng vẫn cần xác nhận bằng quan trắc.",
            }
        )

    wind = _number(measurements.get("wind_speed"))
    if wind is not None and wind <= 10:
        facts.append(
            {
                "code": "low_wind",
                "evidence": f"Tốc độ gió hiện tại {wind:.1f} km/h.",
                "interpretation": "Gió yếu có thể làm giảm khả năng khuếch tán chất ô nhiễm; đây chưa phải bằng chứng nguyên nhân.",
            }
        )
    elif wind is not None and wind >= 15:
        facts.append(
            {
                "code": "dispersive_wind",
                "evidence": f"Tốc độ gió hiện tại {wind:.1f} km/h.",
                "interpretation": (
                    "Gió ở mức cao hơn có thể hỗ trợ khuếch tán tại chỗ, nhưng cũng có thể vận chuyển "
                    "ô nhiễm từ nơi khác nên chưa đủ để dự đoán PM2.5 sẽ giảm."
                ),
            }
        )
    humidity = _number(measurements.get("humidity"))
    if humidity is not None and humidity >= 80:
        facts.append(
            {
                "code": "high_humidity",
                "evidence": f"Độ ẩm hiện tại {humidity:.1f}%.",
                "interpretation": "Độ ẩm cao là điều kiện đi kèm cần xem xét, không đủ để xác nhận nguồn ô nhiễm.",
            }
        )
    precipitation = _number(measurements.get("precipitation"))
    if precipitation is not None and precipitation <= 0.1:
        facts.append(
            {
                "code": "little_rain",
                "evidence": f"Lượng mưa hiện tại {precipitation:.1f} mm.",
                "interpretation": "Ít hoặc không mưa đồng nghĩa không có điều kiện rửa trôi rõ rệt; chưa chứng minh quan hệ nhân quả.",
            }
        )
    elif precipitation is not None:
        facts.append(
            {
                "code": "rain_present",
                "evidence": f"Lượng mưa hiện tại {precipitation:.1f} mm.",
                "interpretation": (
                    "Mưa có thể hỗ trợ loại bỏ hạt khỏi không khí; hiệu quả thực tế còn phụ thuộc "
                    "cường độ, thời gian mưa và vận chuyển khí quyển."
                ),
            }
        )
    return facts


def _traffic_condition_facts(traffic: dict[str, Any] | None) -> list[dict[str, str]]:
    """Create bounded traffic evidence only when TomTom quality is adequate."""
    if not isinstance(traffic, dict) or traffic.get("available") is not True:
        return []
    confidence = _number(traffic.get("confidence"))
    congestion = _number(traffic.get("congestion_ratio"))
    current_speed = _number(traffic.get("current_speed_kmh"))
    free_flow_speed = _number(traffic.get("free_flow_speed_kmh"))
    if confidence is None or confidence < 0.7:
        return []

    facts: list[dict[str, str]] = []
    if (
        congestion is not None
        and congestion >= 0.25
        and current_speed is not None
        and free_flow_speed is not None
    ):
        facts.append(
            {
                "code": "traffic_congestion",
                "evidence": (
                    f"TomTom ghi nhận tốc độ {current_speed:.1f} km/h so với mức thông thoáng "
                    f"{free_flow_speed:.1f} km/h, độ ùn tắc {congestion * 100:.1f}% và độ tin cậy {confidence:.2f}."
                ),
                "interpretation": (
                    "Ùn tắc trên đoạn đường gần điểm lấy mẫu có thể đi kèm phát thải giao thông cao hơn; "
                    "dữ liệu này không chứng minh giao thông là nguyên nhân làm PM2.5 tăng."
                ),
            }
        )
    if traffic.get("road_closure") is True:
        facts.append(
            {
                "code": "nearby_road_closure",
                "evidence": "TomTom đánh dấu đóng đường tại đoạn gần điểm lấy mẫu.",
                "interpretation": (
                    "Đóng đường có thể làm thay đổi phân bố dòng xe xung quanh; cần xác minh hiện trường "
                    "trước khi liên hệ với biến động PM2.5."
                ),
            }
        )
    return facts


def build_forecast_context(
    prediction: dict[str, Any],
    horizon_hours: int,
    *,
    traffic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if horizon_hours not in {1, 3, 6}:
        raise ValueError("horizon_hours must be one of 1, 3 or 6")
    horizon_key = f"{horizon_hours}h"
    predicted_pm25 = _number((prediction.get("forecast_pm25") or {}).get(horizon_key))
    if predicted_pm25 is None:
        raise ValueError(f"ML forecast is unavailable for +{horizon_hours}h")
    measurements = dict(prediction.get("current_measurements") or {})
    current_pm25 = _number(measurements.get("pm25"))
    if current_pm25 is None:
        raise ValueError("Current PM2.5 measurement is unavailable")
    delta = predicted_pm25 - current_pm25
    screening = (prediction.get("forecast_screening_levels") or {}).get(horizon_key)
    if not isinstance(screening, dict):
        screening = classify_hourly_forecast_proxy(predicted_pm25)
    level_code = screening.get("level_code")
    if not isinstance(level_code, int):
        level_code = int(classify_hourly_forecast_proxy(predicted_pm25)["level_code"])
    conditions = [
        *_condition_facts(measurements, delta),
        *_traffic_condition_facts(traffic),
    ]
    knowledge_codes = {item["code"] for item in conditions}
    for field, evidence_code in {
        "wind_speed": "wind_observed",
        "humidity": "humidity_observed",
        "precipitation": "rain_observed",
        "temperature": "temperature_observed",
    }.items():
        if _number(measurements.get(field)) is not None:
            knowledge_codes.add(evidence_code)
    knowledge_graph = build_pm25_knowledge_context(
        knowledge_codes,
        screening_level_code=level_code,
    )

    supported_analysis = None
    cause_analysis = prediction.get("cause_analysis")
    if isinstance(cause_analysis, dict) and cause_analysis.get("available") is True:
        supported_analysis = {
            "hypothesis": cause_analysis.get("top_hypothesis_vi"),
            "evidence_strength": cause_analysis.get("evidence_strength"),
            "evidence": cause_analysis.get("evidence") or [],
            "limitations": cause_analysis.get("limitations") or [],
            "causal_claim_allowed": False,
        }

    return {
        "station_id": prediction.get("station_id"),
        "observation_timestamp": str(prediction.get("timestamp") or ""),
        "ml_model": prediction.get("model"),
        "horizon_hours": horizon_hours,
        "current_pm25": round(current_pm25, 2),
        "predicted_pm25": round(predicted_pm25, 2),
        "change_pm25": round(delta, 2),
        "unit": "µg/m³",
        "hourly_screening": {
            "level_code": level_code,
            "label_vi": screening.get("project_label_vi"),
            "who_band_vi": screening.get("who_band_vi"),
            "screening_only": True,
            "note_vi": screening.get("note_vi"),
        },
        "weather": {
            key: measurements.get(key)
            for key in ("temperature", "humidity", "wind_speed", "precipitation")
            if measurements.get(key) is not None
        },
        "traffic": traffic
        if isinstance(traffic, dict)
        else {
            "available": False,
            "reason": "traffic_context_not_provided",
            "source": "tomtom_flow_segment",
        },
        "observed_conditions": conditions,
        "supported_cause_analysis": supported_analysis,
        "knowledge_graph": knowledge_graph,
        "constraints": {
            "causal_claim_allowed": False,
            "official_aqi_claim_allowed": False,
            "organisation_attribution_allowed": False,
        },
    }


def _fallback_explanation(context: dict[str, Any]) -> dict[str, Any]:
    current = float(context["current_pm25"])
    predicted = float(context["predicted_pm25"])
    delta = float(context["change_pm25"])
    horizon = int(context["horizon_hours"])
    label = context["hourly_screening"].get("label_vi") or "chưa phân loại"
    if delta >= 5:
        trend = "tăng"
    elif delta <= -5:
        trend = "giảm"
    else:
        trend = "ít thay đổi"
    conditions = [item["interpretation"] for item in context["observed_conditions"]]
    if not conditions:
        conditions = ["Chưa có đủ điều kiện quan trắc nổi bật để nêu giả thuyết đóng góp."]
    level_code = int(context["hourly_screening"]["level_code"])
    if level_code <= 1:
        advice = "Nhóm nhạy cảm có thể sinh hoạt bình thường nhưng nên theo dõi cập nhật và triệu chứng cá nhân."
    elif level_code <= 3:
        advice = "Nhóm nhạy cảm nên giảm gắng sức kéo dài ngoài trời nếu xuất hiện khó chịu."
    else:
        advice = "Nhóm nhạy cảm nên giảm hoạt động kéo dài hoặc gắng sức ngoài trời và theo dõi hướng dẫn chính thức."
    actions = [
        "Theo dõi số đo quan trắc ở giờ tiếp theo và so sánh với dự báo.",
        "Kiểm tra chất lượng dữ liệu cảm biến trước khi đưa ra quyết định vận hành.",
        "Xác minh tại hiện trường nếu mức PM2.5 tiếp tục tăng hoặc xuất hiện cảnh báo bất thường.",
    ]
    traffic_available = context.get("traffic", {}).get("available") is True
    traffic_limit = (
        " Dữ liệu TomTom chỉ phản ánh đoạn đường gần điểm lấy mẫu, không đại diện toàn khu vực."
        if traffic_available
        else " Dữ liệu giao thông TomTom phù hợp thời điểm hiện chưa khả dụng."
    )
    return {
        "headline": f"PM2.5 dự báo +{horizon} giờ: {predicted:.1f} µg/m³",
        "summary": (
            f"Mô hình dự báo PM2.5 {trend} từ {current:.1f} lên {predicted:.1f} µg/m³ sau {horizon} giờ. "
            f"Giá trị này thuộc mức sàng lọc “{label}”."
        ),
        "contributing_conditions": conditions,
        "sensitive_group_advice": advice,
        "recommended_actions": actions,
        "uncertainty": (
            "Đây là dự báo ML và mức sàng lọc PM2.5 theo giờ, không phải AQI chính thức hoặc đánh giá tuân thủ WHO; "
            "các điều kiện nêu trên và Knowledge Graph không chứng minh nguyên nhân gây ô nhiễm hoặc chẩn đoán sức khỏe."
            + traffic_limit
        ),
    }


def _prompt(context: dict[str, Any]) -> str:
    schema = {
        "headline": "tiêu đề ngắn",
        "summary": "giải thích dự báo, có giá trị hiện tại, dự báo và chân trời",
        "contributing_conditions": ["các điều kiện có bằng chứng; không gọi là nguyên nhân"],
        "sensitive_group_advice": "khuyến nghị sức khỏe thận trọng ở mức chung, không chẩn đoán",
        "recommended_actions": ["hành động theo dõi và xác minh an toàn; không biến biện pháp quy hoạch MITIGATED_BY thành hành động tự động"],
        "uncertainty": "giới hạn của ML, mức theo giờ và suy luận",
    }
    return (
        "Hãy tạo JSON giải thích dự báo theo đúng schema sau. Mọi số trong câu trả lời phải có trong dữ liệu.\n"
        f"SCHEMA_JSON={json.dumps(schema, ensure_ascii=False)}\n"
        f"DATA_JSON={json.dumps(context, ensure_ascii=False, default=str)}"
    )


def _grounded_numbers(value: Any) -> list[float]:
    """Collect every number already present in structured or textual context."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _grounded_numbers(item)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _grounded_numbers(item)]
    if isinstance(value, str):
        text = re.sub(r"\bPM\s*2[.,]5\b", "PM", value, flags=re.IGNORECASE)
        return [float(match.replace(",", ".")) for match in CONTEXT_NUMBER_PATTERN.findall(text)]
    return []


def explain_forecast(
    prediction: dict[str, Any],
    horizon_hours: int,
    *,
    use_llm: bool = True,
    client: GroqClient | None = None,
    traffic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Explain one forecast with Groq, falling back to deterministic safe text."""
    context = build_forecast_context(
        prediction,
        horizon_hours,
        traffic=traffic,
    )
    fallback = _fallback_explanation(context)
    generation_mode = "deterministic_fallback"
    provider_model = None
    fallback_reason = "llm_disabled" if not use_llm else "groq_not_configured"

    groq = client or GroqClient()
    if use_llm and groq.available:
        try:
            generated = groq.generate_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=_prompt(context),
            )
            allowed_numbers = [2.5, 24, *_grounded_numbers(context)]
            explanation = validate_generated_explanation(
                generated,
                allowed_numbers=allowed_numbers,
                forbidden_terms=unsupported_emission_source_terms(context["knowledge_graph"]),
            )
            generation_mode = "groq"
            provider_model = groq.config.model
            fallback_reason = None
        except (GroqClientError, GuardrailViolation, ValueError, TypeError):
            explanation = fallback
            fallback_reason = "provider_or_guardrail_failure"
    else:
        explanation = fallback

    return {
        "station_id": context["station_id"],
        "observation_timestamp": context["observation_timestamp"],
        "forecast": {
            "model": context["ml_model"],
            "horizon_hours": context["horizon_hours"],
            "current_pm25": context["current_pm25"],
            "predicted_pm25": context["predicted_pm25"],
            "change_pm25": context["change_pm25"],
            "unit": context["unit"],
            "screening": context["hourly_screening"],
        },
        "explanation": explanation,
        "grounding": {
            "observed_conditions": context["observed_conditions"],
            "traffic": context["traffic"],
            "supported_cause_analysis": context["supported_cause_analysis"],
            "knowledge_graph": context["knowledge_graph"],
            "causal_claim_allowed": False,
            "official_aqi_claim_allowed": False,
        },
        "generation": {
            "mode": generation_mode,
            "provider_model": provider_model,
            "fallback_reason": fallback_reason,
        },
    }
