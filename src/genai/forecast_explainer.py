"""Grounded, controlled explanations of internal PM2.5 ML forecasts."""

from __future__ import annotations

import json
import re
from typing import Any

from src.assessment.who_pm25 import classify_hourly_forecast_proxy
from src.genai.dashscope_client import DashScopeClient, DashScopeClientError
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


def _trend_assessment(delta: float) -> dict[str, str]:
    if delta >= 5:
        return {
            "code": "clear_rise",
            "label_vi": "xu hướng tăng rõ",
            "interpretation_vi": (
                "Mức dự báo tăng đủ lớn để ưu tiên theo dõi lần quan trắc tiếp theo."
            ),
        }
    if delta <= -5:
        return {
            "code": "clear_fall",
            "label_vi": "xu hướng giảm rõ",
            "interpretation_vi": (
                "Mô hình cho thấy mức giảm đáng chú ý nhưng vẫn cần quan trắc xác nhận."
            ),
        }
    if abs(delta) < 1:
        return {
            "code": "stable",
            "label_vi": "gần như ổn định",
            "interpretation_vi": (
                "Chênh lệch dự báo nhỏ nên chưa cho thấy biến động rõ rệt trong chân trời đã chọn."
            ),
        }
    return {
        "code": "small_change",
        "label_vi": "thay đổi nhẹ",
        "interpretation_vi": (
            "Mô hình dự báo có thay đổi nhưng biên độ chưa đủ để xem là một xu hướng mạnh."
        ),
    }


def _screening_health_guidance(level_code: int) -> dict[str, str]:
    if level_code <= 1:
        advice = (
            "Nhóm nhạy cảm có thể duy trì sinh hoạt bình thường, đồng thời theo dõi "
            "cập nhật và phản ứng sức khỏe cá nhân."
        )
    elif level_code <= 3:
        advice = (
            "Nhóm nhạy cảm nên giảm gắng sức kéo dài ngoài trời nếu xuất hiện khó chịu "
            "và theo dõi các cập nhật tiếp theo."
        )
    else:
        advice = (
            "Nhóm nhạy cảm nên giảm hoạt động kéo dài hoặc gắng sức ngoài trời và "
            "theo dõi hướng dẫn chính thức."
        )
    return {
        "advice_vi": advice,
        "scope_vi": (
            "Đây là khuyến nghị thận trọng theo mức sàng lọc của dự án, không phải "
            "chẩn đoán y tế hoặc hướng dẫn AQI chính thức."
        ),
    }


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


def _fallback_overall_interpretation(context: dict[str, Any]) -> str:
    trend = context["trend_assessment"]["interpretation_vi"]
    codes = {
        item.get("code")
        for item in context.get("observed_conditions", [])
        if isinstance(item, dict)
    }
    accumulation_signals = codes & {
        "low_wind",
        "high_humidity",
        "traffic_congestion",
    }
    removal_signals = codes & {"rain_present", "dispersive_wind"}
    if accumulation_signals and removal_signals:
        balance = (
            "Các điều kiện quan sát đang cho tín hiệu theo nhiều hướng: một số điều kiện "
            "có thể hạn chế khuếch tán hoặc đi kèm phát thải, trong khi mưa hoặc gió có "
            "thể hỗ trợ làm sạch hay phân tán tại chỗ. Dữ liệu hiện có chưa định lượng "
            "được yếu tố nào chiếm ưu thế."
        )
    elif accumulation_signals:
        balance = (
            "Một số điều kiện quan sát có thể đi kèm khả năng tích tụ hoặc phát thải cao "
            "hơn, nhưng chưa đủ bằng chứng để xác định nguyên nhân hay mức đóng góp."
        )
    elif removal_signals:
        balance = (
            "Mưa hoặc gió có thể hỗ trợ loại bỏ hay khuếch tán hạt tại chỗ, nhưng hiệu "
            "quả cần được xác nhận bằng số đo PM2.5 ở những giờ tiếp theo."
        )
    else:
        balance = (
            "Chưa có tín hiệu quan trắc nổi bật để giải thích biến động; cần tiếp tục "
            "đối chiếu dự báo với số đo mới."
        )
    return f"{trend} {balance}"


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
        "trend_assessment": _trend_assessment(delta),
        "hourly_screening": {
            "level_code": level_code,
            "label_vi": screening.get("project_label_vi"),
            "who_band_vi": screening.get("who_band_vi"),
            "screening_only": True,
            "note_vi": screening.get("note_vi"),
        },
        "health_guidance": _screening_health_guidance(level_code),
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
    conditions = [
        f'{item["evidence"]} {item["interpretation"]}'
        for item in context["observed_conditions"]
    ]
    if not conditions:
        conditions = ["Chưa có đủ điều kiện quan trắc nổi bật để nêu giả thuyết đóng góp."]
    health_guidance = context["health_guidance"]
    advice = f'{health_guidance["advice_vi"]} {health_guidance["scope_vi"]}'
    actions = [
        (
            "Ưu tiên ngay: theo dõi số đo quan trắc ở giờ tiếp theo và so sánh với dự báo "
            "để biết xu hướng có tiếp tục hay không."
        ),
        (
            "Xác minh khi cần: kiểm tra chất lượng cảm biến và điều kiện hiện trường nếu "
            "số đo mới lệch đáng kể với dự báo hoặc xuất hiện cảnh báo bất thường."
        ),
        (
            "Dài hạn: chỉ xem các biện pháp trong Knowledge Graph là định hướng quy hoạch; "
            "cần đánh giá theo địa điểm vì chúng không xử lý tức thời một giờ ô nhiễm."
        ),
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
            f"Giá trị này thuộc mức sàng lọc “{label}”. "
            f'{context["trend_assessment"]["interpretation_vi"]}'
        ),
        "overall_interpretation": _fallback_overall_interpretation(context),
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
        "headline": "một câu ngắn nêu xu hướng và giá trị dự báo",
        "summary": (
            "ba đến bốn câu: mô tả giá trị hiện tại, dự báo, mức thay đổi, ý nghĩa "
            "của trend_assessment và mức sàng lọc; không lặp nguyên văn headline"
        ),
        "overall_interpretation": (
            "hai đến ba câu tổng hợp các điều kiện đang cùng chiều hoặc ngược chiều, "
            "nêu điều có thể giải thích và điều chưa thể kết luận; không chỉ liệt kê lại dữ liệu"
        ),
        "contributing_conditions": [
            (
                "mỗi phần tử gồm bằng chứng quan sát, cơ chế hoặc hướng ảnh hưởng có thể "
                "xảy ra, và giới hạn suy luận; không gọi là nguyên nhân"
            )
        ],
        "sensitive_group_advice": (
            "hai câu, bám sát health_guidance và không khuyến cáo nghiêm ngặt hơn dữ liệu cho phép"
        ),
        "recommended_actions": [
            "Ưu tiên ngay: hành động theo dõi kèm mục đích",
            "Xác minh khi cần: điều kiện kích hoạt việc kiểm tra cảm biến hoặc hiện trường",
            "Dài hạn: biện pháp phù hợp từ Knowledge Graph và giới hạn hiệu quả tức thời",
        ],
        "uncertainty": (
            "hai đến ba câu về sai số ML, phạm vi dữ liệu TomTom, giới hạn nhân quả "
            "và việc đây không phải AQI chính thức"
        ),
    }
    return (
        "Hãy tạo JSON giải thích dự báo theo đúng schema sau. Mọi số trong câu trả lời "
        "phải có trong dữ liệu. Diễn giải mạch lạc và cụ thể cho trường hợp hiện tại, "
        "không chỉ đổi cách viết hoặc liệt kê lại DATA_JSON. Phải giải thích vì sao "
        "mức thay đổi được xem là ổn định, nhẹ hoặc rõ dựa trên trend_assessment. "
        "Khi có cả điều kiện hỗ trợ tích tụ và điều kiện hỗ trợ loại bỏ hoặc khuếch tán, "
        "hãy nói rõ chúng có thể tác động theo các hướng khác nhau và chưa biết yếu tố "
        "nào chiếm ưu thế. recommended_actions phải có đúng ba mục theo thứ tự ưu tiên "
        "ngay, xác minh khi cần và dài hạn.\n"
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
    client: DashScopeClient | None = None,
    traffic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Explain one forecast with DeepSeek on DashScope and a safe fallback."""
    context = build_forecast_context(
        prediction,
        horizon_hours,
        traffic=traffic,
    )
    fallback = _fallback_explanation(context)
    generation_mode = "deterministic_fallback"
    provider_model = None
    fallback_reason = "llm_disabled" if not use_llm else "dashscope_not_configured"

    dashscope = client or DashScopeClient()
    if use_llm and dashscope.available:
        try:
            generated = dashscope.generate_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=_prompt(context),
            )
            grounded_numbers = _grounded_numbers(context)
            allowed_numbers = [
                2.5,
                24,
                *grounded_numbers,
                *(round(number, 1) for number in grounded_numbers),
            ]
            forbidden_terms = unsupported_emission_source_terms(
                context["knowledge_graph"]
            )
            try:
                explanation = validate_generated_explanation(
                    generated,
                    allowed_numbers=allowed_numbers,
                    forbidden_terms=forbidden_terms,
                )
            except GuardrailViolation as error:
                repair_prompt = (
                    f"{_prompt(context)}\n\n"
                    "Lần trả lời trước bị bộ kiểm soát từ chối với lý do: "
                    f"{error}. Hãy tạo lại JSON đúng schema, không thêm dữ kiện, "
                    "số liệu, nguồn phát thải hoặc khẳng định nhân quả ngoài "
                    "context."
                )
                generated = dashscope.generate_json(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=repair_prompt,
                )
                explanation = validate_generated_explanation(
                    generated,
                    allowed_numbers=allowed_numbers,
                    forbidden_terms=forbidden_terms,
                )
            generation_mode = "dashscope"
            provider_model = dashscope.config.model
            fallback_reason = None
        except (DashScopeClientError, GuardrailViolation, ValueError, TypeError):
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
