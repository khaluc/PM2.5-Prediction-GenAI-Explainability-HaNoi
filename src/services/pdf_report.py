"""Polished A4 PDF rendering for deterministic environmental reports."""

from __future__ import annotations

import html
import os
import threading
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.graphics.shapes import Circle, Drawing, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FONT_LOCK = threading.Lock()
FONT_REGULAR = "EnvironmentAI-Regular"
FONT_BOLD = "EnvironmentAI-Bold"

BRAND = colors.HexColor("#1D6F68")
BRAND_DARK = colors.HexColor("#114F4A")
BRAND_SOFT = colors.HexColor("#E8F3F1")
INK = colors.HexColor("#17222E")
MUTED = colors.HexColor("#667483")
LINE = colors.HexColor("#DDE5E7")
CANVAS = colors.HexColor("#F4F7F6")
WARNING_SOFT = colors.HexColor("#FFF4D6")


def _font_candidates() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    configured_regular = os.getenv("PDF_FONT_REGULAR")
    configured_bold = os.getenv("PDF_FONT_BOLD")
    if configured_regular and configured_bold:
        pairs.append((Path(configured_regular), Path(configured_bold)))
    pairs.extend(
        [
            (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
            (Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/segoeuib.ttf")),
            (
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            ),
            (
                Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
                Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
            ),
        ]
    )
    return pairs


def register_pdf_fonts() -> tuple[str, str]:
    """Register a Unicode font family that supports Vietnamese glyphs."""

    with FONT_LOCK:
        if FONT_REGULAR in pdfmetrics.getRegisteredFontNames():
            return FONT_REGULAR, FONT_BOLD
        for regular, bold in _font_candidates():
            if regular.is_file() and bold.is_file():
                pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(regular)))
                pdfmetrics.registerFont(TTFont(FONT_BOLD, str(bold)))
                pdfmetrics.registerFontFamily(
                    "EnvironmentAI",
                    normal=FONT_REGULAR,
                    bold=FONT_BOLD,
                    italic=FONT_REGULAR,
                    boldItalic=FONT_BOLD,
                )
                return FONT_REGULAR, FONT_BOLD
    raise RuntimeError(
        "Không tìm thấy font Unicode để tạo PDF. Hãy cấu hình PDF_FONT_REGULAR và PDF_FONT_BOLD."
    )


def _escape(value: Any) -> str:
    if value is None or value == "":
        return "Chưa có dữ liệu"
    return html.escape(str(value))


def _number(value: Any, digits: int = 1, suffix: str = "") -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "Chưa có dữ liệu"
    rendered = f"{numeric:,.{digits}f}".replace(",", " ")
    return f"{rendered}{suffix}"


def _timestamp(value: Any) -> str:
    if value is None:
        return "Chưa có dữ liệu"
    try:
        parsed = pd.Timestamp(value)
        return parsed.strftime("%d/%m/%Y %H:%M")
    except (TypeError, ValueError):
        return str(value)


def _styles(regular: str, bold: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName=bold,
            fontSize=21,
            leading=25,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=3 * mm,
        ),
        "station": ParagraphStyle(
            "Station",
            parent=base["Heading2"],
            fontName=bold,
            fontSize=13,
            leading=17,
            textColor=BRAND_DARK,
            spaceAfter=1.5 * mm,
        ),
        "meta": ParagraphStyle(
            "Meta",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=8.5,
            leading=12,
            textColor=MUTED,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=base["Heading2"],
            fontName=bold,
            fontSize=12,
            leading=15,
            textColor=INK,
            spaceBefore=5 * mm,
            spaceAfter=2.5 * mm,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=8.5,
            leading=12.5,
            textColor=INK,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=7.5,
            leading=10.5,
            textColor=MUTED,
        ),
        "card_label": ParagraphStyle(
            "CardLabel",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=7,
            leading=9,
            textColor=MUTED,
        ),
        "card_value": ParagraphStyle(
            "CardValue",
            parent=base["BodyText"],
            fontName=bold,
            fontSize=13,
            leading=16,
            textColor=BRAND_DARK,
        ),
        "notice": ParagraphStyle(
            "Notice",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=8,
            leading=12,
            textColor=colors.HexColor("#5F4C16"),
        ),
    }


def _metric_card(label: str, value: str, styles: dict[str, ParagraphStyle]) -> list[Paragraph]:
    return [
        Paragraph(_escape(label).upper(), styles["card_label"]),
        Paragraph(_escape(value), styles["card_value"]),
    ]


def _summary_cards(report: dict[str, Any], styles: dict[str, ParagraphStyle]) -> Table:
    metrics = report["metrics"]
    cards = [
        _metric_card("PM2.5 trung bình", _number(metrics["pm25_mean"], suffix=" µg/m³"), styles),
        _metric_card("PM2.5 cao nhất", _number(metrics["pm25_max"], suffix=" µg/m³"), styles),
        _metric_card("Độ ẩm trung bình", _number(metrics["humidity_mean"], suffix="%"), styles),
        _metric_card("Số quan trắc", str(report["period"]["observation_count"]), styles),
    ]
    table = Table([cards], colWidths=[(A4[0] - 34 * mm) / 4] * 4, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), BRAND_SOFT),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#CFE1DE")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CFE1DE")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return table


def _trend_chart(frame: pd.DataFrame, regular: str, bold: str) -> Drawing:
    width, height = 500, 145
    left, right, bottom, top = 42, 12, 25, 14
    chart_w, chart_h = width - left - right, height - bottom - top
    drawing = Drawing(width, height)
    drawing.add(Rect(0, 0, width, height, rx=8, ry=8, fillColor=CANVAS, strokeColor=LINE))

    series = frame[["timestamp", "pm25"]].copy()
    series["pm25"] = pd.to_numeric(series["pm25"], errors="coerce")
    series = series.dropna(subset=["pm25"])
    if len(series) > 48:
        indexes = [round(index) for index in pd.Series(range(len(series))).iloc[[round(i * (len(series) - 1) / 47) for i in range(48)]]]
        series = series.iloc[indexes]
    if series.empty:
        drawing.add(String(width / 2, height / 2, "Chưa đủ dữ liệu PM2.5", textAnchor="middle", fontName=regular, fontSize=9, fillColor=MUTED))
        return drawing

    maximum = max(15.0, float(series["pm25"].max()) * 1.12)
    for index in range(4):
        fraction = index / 3
        y = bottom + fraction * chart_h
        drawing.add(PolyLine([(left, y), (width - right, y)], strokeColor=LINE, strokeWidth=0.55))
        drawing.add(String(left - 7, y - 2.5, f"{maximum * fraction:.0f}", textAnchor="end", fontName=regular, fontSize=6.5, fillColor=MUTED))

    values = series["pm25"].tolist()
    denominator = max(1, len(values) - 1)
    points = [
        (left + index / denominator * chart_w, bottom + float(value) / maximum * chart_h)
        for index, value in enumerate(values)
    ]
    drawing.add(PolyLine(points, strokeColor=BRAND, strokeWidth=2.1, strokeLineJoin=1))
    for x, y in points[:: max(1, len(points) // 12)]:
        drawing.add(Circle(x, y, 2.2, fillColor=colors.white, strokeColor=BRAND, strokeWidth=1.1))
    peak_index = max(range(len(values)), key=lambda index: values[index])
    peak_x, peak_y = points[peak_index]
    drawing.add(Circle(peak_x, peak_y, 3.1, fillColor=colors.HexColor("#DF594E"), strokeColor=colors.white, strokeWidth=1))
    drawing.add(String(peak_x, min(height - 9, peak_y + 8), f"{values[peak_index]:.1f}", textAnchor="middle", fontName=bold, fontSize=6.5, fillColor=colors.HexColor("#B43F37")))

    first_time = pd.Timestamp(series["timestamp"].iloc[0]).strftime("%d/%m %H:%M")
    last_time = pd.Timestamp(series["timestamp"].iloc[-1]).strftime("%d/%m %H:%M")
    drawing.add(String(left, 8, first_time, fontName=regular, fontSize=6.5, fillColor=MUTED))
    drawing.add(String(width - right, 8, last_time, textAnchor="end", fontName=regular, fontSize=6.5, fillColor=MUTED))
    drawing.add(String(left, height - 10, "PM2.5 (µg/m³)", fontName=bold, fontSize=7, fillColor=BRAND_DARK))
    return drawing


def _data_table(
    rows: list[tuple[str, str]],
    styles: dict[str, ParagraphStyle],
    *,
    widths: tuple[float, float] = (58 * mm, 118 * mm),
) -> Table:
    data = [
        [Paragraph(_escape(label), styles["body"]), Paragraph(_escape(value), styles["body"])]
        for label, value in rows
    ]
    table = Table(data, colWidths=list(widths), repeatRows=0, hAlign="LEFT")
    commands: list[tuple] = [
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.45, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("BACKGROUND", (0, 0), (0, -1), CANVAS),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
    ]
    table.setStyle(TableStyle(commands))
    return table


def _draw_page(canvas, document, regular: str, bold: str, report_id: str) -> None:
    canvas.saveState()
    page_width, page_height = A4
    canvas.setFillColor(BRAND_DARK)
    canvas.rect(0, page_height - 17 * mm, page_width, 17 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.circle(15 * mm, page_height - 8.5 * mm, 4.2 * mm, fill=1, stroke=0)
    canvas.setFillColor(BRAND)
    canvas.setFont(bold, 11)
    canvas.drawString(12.3 * mm, page_height - 10.1 * mm, "EA")
    canvas.setFillColor(colors.white)
    canvas.setFont(bold, 10)
    canvas.drawString(23 * mm, page_height - 7.5 * mm, "ENVIRONMENT AI")
    canvas.setFont(regular, 6.5)
    canvas.setFillColor(colors.HexColor("#B9DEDA"))
    canvas.drawString(23 * mm, page_height - 11.3 * mm, "HANOI AIR INTELLIGENCE")
    canvas.setFont(regular, 7)
    canvas.setFillColor(colors.HexColor("#D7EBE8"))
    canvas.drawRightString(page_width - 15 * mm, page_height - 9.3 * mm, report_id)

    canvas.setStrokeColor(LINE)
    canvas.line(15 * mm, 14 * mm, page_width - 15 * mm, 14 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont(regular, 6.5)
    canvas.drawString(15 * mm, 9.5 * mm, "Báo cáo hỗ trợ ra quyết định - cần xác minh trước khi sử dụng vận hành")
    canvas.drawRightString(page_width - 15 * mm, 9.5 * mm, f"Trang {document.page}")
    canvas.restoreState()


def render_environment_report_pdf(
    report: dict[str, Any],
    frame: pd.DataFrame,
    target: str | Path,
) -> Path:
    """Render a deterministic report to a polished, Unicode-safe PDF."""

    regular, bold = register_pdf_fonts()
    styles = _styles(regular, bold)
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        leftMargin=17 * mm,
        rightMargin=17 * mm,
        topMargin=24 * mm,
        bottomMargin=20 * mm,
        title=f"Báo cáo chất lượng không khí - {report['station']['name']}",
        author="Environment AI",
        subject="Báo cáo giám sát chất lượng không khí Hà Nội",
    )

    story: list[Any] = [
        Paragraph("BÁO CÁO CHẤT LƯỢNG KHÔNG KHÍ", styles["title"]),
        Paragraph(_escape(report["station"]["name"]), styles["station"]),
        Paragraph(
            f"Mã điểm: <b>{_escape(report['station']['station_id'])}</b> &nbsp; | &nbsp; "
            f"Kỳ dữ liệu: {_timestamp(report['period']['start'])} - {_timestamp(report['period']['end'])}",
            styles["meta"],
        ),
        Spacer(1, 5 * mm),
        _summary_cards(report, styles),
        Paragraph("Diễn biến PM2.5", styles["section"]),
        _trend_chart(frame, regular, bold),
        Paragraph("Chi tiết chỉ số", styles["section"]),
        _data_table(
            [
                ("PM2.5 trung bình", _number(report["metrics"]["pm25_mean"], suffix=" µg/m³")),
                ("PM2.5 thấp nhất", _number(report["metrics"]["pm25_min"], suffix=" µg/m³")),
                ("PM2.5 cao nhất", _number(report["metrics"]["pm25_max"], suffix=" µg/m³")),
                ("Thời điểm PM2.5 cao nhất", _timestamp(report["metrics"]["pm25_peak_timestamp"])),
                ("PM10 trung bình", _number(report["metrics"]["pm10_mean"], suffix=" µg/m³")),
                ("Độ ẩm trung bình", _number(report["metrics"]["humidity_mean"], suffix="%")),
                ("Tốc độ gió trung bình", _number(report["metrics"]["wind_speed_mean"], suffix=" km/h")),
            ],
            styles,
        ),
        KeepTogether(
            [
                Paragraph("Chất lượng và nguồn dữ liệu", styles["section"]),
                _data_table(
                    [
                        ("Điểm chất lượng trung bình", _number(report["data_quality"]["mean_score"], digits=3)),
                        ("Tỷ lệ dữ liệu nội suy", _number(float(report["data_quality"]["imputed_fraction"]) * 100, suffix="%")),
                        ("Tỷ lệ cờ ngoại lệ", _number(float(report["data_quality"]["possible_outlier_fraction"]) * 100, suffix="%")),
                        ("Nguồn không khí", str(report["provenance"]["air_source"] or "Chưa có dữ liệu")),
                        ("Nguồn khí tượng", str(report["provenance"]["weather_source"] or "Chưa có dữ liệu")),
                        ("Vị trí", f"{report['station']['latitude']}, {report['station']['longitude']}"),
                    ],
                    styles,
                ),
                Paragraph("Giới hạn sử dụng", styles["section"]),
                *[
                    Paragraph(f"• {_escape(item)}", styles["body"])
                    for item in report["limitations"]
                ],
                Spacer(1, 3 * mm),
                Table(
                    [[Paragraph("Báo cáo mang tính mô tả dữ liệu theo giờ; không phải kết luận pháp lý, y tế hoặc xác định nguồn phát thải.", styles["notice"]) ]],
                    colWidths=[176 * mm],
                    style=TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), WARNING_SOFT),
                            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#E8D59A")),
                            ("LEFTPADDING", (0, 0), (-1, -1), 10),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                            ("TOPPADDING", (0, 0), (-1, -1), 8),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ]
                    ),
                ),
            ]
        ),
    ]

    document.build(
        story,
        onFirstPage=lambda canvas, doc: _draw_page(canvas, doc, regular, bold, report["report_id"]),
        onLaterPages=lambda canvas, doc: _draw_page(canvas, doc, regular, bold, report["report_id"]),
    )
    return destination
