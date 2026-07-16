"""Tests for the source-attributed environmental news collector."""

from __future__ import annotations

import httpx

from src.collection.news_collector import NewsCrawler, parse_news_html


SAMPLE_HTML = """
<html><body>
  <div class="row news-item"><article>
    <div class="imgbg" style="background-image:url(/uploads/air.jpg)"></div>
    <a href="/thong-tin/860/chat-luong-khong-khi"><h3>Chất lượng &amp; không khí Hà Nội</h3></a>
    <div class="byline"><span>07/07/2023</span></div>
    <p>  Bản tin   môi trường công khai. </p>
  </article></div>
  <div class="row news-item"><article>
    <img src="https://example.com/tracker.jpg">
    <a href="/thong-tin/861/song-xanh"><h3>Sống xanh mỗi ngày</h3></a>
    <div class="byline"><span>06/07/2023</span></div>
    <p>Mô tả thứ hai.</p>
  </article></div>
  <div class="row news-item"><article>
    <a href="https://evil.example/article"><h3>Liên kết ngoài không hợp lệ</h3></a>
  </article></div>
</body></html>
"""


def test_parse_news_html_normalises_fields_and_rejects_external_urls() -> None:
    items = parse_news_html(SAMPLE_HTML)
    assert len(items) == 2
    assert items[0]["title"] == "Chất lượng & không khí Hà Nội"
    assert items[0]["excerpt"] == "Bản tin môi trường công khai."
    assert items[0]["published_date"] == "2023-07-07"
    assert items[0]["url"] == "https://moitruongthudo.vn/thong-tin/860/chat-luong-khong-khi"
    assert items[0]["image_url"] == "https://moitruongthudo.vn/uploads/air.jpg"
    assert items[1]["image_url"] is None


def test_crawler_uses_cache_and_falls_back_when_source_is_unavailable(tmp_path) -> None:
    requests = {"count": 0, "fail": False}

    def handler(request: httpx.Request) -> httpx.Response:
        requests["count"] += 1
        if requests["fail"]:
            return httpx.Response(503, request=request)
        return httpx.Response(
            200,
            text=SAMPLE_HTML,
            headers={"content-type": "text/html; charset=utf-8"},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    crawler = NewsCrawler(tmp_path / "news.json", http_client=client)
    first = crawler.fetch("latest", page=1, limit=1, force_refresh=True)
    assert first["cache_status"] == "live"
    assert first["total"] == 1
    assert requests["count"] == 1

    cached = crawler.fetch("latest", page=1, limit=2)
    assert cached["cache_status"] == "fresh_cache"
    assert len(cached["items"]) == 2
    assert requests["count"] == 1

    requests["fail"] = True
    fallback = crawler.fetch("latest", page=1, limit=2, force_refresh=True)
    assert fallback["cache_status"] == "stale_cache"
    assert fallback["stale"] is True
    assert "bản lưu" in fallback["warning"]
    client.close()


def test_crawler_builds_category_and_page_query(tmp_path) -> None:
    seen_url = {"value": ""}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_url["value"] = str(request.url)
        return httpx.Response(
            200,
            text=SAMPLE_HTML,
            headers={"content-type": "text/html"},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    crawler = NewsCrawler(tmp_path / "news.json", http_client=client)
    payload = crawler.fetch("international", page=3)
    assert payload["category_label"] == "Tin quốc tế"
    assert "mode=2" in seen_url["value"]
    assert "page=3" in seen_url["value"]
    client.close()
