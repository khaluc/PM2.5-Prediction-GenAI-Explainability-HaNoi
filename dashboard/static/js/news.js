"use strict";

const state = { category: "latest", page: 1, loading: false, hasNext: true };
const elements = {
  featured: document.getElementById("featured-news"),
  grid: document.getElementById("news-grid"),
  empty: document.getElementById("news-empty"),
  status: document.getElementById("news-status"),
  refresh: document.getElementById("news-refresh"),
  prev: document.getElementById("news-prev"),
  next: document.getElementById("news-next"),
  page: document.getElementById("news-page-number"),
  heading: document.getElementById("news-heading"),
  tabs: Array.from(document.querySelectorAll("[data-category]")),
};

function safeSourceUrl(value) {
  try {
    const url = new URL(value);
    const host = url.hostname.toLowerCase().replace(/^www\./, "");
    return ["http:", "https:"].includes(url.protocol) && host === "moitruongthudo.vn" ? url.href : null;
  } catch (_error) {
    return null;
  }
}

function fallbackImage() {
  const fallback = document.createElement("div");
  fallback.className = "image-fallback";
  fallback.setAttribute("aria-label", "Ảnh tin môi trường");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 80 80");
  const circle = document.createElementNS(svg.namespaceURI, "circle");
  circle.setAttribute("cx", "40"); circle.setAttribute("cy", "40"); circle.setAttribute("r", "27");
  const leaf = document.createElementNS(svg.namespaceURI, "path");
  leaf.setAttribute("d", "M26 48c12-1 13-22 37-23-3 20-14 31-34 27 9-4 17-10 23-18-8 7-17 12-26 14Z");
  svg.append(circle, leaf); fallback.append(svg);
  return fallback;
}

function appendImage(container, item) {
  const source = safeSourceUrl(item.image_url);
  if (!source) { container.append(fallbackImage()); return; }
  const image = document.createElement("img");
  image.src = source;
  image.alt = "";
  image.loading = "lazy";
  image.referrerPolicy = "strict-origin-when-cross-origin";
  image.addEventListener("error", () => image.replaceWith(fallbackImage()), { once: true });
  container.append(image);
}

function articleMeta(item) {
  const meta = document.createElement("div");
  meta.className = "article-meta";
  const source = document.createElement("span");
  source.className = "article-source";
  source.textContent = item.source || "Môi Trường Thủ Đô";
  const date = document.createElement("time");
  if (item.published_date) date.dateTime = item.published_date;
  date.textContent = item.date_display || "Chưa rõ ngày đăng";
  meta.append(source, date);
  return meta;
}

function originalLink(item) {
  const link = document.createElement("a");
  link.className = "read-original";
  link.textContent = "Đọc bài gốc";
  link.href = safeSourceUrl(item.url) || "https://moitruongthudo.vn/thong-tin";
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

function renderFeatured(item) {
  const article = document.createElement("article");
  article.className = "featured-article";
  const image = document.createElement("div"); image.className = "featured-image"; appendImage(image, item);
  const copy = document.createElement("div"); copy.className = "featured-copy";
  const heading = document.createElement("h3"); heading.textContent = item.title || "Bản tin môi trường";
  const excerpt = document.createElement("p"); excerpt.textContent = item.excerpt || "Mở bài viết tại nguồn để xem nội dung đầy đủ.";
  copy.append(articleMeta(item), heading, excerpt, originalLink(item));
  article.append(image, copy);
  elements.featured.replaceChildren(article);
}

function renderCard(item, index) {
  const article = document.createElement("article");
  article.className = "news-card";
  article.style.animationDelay = `${Math.min(index * 55, 330)}ms`;
  const image = document.createElement("div"); image.className = "card-image"; appendImage(image, item);
  const copy = document.createElement("div"); copy.className = "card-copy";
  const heading = document.createElement("h3"); heading.textContent = item.title || "Bản tin môi trường";
  const excerpt = document.createElement("p"); excerpt.textContent = item.excerpt || "Mở bài viết tại nguồn để xem nội dung đầy đủ.";
  copy.append(articleMeta(item), heading, excerpt, originalLink(item));
  article.append(image, copy);
  return article;
}

function setLoading(loading) {
  state.loading = loading;
  elements.refresh.disabled = loading;
  elements.refresh.classList.toggle("loading", loading);
  elements.prev.disabled = loading || state.page <= 1;
  elements.next.disabled = loading || state.page >= 10 || !state.hasNext;
}

function setStatus(payload) {
  const stamp = payload.fetched_at ? new Date(payload.fetched_at) : null;
  const validStamp = stamp && !Number.isNaN(stamp.getTime());
  const timeText = validStamp
    ? stamp.toLocaleString("vi-VN", { timeZone: "Asia/Ho_Chi_Minh", hour: "2-digit", minute: "2-digit", day: "2-digit", month: "2-digit", year: "numeric" })
    : "không rõ thời gian";
  elements.status.className = `news-meta${payload.stale ? " stale" : ""}`;
  elements.status.querySelector("span:last-child").textContent = payload.stale
    ? `${payload.warning || "Đang dùng bản lưu gần nhất"} Cập nhật: ${timeText}.`
    : `${payload.source?.name || "Môi Trường Thủ Đô"} · Cập nhật ${timeText} · ${payload.total || 0} tin`;
}

function render(payload) {
  const items = Array.isArray(payload.items) ? payload.items : [];
  state.hasNext = Boolean(payload.has_next);
  elements.featured.replaceChildren();
  elements.grid.replaceChildren();
  elements.empty.classList.toggle("hidden", items.length > 0);
  elements.featured.classList.toggle("hidden", items.length === 0);
  elements.grid.classList.toggle("hidden", items.length <= 1);
  if (items.length) {
    renderFeatured(items[0]);
    items.slice(1).forEach((item, index) => elements.grid.append(renderCard(item, index)));
  }
  elements.page.textContent = String(state.page);
  elements.heading.textContent = payload.category_label || "Mới cập nhật";
  elements.next.disabled = state.loading || state.page >= 10 || !state.hasNext;
  setStatus(payload);
}

async function loadNews({ refresh = false } = {}) {
  if (state.loading) return;
  setLoading(true);
  elements.status.className = "news-meta";
  elements.status.querySelector("span:last-child").textContent = "Đang đồng bộ bản tin…";
  try {
    const query = new URLSearchParams({ category: state.category, page: String(state.page), limit: "12" });
    if (refresh) query.set("refresh", "true");
    const response = await fetch(`/api/news?${query.toString()}`, { headers: { Accept: "application/json" } });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || payload.error || "Không thể tải bản tin.");
    render(payload);
  } catch (error) {
    elements.featured.replaceChildren(); elements.grid.replaceChildren();
    elements.featured.classList.add("hidden"); elements.grid.classList.add("hidden"); elements.empty.classList.remove("hidden");
    elements.empty.querySelector("h3").textContent = "Không thể tải bản tin";
    elements.empty.querySelector("p").textContent = error instanceof Error ? error.message : "Nguồn tin tạm thời không phản hồi.";
    elements.status.className = "news-meta error";
    elements.status.querySelector("span:last-child").textContent = "Kết nối nguồn tin gặp sự cố.";
  } finally {
    setLoading(false);
  }
}

elements.tabs.forEach((tab) => tab.addEventListener("click", () => {
  if (state.loading || tab.dataset.category === state.category) return;
  state.category = tab.dataset.category;
  state.page = 1;
  elements.tabs.forEach((item) => {
    const active = item === tab;
    item.classList.toggle("active", active);
    item.setAttribute("aria-selected", String(active));
  });
  loadNews();
}));
elements.refresh.addEventListener("click", () => loadNews({ refresh: true }));
elements.prev.addEventListener("click", () => { if (state.page > 1) { state.page -= 1; loadNews(); window.scrollTo({ top: 330, behavior: "smooth" }); } });
elements.next.addEventListener("click", () => { if (state.page < 10) { state.page += 1; loadNews(); window.scrollTo({ top: 330, behavior: "smooth" }); } });

loadNews();
