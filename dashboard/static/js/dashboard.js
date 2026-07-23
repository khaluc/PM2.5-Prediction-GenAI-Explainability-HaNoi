"use strict";

const state = {
  stations: [],
  stationId: null,
  snapshot: null,
  boundary: null,
  map: null,
  boundaryLayer: null,
  markers: new Map(),
  lastHourlySuccess: null,
  hourlyPollTimer: null,
  hourlyFallbackTimer: null,
  hourlyRefreshInFlight: false,
  explanationCache: new Map(),
  explanationInFlightKey: null,
  explanationRequestId: 0,
  explanationRetryTimer: null,
};

const byId = (id) => document.getElementById(id);
const setText = (id, value, fallback = "—") => {
  const element = byId(id);
  if (element) element.textContent = value === null || value === undefined || value === "" ? fallback : String(value);
};
const formatNumber = (value, digits = 1) => {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString("vi-VN", { maximumFractionDigits: digits }) : "—";
};
const formatTime = (value) => {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("vi-VN", {
    hour: "2-digit", minute: "2-digit", day: "2-digit", month: "2-digit", year: "numeric",
  });
};
const formatHour = (value) => {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
};

async function apiFetch(url, options = {}) {
  const response = await fetch(url, { ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
  return payload;
}

function setLoading(active) { byId("loading-layer").classList.toggle("hidden", !active); }
function setSystem(mode, message) {
  const element = byId("system-state");
  element.classList.remove("online", "error");
  if (mode) element.classList.add(mode);
  element.querySelector("span:last-child").textContent = message;
}
function toast(message, isError = false) {
  const element = byId("toast");
  element.textContent = message;
  element.classList.toggle("error", isError);
  element.classList.remove("hidden");
  window.setTimeout(() => element.classList.add("hidden"), 4200);
}

function nextBrowserHourlyRefresh(now = new Date()) {
  const target = new Date(now);
  target.setHours(target.getHours() + 1, 2, 0, 0);
  return target;
}

function scheduleBrowserHourlyRefresh() {
  if (state.hourlyFallbackTimer) window.clearTimeout(state.hourlyFallbackTimer);
  const target = nextBrowserHourlyRefresh();
  state.hourlyFallbackTimer = window.setTimeout(async () => {
    if (state.stationId && !state.hourlyRefreshInFlight) {
      state.hourlyRefreshInFlight = true;
      try { await selectStation(state.stationId, { moveMap: false }); }
      finally { state.hourlyRefreshInFlight = false; }
    }
    await syncHourlyUpdateStatus({ refreshOnChange: false });
    scheduleBrowserHourlyRefresh();
  }, Math.max(1_000, target.getTime() - Date.now()));
}

async function syncHourlyUpdateStatus({ refreshOnChange = true } = {}) {
  const caption = byId("auto-update-status");
  if (!caption) return;
  try {
    const status = await apiFetch("/api/system/hourly-update");
    caption.classList.remove("updating", "error");
    const latestSuccess = status.last_success_at || null;
    if (!status.enabled) {
      caption.textContent = "Cập nhật tự động đang tắt";
      caption.classList.add("error");
    } else if (status.running) {
      caption.textContent = "Đang lấy dữ liệu và chạy lại dự báo ML…";
      caption.classList.add("updating");
    } else if (status.last_error && !latestSuccess) {
      caption.textContent = "Lần cập nhật gần nhất chưa thành công · hệ thống sẽ tự thử lại";
      caption.classList.add("error");
    } else {
      const next = status.next_run_at ? formatTime(status.next_run_at) : "đầu giờ kế tiếp";
      caption.textContent = `Tự động mỗi giờ · kỳ tiếp theo ${next}`;
    }

    const changed = Boolean(
      refreshOnChange
      && latestSuccess
      && state.lastHourlySuccess
      && latestSuccess !== state.lastHourlySuccess
    );
    state.lastHourlySuccess = latestSuccess;
    if (changed && state.stationId && !state.hourlyRefreshInFlight) {
      state.hourlyRefreshInFlight = true;
      try {
        await selectStation(state.stationId, { moveMap: false });
        toast("Đã cập nhật dữ liệu giờ mới và chạy lại dự báo ML.");
      } finally {
        state.hourlyRefreshInFlight = false;
      }
    }
  } catch (_error) {
    caption.textContent = "Chưa đọc được trạng thái cập nhật tự động";
    caption.classList.add("error");
  }
}

function aqiStatus(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return { label: "Thiếu dữ liệu", className: "unknown", health: "Chưa đủ dữ liệu AQI để đưa ra khuyến nghị theo dõi." };
  if (number <= 50) return { label: "Tốt", className: "good", health: "Chất lượng không khí đang ở mức tốt; tiếp tục theo dõi cập nhật địa phương." };
  if (number <= 100) return { label: "Trung bình", className: "moderate", health: "Nhóm nhạy cảm nên theo dõi triệu chứng và hạn chế gắng sức kéo dài ngoài trời khi thấy khó chịu." };
  if (number <= 150) return { label: "Kém cho nhóm nhạy cảm", className: "watch", health: "Trẻ em, người cao tuổi và người có bệnh hô hấp nên giảm hoạt động ngoài trời kéo dài." };
  if (number <= 200) return { label: "Không lành mạnh", className: "unhealthy", health: "Nên giảm hoạt động ngoài trời; nhóm nhạy cảm cần thận trọng hơn và theo dõi hướng dẫn chính thức." };
  if (number <= 300) return { label: "Rất không lành mạnh", className: "severe", health: "Hạn chế hoạt động ngoài trời và kiểm tra thông báo của cơ quan chức năng." };
  return { label: "Nguy hại", className: "hazardous", health: "Tránh hoạt động ngoài trời không cần thiết và theo dõi khuyến cáo khẩn cấp chính thức." };
}

function pm25Level(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return { color: "#9ba7b1", className: "unknown", label: "Thiếu dữ liệu", description: "Chưa có đủ dữ liệu PM2.5 theo giờ để xác định mức sàng lọc." };
  if (number <= 15) return { color: "#35a96b", className: "good", label: "Tốt", description: "Nồng độ PM2.5 theo giờ đang ở vùng thấp trên thang sàng lọc của hệ thống." };
  if (number <= 35) return { color: "#e3ba35", className: "moderate", label: "Cần theo dõi", description: "PM2.5 đã tăng khỏi vùng thấp; nên tiếp tục theo dõi các lần cập nhật tiếp theo." };
  if (number <= 55) return { color: "#ee9238", className: "watch", label: "Cao", description: "PM2.5 theo giờ đang cao; nhóm nhạy cảm nên chú ý triệu chứng và thời gian ở ngoài trời." };
  if (number <= 75) return { color: "#df594e", className: "unhealthy", label: "Rất cao", description: "Nồng độ PM2.5 theo giờ ở vùng rất cao và cần được đối chiếu với trung bình 24 giờ." };
  return { color: "#8652a0", className: "severe", label: "Nghiêm trọng", description: "PM2.5 theo giờ vượt vùng 75 µg/m³; cần theo dõi sát quan trắc và hướng dẫn chính thức." };
}

function pm25ScalePosition(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return 0;
  const bounds = [0, 15, 35, 55, 75, 150];
  for (let index = 0; index < bounds.length - 1; index += 1) {
    if (number <= bounds[index + 1]) {
      const ratio = (number - bounds[index]) / (bounds[index + 1] - bounds[index]);
      return Math.max(0, Math.min(100, index * 20 + ratio * 20));
    }
  }
  return 100;
}

function animateMetric(id, value, digits = 1) {
  const element = byId(id), target = Number(value);
  if (!element) return;
  if (!Number.isFinite(target)) { element.textContent = "—"; delete element.dataset.numericValue; return; }
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  const previous = Number(element.dataset.numericValue);
  element.dataset.numericValue = String(target);
  if (reduceMotion) { element.textContent = formatNumber(target, digits); return; }
  const startValue = Number.isFinite(previous) ? previous : 0;
  const startedAt = performance.now(), duration = 720;
  const draw = (now) => {
    if (Number(element.dataset.numericValue) !== target) return;
    const progress = Math.min(1, (now - startedAt) / duration);
    const eased = 1 - Math.pow(1 - progress, 3);
    element.textContent = formatNumber(startValue + (target - startValue) * eased, digits);
    if (progress < 1) window.requestAnimationFrame(draw);
  };
  window.requestAnimationFrame(draw);
}

function initialiseMap(boundary) {
  const container = byId("station-map");
  if (!window.L) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "Không tải được thư viện bản đồ tương tác.";
    container.replaceChildren(empty);
    return;
  }
  state.map = L.map(container, { zoomControl: false, preferCanvas: true, minZoom: 8, maxZoom: 17 });
  L.control.zoom({ position: "bottomright" }).addTo(state.map);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(state.map);
  if (boundary?.features?.length) {
    state.boundaryLayer = L.geoJSON(boundary, {
      style: { color: "#1d6f68", weight: 2, opacity: .85, fillColor: "#65b7aa", fillOpacity: .08 },
    }).addTo(state.map);
    state.map.fitBounds(state.boundaryLayer.getBounds(), { padding: [16, 16] });
  } else {
    state.map.setView([21.0285, 105.8542], 10);
  }
}

function markerIcon(station) {
  const level = pm25Level(station.latest_pm25);
  const value = Number.isFinite(Number(station.latest_pm25)) ? Math.round(Number(station.latest_pm25)) : "—";
  return L.divIcon({
    className: `air-marker pm-${level.className}${station.station_id === state.stationId ? " selected" : ""}`,
    html: `<div class="air-marker-inner"><span>${value}</span></div>`,
    iconSize: [45, 45],
    iconAnchor: [22, 43],
  });
}

function renderMapStations(stations) {
  if (!state.map || !window.L) return;
  state.markers.forEach((marker) => marker.remove());
  state.markers.clear();
  stations.forEach((station) => {
    const latitude = Number(station.latitude), longitude = Number(station.longitude);
    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return;
    const marker = L.marker([latitude, longitude], { icon: markerIcon(station), keyboard: true }).addTo(state.map);
    const tooltip = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = station.name || station.station_id;
    const reading = document.createElement("div");
    reading.textContent = `PM2.5: ${formatNumber(station.latest_pm25)} µg/m³`;
    tooltip.append(title, reading);
    marker.bindTooltip(tooltip, { direction: "top", offset: [0, -35] });
    marker.on("click", () => {
      byId("station-select").value = station.station_id;
      selectStation(station.station_id, { moveMap: false });
    });
    state.markers.set(station.station_id, marker);
  });
}

function renderStationRanking(stations) {
  const container = byId("station-ranking");
  container.replaceChildren();
  const sorted = [...stations].sort((a, b) => Number(b.latest_pm25 ?? -1) - Number(a.latest_pm25 ?? -1));
  sorted.forEach((station) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `station-rank-item${station.station_id === state.stationId ? " selected" : ""}`;
    button.dataset.stationId = station.station_id;
    const dot = document.createElement("i");
    dot.className = `level-${pm25Level(station.latest_pm25).className}`;
    const name = document.createElement("span");
    name.textContent = station.name || station.station_id;
    const value = document.createElement("strong");
    value.textContent = formatNumber(station.latest_pm25);
    button.append(dot, name, value);
    button.addEventListener("click", () => {
      byId("station-select").value = station.station_id;
      selectStation(station.station_id, { moveMap: true });
    });
    container.append(button);
  });
}

function renderStations(stations) {
  const select = byId("station-select");
  select.replaceChildren();
  stations.forEach((station) => {
    const option = document.createElement("option");
    option.value = station.station_id;
    option.textContent = station.name || station.station_id;
    select.append(option);
  });
  renderMapStations(stations);
  renderStationRanking(stations);
}

function updateMapSelection(moveMap = false) {
  state.markers.forEach((marker, stationId) => {
    const element = marker.getElement();
    if (element) element.classList.toggle("selected", stationId === state.stationId);
  });
  document.querySelectorAll(".station-rank-item").forEach((item) => item.classList.toggle("selected", item.dataset.stationId === state.stationId));
  if (moveMap) {
    const station = state.stations.find((item) => item.station_id === state.stationId);
    if (station && state.map) state.map.flyTo([Number(station.latitude), Number(station.longitude)], 11, { duration: .7 });
  }
}

function renderChart(items) {
  const container = byId("pm-chart");
  container.replaceChildren();
  const data = [...(items || [])].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp)).filter((item) => Number.isFinite(Number(item.pm25)));
  if (!data.length) {
    const empty = document.createElement("div"); empty.className = "empty-state"; empty.textContent = "Không có dữ liệu PM2.5 trong khoảng này."; container.append(empty); return;
  }
  const values = data.map((item) => Number(item.pm25));
  const minimum = Math.min(...values), maximum = Math.max(...values);
  const minimumIndex = values.indexOf(minimum), maximumIndex = values.indexOf(maximum);
  setText("chart-average", `${formatNumber(values.reduce((sum, value) => sum + value, 0) / values.length)} µg/m³`);
  setText("chart-minimum", formatNumber(minimum));
  setText("chart-maximum", formatNumber(maximum));
  setText("chart-minimum-time", formatHour(data[minimumIndex].timestamp));
  setText("chart-maximum-time", formatHour(data[maximumIndex].timestamp));

  const width = 900, height = 310, left = 62, right = 18, top = 20, bottom = 64;
  const threshold = 15;
  const tickStep = Math.max(5, Math.ceil(Math.max(maximum, threshold) * 1.08 / 4 / 5) * 5);
  const yMax = tickStep * 4;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const slotWidth = plotWidth / data.length;
  const barWidth = Math.max(5, Math.min(24, slotWidth * .68));
  const x = (index) => left + slotWidth * (index + .5);
  const y = (value) => top + (yMax - value) * (plotHeight / yMax);
  const chartBottom = height - bottom;
  const createSvg = (name) => document.createElementNS("http://www.w3.org/2000/svg", name);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("aria-hidden", "true");
  const defs = createSvg("defs");
  const gradient = createSvg("linearGradient");
  gradient.id = "pmBarGradient"; gradient.setAttribute("x1", "0"); gradient.setAttribute("y1", "0"); gradient.setAttribute("x2", "0"); gradient.setAttribute("y2", "1");
  [["0%", "#b7e629"], ["100%", "#83c91c"]].forEach(([offset, color]) => {
    const stop = createSvg("stop"); stop.setAttribute("offset", offset); stop.setAttribute("stop-color", color); gradient.append(stop);
  });
  defs.append(gradient); svg.append(defs);

  for (let index = 0; index <= 4; index += 1) {
    const value = tickStep * index, yy = y(value);
    const grid = createSvg("line"); grid.classList.add("chart-grid"); grid.setAttribute("x1", left); grid.setAttribute("x2", width - right); grid.setAttribute("y1", yy); grid.setAttribute("y2", yy); svg.append(grid);
    const label = createSvg("text"); label.classList.add("chart-axis-text"); label.setAttribute("x", left - 11); label.setAttribute("y", yy + 3); label.setAttribute("text-anchor", "end"); label.textContent = formatNumber(value, 0); svg.append(label);
  }

  const thresholdY = y(threshold);
  const thresholdLine = createSvg("line"); thresholdLine.classList.add("chart-threshold"); thresholdLine.setAttribute("x1", left); thresholdLine.setAttribute("x2", width - right); thresholdLine.setAttribute("y1", thresholdY); thresholdLine.setAttribute("y2", thresholdY); svg.append(thresholdLine);
  const thresholdLabel = createSvg("text"); thresholdLabel.classList.add("chart-threshold-label"); thresholdLabel.setAttribute("x", width - right - 3); thresholdLabel.setAttribute("y", thresholdY - 6); thresholdLabel.setAttribute("text-anchor", "end"); thresholdLabel.textContent = "Mốc tham chiếu 15"; svg.append(thresholdLabel);

  const hoverLine = createSvg("line"); hoverLine.classList.add("chart-hover-line"); hoverLine.setAttribute("y1", top); hoverLine.setAttribute("y2", chartBottom); svg.append(hoverLine);
  const hoverMarker = createSvg("circle"); hoverMarker.classList.add("chart-hover-marker"); hoverMarker.setAttribute("r", "5"); svg.append(hoverMarker);

  const bars = [];
  data.forEach((item, index) => {
    const value = Number(item.pm25), yy = y(value);
    const bar = createSvg("rect");
    bar.classList.add("pm-chart-bar");
    bar.setAttribute("x", x(index) - barWidth / 2);
    bar.setAttribute("y", yy);
    bar.setAttribute("width", barWidth);
    bar.setAttribute("height", Math.max(1, chartBottom - yy));
    bar.setAttribute("rx", "2.5");
    bar.setAttribute("tabindex", "0");
    bar.setAttribute("role", "button");
    bar.setAttribute("aria-label", `${formatHour(item.timestamp)}: PM2.5 ${formatNumber(value)} microgam trên mét khối`);
    bars.push(bar); svg.append(bar);
  });

  const tickIndexes = new Set([0, data.length - 1]);
  const tickInterval = Math.max(1, Math.ceil((data.length - 1) / 8));
  for (let index = 0; index < data.length; index += tickInterval) tickIndexes.add(index);
  [...tickIndexes].sort((a, b) => a - b).forEach((index) => {
    const tick = createSvg("text"); tick.classList.add("chart-axis-text", "chart-time-label"); tick.setAttribute("x", x(index)); tick.setAttribute("y", chartBottom + 18); tick.setAttribute("text-anchor", index === 0 ? "start" : index === data.length - 1 ? "end" : "middle"); tick.textContent = formatHour(data[index].timestamp); svg.append(tick);
  });

  const yTitle = createSvg("text"); yTitle.classList.add("chart-axis-title"); yTitle.setAttribute("x", "16"); yTitle.setAttribute("y", top + plotHeight / 2); yTitle.setAttribute("text-anchor", "middle"); yTitle.setAttribute("transform", `rotate(-90 16 ${top + plotHeight / 2})`); yTitle.textContent = "PM2.5 (µg/m³)"; svg.append(yTitle);
  const firstDate = new Date(data[0].timestamp), lastDate = new Date(data[data.length - 1].timestamp);
  const dateFormatter = new Intl.DateTimeFormat("vi-VN", { day: "2-digit", month: "2-digit", year: "numeric" });
  const startDate = createSvg("text"); startDate.classList.add("chart-date-label"); startDate.setAttribute("x", left); startDate.setAttribute("y", height - 5); startDate.textContent = Number.isNaN(firstDate.getTime()) ? "—" : dateFormatter.format(firstDate); svg.append(startDate);
  const endDate = createSvg("text"); endDate.classList.add("chart-date-label"); endDate.setAttribute("x", width - right); endDate.setAttribute("y", height - 5); endDate.setAttribute("text-anchor", "end"); endDate.textContent = Number.isNaN(lastDate.getTime()) ? "—" : dateFormatter.format(lastDate); svg.append(endDate);
  const xTitle = createSvg("text"); xTitle.classList.add("chart-axis-title"); xTitle.setAttribute("x", left + plotWidth / 2); xTitle.setAttribute("y", height - 5); xTitle.setAttribute("text-anchor", "middle"); xTitle.textContent = "Thời gian"; svg.append(xTitle);

  const tooltip = document.createElement("div"); tooltip.className = "chart-tooltip hidden"; tooltip.setAttribute("role", "status");
  const tooltipTime = document.createElement("time");
  const tooltipValue = document.createElement("strong");
  const tooltipStation = document.createElement("span");
  tooltip.append(tooltipTime, tooltipValue, tooltipStation);
  container.append(svg, tooltip);

  const station = state.stations.find((item) => item.station_id === state.stationId);
  const stationName = station?.name || state.snapshot?.latest?.location_name || "Điểm quan trắc";
  const setFocus = (index, showTooltip) => {
    const xx = x(index), yy = y(values[index]);
    hoverLine.setAttribute("x1", xx); hoverLine.setAttribute("x2", xx);
    hoverMarker.setAttribute("cx", xx); hoverMarker.setAttribute("cy", yy);
    bars.forEach((bar, barIndex) => bar.classList.toggle("active", barIndex === index));
    if (!showTooltip) { tooltip.classList.add("hidden"); return; }
    tooltipTime.textContent = formatTime(data[index].timestamp);
    tooltipValue.textContent = `${formatNumber(values[index])} µg/m³`;
    tooltipStation.textContent = stationName;
    tooltip.style.left = `${xx / width * 100}%`;
    tooltip.style.top = `${Math.max(10, yy / height * 100)}%`;
    tooltip.classList.toggle("align-right", xx > width * .72);
    tooltip.classList.remove("hidden");
  };
  bars.forEach((bar, index) => {
    bar.addEventListener("pointerenter", () => setFocus(index, true));
    bar.addEventListener("focus", () => setFocus(index, true));
    bar.addEventListener("pointerleave", () => setFocus(data.length - 1, false));
    bar.addEventListener("blur", () => setFocus(data.length - 1, false));
  });
  setFocus(data.length - 1, false);
}

function renderMlForecast(latest, prediction) {
  const container = byId("hourly-forecast");
  container.replaceChildren();
  const forecast = prediction?.forecast_pm25 || {};
  const horizons = [1, 3, 6].filter((hours) => Number.isFinite(Number(forecast[`${hours}h`])));
  if (!horizons.length) {
    const empty = document.createElement("div"); empty.className = "empty-state"; empty.textContent = "Dự báo ML chưa khả dụng vì chuỗi quan trắc live chưa đủ liên tục."; container.append(empty); return;
  }
  const baseTime = latest.timestamp ? new Date(latest.timestamp).getTime() : null;
  const rows = [
    { ...latest, is_now: true },
    ...horizons.map((hours) => ({
      timestamp: baseTime === null ? null : new Date(baseTime + hours * 60 * 60 * 1000).toISOString(),
      pm25: forecast[`${hours}h`],
      horizon: hours,
      model: prediction.model || "ML",
    })),
  ];
  rows.forEach((item) => {
    const card = document.createElement("article"); card.className = "forecast-hour";
    const time = document.createElement("time"); time.textContent = item.is_now ? "Bây giờ" : formatHour(item.timestamp);
    const badge = document.createElement("span");
    badge.className = item.is_now ? `forecast-aqi aqi-${aqiStatus(item.us_aqi).className}` : "forecast-aqi ml-badge";
    badge.textContent = item.is_now ? formatNumber(item.us_aqi, 0) : `+${item.horizon}h`;
    const pm = document.createElement("strong"); pm.textContent = formatNumber(item.pm25);
    const unit = document.createElement("small"); unit.textContent = "PM2.5 µg/m³";
    const weather = document.createElement("div"); weather.className = "forecast-weather";
    if (item.is_now) {
      const temperature = document.createElement("span"); temperature.textContent = `${formatNumber(item.temperature, 0)}°`;
      const humidity = document.createElement("span"); humidity.textContent = `${formatNumber(item.humidity, 0)}%`;
      const wind = document.createElement("span"); wind.textContent = `${formatNumber(item.wind_speed, 0)} km/h`;
      weather.append(temperature, humidity, wind);
    } else {
      const model = document.createElement("span"); model.textContent = item.model;
      const source = document.createElement("span"); source.textContent = "ML nội bộ";
      weather.append(model, source);
    }
    card.append(time, badge, pm, unit, weather); container.append(card);
  });
}

function renderAlerts(payload) {
  const items = payload?.items || [];
  setText("alert-count", payload?.total || 0);
  const list = byId("alert-list"); list.replaceChildren();
  if (!items.length) { const empty = document.createElement("div"); empty.className = "empty-state"; empty.textContent = "Không có cảnh báo đã lưu cho khu vực này."; list.append(empty); return; }
  items.forEach((alert) => {
    const item = document.createElement("article"); item.className = `alert-item ${alert.severity === "critical" ? "critical" : ""}`;
    const indicator = document.createElement("span"); indicator.className = "alert-indicator";
    const content = document.createElement("div");
    const title = document.createElement("h3"); title.textContent = alert.title_vi || "Cảnh báo môi trường";
    const detail = document.createElement("p"); detail.textContent = alert.evidence?.reason || "Cần xác minh dữ liệu và hiện trường.";
    const meta = document.createElement("div"); meta.className = "alert-meta";
    const severity = document.createElement("span"); severity.textContent = alert.severity || "warning";
    const time = document.createElement("span"); time.textContent = formatTime(alert.event_timestamp);
    meta.append(severity, time); content.append(title, detail, meta); item.append(indicator, content); list.append(item);
  });
}

function resetForecastExplanation(messageText = "Đang chờ dự báo ML +1 giờ để tạo nhận định tự động…") {
  const container = byId("genai-output");
  container.className = "genai-output empty";
  const message = document.createElement("p");
  message.textContent = messageText;
  container.replaceChildren(message);
}

function forecastExplanationKey(stationId = state.stationId, snapshot = state.snapshot) {
  const forecast = snapshot?.prediction?.forecast_pm25?.["1h"];
  if (!stationId || !Number.isFinite(Number(forecast))) return null;
  const timestamp = snapshot?.prediction?.timestamp || snapshot?.latest?.timestamp || "";
  return JSON.stringify([stationId, timestamp, Number(forecast)]);
}

function clearForecastExplanationRetry() {
  if (state.explanationRetryTimer) {
    window.clearTimeout(state.explanationRetryTimer);
    state.explanationRetryTimer = null;
  }
}

function scheduleForecastExplanationRetry(key, payload) {
  clearForecastExplanationRetry();
  const generationMode = payload?.result?.generation?.mode;
  const expiresAt = new Date(payload?.cache?.expires_at || "");
  if (
    generationMode === "dashscope"
    || Number.isNaN(expiresAt.getTime())
  ) return;

  const delay = Math.max(1_000, expiresAt.getTime() - Date.now() + 1_000);
  state.explanationRetryTimer = window.setTimeout(() => {
    state.explanationRetryTimer = null;
    if (key !== forecastExplanationKey()) return;
    state.explanationCache.delete(key);
    void generateForecastExplanation();
  }, delay);
}

function setForecastExplanationLoading() {
  const container = byId("genai-output");
  container.className = "genai-output loading";
  const message = document.createElement("p");
  message.textContent = "Đang tự động giải thích dự báo ML +1 giờ…";
  container.replaceChildren(message);
}

function renderKnowledgeGraph(knowledge) {
  if (!knowledge || !Array.isArray(knowledge.relations)) return null;

  const section = document.createElement("section");
  section.className = "knowledge-graph";
  section.setAttribute("aria-label", "Knowledge Graph PM2.5");

  const heading = document.createElement("div");
  heading.className = "knowledge-graph-heading";
  const title = document.createElement("h4");
  title.textContent = "Knowledge Graph PM2.5";
  const scope = document.createElement("span");
  scope.textContent = "Kiến thức miền · không kết luận nhân quả";
  heading.append(title, scope);

  const flow = document.createElement("div");
  flow.className = "knowledge-graph-flow";
  const focus = document.createElement("div");
  focus.className = "knowledge-focus-node";
  const focusLabel = document.createElement("strong");
  focusLabel.textContent = "PM2.5";
  const focusType = document.createElement("small");
  focusType.textContent = "Chất ô nhiễm trung tâm";
  focus.append(focusLabel, focusType);
  flow.append(focus);

  const supportedSources = knowledge.supported_emission_sources || [];
  const unverifiedSources = knowledge.unverified_emission_sources || [];
  const groups = [
    {
      relation: "EMITS",
      label: "Nguồn phát thải",
      items: [
        ...supportedSources.map((item) => ({ ...item, state: "supported", note: "Có dữ liệu liên quan hiện tại" })),
        ...unverifiedSources.map((item) => ({ ...item, state: "unverified", note: "Chưa có dữ liệu xác minh hiện tại" })),
      ],
    },
    {
      relation: "INFLUENCED_BY",
      label: "Yếu tố khí tượng",
      items: (knowledge.meteorological_factors || []).map((item) => ({
        ...item,
        state: item.currently_observed ? "relevant" : "general",
        note: item.currently_observed ? "Có dữ liệu thời tiết hiện tại" : "Kiến thức khí tượng chung",
      })),
    },
    {
      relation: "MITIGATED_BY",
      label: "Biện pháp giảm nhẹ",
      items: (knowledge.mitigations || []).map((item) => ({
        ...item,
        state: "general",
        note: "Biện pháp quy hoạch/chính sách",
      })),
    },
  ];

  groups.forEach((group) => {
    const column = document.createElement("div");
    column.className = "knowledge-relation-group";
    const relation = document.createElement("b");
    relation.textContent = group.relation;
    const label = document.createElement("p");
    label.textContent = group.label;
    const nodes = document.createElement("div");
    nodes.className = "knowledge-node-list";
    group.items.forEach((item) => {
      const node = document.createElement("div");
      node.className = `knowledge-node ${item.state}`;
      const nodeLabel = document.createElement("span");
      nodeLabel.textContent = item.label_vi || item.id;
      const note = document.createElement("small");
      note.textContent = item.note;
      node.append(nodeLabel, note);
      nodes.append(node);
    });
    column.append(relation, label, nodes);
    flow.append(column);
  });

  const disclaimer = document.createElement("p");
  disclaimer.className = "knowledge-disclaimer";
  disclaimer.textContent = knowledge.disclaimer_vi || "Knowledge Graph chỉ cung cấp kiến thức tham khảo có kiểm soát.";
  section.append(heading, flow, disclaimer);
  return section;
}

function renderForecastExplanation(payload) {
  const result = payload?.result || payload || {};
  const explanation = result.explanation || {};
  const forecast = result.forecast || {};
  const generation = result.generation || {};
  const container = byId("genai-output");
  container.className = "genai-output";
  container.replaceChildren();

  const header = document.createElement("div"); header.className = "genai-result-heading";
  const title = document.createElement("h3"); title.textContent = explanation.headline || "Giải thích dự báo";
  const mode = document.createElement("span"); mode.className = `generation-mode ${generation.mode === "dashscope" ? "live" : "fallback"}`;
  mode.textContent = generation.mode === "dashscope" ? `DashScope · ${generation.provider_model || "DeepSeek"}` : "Giải thích kiểm soát · dự phòng";
  header.append(title, mode);

  const summary = document.createElement("p"); summary.className = "genai-summary"; summary.textContent = explanation.summary || "Không có nội dung giải thích.";
  const interpretation = document.createElement("div"); interpretation.className = "genai-interpretation";
  const interpretationTitle = document.createElement("h4"); interpretationTitle.textContent = "Nhận định tổng hợp";
  const interpretationText = document.createElement("p");
  interpretationText.textContent = explanation.overall_interpretation
    || "Chưa có đủ dữ liệu để tổng hợp mối liên hệ giữa các điều kiện quan sát.";
  interpretation.append(interpretationTitle, interpretationText);
  const columns = document.createElement("div"); columns.className = "genai-columns";
  const conditionBlock = document.createElement("div");
  const conditionTitle = document.createElement("h4"); conditionTitle.textContent = "Điều kiện có thể góp phần";
  const conditionList = document.createElement("ul");
  (explanation.contributing_conditions || []).forEach((item) => { const li = document.createElement("li"); li.textContent = item; conditionList.append(li); });
  conditionBlock.append(conditionTitle, conditionList);
  const adviceBlock = document.createElement("div");
  const adviceTitle = document.createElement("h4"); adviceTitle.textContent = "Khuyến nghị";
  const advice = document.createElement("p"); advice.textContent = explanation.sensitive_group_advice || "—";
  const actions = document.createElement("ul");
  (explanation.recommended_actions || []).forEach((item) => { const li = document.createElement("li"); li.textContent = item; actions.append(li); });
  adviceBlock.append(adviceTitle, advice, actions); columns.append(conditionBlock, adviceBlock);

  const footer = document.createElement("div"); footer.className = "genai-result-footer";
  const uncertainty = document.createElement("p"); uncertainty.textContent = explanation.uncertainty || "Dự báo có sai số và cần được xác minh bằng quan trắc.";
  const facts = document.createElement("span"); facts.textContent = `${forecast.model || "ML"} · +${forecast.horizon_hours || "—"}h · ${formatNumber(forecast.predicted_pm25)} µg/m³`;
  footer.append(uncertainty, facts);
  const knowledgeGraph = renderKnowledgeGraph(result.grounding?.knowledge_graph);
  container.append(header, summary, interpretation, columns);
  if (knowledgeGraph) container.append(knowledgeGraph);
  container.append(footer);
}

function renderSnapshot(snapshot) {
  state.snapshot = snapshot;
  const latest = snapshot.latest || {};
  const station = state.stations.find((item) => item.station_id === state.stationId) || {};
  const stationName = station.name || latest.location_name || state.stationId;
  setText("location-title", stationName);
  setText("breadcrumb-station", stationName);
  setText("station-caption", `${formatTime(latest.timestamp)} · dữ liệu theo giờ`);
  setText("data-source", latest.air_source || "Không rõ nguồn");
  setText("hero-data-source", latest.air_source || "Không rõ nguồn");
  setText("latest-time", formatTime(latest.timestamp));
  setText("current-aqi", formatNumber(latest.us_aqi, 0));
  animateMetric("current-pm25", latest.pm25);
  setText("pollutant-pm25", formatNumber(latest.pm25));
  setText("current-pm10", formatNumber(latest.pm10));
  setText("current-o3", formatNumber(latest.o3));
  setText("current-no2", formatNumber(latest.no2));
  setText("current-so2", formatNumber(latest.so2));
  setText("current-co", formatNumber(latest.co));
  setText("current-temperature", formatNumber(latest.temperature));
  setText("current-humidity", formatNumber(latest.humidity, 0));
  setText("current-wind", formatNumber(latest.wind_speed));
  setText("current-pressure", formatNumber(latest.surface_pressure));

  const status = aqiStatus(latest.us_aqi);
  const particulate = pm25Level(latest.pm25);
  setText("aqi-status", status.label);
  setText("pm25-status", particulate.label);
  setText("pm25-level-label", particulate.label);
  setText("pm25-description", particulate.description);
  setText("health-message", status.health);
  const aqiCard = byId("aqi-card");
  [...aqiCard.classList].filter((name) => name.startsWith("pm-")).forEach((name) => aqiCard.classList.remove(name));
  aqiCard.classList.add(`pm-${particulate.className}`);
  aqiCard.style.setProperty("--pm-position", `${pm25ScalePosition(latest.pm25)}%`);

  renderChart(snapshot.history?.items || []);
  const prediction = snapshot.prediction || {};
  const mlForecast = prediction.forecast_pm25 || {};
  renderMlForecast(latest, prediction);
  const mlAvailable = ["1h", "3h", "6h"].some((key) => Number.isFinite(Number(mlForecast[key])));
  setText("forecast-source-label", mlAvailable ? `${prediction.model || "ML"} · ML NỘI BỘ` : "ML · CHƯA KHẢ DỤNG");
  setText("forecast-note", mlAvailable
    ? "Kết quả từ mô hình ML của hệ thống; không sử dụng dự báo tương lai Open-Meteo/CAMS."
    : "Cần ít nhất 169 giờ quan trắc liên tục để tạo đầy đủ đặc trưng cho mô hình.");
  setText("forecast-1h", formatNumber(mlForecast["1h"]));
  setText("forecast-3h", formatNumber(mlForecast["3h"]));
  setText("forecast-6h", formatNumber(mlForecast["6h"]));

  const anomaly = snapshot.anomaly || {}, anomalyCard = byId("anomaly-card");
  if (anomaly.available === false) {
    anomalyCard.classList.remove("anomaly");
    setText("anomaly-status", "Chưa đánh giá live");
    setText("anomaly-detail", "Thiếu chuỗi feature liên tục; không dùng kết quả cũ 30/05.");
  } else {
    const isAnomaly = Boolean(anomaly.is_anomaly); anomalyCard.classList.toggle("anomaly", isAnomaly);
    setText("anomaly-status", isAnomaly ? "Phát hiện bất thường" : "Không phát hiện bất thường");
    setText("anomaly-detail", anomaly.reason || `Nguồn phát hiện: ${anomaly.detection_source || "none"}`);
  }
  renderAlerts(snapshot.alerts);
  updateMapSelection(false);
}

async function selectStation(stationId, options = {}) {
  const stationChanged = state.stationId !== stationId;
  state.stationId = stationId;
  if (stationChanged) {
    state.snapshot = null;
    state.explanationRequestId += 1;
    state.explanationInFlightKey = null;
    clearForecastExplanationRetry();
    resetForecastExplanation("Đang tải dữ liệu khu vực để giải thích dự báo +1 giờ…");
  }
  updateMapSelection(Boolean(options.moveMap));
  setLoading(true);
  try {
    const snapshot = await apiFetch(`/api/stations/${encodeURIComponent(stationId)}/snapshot`);
    if (state.stationId !== stationId) return;
    renderSnapshot(snapshot);
    void generateForecastExplanation();
    setSystem("online", "Dữ liệu live");
  } catch (error) {
    setSystem("error", "Mất kết nối");
    toast(`Không tải được dữ liệu: ${error.message}`, true);
  } finally { setLoading(false); }
}

async function generateReport() {
  const timestamp = state.snapshot?.latest?.timestamp;
  if (!state.stationId || !timestamp) { toast("Chưa có dữ liệu để tạo báo cáo.", true); return; }
  const end = new Date(timestamp), start = new Date(end.getTime() - 23 * 60 * 60 * 1000), button = byId("report-button");
  button.disabled = true;
  const originalLabel = button.querySelector("span")?.textContent || "Tạo báo cáo PDF";
  if (button.querySelector("span")) button.querySelector("span").textContent = "Đang tạo PDF…";
  try {
    const payload = await apiFetch("/api/reports", { method: "POST", body: JSON.stringify({ station_id: state.stationId, start: start.toISOString(), end: end.toISOString(), format: "pdf", persist: true }) });
    const downloadUrl = `/api/reports/${encodeURIComponent(payload.report_id)}/download`;
    const downloadLink = byId("download-report");
    downloadLink.href = downloadUrl;
    downloadLink.download = `environment-ai-${payload.report_id}.pdf`;
    setText("report-title", `Báo cáo ${payload.report_id}`);
    setText("report-path", payload.output_path || "File PDF đã tạo");
    byId("report-dialog").showModal();
    downloadLink.click();
    toast("Đã tạo báo cáo PDF. Trình duyệt đang tải file.");
  } catch (error) { toast(`Không tạo được báo cáo: ${error.message}`, true); }
  finally {
    button.disabled = false;
    if (button.querySelector("span")) button.querySelector("span").textContent = originalLabel;
  }
}

async function generateForecastExplanation() {
  const stationId = state.stationId;
  const key = forecastExplanationKey(stationId, state.snapshot);
  if (!key) {
    resetForecastExplanation("Dự báo ML +1 giờ chưa khả dụng nên chưa thể tạo giải thích.");
    return;
  }
  if (state.explanationCache.has(key)) {
    const cachedPayload = state.explanationCache.get(key);
    renderForecastExplanation(cachedPayload);
    scheduleForecastExplanationRetry(key, cachedPayload);
    return;
  }
  if (state.explanationInFlightKey === key) return;

  const requestId = state.explanationRequestId + 1;
  state.explanationRequestId = requestId;
  state.explanationInFlightKey = key;
  setForecastExplanationLoading();
  try {
    const payload = await apiFetch("/api/forecast-explanation", {
      method: "POST",
      body: JSON.stringify({ station_id: stationId, horizon_hours: 1, use_llm: true }),
    });
    if (
      requestId !== state.explanationRequestId
      || key !== forecastExplanationKey()
    ) return;
    state.explanationCache.set(key, payload);
    if (state.explanationCache.size > 24) {
      state.explanationCache.delete(state.explanationCache.keys().next().value);
    }
    scheduleForecastExplanationRetry(key, payload);
    renderForecastExplanation(payload);
  } catch (error) {
    if (
      requestId === state.explanationRequestId
      && key === forecastExplanationKey()
    ) {
      resetForecastExplanation(
        "Chưa thể tạo giải thích tự động. Hệ thống sẽ thử lại khi dữ liệu được làm mới.",
      );
      toast(`Không tạo được giải thích: ${error.message}`, true);
    }
  } finally {
    if (state.explanationInFlightKey === key) state.explanationInFlightKey = null;
  }
}

async function init() {
  setLoading(true);
  try {
    const [stations, boundary] = await Promise.all([apiFetch("/api/stations"), apiFetch("/static/data/hanoi_boundary.geojson")]);
    if (!stations.length) throw new Error("Không có khu vực nào trong dữ liệu.");
    state.stations = stations; state.boundary = boundary; state.stationId = stations[0].station_id;
    initialiseMap(boundary); renderStations(stations);
    byId("station-select").value = state.stationId;
    await selectStation(state.stationId, { moveMap: false });
    await syncHourlyUpdateStatus({ refreshOnChange: false });
    state.hourlyPollTimer = window.setInterval(
      () => syncHourlyUpdateStatus({ refreshOnChange: true }),
      30_000,
    );
    scheduleBrowserHourlyRefresh();
  } catch (error) {
    setSystem("error", "Mất kết nối");
    toast(`Dashboard chưa sẵn sàng: ${error.message}`, true);
  } finally { setLoading(false); }
}

byId("station-select").addEventListener("change", (event) => selectStation(event.target.value, { moveMap: true }));
byId("refresh-button").addEventListener("click", () => state.stationId && selectStation(state.stationId));
byId("report-button").addEventListener("click", generateReport);
byId("report-close").addEventListener("click", () => byId("report-dialog").close());
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    syncHourlyUpdateStatus({ refreshOnChange: true });
  }
});
document.addEventListener("DOMContentLoaded", init);
