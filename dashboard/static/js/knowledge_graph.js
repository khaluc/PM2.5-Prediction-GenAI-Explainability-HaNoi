"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const RELATION_CLASS = {
  EMITS: "emits",
  INFLUENCED_BY: "influenced-by",
  MITIGATED_BY: "mitigated-by",
};
const TYPE_LABELS = {
  pollutant: "Chất ô nhiễm",
  emission_source: "Nguồn phát thải",
  meteorological_factor: "Yếu tố khí tượng",
  mitigation_measure: "Biện pháp giảm nhẹ",
};
const DEFAULT_DESCRIPTIONS = {
  emission_source: "Nguồn phát thải tiềm năng trong kiến thức miền chung; cần bằng chứng hiện trường để liên hệ với một sự kiện cụ thể.",
  meteorological_factor: "Yếu tố khí tượng có thể ảnh hưởng đến khuếch tán, vận chuyển, hình thành hoặc loại bỏ chất ô nhiễm ngoài trời.",
  mitigation_measure: "Biện pháp quy hoạch hoặc chính sách có thể hỗ trợ giảm phát thải hay nồng độ ô nhiễm; hiệu quả cần được đánh giá theo bối cảnh.",
};

const state = {
  graph: null,
  nodeById: new Map(),
  sourceById: new Map(),
  positions: new Map(),
  nodeElements: new Map(),
  edgeElements: [],
  activeRelations: new Set(Object.keys(RELATION_CLASS)),
  transform: { x: 0, y: 0, k: 1 },
  selected: null,
  dragNodeId: null,
  panStart: null,
  pointerMoved: false,
};

const byId = (id) => document.getElementById(id);
const svgElement = (tag, attributes = {}) => {
  const element = document.createElementNS(SVG_NS, tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
  return element;
};
const normalise = (value) => String(value || "")
  .normalize("NFD")
  .replace(/[\u0300-\u036f]/g, "")
  .toLocaleLowerCase("vi-VN");

async function apiFetch(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
  return payload;
}

function nodeRadius(node) {
  return node.id === state.graph?.focus_node ? 50 : 35;
}

function layoutGraph() {
  state.positions.clear();
  if (!state.graph) return;
  const focusId = state.graph.focus_node;
  state.positions.set(focusId, { x: 0, y: 0 });

  const columns = {
    emission_source: { x: -390, startY: -170, gap: 170 },
    meteorological_factor: { x: 390, startY: -255, gap: 170 },
    mitigation_measure: { x: 0, startY: 335, gap: 0 },
  };
  Object.entries(columns).forEach(([type, layout]) => {
    const nodes = state.graph.nodes.filter((node) => node.type === type);
    nodes.forEach((node, index) => {
      if (type === "mitigation_measure") {
        state.positions.set(node.id, { x: -230 + index * 230, y: layout.startY });
      } else {
        state.positions.set(node.id, { x: layout.x, y: layout.startY + index * layout.gap });
      }
    });
  });

  const pollutantPositions = {
    pm10: { x: -95, y: -205 },
    no2: { x: 105, y: -205 },
    so2: { x: 165, y: 150 },
    co: { x: -165, y: 150 },
  };
  Object.entries(pollutantPositions).forEach(([id, position]) => {
    if (state.graph.nodes.some((node) => node.id === id)) state.positions.set(id, position);
  });

  const unplaced = state.graph.nodes.filter((node) => !state.positions.has(node.id));
  unplaced.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(1, unplaced.length);
    state.positions.set(node.id, { x: Math.cos(angle) * 360, y: Math.sin(angle) * 300 });
  });
}

function labelLines(label, maxLength = 18) {
  const words = String(label || "").split(/\s+/).filter(Boolean);
  if (!words.length) return ["—"];
  const lines = [];
  let current = "";
  words.forEach((word) => {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length > maxLength && current) {
      lines.push(current);
      current = word;
    } else {
      current = candidate;
    }
  });
  if (current) lines.push(current);
  if (lines.length <= 2) return lines;
  return [lines[0], `${lines.slice(1).join(" ").slice(0, maxLength - 1)}…`];
}

function edgeGeometry(edge) {
  const source = state.positions.get(edge.source);
  const target = state.positions.get(edge.target);
  const sourceNode = state.nodeById.get(edge.source);
  const targetNode = state.nodeById.get(edge.target);
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const distance = Math.max(1, Math.hypot(dx, dy));
  const ux = dx / distance;
  const uy = dy / distance;
  const startRadius = nodeRadius(sourceNode) + 4;
  const endRadius = nodeRadius(targetNode) + 11;
  const start = { x: source.x + ux * startRadius, y: source.y + uy * startRadius };
  const end = { x: target.x - ux * endRadius, y: target.y - uy * endRadius };
  const curveDirection = edge.relation === "MITIGATED_BY" ? 1 : -1;
  const curvature = Math.min(24, distance * 0.07) * curveDirection;
  const control = {
    x: (start.x + end.x) / 2 - uy * curvature,
    y: (start.y + end.y) / 2 + ux * curvature,
  };
  const middle = {
    x: 0.25 * start.x + 0.5 * control.x + 0.25 * end.x,
    y: 0.25 * start.y + 0.5 * control.y + 0.25 * end.y,
  };
  return {
    path: `M ${start.x} ${start.y} Q ${control.x} ${control.y} ${end.x} ${end.y}`,
    middle,
  };
}

function updateViewport() {
  byId("kg-viewport").setAttribute(
    "transform",
    `translate(${state.transform.x} ${state.transform.y}) scale(${state.transform.k})`,
  );
}

function updateGeometry() {
  state.nodeElements.forEach((element, nodeId) => {
    const position = state.positions.get(nodeId);
    element.setAttribute("transform", `translate(${position.x} ${position.y})`);
  });
  state.edgeElements.forEach(({ edge, path, label }) => {
    const geometry = edgeGeometry(edge);
    path.setAttribute("d", geometry.path);
    label.setAttribute("transform", `translate(${geometry.middle.x} ${geometry.middle.y})`);
  });
}

function createEdge(edge, index) {
  const edgeClass = RELATION_CLASS[edge.relation] || "unknown";
  const path = svgElement("path", {
    class: `kg-edge ${edgeClass}`,
    "data-edge-index": index,
    tabindex: 0,
    role: "button",
    "aria-label": `${edge.relation}: ${state.nodeById.get(edge.source)?.label_vi} đến ${state.nodeById.get(edge.target)?.label_vi}`,
  });
  const label = svgElement("g", {
    class: `kg-edge-label ${edgeClass}`,
    "data-edge-index": index,
    tabindex: 0,
    role: "button",
  });
  const width = Math.max(48, edge.relation.length * 5.2 + 13);
  const background = svgElement("rect", { x: -width / 2, y: -10, width, height: 20 });
  const text = svgElement("text", { x: 0, y: 1 });
  text.textContent = edge.relation;
  label.append(background, text);

  const select = (event) => {
    event.stopPropagation();
    selectEdge(index);
  };
  path.addEventListener("click", select);
  label.addEventListener("click", select);
  [path, label].forEach((element) => element.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") select(event);
  }));
  byId("kg-edges").append(path);
  byId("kg-edge-labels").append(label);
  state.edgeElements.push({ edge, path, label, index });
}

function createNode(node) {
  const typeClass = String(node.type || "unknown").replaceAll("_", "-");
  const isFocus = node.id === state.graph.focus_node;
  const group = svgElement("g", {
    class: `kg-node ${typeClass}${isFocus ? " focus" : ""}`,
    "data-node-id": node.id,
    tabindex: 0,
    role: "button",
    "aria-label": `${node.label_vi}, ${TYPE_LABELS[node.type] || node.type}`,
  });
  const circle = svgElement("circle", { r: nodeRadius(node) });
  group.append(circle);

  const lines = labelLines(node.label_vi, isFocus ? 16 : 15);
  const title = svgElement("text", { class: "kg-node-title", y: lines.length === 1 ? -1 : -6 });
  lines.forEach((line, index) => {
    const tspan = svgElement("tspan", { x: 0, dy: index === 0 ? 0 : 11 });
    tspan.textContent = line;
    title.append(tspan);
  });
  const type = svgElement("text", {
    class: "kg-node-type",
    x: 0,
    y: isFocus ? 22 : 24,
  });
  type.textContent = isFocus ? "POLLUTANT" : (TYPE_LABELS[node.type] || node.type);
  group.append(title, type);

  group.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.stopPropagation();
    state.dragNodeId = node.id;
    state.pointerMoved = false;
    group.setPointerCapture(event.pointerId);
  });
  group.addEventListener("click", (event) => {
    event.stopPropagation();
    if (!state.pointerMoved) selectNode(node.id);
  });
  group.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectNode(node.id);
    }
  });
  byId("kg-nodes").append(group);
  state.nodeElements.set(node.id, group);
}

function renderGraph() {
  byId("kg-edges").replaceChildren();
  byId("kg-edge-labels").replaceChildren();
  byId("kg-nodes").replaceChildren();
  state.nodeElements.clear();
  state.edgeElements = [];
  state.graph.edges.forEach(createEdge);
  state.graph.nodes.forEach(createNode);
  updateGeometry();
  applyVisibility();
}

function connectedVisibleNodeIds() {
  const visible = new Set([state.graph.focus_node]);
  state.graph.edges.forEach((edge) => {
    if (state.activeRelations.has(edge.relation)) {
      visible.add(edge.source);
      visible.add(edge.target);
    }
  });
  return visible;
}

function applyVisibility() {
  if (!state.graph) return;
  const visibleNodes = connectedVisibleNodeIds();
  const query = normalise(byId("kg-search-input").value.trim());
  const matches = new Set();
  if (query) {
    state.graph.nodes.forEach((node) => {
      const searchable = normalise([node.id, node.label_vi, node.label_en, node.description_vi].join(" "));
      if (searchable.includes(query)) matches.add(node.id);
    });
  }
  const relevant = new Set(matches);
  if (matches.size) {
    state.graph.edges.forEach((edge) => {
      if (matches.has(edge.source) || matches.has(edge.target)) {
        relevant.add(edge.source);
        relevant.add(edge.target);
      }
    });
  }

  state.nodeElements.forEach((element, nodeId) => {
    element.classList.toggle("hidden", !visibleNodes.has(nodeId));
    element.classList.toggle("dimmed", Boolean(query) && !relevant.has(nodeId));
    element.classList.toggle("selected", state.selected?.type === "node" && state.selected.id === nodeId);
  });
  state.edgeElements.forEach(({ edge, path, label, index }) => {
    const hidden = !state.activeRelations.has(edge.relation);
    const dimmed = Boolean(query) && !(relevant.has(edge.source) && relevant.has(edge.target));
    const selected = state.selected?.type === "edge" && state.selected.index === index;
    [path, label].forEach((element) => {
      element.classList.toggle("hidden", hidden);
      element.classList.toggle("dimmed", dimmed);
      element.classList.toggle("selected", selected);
    });
  });

  if (query && matches.size === 1) {
    const nodeId = [...matches][0];
    selectNode(nodeId, { preserveFilter: true });
  }
}

function relationSummary(edge) {
  const source = state.nodeById.get(edge.source)?.label_vi || edge.source;
  const target = state.nodeById.get(edge.target)?.label_vi || edge.target;
  return `${source} → ${target}`;
}

function appendMeta(container, values) {
  container.replaceChildren();
  values.filter(Boolean).forEach((value) => {
    const chip = document.createElement("span");
    chip.textContent = value;
    container.append(chip);
  });
}

function renderRelationCards(edges) {
  const container = byId("kg-inspector-relations");
  container.replaceChildren();
  if (!edges.length) {
    const empty = document.createElement("p");
    empty.className = "kg-no-data";
    empty.textContent = "Không có quan hệ liên quan trong bộ lọc hiện tại.";
    container.append(empty);
    return;
  }
  edges.forEach((edge) => {
    const card = document.createElement("div");
    card.className = `kg-relation-card ${RELATION_CLASS[edge.relation] || ""}`;
    const title = document.createElement("b");
    title.textContent = `${edge.relation} · ${relationSummary(edge)}`;
    const statement = document.createElement("span");
    statement.textContent = edge.statement_vi;
    card.append(title, statement);
    container.append(card);
  });
}

function renderSources(sourceIds) {
  const container = byId("kg-inspector-sources");
  container.replaceChildren();
  const sources = [...new Set(sourceIds)].map((id) => state.sourceById.get(id)).filter(Boolean);
  if (!sources.length) {
    const empty = document.createElement("p");
    empty.className = "kg-no-data";
    empty.textContent = "Không có nguồn tham chiếu trong phần tử này.";
    container.append(empty);
    return;
  }
  sources.forEach((source) => {
    const link = document.createElement("a");
    link.className = "kg-source-item";
    link.href = source.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    const title = document.createElement("strong");
    title.textContent = source.title;
    const organisation = document.createElement("span");
    organisation.textContent = `${source.organization} ↗`;
    link.append(title, organisation);
    container.append(link);
  });
}

function showInspector() {
  byId("kg-inspector-empty").classList.add("hidden");
  byId("kg-inspector-content").classList.remove("hidden");
}

function selectNode(nodeId, { preserveFilter = false } = {}) {
  const node = state.nodeById.get(nodeId);
  if (!node) return;
  state.selected = { type: "node", id: nodeId };
  showInspector();
  byId("kg-inspector-type").textContent = TYPE_LABELS[node.type] || "NODE";
  byId("kg-inspector-title").textContent = node.label_vi;
  byId("kg-inspector-description").textContent = node.description_vi || DEFAULT_DESCRIPTIONS[node.type] || "Thực thể trong Knowledge Graph PM2.5.";
  const related = state.graph.edges.filter((edge) => edge.source === nodeId || edge.target === nodeId);
  appendMeta(byId("kg-inspector-meta"), [node.label_en, `${related.length} quan hệ`, node.id]);
  renderRelationCards(related);
  renderSources(related.flatMap((edge) => edge.source_refs || []));
  if (!preserveFilter) applyVisibility();
  else {
    state.nodeElements.forEach((element, id) => element.classList.toggle("selected", id === nodeId));
  }
}

function selectEdge(index) {
  const edge = state.graph.edges[index];
  if (!edge) return;
  state.selected = { type: "edge", index };
  showInspector();
  byId("kg-inspector-type").textContent = edge.relation;
  byId("kg-inspector-title").textContent = relationSummary(edge);
  byId("kg-inspector-description").textContent = edge.statement_vi;
  appendMeta(byId("kg-inspector-meta"), [
    edge.claim_scope || edge.mitigation_scope,
    edge.event_claim_allowed === false ? "Không kết luận nhân quả sự kiện" : null,
  ]);
  renderRelationCards([edge]);
  renderSources(edge.source_refs || []);
  applyVisibility();
}

function clearSelection() {
  state.selected = null;
  byId("kg-inspector-empty").classList.remove("hidden");
  byId("kg-inspector-content").classList.add("hidden");
  applyVisibility();
}

function worldPoint(event) {
  const rect = byId("kg-network").getBoundingClientRect();
  return {
    x: (event.clientX - rect.left - state.transform.x) / state.transform.k,
    y: (event.clientY - rect.top - state.transform.y) / state.transform.k,
  };
}

function fitGraph() {
  if (!state.positions.size) return;
  const svg = byId("kg-network");
  const width = svg.clientWidth || 900;
  const height = svg.clientHeight || 640;
  const points = [...state.positions.values()];
  const minX = Math.min(...points.map((point) => point.x)) - 70;
  const maxX = Math.max(...points.map((point) => point.x)) + 70;
  const minY = Math.min(...points.map((point) => point.y)) - 70;
  const maxY = Math.max(...points.map((point) => point.y)) + 70;
  const scale = Math.min(width / (maxX - minX), height / (maxY - minY), 1.25) * 0.92;
  state.transform = {
    k: scale,
    x: width / 2 - ((minX + maxX) / 2) * scale,
    y: height / 2 - ((minY + maxY) / 2) * scale,
  };
  updateViewport();
}

function zoomBy(factor, clientX = null, clientY = null) {
  const svg = byId("kg-network");
  const rect = svg.getBoundingClientRect();
  const px = clientX === null ? rect.width / 2 : clientX - rect.left;
  const py = clientY === null ? rect.height / 2 : clientY - rect.top;
  const previous = state.transform.k;
  const next = Math.min(2.8, Math.max(0.35, previous * factor));
  const worldX = (px - state.transform.x) / previous;
  const worldY = (py - state.transform.y) / previous;
  state.transform.x = px - worldX * next;
  state.transform.y = py - worldY * next;
  state.transform.k = next;
  updateViewport();
}

function bindInteractions() {
  const svg = byId("kg-network");
  svg.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || state.dragNodeId) return;
    state.panStart = {
      clientX: event.clientX,
      clientY: event.clientY,
      x: state.transform.x,
      y: state.transform.y,
    };
    state.pointerMoved = false;
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener("pointermove", (event) => {
    if (state.dragNodeId) {
      const point = worldPoint(event);
      const position = state.positions.get(state.dragNodeId);
      if (Math.hypot(point.x - position.x, point.y - position.y) > 1) state.pointerMoved = true;
      state.positions.set(state.dragNodeId, point);
      updateGeometry();
      return;
    }
    if (state.panStart) {
      const dx = event.clientX - state.panStart.clientX;
      const dy = event.clientY - state.panStart.clientY;
      if (Math.hypot(dx, dy) > 2) state.pointerMoved = true;
      state.transform.x = state.panStart.x + dx;
      state.transform.y = state.panStart.y + dy;
      updateViewport();
    }
  });
  const finishPointer = (event) => {
    if (event.target.hasPointerCapture?.(event.pointerId)) {
      event.target.releasePointerCapture(event.pointerId);
    }
    state.dragNodeId = null;
    state.panStart = null;
  };
  svg.addEventListener("pointerup", finishPointer);
  svg.addEventListener("pointercancel", finishPointer);
  svg.addEventListener("click", () => {
    if (!state.pointerMoved) clearSelection();
  });
  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    zoomBy(event.deltaY < 0 ? 1.12 : 0.89, event.clientX, event.clientY);
  }, { passive: false });

  byId("kg-zoom-in").addEventListener("click", () => zoomBy(1.2));
  byId("kg-zoom-out").addEventListener("click", () => zoomBy(0.82));
  byId("kg-fit").addEventListener("click", fitGraph);
  byId("kg-reset").addEventListener("click", () => {
    layoutGraph();
    updateGeometry();
    fitGraph();
    clearSelection();
  });
  byId("kg-search-input").addEventListener("input", applyVisibility);
  document.querySelectorAll("[data-relation]").forEach((button) => {
    button.addEventListener("click", () => {
      const relation = button.dataset.relation;
      if (state.activeRelations.has(relation)) state.activeRelations.delete(relation);
      else state.activeRelations.add(relation);
      button.classList.toggle("active", state.activeRelations.has(relation));
      button.setAttribute("aria-pressed", String(state.activeRelations.has(relation)));
      applyVisibility();
    });
  });
  window.addEventListener("resize", fitGraph);
}

function updateSummary() {
  byId("kg-node-count").textContent = state.graph.nodes.length;
  byId("kg-edge-count").textContent = state.graph.edges.length;
  byId("kg-source-count").textContent = state.graph.sources.length;
  byId("kg-disclaimer").textContent = state.graph.disclaimer_vi;
}

async function initialise() {
  bindInteractions();
  try {
    const payload = await apiFetch("/api/knowledge-graph/pm25");
    state.graph = payload.graph || payload;
    if (!Array.isArray(state.graph.nodes) || !Array.isArray(state.graph.edges)) {
      throw new Error("Dữ liệu graph không đúng định dạng");
    }
    state.nodeById = new Map(state.graph.nodes.map((node) => [node.id, node]));
    state.sourceById = new Map((state.graph.sources || []).map((source) => [source.id, source]));
    layoutGraph();
    renderGraph();
    updateSummary();
    byId("kg-canvas-status").classList.add("hidden");
    window.requestAnimationFrame(fitGraph);
  } catch (error) {
    const status = byId("kg-canvas-status");
    status.textContent = `Không tải được Knowledge Graph: ${error.message}`;
    status.classList.remove("hidden");
  }
}

document.addEventListener("DOMContentLoaded", initialise);
