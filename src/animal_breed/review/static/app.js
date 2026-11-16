const state = {
  records: [],
  filtered: [],
  activeIndex: 0,
  image: null,
  imageScale: 1,
  dragging: null,
  dragStart: null,
  recordMap: new Map(),
};

const reviewImage = document.getElementById("reviewImage");
const canvas = document.getElementById("reviewCanvas");
const ctx = canvas.getContext("2d");
const statusFilter = document.getElementById("statusFilter");
const searchInput = document.getElementById("searchInput");
const recordList = document.getElementById("recordList");
const metaBlock = document.getElementById("metaBlock");
const summaryText = document.getElementById("summaryText");
const saveStatus = document.getElementById("saveStatus");
const canvasWrap = document.querySelector(".canvas-wrap");
const canvasStage = document.getElementById("canvasStage");
const canvasMessage = document.getElementById("canvasMessage");

const formEls = {
  reviewStatus: document.getElementById("reviewStatus"),
  reviewDecision: document.getElementById("reviewDecision"),
  reviewNotes: document.getElementById("reviewNotes"),
  boxXMin: document.getElementById("boxXMin"),
  boxYMin: document.getElementById("boxYMin"),
  boxXMax: document.getElementById("boxXMax"),
  boxYMax: document.getElementById("boxYMax"),
};

let activeImageUrl = null;
let loadToken = 0;

function activeRecord() {
  return state.filtered[state.activeIndex] || null;
}

function setCanvasMessage(message = "", isVisible = false) {
  canvasMessage.textContent = message;
  canvasMessage.classList.toggle("visible", isVisible);
}

function releaseImageUrl() {
  if (activeImageUrl) {
    URL.revokeObjectURL(activeImageUrl);
    activeImageUrl = null;
  }
}

function ensureCanvasViewport() {
  const width = Math.max(320, canvasWrap.clientWidth - 40);
  const height = Math.max(240, Math.min(window.innerHeight - 160, 720));
  canvasStage.style.width = `${width}px`;
  canvasStage.style.height = `${height}px`;
  reviewImage.style.width = `${width}px`;
  reviewImage.style.height = `${height}px`;
  canvas.width = width;
  canvas.height = height;
}

function clampBox(box) {
  const xmin = Math.min(box.xmin, box.xmax);
  const xmax = Math.max(box.xmin, box.xmax);
  const ymin = Math.min(box.ymin, box.ymax);
  const ymax = Math.max(box.ymin, box.ymax);
  return { xmin, ymin, xmax, ymax };
}

async function loadSession() {
  const response = await fetch("/api/session");
  const payload = await response.json();
  state.records = payload.records;
  state.recordMap = new Map(payload.records.map((r) => [r.candidate_id, r]));
  summaryText.textContent = `${payload.summary.total_records} records | ${payload.summary.candidates_csv}`;
  applyFilters();
}

function applyFilters() {
  const wantedStatus = statusFilter.value;
  const query = searchInput.value.trim().toLowerCase();
  state.filtered = state.records.filter((record) => {
    const statusOk = wantedStatus === "all" || record.review_status === wantedStatus;
    const haystack = [
      record.predicted_species,
      record.source_dataset,
      record.image_path,
      record.conflict_type,
      record.existing_labels_in_image,
    ]
      .join(" ")
      .toLowerCase();
    const queryOk = !query || haystack.includes(query);
    return statusOk && queryOk;
  });
  if (state.activeIndex >= state.filtered.length) {
    state.activeIndex = 0;
  }
  renderRecordList();
  loadActiveRecord();
}

function renderRecordList() {
  recordList.innerHTML = "";
  state.filtered.forEach((record, idx) => {
    const card = document.createElement("button");
    card.className = `record-card${idx === state.activeIndex ? " active" : ""}`;
    card.type = "button";
    card.innerHTML = `
      <div class="topline">
        <span>${record.predicted_species}</span>
        <span>${record.confidence.toFixed(3)}</span>
      </div>
      <div>${record.source_dataset}</div>
      <div>${record.conflict_type}</div>
      <div>${record.review_status}${record.review_decision ? " / " + record.review_decision : ""}</div>
    `;
    card.addEventListener("click", () => {
      state.activeIndex = idx;
      renderRecordList();
      loadActiveRecord();
    });
    recordList.appendChild(card);
  });
}

async function loadActiveRecord() {
  const record = activeRecord();
  if (!record) {
    metaBlock.innerHTML = "<p>No records match the current filter.</p>";
    state.image = null;
    releaseImageUrl();
    reviewImage.removeAttribute("src");
    ensureCanvasViewport();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setCanvasMessage("No records match the current filter.", true);
    return;
  }
  metaBlock.innerHTML = `
    <div><strong>ID:</strong> ${record.candidate_id}</div>
    <div><strong>Species:</strong> ${record.predicted_species}</div>
    <div><strong>Confidence:</strong> ${record.confidence.toFixed(4)}</div>
    <div><strong>Dataset:</strong> ${record.source_dataset}</div>
    <div><strong>Conflict:</strong> ${record.conflict_type}</div>
    <div><strong>Existing:</strong> ${record.existing_labels_in_image || "-"}</div>
    <div><strong>Image:</strong> ${record.image_path}</div>
  `;
  formEls.reviewStatus.value = record.review_status || "unreviewed";
  formEls.reviewDecision.value = record.review_decision || "";
  formEls.reviewNotes.value = record.review_notes || "";
  updateBoxInputs(record.review_box);
  state.image = null;
  reviewImage.removeAttribute("src");
  ensureCanvasViewport();
  drawPlaceholder();
  setCanvasMessage("Loading image...", true);

  const currentLoad = ++loadToken;
  releaseImageUrl();
  const imageUrl = `/api/image?path=${encodeURIComponent(record.image_path)}`;
  reviewImage.onload = () => {
    if (currentLoad !== loadToken) {
      return;
    }
    state.image = reviewImage;
    fitCanvas();
    draw();
    setCanvasMessage("", false);
  };
  reviewImage.onerror = () => {
    if (currentLoad !== loadToken) {
      return;
    }
    const error = new Error("Image element failed to load");
    console.error("Image load failed", record.image_path, error);
    state.image = null;
    reviewImage.removeAttribute("src");
    drawPlaceholder();
    setCanvasMessage(`Image failed to load: ${error.message || error}`, true);
  };
  reviewImage.src = imageUrl;
  if (reviewImage.complete && reviewImage.naturalWidth > 0) {
    reviewImage.onload();
  }
}

function fitCanvas() {
  if (!state.image) return;
  const imageWidth = state.image.naturalWidth || state.image.width;
  const imageHeight = state.image.naturalHeight || state.image.height;
  const maxW = Math.max(320, canvasWrap.clientWidth - 40);
  const maxH = window.innerHeight - 140;
  const scale = Math.min(maxW / imageWidth, maxH / imageHeight, 1);
  state.imageScale = scale;
  const width = Math.max(1, Math.floor(imageWidth * scale));
  const height = Math.max(1, Math.floor(imageHeight * scale));
  canvasStage.style.width = `${width}px`;
  canvasStage.style.height = `${height}px`;
  reviewImage.style.width = `${width}px`;
  reviewImage.style.height = `${height}px`;
  canvas.width = width;
  canvas.height = height;
}

function drawPlaceholder() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function scaledBox(box) {
  return {
    xmin: box.xmin * state.imageScale,
    ymin: box.ymin * state.imageScale,
    xmax: box.xmax * state.imageScale,
    ymax: box.ymax * state.imageScale,
  };
}

function draw() {
  const record = activeRecord();
  if (!record || !state.image) {
    drawPlaceholder();
    return;
  }
  drawPlaceholder();

  const original = scaledBox(record.original_box);
  ctx.strokeStyle = "#f7b267";
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 5]);
  ctx.strokeRect(original.xmin, original.ymin, original.xmax - original.xmin, original.ymax - original.ymin);

  const review = scaledBox(record.review_box);
  ctx.setLineDash([]);
  ctx.strokeStyle = "#63ff63";
  ctx.lineWidth = 3;
  ctx.strokeRect(review.xmin, review.ymin, review.xmax - review.xmin, review.ymax - review.ymin);
  drawHandles(review);
}

function drawHandles(box) {
  const size = 8;
  const points = [
    [box.xmin, box.ymin, "nw"],
    [box.xmax, box.ymin, "ne"],
    [box.xmin, box.ymax, "sw"],
    [box.xmax, box.ymax, "se"],
  ];
  ctx.fillStyle = "#63ff63";
  for (const [x, y] of points) {
    ctx.fillRect(x - size / 2, y - size / 2, size, size);
  }
}

function updateBoxInputs(box) {
  formEls.boxXMin.value = box.xmin.toFixed(1);
  formEls.boxYMin.value = box.ymin.toFixed(1);
  formEls.boxXMax.value = box.xmax.toFixed(1);
  formEls.boxYMax.value = box.ymax.toFixed(1);
}

function readBoxInputs() {
  return clampBox({
    xmin: Number(formEls.boxXMin.value),
    ymin: Number(formEls.boxYMin.value),
    xmax: Number(formEls.boxXMax.value),
    ymax: Number(formEls.boxYMax.value),
  });
}

function setReviewBox(box) {
  const record = activeRecord();
  if (!record) return;
  record.review_box = clampBox(box);
  updateBoxInputs(record.review_box);
  draw();
}

function hitTest(mouseX, mouseY, box) {
  const scaled = scaledBox(box);
  const handleRadius = 10;
  const handles = {
    nw: [scaled.xmin, scaled.ymin],
    ne: [scaled.xmax, scaled.ymin],
    sw: [scaled.xmin, scaled.ymax],
    se: [scaled.xmax, scaled.ymax],
  };
  for (const [name, [hx, hy]] of Object.entries(handles)) {
    const dx = mouseX - hx;
    const dy = mouseY - hy;
    if (Math.hypot(dx, dy) <= handleRadius) return name;
  }
  if (
    mouseX >= scaled.xmin &&
    mouseX <= scaled.xmax &&
    mouseY >= scaled.ymin &&
    mouseY <= scaled.ymax
  ) {
    return "move";
  }
  return null;
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

canvas.addEventListener("mousedown", (event) => {
  const record = activeRecord();
  if (!record) return;
  const point = canvasPoint(event);
  const dragMode = hitTest(point.x, point.y, record.review_box);
  if (!dragMode) return;
  state.dragging = dragMode;
  state.dragStart = {
    point,
    box: { ...record.review_box },
  };
});

window.addEventListener("mousemove", (event) => {
  const record = activeRecord();
  if (!record || !state.dragging || !state.dragStart) return;
  const point = canvasPoint(event);
  const dx = (point.x - state.dragStart.point.x) / state.imageScale;
  const dy = (point.y - state.dragStart.point.y) / state.imageScale;
  const box = { ...state.dragStart.box };
  if (state.dragging === "move") {
    box.xmin += dx;
    box.xmax += dx;
    box.ymin += dy;
    box.ymax += dy;
  } else if (state.dragging === "nw") {
    box.xmin += dx;
    box.ymin += dy;
  } else if (state.dragging === "ne") {
    box.xmax += dx;
    box.ymin += dy;
  } else if (state.dragging === "sw") {
    box.xmin += dx;
    box.ymax += dy;
  } else if (state.dragging === "se") {
    box.xmax += dx;
    box.ymax += dy;
  }
  setReviewBox(box);
});

window.addEventListener("mouseup", () => {
  state.dragging = null;
  state.dragStart = null;
});

for (const key of ["boxXMin", "boxYMin", "boxXMax", "boxYMax"]) {
  formEls[key].addEventListener("change", () => setReviewBox(readBoxInputs()));
}

document.getElementById("reviewForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const record = activeRecord();
  if (!record) return;
  record.review_status = formEls.reviewStatus.value;
  record.review_decision = formEls.reviewDecision.value;
  record.review_notes = formEls.reviewNotes.value;
  const payload = {
    candidate_id: record.candidate_id,
    review_status: record.review_status,
    review_decision: record.review_decision,
    review_notes: record.review_notes,
    review_box: record.review_box,
  };
  const response = await fetch("/api/review", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (result.ok) {
    saveStatus.textContent = `Saved ${record.candidate_id}`;
    renderRecordList();
  } else {
    saveStatus.textContent = result.error || "Save failed";
  }
});

document.getElementById("prevBtn").addEventListener("click", () => {
  if (!state.filtered.length) return;
  state.activeIndex = (state.activeIndex - 1 + state.filtered.length) % state.filtered.length;
  renderRecordList();
  loadActiveRecord();
});

document.getElementById("nextBtn").addEventListener("click", () => {
  if (!state.filtered.length) return;
  state.activeIndex = (state.activeIndex + 1) % state.filtered.length;
  renderRecordList();
  loadActiveRecord();
});

document.getElementById("resetBoxBtn").addEventListener("click", () => {
  const record = activeRecord();
  if (!record) return;
  setReviewBox({ ...record.original_box });
});

document.getElementById("fitBtn").addEventListener("click", () => {
  fitCanvas();
  draw();
});

statusFilter.addEventListener("change", applyFilters);
searchInput.addEventListener("input", applyFilters);
window.addEventListener("resize", () => {
  if (state.image) {
    fitCanvas();
  } else {
    ensureCanvasViewport();
  }
  draw();
});

ensureCanvasViewport();
drawPlaceholder();
loadSession().catch((error) => {
  summaryText.textContent = `Failed to load session: ${error}`;
  ensureCanvasViewport();
  drawPlaceholder();
  setCanvasMessage(`Failed to load session: ${error}`, true);
});
