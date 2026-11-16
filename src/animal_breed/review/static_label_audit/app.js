const state = {
  groups: [],
  filtered: [],
  activeIndex: 0,
};

const summaryText = document.getElementById("summaryText");
const statusFilter = document.getElementById("statusFilter");
const searchInput = document.getElementById("searchInput");
const groupList = document.getElementById("groupList");
const metaBlock = document.getElementById("metaBlock");
const sheetImage = document.getElementById("sheetImage");
const sampleGrid = document.getElementById("sampleGrid");
const saveStatus = document.getElementById("saveStatus");

const formEls = {
  reviewStatus: document.getElementById("reviewStatus"),
  reviewDecision: document.getElementById("reviewDecision"),
  proposedSpecies: document.getElementById("proposedSpecies"),
  reviewNotes: document.getElementById("reviewNotes"),
};

function activeGroup() {
  return state.filtered[state.activeIndex] || null;
}

async function loadSession() {
  const response = await fetch("/api/session");
  const payload = await response.json();
  state.groups = payload.groups;
  summaryText.textContent = `${payload.summary.total_groups} label groups | ${payload.summary.index_csv}`;
  applyFilters();
}

function applyFilters() {
  const wantedStatus = statusFilter.value;
  const query = searchInput.value.trim().toLowerCase();
  state.filtered = state.groups.filter((group) => {
    const statusOk = wantedStatus === "all" || group.review_status === wantedStatus;
    const haystack = [
      group.dataset_id,
      group.label_name,
      group.review_decision,
      group.proposed_species,
      group.review_notes,
    ].join(" ").toLowerCase();
    return statusOk && (!query || haystack.includes(query));
  });
  if (state.activeIndex >= state.filtered.length) {
    state.activeIndex = 0;
  }
  renderGroupList();
  loadActiveGroup();
}

function renderGroupList() {
  groupList.innerHTML = "";
  for (const [idx, group] of state.filtered.entries()) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `record-card${idx === state.activeIndex ? " active" : ""}`;
    card.innerHTML = `
      <div class="topline">
        <span>${group.label_name}</span>
        <span>${group.sample_count}</span>
      </div>
      <div>${group.dataset_id}</div>
      <div>${group.review_status}${group.review_decision ? " / " + group.review_decision : ""}</div>
    `;
    card.addEventListener("click", () => {
      state.activeIndex = idx;
      renderGroupList();
      loadActiveGroup();
    });
    groupList.appendChild(card);
  }
}

function fileUrl(path) {
  return `/api/file?path=${encodeURIComponent(path)}`;
}

function loadActiveGroup() {
  const group = activeGroup();
  if (!group) {
    metaBlock.innerHTML = "<p>No label groups match the current filter.</p>";
    sheetImage.removeAttribute("src");
    sampleGrid.innerHTML = "";
    return;
  }
  metaBlock.innerHTML = `
    <div><strong>Dataset:</strong> ${group.dataset_id}</div>
    <div><strong>Label:</strong> ${group.label_name}</div>
    <div><strong>Samples:</strong> ${group.sample_count}</div>
  `;
  formEls.reviewStatus.value = group.review_status || "unreviewed";
  formEls.reviewDecision.value = group.review_decision || "";
  formEls.proposedSpecies.value = group.proposed_species || "";
  formEls.reviewNotes.value = group.review_notes || "";

  sheetImage.src = fileUrl(group.contact_sheet_path);
  sampleGrid.innerHTML = "";
  for (const cropPath of group.crop_paths) {
    const card = document.createElement("div");
    card.className = "sample-card";
    card.innerHTML = `<img alt="${group.label_name}" src="${fileUrl(cropPath)}" />`;
    sampleGrid.appendChild(card);
  }
}

document.getElementById("reviewForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const group = activeGroup();
  if (!group) return;
  group.review_status = formEls.reviewStatus.value;
  group.review_decision = formEls.reviewDecision.value;
  group.proposed_species = formEls.proposedSpecies.value.trim();
  group.review_notes = formEls.reviewNotes.value;
  const response = await fetch("/api/review", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      group_id: group.group_id,
      review_status: group.review_status,
      review_decision: group.review_decision,
      proposed_species: group.proposed_species,
      review_notes: group.review_notes,
    }),
  });
  const result = await response.json();
  saveStatus.textContent = result.ok ? `Saved ${group.label_name}` : (result.error || "Save failed");
  renderGroupList();
});

document.getElementById("prevBtn").addEventListener("click", () => {
  if (!state.filtered.length) return;
  state.activeIndex = (state.activeIndex - 1 + state.filtered.length) % state.filtered.length;
  renderGroupList();
  loadActiveGroup();
});

document.getElementById("nextBtn").addEventListener("click", () => {
  if (!state.filtered.length) return;
  state.activeIndex = (state.activeIndex + 1) % state.filtered.length;
  renderGroupList();
  loadActiveGroup();
});

statusFilter.addEventListener("change", applyFilters);
searchInput.addEventListener("input", applyFilters);

loadSession().catch((error) => {
  summaryText.textContent = `Failed to load audit session: ${error}`;
});
