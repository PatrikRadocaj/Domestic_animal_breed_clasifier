const state = {
  file: null,
  status: null,
};

const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const runBtn = document.getElementById("runBtn");
const selectedFileName = document.getElementById("selectedFileName");
const statusBlock = document.getElementById("statusBlock");
const resultSummary = document.getElementById("resultSummary");
const resultImage = document.getElementById("resultImage");
const placeholder = document.getElementById("placeholder");
const detectionList = document.getElementById("detectionList");

function setSelectedFile(file) {
  state.file = file || null;
  selectedFileName.textContent = file ? file.name : "No file selected";
}

function setBusy(isBusy) {
  runBtn.disabled = isBusy;
  runBtn.textContent = isBusy ? "Running..." : "Run Inference";
}

function showError(message) {
  detectionList.innerHTML = `<div class="error-banner">${message}</div>`;
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Failed to read selected file."));
    reader.readAsDataURL(file);
  });
}

function renderStatus(status) {
  state.status = status;
  const breedSpecies = status.breed_model_species.length
    ? status.breed_model_species.join(", ")
    : "none";
  statusBlock.innerHTML = `
    <div class="status-item">
      <strong>Detector</strong>
      <span>${status.detector_weights}</span>
    </div>
    <div class="status-item">
      <strong>Device</strong>
      <span>${status.device}</span>
    </div>
    <div class="status-item">
      <strong>Breed Models</strong>
      <span>${status.breed_model_count} species</span>
    </div>
    <div class="status-item">
      <strong>Breed-Capable Species</strong>
      <span>${breedSpecies}</span>
    </div>
  `;
}

function renderDetections(result) {
  if (!result.detections.length) {
    detectionList.innerHTML = `<div class="error-banner">No animals were detected in this image.</div>`;
    return;
  }

  detectionList.innerHTML = result.detections
    .map((detection) => {
      const breed = detection.breed_prediction;
      const altGuesses = breed ? breed.topk.slice(1, 3) : [];
      const topk = breed
        ? breed.topk
            .map((item) => `<span class="pill">${item.display_label} ${item.confidence.toFixed(2)}</span>`)
            .join("")
        : `<span class="pill warn">No breed classifier for ${detection.species_display}</span>`;
      const altGuessBlock = altGuesses.length
        ? `
          <div class="ranked-guesses">
            <strong>Other guesses</strong>
            <ol>
              ${altGuesses
                .map(
                  (item) =>
                    `<li>${item.display_label} <span>${item.confidence.toFixed(3)}</span></li>`,
                )
                .join("")}
            </ol>
          </div>
        `
        : "";

      return `
        <article class="detection-card">
          <h3>#${detection.index} ${detection.species_display}</h3>
          <p>
            Species confidence: <strong>${detection.species_confidence.toFixed(3)}</strong>
            ${
              breed
                ? `<br />Top breed: <strong>${breed.display_label}</strong> (${breed.confidence.toFixed(3)})`
                : ""
            }
          </p>
          <div class="detection-meta">
            <span>BBox: ${detection.bbox_xyxy.map((value) => value.toFixed(1)).join(", ")}</span>
            <span>Breed model: ${detection.breed_model_available ? "available" : "missing"}</span>
          </div>
          ${altGuessBlock}
          <div class="pill-row">${topk}</div>
        </article>
      `;
    })
    .join("");
}

async function loadStatus() {
  const response = await fetch("/api/status");
  const payload = await response.json();
  if (!payload.ok) {
    throw new Error("Failed to load model status.");
  }
  renderStatus(payload.status);
}

async function runInference() {
  if (!state.file) {
    showError("Select an image before running inference.");
    return;
  }

  setBusy(true);
  detectionList.innerHTML = "";
  resultSummary.textContent = "Running detector and classifiers...";

  try {
    const imageBase64 = await fileToDataUrl(state.file);
    const response = await fetch("/api/infer", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        filename: state.file.name,
        image_base64: imageBase64,
      }),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Inference request failed.");
    }

    const result = payload.result;
    resultImage.src = result.annotated_image_url;
    resultImage.classList.add("visible");
    placeholder.classList.add("hidden");
    resultSummary.textContent = `${result.detection_count} detections | species: ${
      result.detected_species.join(", ") || "none"
    }`;
    renderDetections(result);
  } catch (error) {
    resultImage.removeAttribute("src");
    resultImage.classList.remove("visible");
    placeholder.classList.remove("hidden");
    resultSummary.textContent = "Inference failed.";
    showError(error.message || String(error));
  } finally {
    setBusy(false);
  }
}

function bindDropzone() {
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
    const [file] = event.dataTransfer.files;
    if (file) {
      setSelectedFile(file);
    }
  });
}

fileInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) {
    setSelectedFile(file);
  }
});

runBtn.addEventListener("click", runInference);
bindDropzone();
loadStatus().catch((error) => {
  showError(error.message || String(error));
});
