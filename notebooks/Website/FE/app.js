/* app.js – upload page logic */

const API_BASE = "nlp-ner-production.up.railway.app"; // change to prod URL when deployed

const fileInput   = document.getElementById("fileInput");
const dropZone    = document.getElementById("dropZone");
const dropLabel   = document.getElementById("dropLabel");
const fileChosen  = document.getElementById("fileChosen");
const analyzeBtn  = document.getElementById("analyzeBtn");
const loadingOverlay = document.getElementById("loadingOverlay");
const popupOverlay   = document.getElementById("popupOverlay");
const popupMessage   = document.getElementById("popupMessage");
const popupClose     = document.getElementById("popupClose");

let selectedFile = null;

/* ── File selection ─────────────────────────── */
function onFileSelected(file) {
  if (!file) return;
  selectedFile = file;
  fileChosen.textContent = file.name;
  dropLabel.textContent = "File ready";
  dropZone.classList.add("has-file");
  analyzeBtn.disabled = false;
}

fileInput.addEventListener("change", () => onFileSelected(fileInput.files[0]));

/* ── Drag & drop ────────────────────────────── */
dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) {
    fileInput.files = e.dataTransfer.files; // sync input
    onFileSelected(file);
  }
});

/* ── Error popup ────────────────────────────── */
function showError(msg) {
  popupMessage.textContent = msg;
  popupOverlay.hidden = false;
}
popupClose.addEventListener("click", () => popupOverlay.hidden = true);

/* ── Analyze ────────────────────────────────── */
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  loadingOverlay.hidden = false;
  analyzeBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
    });

    const json = await res.json();

    if (!res.ok || json.error) {
      throw new Error(json.error || `Server error ${res.status}`);
    }

    // Store result and filename, then navigate
    sessionStorage.setItem("nerResult", JSON.stringify(json.data));
    sessionStorage.setItem("nerFilename", selectedFile.name);
    window.location.href = "results.html";

  } catch (err) {
    loadingOverlay.hidden = true;
    analyzeBtn.disabled = false;
    showError(err.message || "Could not connect to the server. Make sure the backend is running.");
  }
});
