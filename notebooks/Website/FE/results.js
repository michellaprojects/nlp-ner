/* results.js – render NER results */

// Label config: entity key → display name + render style
const ENTITY_CONFIG = [
  { key: "name",                label: "Name",                style: "items" },
  { key: "designation",         label: "Designation",         style: "items" },
  { key: "companies_worked_at", label: "Companies Worked At", style: "items" },
  { key: "location",            label: "Location",            style: "items" },
  { key: "email",               label: "Email Address",       style: "items" },
  { key: "college_name",        label: "College Name",        style: "items" },
  { key: "degree",              label: "Degree",              style: "items" },
  { key: "graduation_year",     label: "Graduation Year",     style: "items" },
  { key: "skills",              label: "Skills",              style: "pills" },
  { key: "years_of_experience", label: "Years of Experience", style: "items" },
];

const grid     = document.getElementById("resultsGrid");
const popupOverlay  = document.getElementById("popupOverlay");
const filenameLabel = document.getElementById("filenameLabel");
// const copyBtn       = document.getElementById("copyBtn");

/* ── Load data ──────────────────────────────── */
const raw = sessionStorage.getItem("nerResult");
const filename = sessionStorage.getItem("nerFilename") || "";

if (!raw || raw === "undefined") {
  popupOverlay.hidden = false;
} else {
  const data = JSON.parse(raw);
  filenameLabel.textContent = `Source: ${filename}`;
  renderCards(data);
}

/* ── Render ─────────────────────────────────── */
function renderCards(data) {
  // Render known entities in order
  ENTITY_CONFIG.forEach(({ key, label, style }, i) => {
    const values = data[key];
    if (!values || values.length === 0) return;
    grid.appendChild(buildCard(label, values, style, i));
  });

  // Render any extra keys not in ENTITY_CONFIG
  Object.keys(data).forEach((key, i) => {
    const known = ENTITY_CONFIG.some(e => e.key === key);
    if (!known && data[key]?.length) {
      const label = key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
      grid.appendChild(buildCard(label, data[key], "items", ENTITY_CONFIG.length + i));
    }
  });
}

function buildCard(label, values, style, index) {
  const card = document.createElement("div");
  card.className = "result-card";
  card.style.animationDelay = `${index * 0.05}s`;

  const lbl = document.createElement("div");
  lbl.className = "result-card-label";
  lbl.textContent = label;
  card.appendChild(lbl);

  if (style === "pills") {
    const pillBox = document.createElement("div");
    pillBox.className = "result-pills";
    values.forEach(v => {
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = v;
      pillBox.appendChild(pill);
    });
    card.appendChild(pillBox);
  } else {
    const itemBox = document.createElement("div");
    itemBox.className = "result-items";
    values.forEach(v => {
      const item = document.createElement("div");
      item.className = "result-item";
      item.textContent = v;
      itemBox.appendChild(item);
    });
    card.appendChild(itemBox);
  }

  return card;
}

// /* ── Copy JSON ──────────────────────────────── */
// copyBtn.addEventListener("click", async () => {
//   try {
//     await navigator.clipboard.writeText(
//       JSON.stringify(JSON.parse(raw), null, 2)
//     );
//     copyBtn.textContent = "Copied!";
//     setTimeout(() => copyBtn.textContent = "Copy JSON", 2000);
//   } catch {
//     copyBtn.textContent = "Copy failed";
//   }
// });
