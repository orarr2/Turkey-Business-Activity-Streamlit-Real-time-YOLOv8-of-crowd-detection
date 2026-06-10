// Live dashboard: subscribes to Firestore and updates in real time (no polling).
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
import {
  getFirestore, collection, onSnapshot, query, where, orderBy, limit,
} from "https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js";
import { firebaseConfig } from "./firebase-config.js";

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const statusEl = document.getElementById("status");

let selectedCam = null;       // cam_id currently charted
let chart = null;
let chartUnsub = null;        // detach previous history listener when switching cameras

// ---- Live KPI cards: one doc per camera in `latest`, overwritten each sample ----
onSnapshot(collection(db, "latest"), (snap) => {
  statusEl.innerHTML = '<span class="live">● live</span> · updated ' +
    new Date().toLocaleTimeString();
  const cards = document.getElementById("cards");
  const docs = snap.docs.map((d) => ({ id: d.id, ...d.data() }))
    .sort((a, b) => (b.person ?? 0) - (a.person ?? 0));

  cards.innerHTML = "";
  for (const c of docs) {
    const el = document.createElement("div");
    el.className = "card" + (c.id === selectedCam ? " sel" : "");
    el.onclick = () => selectCamera(c.id, c.cam_name);
    el.innerHTML = `
      <h3>${c.cam_name ?? c.id}</h3>
      <div class="row">
        <div><div class="big">${c.person ?? "–"}</div><div class="muted">people</div></div>
        <div><div class="big veh">${c.vehicles ?? "–"}</div><div class="muted">vehicles</div></div>
      </div>
      <div class="muted">${c.ts ? new Date(c.ts).toLocaleTimeString() : ""}</div>`;
    cards.appendChild(el);
  }
  if (!selectedCam && docs.length) selectCamera(docs[0].id, docs[0].cam_name);
}, (err) => { statusEl.textContent = "error: " + err.message; });

// ---- Time-series chart for the selected camera (live, last 200 samples) ----
function selectCamera(camId, camName) {
  selectedCam = camId;
  document.getElementById("chart-title").textContent = `${camName ?? camId} — footfall (live)`;
  document.querySelectorAll(".card").forEach((c) => c.classList.remove("sel"));

  if (chartUnsub) chartUnsub();
  const q = query(
    collection(db, "footfall"),
    where("cam_id", "==", camId),
    orderBy("ts", "desc"),
    limit(200),
  );
  chartUnsub = onSnapshot(q, (snap) => {
    const rows = snap.docs.map((d) => d.data())
      .filter((r) => r.ok)
      .sort((a, b) => a.ts.localeCompare(b.ts));
    renderChart(
      rows.map((r) => new Date(r.ts).toLocaleTimeString()),
      rows.map((r) => r.person),
      rows.map((r) => r.vehicles),
    );
  });
}

function renderChart(labels, people, vehicles) {
  const data = {
    labels,
    datasets: [
      { label: "people", data: people, borderColor: "#4f8cff", tension: 0.25, pointRadius: 0 },
      { label: "vehicles", data: vehicles, borderColor: "#f0a35e", tension: 0.25, pointRadius: 0 },
    ],
  };
  if (chart) { chart.data = data; chart.update("none"); return; }
  chart = new Chart(document.getElementById("chart"), {
    type: "line",
    data,
    options: {
      responsive: true,
      scales: { x: { ticks: { color: "#8b909a", maxTicksLimit: 8 } },
                y: { ticks: { color: "#8b909a" }, beginAtZero: true } },
      plugins: { legend: { labels: { color: "#e7e9ee" } } },
    },
  });
}
