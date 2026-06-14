// 4-camera live HTML dashboard. Data lives in Firestore so it is *persistent
// across visitors*: every footfall write the collector makes accumulates there,
// and any browser visiting this page subscribes via onSnapshot for instant
// (no-polling) updates.
//
// Collections this expects (collector.py writes them):
//   latest/{cam_id}            one doc per camera, overwritten each sample
//   footfall/{...auto}         append-only history (ts, cam_id, person, vehicles, ok)
//   reid_stats/{cam_id}        per-camera unique/sightings/regulars (optional but used)

import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
import {
  getFirestore, collection, onSnapshot, query, where, orderBy, limit,
} from "https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js";
import { GRID_CAMERAS } from "./cameras.js";

// firebase-config.js is gitignored — if it's missing the dashboard still renders
// the layout but shows a config warning.
let firebaseConfig;
try {
  firebaseConfig = (await import("./firebase-config.js")).firebaseConfig;
} catch (_) { /* handled below */ }

const statusEl = document.getElementById("status");
const tilesEl  = document.getElementById("tiles");

const HISTORY_LIMIT  = 360;     // ~6h at 1 sample/min, 1.5h at 1/15s
const ANOMALY_WINDOW = 12;
const ANOMALY_Z      = 2.5;

// ---------- 1. Render tile skeletons (works even without Firebase) ----------

const tileState = {};   // cam_id -> { latestEl, chartEl, chart, anomalyEl, samplesEl, history }

for (const cam of GRID_CAMERAS) {
  const tile = document.createElement("div");
  tile.className = "tile";
  tile.innerHTML = `
    <div>
      <h2>${escapeHtml(cam.name)}</h2>
      <div class="city">${escapeHtml(cam.city ?? "")}</div>
    </div>
    <div class="video-wrap">
      ${cam.embed
        ? `<iframe src="${cam.embed}" allow="autoplay; encrypted-media"
                   allowfullscreen loading="lazy"></iframe>`
        : `<div class="video-fallback">
             Live stream not embeddable from this site —
             <a href="${cam.page}" target="_blank" rel="noopener">open camera page ↗</a>
           </div>`}
    </div>
    <div class="metrics">
      <div class="metric"><div class="lbl">People (now)</div>
        <div class="val" data-k="person">–</div></div>
      <div class="metric vehicles"><div class="lbl">Vehicles (now)</div>
        <div class="val" data-k="vehicles">–</div></div>
      <div class="metric"><div class="lbl">24h avg</div>
        <div class="val" data-k="avg">–</div></div>
      <div class="metric"><div class="lbl">24h peak</div>
        <div class="val" data-k="peak">–</div></div>
    </div>
    <div>
      <span class="anomaly-badge unk" data-anomaly>
        <span class="dot"></span><span data-anomaly-text>no data yet</span>
      </span>
      <span class="footnote" data-samples></span>
    </div>
    <canvas class="chart-mini"></canvas>
  `;
  tilesEl.appendChild(tile);

  tileState[cam.id] = {
    tile,
    latestVals: tile.querySelectorAll(".metric .val"),
    anomalyBadge: tile.querySelector("[data-anomaly]"),
    anomalyText:  tile.querySelector("[data-anomaly-text]"),
    samplesEl:    tile.querySelector("[data-samples]"),
    chartCanvas:  tile.querySelector("canvas"),
    chart: null,
    history: [],
  };
}

// ---------- 2. Bail out cleanly if Firebase isn't configured -----------------

if (!firebaseConfig) {
  document.getElementById("config-warning").style.display = "block";
  statusEl.innerHTML = `<span class="down">● firebase not configured</span>`;
} else {
  start(firebaseConfig);
}

// ---------- 3. Live subscriptions -------------------------------------------

function start(cfg) {
  const app = initializeApp(cfg);
  const db  = getFirestore(app);

  // 3a. latest/{cam_id} -> KPI cards. One snapshot covers all cameras.
  onSnapshot(collection(db, "latest"), (snap) => {
    let alive = 0;
    for (const doc of snap.docs) {
      const cam = doc.id;
      const st  = tileState[cam];
      if (!st) continue;
      const d   = doc.data();
      const ageS = d.ts ? Math.round((Date.now() - new Date(d.ts).getTime()) / 1000) : null;
      if (ageS != null && ageS < 120) alive++;
      setLatest(st, d);
    }
    statusEl.innerHTML = alive === GRID_CAMERAS.length
        ? `<span class="live">● live</span> · ${alive}/${GRID_CAMERAS.length} cameras updating`
        : alive > 0
        ? `<span class="stale">● partial</span> · ${alive}/${GRID_CAMERAS.length} cameras updating`
        : `<span class="down">● no recent writes</span> · is the collector running?`;
  }, (err) => statusEl.textContent = "error: " + err.message);

  // 3b. One history subscription per camera (last 24h).
  const since = new Date(Date.now() - 24 * 3600 * 1000).toISOString();
  for (const cam of GRID_CAMERAS) {
    const q = query(
      collection(db, "footfall"),
      where("cam_id", "==", cam.id),
      where("ts", ">=", since),
      orderBy("ts", "desc"),
      limit(HISTORY_LIMIT),
    );
    onSnapshot(q, (snap) => {
      const rows = snap.docs
          .map((d) => d.data())
          .filter((r) => r.ok)
          .sort((a, b) => a.ts.localeCompare(b.ts));
      const st = tileState[cam.id];
      st.history = rows;
      renderTileChart(cam.id, rows);
      updateAggregates(cam.id, rows);
    });
  }

  // 3c. Combined 24h chart at the bottom — rebuild from each tile's history
  //     each time anything updates. (Cheap; we have at most 4×360 rows.)
  setInterval(renderCombinedChart, 4000);
  renderCombinedChart();

  // 3d. Re-ID summary (optional collection; absent until the collector publishes it).
  onSnapshot(collection(db, "reid_stats"), (snap) => {
    renderReidTable(snap.docs.map((d) => ({ id: d.id, ...d.data() })));
  }, () => { /* collection may not exist yet */ });
}

// ---------- 4. Per-tile rendering -------------------------------------------

function setLatest(st, d) {
  const set = (k, v) => {
    const el = [...st.latestVals].find((x) => x.dataset.k === k);
    if (el) el.textContent = (v ?? v === 0) ? v : "–";
  };
  set("person",   d.person);
  set("vehicles", d.vehicles);
}

function updateAggregates(camId, rows) {
  const st = tileState[camId];
  if (!rows.length) {
    st.samplesEl.textContent = "no samples in the last 24h";
    return;
  }
  const ppl = rows.map((r) => r.person ?? 0);
  const avg  = ppl.reduce((a, b) => a + b, 0) / ppl.length;
  const peak = Math.max(...ppl);
  const setAgg = (k, v) => {
    const el = [...st.latestVals].find((x) => x.dataset.k === k);
    if (el) el.textContent = v;
  };
  setAgg("avg",  avg.toFixed(1));
  setAgg("peak", peak);

  const anomalies = flagAnomalies(rows);
  if (anomalies.length) {
    const last = anomalies[anomalies.length - 1];
    st.anomalyBadge.className = "anomaly-badge warn";
    st.anomalyText.textContent =
        `⚠ anomaly at ${fmtTime(last.ts)} — ${last.person} people (${anomalies.length} in 24h)`;
  } else {
    st.anomalyBadge.className = "anomaly-badge ok";
    st.anomalyText.textContent = "no anomalies in the last 24h";
  }
  st.samplesEl.textContent = ` · ${rows.length} samples in 24h`;
}

function renderTileChart(camId, rows) {
  const st = tileState[camId];
  const labels   = rows.map((r) => fmtTime(r.ts));
  const people   = rows.map((r) => r.person);
  const vehicles = rows.map((r) => r.vehicles);

  if (st.chart) {
    st.chart.data.labels = labels;
    st.chart.data.datasets[0].data = people;
    st.chart.data.datasets[1].data = vehicles;
    st.chart.update("none");
    return;
  }
  st.chart = new Chart(st.chartCanvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "people",   data: people,   borderColor: "#4f8cff",
          tension: 0.25, pointRadius: 0, borderWidth: 2 },
        { label: "vehicles", data: vehicles, borderColor: "#f0a35e",
          tension: 0.25, pointRadius: 0, borderWidth: 2 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      scales: {
        x: { ticks: { color: "#6f7480", maxTicksLimit: 6, font: { size: 10 } },
             grid: { color: "rgba(255,255,255,0.04)" } },
        y: { beginAtZero: true,
             ticks: { color: "#6f7480", font: { size: 10 } },
             grid: { color: "rgba(255,255,255,0.04)" } },
      },
      plugins: {
        legend: { labels: { color: "#8b909a", font: { size: 11 }, boxWidth: 8 } },
      },
    },
  });
}

// ---------- 5. Combined chart of all four cameras ---------------------------

let combinedChart = null;
function renderCombinedChart() {
  // Build a label axis from the union of timestamps, downsampled.
  const seriesByCam = {};
  let labels = [];
  for (const cam of GRID_CAMERAS) {
    const rows = tileState[cam.id].history;
    seriesByCam[cam.id] = rows;
    for (const r of rows) labels.push(r.ts);
  }
  labels = [...new Set(labels)].sort();
  const displayLabels = labels.map(fmtTime);

  const palette = ["#4f8cff", "#36d399", "#f0a35e", "#a78bfa"];
  const datasets = GRID_CAMERAS.map((cam, i) => {
    const byTs = new Map(seriesByCam[cam.id].map((r) => [r.ts, r.person]));
    return {
      label: cam.name.split(" — ")[0],
      data: labels.map((t) => byTs.has(t) ? byTs.get(t) : null),
      borderColor: palette[i % palette.length],
      tension: 0.2, pointRadius: 0, borderWidth: 2, spanGaps: true,
    };
  });

  if (!combinedChart) {
    combinedChart = new Chart(document.getElementById("chart-all"), {
      type: "line",
      data: { labels: displayLabels, datasets },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
          x: { ticks: { color: "#6f7480", maxTicksLimit: 12 },
               grid: { color: "rgba(255,255,255,0.04)" } },
          y: { beginAtZero: true, ticks: { color: "#6f7480" },
               grid: { color: "rgba(255,255,255,0.04)" } },
        },
        plugins: { legend: { labels: { color: "#e7e9ee" } } },
      },
    });
  } else {
    combinedChart.data.labels = displayLabels;
    combinedChart.data.datasets = datasets;
    combinedChart.update("none");
  }
}

// ---------- 6. Re-ID summary table ------------------------------------------

function renderReidTable(docs) {
  const wrap = document.getElementById("reid-table-wrap");
  const rows = docs.filter((d) => GRID_CAMERAS.some((c) => c.id === d.id));
  if (!rows.length) {
    wrap.innerHTML = `<div class="empty">No re-ID stats yet — the collector publishes
      them as detections come in.</div>`;
    return;
  }
  const tr = (cells) => `<tr>${cells.map((c) => `<td>${c}</td>`).join("")}</tr>`;
  const camName = (id) => (GRID_CAMERAS.find((c) => c.id === id) ?? {}).name ?? id;
  const total = (r, k) => Object.values(r.per_class ?? {}).reduce((s, p) => s + (p[k] ?? 0), 0);
  wrap.innerHTML = `
    <table class="reid">
      <thead><tr>
        <th>Camera</th><th>Unique entities</th><th>Total sightings</th>
        <th>Regulars (≥3)</th>
      </tr></thead>
      <tbody>
        ${rows.map((r) => tr([
          escapeHtml(camName(r.id)),
          r.total_unique ?? total(r, "unique") ?? "–",
          r.total_sightings ?? total(r, "total_sightings") ?? "–",
          r.regulars ?? total(r, "regulars") ?? "–",
        ])).join("")}
      </tbody>
    </table>`;
}

// ---------- helpers ---------------------------------------------------------

function flagAnomalies(rows) {
  // Rolling z-score on people counts; flag any |z| > 2.5.
  const xs = rows.map((r) => r.person ?? 0);
  const out = [];
  for (let i = ANOMALY_WINDOW; i < xs.length; i++) {
    const win = xs.slice(i - ANOMALY_WINDOW, i);
    const mu  = win.reduce((a, b) => a + b, 0) / win.length;
    const sd  = Math.sqrt(win.reduce((a, b) => a + (b - mu) ** 2, 0) / win.length);
    if (sd > 0 && Math.abs((xs[i] - mu) / sd) > ANOMALY_Z) out.push(rows[i]);
  }
  return out;
}

function fmtTime(iso) {
  try { return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }
  catch { return String(iso).slice(11, 16); }
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
