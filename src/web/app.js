// 4-slot live HTML dashboard, slot-based since the fallback refactor. Data lives
// in Firestore so it is persistent across visitors. Every visitor subscribes
// via onSnapshot; no polling.
//
// Collections this expects (cloud collector writes them):
//   config/grid            one doc; publishes the current active cam per slot
//   latest/{slot_id}       one doc per slot, overwritten each sample
//   footfall/{auto}        append-only history; each doc has a `slot` field.
//                          TTL policy on `expire_at` deletes docs after 24h.
//   reid_stats/{slot_id}   per-slot unique/sightings/regulars

import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
import {
  initializeAppCheck, ReCaptchaV3Provider,
} from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app-check.js";
import {
  getFirestore, collection, doc, onSnapshot, query, where, orderBy, limit,
} from "https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js";

// Cache-busting for sibling modules; see the old app.js comment for rationale.
const _u = new URL(import.meta.url);
const _ver = (_u.searchParams.get("v") || _u.searchParams.get("ver")
              || new URLSearchParams(location.search).get("ver") || "dev");
const _q = "?v=" + encodeURIComponent(_ver);

const { GRID_SLOTS, hlsUrlForActiveCam } = await import("./cameras.js" + _q);

let firebaseConfig;
try {
  firebaseConfig = (await import("./firebase-config.js" + _q)).firebaseConfig;
} catch (_) { /* handled below */ }

const statusEl = document.getElementById("status");
const tilesEl  = document.getElementById("tiles");

const HISTORY_LIMIT = 360;

// Anomaly gates — mirror AnomalyTracker in app/collector.py.
const ANOMALY_WINDOW    = 30;
const ANOMALY_WARMUP    = 10;
const ANOMALY_Z         = 3.5;
const ANOMALY_MIN_PEOPLE = 5;
const ANOMALY_MIN_DELTA  = 5;
const ANOMALY_MIN_STD    = 0.0;
const ANOMALY_COOLDOWN_S = 300;

// tileState is keyed by slot_id (stable across fallback changes) — the video/
// header re-renders when active_cam changes, but chart history is preserved.
const tileState = {};
let combinedChart = null;
let currentGridConfig = null;   // last config/grid doc — {slots: [...]}

// ---------- 1. Render tile skeletons -----------------------------------------

for (const slot of GRID_SLOTS) {
  const tile = document.createElement("div");
  tile.className = "tile";
  tile.dataset.slot = slot.slot_id;
  tile.innerHTML = `
    <div>
      <h2 data-cam-name>${escapeHtml(slot.placeholder_name)}</h2>
      <div class="city" data-cam-area>${escapeHtml(slot.display_area)}</div>
    </div>
    <div class="video-wrap" data-video-wrap></div>
    <div class="metrics">
      <div class="metric"><div class="lbl">People (now)</div>
        <div class="val" data-k="person">-</div></div>
      <div class="metric vehicles"><div class="lbl">Vehicles (now)</div>
        <div class="val" data-k="vehicles">-</div></div>
      <div class="metric"><div class="lbl">24h avg</div>
        <div class="val" data-k="avg">-</div></div>
      <div class="metric"><div class="lbl">24h peak</div>
        <div class="val" data-k="peak">-</div></div>
    </div>
    <div>
      <span class="activity-badge act-unknown" data-activity>
        <span class="dot"></span><span data-activity-text>activity -/10</span>
      </span>
      <span class="anomaly-badge unk" data-anomaly>
        <span class="dot"></span><span data-anomaly-text>no data yet</span>
      </span>
      <span class="fallback-badge" data-fallback style="display:none"></span>
      <a class="anomaly-thumb" data-anomaly-thumb target="_blank" rel="noopener"
         style="display:none" title="open snapshot of latest anomaly">
        <img alt="" />
      </a>
      <span class="footnote" data-samples></span>
    </div>
    <div class="chart-mini"><canvas></canvas></div>
  `;
  tilesEl.appendChild(tile);

  tileState[slot.slot_id] = {
    slot,
    tile,
    camNameEl:    tile.querySelector("[data-cam-name]"),
    camAreaEl:    tile.querySelector("[data-cam-area]"),
    videoWrap:    tile.querySelector("[data-video-wrap]"),
    latestVals:   tile.querySelectorAll(".metric .val"),
    activityBadge: tile.querySelector("[data-activity]"),
    activityText:  tile.querySelector("[data-activity-text]"),
    anomalyBadge: tile.querySelector("[data-anomaly]"),
    anomalyText:  tile.querySelector("[data-anomaly-text]"),
    fallbackBadge: tile.querySelector("[data-fallback]"),
    anomalyThumb: tile.querySelector("[data-anomaly-thumb]"),
    samplesEl:    tile.querySelector("[data-samples]"),
    chartCanvas:  tile.querySelector("canvas"),
    chart: null,
    history: [],
    currentActiveCam: null,   // updated by applyGridConfig
    currentHlsInstance: null, // hls.js instance we own; destroyed on rebuild
  };
  // Render initial placeholder video so viewers see something before
  // config/grid arrives.
  buildVideoInto(tileState[slot.slot_id],
    { active_hls: slot.placeholder_hls, active_page: slot.placeholder_page },
    slot);
}

// ---------- 2. Video builder (re-runs when active_cam changes) --------------

function buildVideoInto(st, cfg, slot) {
  // Tear down any existing hls.js instance so we don't leak network sockets
  // when a fallback swap replaces the <video> element.
  if (st.currentHlsInstance) {
    try { st.currentHlsInstance.destroy(); } catch (_) {}
    st.currentHlsInstance = null;
  }
  const hlsUrl = hlsUrlForActiveCam(cfg);
  const embed  = cfg.active_embed;
  const page   = cfg.active_page || slot.placeholder_page;

  let markup;
  if (embed && embed.includes("player.tvkur.com")) {
    // tvkur player is iframe-friendly and self-authenticates via referer,
    // so it's the preferred embed for konya/otogar-style cams.
    markup = `<iframe src="${embed}" allow="autoplay; encrypted-media"
                     allowfullscreen loading="lazy"></iframe>`;
  } else if (hlsUrl) {
    markup = `<video data-hls="${hlsUrl}" autoplay muted playsinline
                     controls preload="metadata"></video>`;
  } else if (page) {
    markup = `<div class="video-fallback">
                Live stream not embeddable from this site -
                <a href="${page}" target="_blank" rel="noopener">open camera page ↗</a>
              </div>`;
  } else {
    markup = `<div class="video-fallback">No live video available.</div>`;
  }
  st.videoWrap.innerHTML = markup;
  const video = st.videoWrap.querySelector("video[data-hls]");
  if (video) attachHls(st, video);
}

function attachHls(st, video) {
  const src = video.dataset.hls;
  if (window.Hls && window.Hls.isSupported()) {
    const hls = new window.Hls({ lowLatencyMode: true, liveSyncDuration: 4 });
    hls.loadSource(src);
    hls.attachMedia(video);
    hls.on(window.Hls.Events.ERROR, (_, data) => {
      if (data.fatal) console.warn("hls.js fatal error on", src, data);
    });
    st.currentHlsInstance = hls;
  } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
    video.src = src;
  } else {
    console.warn("No HLS playback support in this browser for", src);
  }
}

// ---------- 3. Bail out cleanly if Firebase isn't configured -----------------

if (!firebaseConfig) {
  document.getElementById("config-warning").style.display = "block";
  statusEl.innerHTML = `<span class="down">● firebase not configured</span>`;
} else {
  start(firebaseConfig);
}

// ---------- 4. Live subscriptions -------------------------------------------

function start(cfg) {
  const app = initializeApp(cfg);
  if (cfg.recaptchaSiteKey) {
    try {
      initializeAppCheck(app, {
        provider: new ReCaptchaV3Provider(cfg.recaptchaSiteKey),
        isTokenAutoRefreshEnabled: true,
      });
    } catch (e) {
      console.warn("App Check init failed — continuing without it:", e);
    }
  }
  const db = getFirestore(app);

  // 4a. config/grid — active cam per slot. Applied on every change.
  onSnapshot(doc(db, "config", "grid"), (snap) => {
    if (!snap.exists()) return;
    currentGridConfig = snap.data();
    applyGridConfig(currentGridConfig);
  }, (err) => console.warn("config/grid subscription failed:", err));

  // 4b. latest/{slot_id} -> KPI cards.
  const slotIds = new Set(GRID_SLOTS.map((s) => s.slot_id));
  onSnapshot(collection(db, "latest"), (snap) => {
    let alive = 0;
    for (const d of snap.docs) {
      if (!slotIds.has(d.id)) continue;
      const st  = tileState[d.id];
      if (!st) continue;
      const rec = d.data();
      const ageS = rec.ts ? Math.round((Date.now() - new Date(rec.ts).getTime()) / 1000) : null;
      if (ageS != null && ageS < 120) alive++;
      setLatest(st, rec);
    }
    statusEl.innerHTML = alive === GRID_SLOTS.length
        ? `<span class="live">● live</span> · ${alive}/${GRID_SLOTS.length} slots updating`
        : alive > 0
        ? `<span class="stale">● partial</span> · ${alive}/${GRID_SLOTS.length} slots updating`
        : `<span class="down">● no recent writes</span> · is the collector running?`;
  }, (err) => statusEl.textContent = "error: " + err.message);

  // 4c. footfall history for the 24h window, one query for all slots.
  const since = new Date(Date.now() - 24 * 3600 * 1000).toISOString();
  const histQ = query(
    collection(db, "footfall"),
    where("ts", ">=", since),
    orderBy("ts", "desc"),
    limit(HISTORY_LIMIT * GRID_SLOTS.length),
  );
  onSnapshot(histQ, (snap) => {
    const bySlot = Object.fromEntries(GRID_SLOTS.map((s) => [s.slot_id, []]));
    for (const d of snap.docs) {
      const r = d.data();
      if (!r.ok) continue;
      if (!bySlot[r.slot]) continue;
      bySlot[r.slot].push(r);
    }
    for (const slot of GRID_SLOTS) {
      const rows = bySlot[slot.slot_id].sort((a, b) => a.ts.localeCompare(b.ts));
      tileState[slot.slot_id].history = rows;
      renderTileChart(slot.slot_id, rows);
      updateAggregates(slot.slot_id, rows);
    }
  }, (err) => console.error("footfall history query failed:", err));

  setInterval(renderCombinedChart, 4000);
  renderCombinedChart();

  // 4d. Re-ID summary.
  onSnapshot(collection(db, "reid_stats"), (snap) => {
    renderReidTable(snap.docs.map((d) => ({ id: d.id, ...d.data() })));
  }, () => {});
}

function applyGridConfig(cfg) {
  if (!cfg || !Array.isArray(cfg.slots)) return;
  for (const slotCfg of cfg.slots) {
    const st = tileState[slotCfg.slot_id];
    if (!st) continue;
    if (st.currentActiveCam !== slotCfg.active_cam) {
      st.currentActiveCam = slotCfg.active_cam;
      st.camNameEl.textContent = slotCfg.active_cam_name || slotCfg.slot_id;
      st.camAreaEl.textContent = slotCfg.display_area || "";
      buildVideoInto(st, slotCfg, st.slot);
    }
    const usingFallback = slotCfg.active_cam !== slotCfg.primary;
    if (usingFallback) {
      st.fallbackBadge.textContent = `↳ fallback: ${slotCfg.active_cam}`;
      st.fallbackBadge.style.display = "inline-block";
    } else {
      st.fallbackBadge.style.display = "none";
    }
  }
}

// ---------- 5. Per-tile rendering -------------------------------------------

function setLatest(st, d) {
  const set = (k, v) => {
    const el = [...st.latestVals].find((x) => x.dataset.k === k);
    if (el) el.textContent = (v ?? v === 0) ? v : "-";
  };
  set("person",   d.person);
  set("vehicles", d.vehicles);
}

function updateAggregates(slotId, rows) {
  const st = tileState[slotId];
  if (!rows.length) {
    st.samplesEl.textContent = "no samples in the last 24h";
    setActivityBadge(st, null);
    return;
  }
  const ppl  = rows.map((r) => r.person ?? 0);
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
        `⚠ anomaly at ${fmtTime(last.ts)} - ${last.person} people (${anomalies.length} in 24h)`;
    const snap = last.snapshot_annotated_url || last.snapshot_url;
    if (snap) {
      st.anomalyThumb.href = snap;
      st.anomalyThumb.querySelector("img").src = snap;
      st.anomalyThumb.style.display = "inline-block";
    } else {
      st.anomalyThumb.style.display = "none";
    }
  } else {
    st.anomalyBadge.className = "anomaly-badge ok";
    st.anomalyText.textContent = "no anomalies in the last 24h";
    st.anomalyThumb.style.display = "none";
  }
  st.samplesEl.textContent = ` · ${rows.length} samples in 24h`;

  setActivityBadge(st, computeActivity(rows));
}

function computeActivity(rows) {
  const ppl = rows.map((r) => r.person ?? 0).filter((x) => x >= 0);
  if (ppl.length < 4) return null;
  const sorted = [...ppl].sort((a, b) => a - b);
  const p90Idx = Math.max(0, Math.floor(sorted.length * 0.9) - 1);
  const p90    = sorted[p90Idx] || 1;
  const now    = ppl[ppl.length - 1];
  const idx    = Math.max(0, Math.min(10, Math.round((now / p90) * 8)));
  const label  = idx <= 2 ? "Quiet"
               : idx <= 5 ? "Moderate"
               : idx <= 7 ? "Busy"
               : "Crowded";
  return { idx, label, now, p90 };
}

function setActivityBadge(st, act) {
  const badge = st.activityBadge;
  const text  = st.activityText;
  if (!act) {
    badge.className = "activity-badge act-unknown";
    text.textContent = "activity -/10";
    return;
  }
  const cls = act.label.toLowerCase();
  badge.className = `activity-badge act-${cls}`;
  text.textContent = `activity ${act.idx}/10 - ${act.label}`;
}

const TILE_CHART_LAST_N = 30;

function renderTileChart(slotId, rows) {
  const st = tileState[slotId];
  const view = rows.slice(-TILE_CHART_LAST_N);
  const labels   = view.map((r) => fmtTime(r.ts));
  const people   = view.map((r) => r.person);
  const vehicles = view.map((r) => r.vehicles);

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
          tension: 0, pointRadius: 2, pointHoverRadius: 4, borderWidth: 2 },
        { label: "vehicles", data: vehicles, borderColor: "#f0a35e",
          tension: 0, pointRadius: 2, pointHoverRadius: 4, borderWidth: 2 },
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

// ---------- 6. Combined chart of all four slots -----------------------------

function renderCombinedChart() {
  const seriesBySlot = {};
  let labels = [];
  for (const slot of GRID_SLOTS) {
    const rows = tileState[slot.slot_id].history;
    seriesBySlot[slot.slot_id] = rows;
    for (const r of rows) labels.push(r.ts);
  }
  labels = [...new Set(labels)].sort();
  const displayLabels = labels.map(fmtTime);

  const palette = ["#4f8cff", "#36d399", "#f0a35e", "#a78bfa"];
  const datasets = GRID_SLOTS.map((slot, i) => {
    const byTs = new Map(seriesBySlot[slot.slot_id].map((r) => [r.ts, r.person]));
    return {
      label: slot.display_area,
      data: labels.map((t) => byTs.has(t) ? byTs.get(t) : null),
      borderColor: palette[i % palette.length],
      tension: 0, pointRadius: 2, pointHoverRadius: 4, borderWidth: 2, spanGaps: true,
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

// ---------- 7. Re-ID summary table ------------------------------------------

function renderReidTable(docs) {
  const wrap = document.getElementById("reid-table-wrap");
  const slotIds = new Set(GRID_SLOTS.map((s) => s.slot_id));
  const rows = docs.filter((d) => slotIds.has(d.id));
  if (!rows.length) {
    wrap.innerHTML = `<div class="empty">No re-ID stats yet - the collector publishes
      them as detections come in.</div>`;
    return;
  }
  const tr = (cells) => `<tr>${cells.map((c) => `<td>${c}</td>`).join("")}</tr>`;
  const slotLabel = (id) => {
    const slot = GRID_SLOTS.find((s) => s.slot_id === id);
    return slot ? slot.display_area : id;
  };
  const total = (r, k) => Object.values(r.per_class ?? {}).reduce((s, p) => s + (p[k] ?? 0), 0);
  wrap.innerHTML = `
    <table class="reid">
      <thead><tr>
        <th>Slot</th><th>Camera (now)</th><th>Unique entities</th><th>Total sightings</th>
        <th>Regulars (≥3)</th>
      </tr></thead>
      <tbody>
        ${rows.map((r) => tr([
          escapeHtml(slotLabel(r.id)),
          escapeHtml(r.cam_id ?? "-"),
          r.total_unique ?? total(r, "unique") ?? "-",
          r.total_sightings ?? total(r, "total_sightings") ?? "-",
          r.regulars ?? total(r, "regulars") ?? "-",
        ])).join("")}
      </tbody>
    </table>`;
}

// ---------- helpers ---------------------------------------------------------

function flagAnomalies(rows) {
  const xs = rows.map((r) => r.person ?? 0);
  const out = [];
  let lastFlaggedTs = -Infinity;
  for (let i = ANOMALY_WARMUP; i < xs.length; i++) {
    const win = xs.slice(Math.max(0, i - ANOMALY_WINDOW), i);
    if (win.length < ANOMALY_WARMUP) continue;
    const mu  = win.reduce((a, b) => a + b, 0) / win.length;
    const sd  = Math.sqrt(win.reduce((a, b) => a + (b - mu) ** 2, 0) / win.length);
    const people = xs[i];
    const delta  = people - mu;
    if (delta <= 0) continue;
    if (people < ANOMALY_MIN_PEOPLE) continue;
    if (sd < ANOMALY_MIN_STD) continue;
    if (delta < ANOMALY_MIN_DELTA) continue;
    if (sd <= 0 || delta / sd < ANOMALY_Z) continue;
    const ts = new Date(rows[i].ts).getTime() / 1000;
    if (ts - lastFlaggedTs < ANOMALY_COOLDOWN_S) continue;
    lastFlaggedTs = ts;
    out.push(rows[i]);
  }
  return out;
}

function fmtTime(iso) {
  try {
    return new Date(iso).toLocaleTimeString([], {
      hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
    });
  } catch { return String(iso).slice(11, 19); }
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
