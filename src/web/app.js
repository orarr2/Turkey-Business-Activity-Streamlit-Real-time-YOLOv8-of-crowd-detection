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
  initializeAppCheck, ReCaptchaV3Provider,
} from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app-check.js";
import {
  getFirestore, collection, onSnapshot, query, where, orderBy, limit,
} from "https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js";

// Cache-busting for sibling modules. Browsers cache ES-module specifiers by exact
// URL - if the HTML's <script src="./cameras.js?v=5"> bumps version but app.js still
// imports "./cameras.js" (no query), the import resolves to a different URL and can
// serve a stale module. Extract the same ?v=N from this file's own URL so both
// imports below share it. Pages can also set `?ver=N` on the page URL to override.
const _u = new URL(import.meta.url);
const _ver = (_u.searchParams.get("v") || _u.searchParams.get("ver")
              || new URLSearchParams(location.search).get("ver") || "dev");
const _q = "?v=" + encodeURIComponent(_ver);

const { GRID_CAMERAS } = await import("./cameras.js" + _q);

// firebase-config.js is gitignored - if it's missing the dashboard still renders
// the layout but shows a config warning.
let firebaseConfig;
try {
  firebaseConfig = (await import("./firebase-config.js" + _q)).firebaseConfig;
} catch (_) { /* handled below */ }

const statusEl = document.getElementById("status");
const tilesEl  = document.getElementById("tiles");

const HISTORY_LIMIT  = 360;     // ~6h at 1 sample/min, 1.5h at 1/15s

// Anomaly gates - keep in lock-step with AnomalyTracker in app/collector.py.
// The Python collector now writes is_anomaly to Firestore so this JS-side
// detector only runs as a fallback for legacy rows that don't have the flag
// (so the dashboard doesn't re-flag noise that the collector wisely skipped).
const ANOMALY_WINDOW    = 30;       // 10 min @ 20s sampling
const ANOMALY_WARMUP    = 10;       // need real history before flagging
const ANOMALY_Z         = 3.5;      // higher significance bar (was 2.5)
const ANOMALY_MIN_PEOPLE = 5;       // absolute people floor for the spike
const ANOMALY_MIN_DELTA  = 5;       // spike must be >= 5 above baseline
const ANOMALY_MIN_STD    = 0.0;     // off by default - other gates handle noise
const ANOMALY_COOLDOWN_S = 300;     // 5 min between flags per series

// ---------- 1. Render tile skeletons (works even without Firebase) ----------

const tileState = {};   // cam_id -> { latestEl, chartEl, chart, anomalyEl, samplesEl, history }
let combinedChart = null;   // declared at module top to avoid TDZ when start() calls
                            // renderCombinedChart before reaching its previous declaration site

for (const cam of GRID_CAMERAS) {
  const tile = document.createElement("div");
  tile.className = "tile";
  // Per-tile video markup. Three shapes:
  //   cam.embed -> tvkur (or any iframe-friendly) live player in an <iframe>.
  //   cam.hls   -> direct HLS .m3u8 played in a <video> via hls.js (no iframe).
  //               Used when the camera's owner page sets X-Frame-Options but
  //               the underlying CDN exposes Access-Control-Allow-Origin: *
  //               (IBB's kamerayayin.ibb.istanbul does both).
  //   neither   -> a clickable "open camera page" fallback.
  let videoMarkup;
  if (cam.embed) {
    videoMarkup =
      `<iframe src="${cam.embed}" allow="autoplay; encrypted-media"
               allowfullscreen loading="lazy"></iframe>`;
  } else if (cam.hls) {
    videoMarkup =
      `<video data-hls="${cam.hls}" autoplay muted playsinline
              controls preload="metadata"></video>`;
  } else {
    videoMarkup =
      `<div class="video-fallback">
         Live stream not embeddable from this site -
         <a href="${cam.page}" target="_blank" rel="noopener">open camera page ↗</a>
       </div>`;
  }
  tile.innerHTML = `
    <div>
      <h2>${escapeHtml(cam.name)}</h2>
      <div class="city">${escapeHtml(cam.city ?? "")}</div>
    </div>
    <div class="video-wrap">${videoMarkup}</div>
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
      <a class="anomaly-thumb" data-anomaly-thumb target="_blank" rel="noopener"
         style="display:none" title="open snapshot of latest anomaly">
        <img alt="" />
      </a>
      <span class="footnote" data-samples></span>
    </div>
    <div class="chart-mini"><canvas></canvas></div>
  `;
  tilesEl.appendChild(tile);

  tileState[cam.id] = {
    tile,
    latestVals:   tile.querySelectorAll(".metric .val"),
    activityBadge: tile.querySelector("[data-activity]"),
    activityText:  tile.querySelector("[data-activity-text]"),
    anomalyBadge: tile.querySelector("[data-anomaly]"),
    anomalyText:  tile.querySelector("[data-anomaly-text]"),
    anomalyThumb: tile.querySelector("[data-anomaly-thumb]"),
    samplesEl:    tile.querySelector("[data-samples]"),
    chartCanvas:  tile.querySelector("canvas"),
    chart: null,
    history: [],
  };
}

// ---------- 1b. Attach hls.js to any <video data-hls=...> tile ---------------
// We use hls.js for Chrome/Edge/Firefox (no native HLS) and fall back to native
// <video src=...> on Safari. The CDN script is in index.html; if it didn't load
// we leave the <video> blank and the user still has the "open camera page" link
// via the title bar.
for (const video of document.querySelectorAll("video[data-hls]")) {
  const src = video.dataset.hls;
  if (window.Hls && window.Hls.isSupported()) {
    const hls = new window.Hls({ lowLatencyMode: true, liveSyncDuration: 4 });
    hls.loadSource(src);
    hls.attachMedia(video);
    hls.on(window.Hls.Events.ERROR, (_, data) => {
      // Recoverable network/media errors hls.js can heal; fatal ones we just log
      // and leave a black tile - the YOLO counts keep flowing regardless.
      if (data.fatal) console.warn("hls.js fatal error on", src, data);
    });
  } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
    video.src = src;   // Safari / iOS native HLS
  } else {
    console.warn("No HLS playback support in this browser for", src);
  }
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

  // App Check (anti-abuse): attest that reads come from your real web app, not a
  // bot scraping the database to burn your read quota. Initialized right after
  // initializeApp() and before getFirestore() so the App Check token rides along
  // with every Firestore request. Only active if a reCAPTCHA v3 site key is set
  // in firebase-config.js; enforcement itself is toggled in the Firebase console
  // (App Check -> Firestore -> Enforce). See docs/firebase_setup.md §6.
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

  // 3b. ONE subscription to the footfall history collection (last 24h), then
  //     fan out to the 4 tiles client-side. This avoids the composite index
  //     Firestore would otherwise require for
  //         where(cam_id, ==) + orderBy(ts, desc) + where(ts, >=)
  //     and reads each Firestore document once instead of four times.
  const since = new Date(Date.now() - 24 * 3600 * 1000).toISOString();
  const gridIds = new Set(GRID_CAMERAS.map((c) => c.id));
  const histQ = query(
    collection(db, "footfall"),
    where("ts", ">=", since),
    orderBy("ts", "desc"),
    limit(HISTORY_LIMIT * GRID_CAMERAS.length),
  );
  onSnapshot(histQ, (snap) => {
    const byCam = Object.fromEntries(GRID_CAMERAS.map((c) => [c.id, []]));
    for (const doc of snap.docs) {
      const r = doc.data();
      if (!r.ok) continue;
      if (!gridIds.has(r.cam_id)) continue;
      byCam[r.cam_id].push(r);
    }
    for (const cam of GRID_CAMERAS) {
      const rows = byCam[cam.id].sort((a, b) => a.ts.localeCompare(b.ts));
      tileState[cam.id].history = rows;
      renderTileChart(cam.id, rows);
      updateAggregates(cam.id, rows);
    }
  }, (err) => console.error("footfall history query failed:", err));

  // 3c. Combined 24h chart at the bottom - rebuild from each tile's history
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
    if (el) el.textContent = (v ?? v === 0) ? v : "-";
  };
  set("person",   d.person);
  set("vehicles", d.vehicles);
}

function updateAggregates(camId, rows) {
  const st = tileState[camId];
  if (!rows.length) {
    st.samplesEl.textContent = "no samples in the last 24h";
    setActivityBadge(st, null);
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

  // Anomalies: prefer the collector's flag (is_anomaly + snapshot_url) when
  // present; fall back to the JS-side rolling z-score for rows the older
  // collector wrote without the flag. Only the LATEST anomaly gets its
  // snapshot thumbnail shown - we don't clutter the tile with all of them.
  const anomalies = pickAnomalies(rows);
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

// Pick anomalies from a 24h row set. We *always* run the JS-side gates rather
// than trusting `is_anomaly` from Firestore: rows written before the gates
// were tightened may have is_anomaly=true under the old lax criteria, and we
// don't want those legacy false positives to keep painting the tile red.
// If the JS-flagged row also carries a snapshot_url, the thumbnail shows up;
// otherwise the badge is still meaningful (collector ran without snapshots
// or the snapshot was pruned).
function pickAnomalies(rows) {
  return flagAnomalies(rows);
}

// ---------- Activity Index ---------------------------------------------------
// Normalize the current people count against the camera's own 24h history so
// each camera grades against its own baseline (a "busy" Konya square is not
// the same scale as a "busy" Sultanahmet square).
//
//   index = round( now / p90_24h * 8 ),  clipped to 0..10.
//
// Why p90 instead of max: peaks can be one-off spikes that would push every
// normal sample to a low index. The 90th percentile is a more stable ceiling.
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
  const cls = act.label.toLowerCase();   // quiet | moderate | busy | crowded
  badge.className = `activity-badge act-${cls}`;
  text.textContent = `activity ${act.idx}/10 - ${act.label}`;
}

// How many recent samples to show in the per-tile chart. The metrics above
// (24h avg / 24h peak / anomaly) still aggregate the full 24h window in `rows`;
// only the chart trims to the most-recent N points so they spread across the
// canvas instead of bunching into a single vertical "spike" on the right.
const TILE_CHART_LAST_N = 30;

function renderTileChart(camId, rows) {
  const st = tileState[camId];
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

// ---------- 5. Combined chart of all four cameras ---------------------------

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
      label: cam.name.split(" - ")[0],
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

// ---------- 6. Re-ID summary table ------------------------------------------

function renderReidTable(docs) {
  const wrap = document.getElementById("reid-table-wrap");
  const rows = docs.filter((d) => GRID_CAMERAS.some((c) => c.id === d.id));
  if (!rows.length) {
    wrap.innerHTML = `<div class="empty">No re-ID stats yet - the collector publishes
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
          r.total_unique ?? total(r, "unique") ?? "-",
          r.total_sightings ?? total(r, "total_sightings") ?? "-",
          r.regulars ?? total(r, "regulars") ?? "-",
        ])).join("")}
      </tbody>
    </table>`;
}

// ---------- helpers ---------------------------------------------------------

function flagAnomalies(rows) {
  // Multi-gate fallback that mirrors AnomalyTracker.push_and_check in
  // app/collector.py. ALL gates must pass for a row to count as an anomaly:
  // z>=ANOMALY_Z (positive only, not drops), people>=ANOMALY_MIN_PEOPLE,
  // delta from baseline mean >= ANOMALY_MIN_DELTA, baseline std >= ANOMALY_MIN_STD,
  // and a per-series cooldown of ANOMALY_COOLDOWN_S between flags.
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
    if (delta <= 0) continue;                              // spike only
    if (people < ANOMALY_MIN_PEOPLE) continue;             // absolute floor
    if (sd < ANOMALY_MIN_STD) continue;                    // quiet baseline
    if (delta < ANOMALY_MIN_DELTA) continue;               // small delta
    if (sd <= 0 || delta / sd < ANOMALY_Z) continue;       // below z bar
    const ts = new Date(rows[i].ts).getTime() / 1000;
    if (ts - lastFlaggedTs < ANOMALY_COOLDOWN_S) continue; // cooldown
    lastFlaggedTs = ts;
    out.push(rows[i]);
  }
  return out;
}

function fmtTime(iso) {
  // HH:MM:SS - keeping seconds means each 20s sample gets a unique x-axis label
  // so the chart doesn't squash same-minute samples into a single vertical spike.
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
