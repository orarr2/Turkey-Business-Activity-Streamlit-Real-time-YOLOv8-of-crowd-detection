// 4-slot live HTML dashboard, slot-based since the fallback refactor. Data lives
// in Firestore so it is persistent across visitors. Every visitor subscribes
// via onSnapshot; no polling.
//
// Collections this expects (cloud collector writes them):
//   config/grid            one doc; publishes the current active cam per slot
//   latest/{slot_id}       one doc per slot, overwritten each sample
//   footfall/{auto}        append-only history; each doc has a `slot` field,
//                          `person`/`vehicles` burst-median counts, and — when
//                          the collector flagged it — `is_anomaly` plus an
//                          `anomaly` map (kind/metric/window/z/observed/
//                          expected/bucket). TTL on `expire_at` deletes docs
//                          after 24h.
//   reid_stats/{slot_id}   per-slot unique/sightings/regulars (estimates)

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
// Shared staleness threshold: the header status pill and the per-tile age
// label must agree, or the same screen claims "live" and "stale" at once.
const STALE_AGE_S = 120;

// Anomalies: the collector is the single source of truth. Every footfall doc
// carries `is_anomaly` and (when flagged) an `anomaly` map with
// kind/metric/window/z/observed/expected computed server-side from robust
// statistics + the hour-of-week profile. The dashboard only RENDERS those
// fields — it no longer recomputes z-scores client-side, so what you see is
// exactly what the collector flagged (and snapshotted) at sample time.

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
      <span class="footnote" data-age title="age of the counts shown - the video is live, the numbers are the collector's most recent sample"></span>
      <span class="footnote" data-crossings title="sampled line crossings during the last burst (cameras with a configured counting line)"></span>
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
    ageEl:        tile.querySelector("[data-age]"),
    crossEl:      tile.querySelector("[data-crossings]"),
    samplesEl:    tile.querySelector("[data-samples]"),
    chartCanvas:  tile.querySelector("canvas"),
    chart: null,
    history: [],
    lastSampleMs: null,   // epoch ms of the last OK sample; drives the age label
    currentActiveCam: null,   // updated by applyGridConfig
    currentHlsInstance: null, // hls.js instance we own; destroyed on rebuild
  };
  // Render initial placeholder video so viewers see something before
  // config/grid arrives.
  buildVideoInto(tileState[slot.slot_id],
    { active_hls: slot.placeholder_hls, active_page: slot.placeholder_page },
    slot);
}

// ---------- 1b. Model-view strip skeleton -----------------------------------
// One tiny annotated-frame card per slot, next to the search panel. The image
// URL is the same `live_annotated_url` the collector publishes on each sample;
// the strip stays put and only its <img>/counts refresh, so the grid keeps the
// live videos and this strip carries the "what the model saw" view - without
// doubling the grid the way the per-tile version did.
const stripEl = document.getElementById("model-strip");
const stripState = {};
if (stripEl) {
  for (const slot of GRID_SLOTS) {
    const cell = document.createElement("div");
    cell.className = "mini";
    cell.innerHTML = `
      <div class="lbl" data-lbl>${escapeHtml(slot.display_area)}</div>
      <a data-link target="_blank" rel="noopener" title="open annotated frame full size">
        <div class="mini-empty" data-empty>waiting for first sample…</div>
        <img alt="annotated detections" loading="lazy" hidden />
      </a>
      <div class="nums">
        <span>👤 <b data-p>-</b></span>
        <span class="v">🚗 <b data-v>-</b></span>
      </div>
      <div class="age" data-age></div>`;
    stripEl.appendChild(cell);
    stripState[slot.slot_id] = {
      cell,
      lbl:   cell.querySelector("[data-lbl]"),
      link:  cell.querySelector("[data-link]"),
      empty: cell.querySelector("[data-empty]"),
      img:   cell.querySelector("img"),
      p:     cell.querySelector("[data-p]"),
      v:     cell.querySelector("[data-v]"),
      age:   cell.querySelector("[data-age]"),
      lastSampleMs: null,
    };
  }
}

function updateStrip(slotId, d) {
  const s = stripState[slotId];
  if (!s) return;
  if (d.person   != null) s.p.textContent = d.person;
  if (d.vehicles != null) s.v.textContent = d.vehicles;
  if (d.ok && d.live_annotated_url) {
    const url = d.live_annotated_url
        + (d.live_annotated_url.includes("?") ? "&" : "?")
        + "t=" + encodeURIComponent(d.ts || Date.now());
    s.img.src = url;
    s.img.hidden = false;
    s.link.href = url;
    if (s.empty) s.empty.style.display = "none";
  }
  if (d.ok && d.ts) s.lastSampleMs = Date.parse(d.ts);
  renderStripAge(slotId);
}

function renderStripAge(slotId) {
  const s = stripState[slotId];
  if (!s || !s.lastSampleMs) return;
  const ageS = Math.max(0, Math.round((Date.now() - s.lastSampleMs) / 1000));
  const stale = ageS > STALE_AGE_S;
  s.age.textContent = ageS < 90 ? `${ageS}s ago` : `${Math.round(ageS / 60)}m ago`;
  s.age.style.color = stale ? "#ef4444" : "";
}

setInterval(() => {
  for (const id of Object.keys(stripState)) renderStripAge(id);
}, 1000);

// Reflect fallback/active-cam label into the strip too so the mini card matches
// what's in the main tile.
function updateStripLabel(slotId, activeCamName, displayArea) {
  const s = stripState[slotId];
  if (!s) return;
  s.lbl.textContent = displayArea || activeCamName || slotId;
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
  if (hlsUrl) {
    // Direct HLS first: <video autoplay muted> starts on its own (tvkur cams
    // route through the local /tvkur/ proxy). The tvkur iframe player shows a
    // click-to-play splash, so it's only the FALLBACK when HLS can't play
    // (e.g. web/ hosted statically without the proxy) - see attachHls.
    markup = `<video data-hls="${hlsUrl}" autoplay muted playsinline
                     controls preload="metadata"></video>`;
  } else if (embed && embed.includes("player.tvkur.com")) {
    markup = `<iframe src="${embed}" allow="autoplay; encrypted-media"
                     allowfullscreen loading="lazy"></iframe>`;
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
  if (video) attachHls(st, video, cfg);
}

function attachHls(st, video, cfg) {
  const src = video.dataset.hls;
  const fallbackToEmbed = () => {
    // HLS is unplayable here (proxy missing / stream refused). If the slot
    // has an iframe player, swap to it so the tile still shows live video.
    const embed = cfg && cfg.active_embed;
    if (!embed) return;
    if (st.currentHlsInstance) {
      try { st.currentHlsInstance.destroy(); } catch (_) {}
      st.currentHlsInstance = null;
    }
    st.videoWrap.innerHTML = `<iframe src="${embed}" allow="autoplay; encrypted-media"
                     allowfullscreen loading="lazy"></iframe>`;
  };
  if (window.Hls && window.Hls.isSupported()) {
    const hls = new window.Hls({ lowLatencyMode: true, liveSyncDuration: 4 });
    hls.loadSource(src);
    hls.attachMedia(video);
    hls.on(window.Hls.Events.ERROR, (_, data) => {
      if (!data.fatal) return;
      console.warn("hls.js fatal error on", src, data);
      fallbackToEmbed();
    });
    st.currentHlsInstance = hls;
  } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
    video.src = src;
    video.addEventListener("error", fallbackToEmbed, { once: true });
  } else {
    console.warn("No HLS playback support in this browser for", src);
    fallbackToEmbed();
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
      if (ageS != null && ageS < STALE_AGE_S) alive++;
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
    renderAnomalyEvents();
  }, (err) => console.error("footfall history query failed:", err));

  setInterval(renderCombinedChart, 4000);
  renderCombinedChart();

  // 4d. Re-ID summary.
  onSnapshot(collection(db, "reid_stats"), (snap) => {
    renderReidTable(snap.docs.map((d) => ({ id: d.id, ...d.data() })));
  }, () => {});

  // 4e. Operational events (loiter / returning) - last 24h, newest first.
  const evQ = query(
    collection(db, "events"),
    where("ts", ">=", since),
    orderBy("ts", "desc"),
    limit(120),
  );
  onSnapshot(evQ, (snap) => {
    renderEventsTable(snap.docs.map((d) => d.data()));
  }, (err) => console.warn("events subscription failed:", err));
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
      updateStripLabel(slotCfg.slot_id, slotCfg.active_cam_name,
                       slotCfg.display_area);
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
    // v != null keeps 0 - an empty street at night is a real count, not
    // missing data.
    if (el) el.textContent = v != null ? v : "-";
  };
  set("person",   d.person);
  set("vehicles", d.vehicles);
  // Sampled line-crossing flow, shown only for cameras with a configured
  // counting line (cameras.py "line"): in/out during this sample's burst.
  if (st.crossEl) {
    const c = d.crossings;
    st.crossEl.textContent = c
        ? ` · line: ${c.in ?? 0} in / ${c.out ?? 0} out`
        : "";
  }
  // "Model view": annotated frame + counts moved to the compact side strip
  // beside the search panel (see updateStrip). The tile itself now holds only
  // the live video + KPIs, so the grid stays 2x2, not 4x2.
  updateStrip(st.slot.slot_id, d);
  // Only a SUCCESSFUL sample refreshes the age: MISS docs (ok=0) also carry a
  // fresh ts, and using it would keep the label green while the camera has
  // produced no real count for hours - the exact case the label must expose.
  if (d.ok && d.ts) st.lastSampleMs = Date.parse(d.ts);
  renderSampleAge(st);
}

// The video tile is (near-)live but the numbers describe the collector's most
// recent sample - tens of seconds old by construction. Showing the age keeps
// the "I count 9 cars, the tile says 4" confusion honest: it labels WHEN the
// number was true, and turns red when the collector has stopped keeping up.
function renderSampleAge(st) {
  if (!st.ageEl) return;
  if (!st.lastSampleMs) { st.ageEl.textContent = ""; return; }
  const ageS = Math.max(0, Math.round((Date.now() - st.lastSampleMs) / 1000));
  const stale = ageS > STALE_AGE_S;
  const label = ageS < 90 ? ` · counts from ${ageS}s ago`
              : ` · counts from ${Math.round(ageS / 60)}m ago`;
  const memo = label + (stale ? "!" : "");
  if (memo !== st._ageMemo) {            // skip no-op DOM writes
    st._ageMemo = memo;
    st.ageEl.textContent = label;
    st.ageEl.style.color = stale ? "#ef4444" : "";
  }
}

setInterval(() => {
  for (const st of Object.values(tileState)) renderSampleAge(st);
}, 1000);

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

  const anomalies = rows.filter((r) => r.is_anomaly);
  if (anomalies.length) {
    const last = anomalies[anomalies.length - 1];
    const d = describeAnomaly(last);
    st.anomalyBadge.className = "anomaly-badge warn";
    st.anomalyText.textContent =
        `⚠ ${d.arrow} ${d.metricLabel} ${d.kindLabel} at ${fmtTime(last.ts)} - ` +
        `${d.observed ?? "?"} vs ~${d.expected ?? "?"} expected (${anomalies.length} in 24h)`;
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
  // Anomalous samples render as enlarged red points on the metric that fired.
  const anomOn = (metric) => (r) =>
      r.is_anomaly && ((r.anomaly?.metric ?? "person") === metric);
  const pplPointBg = view.map((r) => anomOn("person")(r)   ? "#ef4444" : "#4f8cff");
  const vehPointBg = view.map((r) => anomOn("vehicles")(r) ? "#ef4444" : "#f0a35e");
  const pplPointR  = view.map((r) => anomOn("person")(r)   ? 5 : 2);
  const vehPointR  = view.map((r) => anomOn("vehicles")(r) ? 5 : 2);

  if (st.chart) {
    st.chart.data.labels = labels;
    st.chart.data.datasets[0].data = people;
    st.chart.data.datasets[0].pointBackgroundColor = pplPointBg;
    st.chart.data.datasets[0].pointRadius = pplPointR;
    st.chart.data.datasets[1].data = vehicles;
    st.chart.data.datasets[1].pointBackgroundColor = vehPointBg;
    st.chart.data.datasets[1].pointRadius = vehPointR;
    st.chart.update("none");
    return;
  }
  st.chart = new Chart(st.chartCanvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "people",   data: people,   borderColor: "#4f8cff",
          pointBackgroundColor: pplPointBg,
          tension: 0, pointRadius: pplPointR, pointHoverRadius: 6, borderWidth: 2 },
        { label: "vehicles", data: vehicles, borderColor: "#f0a35e",
          pointBackgroundColor: vehPointBg,
          tension: 0, pointRadius: vehPointR, pointHoverRadius: 6, borderWidth: 2 },
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
// ~1,000 raw samples per camera per day drawn as-is turn into unreadable
// spaghetti, and per-sample labels distort the time axis (gaps compress).
// So the combined view averages each camera into fixed 5-minute bins on a
// true shared timeline; the per-tile mini charts keep the raw samples.

const COMBINED_BIN_MIN = 5;

// Each section below the tiles reveals ITSELF the moment it has content, and
// stays gone until then. `hidden` on the wrapper element is toggled here so an
// idle collector doesn't leave four empty placeholder boxes eating half the
// viewport.
function toggleSection(id, hasContent) {
  const el = document.getElementById(id);
  if (el) el.hidden = !hasContent;
}

function renderCombinedChart() {
  const binMs = COMBINED_BIN_MIN * 60 * 1000;
  const binsBySlot = {};
  const allBins = new Set();
  for (const slot of GRID_SLOTS) {
    const bins = new Map();   // bin start (ms) -> {sum, n}
    for (const r of tileState[slot.slot_id].history) {
      if (r.person == null) continue;
      const t = new Date(r.ts).getTime();
      if (!Number.isFinite(t)) continue;
      const b = Math.floor(t / binMs) * binMs;
      const cell = bins.get(b) || { sum: 0, n: 0 };
      cell.sum += r.person; cell.n += 1;
      bins.set(b, cell);
      allBins.add(b);
    }
    binsBySlot[slot.slot_id] = bins;
  }
  const binList = [...allBins].sort((a, b) => a - b);
  toggleSection("chart-section", binList.length > 0);
  if (!binList.length) return;
  const displayLabels = binList.map((b) => fmtTimeShort(b));

  // Anomaly bins per slot (people spike/drop confirmed by the collector).
  // Each bin is anomalous if any raw sample in it carries is_anomaly on the
  // people metric - the combined chart bins to 5 min, so we mark the bin, not
  // the raw sample. Matches the per-tile chart's red-point convention.
  const anomBinsBySlot = {};
  for (const slot of GRID_SLOTS) {
    const set = new Set();
    for (const r of tileState[slot.slot_id].history) {
      if (!r.is_anomaly) continue;
      if ((r.anomaly?.metric || "person") !== "person") continue;
      const t = new Date(r.ts).getTime();
      if (!Number.isFinite(t)) continue;
      set.add(Math.floor(t / binMs) * binMs);
    }
    anomBinsBySlot[slot.slot_id] = set;
  }

  const palette = ["#4f8cff", "#36d399", "#f0a35e", "#a78bfa"];
  const datasets = GRID_SLOTS.map((slot, i) => {
    const bins = binsBySlot[slot.slot_id];
    const anom = anomBinsBySlot[slot.slot_id];
    const pointBg = binList.map((b) => anom.has(b) ? "#ef4444"
                                                   : palette[i % palette.length]);
    const pointR  = binList.map((b) => anom.has(b) ? 5 : 0);
    return {
      label: slot.display_area,
      data: binList.map((b) => bins.has(b)
          ? +(bins.get(b).sum / bins.get(b).n).toFixed(1) : null),
      borderColor: palette[i % palette.length],
      pointBackgroundColor: pointBg,
      tension: 0.25, pointRadius: pointR, pointHoverRadius: 5,
      borderWidth: 2, spanGaps: true,
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

// ---------- 6b. Anomaly events - where exactly, and when ---------------------
// Flat 24h log across all slots, newest first: time, area, direction
// (spike/drop), which metric moved, observed vs expected, snapshot proof.

function renderAnomalyEvents() {
  const wrap = document.getElementById("anomaly-table-wrap");
  if (!wrap) return;
  const events = [];
  for (const slot of GRID_SLOTS) {
    for (const r of tileState[slot.slot_id].history) {
      if (r.is_anomaly) events.push({ area: slot.display_area, r });
    }
  }
  toggleSection("anomaly-section", events.length > 0);
  if (!events.length) return;
  events.sort((a, b) => b.r.ts.localeCompare(a.r.ts));
  const rows = events.slice(0, 60).map(({ area, r }) => {
    const d = describeAnomaly(r);
    const snap = r.snapshot_annotated_url || r.snapshot_url;
    const expected = d.expected != null
        ? `~${d.expected}${d.bucket ? ` <span class="footnote">(${escapeHtml(d.bucket)} norm)</span>` : ""}`
        : "-";
    return `<tr>
      <td>${fmtTime(r.ts)}</td>
      <td>${escapeHtml(area)}</td>
      <td class="${d.dir}">${d.arrow} ${escapeHtml(d.kindLabel)}</td>
      <td>${escapeHtml(d.metricLabel)}</td>
      <td>${d.observed ?? "-"}</td>
      <td>${expected}</td>
      <td>${snap ? `<a href="${snap}" target="_blank" rel="noopener">view</a>` : "-"}</td>
    </tr>`;
  }).join("");
  wrap.innerHTML = `<table class="reid">
    <thead><tr>
      <th>Time</th><th>Area</th><th>Type</th><th>Metric</th>
      <th>Observed</th><th>Expected</th><th>Snapshot</th>
    </tr></thead>
    <tbody>${rows}</tbody></table>`;
}

// ---------- 6c. Operational events (loiter / returning) ----------------------

const EVENT_LABELS = {
  loiter:    { icon: "⏱", label: "prolonged presence" },
  returning: { icon: "↩", label: "returning visitor" },
};

function renderEventsTable(events) {
  const wrap = document.getElementById("events-table-wrap");
  if (!wrap) return;
  const slotLabel = (id) => {
    const slot = GRID_SLOTS.find((s) => s.slot_id === id);
    return slot ? slot.display_area : id;
  };
  toggleSection("events-section", events.length > 0);
  if (!events.length) return;
  const rows = events.slice(0, 60).map((e) => {
    const meta = EVENT_LABELS[e.kind] || { icon: "•", label: e.kind };
    const detail = e.kind === "loiter"
        ? `${e.cls ?? "?"} stationary ${Math.round((e.duration_sec ?? 0) / 60)} min`
        : e.kind === "returning"
        ? `${e.cls ?? "?"} #${e.entity_id ?? "?"} back after ${Math.round((e.gap_seconds ?? 0) / 60)} min`
        : "";
    const snap = e.snapshot_url || e.fullframe_url;
    return `<tr>
      <td>${fmtTime(e.ts)}</td>
      <td>${escapeHtml(slotLabel(e.slot))}</td>
      <td>${meta.icon} ${escapeHtml(meta.label)}</td>
      <td>${escapeHtml(detail)}</td>
      <td>${snap ? `<a href="${snap}" target="_blank" rel="noopener">view</a>` : "-"}</td>
    </tr>`;
  }).join("");
  wrap.innerHTML = `<table class="reid">
    <thead><tr>
      <th>Time</th><th>Area</th><th>Event</th><th>Detail</th><th>Snapshot</th>
    </tr></thead>
    <tbody>${rows}</tbody></table>`;
}

// ---------- 7. Re-ID summary table ------------------------------------------

function renderReidTable(docs) {
  const wrap = document.getElementById("reid-table-wrap");
  const slotIds = new Set(GRID_SLOTS.map((s) => s.slot_id));
  const rows = docs.filter((d) => slotIds.has(d.id));
  toggleSection("reid-section", rows.length > 0);
  if (!rows.length) return;
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
    </table>
    <div class="footnote" style="margin-top:8px">
      Estimates from HSV color-histogram matching (rolling 48h registry) - good
      for trends in daylight, not an identity system. Two people in similar
      clothing can merge; the same person can split after a lighting change.
    </div>`;
}

// ---------- helpers ---------------------------------------------------------

// Normalize a flagged doc's `anomaly` map into display strings. Docs written
// before the anomaly map existed only have the boolean — treated as a people
// spike with no expectation attached.
function describeAnomaly(r) {
  const a = r.anomaly || {};
  const kind = a.kind || "spike";
  const isDrop = kind.includes("drop");
  const hourly = a.window === "hourly" || kind.startsWith("contextual");
  const metric = a.metric === "vehicles" ? "vehicles" : "people";
  return {
    arrow:       isDrop ? "▼" : "▲",
    dir:         isDrop ? "drop" : "spike",
    kindLabel:   (hourly ? "hourly " : "") + (isDrop ? "drop" : "spike"),
    metricLabel: metric,
    observed:    a.observed ?? (metric === "vehicles" ? r.vehicles : r.person),
    expected:    a.expected,
    bucket:      a.bucket || "",
  };
}

function fmtTime(iso) {
  try {
    return new Date(iso).toLocaleTimeString([], {
      hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
    });
  } catch { return String(iso).slice(11, 19); }
}

function fmtTimeShort(ms) {
  try {
    return new Date(ms).toLocaleTimeString([], {
      hour: "2-digit", minute: "2-digit", hour12: false,
    });
  } catch { return ""; }
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
