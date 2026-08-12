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

const { GRID_SLOTS, LOCAL_MODE, hlsUrlForActiveCam } = await import("./cameras.js" + _q);

let firebaseConfig;
try {
  firebaseConfig = (await import("./firebase-config.js" + _q)).firebaseConfig;
} catch (_) { /* handled below */ }

// Dual-mode dashboard. Each notebook's Section 7 launches this page with a
// ?mode= URL param; app.js sets it as an attribute on <html> so the CSS at
// the top of index.html can hide the panels that do not apply. Default:
// main (opens as http://localhost:8000/?mode=main OR no param).
//
// Twin (yolov8n twin notebook, ?mode=twin): hides Send Report From VM,
// per-tile Live Analysis 🔬, Window analysis, and the class/time dropdowns
// in the model-view strip. KEEPS Model view - live (twin IS the VM's own
// data) and the RL tab (twin IS the review tool). Reorders the RL tab so
// the Review section sits above Learning proof.
//
// Main (turkey_business_activity.ipynb, ?mode=main or no param): hides
// Model view - live (hardcoded Turkey cams the picked country may not
// match), Window analysis, model-strip dropdowns, and the RL tab (no
// persistent review data - each run is ephemeral). Shows a new Snapshots
// tab in place of RL, and adds a "📸 Snapshot grid" header button.
const MODE = new URLSearchParams(location.search).get("mode") === "twin"
             ? "twin" : "main";
const TWIN_MODE = MODE === "twin";
const MAIN_MODE = MODE === "main";
document.documentElement.setAttribute("data-mode", MODE);

// Force the default tab to Analysis on every load. Without this, a stale
// localStorage entry from an earlier session could re-open the dashboard
// on RL / Search, which surprised the operator ("why did it open on RL?").
try {
  localStorage.removeItem("activeTab");
  localStorage.removeItem("dashboardActiveTab");
} catch (_) { /* private mode - fine */ }

// The mode-specific hides are gone: each mode has its own HTML file
// (index_main.html / index_twin.html) that physically lacks the panels
// not applicable to it. dashboard_server.py routes / and /index.html to
// the file matching ?mode=. The JS below still uses MODE for one thing
// the tile TEMPLATE builds dynamically (the analyze-btn appended per
// tile inside createTile) - see the ternary in that template further
// down. Everything else that used to be belt-and-suspenders JS hide is
// unnecessary now: the elements simply do not exist.

const statusEl = document.getElementById("status");
const tilesEl  = document.getElementById("tiles");

// Per-slot cap on the 24h footfall query. Must cover the collector's REAL
// cadence or every "24h" widget silently shrinks: at the VM's 40s interval a
// slot writes ~2160 docs/day, and the old cap of 360 meant the "24h" chart,
// avg and peak were computed over the newest ~4 HOURS while labeled 24h.
// 2160 x 4 slots = ~8.6k doc reads per page load - well inside the free
// tier for an operator dashboard opened a handful of times a day.
const HISTORY_LIMIT = 2160;
// Shared staleness threshold: the header status pill and the per-tile age
// label must agree, or the same screen claims "live" and "stale" at once.
const STALE_AGE_S = 120;

// Activity-index bands + combined-chart bin size. Declared up here (not next
// to their consumers below) because start() calls renderCombinedChart and
// computeActivity SYNCHRONOUSLY at file load - a const declared further
// down would still be in TDZ when those first calls run, and the whole
// dashboard init would throw (blank tiles, no video, no search, no review).
const ACTIVITY_BANDS = [
  { max: 0,   idx: 0  }, // truly empty
  { max: 2,   idx: 1  }, // 1-2 people = quiet regardless of history
  { max: 5,   idx: 2  }, // handful passing by = quiet
  { max: 8,   idx: 3  }, // still quiet
  { max: 12,  idx: 5  }, // moderate
  { max: 18,  idx: 6  }, // moderate-to-busy
  { max: 25,  idx: 7  }, // busy
  { max: 35,  idx: 8  }, // crowded starts here
  { max: 50,  idx: 9  }, // crowded
  { max: 1e9, idx: 10 }, // packed
];
// Vehicle side of the activity index. "Business activity" on these cameras
// is foot traffic AND vehicle traffic (the collector's own definition), but
// the index used to read `person` only - a junction moving 9 buses scored
// 0/10 "Quiet". Vehicles get their own weighted load (a bus occupies far
// more street than a bicycle) and their own bands; the final index is the
// busier of the two sides, so pedestrian plazas keep their old behavior.
const VEHICLE_LOAD_WEIGHTS = {
  car: 1.0, truck: 2.5, bus: 2.5, motorcycle: 0.5, bicycle: 0.3, train: 3.0,
};
const VEHICLE_BANDS = [
  { max: 0,   idx: 0  }, // no traffic
  { max: 1,   idx: 1  }, // one vehicle
  { max: 3,   idx: 2  }, // sparse
  { max: 5,   idx: 3  }, // a handful passing
  { max: 8,   idx: 5  }, // steady flow
  { max: 12,  idx: 6  }, // lively junction
  { max: 18,  idx: 7  }, // busy
  { max: 26,  idx: 8  }, // heavy traffic
  { max: 38,  idx: 9  }, // jammed
  { max: 1e9, idx: 10 }, // gridlock
];
const COMBINED_BIN_MIN = 5;

// Anomalies: the collector is the single source of truth. Every footfall doc
// carries `is_anomaly` and (when flagged) an `anomaly` map with
// kind/metric/window/z/observed/expected computed server-side from robust
// statistics + the hour-of-week profile. The dashboard only RENDERS those
// fields — it no longer recomputes z-scores client-side, so what you see is
// exactly what the collector flagged (and snapshotted) at sample time.

// tileState is keyed by slot_id (stable across fallback changes) — the video/
// header re-renders when active_cam changes, but chart history is preserved.
const tileState = {};

// YouTube IFrame Player API state (declared up here because buildVideoInto,
// which runs during the initial tile render below, calls withYouTubeAPI -
// these must be initialized before that first call, not in the helper block
// further down where a `let`/`const` TDZ would throw). See the block near
// mountYouTubePlayer for what they drive.
let _ytApiState = 0;              // 0 unloaded, 1 loading, 2 ready
let _ytHostSeq = 1;              // fallback host-id counter
const _ytReadyQ = [];
const YT_LIVE_MAX_DRIFT_S = 20;  // seconds behind live before we snap forward
let combinedChart = null;
let currentGridConfig = null;   // last config/grid doc — {slots: [...]}

// ---------- 1. Render tile skeletons -----------------------------------------

for (const slot of GRID_SLOTS) {
  const tile = document.createElement("div");
  tile.className = "tile";
  tile.dataset.slot = slot.slot_id;
  // Compact tile: header row above the video, KPIs/age overlaid on the video.
  // The tile no longer stacks metrics/badges/chart below the video — those all
  // moved to the header (badges) and overlay (KPIs + age) so tiles fit two
  // columns × two rows within one viewport.
  tile.innerHTML = `
    <div class="tile-head">
      <div class="tile-head-left">
        <h2 data-cam-name>${escapeHtml(slot.placeholder_name)}</h2>
        <div class="city" data-cam-area>${escapeHtml(slot.display_area)}</div>
      </div>
      <div class="tile-head-right">
        ${TWIN_MODE ? "" : `<button class="analyze-btn" data-analyze
                title="Live analysis - pick one layer for this camera"
                style="cursor:pointer;border:1px solid #334155;background:#1e293b;color:#e2e8f0;border-radius:6px;padding:2px 8px;font-size:13px">🔬</button>`}
        <span class="activity-badge act-unknown" data-activity>
          <span class="dot"></span><span data-activity-text>-/10</span>
        </span>
        <span class="anomaly-badge unk" data-anomaly title="no data yet">
          <span class="dot"></span><span data-anomaly-text>ok</span>
        </span>
        <span class="fallback-badge" data-fallback style="display:none"></span>
        <a class="anomaly-thumb" data-anomaly-thumb target="_blank" rel="noopener"
           style="display:none" title="open snapshot of latest anomaly">
          <img alt="" />
        </a>
      </div>
    </div>
    <div class="video-wrap" data-video-wrap>
      <div class="video-overlay-bottom" data-overlay>
        <span class="kpi"><span class="lbl">People</span>
          <span class="val" data-k="person">-</span></span>
        <span class="kpi vehicles"><span class="lbl">Vehicles</span>
          <span class="val" data-k="vehicles">-</span></span>
        <span class="kpi" data-speed-wrap style="display:none"
              title="median speed of moving vehicles this sample - burst-based estimate (each vehicle scaled by its own class length), roughly ±40%">
          <span class="lbl">~Speed</span>
          <span class="val" data-k="speed">-</span></span>
        <span class="kpi"><span class="lbl">24h avg</span>
          <span class="val" data-k="avg">-</span></span>
        <span class="kpi"><span class="lbl">24h peak</span>
          <span class="val" data-k="peak">-</span></span>
        <span class="age" data-age title="age of the counts - the video is live, the numbers are the collector's most recent sample"></span>
      </div>
    </div>
  `;
  tilesEl.appendChild(tile);

  tileState[slot.slot_id] = {
    slot,
    tile,
    camNameEl:    tile.querySelector("[data-cam-name]"),
    camAreaEl:    tile.querySelector("[data-cam-area]"),
    videoWrap:    tile.querySelector("[data-video-wrap]"),
    overlay:      tile.querySelector("[data-overlay]"),
    latestVals:   tile.querySelectorAll("[data-k]"),
    activityBadge: tile.querySelector("[data-activity]"),
    activityText:  tile.querySelector("[data-activity-text]"),
    speedWrap:     tile.querySelector("[data-speed-wrap]"),
    anomalyBadge: tile.querySelector("[data-anomaly]"),
    anomalyText:  tile.querySelector("[data-anomaly-text]"),
    fallbackBadge: tile.querySelector("[data-fallback]"),
    anomalyThumb: tile.querySelector("[data-anomaly-thumb]"),
    ageEl:        tile.querySelector("[data-age]"),
    // crossings/samples footnotes were removed with the below-video row;
    // line-crossing info still flows through updateStrip on the model-view
    // side card, and 24h sample counts show in the combined chart below.
    crossEl:      null,
    samplesEl:    null,
    chartCanvas:  null,       // per-tile mini chart removed (kept combined 24h chart)
    chart: null,
    history: [],
    lastSampleMs: null,   // epoch ms of the last OK sample; drives the age label
    currentActiveCam: null,   // updated by applyGridConfig
    currentHlsInstance: null, // hls.js instance we own; destroyed on rebuild
  };
  // Render initial placeholder video so viewers see something before
  // config/grid arrives. In local mode the placeholder IS the picked camera
  // (with its own embed/HLS), and no Firestore doc will replace it.
  buildVideoInto(tileState[slot.slot_id],
    { active_hls: slot.placeholder_hls, active_embed: slot.placeholder_embed,
      active_page: slot.placeholder_page },
    slot);
  // The 🔬 Live Analysis button only exists in main mode (see the
  // TWIN_MODE ternary in the template above). In twin mode the button
  // is not rendered, so querySelector returns null - guard so we don't
  // crash the for-loop and end up with a 1-tile grid instead of 2x2.
  const _anBtn = tile.querySelector("[data-analyze]");
  if (_anBtn) _anBtn.addEventListener("click", () =>
    openAnalysisPicker(tileState[slot.slot_id]));
}

// ---------- 1b0. Private-backend probe (fix 2) --------------------------------
// One truth source for "is this the operator's PRIVATE dashboard": ask the
// server. Only dashboard_server.py answers /api/ping with {private:true};
// the hosted public copy has no backend (fetch fails or returns HTML).
// Hostname sniffing is gone - it lied behind proxies. The probe gates the
// send-report field (1b2) and reveals the live-analysis buttons (1c).
let PRIVATE_BACKEND = false;
const _privateProbe = (async () => {
  try {
    const r = await fetch("/api/ping", { cache: "no-store" });
    if (r.ok) {
      const j = await r.json();
      PRIVATE_BACKEND = !!(j && j.private === true);
    }
  } catch (_) { PRIVATE_BACKEND = false; }
  document.body.classList.toggle("private-backend", PRIVATE_BACKEND);
  return PRIVATE_BACKEND;
})();

// ---------- 1b2. Send-report button (2026-08-09) ------------------------------
// One button, two mechanisms: on the operator's PRIVATE dashboard the
// field+button post to this server's /api/send-report; the hosted PUBLIC
// dashboard keeps the link variant that opens the send-report GitHub
// workflow, where GitHub login gates abuse.
(async () => {
  const priv = document.getElementById("send-report-private");
  const pub = document.getElementById("send-report-public");
  if (!priv || !pub) return;
  if (!await _privateProbe) return;        // public stays on the link
  pub.style.display = "none";
  priv.style.display = "flex";
  const toEl = document.getElementById("send-report-to");
  const btn = document.getElementById("send-report-btn");
  const msg = document.getElementById("send-report-msg");
  btn.addEventListener("click", async () => {
    const to = (toEl.value || "").trim();
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(to)) {
      msg.textContent = "Invalid email address";
      msg.style.color = "#f87171";
      return;
    }
    btn.disabled = true;
    msg.style.color = "#94a3b8";
    msg.textContent = "Building & sending... (~2 min)";
    try {
      const r = await fetch(`/api/send-report?to=${encodeURIComponent(to)}`,
                            { method: "POST" });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || r.status);
      msg.style.color = "#4ade80";
      msg.textContent = `Sent to ${to}`;
    } catch (e) {
      msg.style.color = "#f87171";
      msg.textContent = "Failed: " + e.message;
    } finally {
      btn.disabled = false;
    }
  });
})();

// ---------- 1c. fix 2: LIVE advanced analysis ---------------------------------
// One layer per camera, up to four live analyses across the grid (= the
// four tiles, so the cap is structural). Picking a layer morphs THAT tile
// in place: its video is replaced by a live stream of analyzed frames of
// the SAME camera, polled from the local server at ~1 fps - the honest
// pace of CPU inference (four concurrent analyses share one model and
// degrade to ~0.3-0.5 fps each). Unanalyzed tiles keep full-rate video;
// the VM collector is never involved. Switching layers on a running tile
// keeps the session's stream + accumulators (heat map, line counters,
// gesture history) - heat -> gestures -> heat resumes, never restarts.
const ANALYSIS_LAYER_DEFS = [
  ["heat",     "Heat signature"],
  ["paths",    "Paths & speeds"],
  ["pose",     "Pose & skeleton"],
  ["gestures", "Hand gestures"],
  ["body",     "Body anomalies"],
  ["faces",    "Face detection"],
  ["line",     "Line crossing"],
];
const ANALYSIS_POLL_MS = 1000;

const analysisPanel = document.createElement("div");
analysisPanel.style.cssText =
  "display:none;position:fixed;inset:0;z-index:60;background:rgba(2,6,23,.72);" +
  "align-items:center;justify-content:center";
analysisPanel.innerHTML = `
  <div style="background:#0f172a;border:1px solid #334155;border-radius:12px;
              padding:20px 22px;max-width:440px;width:92%;color:#e2e8f0;
              font-size:15px">
    <h3 style="margin:0 0 4px;font-size:17px">Live analysis -
      <span data-an-cam></span></h3>
    <div style="color:#94a3b8;font-size:13px;margin-bottom:12px">
      One layer per camera, up to 4 live analyses across the grid.
      The tile becomes a live analyzed stream (~1 fps on local CPU);
      Stop on the tile returns the video.</div>
    <div data-an-boxes style="display:grid;grid-template-columns:1fr 1fr;
         gap:8px 14px;margin-bottom:14px"></div>
    <div data-an-err style="color:#f87171;font-size:13px;min-height:18px"></div>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <button data-an-run style="cursor:pointer;background:#2563eb;border:0;
              color:#fff;border-radius:8px;padding:7px 18px;font-size:14px">
        Start</button>
      <button data-an-editline style="cursor:pointer;background:#334155;border:0;
              color:#fff;border-radius:8px;padding:7px 14px;font-size:14px;
              display:none">Edit counting line</button>
      <button data-an-cancel style="cursor:pointer;background:#1e293b;
              border:1px solid #334155;color:#e2e8f0;border-radius:8px;
              padding:7px 14px;font-size:14px">Cancel</button>
    </div>
  </div>`;
document.body.appendChild(analysisPanel);

// Show/hide the "Edit counting line" button as the user toggles layers.
analysisPanel.addEventListener("change", (e) => {
  if (e.target && e.target.name === "an-layer") {
    const el = analysisPanel.querySelector("[data-an-editline]");
    el.style.display = (e.target.value === "line") ? "" : "none";
  }
});

analysisPanel.querySelector("[data-an-editline]").addEventListener("click",
  () => {
    if (!_anTarget) return;
    const cam = tileAnalysisCamId(_anTarget);
    if (!cam) { alert("No active camera on this tile yet"); return; }
    analysisPanel.style.display = "none";
    // A live analysis frame is the freshest snapshot the tile can offer;
    // if none is running yet, /api/analysis/frame returns the boot-poster
    // and the editor still works (the line is normalized, not pixel-tied).
    window.openLineEditor(cam,
      `/api/analysis/frame?cam=${encodeURIComponent(cam)}&_=${Date.now()}`);
  });

const _anBoxes = analysisPanel.querySelector("[data-an-boxes]");
for (const [key, label] of ANALYSIS_LAYER_DEFS) {
  const lab = document.createElement("label");
  lab.style.cssText = "display:flex;gap:7px;align-items:center;cursor:pointer";
  lab.innerHTML = `<input type="radio" name="an-layer" value="${key}"> ${label}`;
  _anBoxes.appendChild(lab);
}

let _anTarget = null;   // tileState entry the picker is open for

function tileAnalysisCamId(st) {
  // The tile's OWN camera - the one whose video the operator watches.
  // Local preview: the picked slot (local_grid.json), resolved by the
  // server. Cloud mode: the collector's active camera for this slot.
  return LOCAL_MODE ? st.slot.slot_id : (st.currentActiveCam || null);
}

function openAnalysisPicker(st) {
  _anTarget = st;
  analysisPanel.querySelector("[data-an-cam]").textContent =
    st.camNameEl.textContent || st.slot.display_area;
  analysisPanel.querySelector("[data-an-err]").textContent = "";
  const current = st.analysis ? st.analysis.layer : null;
  for (const rb of _anBoxes.querySelectorAll("input"))
    rb.checked = rb.value === current;
  analysisPanel.querySelector("[data-an-run]").textContent =
    st.analysis ? "Switch layer" : "Start";
  // Match the initial visibility of the "Edit counting line" button to
  // the currently-selected layer (the change listener only fires on
  // subsequent picks).
  analysisPanel.querySelector("[data-an-editline]").style.display =
    (current === "line") ? "" : "none";
  analysisPanel.style.display = "flex";
}

analysisPanel.querySelector("[data-an-cancel]").addEventListener("click",
  () => { analysisPanel.style.display = "none"; });

analysisPanel.querySelector("[data-an-run]").addEventListener("click",
  async () => {
    const errEl = analysisPanel.querySelector("[data-an-err]");
    const picked = _anBoxes.querySelector("input:checked");
    if (!picked) {
      errEl.textContent = "Pick a layer";
      return;
    }
    const st = _anTarget;
    const cam = st && tileAnalysisCamId(st);
    if (!cam) {
      errEl.textContent =
        "Camera not identified yet - wait for the first grid config";
      return;
    }
    const runBtn = analysisPanel.querySelector("[data-an-run]");
    runBtn.disabled = true;
    errEl.textContent = "";
    try {
      const r = await fetch(
        `/api/analysis/start?cam=${encodeURIComponent(cam)}` +
        `&layer=${encodeURIComponent(picked.value)}`,
        { method: "POST" });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || r.status);
      beginTileAnalysis(st, cam, picked.value);
      analysisPanel.style.display = "none";
    } catch (e) {
      errEl.textContent = "Failed to start: " + e.message;
    } finally {
      runBtn.disabled = false;
    }
  });

const _layerLabel = Object.fromEntries(ANALYSIS_LAYER_DEFS);

// CANVAS OVERLAY approach (2026-08-12): live analysis no longer replaces
// the tile's video with a 1-fps JPEG stream. Instead we KEEP the video
// playing at native fps and paint the tick's annotations (boxes /
// skeleton / heatmap / faces / paths / line) on a canvas layered above
// it. Metadata rides the X-Analysis-Meta response header of the same
// /api/analysis/frame endpoint (base64 JSON, produced per-tick by
// live_analysis.Session._build_meta). Operator sees smooth 30 fps video
// PLUS refreshing analysis overlays - no more 'video is frozen' feel.
function beginTileAnalysis(st, cam, layer) {
  if (st.analysis) {
    // Same tile, new layer: server switched in-place (stream + accumulators
    // survive); just relabel the tag, the poller runs on. Also refresh
    // the layer-specific control buttons (line-draw is only shown for
    // the 'line' layer, hidden for the rest).
    st.analysis.layer = layer;
    if (st.analysis.tag)
      st.analysis.tag.firstChild.textContent =
        `🔬 ${_layerLabel[layer] || layer}`;
    _updateLineButton(st);
    return;
  }
  // Video stays. Add canvas overlay + a stop tag, both positioned
  // absolutely inside videoWrap so the tile geometry never shifts.
  const canvas = document.createElement("canvas");
  canvas.className = "analysis-overlay on";
  canvas.width = 640; canvas.height = 360;   // resized to img_w/img_h on 1st meta
  st.videoWrap.appendChild(canvas);

  const tag = document.createElement("span");
  tag.className = "analysis-live-tag on";
  tag.title = "click to stop live analysis and clear the overlay";
  const lbl = document.createElement("span");
  lbl.textContent = `🔬 ${_layerLabel[layer] || layer}`;
  const x = document.createElement("span");
  x.textContent = " ✕";
  x.style.marginLeft = "6px";
  tag.appendChild(lbl); tag.appendChild(x);
  st.videoWrap.appendChild(tag);
  tag.addEventListener("click", () => stopTileAnalysis(st));

  // Line-layer draw button: appears next to the tag when layer='line' so
  // the operator can define the counting line by clicking two points on
  // the video, in place, without an out-of-context modal.
  const lineBtn = document.createElement("span");
  lineBtn.className = "analysis-line-btn";
  lineBtn.style.cssText =
    "position:absolute;top:8px;right:126px;background:rgba(34,197,94,0.9);" +
    "color:#fff;padding:2px 8px;border-radius:6px;font-size:11px;" +
    "font-weight:600;pointer-events:auto;cursor:pointer;z-index:3;display:none";
  lineBtn.textContent = "✎ Draw line";
  lineBtn.title = "click 2 points on the video to place a counting line";
  st.videoWrap.appendChild(lineBtn);

  st.analysis = {
    cam, layer, canvas, ctx: canvas.getContext("2d"), tag, lineBtn,
    lineDrawing: false, linePts: [],
    failures: 0, lastRestart: 0, inflight: false,
    timer: setInterval(() => pollAnalysisFrame(st), ANALYSIS_POLL_MS),
  };
  lineBtn.addEventListener("click", () => _startLineDraw(st));
  canvas.addEventListener("click", (ev) => _onCanvasClick(st, ev));
  _updateLineButton(st);
  pollAnalysisFrame(st);
}

// Show / hide the "Draw line" button based on the current layer.
function _updateLineButton(st) {
  const a = st.analysis;
  if (!a || !a.lineBtn) return;
  a.lineBtn.style.display = (a.layer === "line") ? "" : "none";
  if (a.layer !== "line") {
    a.lineDrawing = false;
    a.linePts = [];
    if (a.canvas) a.canvas.style.pointerEvents = "";
    if (a.canvas) a.canvas.style.cursor = "";
  }
}

function _startLineDraw(st) {
  const a = st.analysis;
  if (!a || a.layer !== "line") return;
  a.lineDrawing = true;
  a.linePts = [];
  a.canvas.style.pointerEvents = "auto";
  a.canvas.style.cursor = "crosshair";
  a.lineBtn.textContent = "click point 1 on video";
  a.lineBtn.style.background = "rgba(37,99,235,0.95)";
}

async function _onCanvasClick(st, ev) {
  const a = st.analysis;
  if (!a || !a.lineDrawing) return;
  const rect = a.canvas.getBoundingClientRect();
  const x = (ev.clientX - rect.left) / rect.width;
  const y = (ev.clientY - rect.top) / rect.height;
  a.linePts.push([Number(x.toFixed(4)), Number(y.toFixed(4))]);
  if (a.linePts.length === 1) {
    a.lineBtn.textContent = "click point 2 on video";
    return;
  }
  // Two points collected: persist + finish.
  const body = JSON.stringify({ line: a.linePts });
  try {
    const r = await fetch(
      `/api/lines?cam=${encodeURIComponent(a.cam)}`,
      { method: "POST",
        headers: { "Content-Type": "application/json" },
        body });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      a.lineBtn.textContent = "✕ " + (j.error || `HTTP ${r.status}`);
      a.lineBtn.style.background = "rgba(239,68,68,0.9)";
    } else {
      a.lineBtn.textContent = "✓ line saved";
      a.lineBtn.style.background = "rgba(34,197,94,0.9)";
      // Reflect the new line on the overlay immediately - the next poll
      // will paint the server-side version in ~1 s.
      drawLineOverlay(a.ctx, { line: a.linePts,
                                cross_in: 0, cross_out: 0 });
    }
  } catch (e) {
    a.lineBtn.textContent = "✕ " + e.message;
    a.lineBtn.style.background = "rgba(239,68,68,0.9)";
  }
  a.lineDrawing = false;
  a.linePts = [];
  a.canvas.style.pointerEvents = "";
  a.canvas.style.cursor = "";
  // Restore "Draw line" label after a moment.
  setTimeout(() => {
    if (a.lineBtn && a.layer === "line") {
      a.lineBtn.textContent = "✎ Redraw line";
      a.lineBtn.style.background = "rgba(34,197,94,0.9)";
    }
  }, 1500);
}

async function pollAnalysisFrame(st) {
  const a = st.analysis;
  if (!a || a.inflight) return;
  a.inflight = true;
  try {
    const r = await fetch(
      `/api/analysis/frame?cam=${encodeURIComponent(a.cam)}&_=${Date.now()}`,
      { cache: "no-store" });
    if (r.status === 200) {
      const metaB64 = r.headers.get("X-Analysis-Meta");
      if (metaB64) {
        try {
          // atob->UTF-8 dance for non-ASCII (class names etc.).
          const bin = atob(metaB64);
          const bytes = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
          const meta = JSON.parse(new TextDecoder("utf-8").decode(bytes));
          drawAnalysisOverlay(a, meta);
          if (a.tag && a.tag.firstChild)
            a.tag.firstChild.textContent =
              `🔬 ${_layerLabel[meta.layer] || meta.layer}`;
          a.failures = 0;
        } catch (e) { console.warn("meta parse failed:", e); }
      }
    } else if (r.status === 202) {
      if (a.tag && a.tag.firstChild)
        a.tag.firstChild.textContent = "🔬 starting...";
    } else if (r.status === 404 || r.status === 410) {
      a.failures += 1;
      if (Date.now() - a.lastRestart > 5000) {
        a.lastRestart = Date.now();
        if (a.tag && a.tag.firstChild)
          a.tag.firstChild.textContent = "🔬 reconnecting...";
        fetch(`/api/analysis/start?cam=${encodeURIComponent(a.cam)}`
              + `&layer=${encodeURIComponent(a.layer)}`,
              { method: "POST" }).catch(() => {});
      }
    } else {
      a.failures += 1;
    }
  } catch (_) {
    a.failures += 1;
  } finally {
    a.inflight = false;
  }
  if (a.failures > 8 && a.tag && a.tag.firstChild)
    a.tag.firstChild.textContent = "🔬 unreachable";
}

function stopTileAnalysis(st) {
  const a = st.analysis;
  if (!a) return;
  clearInterval(a.timer);
  st.analysis = null;
  fetch(`/api/analysis/stop?cam=${encodeURIComponent(a.cam)}`,
        { method: "POST" }).catch(() => {});
  if (a.canvas) a.canvas.remove();
  if (a.tag) a.tag.remove();
  if (a.lineBtn) a.lineBtn.remove();
  // Video was never replaced, nothing to rebuild. Tear down the
  // Line-layer history strip if one was showing on this tile.
  const strip = st.tile && st.tile.querySelector(".crossings-strip");
  if (strip) strip.remove();
}

// ---- Canvas overlay painters ---------------------------------------------
// All coordinates in meta are ORIGINAL frame pixel space (meta.img_w x
// meta.img_h). Canvas backing store is sized to that; CSS scales it to
// fit the displayed video via `.analysis-overlay { inset: 0 }`, so
// coordinates line up 1:1 with the underlying video regardless of the
// tile's actual pixel size.
function drawAnalysisOverlay(a, meta) {
  const canvas = a.canvas, ctx = a.ctx;
  if (canvas.width !== meta.img_w || canvas.height !== meta.img_h) {
    canvas.width = meta.img_w || 640;
    canvas.height = meta.img_h || 360;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const layer = meta.layer;
  if (layer === "heat")      drawHeatOverlay(ctx, meta);
  if (layer !== "heat")      drawBoxesOverlay(ctx, meta.boxes || []);
  if (layer === "pose")      drawSkeletonOverlay(ctx, meta.skeleton || []);
  if (layer === "faces")     drawFacesOverlay(ctx, meta.faces || [], meta);
  if (layer === "line")      drawLineOverlay(ctx, meta);
  if (layer === "paths" || layer === "gestures" || layer === "body")
    drawTracksOverlay(ctx, meta.tracks || []);
}

function drawBoxesOverlay(ctx, boxes) {
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(79,140,255,0.75)";
  ctx.font = "12px sans-serif";
  ctx.fillStyle = "rgba(79,140,255,0.95)";
  for (const b of boxes) {
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    if (b.cls) {
      const label = `${b.cls} ${Math.round((b.conf || 0) * 100)}%`;
      const w = ctx.measureText(label).width + 6;
      ctx.fillStyle = "rgba(15,23,42,0.85)";
      ctx.fillRect(b.x1, b.y1 - 14, w, 14);
      ctx.fillStyle = "rgba(255,255,255,0.95)";
      ctx.fillText(label, b.x1 + 3, b.y1 - 3);
    }
  }
}

function drawHeatOverlay(ctx, meta) {
  const grid = meta.heat, gw = meta.heat_w, gh = meta.heat_h;
  if (!grid || !gw || !gh) return;
  const cw = ctx.canvas.width / gw, ch = ctx.canvas.height / gh;
  let mx = 0;
  for (const row of grid) for (const v of row) if (v > mx) mx = v;
  if (mx <= 0) return;
  for (let y = 0; y < gh; y++) {
    const row = grid[y];
    for (let x = 0; x < gw; x++) {
      const v = row[x] / mx;
      if (v <= 0.05) continue;
      const a = Math.min(0.65, v * 0.75);
      // Blue -> yellow -> red gradient
      const r = Math.round(255 * Math.min(1, v * 1.5));
      const g = Math.round(255 * Math.max(0, 1 - Math.abs(v - 0.5) * 2));
      const b = Math.round(255 * Math.max(0, 1 - v * 1.5));
      ctx.fillStyle = `rgba(${r},${g},${b},${a})`;
      ctx.fillRect(x * cw, y * ch, cw + 1, ch + 1);
    }
  }
}

const _COCO_POSE_LINKS = [
  [5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],
  [11,13],[13,15],[12,14],[14,16],
];
function drawSkeletonOverlay(ctx, skeletons) {
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(255,204,0,0.9)";
  ctx.fillStyle = "rgba(255,204,0,0.95)";
  for (const p of skeletons) {
    const kps = p.kps || [];
    for (const [a, b] of _COCO_POSE_LINKS) {
      const ka = kps[a], kb = kps[b];
      if (!ka || !kb || (ka[2] || 0) < 0.3 || (kb[2] || 0) < 0.3) continue;
      ctx.beginPath();
      ctx.moveTo(ka[0], ka[1]); ctx.lineTo(kb[0], kb[1]);
      ctx.stroke();
    }
    for (const k of kps) {
      if (k && (k[2] || 0) > 0.3) {
        ctx.beginPath();
        ctx.arc(k[0], k[1], 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}

function drawFacesOverlay(ctx, faces, meta) {
  if (meta && meta.faces_available === false) {
    ctx.fillStyle = "rgba(15,23,42,0.85)";
    ctx.fillRect(6, 6, 280, 22);
    ctx.fillStyle = "#f0a35e"; ctx.font = "12px sans-serif";
    ctx.fillText("faces: FACE_MODEL not configured", 12, 21);
    return;
  }
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(236,72,153,0.9)";
  ctx.fillStyle = "rgba(236,72,153,0.95)";
  ctx.font = "11px sans-serif";
  for (const f of faces) {
    const [x1, y1, x2, y2] = f.box;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (f.conf) ctx.fillText(`face ${Math.round(f.conf * 100)}%`, x1 + 2, y1 - 3);
  }
}

function drawLineOverlay(ctx, meta) {
  const W = ctx.canvas.width, H = ctx.canvas.height;
  const line = meta.line;
  if (line && line.length === 2 && line[0].length === 2) {
    const [[x1, y1], [x2, y2]] = line;
    ctx.strokeStyle = "rgba(0,255,136,0.9)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x1 * W, y1 * H); ctx.lineTo(x2 * W, y2 * H);
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(15,23,42,0.85)";
  ctx.fillRect(6, 6, 240, 26);
  ctx.fillStyle = "#e2e8f0"; ctx.font = "14px sans-serif";
  ctx.fillText(`IN ${meta.cross_in || 0} · OUT ${meta.cross_out || 0}`, 12, 25);
}

function drawTracksOverlay(ctx, tracks) {
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(56,189,248,0.85)";
  ctx.fillStyle = "rgba(56,189,248,0.95)";
  ctx.font = "11px sans-serif";
  for (const t of tracks) {
    const p = t.path || [];
    if (p.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(p[0][0], p[0][1]);
    for (let i = 1; i < p.length; i++) ctx.lineTo(p[i][0], p[i][1]);
    ctx.stroke();
    const last = p[p.length - 1];
    ctx.beginPath();
    ctx.arc(last[0], last[1], 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(15,23,42,0.85)";
    ctx.fillRect(last[0] + 5, last[1] - 14, 30, 14);
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.fillText(`#${t.tid}`, last[0] + 8, last[1] - 3);
    ctx.fillStyle = "rgba(56,189,248,0.95)";
  }
}

// Tear down whatever player the tile currently runs (hls.js / YT API /
// plain iframe) without touching the overlay - the same teardown
// buildVideoInto performs before a rebuild, reusable for the analysis
// morph.
function stopTileVideo(st) {
  st._vLastTime = null;
  st._vStrikes = 0;
  if (st.currentHlsInstance) {
    try { st.currentHlsInstance.destroy(); } catch (_) {}
    st.currentHlsInstance = null;
  }
  clearTimeout(st._ytStartTimer);
  if (st.ytPlayer) {
    try { st.ytPlayer.destroy(); } catch (_) {}
    st.ytPlayer = null;
  }
  for (const el of Array.from(st.videoWrap.children)) {
    if (el !== st.overlay) el.remove();
  }
}

// ---------- 1b. Model-view strip skeleton -----------------------------------
// One annotated-frame card per slot, laid out as a 2x2 grid below the search
// area. The image URL is the same `live_annotated_url` the collector publishes
// on each sample; the strip stays put and only its <img>/counts refresh.
//
// Robustness: the four cells are built up-front from GRID_SLOTS so every slot
// has a visible skeleton the moment the page loads. If a slot's Firestore doc
// arrives late (or its collector isn't currently uploading annotated frames),
// its cell shows a graceful "no live view yet" state instead of a broken img.
// If the image URL 404s, onerror rolls back to the empty state so a stale
// Storage URL doesn't leave a broken-image icon.
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
        <img alt="annotated detections" hidden />
      </a>
      <div class="nums">
        <span>👤 <b data-p>-</b></span>
        <span class="v">🚗 <b data-v>-</b></span>
        <button data-heat hidden
                title="toggle the long-horizon presence heatmap for this camera (where activity stands, weighted by dwell time)"
                style="margin-left:auto;background:none;border:1px solid #444;
                       border-radius:4px;color:inherit;cursor:pointer;
                       font-size:11px;padding:0 5px">🔥</button>
      </div>
      <!-- R5 (2026-08-12): tile-footer heat-layer/daypart dropdowns
           physically removed per operator - they were an operator-only
           debug control that cluttered the strip. The heatmap toggle
           button (🔥 above) still works with the default layer+part. -->
      <div class="age" data-age></div>`;
    stripEl.appendChild(cell);
    const s = {
      cell,
      lbl:   cell.querySelector("[data-lbl]"),
      link:  cell.querySelector("[data-link]"),
      empty: cell.querySelector("[data-empty]"),
      img:   cell.querySelector("img"),
      p:     cell.querySelector("[data-p]"),
      v:     cell.querySelector("[data-v]"),
      age:   cell.querySelector("[data-age]"),
      heatBtn: cell.querySelector("[data-heat]"),
      heatControls: cell.querySelector("[data-heat-controls]"),
      heatLayerSel: cell.querySelector("[data-heat-layer]"),
      heatPartSel:  cell.querySelector("[data-heat-part]"),
      camId: null,
      lastSampleMs: null,
      liveUrl: null,
      heatUrl: null,
      showHeat: false,
    };
    // Heatmap toggle: swaps the model-view image between the live
    // annotated frame and the dwell heatmap. On the private dashboard the
    // heat view is rendered on demand by /api/heatmap from the grids the
    // VM publishes, so the layer/daypart selectors work (fix 3); the
    // public copy keeps the single published person overlay.
    s.heatBtn.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      s.showHeat = !s.showHeat;
      s.heatBtn.style.borderColor = s.showHeat ? "#f0a35e" : "#444";
      s.heatControls.hidden = !(s.showHeat && PRIVATE_BACKEND);
      refreshHeatView(s);
    });
    for (const sel of (s.heatLayerSel && s.heatPartSel
                       ? [s.heatLayerSel, s.heatPartSel] : []))
      sel.addEventListener("change", () => refreshHeatView(s));
    s.img.addEventListener("error", () => {
      // Storage URL rotted, or Storage never got this snapshot. Roll back to
      // the empty state so the cell reads as "no live view" instead of a
      // broken image icon.
      s.img.hidden = true;
      if (s.empty) {
        s.empty.textContent = "no live view for this camera";
        s.empty.style.display = "";
      }
    });
    stripState[slot.slot_id] = s;
  }
}

// The heat view for one strip cell: private dashboards render the picked
// layer/daypart from the VM-published grids via /api/heatmap; the public
// copy falls back to the collector's single published overlay. Heat off
// restores the live annotated frame.
function refreshHeatView(s) {
  let url = null;
  if (!s.showHeat) {
    url = s.liveUrl;
  } else if (PRIVATE_BACKEND && s.camId) {
    const layer = (s.heatLayerSel && s.heatLayerSel.value) || "person";
    const part = (s.heatPartSel && s.heatPartSel.value) || "";
    url = `/api/heatmap?cam=${encodeURIComponent(s.camId)}`
        + `&layer=${encodeURIComponent(layer)}`
        + (part ? `&part=${encodeURIComponent(part)}` : "");
  } else if (s.heatUrl) {
    url = s.heatUrl;
  }
  if (!url) return;
  const busted = url + (url.includes("?") ? "&" : "?") + "t=" + Date.now();
  s.img.src = busted;
  s.link.href = busted;
  s.img.hidden = false;
}

// LOCAL_MODE local model-view poll (2026-08-13): main mode reads
// /snapshots/model_view/<slot_id>.json + .jpg produced by the notebook's
// app.local_producers.ModelViewProducer, then feeds updateStrip so the
// Model view - live section shows the operator's PICKED cameras instead
// of the empty "waiting for first sample" cells that used to sit there
// (they never filled because the VM's Firestore keys are Turkey slots,
// not the picked local_* ones). Twin mode is unchanged - twin mirrors
// the VM directly via Firestore.
if (LOCAL_MODE && stripEl) {
  const _pollLocalModelView = async () => {
    for (const slot of GRID_SLOTS) {
      const meta_url = `/snapshots/model_view/${slot.slot_id}.json?_=` + Date.now();
      const jpg_url  = `/snapshots/model_view/${slot.slot_id}.jpg?_=`  + Date.now();
      try {
        const r = await fetch(meta_url, { cache: "no-store" });
        if (!r.ok) continue;
        const j = await r.json();
        updateStrip(slot.slot_id, {
          cam_id:              j.cam_id,
          cam_name:            j.cam_name,
          person:              j.counts?.person,
          vehicles:            j.counts?.vehicles,
          ok:                  true,
          live_annotated_url:  jpg_url,
          ts:                  j.at,
        });
      } catch (_) { /* file not written yet - keep placeholder */ }
    }
  };
  _pollLocalModelView();
  setInterval(_pollLocalModelView, 15000);
}

function updateStrip(slotId, d) {
  const s = stripState[slotId];
  if (!s) return;
  // The strip shows the CLOUD collector's annotated frames ("what the counts
  // came from"), but its label was initialized from the LOCAL pick - so an
  // Istanbul frame sat under a "Bangkok, Thailand" caption. Name the actual
  // counts-source camera the moment a cloud sample identifies itself.
  if (LOCAL_MODE && d.cam_name && s.lbl && s.lbl.textContent !== d.cam_name) {
    s.lbl.textContent = d.cam_name;
  }
  if (d.cam_id) s.camId = d.cam_id;
  if (d.person   != null) s.p.textContent = d.person;
  if (d.vehicles != null) s.v.textContent = d.vehicles;
  if (s.heatBtn && (d.heatmap_url || (PRIVATE_BACKEND && s.camId))) {
    if (d.heatmap_url) s.heatUrl = d.heatmap_url;
    s.heatBtn.hidden = false;
  }
  if (d.ok && d.live_annotated_url) {
    s.liveUrl = d.live_annotated_url;
    if (s.showHeat) {
      // Keep the heat view current (private: re-rendered from the freshest
      // VM grids; public: the published overlay re-busted).
      refreshHeatView(s);
    } else {
      const url = d.live_annotated_url
          + (d.live_annotated_url.includes("?") ? "&" : "?")
          + "t=" + encodeURIComponent(d.ts || Date.now());
      s.img.src = url;
      s.img.hidden = false;
      s.link.href = url;
    }
    if (s.empty) s.empty.style.display = "none";
  } else if (d.ok && !d.live_annotated_url && s.empty
             && s.empty.textContent.startsWith("waiting")) {
    // The slot IS producing samples, just not annotated snapshots (Storage
    // not configured on the collector VM, or this particular sample failed
    // to upload). Say so instead of implying we're still waiting for the
    // very first sample.
    s.empty.textContent = "counts only - no annotated snapshot";
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
  // Remember the build inputs so the keep-live watchdog can rebuild this
  // tile from scratch when it detects a hard stall (stream died, YouTube
  // rotated the live URL, hls.js wedged) - the exact failure the operator
  // hit: a tile frozen on an hours-old frame with nothing left to revive it.
  st.lastVideoBuild = { cfg, slot };
  st._vLastTime = null;
  st._vStrikes = 0;
  st.ytBehindS = null;
  st.ytPlayerState = null;
  // Tear down any existing hls.js instance so we don't leak network sockets
  // when a fallback swap replaces the <video> element.
  if (st.currentHlsInstance) {
    try { st.currentHlsInstance.destroy(); } catch (_) {}
    st.currentHlsInstance = null;
  }
  // Tear down any prior YouTube player (active-cam change / rebuild) so the
  // watchdog never polls a detached player and we don't leak the iframe.
  clearTimeout(st._ytStartTimer);
  if (st.ytPlayer) {
    try { st.ytPlayer.destroy(); } catch (_) {}
    st.ytPlayer = null;
  }
  // Idle slot: the collector narrowed the grid because no country can
  // field this many live cameras right now (explicit idle flag from
  // config/grid - local picker slots carry no active_cam field and must
  // NOT match). Show the honest state instead of a dead player.
  if (cfg.idle) {
    for (const el of Array.from(st.videoWrap.children)) {
      if (el !== st.overlay) el.remove();
    }
    st.videoWrap.insertAdjacentHTML("afterbegin",
      `<div class="video-fallback">slot on standby -
        no additional live camera in any country right now</div>`);
    return;
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
    // preload="auto" tells the browser to start buffering IMMEDIATELY when
    // the element mounts, so the tile shows video the moment the page opens
    // instead of after the first user interaction. Combined with a hls.js
    // load kick below (autoStartLoad + play()) this pins down the case
    // where one tile stayed frozen until the user clicked into it.
    // `controls` back on. The KPI overlay moved to the TOP of the video-wrap
    // (see .video-overlay-bottom CSS which is anchored to top:0 now despite
    // the historical name), so the browser's control chrome at the bottom
    // no longer collides with the KPIs. controlsList strips the pieces we
    // don't want a stream monitor to have (nothing to download; no cast /
    // remote picker for a public dashboard).
    markup = `<video data-hls="${hlsUrl}" autoplay muted playsinline
                     controls controlsList="nodownload noremoteplayback"
                     preload="auto"></video>`;
  } else if (embed && (embed.includes("youtube.com/embed")
                       || embed.includes("youtube-nocookie.com/embed"))) {
    // YouTube live embeds (the Thailand/Japan/USA street cams). These are
    // mounted through the official IFrame Player API - NOT a raw iframe -
    // so the live-edge pinning below has reliable getDuration()/seekTo()
    // instead of the fire-and-forget postMessage that used to let a tile
    // drift to the start of the 12h DVR window ("-11:59:xx / live"). We
    // insert a placeholder div and hand it to mountYouTubePlayer once the
    // API is ready; a data-yt-embed marker carries the URL across the
    // API's async load.
    const vid = _ytVideoId(embed);
    if (vid) {
      const hostId = `yt-${st.slot ? st.slot.slot_id : Math.floor(_ytHostSeq++)}`;
      markup = `<div class="yt-host" id="${hostId}" data-yt-vid="${vid}"></div>`;
      st._ytPendingHost = hostId;
      st._ytPendingVid = vid;
    } else {
      markup = `<iframe src="${embed}" allow="autoplay; encrypted-media"
                       allowfullscreen></iframe>`;
    }
  } else if (embed && embed.includes("player.tvkur.com")) {
    // tvkur splash player (Konya) - a plain iframe; no DVR, no live-edge
    // concern, so it stays a bare embed.
    markup = `<iframe src="${embed}" allow="autoplay; encrypted-media"
                     allowfullscreen></iframe>`;
  } else if (page) {
    markup = `<div class="video-fallback">
                Live stream not embeddable from this site -
                <a href="${page}" target="_blank" rel="noopener">open camera page ↗</a>
              </div>`;
  } else {
    markup = `<div class="video-fallback">No live video available.</div>`;
  }
  // The KPI overlay lives inside video-wrap so its gradient sits on top of
  // the live frame. Replacing videoWrap.innerHTML wholesale would blow it
  // away every time the active cam changes - preserve it by rebuilding the
  // players' host DOM piecewise instead.
  for (const el of Array.from(st.videoWrap.children)) {
    if (el !== st.overlay) el.remove();
  }
  st.videoWrap.insertAdjacentHTML("afterbegin", markup);
  const video = st.videoWrap.querySelector("video[data-hls]");
  if (video) attachHls(st, video, cfg);
  if (st._ytPendingHost) {
    const hostId = st._ytPendingHost, vid = st._ytPendingVid;
    st._ytPendingHost = st._ytPendingVid = null;
    withYouTubeAPI(() => mountYouTubePlayer(st, hostId, vid));
  }
}

// ---------- YouTube IFrame Player API: reliable live-edge pinning ----------
// (state vars _ytApiState / _ytHostSeq / _ytReadyQ / YT_LIVE_MAX_DRIFT_S are
// declared near tileState at the top - they must exist before the initial
// render calls buildVideoInto.)

function _ytVideoId(embed) {
  const m = String(embed).match(/\/embed\/([\w-]{11})/);
  return m ? m[1] : null;
}

// Load the IFrame API exactly once and run queued callbacks when YT.Player
// exists. Multiple tiles share the single script load.
function withYouTubeAPI(cb) {
  if (_ytApiState === 2 && window.YT && window.YT.Player) return cb();
  _ytReadyQ.push(cb);
  if (_ytApiState !== 0) return;
  _ytApiState = 1;
  const prev = window.onYouTubeIframeAPIReady;
  window.onYouTubeIframeAPIReady = () => {
    if (typeof prev === "function") { try { prev(); } catch (_) {} }
    _ytApiState = 2;
    while (_ytReadyQ.length) { try { _ytReadyQ.shift()(); } catch (_) {} }
  };
  const s = document.createElement("script");
  s.src = "https://www.youtube.com/iframe_api";
  s.async = true;
  document.head.appendChild(s);
}

// Honest seconds-behind-live for a YT.Player, or null when unknowable.
// Measured on the live streams themselves (2026-08-05): getDuration() on a
// manifestless live stream returns a FROZEN value ~1h past the real head, so
// duration-minus-currentTime reports "an hour behind" for a player that is
// exactly at the live edge. The player's own progressState is truthful:
// seekableEnd tracks the real head at wall-clock rate and isAtLiveHead is
// authoritative. Without progressState we return null - unknown - instead of
// fabricating a drift that would trigger an endless seek storm.
function _ytBehind(p) {
  try {
    const ps = p.playerInfo && p.playerInfo.progressState;
    if (ps && ps.seekableEnd > 0) {
      if (ps.isAtLiveHead) return 0;
      const cur = p.getCurrentTime ? p.getCurrentTime() : ps.current;
      return Math.max(0, ps.seekableEnd - cur);
    }
  } catch (_) {}
  return null;
}

function _seekLive(p) {
  // Jump to the live head. seekTo(progressState.seekableEnd) verifiably
  // lands at the head; seekTo(getDuration()) is silently IGNORED on
  // manifestless live streams (the frozen duration lies past the seekable
  // range). Rate-limited per player: onStateChange fires after our own
  // seeks too, and an unthrottled handler turns that into a feedback loop.
  try {
    const now = Date.now();
    if (p._lastSeekTs && now - p._lastSeekTs < 5000) return;
    p._lastSeekTs = now;
    const ps = p.playerInfo && p.playerInfo.progressState;
    if (ps && ps.seekableEnd > 0) {
      p.seekTo(ps.seekableEnd - 1, true);
    } else {
      const d = typeof p.getDuration === "function" ? p.getDuration() : 0;
      if (d && isFinite(d) && d > 0) p.seekTo(d, true);
    }
    if (typeof p.playVideo === "function") p.playVideo();
  } catch (_) { /* not ready this instant; the interval retries */ }
}

function mountYouTubePlayer(st, hostId, vid) {
  const host = document.getElementById(hostId);
  if (!host) return;                             // tile was rebuilt already
  try { if (st.ytPlayer && st.ytPlayer.destroy) st.ytPlayer.destroy(); } catch (_) {}
  st._ytStarted = false;
  st.ytPlayer = new window.YT.Player(hostId, {
    width: "100%", height: "100%",
    videoId: vid,
    host: "https://www.youtube.com",
    playerVars: {
      autoplay: 1, mute: 1, playsinline: 1, controls: 1,
      modestbranding: 1, rel: 0,
    },
    events: {
      onReady: (e) => {
        try { e.target.mute(); } catch (_) {}
        try { e.target.playVideo(); } catch (_) {}
        _seekLive(e.target);
      },
      onStateChange: (e) => {
        // BUFFERING (3) / PLAYING (1) both mean the player came alive.
        if (e.data === 1 || e.data === 3) st._ytStarted = true;
        // PAUSED (2) - Chrome throttles multi-iframe autoplay; resume + relive.
        // PLAYING (1) - re-pin only when measurably behind (or unknowable,
        // e.g. the very first PLAYING before progressState exists); a player
        // already at the head must not be re-seeked on every state change.
        if (e.data === 2) {
          _seekLive(e.target);
        } else if (e.data === 1) {
          const b = _ytBehind(e.target);
          if (b == null || b > YT_LIVE_MAX_DRIFT_S) _seekLive(e.target);
        }
      },
    },
  });
  // Autoplay safety net: the IFrame API player is the ONLY way to reliably
  // seek to the live edge, but a few VISIBLE embedded contexts still refuse
  // to start it - those get the plain autoplaying embed after 8s (loses the
  // live-pin, never a black tile). A HIDDEN tab is different: autoplay
  // denial there is normal policy, and falling back would strand the tile
  // on a click-to-play embed forever - so hidden tabs keep the API player
  // armed and re-check until the tab is visible. In a normal foreground
  // Chrome muted autoplay starts well inside 8s and none of this runs.
  clearTimeout(st._ytStartTimer);
  const startCheck = () => {
    let state = -1;
    try { state = st.ytPlayer && st.ytPlayer.getPlayerState(); } catch (_) {}
    if (st._ytStarted || state === 1 || state === 3) return;   // it's alive
    if (document.visibilityState !== "visible") {
      // A hidden tab legitimately refuses autoplay - swapping to a plain
      // embed NOW would freeze the tile on click-to-play forever (the
      // black-tiles-after-background-open bug). Keep the API player and
      // re-check; the visibilitychange handler starts it the moment the
      // tab is seen again.
      st._ytStartTimer = setTimeout(startCheck, 8000);
      return;
    }
    console.warn("keep-live: YT API player did not start - plain-embed fallback", hostId);
    try { st.ytPlayer && st.ytPlayer.destroy(); } catch (_) {}
    st.ytPlayer = null;
    const emb = `https://www.youtube.com/embed/${vid}`
              + `?autoplay=1&mute=1&playsinline=1`;
    for (const el of Array.from(st.videoWrap.children)) {
      if (el !== st.overlay) el.remove();
    }
    st.videoWrap.insertAdjacentHTML("afterbegin",
      `<iframe src="${emb}" allow="autoplay; encrypted-media" allowfullscreen></iframe>`);
  };
  st._ytStartTimer = setTimeout(startCheck, 8000);
}

// Re-pin every live YouTube player to the live edge when the tab regains
// focus - background tabs throttle timers + autoplay, so a dashboard left
// in the background then revisited could sit paused mid-DVR. The periodic
// watchdog covers the steady state; this makes the correction instant on
// return.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "visible") return;
  for (const st of Object.values(tileState)) {
    if (st.ytPlayer) _seekLive(st.ytPlayer);
  }
});

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
    // Same overlay-preserving swap as buildVideoInto.
    for (const el of Array.from(st.videoWrap.children)) {
      if (el !== st.overlay) el.remove();
    }
    st.videoWrap.insertAdjacentHTML("afterbegin",
        `<iframe src="${embed}" allow="autoplay; encrypted-media"
                 allowfullscreen loading="lazy"></iframe>`);
  };
  if (window.Hls && window.Hls.isSupported()) {
    const hls = new window.Hls({ lowLatencyMode: true, liveSyncDuration: 4 });
    hls.loadSource(src);
    hls.attachMedia(video);
    // Kick play() the moment the manifest parses. Chrome allows
    // muted-autoplay but sometimes never fires it if the element was
    // rendered outside the viewport at attach time (which happens for the
    // bottom row before scroll). Explicit .play() removes that dependency
    // on scroll position; the promise-rejection swallow keeps the flow
    // clean when browsers block autoplay in exotic contexts.
    hls.on(window.Hls.Events.MANIFEST_PARSED, () => {
      const p = video.play();
      if (p && p.catch) p.catch(() => { /* autoplay blocked; user clicks play */ });
    });
    hls.on(window.Hls.Events.ERROR, (_, data) => {
      if (!data.fatal) return;
      console.warn("hls.js fatal error on", src, data);
      fallbackToEmbed();
    });
    st.currentHlsInstance = hls;
  } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
    video.src = src;
    video.addEventListener("loadedmetadata", () => {
      const p = video.play();
      if (p && p.catch) p.catch(() => {});
    }, { once: true });
    video.addEventListener("error", fallbackToEmbed, { once: true });
  } else {
    console.warn("No HLS playback support in this browser for", src);
    fallbackToEmbed();
  }
}

// ---------- 2b. Keep-live watchdog -------------------------------------------
// Runs for the page's whole life, every WATCH_EVERY_MS, while the tab is
// visible (background tabs throttle playback - correcting them is pointless
// and would fight the browser):
//   * YouTube: through the real YT.Player - _ytBehind() reads the player's
//     progressState (the only truthful behind-live source; getDuration()
//     is frozen garbage on manifestless streams). More than
//     YT_LIVE_MAX_DRIFT_S back (a DVR-window drift, the "-11:59 / live"
//     bug) or a paused/ended state -> _seekLive() snaps to the real head.
//   * HLS <video>: currentTime flat across two visible checks -> a
//     play()/startLoad kick, then a full rebuild from the remembered
//     build inputs.
const WATCH_EVERY_MS = 8_000;   // tight enough that drift never shows for long

function watchTilesLive() {
  if (document.visibilityState !== "visible") return;
  for (const st of Object.values(tileState)) {
    if (!st.videoWrap) continue;
    if (st.analysis) continue;   // analyzed tile: no player to keep alive
    const video = st.videoWrap.querySelector("video[data-hls]");
    if (video) {
      const t = video.currentTime;
      if (st._vLastTime != null && t <= st._vLastTime + 0.1) {
        st._vStrikes = (st._vStrikes || 0) + 1;
        if (st._vStrikes === 1) {
          try { if (st.currentHlsInstance) st.currentHlsInstance.startLoad(); } catch (_) {}
          const p = video.play();
          if (p && p.catch) p.catch(() => {});
        } else if (st._vStrikes >= 2 && st.lastVideoBuild) {
          console.warn("keep-live: rebuilding stalled HLS tile", st.slot && st.slot.slot_id);
          buildVideoInto(st, st.lastVideoBuild.cfg, st.lastVideoBuild.slot);
          continue;
        }
      } else {
        st._vStrikes = 0;
      }
      st._vLastTime = t;
    } else if (st.ytPlayer) {
      try {
        const p = st.ytPlayer;
        const state = p.getPlayerState ? p.getPlayerState() : 1;
        st.ytBehindS = _ytBehind(p);
        st.ytPlayerState = state;
        // Paused/cued/ended, or measurably drifted into the DVR window ->
        // snap to live. An unknown drift (null) is NOT a snap trigger here:
        // the steady state must be zero seeks, not a seek every 8s.
        if (state === 2 || state === 5 || state === 0
            || (st.ytBehindS != null && st.ytBehindS > YT_LIVE_MAX_DRIFT_S)) {
          _seekLive(p);
        }
      } catch (_) { /* player between states; next tick retries */ }
    }
  }
}
setInterval(watchTilesLive, WATCH_EVERY_MS);

// Console diagnostic: per-tile liveness at a glance.
window.__tileLiveDebug = () => Object.fromEntries(
  Object.entries(tileState).map(([sid, st]) => {
    const video = st.videoWrap && st.videoWrap.querySelector("video[data-hls]");
    const p = st.ytPlayer;
    let behind = null, state = null, cur = video ? Math.round(video.currentTime) : null;
    if (p) {
      try {
        const b = _ytBehind(p);
        behind = b == null ? null : Math.round(b);
        state = p.getPlayerState ? p.getPlayerState() : null;
        cur = p.getCurrentTime ? Math.round(p.getCurrentTime()) : null;
      } catch (_) {}
    }
    return [sid, {
      kind: video ? "hls" : (p ? "youtube" : "none"),
      curTime: cur,
      behindLiveS: behind,       // seconds behind the live edge (want ~0-20)
      playerState: state,        // YT: 1 playing, 2 paused, 3 buffering
      strikes: st._vStrikes || 0,
    }];
  }));

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
  // LOCAL preview keys its tiles local_0..3 while every cloud doc is keyed
  // slot_1..4 - without a join the KPIs, the activity index and the model
  // view all stare at keys that don't exist and sit empty forever (that is
  // exactly what happened after the grid refactor). cloudToTile() joins by
  // the active camera when the local pick matches a cloud slot, else by
  // position, so the cloud counts always land on SOME tile.
  const slotIds = new Set(GRID_SLOTS.map((s) => s.slot_id));
  onSnapshot(collection(db, "latest"), (snap) => {
    let alive = 0;
    for (const d of snap.docs) {
      const tid = cloudToTile(d.id);
      if (!tid || !slotIds.has(tid)) continue;
      const st  = tileState[tid];
      if (!st) continue;
      const rec = d.data();
      const ageS = rec.ts ? Math.round((Date.now() - new Date(rec.ts).getTime()) / 1000) : null;
      if (ageS != null && ageS < STALE_AGE_S) alive++;
      setLatest(st, rec);
    }
    // LOCAL preview mode (the notebook wrote local_grid.json): the tiles show
    // the cameras YOU picked as live video. Their COUNTS come from Firestore,
    // written by the 24/7 cloud collector - which watches its own country
    // ladder, not your local picks - so "no recent writes" here is EXPECTED,
    // not a fault. Say so instead of raising a false "collector down" alarm.
    if (LOCAL_MODE) {
      statusEl.innerHTML = alive > 0
        ? `<span class="live">● local preview</span> · ${GRID_SLOTS.length} picked cameras (live video) · ${alive} also live on the cloud collector`
        : `<span class="live">● local preview</span> · ${GRID_SLOTS.length} picked cameras (live video) · counts + anomalies come from the 24/7 cloud collector (watching its own grid)`;
    } else {
      statusEl.innerHTML = alive === GRID_SLOTS.length
        ? `<span class="live">● live</span> · ${alive}/${GRID_SLOTS.length} slots updating`
        : alive > 0
        ? `<span class="stale">● partial</span> · ${alive}/${GRID_SLOTS.length} slots updating`
        : `<span class="down">● no recent writes</span> · is the collector running?`;
    }
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
      const tid = cloudToTile(r.slot);
      if (!tid || !bySlot[tid]) continue;
      bySlot[tid].push(r);
    }
    for (const slot of GRID_SLOTS) {
      const rows = bySlot[slot.slot_id].sort((a, b) => a.ts.localeCompare(b.ts));
      tileState[slot.slot_id].history = rows;
      // Per-tile sparkline moved into the combined 24h chart to reclaim
      // vertical space; renderTileChart is kept for future re-enabling but
      // no-ops when chartCanvas is null.
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

// cloud slot_id -> local tile slot_id. Rebuilt on every config/grid change:
// match by the active camera first (name or embed - the local picker may
// have picked exactly what the collector runs), else fall back to position
// (slot_N -> the N-th local tile). Identity in cloud mode.
let cloudSlotMap = {};

function cloudToTile(cloudId) {
  if (!LOCAL_MODE) return cloudId;
  if (cloudSlotMap[cloudId]) return cloudSlotMap[cloudId];
  const m = /^slot_(\d+)$/.exec(cloudId || "");
  const byPos = m ? GRID_SLOTS[Number(m[1]) - 1] : null;
  return byPos ? byPos.slot_id : null;
}

function rebuildCloudSlotMap(cfg) {
  cloudSlotMap = {};
  if (!LOCAL_MODE || !cfg || !Array.isArray(cfg.slots)) return;
  const taken = new Set();
  for (const slotCfg of cfg.slots) {
    const hit = GRID_SLOTS.find((s) =>
        !taken.has(s.slot_id) &&
        ((s.placeholder_name && s.placeholder_name === slotCfg.active_cam_name)
         || (s.placeholder_embed && slotCfg.active_embed
             && s.placeholder_embed === slotCfg.active_embed)));
    if (hit) {
      cloudSlotMap[slotCfg.slot_id] = hit.slot_id;
      taken.add(hit.slot_id);
    }
  }
}

function applyGridConfig(cfg) {
  if (!cfg || !Array.isArray(cfg.slots)) return;
  rebuildCloudSlotMap(cfg);
  for (const slotCfg of cfg.slots) {
    const st = tileState[slotCfg.slot_id];
    if (!st) continue;
    if (st.currentActiveCam !== slotCfg.active_cam) {
      st.currentActiveCam = slotCfg.active_cam;
      st.camNameEl.textContent = slotCfg.active_cam_name || slotCfg.slot_id;
      st.camAreaEl.textContent = slotCfg.display_area || "";
      updateStripLabel(slotCfg.slot_id, slotCfg.active_cam_name,
                       slotCfg.display_area);
      if (st.analysis) {
        // Tile is mid-analysis (fix 2): don't stomp the analyzed stream;
        // remember the new video inputs so Stop rebuilds the CURRENT cam.
        st.lastVideoBuild = { cfg: slotCfg, slot: st.slot };
      } else {
        buildVideoInto(st, slotCfg, st.slot);
      }
    }
    if (slotCfg.idle) {
      // Grid narrowed: this slot is deliberately idle, not "on fallback".
      st.fallbackBadge.textContent = "standby";
      st.fallbackBadge.title =
        "grid narrowed - no country fields this many live cameras right now";
      st.fallbackBadge.style.display = "inline-block";
    } else if (slotCfg.active_cam !== slotCfg.primary) {
      st.fallbackBadge.textContent = "↳ fallback";
      st.fallbackBadge.title = `primary cam offline - using fallback: ${slotCfg.active_cam}`;
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
  // fix 2 tile identity (local preview): a tile carries CLOUD numbers only
  // when the cloud camera in this slot IS the picked camera. An unmatched
  // tile shows video + live analysis only - no KPIs, no badges, no age
  // from a DIFFERENT camera; the VM's numbers stay in the clearly-VM
  // areas (model strip, 24h chart, anomaly/events tables), labeled with
  // the cloud camera's own name via st.cloudCamName.
  if (d.cam_name) st.cloudCamName = d.cam_name;
  st.cloudMismatch = !!(LOCAL_MODE && d.cam_name && st.camNameEl
      && st.camNameEl.textContent
      && st.camNameEl.textContent !== d.cam_name);
  if (st.cloudMismatch) {
    st.overlay.style.display = "none";
    st.activityBadge.style.display = "none";
    st.anomalyBadge.style.display = "none";
    st.anomalyThumb.style.display = "none";
    updateStrip(st.slot.slot_id, d);   // the strip IS a cloud area
    return;
  }
  if (!st.analysis) st.overlay.style.display = "";
  st.activityBadge.style.display = "";
  st.anomalyBadge.style.display = "";
  set("person",   d.person);
  set("vehicles", d.vehicles);
  // Vehicle speed chip: shown only when this sample tracked moving vehicles
  // (a burst-based estimate; the tooltip carries the honesty disclaimer).
  if (st.speedWrap) {
    const sp = d.speeds;
    if (sp && sp.moving > 0 && sp.median_kmh > 0) {
      st.speedWrap.style.display = "";
      set("speed", `${sp.median_kmh} km/h`);
      st.speedWrap.title =
          `median of ${sp.moving} moving vehicle(s) this sample - ` +
          `burst estimate ±40% · max ~${sp.max_kmh} km/h` +
          (sp.per_class ? " · " + Object.entries(sp.per_class)
              .map(([c, v]) => `${c} ${v}`).join(", ") : "");
    } else {
      st.speedWrap.style.display = "none";
    }
  }
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
  const label = ageS < 90 ? `${ageS}s ago`
              : `${Math.round(ageS / 60)}m ago`;
  const memo = label + (stale ? "!" : "");
  if (memo !== st._ageMemo) {            // skip no-op DOM writes
    st._ageMemo = memo;
    st.ageEl.textContent = label;
    st.ageEl.classList.toggle("stale", stale);
  }
}

setInterval(() => {
  for (const st of Object.values(tileState)) renderSampleAge(st);
}, 1000);

function updateAggregates(slotId, rows) {
  const st = tileState[slotId];
  if (st.cloudMismatch) {
    // Unmatched local tile (fix 2): its 24h history belongs to the CLOUD
    // camera - it feeds the chart + tables, never this tile's widgets.
    return;
  }
  if (!rows.length) {
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

  const anomalies = rows.filter(isShownAnomaly);
  if (anomalies.length) {
    const last = anomalies[anomalies.length - 1];
    const d = describeAnomaly(last);
    st.anomalyBadge.className = "anomaly-badge warn";
    // Compact badge in the tile header - full detail on hover via title attr.
    st.anomalyText.textContent = `⚠ ${d.arrow} ${anomalies.length}`;
    st.anomalyBadge.title =
        `${d.arrow} ${d.metricLabel} ${d.kindLabel} at ${fmtTime(last.ts)} - ` +
        `${d.observed ?? "?"} vs ~${d.expected ?? "?"} expected ` +
        `(${anomalies.length} in 24h)`;
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
    st.anomalyText.textContent = "ok";
    st.anomalyBadge.title = `no anomalies in the last 24h (${rows.length} samples)`;
    st.anomalyThumb.style.display = "none";
  }

  setActivityBadge(st, computeActivity(rows));
}

// Absolute activity scale in FIXED bands. Replaces the old
// `(now / p90) * 8` formula which was broken in two ways:
//   1. On a quiet street the 24h p90 collapsed to 1, so a single
//      false-positive detection (a lamp post read as "person") produced
//      (2/1)*8 = 16 → clamped to 10/10 "Crowded". The user saw exactly
//      this on empty streets.
//   2. When there was steady daylong traffic, a modest instantaneous
//      dip below the p90 was scored "Quiet" even though 12 people is
//      objectively a busy scene.
// The activity bands sit on absolute person counts and reflect what
// "business activity" means for a downtown street camera - no history, no
// p90, no fabricated crowds on empty scenes. Table lives at module top for
// TDZ safety (see ACTIVITY_BANDS declaration near the file header).
function _bandIndex(n, bands = ACTIVITY_BANDS) {
  for (const b of bands) if (n <= b.max) return b.idx;
  return 10;
}
// Weighted vehicle load for one footfall row. Prefers the per-class
// `counts` map every collector record carries; falls back to the flat
// `vehicles` field (all treated as cars) for legacy docs.
function _vehicleLoad(r) {
  const c = r.counts;
  if (c && typeof c === "object") {
    let load = 0, seen = false;
    for (const [cls, w] of Object.entries(VEHICLE_LOAD_WEIGHTS)) {
      const n = c[cls];
      if (typeof n === "number" && n > 0) { load += w * n; }
      if (n != null) seen = true;
    }
    if (seen) return load;
  }
  return (r.vehicles ?? 0) * 1.0;
}
function _median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
}
function computeActivity(rows) {
  if (!rows.length) return null;
  // Median of the last 3 samples (~2 min of wall time): one glitchy burst
  // can no longer swing the badge, while "now" still means "right now".
  const tail   = rows.slice(-3);
  const people = Math.round(_median(tail.map((r) => Math.max(0, r.person ?? 0))));
  const load   = _median(tail.map(_vehicleLoad));
  const pIdx   = _bandIndex(people, ACTIVITY_BANDS);
  const vIdx   = _bandIndex(load, VEHICLE_BANDS);
  const idx    = Math.max(pIdx, vIdx);
  const label = idx <= 3 ? "Quiet"
              : idx <= 6 ? "Moderate"
              : idx <= 8 ? "Busy"
              : "Crowded";
  const last = rows[rows.length - 1];
  return { idx, label, pIdx, vIdx,
           now: last.person ?? 0,
           veh: last.vehicles ?? 0,
           load: Math.round(load * 10) / 10 };
}

function setActivityBadge(st, act) {
  const badge = st.activityBadge;
  const text  = st.activityText;
  if (!act) {
    badge.className = "activity-badge act-unknown";
    text.textContent = "-/10";
    badge.title = "activity index - not enough samples yet";
    return;
  }
  const cls = act.label.toLowerCase();
  badge.className = `activity-badge act-${cls}`;
  text.textContent = `${act.idx}/10`;
  badge.title = `activity ${act.idx}/10 - ${act.label} · ` +
      `people ${act.now} (${act.pIdx}/10) · ` +
      `vehicle load ${act.load} (${act.vIdx}/10, bus/truck weigh more) · ` +
      `index = busier of the two, median of last 3 samples`;
}

const TILE_CHART_LAST_N = 30;

function renderTileChart(slotId, rows) {
  const st = tileState[slotId];
  // Per-tile sparkline was removed to reclaim vertical space; the combined
  // 24h chart below the tiles carries the same story with more legibility.
  // Leaving the function in place so a future re-enable is just a skeleton
  // edit + this early return removal.
  if (!st.chartCanvas) return;
  const view = rows.slice(-TILE_CHART_LAST_N);
  const labels   = view.map((r) => fmtTime(r.ts));
  const people   = view.map((r) => r.person);
  const vehicles = view.map((r) => r.vehicles);
  // Anomalous samples render as enlarged red points on the metric that fired.
  const anomOn = (metric) => (r) =>
      isShownAnomaly(r) && ((r.anomaly?.metric ?? "person") === metric);
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
// COMBINED_BIN_MIN is declared at module top for TDZ safety.

// Each section below the tiles reveals ITSELF the moment it has content, and
// stays gone until then. `hidden` on the wrapper element is toggled here so an
// idle collector doesn't leave four empty placeholder boxes eating half the
// viewport.
function toggleSection(id, hasContent) {
  const el = document.getElementById(id);
  if (el) el.hidden = !hasContent;
}

// Cloud-area label for a tile's data series: in local-preview mode the
// 24h history routed to a tile belongs to the CLOUD camera, so the chart
// legend and the anomaly/events tables must carry the CLOUD camera's name
// - never the local pick's title (fix 2: no Bangkok titles on Istanbul
// curves).
function tileCloudLabel(slot) {
  const st = tileState[slot.slot_id];
  return (LOCAL_MODE && st && st.cloudCamName) ? st.cloudCamName
                                               : slot.display_area;
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
      if (!isShownAnomaly(r)) continue;
      if ((r.anomaly?.metric || "person") !== "person") continue;
      const t = new Date(r.ts).getTime();
      if (!Number.isFinite(t)) continue;
      set.add(Math.floor(t / binMs) * binMs);
    }
    anomBinsBySlot[slot.slot_id] = set;
  }

  const palette = ["#4f8cff", "#36d399", "#f0a35e", "#a78bfa", "#ff6b9d"];
  const datasets = GRID_SLOTS.map((slot, i) => {
    const bins = binsBySlot[slot.slot_id];
    const anom = anomBinsBySlot[slot.slot_id];
    const pointBg = binList.map((b) => anom.has(b) ? "#ef4444"
                                                   : palette[i % palette.length]);
    const pointR  = binList.map((b) => anom.has(b) ? 5 : 0);
    return {
      label: tileCloudLabel(slot),
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
      if (isShownAnomaly(r)) events.push({ area: tileCloudLabel(slot), r });
    }
  }
  toggleSection("anomaly-section", events.length > 0);
  if (!events.length) return;
  events.sort((a, b) => b.r.ts.localeCompare(a.r.ts));
  // AGGREGATE repeats: the same (kind, camera) firing again and again is one
  // STORY, not sixty rows - a stuck false trigger was drowning out every
  // other anomaly type. One row per (kind, area, metric), carrying the
  // latest occurrence, a repeat counter and the first-seen time.
  const groups = new Map();
  for (const { area, r } of events) {
    const a = r.anomaly || {};
    const key = `${a.kind}|${area}|${a.metric || ""}`;
    const g = groups.get(key);
    if (!g) {
      groups.set(key, { area, latest: r, count: 1, firstTs: r.ts });
    } else {
      g.count += 1;
      if (r.ts < g.firstTs) g.firstTs = r.ts;   // events sorted desc; track span
    }
  }
  const rows = [...groups.values()].slice(0, 30).map((g) => {
    const r = g.latest;
    const d = describeAnomaly(r);
    const snap = r.snapshot_annotated_url || r.snapshot_url;
    const expected = d.expected != null
        ? `~${d.expected}${d.bucket ? ` <span class="footnote">(${escapeHtml(d.bucket)} norm)</span>` : ""}`
        : "-";
    const times = g.count > 1
        ? `${fmtTime(r.ts)} <span class="footnote">(×${g.count} since ${fmtTime(g.firstTs)})</span>`
        : fmtTime(r.ts);
    return `<tr>
      <td>${times}</td>
      <td>${escapeHtml(g.area)}</td>
      <td class="${d.dir}">${d.arrow} ${escapeHtml(d.kindLabel)}</td>
      <td>${escapeHtml(d.metricLabel)}</td>
      <td>${d.observed ?? "-"}</td>
      <td>${expected}</td>
      <td>${snap ? `<a href="${snap}" target="_blank" rel="noopener">view</a>` : "-"}</td>
    </tr>`;
  }).join("");
  wrap.innerHTML = `<table class="reid">
    <thead><tr>
      <th>Latest</th><th>Area</th><th>Type</th><th>Metric</th>
      <th>Observed</th><th>Expected</th><th>Snapshot</th>
    </tr></thead>
    <tbody>${rows}</tbody></table>`;
}

// ---------- 6c. Operational events (loiter / returning) ----------------------

const EVENT_LABELS = {
  loiter:           { icon: "⏱", label: "prolonged presence" },
  returning:        { icon: "↩", label: "returning visitor" },
  static_departed:  { icon: "📤", label: "static object left" },
};

// Keep the full events list in module scope so the accordion can look up
// prior sightings of the same entity without re-querying Firestore.
let _ALL_EVENTS = [];

function renderEventsTable(events) {
  const wrap = document.getElementById("events-table-wrap");
  if (!wrap) return;
  _ALL_EVENTS = events;
  const slotLabel = (id) => {
    if (LOCAL_MODE) {
      // Events are CLOUD data - name the cloud camera, not the local pick.
      const tid = cloudToTile(id);
      const st = tid && tileState[tid];
      return (st && st.cloudCamName) || id;
    }
    const slot = GRID_SLOTS.find((s) => s.slot_id === id);
    return slot ? slot.display_area : id;
  };
  toggleSection("events-section", events.length > 0);
  if (!events.length) return;
  const rows = events.slice(0, 60).map((e, i) => {
    const meta = EVENT_LABELS[e.kind] || { icon: "•", label: e.kind };
    const detail = e.kind === "loiter"
        ? `${e.cls ?? "?"} stationary ${Math.round((e.duration_sec ?? 0) / 60)} min`
        : e.kind === "returning"
        ? `${e.cls ?? "?"} #${e.entity_id ?? "?"} back after ${Math.round((e.gap_seconds ?? 0) / 60)} min`
        : e.kind === "static_departed"
        ? `${e.cls ?? "?"} static ${Math.round((e.dwell_sec ?? 0) / 60)} min - now gone`
        : "";
    const snap = e.snapshot_url || e.fullframe_url;
    // Every row with an entity_id gets an expand-toggle - clicking it opens
    // an inline accordion that shows every past sighting of the same entity
    // at the same slot, so the user can eyeball whether the "back after N min"
    // claim really is the same object rather than a lookalike.
    const canExpand = e.entity_id != null;
    const toggle = canExpand
        ? `<span class="row-toggle" data-idx="${i}" title="show all sightings of this entity">▸</span>`
        : "";
    return `<tr class="ev-row">
      <td>${toggle} ${fmtTime(e.ts)}</td>
      <td>${escapeHtml(slotLabel(e.slot))}</td>
      <td>${meta.icon} ${escapeHtml(meta.label)}</td>
      <td>${escapeHtml(detail)}</td>
      <td>${snap ? `<a href="${snap}" target="_blank" rel="noopener">view</a>` : "-"}</td>
    </tr>
    <tr class="ev-accordion" data-idx="${i}" hidden><td colspan="5"></td></tr>`;
  }).join("");
  wrap.innerHTML = `<table class="reid">
    <thead><tr>
      <th>Time</th><th>Area</th><th>Event</th><th>Detail</th><th>Snapshot</th>
    </tr></thead>
    <tbody>${rows}</tbody></table>`;
  // Wire up expand clicks
  wrap.querySelectorAll(".row-toggle").forEach((t) => {
    t.addEventListener("click", (ev) => {
      ev.preventDefault();
      toggleEventAccordion(parseInt(t.dataset.idx, 10), t);
    });
  });
}

function toggleEventAccordion(idx, toggleEl) {
  const wrap = document.getElementById("events-table-wrap");
  const row = wrap.querySelector(`.ev-accordion[data-idx="${idx}"]`);
  if (!row) return;
  if (!row.hidden) {
    row.hidden = true;
    toggleEl.textContent = "▸";
    return;
  }
  const target = _ALL_EVENTS[idx];
  if (!target || target.entity_id == null) return;
  // Same-slot, same-entity_id sightings, oldest first so the story reads
  // left-to-right in the accordion strip.
  const related = _ALL_EVENTS
      .filter((e) => e.entity_id === target.entity_id && e.slot === target.slot)
      .sort((a, b) => (a.ts || "").localeCompare(b.ts || ""));
  const cell = row.querySelector("td");
  // The per-entity gallery holds a crop from EVERY sighting (not just the
  // ones that fired a returning-event), served by the local API from the
  // synced entities/ pool. Appended async under the event cards.
  const appendGallery = () => {
    if (!target.cam_id) return;
    fetch(`/api/entity-gallery?cam_id=${encodeURIComponent(target.cam_id)}` +
          `&entity_id=${encodeURIComponent(target.entity_id)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((g) => {
        if (!g || !(g.sightings || []).length || row.hidden) return;
        const thumbs = g.sightings.map((s, i) => `
          <a href="${s.url}" target="_blank" rel="noopener" class="ev-card">
            <img src="${s.url}" loading="lazy" alt="appearance ${i + 1}"/>
            <div class="ev-ts">${s.ts ? fmtTime(s.ts) : ""}</div>
          </a>`).join("");
        const div = document.createElement("div");
        div.className = "ev-strip";
        div.innerHTML = `<div class="ev-note">Every stored appearance of
            #${target.entity_id} (${g.sightings.length} crops, newest first)
            - the full gallery, not only event moments.</div>
          <div class="ev-cards">${thumbs}</div>`;
        cell.appendChild(div);
      })
      .catch(() => {});
  };
  if (related.length <= 1) {
    cell.innerHTML = `<div class="ev-empty">
      Only this sighting fired an event in the last 24h window -
      the appearance gallery below shows every stored look at it.
    </div>`;
    appendGallery();
  } else {
    const cards = related.map((e, k) => {
      const url = e.snapshot_url || e.fullframe_url;
      const badge = e === target ? "this event" : `#${k + 1}`;
      const sim = e.similarity != null
          ? `<div class="ev-sim">similarity ${Math.round(e.similarity * 100)}%</div>`
          : "";
      return `<div class="ev-card ${e === target ? "current" : ""}">
        <div class="ev-badge">${badge}</div>
        ${url ? `<a href="${url}" target="_blank" rel="noopener">
                  <img src="${url}" loading="lazy" alt="sighting ${k+1}"/>
                </a>` : `<div class="ev-nosnap">no snapshot saved</div>`}
        <div class="ev-ts">${fmtTime(e.ts)}</div>
        ${sim}
      </div>`;
    }).join("");
    cell.innerHTML = `<div class="ev-strip">
      <div class="ev-note">All ${related.length} sightings of
        <b>${target.cls ?? "?"} #${target.entity_id}</b>
        at ${escapeHtml((GRID_SLOTS.find(s=>s.slot_id===target.slot)||{}).display_area || target.slot)}
        in the last 24h - compare side by side.</div>
      <div class="ev-cards">${cards}</div>
    </div>`;
    appendGallery();
  }
  row.hidden = false;
  toggleEl.textContent = "▾";
}

// ---------- 7. Re-ID summary table ------------------------------------------

function renderReidTable(docs) {
  const wrap = document.getElementById("reid-table-wrap");
  const slotIds = new Set(GRID_SLOTS.map((s) => s.slot_id));
  // Re-id docs are keyed by CLOUD slot ids; route through cloudToTile so
  // the table renders in local-preview mode too (it silently vanished
  // there before - the cloud ids never matched the local_N tile ids).
  const rows = docs.filter((d) => {
    const tid = cloudToTile(d.id);
    return tid && slotIds.has(tid);
  });
  toggleSection("reid-section", rows.length > 0);
  if (!rows.length) return;
  const tr = (cells) => `<tr>${cells.map((c) => `<td>${c}</td>`).join("")}</tr>`;
  const slotLabel = (id) => {
    if (LOCAL_MODE) {
      const tid = cloudToTile(id);
      const st = tid && tileState[tid];
      return (st && st.cloudCamName) || id;
    }
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
      Estimates from the OSNet appearance embedder (rolling 48h registry) -
      robust to lighting and viewpoint changes, still an estimate rather than
      a biometric identity system. Counts reset once on 2026-07-10 when the
      embedder was upgraded from color histograms; entities age out after 48h
      of absence, newest-in oldest-out - there is no daily wipe.
    </div>`;
}

// ---------- helpers ---------------------------------------------------------

// Normalize a flagged doc's `anomaly` map into display strings. Docs written
// before the anomaly map existed only have the boolean — treated as a people
// spike with no expectation attached.
// Anomaly kinds the dashboard SHOWS. Statistical spike/drop verdicts were
// dropped by operator decision (2026-07): "busier than this hour usually
// is" is weather, not an event worth an alert. Legacy docs inside the 24h
// TTL window may still carry the old kinds - the filter hides them.
const ANOMALY_KINDS = new Set(["extreme_load", "camera_obstructed",
                               "camera_dark"]);
function isShownAnomaly(r) {
  return !!(r.is_anomaly && ANOMALY_KINDS.has(r.anomaly?.kind));
}

const _ANOMALY_KIND_LABELS = {
  extreme_load:     { arrow: "▲", dir: "spike", label: "extreme crowd/traffic" },
  camera_obstructed:{ arrow: "⛔", dir: "spike", label: "camera blocked - object at lens" },
  camera_dark:      { arrow: "⛔", dir: "drop",  label: "view went dark" },
};

function describeAnomaly(r) {
  const a = r.anomaly || {};
  const k = _ANOMALY_KIND_LABELS[a.kind]
        || { arrow: "▲", dir: "spike", label: a.kind || "anomaly" };
  const metric = a.metric === "vehicles" ? "vehicles"
               : a.metric === "person" ? "people" : (a.metric || "");
  return {
    arrow:       k.arrow,
    dir:         k.dir,
    kindLabel:   k.label,
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

// ---- Active-learning curve (plan WS5) --------------------------------------
// One point per training run from /api/al-curve (gate history). Local-mode
// only: on the hosted dashboard the fetch 404s and the panel stays hidden.
let alCurveChart = null;
async function renderAlCurve() {
  const wrap = document.getElementById("al-curve-wrap");
  const canvas = document.getElementById("al-curve");
  if (!wrap || !canvas || typeof Chart === "undefined") return;
  let data;
  try {
    const r = await fetch("/api/al-curve");
    if (!r.ok) return;
    data = await r.json();
  } catch { return; }
  const pts = data.points || [];
  if (!pts.length) return;
  wrap.hidden = false;
  const labels = pts.map((p, i) =>
      p.labels_total != null ? `${p.labels_total} lbl` : `run ${i + 1}`);
  const promoted = pts.map((p) => (p.promoted ? p.map50 : null));
  const rejected = pts.map((p) => (p.promoted ? null : p.map50));
  const baseline = pts.map(() => data.baseline_map50);
  if (alCurveChart) alCurveChart.destroy();
  alCurveChart = new Chart(canvas, {
    type: "line",
    data: { labels, datasets: [
      { label: "promoted", data: promoted, borderColor: "#4cc9f0",
        backgroundColor: "#4cc9f0", spanGaps: true, pointRadius: 5 },
      { label: "rejected", data: rejected, borderColor: "#6f7480",
        backgroundColor: "#6f7480", spanGaps: true, pointRadius: 5,
        pointStyle: "crossRot" },
      { label: "base model", data: baseline, borderColor: "#e7e9ee",
        borderDash: [6, 4], pointRadius: 0 },
    ]},
    options: {
      animation: false, responsive: true,
      scales: {
        y: { title: { display: true, text: "mAP50 (val)" },
             ticks: { color: "#9aa0ab" }, grid: { color: "#2a2d33" } },
        x: { ticks: { color: "#9aa0ab" }, grid: { color: "#2a2d33" } },
      },
      plugins: { legend: { labels: { color: "#9aa0ab" } } },
    },
  });
}
renderAlCurve();
setInterval(renderAlCurve, 300000);


// -----------------------------------------------------------------------------
// Counting-line editor + crossing alerts
// -----------------------------------------------------------------------------
// Two things live here that the previous Line layer never had:
//
//   1. Editor modal - drag on the last-known snapshot to pick two points;
//      Save posts to /api/lines?cam=<id>. From that moment the collector's
//      resolve_line() picks the override up on the next round (no restart
//      needed).
//   2. Alert loop - while a Line-layer analysis is active on any tile, poll
//      /api/crossings?cam=<id> and, on every new event, show a red toast
//      and flash the tile briefly. Small history strip below the tile
//      shows the last few event snapshots.
//
// Both are strictly ADDITIVE - if you never open the editor and never run a
// Line-layer analysis, nothing here changes the existing dashboard.

const lineEditor = document.createElement("div");
lineEditor.style.cssText =
  "display:none;position:fixed;inset:0;z-index:70;background:rgba(2,6,23,.82);" +
  "align-items:center;justify-content:center";
lineEditor.innerHTML = `
  <div style="background:#0f172a;border:1px solid #334155;border-radius:12px;
              padding:16px 18px;max-width:800px;width:94%;color:#e2e8f0">
    <h3 style="margin:0 0 4px;font-size:17px">Counting line -
      <span data-le-cam></span></h3>
    <div style="color:#94a3b8;font-size:13px;margin-bottom:10px">
      Drag on the snapshot to place a counting line. Save persists it
      per-camera; a running Line-layer session picks it up within a few
      seconds without a restart.</div>
    <div style="position:relative;background:#020617;border:1px solid #334155;
                border-radius:8px;overflow:hidden">
      <img data-le-img style="display:block;width:100%;height:auto;
                              user-select:none;-webkit-user-drag:none">
      <canvas data-le-canvas style="position:absolute;inset:0;width:100%;
                                     height:100%;cursor:crosshair"></canvas>
    </div>
    <div data-le-classes style="display:flex;flex-wrap:wrap;gap:10px 16px;
                                 margin-top:10px;font-size:13px;
                                 color:#cbd5e1"></div>
    <div style="color:#94a3b8;font-size:12px;margin-top:4px">
      Nothing checked = count every tracked class.</div>
    <div data-le-err style="color:#f87171;font-size:13px;min-height:18px;
                            margin-top:8px"></div>
    <div style="display:flex;gap:10px;margin-top:6px">
      <button data-le-save style="cursor:pointer;background:#2563eb;border:0;
              color:#fff;border-radius:8px;padding:7px 18px">Save line</button>
      <button data-le-clear style="cursor:pointer;background:#334155;border:0;
              color:#fff;border-radius:8px;padding:7px 14px">Clear override</button>
      <button data-le-cancel style="cursor:pointer;background:#1e293b;
              border:1px solid #334155;color:#e2e8f0;border-radius:8px;
              padding:7px 14px">Close</button>
    </div>
  </div>`;
document.body.appendChild(lineEditor);

const _leImg = lineEditor.querySelector("[data-le-img]");
const _leCanvas = lineEditor.querySelector("[data-le-canvas]");
const _leErr = lineEditor.querySelector("[data-le-err]");
const _leClasses = lineEditor.querySelector("[data-le-classes]");
let _leCam = null;
let _lePts = [];   // [[x_norm, y_norm], [x_norm, y_norm]]

function _leRenderClasses(allowed, picked) {
  // Checkbox row for the class filter. `allowed` comes from the server
  // (mirrors cameras.LINE_ALLOWED_CLASSES); `picked` is the current
  // override or null / [] for "count everything".
  _leClasses.innerHTML = "";
  const set = new Set(picked || []);
  for (const name of (allowed || [])) {
    const id = "le-cls-" + name;
    const lab = document.createElement("label");
    lab.style.cssText = "display:inline-flex;align-items:center;gap:5px;" +
                        "cursor:pointer";
    lab.innerHTML = `<input type="checkbox" data-le-cls value="${name}" ` +
      `id="${id}"${set.has(name) ? " checked" : ""}> ${name}`;
    _leClasses.appendChild(lab);
  }
}

function _leCollectClasses() {
  // Return the picked filter or null when the user checked nothing
  // (server treats null as "count every class").
  const boxes = _leClasses.querySelectorAll("[data-le-cls]:checked");
  if (!boxes.length) return null;
  return Array.from(boxes, (b) => b.value);
}

function _leDraw() {
  const c = _leCanvas;
  c.width = _leImg.clientWidth || _leImg.naturalWidth || 640;
  c.height = _leImg.clientHeight || _leImg.naturalHeight || 360;
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (_lePts.length === 0) return;
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 3;
  const p0 = [_lePts[0][0] * c.width, _lePts[0][1] * c.height];
  if (_lePts.length === 1) {
    ctx.beginPath(); ctx.arc(p0[0], p0[1], 6, 0, Math.PI * 2); ctx.fillStyle = "#f59e0b"; ctx.fill();
  } else {
    const p1 = [_lePts[1][0] * c.width, _lePts[1][1] * c.height];
    ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.stroke();
    for (const [x, y] of [p0, p1]) {
      ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fillStyle = "#f59e0b"; ctx.fill();
    }
  }
}

let _leDragging = false;
_leCanvas.addEventListener("mousedown", (e) => {
  const r = _leCanvas.getBoundingClientRect();
  _lePts = [[(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]];
  _leDragging = true;
  _leDraw();
});
_leCanvas.addEventListener("mousemove", (e) => {
  if (!_leDragging) return;
  const r = _leCanvas.getBoundingClientRect();
  const pt = [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
  _lePts = _lePts.length === 0 ? [pt] : [_lePts[0], pt];
  _leDraw();
});
_leCanvas.addEventListener("mouseup", () => { _leDragging = false; });
window.addEventListener("resize", _leDraw);

lineEditor.querySelector("[data-le-cancel]").addEventListener("click",
  () => { lineEditor.style.display = "none"; });

lineEditor.querySelector("[data-le-save]").addEventListener("click", async () => {
  _leErr.textContent = "";
  if (_lePts.length !== 2) { _leErr.textContent = "Draw a line first (drag on the image)"; return; }
  const classes = _leCollectClasses();
  try {
    const r = await fetch(`/api/lines?cam=${encodeURIComponent(_leCam)}`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({line: _lePts, classes}),
    });
    if (!r.ok) throw new Error(await r.text());
    _leErr.style.color = "#4ade80";
    _leErr.textContent = "Saved - a running session picks it up in a few seconds";
    setTimeout(() => { _leErr.style.color = "#f87171"; _leErr.textContent = ""; }, 2500);
  } catch (e) {
    _leErr.style.color = "#f87171";
    _leErr.textContent = "Save failed: " + e.message;
  }
});

lineEditor.querySelector("[data-le-clear]").addEventListener("click", async () => {
  _leErr.textContent = "";
  try {
    const r = await fetch(`/api/lines/clear?cam=${encodeURIComponent(_leCam)}`,
                          {method: "POST"});
    if (!r.ok) throw new Error(await r.text());
    _lePts = [];
    _leRenderClasses(_leLastAllowed, []);   // reset checkboxes too
    _leDraw();
    _leErr.style.color = "#4ade80";
    _leErr.textContent = "Override cleared - falling back to cameras.py";
    setTimeout(() => { _leErr.style.color = "#f87171"; _leErr.textContent = ""; }, 2000);
  } catch (e) {
    _leErr.style.color = "#f87171";
    _leErr.textContent = "Clear failed: " + e.message;
  }
});

// Cache the allowed-classes list from the last /api/lines call so
// Clear can re-render an empty checkbox row without a second fetch.
let _leLastAllowed = ["person", "bicycle", "car", "motorcycle", "bus", "truck"];

async function openLineEditor(cam, snapshotUrl) {
  _leCam = cam;
  _lePts = [];
  lineEditor.querySelector("[data-le-cam]").textContent = cam;
  _leImg.src = snapshotUrl || `/api/analysis/frame?cam=${encodeURIComponent(cam)}&_=${Date.now()}`;
  _leRenderClasses(_leLastAllowed, []);    // placeholder until fetch lands
  _leImg.onload = () => {
    // Load existing line + class filter + allowed vocabulary.
    fetch(`/api/lines?cam=${encodeURIComponent(cam)}`).then(r => r.json()).then(d => {
      if (d && d.line) _lePts = d.line;
      if (d && Array.isArray(d.allowed_classes) && d.allowed_classes.length)
        _leLastAllowed = d.allowed_classes;
      _leRenderClasses(_leLastAllowed, (d && d.classes) || []);
      _leDraw();
    }).catch(() => _leDraw());
  };
  lineEditor.style.display = "flex";
}

// Expose so the picker's "Edit line" button can call it.
window.openLineEditor = openLineEditor;

// ---- Crossing toast + tile flash ---------------------------------------

const _crossToast = document.createElement("div");
_crossToast.style.cssText =
  "position:fixed;top:16px;right:16px;z-index:80;display:none;" +
  "background:#dc2626;color:#fff;padding:10px 14px;border-radius:8px;" +
  "font-size:14px;font-weight:600;box-shadow:0 6px 20px rgba(0,0,0,.4);" +
  "max-width:320px";
document.body.appendChild(_crossToast);

let _crossToastTimer = null;
function showCrossToast(msg) {
  _crossToast.textContent = msg;
  _crossToast.style.display = "block";
  if (_crossToastTimer) clearTimeout(_crossToastTimer);
  _crossToastTimer = setTimeout(() => { _crossToast.style.display = "none"; }, 3500);
}

// Poll /api/crossings?cam=<id> for every tile whose analysis layer is
// "line". Three jobs per tick per tile: (1) fire a toast + red flash on
// every unseen event, (2) render the last N snapshot thumbs under the
// tile, (3) drop that strip the moment the layer changes off "line".
const _seenCrossings = new Map();  // cam_id -> Set of "ts|tid|dir" keys
const CROSSINGS_STRIP_MAX = 8;
const CROSSINGS_POLL_LIMIT = CROSSINGS_STRIP_MAX + 4;

function _ensureCrossingsStrip(st) {
  if (!st || !st.tile) return null;
  let strip = st.tile.querySelector(".crossings-strip");
  if (strip) return strip;
  strip = document.createElement("div");
  strip.className = "crossings-strip";
  strip.style.cssText =
    "display:flex;gap:6px;overflow-x:auto;padding:6px 8px;background:#0b1220;" +
    "border-top:1px solid #1f2937";
  strip.dataset.empty = "1";
  strip.textContent = "waiting for the first crossing...";
  strip.style.color = "#64748b";
  strip.style.fontSize = "12px";
  st.tile.appendChild(strip);
  return strip;
}

function _renderCrossingsStrip(strip, events) {
  // events are newest-first from the API; render the newest ones on the
  // LEFT so a new arrival visibly slides in from that side. Only rebuild
  // when the top event changes to avoid thrashing the DOM every tick.
  if (!strip) return;
  const top = events[0];
  const topKey = top ? (top.ts + "|" + top.tid + "|" + top.direction) : "";
  if (strip.dataset.topKey === topKey) return;
  strip.dataset.topKey = topKey;
  strip.style.color = "";
  strip.style.fontSize = "";
  strip.innerHTML = "";
  if (!top) {
    strip.dataset.empty = "1";
    strip.style.color = "#64748b";
    strip.style.fontSize = "12px";
    strip.textContent = "waiting for the first crossing...";
    return;
  }
  strip.dataset.empty = "0";
  for (const ev of events.slice(0, CROSSINGS_STRIP_MAX)) {
    const card = document.createElement("div");
    card.style.cssText =
      "flex:0 0 auto;width:88px;background:#111827;border:1px solid #1f2937;" +
      "border-radius:6px;overflow:hidden;text-align:center";
    const dir = (ev.direction === "in") ? "IN" : "OUT";
    const color = (ev.direction === "in") ? "#22c55e" : "#f97316";
    const hhmmss = (ev.ts || "").substr(11, 8);
    if (ev.snap) {
      const img = document.createElement("img");
      img.src = "/" + ev.snap;
      img.alt = dir;
      img.style.cssText = "display:block;width:100%;height:56px;object-fit:cover";
      card.appendChild(img);
    } else {
      const placeholder = document.createElement("div");
      placeholder.style.cssText = "height:56px;background:#020617;" +
        "display:flex;align-items:center;justify-content:center;" +
        "color:#475569;font-size:11px";
      placeholder.textContent = "no crop";
      card.appendChild(placeholder);
    }
    const meta = document.createElement("div");
    meta.style.cssText = "padding:3px 4px;font-size:11px;line-height:1.2;" +
                         "color:#e2e8f0";
    meta.innerHTML =
      `<div style="color:${color};font-weight:600">${dir} · ${escapeHtml(ev.cls || "obj")}</div>` +
      `<div style="color:#94a3b8">${escapeHtml(hhmmss)}</div>`;
    card.appendChild(meta);
    strip.appendChild(card);
  }
}

setInterval(async () => {
  if (typeof tileState !== "object" || !tileState) return;
  for (const st of Object.values(tileState)) {
    if (!st) continue;
    // Strip lives only while the tile is on the Line layer. Tear it down
    // (and the seen-set) on any other layer / no analysis.
    const onLine = st.analysis && st.analysis.layer === "line";
    if (!onLine) {
      const stale = st.tile && st.tile.querySelector(".crossings-strip");
      if (stale) stale.remove();
      continue;
    }
    const cam = st.analysis.cam;
    if (!cam) continue;
    let seen = _seenCrossings.get(cam);
    if (!seen) { seen = new Set(); _seenCrossings.set(cam, seen); }
    const strip = _ensureCrossingsStrip(st);
    try {
      const r = await fetch(`/api/crossings?cam=${encodeURIComponent(cam)}` +
                            `&limit=${CROSSINGS_POLL_LIMIT}`);
      if (!r.ok) continue;
      const data = await r.json();
      const eventsNewestFirst = data.events || [];
      _renderCrossingsStrip(strip, eventsNewestFirst);
      const events = eventsNewestFirst.slice().reverse();  // oldest first for toast dedup
      const boot = seen.size === 0;    // first poll: don't alarm on backlog
      for (const ev of events) {
        const key = ev.ts + "|" + ev.tid + "|" + ev.direction;
        if (seen.has(key)) continue;
        seen.add(key);
        if (boot) continue;
        showCrossToast(`${ev.direction === "in" ? "-> IN" : "OUT ->"}  ` +
                       `${ev.cls || "object"}  @ ${ev.ts.substr(11, 8)}`);
        // Flash the tile red briefly. tileState entries expose the DOM
        // node as `tile` (see the tile-render block near the top of this
        // file); fall back to videoWrap so the flash still lands if the
        // tile schema ever changes.
        const el = st.tile || st.videoWrap || null;
        if (el && el.style) {
          const prev = el.style.boxShadow;
          el.style.boxShadow = "0 0 0 4px #dc2626 inset";
          setTimeout(() => { el.style.boxShadow = prev; }, 800);
        }
      }
      if (seen.size > 200) {  // bound memory
        const arr = Array.from(seen);
        _seenCrossings.set(cam, new Set(arr.slice(arr.length - 100)));
      }
    } catch (_e) { /* transient - retry next tick */ }
  }
}, 4000);


// -----------------------------------------------------------------------------
// Snapshots (main-mode only): "📸 Snapshot grid" header button saves the four
// Analysis tiles as ONE 2x2 PNG under data/snapshots/<timestamp>.png. The
// Snapshots tab lists every saved PNG as a thumbnail card - click to open in
// a new tab / download, 🗑 deletes one, "Clear all" nukes the folder.
// Twin mode: nothing wires (button and tab are hidden by CSS/JS above).
// -----------------------------------------------------------------------------

(function initSnapshots() {
  if (!MAIN_MODE) return;

  const captureBtn = document.getElementById("snap-capture-btn");
  const refreshBtn = document.getElementById("snap-refresh");
  const clearBtn   = document.getElementById("snap-clear-all");
  const gridEl     = () => document.getElementById("snap-grid");
  const statusEl   = () => document.getElementById("snap-status");

  function _status(msg, kind = "ok") {
    const el = statusEl(); if (!el) return;
    el.textContent = msg;
    el.style.color = kind === "err" ? "#ef4444"
                   : kind === "ok"  ? "#4ade80"
                                    : "#94a3b8";
    if (msg) setTimeout(() => { el.textContent = ""; }, 3500);
  }

  async function captureGrid() {
    // tileState is a module-level object keyed by slot_id - each entry has
    // a `tile` DOM node (see the createTile block above). Drill down for
    // each tile's <video>, drawImage into a 2x2 canvas, POST as PNG.
    const st = (typeof tileState === "object" && tileState) ? tileState : {};
    const tiles = Object.values(st).slice(0, 4);
    if (!tiles.length) { _status("No tiles yet - wait a few seconds.", "err"); return; }
    const videos = tiles.map(t =>
      (t.tile || t.videoWrap || document).querySelector("video"));
    const usable = videos.map((v, i) => ({v, i}))
                         .filter(o => o.v && o.v.videoWidth > 0 && o.v.videoHeight > 0);
    if (!usable.length) {
      _status("No decoded video frames yet - reload if the tiles never fill in.", "err");
      return;
    }
    const cw = Math.max(...usable.map(o => o.v.videoWidth));
    const ch = Math.max(...usable.map(o => o.v.videoHeight));
    const canvas = document.createElement("canvas");
    canvas.width  = cw * 2;
    canvas.height = ch * 2;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#0f1115";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    // Cell labels overlay (top-left of each cell) - name from the tile head.
    ctx.font = "18px system-ui, -apple-system, Segoe UI, sans-serif";
    for (let i = 0; i < Math.min(videos.length, 4); i++) {
      const v = videos[i];
      const col = i % 2, row = Math.floor(i / 2);
      const x = col * cw, y = row * ch;
      if (v && v.videoWidth) {
        try { ctx.drawImage(v, x, y, cw, ch); }
        catch (e) { console.warn("snap drawImage failed", i, e); }
      }
      const name = (tiles[i]?.tile?.querySelector?.("h2")?.textContent
                 || tiles[i]?.slot?.placeholder_name
                 || `slot ${i + 1}`).slice(0, 60);
      // Semi-transparent label strip
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.fillRect(x, y, cw, 32);
      ctx.fillStyle = "#e7e9ee";
      ctx.fillText(name, x + 10, y + 22);
    }
    _status("capturing...", "info");
    const blob = await new Promise(res => canvas.toBlob(res, "image/png"));
    if (!blob) { _status("canvas toBlob failed - browser refused.", "err"); return; }
    const fd = new FormData();
    fd.append("png", blob, "grid.png");
    try {
      const r = await fetch("/api/snapshot", { method: "POST", body: fd });
      if (!r.ok) {
        const t = await r.text();
        _status(`save failed: HTTP ${r.status} - ${t.slice(0, 120)}`, "err");
        return;
      }
      const d = await r.json();
      _status(`saved ${d.name} (${(d.bytes / 1024).toFixed(0)} KB)`, "ok");
      const active = document.querySelector("main")?.dataset?.activeTab;
      if (active === "snapshots") loadSnaps();
    } catch (e) {
      _status("network error: " + e.message, "err");
    }
  }

  async function loadSnaps() {
    const g = gridEl(); if (!g) return;
    g.innerHTML = '<div class="snap-empty">loading...</div>';
    try {
      const r = await fetch("/api/snapshots-list");
      if (!r.ok) { g.innerHTML = `<div class="snap-empty">list failed: HTTP ${r.status}</div>`; return; }
      const d = await r.json();
      if (!d.items || !d.items.length) {
        g.innerHTML = '<div class="snap-empty">No snapshots yet. Hit "📸 Snapshot grid" in the header.</div>';
        return;
      }
      g.innerHTML = "";
      for (const it of d.items) {
        const card = document.createElement("div");
        card.className = "snap-card";
        const safe = escapeHtml(it.name);
        card.innerHTML = `
          <a href="${it.url}" target="_blank" rel="noopener" title="Open / download">
            <img src="${it.url}" loading="lazy" alt="${safe}"></a>
          <div class="snap-card-body">
            <span class="snap-card-ts">${safe}</span>
            <button class="snap-card-del" title="Delete this snapshot">🗑</button>
          </div>`;
        card.querySelector(".snap-card-del").addEventListener("click", async () => {
          if (!confirm(`Delete ${it.name}?`)) return;
          const dr = await fetch("/api/snapshot?path=" + encodeURIComponent(it.path),
                                 { method: "DELETE" });
          if (dr.ok) loadSnaps(); else alert("delete failed: HTTP " + dr.status);
        });
        g.appendChild(card);
      }
    } catch (e) {
      g.innerHTML = `<div class="snap-empty">network error: ${escapeHtml(e.message)}</div>`;
    }
  }

  if (captureBtn) captureBtn.addEventListener("click", () =>
    captureGrid().catch(e => _status("capture failed: " + e.message, "err")));
  if (refreshBtn) refreshBtn.addEventListener("click", loadSnaps);
  if (clearBtn) clearBtn.addEventListener("click", async () => {
    if (!confirm("Delete ALL saved snapshots on disk? This cannot be undone.")) return;
    const r = await fetch("/api/snapshot?path=*", { method: "DELETE" });
    if (r.ok) loadSnaps(); else alert("clear-all failed: HTTP " + r.status);
  });

  // Whenever the user clicks the Snapshots tab button, refresh the grid.
  document.getElementById("tabbar")?.addEventListener("click", (ev) => {
    const b = ev.target?.closest?.("[data-tab-btn]");
    if (b && b.dataset.tabBtn === "snapshots") loadSnaps();
  });
})();
