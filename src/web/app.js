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
        <button class="analyze-btn" data-analyze
                title="Live advanced analysis - pick one layer for this camera"
                style="cursor:pointer;border:1px solid #334155;background:#1e293b;color:#e2e8f0;border-radius:6px;padding:2px 8px;font-size:13px">🔬</button>
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
  // Per-tile click-to-play overlay for YouTube iframes. Chrome's autoplay
  // policy blackens muted-autoplay iframes on multi-embed pages after the
  // first one, and YouTube's own JS may also queue a pre-roll ad that
  // needs a user gesture. Overlay a big "play" button that on click calls
  // YT.Player.playVideo() through the iframe API (buildVideoInto already
  // mounted a Player with enablejsapi=1 so postMessage works). Once the
  // player fires onStateChange=1 (playing), the overlay fades out.
  // Click-to-play overlay only where an iframe would actually mount:
  // slots with a placeholder_hls (the /ytproxy relay) autoplay muted on
  // their own, and a "Play live" button floating over an already-playing
  // video just dims it and confuses.
  const _isYtEmbed = /youtube\.com\/embed/.test(slot.placeholder_embed || "")
                     && !slot.placeholder_hls;
  if (_isYtEmbed) {
    const _vw = tileState[slot.slot_id].videoWrap;
    _vw.style.position = _vw.style.position || "relative";
    const _play = document.createElement("button");
    _play.className = "play-overlay";
    _play.type = "button";
    _play.title = "Play (Chrome blocks multi-iframe autoplay - one tap starts it)";
    _play.innerHTML =
      '<span style="display:inline-flex;align-items:center;gap:10px;'
      + 'padding:14px 26px;border-radius:14px;background:rgba(37,99,235,0.92);'
      + 'color:#f8fafc;font-size:20px;font-weight:600;'
      + 'box-shadow:0 6px 20px rgba(0,0,0,0.4);pointer-events:none;">'
      + '<span style="font-size:26px;line-height:1;">▶</span>'
      + '<span>Play live</span></span>';
    _play.style.cssText =
      "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;"
      + "background:linear-gradient(180deg,rgba(15,23,42,0.35),rgba(15,23,42,0.7));"
      + "border:0;cursor:pointer;z-index:5;transition:opacity .35s ease;"
      + "backdrop-filter:blur(1px);";
    _play.addEventListener("click", () => {
      _play.style.opacity = "0";
      setTimeout(() => _play.remove(), 400);
      // Reach into the iframe player and call playVideo() through the
      // official IFrame API (the wrapper is stashed on tileState by
      // mountYouTubePlayer when it's ready). Fall back to a postMessage
      // if the wrapper is not there yet.
      const st = tileState[slot.slot_id];
      const p = st?.ytPlayer;
      if (p && typeof p.playVideo === "function") {
        try { p.unMute && p.unMute(); } catch (_) {}
        try { p.playVideo(); } catch (_) {}
      } else {
        const _if = _vw.querySelector("iframe");
        if (_if && _if.contentWindow) {
          _if.contentWindow.postMessage(
            JSON.stringify({ event: "command", func: "playVideo", args: [] }),
            "*");
        }
      }
    });
    _vw.appendChild(_play);
    tileState[slot.slot_id].playOverlay = _play;
  }
  // Per-tile 🔬 advanced-analysis button (restored on top of 27bced9 baseline).
  // The 27bced9 commit removed the button from the tile template but left this
  // event-binding line intact; the resulting null.addEventListener() threw and
  // aborted the tile-render loop after the first tile - hence the "only 1
  // camera appears" symptom. The template now carries the button back, and
  // the null-guard here prevents any future template drift from breaking the
  // whole grid the same way.
  const _analyzeBtn = tile.querySelector("[data-analyze]");
  if (_analyzeBtn) {
    _analyzeBtn.addEventListener("click", () =>
      openAnalysisPicker(tileState[slot.slot_id]));
  }
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
  ["gestures", "Static postures"],
  ["body",     "Body anomalies"],
  ["faces",    "Face detection"],
  ["line",     "Line crossing"],
  ["loiter",   "Zone & loitering"],
  ["parking",  "Parking occupancy"],
  ["plates",   "License plates (LPR)"],
];
// Layers whose geometry the operator draws on the frame themselves.
const DRAWABLE_LAYERS = { line: "✏ Draw line",
                          loiter: "✏ Draw zones",
                          parking: "✏ Draw spots" };
const ANALYSIS_POLL_MS = 500;

// ---- YouTube iframe lag (operator-tunable) ---------------------------------
// YouTube's live iframe plays several seconds BEHIND the actual live edge
// (typically 4-6s of player-side buffer). Our server yt-dlp grabs frames
// with a much smaller lag (~2s). The difference is what the operator sees
// as "the box is 0.5-1s ahead of where the object actually is on screen".
// We compensate on the client by picking an OLDER buffered tick when
// drawing over the iframe. The exact iframe lag varies per YouTube stream,
// per network, and even per session, so the operator tunes it live with
// the bracket keys: `[` shifts boxes EARLIER (increase lag), `]` shifts
// LATER (decrease lag). Setting persists in localStorage.
const YT_LAG_KEY = "ytIframeLagS";
const YT_LAG_DEFAULT_S = 4.5;
function _ytIframeLagS() {
  const v = parseFloat(localStorage.getItem(YT_LAG_KEY));
  if (!isFinite(v) || v < 0 || v > 25) return YT_LAG_DEFAULT_S;
  return v;
}
function _ytIframeLagBump(deltaS) {
  const now = _ytIframeLagS();
  const next = Math.max(0, Math.min(20, now + deltaS));
  localStorage.setItem(YT_LAG_KEY, String(next.toFixed(2)));
  // Force every active session to re-pin on the next tick.
  for (const st of Object.values(tileState || {})) {
    if (st && st.analysis) st.analysis._ytPin = null;
  }
  return next;
}
window.addEventListener("keydown", (e) => {
  if (e.target && (e.target.tagName === "INPUT"
                   || e.target.tagName === "TEXTAREA"
                   || e.target.isContentEditable)) return;
  let msg = null;
  if (e.key === "[") msg = `iframe lag ${_ytIframeLagBump(0.25).toFixed(2)}s`
                         + " - boxes SHIFT EARLIER (use if boxes are AHEAD)";
  else if (e.key === "]") msg = `iframe lag ${_ytIframeLagBump(-0.25).toFixed(2)}s`
                              + " - boxes SHIFT LATER (use if boxes are BEHIND)";
  if (msg && typeof showCrossToast === "function") showCrossToast(msg);
});

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
      // Belt-and-braces against a lost switch (seen once in testing: a
      // fast tile-to-tile flow left one session on its old layer): the
      // server echoes the layer it actually runs - retry once if it
      // doesn't match what the operator picked.
      if (data.layer && data.layer !== picked.value) {
        await new Promise((res) => setTimeout(res, 300));
        const r2 = await fetch(
          `/api/analysis/start?cam=${encodeURIComponent(cam)}` +
          `&layer=${encodeURIComponent(picked.value)}`,
          { method: "POST" });
        const d2 = await r2.json();
        if (!r2.ok || (d2.layer && d2.layer !== picked.value))
          throw new Error("layer switch did not take - try again");
      }
      beginTileAnalysis(st, cam, picked.value);
      analysisPanel.style.display = "none";
      // Deferred confirmation: a second later, ask the server which layer
      // is ACTUALLY running and silently re-issue the switch if it lost
      // the race (observed under rapid tile-to-tile switching).
      setTimeout(async () => {
        try {
          const chk = await fetch(
            `/api/analysis/data?cam=${encodeURIComponent(cam)}`)
            .then((x) => x.status === 200 ? x.json() : null);
          if (chk && chk.layer && chk.layer !== picked.value) {
            await fetch(
              `/api/analysis/start?cam=${encodeURIComponent(cam)}` +
              `&layer=${encodeURIComponent(picked.value)}`,
              { method: "POST" });
          }
        } catch (_) {}
      }, 1200);
    } catch (e) {
      errEl.textContent = "Failed to start: " + e.message;
    } finally {
      runBtn.disabled = false;
    }
  });

const _layerLabel = Object.fromEntries(ANALYSIS_LAYER_DEFS);

function beginTileAnalysis(st, cam, layer) {
  if (st.analysis) {
    // Same tile, new layer: the session already switched server-side
    // (stream + accumulators kept) - just relabel; the poller runs on.
    st.analysis.layer = layer;
    st.analysis.tickBuf.length = 0;   // old-layer ticks are stale now
    // Layer switch keeps the iframe playing; existing YT<->server pin
    // still valid. No re-sync needed.
    const tag = st.videoWrap.querySelector(".analysis-live-tag");
    if (tag) tag.textContent = `LIVE ANALYSIS · ${_layerLabel[layer] || layer}`;
    const lb = st.videoWrap.querySelector(".analysis-drawline");
    if (lb) {
      lb.style.display = DRAWABLE_LAYERS[layer] ? "" : "none";
      if (DRAWABLE_LAYERS[layer]) lb.textContent = DRAWABLE_LAYERS[layer];
    }
    return;
  }
  // Deliberate "delayed playback" mode (operator ask 2026-08-15,
  // "professional CV standard"): the iframe is held ~20 s BEHIND
  // YouTube's live edge for the duration of the analysis session. The
  // server keeps analyzing at real-time from the yt-dlp grab, so by the
  // time the iframe plays a given frame the server has processed it
  // long ago and the corresponding tick already sits in the buffer.
  // The client's DVR-based alignment (getDuration - getCurrentTime)
  // then picks the exact matching tick and lerps between its bracket
  // for smooth on-frame boxes. Cost: the video shown is 20 s stale -
  // the operator explicitly OK'd this trade to get accurate overlays.
  const ANALYSIS_HOLD_BEHIND_S = 20;
  if (st.ytPlayer && !st._analysisHoldSeek) {
    const tryHold = () => {
      try {
        const ct = st.ytPlayer.getCurrentTime();
        if (typeof ct === "number" && ct > ANALYSIS_HOLD_BEHIND_S + 2) {
          st.ytPlayer.seekTo(ct - ANALYSIS_HOLD_BEHIND_S, true);
          st._analysisHoldSeek = { at: Date.now() };
        } else {
          // Player not fully warm yet (ct too low); retry shortly.
          setTimeout(tryHold, 400);
        }
      } catch (_) { setTimeout(tryHold, 400); }
    };
    tryHold();
  }
  // Canvas-overlay mode: keep the iframe playing at native fps, draw
  // YOLO boxes / heat / line on a transparent canvas above it. The old
  // design tore down the video and replaced it with an analyzed-JPEG
  // slideshow, capping the tile at ~1 fps - the operator read that as
  // "the video froze the moment I clicked heat".
  st._overlayWasHidden = st.overlay.style.display === "none";
  st.overlay.style.display = "none";
  const wrap = document.createElement("div");
  wrap.className = "analysis-wrap analysis-overlay-mode";
  // background:transparent overrides the stylesheet's .analysis-wrap
  // {background:#000} - that rule belongs to the old replace-the-video
  // design and would paint solid black over the smooth /ytproxy video
  // this wrap now floats above.
  wrap.style.cssText = "position:absolute;inset:0;pointer-events:none;"
                     + "z-index:4;background:transparent;";
  // Remove any lingering "Play live" overlay from tile-creation - it
  // sits at z-index:5 above the analysis wrap and would visually block
  // the whole tile once analysis is on. The overlay was there so the
  // operator could kick YouTube autoplay; inside Advanced Analysis the
  // canvas + bg fallback are what the operator wants to see.
  const _leftoverPlay = st.videoWrap.querySelector(".play-overlay");
  if (_leftoverPlay) _leftoverPlay.remove();
  st.playOverlay = null;
  // Best-effort autoplay kick inside the operator's Start-click gesture
  // so YouTube plays too (if Chrome allows) - the ytPlayer state is not
  // used to hide the bg anymore, so this is purely a nice-to-have.
  try {
    if (st.ytPlayer && typeof st.ytPlayer.playVideo === "function") {
      try { st.ytPlayer.mute && st.ytPlayer.mute(); } catch (_) {}
      st.ytPlayer.playVideo();
    }
  } catch (_) {}
  wrap.innerHTML = `
    <img class="analysis-bg" alt="" draggable="false"
         style="position:absolute;inset:0;width:100%;height:100%;
                object-fit:cover;background:#0f172a;display:block;"
         data-hidden-when-playing="1">
    <canvas class="analysis-canvas"
            style="position:absolute;inset:0;width:100%;height:100%;
                   pointer-events:none;background:transparent;"></canvas>
    <div class="analysis-status"
         style="position:absolute;left:8px;top:8px;padding:4px 10px;
                background:rgba(15,23,42,0.85);color:#e2e8f0;border-radius:6px;
                font-size:12px;pointer-events:none;">starting live analysis...</div>
    <span class="analysis-live-tag"
          style="position:absolute;right:8px;top:8px;padding:4px 10px;
                 background:rgba(37,99,235,0.9);color:#f8fafc;border-radius:6px;
                 font-size:12px;font-weight:600;pointer-events:none;">LIVE ·
      ${escapeHtml(_layerLabel[layer] || layer)}</span>
    <button class="analysis-drawline"
            style="position:absolute;right:78px;bottom:8px;padding:6px 12px;
                   background:#2563eb;color:#f8fafc;border:0;border-radius:6px;
                   cursor:pointer;font-size:13px;pointer-events:auto;
                   display:${DRAWABLE_LAYERS[layer] ? "" : "none"};">
      ${DRAWABLE_LAYERS[layer] || "✏ Draw"}</button>
    <button class="analysis-stop"
            style="position:absolute;right:8px;bottom:8px;padding:6px 12px;
                   background:#dc2626;color:#f8fafc;border:0;border-radius:6px;
                   cursor:pointer;font-size:13px;pointer-events:auto;">
      ■ Stop</button>`;
  st.videoWrap.style.position = st.videoWrap.style.position || "relative";
  st.videoWrap.appendChild(wrap);
  wrap.querySelector(".analysis-stop").addEventListener("click",
    () => stopTileAnalysis(st));
  wrap.querySelector(".analysis-drawline").addEventListener("click", () => {
    const snap =
      `/api/analysis/frame?cam=${encodeURIComponent(cam)}&_=${Date.now()}`;
    const lay = st.analysis ? st.analysis.layer : layer;
    if (lay === "line") window.openLineEditor(cam, snap);
    else openZoneEditor(cam, lay, snap);
  });
  st.analysis = {
    cam, layer,
    wrap,
    bg: wrap.querySelector(".analysis-bg"),
    canvas: wrap.querySelector(".analysis-canvas"),
    status: wrap.querySelector(".analysis-status"),
    lastBgUrl: null,
    // Ring buffer of recent analysis ticks (each stamped with `at`, the
    // capture time on the stream's own clock). The rAF draw loop picks
    // the tick matching the video time on screen and extrapolates box
    // positions by track velocity - see _analysisDrawLoop.
    tickBuf: [],
    failures: 0, lastRestart: 0, inflight: false, lastSeq: -1,
    evSeen: new Set(),
    evTimer: setInterval(() => pollAnalysisEvents(st), 2500),
    timer: setInterval(() => pollAnalysisFrame(st), ANALYSIS_POLL_MS),
    // Twice a second, decide which layer the operator sees UNDER the
    // canvas boxes: the smooth /ytproxy <video> when it is genuinely
    // advancing, or the analyzed-frame JPEG otherwise. This check is
    // trustworthy ONLY because the video is now our own same-origin
    // <video> element (currentTime cannot advance without pixels being
    // decoded) - the old YT-iframe API happily reported PLAYING while
    // Chrome rendered a black surface, which is why every iframe-based
    // heuristic before this failed.
    videoStateTimer: setInterval(() => _syncAnalysisBgVisibility(st), 500),
  };
  // Analyzed frame visible from the first paint; the smooth video is
  // the upgrade once it proves it is actually advancing.
  st.analysis.bg.style.display = "block";
  st.analysis.canvas.style.display = "none";
  // Detection hot-strip: a rolling feed of this layer's discrete events
  // (a plate read, a crossing, a loiter alert, a posture...) UNDER the
  // video. New events push old ones off; only an explicit 💾 persists
  // the full frame server-side for later study.
  const evStrip = document.createElement("div");
  evStrip.className = "events-strip";
  evStrip.style.cssText =
    "display:flex;gap:6px;align-items:stretch;overflow-x:auto;" +
    "padding:6px 4px;background:#0b1220;border-radius:6px;margin-top:6px;" +
    "min-height:76px;scrollbar-width:thin;";
  evStrip.innerHTML = `<button class="events-saved-btn" style="flex:0 0 auto;
      background:#1e293b;color:#94a3b8;border:0;border-radius:6px;
      padding:0 10px;cursor:pointer;font-size:12px">Saved ▤</button>
    <div class="events-empty" style="color:#475569;font-size:12px;
      align-self:center;padding:0 8px">detections will appear here...</div>`;
  evStrip.querySelector(".events-saved-btn")
    .addEventListener("click", openSavedDetections);
  st.videoWrap.insertAdjacentElement("afterend", evStrip);
  st.analysis.evStrip = evStrip;
  pollAnalysisFrame(st);
  pollAnalysisEvents(st);
  // Kick a first bg fetch immediately so we don't wait a whole poll
  // interval before painting anything visible.
  _refreshAnalysisBg(st.analysis);
  _analysisDrawLoop(st, st.analysis);
}

// -- detection hot-strip -----------------------------------------------------

async function pollAnalysisEvents(st) {
  const a = st.analysis;
  if (!a || !a.evStrip) return;
  let d;
  try {
    const r = await fetch(
      `/api/analysis/events?cam=${encodeURIComponent(a.cam)}`);
    if (!r.ok) return;
    d = await r.json();
  } catch (_) { return; }
  if (!st.analysis || st.analysis !== a) return;
  const evs = d.events || [];
  if (evs.length) {
    const empty = a.evStrip.querySelector(".events-empty");
    if (empty) empty.remove();
  }
  // Server sends newest first; walk oldest-unseen-first so insertion
  // right after the Saved button keeps the strip newest-on-the-left.
  const anchor = a.evStrip.querySelector(".events-saved-btn");
  for (let i = evs.length - 1; i >= 0; i--) {
    const ev = evs[i];
    if (a.evSeen.has(ev.id)) continue;
    a.evSeen.add(ev.id);
    anchor.insertAdjacentElement("afterend", _eventChip(a, ev));
  }
  const chips = a.evStrip.querySelectorAll(".event-chip");
  for (let i = 30; i < chips.length; i++) chips[i].remove();
  // Show only the CURRENT layer's chips (server truth) - a heat view
  // scrolling old loiter alerts read as "heat is showing loitering".
  // Other layers' chips stay in the DOM and reappear on switch-back.
  const cur = a.actualLayer || a.layer;
  for (const c of a.evStrip.querySelectorAll(".event-chip")) {
    c.style.display = (!c.dataset.layer || c.dataset.layer === cur)
      ? "" : "none";
  }
}

function _eventChip(a, ev) {
  const chip = document.createElement("div");
  chip.className = "event-chip";
  chip.dataset.layer = ev.layer || "";
  chip.style.cssText =
    "flex:0 0 auto;display:flex;gap:6px;align-items:center;" +
    "background:#111a2e;border:1px solid #1e293b;border-radius:6px;" +
    "padding:4px 6px;max-width:250px;";
  const t = new Date(ev.ts * 1000);
  const pad = (n) => String(n).padStart(2, "0");
  chip.innerHTML = `
    <img src="data:image/jpeg;base64,${ev.thumb}" alt=""
         style="height:56px;border-radius:4px;flex:0 0 auto">
    <div style="min-width:0">
      <div style="font-size:11px;color:#e2e8f0;white-space:nowrap;
                  overflow:hidden;text-overflow:ellipsis">${escapeHtml(ev.text)}</div>
      <div style="font-size:10px;color:#64748b">${pad(t.getHours())}:${pad(t.getMinutes())}:${pad(t.getSeconds())}</div>
    </div>
    <button class="event-save" title="save this detection for later study"
            style="background:#1d4ed8;color:#fff;border:0;border-radius:5px;
                   padding:4px 7px;cursor:pointer;font-size:11px;flex:0 0 auto">
      ${ev.saved ? "✓" : "💾"}</button>`;
  const btn = chip.querySelector(".event-save");
  if (ev.saved) btn.disabled = true;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    try {
      const r = await fetch(
        `/api/analysis/event/save?cam=${encodeURIComponent(a.cam)}` +
        `&id=${encodeURIComponent(ev.id)}`, { method: "POST" });
      btn.textContent = r.ok ? "✓" : "✗";
      if (!r.ok) btn.disabled = false;
    } catch (_) { btn.textContent = "✗"; btn.disabled = false; }
  });
  return chip;
}

async function openSavedDetections() {
  let items = [];
  try {
    const r = await fetch("/api/analysis/saved");
    items = (await r.json()).items || [];
  } catch (_) {}
  const bg = document.createElement("div");
  bg.style.cssText = "position:fixed;inset:0;background:rgba(2,6,23,0.8);" +
    "z-index:60;display:flex;align-items:center;justify-content:center";
  const box = document.createElement("div");
  box.style.cssText = "background:#0f172a;border:1px solid #1e293b;" +
    "border-radius:10px;max-width:820px;width:92%;max-height:80vh;" +
    "overflow:auto;padding:16px";
  box.innerHTML = `<div style="display:flex;justify-content:space-between;
      align-items:center;margin-bottom:10px">
      <b style="color:#e2e8f0">Saved detections (${items.length})</b>
      <button class="saved-close" style="background:#1e293b;color:#94a3b8;
        border:0;border-radius:6px;padding:6px 12px;cursor:pointer">close</button>
    </div>` + (items.length ? "" :
    `<div style="color:#64748b;font-size:13px">nothing saved yet - use the
     💾 button on a detection chip</div>`);
  for (const it of items) {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;gap:10px;align-items:center;" +
      "border-top:1px solid #1e293b;padding:8px 0";
    const t = new Date(it.ts * 1000);
    row.innerHTML = `
      <a href="${it.image}" target="_blank" rel="noopener">
        <img src="${it.image}" alt="" style="height:64px;border-radius:6px"></a>
      <div style="min-width:0">
        <div style="color:#e2e8f0;font-size:13px">${escapeHtml(it.text)}</div>
        <div style="color:#64748b;font-size:11px">${escapeHtml(it.cam_name || it.cam)}
          · ${escapeHtml(it.layer)} · ${t.toLocaleString()}</div>
      </div>`;
    box.appendChild(row);
  }
  bg.appendChild(box);
  bg.addEventListener("click", (e) => { if (e.target === bg) bg.remove(); });
  box.querySelector(".saved-close").addEventListener("click",
    () => bg.remove());
  document.body.appendChild(bg);
}

// Investigation tab: the standing gallery of saved detection samples -
// reachable with or without a running analysis (the strip's modal only
// exists while a wrap is mounted; this one always does).
async function renderGallery() {
  const wrap = document.getElementById("gallery-wrap");
  const title = document.getElementById("gallery-title");
  if (!wrap) return;
  let items = [];
  try {
    const r = await fetch("/api/analysis/saved", { cache: "no-store" });
    items = (await r.json()).items || [];
  } catch (_) { return; }
  if (title) title.textContent =
    `Detections gallery - ${items.length} saved sample(s)`;
  if (!items.length) {
    wrap.innerHTML = `<div class="sub">nothing saved yet - press 💾 on a
      live detection chip, or let the proof collector fill this up.</div>`;
    return;
  }
  const order = ["plates", "line", "loiter", "parking", "gestures",
                 "body", "pose", "faces", "heat", "paths"];
  items.sort((a, b) => order.indexOf(a.layer) - order.indexOf(b.layer)
                       || (b.ts || 0) - (a.ts || 0));
  wrap.innerHTML = items.map((it) => {
    const t = new Date((it.ts || 0) * 1000);
    return `<figure style="margin:0;background:#0c0e13;border:1px solid
        #232733;border-radius:8px;overflow:hidden">
      <a href="${it.image}" target="_blank" rel="noopener">
        <img src="${it.image}" alt="" loading="lazy"
             style="width:100%;height:130px;object-fit:cover;display:block"></a>
      <figcaption style="padding:6px 8px">
        <div style="font-size:11px;color:#e7e9ee;white-space:nowrap;
             overflow:hidden;text-overflow:ellipsis">${escapeHtml(it.text)}</div>
        <div style="font-size:10px;color:#8b909a">${escapeHtml(it.layer)} ·
          ${escapeHtml(it.cam_name || it.cam)} · ${t.toLocaleTimeString()}</div>
      </figcaption>
    </figure>`;
  }).join("");
}
renderGallery();
setInterval(renderGallery, 20000);

// 60fps draw loop for one tile's analysis overlay. Runs only while that
// tile's analysis is active (self-terminates when st.analysis changes).
// Each frame: pick the buffered tick whose capture time matches the video
// time currently on screen (hls.js exposes it as playingDate, driven by
// the stream's EXT-X-PROGRAM-DATE-TIME tags), then shift every box by its
// track velocity times the residual dt. Boxes glide with the traffic
// instead of jumping once a second onto pixels the object already left.
function _analysisDrawLoop(st, a) {
  if (!st.analysis || st.analysis !== a) return;   // stopped or restarted
  // ~15fps, not 60: four tiles at 60fps of canvas work measurably janked
  // the main thread that hls.js needs for MSE segment appends - the
  // videos decayed to ~0.7x realtime and never caught up (the operator's
  // "stuck video"). 15fps keeps box motion visually smooth at a quarter
  // of the main-thread cost. requestAnimationFrame still paces us to
  // the display, we just skip frames until 66ms have passed.
  requestAnimationFrame(() => _analysisDrawLoop(st, a));
  const nowMs = performance.now();
  if (a._lastDrawMs && nowMs - a._lastDrawMs < 66) return;
  a._lastDrawMs = nowMs;
  if (a.canvas.style.display === "none") return;   // JPEG mode covers it
  const buf = a.tickBuf;
  if (!buf.length) return;
  let merged = buf[buf.length - 1];
  // Video clock in wall-clock seconds. Two paths:
  // (1) hls.js: playingDate is PDT and IS wall clock, use directly.
  // (2) YouTube iframe: getCurrentTime() is DVR seconds - pin it to the
  //     first server tick's `at` once, then `at + (nowCT - pinCT)` gives
  //     us a wall-clock proxy. The player runs at 1s per real second, so
  //     the pin holds for hours; a big drift (state change / seek) is
  //     detected by monitoring the CT-vs-wall skew and re-pinning.
  const hls = st.currentHlsInstance;
  const pd = hls && hls.playingDate;
  let vidT = null;
  if (pd instanceof Date && !isNaN(pd)) {
    vidT = pd.getTime() / 1000;
  } else if (st.ytPlayer && typeof st.ytPlayer.getCurrentTime === "function") {
    try {
      const ct = st.ytPlayer.getCurrentTime();
      const dur = typeof st.ytPlayer.getDuration === "function"
        ? st.ytPlayer.getDuration() : 0;
      if (typeof ct === "number" && isFinite(ct) && ct > 0) {
        const nowWall = Date.now() / 1000;
        // Preferred (auto): YouTube live with DVR reports its own live-
        // edge lag as getDuration()-getCurrentTime(). That IS the exact
        // number of seconds the iframe is playing BEHIND real-time. The
        // server publishes tick.at as wall-clock too, so alignment is
        // direct: vidT = nowWall - iframeLagS. The draw loop then picks
        // the OLDER tick whose `at` matches the on-screen moment and
        // lerps between it and its successor for smooth motion. No
        // tuning needed on any DVR-enabled YouTube stream (the default).
        const dvrLag = dur - ct;
        if (dur > 1 && dvrLag > 0.2 && dvrLag < 60) {
          vidT = nowWall - dvrLag;
        } else {
          // No DVR: use the operator-tunable lag knob as a coarse
          // fallback. `[` / `]` shift it live.
          const newest = buf[buf.length - 1];
          const serverAge = Math.max(0, nowWall - (newest?.at || nowWall));
          const iframeLag = _ytIframeLagS();
          const boxDelayS = Math.max(0, iframeLag - serverAge);
          vidT = (newest?.at || nowWall) - boxDelayS
                 + (ct - (a._ytPin?.ct || ct));
          if (!a._ytPin) a._ytPin = { ct, at: vidT };
        }
      }
    } catch (_) { /* getCurrentTime rejects before ready */ }
  }
  if (vidT != null) {
    // Newest tick at or before the video clock...
    let i = buf.length - 1;
    while (i > 0 && (buf[i].at || 0) > vidT + 0.25) i--;
    const d = buf[i];
    const nxt = i + 1 < buf.length ? buf[i + 1] : null;
    const t0 = d.at || vidT;
    // Adaptive glide window: ~1.3 measured tick intervals (the old fixed
    // TAU=5s + 10s cap froze boxes mid-gap at 12-15s tick cadence - the
    // operator's "box stops, then snaps").
    // Extrapolation is tightly bounded on purpose (audit 2026-08-15:
    // boxes flew off screen and stayed there when the previous 20s
    // limit let stale velocities keep running). Real limits from what
    // a street object physically does between analysis ticks:
    //   * time: 1.5s past the newest real position, then STOP drawing
    //   * displacement: never further than half the object's own
    //     diagonal - a velocity that puts the box beyond that is
    //     stale, and shifting it there paints on empty pixels.
    // A missed track (coast=true) never extrapolates: we don't know it
    // moved at all.
    const EXTRAP_MAX_S = 1.5;
    const EXTRAP_MAX_DIAG = 0.5;
    const _capShift = (b, dt) => {
      if (b.coast || !dt || dt <= 0) return _shiftBox(b, 0);
      const diag = Math.hypot(b.x2 - b.x1, b.y2 - b.y1) || 1;
      const disp = Math.hypot((b.vx || 0) * dt, (b.vy || 0) * dt);
      const cap = EXTRAP_MAX_DIAG * diag;
      const useDt = disp > cap && disp > 0 ? dt * (cap / disp) : dt;
      return _shiftBox(b, Math.min(useDt, EXTRAP_MAX_S));
    };
    let fade = 1;
    const boxes = [];
    if (nxt && (nxt.at || 0) > t0 + 0.05) {
      // TRUE interpolation: the player runs behind the live edge, so
      // the tick AFTER the on-screen moment usually already exists.
      // Same-track boxes lerp between their two real positions - box
      // rides the object's actual path, no prediction, no freeze.
      const al = Math.max(0, Math.min(1, (vidT - t0) / ((nxt.at || 0) - t0)));
      const byTid = new Map();
      for (const nb of nxt.boxes || []) {
        if (nb.tid !== undefined) byTid.set(nb.tid, nb);
      }
      for (const b of d.boxes || []) {
        const nb = b.tid !== undefined ? byTid.get(b.tid) : null;
        boxes.push(nb ? _lerpBox(b, nb, al) : _capShift(b, vidT - t0));
      }
    } else {
      const rawDt = Math.max(0, vidT - t0);
      // Hard cutoff: if the newest tick is older than EXTRAP_MAX_S,
      // publish NO boxes for this frame. The operator sees the analysis
      // temporarily go dark (correct: server hasn't confirmed anything
      // this fresh) instead of a rigid box drifting off after a stopped
      // object. Next tick brings truth back within one poll interval.
      if (rawDt <= EXTRAP_MAX_S) {
        for (const b of d.boxes || []) boxes.push(_capShift(b, rawDt));
        if (rawDt > EXTRAP_MAX_S * 0.7) {
          fade = 1 - 0.5 * (rawDt - EXTRAP_MAX_S * 0.7)
                     / (EXTRAP_MAX_S * 0.3);
        }
      }
    }
    merged = Object.assign({}, d, { boxes, _fade: fade });
  }
  _drawAnalysisOverlay(a.canvas, merged, 0);
}

// Interpolated box: geometry lerped between the SAME track's two real
// tick positions; label/conf/flags from the newer tick. Keypoints lerp
// joint-by-joint when both ticks carry them, else ride the box shift.
function _lerpBox(b, nb, al) {
  const o = Object.assign({}, nb);
  o.x1 = b.x1 + (nb.x1 - b.x1) * al;
  o.y1 = b.y1 + (nb.y1 - b.y1) * al;
  o.x2 = b.x2 + (nb.x2 - b.x2) * al;
  o.y2 = b.y2 + (nb.y2 - b.y2) * al;
  const dx = (o.x1 + o.x2 - b.x1 - b.x2) / 2;
  const dy = (o.y1 + o.y2 - b.y1 - b.y2) / 2;
  if (b.kps && nb.kps && nb.kps.length === b.kps.length) {
    o.kps = b.kps.map((k, j) => [k[0] + (nb.kps[j][0] - k[0]) * al,
                                 k[1] + (nb.kps[j][1] - k[1]) * al,
                                 Math.min(k[2], nb.kps[j][2])]);
  } else if (b.kps) {
    o.kps = b.kps.map((k) => [k[0] + dx, k[1] + dy, k[2]]);
  }
  if (b.trail) o.trail = b.trail;
  o.vx = 0; o.vy = 0;
  return o;
}

// Extrapolated box: rigid shift along the tracker velocity.
function _shiftBox(b, dt) {
  const dx = (b.vx || 0) * dt, dy = (b.vy || 0) * dt;
  const o = Object.assign({}, b);
  o.x1 = b.x1 + dx; o.y1 = b.y1 + dy;
  o.x2 = b.x2 + dx; o.y2 = b.y2 + dy;
  if (b.kps) o.kps = b.kps.map((k) => [k[0] + dx, k[1] + dy, k[2]]);
  o.vx = 0; o.vy = 0;
  return o;
}

function _syncAnalysisBgVisibility(st) {
  const a = st.analysis;
  if (!a || !a.bg) return;
  let playing = false;
  const v = st.videoWrap.querySelector("video");
  if (v && !v.paused && !v.ended && v.readyState >= 2) {
    const t = v.currentTime;
    playing = (a._lastVidT !== undefined) && (t > a._lastVidT + 0.05);
    a._lastVidT = t;
  } else if (st.ytPlayer) {
    // YouTube iframe: no <video> we can inspect (cross-origin), so trust
    // the IFrame API. PLAYING(1) OR BUFFERING(3) mean pixels are moving;
    // getCurrentTime advancing between calls confirms it under load.
    try {
      const state = st.ytPlayer.getPlayerState();
      const ct = st.ytPlayer.getCurrentTime();
      if (state === 1 || state === 3) {
        playing = (a._lastYtCt !== undefined) && (ct > a._lastYtCt + 0.05);
        // If we have no delta yet (first tick), trust the state.
        if (a._lastYtCt === undefined) playing = true;
        a._lastYtCt = ct;
      } else {
        a._lastYtCt = undefined;
      }
    } catch (_) {}
  } else {
    a._lastVidT = undefined;
  }
  const want = playing ? "none" : "block";
  if (a.bg.style.display !== want) {
    a.bg.style.display = want;
    // The canvas draws over live video ONLY - the JPEG fallback already
    // carries the server-rendered overlay burned in, and stacking the
    // canvas on top of it double-draws every box.
    a.canvas.style.display = playing ? "" : "none";
    // Falling back to the frame view: refresh right away so the
    // operator doesn't stare at a stale image for a poll interval.
    if (want === "block") _refreshAnalysisBg(a);
  }
}

async function pollAnalysisFrame(st) {
  const a = st.analysis;
  if (!a || a.inflight) return;
  a.inflight = true;
  try {
    const r = await fetch(
      `/api/analysis/data?cam=${encodeURIComponent(a.cam)}&_=${Date.now()}`,
      { cache: "no-store" });
    if (r.status === 200) {
      const d = await r.json();
      // Layer-truth sync: the blue tag reflects what the SERVER runs,
      // not what was clicked. While they differ, say "switching..." and
      // re-POST the switch (throttled) until the echo confirms - a
      // silently failed switch used to leave the tag lying (a "Pose"
      // tag over loiter zones).
      a.actualLayer = d.layer;
      const liveTag = st.videoWrap.querySelector(".analysis-live-tag");
      if (d.layer === a.layer) {
        if (liveTag) liveTag.textContent =
          `LIVE · ${_layerLabel[d.layer] || d.layer}`;
      } else {
        if (liveTag) liveTag.textContent =
          `switching to ${_layerLabel[a.layer] || a.layer}…`;
        if (Date.now() - (a._switchPost || 0) > 4000) {
          a._switchPost = Date.now();
          fetch(`/api/analysis/start?cam=${encodeURIComponent(a.cam)}`
                + `&layer=${encodeURIComponent(a.layer)}`,
                { method: "POST" }).catch(() => {});
        }
      }
      if (d.seq !== a.lastSeq) {
        a.lastSeq = d.seq;
        // Live tick-interval estimate (EMA) - the draw loop scales its
        // glide/fade window to the cadence the session actually delivers.
        const prevTick = a.tickBuf[a.tickBuf.length - 1];
        if (prevTick) {
          const gap = (d.at || 0) - (prevTick.at || 0);
          if (gap > 0.2 && gap < 60) {
            a._gapEma = a._gapEma ? 0.7 * a._gapEma + 0.3 * gap : gap;
          }
        }
        a.tickBuf.push(d);
        // 60 ticks x ~12s = ~12 minutes of history, deep enough that a
        // 20 s hold-back seek always finds bracket ticks around vidT.
        if (a.tickBuf.length > 60) a.tickBuf.shift();
        // Old seek-based sync removed 2026-08-15: YouTube live streams
        // without DVR clamp seekTo back to the live edge, so seeking never
        // held. Alignment is now done in _analysisDrawLoop by shifting
        // vidT backward by the operator-tunable iframe-lag estimate.
        // A first tick just resets the pin so the new offset takes effect.
        if (st.ytPlayer) a._ytPin = null;
        // Skip the JPEG round-trip while the smooth video is confirmed
        // playing (the bg is hidden then and the bytes would be wasted).
        if (a.bg && a.bg.style.display !== "none") _refreshAnalysisBg(a);
        // Loitering alerts: toast + tile flash once per zone-episode
        // (re-arms when the zone drops back below its threshold).
        if (d.layer === "loiter" && Array.isArray(d.zones)) {
          a._loiterAlerted = a._loiterAlerted || new Set();
          for (const z of d.zones) {
            if (z.alert && !a._loiterAlerted.has(z.name)) {
              a._loiterAlerted.add(z.name);
              showCrossToast(`⚠ loitering in ${z.name} - `
                             + `${z.max_dwell}s (${st.slot.placeholder_name})`);
              const t = st.tile;
              if (t) {
                t.style.outline = "3px solid #ef4444";
                setTimeout(() => { t.style.outline = ""; }, 2500);
              }
            } else if (!z.alert) {
              a._loiterAlerted.delete(z.name);
            }
          }
        }
      }
      a.status.style.display = "none";
      a.failures = 0;
    } else if (r.status === 202) {
      const j = await r.json();
      a.status.style.display = "";
      a.status.textContent = j.note || "starting...";
    } else if (r.status === 404) {
      // Session ended server-side (idle stop / server restart): restart
      // it, at most once per 5s so a dead backend isn't hammered.
      a.failures += 1;
      if (Date.now() - a.lastRestart > 5000) {
        a.lastRestart = Date.now();
        a.status.style.display = "";
        a.status.textContent = "analysis session ended - restarting...";
        fetch(`/api/analysis/start?cam=${encodeURIComponent(a.cam)}`
              + `&layer=${encodeURIComponent(a.layer)}`,
              { method: "POST" }).catch(() => {});
      }
    } else if (r.status === 410) {
      // Session CRASHED server-side: surface the recorded reason instead
      // of burying it under a generic "unreachable" after blind retries.
      let reason = "";
      try { reason = (await r.json()).error || ""; } catch (_) {}
      a.status.style.display = "";
      a.status.textContent = "analysis ended"
        + (reason ? ` - ${reason}` : "") + " - pick a layer to restart";
      a.failures += 1;
    } else {
      a.failures += 1;
    }
  } catch (_) {
    a.failures += 1;
  } finally {
    a.inflight = false;
  }
  if (a.failures > 8) {
    a.status.style.display = "";
    a.status.textContent =
      "analysis unreachable - press Stop to return to video";
  }
}

// Draw one analysis tick (boxes + optional heat/line) onto the tile's
// overlay canvas. `dtExtra` is how many seconds of video time have passed
// since this tick's frame was captured - every box shifts by its track
// velocity times that, so boxes ride along with the traffic between YOLO
// updates instead of sitting on vacated pixels.
function _drawAnalysisOverlay(canvas, d, dtExtra = 0) {
  // getBoundingClientRect forces layout - at 15fps x 4 tiles that is
  // still 60 forced layouts/s. Cache the measurement for 1.5s; tile
  // sizes only change on window resroll/zoom, and a stale size for a
  // second costs one slightly-misscaled frame, not a jank storm.
  const nowMs = performance.now();
  let m = canvas._sizeCache;
  if (!m || nowMs - m.t > 1500) {
    const rect = canvas.parentElement.getBoundingClientRect();
    m = canvas._sizeCache = {
      t: nowMs,
      cw: Math.max(1, Math.round(rect.width)),
      ch: Math.max(1, Math.round(rect.height)),
    };
  }
  const cw = m.cw, ch = m.ch;
  if (canvas.width !== cw)  canvas.width = cw;
  if (canvas.height !== ch) canvas.height = ch;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, cw, ch);
  const fw = Math.max(1, Number(d.frame_w) || cw);
  const fh = Math.max(1, Number(d.frame_h) || ch);
  const sx = cw / fw, sy = ch / fh;

  // Heat layer paints a semi-transparent grid over the whole tile.
  // The grid only changes once per analysis tick, but this draw runs at
  // 60fps for the box extrapolation - so the ~400 fillRects render into
  // an offscreen canvas once per (tick, size) and every rAF just blits.
  if (d.layer === "heat" && Array.isArray(d.heat) && d.heat.length) {
    let hc = canvas._heatCache;
    if (!hc || hc.seq !== d.seq || hc.cw !== cw || hc.ch !== ch) {
      const off = document.createElement("canvas");
      off.width = cw; off.height = ch;
      const octx = off.getContext("2d");
      const gh = d.heat.length, gw = d.heat[0].length;
      const cellW = cw / gw, cellH = ch / gh;
      // p99 normalization instead of max: one nuclear cell (a bus stop)
      // used to flatten every other cell to invisible - percentile
      // clipping is the standard colormap fix.
      const vals = [];
      for (const row of d.heat) for (const v of row) if (v > 0) vals.push(v);
      vals.sort((a, b) => a - b);
      const peak = vals.length
        ? vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.99))]
        : 0;
      if (peak > 0) {
        for (let gy = 0; gy < gh; gy++) {
          for (let gx = 0; gx < gw; gx++) {
            const v = Math.min(1, d.heat[gy][gx] / peak);
            if (v < 0.05) continue;
            const alpha = Math.min(0.65, v * 0.7);
            octx.fillStyle = _heatColor(v, alpha);
            octx.fillRect(gx * cellW, gy * cellH, cellW + 1, cellH + 1);
          }
        }
      }
      hc = canvas._heatCache = { seq: d.seq, cw, ch, off };
    }
    ctx.drawImage(hc.off, 0, 0);
  }

  // Line layer: the crossing line + running counts. Line points are
  // NORMALIZED (0..1 of the frame) when they come from the line editor
  // or the default - detect and scale accordingly (a <=1 coordinate on a
  // 1920px frame can only be normalized).
  if (d.layer === "line" && Array.isArray(d.line) && d.line.length === 2) {
    const norm = d.line.every((p) => p[0] <= 1.001 && p[1] <= 1.001);
    const lx = (p) => norm ? p[0] * cw : p[0] * sx;
    const ly = (p) => norm ? p[1] * ch : p[1] * sy;
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(59,130,246,0.95)";
    ctx.setLineDash([10, 6]);
    ctx.beginPath();
    ctx.moveTo(lx(d.line[0]), ly(d.line[0]));
    ctx.lineTo(lx(d.line[1]), ly(d.line[1]));
    ctx.stroke();
    ctx.setLineDash([]);
    if (d.cross) {
      ctx.fillStyle = "rgba(15,23,42,0.85)";
      ctx.fillRect(8, ch - 30, 150, 22);
      ctx.fillStyle = "#f8fafc";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText(`in: ${d.cross.in || 0}   out: ${d.cross.out || 0}`,
                   14, ch - 14);
    }
  }

  // Loiter zones / parking spots - normalized polygons.
  if ((d.layer === "loiter" && Array.isArray(d.zones))
      || (d.layer === "parking" && Array.isArray(d.spots))) {
    const entries = d.layer === "loiter" ? d.zones : d.spots;
    ctx.font = "12px system-ui, sans-serif";
    for (const z of entries) {
      const hot = d.layer === "loiter" ? z.alert : z.occupied;
      const col = hot ? "239,68,68" : "74,222,128";
      ctx.beginPath();
      ctx.moveTo(z.points[0][0] * cw, z.points[0][1] * ch);
      for (let i = 1; i < z.points.length; i++)
        ctx.lineTo(z.points[i][0] * cw, z.points[i][1] * ch);
      ctx.closePath();
      ctx.fillStyle = `rgba(${col},${hot ? 0.22 : 0.12})`;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = `rgba(${col},0.95)`;
      ctx.stroke();
      const label = d.layer === "loiter"
        ? `${z.name}: ${z.count} inside · max ${z.max_dwell}s`
        : `${z.name}: ${z.occupied ? (z.cls || "occupied") : "free"}`;
      const zx = z.points[0][0] * cw, zy = z.points[0][1] * ch;
      const tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = "rgba(15,23,42,0.85)";
      ctx.fillRect(zx, Math.max(0, zy - 16), tw, 16);
      ctx.fillStyle = "#f8fafc";
      ctx.fillText(label, zx + 4, Math.max(12, zy - 4));
    }
    if (d.layer === "parking" && d.parking) {
      const t = `parking: ${d.parking.occupied}/${d.parking.total} occupied`;
      ctx.fillStyle = "rgba(15,23,42,0.85)";
      ctx.fillRect(8, ch - 30, ctx.measureText(t).width + 14, 22);
      ctx.fillStyle = "#f8fafc";
      ctx.fillText(t, 14, ch - 14);
    }
    if (!entries.length) {
      const t = "no zones drawn yet - press the Draw button";
      ctx.fillStyle = "rgba(15,23,42,0.85)";
      ctx.fillRect(8, 8, ctx.measureText(t).width + 14, 22);
      ctx.fillStyle = "#fbbf24";
      ctx.fillText(t, 14, 23);
    }
  }

  // Trails first (paths layer) so boxes draw over them.
  if (d.layer === "paths") {
    for (const b of d.boxes || []) {
      if (!Array.isArray(b.trail) || b.trail.length < 2) continue;
      ctx.lineWidth = 2;
      ctx.strokeStyle = _trailColor(b.tid || 0);
      ctx.beginPath();
      ctx.moveTo(b.trail[0][0] * sx, b.trail[0][1] * sy);
      for (let i = 1; i < b.trail.length; i++)
        ctx.lineTo(b.trail[i][0] * sx, b.trail[i][1] * sy);
      // extend the trail tip to the extrapolated current position
      ctx.lineTo((b.x1 + b.x2) / 2 * sx + (b.vx || 0) * dtExtra * sx,
                 (b.y1 + b.y2) / 2 * sy + (b.vy || 0) * dtExtra * sy);
      ctx.stroke();
    }
  }

  // Face rectangles (faces layer) - not tracked, so no extrapolation.
  if (d.layer === "faces") {
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(250,204,21,0.95)";
    for (const f of d.faces || [])
      ctx.strokeRect(f.x1 * sx, f.y1 * sy,
                     (f.x2 - f.x1) * sx, (f.y2 - f.y1) * sy);
    if (d.faces_ok === false) {
      ctx.fillStyle = "rgba(15,23,42,0.85)";
      ctx.fillRect(8, 8, 220, 22);
      ctx.fillStyle = "#fbbf24";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText("face backend unavailable", 14, 23);
    }
  }

  // Boxes - tracker-confirmed only (server filters), extrapolated
  // forward by track velocity. Pose-ish layers only box people;
  // body layer only boxes FLAGGED people (matching the server render).
  ctx.font = "12px system-ui, sans-serif";
  // Stale-tick fade (set by the draw loop once extrapolation runs past
  // its glide window): boxes dim instead of freezing bright.
  const boxAlpha = Number(d._fade) || 1;
  if (boxAlpha < 1) ctx.globalAlpha = boxAlpha;
  let alertOn = false;
  for (const b of d.boxes || []) {
    // Faces layer draws ONLY face rectangles (above) - generic tracked
    // boxes (vehicles, people) do not belong on it; the server JPEG
    // render has always been faces-only and the canvas must match.
    if (d.layer === "faces") continue;
    // Heat layer draws ONLY the heat field - the layer's whole point is
    // WHERE presence accumulates, and a wall of labeled boxes on top
    // read as "heat shows detections" (it does not).
    if (d.layer === "heat") break;
    const isPose = (d.layer === "pose" || d.layer === "gestures");
    if (isPose && b.cls !== "person") continue;
    if (d.layer === "body" && !b.flag) continue;
    if (d.layer === "gestures" && !b.gestures && !b.kps) continue;
    if (d.layer === "plates" && b.cls === "person") continue;
    const ox = (b.vx || 0) * dtExtra, oy = (b.vy || 0) * dtExtra;
    const x = (b.x1 + ox) * sx, y = (b.y1 + oy) * sy;
    const w = (b.x2 - b.x1) * sx, h = (b.y2 - b.y1) * sy;
    if (x + w < 0 || y + h < 0 || x > cw || y > ch) continue;
    let color = b.cls === "person"
      ? "rgba(74,222,128,0.95)" : "rgba(251,146,60,0.95)";
    let label = `${b.cls} ${Math.round((b.conf || 0) * 100)}%`;
    if (d.layer === "paths" && b.tier)
      label += ` · ${b.tier}`;
    if (d.layer === "gestures" && b.gestures)
      label = `#${b.tid} ${b.gestures.join("+")}`;
    if (d.layer === "body" && b.flag) {
      color = b.alert ? "rgba(239,68,68,0.95)" : "rgba(234,140,8,0.95)";
      label = `#${b.tid} ${String(b.flag).toUpperCase()}`
        + (b.flags ? " " + b.flags.join("+") : "");
      if (b.alert) alertOn = true;
    }
    if (d.layer === "loiter" && b.dwell != null) {
      label += ` · ${b.dwell}s in zone`;
      if ((d.zones || []).some((z) => z.alert)) color = "rgba(239,68,68,0.95)";
    }
    if (d.layer === "plates" && b.plate) {
      color = "rgba(74,222,128,0.95)";
      label = `${b.plate} · ${Math.round((b.plate_conf || 0) * 100)}%`;
    }
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    if (b.coast) ctx.setLineDash([6, 5]);   // coasting on prediction only
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    if (b.kps) _drawSkeleton(ctx, b.kps, sx, sy, ox, oy);
    const tw = ctx.measureText(label).width + 8;
    ctx.fillStyle = "rgba(15,23,42,0.85)";
    ctx.fillRect(x, Math.max(0, y - 16), tw, 16);
    ctx.fillStyle = "#f8fafc";
    ctx.fillText(label, x + 4, Math.max(12, y - 4));
  }
  if (boxAlpha < 1) ctx.globalAlpha = 1;

  // Operating-envelope note: what the layer can physically see right
  // now ("7 people, skeletons on 2 (>=96px only)") - silent emptiness
  // used to read as breakage.
  if (d.envelope) {
    ctx.font = "11px system-ui, sans-serif";
    const tw = ctx.measureText(d.envelope).width + 14;
    ctx.fillStyle = "rgba(15,23,42,0.8)";
    ctx.fillRect(8, 8, tw, 20);
    ctx.fillStyle = "#94a3b8";
    ctx.fillText(d.envelope, 15, 22);
    ctx.font = "12px system-ui, sans-serif";
  }

  // Gesture session tally (bottom-left chip, mirrors the JPEG caption).
  if (d.layer === "gestures" && d.gesture_counts) {
    const txt = "session: " + Object.entries(d.gesture_counts)
      .map(([g, n]) => `${g} x${n}`).join(", ");
    ctx.fillStyle = "rgba(15,23,42,0.85)";
    ctx.fillRect(8, ch - 30, ctx.measureText(txt).width + 14, 22);
    ctx.fillStyle = "#f8fafc";
    ctx.fillText(txt, 14, ch - 14);
  }

  // Body-anomaly alert: burn a red frame so it can't be missed.
  if (alertOn) {
    ctx.lineWidth = 6;
    ctx.strokeStyle = "rgba(239,68,68,0.9)";
    ctx.strokeRect(3, 3, cw - 6, ch - 6);
  }
}

// COCO-17 keypoint skeleton edges (indices into the kps array).
const _SKELETON_EDGES = [
  [5, 7], [7, 9], [6, 8], [8, 10], [5, 6], [5, 11], [6, 12],
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [0, 5], [0, 6],
];

function _drawSkeleton(ctx, kps, sx, sy, ox, oy) {
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(96,165,250,0.95)";
  for (const [a, b] of _SKELETON_EDGES) {
    const p = kps[a], q = kps[b];
    if (!p || !q || p[2] < 0.3 || q[2] < 0.3) continue;
    ctx.beginPath();
    ctx.moveTo((p[0] + ox) * sx, (p[1] + oy) * sy);
    ctx.lineTo((q[0] + ox) * sx, (q[1] + oy) * sy);
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(219,234,254,0.95)";
  for (const k of kps) {
    if (!k || k[2] < 0.3) continue;
    ctx.beginPath();
    ctx.arc((k[0] + ox) * sx, (k[1] + oy) * sy, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

const _TRAIL_PALETTE = [
  "rgba(96,165,250,0.9)", "rgba(74,222,128,0.9)", "rgba(251,146,60,0.9)",
  "rgba(232,121,249,0.9)", "rgba(250,204,21,0.9)", "rgba(45,212,191,0.9)",
];
function _trailColor(tid) {
  return _TRAIL_PALETTE[Math.abs(tid) % _TRAIL_PALETTE.length];
}

async function _refreshAnalysisBg(a) {
  if (!a || !a.bg) return;
  try {
    const r = await fetch(
      `/api/analysis/frame?cam=${encodeURIComponent(a.cam)}&_=${Date.now()}`,
      { cache: "no-store" });
    if (r.status !== 200
        || !(r.headers.get("Content-Type") || "").includes("image")) return;
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    a.bg.src = url;
    if (a.lastBgUrl) URL.revokeObjectURL(a.lastBgUrl);
    a.lastBgUrl = url;
  } catch (_) { /* transient - next tick tries again */ }
}

function _heatColor(v, alpha) {
  // v in [0,1]: cold (blue) -> warm (yellow) -> hot (red).
  const r = Math.round(255 * Math.min(1, v * 2));
  const g = Math.round(255 * Math.min(1, (1 - Math.abs(v - 0.5) * 2)));
  const b = Math.round(255 * Math.max(0, 1 - v * 2));
  return `rgba(${r},${g},${b},${alpha})`;
}

function stopTileAnalysis(st) {
  const a = st.analysis;
  if (!a) return;
  clearInterval(a.timer);
  if (a.videoStateTimer) clearInterval(a.videoStateTimer);
  if (a.evTimer) clearInterval(a.evTimer);
  if (a.evStrip) a.evStrip.remove();
  if (a.lastBgUrl) URL.revokeObjectURL(a.lastBgUrl);
  st.analysis = null;
  fetch(`/api/analysis/stop?cam=${encodeURIComponent(a.cam)}`,
        { method: "POST" }).catch(() => {});
  const wrap = st.videoWrap.querySelector(".analysis-wrap");
  if (wrap) wrap.remove();
  // Tear down the Line-layer history strip if one was showing on this tile.
  const strip = st.tile && st.tile.querySelector(".crossings-strip");
  if (strip) strip.remove();
  if (!st._overlayWasHidden) st.overlay.style.display = "";
  // Overlay mode never tore down the video, so no rebuild needed.
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
      <!-- fix 3: the stored heat depth, finally selectable. Only shown in
           heat mode on the PRIVATE dashboard (the API renders any combo
           from the VM-published grids; the public copy keeps the single
           published overlay). -->
      <div class="heat-controls" data-heat-controls hidden>
        <select data-heat-layer title="which detections feed the map">
          <option value="person">people</option>
          <option value="vehicles">vehicles</option>
          <option value="other">other</option>
        </select>
        <select data-heat-part title="local-time daypart">
          <option value="">all day</option>
          <option value="night">night</option>
          <option value="morning">morning</option>
          <option value="afternoon">afternoon</option>
          <option value="evening">evening</option>
        </select>
      </div>
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
        // Don't fight our own hold-back seek: when an analysis session
        // deliberately parked the player 20 s behind live for overlay
        // alignment, snapping back to live edge here would undo the
        // sync every state change (Chrome buffers, tab visibility flip,
        // network hiccup).
        if (st._analysisHoldSeek) return;
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
    const hls = new window.Hls({ lowLatencyMode: true, liveSyncDuration: 3 });
    hls.loadSource(src);
    hls.attachMedia(video);
    // Kick play() the moment the manifest parses. Chrome allows
    // muted-autoplay but sometimes never fires it if the element was
    // rendered outside the viewport at attach time (which happens for the
    // bottom row before scroll). Explicit .play() removes that dependency
    // on scroll position; the promise-rejection swallow keeps the flow
    // clean when browsers block autoplay in exotic contexts.
    hls.on(window.Hls.Events.MANIFEST_PARSED, () => {
      st._ytproxyRetries = 0;
      const p = video.play();
      if (p && p.catch) p.catch(() => { /* autoplay blocked; user clicks play */ });
    });
    hls.on(window.Hls.Events.ERROR, (_, data) => {
      if (!data.fatal) return;
      console.warn("hls.js fatal error on", src, data);
      // /ytproxy sources: googlevideo's signed URLs rotate every few
      // hours - a rebuild re-asks the server, which re-resolves a fresh
      // one. Retry FOREVER with exponential backoff (2s..30s): a long
      // overnight run must survive every rotation, and the server's
      // negative-cache makes each retry nearly free while the camera
      // is genuinely down.
      if (src.startsWith("/ytproxy") && st.lastVideoBuild) {
        st._ytproxyRetries = (st._ytproxyRetries || 0) + 1;
        const delay = Math.min(30000,
                               2000 * Math.pow(2, st._ytproxyRetries - 1));
        setTimeout(() => {
          if (st.analysis) return;   // tile switched to analysis meanwhile
          buildVideoInto(st, st.lastVideoBuild.cfg, st.lastVideoBuild.slot);
        }, delay);
        return;
      }
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
  // CLOUD MODE ONLY: in local mode the 24h history comes from the LOCAL
  // producers' per-slot history files (see _pollLocalHistory below) - the
  // cloud collector watches its own country ladder, and its rows have no
  // business on a chart titled with the operator's picked cameras.
  if (!LOCAL_MODE) {
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
  }

  setInterval(renderCombinedChart, 4000);
  renderCombinedChart();

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

// Series label for a tile's 24h history. LOCAL mode now plots the LOCAL
// producers' history of the operator's own picked cameras (the cloud
// join was severed in the Turkey-cleanup pass), so the legend carries
// the pick's own name - a Bangkok curve gets a Bangkok title. Cloud
// mode keeps the cloud slot's area name.
function tileCloudLabel(slot) {
  if (LOCAL_MODE) return slot.placeholder_name || slot.display_area;
  return slot.display_area;
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

// -----------------------------------------------------------------------------
// Zones editor (loiter areas + parking spots)
// -----------------------------------------------------------------------------
// Click to drop vertices on the snapshot, "Close polygon" (or double-click)
// seals the current shape, Save POSTs the full set to /api/zones. Zones of
// the OTHER kind are preserved untouched - drawing parking spots never
// clobbers loiter areas and vice versa. A running session hot-reloads the
// file within a few seconds, no restart.

const zoneEditor = document.createElement("div");
zoneEditor.style.cssText =
  "display:none;position:fixed;inset:0;z-index:70;background:rgba(2,6,23,.82);" +
  "align-items:center;justify-content:center";
zoneEditor.innerHTML = `
  <div style="background:#0f172a;border:1px solid #334155;border-radius:12px;
              padding:16px 18px;max-width:800px;width:94%;color:#e2e8f0">
    <h3 style="margin:0 0 4px;font-size:17px"><span data-ze-title></span> -
      <span data-ze-cam></span></h3>
    <div style="color:#94a3b8;font-size:13px;margin-bottom:10px">
      Click the snapshot to drop polygon corners; double-click (or the
      button) closes the shape. Repeat for more zones, then Save.</div>
    <div style="position:relative;background:#020617;border:1px solid #334155;
                border-radius:8px;overflow:hidden">
      <img data-ze-img style="display:block;width:100%;height:auto;
                              user-select:none;-webkit-user-drag:none">
      <canvas data-ze-canvas style="position:absolute;inset:0;width:100%;
                                     height:100%;cursor:crosshair"></canvas>
    </div>
    <div data-ze-dwellrow style="margin-top:10px;font-size:13px;color:#cbd5e1">
      Loiter alert after <input data-ze-dwell type="number" min="5" max="3600"
        value="30" style="width:70px;background:#1e293b;color:#e2e8f0;
        border:1px solid #334155;border-radius:6px;padding:3px 6px"> seconds
      inside a zone.</div>
    <div data-ze-err style="color:#f87171;font-size:13px;min-height:18px;
                            margin-top:8px"></div>
    <div style="display:flex;gap:10px;margin-top:6px;flex-wrap:wrap">
      <button data-ze-closepoly style="cursor:pointer;background:#334155;
              border:0;color:#fff;border-radius:8px;padding:7px 14px">
        Close polygon</button>
      <button data-ze-undo style="cursor:pointer;background:#334155;border:0;
              color:#fff;border-radius:8px;padding:7px 14px">Undo point</button>
      <button data-ze-clear style="cursor:pointer;background:#7f1d1d;border:0;
              color:#fff;border-radius:8px;padding:7px 14px">Clear all</button>
      <button data-ze-save style="cursor:pointer;background:#2563eb;border:0;
              color:#fff;border-radius:8px;padding:7px 18px">Save</button>
      <button data-ze-cancel style="cursor:pointer;background:#1e293b;
              border:1px solid #334155;color:#e2e8f0;border-radius:8px;
              padding:7px 14px">Close</button>
    </div>
  </div>`;
document.body.appendChild(zoneEditor);

const _zeImg = zoneEditor.querySelector("[data-ze-img]");
const _zeCanvas = zoneEditor.querySelector("[data-ze-canvas]");
const _zeErr = zoneEditor.querySelector("[data-ze-err]");
let _zeCam = null, _zeKind = "loiter";
let _zeZones = [];     // zones of the edited kind (editable)
let _zeOthers = [];    // zones of the other kind (preserved on save)
let _zeCurrent = [];   // in-progress polygon, normalized points

function _zeRedraw() {
  const r = _zeImg.getBoundingClientRect();
  _zeCanvas.width = Math.max(1, Math.round(r.width));
  _zeCanvas.height = Math.max(1, Math.round(r.height));
  const cw = _zeCanvas.width, ch = _zeCanvas.height;
  const ctx = _zeCanvas.getContext("2d");
  ctx.clearRect(0, 0, cw, ch);
  ctx.font = "12px system-ui, sans-serif";
  for (const z of _zeZones) {
    ctx.beginPath();
    ctx.moveTo(z.points[0][0] * cw, z.points[0][1] * ch);
    for (let i = 1; i < z.points.length; i++)
      ctx.lineTo(z.points[i][0] * cw, z.points[i][1] * ch);
    ctx.closePath();
    ctx.fillStyle = "rgba(74,222,128,0.15)";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(74,222,128,0.95)";
    ctx.stroke();
    ctx.fillStyle = "#f8fafc";
    ctx.fillText(z.name || "?",
                 z.points[0][0] * cw + 4, z.points[0][1] * ch + 14);
  }
  if (_zeCurrent.length) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(250,204,21,0.95)";
    ctx.beginPath();
    ctx.moveTo(_zeCurrent[0][0] * cw, _zeCurrent[0][1] * ch);
    for (let i = 1; i < _zeCurrent.length; i++)
      ctx.lineTo(_zeCurrent[i][0] * cw, _zeCurrent[i][1] * ch);
    ctx.stroke();
    ctx.fillStyle = "rgba(250,204,21,0.95)";
    for (const p of _zeCurrent) {
      ctx.beginPath();
      ctx.arc(p[0] * cw, p[1] * ch, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function _zeClosePoly() {
  if (_zeCurrent.length < 3) {
    _zeErr.textContent = "a polygon needs at least 3 points";
    return;
  }
  const prefix = _zeKind === "parking" ? "P" : "Z";
  _zeZones.push({
    kind: _zeKind,
    name: prefix + (_zeZones.length + 1),
    points: _zeCurrent.slice(),
  });
  _zeCurrent = [];
  _zeErr.textContent = "";
  _zeRedraw();
}

_zeCanvas.addEventListener("click", (e) => {
  const r = _zeCanvas.getBoundingClientRect();
  _zeCurrent.push([
    Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
    Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))]);
  _zeRedraw();
});
_zeCanvas.addEventListener("dblclick", (e) => {
  e.preventDefault();
  // The dblclick already delivered two click events for the same spot -
  // drop the duplicate vertex before sealing.
  if (_zeCurrent.length >= 2) _zeCurrent.pop();
  _zeClosePoly();
});
zoneEditor.querySelector("[data-ze-closepoly]")
  .addEventListener("click", _zeClosePoly);
zoneEditor.querySelector("[data-ze-undo]").addEventListener("click", () => {
  if (_zeCurrent.length) _zeCurrent.pop();
  else _zeZones.pop();
  _zeRedraw();
});
zoneEditor.querySelector("[data-ze-clear]").addEventListener("click", () => {
  _zeZones = []; _zeCurrent = [];
  _zeRedraw();
});
zoneEditor.querySelector("[data-ze-cancel]").addEventListener("click",
  () => { zoneEditor.style.display = "none"; });
zoneEditor.querySelector("[data-ze-save]").addEventListener("click",
  async () => {
    if (_zeCurrent.length) _zeClosePoly();
    const dwell = Number(zoneEditor.querySelector("[data-ze-dwell]").value)
                  || 30;
    const mine = _zeZones.map((z) => (_zeKind === "loiter"
      ? { ...z, dwell_s: Math.min(3600, Math.max(5, dwell)) } : z));
    try {
      const r = await fetch(`/api/zones?cam=${encodeURIComponent(_zeCam)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ zones: [..._zeOthers, ...mine] }),
      });
      if (!r.ok) throw new Error((await r.text()).slice(0, 120));
      zoneEditor.style.display = "none";
    } catch (e) {
      _zeErr.textContent = "save failed: " + e.message;
    }
  });

async function openZoneEditor(cam, kind, snapshotUrl) {
  _zeCam = cam;
  _zeKind = kind === "parking" ? "parking" : "loiter";
  _zeCurrent = [];
  _zeErr.textContent = "";
  zoneEditor.querySelector("[data-ze-title]").textContent =
    _zeKind === "parking" ? "Parking spots" : "Loitering zones";
  zoneEditor.querySelector("[data-ze-cam]").textContent = cam;
  zoneEditor.querySelector("[data-ze-dwellrow]").style.display =
    _zeKind === "loiter" ? "" : "none";
  try {
    const j = await fetch(`/api/zones?cam=${encodeURIComponent(cam)}`)
      .then((r) => r.json());
    const all = Array.isArray(j.zones) ? j.zones : [];
    _zeZones = all.filter((z) => z.kind === _zeKind);
    _zeOthers = all.filter((z) => z.kind !== _zeKind);
    const dz = _zeZones.find((z) => z.dwell_s);
    if (dz) zoneEditor.querySelector("[data-ze-dwell]").value = dz.dwell_s;
  } catch (_) { _zeZones = []; _zeOthers = []; }
  _zeImg.onload = _zeRedraw;
  _zeImg.src = snapshotUrl;
  zoneEditor.style.display = "flex";
  if (_zeImg.complete) _zeRedraw();
}

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

// ============================================================================
// LOCAL_MODE model-view poll (2026-08-13, backported on 27bced9 baseline).
//
// Fires the tile's Activity Index badge + anomaly badge + KPI overlay from
// the ModelViewProducer JSON that app/local_producers.py writes every ~25 s
// to web/snapshots/model_view/local_*.json when the operator picked cameras
// that are NOT the VM's active grid (typical for a Thailand pick while the
// VM watches Turkey - the cloudMismatch guard hides badges in that case).
//
// Rolling window of the last 6 rounds (~2.5 min) drives computeActivity so a
// single glitchy sample cannot swing the badge. Same absolute bands as the
// cloud path (people bucket + weighted vehicle load).
// ============================================================================
const _LOCAL_ACTIVITY_WINDOW = 6;
const _LOCAL_HISTORY = Object.create(null);   // slot_id -> [{person, vehicles, ts}, ...]

function _updateLocalTileBadges(slotId, j) {
  const st = tileState[slotId];
  if (!st) return;
  const person   = Number(j?.counts?.person   ?? 0);
  const vehicles = Number(j?.counts?.vehicles ?? 0);
  const ts       = j?.at ? j.at * 1000 : Date.now();
  const hist = _LOCAL_HISTORY[slotId] || (_LOCAL_HISTORY[slotId] = []);
  hist.push({ person, vehicles, ts, counts: j?.counts || null });
  if (hist.length > _LOCAL_ACTIVITY_WINDOW) hist.shift();

  // Activity Index (X/10 + label) - same math as the cloud path.
  const act = computeActivity(hist);
  st.activityBadge.style.display = "";
  setActivityBadge(st, act);

  // LIVE per-camera ANOMALY indicator. Load stays NUMBERS-ONLY on the
  // activity badge (X/10) - a busy street is not an anomaly, and the
  // old idx>=8 trigger double-counted it here. This badge flags only
  // true anomaly states, same kinds as the collector's table:
  //   camera_obstructed - one confident box covers 50%+ of the view
  //   camera_dark       - the view went black (covered lens, power cut)
  //   extreme_load      - absolute EVENT scale: 50+ people / 38+ units
  const load = act?.load ?? 0;
  const now  = act?.now  ?? 0;
  let mood = "unk", msg = "no data yet";
  if (hist.length >= 2) {
    mood = "ok";
    msg = act ? `no anomaly · activity ${act.idx}/10 ${act.label}` : "ok";
    if (now >= 50 || load >= 38) {
      mood = "spike";
      msg = `extreme load - ${now} people + ${load} veh-load units`;
    }
  }
  if (j?.dark != null) {
    mood = "spike";
    msg = `camera dark - view went black (mean luma ${j.dark})`;
  }
  if (j?.obstructed) {
    mood = "spike";
    msg = `camera obstructed - ${j.obstructed.cls} covers `
        + `${Math.round((j.obstructed.frac || 0) * 100)}% of view`;
  }
  st.anomalyBadge.style.display = "";
  st.anomalyBadge.className = `anomaly-badge ${mood}`;
  const anomalyTextEl = st.anomalyBadge.querySelector("[data-anomaly-text]");
  if (anomalyTextEl) {
    anomalyTextEl.textContent = mood === "spike" ? "!"
                              : mood === "unk"   ? "-"
                              : "ok";
  }
  st.anomalyBadge.title = msg;

  // KPI overlay (People / Vehicles) - never cleared by cloudMismatch anymore.
  if (!st.analysis) st.overlay.style.display = "";
  const setK = (k, v) => {
    const el = [...st.latestVals].find((x) => x.dataset.k === k);
    if (el) el.textContent = v != null ? v : "-";
  };
  setK("person", person);
  setK("vehicles", vehicles);
  st.lastSampleMs = ts;
  if (typeof renderSampleAge === "function") renderSampleAge(st);
}

if (LOCAL_MODE) {
  const _pollLocalModelView = async () => {
    for (const slot of GRID_SLOTS) {
      const meta_url = `/snapshots/model_view/${slot.slot_id}.json?_=` + Date.now();
      try {
        const r = await fetch(meta_url, { cache: "no-store" });
        if (!r.ok) continue;
        const j = await r.json();
        _updateLocalTileBadges(slot.slot_id, j);
      } catch (_) { /* file not written yet - keep placeholder */ }
    }
  };
  _pollLocalModelView();
  setInterval(_pollLocalModelView, 8000);

  // 24h footfall history from the LOCAL producers (one JSONL per slot,
  // one row per ~30s round) - feeds the combined chart and the per-tile
  // aggregates with the operator's OWN cameras instead of cloud rows.
  const _pollLocalHistory = async () => {
    for (const slot of GRID_SLOTS) {
      const url = `/snapshots/model_view/${slot.slot_id}_history.jsonl?_=`
                + Date.now();
      try {
        const r = await fetch(url, { cache: "no-store" });
        if (!r.ok) continue;
        const text = await r.text();
        const cutoff = Date.now() - 24 * 3600 * 1000;
        const rows = [];
        for (const line of text.split("\n")) {
          if (!line.trim()) continue;
          try {
            const row = JSON.parse(line);
            if (new Date(row.ts).getTime() >= cutoff) rows.push(row);
          } catch (_) { /* torn write mid-append - skip the line */ }
        }
        if (!rows.length) continue;
        tileState[slot.slot_id].history = rows;
        renderTileChart(slot.slot_id, rows);
        updateAggregates(slot.slot_id, rows);
      } catch (_) { /* history not written yet */ }
    }
  };
  _pollLocalHistory();
  setInterval(_pollLocalHistory, 60000);
}
