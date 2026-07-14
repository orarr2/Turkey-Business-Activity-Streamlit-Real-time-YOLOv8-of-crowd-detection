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
  // config/grid arrives.
  buildVideoInto(tileState[slot.slot_id],
    { active_hls: slot.placeholder_hls, active_page: slot.placeholder_page },
    slot);
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
      lastSampleMs: null,
    };
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
  } else if (embed && embed.includes("player.tvkur.com")) {
    // loading="lazy" postponed the iframe request until scroll, which is
    // exactly the wrong behavior for the top row of the dashboard - one of
    // the four cams could stay dark on a short viewport. Load eagerly.
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
      if (isShownAnomaly(r)) events.push({ area: slot.display_area, r });
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
  loiter:    { icon: "⏱", label: "prolonged presence" },
  returning: { icon: "↩", label: "returning visitor" },
};

// Keep the full events list in module scope so the accordion can look up
// prior sightings of the same entity without re-querying Firestore.
let _ALL_EVENTS = [];

function renderEventsTable(events) {
  const wrap = document.getElementById("events-table-wrap");
  if (!wrap) return;
  _ALL_EVENTS = events;
  const slotLabel = (id) => {
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
