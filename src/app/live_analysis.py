"""Live advanced-analysis engine for the private dashboard (fix 2).

The fix1-B picker returned ONE static analyzed frame (and aimed it at the
CLOUD camera paired with the tile, not the camera the operator was
watching). The fix 2 requirement replaces that completely: analysis is
LIVE and CONTINUOUS on the exact camera whose tile was clicked - the tile
morphs in place into a stream of analyzed frames while the VM collector
runs untouched. This module owns that:

  * one LiveSession per camera (max MAX_SESSIONS = the four grid tiles),
    holding the SAME stream the tile plays (registry camera or a
    local-picker slot resolved from web/local_grid.json), pacing one
    detection tick roughly every TICK_TARGET_S;
  * ONE analysis layer per session - the fix 2 semantics: a single layer
    per camera, up to four live analyses across the grid, duplicates
    fine. Switching the layer MUTATES the running session: the stream,
    the tracker and every accumulator survive, so heat -> gestures ->
    heat resumes the accumulated map instead of restarting;
  * per-layer rendering that draws ONLY that layer's semantics (pose =
    skeletons on close-enough people, never detection boxes) and says
    so honestly when a layer finds nothing ("none detected right now");
  * a latest-JPEG buffer the dashboard polls (~1/s). The client never
    touches the model directly and the VM is never involved.

Compute reality on an operator PC (CPU): one active session runs about
1-2 fps; four concurrent sessions about 0.3-0.5 fps each - INFER_LOCK
serializes model access so four sessions degrade gracefully instead of
thrashing the same weights from four threads.

The draw_* functions are pure (frame + data in, frame out) - since
fix 3 removed the one-shot layers branch, this module is the ONLY place
a layer's look is defined.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from app.heatmap import GRID_H, GRID_W

_SRC_ROOT = Path(__file__).resolve().parent.parent

# The seven analysis layers an operator can run live. "line" is the
# threshold-crossing layer added in fix 2.
LIVE_LAYERS = ("paths", "pose", "gestures", "body", "faces", "heat", "line",
               "loiter", "parking")
DEFAULT_LOITER_DWELL_S = 30.0
_VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle", "bicycle")
LAYER_TITLES = {
    "paths":    "Paths & speeds",
    "pose":     "Pose & skeleton",
    "gestures": "Hand gestures",
    "body":     "Body anomalies",
    "faces":    "Face detection",
    "heat":     "Heat signature",
    "line":     "Line crossing",
}

MAX_SESSIONS = 4          # one per grid tile - the fix 2 cap
IDLE_STOP_S = 60.0        # no client poll this long -> session shuts down
TICK_TARGET_S = 0.8       # pacing floor between inference ticks
LIVE_IMGSZ = 640

# ---- overlay display filters (2026-08 accuracy pass) ----------------------
# Raw single-frame detections flicker: one-tick ghosts, low-conf floaters,
# and COCO classes that make no sense on a street cam ("train" on a fence).
# The overlay publishes tracker-CONFIRMED objects instead - seen on at
# least DISPLAY_MIN_HITS ticks, recent conf at or above DISPLAY_MIN_CONF,
# class not blacklisted. The analytics accumulators (heat, crossings,
# counts) still consume every raw detection - display strictness must not
# starve the statistics.
DISPLAY_MIN_HITS = 2
DISPLAY_MIN_CONF = 0.40
DISPLAY_MAX_MISSES = 1     # allow 1-tick coasting through brief occlusion
DISPLAY_CLASS_BLACKLIST = {"train", "boat", "airplane"}

# cam_id -> seconds between wall clock and the stream's PROGRAM-DATE-TIME
# live edge, measured by dashboard_server's /ytproxy manifest handler on
# every playlist refresh. Lets _publish_data stamp each tick with the
# capture time in the VIDEO's own clock, so the browser can align boxes
# with the exact frame the operator is watching.
STREAM_PDT_OFFSET: dict[str, float] = {}
JPEG_MAX_W = 960
JPEG_QUALITY = 80
TRACK_KEEP = 48           # per-track box history cap (live runs are open-ended)
TRAIL_MAX_PTS = 40
GRAB_FAIL_REFRESH = 3     # consecutive grab failures before re-resolving
# Default counting line for cameras without a configured "line" (the local
# picker's cameras): horizontal, at 62% height - the sidewalk band on a
# typical street view. Same normalized [[x,y],[x,y]] convention as
# cameras.py; crossing negative -> positive side of A->B counts as "in".
DEFAULT_LINE = [[0.10, 0.62], [0.90, 0.62]]
# Hot-reload cadence: every N seconds the session restats the line JSON
# and picks up any change the operator saved while a session is running,
# so redrawing the line takes effect without a stop/start round-trip.
LINE_RELOAD_POLL_S = 5.0
# Per-tid crossing cooldown: a foot point that jitters within a few pixels
# of the line can produce a real neg->pos->neg burst in one second. This
# rejects any crossing for a tid that already crossed within N seconds,
# regardless of direction. 2 s is small enough that a person who really
# doubled back is still counted twice, and large enough to eat the jitter.
CROSSING_COOLDOWN_S = 2.0

# Serializes EVERY model call in this process (detection + pose, live
# sessions + the one-shot deep window): ultralytics predict is not
# thread-safe on a shared model object.
INFER_LOCK = threading.Lock()

# Body-anomaly layer: which behavior labels count as an anomaly worth
# drawing (everything else - walking/standing/dwelling/driving/parked -
# is normal street life).
BODY_ANOMALY_LABELS = frozenset({"fall_suspect", "erratic", "running"})


class BusyError(RuntimeError):
    """All MAX_SESSIONS live-analysis slots are taken."""


# ---------------------------------------------------------------------------
# Camera resolution: registry cameras by id, local-picker slots by slot_id.
# ---------------------------------------------------------------------------

def resolve_cam(cam_id: str, grid_path: Path | None = None) -> dict:
    """Return an analyzable camera dict for `cam_id`.

    Registry cameras (app/cameras.py) win; otherwise the local picker's
    web/local_grid.json is searched by slot_id and a stream-resolvable
    dict is synthesized from the slot's embed/HLS/page fields. Raises
    ValueError when the id is unknown or the slot has no usable stream.
    """
    from app.cameras import CAMERAS
    cam = CAMERAS.get(cam_id)
    if cam is not None:
        return {"id": cam_id, **cam}
    p = grid_path or (_SRC_ROOT / "web" / "local_grid.json")
    if p.exists():
        try:
            slots = json.loads(p.read_text(encoding="utf-8")).get("slots") or []
        except (OSError, ValueError):
            slots = []
        for slot in slots:
            if slot.get("slot_id") == cam_id:
                return _cam_from_slot(slot)
    raise ValueError(f"unknown camera {cam_id!r}")


def _cam_from_slot(slot: dict) -> dict:
    cam_id = slot["slot_id"]
    name = slot.get("placeholder_name") or cam_id
    emb = slot.get("placeholder_embed") or ""
    m = re.search(r"/embed/([\w-]{11})", emb)
    if m:
        return {"id": cam_id, "name": name, "kind": "youtube",
                "url": f"https://www.youtube.com/watch?v={m.group(1)}"}
    hls = slot.get("placeholder_hls") or ""
    m = re.match(r"^/tvkur/([^/]+)/", hls)
    if m:
        # The dashboard plays tvkur through its local proxy; the analysis
        # loop talks to the upstream directly (grab_frame carries the
        # Referer/Origin the host demands).
        return {"id": cam_id, "name": name, "kind": "hls",
                "url": f"https://content.tvkur.com/l/{m.group(1)}/master.m3u8"}
    if hls.startswith("http"):
        return {"id": cam_id, "name": name, "kind": "hls", "url": hls}
    page = slot.get("placeholder_page") or ""
    if "youtube.com/watch" in page:
        return {"id": cam_id, "name": name, "kind": "youtube", "url": page}
    if "webcamera24.com" in page:
        return {"id": cam_id, "name": name, "kind": "webcamera24",
                "url": page, "page": page}
    # skylinewebcams pages resolve through detect_core.resolve_skyline; the
    # picker writes them as a plain page link (no HLS/embed hint), so match
    # on the host and hand back a kind="skyline" dict.
    if "skylinewebcams.com" in page:
        return {"id": cam_id, "name": name, "kind": "skyline",
                "url": page, "page": page}
    raise ValueError(f"camera {cam_id!r} has no analyzable stream")


# ---------------------------------------------------------------------------
# Shared accumulators (pure - unit-testable without streams or a model).
# ---------------------------------------------------------------------------

def bump_heat(grid: list, boxes: list[dict], frame_shape, weight: float) -> None:
    """Bank each box's foot point into the session dwell grid."""
    H, W = frame_shape[:2]
    if not (H and W):
        return
    for b in boxes:
        fx = (b["x1"] + b["x2"]) / 2.0
        fy = b["y2"]
        if not (0 <= fx <= W and 0 <= fy <= H):
            continue
        gx = min(GRID_W - 1, int(fx / W * GRID_W))
        gy = min(GRID_H - 1, int(fy / H * GRID_H))
        grid[gy][gx] += weight


def grid_from_tracks(tracks, frame_shape) -> list:
    """One-shot dwell grid from a closed window's tracks (behavior.py's
    heat layer - same accumulation, no session)."""
    grid = [[0.0] * GRID_W for _ in range(GRID_H)]
    for tr in tracks:
        bump_heat(grid, tr.boxes, frame_shape, 1.0)
    return grid


def update_crossings(side_state: dict, tracks, frame_shape, line: list,
                     cross: dict, on_event=None, frame=None,
                     cam_id: str | None = None,
                     classes: list | set | None = None,
                     last_cross_ts: dict | None = None,
                     cooldown_s: float = CROSSING_COOLDOWN_S,
                     now: float | None = None) -> None:
    """Advance the session in/out counters from each visible track's
    NEWEST foot point. `side_state` remembers the last STRICTLY-signed
    side per track id (side == 0 means "on the line" and is stored as
    None - the next tick with a real sign starts the comparison from
    there). A crossing = a strict sign flip between two consecutive
    signed observations. Same convention as
    detect_core.count_line_crossings: negative -> positive side of the
    A->B line = "in".

    on_event(direction, track, frame): optional callback fired on each
    crossing so the caller can persist an event + snapshot to
    data/crossings/<cam>.jsonl (see log_crossing_event below). cam_id +
    frame are forwarded to the callback so it can crop the mover for the
    event image. Absent callback -> counters only, backward-compatible.

    `classes`: iterable of class names to count (None = every class).
    Tracks whose `cls` is not in the set are skipped BEFORE side tracking
    so their sign changes never update `side_state` and never fire a
    counter or event.

    `last_cross_ts` + `cooldown_s`: per-tid cooldown to swallow the
    jitter burst you get when a foot point rides right on the line. If a
    tid already crossed within `cooldown_s` seconds, the next crossing
    (either direction) is dropped. Pass None for `last_cross_ts` to
    disable cooldown (the pre-cooldown behavior)."""
    from app.detect_core import _line_side
    H, W = frame_shape[:2]
    if not (H and W):
        return
    cls_filter = set(classes) if classes else None
    if now is None:
        now = time.time()
    for tr in tracks:
        if getattr(tr, "misses", 0):
            continue
        if cls_filter is not None and getattr(tr, "cls", None) not in cls_filter:
            continue
        b = tr.boxes[-1]
        fx = (b["x1"] + b["x2"]) / 2.0
        fy = b["y2"]
        side = _line_side(fx / W, fy / H, line)
        prev = side_state.get(tr.tid)
        # Landing exactly on the line is ambiguous: don't classify it as
        # either side, and don't reset the last known side either - a
        # track that jitters neg -> 0 -> neg should count zero crossings.
        if side == 0:
            continue
        side_state[tr.tid] = side
        if prev is None or prev == 0:
            continue
        direction = None
        if prev < 0 and side > 0:
            direction = "in"
        elif prev > 0 and side < 0:
            direction = "out"
        if not direction:
            continue
        if last_cross_ts is not None:
            prev_ts = last_cross_ts.get(tr.tid)
            if prev_ts is not None and (now - prev_ts) < cooldown_s:
                # Jitter suppression: same tid crossed less than
                # cooldown_s ago. Skip without touching the counter or
                # firing an event, but keep the newest side in
                # side_state so the tid can cross again once it moves
                # off the line.
                continue
            last_cross_ts[tr.tid] = now
        if direction == "in":
            cross["in"] = cross.get("in", 0) + 1
        else:
            cross["out"] = cross.get("out", 0) + 1
        if on_event is not None:
            try:
                on_event(direction=direction, track=tr, frame=frame,
                         cam_id=cam_id)
            except Exception as e:
                # Event persistence must never break the session's counter
                # loop. Log and move on.
                print(f"live_analysis: crossing on_event failed: "
                      f"{type(e).__name__}: {e}")


# ---- crossing-event log --------------------------------------------------

# Per-camera JSONL of the most recent line-crossing events. The dashboard's
# /api/crossings?cam=<id> reads this to render toasts + a history strip on
# the Line layer. Bounded rewrite: keep the newest CROSSING_LOG_KEEP rows;
# a full rewrite of ~50 rows is cheap and avoids indefinite growth on a
# camera with heavy traffic.
CROSSING_LOG_KEEP = 50


def _crossings_dir():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent / "data" / "crossings"


def log_crossing_event(cam_id: str, direction: str, track, frame=None) -> None:
    """Append a crossing event to data/crossings/<cam>.jsonl (bounded).
    When `frame` is provided we also save a small jpeg crop of the mover.

    The event fields the frontend reads:
      ts        - ISO-8601 UTC
      direction - "in" | "out"
      cls       - track class (person/car/bus/...)
      snap      - relative URL of the crop, or None
    """
    import json as _json
    import time as _t
    d = _crossings_dir()
    d.mkdir(parents=True, exist_ok=True)
    ts = _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime())
    snap_rel = None
    if frame is not None:
        try:
            import cv2 as _cv
            from pathlib import Path as _P
            b = track.boxes[-1]
            H, W = frame.shape[:2]
            pad = 20
            x1 = max(0, int(b["x1"]) - pad); y1 = max(0, int(b["y1"]) - pad)
            x2 = min(W, int(b["x2"]) + pad); y2 = min(H, int(b["y2"]) + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                # Snaps go under src/web/snapshots/crossings/ so the
                # dashboard's static handler serves them at
                # /snapshots/crossings/<cam>/<file>.jpg with no extra route.
                web_snaps = (_P(__file__).resolve().parent.parent
                             / "web" / "snapshots" / "crossings" / cam_id)
                web_snaps.mkdir(parents=True, exist_ok=True)
                fname = f"{ts.replace(':', '')}_{track.tid}_{direction}.jpg"
                _cv.imwrite(str(web_snaps / fname), crop,
                            [_cv.IMWRITE_JPEG_QUALITY, 72])
                snap_rel = f"snapshots/crossings/{cam_id}/{fname}"
        except Exception as e:
            print(f"log_crossing_event: snap failed: {type(e).__name__}: {e}")
    row = {"ts": ts, "direction": direction,
           "cls": getattr(track, "cls", None),
           "tid": getattr(track, "tid", None), "snap": snap_rel}

    log = d / f"{cam_id}.jsonl"
    lines = []
    if log.exists():
        try:
            lines = log.read_text().splitlines()[-CROSSING_LOG_KEEP + 1:]
        except OSError:
            lines = []
    lines.append(_json.dumps(row))
    tmp = log.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(log)


def read_crossing_events(cam_id: str, limit: int = 20) -> list:
    """Newest-first read of the last N crossing events for a camera.
    Returns [] if the log doesn't exist (nothing has crossed yet)."""
    import json as _json
    log = _crossings_dir() / f"{cam_id}.jsonl"
    if not log.exists():
        return []
    try:
        rows = [_json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        return []
    return list(reversed(rows))[:limit]


# ---------------------------------------------------------------------------
# Layer renderers - each draws ONLY its layer's semantics + an honest
# caption. All mutate/return the given BGR frame.
# ---------------------------------------------------------------------------

def _caption(img, lines) -> "object":
    """Darkened strip at the top with the layer verdict ("no gestures
    detected right now" is a legitimate, expected outcome - fix 2)."""
    import cv2
    if isinstance(lines, str):
        lines = [lines]
    lh, pad = 22, 8
    h = min(img.shape[0], pad * 2 + lh * len(lines) - 8)
    img[0:h] = (img[0:h] * 0.35).astype(img.dtype)
    y = pad + 12
    for i, t in enumerate(lines):
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 if i == 0 else 0.46,
                    (255, 255, 255) if i == 0 else (205, 205, 205),
                    1, cv2.LINE_AA)
        y += lh
    return img


def _chip(img, b: dict, txt: str, color) -> None:
    import cv2
    x1, y2 = int(b["x1"]), int(b["y2"])
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y2 + 2), (x1 + tw + 6, y2 + th + 8), color, -1)
    cv2.putText(img, txt, (x1 + 3, y2 + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)


def _hud_panel(img, lines: list[str], alert: bool = False) -> None:
    """Bordered status panel, top-left - the fall-detection-reference HUD
    (system name, persons in view, flagged count). Red border on alert."""
    import cv2
    lh, pad = 18, 8
    w = 240
    h = pad * 2 + lh * len(lines) - 4
    x0, y0 = 8, 8
    roi = img[y0:y0 + h, x0:x0 + w]
    roi[:] = (roi * 0.25).astype(img.dtype)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h),
                  (0, 0, 230) if alert else (160, 160, 160), 2)
    y = y0 + pad + 8
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x0 + 8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48 if i == 0 else 0.42,
                    (255, 255, 255) if i == 0 else (210, 210, 210),
                    1, cv2.LINE_AA)
        y += lh


def _alert_banner(img, txt: str) -> None:
    """Loud red banner, top-center - fires only while an alert-grade flag
    (fall/erratic) is live, exactly like the operator's reference clip."""
    import cv2
    H, W = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    x0 = max(8, (W - tw) // 2 - 12)
    y0 = 8
    cv2.rectangle(img, (x0, y0), (min(W - 8, x0 + tw + 24), y0 + th + 18),
                  (0, 0, 210), -1)
    cv2.putText(img, txt, (x0 + 12, y0 + th + 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
                cv2.LINE_AA)


def draw_paths_layer(img, tracks, last_boxes: list[dict],
                     stats_by_id: dict):
    """Trails + id boxes + km/h chips - the one layer that legitimately
    shows detection boxes for every class."""
    import cv2
    from app.behavior import _TRAIL_COLORS
    from app.detect_core import draw_boxes
    for tr in tracks:
        color = _TRAIL_COLORS[(tr.tid - 1) % len(_TRAIL_COLORS)]
        pts = [(int((b["x1"] + b["x2"]) / 2), int((b["y1"] + b["y2"]) / 2))
               for b in tr.boxes[-TRAIL_MAX_PTS:]]
        for p0, p1 in zip(pts, pts[1:]):
            cv2.line(img, p0, p1, color, 2, cv2.LINE_AA)
        if pts:
            cv2.circle(img, pts[0], 4, color, -1, cv2.LINE_AA)
    img = draw_boxes(img, last_boxes)
    for b in last_boxes:
        s = stats_by_id.get(b.get("track_id"))
        if s and s.get("kmh_est"):
            _chip(img, b, f"{s['kmh_est']} km/h", (90, 90, 90))
    note = (f"Paths & speeds - {len(last_boxes)} tracked now"
            if last_boxes else "Paths & speeds - nothing tracked yet")
    return _caption(img, [note])


def draw_zones_layer(img, entries: list[dict], kind: str):
    """Polygons + occupancy caption for the loiter / parking layers - the
    JPEG-fallback rendering; the canvas overlay is the primary view."""
    import cv2
    import numpy as np
    H, W = img.shape[:2]
    overlay = img.copy()
    for e in entries:
        pts = np.array([[int(p[0] * W), int(p[1] * H)]
                        for p in e["points"]], dtype=np.int32)
        if kind == "parking":
            hot = bool(e.get("occupied"))
            label = f"{e['name']}: {'occupied' if hot else 'free'}"
        else:
            hot = bool(e.get("alert"))
            label = (f"{e['name']}: {e.get('count', 0)} inside"
                     f", max {int(e.get('max_dwell', 0))}s")
        color = (0, 0, 220) if hot else (0, 200, 80)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
        x0, y0 = int(pts[0][0]), int(pts[0][1])
        cv2.putText(img, label, (x0, max(14, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
    if not entries:
        note = (f"{'Parking' if kind == 'parking' else 'Zone & loitering'}"
                " - nothing drawn yet (use the Draw zones button)")
    elif kind == "parking":
        occ = sum(1 for e in entries if e.get("occupied"))
        note = f"Parking - {occ}/{len(entries)} occupied"
    else:
        note = (f"Zone & loitering - "
                f"{sum(e.get('count', 0) for e in entries)} inside, "
                f"{sum(1 for e in entries if e.get('alert'))} alert(s)")
    return _caption(img, [note])


def draw_pose_layer(img, boxes: list[dict]):
    """Skeletons ONLY, on people close enough for the per-crop pose pass.
    No detection boxes, no vehicles - fix 2's core layer-correctness
    complaint."""
    from app.pose import draw_skeleton
    persons = [b for b in boxes if b.get("cls") == "person"]
    withk = [b for b in persons if b.get("kps")]
    if withk:
        draw_skeleton(img, withk)
    if not persons:
        note = "Pose & skeleton - no people in frame"
    elif not withk:
        note = (f"Pose & skeleton - no skeletons "
                f"({len(persons)} people too far/small for pose)")
    else:
        note = (f"Pose & skeleton - skeletons on {len(withk)} "
                f"of {len(persons)} people")
        if len(withk) < len(persons):
            note += " (rest too far)"
    return _caption(img, [note])


def draw_gestures_layer(img, boxes: list[dict], stats_by_id: dict,
                        session_counts: dict | None = None):
    """Skeleton + gesture chip only for people with a DETECTED gesture."""
    from app.pose import draw_skeleton
    active = []
    for b in boxes:
        if b.get("cls") != "person" or not b.get("kps"):
            continue
        s = stats_by_id.get(b.get("track_id"))
        if s and s.get("gestures"):
            active.append((b, s))
    for b, s in active:
        draw_skeleton(img, [b])
        _chip(img, b, "+".join(s["gestures"]), (190, 120, 0))
    note = ("Hand gestures - "
            + ", ".join(f"#{s.get('id', '?')} {'+'.join(s['gestures'])}"
                        for _, s in active)
            if active else "Hand gestures - none detected right now")
    lines = [note]
    if session_counts is not None:
        tot = ", ".join(f"{g} x{n}"
                        for g, n in sorted(session_counts.items()))
        lines.append(f"session: {tot}" if tot else "session: none yet")
    return _caption(img, lines)


def draw_body_layer(img, boxes: list[dict], stats_by_id: dict):
    """Fall-detection-style live view (the operator's reference clip):
    a status HUD tallies everyone, flagged people get a red box +
    skeleton + verdict chip, and an ALERT banner burns while a
    fall/erratic flag is live. Normal street life stays unmarked."""
    import cv2
    persons = [b for b in boxes if b.get("cls") == "person"]
    flagged = []
    for b in persons:
        s = stats_by_id.get(b.get("track_id"))
        if not s:
            continue
        if s.get("label") in BODY_ANOMALY_LABELS or s.get("pose_flags"):
            flagged.append((b, s))
    for b, s in flagged:
        color = (0, 0, 220) if s.get("alert") else (0, 150, 230)
        cv2.rectangle(img, (int(b["x1"]), int(b["y1"])),
                      (int(b["x2"]), int(b["y2"])), color, 2)
        if b.get("kps"):
            from app.pose import draw_skeleton
            draw_skeleton(img, [b])
        txt = f"#{s.get('id', '?')} {(s.get('label') or '').upper()}"
        extra = [f for f in (s.get("pose_flags") or [])
                 if f and f != s.get("label")]
        if extra:
            txt += " " + "+".join(extra)
        _chip(img, b, txt, color)
    alerts = [s for _, s in flagged if s.get("alert")]
    _hud_panel(img, ["BODY ANOMALIES",
                     f"persons in view: {len(persons)}",
                     f"flagged: {len(flagged)}"
                     + ("" if flagged else " (none right now)")],
               alert=bool(alerts))
    if alerts:
        kinds: dict[str, int] = {}
        for s in alerts:
            k = (s.get("label") or "?").upper().replace("_", " ")
            kinds[k] = kinds.get(k, 0) + 1
        _alert_banner(img, "ALERT! " + ", ".join(
            f"{n} {k}" for k, n in sorted(kinds.items())))
    return img


def draw_faces_layer_img(img, faces_list: list[dict], available: bool = True):
    if faces_list:
        from app.faces import draw_faces
        draw_faces(img, faces_list)
        note = f"Face detection - {len(faces_list)} face(s)"
    elif available:
        note = "Face detection - no faces at this distance/resolution"
    else:
        note = "Face detection - face model not available on this machine"
    return _caption(img, [note])


def draw_heat_layer(img, grid: list, since: float | None = None):
    """Full heat-vision view (fix 3, per the operator's requirement that
    picking heat CHANGES THE WHOLE PICTURE): the frame is re-rendered as
    a thermal-style colormap driven by its own brightness, and the
    session's dwell accumulation on THIS camera burns its zones toward
    the hot end. Not a thermal sensor - the caption says exactly what
    drives the colors. The accumulation itself never belongs to another
    camera (the fix 2 rule)."""
    import cv2
    import numpy as np
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    signal = gray * 0.72
    g = np.asarray(grid, dtype=np.float32)
    peak = float(g.max())
    if peak > 0:
        dwell = np.sqrt(g / peak)
        dwell = cv2.resize(dwell, (W, H), interpolation=cv2.INTER_LINEAR)
        dwell = cv2.GaussianBlur(dwell, (0, 0), sigmaX=max(2.0, W / 96.0))
        m = float(dwell.max())
        if m > 0:
            dwell /= m
        signal = np.clip(signal + dwell * 0.55, 0.0, 1.0)
    out = cv2.applyColorMap((signal * 255).astype(np.uint8),
                            cv2.COLORMAP_INFERNO)
    if since:
        el = int(time.time() - since)
        mm, ss = divmod(el, 60)
        note = (f"Heat vision - dwell accumulating since "
                f"{time.strftime('%H:%M:%S', time.localtime(since))} "
                f"({mm}m{ss:02d}s)")
    else:
        note = "Heat vision - dwell over this window"
    if peak <= 0:
        note += " - no dwell banked yet (brightness only)"
    return _caption(out, [note,
                          "stylized: brightness + dwell, not a thermal "
                          "sensor"])


def draw_line_layer(img, line: list, cross: dict):
    import cv2
    H, W = img.shape[:2]
    (ax, ay), (bx, by) = line
    p0 = (int(ax * W), int(ay * H))
    p1 = (int(bx * W), int(by * H))
    cv2.line(img, p0, p1, (0, 215, 255), 3, cv2.LINE_AA)
    for p in (p0, p1):
        cv2.circle(img, p, 6, (0, 215, 255), -1, cv2.LINE_AA)
    mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
    txt = f"IN {cross.get('in', 0)}  OUT {cross.get('out', 0)}"
    cv2.putText(img, txt, (max(8, mid[0] - 70), max(24, mid[1] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2,
                cv2.LINE_AA)
    return _caption(img, [f"Line crossing - IN {cross.get('in', 0)} / "
                          f"OUT {cross.get('out', 0)} (session total)"])


# ---------------------------------------------------------------------------
# The live session.
# ---------------------------------------------------------------------------

def _pt_in_poly(x: float, y: float, pts: list) -> bool:
    """Ray-casting point-in-polygon on normalized coords."""
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i][0], pts[i][1]
        xj, yj = pts[j][0], pts[j][1]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


class _InferBatcher(threading.Thread):
    """Coalesces concurrent sessions' YOLO requests into one batched call.

    With N sessions free-running behind INFER_LOCK each session ticked
    every N x 1.4 s (measured: 4 cams -> a new frame every 5-8 s). Here a
    session hands its frame in and blocks; the batcher waits a short
    COLLECT_S for the other active sessions (they pace off the same
    previous round, so after one round they arrive nearly together), then
    runs detect_with_boxes_batch once and fans the per-frame results
    back. Single-session cost: +COLLECT_S. Four-session cost: one
    batch-of-4 forward (~2.5x single) instead of 4x serial.
    """

    COLLECT_S = 0.10

    def __init__(self):
        super().__init__(daemon=True, name="live-infer-batcher")
        self._lock = threading.Lock()
        self._reqs: list[dict] = []
        self._wake = threading.Event()
        self.start()

    def infer(self, model, frame, conf: float, gates: dict) -> list[dict]:
        req = {"model": model, "frame": frame, "conf": conf,
               "gates": gates, "done": threading.Event(),
               "out": None, "err": None}
        with self._lock:
            self._reqs.append(req)
        self._wake.set()
        req["done"].wait(timeout=90)
        if req["err"] is not None:
            raise req["err"]
        return req["out"] or []

    def run(self) -> None:
        from app.detect_core import detect_with_boxes_batch
        while True:
            self._wake.wait()
            time.sleep(self.COLLECT_S)
            with self._lock:
                batch, self._reqs = self._reqs, []
                self._wake.clear()
            if not batch:
                continue
            try:
                with INFER_LOCK:
                    outs = detect_with_boxes_batch(
                        batch[0]["model"],
                        [r["frame"] for r in batch],
                        imgsz=LIVE_IMGSZ,
                        per_class_conf_list=[r["gates"] for r in batch],
                        conf_list=[r["conf"] for r in batch])
                for r, (_counts, boxes) in zip(batch, outs):
                    r["out"] = boxes
            except Exception as e:  # noqa: BLE001 - deliver, don't die
                for r in batch:
                    r["err"] = e
            finally:
                for r in batch:
                    r["done"].set()


BATCHER = _InferBatcher()


class _StreamReader(threading.Thread):
    """Continuously tracks the live edge of one direct-HLS stream.

    The old per-tick path opened a fresh cv2.VideoCapture for every
    analyzed frame - measured at 1.0-2.1 s per open on the operator's
    laptop, which alone made a ~2.5 s floor between analysis updates.
    This thread opens the capture ONCE, then grab()s every source frame
    (decode-only, no BGR conversion - the cheap half of read()) to stay
    pinned to the live edge, and retrieve()s a full frame a few times a
    second into `latest`. The analysis tick takes the newest frame
    instantly instead of paying the open cost again and again.
    """

    RETRIEVE_EVERY = 6      # ~4 fresh BGR frames/s at a 25 fps source

    def __init__(self, url: str):
        super().__init__(daemon=True, name="live-analysis-reader")
        self.url = url
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest = None          # newest BGR frame (ndarray)
        self.latest_ts = 0.0
        self.dead = False

    def run(self) -> None:
        from app.detect_core import _open_cap   # applies ffmpeg timeouts
        cap = _open_cap(self.url)
        if not cap.isOpened():
            self.dead = True
            return
        n = 0
        try:
            while not self.stop_event.is_set():
                if not cap.grab():
                    # Live edge starved or the signed manifest rotated:
                    # one in-place reopen attempt, then declare dead and
                    # let LiveSession._grab rebuild us on a fresh URL.
                    cap.release()
                    time.sleep(0.5)
                    cap = _open_cap(self.url)
                    if not cap.isOpened() or not cap.grab():
                        self.dead = True
                        return
                n += 1
                if n % self.RETRIEVE_EVERY == 0:
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        with self.lock:
                            self.latest = frame
                            self.latest_ts = time.time()
        finally:
            cap.release()

    def snapshot(self):
        self.last_used = time.time()
        with self.lock:
            return self.latest

    def snapshot_wait(self, timeout: float = 8.0):
        """snapshot(), but block up to `timeout` for the FIRST frame of a
        freshly-started reader instead of returning None."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            fr = self.snapshot()
            if fr is not None or self.dead:
                return fr
            time.sleep(0.1)
        return self.snapshot()

    def stop(self) -> None:
        self.stop_event.set()


# ---- Shared reader pool ----------------------------------------------------
# ONE persistent live-edge reader per camera, shared by every consumer:
# live-analysis sessions AND local_producers' KPI rounds. Before the pool,
# producers re-opened the HLS stream from scratch every 8 s round (measured
# 1-2 s per open, times four cams) while the sessions each held their own
# open reader for the very same streams - double infrastructure that
# saturated the CPU and slowed everyone's ticks. Readers idle-stop after
# _READER_IDLE_STOP_S without a snapshot() so an unwatched camera doesn't
# decode forever.

_READER_POOL: dict[str, _StreamReader] = {}
_READER_POOL_LOCK = threading.Lock()
_READER_IDLE_STOP_S = 180.0


def get_shared_reader(cam: dict, cam_id: str):
    """The pool's reader for this camera, (re)built as needed.

    Returns None for header-required hosts (tvkur/ibb/skyline) - those
    can't ride a plain VideoCapture and keep their per-tick segment path.
    Raises when the stream can't be resolved (caller counts the failure;
    resolve_stream's negative cache keeps the retry cost near zero).
    """
    from app.detect_core import HEADER_HOSTS, resolve_stream
    url = resolve_stream(cam)
    if any(h in url for h in HEADER_HOSTS):
        return None
    now = time.time()
    with _READER_POOL_LOCK:
        # Idle-stop readers nobody snapshots anymore.
        for key, r in list(_READER_POOL.items()):
            if key != cam_id and now - getattr(r, "last_used", now) \
                    > _READER_IDLE_STOP_S:
                r.stop()
                _READER_POOL.pop(key, None)
        r = _READER_POOL.get(cam_id)
        stale = (r is not None and r.latest is not None
                 and now - r.latest_ts > 10)
        if r is None or r.dead or not r.is_alive() or r.url != url or stale:
            if r is not None:
                r.stop()
            r = _StreamReader(url)
            r.last_used = now
            r.start()
            _READER_POOL[cam_id] = r
        return r


class LiveSession(threading.Thread):
    """One camera's live analysis: stream -> detect -> track -> layer."""

    def __init__(self, cam: dict, model, layer: str):
        super().__init__(daemon=True, name=f"live-analysis-{cam['id']}")
        self.cam = cam
        self.cam_id = cam["id"]
        self.cam_name = cam.get("name", cam["id"])
        self.model = model
        self.layer = layer            # mutated by the manager on switch
        self.created = time.time()
        self.last_poll = time.time()  # touched by every /frame poll
        self.stop_event = threading.Event()
        self.lock = threading.Lock()  # guards latest/seq/note
        self.latest: bytes | None = None
        # Structured snapshot for the canvas-overlay renderer. The
        # frontend fetches this every ~800 ms and draws boxes/heat/line
        # on a canvas positioned over the live iframe, so the video
        # stays 25 fps while the analysis overlay ticks at YOLO's pace.
        # Same PID as the poll handler, so a plain dict is safe.
        self.latest_data: dict | None = None
        self.seq = 0
        self.note = "starting stream..."
        self.err: str | None = None
        # Rolling state that SURVIVES layer switches (fix 2 point 9):
        self.tracker = None
        self.heat = [[0.0] * GRID_W for _ in range(GRID_H)]
        self.heat_since: float | None = None
        # User-drawn override (data/lines/<cam>.json) wins over cameras.py.
        # The line + its class filter are hot-reloaded on the fly (see
        # _maybe_reload_line) so redrawing while a session runs takes
        # effect within LINE_RELOAD_POLL_S seconds without restart.
        from app.cameras import resolve_line as _resolve_line
        from app.cameras import resolve_line_classes as _resolve_classes
        from app.cameras import resolve_zones as _resolve_zones
        self.line = _resolve_line(self.cam_id) or cam.get("line") or DEFAULT_LINE
        self.line_classes = _resolve_classes(self.cam_id)
        self._line_mtime = self._line_json_mtime()
        self._next_line_check = time.time() + LINE_RELOAD_POLL_S
        # User-drawn zones (loiter areas + parking spots) - same hot-reload
        # contract as the counting line.
        self.zones = _resolve_zones(self.cam_id)
        self._zones_mtime = self._zones_json_mtime()
        self._next_zones_check = time.time() + LINE_RELOAD_POLL_S
        self._zone_since: dict[tuple, float] = {}   # (tid, zone_idx) -> t0
        self.cross = {"in": 0, "out": 0}
        self._line_sides: dict[int, float] = {}
        self._last_cross_ts: dict[int, float] = {}
        self.gesture_counts: dict[str, int] = {}
        self._track_gestures: dict[int, set] = {}
        self._faces_ok: bool | None = None
        self._fail = 0
        self._last_tick: float | None = None

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> None:
        try:
            while (not self.stop_event.is_set()
                   and time.time() - self.last_poll < IDLE_STOP_S):
                t0 = time.time()
                frame = self._grab()
                if frame is None:
                    self._publish_note("stream unavailable - retrying...")
                    if self.stop_event.wait(2.0):
                        break
                    continue
                boxes = self._infer(frame)
                now = time.time()
                if self.tracker is None:
                    from app.tracker import BurstTracker
                    self.tracker = BurstTracker(frame.shape)
                self.tracker.update(boxes, now)
                self._trim()
                layer = self.layer
                if layer in ("pose", "gestures", "body"):
                    self._pose_pass(frame, boxes)
                faces_list: list[dict] = []
                if layer == "faces":
                    faces_list = self._faces_pass(frame)
                self._accumulate(frame, boxes, now)
                img = self._render(frame, faces_list, layer)
                self._publish(img)
                self._publish_data(frame.shape, boxes, layer, faces_list)
                dt = time.time() - t0
                wait = max(0.0, TICK_TARGET_S - dt)
                if wait and self.stop_event.wait(wait):
                    break
        except Exception as e:  # noqa: BLE001 - the session must not die silently
            self.err = f"{type(e).__name__}: {e}"
            self._publish_note(f"analysis stopped: {self.err}")
            print(f"live-analysis {self.cam_id}: crashed ({self.err})")
        # No reader cleanup here: readers live in the shared pool now and
        # idle-stop on their own when nothing snapshots them anymore.

    # -- pipeline stages ---------------------------------------------------

    def _grab(self):
        from app.detect_core import (HEADER_HOSTS, grab_frame,
                                     invalidate_stream, resolve_stream)
        try:
            url = resolve_stream(self.cam)
        except Exception:
            self._fail += 1
            return None
        # Header-required hosts (tvkur, ibb, skyline) can't ride a plain
        # persistent VideoCapture - every segment request needs Referer/
        # Origin headers - so they keep the old per-tick segment path.
        if any(h in url for h in HEADER_HOSTS):
            frame = grab_frame(url)
            if frame is None:
                self._fail += 1
                if self._fail % GRAB_FAIL_REFRESH == 0:
                    invalidate_stream(self.cam_id)
            else:
                self._fail = 0
                self._last_frame_ts = time.time()
            return frame
        try:
            r = get_shared_reader(self.cam, self.cam_id)
        except Exception:
            self._fail += 1
            return None
        if r is None:      # header host slipped through - segment path
            return grab_frame(url)
        frame = r.snapshot_wait(timeout=4.0)
        # A reader whose frames stopped aging forward is wedged (stalled
        # stream that still holds its last decode) - rebuild next tick.
        if frame is not None and time.time() - r.latest_ts > 10:
            r.stop()
            r.dead = True
            frame = None
        if frame is None:
            self._fail += 1
            if self._fail % GRAB_FAIL_REFRESH == 0:
                # Expired manifest / rotated token: force a fresh resolve.
                invalidate_stream(self.cam_id)
        else:
            self._fail = 0
            self._last_frame_ts = r.latest_ts
        return frame

    def _infer(self, frame) -> list[dict]:
        from app.detect_core import DEFAULT_PER_CLASS_CONF, filter_boxes_roi
        gates = dict(self.cam.get("per_class_conf") or DEFAULT_PER_CLASS_CONF)
        # All sessions funnel through the batcher: concurrent ticks share
        # one batched forward pass instead of queueing on INFER_LOCK.
        boxes = BATCHER.infer(self.model, frame,
                              self.cam.get("conf", 0.30), gates)
        if (self.cam.get("roi") or self.cam.get("roi_exclude")
                or self.cam.get("roi_exclude_class")):
            boxes = filter_boxes_roi(boxes, frame.shape, self.cam.get("roi"),
                                     self.cam.get("roi_exclude"),
                                     self.cam.get("roi_exclude_class"))
        return boxes

    def _pose_pass(self, frame, boxes) -> None:
        from app.pose import attach_keypoints_crops, load_pose_model
        with INFER_LOCK:
            attach_keypoints_crops(load_pose_model(), frame, boxes)

    def _faces_pass(self, frame) -> list[dict]:
        from app import faces as _faces
        if self._faces_ok is None:
            self._faces_ok = _faces.available()
        if not self._faces_ok:
            return []
        out = _faces.detect_faces(frame)
        # Night frames made the raw detector spray hundreds of noise
        # rects (measured 440 on one tick) - a conf floor, a minimum
        # size and a hard cap keep both the render and the JSON sane.
        out = [f for f in out
               if float(f.get("conf") or 0) >= 0.6
               and (f["x2"] - f["x1"]) >= 8 and (f["y2"] - f["y1"]) >= 8]
        out.sort(key=lambda f: -float(f.get("conf") or 0))
        return out[:32]

    def _zones_json_mtime(self) -> float | None:
        from app.cameras import _zones_dir
        p = _zones_dir() / f"{self.cam_id}.json"
        try:
            return p.stat().st_mtime
        except OSError:
            return None

    def _maybe_reload_zones(self, now: float) -> None:
        """Hot-reload user-drawn zones on the same cadence as the line.
        The dwell clocks restart on an edit (indices may have shifted);
        occupancy recovers within one tick."""
        if now < self._next_zones_check:
            return
        self._next_zones_check = now + LINE_RELOAD_POLL_S
        mtime = self._zones_json_mtime()
        if mtime == self._zones_mtime:
            return
        self._zones_mtime = mtime
        from app.cameras import resolve_zones as _resolve_zones
        self.zones = _resolve_zones(self.cam_id)
        self._zone_since.clear()

    def _line_json_mtime(self) -> float | None:
        """Current mtime of data/lines/<cam>.json, or None when the file
        does not exist. Used by _maybe_reload_line to detect a fresh save
        or a clear that happened while the session is running."""
        from app.cameras import _lines_dir
        p = _lines_dir() / f"{self.cam_id}.json"
        try:
            return p.stat().st_mtime
        except OSError:
            return None

    def _maybe_reload_line(self, now: float) -> None:
        """Hot-reload the line + class filter if the JSON has been
        rewritten (or removed) since the last check.

        Cadence is bounded by LINE_RELOAD_POLL_S so this is at most one
        stat call every few seconds - cheap next to the inference tick.
        When the line moves, side_state is dropped so stale side
        observations from the OLD line can't fabricate a fake crossing
        against the NEW line; the counter itself is preserved (session
        totals keep accumulating across edits). The per-tid cooldown
        map is dropped too - it's per-line by design."""
        if now < self._next_line_check:
            return
        self._next_line_check = now + LINE_RELOAD_POLL_S
        mtime = self._line_json_mtime()
        if mtime == self._line_mtime:
            return
        self._line_mtime = mtime
        from app.cameras import (resolve_line as _resolve_line,
                                 resolve_line_classes as _resolve_classes)
        new_line = (_resolve_line(self.cam_id)
                    or self.cam.get("line") or DEFAULT_LINE)
        new_classes = _resolve_classes(self.cam_id)
        if new_line != self.line or new_classes != self.line_classes:
            self.line = new_line
            self.line_classes = new_classes
            self._line_sides.clear()
            self._last_cross_ts.clear()

    def _accumulate(self, frame, boxes, now: float) -> None:
        # First tick has no prior timestamp to measure against, so it
        # borrows the pacing target instead of the old arbitrary 1.0 - the
        # boot sample now weighs about as much as a normal tick (TICK_TARGET_S
        # is the pacing floor the run loop already sleeps to). Later ticks
        # use the real elapsed time, clamped so a long stall doesn't inflate
        # one bin.
        frame_shape = frame.shape
        if self._last_tick is None:
            w = TICK_TARGET_S
        else:
            w = min(5.0, max(0.2, now - self._last_tick))
        self._last_tick = now
        if self.heat_since is None:
            self.heat_since = now
        bump_heat(self.heat, boxes, frame_shape, w)
        self._maybe_reload_line(now)
        self._maybe_reload_zones(now)
        # Persist an event per crossing (bounded JSONL + optional snap).
        # The dashboard's Line layer polls /api/crossings?cam=<id> for the
        # toast + history strip. Frame is passed so the crop of the mover
        # captures the moment they crossed.
        def _on_cross(direction, track, frame, cam_id):
            log_crossing_event(cam_id, direction, track, frame=frame)
        update_crossings(self._line_sides, self.tracker.open, frame_shape,
                         self.line, self.cross,
                         on_event=_on_cross, frame=frame,
                         cam_id=self.cam_id,
                         classes=self.line_classes,
                         last_cross_ts=self._last_cross_ts, now=now)

    def _trim(self) -> None:
        for tr in self.tracker.open:
            if len(tr.boxes) > TRACK_KEEP:
                del tr.boxes[:-TRACK_KEEP]
                del tr.times[:-TRACK_KEEP]
        # Retired tracks are never revisited live - drop them.
        if self.tracker.done:
            self.tracker.done.clear()
        if len(self._line_sides) > 256 or len(self._track_gestures) > 256:
            keep = {tr.tid for tr in self.tracker.open}
            for store in (self._line_sides, self._track_gestures):
                for k in list(store):
                    if k not in keep:
                        store.pop(k, None)

    def _render(self, frame, faces_list: list[dict], layer: str):
        img = frame.copy()
        open_now = list(self.tracker.open)
        visible = [tr.boxes[-1] for tr in open_now if not tr.misses]
        stats_by_id: dict[int, dict] = {}
        if layer in ("paths", "gestures", "body"):
            from app.behavior import track_stats
            for tr in open_now:
                if tr.misses:
                    continue
                row = track_stats(tr.cls, tr.boxes, tr.times, frame.shape)
                row["id"] = tr.tid
                if layer in ("gestures", "body"):
                    from app.behavior_labels import label_track
                    from app.gestures import detect_gestures
                    kseq = [b.get("kps") for b in tr.boxes[-16:]]
                    has_kps = any(kseq)
                    row.update(label_track(row, frame.shape,
                                           kseq if has_kps else None))
                    row["gestures"] = detect_gestures(kseq) if has_kps else []
                    for g in row["gestures"]:
                        seen = self._track_gestures.setdefault(tr.tid, set())
                        if g not in seen:
                            seen.add(g)
                            self.gesture_counts[g] = \
                                self.gesture_counts.get(g, 0) + 1
                stats_by_id[tr.tid] = row
        if layer == "paths":
            return draw_paths_layer(img, open_now, visible, stats_by_id)
        if layer == "pose":
            return draw_pose_layer(img, visible)
        if layer == "gestures":
            return draw_gestures_layer(img, visible, stats_by_id,
                                       self.gesture_counts)
        if layer == "body":
            return draw_body_layer(img, visible, stats_by_id)
        if layer == "faces":
            return draw_faces_layer_img(img, faces_list,
                                        available=bool(self._faces_ok))
        if layer == "heat":
            return draw_heat_layer(img, self.heat, since=self.heat_since)
        if layer == "line":
            return draw_line_layer(img, self.line, self.cross)
        if layer in ("loiter", "parking"):
            lo, pk, _dwell = self._zone_stats(frame.shape)
            return draw_zones_layer(img, lo if layer == "loiter" else pk,
                                    layer)
        return img

    def _zone_stats(self, frame_shape):
        """Occupancy + dwell for loiter zones, occupancy for parking spots,
        computed from the tracker's confirmed tracks. Cached per tick
        (keyed on the frame's capture stamp) so the JPEG render and the
        JSON publish share ONE computation instead of walking every
        track against every polygon twice."""
        key = getattr(self, "_last_frame_ts", None)
        if key is not None and getattr(self, "_zone_cache_key", None) == key:
            return self._zone_cache
        H, W = int(frame_shape[0]), int(frame_shape[1])
        now_t = time.time()
        loiter, parking = [], []
        for zi, z in enumerate(self.zones or []):
            entry = {"name": z.get("name") or f"Z{zi + 1}",
                     "points": z["points"]}
            if z.get("kind") == "parking":
                entry.update(occupied=False, cls=None)
                parking.append((zi, entry))
            else:
                entry.update(count=0, max_dwell=0.0, alert=False,
                             dwell_s=float(z.get("dwell_s")
                                           or DEFAULT_LOITER_DWELL_S))
                loiter.append((zi, entry))
        dwell_by_tid: dict[int, float] = {}
        active: set[tuple] = set()
        for tr in (self.tracker.open if self.tracker else []):
            if tr.misses > DISPLAY_MAX_MISSES or tr.hits < DISPLAY_MIN_HITS:
                continue
            b = tr.boxes[-1]
            if tr.cls == "person" and loiter:
                cx = ((b["x1"] + b["x2"]) / 2) / W
                by = b["y2"] / H            # feet, not head
                for zi, e in loiter:
                    if _pt_in_poly(cx, by, e["points"]):
                        key = (tr.tid, zi)
                        active.add(key)
                        dw = now_t - self._zone_since.setdefault(key, now_t)
                        e["count"] += 1
                        e["max_dwell"] = max(e["max_dwell"], dw)
                        if dw >= e["dwell_s"]:
                            e["alert"] = True
                        dwell_by_tid[tr.tid] = max(
                            dwell_by_tid.get(tr.tid, 0.0), dw)
            if tr.cls in _VEHICLE_CLASSES and parking:
                cx = ((b["x1"] + b["x2"]) / 2) / W
                cy = ((b["y1"] + b["y2"]) / 2) / H
                for zi, e in parking:
                    if not e["occupied"] and _pt_in_poly(cx, cy,
                                                         e["points"]):
                        e["occupied"] = True
                        e["cls"] = tr.cls
        # Drop clocks of tracks that left their zone (or died) so a
        # RETURN starts a fresh dwell instead of resuming the old one.
        self._zone_since = {k: v for k, v in self._zone_since.items()
                            if k in active}
        result = ([e for _, e in loiter], [e for _, e in parking],
                  dwell_by_tid)
        self._zone_cache_key = key
        self._zone_cache = result
        return result

    def _publish(self, img) -> None:
        import cv2
        H, W = img.shape[:2]
        if W > JPEG_MAX_W:
            img = cv2.resize(img, (JPEG_MAX_W, int(H * JPEG_MAX_W / W)),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with self.lock:
                self.latest = buf.tobytes()
                self.seq += 1
                self.note = ""

    def _publish_note(self, note: str) -> None:
        with self.lock:
            self.note = note

    def _publish_data(self, frame_shape, boxes, layer: str,
                      faces_list: list[dict] | None = None) -> None:
        """Snapshot the just-inferred tick as JSON for the overlay canvas.

        Boxes come from the TRACKER, not the raw detections: only objects
        confirmed across DISPLAY_MIN_HITS ticks are published, each with
        its track id and centroid velocity (px/s) so the client can
        extrapolate positions between ticks and glide boxes with the
        video instead of letting them sit on vacated pixels. `at` is the
        capture time translated into the stream's PROGRAM-DATE-TIME clock
        (when known), which is the clock hls.js reports for the frame the
        operator is currently watching.
        """
        H, W = int(frame_shape[0]), int(frame_shape[1])
        js_boxes = []
        for tr in (self.tracker.open if self.tracker else []):
            last = tr.boxes[-1]
            conf = max(float(b.get("conf") or 0) for b in tr.boxes[-2:])
            if (tr.hits < DISPLAY_MIN_HITS
                    or tr.misses > DISPLAY_MAX_MISSES
                    or conf < DISPLAY_MIN_CONF
                    or (tr.cls or "?") in DISPLAY_CLASS_BLACKLIST):
                continue
            jb = {
                "tid": tr.tid,
                "x1": int(last.get("x1", 0)),
                "y1": int(last.get("y1", 0)),
                "x2": int(last.get("x2", 0)),
                "y2": int(last.get("y2", 0)),
                "cls": tr.cls or "?",
                "conf": round(conf, 3),
                "vx": round(float(tr.vx), 1),
                "vy": round(float(tr.vy), 1),
            }
            # Layer-specific extras ride on each box so the canvas can
            # draw the REAL layer, not just generic rectangles - this
            # was the "analysis is not logically right" gap: trails,
            # skeletons, gesture chips and anomaly flags existed only
            # inside the server-rendered JPEG.
            if layer == "paths":
                jb["trail"] = [
                    [int((b["x1"] + b["x2"]) / 2),
                     int((b["y1"] + b["y2"]) / 2)]
                    for b in tr.boxes[-12:]]
                try:
                    from app.behavior import track_stats
                    row = track_stats(tr.cls, tr.boxes, tr.times,
                                      frame_shape)
                    if row.get("kmh_est"):
                        jb["kmh"] = row["kmh_est"]
                except Exception:
                    pass
            elif layer in ("pose", "gestures", "body"):
                kps = last.get("kps")
                if kps:
                    jb["kps"] = [[int(k[0]), int(k[1]), round(k[2], 2)]
                                 for k in kps]
                if layer == "gestures" and tr.cls == "person":
                    try:
                        from app.gestures import detect_gestures
                        kseq = [b.get("kps") for b in tr.boxes[-16:]]
                        if any(kseq):
                            g = detect_gestures(kseq)
                            if g:
                                jb["gestures"] = g
                    except Exception:
                        pass
                if layer == "body" and tr.cls == "person":
                    try:
                        from app.behavior import track_stats
                        from app.behavior_labels import label_track
                        row = track_stats(tr.cls, tr.boxes, tr.times,
                                          frame_shape)
                        kseq = [b.get("kps") for b in tr.boxes[-16:]]
                        row.update(label_track(row, frame_shape,
                                               kseq if any(kseq) else None))
                        if (row.get("label") in BODY_ANOMALY_LABELS
                                or row.get("pose_flags")):
                            jb["flag"] = row.get("label") or ""
                            jb["alert"] = bool(row.get("alert"))
                            flags = [f for f in (row.get("pose_flags") or [])
                                     if f and f != row.get("label")]
                            if flags:
                                jb["flags"] = flags
                    except Exception:
                        pass
            js_boxes.append(jb)
        cap_ts = getattr(self, "_last_frame_ts", None) or time.time()
        pdt_off = STREAM_PDT_OFFSET.get(self.cam_id, 3.0)
        data: dict = {
            "seq": self.seq + 1,        # matches _publish's post-bump seq
            "layer": layer,
            "frame_w": W,
            "frame_h": H,
            # capture time on the video's own clock; clamp the measured
            # ingest offset to something sane so one bad manifest parse
            # can't shove every box seconds off.
            "at": round(cap_ts - min(15.0, max(0.0, pdt_off)), 3),
            "boxes": js_boxes,
            "person": sum(1 for b in (boxes or [])
                          if b.get("cls") == "person"),
            "vehicles": sum(1 for b in (boxes or [])
                            if b.get("cls") in ("car", "truck", "bus",
                                                "motorcycle", "bicycle")),
        }
        if layer == "heat":
            data["heat"] = self.heat
        if layer == "line":
            data["line"] = self.line
            data["cross"] = dict(self.cross)
        if layer == "gestures" and self.gesture_counts:
            data["gesture_counts"] = dict(self.gesture_counts)
        if layer in ("loiter", "parking"):
            lo, pk, dwell_by_tid = self._zone_stats(frame_shape)
            if layer == "loiter":
                data["zones"] = [{**e, "max_dwell": int(e["max_dwell"])}
                                 for e in lo]
                for jb in js_boxes:
                    if (jb["cls"] == "person"
                            and jb["tid"] in dwell_by_tid):
                        jb["dwell"] = int(dwell_by_tid[jb["tid"]])
            else:
                data["spots"] = pk
                data["parking"] = {
                    "total": len(pk),
                    "occupied": sum(1 for e in pk if e["occupied"])}
        if layer == "faces":
            data["faces"] = [
                {"x1": int(f["x1"]), "y1": int(f["y1"]),
                 "x2": int(f["x2"]), "y2": int(f["y2"]),
                 "conf": round(float(f.get("conf") or 0), 2)}
                for f in (faces_list or [])]
            data["faces_ok"] = bool(self._faces_ok)
        with self.lock:
            self.latest_data = data


# ---------------------------------------------------------------------------
# The manager the dashboard endpoints talk to.
# ---------------------------------------------------------------------------

class LiveAnalysisManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: dict[str, LiveSession] = {}
        # Last crash reason per cam_id, kept until the NEXT frame() poll so
        # the client can render "analysis stopped: <reason>" instead of the
        # bare 404 the old reap loop returned. Bounded to MAX_SESSIONS
        # entries by _remember_error_locked (one per possible camera slot).
        self._errors: dict[str, str] = {}

    def start(self, cam_id: str, layer: str, model) -> dict:
        """Start a session for `cam_id`, or switch the layer of a running
        one (stream + accumulators survive the switch)."""
        if layer not in LIVE_LAYERS:
            raise ValueError(f"unknown layer {layer!r}")
        cam = resolve_cam(cam_id)     # raises ValueError on unknown ids
        with self._lock:
            self._reap_locked()
            # A fresh start clears any stale error remembered from the
            # previous session on this camera.
            self._errors.pop(cam_id, None)
            s = self._sessions.get(cam_id)
            if s is not None and s.is_alive():
                switched = s.layer != layer
                s.layer = layer
                s.last_poll = time.time()
                return {"cam": cam_id, "cam_name": s.cam_name,
                        "layer": layer, "switched": switched,
                        "active": len(self._sessions)}
            if len(self._sessions) >= MAX_SESSIONS:
                raise BusyError(
                    f"{MAX_SESSIONS} live analyses already running - "
                    f"stop one first")
            s = LiveSession(cam, model, layer)
            self._sessions[cam_id] = s
            s.start()
            return {"cam": cam_id, "cam_name": s.cam_name, "layer": layer,
                    "switched": False, "active": len(self._sessions)}

    def frame(self, cam_id: str) -> dict | None:
        """Latest JPEG + metadata, or None when no session runs. Every
        call refreshes the idle clock. A session that crashed is popped
        AND its error is returned once via {"error": "..."} so the client
        sees WHY analysis stopped instead of a bare 404."""
        with self._lock:
            s = self._sessions.get(cam_id)
            if s is None:
                err = self._errors.pop(cam_id, None)
                return {"error": err} if err else None
            if not s.is_alive():
                self._sessions.pop(cam_id, None)
                self._remember_error_locked(cam_id, s.err
                                            or "session ended unexpectedly")
                return {"error": self._errors.pop(cam_id, None)}
        s.last_poll = time.time()
        with s.lock:
            return {"jpeg": s.latest, "seq": s.seq, "layer": s.layer,
                    "note": s.note}

    def any_alive(self) -> bool:
        """True while at least one session thread is actually running.
        local_producers reads this to yield the CPU during analysis;
        thread state (not a bookkeeping set) means an idle-timed-out
        session releases the pause automatically."""
        with self._lock:
            return any(s.is_alive() for s in self._sessions.values())

    def data(self, cam_id: str) -> dict | None:
        """Same idle-clock refresh as frame(), but returns the JSON snapshot
        used by the canvas-overlay renderer instead of the annotated JPEG."""
        with self._lock:
            s = self._sessions.get(cam_id)
            if s is None:
                err = self._errors.pop(cam_id, None)
                return {"error": err} if err else None
            if not s.is_alive():
                self._sessions.pop(cam_id, None)
                self._remember_error_locked(cam_id, s.err
                                            or "session ended unexpectedly")
                return {"error": self._errors.pop(cam_id, None)}
        s.last_poll = time.time()
        with s.lock:
            return {"data": s.latest_data, "seq": s.seq, "layer": s.layer,
                    "note": s.note}

    def stop(self, cam_id: str) -> bool:
        with self._lock:
            s = self._sessions.pop(cam_id, None)
            # An operator-initiated stop is not a crash; drop any pending
            # error so the next start on this camera is a clean slate.
            self._errors.pop(cam_id, None)
        if s is None:
            return False
        s.stop_event.set()
        return True

    def stop_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._errors.clear()
        for s in sessions:
            s.stop_event.set()

    def _reap_locked(self) -> None:
        for cam_id in [c for c, s in self._sessions.items()
                       if not s.is_alive()]:
            s = self._sessions.pop(cam_id, None)
            if s is not None:
                self._remember_error_locked(
                    cam_id, getattr(s, "err", None) or "session ended unexpectedly")

    def _remember_error_locked(self, cam_id: str, err: str) -> None:
        """Cap the error dict at MAX_SESSIONS entries (one per possible
        camera slot) so a runaway crash loop can never grow it unbounded."""
        self._errors[cam_id] = err
        if len(self._errors) > MAX_SESSIONS:
            # FIFO eviction: pop the oldest remembered error.
            oldest = next(iter(self._errors))
            if oldest != cam_id:
                self._errors.pop(oldest, None)


MANAGER = LiveAnalysisManager()
