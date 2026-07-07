"""Shared detection + stream-access core.

Imported by the notebook, the collector daemon, and the Streamlit app so the
detection logic lives in exactly one place.
"""
from __future__ import annotations

import os
import re
import ssl
import tempfile
import urllib.request

import cv2
import numpy as np

# COCO class ids we care about for *business activity* (footfall + vehicles
# + rail). `train` was added after a metro train crossing the frame went
# unlabeled - the model classifies it as class 6, but without id 6 in the
# `classes=` filter YOLO silently drops it before we ever see the box.
CLASSES_OF_INTEREST = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "train": 6,
    "truck": 7,
}
NAME_BY_ID = {v: k for k, v in CLASSES_OF_INTEREST.items()}
# `train` is intentionally excluded from `vehicles`: a metro/tram flows at a
# completely different rate than road traffic and mixing them would corrupt
# the per-camera baselines. It shows up as its own count on cameras that
# look at rail.
VEHICLE_NAMES = ("bicycle", "car", "motorcycle", "bus", "truck")

# Model input size the collector runs at. YOLO's 640 default shrinks a distant
# pedestrian on these wide street shots to a handful of pixels and the model
# undercounts badly (3 visible cars -> 1 detected). 960 costs ~2.3x the CPU
# time per frame - still a fraction of the collector's sampling interval - and
# recovers most of the small/far objects. Pass imgsz=None to use the model's
# own default (the notebook's quick cells do that).
DEFAULT_IMGSZ = 960


def load_model(weights: str = "yolov8s.pt"):
    """Load a YOLO model once and reuse it.

    Default is `yolov8s` (small) rather than `yolov8n` (nano). Nano's
    recall on the wide overhead street views these cameras produce is too
    low: it silently drops distant/static vehicles, mis-fires `person` on
    upright thin road furniture, and often mis-classifies a partially-cropped
    car at the frame edge as `bicycle`. Small is the smallest tier where those
    three failure modes back off to acceptable levels. CPU cost is ~3x nano
    per burst - still a fraction of the collector's sampling interval.
    """
    from ultralytics import YOLO

    return YOLO(weights)


def resolve_youtube(url: str) -> str:
    """Resolve a YouTube Live (or webcamera24 YouTube-backed) page to an HLS .m3u8 URL."""
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "format": "best[protocol^=m3u8]/best"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info["url"]


# Browser-ish headers: webcamera24 and skylinewebcams both 403 bare urllib fetchers.
_BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

# skylinewebcams page -> the tokenized HLS it points at (token rotates, so resolve live).
_SKYLINE_RE = re.compile(r'(?:source|src)\s*[:=]\s*["\']([^"\']*?live[^"\']*?\.m3u8[^"\']*)["\']',
                         re.IGNORECASE)
_SKYLINE_HOST = "https://hd-auth.skylinewebcams.com/"

# webcamera24 pages embed a tvkur player; pull its id and build the master playlist.
_TVKUR_ID_RE = re.compile(r'(?:player\.tvkur\.com/l/|content\.tvkur\.com/l/)([a-z0-9]+)',
                          re.IGNORECASE)
_YOUTUBE_RE = re.compile(r'(?:youtube\.com/(?:embed/|watch\?v=)|youtu\.be/)([\w-]{11})')


def resolve_skyline(page_url: str) -> str:
    """Resolve a skylinewebcams.com webcam page to its tokenized HLS .m3u8 URL.

    The page embeds the playlist as `source:"livee.m3u8?a=<token>"` (relative to
    hd-auth.skylinewebcams.com). The token rotates, so call this each cycle.
    """
    html = _http_get(page_url, _BROWSER_HEADERS).decode("utf-8", "replace")
    m = _SKYLINE_RE.search(html)
    if not m:
        raise RuntimeError("skyline: no HLS source found on page (layout changed or geo-blocked)")
    src = m.group(1)
    if src.startswith("http"):
        return src
    return _SKYLINE_HOST + src.lstrip("/")


def resolve_webcamera24(page_url: str) -> str:
    """Resolve a webcamera24.com page to an HLS URL via its embedded tvkur/YouTube player."""
    html = _http_get(page_url, _BROWSER_HEADERS).decode("utf-8", "replace")
    m = _TVKUR_ID_RE.search(html)
    if m:
        return f"https://content.tvkur.com/l/{m.group(1)}/master.m3u8"
    y = _YOUTUBE_RE.search(html)
    if y:
        return resolve_youtube(f"https://www.youtube.com/watch?v={y.group(1)}")
    raise RuntimeError("webcamera24: no tvkur/YouTube player found on page")


def resolve_stream(cam: dict) -> str:
    """Resolve any catalog camera dict to a directly-openable stream URL by `kind`.

    Direct HLS is returned as-is; YouTube/skyline/webcamera24 pages are resolved live.
    """
    kind = cam.get("kind", "hls")
    url = cam["url"]
    if kind == "hls":
        return url
    if kind == "youtube":
        return resolve_youtube(url)
    if kind == "skyline":
        return resolve_skyline(cam.get("page", url))
    if kind == "webcamera24":
        return resolve_webcamera24(cam.get("page", url))
    raise ValueError(f"unknown camera kind: {kind!r}")


_SSL_CTX = ssl._create_unverified_context()

# Some live-CDN HLS endpoints (e.g. content.tvkur.com) require a Referer/Origin header
# that ffmpeg-via-cv2 can't always pass on Windows. For those hosts we fetch the latest
# .ts segment manually and decode locally.
HEADER_HOSTS = {
    "content.tvkur.com":          {"Referer": "https://player.tvkur.com/",
                                   "Origin":  "https://player.tvkur.com"},
    "livestream.ibb.gov.tr":      {"Referer": "https://istanbuluseyret.ibb.gov.tr/",
                                   "Origin":  "https://istanbuluseyret.ibb.gov.tr"},
    "kamerayayin.ibb.istanbul":   {"Referer": "https://istanbuluseyret.ibb.gov.tr/",
                                   "Origin":  "https://istanbuluseyret.ibb.gov.tr"},
    "skylinewebcams.com":         {"Referer": "https://www.skylinewebcams.com/",
                                   "Origin":  "https://www.skylinewebcams.com"},
}

def _http_get(url: str, extra_headers: dict | None = None) -> bytes:
    h = {"User-Agent": "Mozilla/5.0"}
    if extra_headers:
        h.update(extra_headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as r:
        return r.read()

def _grab_via_segment(stream_url: str, headers: dict) -> np.ndarray | None:
    """Download the most recent .ts segment with the right headers and decode it."""
    base = stream_url.rsplit("/", 1)[0] + "/"
    pl = _http_get(stream_url, headers).decode("utf-8", "replace")
    if "#EXT-X-STREAM-INF" in pl:
        variant = next((l.strip() for l in pl.splitlines()
                        if l.strip() and not l.startswith("#")), None)
        if not variant:
            return None
        variant_url = variant if variant.startswith("http") else base + variant
        pl = _http_get(variant_url, headers).decode("utf-8", "replace")
        base = variant_url.rsplit("/", 1)[0] + "/"
    segs = [l.strip() for l in pl.splitlines() if l.strip() and not l.startswith("#")]
    if not segs:
        return None
    seg = segs[-1]
    seg_url = seg if seg.startswith("http") else base + seg
    data = _http_get(seg_url, headers)
    with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as f:
        f.write(data); tmp = f.name
    try:
        cap = cv2.VideoCapture(tmp)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    finally:
        try: os.unlink(tmp)
        except OSError: pass


def grab_frame(stream_url: str):
    """Open an HLS/RTSP stream, read a single frame (BGR ndarray), close. None on failure.

    For hosts that need referer/origin headers, route via _grab_via_segment.
    """
    for host, headers in HEADER_HOSTS.items():
        if host in stream_url:
            try:
                return _grab_via_segment(stream_url, headers)
            except Exception:
                return None
    cap = cv2.VideoCapture(stream_url)
    try:
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def iter_frames(stream_url: str, max_frames: int):
    """Yield up to `max_frames` consecutive frames from a live HLS stream.

    For header-required hosts (content.tvkur.com, livestream.ibb.gov.tr, skylinewebcams.com)
    cv2.VideoCapture(url) can't pass Referer/Origin on Windows, so we download the latest
    few .ts segments with the right headers and decode them locally - yielding frames in
    arrival order. For normal HLS we open the URL directly with cv2 and read.

    Used by the dwell-time / tracking section of the notebook so ByteTrack can see the
    consecutive frames it needs.
    """
    # header-required host: fetch enough segments to cover max_frames at ~25-30 fps
    matching_headers = None
    for host, headers in HEADER_HOSTS.items():
        if host in stream_url:
            matching_headers = headers
            break

    if matching_headers is not None:
        base = stream_url.rsplit("/", 1)[0] + "/"
        try:
            pl = _http_get(stream_url, matching_headers).decode("utf-8", "replace")
        except Exception:
            return
        if "#EXT-X-STREAM-INF" in pl:
            variant = next((l.strip() for l in pl.splitlines()
                            if l.strip() and not l.startswith("#")), None)
            if not variant:
                return
            variant_url = variant if variant.startswith("http") else base + variant
            try:
                pl = _http_get(variant_url, matching_headers).decode("utf-8", "replace")
            except Exception:
                return
            base = variant_url.rsplit("/", 1)[0] + "/"

        segs = [l.strip() for l in pl.splitlines() if l.strip() and not l.startswith("#")]
        if not segs:
            return
        # tail segments give the freshest live view; pull ~enough to cover the request
        approx_frames_per_seg = 60   # 2 s @ 30 fps is a typical segment
        n_segs = max(1, min(len(segs), -(-max_frames // approx_frames_per_seg)))
        yielded = 0
        for seg in segs[-n_segs:]:
            if yielded >= max_frames:
                break
            seg_url = seg if seg.startswith("http") else base + seg
            try:
                data = _http_get(seg_url, matching_headers)
            except Exception:
                continue
            with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as f:
                f.write(data); tmp = f.name
            try:
                cap = cv2.VideoCapture(tmp)
                while yielded < max_frames:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    yielded += 1
                    yield fr
                cap.release()
            finally:
                try: os.unlink(tmp)
                except OSError: pass
        return

    # normal HLS / RTSP: stream directly
    cap = cv2.VideoCapture(stream_url)
    yielded = 0
    try:
        while yielded < max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            yielded += 1
            yield fr
    finally:
        cap.release()


# ---- Region-of-interest (ROI) filtering --------------------------------------
# A camera entry may carry a "roi" polygon (and/or "roi_exclude" polygons) in
# NORMALIZED coordinates (0..1 relative to frame width/height), so one config
# works across stream resolutions. A detection belongs to the ROI when its
# FOOT POINT - bottom-center of the box, where the object touches the ground -
# is inside the polygon. That excludes parked-car lots, sky, and neighboring
# roofs without clipping pedestrians whose heads poke outside the zone.

def point_in_polygon(x: float, y: float, poly: list) -> bool:
    """Ray-casting test; poly is [[x1,y1], [x2,y2], ...] in any unit."""
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _foot_point(b: dict) -> tuple[float, float]:
    return (b["x1"] + b["x2"]) / 2.0, b["y2"]


def filter_boxes_roi(boxes: list[dict], frame_shape,
                     roi: list | None,
                     roi_exclude: list | None = None,
                     roi_exclude_class: dict | None = None) -> list[dict]:
    """Keep boxes whose foot point is inside `roi` (if set) and outside every
    polygon in `roi_exclude`. Polygons use normalized 0..1 coordinates.

    `roi_exclude_class` is per-class: `{cls_name: [poly, poly, ...]}`. A box
    is dropped when its foot point falls inside ANY polygon listed for its
    own class. This lets a camera's config say "never accept `person` in the
    top-left corner (there's a lamp post there)" without hiding real cars
    that pass through the same pixels.
    """
    if not roi and not roi_exclude and not roi_exclude_class:
        return boxes
    H, W = frame_shape[:2]
    kept = []
    for b in boxes:
        fx, fy = _foot_point(b)
        nx, ny = fx / W, fy / H
        if roi and not point_in_polygon(nx, ny, roi):
            continue
        if roi_exclude and any(point_in_polygon(nx, ny, p) for p in roi_exclude):
            continue
        if roi_exclude_class:
            polys = roi_exclude_class.get(b.get("cls")) or ()
            if any(point_in_polygon(nx, ny, p) for p in polys):
                continue
        kept.append(b)
    return kept


def counts_from_boxes(boxes: list[dict]) -> dict:
    """Recompute the per-class count dict (incl. 'vehicles') from a box list."""
    counts = {name: 0 for name in CLASSES_OF_INTEREST}
    for b in boxes:
        if b.get("cls") in counts:
            counts[b["cls"]] += 1
    counts["vehicles"] = sum(counts[v] for v in VEHICLE_NAMES)
    return counts


# ---- Burst tracking + virtual-line crossing -----------------------------------
# The burst gives a short consecutive window (~n frames, ~1s apart). Matching
# detections across those frames by nearest centroid yields short tracks; a
# camera with a configured "line" ([[x1,y1],[x2,y2]] normalized) then counts
# how many tracks CROSSED it, and in which direction. Because the collector
# only observes ~2-3s out of every interval, the numbers are a SAMPLED flow
# rate - comparable over time on the same camera, not an absolute turnstile.

def _centroid(b: dict) -> tuple[float, float]:
    return (b["x1"] + b["x2"]) / 2.0, (b["y1"] + b["y2"]) / 2.0


def track_burst(frames_boxes: list[list[dict]], frame_shape,
                max_move_frac: float = 0.12) -> list[list[dict]]:
    """Greedy nearest-centroid matching of boxes across burst frames.

    Returns tracks: each a list of same-class boxes, one per frame the object
    was matched in (consecutive frames only - a miss ends the track).
    `max_move_frac` caps the allowed centroid move between burst frames as a
    fraction of the frame diagonal (objects teleporting further are different
    objects).
    """
    if not frames_boxes:
        return []
    H, W = frame_shape[:2]
    budget = max_move_frac * (H * H + W * W) ** 0.5
    tracks: list[list[dict]] = [[b] for b in frames_boxes[0]]
    open_tracks = list(tracks)
    for boxes in frames_boxes[1:]:
        candidates = []
        for ti, t in enumerate(open_tracks):
            tx, ty = _centroid(t[-1])
            for bi, b in enumerate(boxes):
                if b["cls"] != t[-1]["cls"]:
                    continue
                bx, by = _centroid(b)
                d = ((bx - tx) ** 2 + (by - ty) ** 2) ** 0.5
                if d <= budget:
                    candidates.append((d, ti, bi))
        candidates.sort()
        used_t, used_b = set(), set()
        next_open: list[list[dict]] = []
        for d, ti, bi in candidates:
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti); used_b.add(bi)
            open_tracks[ti].append(boxes[bi])
            next_open.append(open_tracks[ti])
        for bi, b in enumerate(boxes):
            if bi not in used_b:
                t = [b]
                tracks.append(t)
                next_open.append(t)
        open_tracks = next_open
    return tracks


def _line_side(px: float, py: float, line: list) -> float:
    """Signed side of point vs the line A->B (cross product z)."""
    (ax, ay), (bx, by) = line
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def count_line_crossings(tracks: list[list[dict]], frame_shape,
                         line: list) -> dict:
    """Count tracks whose FOOT POINT crossed the normalized line during the
    burst. Returns {"in": n, "out": n, "person_in": ..., "vehicles_in": ...}.
    "in" is a crossing from the negative to the positive side of A->B (pick
    the line's point order so that "in" means into your area of interest).
    """
    H, W = frame_shape[:2]
    res = {"in": 0, "out": 0,
           "person_in": 0, "person_out": 0,
           "vehicles_in": 0, "vehicles_out": 0}
    for t in tracks:
        if len(t) < 2:
            continue
        sides = []
        for b in t:
            fx, fy = _foot_point(b)
            sides.append(_line_side(fx / W, fy / H, line))
        crossed = None
        for s0, s1 in zip(sides, sides[1:]):
            if s0 < 0 <= s1:
                crossed = "in"
            elif s0 >= 0 > s1:
                crossed = "out"
        if crossed is None:
            continue
        res[crossed] += 1
        metric = "person" if t[0].get("cls") == "person" else "vehicles"
        res[f"{metric}_{crossed}"] += 1
    return res


# Per-class confidence gate applied AFTER the model's own conf filter.
# `person`, `car`, `bus`, `truck` at 0.35 keep the model honest on the classes
# it's usually confident about. `train` sits at 0.25 because a partial-view
# tram or metro car at street-camera angles rarely lands above 0.35 in
# practice, and losing it entirely (as the user reported) is a worse failure
# than the occasional low-confidence false positive at that class.
DEFAULT_PER_CLASS_CONF = {
    "person":     0.35,
    "bicycle":    0.22,
    "motorcycle": 0.22,
    "car":        0.35,
    "bus":        0.35,
    "train":      0.25,
    "truck":      0.35,
}

# Person shape / size gates. A real pedestrian on a street cam is TALLER
# than wide (aspect >= 0.90) but NOT wildly so (aspect <= 3.0); has at
# least a couple of dozen pixels of vertical extent; and never spans more
# than a fraction of the frame's width (a "person" box wider than that is
# almost always the model misfiring on a metro car, a bus at close range,
# or a large signage board mistaken for a human).
#
# Values below were pulled in after user reports of
#   * a metro / light-rail car (spanning ~half the frame) labeled `person`;
#   * a separator pole (very narrow, tall) labeled `person` at conf 0.4;
#   * false positives on distant road furniture that clear MIN_ASPECT but
#     look nothing like a person.
DEFAULT_PERSON_MIN_ASPECT = 0.90
DEFAULT_PERSON_MAX_ASPECT = 3.0
DEFAULT_PERSON_MIN_HEIGHT_PX = 24     # smaller than this and there's no
                                       # meaningful person signal anyway
DEFAULT_PERSON_MAX_WIDTH_FRAC = 0.30   # any "person" wider than 30% of
                                       # frame width is misfiring

# "Rider co-detection": if a person box overlaps a two-wheeler that YOLO
# reported ABOVE its own gate but BELOW the per-class gate, resurrect the
# two-wheeler - a person on a motorcycle is a rider AND a vehicle, but the
# nano model often reports the person confidently and the vehicle just below
# threshold, so counting only the person under-reports vehicle traffic.
DEFAULT_RIDER_IOU = 0.30
_TWO_WHEELERS = ("bicycle", "motorcycle")


def _box_wh(b: dict) -> tuple[float, float]:
    return b["x2"] - b["x1"], b["y2"] - b["y1"]


def detect_and_count(model, frame, conf: float = 0.35, imgsz: int | None = None) -> dict:
    """Run YOLO on one frame -> {class_name: count} for the classes we track."""
    counts, _ = detect_with_boxes(model, frame, conf=conf, imgsz=imgsz)
    return counts


def detect_with_boxes(model, frame, conf: float = 0.35,
                      imgsz: int | None = None,
                      per_class_conf: dict | None = None,
                      person_min_aspect: float | None = DEFAULT_PERSON_MIN_ASPECT,
                      person_max_aspect: float | None = DEFAULT_PERSON_MAX_ASPECT,
                      person_min_height_px: int | None = DEFAULT_PERSON_MIN_HEIGHT_PX,
                      person_max_width_frac: float | None = DEFAULT_PERSON_MAX_WIDTH_FRAC,
                      rider_iou: float | None = DEFAULT_RIDER_IOU
                      ) -> tuple[dict, list[dict]]:
    """Like detect_and_count but also returns per-detection boxes.

    Detection is a two-stage filter: the model runs at the MOST PERMISSIVE
    threshold in `per_class_conf` (so nothing that any class needs is dropped
    before we can see it), then each raw detection is kept iff its confidence
    clears that class's per-class threshold. `person_min_aspect` and
    `person_max_aspect` reject person boxes whose height/width falls outside
    the plausible band - respectively the "stroller mis-read as person" case
    and the "lamp post/traffic sign mis-read as person" case. `rider_iou`
    resurrects a two-wheeler box that survived the model gate but not its
    per-class gate when it overlaps a surviving person box - a rider is a
    person AND a vehicle.

    Set `per_class_conf=None` to fall back to the single `conf` (legacy).
    Set `person_min_aspect=None` / `person_max_aspect=None` / `rider_iou=None`
    to skip that filter.

    Returns:
        counts: {class_name: int, "vehicles": int}
        boxes:  [{x1,y1,x2,y2,cls,conf}, ...] in pixel coords (BGR frame).
    """
    # Assemble the effective per-class gate. When the caller passes a single
    # legacy `conf`, per_class_conf=None means "same threshold everywhere" -
    # older callers keep their exact behavior. When per_class_conf is used,
    # the incoming `conf` is a global floor (nothing below is asked for).
    if per_class_conf is None:
        per_cls_gate = {c: conf for c in CLASSES_OF_INTEREST}
        model_gate = conf
    else:
        per_cls_gate = {c: float(per_class_conf.get(c, conf))
                        for c in CLASSES_OF_INTEREST}
        model_gate = max(0.001, min(min(per_cls_gate.values()), conf))

    kwargs = dict(conf=model_gate,
                  classes=list(CLASSES_OF_INTEREST.values()), verbose=False)
    if imgsz:
        kwargs["imgsz"] = imgsz
    res = model.predict(frame, **kwargs)[0]
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()

    # Stage 1: build the raw candidate list (everything the model returned).
    raw: list[dict] = []
    for i, c in enumerate(cls_ids):
        name = NAME_BY_ID.get(int(c))
        if not name:
            continue
        x1, y1, x2, y2 = xyxy[i].tolist()
        raw.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cls": name, "conf": float(confs[i])})

    # Stage 2: apply the per-class gate + person shape filter.
    kept: list[dict] = []
    below_gate: list[dict] = []
    for b in raw:
        gate = per_cls_gate.get(b["cls"], conf)
        if b["conf"] < gate:
            below_gate.append(b)
            continue
        if b["cls"] == "person":
            w, h = _box_wh(b)
            if person_min_height_px is not None and h < person_min_height_px:
                # Below this the box is too small to carry meaningful person
                # signal; usually a false pop on textured background.
                b["_dropped_reason"] = "person_too_small"
                continue
            if person_max_width_frac is not None:
                frame_w = frame.shape[1] if hasattr(frame, "shape") else None
                if frame_w and w > frame_w * person_max_width_frac:
                    # A "person" box wider than 30% of the frame is a metro
                    # car, a bus at close range, or a large signage board -
                    # never an actual pedestrian.
                    b["_dropped_reason"] = "person_too_wide"
                    continue
            if w > 0 and (person_min_aspect is not None
                          or person_max_aspect is not None):
                aspect = h / w
                if person_min_aspect is not None and aspect < person_min_aspect:
                    # Stroller / banner / cart shaped like a person to the model.
                    b["_dropped_reason"] = "person_aspect_low"
                    continue
                if person_max_aspect is not None and aspect > person_max_aspect:
                    # Lamp post / traffic sign / bollard: taller-and-thinner
                    # than any real pedestrian is.
                    b["_dropped_reason"] = "person_aspect_high"
                    continue
        kept.append(b)

    # Stage 3: rider co-detection - resurrect below-gate two-wheelers that
    # overlap a surviving person box. Person + motorcycle is a rider, and both
    # should be counted; without this the rider inflates the person count but
    # the vehicle disappears.
    if rider_iou is not None:
        persons = [b for b in kept if b["cls"] == "person"]
        for b in below_gate:
            if b["cls"] not in _TWO_WHEELERS:
                continue
            for p in persons:
                if box_iou(p, b) >= rider_iou:
                    b["_rescued_by_rider"] = True
                    kept.append(b)
                    break

    counts = {name: 0 for name in CLASSES_OF_INTEREST}
    boxes: list[dict] = []
    for b in kept:
        counts[b["cls"]] += 1
        boxes.append({k: b[k] for k in ("x1", "y1", "x2", "y2", "cls", "conf")})
    counts["vehicles"] = sum(counts[v] for v in VEHICLE_NAMES)
    return counts, boxes


def annotate(model, frame, conf: float = 0.35, imgsz: int | None = None):
    """Run detection and return the annotated frame (BGR ndarray).

    Runs a FRESH inference - fine for notebook one-offs. The collector calls
    draw_boxes() with the detections it already has instead, so an anomalous
    sample doesn't cost a second model pass on the VM. Both paths render via
    draw_boxes so calibration images and dashboard snapshots look identical.
    """
    _, boxes = detect_with_boxes(model, frame, conf=conf, imgsz=imgsz)
    return draw_boxes(frame, boxes)


def box_iou(a: dict | None, b: dict | None) -> float:
    """IoU of two {x1,y1,x2,y2} boxes; 0.0 if either is missing/degenerate."""
    if not a or not b:
        return 0.0
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


_BOX_COLORS = {
    "person":     (80, 175, 76),    # green (BGR)
    "bicycle":    (200, 130, 0),
    "car":        (60, 130, 246),
    "motorcycle": (200, 130, 0),
    "bus":        (0, 90, 230),
    "train":      (200, 60, 200),   # magenta - rail is neither road nor foot
    "truck":      (0, 90, 230),
}


def draw_boxes(frame: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """Annotate a COPY of `frame` with already-computed detection boxes.

    Same information as annotate() (class + confidence per box) without
    re-running the model.
    """
    out = frame.copy()
    for b in boxes:
        x1, y1 = int(b["x1"]), int(b["y1"])
        x2, y2 = int(b["x2"]), int(b["y2"])
        color = _BOX_COLORS.get(b.get("cls", ""), (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f'{b.get("cls", "?")} {b.get("conf", 0):.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = y1 - 4 if y1 - th - 6 >= 0 else y2 + th + 4
        cv2.rectangle(out, (x1, ty - th - 3), (x1 + tw + 4, ty + 3), color, -1)
        cv2.putText(out, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---- Burst sampling: several frames per sample, median count -----------------
# A single frame is a noisy estimator: a pedestrian occluded for one moment, or
# a car sitting at the edge of the confidence band, flips the count between
# consecutive frames. The collector therefore detects on a short burst and
# keeps the MEDIAN count - per-frame flicker cancels out while "now" still
# means "this handful of seconds".

def grab_burst(stream_url: str, n: int = 3, stride: int = 25) -> list[np.ndarray]:
    """Grab up to `n` frames spaced ~`stride` frames (~1 s at 25 fps) apart.

    Rides on iter_frames(), which already handles the header-required HLS hosts
    (tvkur / IBB / skyline) by decoding recent .ts segments locally. Falls back
    to the single-frame grab if the iterator yields nothing. May return fewer
    than `n` frames (short segments); callers should handle 1..n.
    """
    if n <= 1:
        f = grab_frame(stream_url)
        return [] if f is None else [f]
    frames: list[np.ndarray] = []
    try:
        for i, fr in enumerate(iter_frames(stream_url, max_frames=n * stride)):
            if i % stride == 0:
                frames.append(fr)
                if len(frames) >= n:
                    break
    except Exception:
        pass
    if not frames:
        f = grab_frame(stream_url)
        if f is not None:
            frames = [f]
    return frames


def median_counts(counts_list: list[dict]) -> dict:
    """Element-wise median over per-frame count dicts, rounded to int.

    Median (not mean) so one bad frame in the burst - a decode glitch, a bus
    covering the lens - cannot drag the reported count.
    """
    if not counts_list:
        return {}
    keys: set = set()
    for c in counts_list:
        keys.update(c.keys())
    out: dict = {}
    for k in keys:
        vals = sorted((c.get(k) or 0) for c in counts_list)
        m = len(vals)
        mid = vals[m // 2] if m % 2 == 1 else (vals[m // 2 - 1] + vals[m // 2]) / 2
        out[k] = int(round(mid))
    return out


def detect_burst(model, frames: list[np.ndarray], conf: float = 0.35,
                 imgsz: int | None = None,
                 roi: list | None = None,
                 roi_exclude: list | None = None,
                 roi_exclude_class: dict | None = None,
                 line: list | None = None,
                 per_class_conf: dict | None = None) -> tuple[dict, list[dict], np.ndarray, dict]:
    """Run detection over a burst of frames and aggregate counts by median.

    Returns (counts, boxes, frame, debug):
      counts - per-class median across the burst (plus "vehicles");
      boxes/frame - the representative frame (person count closest to the
        median, latest wins ties) and its detections. Re-ID and snapshots use
        that frame so they stay consistent with the reported counts;
      debug - raw per-frame person/vehicle series for the Firestore record,
        plus "crossings" (in/out per metric) when a `line` is configured.

    `roi`/`roi_exclude` (normalized polygons) drop detections whose foot point
    is outside the camera's area of interest BEFORE counting, so a parking lot
    or a neighboring roof can't inflate business-activity numbers.
    `roi_exclude_class` (dict cls -> [poly,...]) is the per-class variant:
    "in this zone, never accept `person`" - use it for known false-positive
    hotspots (traffic sign in the middle of the intersection, lamp post in
    the corner) without hiding other classes that legitimately pass through.
    `line` (normalized [[x1,y1],[x2,y2]]) counts burst-window crossings via
    short centroid tracks - a sampled entry/exit flow rate.
    """
    # Opt the collector into class-aware confidence gating by default: the
    # nano model on overhead cams loses two-wheelers and small pedestrians at a
    # single 0.35 threshold. Callers who want the legacy single-conf behavior
    # can pass per_class_conf={} explicitly.
    if per_class_conf is None:
        per_class_conf = DEFAULT_PER_CLASS_CONF
    per: list[tuple[dict, list[dict], np.ndarray]] = []
    for fr in frames:
        c, b = detect_with_boxes(model, fr, conf=conf, imgsz=imgsz,
                                 per_class_conf=per_class_conf or None)
        if roi or roi_exclude or roi_exclude_class:
            b = filter_boxes_roi(b, fr.shape, roi, roi_exclude,
                                 roi_exclude_class)
            c = counts_from_boxes(b)
        per.append((c, b, fr))
    counts = median_counts([c for c, _, _ in per])
    target = counts.get("person", 0)
    best = min(reversed(per), key=lambda t: abs((t[0].get("person") or 0) - target))
    debug = {
        "burst_person":   [c.get("person") for c, _, _ in per],
        "burst_vehicles": [c.get("vehicles") for c, _, _ in per],
    }
    if line and len(per) >= 2:
        tracks = track_burst([b for _, b, _ in per], per[0][2].shape)
        debug["crossings"] = count_line_crossings(tracks, per[0][2].shape, line)
    return counts, best[1], best[2], debug


if __name__ == "__main__":  # one-time stream-resolution check (run on an open network)
    import argparse

    from app.cameras import CAMERAS

    ap = argparse.ArgumentParser(description="Resolve a catalog camera to its live HLS URL")
    ap.add_argument("--resolve", default="", help="comma-separated cam ids (default: all)")
    args = ap.parse_args()

    ids = [c.strip() for c in args.resolve.split(",") if c.strip()] or list(CAMERAS)
    for cid in ids:
        cam = CAMERAS.get(cid)
        if not cam:
            print(f"{cid:16s} -> UNKNOWN camera id")
            continue
        try:
            print(f"{cid:16s} -> {resolve_stream(cam)}")
        except Exception as e:
            print(f"{cid:16s} -> FAILED ({e})")
