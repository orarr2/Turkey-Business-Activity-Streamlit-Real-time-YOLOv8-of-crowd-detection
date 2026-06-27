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

# COCO class ids we care about for *business activity* (footfall + vehicles).
CLASSES_OF_INTEREST = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}
NAME_BY_ID = {v: k for k, v in CLASSES_OF_INTEREST.items()}
VEHICLE_NAMES = ("bicycle", "car", "motorcycle", "bus", "truck")


def load_model(weights: str = "yolov8n.pt"):
    """Load a YOLO model once and reuse it. nano runs on CPU; bump to s/m for accuracy."""
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


def detect_and_count(model, frame, conf: float = 0.35) -> dict:
    """Run YOLO on one frame -> {class_name: count} for the classes we track."""
    counts, _ = detect_with_boxes(model, frame, conf=conf)
    return counts


def detect_with_boxes(model, frame, conf: float = 0.35) -> tuple[dict, list[dict]]:
    """Like detect_and_count but also returns per-detection boxes.

    Returns:
        counts: {class_name: int, "vehicles": int}
        boxes:  [{x1,y1,x2,y2,cls,conf}, ...] in pixel coords (BGR frame).
    """
    counts = {name: 0 for name in CLASSES_OF_INTEREST}
    boxes: list[dict] = []
    res = model.predict(
        frame, conf=conf, classes=list(CLASSES_OF_INTEREST.values()), verbose=False
    )[0]
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    for i, c in enumerate(cls_ids):
        name = NAME_BY_ID.get(int(c))
        if not name:
            continue
        counts[name] += 1
        x1, y1, x2, y2 = xyxy[i].tolist()
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "cls": name, "conf": float(confs[i])})
    counts["vehicles"] = sum(counts[v] for v in VEHICLE_NAMES)
    return counts, boxes


def annotate(model, frame, conf: float = 0.35):
    """Return the YOLO-annotated frame (BGR ndarray) for visualization."""
    res = model.predict(
        frame, conf=conf, classes=list(CLASSES_OF_INTEREST.values()), verbose=False
    )[0]
    return res.plot()


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
