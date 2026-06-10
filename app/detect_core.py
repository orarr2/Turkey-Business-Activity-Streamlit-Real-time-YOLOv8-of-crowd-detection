"""Shared detection + stream-access core.

Imported by the notebook, the collector daemon, and the Streamlit app so the
detection logic lives in exactly one place.
"""
from __future__ import annotations

import os
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


_SSL_CTX = ssl._create_unverified_context()

# Some live-CDN HLS endpoints (e.g. content.tvkur.com) require a Referer/Origin header
# that ffmpeg-via-cv2 can't always pass on Windows. For those hosts we fetch the latest
# .ts segment manually and decode locally.
HEADER_HOSTS = {
    "content.tvkur.com":     {"Referer": "https://player.tvkur.com/",
                              "Origin":  "https://player.tvkur.com"},
    "livestream.ibb.gov.tr": {"Referer": "https://istanbuluseyret.ibb.gov.tr/",
                              "Origin":  "https://istanbuluseyret.ibb.gov.tr"},
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
