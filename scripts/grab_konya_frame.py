"""Fetch a .ts segment from the Konya HLS stream and decode a frame with cv2.

Workaround: cv2 on this build won't pass the Referer header to ffmpeg via env opts,
so we download a segment manually (with Referer) then decode it locally.
"""
import os, re, ssl, sys, time, urllib.request, tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ctx = ssl._create_unverified_context()

MASTER = "https://content.tvkur.com/l/c77i84vbb2nj4i0fr80g/master.m3u8"
BASE = MASTER.rsplit("/", 1)[0] + "/"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer":    "https://player.tvkur.com/",
    "Origin":     "https://player.tvkur.com",
}

def http_get(url, **extra):
    h = HEADERS.copy(); h.update(extra)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=15, context=ctx) as r:
        return r.read()

def grab_one_frame():
    """Return the latest frame from the live stream as a BGR ndarray (or None)."""
    pl = http_get(MASTER).decode("utf-8", "replace")
    # Some streams (this one) only return media playlists at master.m3u8. If we see
    # #EXT-X-STREAM-INF lines, follow the first variant; otherwise treat as media.
    segs = []
    if "#EXT-X-STREAM-INF" in pl:
        variant = next((l.strip() for l in pl.splitlines()
                        if l.strip() and not l.startswith("#")), None)
        if variant:
            variant_url = variant if variant.startswith("http") else BASE + variant
            pl = http_get(variant_url).decode("utf-8", "replace")
    for line in pl.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            segs.append(line)
    if not segs:
        print("no segments in playlist"); return None
    seg = segs[-1]               # the most recent segment
    seg_url = seg if seg.startswith("http") else BASE + seg
    print(f"fetching {seg_url}")
    data = http_get(seg_url)
    print(f"  {len(data)} bytes")

    # Write the .ts to a temp file and let cv2 open it locally.
    with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as f:
        f.write(data); tmp = f.name
    try:
        import cv2
        cap = cv2.VideoCapture(tmp)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    finally:
        os.unlink(tmp)

frame = grab_one_frame()
if frame is None:
    print("frame grab failed"); sys.exit(1)
print(f"frame shape: {frame.shape}")

# YOLO detect
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.detect_core import load_model, detect_and_count, CLASSES_OF_INTEREST
import cv2

model = load_model("yolov8n.pt")
counts = detect_and_count(model, frame)
print("counts:", counts)

res = model.predict(frame, conf=0.35, classes=list(CLASSES_OF_INTEREST.values()), verbose=False)[0]
out = Path(__file__).resolve().parent.parent / "data" / "konya_live_detection.jpg"
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), res.plot())
print("annotated ->", out)
