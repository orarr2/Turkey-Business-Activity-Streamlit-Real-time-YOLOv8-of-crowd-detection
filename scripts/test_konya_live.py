"""Verify the Konya Sarraflar Yeralti Carsisi live HLS stream end-to-end:
   playlist HTTP -> cv2 frame grab -> YOLO detection."""
import os, ssl, sys, time, urllib.request
from pathlib import Path

# OpenCV/ffmpeg: allow self-signed CDN certs, send the Referer that tvkur expects
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "tls_verify;0|"
    "headers;Referer: https://player.tvkur.com/\\r\\n"
    "Origin: https://player.tvkur.com\\r\\n"
    "User-Agent: Mozilla/5.0\\r\\n"
)
sys.stdout.reconfigure(encoding="utf-8")

import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.detect_core import load_model, detect_and_count, CLASSES_OF_INTEREST

MASTER = "https://content.tvkur.com/l/c77i84vbb2nj4i0fr80g/master.m3u8"
ctx = ssl._create_unverified_context()

# 1) Confirm master playlist returns an EXTM3U body
req = urllib.request.Request(MASTER, headers={
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://player.tvkur.com/",
    "Origin":  "https://player.tvkur.com",
})
with urllib.request.urlopen(req, timeout=10, context=ctx) as r:
    body = r.read().decode("utf-8", "replace")
print(f"HTTP {r.status}  len={len(body)}")
print(body[:300])

# 2) cv2 open + read first frame
print("\nOpening with cv2.VideoCapture ...")
t0 = time.time()
cap = cv2.VideoCapture(MASTER)
ok, frame = cap.read()
cap.release()
print(f"cv2_ok={ok}  shape={None if not ok else frame.shape}  took {(time.time()-t0)*1000:.0f}ms")
if not ok:
    sys.exit(1)

# 3) YOLO detect + save annotated frame
print("\nLoading YOLO ...")
model = load_model("yolov8n.pt")
counts = detect_and_count(model, frame)
print("counts:", counts)

res = model.predict(frame, conf=0.35, classes=list(CLASSES_OF_INTEREST.values()), verbose=False)[0]
out = Path(__file__).resolve().parent.parent / "data" / "konya_live_detection.jpg"
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), res.plot())
print("annotated frame ->", out)
