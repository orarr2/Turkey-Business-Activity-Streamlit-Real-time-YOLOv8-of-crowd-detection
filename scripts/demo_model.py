"""Demonstrate the full YOLO + detect_and_count pipeline on a known crowd image.

We can't hit IBB streams from this network (404 to every b_*.stream path; either GEO-
restricted or transiently offline). To still prove the model and the project's detection
core work end-to-end, this script:

  1. Pulls a canonical Ultralytics test image (bus + people).
  2. Runs the same detect_and_count used by app/collector.py.
  3. Saves the annotated frame to data/demo_detection.jpg.
  4. Writes a fake "footfall" record to local SQLite (data/footfall.db) so the
     downstream dashboard pipeline can be exercised.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import ssl
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from app.detect_core import CLASSES_OF_INTEREST, detect_and_count, load_model

IMAGES = {
    "bus":    "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg",
    "zidane": "https://raw.githubusercontent.com/ultralytics/assets/main/im/zidane.jpg",
}

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(parents=True, exist_ok=True)

ctx = ssl._create_unverified_context()

def download(url: str, dest: Path) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20, context=ctx) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest

def init_db(p: Path) -> sqlite3.Connection:
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS footfall (
        ts TEXT NOT NULL, cam_id TEXT NOT NULL, cam_name TEXT,
        person INTEGER, vehicles INTEGER, counts TEXT, ok INTEGER NOT NULL);
    CREATE INDEX IF NOT EXISTS idx_footfall_ts ON footfall(ts);
    """)
    conn.commit()
    return conn

print("=== Loading YOLO model ===")
model = load_model("yolov8n.pt")
print("Classes tracked:", list(CLASSES_OF_INTEREST))

db = init_db(DATA / "footfall.db")

for name, url in IMAGES.items():
    print(f"\n=== Sample image: {name} ===")
    img_path = DATA / f"sample_{name}.jpg"
    try:
        download(url, img_path)
        print(f"  downloaded {img_path.stat().st_size} bytes")
    except Exception as e:
        print(f"  download FAILED: {e}")
        continue
    frame = cv2.imread(str(img_path))
    if frame is None:
        print("  decode failed"); continue
    print(f"  frame shape: {frame.shape}")
    counts = detect_and_count(model, frame)
    print(f"  counts: {counts}")

    # save annotated frame
    res = model.predict(frame, conf=0.35, classes=list(CLASSES_OF_INTEREST.values()),
                        verbose=False)[0]
    annot = res.plot()
    out = DATA / f"demo_detection_{name}.jpg"
    cv2.imwrite(str(out), annot)
    print(f"  annotated -> {out}")

    # also write a footfall row (so the storage path is exercised too)
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    db.execute(
        "INSERT INTO footfall (ts, cam_id, cam_name, person, vehicles, counts, ok) "
        "VALUES (?, ?, ?, ?, ?, ?, 1)",
        (ts, f"demo_{name}", f"Ultralytics sample: {name}",
         counts.get("person"), counts.get("vehicles"), json.dumps(counts)),
    )

db.commit()
n = db.execute("SELECT COUNT(*) FROM footfall").fetchone()[0]
print(f"\nfootfall.db now has {n} rows.")
db.close()
