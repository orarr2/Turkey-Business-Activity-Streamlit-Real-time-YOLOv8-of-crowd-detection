"""Continuous footfall collector — the answer to "how is the data live?".

A Jupyter cell runs once and stops. This process runs forever: every `--interval`
seconds it samples each camera, runs YOLO, and appends the counts to a SQLite
database. The dashboard reads that same DB, so the data is always fresh without
ever re-running a notebook cell.

Run it next to your app (open network required — IBB hosts are blocked from
restricted sandboxes):

    python -m app.collector --db data/footfall.db --interval 20

Leave it running (or put it under systemd / a container / `nohup`). It is fully
decoupled from any UI.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import time
from pathlib import Path

import cv2

from app.cameras import active_cameras
from app.detect_core import (
    CLASSES_OF_INTEREST,
    annotate,
    detect_and_count,
    detect_with_boxes,
    grab_frame,
    load_model,
    resolve_stream,
)
from app.reid import ReidStore

SCHEMA = """
CREATE TABLE IF NOT EXISTS footfall (
    ts        TEXT NOT NULL,        -- ISO8601 UTC
    cam_id    TEXT NOT NULL,
    cam_name  TEXT,
    person    INTEGER,
    vehicles  INTEGER,
    counts    TEXT,                 -- full per-class JSON
    ok        INTEGER NOT NULL      -- 1 = frame decoded, 0 = miss
);
CREATE INDEX IF NOT EXISTS idx_footfall_ts ON footfall(ts);
CREATE INDEX IF NOT EXISTS idx_footfall_cam ON footfall(cam_id);
"""


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def resolve_url(cam: dict) -> str:
    """Resolve any camera (hls / youtube / skyline / webcamera24) to a stream URL.

    Page-backed kinds (skyline, webcamera24, youtube) are re-resolved each cycle because
    their tokenized HLS URLs rotate.
    """
    return resolve_stream(cam)


def sample_once(model, conn, cam_id: str, cam: dict, firebase=None,
                reid: ReidStore | None = None, frames_dir: Path | None = None,
                conf: float = 0.35) -> None:
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    new_ids: list[int] = []
    seen_again: list[int] = []
    try:
        frame = grab_frame(resolve_url(cam))
        if frame is None:
            raise RuntimeError("empty frame")
        counts, boxes = detect_with_boxes(model, frame, conf=conf)
        ok = 1
        # re-ID: which detections are new vs already-seen entities?
        if reid is not None and boxes:
            for r in reid.update_from_frame(cam_id, frame, boxes):
                (new_ids if r.is_new else seen_again).append(r.entity_id)
        # save the annotated frame so the dashboard can show the model's prediction
        if frames_dir is not None:
            try:
                ann = annotate(model, frame, conf=conf)
                cv2.imwrite(str(frames_dir / f"latest_{cam_id}.jpg"), ann)
            except Exception as e:
                print(f"[{ts}] {cam_id}: annotated-frame save failed ({e})")
    except Exception as e:  # network blip / stream hiccup -> record a miss, keep going
        print(f"[{ts}] {cam_id}: MISS ({e})")
        counts = {name: None for name in CLASSES_OF_INTEREST}
        counts["vehicles"] = None
        ok = 0

    record = {
        "ts": ts, "cam_id": cam_id, "cam_name": cam["name"],
        "person": counts.get("person"), "vehicles": counts.get("vehicles"),
        "counts": counts, "ok": ok,
        "new_entities":  len(new_ids),
        "seen_entities": len(seen_again),
    }

    if conn is not None:
        conn.execute(
            "INSERT INTO footfall (ts, cam_id, cam_name, person, vehicles, counts, ok) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ts, cam_id, cam["name"], record["person"], record["vehicles"],
             json.dumps(counts), ok),
        )
        conn.commit()
    if firebase is not None:
        try:
            firebase.write(record)
        except Exception as e:  # never let a backend hiccup kill the collector
            print(f"[{ts}] {cam_id}: firebase write failed ({e})")
    if ok:
        extra = ""
        if reid is not None:
            extra = f"  new={len(new_ids)} seen_again={len(seen_again)}"
        print(f"[{ts}] {cam_id}: person={counts['person']} vehicles={counts['vehicles']}{extra}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Continuous YOLO footfall collector")
    ap.add_argument("--db", default="data/footfall.db")
    ap.add_argument("--interval", type=int, default=20, help="seconds between sampling rounds")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--backend", choices=["sqlite", "firebase", "both"], default="sqlite",
                    help="where to write counts (firebase needs FIREBASE_CREDENTIALS)")
    ap.add_argument("--only", default="", help="comma-separated cam ids to restrict to")
    ap.add_argument("--reid-db", default="data/reid.db",
                    help="SQLite path for the appearance-based re-ID registry "
                         "(set --no-reid to disable)")
    ap.add_argument("--no-reid", action="store_true",
                    help="disable re-identification (just count, don't track identities)")
    ap.add_argument("--reid-threshold", type=float, default=0.92,
                    help="cosine similarity above which a detection is judged 'seen before' "
                         "(lower = more aggressive merging, more false matches)")
    ap.add_argument("--frames-dir", default="data/frames",
                    help="directory to save latest_{cam_id}.jpg annotated frames "
                         "(consumed by streamlit dashboard)")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="YOLO confidence threshold (lower = catches more small/distant objects)")
    args = ap.parse_args()

    conn = init_db(Path(args.db)) if args.backend in ("sqlite", "both") else None

    firebase = None
    if args.backend in ("firebase", "both"):
        from app.firebase_store import FirebaseStore
        firebase = FirebaseStore()
        print("Firebase backend initialized.")

    model = load_model(args.weights)
    cams = active_cameras()
    if args.only:
        wanted = {c.strip() for c in args.only.split(",")}
        cams = {k: v for k, v in cams.items() if k in wanted}

    reid = None
    if not args.no_reid:
        reid = ReidStore(args.reid_db, threshold=args.reid_threshold)

    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collector started. {len(cams)} camera(s): {list(cams)}")
    print(f"interval={args.interval}s, backend={args.backend}, "
          f"reid={'on' if reid else 'off'}, conf={args.conf}")
    print("Ctrl+C to stop.\n")

    try:
        while True:
            round_start = time.time()
            for cam_id, cam in cams.items():
                sample_once(model, conn, cam_id, cam, firebase,
                            reid=reid, frames_dir=frames_dir, conf=args.conf)
            # keep a steady cadence regardless of how long the round took
            time.sleep(max(0, args.interval - (time.time() - round_start)))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if conn is not None:
            conn.close()
        if reid is not None:
            reid.close()


if __name__ == "__main__":
    main()
