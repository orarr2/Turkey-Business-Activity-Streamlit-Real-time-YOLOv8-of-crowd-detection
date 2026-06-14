"""Continuous footfall collector — pushes live YOLO counts to Firestore.

A Jupyter cell runs once and stops. This process runs forever: every `--interval`
seconds it samples each camera, runs YOLO, updates the re-ID registry, and
writes the result to Firestore. The HTML dashboard at web/index.html subscribes
to those Firestore collections via onSnapshot and updates in real time — so
the data is genuinely shared and aggregative across visitors.

    python -m app.collector --interval 20 \\
        --only konya_hukumet,giresun_gazi,otogar_kavsagi,kadikoy

Requires FIREBASE_CREDENTIALS to point at your Firebase Admin SDK service-account
JSON. Run it on an open network (IBB hosts and skylinewebcams are blocked from
restricted sandboxes). Leave it under systemd / Docker / `nohup`.

Local persistence:
- `data/reid.db` (SQLite) holds the appearance registry the re-ID logic needs
  to recognise the same person/car across samples. It is the only piece of state
  this process keeps on disk; everything user-facing lives in Firestore.
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

from app.cameras import active_cameras
from app.detect_core import (
    CLASSES_OF_INTEREST,
    detect_with_boxes,
    grab_frame,
    load_model,
    resolve_stream,
)
from app.reid import ReidStore


def resolve_url(cam: dict) -> str:
    """Resolve any camera (hls / youtube / skyline / webcamera24) to a stream URL.

    Page-backed kinds (skyline, webcamera24, youtube) are re-resolved each cycle because
    their tokenized HLS URLs rotate.
    """
    return resolve_stream(cam)


def sample_once(model, cam_id: str, cam: dict, firebase,
                reid: ReidStore | None = None, conf: float = 0.35) -> None:
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

    try:
        firebase.write(record)
        if reid is not None and ok:
            # Push the per-camera re-ID summary too, so the HTML dashboard's
            # bottom table updates in real time alongside the count/anomaly tiles.
            firebase.write_reid_stats(cam_id, reid.stats(cam_id))
    except Exception as e:  # never let a backend hiccup kill the collector
        print(f"[{ts}] {cam_id}: firebase write failed ({e})")

    if ok:
        extra = f"  new={len(new_ids)} seen_again={len(seen_again)}" if reid is not None else ""
        print(f"[{ts}] {cam_id}: person={counts['person']} vehicles={counts['vehicles']}{extra}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Continuous YOLO footfall collector "
                                             "(writes to Firestore for the HTML dashboard)")
    ap.add_argument("--interval", type=int, default=20, help="seconds between sampling rounds")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--only", default="", help="comma-separated cam ids to restrict to")
    ap.add_argument("--reid-db", default="data/reid.db",
                    help="local SQLite path for the appearance-based re-ID registry "
                         "(set --no-reid to disable)")
    ap.add_argument("--no-reid", action="store_true",
                    help="disable re-identification (just count, don't track identities)")
    ap.add_argument("--reid-threshold", type=float, default=0.92,
                    help="cosine similarity above which a detection is judged 'seen before' "
                         "(lower = more aggressive merging, more false matches)")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="YOLO confidence threshold (lower = catches more small/distant objects)")
    args = ap.parse_args()

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

    print(f"Collector started. {len(cams)} camera(s): {list(cams)}")
    print(f"interval={args.interval}s, reid={'on' if reid else 'off'}, conf={args.conf}")
    print("Ctrl+C to stop.\n")

    try:
        while True:
            round_start = time.time()
            for cam_id, cam in cams.items():
                sample_once(model, cam_id, cam, firebase, reid=reid, conf=args.conf)
            # keep a steady cadence regardless of how long the round took
            time.sleep(max(0, args.interval - (time.time() - round_start)))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if reid is not None:
            reid.close()


if __name__ == "__main__":
    main()
