"""Continuous footfall collector - pushes live YOLO counts to Firestore.

A Jupyter cell runs once and stops. This process runs forever: every `--interval`
seconds it samples each camera, runs YOLO, updates the re-ID registry, and
writes the result to Firestore. The HTML dashboard at web/index.html subscribes
to those Firestore collections via onSnapshot and updates in real time - so
the data is genuinely shared and aggregative across visitors.

    python -m app.collector --interval 20 \\
        --only konya_hukumet,otogar_kavsagi,sultanahmet_1_yeni,taksim_yeni

Requires FIREBASE_CREDENTIALS to point at your Firebase Admin SDK service-account
JSON. Run it on an open network (IBB hosts and skylinewebcams are blocked from
restricted sandboxes). Leave it under systemd / Docker / `nohup`.

Local persistence:
- `data/reid.db` (SQLite) holds the appearance registry the re-ID logic needs
  to recognise the same person/car across samples. It is the only piece of state
  this process keeps on disk; everything user-facing lives in Firestore.
- `web/snapshots/anomalies/{cam_id}/<ts>.jpg` and `<ts>_annotated.jpg` for any
  sample where the rolling z-score on the people series trips the anomaly
  threshold - the dashboard renders these as a clickable thumbnail.
- `web/snapshots/returning/{cam_id}/eid<N>_seen<K>_<ts>.jpg` (bbox crop) +
  `_full.jpg` (full frame) when re-ID matches an entity it hasn't seen for at
  least RETURNING_GAP_SEC (default 300 = 5 min).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import cv2

from app.cameras import active_cameras
from app.detect_core import (
    CLASSES_OF_INTEREST,
    annotate,
    detect_with_boxes,
    grab_frame,
    load_model,
    resolve_stream,
)
from app.reid import ReidStore

# --- Write rate-limit guard (protects your Firestore write quota / billing) ---
# The Firestore free (Spark) tier allows ~20k document writes/day. Each camera
# makes ~2 writes/round (footfall history + latest), or ~3 with re-ID on
# (+reid_stats). A too-small --interval (e.g. a typo of 1) could blow past that
# in minutes, so we clamp to a floor and warn on the projected daily total.
MIN_INTERVAL_S = 5
FREE_TIER_WRITES_PER_DAY = 20_000

# Roots for the runtime snapshot folders. These sit under web/ on purpose so
# serve.py exposes them at /snapshots/... without any extra route.
SNAPSHOTS_ROOT     = Path("web/snapshots")
ANOMALY_DIR        = SNAPSHOTS_ROOT / "anomalies"
RETURNING_DIR      = SNAPSHOTS_ROOT / "returning"
RETURNING_GAP_SEC  = 300   # save a returning-visitor image only when gap >= 5 min


class AnomalyTracker:
    """Per-camera rolling-window z-score detector.

    Matches the dashboard's flagAnomalies() so the same samples get flagged in
    Python and JS: window=12 most recent people counts, trip if |z| > 2.5.
    """

    def __init__(self, window: int = 12, z_threshold: float = 2.5, warmup: int = 6):
        self.window  = window
        self.z       = z_threshold
        self.warmup  = warmup
        self._history: dict[str, list[int]] = {}

    def push_and_check(self, cam_id: str, people: int | None) -> tuple[bool, dict]:
        """Append `people` and report whether it is an anomaly vs the window.

        Returns (is_anomaly, debug) where debug has window stats for logging.
        """
        if people is None:
            return False, {}
        hist = self._history.setdefault(cam_id, [])
        # Score *before* appending - we compare the new value to the window
        # that preceded it (otherwise the value would skew its own baseline).
        is_anom = False
        debug = {"window_size": len(hist)}
        if len(hist) >= self.warmup:
            mu = sum(hist) / len(hist)
            sd = (sum((x - mu) ** 2 for x in hist) / len(hist)) ** 0.5
            if sd > 0:
                z = abs((people - mu) / sd)
                debug.update({"mean": round(mu, 2), "std": round(sd, 2), "z": round(z, 2)})
                is_anom = z > self.z
        hist.append(int(people))
        if len(hist) > self.window:
            hist.pop(0)
        return is_anom, debug


def _ts_filename(ts_iso: str) -> str:
    # 2026-06-27T11:42:07.123456+00:00 -> 20260627_114207
    return ts_iso.replace("-", "").replace(":", "").replace("T", "_")[:15]


def _save_anomaly_snapshot(cam_id: str, ts_iso: str, frame, model, conf: float) -> dict:
    """Write raw + annotated frames; return {snapshot_url, snapshot_annotated_url}."""
    cam_dir = ANOMALY_DIR / cam_id
    cam_dir.mkdir(parents=True, exist_ok=True)
    stem = _ts_filename(ts_iso)
    raw_path = cam_dir / f"{stem}.jpg"
    cv2.imwrite(str(raw_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated_url = None
    try:
        annotated = annotate(model, frame, conf=conf)
        ann_path = cam_dir / f"{stem}_annotated.jpg"
        cv2.imwrite(str(ann_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_url = f"/snapshots/anomalies/{cam_id}/{stem}_annotated.jpg"
    except Exception:
        pass
    return {
        "snapshot_url":           f"/snapshots/anomalies/{cam_id}/{stem}.jpg",
        "snapshot_annotated_url": annotated_url,
    }


def _save_returning_visitor(cam_id: str, ts_iso: str, entity_id: int,
                            sightings: int, gap_sec: float, frame, box: dict) -> None:
    """Write the bbox crop + full frame; append to per-camera manifest.json."""
    cam_dir = RETURNING_DIR / cam_id
    cam_dir.mkdir(parents=True, exist_ok=True)
    stem  = _ts_filename(ts_iso)
    base  = f"eid{entity_id:04d}_seen{sightings:02d}_{stem}"
    x1 = max(0, int(box["x1"])); y1 = max(0, int(box["y1"]))
    x2 = min(frame.shape[1], int(box["x2"])); y2 = min(frame.shape[0], int(box["y2"]))
    if x2 > x1 and y2 > y1:
        crop = frame[y1:y2, x1:x2]
        cv2.imwrite(str(cam_dir / f"{base}.jpg"),      crop,  [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(cam_dir / f"{base}_full.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    manifest = cam_dir / "manifest.json"
    items = []
    if manifest.is_file():
        try:    items = json.loads(manifest.read_text())
        except Exception: items = []
    items.append({
        "ts": ts_iso, "entity_id": entity_id, "cls": box.get("cls"),
        "sightings": sightings, "gap_seconds": round(gap_sec, 1),
        "crop_url":      f"/snapshots/returning/{cam_id}/{base}.jpg",
        "fullframe_url": f"/snapshots/returning/{cam_id}/{base}_full.jpg",
    })
    manifest.write_text(json.dumps(items, indent=2))


def sample_once(model, cam_id: str, cam: dict, firebase,
                reid: ReidStore | None = None, conf: float = 0.35,
                anomaly: AnomalyTracker | None = None,
                save_snapshots: bool = True) -> None:
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    new_ids: list[int] = []
    seen_again: list[int] = []
    frame = None
    try:
        frame = grab_frame(resolve_stream(cam))
        if frame is None:
            raise RuntimeError("empty frame")
        counts, boxes = detect_with_boxes(model, frame, conf=conf)
        ok = 1
        # re-ID: which detections are new vs already-seen entities?
        if reid is not None and boxes:
            results = reid.update_from_frame(cam_id, frame, boxes)
            for i, r in enumerate(results):
                (new_ids if r.is_new else seen_again).append(r.entity_id)
                # Save a "returning visitor" image only when the entity hasn't
                # been seen for at least RETURNING_GAP_SEC: this gates out the
                # noisy short-term re-matches (someone lingering, neighbours in
                # the same frame) and keeps the folder full of genuine returns.
                if save_snapshots and (not r.is_new) \
                        and r.gap_seconds is not None \
                        and r.gap_seconds >= RETURNING_GAP_SEC \
                        and i < len(boxes):
                    try:
                        _save_returning_visitor(cam_id, ts, r.entity_id,
                                                r.sightings, r.gap_seconds,
                                                frame, boxes[i])
                    except Exception as e:
                        print(f"  ! returning save failed for {cam_id} eid{r.entity_id}: {e}")
    except Exception as e:
        # network blip / stream hiccup -> record a miss, keep going
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

    # Anomaly check + snapshot. Done after the YOLO call so the tracker only
    # sees real samples (misses are skipped via people=None).
    if anomaly is not None and ok:
        is_anom, dbg = anomaly.push_and_check(cam_id, counts.get("person"))
        record["is_anomaly"] = bool(is_anom)
        if is_anom and save_snapshots and frame is not None:
            try:
                record.update(_save_anomaly_snapshot(cam_id, ts, frame, model, conf))
                print(f"  ! anomaly @ {cam_id} - z={dbg.get('z')}, mu={dbg.get('mean')}, "
                      f"people={counts['person']} - snapshot saved")
            except Exception as e:
                print(f"  ! anomaly snapshot save failed for {cam_id}: {e}")

    try:
        firebase.write(record)
        if reid is not None and ok:
            firebase.write_reid_stats(cam_id, reid.stats(cam_id))
    except Exception as e:
        print(f"[{ts}] {cam_id}: firebase write failed ({e})")

    if ok:
        extra = f"  new={len(new_ids)} seen_again={len(seen_again)}" if reid is not None else ""
        flag  = "  ANOMALY" if record.get("is_anomaly") else ""
        print(f"[{ts}] {cam_id}: person={counts['person']} vehicles={counts['vehicles']}{extra}{flag}")


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
    ap.add_argument("--no-snapshots", action="store_true",
                    help="skip writing anomaly / returning-visitor images to web/snapshots/")
    args = ap.parse_args()

    # Rate-limit guard: never let the collector hammer Firestore faster than the
    # floor, regardless of what the user passed.
    if args.interval < MIN_INTERVAL_S:
        print(f"--interval {args.interval}s is below the {MIN_INTERVAL_S}s floor; "
              f"clamping to {MIN_INTERVAL_S}s to protect your Firestore write quota.")
        args.interval = MIN_INTERVAL_S

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

    anomaly        = AnomalyTracker()
    save_snapshots = not args.no_snapshots

    print(f"Collector started. {len(cams)} camera(s): {list(cams)}")
    print(f"interval={args.interval}s, reid={'on' if reid else 'off'}, "
          f"conf={args.conf}, snapshots={'on' if save_snapshots else 'off'}")

    # Project the daily write volume and warn if it would exceed the free tier.
    writes_per_round = len(cams) * (3 if reid else 2)
    projected = writes_per_round * (86400 / args.interval)
    print(f"~{projected:,.0f} Firestore writes/day projected "
          f"(free tier ~ {FREE_TIER_WRITES_PER_DAY:,}).")
    if projected > FREE_TIER_WRITES_PER_DAY:
        print("  ! Above the free tier - raise --interval, run fewer cameras, or set a "
              "billing budget alert / daily cap (see docs/firebase_setup.md sec.7).")
    print("Ctrl+C to stop.\n")

    try:
        while True:
            round_start = time.time()
            for cam_id, cam in cams.items():
                sample_once(model, cam_id, cam, firebase, reid=reid,
                            conf=args.conf, anomaly=anomaly,
                            save_snapshots=save_snapshots)
            time.sleep(max(0, args.interval - (time.time() - round_start)))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if reid is not None:
            reid.close()


if __name__ == "__main__":
    main()
