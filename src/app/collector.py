"""Continuous footfall collector - pushes live YOLO counts to Firestore.

A Jupyter cell runs once and stops. This process runs forever: every `--interval`
seconds it iterates the four GRID_SLOTS, picks each slot's currently-healthy
camera (with fallback), runs YOLO, updates the re-ID registry, and writes the
result to Firestore (keyed by slot_id, not cam_id). The HTML dashboard subscribes
via onSnapshot and updates in real time.

    python -m app.collector --interval 20

Requires FIREBASE_CREDENTIALS to point at the Firebase Admin SDK service-account
JSON. Optional FIREBASE_STORAGE_BUCKET to upload anomaly / returning-visitor
snapshots to Firebase Storage; without it, snapshots are written to
web/snapshots/ on local disk (the notebook/laptop mode).

Local persistence:
- `data/reid.db` (SQLite) — appearance registry the re-ID logic needs to
  recognise the same person/car across samples. Only piece of state kept on
  disk; everything user-facing lives in Firestore + Storage.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import cv2

from app.cameras import CAMERAS, GRID_SLOTS
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
# Free tier: ~20k writes/day. Each slot per round = 3 writes (footfall + latest +
# reid_stats). 4 slots @ 20s => ~52k writes/day. Blaze free tier covers it, but
# we still enforce a floor to prevent typos like --interval 1.
MIN_INTERVAL_S = 5
FREE_TIER_WRITES_PER_DAY = 20_000

# --- Fallback picker knobs ----------------------------------------------------
FALLBACK_MAX_FAILURES  = 3   # consecutive failures before advancing the chain
FALLBACK_RETRY_MINUTES = 15  # after this long on a fallback, re-try the primary

# Roots for LOCAL snapshot mode (used when FIREBASE_STORAGE_BUCKET isn't set).
# These sit under web/ so serve.py exposes them at /snapshots/... automatically.
# The cloud collector on the VM sets FIREBASE_STORAGE_BUCKET and never touches
# these paths — snapshots go straight to Storage.
_SRC_ROOT      = Path(__file__).resolve().parent.parent
SNAPSHOTS_ROOT = _SRC_ROOT / "web" / "snapshots"
ANOMALY_DIR    = SNAPSHOTS_ROOT / "anomalies"
RETURNING_DIR  = SNAPSHOTS_ROOT / "returning"

# ---- Returning-visitor gates (each saved image is a real return event) -----
RETURNING_GAP_SEC              = 900   # >= 15 min absence
RETURNING_MIN_SIMILARITY       = 0.96  # >= 0.96 cosine
RETURNING_MIN_PRIOR_SIGHTINGS  = 2     # entity must have been seen >= 2 times
RETURNING_PER_ENTITY_COOLDOWN  = 1800  # same eid at most once per 30 min


class SlotStreamPicker:
    """Per-slot circuit-breaker over an ordered fallback chain.

      * `current_cam()` — the cam_id we should sample right now.
      * `record_result(ok)` — feed back whether the last sample succeeded.

    After `max_failures` consecutive misses we advance one step down the chain.
    Every `retry_minutes` we retry the primary regardless of where we are; if
    the primary works, we snap back to index 0 (primary is always preferred).
    """

    def __init__(self, slot: dict,
                 max_failures: int = FALLBACK_MAX_FAILURES,
                 retry_minutes: int = FALLBACK_RETRY_MINUTES):
        self.slot_id       = slot["slot_id"]
        self.display_area  = slot["display_area"]
        self.chain         = [slot["primary"]] + list(slot["fallbacks"])
        self.primary       = slot["primary"]
        self.idx           = 0
        self.failures      = 0
        self.max_failures  = max_failures
        self.retry_seconds = retry_minutes * 60
        self.last_primary_check = time.time()

    def current_cam(self) -> str:
        # Periodic retry of the primary — if we drifted onto a fallback and it's
        # been long enough, put us back at the top of the chain so the next
        # sample tests the primary again.
        if self.idx > 0 and (time.time() - self.last_primary_check) >= self.retry_seconds:
            self.idx = 0
            self.failures = 0
            self.last_primary_check = time.time()
        return self.chain[self.idx]

    def record_result(self, ok: bool) -> str | None:
        """Return the new active cam_id if it changed this call, else None."""
        prev = self.chain[self.idx]
        if ok:
            self.failures = 0
            if self.chain[self.idx] == self.primary:
                self.last_primary_check = time.time()
        else:
            self.failures += 1
            if self.failures >= self.max_failures and self.idx < len(self.chain) - 1:
                self.idx += 1
                self.failures = 0
        return self.chain[self.idx] if self.chain[self.idx] != prev else None


class AnomalyTracker:
    """Per-slot anomaly detector with layered gates that ALL must pass.

    Six gates (window, z_threshold, spike_only, min_people, min_delta, min_std)
    plus a cooldown suppress the vast majority of low-activity noise so only
    meaningful crowd spikes light up. See docstrings in _passes.
    """

    def __init__(self, window: int = 30, z_threshold: float = 3.5, warmup: int = 10,
                 min_people: int = 5, min_delta: float = 5.0, min_std: float = 0.0,
                 spike_only: bool = True, cooldown_sec: float = 300):
        self.window       = window
        self.z            = z_threshold
        self.warmup       = warmup
        self.min_people   = min_people
        self.min_delta    = min_delta
        self.min_std      = min_std
        self.spike_only   = spike_only
        self.cooldown_sec = cooldown_sec
        self._history: dict[str, list[int]] = {}
        self._last_flagged: dict[str, float] = {}

    def push_and_check(self, key: str, people: int | None) -> tuple[bool, dict]:
        if people is None:
            return False, {"reason": "no_sample"}
        hist = self._history.setdefault(key, [])
        debug: dict = {"window_size": len(hist), "people": int(people)}
        try:
            if len(hist) < self.warmup:
                return False, {**debug, "reason": "warmup"}
            mu  = sum(hist) / len(hist)
            sd  = (sum((x - mu) ** 2 for x in hist) / len(hist)) ** 0.5
            delta = people - mu
            z = (delta / sd) if sd > 0 else 0.0
            debug.update({"mean": round(mu, 2), "std": round(sd, 2),
                          "delta": round(delta, 2), "z": round(z, 2)})
            if self.spike_only and delta <= 0:
                return False, {**debug, "reason": "not_a_spike"}
            if people < self.min_people:
                return False, {**debug, "reason": "below_min_people"}
            if sd < self.min_std:
                return False, {**debug, "reason": "quiet_baseline"}
            if abs(delta) < self.min_delta:
                return False, {**debug, "reason": "small_delta"}
            z_check = z if self.spike_only else abs(z)
            if z_check < self.z:
                return False, {**debug, "reason": "below_z"}
            now  = time.time()
            last = self._last_flagged.get(key, 0.0)
            if now - last < self.cooldown_sec:
                return False, {**debug, "reason": "cooldown",
                               "cooldown_remaining": round(self.cooldown_sec - (now - last), 1)}
            self._last_flagged[key] = now
            return True, {**debug, "reason": "anomaly"}
        finally:
            hist.append(int(people))
            if len(hist) > self.window:
                hist.pop(0)


def _ts_filename(ts_iso: str) -> str:
    return ts_iso.replace("-", "").replace(":", "").replace("T", "_")[:15]


def _slot_metadata(slot: dict, active_cam: str) -> dict:
    """Snapshot of what the dashboard needs about a slot right now."""
    cam = CAMERAS.get(active_cam, {})
    return {
        "slot_id":         slot["slot_id"],
        "primary":         slot["primary"],
        "active_cam":      active_cam,
        "active_cam_name": cam.get("name", active_cam),
        "active_embed":    cam.get("embed"),
        "active_page":     cam.get("page"),
        "active_hls":      cam.get("url") if cam.get("kind") == "hls" else None,
        "active_kind":     cam.get("kind"),
        "display_area":    slot["display_area"],
    }


def _save_anomaly_snapshot(slot_id: str, cam_id: str, ts_iso: str,
                           frame, model, conf: float, firebase) -> dict:
    """Save raw + annotated frames. Uses Storage if configured, else local disk."""
    stem = _ts_filename(ts_iso)
    raw_ok, raw_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not raw_ok:
        return {}
    urls = {"snapshot_url": None, "snapshot_annotated_url": None}
    try:
        annotated_frame = annotate(model, frame, conf=conf)
    except Exception:
        annotated_frame = None

    if firebase.storage is not None:
        urls["snapshot_url"] = firebase.upload_snapshot(
            f"anomalies/{slot_id}/{stem}.jpg", raw_buf.tobytes())
        if annotated_frame is not None:
            ok, ann_buf = cv2.imencode(".jpg", annotated_frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                urls["snapshot_annotated_url"] = firebase.upload_snapshot(
                    f"anomalies/{slot_id}/{stem}_annotated.jpg", ann_buf.tobytes())
    else:
        cam_dir = ANOMALY_DIR / slot_id
        cam_dir.mkdir(parents=True, exist_ok=True)
        raw_path = cam_dir / f"{stem}.jpg"
        raw_path.write_bytes(raw_buf.tobytes())
        urls["snapshot_url"] = f"/snapshots/anomalies/{slot_id}/{stem}.jpg"
        if annotated_frame is not None:
            ok, ann_buf = cv2.imencode(".jpg", annotated_frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                (cam_dir / f"{stem}_annotated.jpg").write_bytes(ann_buf.tobytes())
                urls["snapshot_annotated_url"] = f"/snapshots/anomalies/{slot_id}/{stem}_annotated.jpg"
    return urls


def _save_returning_visitor(slot_id: str, cam_id: str, ts_iso: str,
                            entity_id: int, sightings: int, gap_sec: float,
                            frame, box: dict, firebase) -> None:
    """Save the bbox crop + full frame. Uses Storage if configured, else local."""
    stem  = _ts_filename(ts_iso)
    base  = f"eid{entity_id:04d}_seen{sightings:02d}_{stem}"
    x1 = max(0, int(box["x1"])); y1 = max(0, int(box["y1"]))
    x2 = min(frame.shape[1], int(box["x2"])); y2 = min(frame.shape[0], int(box["y2"]))
    if not (x2 > x1 and y2 > y1):
        return

    crop = frame[y1:y2, x1:x2]
    ok_c, crop_buf = cv2.imencode(".jpg", crop,  [cv2.IMWRITE_JPEG_QUALITY, 85])
    ok_f, full_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not (ok_c and ok_f):
        return

    if firebase.storage is not None:
        crop_url = firebase.upload_snapshot(
            f"returning/{slot_id}/{base}.jpg", crop_buf.tobytes())
        full_url = firebase.upload_snapshot(
            f"returning/{slot_id}/{base}_full.jpg", full_buf.tobytes())
    else:
        cam_dir = RETURNING_DIR / slot_id
        cam_dir.mkdir(parents=True, exist_ok=True)
        (cam_dir / f"{base}.jpg").write_bytes(crop_buf.tobytes())
        (cam_dir / f"{base}_full.jpg").write_bytes(full_buf.tobytes())
        crop_url = f"/snapshots/returning/{slot_id}/{base}.jpg"
        full_url = f"/snapshots/returning/{slot_id}/{base}_full.jpg"
        # Local-only manifest for the serve.py dashboard.
        manifest = cam_dir / "manifest.json"
        items = []
        if manifest.is_file():
            try:    items = json.loads(manifest.read_text())
            except Exception: items = []
        items.append({
            "ts": ts_iso, "entity_id": entity_id, "cls": box.get("cls"),
            "sightings": sightings, "gap_seconds": round(gap_sec, 1),
            "crop_url": crop_url, "fullframe_url": full_url,
        })
        manifest.write_text(json.dumps(items, indent=2))


def _passes_returning_gates(r, gap_min_sec: float, sim_min: float,
                            min_prior: int, cooldown_sec: float,
                            last_save_for_eid: dict) -> tuple[bool, str]:
    if r.is_new:                              return False, "new_entity"
    if r.gap_seconds is None:                 return False, "no_gap"
    if r.gap_seconds < gap_min_sec:           return False, "short_gap"
    if r.similarity is not None and r.similarity < sim_min:
                                              return False, "weak_match"
    prior = max(0, (r.sightings or 1) - 1)
    if prior < min_prior:                     return False, "few_prior_sightings"
    now  = time.time()
    last = last_save_for_eid.get(r.entity_id, 0.0)
    if now - last < cooldown_sec:             return False, "per_entity_cooldown"
    last_save_for_eid[r.entity_id] = now
    return True, "save"


def sample_slot(model, slot: dict, cam_id: str, firebase,
                reid: ReidStore | None = None, conf: float = 0.35,
                anomaly: AnomalyTracker | None = None,
                save_snapshots: bool = True,
                returning_gap_sec: float = RETURNING_GAP_SEC,
                returning_sim_min: float = RETURNING_MIN_SIMILARITY,
                returning_min_prior: int  = RETURNING_MIN_PRIOR_SIGHTINGS,
                returning_cooldown_sec: float = RETURNING_PER_ENTITY_COOLDOWN,
                _returning_last_save: dict | None = None) -> bool:
    """Sample the currently-active cam for a slot and write to Firestore.

    Returns True iff the frame was grabbed and processed successfully. The
    caller feeds this back to the SlotStreamPicker to decide whether to
    advance the fallback chain.
    """
    slot_id = slot["slot_id"]
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    new_ids: list[int] = []
    seen_again: list[int] = []
    frame = None
    cam = CAMERAS.get(cam_id)
    if cam is None:
        print(f"[{ts}] {slot_id}: unknown cam_id {cam_id!r}, skipping")
        return False

    if _returning_last_save is None:
        _returning_last_save = getattr(sample_slot, "_returning_state",
                                       {}).setdefault(slot_id, {})
        sample_slot._returning_state = getattr(sample_slot, "_returning_state", {})
        sample_slot._returning_state[slot_id] = _returning_last_save

    try:
        frame = grab_frame(resolve_stream(cam))
        if frame is None:
            raise RuntimeError("empty frame")
        counts, boxes = detect_with_boxes(model, frame, conf=conf)
        ok = 1
        if reid is not None and boxes:
            results = reid.update_from_frame(cam_id, frame, boxes)
            for i, r in enumerate(results):
                (new_ids if r.is_new else seen_again).append(r.entity_id)
                if save_snapshots and i < len(boxes):
                    passes, _why = _passes_returning_gates(
                        r, returning_gap_sec, returning_sim_min,
                        returning_min_prior, returning_cooldown_sec,
                        _returning_last_save)
                    if passes:
                        try:
                            _save_returning_visitor(slot_id, cam_id, ts,
                                                    r.entity_id, r.sightings,
                                                    r.gap_seconds, frame,
                                                    boxes[i], firebase)
                        except Exception as e:
                            print(f"  ! returning save failed for {slot_id}/{cam_id} "
                                  f"eid{r.entity_id}: {e}")
    except Exception as e:
        print(f"[{ts}] {slot_id} ({cam_id}): MISS ({e})")
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

    # Anomaly gating keyed by slot (not cam) so a fallback swap doesn't reset the
    # rolling window — the DASHBOARD tile is what we're comparing to itself.
    if anomaly is not None and ok:
        is_anom, dbg = anomaly.push_and_check(slot_id, counts.get("person"))
        record["is_anomaly"] = bool(is_anom)
        if is_anom and save_snapshots and frame is not None:
            try:
                record.update(_save_anomaly_snapshot(slot_id, cam_id, ts, frame,
                                                    model, conf, firebase))
                print(f"  ! anomaly @ {slot_id}/{cam_id} - z={dbg.get('z')}, "
                      f"mu={dbg.get('mean')}, people={counts['person']} - snapshot saved")
            except Exception as e:
                print(f"  ! anomaly snapshot save failed for {slot_id}: {e}")

    try:
        firebase.write(slot_id, record)
        if reid is not None and ok:
            firebase.write_reid_stats(slot_id, cam_id, reid.stats(cam_id))
    except Exception as e:
        print(f"[{ts}] {slot_id}: firebase write failed ({e})")

    if ok:
        extra = f"  new={len(new_ids)} seen_again={len(seen_again)}" if reid is not None else ""
        flag  = "  ANOMALY" if record.get("is_anomaly") else ""
        print(f"[{ts}] {slot_id} ({cam_id}): person={counts['person']} "
              f"vehicles={counts['vehicles']}{extra}{flag}")
    return bool(ok)


# --- Legacy shim for the viewer/admin notebooks and single-cam smoke tests. ---
# Keeps the old signature working; internally wraps sample_slot in a
# "one-off slot" so the record still lands in Firestore under a stable key.
def sample_once(model, cam_id: str, cam: dict, firebase,
                reid: ReidStore | None = None, conf: float = 0.35,
                anomaly: AnomalyTracker | None = None,
                save_snapshots: bool = True,
                slot_id: str | None = None,
                **kwargs) -> bool:
    slot = {"slot_id": slot_id or f"cam_{cam_id}",
            "display_area": cam.get("name", cam_id),
            "primary":      cam_id,
            "fallbacks":    []}
    return sample_slot(model, slot, cam_id, firebase, reid=reid, conf=conf,
                       anomaly=anomaly, save_snapshots=save_snapshots, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Continuous YOLO footfall collector "
                                             "(writes to Firestore + Storage for the HTML dashboard)")
    ap.add_argument("--interval", type=int, default=20, help="seconds between sampling rounds")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--reid-db", default="data/reid.db",
                    help="local SQLite path for the appearance-based re-ID registry")
    ap.add_argument("--no-reid", action="store_true", help="disable re-identification")
    ap.add_argument("--reid-threshold", type=float, default=0.92,
                    help="cosine similarity above which a detection is 'seen before'")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    ap.add_argument("--no-snapshots", action="store_true",
                    help="skip anomaly / returning-visitor image saves")
    ap.add_argument("--prune-snapshots", action="store_true",
                    help="delete every file under web/snapshots/{anomalies,returning}/* "
                         "before starting (local mode only; Storage cleanup uses the "
                         "lifecycle rule)")
    ag = ap.add_argument_group("anomaly gating (each gate must pass for a snapshot)")
    ag.add_argument("--anomaly-z",          type=float, default=3.5)
    ag.add_argument("--anomaly-window",     type=int,   default=30)
    ag.add_argument("--anomaly-min-people", type=int,   default=5)
    ag.add_argument("--anomaly-min-delta",  type=float, default=5.0)
    ag.add_argument("--anomaly-min-std",    type=float, default=0.0)
    ag.add_argument("--anomaly-cooldown",   type=float, default=300.0)
    ag.add_argument("--anomaly-allow-drops", action="store_true")
    rg = ap.add_argument_group("returning-visitor gating")
    rg.add_argument("--returning-gap-min",       type=float, default=15.0)
    rg.add_argument("--returning-min-similarity", type=float, default=0.96)
    rg.add_argument("--returning-min-prior",     type=int, default=2)
    rg.add_argument("--returning-per-entity-cooldown-min", type=float, default=30.0)
    args = ap.parse_args()

    if args.prune_snapshots:
        import shutil
        n = 0
        for d in [ANOMALY_DIR, RETURNING_DIR]:
            if d.is_dir():
                for sub in d.iterdir():
                    if sub.is_dir():
                        for f in sub.iterdir():
                            if f.is_file():
                                try: f.unlink(); n += 1
                                except Exception: pass
        print(f"Pruned {n} existing snapshot files from web/snapshots/.")

    if args.interval < MIN_INTERVAL_S:
        print(f"--interval {args.interval}s is below the {MIN_INTERVAL_S}s floor; "
              f"clamping to {MIN_INTERVAL_S}s.")
        args.interval = MIN_INTERVAL_S

    from app.firebase_store import FirebaseStore
    firebase = FirebaseStore()
    print(f"Firebase backend initialized. "
          f"Storage: {'ON' if firebase.storage else 'off (local disk fallback)'}")

    model = load_model(args.weights)

    # Build one picker per slot.
    pickers = {s["slot_id"]: SlotStreamPicker(s) for s in GRID_SLOTS}

    reid = None
    if not args.no_reid:
        reid = ReidStore(args.reid_db, threshold=args.reid_threshold)

    anomaly = AnomalyTracker(
        window       = args.anomaly_window,
        z_threshold  = args.anomaly_z,
        min_people   = args.anomaly_min_people,
        min_delta    = args.anomaly_min_delta,
        min_std      = args.anomaly_min_std,
        spike_only   = not args.anomaly_allow_drops,
        cooldown_sec = args.anomaly_cooldown,
    )
    save_snapshots          = not args.no_snapshots
    returning_gap_sec       = args.returning_gap_min * 60
    returning_cooldown_sec  = args.returning_per_entity_cooldown_min * 60

    # Publish the initial grid config so the dashboard renders immediately.
    slots_meta = [_slot_metadata(s, pickers[s["slot_id"]].current_cam()) for s in GRID_SLOTS]
    firebase.write_grid_config(slots_meta)

    print(f"Collector started. {len(GRID_SLOTS)} slot(s):")
    for slot in GRID_SLOTS:
        chain = " -> ".join([slot["primary"], *slot["fallbacks"]])
        print(f"  {slot['slot_id']:20s} = {chain}")
    print(f"interval={args.interval}s, reid={'on' if reid else 'off'}, "
          f"conf={args.conf}, snapshots={'on' if save_snapshots else 'off'}")
    print(f"fallback: {FALLBACK_MAX_FAILURES} misses to advance, "
          f"retry primary every {FALLBACK_RETRY_MINUTES} min.")

    writes_per_round = len(GRID_SLOTS) * (3 if reid else 2)
    projected = writes_per_round * (86400 / args.interval)
    print(f"~{projected:,.0f} Firestore writes/day projected "
          f"(free tier ~ {FREE_TIER_WRITES_PER_DAY:,}).")
    if projected > FREE_TIER_WRITES_PER_DAY:
        print("  ! Above the Spark free tier - on Blaze this stays in the free tier "
              "quotas; set a budget alert to be safe.")
    print("Ctrl+C to stop.\n")

    try:
        while True:
            round_start = time.time()
            for slot in GRID_SLOTS:
                picker = pickers[slot["slot_id"]]
                cam_id = picker.current_cam()
                ok = sample_slot(model, slot, cam_id, firebase, reid=reid,
                                 conf=args.conf, anomaly=anomaly,
                                 save_snapshots=save_snapshots,
                                 returning_gap_sec      = returning_gap_sec,
                                 returning_sim_min      = args.returning_min_similarity,
                                 returning_min_prior    = args.returning_min_prior,
                                 returning_cooldown_sec = returning_cooldown_sec)
                changed = picker.record_result(ok)
                if changed is not None:
                    print(f"  * {slot['slot_id']}: fallback -> {changed}")
                    slots_meta = [_slot_metadata(s, pickers[s["slot_id"]].current_cam())
                                  for s in GRID_SLOTS]
                    try:
                        firebase.write_grid_config(slots_meta)
                    except Exception as e:
                        print(f"  ! grid config write failed: {e}")
            time.sleep(max(0, args.interval - (time.time() - round_start)))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if reid is not None:
            reid.close()


if __name__ == "__main__":
    main()
