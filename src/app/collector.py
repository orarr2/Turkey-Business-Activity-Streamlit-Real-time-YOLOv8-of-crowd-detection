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
# Anchored relative to this file (src/app/collector.py -> src/web/snapshots)
# so the paths resolve correctly regardless of which directory the user runs
# the collector from (`python -m app.collector` from src/, or via the notebook).
_SRC_ROOT          = Path(__file__).resolve().parent.parent
SNAPSHOTS_ROOT     = _SRC_ROOT / "web" / "snapshots"
ANOMALY_DIR        = SNAPSHOTS_ROOT / "anomalies"
RETURNING_DIR      = SNAPSHOTS_ROOT / "returning"

# ---- Returning-visitor gates (each saved image is a real return event) -----
# The previous defaults (gap >= 5 min, no other gates) flooded the folders.
# These four gates intersect to require: long absence + strong appearance
# match + a real history (not a first-time match) + per-entity cooldown so
# the same entity doesn't get saved every cycle.
RETURNING_GAP_SEC              = 900   # >= 15 min absence (was 5 min)
RETURNING_MIN_SIMILARITY       = 0.96  # >= 0.96 cosine (was 0.92 default match)
RETURNING_MIN_PRIOR_SIGHTINGS  = 2     # entity must have been seen >= 2 times before
RETURNING_PER_ENTITY_COOLDOWN  = 1800  # don't save same eid more than once per 30 min


class AnomalyTracker:
    """Per-camera anomaly detector with layered gates that ALL must pass.

    The bare z-score (the previous default of z>2.5 over a 12-sample window)
    fires on every minor wobble during quiet periods - 1 person walking past
    an empty plaza yields z=8 because the baseline std was 0.3. This class
    intersects six gates so only meaningful crowd spikes light up:

      * window           : larger context (default 30 samples = 10 min)
      * z_threshold      : higher significance bar (default 3.5, was 2.5)
      * spike_only       : ignore drops to zero (occlusion / detection misses)
      * min_people       : absolute floor on the spike value (default 5 people)
      * min_delta        : spike must exceed baseline by at least this many
                           people (default 5) - kills the "noise math" gate
      * min_std          : optional opt-in gate. Set > 0 only when you also
                           want to suppress *busy-baseline-but-low-variance*
                           cameras; the other gates already kill the
                           low-activity noise cases without it (default 0).
      * cooldown_sec     : after a flagged anomaly, suppress flags on the same
                           camera for cooldown_sec (default 300 = 5 min) - one
                           real event no longer paints the next 4 frames red

    All thresholds are CLI-tunable (see --anomaly-* flags). The dashboard's
    JS-side fallback in web/app.js applies the same gates so legacy rows
    written before this fix don't show up as anomalies either.
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
        # last "real time" we flagged a given camera, for the cooldown gate.
        self._last_flagged: dict[str, float] = {}

    def push_and_check(self, cam_id: str, people: int | None) -> tuple[bool, dict]:
        """Append `people` and report whether it is a *real* anomaly.

        Returns (is_anomaly, debug) where debug names the gate that decided.
        """
        if people is None:
            return False, {"reason": "no_sample"}
        hist = self._history.setdefault(cam_id, [])
        debug: dict = {"window_size": len(hist), "people": int(people)}

        # Score *before* appending - we compare the new value to the window
        # that preceded it (otherwise the value would skew its own baseline).
        try:
            if len(hist) < self.warmup:
                return False, {**debug, "reason": "warmup"}
            mu  = sum(hist) / len(hist)
            sd  = (sum((x - mu) ** 2 for x in hist) / len(hist)) ** 0.5
            delta = people - mu
            z = (delta / sd) if sd > 0 else 0.0
            debug.update({"mean": round(mu, 2), "std": round(sd, 2),
                          "delta": round(delta, 2), "z": round(z, 2)})

            # --- Gate 1: spike only (drops to zero are usually misses) ------
            if self.spike_only and delta <= 0:
                return False, {**debug, "reason": "not_a_spike"}
            # --- Gate 2: absolute floor on people in the spike --------------
            if people < self.min_people:
                return False, {**debug, "reason": "below_min_people"}
            # --- Gate 3: baseline must actually have variation --------------
            if sd < self.min_std:
                return False, {**debug, "reason": "quiet_baseline"}
            # --- Gate 4: spike must be substantial in absolute terms --------
            if abs(delta) < self.min_delta:
                return False, {**debug, "reason": "small_delta"}
            # --- Gate 5: z must clear the significance bar ------------------
            z_check = z if self.spike_only else abs(z)
            if z_check < self.z:
                return False, {**debug, "reason": "below_z"}
            # --- Gate 6: cooldown since the previous flag on this camera ---
            now  = time.time()
            last = self._last_flagged.get(cam_id, 0.0)
            if now - last < self.cooldown_sec:
                return False, {**debug, "reason": "cooldown",
                               "cooldown_remaining": round(self.cooldown_sec - (now - last), 1)}

            self._last_flagged[cam_id] = now
            return True, {**debug, "reason": "anomaly"}
        finally:
            hist.append(int(people))
            if len(hist) > self.window:
                hist.pop(0)


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


def _passes_returning_gates(r, gap_min_sec: float, sim_min: float,
                            min_prior: int, cooldown_sec: float,
                            last_save_for_eid: dict) -> tuple[bool, str]:
    """Return (passes, reason). All four gates must pass for a save.

    - r.gap_seconds  : absence since previous sighting at this camera
    - r.similarity   : cosine of the match (1.0 = brand new entity)
    - r.sightings    : sightings AFTER this match (so prior == sightings-1)
    - cooldown       : per-entity rate-limit to avoid runs of identical saves
    """
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


def sample_once(model, cam_id: str, cam: dict, firebase,
                reid: ReidStore | None = None, conf: float = 0.35,
                anomaly: AnomalyTracker | None = None,
                save_snapshots: bool = True,
                returning_gap_sec: float = RETURNING_GAP_SEC,
                returning_sim_min: float = RETURNING_MIN_SIMILARITY,
                returning_min_prior: int  = RETURNING_MIN_PRIOR_SIGHTINGS,
                returning_cooldown_sec: float = RETURNING_PER_ENTITY_COOLDOWN,
                _returning_last_save: dict | None = None) -> None:
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    new_ids: list[int] = []
    seen_again: list[int] = []
    frame = None
    # Per-entity-cooldown state. Caller can pin a shared dict across calls
    # via _returning_last_save; if omitted we attach one to sample_once itself.
    if _returning_last_save is None:
        _returning_last_save = getattr(sample_once, "_returning_state",
                                       {}).setdefault(cam_id, {})
        sample_once._returning_state = getattr(sample_once, "_returning_state", {})
        sample_once._returning_state[cam_id] = _returning_last_save
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
                # Save a "returning visitor" image only when ALL four gates
                # pass: long absence + strong appearance match + the entity
                # already has a history + per-entity cooldown isn't active.
                # This kills the previous flood where every consecutive match
                # produced a folder entry.
                if save_snapshots and i < len(boxes):
                    passes, _why = _passes_returning_gates(
                        r, returning_gap_sec, returning_sim_min,
                        returning_min_prior, returning_cooldown_sec,
                        _returning_last_save)
                    if passes:
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
    ap.add_argument("--prune-snapshots", action="store_true",
                    help="delete every file under web/snapshots/{anomalies,returning}/* "
                         "before starting - useful after tuning the gates")
    # ---- anomaly tuning ---------------------------------------------------
    ag = ap.add_argument_group("anomaly gating (each gate must pass for a snapshot)")
    ag.add_argument("--anomaly-z",          type=float, default=3.5,
                    help="z-score significance bar (default 3.5; was 2.5)")
    ag.add_argument("--anomaly-window",     type=int,   default=30,
                    help="rolling window length in samples (default 30 = 10 min @ 20s)")
    ag.add_argument("--anomaly-min-people", type=int,   default=5,
                    help="absolute people-count floor for the spike (default 5)")
    ag.add_argument("--anomaly-min-delta",  type=float, default=5.0,
                    help="spike must exceed baseline by this many people (default 5)")
    ag.add_argument("--anomaly-min-std",    type=float, default=0.0,
                    help="baseline must have std >= this (default 0 = no gate; only "
                         "enable if a stable busy camera is over-flagging)")
    ag.add_argument("--anomaly-cooldown",   type=float, default=300.0,
                    help="seconds between flagged anomalies per camera (default 300 = 5 min)")
    ag.add_argument("--anomaly-allow-drops", action="store_true",
                    help="also flag sharp DROPS (z<0); default is spikes only")
    # ---- returning-visitor tuning ----------------------------------------
    rg = ap.add_argument_group("returning-visitor gating")
    rg.add_argument("--returning-gap-min",       type=float, default=15.0,
                    help="minimum absence before a re-match counts as 'returning', "
                         "in minutes (default 15)")
    rg.add_argument("--returning-min-similarity", type=float, default=0.96,
                    help="match must be >= this cosine for save (default 0.96)")
    rg.add_argument("--returning-min-prior",     type=int, default=2,
                    help="entity must have >= this many prior sightings before save (default 2)")
    rg.add_argument("--returning-per-entity-cooldown-min", type=float, default=30.0,
                    help="don't save the same entity_id more than once per N minutes (default 30)")
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

    print(f"Collector started. {len(cams)} camera(s): {list(cams)}")
    print(f"interval={args.interval}s, reid={'on' if reid else 'off'}, "
          f"conf={args.conf}, snapshots={'on' if save_snapshots else 'off'}")
    print(f"anomaly gates: z>={args.anomaly_z}, window={args.anomaly_window}, "
          f"min_people={args.anomaly_min_people}, min_delta={args.anomaly_min_delta}, "
          f"min_std={args.anomaly_min_std}, cooldown={args.anomaly_cooldown}s")
    print(f"returning gates: gap>={args.returning_gap_min}min, "
          f"similarity>={args.returning_min_similarity}, "
          f"prior_sightings>={args.returning_min_prior}, "
          f"per-entity cooldown={args.returning_per_entity_cooldown_min}min")

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
                            save_snapshots=save_snapshots,
                            returning_gap_sec      = returning_gap_sec,
                            returning_sim_min      = args.returning_min_similarity,
                            returning_min_prior    = args.returning_min_prior,
                            returning_cooldown_sec = returning_cooldown_sec)
            time.sleep(max(0, args.interval - (time.time() - round_start)))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if reid is not None:
            reid.close()


if __name__ == "__main__":
    main()
