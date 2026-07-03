"""Continuous footfall collector - pushes live YOLO counts to Firestore.

A Jupyter cell runs once and stops. This process runs forever: every `--interval`
seconds it iterates the four GRID_SLOTS, picks each slot's currently-healthy
camera (with fallback), runs YOLO on a short frame burst (median count), updates
the re-ID registry, and writes the result to Firestore (keyed by slot_id, not
cam_id). The HTML dashboard subscribes via onSnapshot and updates in real time.

Anomaly detection, two layers x two metrics (people AND vehicles):
  * rolling window (median + MAD robust z): sudden spikes, and drops below a
    busy baseline - "the street just emptied / flooded vs 20 minutes ago";
  * hour-of-week profile (Welford mean/std per (dow, hour), Turkey local time):
    contextual anomalies - "this is not what a Wednesday 14:00 looks like
    here". Persisted to Firestore so restarts keep the learned baseline.
Both layers restore their state from Firestore on startup, so a service
restart doesn't re-warm from zero.

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
import math
import time
from pathlib import Path

import cv2
import numpy as np

from app.cameras import CAMERAS, GRID_SLOTS
from app.detect_core import (
    CLASSES_OF_INTEREST,
    DEFAULT_IMGSZ,
    annotate,
    detect_burst,
    grab_burst,
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

# ---- Anomaly metrics ---------------------------------------------------------
# "Business activity" on these cameras is foot traffic AND vehicle traffic, so
# the collector tracks the two series independently, each with gates scaled to
# its typical magnitude. A spike of buses at the otogar is exactly as
# reportable as a crowd at the market.
ANOMALY_METRICS = {
    "person":   dict(min_value=5, min_delta=5.0, drop_min_baseline=8.0),
    "vehicles": dict(min_value=4, min_delta=4.0, drop_min_baseline=6.0),
}

# Mean-gray threshold under which a frame is tagged `is_night` (the Konya cams
# switch to sodium lighting; day/night baselines differ a lot).
NIGHT_MEAN_GRAY = 60.0

TURKEY_TZ = dt.timezone(dt.timedelta(hours=3))  # permanent UTC+3, no DST
_DOW = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


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


def _median(xs) -> float:
    s = sorted(xs)
    m = len(s)
    return float(s[m // 2]) if m % 2 else (s[m // 2 - 1] + s[m // 2]) / 2.0


def robust_stats(xs) -> tuple[float, float]:
    """(median, MAD-based spread). 1.4826*MAD estimates std for normal data,
    but unlike mean/std it barely moves when a few outliers sit in the window -
    so one previous spike in the history can no longer mask the next one, and
    a single decode glitch can no longer trigger a false alarm."""
    med = _median(xs)
    mad = _median([abs(x - med) for x in xs])
    return med, 1.4826 * mad


class AnomalyTracker:
    """Rolling-window detector for ONE metric using robust (median/MAD) z-scores.

    Verdict kinds:
      'spike' - value far ABOVE the recent window (crowd surge, gathering);
      'drop'  - value far BELOW a busy window (street suddenly emptied). Only
                fires when the recent median is itself substantial
                (drop_min_baseline), so a quiet street going to zero is silent.

    All gates must pass, then a per-key cooldown throttles repeats. Keys are
    slot_ids: the window follows the dashboard tile, not the physical camera,
    so a fallback swap doesn't reset it.
    """

    def __init__(self, metric: str = "person", window: int = 30, warmup: int = 10,
                 z_spike: float = 3.5, z_drop: float = 3.0,
                 min_value: float = 5, min_delta: float = 5.0,
                 drop_min_baseline: float = 8.0, mad_floor: float = 1.0,
                 cooldown_sec: float = 300, **legacy):
        # Pre-robust kwarg names still arrive from older callers/notebooks.
        if "z_threshold" in legacy:
            z_spike = legacy.pop("z_threshold")
        if "min_people" in legacy:
            min_value = legacy.pop("min_people")
        legacy.pop("spike_only", None)   # drops now have their own gates
        legacy.pop("min_std", None)      # superseded by mad_floor
        if legacy:
            raise TypeError(f"unknown AnomalyTracker kwargs: {sorted(legacy)}")
        self.metric            = metric
        self.window            = window
        self.warmup            = warmup
        self.z_spike           = z_spike
        self.z_drop            = z_drop
        self.min_value         = min_value
        self.min_delta         = min_delta
        self.drop_min_baseline = drop_min_baseline
        self.mad_floor         = mad_floor
        self.cooldown_sec      = cooldown_sec
        self._history: dict[str, list[float]] = {}
        self._last_flagged: dict[str, float] = {}

    def seed(self, key: str, values) -> int:
        """Preload the rolling window from persisted history (restart recovery)
        without evaluating any of the values. Returns how many were kept."""
        vals = [float(v) for v in values if v is not None][-self.window:]
        self._history[key] = vals
        return len(vals)

    def push_and_check(self, key: str, value: float | None) -> tuple[bool, dict]:
        if value is None:
            return False, {"reason": "no_sample"}
        hist = self._history.setdefault(key, [])
        debug: dict = {"metric": self.metric, "window_size": len(hist),
                       "value": float(value)}
        try:
            if len(hist) < self.warmup:
                return False, {**debug, "reason": "warmup"}
            med, spread = robust_stats(hist)
            spread = max(spread, self.mad_floor)
            delta = value - med
            z = delta / spread
            debug.update({"median": round(med, 2), "spread": round(spread, 2),
                          "delta": round(delta, 2), "z": round(z, 2)})
            kind = None
            if (delta > 0 and value >= self.min_value
                    and delta >= self.min_delta and z >= self.z_spike):
                kind = "spike"
            elif (delta < 0 and med >= self.drop_min_baseline
                    and -delta >= self.min_delta and z <= -self.z_drop):
                kind = "drop"
            if kind is None:
                return False, {**debug, "reason": "within_norm"}
            now  = time.time()
            last = self._last_flagged.get(key, 0.0)
            if now - last < self.cooldown_sec:
                return False, {**debug, "reason": "cooldown", "suppressed_kind": kind,
                               "cooldown_remaining": round(self.cooldown_sec - (now - last), 1)}
            self._last_flagged[key] = now
            return True, {**debug, "reason": "anomaly", "kind": kind,
                          "expected": round(med, 1)}
        finally:
            hist.append(float(value))
            if len(hist) > self.window:
                hist.pop(0)


class HourlyProfile:
    """Hour-of-week baseline per (slot, metric): what is NORMAL here on a
    Wednesday at 14:00?

    The rolling window only remembers the last ~20 minutes, so a slow build-up
    to an abnormal level - or a street inexplicably dead at rush hour - passes
    it silently. This profile keeps a running mean/std (Welford) for each of
    the 7x24 hour buckets in Turkey local time (UTC+3) and flags values far
    outside the bucket's history once the bucket has enough samples to trust
    (min_samples). Verdict kinds: 'contextual_spike' / 'contextual_drop'.

    Persisted to Firestore (config/profile_{slot_id}) so restarts don't lose
    days of learned baseline.
    """

    def __init__(self, min_samples: int = 10, z_spike: float = 3.5,
                 z_drop: float = 3.0, std_floor: float = 1.0,
                 cooldown_sec: float = 1800):
        self.min_samples  = min_samples
        self.z_spike      = z_spike
        self.z_drop       = z_drop
        self.std_floor    = std_floor
        self.cooldown_sec = cooldown_sec
        # slot_id -> metric -> "dow_hour" -> [n, mean, m2]  (Welford accumulator)
        self._slots: dict[str, dict[str, dict[str, list[float]]]] = {}
        self._last_flagged: dict[tuple[str, str], float] = {}

    @staticmethod
    def bucket_of(ts_utc: dt.datetime) -> tuple[str, str]:
        local = ts_utc.astimezone(TURKEY_TZ)
        return f"{local.weekday()}_{local.hour}", f"{_DOW[local.weekday()]} {local.hour:02d}:00"

    def stats(self, slot_id: str, metric: str,
              ts_utc: dt.datetime) -> tuple[str, str, int, float, float]:
        """(bucket, label, n, mean, std) for the bucket this timestamp falls in."""
        bucket, label = self.bucket_of(ts_utc)
        cell = self._slots.get(slot_id, {}).get(metric, {}).get(bucket)
        if not cell or cell[0] < 1:
            return bucket, label, 0, 0.0, 0.0
        n, mean, m2 = cell
        std = math.sqrt(m2 / n) if n > 1 else 0.0
        return bucket, label, int(n), mean, std

    def check(self, slot_id: str, metric: str, ts_utc: dt.datetime,
              value: float | None, *, min_delta: float,
              drop_min_baseline: float) -> tuple[bool, dict]:
        """Evaluate `value` against its hour-of-week bucket (does NOT update the
        bucket - call update() afterwards so a value never scores itself)."""
        bucket, label, n, mean, std = self.stats(slot_id, metric, ts_utc)
        debug: dict = {"metric": metric, "bucket": label, "bucket_n": n,
                       "bucket_mean": round(mean, 2), "bucket_std": round(std, 2)}
        if value is None:
            return False, {**debug, "reason": "no_sample"}
        if n < self.min_samples:
            return False, {**debug, "reason": "bucket_warmup"}
        # Floor the spread: tiny-count buckets are near-deterministic and a
        # +2 change would otherwise score as a huge z.
        spread = max(std, self.std_floor, 0.15 * mean)
        delta = value - mean
        z = delta / spread
        debug.update({"z": round(z, 2), "delta": round(delta, 2)})
        kind = None
        if delta > 0 and delta >= min_delta and z >= self.z_spike:
            kind = "contextual_spike"
        elif (delta < 0 and mean >= drop_min_baseline
                and -delta >= min_delta and z <= -self.z_drop):
            kind = "contextual_drop"
        if kind is None:
            return False, {**debug, "reason": "within_norm"}
        now  = time.time()
        key  = (slot_id, metric)
        last = self._last_flagged.get(key, 0.0)
        if now - last < self.cooldown_sec:
            return False, {**debug, "reason": "cooldown", "suppressed_kind": kind}
        self._last_flagged[key] = now
        return True, {**debug, "reason": "anomaly", "kind": kind,
                      "expected": round(mean, 1)}

    def update(self, slot_id: str, metric: str, ts_utc: dt.datetime,
               value: float | None) -> None:
        if value is None:
            return
        bucket, _ = self.bucket_of(ts_utc)
        cell = (self._slots.setdefault(slot_id, {})
                .setdefault(metric, {})
                .setdefault(bucket, [0, 0.0, 0.0]))
        cell[0] += 1
        d = value - cell[1]
        cell[1] += d / cell[0]
        cell[2] += d * (value - cell[1])

    # ---- persistence -------------------------------------------------------

    def to_payload(self, slot_id: str) -> dict:
        metrics = {}
        for metric, buckets in self._slots.get(slot_id, {}).items():
            metrics[metric] = {b: {"n": c[0], "mean": c[1], "m2": c[2]}
                               for b, c in buckets.items()}
        return {"slot": slot_id, "tz": "UTC+3", "metrics": metrics}

    def load_payload(self, slot_id: str, payload: dict) -> int:
        """Merge a persisted payload back in. Returns #buckets loaded."""
        loaded = 0
        for metric, buckets in (payload.get("metrics") or {}).items():
            dst = self._slots.setdefault(slot_id, {}).setdefault(metric, {})
            for b, c in (buckets or {}).items():
                try:
                    dst[b] = [int(c["n"]), float(c["mean"]), float(c["m2"])]
                    loaded += 1
                except (KeyError, TypeError, ValueError):
                    continue
        return loaded


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
                           frame, model, conf: float, firebase,
                           imgsz: int | None = None) -> dict:
    """Save raw + annotated frames. Uses Storage if configured, else local disk."""
    stem = _ts_filename(ts_iso)
    raw_ok, raw_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not raw_ok:
        return {}
    urls = {"snapshot_url": None, "snapshot_annotated_url": None}
    try:
        annotated_frame = annotate(model, frame, conf=conf, imgsz=imgsz)
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
                reid: ReidStore | None = None, conf: float = 0.30,
                anomaly=None,
                profile: HourlyProfile | None = None,
                imgsz: int | None = DEFAULT_IMGSZ,
                burst: int = 3, burst_stride: int = 25,
                save_snapshots: bool = True,
                returning_gap_sec: float = RETURNING_GAP_SEC,
                returning_sim_min: float = RETURNING_MIN_SIMILARITY,
                returning_min_prior: int  = RETURNING_MIN_PRIOR_SIGHTINGS,
                returning_cooldown_sec: float = RETURNING_PER_ENTITY_COOLDOWN,
                _returning_last_save: dict | None = None) -> bool:
    """Sample the currently-active cam for a slot and write to Firestore.

    Detection runs on a short frame burst and keeps the median count (see
    detect_core.detect_burst), so a single noisy frame can no longer move the
    number the dashboard shows. `anomaly` accepts either one AnomalyTracker
    (legacy people-only callers) or a {metric: tracker} dict; `profile` adds
    the hour-of-week contextual check on top of the rolling window.

    A camera can carry its own calibrated "conf" (see cameras.py); otherwise
    the global `conf` applies.

    Returns True iff a frame was grabbed and processed successfully. The
    caller feeds this back to the SlotStreamPicker to decide whether to
    advance the fallback chain.
    """
    slot_id = slot["slot_id"]
    now_utc = dt.datetime.now(dt.timezone.utc)
    ts = now_utc.isoformat()
    new_ids: list[int] = []
    seen_again: list[int] = []
    frame = None
    burst_dbg: dict = {}
    cam = CAMERAS.get(cam_id)
    if cam is None:
        print(f"[{ts}] {slot_id}: unknown cam_id {cam_id!r}, skipping")
        return False
    cam_conf = cam.get("conf", conf)

    trackers: dict[str, AnomalyTracker] | None
    if anomaly is None:
        trackers = None
    elif isinstance(anomaly, AnomalyTracker):
        trackers = {anomaly.metric: anomaly}
    else:
        trackers = anomaly

    if _returning_last_save is None:
        _returning_last_save = getattr(sample_slot, "_returning_state",
                                       {}).setdefault(slot_id, {})
        sample_slot._returning_state = getattr(sample_slot, "_returning_state", {})
        sample_slot._returning_state[slot_id] = _returning_last_save

    try:
        frames = grab_burst(resolve_stream(cam), n=burst, stride=burst_stride)
        if not frames:
            raise RuntimeError("empty frame")
        counts, boxes, frame, burst_dbg = detect_burst(model, frames,
                                                       conf=cam_conf, imgsz=imgsz)
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
    if ok:
        record["burst"] = burst_dbg
        # Day/night tag: lets the dashboard and any offline analysis split
        # baselines - the same street has very different "normal" after dark.
        record["is_night"] = bool(
            float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) < NIGHT_MEAN_GRAY)

    # Anomaly gating keyed by slot (not cam) so a fallback swap doesn't reset
    # the windows — the DASHBOARD tile is what we're comparing to itself.
    if trackers is not None and ok:
        verdicts: list[dict] = []
        for metric, tracker in trackers.items():
            value = counts.get(metric)
            flagged, dbg = tracker.push_and_check(slot_id, value)
            if flagged:
                verdicts.append({"kind": dbg["kind"], "metric": metric,
                                 "window": "rolling", "z": dbg.get("z"),
                                 "observed": value, "expected": dbg.get("expected")})
            ctx_flagged, ctx_dbg = False, {}
            if profile is not None:
                gates = ANOMALY_METRICS.get(metric, ANOMALY_METRICS["person"])
                ctx_flagged, ctx_dbg = profile.check(
                    slot_id, metric, now_utc, value,
                    min_delta=gates["min_delta"],
                    drop_min_baseline=gates["drop_min_baseline"])
                if ctx_flagged:
                    verdicts.append({"kind": ctx_dbg["kind"], "metric": metric,
                                     "window": "hourly", "z": ctx_dbg.get("z"),
                                     "observed": value,
                                     "expected": ctx_dbg.get("expected"),
                                     "bucket": ctx_dbg.get("bucket")})
                # Keep flagged (or cooldown-suppressed) samples OUT of the
                # baseline so an ongoing event can't normalize itself into it.
                if (not flagged and not ctx_flagged
                        and "suppressed_kind" not in dbg
                        and "suppressed_kind" not in ctx_dbg):
                    profile.update(slot_id, metric, now_utc, value)
        record["is_anomaly"] = bool(verdicts)
        if verdicts:
            # Rolling verdicts outrank hourly ones for the headline fields;
            # everything else lands under "also".
            verdicts.sort(key=lambda v: 0 if v["window"] == "rolling" else 1)
            primary = dict(verdicts[0])
            if len(verdicts) > 1:
                primary["also"] = [{k: v.get(k) for k in ("kind", "metric", "z")}
                                   for v in verdicts[1:]]
            record["anomaly"] = primary
            if save_snapshots and frame is not None:
                try:
                    record.update(_save_anomaly_snapshot(
                        slot_id, cam_id, ts, frame, model, cam_conf, firebase,
                        imgsz=imgsz))
                    print(f"  ! {primary['kind']} @ {slot_id}/{cam_id} "
                          f"[{primary['metric']}] z={primary.get('z')}, "
                          f"observed={primary.get('observed')} vs "
                          f"~{primary.get('expected')} expected - snapshot saved")
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
        flag  = (f"  ANOMALY[{record['anomaly']['kind']}/{record['anomaly']['metric']}]"
                 if record.get("is_anomaly") else "")
        n_burst = len(burst_dbg.get("burst_person") or [])
        print(f"[{ts}] {slot_id} ({cam_id}): person={counts['person']} "
              f"vehicles={counts['vehicles']} burst={n_burst}{extra}{flag}")
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


def _restore_state(firebase, trackers: dict[str, AnomalyTracker],
                   profile: HourlyProfile | None, slot_ids: set[str]) -> None:
    """Rebuild in-memory analysis state from Firestore after a (re)start.

    1. Rolling windows: the last hour of footfall docs reseeds each slot's
       window, so anomaly detection resumes on the first sample instead of
       re-warming for `warmup` samples after every service restart.
    2. Hourly profiles: persisted per-slot profile docs are loaded; slots
       without one (first deploy) bootstrap from the last 24h of history and
       the bootstrap result is saved back immediately.
    """
    now = dt.datetime.now(dt.timezone.utc)
    try:
        since = (now - dt.timedelta(hours=1)).isoformat()
        docs = firebase.recent_history(since, limit_docs=600)
        by_slot: dict[str, list[dict]] = {}
        for d in docs:
            if d.get("ok") and d.get("slot") in slot_ids:
                by_slot.setdefault(d["slot"], []).append(d)
        for sid, rows in by_slot.items():
            rows.sort(key=lambda r: r.get("ts") or "")
            for metric, tracker in trackers.items():
                kept = tracker.seed(sid, [r.get(metric) for r in rows])
                if kept:
                    print(f"  restored {kept} samples -> {sid}/{metric} window")
    except Exception as e:
        print(f"  ! rolling-window restore skipped ({e})")

    if profile is None:
        return
    missing: list[str] = []
    for sid in sorted(slot_ids):
        payload = None
        try:
            payload = firebase.load_slot_profile(sid)
        except Exception as e:
            print(f"  ! profile load failed for {sid} ({e})")
        if payload and profile.load_payload(sid, payload):
            print(f"  loaded hourly profile for {sid}")
        else:
            missing.append(sid)
    if not missing:
        return
    try:
        since = (now - dt.timedelta(hours=24)).isoformat()
        docs = firebase.recent_history(since, limit_docs=10_000)
        n = 0
        for d in docs:
            sid = d.get("slot")
            if not d.get("ok") or sid not in missing:
                continue
            try:
                ts = dt.datetime.fromisoformat(str(d.get("ts")).replace("Z", "+00:00"))
            except ValueError:
                continue
            for metric in trackers:
                profile.update(sid, metric, ts, d.get(metric))
            n += 1
        print(f"  bootstrapped hourly profile for {', '.join(missing)} "
              f"from {n} history docs")
        for sid in missing:
            try:
                firebase.save_slot_profile(sid, profile.to_payload(sid))
            except Exception as e:
                print(f"  ! profile save failed for {sid} ({e})")
    except Exception as e:
        print(f"  ! profile bootstrap skipped ({e})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Continuous YOLO footfall collector "
                                             "(writes to Firestore + Storage for the HTML dashboard)")
    ap.add_argument("--interval", type=int, default=20, help="seconds between sampling rounds")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                    help="YOLO input size. 960 recovers the small/distant "
                         "pedestrians and cars these wide street shots are full "
                         "of; set 640 if CPU/RAM constrained")
    ap.add_argument("--burst", type=int, default=3,
                    help="frames per sample; the reported count is the burst median")
    ap.add_argument("--burst-stride", type=int, default=25,
                    help="frames skipped between burst frames (~1s at 25fps)")
    ap.add_argument("--reid-db", default="data/reid.db",
                    help="local SQLite path for the appearance-based re-ID registry")
    ap.add_argument("--no-reid", action="store_true", help="disable re-identification")
    ap.add_argument("--reid-threshold", type=float, default=0.92,
                    help="cosine similarity above which a detection is 'seen before'")
    ap.add_argument("--reid-prune-hours", type=float, default=48.0,
                    help="delete re-ID entities not seen for this many hours")
    ap.add_argument("--conf", type=float, default=0.30,
                    help="YOLO confidence threshold (cameras.py entries may "
                         "override per camera)")
    ap.add_argument("--no-snapshots", action="store_true",
                    help="skip anomaly / returning-visitor image saves")
    ap.add_argument("--prune-snapshots", action="store_true",
                    help="delete every file under web/snapshots/{anomalies,returning}/* "
                         "before starting (local mode only; Storage cleanup uses the "
                         "lifecycle rule)")
    ag = ap.add_argument_group("anomaly gating (rolling window, robust z on median+MAD)")
    ag.add_argument("--anomaly-z",          type=float, default=3.5,
                    help="robust z a spike must reach")
    ag.add_argument("--anomaly-drop-z",     type=float, default=3.0,
                    help="robust |z| a drop must reach (below a busy baseline)")
    ag.add_argument("--anomaly-window",     type=int,   default=30)
    ag.add_argument("--anomaly-min-people", type=int,   default=5)
    ag.add_argument("--anomaly-min-delta",  type=float, default=5.0)
    ag.add_argument("--anomaly-cooldown",   type=float, default=300.0)
    pg = ap.add_argument_group("hour-of-week contextual baseline")
    pg.add_argument("--no-profile", action="store_true",
                    help="disable the hour-of-week contextual anomaly check")
    pg.add_argument("--profile-min-samples", type=int, default=10,
                    help="bucket samples required before contextual checks fire")
    pg.add_argument("--profile-save-min", type=float, default=30.0,
                    help="minutes between profile persists to Firestore")
    rg = ap.add_argument_group("returning-visitor gating")
    rg.add_argument("--returning-gap-min",       type=float, default=15.0)
    rg.add_argument("--returning-min-similarity", type=float, default=0.96)
    rg.add_argument("--returning-min-prior",     type=int, default=2)
    rg.add_argument("--returning-per-entity-cooldown-min", type=float, default=30.0)
    args = ap.parse_args()

    if args.prune_snapshots:
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
        try:
            removed = reid.prune(max_age_hours=args.reid_prune_hours)
            if removed:
                print(f"reid: pruned {removed} entities idle > {args.reid_prune_hours:g}h")
        except Exception as e:
            print(f"reid: prune failed ({e})")

    # One rolling tracker per metric; gates scale to each metric's magnitude.
    trackers: dict[str, AnomalyTracker] = {}
    for metric, gates in ANOMALY_METRICS.items():
        overrides = (dict(min_value=args.anomaly_min_people,
                          min_delta=args.anomaly_min_delta)
                     if metric == "person"
                     else dict(min_value=gates["min_value"],
                               min_delta=gates["min_delta"]))
        trackers[metric] = AnomalyTracker(
            metric            = metric,
            window            = args.anomaly_window,
            z_spike           = args.anomaly_z,
            z_drop            = args.anomaly_drop_z,
            drop_min_baseline = gates["drop_min_baseline"],
            cooldown_sec      = args.anomaly_cooldown,
            **overrides,
        )

    profile = None if args.no_profile else HourlyProfile(
        min_samples = args.profile_min_samples,
        z_spike     = args.anomaly_z,
        z_drop      = args.anomaly_drop_z,
    )

    save_snapshots          = not args.no_snapshots
    returning_gap_sec       = args.returning_gap_min * 60
    returning_cooldown_sec  = args.returning_per_entity_cooldown_min * 60

    print("Restoring analysis state from Firestore...")
    _restore_state(firebase, trackers, profile, set(pickers))

    # Publish the initial grid config so the dashboard renders immediately.
    slots_meta = [_slot_metadata(s, pickers[s["slot_id"]].current_cam()) for s in GRID_SLOTS]
    firebase.write_grid_config(slots_meta)

    print(f"Collector started. {len(GRID_SLOTS)} slot(s):")
    for slot in GRID_SLOTS:
        chain = " -> ".join([slot["primary"], *slot["fallbacks"]])
        print(f"  {slot['slot_id']:20s} = {chain}")
    print(f"interval={args.interval}s, imgsz={args.imgsz}, burst={args.burst}, "
          f"reid={'on' if reid else 'off'}, conf={args.conf}, "
          f"snapshots={'on' if save_snapshots else 'off'}")
    print(f"anomaly metrics: {', '.join(trackers)} | rolling robust-z "
          f"(spike>={args.anomaly_z}, drop<=-{args.anomaly_drop_z}) | "
          f"hour-of-week profile: {'on' if profile else 'off'}")
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

    profile_save_s   = max(60.0, args.profile_save_min * 60)
    reid_prune_s     = 6 * 3600
    last_profile_save = time.time()
    last_reid_prune   = time.time()

    def _persist_profiles() -> None:
        if profile is None:
            return
        for sid in pickers:
            try:
                firebase.save_slot_profile(sid, profile.to_payload(sid))
            except Exception as e:
                print(f"  ! profile save failed for {sid} ({e})")

    try:
        while True:
            round_start = time.time()
            for slot in GRID_SLOTS:
                picker = pickers[slot["slot_id"]]
                cam_id = picker.current_cam()
                ok = sample_slot(model, slot, cam_id, firebase, reid=reid,
                                 conf=args.conf, anomaly=trackers,
                                 profile=profile, imgsz=args.imgsz,
                                 burst=args.burst, burst_stride=args.burst_stride,
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
            if profile is not None and time.time() - last_profile_save >= profile_save_s:
                _persist_profiles()
                last_profile_save = time.time()
            if reid is not None and time.time() - last_reid_prune >= reid_prune_s:
                try:
                    removed = reid.prune(max_age_hours=args.reid_prune_hours)
                    if removed:
                        print(f"  reid: pruned {removed} stale entities")
                except Exception as e:
                    print(f"  ! reid prune failed: {e}")
                last_reid_prune = time.time()
            time.sleep(max(0, args.interval - (time.time() - round_start)))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        _persist_profiles()   # don't lose up to 30 min of learned baseline
        if reid is not None:
            reid.close()


if __name__ == "__main__":
    main()
