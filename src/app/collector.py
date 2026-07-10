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
Operational gating on top of the statistics: verdicts must persist for
`--anomaly-confirm` consecutive samples, the move must be large relative to
the scene's own baseline (not just in absolute terms), and both layers are
keyed to the PHYSICAL CAMERA - a fallback swap starts a short warmup on the
new scene instead of comparing two different streets against each other.
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
import bisect
import datetime as dt
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from app.cameras import CAMERAS, GRID_SLOTS
from app.alerts import AlertSink
from app.detect_core import (
    CLASSES_OF_INTEREST,
    DEFAULT_IMGSZ,
    box_iou,
    detect_burst,
    draw_boxes,
    grab_burst,
    load_model,
    resolve_stream,
)
from app.presence import PresenceTracker
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
EVENTS_DIR     = SNAPSHOTS_ROOT / "events"

# ---- Returning-visitor gates (each saved image is a real return event) -----
RETURNING_GAP_SEC              = 300   # >= 5 min absence (the declared behavior)
RETURNING_MIN_SIMILARITY       = 0.96  # >= 0.96 cosine
RETURNING_MIN_PRIOR_SIGHTINGS  = 2     # entity must have been seen >= 2 times
RETURNING_PER_ENTITY_COOLDOWN  = 1800  # same eid at most once per 30 min
# A "return" is only meaningful if we were actually watching during the
# absence. If the camera itself wasn't sampled for most of the entity's gap
# (outage, fallback episode), the entity never "left" - we just weren't
# looking. Reject the save when the unobserved fraction of the gap exceeds:
RETURNING_MAX_UNOBSERVED_FRAC  = 0.5
# Static objects (parked cars, banners detected as trucks) re-match in the
# same spot forever; a genuine return walks/drives back INTO the scene. If the
# entity's new box overlaps its previous box above this IoU, it never left.
RETURNING_STATIC_IOU           = 0.5

# ---- Anomaly metrics ---------------------------------------------------------
# "Business activity" on these cameras is foot traffic AND vehicle traffic, so
# the collector tracks the two series independently, each with gates scaled to
# its typical magnitude. A spike of buses at the otogar is exactly as
# reportable as a crowd at the market.
#
# Gates are deliberately conservative for OPERATIONAL use: an event must be
# large in absolute terms (min_value/min_delta), large RELATIVE to the scene's
# own baseline (rel_delta x median), and must persist for `confirm_samples`
# consecutive samples. A single family stepping off a bus is not an event; a
# crowd that is still there 40 s later is.
ANOMALY_METRICS = {
    "person":   dict(min_value=8, min_delta=5.0, drop_min_baseline=8.0),
    "vehicles": dict(min_value=6, min_delta=4.0, drop_min_baseline=6.0),
}
# How many anomaly verdicts per physical camera per day is "normal operations".
# Beyond this the collector logs a loud warning - the gates are miscalibrated
# for that scene and the operator should know before trusting the feed.
ANOMALY_BUDGET_PER_DAY = 8

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


class CamObservationLog:
    """Tracks WHEN each physical camera was successfully sampled, so the
    returning-visitor gate can tell a genuine absence ("we watched the scene
    the whole time and the entity was gone") from a blind spot ("the camera
    was down / the slot was on a fallback - nothing actually returned").

    Success timestamps are kept per cam (sorted) for `horizon_sec`; everything
    before the earliest known success is blind (we can't vouch for what
    happened before we were looking; seed() backfills from Firestore history
    after a restart so long-gap returns aren't suppressed for hours).

    A gap between consecutive successes counts as unobserved when it exceeds
    the hole threshold, which ADAPTS to the camera's real cadence
    (max(hole_threshold_sec, 4x the recent median inter-sample gap)) - on an
    undersized machine where every round takes minutes, a fixed 180s would
    classify normal sampling as one long hole and silently disable the
    returning-visitor feature.
    """

    def __init__(self, horizon_sec: float = 48 * 3600,
                 hole_threshold_sec: float = 180.0):
        self.horizon = horizon_sec
        self.hole_threshold = hole_threshold_sec
        self._successes: dict[str, list[float]] = {}

    def record_success(self, cam_id: str, ts: float | None = None) -> None:
        lst = self._successes.setdefault(cam_id, [])
        t = time.time() if ts is None else ts
        if lst and t < lst[-1]:
            bisect.insort(lst, t)          # out-of-order seed
        else:
            lst.append(t)
        # Amortized prune: only compact once >10% of the horizon is expired,
        # so a long-running collector isn't rebuilding a ~4k list every round.
        cutoff = t - self.horizon
        if lst[0] < cutoff - 0.1 * self.horizon:
            del lst[: bisect.bisect_left(lst, cutoff)]

    def seed(self, cam_id: str, epochs: list[float]) -> int:
        """Backfill success timestamps (restart recovery from history docs)."""
        if not epochs:
            return 0
        lst = self._successes.setdefault(cam_id, [])
        lst.extend(epochs)
        lst.sort()
        return len(epochs)

    def _hole_threshold_for(self, lst: list[float]) -> float:
        if len(lst) < 4:
            return self.hole_threshold
        recent = lst[-12:]
        gaps = sorted(b - a for a, b in zip(recent, recent[1:]))
        median_gap = gaps[len(gaps) // 2]
        return max(self.hole_threshold, 4.0 * median_gap)

    def unobserved_during(self, cam_id: str, window_sec: float) -> float:
        """Seconds of [now - window_sec, now] this camera was NOT observed."""
        now = time.time()
        t0 = now - window_sec
        lst = self._successes.get(cam_id) or []
        known_from = lst[0] if lst else now
        unobserved = max(0.0, min(known_from, now) - t0)   # pre-knowledge = blind
        prev = max(t0, known_from)
        hole = self._hole_threshold_for(lst)
        for i in range(bisect.bisect_left(lst, prev), len(lst)):
            gap = lst[i] - prev
            if gap > hole:
                unobserved += gap
            prev = lst[i]
        tail = now - prev
        if tail > hole:
            unobserved += tail
        return min(unobserved, window_sec)


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

    Keys are "{slot_id}|{cam_id}": the window belongs to the PHYSICAL SCENE.
    Keying by slot alone (the old behavior) meant a fallback swap compared a
    quiet park against a busy market's baseline and flagged a storm of fake
    drops/spikes in both directions; a swap now simply starts a short warmup
    on the new camera's own window.

    Gates, all of which must pass:
      * value >= min_value (absolute - tiny scenes can't alarm);
      * |delta| >= max(min_delta, rel_delta * median) - the move must be large
        relative to the scene's own level, so a group of 6 on a street whose
        median is 2 no longer counts as an "event";
      * robust |z| >= z_spike / z_drop with spread floored at mad_floor
        (near-constant windows have MAD 0; without a real floor any +2 change
        scores an absurd z);
      * the same verdict kind must repeat for `confirm_samples` CONSECUTIVE
        samples - a one-sample blip (bus unloading, decode glitch) is not an
        operational event;
    then a per-key cooldown throttles repeats.
    """

    def __init__(self, metric: str = "person", window: int = 30, warmup: int = 10,
                 z_spike: float = 3.5, z_drop: float = 3.0,
                 min_value: float = 5, min_delta: float = 5.0,
                 rel_delta: float = 0.8, confirm_samples: int = 2,
                 drop_min_baseline: float = 8.0, mad_floor: float = 2.0,
                 cooldown_sec: float = 300, stale_after_sec: float = 600,
                 **legacy):
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
        self.rel_delta         = rel_delta
        self.confirm_samples   = max(1, int(confirm_samples))
        self.drop_min_baseline = drop_min_baseline
        self.mad_floor         = mad_floor
        self.cooldown_sec      = cooldown_sec
        # A window that hasn't been fed for this long describes a different
        # regime (the slot was on a fallback for hours, the stream was down
        # over a day/night transition). It is cleared and re-warms instead of
        # scoring the present against a stale past.
        self.stale_after_sec   = stale_after_sec
        self._history: dict[str, list[float]] = {}
        self._last_flagged: dict[str, float] = {}
        self._last_push: dict[str, float] = {}
        # key -> (candidate kind, consecutive samples it has persisted)
        self._pending: dict[str, tuple[str, int]] = {}

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
        now_push  = time.time()
        last_push = self._last_push.get(key)
        self._last_push[key] = now_push
        stale = (last_push is not None
                 and now_push - last_push > self.stale_after_sec)
        if stale and hist:
            # e.g. the slot spent hours on a fallback and just returned to
            # this camera: the stored window is another time-of-day's regime.
            hist.clear()
            self._pending.pop(key, None)
        debug: dict = {"metric": self.metric, "window_size": len(hist),
                       "value": float(value)}
        if stale:
            debug["window_was_stale"] = True
        try:
            if len(hist) < self.warmup:
                self._pending.pop(key, None)
                return False, {**debug, "reason": "warmup"}
            med, spread = robust_stats(hist)
            spread = max(spread, self.mad_floor)
            delta = value - med
            z = delta / spread
            # Both the absolute AND the scene-relative move floors must pass.
            eff_min_delta = max(self.min_delta, self.rel_delta * med)
            debug.update({"median": round(med, 2), "spread": round(spread, 2),
                          "delta": round(delta, 2), "z": round(z, 2),
                          "min_delta_effective": round(eff_min_delta, 2)})
            kind = None
            if (delta > 0 and value >= self.min_value
                    and delta >= eff_min_delta and z >= self.z_spike):
                kind = "spike"
            elif (delta < 0 and med >= self.drop_min_baseline
                    and -delta >= eff_min_delta and z <= -self.z_drop):
                kind = "drop"
            if kind is None:
                self._pending.pop(key, None)
                return False, {**debug, "reason": "within_norm"}
            # Persistence gate: the SAME verdict kind must hold for
            # confirm_samples consecutive samples before it becomes an event.
            prev_kind, streak = self._pending.get(key, (kind, 0))
            streak = streak + 1 if prev_kind == kind else 1
            if streak < self.confirm_samples:
                self._pending[key] = (kind, streak)
                return False, {**debug, "reason": "pending_confirmation",
                               "candidate_kind": kind, "streak": streak,
                               "needed": self.confirm_samples}
            self._pending.pop(key, None)
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
    """Hour-of-week baseline per (camera, metric): what is NORMAL here on a
    Wednesday at 14:00?

    The rolling window only remembers the last ~20 minutes, so a slow build-up
    to an abnormal level - or a street inexplicably dead at rush hour - passes
    it silently. This profile keeps a running mean/std (Welford) for each of
    the 7x24 hour buckets in Turkey local time (UTC+3) and flags values far
    outside the bucket's history once the bucket has enough samples to trust
    (min_samples). Verdict kinds: 'contextual_spike' / 'contextual_drop'.
    Buckets are hour-of-week, so day/night regimes get separate baselines by
    construction (the `is_night` tag on records stays for offline analysis).

    Keys are cam_ids (the physical scene), persisted to Firestore
    (config/profile_{cam_id}) so restarts don't lose days of learned baseline.

    Learning uses CLIPPED updates (see update()): the old exclude-flagged
    policy meant a street that legitimately became busier never re-entered its
    own baseline and kept flagging forever - a measured, self-reinforcing
    false-positive loop. Clipping lets the bucket converge to a genuine new
    regime over days while a short spike still can't drag it.
    """

    # Self-healing rebase: when the DETECTOR itself changes regime (a
    # confidence threshold tuned looser by reviews, a higher imgsz, a better
    # model), every count jumps and every hour bucket's old mean is simply
    # wrong - the profile then cries "spike" on ordinary traffic for the
    # weeks the clipped updates need to converge. That exact failure was
    # observed live: 13+ "hourly spike vehicles observed 8-15 expected ~2-3"
    # events per day. When a (cam, metric) pair accumulates REBASE_AFTER
    # spike verdicts (including cooldown-suppressed ones) inside
    # REBASE_WINDOW_SEC, the pair's buckets get their effective sample count
    # cut to REBASE_N: new samples then land ~4x harder and the baseline
    # re-converges within a day instead of weeks. A real once-a-day spike
    # can't trigger this; sustained systematic disagreement does.
    REBASE_WINDOW_SEC = 24 * 3600
    REBASE_AFTER = 8
    REBASE_N = 30

    def __init__(self, min_samples: int = 30, z_spike: float = 3.5,
                 z_drop: float = 3.0, std_floor: float = 1.0,
                 cooldown_sec: float = 1800, clip_z: float = 3.0,
                 n_max: int = 120):
        self.min_samples  = min_samples
        self.z_spike      = z_spike
        self.z_drop       = z_drop
        self.std_floor    = std_floor
        self.cooldown_sec = cooldown_sec
        self.clip_z       = clip_z
        # Effective memory of a bucket. Without a cap, a new sample's weight
        # (1/n) decays forever and an old bucket can no longer learn anything;
        # capping n turns the accumulator into an exponentially-weighted
        # Welford with a ~n_max-sample time constant. 120 ~= 1.3 occurrences
        # of an hour-of-week bucket at a 40s interval: measured to silence a
        # genuine regime change within ~a week (1 alert/day meanwhile) while
        # a single wild sample still moves the mean by only ~2%.
        self.n_max        = max(min_samples, n_max)
        # key (cam_id) -> metric -> "dow_hour" -> [n, mean, m2]  (Welford)
        self._slots: dict[str, dict[str, dict[str, list[float]]]] = {}
        self._last_flagged: dict[tuple[str, str], float] = {}
        # (key, metric) -> recent spike-verdict epochs / last rebase epoch
        self._spike_times: dict[tuple[str, str], list[float]] = {}
        self._last_rebase: dict[tuple[str, str], float] = {}

    def _spread(self, mean: float, std: float) -> float:
        """One spread definition for BOTH scoring and clip bounds - if these
        ever diverge, values get clipped by a different envelope than the one
        they're scored against and the baseline drifts from the detector."""
        return max(std, self.std_floor, 0.15 * mean)

    @staticmethod
    def bucket_of(ts_utc: dt.datetime) -> tuple[str, str]:
        local = ts_utc.astimezone(TURKEY_TZ)
        return f"{local.weekday()}_{local.hour}", f"{_DOW[local.weekday()]} {local.hour:02d}:00"

    def stats(self, key: str, metric: str,
              ts_utc: dt.datetime) -> tuple[str, str, int, float, float]:
        """(bucket, label, n, mean, std) for the bucket this timestamp falls in."""
        bucket, label = self.bucket_of(ts_utc)
        cell = self._slots.get(key, {}).get(metric, {}).get(bucket)
        if not cell or cell[0] < 1:
            return bucket, label, 0, 0.0, 0.0
        n, mean, m2 = cell
        std = math.sqrt(m2 / n) if n > 1 else 0.0
        return bucket, label, int(n), mean, std

    def check(self, key: str, metric: str, ts_utc: dt.datetime,
              value: float | None, *, min_delta: float,
              drop_min_baseline: float) -> tuple[bool, dict]:
        """Evaluate `value` against its hour-of-week bucket (does NOT update the
        bucket - call update() afterwards so a value never scores itself).
        `key` is the physical cam_id since the scene-keyed refactor."""
        bucket, label, n, mean, std = self.stats(key, metric, ts_utc)
        debug: dict = {"metric": metric, "bucket": label, "bucket_n": n,
                       "bucket_mean": round(mean, 2), "bucket_std": round(std, 2)}
        if value is None:
            return False, {**debug, "reason": "no_sample"}
        if n < self.min_samples:
            return False, {**debug, "reason": "bucket_warmup"}
        # Floor the spread: tiny-count buckets are near-deterministic and a
        # +2 change would otherwise score as a huge z.
        spread = self._spread(mean, std)
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
        cool_key = (key, metric)
        # Feed the rebase detector on EVERY spike verdict - suppressed ones
        # included, because during a detector regime change the cooldown
        # hides most of the evidence that the baseline is systematically off.
        if kind == "contextual_spike":
            times = self._spike_times.setdefault(cool_key, [])
            times.append(now)
            cutoff = now - self.REBASE_WINDOW_SEC
            self._spike_times[cool_key] = times = [t for t in times if t >= cutoff]
            if (len(times) >= self.REBASE_AFTER
                    and now - self._last_rebase.get(cool_key, float("-inf"))
                        >= self.REBASE_WINDOW_SEC):
                self._rebase(key, metric)
                self._last_rebase[cool_key] = now
                self._spike_times[cool_key] = []
                debug["rebased"] = True
                print(f"  * profile rebase: {key}/{metric} - "
                      f"{self.REBASE_AFTER}+ spikes in 24h means the baseline "
                      f"is stale (detector regime change); bucket weights cut "
                      f"to {self.REBASE_N} for fast re-convergence")
        last = self._last_flagged.get(cool_key, 0.0)
        if now - last < self.cooldown_sec:
            return False, {**debug, "reason": "cooldown", "suppressed_kind": kind}
        self._last_flagged[cool_key] = now
        return True, {**debug, "reason": "anomaly", "kind": kind,
                      "expected": round(mean, 1)}

    def _rebase(self, key: str, metric: str) -> None:
        """Cut every bucket's effective sample count to REBASE_N so incoming
        samples carry ~4x their previous weight. m2 is scaled by the same
        factor (m2 is a SUM of squared deviations, proportional to n) so the
        bucket's std - and therefore its clip envelope - is preserved."""
        for cell in self._slots.get(key, {}).get(metric, {}).values():
            n = cell[0]
            if n > self.REBASE_N:
                cell[2] *= self.REBASE_N / n
                cell[0] = self.REBASE_N

    def update(self, key: str, metric: str, ts_utc: dt.datetime,
               value: float | None) -> None:
        """Feed a sample into its hour-of-week bucket (`key` = cam_id).

        Once a bucket is mature (n >= min_samples) the value is CLIPPED to
        mean +/- clip_z * spread before entering the accumulator. A one-off
        spike therefore barely moves the baseline, but a persistent regime
        change keeps pushing the mean toward itself every sample and the
        bucket converges within days - instead of the old behavior where
        flagged samples were excluded outright and a legitimately-busier
        street stayed "anomalous" forever.
        """
        if value is None:
            return
        bucket, _ = self.bucket_of(ts_utc)
        cell = (self._slots.setdefault(key, {})
                .setdefault(metric, {})
                .setdefault(bucket, [0, 0.0, 0.0]))
        n, mean, m2 = cell
        if n >= self.min_samples:
            std = math.sqrt(m2 / n) if n > 1 else 0.0
            spread = self._spread(mean, std)
            lo, hi = mean - self.clip_z * spread, mean + self.clip_z * spread
            value = min(max(value, lo), hi)
        if cell[0] >= self.n_max:
            # Cap the effective sample count: shed one sample's worth of
            # weight so the incoming value always carries >= 1/n_max.
            f = (self.n_max - 1) / cell[0]
            cell[2] *= f
            cell[0] = self.n_max - 1
        cell[0] += 1
        d = value - cell[1]
        cell[1] += d / cell[0]
        cell[2] += d * (value - cell[1])

    # ---- persistence -------------------------------------------------------

    def to_payload(self, key: str) -> dict:
        metrics = {}
        for metric, buckets in self._slots.get(key, {}).items():
            metrics[metric] = {b: {"n": c[0], "mean": c[1], "m2": c[2]}
                               for b, c in buckets.items()}
        return {"slot": key, "tz": "UTC+3", "metrics": metrics}

    def load_payload(self, key: str, payload: dict) -> int:
        """Merge a persisted payload back in. Returns #buckets loaded."""
        loaded = 0
        for metric, buckets in (payload.get("metrics") or {}).items():
            dst = self._slots.setdefault(key, {}).setdefault(metric, {})
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
                           frame, boxes: list[dict], firebase) -> dict:
    """Save raw + annotated frames. Uses Storage if configured, else local disk.

    Annotation draws the ALREADY-COMPUTED detection boxes instead of running
    the model a second time - the old extra inference doubled the CPU cost of
    every anomalous sample on the VM, exactly when the round was already slow.
    """
    stem = _ts_filename(ts_iso)
    raw_ok, raw_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not raw_ok:
        return {}
    urls = {"snapshot_url": None, "snapshot_annotated_url": None}
    try:
        annotated_frame = draw_boxes(frame, boxes)
    except Exception:
        annotated_frame = None

    ann_bytes = None
    if annotated_frame is not None:
        ok, ann_buf = cv2.imencode(".jpg", annotated_frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            ann_bytes = ann_buf.tobytes()

    if firebase.storage is not None:
        urls["snapshot_url"] = firebase.upload_snapshot(
            f"anomalies/{slot_id}/{stem}.jpg", raw_buf.tobytes())
        if ann_bytes is not None:
            urls["snapshot_annotated_url"] = firebase.upload_snapshot(
                f"anomalies/{slot_id}/{stem}_annotated.jpg", ann_bytes)
    else:
        cam_dir = ANOMALY_DIR / slot_id
        cam_dir.mkdir(parents=True, exist_ok=True)
        (cam_dir / f"{stem}.jpg").write_bytes(raw_buf.tobytes())
        urls["snapshot_url"] = f"/snapshots/anomalies/{slot_id}/{stem}.jpg"
        if ann_bytes is not None:
            (cam_dir / f"{stem}_annotated.jpg").write_bytes(ann_bytes)
            urls["snapshot_annotated_url"] = f"/snapshots/anomalies/{slot_id}/{stem}_annotated.jpg"
    # For the caller's alert push only - popped before the record is written.
    urls["_annotated_jpeg"] = ann_bytes
    return urls


def _save_live_view(slot_id: str, frame, boxes: list[dict], firebase) -> str | None:
    """Publish the annotated "what the model sees" frame for a slot.

    ONE fixed object per slot (snapshots/live/{slot_id}.jpg), overwritten on
    every sample - the dashboard shows it under the live video with a
    cache-busting timestamp, so viewers can compare the video against the
    exact boxes the counts came from. Storage's 24h lifecycle never removes
    it because each overwrite resets the object's age. Cheap: draws the
    already-computed boxes, no extra inference.
    """
    annotated = draw_boxes(frame, boxes)
    okj, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not okj:
        return None
    if firebase.storage is not None:
        return firebase.upload_snapshot(f"live/{slot_id}.jpg", buf.tobytes())
    live_dir = SNAPSHOTS_ROOT / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / f"{slot_id}.jpg").write_bytes(buf.tobytes())
    return f"/snapshots/live/{slot_id}.jpg"


def _save_event_crop(kind: str, slot_id: str, base: str, frame, box: dict,
                     firebase) -> tuple[str | None, str | None, bytes | None]:
    """Save a bbox crop + full frame for an event under snapshots/{kind}/...
    Returns (crop_url, full_url, crop_jpeg_bytes) - bytes for alert pushes.
    Uses Storage if configured, else local disk under web/snapshots/."""
    x1 = max(0, int(box["x1"])); y1 = max(0, int(box["y1"]))
    x2 = min(frame.shape[1], int(box["x2"])); y2 = min(frame.shape[0], int(box["y2"]))
    if not (x2 > x1 and y2 > y1):
        return None, None, None
    crop = frame[y1:y2, x1:x2]
    ok_c, crop_buf = cv2.imencode(".jpg", crop,  [cv2.IMWRITE_JPEG_QUALITY, 85])
    ok_f, full_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not (ok_c and ok_f):
        return None, None, None
    crop_bytes = crop_buf.tobytes()
    if firebase.storage is not None:
        crop_url = firebase.upload_snapshot(
            f"{kind}/{slot_id}/{base}.jpg", crop_bytes)
        full_url = firebase.upload_snapshot(
            f"{kind}/{slot_id}/{base}_full.jpg", full_buf.tobytes())
    else:
        local_root = {"returning": RETURNING_DIR}.get(kind, EVENTS_DIR / kind)
        cam_dir = local_root / slot_id
        cam_dir.mkdir(parents=True, exist_ok=True)
        (cam_dir / f"{base}.jpg").write_bytes(crop_bytes)
        (cam_dir / f"{base}_full.jpg").write_bytes(full_buf.tobytes())
        rel = str(cam_dir.relative_to(SNAPSHOTS_ROOT)).replace("\\", "/")
        crop_url = f"/snapshots/{rel}/{base}.jpg"
        full_url = f"/snapshots/{rel}/{base}_full.jpg"
    return crop_url, full_url, crop_bytes


def _save_returning_visitor(slot_id: str, cam_id: str, ts_iso: str,
                            entity_id: int, sightings: int, gap_sec: float,
                            frame, box: dict,
                            firebase) -> tuple[str | None, str | None, bytes | None]:
    """Save the bbox crop + full frame; returns (crop_url, full_url, bytes)."""
    stem  = _ts_filename(ts_iso)
    base  = f"eid{entity_id:04d}_seen{sightings:02d}_{stem}"
    crop_url, full_url, crop_bytes = _save_event_crop(
        "returning", slot_id, base, frame, box, firebase)
    if crop_url and firebase.storage is None:
        # Local-only manifest for the serve.py dashboard.
        cam_dir = RETURNING_DIR / slot_id
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
    return crop_url, full_url, crop_bytes


def _passes_returning_gates(r, gap_min_sec: float, sim_min: float,
                            min_prior: int, cooldown_sec: float,
                            last_save_for_eid: dict,
                            unobserved_sec: float | None = None,
                            max_unobserved_frac: float = RETURNING_MAX_UNOBSERVED_FRAC,
                            prev_box: dict | None = None,
                            new_box: dict | None = None,
                            static_iou: float = RETURNING_STATIC_IOU) -> tuple[bool, str]:
    """Decide whether a re-ID match is a REAL return event worth saving.

    Beyond the identity gates (gap / similarity / prior sightings / cooldown),
    two authenticity gates kill the artifact classes that used to dominate:

    * unobserved gap - if the camera itself wasn't sampled for most of the
      entity's absence (stream outage, fallback episode), nothing "returned";
      we just looked away. Without this, every fallback longer than the gap
      threshold manufactured a return event for each static object in view.
    * static object - if the entity re-appears in (almost) the same box it
      occupied last time, it never left the scene (parked car, banner).
    """
    if r.is_new:                              return False, "new_entity"
    if r.gap_seconds is None:                 return False, "no_gap"
    if r.gap_seconds < gap_min_sec:           return False, "short_gap"
    if r.similarity is not None and r.similarity < sim_min:
                                              return False, "weak_match"
    prior = max(0, (r.sightings or 1) - 1)
    if prior < min_prior:                     return False, "few_prior_sightings"
    if unobserved_sec is None:                return False, "no_observation_history"
    if unobserved_sec > max_unobserved_frac * r.gap_seconds:
                                              return False, "unobserved_gap"
    if box_iou(prev_box, new_box) >= static_iou:
                                              return False, "static_object"
    now  = time.time()
    last = last_save_for_eid.get(r.entity_id, 0.0)
    if now - last < cooldown_sec:             return False, "per_entity_cooldown"
    last_save_for_eid[r.entity_id] = now
    return True, "save"


def _window_key(slot_id: str, cam_id: str) -> str:
    """Rolling-window key: the physical scene as seen from a grid slot. The
    ONE place this format lives - sample_slot and _restore_state must agree
    or restarts silently seed windows nobody reads."""
    return f"{slot_id}|{cam_id}"


# Process-wide state for the returning gates and operator warnings.
# _OBS_LOG starts blind and is backfilled from Firestore history on startup
# (_restore_state), so a restart doesn't suppress long-gap returns for hours;
# anything before the earliest known sample stays conservatively unobserved.
_OBS_LOG = CamObservationLog()
_ENTITY_LAST_BOX: dict[tuple[str, int], tuple[dict, float]] = {}
_ENTITY_BOX_CAP = 20_000
_RETURNING_LAST_SAVE: dict[str, dict] = {}   # slot_id -> {eid: last_save_ts}
# Two slots that fell back onto the same cam would double-feed its hourly
# profile every round; (cam_id, metric) -> wall-clock of last accepted feed.
_PROFILE_LAST_FEED: dict[tuple[str, str], float] = {}
PROFILE_FEED_MIN_GAP_S = 15.0
# cam_id -> [turkey-local-date, count, warned] for the daily budget warning.
# In-memory: a restart resets the count, so treat the warning as a floor -
# the dashboard's "(N in 24h)" badge is the authoritative daily number.
_ANOMALY_DAYCOUNT: dict[str, list] = {}


def _prune_entity_boxes(max_age_sec: float | None = None) -> None:
    """Bound _ENTITY_LAST_BOX. Age-based pruning runs off the hot path (from
    main's periodic maintenance block); the cap-sort is only a backstop."""
    if max_age_sec is not None:
        cutoff = time.time() - max_age_sec
        for k in [k for k, (_, ts) in _ENTITY_LAST_BOX.items() if ts < cutoff]:
            _ENTITY_LAST_BOX.pop(k, None)
        return
    if len(_ENTITY_LAST_BOX) <= _ENTITY_BOX_CAP:
        return
    by_age = sorted(_ENTITY_LAST_BOX.items(), key=lambda kv: kv[1][1])
    for k, _ in by_age[: len(by_age) // 2]:
        _ENTITY_LAST_BOX.pop(k, None)


def _emit_event(firebase, alerts: AlertSink | None, event: dict, *,
                title: str, body: str = "",
                image_jpeg: bytes | None = None) -> None:
    """Persist an operational event and push it. Failures never propagate."""
    try:
        firebase.write_event(event)
    except Exception as e:
        print(f"  ! event write failed ({event.get('kind')}): {e}")
    if alerts is not None:
        alerts.send(event.get("kind", "event"), event.get("cam_id", "?"),
                    event.get("slot", "?"), event.get("ts", ""),
                    title, body, image_jpeg)


def _handle_loiter(firebase, alerts: AlertSink | None, slot: dict,
                   cam_id: str, ts: str, frame, box: dict, loiter: dict,
                   save_snapshots: bool = True) -> None:
    crop_url = full_url = None
    crop_bytes = None
    if save_snapshots:
        try:
            base = (f"loiter_eid{loiter['entity_id']:04d}_"
                    f"{int(loiter['duration_sec'])}s_{_ts_filename(ts)}")
            crop_url, full_url, crop_bytes = _save_event_crop(
                "events/loiter", slot["slot_id"], base, frame, box, firebase)
        except Exception as e:
            print(f"  ! loiter snapshot save failed: {e}")
    minutes = loiter["duration_sec"] / 60
    _emit_event(firebase, alerts, {
        "kind": "loiter", "slot": slot["slot_id"], "cam_id": cam_id, "ts": ts,
        "cls": loiter["cls"], "entity_id": loiter["entity_id"],
        "duration_sec": loiter["duration_sec"],
        "snapshot_url": crop_url, "fullframe_url": full_url,
    }, title=f"Prolonged presence @ {slot['display_area']}",
       body=(f"{loiter['cls']} #{loiter['entity_id']} stationary "
             f"for {minutes:.0f} min"),
       image_jpeg=crop_bytes)
    print(f"  ! LOITER {loiter['cls']} eid{loiter['entity_id']} "
          f"@ {slot['slot_id']}/{cam_id} for {minutes:.0f} min")


def sample_slot(model, slot: dict, cam_id: str, firebase,
                reid: ReidStore | None = None, conf: float = 0.30,
                anomaly=None,
                profile: HourlyProfile | None = None,
                presence: PresenceTracker | None = None,
                alerts: AlertSink | None = None,
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
        _returning_last_save = _RETURNING_LAST_SAVE.setdefault(slot_id, {})

    try:
        frames = grab_burst(resolve_stream(cam), n=burst, stride=burst_stride)
        if not frames:
            raise RuntimeError("empty frame")
        counts, boxes, frame, burst_dbg = detect_burst(
            model, frames, conf=cam_conf, imgsz=imgsz,
            roi=cam.get("roi"), roi_exclude=cam.get("roi_exclude"),
            roi_exclude_class=cam.get("roi_exclude_class"),
            line=cam.get("line"), burst_stride=burst_stride)
        ok = 1
        # Live-sample pool: save one random detection per LIVE_SAMPLE_EVERY_N
        # bursts so the review UI has fresh material even on cameras that
        # don't trigger returning / events / anomalies. Best-effort; a
        # failure here must never abort a successful sample write.
        try:
            from app.live_samples import should_sample as _ls_should, save_crop as _ls_save
            if boxes and _ls_should(cam_id):
                _ls_save(cam_id, frame, boxes)
        except Exception as _ls_err:
            print(f"[{ts}] live_samples skipped: {_ls_err}")
        # Frame-based review pool: save the WHOLE frame + all boxes so the
        # canvas review UI can present the full scene with clickable boxes
        # and gather multiple verdicts per frame (including "missed
        # detection" - the input we need for real recall).
        try:
            from app.review_frames import should_save as _rf_should, save_frame as _rf_save
            if boxes and _rf_should(cam_id):
                _rf_save(cam_id, frame, boxes)
        except Exception as _rf_err:
            print(f"[{ts}] review_frames skipped: {_rf_err}")
        if reid is not None and boxes:
            results = reid.update_from_frame(cam_id, frame, boxes)
            for r in results:
                (new_ids if r.is_new else seen_again).append(r.entity_id)
                box = boxes[r.box_index] if r.box_index is not None else None
                prev_box, _prev_ts = _ENTITY_LAST_BOX.get((cam_id, r.entity_id),
                                                          (None, None))
                if save_snapshots and box is not None:
                    # The obs-log scan is the expensive gate input; compute it
                    # only for entities whose gap can actually pass the cheap
                    # short_gap check (~1% of matches on a busy cam).
                    unobs = (_OBS_LOG.unobserved_during(cam_id, r.gap_seconds)
                             if (r.gap_seconds is not None
                                 and r.gap_seconds >= returning_gap_sec)
                             else None)
                    passes, _why = _passes_returning_gates(
                        r, returning_gap_sec, returning_sim_min,
                        returning_min_prior, returning_cooldown_sec,
                        _returning_last_save,
                        unobserved_sec=unobs,
                        prev_box=prev_box, new_box=box)
                    if passes:
                        try:
                            crop_url, full_url, crop_bytes = _save_returning_visitor(
                                slot_id, cam_id, ts, r.entity_id, r.sightings,
                                r.gap_seconds, frame, box, firebase)
                            _emit_event(firebase, alerts, {
                                "kind": "returning", "slot": slot_id,
                                "cam_id": cam_id, "ts": ts,
                                "cls": box.get("cls"),
                                "entity_id": r.entity_id,
                                "gap_seconds": round(r.gap_seconds, 1),
                                "sightings": r.sightings,
                                "snapshot_url": crop_url,
                                "fullframe_url": full_url,
                            }, title=(f"Returning {box.get('cls')} @ "
                                      f"{slot['display_area']}"),
                               body=(f"entity #{r.entity_id} back after "
                                     f"{r.gap_seconds/60:.0f} min "
                                     f"({r.sightings} sightings)"),
                               image_jpeg=crop_bytes)
                        except Exception as e:
                            print(f"  ! returning save failed for {slot_id}/{cam_id} "
                                  f"eid{r.entity_id}: {e}")
                # Prolonged presence (loitering / parked-too-long).
                if presence is not None and box is not None:
                    loiter = presence.observe(cam_id, r.entity_id,
                                              box.get("cls", "person"), box,
                                              frame.shape, cam)
                    if loiter is not None:
                        _handle_loiter(firebase, alerts, slot, cam_id, ts,
                                       frame, box, loiter,
                                       save_snapshots=save_snapshots)
                if box is not None:
                    _ENTITY_LAST_BOX[(cam_id, r.entity_id)] = (box, time.time())
            if len(_ENTITY_LAST_BOX) > _ENTITY_BOX_CAP:
                _prune_entity_boxes()   # backstop; age prune runs in main()
    except Exception as e:
        print(f"[{ts}] {slot_id} ({cam_id}): MISS ({e})")
        counts = {name: None for name in CLASSES_OF_INTEREST}
        counts["vehicles"] = None
        ok = 0
    if ok:
        _OBS_LOG.record_success(cam_id)

    record = {
        "ts": ts, "cam_id": cam_id, "cam_name": cam["name"],
        "person": counts.get("person"), "vehicles": counts.get("vehicles"),
        "counts": counts, "ok": ok,
        "new_entities":  len(new_ids),
        "seen_entities": len(seen_again),
    }
    if ok:
        # Vehicle speed estimates surface top-level so the dashboard tiles
        # can read them without digging through burst debug internals.
        speeds = burst_dbg.pop("speeds", None)
        if speeds:
            record["speeds"] = speeds
        record["burst"] = burst_dbg
        # Day/night tag: lets the dashboard and any offline analysis split
        # baselines - the same street has very different "normal" after dark.
        record["is_night"] = bool(
            float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) < NIGHT_MEAN_GRAY)
        # Sampled line-crossing flow (only present when the cam has a "line").
        if "crossings" in burst_dbg:
            record["crossings"] = burst_dbg["crossings"]
        # "Model view": the annotated frame these counts came from, shown by
        # the dashboard under the live video and refreshed every sample.
        if save_snapshots:
            try:
                url = _save_live_view(slot_id, frame, boxes, firebase)
                if url:
                    record["live_annotated_url"] = url
            except Exception as e:
                print(f"  ! live view save failed for {slot_id}: {e}")

    # Anomaly gating keyed by the PHYSICAL SCENE. Rolling windows use
    # "{slot_id}|{cam_id}" (fresh short warmup after a fallback swap instead of
    # comparing two different streets against each other); hourly profiles use
    # cam_id so the learned week-shape belongs to the camera, and two slots
    # that fall back onto the same cam share one baseline + cooldown (no
    # duplicate alerts for the same scene).
    if trackers is not None and ok:
        wkey = _window_key(slot_id, cam_id)
        verdicts: list[dict] = []
        for metric, tracker in trackers.items():
            value = counts.get(metric)
            flagged, dbg = tracker.push_and_check(wkey, value)
            if flagged:
                verdicts.append({"kind": dbg["kind"], "metric": metric,
                                 "window": "rolling", "z": dbg.get("z"),
                                 "observed": value, "expected": dbg.get("expected")})
            if profile is not None:
                gates = ANOMALY_METRICS.get(metric, ANOMALY_METRICS["person"])
                ctx_flagged, ctx_dbg = profile.check(
                    cam_id, metric, now_utc, value,
                    min_delta=gates["min_delta"],
                    drop_min_baseline=gates["drop_min_baseline"])
                if ctx_flagged:
                    verdicts.append({"kind": ctx_dbg["kind"], "metric": metric,
                                     "window": "hourly", "z": ctx_dbg.get("z"),
                                     "observed": value,
                                     "expected": ctx_dbg.get("expected"),
                                     "bucket": ctx_dbg.get("bucket")})
                # Feed the bucket unless the rolling layer CONFIRMED an event
                # this very sample. Mature buckets are clip-protected anyway;
                # the exclusion mainly shields IMMATURE buckets (n<30) from
                # ingesting a confirmed event raw during their first days.
                # Cooldown-suppressed and pending samples still flow in, so a
                # genuine regime change converges (rolling adapts within ~30
                # samples and stops flagging) - no exclude-forever loop.
                # The wall-clock dedup stops two slots that fell back onto the
                # same cam from double-feeding the bucket each round.
                feed_key = (cam_id, metric)
                now_wall = time.time()
                if (not flagged and now_wall - _PROFILE_LAST_FEED.get(feed_key, 0.0)
                        >= PROFILE_FEED_MIN_GAP_S):
                    _PROFILE_LAST_FEED[feed_key] = now_wall
                    profile.update(cam_id, metric, now_utc, value)
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
            annotated_jpeg = None
            if save_snapshots and frame is not None:
                try:
                    snap = _save_anomaly_snapshot(slot_id, cam_id, ts, frame,
                                                  boxes, firebase)
                    annotated_jpeg = snap.pop("_annotated_jpeg", None)
                    record.update(snap)
                    print(f"  ! {primary['kind']} @ {slot_id}/{cam_id} "
                          f"[{primary['metric']}] z={primary.get('z')}, "
                          f"observed={primary.get('observed')} vs "
                          f"~{primary.get('expected')} expected - snapshot saved")
                except Exception as e:
                    print(f"  ! anomaly snapshot save failed for {slot_id}: {e}")
            if alerts is not None:
                alerts.send("anomaly", cam_id, slot_id, ts,
                            title=(f"{primary['kind'].replace('_', ' ')} @ "
                                   f"{slot['display_area']}"),
                            body=(f"{primary['metric']}: observed "
                                  f"{primary.get('observed')} vs "
                                  f"~{primary.get('expected')} expected "
                                  f"(z={primary.get('z')})"),
                            image_jpeg=annotated_jpeg)

    try:
        firebase.write(slot_id, record)
        if reid is not None and ok:
            firebase.write_reid_stats(slot_id, cam_id, reid.stats(cam_id))
    except Exception as e:
        print(f"[{ts}] {slot_id}: firebase write failed ({e})")

    if record.get("is_anomaly"):
        # Operational day in Turkey local time (the profile layer's timezone).
        local_day = now_utc.astimezone(TURKEY_TZ).date().isoformat()
        cell = _ANOMALY_DAYCOUNT.setdefault(cam_id, [local_day, 0, False])
        if cell[0] != local_day:
            cell[:] = [local_day, 0, False]
        cell[1] += 1
        if cell[1] > ANOMALY_BUDGET_PER_DAY and not cell[2]:
            cell[2] = True
            print(f"  !! {cam_id}: {cell[1]} anomalies today exceeds the "
                  f"{ANOMALY_BUDGET_PER_DAY}/day budget - gates are likely "
                  f"miscalibrated for this scene; review before trusting alerts")

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


def _parse_ts(ts_iso) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(ts_iso).replace("Z", "+00:00"))
    except ValueError:
        return None


def _restore_state(firebase, trackers: dict[str, AnomalyTracker],
                   profile: HourlyProfile | None, slot_ids: set[str],
                   cam_ids: set[str],
                   legacy_slot_of_primary: dict[str, str] | None = None) -> None:
    """Rebuild in-memory analysis state from Firestore after a (re)start.

    1. Rolling windows: RECENT footfall docs (within the trackers' staleness
       horizon - older samples describe another regime and would be cleared
       on the first push anyway) reseed each (slot, cam) window.
    2. Camera observation log: every ok sample's timestamp is replayed so the
       returning-visitor gate knows the cameras WERE being watched before the
       restart - otherwise long-gap returns are suppressed for hours after
       every service bounce.
    3. Hourly profiles: persisted per-CAMERA profile docs are loaded; a
       primary cam with no cam-keyed doc falls back to its slot's legacy
       pre-refactor doc (weeks of learned baseline beat a 24h bootstrap),
       and only then to bootstrapping from the last 24h of history. Empty
       bootstrap results are NOT saved - persisting a {} profile would make
       the cam permanently "missing" and re-trigger the 10k-doc bootstrap
       read on every restart.
    """
    now = dt.datetime.now(dt.timezone.utc)
    stale_s = max(t.stale_after_sec for t in trackers.values()) if trackers else 600
    try:
        since = (now - dt.timedelta(hours=1)).isoformat()
        docs = firebase.recent_history(since, limit_docs=600)
        window_since = (now - dt.timedelta(seconds=stale_s)).isoformat()
        by_key: dict[str, list[dict]] = {}
        obs_epochs: dict[str, list[float]] = {}
        for d in docs:
            if not (d.get("ok") and d.get("slot") in slot_ids and d.get("cam_id")):
                continue
            t = _parse_ts(d.get("ts"))
            if t is not None:
                obs_epochs.setdefault(d["cam_id"], []).append(t.timestamp())
            if (d.get("ts") or "") >= window_since:
                by_key.setdefault(_window_key(d["slot"], d["cam_id"]), []).append(d)
        for cid, epochs in obs_epochs.items():
            _OBS_LOG.seed(cid, epochs)
        if obs_epochs:
            print(f"  observation log seeded for {len(obs_epochs)} cam(s)")
        for key, rows in by_key.items():
            rows.sort(key=lambda r: r.get("ts") or "")
            for metric, tracker in trackers.items():
                kept = tracker.seed(key, [r.get(metric) for r in rows])
                if kept:
                    print(f"  restored {kept} samples -> {key}/{metric} window")
    except Exception as e:
        print(f"  ! rolling-window restore skipped ({e})")

    if profile is None:
        return
    legacy_slot_of_primary = legacy_slot_of_primary or {}
    missing: list[str] = []
    for cid in sorted(cam_ids):
        payload = None
        try:
            payload = firebase.load_slot_profile(cid)
            if not payload and cid in legacy_slot_of_primary:
                payload = firebase.load_slot_profile(legacy_slot_of_primary[cid])
        except Exception as e:
            print(f"  ! profile load failed for {cid} ({e})")
        if payload and profile.load_payload(cid, payload):
            print(f"  loaded hourly profile for cam {cid}")
        else:
            missing.append(cid)
    if not missing:
        return
    try:
        since = (now - dt.timedelta(hours=24)).isoformat()
        docs = firebase.recent_history(since, limit_docs=10_000)
        n = 0
        for d in docs:
            cid = d.get("cam_id")
            if not d.get("ok") or cid not in missing:
                continue
            ts = _parse_ts(d.get("ts"))
            if ts is None:
                continue
            for metric in trackers:
                profile.update(cid, metric, ts, d.get(metric))
            n += 1
        print(f"  bootstrapped hourly profile for {', '.join(missing)} "
              f"from {n} history docs")
        for cid in missing:
            payload = profile.to_payload(cid)
            if not payload.get("metrics"):
                continue   # nothing learned - don't persist an empty doc
            try:
                firebase.save_slot_profile(cid, payload)
            except Exception as e:
                print(f"  ! profile save failed for {cid} ({e})")
    except Exception as e:
        print(f"  ! profile bootstrap skipped ({e})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Continuous YOLO footfall collector "
                                             "(writes to Firestore + Storage for the HTML dashboard)")
    ap.add_argument("--interval", type=int, default=20, help="seconds between sampling rounds")
    ap.add_argument("--weights", default="yolov8s.pt",
                    help="YOLO weights. Default `yolov8s` recovers small/static "
                         "objects nano misses (parked cars, distant pedestrians) "
                         "and mis-classifies less at frame edges. Set to "
                         "yolov8n.pt to trade accuracy for ~3x less CPU.")
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
    ap.add_argument("--reid-model", default=None,
                    help="path to an OSNet .onnx for real cross-lighting "
                         "re-ID; default: HSV histogram")
    ap.add_argument("--reid-threshold", type=float, default=None,
                    help="cosine similarity above which a detection is 'seen "
                         "before' (default: the embedder's own default - 0.92 "
                         "histogram, 0.65 OSNet)")
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
    ag.add_argument("--anomaly-min-people", type=int,
                    default=ANOMALY_METRICS["person"]["min_value"])
    ag.add_argument("--anomaly-min-delta",  type=float,
                    default=ANOMALY_METRICS["person"]["min_delta"])
    ag.add_argument("--anomaly-cooldown",   type=float, default=300.0)
    ag.add_argument("--anomaly-confirm", type=int, default=2,
                    help="consecutive abnormal samples required before flagging")
    pg = ap.add_argument_group("hour-of-week contextual baseline")
    pg.add_argument("--no-profile", action="store_true",
                    help="disable the hour-of-week contextual anomaly check")
    pg.add_argument("--profile-min-samples", type=int, default=30,
                    help="bucket samples required before contextual checks fire")
    pg.add_argument("--profile-save-min", type=float, default=30.0,
                    help="minutes between profile persists to Firestore")
    rg = ap.add_argument_group("returning-visitor gating")
    rg.add_argument("--returning-gap-min",       type=float,
                    default=RETURNING_GAP_SEC / 60.0,
                    help="minimum absence (minutes) for a return event")
    rg.add_argument("--returning-min-similarity", type=float, default=0.96)
    rg.add_argument("--returning-min-prior",     type=int, default=2)
    rg.add_argument("--returning-per-entity-cooldown-min", type=float, default=30.0)
    og = ap.add_argument_group("operational events (loitering, alert push)")
    og.add_argument("--no-loiter", action="store_true",
                    help="disable prolonged-presence (loiter/parked) events")
    og.add_argument("--loiter-person-min", type=float, default=5.0,
                    help="minutes a person must stay in place before an event "
                         "(cameras.py loiter_person_sec overrides per cam)")
    og.add_argument("--loiter-vehicle-min", type=float, default=15.0,
                    help="minutes a vehicle must stay in place before an event")
    og.add_argument("--no-alerts", action="store_true",
                    help="disable Telegram/webhook pushes even if the env vars "
                         "(TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID/ALERT_WEBHOOK_URL) "
                         "are set")
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
        from app.reid_embed import make_embedder
        embedder = make_embedder(args.reid_model)
        reid = ReidStore(args.reid_db, threshold=args.reid_threshold,
                         embedder=embedder)
        print(f"reid: embedder={embedder.embedder_id}, "
              f"threshold={reid.threshold}")
        try:
            removed = reid.prune(max_age_hours=args.reid_prune_hours)
            if removed:
                print(f"reid: pruned {removed} entities idle > {args.reid_prune_hours:g}h")
        except Exception as e:
            print(f"reid: prune failed ({e})")

    # CLI overrides flow INTO ANOMALY_METRICS so both detection layers (the
    # rolling trackers here and profile.check inside sample_slot, which reads
    # the same dict) see identical gates - previously --anomaly-min-delta
    # silently applied to the rolling layer only.
    ANOMALY_METRICS["person"]["min_value"] = args.anomaly_min_people
    ANOMALY_METRICS["person"]["min_delta"] = args.anomaly_min_delta

    # One rolling tracker per metric; gates scale to each metric's magnitude.
    trackers: dict[str, AnomalyTracker] = {}
    for metric, gates in ANOMALY_METRICS.items():
        trackers[metric] = AnomalyTracker(
            metric            = metric,
            window            = args.anomaly_window,
            z_spike           = args.anomaly_z,
            z_drop            = args.anomaly_drop_z,
            min_value         = gates["min_value"],
            min_delta         = gates["min_delta"],
            drop_min_baseline = gates["drop_min_baseline"],
            cooldown_sec      = args.anomaly_cooldown,
            confirm_samples   = args.anomaly_confirm,
        )

    profile = None if args.no_profile else HourlyProfile(
        min_samples = args.profile_min_samples,
        z_spike     = args.anomaly_z,
        z_drop      = args.anomaly_drop_z,
    )

    presence = None if args.no_loiter else PresenceTracker(
        person_sec  = args.loiter_person_min * 60,
        vehicle_sec = args.loiter_vehicle_min * 60,
    )

    alerts = None
    if not args.no_alerts:
        sink = AlertSink()
        if sink.enabled:
            alerts = sink
            backends = [b for b, on in (
                ("telegram", sink.telegram_token and sink.telegram_chat_id),
                ("webhook", sink.webhook_url)) if on]
            print(f"Alert push enabled -> {', '.join(backends)}")
        else:
            print("Alert push disabled (set TELEGRAM_BOT_TOKEN+TELEGRAM_CHAT_ID "
                  "and/or ALERT_WEBHOOK_URL to enable).")

    save_snapshots          = not args.no_snapshots
    returning_gap_sec       = args.returning_gap_min * 60
    returning_cooldown_sec  = args.returning_per_entity_cooldown_min * 60

    # Every camera that can appear in any slot's chain owns its own baseline.
    all_cam_ids = {c for s in GRID_SLOTS for c in [s["primary"], *s["fallbacks"]]}
    # Pre-refactor profiles were keyed by slot; a slot's learned weeks of
    # baseline belong to its PRIMARY cam.
    legacy_slot_of_primary = {s["primary"]: s["slot_id"] for s in GRID_SLOTS}

    print("Restoring analysis state from Firestore...")
    _restore_state(firebase, trackers, profile, set(pickers), all_cam_ids,
                   legacy_slot_of_primary)

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
          f"(spike>={args.anomaly_z}, drop<=-{args.anomaly_drop_z}, "
          f"confirm={args.anomaly_confirm} consecutive) | "
          f"hour-of-week profile: {'on' if profile else 'off'} | "
          f"budget warn: >{ANOMALY_BUDGET_PER_DAY}/cam/day")
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
        for cid in all_cam_ids:
            payload = profile.to_payload(cid)
            if not payload.get("metrics"):
                # Never write an empty profile: it would shadow nothing useful
                # and (worse, after a transient load failure) could clobber a
                # good persisted baseline with {}.
                continue
            try:
                firebase.save_slot_profile(cid, payload)
            except Exception as e:
                print(f"  ! profile save failed for {cid} ({e})")

    # Hot-reload cadence for review-driven camera overrides. Every N rounds
    # the collector re-reads data/blacklist_auto.json + data/confidence_boost.json
    # so a fresh "correct" / "wrong" verdict in the review UI actually changes
    # what the next burst sees, without a service restart.
    _REVIEW_RELOAD_EVERY_ROUNDS = 10
    # Position-persistence pass runs less often (expensive walk over
    # review_frames/*.json). Every ~1h at 40 s/round.
    _STATIC_LEARN_EVERY_ROUNDS = 90
    _round_counter = 0
    try:
        while True:
            round_start = time.time()
            _round_counter += 1
            if _round_counter % _REVIEW_RELOAD_EVERY_ROUNDS == 0:
                try:
                    from app.cameras import reload_review_overrides
                    reload_review_overrides()
                except Exception as e:
                    print(f"  ! review overrides reload failed: {e}")
            if _round_counter % _STATIC_LEARN_EVERY_ROUNDS == 0:
                # Positions where the model consistently fires on the same
                # background feature (buildings, lampposts, road furniture)
                # become auto-blacklist polygons with no user in the loop.
                try:
                    from app.auto_blacklist import learn_from_positions
                    from app.review_frames import _dir as _rf_dir
                    added = learn_from_positions(_rf_dir())
                    if added:
                        print(f"  * static-position: added {len(added)} "
                              f"blacklist polygon(s) - {', '.join(sorted({e['cam_id']+':'+e['cls'] for e in added}))}")
                        # Rebuild cameras.CAMERAS so the next burst uses them.
                        from app.cameras import reload_review_overrides
                        reload_review_overrides()
                except Exception as e:
                    print(f"  ! static-position learn failed: {e}")
            for slot in GRID_SLOTS:
                picker = pickers[slot["slot_id"]]
                cam_id = picker.current_cam()
                ok = sample_slot(model, slot, cam_id, firebase, reid=reid,
                                 conf=args.conf, anomaly=trackers,
                                 profile=profile, presence=presence,
                                 alerts=alerts, imgsz=args.imgsz,
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
            # Mirror the review pools (frames/crops just saved this round) up
            # to Storage so the operator's local dashboard can search and
            # review what the cameras actually captured. No-op without a
            # bucket; cheap no-change rounds cost one dict compare.
            try:
                from app.pool_sync import sync_up as _pool_sync_up
                from app.visual_search import SNAPSHOTS_ROOT as _snap_root
                from app.visual_search import DEFAULT_DB as _reid_db
                stats = _pool_sync_up(firebase, _snap_root, reid_db_path=_reid_db)
                if stats and stats.get("uploaded"):
                    print(f"  * pool sync: +{stats['uploaded']} "
                          f"-{stats.get('deleted', 0)} file(s)")
            except Exception as e:
                print(f"  ! pool sync failed: {e}")
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
                # Age out last-box entries alongside their reid entities, off
                # the per-sample hot path.
                _prune_entity_boxes(max_age_sec=args.reid_prune_hours * 3600)
                if presence is not None:
                    presence.prune()
                last_reid_prune = time.time()
            round_dur = time.time() - round_start
            if round_dur > args.interval:
                # The effective refresh rate the dashboard sees is the ROUND
                # time, not --interval. If this repeats, the machine is
                # undersized (or lower --imgsz / --burst).
                print(f"  ! round took {round_dur:.0f}s > interval "
                      f"{args.interval}s - tiles refresh every ~{round_dur:.0f}s")
            time.sleep(max(0, args.interval - round_dur))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        _persist_profiles()   # don't lose up to 30 min of learned baseline
        if reid is not None:
            reid.close()


if __name__ == "__main__":
    main()
