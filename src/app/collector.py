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
import os
import time
import urllib.error
from pathlib import Path

import cv2
import numpy as np

from app.cameras import CAMERAS, GRID_SLOTS
from app.alerts import AlertSink
from app.detect_core import (
    CLASSES_OF_INTEREST,
    DEFAULT_IMGSZ,
    DEFAULT_PER_CLASS_CONF,
    box_iou,
    detect_burst,
    draw_boxes,
    grab_burst,
    invalidate_resolved,
    last_grab_error,
    last_grab_http,
    load_model,
    night_adjusted_conf,
    resolve_stream,
)
from app.presence import PresenceTracker
from app.reid import ReidStore

# --- Write rate-limit guard (protects your Firestore write quota / billing) ---
# Free tier: ~20k writes/day. Each slot per round = 2 writes (footfall +
# latest); reid_stats adds a 3rd but only every REID_STATS_EVERY_ROUNDS
# rounds - it is a slow-moving aggregate (total uniques/sightings) and
# writing it every round was exactly what pushed 4 slots @ 40s to ~25.9k
# writes/day, past the free quota. Throttled: 4x2x2160 + 4x2160/5 = ~19k/day,
# UNDER the free tier at the same 40s cadence - the dashboard's re-ID table
# now refreshes every ~3-4 min instead of every 40s, which nobody can tell.
# We still enforce an interval floor to prevent typos like --interval 1.
MIN_INTERVAL_S = 5
FREE_TIER_WRITES_PER_DAY = 20_000
REID_STATS_EVERY_ROUNDS = 5

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
# Local-clock night window (Turkey). Measured on live frames at 01:46 local:
# the lit city streets hold mean-gray 105-120 - far ABOVE the darkness
# threshold - so brightness alone never declared night and the night gates
# sat inert through the exact hours they were built for. Clock wins.
NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 6


def is_night(luma: float | None, now_utc: dt.datetime, tz=None) -> bool:
    """Night = local-clock night OR a genuinely dark frame (tunnel-dark
    daytime failure counts too). Drives the per-class gate bump and the
    record's is_night analysis tag. `tz` is the CAMERA's timezone (a US
    Pacific street and an Istanbul square hit 20:00 hours apart); defaults
    to Turkey/UTC+3 for back-compat when the caller has no camera tz."""
    h = now_utc.astimezone(tz or TURKEY_TZ).hour
    if h >= NIGHT_START_HOUR or h < NIGHT_END_HOUR:
        return True
    return luma is not None and luma < NIGHT_MEAN_GRAY

# ---- Scene anomalies (operator definition, 2026-07) ---------------------------
# Statistical spike/drop verdicts are NOT anomalies to the operator - "more
# traffic than Friday 13:00 usually has" is weather, not an event. An anomaly
# is something you would walk over to the screen for:
#   * extreme_load        - the place is genuinely packed (top activity band);
#   * camera_obstructed   - one object fills most of the view (something
#                           parked against / held up to the lens);
#   * camera_dark         - the view went dark while the previous sample was
#                           bright (covered lens, power cut, tampering).
# Returning visitors and prolonged presence remain events (they already are).
EXTREME_PERSON = 50           # matches the web activity scale's 9-10/10 band
EXTREME_VEH_LOAD = 38.0       # weighted vehicles, same 9-10/10 band
_VEH_LOAD_W = {"car": 1.0, "truck": 2.5, "bus": 2.5,
               "motorcycle": 0.5, "bicycle": 0.3, "train": 3.0}
OBSTRUCTION_AREA_FRAC = 0.5   # one box covering half the frame
# A giant box must also be CONFIDENT to count as obstruction: observed live,
# the barely-above-gate `train` class (gate 0.25) hallucinated half-frame
# boxes at the bus terminal five times in one morning and drowned the
# anomaly table. Real lens blockage produces a high-confidence object.
OBSTRUCTION_MIN_CONF = 0.45
DARK_FROM_LUMA = 90.0         # was clearly daylight...
DARK_TO_LUMA = 25.0           # ...and is now near-black
SCENE_ANOMALY_COOLDOWN_S = 1800.0
_SCENE_ANOMALY_LAST: dict[tuple[str, str], float] = {}
_LAST_LUMA: dict[str, float] = {}


def weighted_vehicle_load(counts: dict) -> float:
    """Street presence of the vehicle mix (bus/truck weigh ~2.5 cars) -
    mirrors the dashboard's VEHICLE_LOAD_WEIGHTS so both sides agree on
    what 'packed' means."""
    return sum(w * (counts.get(cls) or 0) for cls, w in _VEH_LOAD_W.items())


def check_scene_anomalies(cam_id: str, counts: dict, boxes: list[dict],
                          frame_shape, luma: float | None,
                          now: float | None = None) -> list[dict]:
    """Evaluate the operator-defined anomaly kinds for one sample.

    Returns verdict dicts shaped like the old statistical ones (kind /
    metric / observed / expected) so the dashboard schema is unchanged.
    Each (cam, kind) pair has its own cooldown.
    """
    now = time.time() if now is None else now
    out: list[dict] = []

    def _fire(kind: str, metric: str, observed, expected) -> None:
        key = (cam_id, kind)
        # -inf default: "never fired" must always pass the cooldown check,
        # regardless of how small the clock is (tests use tiny epochs).
        if (now - _SCENE_ANOMALY_LAST.get(key, float("-inf"))
                < SCENE_ANOMALY_COOLDOWN_S):
            return
        _SCENE_ANOMALY_LAST[key] = now
        out.append({"kind": kind, "metric": metric, "window": "scene",
                    "observed": observed, "expected": expected})

    person = counts.get("person") or 0
    load = weighted_vehicle_load(counts)
    if person >= EXTREME_PERSON:
        _fire("extreme_load", "person", person, f"<{EXTREME_PERSON}")
    elif load >= EXTREME_VEH_LOAD:
        _fire("extreme_load", "vehicles", round(load, 1),
              f"<{EXTREME_VEH_LOAD:g}")

    if frame_shape is not None and boxes:
        H, W = frame_shape[:2]
        area = float(H * W) or 1.0
        for b in boxes:
            frac = max(0.0, (b["x2"] - b["x1"])) * max(0.0, (b["y2"] - b["y1"])) / area
            if (frac >= OBSTRUCTION_AREA_FRAC
                    and float(b.get("conf") or 0.0) >= OBSTRUCTION_MIN_CONF):
                _fire("camera_obstructed", b.get("cls", "?"),
                      f"{frac:.0%} of view", f"<{OBSTRUCTION_AREA_FRAC:.0%}")
                break

    if luma is not None:
        prev = _LAST_LUMA.get(cam_id)
        _LAST_LUMA[cam_id] = luma
        if prev is not None and prev >= DARK_FROM_LUMA and luma <= DARK_TO_LUMA:
            _fire("camera_dark", "brightness", round(luma, 1),
                  f">{DARK_TO_LUMA:g} (was {prev:.0f})")
    return out

TURKEY_TZ = dt.timezone(dt.timedelta(hours=3))  # permanent UTC+3, no DST
_DOW = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Per-camera timezone for the day/night gate and the hour-of-week profile.
# The country-generic grid (2026-07-17) runs cameras from Thailand, Japan,
# and a US bench that spans Eastern/Central/Pacific - bucketing all of them
# under Istanbul's clock would smear every "normal for this hour" baseline
# and fire the night gate at the wrong local hours. Resolve each camera's
# tz (its own "tz", else its country default) to a tzinfo, preferring the
# stdlib zoneinfo (correct DST) and falling back to a fixed-offset table
# when the tz database is absent (bare Windows without `tzdata`).
_FIXED_OFFSETS = {
    "Europe/Istanbul": 3, "Asia/Bangkok": 7, "Asia/Tokyo": 9,
    "America/New_York": -5, "America/Chicago": -6, "America/Los_Angeles": -8,
}
_TZ_CACHE: dict = {}


def _tzinfo(name: str):
    if name in _TZ_CACHE:
        return _TZ_CACHE[name]
    tz = None
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(name)
    except Exception:
        off = _FIXED_OFFSETS.get(name)
        if off is not None:
            tz = dt.timezone(dt.timedelta(hours=off))
    tz = tz or TURKEY_TZ
    _TZ_CACHE[name] = tz
    return tz


def cam_tzinfo(cam_id: str):
    """tzinfo for a camera, via cameras.camera_timezone (its own tz or its
    country default). Cheap and cached; safe on unknown ids (-> Turkey)."""
    try:
        from app.cameras import camera_timezone
        return _tzinfo(camera_timezone(cam_id))
    except Exception:
        return TURKEY_TZ


class CameraPool:
    """ONE health-tracked priority ladder shared by all slots (operator
    spec, 2026-07-16): tier 1 the four Konya cams, tier 2 the preferred
    Istanbul four (sultanahmet, beyazit, eyup, buyuk camlica), tier 3 the
    rest of the live catalog. Each round the pool assigns the FIRST N
    healthy cameras, in priority order, one per slot - so the grid always
    runs N DISTINCT cameras (the per-slot chains this replaces let two
    slots drift onto the same stream).

    Health model per camera:
      * fresh cameras walk in with `max_failures` grace (a transient miss
        does not kill a scene);
      * after `max_failures` consecutive misses the camera goes on
        cooldown for `retry_minutes`;
      * when cooldown expires the camera is probed again - but a camera
        that has already proven dead re-enters cooldown after ONE miss,
        so a dead tvkur backend costs one sample per 15 minutes, not a
        3-miss re-walk (the July 2026 outage burned ~10 min of every 15
        that way);
      * `fast_fail` cameras skip the grace entirely: ONE miss rests them
        even on first contact. Operator spec 2026-07-17 - the tvkur
        (Konya) cams are cheap, low-risk probes (the host never throttled
        this project, and a playlist-level 404 is a definitive "channel
        gone"), so the whole Konya sweep must cost one round, not three;
      * a success fully rehabilitates the camera.

    If fewer than N cameras are healthy the assignment is padded with the
    least-recently-failed cooldown cameras: every slot keeps sampling
    SOMETHING, which is also how dead cameras get re-discovered.
    """

    def __init__(self, pool: list[str], n_slots: int,
                 max_failures: int = FALLBACK_MAX_FAILURES,
                 retry_minutes: int = FALLBACK_RETRY_MINUTES,
                 fast_fail=()):
        self.pool          = list(pool)
        self.n_slots       = n_slots
        self.max_failures  = max_failures
        self.retry_seconds = retry_minutes * 60
        self.fast_fail     = set(fast_fail) & set(self.pool)
        self.failures      = {c: 0 for c in self.pool}       # consecutive misses
        self.cooldown_until = {c: 0.0 for c in self.pool}    # epoch; 0 = eligible
        self.proven_dead   = {c: False for c in self.pool}   # failed a full grace once

    def _eligible(self, cam: str, now: float) -> bool:
        return now >= self.cooldown_until[cam]

    def assign(self, now: float | None = None,
               blocked: frozenset | set = frozenset()) -> list[str]:
        """First n_slots eligible cameras in priority order (distinct by
        construction). When the healthy set is too small the assignment is
        padded with resting cameras in the SAME priority order - stable
        across rounds, so an all-dead pool holds the grid steady on the
        top of the ladder instead of churning tiles every round.

        `blocked` (from the host circuit breaker) removes cameras whose
        HOST is currently refusing access: they are skipped outright and
        used for padding only as the very last resort - a blocked host
        must stay untouched while it rests."""
        now = time.time() if now is None else now
        blocked = set(blocked)
        picked = [c for c in self.pool
                  if c not in blocked and self._eligible(c, now)][: self.n_slots]
        for c in self.pool:                      # pad: unblocked resting cams
            if len(picked) >= self.n_slots:
                break
            if c not in picked and c not in blocked:
                picked.append(c)
        for c in self.pool:                      # last resort: blocked cams
            if len(picked) >= self.n_slots:
                break
            if c not in picked:
                picked.append(c)
        return picked

    def forgive(self, cams) -> None:
        """Wipe per-camera strikes and cooldowns. Used when a HOST-level
        block is identified: the individual cameras were never dead - the
        host refused everyone - and their accumulated 15-min cooldowns
        would otherwise outlive the block itself and stagger recovery."""
        for c in cams:
            if c in self.failures:
                self.failures[c] = 0
                self.cooldown_until[c] = 0.0
                self.proven_dead[c] = False

    def record(self, cam: str, ok: bool, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if cam not in self.failures:      # cam outside the pool (manual runs)
            return
        if ok:
            self.failures[cam] = 0
            self.cooldown_until[cam] = 0.0
            self.proven_dead[cam] = False
            return
        if now < self.cooldown_until[cam]:
            # Forced padding sample of a camera that is still resting: the
            # miss is expected and must NOT push its recovery further out,
            # or an all-dead pool would keep every cooldown sliding forever.
            return
        self.failures[cam] += 1
        # A camera that already burned through its grace once is on probation:
        # a single miss sends it straight back to cooldown. fast_fail cams
        # (tvkur) never get grace in the first place.
        strikes = (1 if (self.proven_dead[cam] or cam in self.fast_fail)
                   else self.max_failures)
        if self.failures[cam] >= strikes:
            self.failures[cam] = 0
            self.proven_dead[cam] = True
            self.cooldown_until[cam] = now + self.retry_seconds

    def all_fast_fail(self, cams) -> bool:
        """True when EVERY camera in `cams` is a low-risk fast-fail probe.
        The main loop uses this to skip the politeness backoff for rounds
        that never touched a rate-limited CDN (an all-Konya probe round
        must not delay the ladder's descent to the Istanbul tier)."""
        cams = list(cams)
        return bool(cams) and all(c in self.fast_fail for c in cams)


class HostBreaker:
    """Host-level circuit breaker for access blocks (HTTP 403/429).

    Per-camera strikes are the wrong tool for a WAF/geo block: on
    2026-07-17 kamerayayin refused EVERY playlist with 403, yet the pool
    dutifully walked 13 Istanbul cameras x 3 strikes each - dozens of
    requests an hour knocking on a door that reputation-based blocks
    count against this address. The breaker treats the HOST as the unit:

      * `threshold` consecutive block-signature failures (403/429, any
        stage) across a host's cameras trip it: ALL its cameras leave the
        rotation for `rest_minutes` and their per-camera strikes are
        forgiven (they were never dead - the host refused everyone);
      * when the rest expires, ONE probe camera (highest priority) is
        allowed through. A success - or any non-block failure, which
        means the host is answering again - reopens the host instantly;
        another 403 re-arms the rest;
      * forced padding samples of a resting host never extend its rest
        (same rule the pool applies to camera cooldowns).

    So a blocked window costs ~3 requests/hour instead of ~120, and an
    unblock (like the open flap observed at 21:15 the day before) is
    caught by the very next probe.
    """

    BLOCK_CODES = frozenset({403, 429})

    def __init__(self, host_of: dict, threshold: int = 4,
                 rest_minutes: int = 20):
        self.host_of = dict(host_of)          # cam_id -> host, priority order
        self.threshold = threshold
        self.rest_seconds = rest_minutes * 60
        self.consec = {}                      # host -> consecutive refusals
        self.rest_until = {}                  # host -> epoch (present = tripped)

    def cams_of(self, host: str) -> list[str]:
        return [c for c, h in self.host_of.items() if h == host]

    def blocked_cams(self, now: float | None = None) -> set:
        """Cameras the pool must not assign this round. While a host
        rests, every one of its cameras is out; once the rest expires the
        FIRST camera becomes the probe and the rest stay out until the
        probe's verdict."""
        now = time.time() if now is None else now
        out = set()
        for host, until in self.rest_until.items():
            cams = self.cams_of(host)
            if now < until:
                out.update(cams)
            else:
                out.update(cams[1:])          # probing: free only the first
        return out

    def note(self, cam: str, ok: bool, http: int | None,
             now: float | None = None) -> str | None:
        """Feed one sample result; returns the event it caused, if any:
        'tripped' / 'rearmed' / 'reopened' / None."""
        host = self.host_of.get(cam)
        if host is None:
            return None
        now = time.time() if now is None else now
        if not ok and http in self.BLOCK_CODES:
            self.consec[host] = self.consec.get(host, 0) + 1
            if host in self.rest_until:
                if now < self.rest_until[host]:
                    return None               # forced sample during rest
                self.rest_until[host] = now + self.rest_seconds
                return "rearmed"              # probe refused: rest again
            if self.consec[host] >= self.threshold:
                self.rest_until[host] = now + self.rest_seconds
                return "tripped"
            return None
        # success, or a non-block failure (404 / timeout): the host is
        # answering - per-camera health can take it from here.
        self.consec[host] = 0
        if host in self.rest_until:
            del self.rest_until[host]
            return "reopened"
        return None


class CountryDirector:
    """Country-generic grid controller (operator spec, 2026-07-17).

    The grid always runs FOUR cameras from ONE country. Each country owns
    its own CameraPool (priority ladder over that country's cameras) and
    HostBreaker (per-host 403/429 circuit breaker). Countries are tried in
    a fixed priority order; the director stays on a country as long as it
    can field live cameras, backfilling a dead camera from deeper in the
    SAME country's bench, and only advances to the next country when the
    active one is fully dark (`min_live` live cameras, default 1).

    Recovery: a country higher in the order than the active one is re-probed
    shortly before each scheduled report; if it delivers, the grid switches
    back to it (Turkey is the project's subject - the point is to spend the
    maximum time on it and only visit Thailand/Japan/USA while it is down).

    Pure control logic - the actual frame grabs happen in the main loop,
    which feeds results back through record(). Fully unit-testable offline.
    """

    def __init__(self, countries: dict, order: list[str], n_slots: int,
                 fast_fail_host_substr: str = "tvkur", min_live: int = 1,
                 max_failures: int = FALLBACK_MAX_FAILURES,
                 retry_minutes: int = FALLBACK_RETRY_MINUTES,
                 breaker_threshold: int = 4, breaker_rest_minutes: int = 20):
        from urllib.parse import urlparse
        self.order = [c for c in order if c in countries and countries[c]]
        if not self.order:
            raise ValueError("CountryDirector needs at least one non-empty country")
        self.n_slots = n_slots
        self.min_live = min_live
        self.pools: dict = {}
        self.breakers: dict = {}
        for country in self.order:
            cams = list(countries[country])
            fast = [c for c in cams
                    if fast_fail_host_substr in (CAMERAS.get(c, {}).get("url") or "")]
            self.pools[country] = CameraPool(
                cams, n_slots=n_slots, max_failures=max_failures,
                retry_minutes=retry_minutes, fast_fail=fast)
            host_of = {c: (urlparse(CAMERAS.get(c, {}).get("url") or "").hostname or "?")
                       for c in cams}
            self.breakers[country] = HostBreaker(
                host_of, threshold=breaker_threshold,
                rest_minutes=breaker_rest_minutes)
        self.active = self.order[0]

    # ---- per-round assignment -------------------------------------------
    def assign(self, now: float) -> tuple[str, list[str]]:
        """(active_country, [4 cam_ids]) for this round, honoring the active
        country's host breaker."""
        pool, br = self.pools[self.active], self.breakers[self.active]
        return self.active, pool.assign(now=now, blocked=br.blocked_cams(now))

    def record(self, cam: str, ok: bool, http: int | None, now: float,
               country: str | None = None) -> str | None:
        """Feed one sample result into a country's pool + breaker (defaults
        to the active country). Returns the breaker event, if any."""
        country = country or self.active
        pool, br = self.pools[country], self.breakers[country]
        pool.record(cam, ok, now=now)
        event = br.note(cam, ok, http, now=now)
        if event == "tripped":
            pool.forgive(br.cams_of(br.host_of[cam]))
        elif event == "rearmed":
            pool.forgive([cam])
        return event

    # ---- liveness + advancement -----------------------------------------
    def live_count(self, country: str, now: float) -> int:
        """Cameras currently eligible (not resting) AND not host-blocked."""
        pool, br = self.pools[country], self.breakers[country]
        blocked = br.blocked_cams(now)
        return sum(1 for c in pool.pool
                   if pool._eligible(c, now) and c not in blocked)

    def maybe_advance(self, now: float) -> tuple[str, str] | None:
        """If the active country is dark (< min_live live cameras), rotate to
        the next country in priority order that has live cameras. Returns
        (from, to) on a switch, else None. If nobody has live cameras the
        active country is kept (the grid holds steady rather than churning)."""
        if self.live_count(self.active, now) >= self.min_live:
            return None
        start = self.order.index(self.active)
        for step in range(1, len(self.order) + 1):
            cand = self.order[(start + step) % len(self.order)]
            if cand == self.active:
                continue
            if self.live_count(cand, now) >= self.min_live:
                prev, self.active = self.active, cand
                return prev, cand
        return None

    def countries_above(self, country: str | None = None) -> list[str]:
        """Higher-priority countries than the active one, best-first - the
        recovery-probe candidates for the pre-report check."""
        idx = self.order.index(country or self.active)
        return self.order[:idx]

    def switch_to(self, country: str) -> None:
        """Force the active country (used after a successful recovery probe).
        Forgives that country's accumulated strikes so it starts clean."""
        if country not in self.pools:
            return
        self.active = country
        pool = self.pools[country]
        pool.forgive(list(pool.pool))


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
    def bucket_of(ts_utc: dt.datetime, tz=None) -> tuple[str, str]:
        local = ts_utc.astimezone(tz or TURKEY_TZ)
        return f"{local.weekday()}_{local.hour}", f"{_DOW[local.weekday()]} {local.hour:02d}:00"

    def stats(self, key: str, metric: str,
              ts_utc: dt.datetime) -> tuple[str, str, int, float, float]:
        """(bucket, label, n, mean, std) for the bucket this timestamp falls in."""
        bucket, label = self.bucket_of(ts_utc, cam_tzinfo(key))
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
        bucket, _ = self.bucket_of(ts_utc, cam_tzinfo(key))
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
    """Snapshot of what the dashboard needs about a slot right now. The
    human label is the active CAMERA's own name (the grid is country-generic
    now - the slot_id stays generic and the tile title follows whatever
    camera is live), and `country` lets the dashboard/report state which
    country the grid is currently watching."""
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
        "country":         cam.get("country", "turkey"),
        "city":            cam.get("city", ""),
        "display_area":    cam.get("name", slot["display_area"]),
    }


# ---- Pre-report country recovery ----------------------------------------
# Turkey is the project's subject; the grid only visits Thailand/Japan/USA
# while Turkey is blocked. A few minutes before each scheduled digest (and
# periodically as a safety net) the collector re-probes higher-priority
# countries even though their cameras are resting - so the report reflects
# Turkey the moment its block lifts, instead of waiting out a 15-min
# camera cooldown. Times are Israel-local (the digest timer's timezone).
REPORT_TIMES_ISRAEL = os.environ.get("REPORT_TIMES_ISRAEL", "12:00,20:00")
RECOVERY_PRE_REPORT_MIN = int(os.environ.get("RECOVERY_PRE_REPORT_MIN") or 5)
_RECOVERY_STATE = {"last": 0.0}


def _minutes_to_next_report(now_ts: float, times_str: str | None = None) -> float:
    times_str = times_str or REPORT_TIMES_ISRAEL
    tz = _tzinfo("Asia/Jerusalem")
    now = dt.datetime.fromtimestamp(now_ts, tz)
    best = None
    for t in times_str.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            hh, mm = (int(x) for x in t.split(":"))
        except ValueError:
            continue
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if target <= now:
            target += dt.timedelta(days=1)
        mins = (target - now).total_seconds() / 60.0
        best = mins if best is None else min(best, mins)
    return best if best is not None else 1e9


def _recovery_due(now_ts: float) -> bool:
    """True at most once per pre-report window. Operator spec (2026-07-17):
    re-probe higher-priority countries ONLY in the few minutes before a
    scheduled report - NOT periodically. Once the grid has settled on a
    fallback country it must not keep sniffing the blocked one every few
    minutes; the block is re-checked twice a day, right before each report."""
    if _minutes_to_next_report(now_ts) <= RECOVERY_PRE_REPORT_MIN:
        # Fire once per window (guard against re-firing every round inside it).
        if now_ts - _RECOVERY_STATE["last"] >= RECOVERY_PRE_REPORT_MIN * 60:
            _RECOVERY_STATE["last"] = now_ts
            return True
    return False


def _recover_higher_priority(director, model, args, firebase,
                             slot_ids, now_ts: float) -> bool:
    """Force-probe the lead cameras of each higher-priority country (best
    first), bypassing cooldowns. On the first live frame, switch the grid
    back to that country. Returns True on a switch."""
    for country in director.countries_above():
        for cam_id in director.pools[country].pool[:2]:   # top 2 cams
            cam = CAMERAS.get(cam_id)
            if not cam:
                continue
            try:
                frames = grab_burst(resolve_stream(cam, now_ts),
                                    n=1, stride=1)
            except Exception:
                frames = None
            ok = bool(frames)
            http = None if ok else last_grab_http()[1]
            director.record(cam_id, ok, http, now_ts, country=country)
            if ok:
                director.switch_to(country)
                print(f"  * recovery: {country} answered ({cam_id}) - "
                      "switching the grid back to it.")
                return True
    return False


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
    fh, fw = (frame.shape[:2] if hasattr(frame, "shape") else (0, 0))
    _emit_event(firebase, alerts, {
        "kind": "loiter", "slot": slot["slot_id"], "cam_id": cam_id, "ts": ts,
        "cls": loiter["cls"], "entity_id": loiter["entity_id"],
        "duration_sec": loiter["duration_sec"],
        "snapshot_url": crop_url, "fullframe_url": full_url,
        # Box coordinates + frame dimensions so the emailed report can draw
        # a red overlay on the fullframe. Without this the operator sees
        # "someone loitered in this scene" but has to hunt the person.
        "box": [float(box.get("x1", 0)), float(box.get("y1", 0)),
                float(box.get("x2", 0)), float(box.get("y2", 0))],
        "frame_w": int(fw), "frame_h": int(fh),
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
                _returning_last_save: dict | None = None,
                write_reid_stats: bool = True) -> bool:
    """Sample the currently-active cam for a slot and write to Firestore.

    Detection runs on a short frame burst and keeps the median count (see
    detect_core.detect_burst), so a single noisy frame can no longer move the
    number the dashboard shows. `anomaly` accepts either one AnomalyTracker
    (legacy people-only callers) or a {metric: tracker} dict; `profile` adds
    the hour-of-week contextual check on top of the rolling window.

    A camera can carry its own calibrated "conf" (see cameras.py); otherwise
    the global `conf` applies.

    Returns True iff a frame was grabbed and processed successfully. The
    caller feeds this back to the CameraPool to decide whether the camera
    should rest and the ladder should advance.
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

    luma = None
    night = False
    try:
        frames = grab_burst(resolve_stream(cam), n=burst, stride=burst_stride)
        if not frames:
            # Name the REAL failure stage (playlist/chunklist/segment/decode)
            # - a bare "empty frame" masked four different root causes.
            why = last_grab_error()
            raise RuntimeError(f"empty frame - {why}" if why else "empty frame")
        # Day/night decided BEFORE detection so the gates can react to it.
        # Use the CAMERA's timezone: a Bangkok street and an Istanbul square
        # cross into night at different UTC hours.
        luma = float(np.mean(cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)))
        night = is_night(luma, now_utc, cam_tzinfo(cam_id))
        # Effective per-class gates. This is ALSO the fix for a silent bug:
        # cameras.py merges the review-driven confidence boosts into
        # cam["per_class_conf"], but the value was never handed to
        # detect_burst - the learning ledger updated while detection kept
        # running on the shipped defaults. At night every gate additionally
        # rises (see night_adjusted_conf): point-lit storefronts and shrubs
        # were firing as bus/motorcycle on the operator's screenshots.
        gates = dict(cam.get("per_class_conf") or DEFAULT_PER_CLASS_CONF)
        if night:
            gates = night_adjusted_conf(gates)
        counts, boxes, frame, burst_dbg = detect_burst(
            model, frames, conf=cam_conf, imgsz=imgsz,
            roi=cam.get("roi"), roi_exclude=cam.get("roi_exclude"),
            roi_exclude_class=cam.get("roi_exclude_class"),
            line=cam.get("line"), per_class_conf=gates,
            burst_stride=burst_stride)
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
                            fh, fw = (frame.shape[:2]
                                      if hasattr(frame, "shape") else (0, 0))
                            _emit_event(firebase, alerts, {
                                "kind": "returning", "slot": slot_id,
                                "cam_id": cam_id, "ts": ts,
                                "cls": box.get("cls"),
                                "entity_id": r.entity_id,
                                "gap_seconds": round(r.gap_seconds, 1),
                                "sightings": r.sightings,
                                "snapshot_url": crop_url,
                                "fullframe_url": full_url,
                                # Same overlay data as loiter, so the report
                                # PDF can circle the returning entity in
                                # the scene rather than just captioning it.
                                "box": [float(box.get("x1", 0)),
                                        float(box.get("y1", 0)),
                                        float(box.get("x2", 0)),
                                        float(box.get("y2", 0))],
                                "frame_w": int(fw), "frame_h": int(fh),
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
                # Sighting gallery: once an entity is established (3+
                # sightings) every further look at it banks a small crop,
                # so a "returning visitor" event can show ALL appearances
                # side by side instead of a single snapshot.
                if save_snapshots and box is not None and r.sightings >= 3:
                    try:
                        from app.entity_gallery import save_sighting
                        save_sighting(cam_id, r.entity_id, frame, box)
                    except Exception as _eg_err:
                        print(f"  ! entity gallery skipped: {_eg_err}")
                if box is not None:
                    _ENTITY_LAST_BOX[(cam_id, r.entity_id)] = (box, time.time())
            if len(_ENTITY_LAST_BOX) > _ENTITY_BOX_CAP:
                _prune_entity_boxes()   # backstop; age prune runs in main()
    except (RuntimeError, OSError, cv2.error, urllib.error.URLError,
            ConnectionError, TimeoutError, ValueError) as e:
        # Narrowed from `except Exception` (2026-07): the old catch-all
        # swallowed KeyError / AttributeError from any code inside this
        # block and rendered them as an indistinguishable "MISS", so a
        # programming bug looked identical to a dead stream. The set above
        # covers the real stream failures we want to keep going through:
        # RuntimeError("empty frame") from grab_burst, cv2 decode errors,
        # HTTP failures inside _http_get, TCP timeouts, and int/float
        # conversion errors from a malformed playlist.
        print(f"[{ts}] {slot_id} ({cam_id}): MISS ({type(e).__name__}: {e})")
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
        # Decided pre-detection (it also drove the night gates above).
        record["is_night"] = night
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
    # The statistical layers (rolling z + hour-of-week profile) keep LEARNING
    # here - their windows, baselines and self-rebase stay warm - but their
    # spike/drop verdicts no longer flag anything: the operator ruled that
    # "busier than this hour usually is" is weather, not an anomaly. What
    # counts as an anomaly is decided by check_scene_anomalies (extreme
    # load / camera obstructed / camera gone dark); returning visitors and
    # prolonged presence continue to flow through the events feed.
    if trackers is not None and ok:
        wkey = _window_key(slot_id, cam_id)
        for metric, tracker in trackers.items():
            value = counts.get(metric)
            flagged, _dbg = tracker.push_and_check(wkey, value)
            if profile is not None:
                gates = ANOMALY_METRICS.get(metric, ANOMALY_METRICS["person"])
                profile.check(cam_id, metric, now_utc, value,
                              min_delta=gates["min_delta"],
                              drop_min_baseline=gates["drop_min_baseline"])
                # Feed the bucket unless the rolling layer confirmed an
                # outlier this very sample (shields immature buckets); the
                # wall-clock dedup stops two slots that fell back onto the
                # same cam from double-feeding the bucket each round.
                feed_key = (cam_id, metric)
                now_wall = time.time()
                if (not flagged and now_wall - _PROFILE_LAST_FEED.get(feed_key, 0.0)
                        >= PROFILE_FEED_MIN_GAP_S):
                    _PROFILE_LAST_FEED[feed_key] = now_wall
                    profile.update(cam_id, metric, now_utc, value)

    if ok:
        scene = check_scene_anomalies(cam_id, counts, boxes,
                                      frame.shape if frame is not None else None,
                                      luma)
        record["is_anomaly"] = bool(scene)
        if scene:
            primary = dict(scene[0])
            if len(scene) > 1:
                primary["also"] = [{k: v.get(k) for k in ("kind", "metric")}
                                   for v in scene[1:]]
            record["anomaly"] = primary
            annotated_jpeg = None
            if save_snapshots and frame is not None:
                try:
                    snap = _save_anomaly_snapshot(slot_id, cam_id, ts, frame,
                                                  boxes, firebase)
                    annotated_jpeg = snap.pop("_annotated_jpeg", None)
                    record.update(snap)
                    print(f"  ! {primary['kind']} @ {slot_id}/{cam_id} "
                          f"[{primary['metric']}] observed="
                          f"{primary.get('observed')} (expected "
                          f"{primary.get('expected')}) - snapshot saved")
                except Exception as e:
                    print(f"  ! anomaly snapshot save failed for {slot_id}: {e}")
            if alerts is not None:
                alerts.send("anomaly", cam_id, slot_id, ts,
                            title=(f"{primary['kind'].replace('_', ' ')} @ "
                                   f"{slot['display_area']}"),
                            body=(f"{primary['metric']}: observed "
                                  f"{primary.get('observed')} vs "
                                  f"{primary.get('expected')} expected"),
                            image_jpeg=annotated_jpeg)

    try:
        firebase.write(slot_id, record)
        if reid is not None and ok and write_reid_stats:
            firebase.write_reid_stats(slot_id, cam_id, reid.stats(cam_id))
    except Exception as e:
        print(f"[{ts}] {slot_id}: firebase write failed ({e})")

    if record.get("is_anomaly"):
        # Operational day in the CAMERA's local time (the profile timezone).
        local_day = now_utc.astimezone(cam_tzinfo(cam_id)).date().isoformat()
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
    ap.add_argument("--country", default=None,
                    help="start the grid on this country (turkey/thailand/"
                         "japan/usa). Default: the top of the priority ladder "
                         "(turkey). The collector still rotates to the next "
                         "country automatically when the active one goes dark.")
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

    # ONE shared camera pool for all slots: Konya first, then the preferred
    # Istanbul four, then the rest of the catalog - always N distinct cams.
    # tvkur (Konya) cameras ride the fast-fail lane: one miss rests them,
    # so a dead Konya backend costs ONE round (<1 min) before the ladder
    # reaches the Istanbul tier - not three 3-strike rounds with growing
    # backoff (~6 min). The IBB cams keep the full grace: their misses can
    # be transient, and probing THEM aggressively is what got the VM
    # throttled on 2026-07-16.
    # Country-generic grid (2026-07-17): the grid runs 4 cameras from ONE
    # country; the CountryDirector holds a per-country priority ladder
    # (CameraPool) and per-host circuit breaker (HostBreaker), stays on a
    # country while it can field live cameras, and only advances to the next
    # country when the active one is fully dark. tvkur (Konya) cams keep the
    # fast-fail lane. `--country` pins a starting country; otherwise it
    # begins at the top of the priority order (Turkey).
    from app.cameras import COUNTRIES, COUNTRY_ORDER, country_pool
    country_pools = {c: country_pool(c) for c in COUNTRY_ORDER}
    director = CountryDirector(country_pools, COUNTRY_ORDER,
                               n_slots=len(GRID_SLOTS))
    if getattr(args, "country", None) and args.country in director.pools:
        director.switch_to(args.country)
    slot_ids = [s["slot_id"] for s in GRID_SLOTS]
    _pool0 = director.pools[director.active]
    print("country grid: "
          f"{' -> '.join(COUNTRY_ORDER)} | active={director.active} "
          f"({len(country_pools[director.active])} cams); advance to the next "
          f"country only when the active one has < {director.min_live} live "
          "camera.")
    print(f"per-country pool: {_pool0.max_failures} misses rest a camera "
          f"{_pool0.retry_seconds // 60:.0f} min; tvkur cams + probation cams "
          "rest after a single miss.")
    _br0 = director.breakers[director.active]
    print(f"host breaker: {_br0.threshold} consecutive 403/429s rest ALL of "
          f"a host's cameras for {_br0.rest_seconds // 60} min, then a "
          "single probe request decides (answer = back in rotation).")

    _active_country, _assigned = director.assign(time.time())
    assignment = dict(zip(slot_ids, _assigned))

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

    # Every camera that can appear in ANY country's grid owns its own
    # baseline (the grid is country-generic - a Thailand or Japan camera
    # needs its hour-of-week profile restored/persisted too, not just
    # Turkey's).
    all_cam_ids = {c for pool in director.pools.values() for c in pool.pool}
    # Pre-refactor profiles were keyed by slot; a slot's learned weeks of
    # baseline belong to its PRIMARY cam.
    legacy_slot_of_primary = {s["primary"]: s["slot_id"] for s in GRID_SLOTS}

    print("Restoring analysis state from Firestore...")
    _restore_state(firebase, trackers, profile, set(slot_ids), all_cam_ids,
                   legacy_slot_of_primary)

    # Publish the initial grid config so the dashboard renders immediately.
    slots_meta = [_slot_metadata(s, assignment[s["slot_id"]]) for s in GRID_SLOTS]
    firebase.write_grid_config(slots_meta, country=director.active)

    print(f"Collector started. {len(GRID_SLOTS)} slot(s), active country = "
          f"{director.active}:")
    print("  priority: " + " -> ".join(director.pools[director.active].pool))
    for slot in GRID_SLOTS:
        print(f"  {slot['slot_id']:20s} starts on {assignment[slot['slot_id']]}")
    print(f"interval={args.interval}s, imgsz={args.imgsz}, burst={args.burst}, "
          f"reid={'on' if reid else 'off'}, conf={args.conf}, "
          f"snapshots={'on' if save_snapshots else 'off'}")
    print(f"anomaly metrics: {', '.join(trackers)} | rolling robust-z "
          f"(spike>={args.anomaly_z}, drop<=-{args.anomaly_drop_z}, "
          f"confirm={args.anomaly_confirm} consecutive) | "
          f"hour-of-week profile: {'on' if profile else 'off'} | "
          f"budget warn: >{ANOMALY_BUDGET_PER_DAY}/cam/day")
    print(f"fallback: {FALLBACK_MAX_FAILURES} misses to advance (tvkur: 1), "
          f"retry primary every {FALLBACK_RETRY_MINUTES} min.")

    writes_per_round = len(GRID_SLOTS) * 2
    if reid:
        writes_per_round += len(GRID_SLOTS) / REID_STATS_EVERY_ROUNDS
    projected = writes_per_round * (86400 / args.interval)
    print(f"~{projected:,.0f} Firestore writes/day projected "
          f"(free tier ~ {FREE_TIER_WRITES_PER_DAY:,}).")
    if projected > FREE_TIER_WRITES_PER_DAY:
        print("  ! Above the free-tier write quota - the overage is billed on "
              "Blaze. Raise --interval or REID_STATS_EVERY_ROUNDS to get back "
              "under it.")
    else:
        print("  within the daily free-tier write quota - Firestore cost $0.")
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
    # Promoted-adapter poll: one tiny pointer read from Storage every ~20
    # min; a download (~5 MB) happens only when the trainer actually
    # promoted a new head. The swap is load_state_dict on the LIVE Detect
    # module - no model reload, no restart, no RAM spike on the 1 GB host.
    _ADAPTER_CHECK_EVERY_ROUNDS = 30
    _round_counter = 0
    _all_miss_rounds = 0
    _active_country = director.active
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
            if _round_counter % _ADAPTER_CHECK_EVERY_ROUNDS == 0:
                try:
                    from app import adapters
                    fetched = adapters.refresh_from_storage(
                        getattr(firebase, "storage", None) if firebase else None)
                    if fetched:
                        n = adapters.apply_current(
                            model, expected_base=args.weights)
                        print(f"  * adapter: hot-loaded {fetched} "
                              f"({n} head tensors, no restart)")
                except Exception as e:
                    print(f"  ! adapter refresh failed: {e}")
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
            # Pre-report recovery: a few minutes before each scheduled digest,
            # re-probe higher-priority countries even while their cameras are
            # resting - Turkey is the subject, so the grid should jump back to
            # it the moment its block lifts, not wait out a 15-min cooldown.
            if _recovery_due(round_start):
                switched = _recover_higher_priority(
                    director, model, args, firebase, slot_ids, round_start)
                if switched:
                    _all_miss_rounds = 0
            # If the active country is fully dark, advance to the next country
            # that can field live cameras (deep same-country bench first, so
            # this only fires when the whole country is down).
            adv = director.maybe_advance(round_start)
            if adv:
                print(f"  * country: {adv[0]} is dark - switching grid to "
                      f"{adv[1]}.")
                _all_miss_rounds = 0

            # One pool decision per round for the active country: the first N
            # healthy cameras in priority order, one per slot, never a
            # duplicate. Cameras of a breaker-tripped host stay out entirely.
            active_country, _assigned = director.assign(round_start)
            round_cams = dict(zip(slot_ids, _assigned))
            country_changed = active_country != _active_country
            _active_country = active_country
            moved = [sid for sid in slot_ids
                     if round_cams[sid] != assignment.get(sid)]
            assignment = round_cams
            if moved or country_changed:
                for sid in moved:
                    print(f"  * {sid}: -> {assignment[sid]} "
                          f"({CAMERAS.get(assignment[sid], {}).get('country', '?')})")
                slots_meta = [_slot_metadata(s, assignment[s["slot_id"]])
                              for s in GRID_SLOTS]
                try:
                    firebase.write_grid_config(slots_meta, country=active_country)
                except Exception as e:
                    print(f"  ! grid config write failed: {e}")
            round_had_ok = False
            for slot in GRID_SLOTS:
                cam_id = assignment[slot["slot_id"]]
                ok = sample_slot(model, slot, cam_id, firebase, reid=reid,
                                 conf=args.conf, anomaly=trackers,
                                 profile=profile, presence=presence,
                                 alerts=alerts, imgsz=args.imgsz,
                                 burst=args.burst, burst_stride=args.burst_stride,
                                 save_snapshots=save_snapshots,
                                 returning_gap_sec      = returning_gap_sec,
                                 returning_sim_min      = args.returning_min_similarity,
                                 returning_min_prior    = args.returning_min_prior,
                                 returning_cooldown_sec = returning_cooldown_sec,
                                 write_reid_stats=(_round_counter
                                                   % REID_STATS_EVERY_ROUNDS == 0))
                round_had_ok = round_had_ok or ok
                if not ok:
                    # Failed grab: drop the cached (stale) resolved URL so the
                    # next attempt re-resolves instead of retrying dead token.
                    invalidate_resolved(cam_id)
                # Route the result through the active country's pool + host
                # breaker (one 403/429 is a data point; `threshold` in a row
                # across a host is an access block on this address).
                _stage, _http = (None, None) if ok else last_grab_http()
                _br = director.breakers[active_country]
                event = director.record(cam_id, ok, _http, round_start,
                                        country=active_country)
                if event == "tripped":
                    _host = _br.host_of[cam_id]
                    print(f"  ! {_host}: {_br.threshold} consecutive access "
                          f"refusals (HTTP {_http}) - this address is blocked. "
                          f"Resting all {len(_br.cams_of(_host))} of its cameras "
                          f"for {_br.rest_seconds // 60} min, then probing with "
                          "ONE request.")
                elif event == "rearmed":
                    print(f"  ! {_br.host_of[cam_id]}: probe still refused "
                          f"(HTTP {_http}) - resting another "
                          f"{_br.rest_seconds // 60} min.")
                elif event == "reopened":
                    print(f"  * {_br.host_of[cam_id]}: answering again - "
                          "host back in rotation.")
            # Politeness backoff: rounds where EVERY camera missed are almost
            # always an upstream outage or the CDN rate-limiting this IP -
            # hammering ~4 cams x 3 requests every round from the same
            # address is exactly how the VM got throttled by kamerayayin on
            # 2026-07-16. Slow the scan down until something delivers again.
            # All-tvkur rounds are exempt: those are the cheap dead-channel
            # probes (zero IBB traffic), and sleeping after them only delays
            # the ladder's descent to the Istanbul tier.
            _active_dark = director.live_count(active_country, round_start) == 0
            _next_country_live = any(
                director.live_count(c, round_start) >= director.min_live
                for c in director.order if c != active_country)
            if round_had_ok:
                _all_miss_rounds = 0
            elif _active_dark and _next_country_live:
                # The active country is dark and a lower-priority country is
                # live - we advance NEXT round anyway. Don't crawl on the way
                # out (blocked hosts are already resting under their breakers);
                # this is the descent, not steady-state hammering.
                _all_miss_rounds = 0
            elif director.pools[active_country].all_fast_fail(round_cams.values()):
                pass
            else:
                _all_miss_rounds += 1
                backoff = min(args.interval * _all_miss_rounds, 240)
                print(f"  ! whole round missed ({_all_miss_rounds}x) - "
                      f"backing off {backoff}s to stay polite to the CDN")
                time.sleep(backoff)
            # Mirror the review pools (frames/crops just saved this round) up
            # to Storage so the operator's local dashboard can search and
            # review what the cameras actually captured. No-op without a
            # bucket; cheap no-change rounds cost one dict compare.
            try:
                from app.pool_sync import sync_up as _pool_sync_up
                from app.visual_search import SNAPSHOTS_ROOT as _snap_root
                from app.visual_search import DEFAULT_DB as _reid_db
                stats = _pool_sync_up(firebase, _snap_root, reid_db_path=_reid_db)
                if stats and (stats.get("uploaded") or stats.get("pending")):
                    print(f"  * pool sync: +{stats['uploaded']} "
                          f"-{stats.get('deleted', 0)} file(s)"
                          + (f", {stats['pending']} queued for next rounds"
                             if stats.get("pending") else ""))
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
