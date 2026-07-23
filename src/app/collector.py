"""Continuous footfall collector - pushes live YOLO counts to Firestore.

A Jupyter cell runs once and stops. This process runs forever: every `--interval`
seconds it iterates the four GRID_SLOTS, picks each slot's currently-healthy
camera (with fallback), runs YOLO on a short frame burst (median count), updates
the re-ID registry, and writes the result to Firestore (keyed by slot_id, not
cam_id). The HTML dashboard subscribes via onSnapshot and updates in real time.

Anomaly detection is OPERATIONAL, not statistical: check_scene_anomalies
flags extreme load (crowd/vehicle surges past hard gates), an obstructed
lens and a feed gone dark, each with per-camera cooldowns and a daily
budget warning. (An earlier rolling-z + hour-of-week baseline pair was
removed 2026-07-18: its verdicts had been muted since the operator ruled
"busier than usual" is weather, not an anomaly - it only cost Firestore
writes.)

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
    """Country-generic grid controller (operator spec, sharpened 2026-07-18).

    The grid runs cameras from ONE country, at the widest width any country
    can field. Each country owns its own CameraPool (priority ladder over
    that country's cameras) and HostBreaker (per-host 403/429 circuit
    breaker). The rule, per the operator:

      1. a dead camera backfills from deeper in the SAME country's bench
         first - the country is only abandoned when its whole ladder
         cannot field the target width;
      2. the target width starts at 4: the first country in priority order
         (Turkey -> Thailand -> Japan -> USA) that can field 4 live
         cameras wins;
      3. only after a full loop finds NO country fielding 4 does the grid
         narrow to 3 - again scanning the full priority order - then 2,
         then 1. Width grows back automatically as cameras recover
         (cooldowns expire / benches refill);
      4. everything dark: hold the active country at full width - the
         padded assignment keeps probing, which is how dead cameras get
         re-discovered.

    Recovery upward to a BLOCKED country (Turkey behind its geo-block) is
    the pre-report probe's job, not this bookkeeping's: blocked hosts stay
    out of live_count while their breaker rests, so the director never
    sniffs them mid-day. The report itself always ships from whatever
    country the closing window actually ran on; a successful pre-report
    probe only re-aims the NEXT window.

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
        self.n_active = n_slots        # current grid width (spec rule 3)

    # ---- per-round assignment -------------------------------------------
    def assign(self, now: float) -> tuple[str, list[str]]:
        """(active_country, cam_ids) for this round, honoring the active
        country's host breaker. The list is n_active wide - fewer than
        n_slots when the whole ladder is narrower than the grid (the
        remaining slots idle instead of hammering proven-dead cameras
        every round). All-dark hold keeps full width so the padded
        assignment can re-discover recoveries."""
        pool, br = self.pools[self.active], self.breakers[self.active]
        cams = pool.assign(now=now, blocked=br.blocked_cams(now))
        return self.active, cams[: max(1, self.n_active)]

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

    def desired_state(self, now: float) -> tuple[str, int]:
        """(country, width): the widest grid any country can field right
        now, ties broken by priority order (spec rules 2-3). Width 0 means
        everything is dark. Pure bookkeeping - no network; blocked hosts
        are already excluded from live_count, so a geo-blocked Turkey is
        skipped without a single request."""
        for n in range(self.n_slots, 0, -1):
            for country in self.order:
                if self.live_count(country, now) >= n:
                    return country, n
        return self.active, 0

    def maybe_advance(self, now: float) -> tuple[str, str] | None:
        """Re-aim the grid at desired_state. Returns (from, to) on a
        country switch, else None (width changes surface via n_active).
        Everything dark: hold the active country at FULL width - the
        padded assignment keeps probing resting cameras, which is how the
        grid re-discovers a recovery (spec rule 4)."""
        country, n = self.desired_state(now)
        if n <= 0:
            self.n_active = self.n_slots
            return None
        self.n_active = n
        if country == self.active:
            return None
        prev, self.active = self.active, country
        return prev, country

    def countries_above(self, country: str | None = None) -> list[str]:
        """Higher-priority countries than the active one, best-first - the
        recovery-probe candidates for the pre-report check."""
        idx = self.order.index(country or self.active)
        return self.order[:idx]

    def switch_to(self, country: str) -> None:
        """Force the active country (used after a successful recovery probe).
        Forgives that country's accumulated strikes so it starts clean, and
        restores full grid width - the fresh country gets its fair shot at
        fielding 4 before desired_state narrows anything."""
        if country not in self.pools:
            return
        self.active = country
        self.n_active = self.n_slots
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


def _ts_filename(ts_iso: str) -> str:
    return ts_iso.replace("-", "").replace(":", "").replace("T", "_")[:15]


def _slot_metadata(slot: dict, active_cam: str | None) -> dict:
    """Snapshot of what the dashboard needs about a slot right now. The
    human label is the active CAMERA's own name (the grid is country-generic
    now - the slot_id stays generic and the tile title follows whatever
    camera is live), and `country` lets the dashboard/report state which
    country the grid is currently watching.

    active_cam=None marks an IDLE slot: the grid narrowed below 4 because
    no country can field that many live cameras right now (spec rule 3).
    The dashboard shows the honest state instead of a dead player."""
    if active_cam is None:
        return {
            "slot_id":         slot["slot_id"],
            "primary":         slot["primary"],
            "active_cam":      None,
            "active_cam_name": "standby - grid narrowed",
            "active_embed":    None,
            "active_page":     None,
            "active_hls":      None,
            "active_kind":     None,
            "country":         None,
            "city":            "",
            "display_area":    slot["display_area"],
            "idle":            True,
        }
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


def _save_heatmap_view(cam_id: str, frame, firebase) -> str | None:
    """Publish the presence-heatmap overlay for a camera.

    ONE fixed object per CAMERA (snapshots/heatmaps/{cam_id}.jpg) -
    cam-keyed rather than slot-keyed because the map is a property of the
    scene, and the grid re-seats cameras across slots. Overwritten every
    ~30 samples; same lifecycle trick as the live view (each overwrite
    resets the Storage object's age)."""
    from app import heatmap as _hm
    img = _hm.render(cam_id, base_frame=frame)
    okj, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not okj:
        return None
    if firebase.storage is not None:
        return firebase.upload_snapshot(f"heatmaps/{cam_id}.jpg",
                                        buf.tobytes())
    hm_dir = SNAPSHOTS_ROOT / "heatmaps"
    hm_dir.mkdir(parents=True, exist_ok=True)
    (hm_dir / f"{cam_id}.jpg").write_bytes(buf.tobytes())
    return f"/snapshots/heatmaps/{cam_id}.jpg"


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


# Process-wide state for the returning gates and operator warnings.
# _OBS_LOG starts blind and is backfilled from Firestore history on startup
# (_restore_state), so a restart doesn't suppress long-gap returns for hours;
# anything before the earliest known sample stays conservatively unobserved.
_OBS_LOG = CamObservationLog()
_ENTITY_LAST_BOX: dict[tuple[str, int], tuple[dict, float]] = {}
_ENTITY_BOX_CAP = 20_000
_RETURNING_LAST_SAVE: dict[str, dict] = {}   # slot_id -> {eid: last_save_ts}
# cam_id -> [turkey-local-date, count, warned] for the daily budget warning.
# In-memory: a restart resets the count, so treat the warning as a floor -
# the dashboard's "(N in 24h)" badge is the authoritative daily number.
_ANOMALY_DAYCOUNT: dict[str, list] = {}
# Same shape for loiter/returning events: (cam_id, kind) -> [date, n, warned].
_EVENT_DAYCOUNT: dict[tuple[str, str], list] = {}
EVENT_BUDGET_PER_DAY = 10
# Last published heatmap-overlay URL per camera. The object is fixed
# (heatmaps/<cam_id>.jpg, overwritten in place), so once a render exists
# every subsequent record can carry the URL, not just the render sample.
_HEATMAP_URLS: dict[str, str] = {}


def _event_evidence_ok(box: dict, kind: str, cam_id: str) -> bool:
    """Anti-flood gates for loiter/returning events (2026-07-18).

    1. Evidence floor: the triggering detection must clear its class's
       UN-boosted default gate. Review-loop boosts may loosen a camera's
       gate to 0.20 to recover misses - right for COUNTING, wrong for
       ALERTING: the sub-default-conf "person" that sits still forever is
       a lamppost, not a loiterer.
    2. Daily budget: at most EVENT_BUDGET_PER_DAY events per (cam, kind)
       per local day; beyond it the event is dropped and one loud line
       flags the miscalibration (mirrors the anomaly budget).
    """
    cls = box.get("cls", "person")
    try:
        conf = float(box.get("conf") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    if conf < DEFAULT_PER_CLASS_CONF.get(cls, 0.35):
        return False
    local_day = (dt.datetime.now(dt.timezone.utc)
                 .astimezone(cam_tzinfo(cam_id)).date().isoformat())
    cell = _EVENT_DAYCOUNT.setdefault((cam_id, kind), [local_day, 0, False])
    if cell[0] != local_day:
        cell[:] = [local_day, 0, False]
    if cell[1] >= EVENT_BUDGET_PER_DAY:
        if not cell[2]:
            cell[2] = True
            print(f"  !! {cam_id}: {kind} events hit the "
                  f"{EVENT_BUDGET_PER_DAY}/day budget - suppressing the "
                  f"rest of today (thresholds likely too loose for this "
                  f"scene; review before trusting the feed)")
        return False
    cell[1] += 1
    return True


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


def _save_static_departed_images(slot_id: str, base: str,
                                 crop_bytes: bytes | None, after_frame,
                                 firebase) -> tuple[str | None, str | None]:
    """Persist a static_departed event's evidence pair.

    The object is GONE from the current frame, so the crop is the one the
    watch captured at settle time (the last good look); the CURRENT frame
    is saved alongside as the "after" shot - the empty spot."""
    after_bytes = None
    if after_frame is not None:
        okf, full_buf = cv2.imencode(".jpg", after_frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, 80])
        if okf:
            after_bytes = full_buf.tobytes()
    crop_url = after_url = None
    if firebase.storage is not None:
        if crop_bytes:
            crop_url = firebase.upload_snapshot(
                f"events/static/{slot_id}/{base}.jpg", crop_bytes)
        if after_bytes:
            after_url = firebase.upload_snapshot(
                f"events/static/{slot_id}/{base}_after.jpg", after_bytes)
    else:
        cam_dir = EVENTS_DIR / "static" / slot_id
        cam_dir.mkdir(parents=True, exist_ok=True)
        rel = str(cam_dir.relative_to(SNAPSHOTS_ROOT)).replace("\\", "/")
        if crop_bytes:
            (cam_dir / f"{base}.jpg").write_bytes(crop_bytes)
            crop_url = f"/snapshots/{rel}/{base}.jpg"
        if after_bytes:
            (cam_dir / f"{base}_after.jpg").write_bytes(after_bytes)
            after_url = f"/snapshots/{rel}/{base}_after.jpg"
    return crop_url, after_url


def _handle_static_departed(firebase, alerts: AlertSink | None, slot: dict,
                            cam_id: str, ts: str, frame, dep: dict,
                            save_snapshots: bool = True) -> None:
    crop_url = after_url = None
    crop_bytes = dep.get("crop_jpeg")
    if save_snapshots:
        try:
            base = (f"static_a{dep['anchor_id']:04d}_"
                    f"{int(dep['dwell_sec'])}s_{_ts_filename(ts)}")
            crop_url, after_url = _save_static_departed_images(
                slot["slot_id"], base, crop_bytes, frame, firebase)
        except Exception as e:
            print(f"  ! static-departed snapshot save failed: {e}")
    minutes = dep["dwell_sec"] / 60
    fh, fw = (frame.shape[:2] if hasattr(frame, "shape") else (0, 0))
    box = dep["box"]
    _emit_event(firebase, alerts, {
        "kind": "static_departed", "slot": slot["slot_id"],
        "cam_id": cam_id, "ts": ts, "cls": dep["cls"],
        "dwell_sec": dep["dwell_sec"], "sightings": dep.get("hits"),
        "snapshot_url": crop_url, "fullframe_url": after_url,
        # Where the object USED to stand, on the "after" frame - the
        # report can circle the now-empty spot.
        "box": [float(box["x1"]), float(box["y1"]),
                float(box["x2"]), float(box["y2"])],
        "frame_w": int(fw), "frame_h": int(fh),
    }, title=f"Static object left @ {slot['display_area']}",
       body=f"{dep['cls']} static for {minutes:.0f} min - now gone",
       image_jpeg=crop_bytes)
    print(f"  ! STATIC-DEPARTED {dep['cls']} a{dep['anchor_id']} "
          f"@ {slot['slot_id']}/{cam_id} after {minutes:.0f} min")


def sample_slot(model, slot: dict, cam_id: str, firebase,
                reid: ReidStore | None = None, conf: float = 0.30,
                presence: PresenceTracker | None = None,
                static_watch=None,
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
    number the dashboard shows. Anomalies are decided by
    check_scene_anomalies (extreme load / obstruction / darkness).

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

    if _returning_last_save is None:
        _returning_last_save = _RETURNING_LAST_SAVE.setdefault(slot_id, {})

    luma = None
    night = False
    heatmap_url = None
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
        # WS1: every box that reaches a review pool carries `uncertainty`,
        # scored against the EFFECTIVE gates this burst ran with (boosted +
        # night) - not the shipped defaults. The optional flip pass (one
        # extra inference on the mirrored frame, UNCERTAINTY_FLIP=1) only
        # runs on bursts that are actually being sampled, so the regular
        # round cost is untouched. Best-effort; a failure here must never
        # abort a successful sample write.
        ls_due = rf_due = False
        try:
            from app.live_samples import should_sample as _ls_should
            from app.review_frames import should_save as _rf_should
            ls_due = bool(boxes) and _ls_should(cam_id)
            rf_due = bool(boxes) and _rf_should(cam_id)
            if boxes:
                from app.uncertainty import attach_uncertainty, flip_delta
                flip = None
                if ((ls_due or rf_due)
                        and os.environ.get("UNCERTAINTY_FLIP") == "1"):
                    flip = flip_delta(model, frame, boxes, imgsz)
                attach_uncertainty(boxes, gates, flip)
        except Exception as _u_err:
            print(f"[{ts}] uncertainty skipped: {_u_err}")
        # Live-sample pool: save one random detection per LIVE_SAMPLE_EVERY_N
        # bursts so the review UI has fresh material even on cameras that
        # don't trigger returning / events / anomalies.
        try:
            if ls_due:
                from app.live_samples import save_crop as _ls_save
                _ls_save(cam_id, frame, boxes)
        except Exception as _ls_err:
            print(f"[{ts}] live_samples skipped: {_ls_err}")
        # Frame-based review pool: save the WHOLE frame + all boxes so the
        # canvas review UI can present the full scene with clickable boxes
        # and gather multiple verdicts per frame (including "missed
        # detection" - the input we need for real recall).
        try:
            if rf_due:
                from app.review_frames import save_frame as _rf_save
                _rf_save(cam_id, frame, boxes)
        except Exception as _rf_err:
            print(f"[{ts}] review_frames skipped: {_rf_err}")
        # Presence heatmap (HEATMAP=0 disables): bank WHERE this sample's
        # activity stood - foot points weighted by the observed interval -
        # and refresh the published overlay every ~30 samples. Pure
        # bookkeeping from already-computed boxes; no extra inference.
        try:
            if boxes and os.environ.get("HEATMAP", "1") != "0":
                from app import heatmap as _hm
                _hm.accumulate(cam_id, boxes, frame.shape,
                               tz=cam_tzinfo(cam_id))
                if save_snapshots and _hm.render_due(cam_id):
                    url = _save_heatmap_view(cam_id, frame, firebase)
                    if url:
                        _HEATMAP_URLS[cam_id] = url
                heatmap_url = _HEATMAP_URLS.get(cam_id)
        except Exception as _hm_err:
            print(f"[{ts}] heatmap skipped: {_hm_err}")
        # Static-object watch: objects that settled >= 5 min and then
        # vanished (parked car pulling out, stall packing up, bag gone).
        # Runs ONLY on successful samples, so a stream outage can never
        # fake a departure; dark frames and scene wipes are guarded inside.
        try:
            if static_watch is not None:
                for _dep in static_watch.observe(cam_id, boxes, frame.shape,
                                                 luma=luma, frame=frame):
                    _pseudo = dict(_dep["box"], cls=_dep["cls"],
                                   conf=_dep["conf_median"])
                    if _event_evidence_ok(_pseudo, "static_departed",
                                          cam_id):
                        _handle_static_departed(
                            firebase, alerts, slot, cam_id, ts, frame,
                            _dep, save_snapshots=save_snapshots)
        except Exception as _sw_err:
            print(f"[{ts}] static watch skipped: {_sw_err}")
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
                    if passes and not _event_evidence_ok(box, "returning",
                                                         cam_id):
                        passes = False
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
                # Evidence floor (2026-07-18): the event must be backed by a
                # detection that clears its class's UN-boosted default gate.
                # The review loop legitimately loosens per-cam gates down to
                # 0.20 to recover misses, but a "person" that only exists at
                # conf 0.22 on a loosened gate is exactly the lamppost/banner
                # class of FP that flooded the report with fake loiters -
                # weak evidence may count, it must never alert. Plus a daily
                # per-camera budget, same idea as the anomaly budget.
                if presence is not None and box is not None:
                    loiter = presence.observe(cam_id, r.entity_id,
                                              box.get("cls", "person"), box,
                                              frame.shape, cam)
                    if loiter is not None and not _event_evidence_ok(
                            box, "loiter", cam_id):
                        loiter = None
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
        # Presence-heatmap overlay URL (refreshed every ~30 samples; the
        # dashboard strip offers it as a toggle on the model view).
        if heatmap_url:
            record["heatmap_url"] = heatmap_url
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

    # What counts as an anomaly is decided by check_scene_anomalies below
    # (extreme load / camera obstructed / camera gone dark); returning
    # visitors and prolonged presence flow through the events feed.
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
            # Scene anomalies (extreme_load / camera_obstructed / camera_dark)
            # were only being written to the footfall record's `is_anomaly`
            # flag - the digest reads `events`, so the daily report NEVER
            # surfaced 'camera_obstructed' or 'camera_dark' even when the
            # collector's log had shouted about them. Mirror one row per
            # verdict into `events` so they appear in the anomalies table
            # exactly like loiter/returning. The per-(cam, kind) 30-min
            # cooldown lives inside check_scene_anomalies, so this write
            # is already deduped upstream.
            try:
                firebase.write_event({
                    "kind":         primary["kind"],
                    "slot":         slot_id,
                    "cam_id":       cam_id,
                    "cam_name":     cam.get("name", cam_id),
                    "ts":           ts,
                    "metric":       primary.get("metric"),
                    "observed":     primary.get("observed"),
                    "expected":     primary.get("expected"),
                    "snapshot_url": record.get("snapshot_url"),
                    "fullframe_url": record.get("snapshot_annotated_url")
                                     or record.get("snapshot_url"),
                })
            except Exception as _e:
                print(f"  ! scene-anomaly event write failed: {_e}")

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
                save_snapshots: bool = True,
                slot_id: str | None = None,
                **kwargs) -> bool:
    slot = {"slot_id": slot_id or f"cam_{cam_id}",
            "display_area": cam.get("name", cam_id),
            "primary":      cam_id,
            "fallbacks":    []}
    return sample_slot(model, slot, cam_id, firebase, reid=reid, conf=conf,
                       save_snapshots=save_snapshots, **kwargs)


def _parse_ts(ts_iso) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(ts_iso).replace("Z", "+00:00"))
    except ValueError:
        return None


def _restore_state(firebase, slot_ids: set[str]) -> None:
    """Rebuild in-memory analysis state from Firestore after a (re)start.

    Camera observation log: every ok sample's timestamp is replayed so the
    returning-visitor gate knows the cameras WERE being watched before the
    restart - otherwise long-gap returns are suppressed for hours after
    every service bounce.
    """
    now = dt.datetime.now(dt.timezone.utc)
    try:
        since = (now - dt.timedelta(hours=1)).isoformat()
        docs = firebase.recent_history(since, limit_docs=600)
        obs_epochs: dict[str, list[float]] = {}
        for d in docs:
            if not (d.get("ok") and d.get("slot") in slot_ids and d.get("cam_id")):
                continue
            ts = _parse_ts(d.get("ts"))
            if ts is not None:
                obs_epochs.setdefault(d["cam_id"], []).append(ts.timestamp())
        for cid, epochs in obs_epochs.items():
            _OBS_LOG.seed(cid, epochs)
        if obs_epochs:
            print(f"  observation log seeded for {len(obs_epochs)} cam(s)")
    except Exception as e:
        print(f"  ! observation-log restore skipped ({e})")


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
          f"({len(country_pools[director.active])} cams); widest-grid rule: "
          "first country (by priority) fielding 4 live cams wins; none -> "
          "the grid narrows to 3, then 2, then 1, and widens back as "
          "cameras recover.")
    print(f"per-country pool: {_pool0.max_failures} misses rest a camera "
          f"{_pool0.retry_seconds // 60:.0f} min; tvkur cams + probation cams "
          "rest after a single miss.")
    _br0 = director.breakers[director.active]
    print(f"host breaker: {_br0.threshold} consecutive 403/429s rest ALL of "
          f"a host's cameras for {_br0.rest_seconds // 60} min, then a "
          "single probe request decides (answer = back in rotation).")

    _active_country, _assigned = director.assign(time.time())
    # Width-aware map: slots beyond the grid's current width idle at None.
    assignment = {sid: (_assigned[i] if i < len(_assigned) else None)
                  for i, sid in enumerate(slot_ids)}
    _active_width = len(_assigned)

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

    presence = None if args.no_loiter else PresenceTracker(
        person_sec  = args.loiter_person_min * 60,
        vehicle_sec = args.loiter_vehicle_min * 60,
    )

    # Static-object watch (STATIC_WATCH=0 disables): objects that settle in
    # place for 5+ minutes and then vanish become `static_departed` events -
    # the exact population the loiter path's static-IoU gate refuses on
    # purpose. Same evidence floor as loiter (UN-boosted default gates).
    static_watch = None
    if os.environ.get("STATIC_WATCH", "1") != "0":
        from app.static_watch import StaticWatch
        try:
            _stay_s = float(os.environ.get("STATIC_MIN_STAY_SEC") or 300)
        except ValueError:
            _stay_s = 300.0
        static_watch = StaticWatch(min_stay_sec=_stay_s,
                                   evidence_gates=DEFAULT_PER_CLASS_CONF)
        print(f"static watch: on (settle >= {_stay_s:.0f}s; "
              "STATIC_WATCH=0 to disable)")

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

    print("Restoring analysis state from Firestore...")
    _restore_state(firebase, set(slot_ids))

    # Publish the initial grid config so the dashboard renders immediately.
    slots_meta = [_slot_metadata(s, assignment[s["slot_id"]]) for s in GRID_SLOTS]
    firebase.write_grid_config(slots_meta, country=director.active)

    print(f"Collector started. {len(GRID_SLOTS)} slot(s), active country = "
          f"{director.active}:")
    print("  priority: " + " -> ".join(director.pools[director.active].pool))
    for slot in GRID_SLOTS:
        print(f"  {slot['slot_id']:20s} starts on "
              f"{assignment[slot['slot_id']] or 'idle (grid narrowed)'}")
    print(f"interval={args.interval}s, imgsz={args.imgsz}, burst={args.burst}, "
          f"reid={'on' if reid else 'off'}, conf={args.conf}, "
          f"snapshots={'on' if save_snapshots else 'off'}")
    print(f"anomalies: scene gates (extreme load / obstruction / dark) | "
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

    reid_prune_s     = 6 * 3600
    last_reid_prune   = time.time()

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
            # Re-aim the grid (operator spec 2026-07-18): widest width any
            # country can field, priority order breaking ties, full-order
            # rescan per width - deep same-country bench first, then the
            # next country at width 4, and only after a full loop finds no
            # 4-capable country does the grid narrow to 3, 2, 1.
            adv = director.maybe_advance(round_start)
            if adv:
                print(f"  * country: {adv[0]} cannot field "
                      f"{director.n_active} camera(s) - switching grid to "
                      f"{adv[1]}.")
                _all_miss_rounds = 0

            # One pool decision per round for the active country: the first N
            # healthy cameras in priority order, one per slot, never a
            # duplicate. Cameras of a breaker-tripped host stay out entirely.
            active_country, _assigned = director.assign(round_start)
            round_cams = {sid: (_assigned[i] if i < len(_assigned) else None)
                          for i, sid in enumerate(slot_ids)}
            country_changed = active_country != _active_country
            width_changed = len(_assigned) != _active_width
            if width_changed:
                trend = ("narrowed - no country fields more"
                         if len(_assigned) < _active_width else "recovered")
                print(f"  * grid width: {_active_width} -> {len(_assigned)} "
                      f"slot(s) ({trend})")
            _active_country = active_country
            _active_width = len(_assigned)
            moved = [sid for sid in slot_ids
                     if round_cams[sid] != assignment.get(sid)]
            assignment = round_cams
            if moved or country_changed or width_changed:
                for sid in moved:
                    cam_lbl = assignment[sid]
                    print(f"  * {sid}: -> "
                          + (f"{cam_lbl} ({CAMERAS.get(cam_lbl, {}).get('country', '?')})"
                             if cam_lbl else "idle"))
                slots_meta = [_slot_metadata(s, assignment[s["slot_id"]])
                              for s in GRID_SLOTS]
                try:
                    firebase.write_grid_config(slots_meta, country=active_country)
                except Exception as e:
                    print(f"  ! grid config write failed: {e}")
            round_had_ok = False
            for slot in GRID_SLOTS:
                cam_id = assignment[slot["slot_id"]]
                if cam_id is None:
                    continue          # idle slot: grid is narrower this round
                ok = sample_slot(model, slot, cam_id, firebase, reid=reid,
                                 conf=args.conf, presence=presence,
                                 static_watch=static_watch,
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
                if static_watch is not None:
                    static_watch.prune()
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
        if reid is not None:
            reid.close()
        try:
            from app import heatmap as _hm
            _hm.save_all()
        except Exception:
            pass


if __name__ == "__main__":
    main()
