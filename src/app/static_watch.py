"""Static-object watch: things that SETTLED in place, then vanished.

The mirror image of presence.py. The loiter path deliberately REFUSES
perfectly static stays (LOITER_MAX_STATIC_IOU - a box that never moved is
a kiosk, not a loiterer), so the information "this object has been parked
here for 20 minutes" was being accumulated and then thrown away. This
module keeps it, and answers the question the loiter path can't: **when
did the static thing LEAVE?** A parked car pulling out, a market stall
packing up, a bag that sat by a wall and is suddenly gone - each becomes a
`static_departed` event carrying how long the object stayed and the crop
captured while it was still there.

Life cycle of an ANCHOR:
  candidate  - a detection with no matching anchor starts one;
  (grows)    - a same-class detection overlapping >= `match_iou` continues
               it: hits += 1, box drifts slowly (EMA) so slight jitter or
               a lighting shift doesn't shed the anchor;
  settled    - stay >= `min_stay_sec` AND >= `min_hits` sightings AND
               median confidence clears the class's UN-boosted default
               gate (same evidence floor as loiter alerts - a conf-0.23
               "car" that only exists on a loosened gate is a shadow, not
               a vehicle). At settle time a crop is captured - the LAST
               look at the object, available later when it's gone;
  departed   - a SETTLED anchor unmatched for `depart_misses` consecutive
               successful samples emits the event and retires. Candidates
               that fizzle just evaporate.

Honesty guards (each one maps to a real failure mode):
  * misses only count on samples that actually RAN - the collector calls
    observe() on successful grabs only, so a stream outage can never fake
    a departure (the anchor just waits; `stale_sec` eventually clears
    anchors whose camera left the grid for hours);
  * a DARK frame (luma < dark_luma) skips miss-counting - losing sight of
    everything at night is lighting, not departure;
  * a SCENE WIPE - most settled anchors unmatched at once - skips
    miss-counting too: the camera moved, refocused, or switched source;
    one object leaving is an event, all of them "leaving" is a cut.
"""
from __future__ import annotations

import time

from app.detect_core import box_iou

STATIC_MIN_STAY_SEC = 300.0   # the operator's "more than five minutes"
STATIC_MATCH_IOU = 0.55       # continuity: static boxes jitter a few px
STATIC_MIN_HITS = 4           # sightings before a stay can settle
STATIC_DEPART_MISSES = 2      # consecutive observed samples without it
STATIC_EMA = 0.9              # old-box weight when drifting the anchor
STATIC_STALE_SEC = 2 * 3600.0 # drop anchors idle this long (camera left)
STATIC_MAX_ANCHORS = 60       # per camera - a packed lot stays bounded
SCENE_WIPE_MIN = 2            # a wipe needs at least this many settled...
SCENE_WIPE_FRAC = 0.5         # ...and >= this fraction gone at once
DARK_LUMA = 40.0              # below this the frame can't testify


class StaticWatch:
    """Per-camera anchor registry. Feed every successful sample's boxes."""

    def __init__(self,
                 min_stay_sec: float = STATIC_MIN_STAY_SEC,
                 match_iou: float = STATIC_MATCH_IOU,
                 min_hits: int = STATIC_MIN_HITS,
                 depart_misses: int = STATIC_DEPART_MISSES,
                 evidence_gates: dict | None = None,
                 dark_luma: float = DARK_LUMA,
                 max_anchors: int = STATIC_MAX_ANCHORS):
        self.min_stay_sec = min_stay_sec
        self.match_iou = match_iou
        self.min_hits = min_hits
        self.depart_misses = depart_misses
        # cls -> conf floor a stay must clear (median) to ever settle.
        # None skips the check (tests); the collector passes detect_core's
        # DEFAULT_PER_CLASS_CONF - deliberately the UN-boosted defaults.
        self.evidence_gates = evidence_gates
        self.dark_luma = dark_luma
        self.max_anchors = max_anchors
        self._anchors: dict[str, list[dict]] = {}
        self._next_id = 1

    # -- internals ---------------------------------------------------------

    def _settle_ok(self, a: dict, now: float) -> bool:
        if a["settled"] or now - a["first_ts"] < self.min_stay_sec \
                or a["hits"] < self.min_hits:
            return False
        if self.evidence_gates is not None:
            confs = sorted(a["confs"])
            med = confs[len(confs) // 2] if confs else 0.0
            if med < float(self.evidence_gates.get(a["cls"], 0.35)):
                return False
        return True

    @staticmethod
    def _crop_jpeg(frame, box: dict) -> bytes | None:
        """Encode the anchor's crop - the evidence shown when it departs.
        Best-effort: None on any failure (headless test envs pass frame=None)."""
        if frame is None:
            return None
        try:
            import cv2
            H, W = frame.shape[:2]
            x1 = max(0, int(box["x1"])); y1 = max(0, int(box["y1"]))
            x2 = min(W, int(box["x2"])); y2 = min(H, int(box["y2"]))
            if not (x2 > x1 and y2 > y1):
                return None
            ok, buf = cv2.imencode(".jpg", frame[y1:y2, x1:x2],
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buf.tobytes() if ok else None
        except Exception:
            return None

    # -- main entry --------------------------------------------------------

    def observe(self, cam_id: str, boxes: list[dict], frame_shape,
                luma: float | None = None, frame=None,
                now: float | None = None) -> list[dict]:
        """One successful sample. Returns `static_departed` event dicts."""
        now = time.time() if now is None else now
        anchors = self._anchors.setdefault(cam_id, [])

        # Greedy same-class IoU matching, best overlap first - one
        # detection continues at most one anchor and vice versa.
        cands: list[tuple[float, int, int]] = []
        for ai, a in enumerate(anchors):
            for bi, b in enumerate(boxes):
                if b.get("cls") != a["cls"]:
                    continue
                iou = box_iou(a["box"], b)
                if iou >= self.match_iou:
                    cands.append((-iou, ai, bi))
        cands.sort()
        used_a: set[int] = set()
        used_b: set[int] = set()
        for _niou, ai, bi in cands:
            if ai in used_a or bi in used_b:
                continue
            used_a.add(ai)
            used_b.add(bi)
            a, b = anchors[ai], boxes[bi]
            w = STATIC_EMA
            a["box"] = {k: w * a["box"][k] + (1 - w) * float(b[k])
                        for k in ("x1", "y1", "x2", "y2")}
            a["last_ts"] = now
            a["hits"] += 1
            a["misses"] = 0
            confs = a["confs"]
            confs.append(float(b.get("conf") or 0.0))
            if len(confs) > 20:
                del confs[0]
            if self._settle_ok(a, now):
                a["settled"] = True
                a["settle_ts"] = now
                a["crop_jpeg"] = self._crop_jpeg(frame, a["box"])

        # New candidates from unmatched detections. Anchors born this very
        # sample sit at index >= n_before and are exempt from this sample's
        # miss accounting - being new is not being missed.
        n_before = len(anchors)
        for bi, b in enumerate(boxes):
            if bi in used_b:
                continue
            anchors.append({
                "id": self._next_id, "cls": b.get("cls"),
                "box": {k: float(b[k]) for k in ("x1", "y1", "x2", "y2")},
                "first_ts": now, "last_ts": now,
                "hits": 1, "misses": 0,
                "confs": [float(b.get("conf") or 0.0)],
                "settled": False, "settle_ts": None, "crop_jpeg": None,
            })
            self._next_id += 1

        # Miss accounting. A dark frame can't testify about absence.
        events: list[dict] = []
        dark = luma is not None and luma < self.dark_luma
        settled_ids = [a["id"] for a in anchors if a["settled"]]
        settled_missed = [a["id"] for ai, a in enumerate(anchors[:n_before])
                          if a["settled"] and ai not in used_a]
        wipe = (len(settled_missed) >= SCENE_WIPE_MIN
                and len(settled_ids) > 0
                and len(settled_missed) / len(settled_ids) >= SCENE_WIPE_FRAC)
        if not dark and not wipe:
            survivors: list[dict] = []
            for ai, a in enumerate(anchors):
                if ai in used_a or ai >= n_before:
                    survivors.append(a)
                    continue
                a["misses"] += 1
                if a["settled"] and a["misses"] >= self.depart_misses:
                    confs = sorted(a["confs"])
                    events.append({
                        "kind": "static_departed",
                        "cls": a["cls"],
                        "anchor_id": a["id"],
                        "box": dict(a["box"]),
                        "dwell_sec": round(a["last_ts"] - a["first_ts"], 1),
                        "first_ts": a["first_ts"],
                        "last_ts": a["last_ts"],
                        "settle_ts": a["settle_ts"],
                        "hits": a["hits"],
                        "conf_median": (confs[len(confs) // 2]
                                        if confs else 0.0),
                        "crop_jpeg": a["crop_jpeg"],
                    })
                    continue                       # departed: retire
                if not a["settled"] and a["misses"] >= self.depart_misses:
                    continue                       # fizzled candidate
                survivors.append(a)
            anchors[:] = survivors

        # Bound the registry: shed the weakest unsettled candidates first.
        if len(anchors) > self.max_anchors:
            anchors.sort(key=lambda a: (a["settled"], a["hits"]))
            del anchors[: len(anchors) - self.max_anchors]

        return events

    def prune(self, max_age_sec: float = STATIC_STALE_SEC,
              now: float | None = None) -> int:
        """Drop anchors not refreshed for `max_age_sec` (camera off-grid).
        Silent by design: an unobserved anchor proves nothing either way."""
        now = time.time() if now is None else now
        dropped = 0
        for cam_id, anchors in self._anchors.items():
            keep = [a for a in anchors if now - a["last_ts"] <= max_age_sec]
            dropped += len(anchors) - len(keep)
            self._anchors[cam_id] = keep
        return dropped

    def counts(self, cam_id: str | None = None) -> dict:
        """Observability: {"anchors": n, "settled": n} (one cam or all)."""
        pools = ([self._anchors.get(cam_id, [])] if cam_id is not None
                 else list(self._anchors.values()))
        n = sum(len(p) for p in pools)
        s = sum(1 for p in pools for a in p if a["settled"])
        return {"anchors": n, "settled": s}
