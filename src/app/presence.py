"""Prolonged-presence detection: "this person/vehicle has been HERE too long".

Built on what the sampling architecture already produces - re-ID matches every
interval - rather than on continuous tracking the collector can't afford:

  * an entity re-matched with its box overlapping its previous box is a
    continuation of the same stay (stationary objects re-match reliably even
    with the histogram embedder: same spot, same lighting, same pose);
  * a move (low IoU) or a continuity gap (entity unmatched for longer than
    `continuity_gap_sec`) resets the stay;
  * once a stay exceeds the per-class threshold - inside the camera's
    loiter polygon, when one is configured - a `loiter` event fires (once,
    then re-arms after `realert_sec`).

For a market street this reads as: a person planted in front of a shop for
5+ minutes, or a vehicle parked/idling in a no-stopping zone for 15+ minutes.
"""
from __future__ import annotations

import time

from app.detect_core import box_iou, point_in_polygon

LOITER_PERSON_SEC_DEFAULT  = 300.0
LOITER_VEHICLE_SEC_DEFAULT = 900.0


class PresenceTracker:
    """Per-(cam, entity) stay accumulator. Feed it every re-ID result."""

    def __init__(self,
                 person_sec: float = LOITER_PERSON_SEC_DEFAULT,
                 vehicle_sec: float = LOITER_VEHICLE_SEC_DEFAULT,
                 match_iou: float = 0.30,
                 continuity_gap_sec: float = 180.0,
                 realert_sec: float = 1800.0):
        self.person_sec  = person_sec
        self.vehicle_sec = vehicle_sec
        self.match_iou   = match_iou
        self.continuity_gap_sec = continuity_gap_sec
        self.realert_sec = realert_sec
        # (cam_id, entity_id) -> [stay_start, last_seen, box, last_alert_ts]
        # last_alert_ts is None until the first alert - an epoch-0 sentinel
        # would silently suppress alerts whenever now < realert_sec (tests,
        # simulated clocks).
        self._stays: dict[tuple[str, int], list] = {}

    def threshold_for(self, cls: str, cam: dict | None = None) -> float:
        cam = cam or {}
        if cls == "person":
            return float(cam.get("loiter_person_sec", self.person_sec))
        return float(cam.get("loiter_vehicle_sec", self.vehicle_sec))

    def observe(self, cam_id: str, entity_id: int, cls: str, box: dict,
                frame_shape, cam: dict | None = None,
                now: float | None = None) -> dict | None:
        """Feed one matched detection; returns a loiter event dict when a stay
        crosses its threshold, else None."""
        now = time.time() if now is None else now
        key = (cam_id, entity_id)
        stay = self._stays.get(key)

        if stay is not None:
            _, last_seen, prev_box, _ = stay
            if (now - last_seen > self.continuity_gap_sec
                    or box_iou(prev_box, box) < self.match_iou):
                stay = None                      # moved, or continuity broken

        if stay is None:
            self._stays[key] = [now, now, box, None]
            return None
        stay[1] = now
        stay[2] = box

        duration = now - stay[0]
        if duration < self.threshold_for(cls, cam):
            return None
        if stay[3] is not None and now - stay[3] < self.realert_sec:
            return None

        loiter_roi = (cam or {}).get("loiter_roi")
        if loiter_roi:
            H, W = frame_shape[:2]
            fx = (box["x1"] + box["x2"]) / 2.0 / W
            fy = box["y2"] / H
            if not point_in_polygon(fx, fy, loiter_roi):
                return None

        stay[3] = now
        return {
            "kind": "loiter",
            "cls": cls,
            "entity_id": entity_id,
            "cam_id": cam_id,
            "duration_sec": round(duration, 1),
            "box": {k: box[k] for k in ("x1", "y1", "x2", "y2")},
        }

    def prune(self, max_age_sec: float = 6 * 3600,
              now: float | None = None) -> int:
        """Drop stays not refreshed for `max_age_sec`. Returns #dropped."""
        now = time.time() if now is None else now
        stale = [k for k, s in self._stays.items() if now - s[1] > max_age_sec]
        for k in stale:
            self._stays.pop(k, None)
        return len(stale)
