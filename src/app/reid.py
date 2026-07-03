"""Appearance-based re-identification ("have I seen this person/car before?").

Approach - deliberately dependency-free so the notebook + collector + dashboard can all
use it without a torch detour:

  1. Per detection, crop the bounding box and normalize the crop size.
  2. Build a *masked* HSV color histogram (8x8x8 bins) - pixels with V<30 are dropped
     so the night-time sodium-light dominant background doesn't swamp the signature.
  3. Append (aspect_ratio, normalized_area) so persons of similar color but different
     build don't collide.
  4. L2-normalize -> 514-dim unit vector.
  5. Store in a SQLite table; on new detection, cosine-similarity-match against the
     same-class entities; if best match >= THRESH update its (last_seen, sightings)
     and return its id, otherwise insert a fresh entity.

This is a *demo-grade* signature. It will work well in daylight where each person has
distinct clothing colour; it will produce false matches at night (the saved frame from
the Konya camera shows the whole scene is yellow-tinted). For production-grade re-ID
swap embed_crop() for an OSNet/torchreid forward pass - the rest of the registry stays.
"""
from __future__ import annotations

import io
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# How aggressive about declaring a match. 0.92 is conservative (more new IDs);
# drop to 0.85 if you'd rather merge more aggressively.
DEFAULT_THRESHOLD = 0.92
# Match against the K nearest stored entities of the same class (top-1 only kept).
TOPK = 25
# Upper bound on candidates scored per query (most-recently-seen first).
MAX_SCAN = 400
PERSON_CROP = (64, 128)   # w x h
VEHICLE_CROP = (96, 96)

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    entity_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    cam_id       TEXT NOT NULL,
    cls          TEXT NOT NULL,              -- 'person' | 'car' | 'truck' | ...
    first_seen   TEXT NOT NULL,
    last_seen    TEXT NOT NULL,
    sightings    INTEGER NOT NULL DEFAULT 1,
    embedding    BLOB NOT NULL                -- float32 L2-normalized vector
);
CREATE INDEX IF NOT EXISTS idx_entities_cam_cls ON entities(cam_id, cls);

CREATE TABLE IF NOT EXISTS sightings (
    sighting_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id    INTEGER NOT NULL,
    ts           TEXT NOT NULL,
    similarity   REAL,                        -- match score; NULL if new entity
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
);
CREATE INDEX IF NOT EXISTS idx_sightings_entity ON sightings(entity_id);
CREATE INDEX IF NOT EXISTS idx_sightings_ts ON sightings(ts);
"""


@dataclass
class ReidResult:
    entity_id: int
    cls: str
    is_new: bool
    sightings: int
    similarity: float            # 1.0 for brand-new entities (self-similarity)
    # Seconds since the *previous* sighting at this camera, or None for new
    # entities. Lets the collector save a "returning visitor" image only when
    # the gap is meaningful (>= 5 min) instead of every consecutive sample.
    gap_seconds: float | None = None


def embed_crop(crop_bgr: np.ndarray, cls: str) -> np.ndarray | None:
    """Return a 514-d unit-norm appearance vector for the crop, or None if too small."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 8:
        return None
    target = PERSON_CROP if cls == "person" else VEHICLE_CROP
    resized = cv2.resize(crop_bgr, target, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    # mask out very dark pixels (night-light gutter) so the signature reflects the object
    mask = cv2.inRange(hsv[..., 2], 30, 255)
    if int(mask.sum()) == 0:
        mask = None  # crop is entirely dark - use everything
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256]).flatten().astype(np.float32)
    # geometric features make persons vs cars more discriminable
    aspect = w / max(1, h)
    area = (w * h) / (1920 * 1080)
    vec = np.concatenate([hist, np.array([aspect, area], dtype=np.float32)])
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else None


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _gap_seconds(prev_iso: str | None, now_iso: str) -> float | None:
    """Seconds between two stored timestamp strings (UTC), None if unparseable."""
    if not prev_iso:
        return None
    try:
        from datetime import datetime
        # stored strings use either '%Y-%m-%dT%H:%M:%SZ' (this module) or the
        # full ISO including microseconds (the collector record). Both parse.
        def _parse(s: str):
            s2 = s.replace("Z", "+00:00")
            return datetime.fromisoformat(s2)
        return max(0.0, (_parse(now_iso) - _parse(prev_iso)).total_seconds())
    except Exception:
        return None


class ReidStore:
    """SQLite-backed appearance memory. Safe to share across processes via the file."""

    def __init__(self, db_path: str | Path = "data/reid.db",
                 threshold: float = DEFAULT_THRESHOLD):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.conn = sqlite3.connect(str(self.path))
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---- write path ------------------------------------------------------

    def query(self, cam_id: str, cls: str, embedding: np.ndarray) -> ReidResult:
        """Match `embedding` to the stored entities (same cam + class). Either
        update an existing entity or insert a new one. Always returns a result.
        """
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # Pull the same-cam same-class candidates and score them. Bounded to
        # the most recently seen MAX_SCAN entities so a busy day can't turn
        # every query into a full-table scan (prune() keeps the table small,
        # this is the backstop). We pull `last_seen` too so we can report the
        # gap to the caller, which uses it to decide whether to save a
        # "returning visitor" image.
        cur = self.conn.execute(
            "SELECT entity_id, sightings, last_seen, embedding FROM entities "
            "WHERE cam_id=? AND cls=? ORDER BY last_seen DESC LIMIT ?",
            (cam_id, cls, MAX_SCAN))
        best_id, best_sim, best_sight, best_last = None, -1.0, 0, None
        for eid, n_sight, last_seen, blob in cur.fetchall():
            vec = _blob_to_vec(blob)
            sim = float(np.dot(vec, embedding))  # both L2-normalized -> cosine
            if sim > best_sim:
                best_sim, best_id, best_sight, best_last = sim, eid, n_sight, last_seen

        if best_id is not None and best_sim >= self.threshold:
            # match: compute the gap from the prior last_seen *before* we
            # overwrite it, then bump sightings + last_seen.
            gap = _gap_seconds(best_last, ts)
            new_sight = best_sight + 1
            self.conn.execute(
                "UPDATE entities SET last_seen=?, sightings=? WHERE entity_id=?",
                (ts, new_sight, best_id))
            self.conn.execute(
                "INSERT INTO sightings (entity_id, ts, similarity) VALUES (?, ?, ?)",
                (best_id, ts, best_sim))
            self.conn.commit()
            return ReidResult(best_id, cls, is_new=False, sightings=new_sight,
                              similarity=best_sim, gap_seconds=gap)

        # new entity
        cur = self.conn.execute(
            "INSERT INTO entities (cam_id, cls, first_seen, last_seen, sightings, embedding)"
            " VALUES (?, ?, ?, ?, 1, ?)",
            (cam_id, cls, ts, ts, embedding.astype(np.float32).tobytes()))
        eid = cur.lastrowid
        self.conn.execute(
            "INSERT INTO sightings (entity_id, ts, similarity) VALUES (?, ?, NULL)",
            (eid, ts))
        self.conn.commit()
        return ReidResult(eid, cls, is_new=True, sightings=1, similarity=1.0,
                          gap_seconds=None)

    def update_from_frame(self, cam_id: str, frame_bgr: np.ndarray,
                          boxes: list[dict]) -> list[ReidResult]:
        """Convenience: embed every box's crop and query the registry.

        boxes is a list of {x1,y1,x2,y2,cls,conf} dicts.
        """
        results = []
        H, W = frame_bgr.shape[:2]
        for b in boxes:
            x1 = max(0, int(b["x1"])); y1 = max(0, int(b["y1"]))
            x2 = min(W, int(b["x2"])); y2 = min(H, int(b["y2"]))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            emb = embed_crop(crop, b["cls"])
            if emb is None:
                continue
            results.append(self.query(cam_id, b["cls"], emb))
        return results

    def prune(self, max_age_hours: float = 48.0) -> int:
        """Delete entities (and their sightings) not seen for `max_age_hours`.

        A public street cam sees thousands of one-off passers-by a day; without
        pruning, the registry grows unbounded and every query() slows down while
        the "unique entities" stat inflates forever. 48h keeps the
        "came back the next day" case the dashboard reports, drops the rest.
        `last_seen` is stored as a fixed-width UTC ISO string, so plain string
        comparison against the cutoff is correct.
        """
        cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                               time.gmtime(time.time() - max_age_hours * 3600))
        self.conn.execute(
            "DELETE FROM sightings WHERE entity_id IN "
            "(SELECT entity_id FROM entities WHERE last_seen < ?)", (cutoff,))
        cur = self.conn.execute(
            "DELETE FROM entities WHERE last_seen < ?", (cutoff,))
        removed = cur.rowcount or 0
        self.conn.commit()
        return removed

    # ---- read path -------------------------------------------------------

    def stats(self, cam_id: str | None = None) -> dict:
        """Roll-up for the dashboard: unique entities, regulars, sightings distribution."""
        where = "WHERE cam_id=?" if cam_id else ""
        args = (cam_id,) if cam_id else ()
        cur = self.conn.execute(
            f"SELECT cls, COUNT(*) AS uniq, SUM(sightings) AS total, "
            f"       SUM(CASE WHEN sightings >= 3 THEN 1 ELSE 0 END) AS regulars "
            f"FROM entities {where} GROUP BY cls",
            args,
        )
        per_class = {row[0]: {"unique": row[1], "total_sightings": row[2],
                              "regulars": row[3]} for row in cur.fetchall()}
        cur = self.conn.execute(
            f"SELECT COUNT(*), SUM(sightings) FROM entities {where}", args)
        total_uniq, total_sight = cur.fetchone()
        return {
            "total_unique":    total_uniq or 0,
            "total_sightings": total_sight or 0,
            "per_class":       per_class,
        }

    def top_regulars(self, cam_id: str | None = None, n: int = 10) -> list[dict]:
        """Return the n entities with the most sightings."""
        where = "WHERE cam_id=?" if cam_id else ""
        args = (cam_id,) if cam_id else ()
        cur = self.conn.execute(
            f"SELECT entity_id, cls, sightings, first_seen, last_seen "
            f"FROM entities {where} ORDER BY sightings DESC LIMIT ?",
            args + (n,))
        return [{"entity_id": r[0], "cls": r[1], "sightings": r[2],
                 "first_seen": r[3], "last_seen": r[4]} for r in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()
