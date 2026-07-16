"""Per-entity sighting gallery - the evidence behind "returning visitor".

A returning-visitor event used to carry exactly ONE snapshot (the moment of
return), so the operator had nothing to compare it against. This module
gives every entity the re-ID registry tracks a small rolling gallery: the
collector drops a crop here on each sighting once an entity has proven
itself (3+ sightings), capped per entity and globally, and the events
accordion shows the whole gallery side by side.

Layout:  web/snapshots/entities/<cam_id>/<entity_id>/<ts_us>.jpg
Caps:    PER_ENTITY_CAP newest crops per entity (oldest deleted),
         GLOBAL_CAP_FILES across the tree (LRU) - the pool is bounded no
         matter how many entities a busy week produces.
Synced:  the tree rides pool_sync like the other pools, so the operator's
         machine mirrors a bounded newest slice of it.
"""
from __future__ import annotations

import itertools
import os
import time
from pathlib import Path

from app.visual_search import SNAPSHOTS_ROOT

GALLERY_SUBDIR = "entities"
PER_ENTITY_CAP = int(os.environ.get("ENTITY_GALLERY_PER_ENTITY") or 8)
GLOBAL_CAP_FILES = int(os.environ.get("ENTITY_GALLERY_MAX_FILES") or 1200)
MIN_CROP_SIDE = 24
# One crop per entity per this many seconds - a 2-frame burst must not burn
# two gallery slots on near-identical crops.
PER_ENTITY_MIN_GAP_S = 60.0

_LAST_SAVE: dict[tuple[str, int], float] = {}
_SAVE_SEQ = itertools.count()


def _dir(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> Path:
    return Path(snapshots_root) / GALLERY_SUBDIR


def save_sighting(cam_id: str, entity_id: int, frame, box: dict,
                  snapshots_root: str | Path = SNAPSHOTS_ROOT) -> str | None:
    """Crop `box` out of `frame` into the entity's gallery. Returns the rel
    path, or None when skipped (tiny crop / gap throttle / write failure)."""
    now = time.time()
    key = (cam_id, int(entity_id))
    if now - _LAST_SAVE.get(key, 0.0) < PER_ENTITY_MIN_GAP_S:
        return None
    import cv2
    H, W = frame.shape[:2]
    x1 = max(0, int(box["x1"])); y1 = max(0, int(box["y1"]))
    x2 = min(W, int(box["x2"])); y2 = min(H, int(box["y2"]))
    if x2 - x1 < MIN_CROP_SIDE or y2 - y1 < MIN_CROP_SIDE:
        return None
    out_dir = _dir(snapshots_root) / cam_id / str(int(entity_id))
    out_dir.mkdir(parents=True, exist_ok=True)
    # Windows time.time() ticks can repeat across rapid saves; the counter
    # suffix keeps names unique (and zero-padded so string sort stays
    # chronological) instead of silently overwriting the previous crop.
    out_path = out_dir / f"{int(now * 1_000_000)}_{next(_SAVE_SEQ):04d}.jpg"
    if not cv2.imwrite(str(out_path), frame[y1:y2, x1:x2],
                       [cv2.IMWRITE_JPEG_QUALITY, 85]):
        return None
    _LAST_SAVE[key] = now
    # per-entity cap: newest PER_ENTITY_CAP survive
    crops = sorted(out_dir.glob("*.jpg"))
    for p in crops[:-PER_ENTITY_CAP]:
        try:
            p.unlink()
        except OSError:
            pass
    _enforce_global_cap(snapshots_root)
    return str(out_path.relative_to(snapshots_root)).replace("\\", "/")


def _enforce_global_cap(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> int:
    root = _dir(snapshots_root)
    if not root.is_dir():
        return 0
    files = sorted(root.rglob("*.jpg"), key=lambda p: p.stat().st_mtime)
    n = 0
    for p in files[:max(0, len(files) - GLOBAL_CAP_FILES)]:
        try:
            p.unlink()
            n += 1
        except OSError:
            continue
    return n


def list_sightings(cam_id: str, entity_id: int,
                   snapshots_root: str | Path = SNAPSHOTS_ROOT) -> list[dict]:
    """Newest-first gallery entries for one entity:
    [{"url": "/snapshots/entities/...", "ts": iso}, ...]"""
    d = _dir(snapshots_root) / cam_id / str(int(entity_id))
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.jpg"), reverse=True):
        try:
            us = int(p.stem.split("_")[0])
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(us / 1_000_000))
        except ValueError:
            ts = ""
        rel = str(p.relative_to(snapshots_root)).replace("\\", "/")
        out.append({"url": f"/snapshots/{rel}", "ts": ts})
    return out
