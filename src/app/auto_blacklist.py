"""Turn repeated user rejects into automatic per-camera blacklist polygons.

The user reviews saved crops (see ``app.labels`` + the dashboard's "Review
detections" panel). When enough consecutive reviews for the same
(camera, class, screen area) come back as ``wrong_label`` /
``not_an_object``, we translate those into a polygon in the camera's
``roi_exclude_class`` config so future collector bursts drop the same
false positive automatically - no code change, no restart, no GPU.

Storage & schema
----------------
Auto-generated polygons live at ``data/blacklist_auto.json``:

    {
        "generated_at": "2026-07-06T18:00:00Z",
        "entries": [
            {
                "cam_id":  "konya_hukumet",
                "cls":     "person",
                "polygon": [[0.10,0.15], [0.28,0.15], [0.28,0.42], [0.10,0.42]],
                "reason":  "3 wrong-label rejects in same bbox area",
                "created_at": "..."
            },
            ...
        ]
    }

Reads: the collector's ``cameras.py`` loads this file on start and merges
each entry into ``cam["roi_exclude_class"]``.

Detection rule
--------------
For each new "reject" review (verdict != correct), look at the last N
rejects of the SAME (cam_id, cls). If M of the last N sit inside the same
20%×20% quadrant of the frame, emit a polygon covering that quadrant with
a small margin. M/N defaults are 3/5.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from app.labels import Review

_SRC_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BLACKLIST_PATH = _SRC_ROOT / "data" / "blacklist_auto.json"

# Sliding-window bookkeeping: read the last N rejects of the same
# (cam_id, cls) and require at least M of them in the same coarse grid cell.
WINDOW_N = 5
QUORUM_M = 3

# Coarse grid the frame is divided into for "same area" detection.
GRID_ROWS = 5
GRID_COLS = 5

# Bounding-box parser for crop paths that carry the source box in the
# filename. The anomaly-crop pipeline names files
# ``<cam>/<frame_stem>__NN_<cls>.jpg`` - no bbox in the name - so we can't
# always recover the pixel box from the path. When we can't, we fall back
# to opening the source frame + running YOLO once; but that path is not
# needed for the common case.
_BBOX_IN_NAME_RE = re.compile(
    r"__(?:\d+_)?(?P<cls>\w+)_(?P<x1>\d+)_(?P<y1>\d+)_(?P<x2>\d+)_(?P<y2>\d+)")


def _load_store(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {"generated_at": "", "entries": []}


def _save_store(store: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(store, indent=2))
    tmp.replace(path)


def _cam_id_from_crop(crop_path: str) -> str | None:
    """Anomaly crops are stored under ``anomalies_crops/<cam_id>/...``; other
    crops don't carry cam_id in the path. Best-effort: return the first
    directory component when we're inside anomalies_crops, else None."""
    parts = crop_path.split("/")
    if len(parts) >= 3 and parts[0] == "anomalies_crops":
        return parts[1]
    return None


def _grid_cell(x: float, y: float) -> tuple[int, int]:
    """Normalized (x,y) -> (row, col) in a GRID_ROWS x GRID_COLS grid."""
    col = min(GRID_COLS - 1, max(0, int(x * GRID_COLS)))
    row = min(GRID_ROWS - 1, max(0, int(y * GRID_ROWS)))
    return row, col


def _cell_polygon(row: int, col: int, margin: float = 0.02) -> list[list[float]]:
    """Normalized polygon of a grid cell, with a small margin so the
    detected foot point stays inside even when a box wobbles at the edge."""
    x0 = col / GRID_COLS - margin
    y0 = row / GRID_ROWS - margin
    x1 = (col + 1) / GRID_COLS + margin
    y1 = (row + 1) / GRID_ROWS + margin
    x0 = max(0.0, x0); y0 = max(0.0, y0)
    x1 = min(1.0, x1); y1 = min(1.0, y1)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _has_polygon(store: dict, cam_id: str, cls: str,
                 polygon: list[list[float]]) -> bool:
    """Two polygons are treated as equal if their (row, col) cell matches."""
    for entry in store.get("entries", []):
        if entry.get("cam_id") == cam_id and entry.get("cls") == cls \
                and entry.get("polygon") == polygon:
            return True
    return False


def consider_review(review_store, latest: Review,
                    blacklist_path: str | Path = DEFAULT_BLACKLIST_PATH,
                    window_n: int = WINDOW_N,
                    quorum_m: int = QUORUM_M) -> dict | None:
    """Called on every submit. Looks at the tail of the review store and
    emits a polygon if the quorum is met.

    Returns the newly-created entry dict, or None when nothing was added.
    """
    if latest.verdict == "correct":
        return None    # accepted labels are not the signal we're after
    cam_id = _cam_id_from_crop(latest.crop_path)
    cls = latest.original_cls
    if not cam_id or not cls:
        return None    # can't attribute this to a camera+class - skip

    # Walk the most recent WINDOW_N rejects for (cam_id, cls).
    all_reviews = list(review_store._by_path.values())  # noqa: SLF001
    all_reviews.sort(key=lambda r: r.reviewed_at, reverse=True)
    matching = [
        r for r in all_reviews
        if r.original_cls == cls
        and r.verdict != "correct"
        and _cam_id_from_crop(r.crop_path) == cam_id
    ][:window_n]
    if len(matching) < quorum_m:
        return None

    # Bucket their inferred grid cells; if any cell has quorum, we ship it.
    cells: dict[tuple[int, int], int] = {}
    for r in matching:
        cell = _cell_from_review(r)
        if cell is None:
            continue
        cells[cell] = cells.get(cell, 0) + 1
    hot = [(c, n) for c, n in cells.items() if n >= quorum_m]
    if not hot:
        return None
    hot.sort(key=lambda t: -t[1])
    (row, col), n = hot[0]
    polygon = _cell_polygon(row, col)

    store_path = Path(blacklist_path)
    store = _load_store(store_path)
    if _has_polygon(store, cam_id, cls, polygon):
        return None    # already there - user is confirming a known bad zone
    entry = {
        "cam_id":     cam_id,
        "cls":        cls,
        "polygon":    polygon,
        "reason":     f"{n} rejects in same area (last {len(matching)} reviews)",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    store.setdefault("entries", []).append(entry)
    _save_store(store, store_path)
    return entry


def _cell_from_review(r: Review,
                      snapshots_root: Path | None = None) -> tuple[int, int] | None:
    """Estimate which grid cell the crop's FOOT POINT lived in on the
    source frame. Returns None when neither shortcut can pin it.

    Three shortcuts, tried in order:
    1. filename-encoded bbox (``__NN_cls_x1_y1_x2_y2.jpg``, used by the
       returning/events writers) - reads pixel-space and normalizes
       against a nominal 1920x1080;
    2. anomaly-crops manifest carries ``foot_norm`` (0..1) per crop -
       written by ``anomaly_crops.refresh`` so a rebuild isn't needed;
    3. fallback: if the crop lives under anomalies_crops/ but we can't
       find its foot_norm, use ``(0.5, 0.5)`` - the center. This is
       aggressive but only fires after the quorum sees N such rejects,
       so it stays conservative in aggregate.
    """
    # Shortcut 1: bbox in filename
    m = _BBOX_IN_NAME_RE.search(r.crop_path)
    if m:
        try:
            x1 = float(m.group("x1")); y1 = float(m.group("y1"))
            x2 = float(m.group("x2")); y2 = float(m.group("y2"))
        except ValueError:
            x1 = y1 = x2 = y2 = -1
        if x2 > x1 and y2 > y1:
            fx = (x1 + x2) / 2.0 / 1920.0
            fy = y2 / 1080.0
            return _grid_cell(fx, fy)

    # Shortcuts 2 & 3: anomaly-crops path
    if not r.crop_path.startswith("anomalies_crops/"):
        return None
    root = snapshots_root or (_SRC_ROOT / "web" / "snapshots")
    manifest_path = root / "anomalies_crops" / ".anomaly_crops.json"
    try:
        data = json.loads(manifest_path.read_text())
        meta = (data.get("crops") or {}).get(r.crop_path) or {}
    except (OSError, ValueError):
        meta = {}
    foot = meta.get("foot_norm")
    if isinstance(foot, list) and len(foot) == 2:
        try:
            return _grid_cell(float(foot[0]), float(foot[1]))
        except (TypeError, ValueError):
            pass
    # Fallback: use frame center. Conservative because the quorum still
    # requires N rejects landing in the same cell.
    return _grid_cell(0.5, 0.5)


def load_auto_blacklist(path: str | Path | None = None) -> dict:
    """Read the current auto-blacklist file as ``{cam_id: {cls: [poly,...]}}``.

    Cameras that don't appear are absent from the dict entirely, so callers
    can merge with an empty default without special-casing. When ``path`` is
    None the current module attribute ``DEFAULT_BLACKLIST_PATH`` is looked
    up at call time - so a caller that monkey-patches it (in tests, or in a
    non-standard install layout) sees the new value.
    """
    store = _load_store(Path(path if path is not None else DEFAULT_BLACKLIST_PATH))
    out: dict[str, dict[str, list]] = {}
    for entry in store.get("entries", []):
        cam = entry.get("cam_id"); cls = entry.get("cls")
        poly = entry.get("polygon")
        if not (cam and cls and poly):
            continue
        out.setdefault(cam, {}).setdefault(cls, []).append(poly)
    return out
