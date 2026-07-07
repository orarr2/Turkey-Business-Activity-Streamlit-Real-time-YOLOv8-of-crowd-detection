"""Continuous per-object crop pool for the review UI.

Motivation: the ``returning/`` / ``events/`` / ``anomalies/`` subtrees only
populate when specific things happen (a re-ID rematch, a loitering event,
a z-score anomaly). Cameras that don't produce those events keep the
review pool empty forever, so the user sees "every stored crop has been
reviewed" the moment the dashboard loads and has nothing to teach the
system with.

This module gives the collector a very cheap, bounded, per-camera crop
pool. On every ``LIVE_SAMPLE_EVERY_N``-th sample burst it saves ONE
randomly chosen detection crop from the burst's representative frame to
``web/snapshots/live_samples/<cam_id>/<mtime>_<cls>_<conf>.jpg`` and
enforces an LRU cap of ``LIVE_SAMPLE_MAX_FILES`` files across the tree.
No Firestore writes, no Storage uploads, no extra egress - the crops
just sit on the VM disk, ready for the review UI to sample and for the
search UI to browse.

Size budget: at the shipped defaults (200 files × ~50 KB) the tree stays
around ~10 MB - well inside the e2-micro's 30 GB Always-Free disk quota.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

from app.visual_search import SNAPSHOTS_ROOT

LIVE_SAMPLES_SUBDIR = "live_samples"

# Save one crop every N bursts, per camera. Balances "steady supply" (so
# the review UI has fresh material within minutes of a restart) against
# "not overwhelming the disk" (LRU eviction is cheap but each write costs
# CPU and adds to the pool of crops the user has to grind through).
LIVE_SAMPLE_EVERY_N = int(os.environ.get("LIVE_SAMPLE_EVERY_N") or 5)

# Hard cap on the total file count in the pool. Oldest files get evicted
# first once the pool grows past this. At ~50 KB per crop this is ~10 MB
# on disk for the default 200 files.
LIVE_SAMPLE_MAX_FILES = int(os.environ.get("LIVE_SAMPLE_MAX_FILES") or 200)

MIN_CROP_SIDE = 24     # matches anomaly_crops - anything smaller is noise


def _dir(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> Path:
    return Path(snapshots_root) / LIVE_SAMPLES_SUBDIR


def _round_counter_path(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> Path:
    """Per-camera burst counter file so ``should_sample`` doesn't need
    global state - the collector process may restart mid-day and we want
    the sampling cadence to survive. One tiny int per line, keyed by cam."""
    return _dir(snapshots_root) / ".burst_counts.txt"


def _read_counts(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in path.read_text().splitlines():
            k, _, v = line.strip().partition(" ")
            if k and v.isdigit():
                out[k] = int(v)
    except OSError:
        pass
    return out


def _write_counts(path: Path, counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(f"{k} {v}" for k, v in sorted(counts.items())))
    tmp.replace(path)


def should_sample(cam_id: str,
                  every_n: int = LIVE_SAMPLE_EVERY_N,
                  snapshots_root: str | Path = SNAPSHOTS_ROOT) -> bool:
    """Called once per burst per camera. Returns True on every Nth burst.

    Uses a persisted counter file so restarts don't skew the phase and
    every camera keeps its own cadence independently.
    """
    if every_n <= 1:
        return True
    p = _round_counter_path(snapshots_root)
    counts = _read_counts(p)
    counts[cam_id] = counts.get(cam_id, 0) + 1
    _write_counts(p, counts)
    return counts[cam_id] % every_n == 0


def save_crop(cam_id: str, frame, boxes: list[dict],
              snapshots_root: str | Path = SNAPSHOTS_ROOT,
              cap_files: int | None = None,
              rng=None) -> str | None:
    """Save one random detection crop from ``boxes`` to the pool.

    Returns the relative-to-``snapshots_root`` path of the saved file, or
    None when there was nothing worth saving. Enforces the LRU cap after
    writing so the pool stays bounded.
    """
    if not boxes:
        return None
    import cv2   # heavy import gated to the function
    rng = rng or random
    # Prefer classes users care about (person + vehicles + train), not
    # every-random-bicycle. Weight person 3x so pedestrian pools stay dense
    # even on car-heavy cameras.
    weighted = []
    for b in boxes:
        w = 3 if b.get("cls") == "person" else 1
        weighted.extend([b] * w)
    b = rng.choice(weighted)
    H, W = frame.shape[:2]
    x1 = max(0, int(b["x1"])); y1 = max(0, int(b["y1"]))
    x2 = min(W, int(b["x2"])); y2 = min(H, int(b["y2"]))
    if x2 - x1 < MIN_CROP_SIDE or y2 - y1 < MIN_CROP_SIDE:
        return None
    crop = frame[y1:y2, x1:x2]
    out_dir = _dir(snapshots_root) / cam_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    conf_pct = int(round((b.get("conf") or 0) * 100))
    name = f"{ts}_{b.get('cls','?')}_{conf_pct:02d}.jpg"
    out_path = out_dir / name
    if not cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        return None
    rel = str(out_path.relative_to(snapshots_root)).replace("\\", "/")
    enforce_cap(snapshots_root, cap_files)
    return rel


def enforce_cap(snapshots_root: str | Path = SNAPSHOTS_ROOT,
                cap_files: int | None = None) -> tuple[int, int]:
    """Delete oldest crops until file count <= cap. Returns
    ``(deleted_count, freed_bytes)``."""
    if cap_files is None:
        cap_files = LIVE_SAMPLE_MAX_FILES
    root = _dir(snapshots_root)
    if not root.is_dir():
        return 0, 0
    files = [p for p in root.rglob("*.jpg")]
    if len(files) <= cap_files:
        return 0, 0
    files.sort(key=lambda p: p.stat().st_mtime)
    to_delete = files[: len(files) - cap_files]
    freed = 0; n = 0
    for p in to_delete:
        try:
            freed += p.stat().st_size
            p.unlink()
            n += 1
        except OSError:
            continue
    return n, freed


def usage_stats(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> dict:
    root = _dir(snapshots_root)
    if not root.is_dir():
        return {"count": 0, "bytes": 0,
                "cap_files": LIVE_SAMPLE_MAX_FILES,
                "every_n": LIVE_SAMPLE_EVERY_N}
    files = list(root.rglob("*.jpg"))
    total = sum(p.stat().st_size for p in files if p.is_file())
    return {
        "count":     len(files),
        "bytes":     total,
        "cap_files": LIVE_SAMPLE_MAX_FILES,
        "every_n":   LIVE_SAMPLE_EVERY_N,
        "path":      str(root),
    }


def clear_all(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> dict:
    root = _dir(snapshots_root)
    if not root.is_dir():
        return {"deleted": 0, "bytes_freed": 0}
    freed = 0; n = 0
    for p in list(root.rglob("*.jpg")):
        try:
            freed += p.stat().st_size
            p.unlink()
            n += 1
        except OSError:
            continue
    # Prune per-cam sub-dirs
    for d in sorted(root.rglob("*"), reverse=True):
        if d.is_dir():
            try: d.rmdir()
            except OSError: pass
    # Reset burst counter so cadence starts fresh from a known point
    p = _round_counter_path(snapshots_root)
    try:
        if p.is_file(): p.unlink()
    except OSError:
        pass
    return {"deleted": n, "bytes_freed": freed}
