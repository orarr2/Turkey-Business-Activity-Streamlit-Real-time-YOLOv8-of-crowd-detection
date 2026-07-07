"""Extract per-object crops from stored anomaly frames.

The collector saves the full frame of every flagged anomaly under
``web/snapshots/anomalies/`` (a whole street, not one object). The search
and review UIs work on per-object crops, so anomaly frames were invisible
to both surfaces until this module ran YOLO once per new frame and wrote
per-object crops to ``web/snapshots/anomalies_crops/``.

The extractor manages the on-disk directory as a bounded cache:

* **Size cap** - the directory total is kept under ``ANOMALY_CROPS_MAX_MB``
  (default 300 MB, small compared to the e2-micro 30 GB Free-Tier disk).
  When a new crop would push the total over, the oldest crops are deleted
  first (LRU by mtime).
* **De-duplication** - a new crop is compared cosine-similarity-wise (via
  the same embedder the rest of the system uses) against recent crops from
  the same camera+class. A crop within ``DEDUP_THRESHOLD`` of an existing
  one is dropped, so "same car parked all day" does not flood the store.
* **Transparency** - ``usage_stats()`` returns bytes/count/cap so the
  dashboard can show an indicator instead of surprising the user with a
  full disk.
* **Manual cleanup** - ``clear_all()`` wipes the tree (used by the
  dashboard's "clear anomaly crops" button and by the collector-tests).

The manifest ``.anomaly_crops.json`` at the tree root records, per crop,
which source frame it came from and the cam_id + cls + conf the model
gave it, so downstream code (search index, review sampler) can present
the crop with full context instead of a bare JPEG.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from app.detect_core import DEFAULT_IMGSZ, DEFAULT_PER_CLASS_CONF
from app.visual_search import SNAPSHOTS_ROOT

# --- constants -----------------------------------------------------------------

ANOMALY_FRAMES_SUBDIR = "anomalies"
CROPS_SUBDIR          = "anomalies_crops"

# Env-var override so admins can size the cache to their VM. 300 MB fits the
# e2-micro with a huge margin; larger hosts can raise it freely.
_DEFAULT_CAP_MB = 300
CAP_MB = int(os.environ.get("ANOMALY_CROPS_MAX_MB") or _DEFAULT_CAP_MB)
CAP_BYTES = CAP_MB * 1024 * 1024

# Cosine similarity above which a candidate crop is treated as a duplicate of
# an existing crop (same camera + same class). Well above the "same appearance"
# identity threshold most embedders use for re-ID, so it only kills near-copies.
DEDUP_THRESHOLD = 0.95

# The manifest lives next to the crops directory - one JSON per tree so
# reads/writes stay atomic even without a DB.
MANIFEST_NAME = ".anomaly_crops.json"

# Minimum on-side pixels for a crop worth indexing. Anomaly frames are
# high-res so tiny detections at the horizon are usually noise; keeping the
# floor at 24 px per side matches what the re-ID crop paths already accept.
MIN_CROP_SIDE = 24


# --- helpers -------------------------------------------------------------------


def _dir(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> Path:
    return Path(snapshots_root) / CROPS_SUBDIR


def _manifest_path(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> Path:
    return _dir(snapshots_root) / MANIFEST_NAME


def _load_manifest(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> dict:
    """Manifest schema:
        {
          "processed_frames": {rel_frame_path: mtime, ...},
          "crops": {rel_crop_path: {cam_id, cls, conf, source_frame, saved_at}}
        }
    """
    p = _manifest_path(snapshots_root)
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {"processed_frames": {}, "crops": {}}


def _save_manifest(m: dict, snapshots_root: str | Path = SNAPSHOTS_ROOT) -> None:
    p = _manifest_path(snapshots_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(m))
    tmp.replace(p)


def _cam_id_from_frame(rel_frame: str) -> str:
    """Anomaly frames are stored as ``anomalies/<cam_id>/<file>.jpg``. Recover
    the cam from the first path component so downstream code can look up its
    per-camera config."""
    parts = rel_frame.split("/")
    return parts[1] if len(parts) >= 3 else "unknown"


def _extract_boxes(model, image_bgr, imgsz: int | None) -> list[dict]:
    from app.detect_core import detect_with_boxes
    _, boxes = detect_with_boxes(model, image_bgr, conf=0.30, imgsz=imgsz,
                                 per_class_conf=DEFAULT_PER_CLASS_CONF)
    return boxes


def _crop_from_frame(image_bgr, box: dict):
    H, W = image_bgr.shape[:2]
    x1 = max(0, int(box["x1"])); y1 = max(0, int(box["y1"]))
    x2 = min(W, int(box["x2"])); y2 = min(H, int(box["y2"]))
    if x2 - x1 < MIN_CROP_SIDE or y2 - y1 < MIN_CROP_SIDE:
        return None
    return image_bgr[y1:y2, x1:x2]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b))


def _is_duplicate(vec: np.ndarray, cam_id: str, cls: str,
                  seen_by_cam_cls: dict[tuple[str, str], list[np.ndarray]]) -> bool:
    """Compare against every prior crop from the same (cam, cls). O(N) per
    check but N is small (the cache is capped and we only compare within one
    cam+cls). Kept simple; if this ever becomes a bottleneck, swap in
    faiss/HNSW."""
    key = (cam_id, cls)
    for prev in seen_by_cam_cls.get(key, ()):
        if _cosine(prev, vec) >= DEDUP_THRESHOLD:
            return True
    return False


# --- LRU eviction --------------------------------------------------------------


def _dir_bytes(root: Path) -> int:
    total = 0
    for p in root.rglob("*.jpg"):
        try:
            total += p.stat().st_size
        except OSError:
            continue
    return total


def enforce_size_cap(snapshots_root: str | Path = SNAPSHOTS_ROOT,
                     cap_bytes: int | None = None) -> tuple[int, int]:
    """Delete oldest crops until the tree total is <= cap. Returns
    ``(deleted_count, freed_bytes)``. Manifest entries for deleted crops
    are pruned in the same pass."""
    if cap_bytes is None:
        cap_bytes = CAP_BYTES
    root = _dir(snapshots_root)
    if not root.is_dir():
        return 0, 0
    used = _dir_bytes(root)
    if used <= cap_bytes:
        return 0, 0
    # oldest first
    files = sorted(root.rglob("*.jpg"), key=lambda p: p.stat().st_mtime)
    freed = 0
    deleted_paths: list[str] = []
    for p in files:
        if used - freed <= cap_bytes:
            break
        try:
            sz = p.stat().st_size
            p.unlink()
            freed += sz
            deleted_paths.append(str(p.relative_to(root.parent)).replace("\\", "/"))
        except OSError:
            continue
    if deleted_paths:
        m = _load_manifest(snapshots_root)
        for rel in deleted_paths:
            m["crops"].pop(rel, None)
        _save_manifest(m, snapshots_root)
    return len(deleted_paths), freed


# --- main extractor ------------------------------------------------------------


def refresh(model, embedder,
            snapshots_root: str | Path = SNAPSHOTS_ROOT,
            imgsz: int | None = DEFAULT_IMGSZ,
            cap_bytes: int | None = None) -> dict:
    """Bring the crops tree up to date with the anomaly frames tree.

    Idempotent: frames already listed in the manifest are skipped, so a
    routine refresh() only pays YOLO for genuinely new frames. Returns a
    small summary dict (frames processed, crops added, crops skipped as
    duplicates, evictions).
    """
    if cap_bytes is None:
        cap_bytes = CAP_BYTES
    import cv2  # heavy import gated to the function so callers can smoke-test
    root_frames = Path(snapshots_root) / ANOMALY_FRAMES_SUBDIR
    root_crops  = _dir(snapshots_root)
    root_crops.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(snapshots_root)
    processed = manifest.setdefault("processed_frames", {})
    crops_map = manifest.setdefault("crops", {})

    # Pre-load per-(cam,cls) vectors of the crops that already exist, so we
    # can dedup new crops against them cheaply.
    seen_by_cam_cls: dict[tuple[str, str], list[np.ndarray]] = {}
    for rel, meta in crops_map.items():
        p = Path(snapshots_root) / rel
        if not p.is_file():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        emb_cls = "person" if meta.get("cls") == "person" else "vehicle"
        vec = embedder.embed(img, emb_cls)
        if vec is None:
            continue
        seen_by_cam_cls.setdefault((meta.get("cam_id", "?"), meta.get("cls", "?")), []).append(vec)

    added = 0
    skipped_dup = 0
    frames_touched = 0

    if root_frames.is_dir():
        for frame_path in sorted(root_frames.rglob("*.jpg")):
            if frame_path.name.endswith("_full.jpg"):
                # already the full-frame path we look at; but if a variant
                # naming shows up too, still fine to process.
                pass
            rel_frame = str(frame_path.relative_to(snapshots_root)).replace("\\", "/")
            mtime = frame_path.stat().st_mtime
            if processed.get(rel_frame) == mtime:
                continue     # already extracted
            frames_touched += 1
            image = cv2.imread(str(frame_path))
            if image is None:
                processed[rel_frame] = mtime
                continue
            cam_id = _cam_id_from_frame(rel_frame)
            boxes = _extract_boxes(model, image, imgsz)
            for i, b in enumerate(boxes):
                crop = _crop_from_frame(image, b)
                if crop is None:
                    continue
                emb_cls = "person" if b["cls"] == "person" else "vehicle"
                vec = embedder.embed(crop, emb_cls)
                if vec is None:
                    continue
                if _is_duplicate(vec, cam_id, b["cls"], seen_by_cam_cls):
                    skipped_dup += 1
                    continue
                seen_by_cam_cls.setdefault((cam_id, b["cls"]), []).append(vec)

                # File path: anomalies_crops/<cam>/<frame_stem>__<idx>_<cls>.jpg
                out_dir = root_crops / cam_id
                out_dir.mkdir(parents=True, exist_ok=True)
                out_name = f"{frame_path.stem}__{i:02d}_{b['cls']}.jpg"
                out_path = out_dir / out_name
                if not cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                    continue
                rel_crop = str(out_path.relative_to(snapshots_root)).replace("\\", "/")
                H, W = image.shape[:2]
                crops_map[rel_crop] = {
                    "cam_id":       cam_id,
                    "cls":          b["cls"],
                    "conf":         round(float(b.get("conf") or 0.0), 3),
                    "source_frame": rel_frame,
                    "saved_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    # Normalized (0..1) foot-point on the source frame so
                    # auto_blacklist can bin it into a grid cell without
                    # re-opening the image.
                    "foot_norm":    [round((b["x1"] + b["x2"]) / 2.0 / W, 4),
                                     round(b["y2"] / H, 4)],
                }
                added += 1
            processed[rel_frame] = mtime

    _save_manifest(manifest, snapshots_root)

    # Enforce cap AFTER the pass so a burst of a hundred new crops all get a
    # chance to be considered before eviction runs.
    evicted, freed = enforce_size_cap(snapshots_root, cap_bytes)

    return {
        "frames_touched": frames_touched,
        "crops_added":    added,
        "crops_skipped_dup": skipped_dup,
        "crops_evicted":  evicted,
        "bytes_freed":    freed,
    }


# --- public utilities ---------------------------------------------------------


def usage_stats(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> dict:
    root = _dir(snapshots_root)
    if not root.is_dir():
        return {"count": 0, "bytes": 0, "cap_bytes": CAP_BYTES, "cap_mb": CAP_MB}
    files = list(root.rglob("*.jpg"))
    total = sum(p.stat().st_size for p in files if p.is_file())
    return {
        "count":     len(files),
        "bytes":     total,
        "cap_bytes": CAP_BYTES,
        "cap_mb":    CAP_MB,
        "path":      str(root),
    }


def clear_all(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> dict:
    """Delete every extracted crop AND reset the manifest. The source
    anomaly frames themselves are left alone - only the extracted crops go."""
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
    # Reset manifest so refresh() will re-process every frame next time.
    _save_manifest({"processed_frames": {}, "crops": {}}, snapshots_root)
    # Prune the now-empty per-cam sub-dirs
    for d in sorted(root.rglob("*"), reverse=True):
        if d.is_dir():
            try: d.rmdir()
            except OSError: pass
    return {"deleted": n, "bytes_freed": freed}
