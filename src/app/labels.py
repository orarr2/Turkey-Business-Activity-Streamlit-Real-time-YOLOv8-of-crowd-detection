"""Human-in-the-loop review of stored detections.

YOLOv8 is inference-only at runtime and does not learn from the stream, so
the only way to actually improve what the system shows the user is to let
them tell us when a label is wrong and to remember what they said. This
module is the persistence layer for that feedback loop:

  1. sample_crop()   - pick a saved crop the user has not reviewed yet
  2. submit_review() - persist the user's verdict (correct / wrong-label /
                       not-an-object) with an optional corrected class
  3. summary()       - count how many crops the user has reviewed so far,
                       broken down by verdict, so the UI can show progress

The store is a plain JSON file under ``data/reviews.json`` - append-only in
practice, keyed by ``crop_path``. That keeps the store trivially inspectable
and avoids a new DB dependency for a feature that produces at most a few
hundred rows per week of active use.

Downstream uses of the collected reviews:
  * flag known-bad crop paths so the collector's static-blacklist helper
    (see visual_search / cameras.py ``roi_exclude_class``) can be updated
    manually or via a small offline script;
  * later, once enough labels accumulate, export them as a COCO-format
    dataset for a real fine-tuning pass.
"""
from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.visual_search import CROP_SUBDIRS, SNAPSHOTS_ROOT

_SRC_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REVIEWS_PATH = _SRC_ROOT / "data" / "reviews.json"

# Verdict values the UI is allowed to POST. Anything else is rejected.
VERDICTS = ("correct", "wrong_label", "not_an_object")


@dataclass
class Review:
    crop_path: str             # relative to SNAPSHOTS_ROOT, forward-slash form
    verdict: str               # one of VERDICTS (label opinion)
    original_cls: str          # what the detector said
    corrected_cls: str | None  # what the user says it actually is (wrong_label)
    anomaly_verdict: str | None  # "yes" / "no" - was this really an anomaly?
    note: str | None
    reviewed_at: str           # ISO-8601 UTC

    def to_public(self) -> dict:
        d = {"crop_path": self.crop_path, "verdict": self.verdict,
             "original_cls": self.original_cls, "reviewed_at": self.reviewed_at}
        if self.corrected_cls:
            d["corrected_cls"] = self.corrected_cls
        if self.anomaly_verdict:
            d["anomaly_verdict"] = self.anomaly_verdict
        if self.note:
            d["note"] = self.note
        return d


class ReviewStore:
    """Thread-safe on-disk store keyed by crop_path (relative to SNAPSHOTS_ROOT).

    The JSON file is loaded once on construction and rewritten wholesale on
    each submit. Fine for the expected write volume (interactive UI); if
    labeling ever scales up, swap to sqlite without touching callers.
    """

    def __init__(self, path: str | Path = DEFAULT_REVIEWS_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._by_path: dict[str, Review] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        for row in data.get("reviews", []):
            try:
                r = Review(
                    crop_path=str(row["crop_path"]),
                    verdict=str(row["verdict"]),
                    original_cls=str(row.get("original_cls", "?")),
                    corrected_cls=row.get("corrected_cls") or None,
                    anomaly_verdict=row.get("anomaly_verdict") or None,
                    note=row.get("note") or None,
                    reviewed_at=str(row.get("reviewed_at", "")))
                self._by_path[r.crop_path] = r
            except (KeyError, TypeError):
                continue

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reviews": [r.to_public() for r in self._by_path.values()],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)

    def is_reviewed(self, crop_path: str) -> bool:
        return crop_path in self._by_path

    def submit(self, crop_path: str, verdict: str, *,
               original_cls: str = "?",
               corrected_cls: str | None = None,
               anomaly_verdict: str | None = None,
               note: str | None = None) -> Review:
        if verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {verdict!r}; expected one of "
                             f"{VERDICTS}")
        if anomaly_verdict is not None and anomaly_verdict not in ("yes", "no"):
            raise ValueError(f"invalid anomaly_verdict {anomaly_verdict!r}; "
                             f"expected 'yes' | 'no' | None")
        r = Review(
            crop_path=str(crop_path), verdict=verdict,
            original_cls=str(original_cls),
            corrected_cls=(str(corrected_cls) if corrected_cls else None),
            anomaly_verdict=(str(anomaly_verdict) if anomaly_verdict else None),
            note=(str(note) if note else None),
            reviewed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with self._lock:
            self._by_path[r.crop_path] = r
            self._save_locked()
        return r

    def summary(self) -> dict:
        counts = {v: 0 for v in VERDICTS}
        for r in self._by_path.values():
            counts[r.verdict] = counts.get(r.verdict, 0) + 1
        return {"total_reviewed": len(self._by_path), "by_verdict": counts}

    def rejects_for_cls(self, cls: str) -> list[str]:
        """Crop paths the user rejected as `wrong_label` or `not_an_object`
        for a given class. Useful when auditing where false positives cluster."""
        out = []
        for r in self._by_path.values():
            if r.original_cls != cls:
                continue
            if r.verdict in ("wrong_label", "not_an_object"):
                out.append(r.crop_path)
        return out


def sample_crop(store: ReviewStore,
                snapshots_root: str | Path = SNAPSHOTS_ROOT,
                seed: int | None = None) -> dict | None:
    """Pick one crop the user has not reviewed yet.

    Returns {"path", "url", "cls", "from_anomaly", "remaining"} or None
    when every stored crop has been reviewed.

    Sampling favors crops the user has NOT already reviewed AND that came
    from anomaly frames (they are the ones most likely to teach the system
    something new). Anomaly candidates are picked with 70% probability
    when they exist; otherwise fall back to routine returning/events crops.
    """
    from app.visual_search import SnapshotIndex, _is_from_anomaly

    root = Path(snapshots_root)
    idx = SnapshotIndex(root)
    manifest_cls = idx._manifest_cls()  # noqa: SLF001 - deliberate reuse

    anomaly_pool: list[tuple[str, str]] = []
    routine_pool: list[tuple[str, str]] = []
    for sub in CROP_SUBDIRS:
        base = root / sub
        if not base.is_dir():
            continue
        for p in base.rglob("*.jpg"):
            if p.name.endswith("_full.jpg"):
                continue
            rel = str(p.relative_to(root)).replace("\\", "/")
            if store.is_reviewed(rel):
                continue
            cls = manifest_cls.get(rel)
            if not cls:
                import cv2 as _cv2
                img = _cv2.imread(str(p))
                cls = idx._guess_cls(p, img.shape) if img is not None else "?"  # noqa: SLF001
            (anomaly_pool if _is_from_anomaly(rel) else routine_pool).append((rel, cls))

    total = len(anomaly_pool) + len(routine_pool)
    if total == 0:
        return None
    rng = random.Random(seed) if seed is not None else random
    prefer_anomaly = anomaly_pool and rng.random() < 0.7
    pool = anomaly_pool if prefer_anomaly else (routine_pool or anomaly_pool)
    rel, cls = rng.choice(pool)
    return {
        "path": rel,
        "url": f"/snapshots/{rel}",
        "cls": cls,
        "from_anomaly": _is_from_anomaly(rel),
        "remaining": total,
    }
