"""BADGE crop sampler (plan WS2, OSNet edition).

Serve the reviewer the crops the model is most unsure about, WITH visual
diversity, instead of ``random.choice``. BADGE's core property - pick by
gradient direction x magnitude - is preserved cheaply: the "direction" is
the crop's OSNet identity embedding (SnapshotIndex cache, so nothing is
re-embedded), the "magnitude" is the capture-time uncertainty WS1 encodes
into the ``_uNN`` filename suffix; only the k-means++ INITIALIZATION step
runs (hand-rolled numpy, ~30 lines - sklearn stays off the VM).

Feature flag: ``REVIEW_SAMPLER=badge|naive`` (default naive), per-request
override via ``/api/review-sample?strategy=``. Crops that predate WS1 carry
no uncertainty and fall back to a neutral weight - a soft failure, never a
wrong signal (plan risk table).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from app.visual_search import CROP_SUBDIRS, SNAPSHOTS_ROOT, SnapshotIndex

# Crops saved before WS1 have no _uNN suffix; treat them as mid-uncertainty
# rather than skipping them (the pool must stay reviewable end-to-end).
NEUTRAL_UNCERTAINTY = 0.5
_U_SUFFIX = re.compile(r"_u(\d{2,3})\.jpg$")


def crop_uncertainty(rel_path: str) -> float | None:
    """Parse the WS1 ``_uNN`` suffix out of a crop filename (None = absent)."""
    m = _U_SUFFIX.search(rel_path)
    if not m:
        return None
    return min(1.0, int(m.group(1)) / 100.0)


def kmeanspp_pick(vectors: np.ndarray, weights: np.ndarray, k: int,
                  seed: int | None = None) -> list[int]:
    """k-means++ INIT only: return k indices, spread apart in embedding
    space and biased toward high weight.

    First pick: probability proportional to weight (uniform when all
    weights are 0 - degenerates to plain spread-only seeding). Every next
    pick: proportional to D^2 to the nearest already-picked vector, scaled
    by weight. Deterministic under a fixed seed.
    """
    n = len(vectors)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    rng = np.random.default_rng(seed)
    w = np.asarray(weights, dtype=np.float64).clip(min=0.0)
    if w.sum() <= 0:
        w = np.ones(n)
    picks = [int(rng.choice(n, p=w / w.sum()))]
    if k == 1:
        return picks
    v = np.asarray(vectors, dtype=np.float64)
    d2 = ((v - v[picks[0]]) ** 2).sum(axis=1)
    for _ in range(k - 1):
        score = d2 * w
        score[picks] = 0.0
        total = score.sum()
        if total <= 0:  # every remaining vector is identical to a pick
            remaining = [i for i in range(n) if i not in picks]
            picks.append(int(rng.choice(remaining)))
        else:
            picks.append(int(rng.choice(n, p=score / total)))
        d2 = np.minimum(d2, ((v - v[picks[-1]]) ** 2).sum(axis=1))
    return picks


def sample_crop_badge(store, snapshots_root: str | Path = SNAPSHOTS_ROOT,
                      batch: int = 30, seed: int | None = None,
                      index: SnapshotIndex | None = None) -> dict | None:
    """BADGE-ranked batch of un-reviewed crops.

    Returns ``{"batch": [crop, ...], "sampler": "badge"}`` where each crop
    has the same shape the naive ``sample_crop`` serves (path/url/cls/
    from_anomaly/remaining) plus ``uncertainty``. None when the pool is
    empty. ``index`` lets the dashboard reuse its live SnapshotIndex.
    """
    from app.visual_search import _is_from_anomaly

    root = Path(snapshots_root)
    idx = index if index is not None else SnapshotIndex(root)
    idx.refresh()

    rels, vecs, weights = [], [], []
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
            entry = idx._entries.get(rel)  # noqa: SLF001 - same app family
            if entry is None:
                continue
            rels.append(rel)
            vecs.append(entry["vec"])
            u = crop_uncertainty(rel)
            weights.append(NEUTRAL_UNCERTAINTY if u is None else u)
    if not rels:
        return None

    picks = kmeanspp_pick(np.stack(vecs), np.asarray(weights),
                          k=min(batch, len(rels)), seed=seed)
    batch_out = []
    for i in picks:
        rel = rels[i]
        entry = idx._entries[rel]  # noqa: SLF001
        batch_out.append({
            "path": rel,
            "url": f"/snapshots/{rel}",
            "cls": entry["cls"],
            "from_anomaly": _is_from_anomaly(rel),
            "uncertainty": round(float(weights[i]), 4),
            "remaining": len(rels),
        })
    return {"batch": batch_out, "sampler": "badge"}
