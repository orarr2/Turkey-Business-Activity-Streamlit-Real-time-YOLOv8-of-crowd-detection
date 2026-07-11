"""Push the operator's training data to cloud Storage - at tag time.

The verdicts (data/reviews.json) and the reviewed frames live on the
operator's machine; the automated trainer (GitHub Actions) runs at night in
the cloud with the operator's PC off. This module bridges the gap at the
only moment the PC is guaranteed on: the instant a review is submitted.
The dashboard calls ``push_async()`` after every submit; a background
thread uploads

  training/reviews.json                  (mutable -> no-store, re-pushed on change)
  training/snapshots/review_frames/...   (reviewed jpg + metadata json, once each)

mirroring exactly the layout ``tools/export_labels.py`` reads, so the
trainer's fetch step reconstructs it 1:1.

Credentials: FIREBASE_CREDENTIALS, or the gitignored Admin-SDK key that
already sits at the repo root on the operator's machine (auto-detected).
No key = the module stays silent after one printed line and the dashboard
works exactly as before - uploading is an enhancement, never a dependency.

Quota: a review session touches tens of files of ~100 KB - noise against
the free tier (uploads are capped per pass like pool_sync, the backlog
drains across submits).
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from app.pool_sync import (_bucket_name, _read_state, _write_state,
                           _reviewed_frame_rels)

_SRC_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SRC_ROOT.parent

TRAIN_PREFIX = "training"
STATE_NAME = ".training_push_state.json"
MAX_UPLOADS_PER_PASS = 40

_LOCK = threading.Lock()
_RUNNING = False
_WARNED = False


def find_service_account() -> str | None:
    """FIREBASE_CREDENTIALS first; else the Admin-SDK json at the repo root
    (gitignored, present on the operator/admin machine only)."""
    env = os.environ.get("FIREBASE_CREDENTIALS")
    if env and Path(env).is_file():
        return env
    for pattern in ("*firebase-adminsdk*.json", "firebase-service-account.json"):
        for p in sorted(_REPO_ROOT.glob(pattern)):
            if p.is_file():
                return str(p)
    return None


_BUCKET = None


def _bucket():
    """Lazy Admin-SDK Storage bucket; None (with one printed line) when the
    machine has no key or no bucket - the viewer-only case."""
    global _BUCKET, _WARNED
    if _BUCKET is not None:
        return _BUCKET
    cred = find_service_account()
    name = os.environ.get("FIREBASE_STORAGE_BUCKET") or _bucket_name()
    if not cred or not name:
        if not _WARNED:
            _WARNED = True
            missing = "service-account key" if not cred else "storage bucket"
            print(f"training_sync: no {missing} - verdicts stay local "
                  f"(trainer uploads disabled)")
        return None
    try:
        import firebase_admin
        from firebase_admin import credentials, storage
        if not firebase_admin._apps:
            firebase_admin.initialize_app(credentials.Certificate(cred),
                                          {"storageBucket": name})
        _BUCKET = storage.bucket(name)
        return _BUCKET
    except Exception as e:
        if not _WARNED:
            _WARNED = True
            print(f"training_sync: storage init failed "
                  f"({type(e).__name__}: {e}) - uploads disabled")
        return None


def push_training_data(snapshots_root: str | Path | None = None,
                       reviews_path: str | Path | None = None,
                       max_uploads: int = MAX_UPLOADS_PER_PASS,
                       state_path: str | Path | None = None) -> dict:
    """One reconcile pass: reviews.json when changed + any reviewed frame
    (jpg/json) not uploaded yet. Ledger-diffed, budget-capped, never raises."""
    bucket = _bucket()
    if bucket is None:
        return {"disabled": True}
    root = Path(snapshots_root) if snapshots_root else _SRC_ROOT / "web" / "snapshots"
    reviews = Path(reviews_path) if reviews_path else _SRC_ROOT / "data" / "reviews.json"
    sp = Path(state_path) if state_path else reviews.parent / STATE_NAME
    state = _read_state(sp)
    uploaded = pending = 0
    changed = False
    try:
        if reviews.is_file():
            mtime = reviews.stat().st_mtime
            if float(state.get("_reviews", {}).get("mtime", -1)) != mtime:
                blob = bucket.blob(f"{TRAIN_PREFIX}/reviews.json")
                blob.cache_control = "no-store"     # mutable name
                blob.upload_from_string(reviews.read_bytes(),
                                        content_type="application/json")
                blob.make_public()
                state["_reviews"] = {"mtime": mtime}
                uploaded += 1
                changed = True

        for rel in sorted(_reviewed_frame_rels(root)):
            p = root / rel
            if not p.is_file():
                continue        # reviewed long ago, image already cloud-only
            mtime = p.stat().st_mtime
            if float(state.get(rel, {}).get("mtime", -1)) == mtime:
                continue
            if uploaded >= max_uploads:
                pending += 1
                continue
            blob = bucket.blob(f"{TRAIN_PREFIX}/snapshots/{rel}")
            blob.upload_from_string(
                p.read_bytes(),
                content_type="image/jpeg" if rel.endswith(".jpg")
                else "application/json")
            blob.make_public()
            state[rel] = {"mtime": mtime}
            uploaded += 1
            changed = True
        if changed:
            _write_state(sp, state)
    except Exception as e:
        print(f"training_sync: push failed ({type(e).__name__}: {e})")
        return {"uploaded": uploaded, "pending": pending, "error": str(e)}
    if uploaded:
        print(f"training_sync: uploaded {uploaded} object(s)"
              + (f", {pending} queued for next submit" if pending else ""))
    return {"uploaded": uploaded, "pending": pending}


def push_async() -> bool:
    """Fire-and-forget push from a request handler. One pusher at a time;
    a submit that lands mid-push is covered by the NEXT submit's pass."""
    global _RUNNING
    with _LOCK:
        if _RUNNING:
            return False
        _RUNNING = True

    def _run() -> None:
        global _RUNNING
        try:
            push_training_data()
        finally:
            with _LOCK:
                _RUNNING = False

    threading.Thread(target=_run, name="training-sync-push",
                     daemon=True).start()
    return True
