"""Pull the operator-uploaded training data down from cloud Storage.

    python -m tools.fetch_training_data --dest data/training_pull

First step of the automated trainer (the GitHub Actions job): the operator's
dashboard uploaded verdicts + reviewed frames to ``training/`` at tag time
(app/training_sync.py); this reconstructs the exact local layout
``tools/export_labels.py`` expects:

    <dest>/reviews.json
    <dest>/snapshots/review_frames/<cam>/<ts>.jpg + .json

It ALSO restores the trainer's own cumulative state into data/adapters/
(current pointer + history.jsonl), so a fresh CI runner appends to the real
promotion history instead of starting a parallel one, and the new pointer's
``previous`` field chains correctly for rollbacks.

Needs write-less Storage access via the Admin SDK: FIREBASE_CREDENTIALS
(service-account json) and the bucket from FIREBASE_STORAGE_BUCKET or
web/firebase-config.js. Exits non-zero with a plain sentence when there is
nothing to train on - a manually dispatched run should say WHY it stopped.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PREFIX = "training"


def _bucket():
    import firebase_admin
    from firebase_admin import credentials, storage
    from app.pool_sync import _bucket_name
    cred = os.environ.get("FIREBASE_CREDENTIALS")
    name = os.environ.get("FIREBASE_STORAGE_BUCKET") or _bucket_name()
    if not cred or not Path(cred).is_file():
        raise SystemExit("FIREBASE_CREDENTIALS must point at the "
                         "service-account json")
    if not name:
        raise SystemExit("no storage bucket (set FIREBASE_STORAGE_BUCKET or "
                         "keep web/firebase-config.js)")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(cred),
                                      {"storageBucket": name})
    return storage.bucket(name)


def main() -> None:
    from app import adapters

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=str(_SRC_ROOT / "data" / "training_pull"))
    ap.add_argument("--adapters-dir", default=str(adapters.ADAPTERS_DIR))
    args = ap.parse_args()
    dest = Path(args.dest)
    bucket = _bucket()

    # 1. Verdicts - without them there is nothing to do.
    rb = bucket.blob(f"{TRAIN_PREFIX}/reviews.json")
    if not rb.exists():
        raise SystemExit(
            "no training/reviews.json in Storage yet - tag a few frames in "
            "the dashboard first (uploads happen automatically on submit)")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "reviews.json").write_bytes(rb.download_as_bytes())

    # 2. Reviewed frames + their metadata jsons.
    frames = 0
    prefix = f"{TRAIN_PREFIX}/snapshots/"
    for blob in bucket.list_blobs(prefix=prefix):
        rel = blob.name[len(prefix):]
        parts = rel.split("/")
        if not rel or ".." in parts or rel.startswith("/") or "\\" in rel:
            continue
        p = dest / "snapshots" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blob.download_as_bytes())
        frames += rel.endswith(".jpg")

    if frames == 0:
        raise SystemExit(
            "reviews.json exists but no reviewed frame images are in "
            "Storage yet - submit at least one FRAME review (the crop-level "
            "reviews carry no coordinates and cannot train the detector)")

    # 3. Cumulative trainer state (may not exist on the very first run).
    adir = Path(args.adapters_dir)
    restored = []
    for src_name, local in ((adapters.STORAGE_POINTER, adapters.POINTER_NAME),
                            (f"{TRAIN_PREFIX}/{adapters.HISTORY_NAME}",
                             adapters.HISTORY_NAME)):
        b = bucket.blob(src_name)
        if b.exists():
            adir.mkdir(parents=True, exist_ok=True)
            (adir / local).write_bytes(b.download_as_bytes())
            restored.append(local)

    print(f"fetched: reviews.json + {frames} reviewed frame(s) -> {dest}")
    print(f"restored trainer state: {', '.join(restored) or 'none (first run)'}")


if __name__ == "__main__":
    main()
