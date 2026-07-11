"""Tag-time upload of verdicts + reviewed frames to the trainer's prefix."""
import json

import pytest
from conftest import FakeBucket as _FakeBucket

from app import training_sync


@pytest.fixture()
def tree(tmp_path):
    """Minimal operator layout: src/data/reviews.json + snapshots tree with
    one reviewed frame (jpg + metadata json) and one UNreviewed frame."""
    src = tmp_path / "src"
    snaps = src / "web" / "snapshots"
    frames = snaps / "review_frames" / "camA"
    frames.mkdir(parents=True)
    (frames / "1000.jpg").write_bytes(b"jpegbytes")
    (frames / "1000.json").write_text(json.dumps({"boxes": []}))
    (frames / "2000.jpg").write_bytes(b"unreviewed")
    data = src / "data"
    data.mkdir(parents=True)
    (data / "reviews.json").write_text(json.dumps({
        "frame_reviews": [{"frame_path": "review_frames/camA/1000.jpg",
                           "cam_id": "camA"}]}))
    return {"snaps": snaps, "reviews": data / "reviews.json",
            "state": data / ".training_push_state.json"}


def _push(tree):
    return training_sync.push_training_data(
        snapshots_root=tree["snaps"], reviews_path=tree["reviews"],
        state_path=tree["state"])


def test_push_uploads_reviews_and_reviewed_frames_only(tree, monkeypatch):
    bucket = _FakeBucket()
    monkeypatch.setattr(training_sync, "_BUCKET", bucket)
    stats = _push(tree)
    assert stats == {"uploaded": 3, "pending": 0}
    assert set(bucket.store) == {
        "training/reviews.json",
        "training/snapshots/review_frames/camA/1000.jpg",
        "training/snapshots/review_frames/camA/1000.json",
    }   # 2000.jpg has no verdict -> NOT training data, stays local
    assert bucket.store["training/snapshots/review_frames/camA/1000.jpg"] \
        == b"jpegbytes"

    # second pass: ledger says everything is current -> zero uploads
    assert _push(tree) == {"uploaded": 0, "pending": 0}

    # a new review verdict re-pushes reviews.json (mtime moved)
    import os
    os.utime(tree["reviews"], (1, 2_000_000_000))
    stats = _push(tree)
    assert stats["uploaded"] == 1


def test_push_without_credentials_is_a_quiet_noop(tree, monkeypatch):
    monkeypatch.setattr(training_sync, "_BUCKET", None)
    monkeypatch.setattr(training_sync, "find_service_account", lambda: None)
    assert _push(tree) == {"disabled": True}


def test_find_service_account_env_first(tmp_path, monkeypatch):
    key = tmp_path / "k.json"
    key.write_text("{}")
    monkeypatch.setenv("FIREBASE_CREDENTIALS", str(key))
    assert training_sync.find_service_account() == str(key)
    monkeypatch.setenv("FIREBASE_CREDENTIALS", str(tmp_path / "missing.json"))
    monkeypatch.setattr(training_sync, "_REPO_ROOT", tmp_path)
    assert training_sync.find_service_account() is None
    auto = tmp_path / "proj-firebase-adminsdk-x.json"
    auto.write_text("{}")
    assert training_sync.find_service_account() == str(auto)
