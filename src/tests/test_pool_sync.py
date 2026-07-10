"""pool_sync: VM-side reconcile (mock bucket) + local-side pull (mock HTTP).

No network, no Firebase SDK: the VM side gets a fake bucket that records
calls; the local side gets a monkeypatched ``_http_get`` serving a canned
manifest + file bytes.
"""
import json
import time
from pathlib import Path

import pytest

from app import pool_sync


# ---- fakes ---------------------------------------------------------------------

class FakeBlob:
    def __init__(self, name, store):
        self.name = name
        self._store = store
        self.public_url = f"https://storage.example/{name}"

    def upload_from_string(self, data, content_type=None):
        self._store[self.name] = data if isinstance(data, bytes) else data.encode()

    def make_public(self):
        pass

    def delete(self):
        self._store.pop(self.name, None)


class FakeBucket:
    def __init__(self):
        self.blobs: dict[str, bytes] = {}

    def blob(self, name):
        return FakeBlob(name, self.blobs)


class FakeFirebase:
    def __init__(self):
        self.storage = FakeBucket()


def _seed_pool(root: Path):
    (root / "review_frames" / "camA").mkdir(parents=True)
    (root / "review_frames" / "camA" / "1000.jpg").write_bytes(b"jpegA")
    (root / "review_frames" / "camA" / "1000.json").write_text(
        json.dumps({"cam_id": "camA", "boxes": []}))
    (root / "live_samples" / "camA").mkdir(parents=True)
    (root / "live_samples" / "camA" / "2000_car_55.jpg").write_bytes(b"jpegB")
    # dotfiles + non-pool dirs must be ignored
    (root / "live_samples" / ".burst_counts.txt").write_text("camA 3")
    (root / "anomalies").mkdir()
    (root / "anomalies" / "x.jpg").write_bytes(b"nope")


# ---- VM side -------------------------------------------------------------------

def test_sync_up_uploads_pools_and_manifest(tmp_path):
    _seed_pool(tmp_path)
    fb = FakeFirebase()
    stats = pool_sync.sync_up(fb, tmp_path)
    assert stats == {"uploaded": 3, "deleted": 0, "pending": 0}
    names = set(fb.storage.blobs)
    assert f"{pool_sync.PREFIX}/review_frames/camA/1000.jpg" in names
    assert f"{pool_sync.PREFIX}/review_frames/camA/1000.json" in names
    assert f"{pool_sync.PREFIX}/live_samples/camA/2000_car_55.jpg" in names
    assert f"{pool_sync.PREFIX}/{pool_sync.MANIFEST_NAME}" in names
    # anomalies/ and dotfiles are NOT synced
    assert not any("anomalies" in n or ".burst_counts" in n for n in names)
    manifest = json.loads(
        fb.storage.blobs[f"{pool_sync.PREFIX}/{pool_sync.MANIFEST_NAME}"])
    assert len(manifest["files"]) == 3


def test_sync_up_second_call_is_noop(tmp_path):
    _seed_pool(tmp_path)
    fb = FakeFirebase()
    pool_sync.sync_up(fb, tmp_path)
    stats = pool_sync.sync_up(fb, tmp_path)
    assert stats == {"uploaded": 0, "deleted": 0, "pending": 0}


def test_sync_up_mirrors_lru_eviction(tmp_path):
    _seed_pool(tmp_path)
    fb = FakeFirebase()
    pool_sync.sync_up(fb, tmp_path)
    (tmp_path / "live_samples" / "camA" / "2000_car_55.jpg").unlink()
    stats = pool_sync.sync_up(fb, tmp_path)
    assert stats["deleted"] == 1
    assert (f"{pool_sync.PREFIX}/live_samples/camA/2000_car_55.jpg"
            not in fb.storage.blobs)
    manifest = json.loads(
        fb.storage.blobs[f"{pool_sync.PREFIX}/{pool_sync.MANIFEST_NAME}"])
    assert "live_samples/camA/2000_car_55.jpg" not in manifest["files"]


def test_sync_up_without_storage_is_none(tmp_path):
    class NoStorage:
        storage = None
    assert pool_sync.sync_up(NoStorage(), tmp_path) is None
    assert pool_sync.sync_up(None, tmp_path) is None


def test_sync_up_batches_large_backlog(tmp_path, monkeypatch):
    """First sync against a big accumulated pool must NOT upload everything
    in one pass (that burst oom-killed the 1 GB VM); it drains over rounds
    and the manifest only ever lists what is actually uploaded."""
    monkeypatch.setattr(pool_sync, "MAX_UPLOADS_PER_PASS", 5)
    d = tmp_path / "live_samples" / "camA"
    d.mkdir(parents=True)
    for i in range(12):
        (d / f"{i:04d}_car_50.jpg").write_bytes(b"x" * 10)
    fb = FakeFirebase()

    s1 = pool_sync.sync_up(fb, tmp_path)
    assert s1["uploaded"] == 5 and s1["pending"] == 7
    manifest = json.loads(
        fb.storage.blobs[f"{pool_sync.PREFIX}/{pool_sync.MANIFEST_NAME}"])
    assert len(manifest["files"]) == 5      # only uploaded files listed

    s2 = pool_sync.sync_up(fb, tmp_path)
    assert s2["uploaded"] == 5 and s2["pending"] == 2
    s3 = pool_sync.sync_up(fb, tmp_path)
    assert s3["uploaded"] == 2 and s3["pending"] == 0
    manifest = json.loads(
        fb.storage.blobs[f"{pool_sync.PREFIX}/{pool_sync.MANIFEST_NAME}"])
    assert len(manifest["files"]) == 12     # backlog fully drained

    s4 = pool_sync.sync_up(fb, tmp_path)
    assert s4 == {"uploaded": 0, "deleted": 0, "pending": 0}


# ---- local side ----------------------------------------------------------------

def _fake_http(manifest: dict, files: dict[str, bytes]):
    """Return an _http_get replacement serving the manifest + file bodies."""
    def get(url, timeout=30):
        if url.endswith("/" + pool_sync.MANIFEST_NAME):
            return json.dumps(manifest).encode()
        for rel, body in files.items():
            if url.endswith(rel):
                return body
        raise OSError(f"404 {url}")
    return get


def test_pull_once_downloads_and_evicts(tmp_path, monkeypatch):
    manifest = {
        "generated_at": "2026-07-10T12:00:00Z",
        "files": {
            "review_frames/camA/1000.jpg":
                {"mtime": 1.0, "url": "https://s.example/review_frames/camA/1000.jpg"},
            "review_frames/camA/1000.json":
                {"mtime": 1.0, "url": "https://s.example/review_frames/camA/1000.json"},
        },
    }
    bodies = {
        "review_frames/camA/1000.jpg": b"framejpeg",
        "review_frames/camA/1000.json": b"{\"boxes\": []}",
    }
    monkeypatch.setattr(pool_sync, "_http_get", _fake_http(manifest, bodies))

    # a purely-local file that must survive manifest eviction passes
    local_only = tmp_path / "review_frames" / "camB" / "local.jpg"
    local_only.parent.mkdir(parents=True)
    local_only.write_bytes(b"local")

    stats = pool_sync.pull_once(tmp_path, bucket="b")
    assert stats["downloaded"] == 2 and stats["errors"] == 0
    assert (tmp_path / "review_frames" / "camA" / "1000.jpg").read_bytes() == b"framejpeg"

    # second pass: no changes -> no downloads
    stats2 = pool_sync.pull_once(tmp_path, bucket="b")
    assert stats2["downloaded"] == 0 and stats2["removed"] == 0

    # VM evicts the frame -> manifest shrinks -> local copy (pulled by us)
    # is removed, while the local-only file is untouched
    manifest["files"] = {}
    stats3 = pool_sync.pull_once(tmp_path, bucket="b")
    assert stats3["removed"] == 2
    assert not (tmp_path / "review_frames" / "camA" / "1000.jpg").exists()
    assert local_only.exists()


def test_pull_once_rejects_path_traversal(tmp_path, monkeypatch):
    manifest = {"files": {
        "../evil.jpg": {"mtime": 1.0, "url": "https://s.example/../evil.jpg"},
        "not_a_pool/x.jpg": {"mtime": 1.0, "url": "https://s.example/not_a_pool/x.jpg"},
    }}
    monkeypatch.setattr(pool_sync, "_http_get",
                        _fake_http(manifest, {"../evil.jpg": b"x",
                                              "not_a_pool/x.jpg": b"x"}))
    stats = pool_sync.pull_once(tmp_path, bucket="b")
    assert stats["downloaded"] == 0
    assert not (tmp_path.parent / "evil.jpg").exists()
    assert not (tmp_path / "not_a_pool").exists()


def test_pull_once_reports_manifest_failure(tmp_path, monkeypatch):
    def boom(url, timeout=30):
        raise OSError("offline")
    monkeypatch.setattr(pool_sync, "_http_get", boom)
    stats = pool_sync.pull_once(tmp_path, bucket="b")
    assert "manifest fetch failed" in stats["error"]


def test_bucket_name_from_config_js(tmp_path, monkeypatch):
    monkeypatch.delenv("FIREBASE_STORAGE_BUCKET", raising=False)
    web = tmp_path / "web"
    web.mkdir()
    (web / "firebase-config.js").write_text(
        'export const firebaseConfig = { storageBucket: "proj.appspot.com" };')
    assert pool_sync._bucket_name(web) == "proj.appspot.com"
