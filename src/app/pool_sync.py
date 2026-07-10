"""Mirror the collector's review pools to Firebase Storage - and pull them
back down on the operator's machine.

Why this exists: the collector runs 24/7 on the GCP VM and writes
``review_frames/`` + ``live_samples/`` (and ``data/reid.db``) to the VM's
own disk. The operator's dashboard runs on their local machine and serves
search/review from the LOCAL snapshots tree - which, without this module,
only ever contains the bootstrap fixtures. The result the operator saw:
search always returned the same six demo crops, the review pool exhausted
after four frames and never refilled, and nothing the cameras actually
captured was reachable.

Two halves, one file:

* VM side  - ``sync_up(firebase, ...)``: reconcile Storage under the
  ``review_sync/`` prefix with the local pool tree. Upload new/changed
  files, delete remote objects whose local counterpart was LRU-evicted,
  and publish a ``manifest.json`` describing the current pool. Cheap when
  nothing changed (state diff against a local ledger, zero network calls).
* Local side - ``pull_once(...)`` / ``start_pull_thread(...)``: fetch the
  public manifest over plain HTTPS (no credentials needed - the objects
  are public, same model as the anomaly snapshots the dashboard already
  renders), download new/changed files into the same snapshots layout,
  and remove local copies of files that left the manifest. Only files this
  module pulled are ever deleted locally - bootstrap fixtures and anything
  a locally-run collector saved are left alone.

Free-tier envelope: the pools are LRU-capped (100 frames + 200 crops
~= 35 MB), uploads happen only on change (~5-8K ops/day, quota 20K/day),
and the manifest poll is ~10 KB every ``PULL_INTERVAL_S``. The re-ID
registry (a few MB of SQLite) is throttled to one push per
``REID_PUSH_EVERY_S`` so it cannot eat the transfer quota.

The ``review_sync/`` prefix is deliberately OUTSIDE the ``snapshots/``
prefix: the bucket's 24h lifecycle rule targets ``snapshots/`` and must
not garbage-collect the review pool, whose LRU horizon spans days.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from pathlib import Path

PREFIX = "review_sync"
POOL_SUBDIRS = ("review_frames", "live_samples")
MANIFEST_NAME = "manifest.json"

# VM-side ledger of what we already uploaded (rel -> mtime). Lives next to
# the pools so a collector restart resumes instead of re-uploading 300 files.
PUSH_STATE_NAME = ".sync_push_state.json"
# Local-side ledger of what we pulled (rel -> mtime). Distinguishes "file
# this module downloaded" (safe to delete when it leaves the manifest) from
# "file that was always local" (bootstrap fixtures - never touched).
PULL_STATE_NAME = ".sync_pull_state.json"

PULL_INTERVAL_S = int(os.environ.get("POOL_SYNC_PULL_INTERVAL_S") or 120)
REID_PUSH_EVERY_S = int(os.environ.get("REID_PUSH_EVERY_S") or 1800)
# Hard per-file ceiling: nothing in these pools is legitimately bigger.
MAX_FILE_BYTES = 20 * 1024 * 1024
# Upload budget per sync pass. The FIRST sync on a long-running VM faces the
# whole accumulated pool (~300 files); uploading them in one post-round
# burst held the collector in a minutes-long network window whose SSL/HTTP
# buffers pushed a 1 GB host into the KERNEL oom-killer (observed live:
# round done 20:28:42, oom-kill 20:31:07, peak 696M - under the cgroup cap,
# over the machine). A bounded batch keeps each pass to a few seconds; the
# backlog drains over the next rounds and the manifest grows with it.
MAX_UPLOADS_PER_PASS = int(os.environ.get("POOL_SYNC_MAX_UPLOADS") or 40)

_CONTENT_TYPES = {".jpg": "image/jpeg", ".json": "application/json",
                  ".db": "application/octet-stream"}


def _read_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def _pool_files(snapshots_root: Path) -> dict[str, float]:
    """rel path (forward slashes, pool-prefixed) -> mtime, for every jpg and
    metadata json in the synced pools. Ledger/marker dotfiles are skipped."""
    out: dict[str, float] = {}
    for sub in POOL_SUBDIRS:
        base = snapshots_root / sub
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix not in (".jpg", ".json"):
                continue
            if p.name.startswith("."):
                continue
            rel = str(p.relative_to(snapshots_root)).replace("\\", "/")
            try:
                out[rel] = p.stat().st_mtime
            except OSError:
                continue
    return out


# ---- VM side ------------------------------------------------------------------

def _compact_sqlite_copy(db_path: Path) -> bytes | None:
    """Point-in-time compacted snapshot of a live sqlite db (VACUUM INTO).
    Returns the bytes, or None when the copy could not be produced."""
    import sqlite3
    import tempfile
    tmp = Path(tempfile.gettempdir()) / f".{db_path.stem}_sync_copy.db"
    try:
        tmp.unlink()
    except OSError:
        pass
    try:
        con = sqlite3.connect(str(db_path))
        try:
            con.execute("VACUUM INTO ?", (str(tmp),))
        finally:
            con.close()
        data = tmp.read_bytes()
        return data
    except Exception as e:
        print(f"pool_sync: reid.db compact failed ({type(e).__name__}: {e})")
        return None
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def sync_up(firebase, snapshots_root: str | Path,
            reid_db_path: str | Path | None = None,
            state_path: str | Path | None = None) -> dict | None:
    """Reconcile Storage's ``review_sync/`` prefix with the local pools.

    Returns a stats dict, or None when Storage isn't configured (local-mode
    collector - nothing to do). Never raises: a sync failure must not cost
    a collector round; the next round retries naturally.
    """
    if firebase is None or getattr(firebase, "storage", None) is None:
        return None
    root = Path(snapshots_root)
    sp = Path(state_path) if state_path else root / PUSH_STATE_NAME
    state = _read_state(sp)
    current = _pool_files(root)

    uploaded = deleted = pending = 0
    manifest_files: dict[str, dict] = {}
    changed = False
    bucket = firebase.storage
    try:
        for rel, mtime in sorted(current.items()):
            entry = state.get(rel)
            if entry is None or float(entry.get("mtime", -1)) != mtime:
                if uploaded >= MAX_UPLOADS_PER_PASS:
                    # Budget spent - the backlog drains next round. A file
                    # with an older uploaded version keeps serving that
                    # version via the manifest until its turn comes.
                    pending += 1
                    if entry is None:
                        continue
                else:
                    p = root / rel
                    try:
                        data = p.read_bytes()
                    except OSError:
                        continue
                    if len(data) > MAX_FILE_BYTES:
                        continue
                    blob = bucket.blob(f"{PREFIX}/{rel}")
                    blob.upload_from_string(
                        data, content_type=_CONTENT_TYPES.get(
                            p.suffix, "application/octet-stream"))
                    blob.make_public()
                    entry = {"mtime": mtime, "url": blob.public_url,
                             "size": len(data)}
                    state[rel] = entry
                    uploaded += 1
                    changed = True
                    # Persist progress mid-pass: on a memory-starved host the
                    # upload window is exactly when the process is most likely
                    # to be killed, and losing the ledger meant re-uploading
                    # the same files every life while the manifest stayed
                    # frozen at the last COMPLETED pass.
                    if uploaded % 10 == 0:
                        _write_state(sp, state)
            manifest_files[rel] = {"mtime": entry["mtime"],
                                   "url": entry["url"],
                                   "size": entry.get("size", 0)}

        # LRU evicted locally -> drop the remote copy so the manifest (and
        # therefore every operator machine) forgets it too.
        for rel in [r for r in state if r not in current and r != "_reid_db"]:
            try:
                bucket.blob(f"{PREFIX}/{rel}").delete()
            except Exception:
                pass   # already gone / transient - manifest omission is what matters
            state.pop(rel, None)
            deleted += 1
            changed = True

        # Re-ID registry: throttled, it changes every round but a snapshot
        # every REID_PUSH_EVERY_S is plenty for the local search/registry view.
        # Pushed as a VACUUM INTO compact copy: the live db accumulates free
        # pages between prunes and can exceed the per-file cap (which silently
        # blocked the push forever); the compacted snapshot is both smaller on
        # the wire and a consistent point-in-time read of a live WAL database.
        reid_entry = state.get("_reid_db") or {}
        if reid_db_path:
            rp = Path(reid_db_path)
            if rp.is_file():
                now = time.time()
                try:
                    r_mtime = rp.stat().st_mtime
                except OSError:
                    r_mtime = None
                if (r_mtime is not None
                        and now - float(reid_entry.get("pushed_at", 0)) >= REID_PUSH_EVERY_S
                        and r_mtime != reid_entry.get("mtime")):
                    data = _compact_sqlite_copy(rp)
                    if data is not None and len(data) <= MAX_FILE_BYTES:
                        blob = bucket.blob(f"{PREFIX}/reid.db")
                        blob.upload_from_string(
                            data, content_type="application/octet-stream")
                        blob.make_public()
                        reid_entry = {"mtime": r_mtime, "pushed_at": now,
                                      "url": blob.public_url, "size": len(data)}
                        state["_reid_db"] = reid_entry
                        uploaded += 1
                        changed = True
                    elif data is not None:
                        print(f"pool_sync: reid.db compact copy still "
                              f"{len(data)} bytes > cap {MAX_FILE_BYTES} - "
                              f"skipping push")

        if changed:
            manifest = {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "files": manifest_files,
            }
            if reid_entry.get("url"):
                manifest["reid_db"] = {"mtime": reid_entry["mtime"],
                                       "url": reid_entry["url"],
                                       "size": reid_entry.get("size", 0)}
            mblob = bucket.blob(f"{PREFIX}/{MANIFEST_NAME}")
            mblob.upload_from_string(json.dumps(manifest),
                                     content_type="application/json")
            mblob.make_public()
            _write_state(sp, state)
    except Exception as e:
        print(f"pool_sync: sync_up failed ({type(e).__name__}: {e})")
        return {"uploaded": uploaded, "deleted": deleted, "pending": pending,
                "error": str(e)}
    return {"uploaded": uploaded, "deleted": deleted, "pending": pending}


# ---- local side ----------------------------------------------------------------

def _bucket_name(web_dir: Path | None = None) -> str | None:
    """Resolve the Storage bucket: env first, then the storageBucket field of
    web/firebase-config.js (the dashboard's own config - always present on a
    working install, so the operator configures nothing new)."""
    env = os.environ.get("FIREBASE_STORAGE_BUCKET")
    if env:
        return env
    if web_dir is None:
        web_dir = Path(__file__).resolve().parent.parent / "web"
    cfg = web_dir / "firebase-config.js"
    try:
        text = cfg.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    import re
    m = re.search(r"storageBucket\s*:\s*[\"']([^\"']+)[\"']", text)
    return m.group(1) if m else None


def _http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pool-sync/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def pull_once(snapshots_root: str | Path,
              bucket: str | None = None,
              state_path: str | Path | None = None) -> dict:
    """One reconcile pass against the public manifest. Safe to call from a
    background thread; every failure is contained and reported in the dict."""
    root = Path(snapshots_root)
    bucket = bucket or _bucket_name()
    if not bucket:
        return {"error": "no storage bucket configured"}
    sp = Path(state_path) if state_path else root / PULL_STATE_NAME
    state = _read_state(sp)

    manifest_url = f"https://storage.googleapis.com/{bucket}/{PREFIX}/{MANIFEST_NAME}"
    try:
        manifest = json.loads(_http_get(manifest_url).decode("utf-8"))
    except Exception as e:
        return {"error": f"manifest fetch failed: {type(e).__name__}"}

    files = manifest.get("files") or {}
    downloaded = removed = 0
    errors = 0
    for rel, meta in files.items():
        # Path hardening: rel comes from the network; keep it inside the pools.
        parts = rel.split("/")
        if (".." in parts or rel.startswith("/") or "\\" in rel
                or parts[0] not in POOL_SUBDIRS):
            continue
        known = state.get(rel)
        # "Already pulled" only counts if the file still EXISTS locally: the
        # dashboard's clear-all buttons delete pool files, and a ledger that
        # keeps claiming them current would leave the pool empty until the
        # remote mtime happened to change.
        if (known is not None
                and float(known.get("mtime", -1)) == float(meta.get("mtime", -2))
                and (root / rel).is_file()):
            continue
        url = meta.get("url") or f"https://storage.googleapis.com/{bucket}/{PREFIX}/{rel}"
        try:
            data = _http_get(url)
        except Exception:
            errors += 1
            continue
        if len(data) > MAX_FILE_BYTES:
            continue
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            tmp.write_bytes(data)
            tmp.replace(dest)
        except OSError:
            errors += 1
            continue
        state[rel] = {"mtime": meta.get("mtime")}
        downloaded += 1

    # Files WE pulled earlier that the manifest no longer lists were LRU
    # -evicted on the VM - mirror the eviction. Never touches local-only files.
    for rel in [r for r in state if r not in files and r != "_reid_db"]:
        p = root / rel
        try:
            if p.is_file():
                p.unlink()
                removed += 1
        except OSError:
            pass
        state.pop(rel, None)

    # Re-ID registry snapshot -> data/reid.db so the search panel's registry
    # source (and the entity accordion) reflect what the VM has learned.
    rd = manifest.get("reid_db") or {}
    if rd.get("url"):
        known = state.get("_reid_db") or {}
        if float(known.get("mtime", -1)) != float(rd.get("mtime", -2)):
            try:
                data = _http_get(rd["url"])
                if len(data) <= MAX_FILE_BYTES:
                    dbp = Path(__file__).resolve().parent.parent / "data" / "reid.db"
                    dbp.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dbp.with_suffix(".db.part")
                    tmp.write_bytes(data)
                    tmp.replace(dbp)
                    state["_reid_db"] = {"mtime": rd.get("mtime")}
                    downloaded += 1
            except Exception:
                errors += 1

    if downloaded or removed:
        _write_state(sp, state)
    return {"downloaded": downloaded, "removed": removed, "errors": errors,
            "manifest_files": len(files)}


_PULL_THREAD: threading.Thread | None = None


def start_pull_thread(snapshots_root: str | Path,
                      interval_s: int = PULL_INTERVAL_S) -> bool:
    """Start the background puller (idempotent). Returns False when no bucket
    is resolvable - the dashboard then simply serves local-only pools, which
    is exactly the pre-sync behavior."""
    global _PULL_THREAD
    if _PULL_THREAD is not None and _PULL_THREAD.is_alive():
        return True
    if os.environ.get("POOL_SYNC_DISABLE") == "1":
        print("pool_sync: disabled via POOL_SYNC_DISABLE=1")
        return False
    bucket = _bucket_name()
    if not bucket:
        print("pool_sync: no storage bucket found (env or firebase-config.js) - "
              "pull disabled, serving local pools only")
        return False

    def _loop() -> None:
        while True:
            try:
                stats = pull_once(snapshots_root, bucket=bucket)
                if stats.get("downloaded") or stats.get("removed"):
                    print(f"pool_sync: pulled {stats.get('downloaded', 0)} file(s), "
                          f"removed {stats.get('removed', 0)} "
                          f"({stats.get('manifest_files', 0)} in manifest)")
                elif stats.get("error"):
                    # One line, not a stack trace: a VM that has not pushed a
                    # manifest yet (or an offline laptop) is a normal state.
                    print(f"pool_sync: {stats['error']}")
            except Exception as e:
                print(f"pool_sync: pull crashed ({type(e).__name__}: {e})")
            time.sleep(max(30, interval_s))

    _PULL_THREAD = threading.Thread(target=_loop, name="pool-sync-pull",
                                    daemon=True)
    _PULL_THREAD.start()
    print(f"pool_sync: pulling from gs://{bucket}/{PREFIX} every {interval_s}s")
    return True
