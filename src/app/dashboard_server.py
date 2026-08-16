"""Dashboard HTTP server building blocks shared by serve.py and the notebook.

Serves web/ statically AND proxies tvkur/IBB streams the browser can't reach
directly due to Referer/CORS requirements:

    GET /tvkur/<stream_id>/<path>           -> content.tvkur.com/l/<stream_id>/<path>
                                               with Referer/Origin=player.tvkur.com
    GET /snapshots/...                      -> web/snapshots/... (anomaly + returning frames)
    POST /api/visual-search                 -> search-by-example: body = an uploaded
                                               image, response = JSON ranking of saved
                                               snapshot crops + re-ID registry entities
                                               by visual similarity (app/visual_search).
                                               UI at /search.html.

The proxy adds Access-Control-Allow-Origin:* so hls.js in the dashboard can
fetch the master playlist and segments without browser CORS errors.

Visual-search knobs (env, all optional):
    REID_MODEL   path to an OSNet .onnx - upgrades the similarity signature
                 (must match the collector's --reid-model or the registry
                 search part silently no-ops on embedder mismatch);
    REID_DB      path to the collector's reid.db (default data/reid.db);
    SEARCH_YOLO  YOLO weights for query-object extraction (default yolov8s.pt;
                 set to "off" to skip detection and embed uploads whole).
"""
from __future__ import annotations

import http.server
import json
import os
import socket
import socketserver
import ssl
import sys
import threading
import time
import urllib.request
from pathlib import Path

# ThreadingHTTPServer is what we need: with 4 cameras each polling the HLS
# chunklist and pulling new .ts segments every few seconds (8-12 concurrent
# requests bursting in parallel), a single-threaded TCPServer queues them
# serially and the videos stall on "loading...". ThreadingHTTPServer hands
# each request to its own thread, which is what hls.js expects from a CDN.

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
SNAPSHOTS_DIR = WEB_DIR / "snapshots"
# Fixture frames the review-pool bootstrap seeds from. They're real
# captures from the four production cameras (see src/docs/images/), so
# the crops the first-time user reviews look exactly like what the
# collector will produce a few minutes later.
DOCS_IMAGES_DIR = ROOT / "docs" / "images"

_TVKUR_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; turkey-footfall-dashboard)",
    "Referer":    "https://player.tvkur.com/",
    "Origin":     "https://player.tvkur.com",
}
_SSL_CTX = ssl._create_unverified_context()

# Uploaded query images larger than this are rejected outright (a phone photo
# is ~3-6 MB; anything beyond 12 MB is not a search query).
MAX_UPLOAD_BYTES = 12 * 1024 * 1024


def _parse_time(v: str) -> float | None:
    """Accept ISO-8601 (`2026-07-06T18:00:00Z`), the browser's datetime-local
    format (`2026-07-06T18:00`), or a bare epoch-seconds number. Return
    epoch seconds. Empty / unparseable input returns None (open bound)."""
    v = (v or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        pass
    import datetime as _dt
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            t = _dt.datetime.strptime(v, fmt)
            # datetime-local sends naive strings; treat as UTC so the API
            # is timezone-stable across browsers.
            return t.replace(tzinfo=_dt.timezone.utc).timestamp()
        except ValueError:
            continue
    return None


class _VisualSearchState:
    """Lazily-built, process-wide search context shared across requests.

    Nothing here is touched until the FIRST /api/visual-search request, so a
    plain dashboard session never imports numpy/cv2/ultralytics. The YOLO
    model load (and its one-time weight download) happens once, behind a lock
    - ThreadingHTTPServer would otherwise race concurrent first requests into
    loading the model twice.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # Serializes SnapshotIndex.refresh() between the background
        # refresher thread and request handlers - two concurrent refreshes
        # would race the entries dict and double-embed the same backlog.
        self.refresh_lock = threading.Lock()
        self._ready = False
        self._model_lock = threading.Lock()
        self._model_ready = False
        self.embedder = None
        self.model = None
        self.index = None
        self.db_path = None

    def get_model(self):
        """The YOLO model ALONE, loaded on first call (~5-15s).

        Live analysis and the deep window need only the model; the full
        get() also builds the search index + crop bootstraps, which can
        take minutes on a cold start with a large synced pool - fix 2
        decouples them so an analysis click never waits on embedding
        backlogs. get() reuses this loader, so the model is still loaded
        exactly once per process.
        """
        with self._model_lock:
            if not self._model_ready:
                weights = os.environ.get("SEARCH_YOLO", "yolov8s.pt")
                if weights.lower() not in ("off", "none", ""):
                    try:
                        from app.detect_core import load_model
                        self.model = load_model(weights)
                    except Exception as e:
                        print(f"visual-search: YOLO unavailable ({e}) - "
                              f"uploads will be embedded whole (no object "
                              f"extraction). pip install ultralytics to fix.")
                self._model_ready = True
        return self.model

    def get(self):
        with self._lock:
            if not self._ready:
                from app.visual_search import DEFAULT_DB, SnapshotIndex
                from app.reid_embed import make_embedder
                self.embedder = make_embedder(os.environ.get("REID_MODEL") or None)
                self.db_path = os.environ.get("REID_DB") or DEFAULT_DB
                self.get_model()
                self.index = SnapshotIndex(SNAPSHOTS_DIR, embedder=self.embedder)
                # Extract per-object crops from the accumulated anomaly frames
                # so search + review can see them. Safe to fail silently: the
                # rest of the pipeline just doesn't pick up anomaly candidates
                # until YOLO is available on the next boot.
                if self.model is not None:
                    try:
                        from app.anomaly_crops import refresh as _anomaly_refresh
                        summary = _anomaly_refresh(
                            self.model, self.embedder, SNAPSHOTS_DIR)
                        print(f"visual-search: anomaly-crops refresh {summary}")
                    except Exception as e:
                        print(f"visual-search: anomaly-crops refresh failed "
                              f"({type(e).__name__}: {e}) - continuing")
                    # One-shot bootstrap: seed the review pool from the shipped
                    # camera fixture frames so the user sees ~8 real crops
                    # within seconds of dashboard startup, instead of waiting
                    # 3-5 minutes for the collector's first live samples.
                    try:
                        from app.live_samples import bootstrap_from_fixtures
                        n = bootstrap_from_fixtures(
                            self.model, DOCS_IMAGES_DIR, SNAPSHOTS_DIR)
                        if n:
                            print(f"visual-search: bootstrapped {n} demo "
                                  f"crops into live_samples/ so the review "
                                  f"UI has material on the first request")
                    except Exception as e:
                        print(f"visual-search: bootstrap skipped "
                              f"({type(e).__name__}: {e})")
                    # Same idea for the FRAME-based review pool (review_frames/):
                    # a fresh install had zero frames until the collector wrote
                    # one, so the Review-detections panel opened on "no frames
                    # in the pool yet" and could not teach the user anything.
                    try:
                        from app.review_frames import bootstrap_from_fixtures as _rf_bootstrap
                        n = _rf_bootstrap(self.model, DOCS_IMAGES_DIR, SNAPSHOTS_DIR)
                        if n:
                            print(f"visual-search: bootstrapped {n} demo "
                                  f"frames into review_frames/ so the Review "
                                  f"panel opens on real content")
                    except Exception as e:
                        print(f"visual-search: review-frames bootstrap skipped "
                              f"({type(e).__name__}: {e})")
                # NOTE: the per-object extraction + OSNet embedding of the
                # review-frames backlog (frame_crops.refresh) used to run
                # HERE, synchronously, at boot. With days of pool-sync
                # backlog that is MINUTES of onnxruntime grinding - and
                # py-spy caught it running head-to-head with OpenVINO
                # inference while an operator watched a live layer,
                # stretching 0.7s ticks to 6-12s. It now runs in the
                # background refresh loop, which yields while any live
                # analysis session is active. Search results simply
                # backfill a few minutes later on a fresh boot.
                self._ready = True
            return self


_VISUAL_SEARCH = _VisualSearchState()

# One deep-window analysis at a time: each run costs `frames` inferences,
# and ThreadingHTTPServer would happily start several in parallel on a
# double-clicked button - exactly the CPU spike the round budget forbids.
_DEEP_ANALYZE_LOCK = threading.Lock()
# Single-flight for the private dashboard's report sends (fix 2026-08-09).
_SEND_REPORT_LOCK = threading.Lock()

# Review store - lazily constructed on the first labels endpoint hit. The
# store is thread-safe (single lock around its dict + rewrite), so all
# handler threads share the one instance.
_REVIEW_STORE = None
_REVIEW_STORE_LOCK = threading.Lock()
# crop_path -> (sampler, uncertainty_at_selection) for crops served but not
# yet judged, so the submit row can record HOW the crop was chosen (spec
# 9.1) without any client-side change. Small and self-cleaning: entries pop
# on submit and the dict resets with the process.
_LAST_SERVED_CROPS: dict[str, tuple] = {}

# 60s memory cache for the cloud training_events pull (al-curve merge).
_CLOUD_TRAIN_CACHE: dict = {"at": 0.0, "points": []}


def _find_admin_key() -> str | None:
    """Admin service-account json: env first, then the operator's repo root
    (the same auto-detect the notebooks use)."""
    key = os.environ.get("FIREBASE_CREDENTIALS")
    if key and Path(key).is_file():
        return key
    for base in (ROOT.parent, ROOT):
        hits = sorted(base.glob("*adminsdk*.json"))
        if hits:
            return str(hits[0])
    return None


def _cloud_training_points() -> list[dict]:
    """Gate records from Firestore `training_events`, shaped like local
    al-curve points. Cached 60s; [] whenever creds/network are absent."""
    import time as _time
    now = _time.time()
    if now - _CLOUD_TRAIN_CACHE["at"] < 60:
        return _CLOUD_TRAIN_CACHE["points"]
    points: list[dict] = []
    key = _find_admin_key()
    if key:
        import firebase_admin
        from firebase_admin import credentials as _fbc, firestore as _fbf
        if not firebase_admin._apps:
            firebase_admin.initialize_app(_fbc.Certificate(key))
        for doc in _fbf.client().collection("training_events").stream():
            r = doc.to_dict() or {}
            cand_map = (r.get("metrics") or {}).get("map50")
            if r.get("event") != "gate" or cand_map is None:
                continue
            points.append({
                "at":           r.get("at"),
                "adapter":      r.get("candidate"),
                "labels_total": r.get("labels_total"),
                "map50":        cand_map,
                "promoted":     bool(r.get("promoted")),
                "baseline_map50": (r.get("baseline") or {}).get("map50"),
                "source":       "cloud",
            })
    _CLOUD_TRAIN_CACHE.update(at=now, points=points)
    return points


# fix 3: the VM publishes its heatmap grids (snapshots/heatmaps/<cam>.json)
# next to the overlay JPEG; /api/heatmap renders any layer x daypart from
# them when THIS machine has no local accumulation for the camera. Cached
# per camera so flipping the strip's selectors doesn't hammer Storage.
_VM_HEAT_CACHE: dict[str, tuple[float, dict]] = {}
_VM_HEAT_TTL_S = 60.0


def _vm_heat_state(cam: str) -> dict | None:
    now = time.time()
    hit = _VM_HEAT_CACHE.get(cam)
    if hit and now - hit[0] < _VM_HEAT_TTL_S:
        return hit[1]
    try:
        import json as _json
        from app.pool_sync import _bucket_name, _http_get
        bucket = _bucket_name()
        if not bucket:
            return hit[1] if hit else None
        raw = _http_get(f"https://storage.googleapis.com/{bucket}"
                        f"/snapshots/heatmaps/{cam}.json?t={int(now)}")
        state = _json.loads(raw.decode("utf-8"))
        _VM_HEAT_CACHE[cam] = (now, state)
        return state
    except Exception:
        return hit[1] if hit else None


def _grid_from_state(state: dict, layer: str, part: str | None):
    """Requested grid out of a published state dict; dayparts summed when
    `part` is None. Returns None when the state holds nothing usable."""
    layer_grids = (state.get("layers") or {}).get(layer) or {}
    if part:
        g = layer_grids.get(part)
        return [row[:] for row in g] if g else None
    out = None
    for g in layer_grids.values():
        if out is None:
            out = [[0.0] * len(g[0]) for _ in range(len(g))]
        for y, row in enumerate(g):
            for x, v in enumerate(row):
                out[y][x] += v
    return out


def _review_store():
    global _REVIEW_STORE
    with _REVIEW_STORE_LOCK:
        if _REVIEW_STORE is None:
            from app.labels import ReviewStore
            _REVIEW_STORE = ReviewStore()
        return _REVIEW_STORE


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Static handler for web/ + transparent tvkur HLS proxy.

    Browsers can't fetch content.tvkur.com directly:
    1. The CDN returns 403 without a Referer header (the browser sets Referer
       to the page origin, not player.tvkur.com).
    2. The CDN does NOT send Access-Control-Allow-Origin, so even if we got
       past 403, hls.js's fetch would fail browser CORS.

    Solution: when the browser asks for /tvkur/<id>/master.m3u8 we relay it
    server-side with the right Referer and add ACAO:* on the way back.
    """

    def end_headers(self) -> None:
        # No-cache for static files so JS edits show on reload (the proxy
        # path sets its own headers and returns early before reaching here).
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        # Under a Jupyter kernel this access log lands inside the notebook
        # cell's output area AND gets saved into the .ipynb (hundreds of
        # "GET /snapshots/..." lines per run bloated the file). Terminal
        # runs (serve.py) keep the log; notebook runs stay quiet.
        if "ipykernel" in sys.modules:
            return
        sys.stdout.write("  " + (fmt % args) + "\n")

    def do_GET(self) -> None:
        if self.path.startswith("/ytproxy"):
            self._proxy_yt()
            return
        if self.path.startswith("/tvkur/"):
            self._proxy_tvkur()
            return
        path = self.path.split("?")[0]
        if path == "/api/ping":
            # Capability probe: only THIS private server answers it, so the
            # frontend can tell "operator dashboard with a backend" from the
            # hosted public copy without sniffing hostnames (which lied
            # behind proxies). Gates the send-report field + live analysis.
            self._send_json(200, {"ok": True, "private": True})
            return
        if path == "/api/analysis/frame":
            self._analysis_frame()
            return
        if path == "/api/analysis/data":
            self._analysis_data()
            return
        if path == "/api/analysis/events":
            self._analysis_events()
            return
        if path == "/api/analysis/saved":
            self._analysis_saved()
            return
        if path == "/api/review-sample":
            self._review_sample()
            return
        if path == "/api/al-curve":
            self._al_curve()
            return
        if path == "/api/review-stats":
            self._review_stats()
            return
        if path == "/api/anomaly-crops-stats":
            self._anomaly_crops_stats()
            return
        if path == "/api/live-samples-stats":
            self._live_samples_stats()
            return
        if path == "/api/model-metrics":
            self._model_metrics()
            return
        if path == "/api/boost-status":
            self._boost_status()
            return
        if path == "/api/review-frame":
            self._review_frame_get()
            return
        if path == "/api/review-frames-list":
            self._review_frames_list()
            return
        if path == "/api/entity-gallery":
            self._entity_gallery()
            return
        if path == "/api/review-frames-stats":
            self._review_frames_stats()
            return
        if path == "/api/heatmap":
            self._heatmap()
            return
        if path == "/api/lines":
            self._get_line()
            return
        if path == "/api/zones":
            self._get_zones()
            return
        if path == "/api/crossings":
            self._get_crossings()
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        # /api/search is the current entry point (image + browse modes).
        # /api/visual-search is the compat alias for the legacy image-only
        # endpoint - the frontend and tools/search_by_image.py both used it
        # before the browse mode existed, so keep serving them from the same
        # handler.
        if path in ("/api/search", "/api/visual-search"):
            self._visual_search()
            return
        if path == "/api/review-submit":
            self._review_submit()
            return
        if path == "/api/anomaly-crops-clear":
            self._anomaly_crops_clear()
            return
        if path == "/api/live-samples-clear":
            self._live_samples_clear()
            return
        if path == "/api/review-frame-submit":
            self._review_frame_submit()
            return
        if path == "/api/review-frames-clear":
            self._review_frames_clear()
            return
        if path == "/api/blacklist-add":
            self._blacklist_add()
            return
        if path == "/api/deep-analyze":
            self._deep_analyze()
            return
        if path == "/api/analysis/start":
            self._analysis_start()
            return
        if path == "/api/analysis/stop":
            self._analysis_stop()
            return
        if path == "/api/analysis/event/save":
            self._analysis_event_save()
            return
        if path == "/api/send-report":
            self._send_report()
            return
        if path == "/api/lines":
            self._save_line()
            return
        if path == "/api/lines/clear":
            self._clear_line()
            return
        if path == "/api/zones":
            self._save_zones()
            return
        if path == "/api/zones/clear":
            self._clear_zones()
            return
        self.send_error(404, "unknown POST endpoint")

    # ---- Line-crossing config + event log --------------------------------
    # The Line layer in the dashboard lets the operator draw a virtual
    # counting line on a snapshot; every crossing then produces a toast +
    # a crop in the history strip. Three endpoints:
    #   GET  /api/lines?cam=<id>       -> {"line": [[x,y],[x,y]] | null, "set_at": ...}
    #   POST /api/lines?cam=<id>       body: {"line": [[x,y],[x,y]]}
    #   POST /api/lines/clear?cam=<id> -> delete the override, back to cameras.py
    #   GET  /api/crossings?cam=<id>&limit=20 -> newest-first events

    def _q_cam(self):
        """Extract the ?cam= query arg. Returns cam_id or None (and writes 400)."""
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(self.path).query)
        cam = (q.get("cam") or [""])[0].strip()
        if not cam:
            self.send_error(400, "missing ?cam=")
            return None
        return cam

    def _get_line(self) -> None:
        cam = self._q_cam()
        if cam is None:
            return
        # Route through resolve_line + resolve_line_classes so a malformed
        # override on disk falls back to the CAMERAS catalog silently -
        # the same rule the collector follows on the next round. Reading
        # the JSON here without the validator would let a bad hand-edit
        # paint a line the frontend believes in but the counter never uses.
        from app.cameras import (LINE_ALLOWED_CLASSES, _lines_dir,
                                 resolve_line, resolve_line_classes)
        p = _lines_dir() / f"{cam}.json"
        set_at = None
        if p.exists():
            try:
                set_at = json.loads(p.read_text()).get("set_at")
            except (OSError, ValueError):
                set_at = None
        line = resolve_line(cam)
        classes = resolve_line_classes(cam)
        body = json.dumps({"cam": cam, "line": line,
                           "classes": classes,
                           "allowed_classes": sorted(LINE_ALLOWED_CLASSES),
                           "set_at": set_at,
                           "user_override": p.exists()}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def _save_line(self) -> None:
        cam = self._q_cam()
        if cam is None:
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 1024:
            self.send_error(400, "empty or oversized body"); return
        try:
            data = json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.send_error(400, "body must be JSON"); return
        line = data.get("line")
        classes = data.get("classes")
        from app.cameras import save_line
        try:
            save_line(cam, line, classes=classes)
        except ValueError as e:
            self.send_error(400, str(e)); return
        body = json.dumps({"ok": True, "cam": cam, "line": line,
                           "classes": classes}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def _clear_line(self) -> None:
        cam = self._q_cam()
        if cam is None:
            return
        from app.cameras import clear_line
        removed = clear_line(cam)
        body = json.dumps({"ok": True, "cam": cam, "removed": removed}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    # ---- Analysis zones (loiter areas + parking spots) -------------------
    #   GET  /api/zones?cam=<id>        -> {"zones": [...]}
    #   POST /api/zones?cam=<id>        body: {"zones": [...]} (full replace)
    #   POST /api/zones/clear?cam=<id>  -> delete the file
    # Running loiter/parking sessions hot-reload within a few seconds.

    def _get_zones(self) -> None:
        cam = self._q_cam()
        if cam is None:
            return
        from app.cameras import resolve_zones
        self._send_json(200, {"zones": resolve_zones(cam)})

    def _save_zones(self) -> None:
        cam = self._q_cam()
        if cam is None:
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 64 * 1024:
            self.send_error(400, "empty or oversized body"); return
        try:
            data = json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.send_error(400, "body must be JSON"); return
        from app.cameras import save_zones
        try:
            save_zones(cam, data.get("zones"))
        except ValueError as e:
            self.send_error(400, str(e)); return
        self._send_json(200, {"ok": True, "cam": cam,
                              "count": len(data.get("zones") or [])})

    def _clear_zones(self) -> None:
        cam = self._q_cam()
        if cam is None:
            return
        from app.cameras import clear_zones
        self._send_json(200, {"ok": True, "removed": clear_zones(cam)})

    def _get_crossings(self) -> None:
        cam = self._q_cam()
        if cam is None:
            return
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(self.path).query)
        limit = 20
        try:
            limit = max(1, min(200, int((q.get("limit") or ["20"])[0])))
        except ValueError:
            pass
        from app.live_analysis import read_crossing_events
        events = read_crossing_events(cam, limit=limit)
        body = json.dumps({"cam": cam, "events": events}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def _send_report(self) -> None:
        """POST /api/send-report?to=<email>[&window_hours=12]

        The PRIVATE dashboard's send button (2026-08-09): composes the
        situation report from the live cloud data and mails it FROM the
        project mailbox to the given address, CC the project mailbox so
        the archive stays complete. This endpoint exists only on the
        operator-controlled server - the PUBLIC dashboard's button goes
        through the send-report GitHub workflow, where GitHub login gates
        abuse. Credentials resolve from data/mailer.env, falling back to
        /etc/turkey-footfall/digest.env (the VM). One send at a time with
        a 60s cooldown."""
        import re
        import subprocess
        import sys as _sys
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        to = (q.get("to") or [""])[0].strip()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", to):
            self._send_json(400, {"error": "invalid destination email"})
            return
        try:
            hours = max(1, min(24, int((q.get("window_hours") or ["12"])[0])))
        except ValueError:
            hours = 12

        creds: dict[str, str] = {}
        for env_path in (Path(__file__).resolve().parent.parent
                         / "data" / "mailer.env",
                         Path("/etc/turkey-footfall/digest.env")):
            try:
                for line in env_path.read_text().splitlines():
                    k, _, v = line.strip().partition("=")
                    if k and v and k not in creds:
                        creds[k] = v
                if creds.get("GMAIL_USER"):
                    break
            except OSError:
                continue
        if not creds.get("GMAIL_USER") or not creds.get("GMAIL_APP_PASSWORD"):
            self._send_json(503, {"error": "no mail credential - create "
                                           "data/mailer.env with GMAIL_USER "
                                           "+ GMAIL_APP_PASSWORD"})
            return
        from app.training_sync import find_service_account
        sa = find_service_account()
        if not sa:
            self._send_json(503, {"error": "no Firebase service-account key "
                                           "on this machine"})
            return

        now = time.time()
        if now - getattr(self.__class__, "_last_report_send", 0.0) < 60.0:
            self._send_json(429, {"error": "a report was sent less than a "
                                           "minute ago - try again shortly"})
            return
        if not _SEND_REPORT_LOCK.acquire(blocking=False):
            self._send_json(409, {"error": "a send is already running"})
            return
        try:
            self.__class__._last_report_send = now
            archive = creds["GMAIL_USER"]
            env = dict(os.environ,
                       FIREBASE_CREDENTIALS=sa,
                       GMAIL_USER=creds["GMAIL_USER"],
                       GMAIL_APP_PASSWORD=creds["GMAIL_APP_PASSWORD"],
                       DIGEST_TO=(to if to == archive
                                  else f"{to},{archive}"))
            proc = subprocess.run(
                [_sys.executable, "-m", "tools.daily_digest",
                 "--window-hours", str(hours)],
                cwd=str(Path(__file__).resolve().parent.parent),
                env=env, capture_output=True, text=True, timeout=300)
            sent_line = next((l for l in (proc.stdout or "").splitlines()
                              if "digest sent" in l), "")
            if proc.returncode == 0 and sent_line:
                print(f"  report sent to {to} ({sent_line.strip()})")
                self._send_json(200, {"ok": True, "to": to,
                                      "detail": sent_line.strip()})
            else:
                tail = ((proc.stderr or "") + (proc.stdout or ""))[-400:]
                self._send_json(502, {"error": f"send failed: {tail}"})
        except subprocess.TimeoutExpired:
            self._send_json(504, {"error": "report composition timed out"})
        except Exception as e:
            self._send_json(502, {"error": f"{type(e).__name__}: {e}"})
        finally:
            _SEND_REPORT_LOCK.release()

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()   # skip our no-cache re-header dance
        self.wfile.write(body)

    def _send_bytes(self, status: int, content_type: str,
                    data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()
        self.wfile.write(data)

    def _heatmap(self) -> None:
        """GET /api/heatmap?cam=<id>[&layer=person|vehicles|other]
        [&part=night|morning|afternoon|evening][&format=json]

        Default response is the rendered overlay JPEG. The base image is
        the camera's freshest review frame when one exists (scene context);
        otherwise the map renders on a dark canvas. format=json returns
        the raw grid + stats for client-side rendering."""
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)

        def _one(name, default=None):
            try:
                return q[name][0]
            except (KeyError, IndexError):
                return default

        cam = _one("cam", "")
        if not cam or "/" in cam or "\\" in cam or ".." in cam:
            self._send_json(400, {"error": "missing or invalid ?cam="})
            return
        layer = _one("layer", "person")
        part = _one("part") or None
        try:
            from app import heatmap as hm
        except Exception as e:
            self._send_json(500, {"error": f"heatmap unavailable: {e}"})
            return
        if layer not in ("person", "vehicles", "other"):
            self._send_json(400, {"error": "layer must be person|vehicles|other"})
            return
        if part is not None and part not in hm.DAYPARTS:
            self._send_json(400, {"error": f"part must be one of {hm.DAYPARTS}"})
            return
        # Local accumulation first (a collector/notebook run on THIS
        # machine); empty -> the state the VM publishes next to its
        # overlay (fix 3), so the operator sees the cloud's depth.
        grid = hm.grid_for(cam, layer=layer, daypart=part)
        source = "local"
        if not any(v for row in grid for v in row):
            state = _vm_heat_state(cam)
            vm_grid = _grid_from_state(state, layer, part) if state else None
            if vm_grid:
                grid = vm_grid
                source = "vm"
        if _one("format") == "json":
            payload = hm.stats(cam)
            payload["layer"] = layer
            payload["part"] = part
            payload["source"] = source
            payload["grid"] = grid
            self._send_json(200, payload)
            return
        base = None
        frames_dir = SNAPSHOTS_DIR / "review_frames" / cam
        if frames_dir.is_dir():
            jpgs = sorted(frames_dir.glob("*.jpg"),
                          key=lambda p: p.stat().st_mtime)
            if jpgs:
                try:
                    import cv2
                    base = cv2.imread(str(jpgs[-1]))
                except Exception:
                    base = None
        try:
            import cv2
            img = hm.overlay(grid, base_frame=base)
            okj, buf = cv2.imencode(".jpg", img,
                                    [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not okj:
                raise RuntimeError("jpeg encode failed")
        except Exception as e:
            self._send_json(500, {"error": f"render failed: {e}"})
            return
        self._send_bytes(200, "image/jpeg", buf.tobytes())

    def _deep_analyze(self) -> None:
        """POST /api/deep-analyze?cam=<id>[&frames=12][&stride=12][&imgsz=640]
                                 [&pose=1][&faces=1][&lock=auto|<track id>]

        Operator-triggered deep window: grabs `frames` frames from ONE
        camera, tracks every individual (position + motion) and returns
        the per-individual behavior profile + the trails image URL.
        `pose=1` adds the skeleton pass (posture labels + gestures),
        `faces=1` adds face-detection boxes on the final frame, `lock`
        draws the crosshair target-lock overlay on one individual. Costs
        `frames` inferences (double with pose), so one analysis runs at a
        time - a second request while one is in flight gets 409."""
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)

        def _one(name, cast, default, lo, hi):
            try:
                return max(lo, min(hi, cast(q[name][0])))
            except (KeyError, IndexError, ValueError):
                return default

        cam = (q.get("cam") or [""])[0]
        if not cam:
            self._send_json(400, {"error": "missing ?cam="})
            return
        n_frames = _one("frames", int, 12, 4, 24)
        stride = _one("stride", int, 12, 4, 40)
        imgsz = _one("imgsz", int, 640, 320, 960)
        pose = _one("pose", int, 0, 0, 1) == 1
        want_faces = _one("faces", int, 0, 0, 1) == 1
        lock = (q.get("lock") or [None])[0] or None

        model = _VISUAL_SEARCH.get_model()
        if model is None:
            self._send_json(503, {"error": "no detection model loaded "
                                           "(SEARCH_YOLO=off?)"})
            return
        if not _DEEP_ANALYZE_LOCK.acquire(blocking=False):
            self._send_json(409, {"error": "an analysis is already running - "
                                           "try again in a few seconds"})
            return
        try:
            from app.behavior import analyze_window
            from app.live_analysis import INFER_LOCK
            # INFER_LOCK: the live-analysis sessions share this exact model
            # object; ultralytics predict is not thread-safe, so the deep
            # window holds the same lock (live tiles pause for its ~10-20s
            # and resume - visible as a longer gap, never a crash).
            with INFER_LOCK:
                result = analyze_window(cam, model, n_frames=n_frames,
                                        stride=stride, imgsz=imgsz,
                                        pose=pose, lock=lock,
                                        want_faces=want_faces)
            self._send_json(200, result)
        except ValueError as e:
            self._send_json(404, {"error": str(e)})
        except Exception as e:
            self._send_json(502, {"error": f"{type(e).__name__}: {e}"})
        finally:
            _DEEP_ANALYZE_LOCK.release()

    # -- fix 2: live advanced analysis (app/live_analysis.py) --------------

    def _analysis_start(self) -> None:
        """POST /api/analysis/start?cam=<id>&layer=<layer>

        Starts a live-analysis session on ONE camera (registry id or a
        local-picker slot id), or switches the layer of a running session
        in place - stream, tracker and accumulators survive the switch.
        At most live_analysis.MAX_SESSIONS run concurrently (409 beyond).
        """
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        cam = (q.get("cam") or [""])[0]
        layer = (q.get("layer") or [""])[0]
        if not cam or not layer:
            self._send_json(400, {"error": "need ?cam= and ?layer="})
            return
        model = _VISUAL_SEARCH.get_model()
        if model is None:
            self._send_json(503, {"error": "no detection model loaded "
                                           "(SEARCH_YOLO=off?)"})
            return
        from app.live_analysis import MANAGER, BusyError
        try:
            info = MANAGER.start(cam, layer, model)
            self._send_json(200, {"ok": True, **info})
        except BusyError as e:
            self._send_json(409, {"error": str(e)})
        except ValueError as e:
            self._send_json(404, {"error": str(e)})
        except Exception as e:
            self._send_json(502, {"error": f"{type(e).__name__}: {e}"})

    def _analysis_frame(self) -> None:
        """GET /api/analysis/frame?cam=<id>

        Latest analyzed JPEG of the session (200 image/jpeg with X-Seq /
        X-Layer / X-Note headers), 202 JSON while the first frame is
        still being produced, 410 JSON when the session died with a
        reported reason (so the operator sees WHY, not a bare 404), and
        404 when no session ever ran for this camera. Polling this keeps
        the session's idle clock alive.
        """
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        cam = (q.get("cam") or [""])[0]
        from app.live_analysis import MANAGER
        fr = MANAGER.frame(cam) if cam else None
        if fr is None:
            self._send_json(404, {"error": "no live analysis for this "
                                           "camera"})
            return
        if fr.get("error"):
            # Session crashed / ended - report the reason once so the UI
            # can distinguish a fatal analysis error from "never started".
            self._send_json(410, {"error": fr["error"], "ended": True})
            return
        if not fr["jpeg"]:
            self._send_json(202, {"ok": True, "pending": True,
                                  "note": fr["note"] or "starting..."})
            return
        body = fr["jpeg"]
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Seq", str(fr["seq"]))
        self.send_header("X-Layer", fr["layer"])
        note = (fr["note"] or "").encode("ascii", "replace").decode("ascii")
        if note:
            self.send_header("X-Note", note)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # poller gave up mid-frame - the next poll catches up

    def _analysis_data(self) -> None:
        """GET /api/analysis/data?cam=<id>

        JSON snapshot for the canvas-overlay renderer: boxes+heat+line
        instead of a rendered JPEG. The client draws these on a canvas
        positioned over the live iframe so the video stays smooth while
        the overlay ticks at YOLO pace. Same idle-clock behaviour as
        /api/analysis/frame: polling keeps the session alive.
        """
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        cam = (q.get("cam") or [""])[0]
        from app.live_analysis import MANAGER
        d = MANAGER.data(cam) if cam else None
        if d is None:
            self._send_json(404, {"error": "no live analysis for this "
                                           "camera"})
            return
        if d.get("error"):
            self._send_json(410, {"error": d["error"], "ended": True})
            return
        if not d.get("data"):
            self._send_json(202, {"ok": True, "pending": True,
                                  "note": d.get("note") or "starting..."})
            return
        payload = dict(d["data"])
        payload["seq"] = d["seq"]
        payload["layer"] = d["layer"]
        self._send_json(200, payload)

    def _analysis_stop(self) -> None:
        """POST /api/analysis/stop?cam=<id> - back to plain video."""
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        cam = (q.get("cam") or [""])[0]
        from app.live_analysis import MANAGER
        stopped = MANAGER.stop(cam) if cam else False
        self._send_json(200, {"ok": True, "stopped": stopped})

    def _analysis_events(self) -> None:
        """GET /api/analysis/events?cam=<id> - the session's detection
        event ring (newest first, thumbs only; full frames stay on the
        server until an explicit save)."""
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        cam = (q.get("cam") or [""])[0]
        from app.live_analysis import MANAGER
        evs = MANAGER.events(cam) if cam else None
        if evs is None:
            self._send_json(404, {"error": "no live analysis for this "
                                           "camera"})
            return
        self._send_json(200, {"events": evs})

    def _analysis_event_save(self) -> None:
        """POST /api/analysis/event/save?cam=<id>&id=<event_id> - persist
        one ring event (full frame + manifest row) for later study."""
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        cam = (q.get("cam") or [""])[0]
        eid = (q.get("id") or [""])[0]
        from app.live_analysis import MANAGER
        row = MANAGER.save_event(cam, eid) if cam and eid else None
        if row is None:
            self._send_json(404, {"error": "event not found (ring may "
                                           "have rolled past it)"})
            return
        self._send_json(200, {"ok": True, "saved": row})

    def _analysis_saved(self) -> None:
        """GET /api/analysis/saved[?country=turkey] - manifest of saved
        detection events. Optional country filter keeps only events whose
        cam_id belongs to the given country pool (Repo #1 default is
        turkey - the VM's auto-fallback to TH/JP/US leaves foreign items
        in the shared bank; the filter hides them from a country-locked
        dashboard)."""
        import json as _json
        from pathlib import Path
        from urllib.parse import parse_qs, urlparse
        man = (Path(__file__).resolve().parent.parent / "web" / "snapshots"
               / "detections" / "saved.json")
        try:
            items = _json.loads(man.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            items = []
        q = parse_qs(urlparse(self.path).query)
        allowed = self._resolve_scope(q)
        if allowed is not None:
            items = [it for it in items
                     if (it.get("cam_id") or it.get("cam") or "") in allowed]
        self._send_json(200, {"items": items})

    def _visual_search(self) -> None:
        """POST /api/search  (or /api/visual-search - the legacy alias).

        Query params (all optional):
            top=12               how many results to return
            min_sim=0.30         image mode: minimum cosine similarity floor
            classes=person,car   restrict candidates to these classes
            from=<iso|epoch>     filter: seen at or after this time
            to=<iso|epoch>       filter: seen at or before this time
            order=time_desc      browse mode: time_desc | time_asc

        Body:
            when non-empty: raw image bytes → image mode (rank by similarity)
            when empty:     browse mode → list crops matching class/time
                            filters ordered by time
        """
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length > MAX_UPLOAD_BYTES:
            self._send_json(413, {"error": f"image too large (>{MAX_UPLOAD_BYTES} bytes)"})
            return
        data = self.rfile.read(length) if length > 0 else b""

        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)

        def _one(name, cast, default):
            try:
                return cast(q[name][0])
            except (KeyError, IndexError, ValueError):
                return default

        top_n   = max(1, min(200, _one("top", int, 12)))
        # Default floor lifted from 0.30 to 0.55 to cut color-similar noise
        # from the results; see visual_search.MIN_SIMILARITY_FLOOR.
        min_sim = _one("min_sim", float, 0.55)
        classes = {c.strip() for c in (q.get("classes", [""])[0]).split(",")
                   if c.strip()} or None
        time_from = _parse_time(q.get("from", [""])[0])
        time_to   = _parse_time(q.get("to", [""])[0])
        order     = q.get("order", ["time_desc"])[0]
        try:
            st = _VISUAL_SEARCH.get()
            # Fold any review frames that arrived since the last search (the
            # pool-sync puller drops new ones in every couple of minutes) into
            # review_crops/ so they are searchable RIGHT NOW. No-op when the
            # frames pool hasn't changed - one directory listing.
            try:
                from app.frame_crops import refresh as _fc_refresh
                fc = _fc_refresh(st.embedder, SNAPSHOTS_DIR)
                if fc.get("crops_added"):
                    print(f"  * review-crops: +{fc['crops_added']} "
                          f"({fc.get('crops_skipped_dup', 0)} dup-skipped)")
            except Exception as ex:
                print(f"  ! review-crops refresh skipped: {type(ex).__name__}: {ex}")
            if data:
                from app.visual_search import search_image_bytes
                result = search_image_bytes(
                    data, model=st.model, embedder=st.embedder,
                    snapshot_index=st.index, db_path=st.db_path,
                    top_n=top_n, min_sim=min_sim, classes=classes,
                    time_from=time_from, time_to=time_to)
                result["detector"] = "yolo" if st.model is not None else "whole-image"
                # Auto-Loose fallback: when the user picks Balanced/Strict and
                # gets NOTHING back, silently retry at the Loose floor and tag
                # the response. Better UX than making the user notice the empty
                # state and click Loose themselves. Only fires when the user
                # didn't already pick 0.30 - we do not want to hide a genuine
                # "no similar crops anywhere at any strictness" state.
                total = (len(result.get("snapshot_matches") or [])
                         + len(result.get("registry_matches") or []))
                if total == 0 and min_sim > 0.30:
                    loose = search_image_bytes(
                        data, model=st.model, embedder=st.embedder,
                        snapshot_index=st.index, db_path=st.db_path,
                        top_n=top_n, min_sim=0.30, classes=classes,
                        time_from=time_from, time_to=time_to)
                    loose_total = (len(loose.get("snapshot_matches") or [])
                                   + len(loose.get("registry_matches") or []))
                    if loose_total > 0:
                        result["snapshot_matches"] = loose.get("snapshot_matches") or []
                        result["registry_matches"] = loose.get("registry_matches") or []
                        result["fallback"] = {"from_min_sim": min_sim,
                                              "to_min_sim": 0.30,
                                              "note": "auto-loose retry"}
            else:
                # Browse mode: no reference photo. The user asked for
                # "N cars between X and Y" - list crops in time order.
                from app.visual_search import browse_snapshots
                result = browse_snapshots(
                    embedder=st.embedder, snapshot_index=st.index,
                    classes=classes, time_from=time_from, time_to=time_to,
                    limit=top_n, order=order)
                result["detector"] = "browse"
            self._send_json(200, result)
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
        except Exception as e:
            print(f"  ! visual-search failed: {type(e).__name__}: {e}")
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    # ---- human-in-the-loop review endpoints ------------------------------
    # Backing the "Review detections" panel in index.html. The user is shown
    # one un-reviewed crop with its current label and picks correct /
    # wrong-label / not-an-object. Answers persist to data/reviews.json via
    # ReviewStore. Sampler: REVIEW_SAMPLER=badge|naive (default naive),
    # overridable per request with ?strategy= (plan WS2). The response
    # carries "sampler" so the UI can badge it, and the server remembers
    # what it served so the submit row records sampler +
    # uncertainty_at_selection (spec 9.1) without any client change.
    def _review_sample(self) -> None:
        try:
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            strategy = ((q.get("strategy") or [""])[0]
                        or os.environ.get("REVIEW_SAMPLER") or "naive").lower()
            s = None
            if strategy == "badge":
                try:
                    from app.badge import sample_crop_badge
                    ranked = sample_crop_badge(_review_store(), SNAPSHOTS_DIR)
                    if ranked and ranked.get("batch"):
                        s = {**ranked["batch"][0], "sampler": "badge"}
                except Exception as ex:
                    print(f"  ! badge sampler failed, falling back to naive: "
                          f"{type(ex).__name__}: {ex}")
            if s is None:
                from app.labels import sample_crop
                s = sample_crop(_review_store(), SNAPSHOTS_DIR)
                if s is not None:
                    s["sampler"] = "naive"
            if s is None:
                self._send_json(200, {"done": True,
                                      "message": "no un-reviewed crops in the store"})
                return
            _LAST_SERVED_CROPS[s["path"]] = (s.get("sampler"),
                                             s.get("uncertainty"))
            self._send_json(200, s)
        except Exception as e:
            print(f"  ! review-sample failed: {type(e).__name__}: {e}")
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _review_submit(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > 32 * 1024:
            self._send_json(400, {"error": "empty or oversized body"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send_json(400, {"error": "body must be JSON"})
            return
        crop_path = str(payload.get("crop_path", "")).strip()
        verdict   = str(payload.get("verdict", "")).strip()
        if not crop_path or not verdict:
            self._send_json(400, {"error": "crop_path and verdict are required"})
            return
        # crop_path must stay inside snapshots dir - reject anything with a
        # backslash or path escape ("../"). The store already treats it as a
        # relative key but we harden the input surface too.
        if ".." in crop_path.split("/") or crop_path.startswith("/") \
                or "\\" in crop_path:
            self._send_json(400, {"error": "invalid crop_path"})
            return
        try:
            # A re-submission overwrites the stored review (keyed by path) but
            # must NOT nudge confidence a second time - otherwise clicking
            # through the same crop twice counts as two learning events and
            # the boost ledger drifts away from the review store.
            was_reviewed = _review_store().is_reviewed(crop_path)
            served_sampler, served_u = _LAST_SERVED_CROPS.pop(
                crop_path, (None, None))
            r = _review_store().submit(
                crop_path,
                verdict,
                original_cls=str(payload.get("original_cls", "?")),
                corrected_cls=payload.get("corrected_cls") or None,
                anomaly_verdict=payload.get("anomaly_verdict") or None,
                note=payload.get("note") or None,
                sampler=payload.get("sampler") or served_sampler,
                uncertainty_at_selection=(
                    payload.get("uncertainty_at_selection")
                    if payload.get("uncertainty_at_selection") is not None
                    else served_u))
            # After each submit, let the auto-blacklist accumulator decide
            # whether N repeated rejects in one area now justify auto-adding
            # a polygon. Silent failure - we never want a blacklist step to
            # break a save.
            try:
                from app.auto_blacklist import consider_review
                consider_review(_review_store(), r)
            except Exception as ex:
                print(f"  ! auto_blacklist skipped: {type(ex).__name__}: {ex}")
            # Positive/negative confidence boost. Correct verdicts lower
            # per-cam per-cls conf (missing real ones); wrong verdicts raise
            # it (false positives). Value is persisted so the collector
            # picks it up on its next hot-reload without a restart.
            try:
                if not was_reviewed:
                    from app.confidence_boost import apply_review
                    from app.auto_blacklist import _cam_id_from_crop
                    cam_id_from_crop = _cam_id_from_crop(crop_path)
                    if cam_id_from_crop:
                        apply_review(cam_id_from_crop,
                                     str(payload.get("original_cls", "?")),
                                     verdict)
            except Exception as ex:
                print(f"  ! confidence_boost skipped: {type(ex).__name__}: {ex}")
            # Ship the fresh verdicts to cloud Storage so the nightly
            # trainer sees them with the operator's PC off. Fire-and-forget.
            try:
                from app.training_sync import push_async
                push_async()
            except Exception as ex:
                print(f"  ! training_sync skipped: {type(ex).__name__}: {ex}")
            self._send_json(200, {"ok": True, "review": r.to_public(),
                                  "summary": _review_store().summary()})
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
        except Exception as e:
            print(f"  ! review-submit failed: {type(e).__name__}: {e}")
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _review_stats(self) -> None:
        try:
            summary = _review_store().summary()
            try:
                from app.confidence_boost import summary as _cb_summary
                summary["boost"] = _cb_summary()
            except Exception as ex:
                summary["boost"] = {"error": f"{type(ex).__name__}"}
            self._send_json(200, summary)
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _anomaly_crops_stats(self) -> None:
        try:
            from app.anomaly_crops import usage_stats
            self._send_json(200, usage_stats(SNAPSHOTS_DIR))
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _anomaly_crops_clear(self) -> None:
        # Clear the on-disk crops then rebuild whatever the live anomaly
        # frames still cover, so the user isn't left with an empty pool. The
        # rebuild happens IN-PROCESS on the visual-search state's already
        # -loaded model - no second YOLO load, no cold start.
        try:
            from app.anomaly_crops import clear_all, refresh
            result = clear_all(SNAPSHOTS_DIR)
            if _VISUAL_SEARCH._ready and _VISUAL_SEARCH.model is not None:
                try:
                    reseeded = refresh(_VISUAL_SEARCH.model,
                                       _VISUAL_SEARCH.embedder,
                                       SNAPSHOTS_DIR)
                    result["reseeded"] = reseeded
                except Exception as e:
                    result["reseed_error"] = f"{type(e).__name__}: {e}"
            self._send_json(200, result)
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _live_samples_stats(self) -> None:
        try:
            from app.live_samples import usage_stats
            self._send_json(200, usage_stats(SNAPSHOTS_DIR))
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _live_samples_clear(self) -> None:
        # "Clear" now means "clear + reseed", so the review UI doesn't die
        # the moment the user clicks it locally. clear_all already drops the
        # bootstrap marker; bootstrap_from_fixtures sees the missing marker
        # and re-seeds fresh crops from the shipped model_view_*.jpg frames.
        try:
            from app.live_samples import (clear_all as ls_clear,
                                          bootstrap_from_fixtures)
            result = ls_clear(SNAPSHOTS_DIR)
            if _VISUAL_SEARCH._ready and _VISUAL_SEARCH.model is not None:
                try:
                    reseeded = bootstrap_from_fixtures(
                        _VISUAL_SEARCH.model, DOCS_IMAGES_DIR, SNAPSHOTS_DIR)
                    result["reseeded"] = reseeded
                except Exception as e:
                    result["reseed_error"] = f"{type(e).__name__}: {e}"
            self._send_json(200, result)
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    # ---- Frame-based review endpoints ----------------------------------
    # The new canvas UX: one frame carries multiple detections, the user
    # gives a verdict per BOX, plus optional "missed" boxes drawn on the
    # canvas. That last piece is what finally gives us FN → recall → F1.
    def _review_frame_get(self) -> None:
        """GET /api/review-frame            -> next un-reviewed frame (sampler)
           GET /api/review-frame?path=<rel> -> that SPECIFIC frame, reviewed or
        not, with any prior verdicts under ``existing`` so the UI can prefill
        and let the user amend a past review."""
        try:
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            rel = (q.get("path") or [""])[0].strip()
            if rel:
                if ".." in rel.split("/") or rel.startswith("/") or "\\" in rel:
                    self._send_json(400, {"error": "invalid path"})
                    return
                from app.labels import load_frame
                s = load_frame(_review_store(), rel, SNAPSHOTS_DIR)
                if s is None:
                    self._send_json(404, {"error": "frame not found"})
                    return
                self._send_json(200, s)
                return
            from app.labels import sample_frame
            allowed = self._resolve_scope(q)
            s = sample_frame(_review_store(), SNAPSHOTS_DIR,
                             allowed_cams=(allowed or None))
            if s is None:
                self._send_json(200, {"done": True,
                                      "message": "no un-reviewed frames yet"})
                return
            self._send_json(200, s)
        except Exception as e:
            print(f"  ! review-frame failed: {type(e).__name__}: {e}")
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    @staticmethod
    def _local_pick_ids() -> set:
        """The MAIN edition's camera universe: the picked slots' ids plus
        their catalog ids (frames get saved under either)."""
        import json as _json
        try:
            grid = _json.loads((WEB_DIR / "local_grid.json").read_text(
                encoding="utf-8"))
        except (OSError, ValueError):
            return set()
        out = set()
        for s in grid.get("slots") or []:
            if s.get("slot_id"):
                out.add(s["slot_id"])
            if s.get("cam_id"):
                out.add(s["cam_id"])
        return out

    @staticmethod
    def _frame_cam(f: dict) -> str:
        cam = f.get("cam_id") or f.get("cam") or ""
        if not cam:
            parts = str(f.get("frame_path") or f.get("path") or "").split("/")
            if len(parts) >= 2:
                cam = parts[1]
        return cam

    @staticmethod
    def _country_cam_ids(country: str) -> set:
        """Cam ids belonging to the given country per app/cameras.py
        country_pool. Empty set on unknown country or import error."""
        try:
            from app.cameras import country_pool
            return set(country_pool((country or "").lower()))
        except Exception:
            return set()

    def _resolve_scope(self, q) -> set | None:
        """Resolve the allowed cam_id set from a request's query string.
        Order: scope=local (only if the operator has picked cameras) ->
        country=<name> (Turkey by contract of Repo #1) -> no filter.
        Returns None when no filter should apply."""
        scope = (q.get("scope") or [""])[0].strip()
        country = (q.get("country") or [""])[0].strip().lower()
        if scope == "local":
            picks = self._local_pick_ids()
            if picks:
                return picks
        if country:
            pool = self._country_cam_ids(country)
            if pool:
                return pool
        return None

    def _review_frames_list(self) -> None:
        """GET /api/review-frames-list[?scope=local][&country=turkey] ->
        every stored frame + review status, newest first. Powers the strip
        that re-opens reviewed frames. Filter precedence: scope=local (only
        when the operator has picked cameras) -> country pool -> no filter.
        Frames from the VM's auto-fallback pools (th_*, jp_*, us_*) are
        excluded when country=turkey is set."""
        try:
            from urllib.parse import parse_qs, urlparse
            from app.labels import list_frames
            frames = list_frames(_review_store(), SNAPSHOTS_DIR)
            q = parse_qs(urlparse(self.path).query)
            allowed = self._resolve_scope(q)
            if allowed is not None:
                frames = [f for f in frames
                          if self._frame_cam(f) in allowed]
            self._send_json(200, {"frames": frames})
        except Exception as e:
            print(f"  ! review-frames-list failed: {type(e).__name__}: {e}")
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _review_frame_submit(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > 128 * 1024:
            self._send_json(400, {"error": "empty or oversized body"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send_json(400, {"error": "body must be JSON"})
            return
        frame_path = str(payload.get("frame_path", "")).strip()
        cam_id     = str(payload.get("cam_id", "")).strip() or "?"
        if not frame_path:
            self._send_json(400, {"error": "frame_path required"})
            return
        # Path harden: keep it inside snapshots dir
        if ".." in frame_path.split("/") or frame_path.startswith("/") \
                or "\\" in frame_path:
            self._send_json(400, {"error": "invalid frame_path"})
            return
        try:
            # Same re-submission rule as the crop path: editing an already
            # -reviewed frame updates the stored verdicts but fires NO second
            # round of confidence nudges (the first submission already spent
            # this frame's learning signal; re-counting it double-boosts).
            was_reviewed = _review_store().is_frame_reviewed(frame_path)
            r = _review_store().submit_frame(
                frame_path=frame_path, cam_id=cam_id,
                box_verdicts=payload.get("box_verdicts") or {},
                missed_detections=payload.get("missed_detections") or [],
                note=payload.get("note") or None)
            # Confidence boost per-box: each correct verdict lowers the
            # per-cam per-cls conf; each wrong verdict raises it. Same
            # nudges as the crop-level submit path. Skipped entirely on a
            # re-submission - the frame's learning signal was already spent.
            if not was_reviewed:
                try:
                    from app.confidence_boost import apply_review
                    # Metadata (class per box_id) sits next to the frame - reload it.
                    from app.review_frames import load_metadata
                    meta = load_metadata(frame_path, SNAPSHOTS_DIR) or {}
                    cls_by_id = {str(b["id"]): b.get("cls", "?")
                                 for b in (meta.get("boxes") or [])}
                    for box_id, verdict in (r.box_verdicts or {}).items():
                        cls = cls_by_id.get(str(box_id))
                        if not cls: continue
                        if verdict.startswith("relabel:"):
                            # The object is real but the class was wrong:
                            # stricter on the class the model claimed, looser
                            # on the class the user says is actually there.
                            apply_review(cam_id, cls, "wrong_label")
                            new_cls = verdict.split(":", 1)[1]
                            apply_review(cam_id, new_cls, "correct")
                            continue
                        v = "correct" if verdict == "correct" else "wrong_label"
                        apply_review(cam_id, cls, v)
                    # Missed detections signal: the model needs to be LESS strict
                    # for the missed class in this camera. Treat each miss like a
                    # user-confirmed "correct" verdict for its class - it lowers
                    # conf so the next burst catches similar objects.
                    for miss in (r.missed_detections or []):
                        cls = miss.get("cls")
                        if cls:
                            apply_review(cam_id, cls, "correct")
                except Exception as ex:
                    print(f"  ! frame confidence_boost skipped: {type(ex).__name__}: {ex}")
            # Verdicts + this frame's jpg/json go to Storage training/ in a
            # background thread - the nightly cloud trainer's input.
            try:
                from app.training_sync import push_async
                push_async()
            except Exception as ex:
                print(f"  ! training_sync skipped: {type(ex).__name__}: {ex}")
            self._send_json(200, {"ok": True, "frame_review": r.to_public(),
                                  "summary": _review_store().summary()})
        except Exception as e:
            print(f"  ! review-frame-submit failed: {type(e).__name__}: {e}")
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _entity_gallery(self) -> None:
        """GET /api/entity-gallery?cam_id=<>&entity_id=<int> -> every stored
        sighting crop of one tracked entity, newest first. Powers the
        appearance comparison inside the events accordion."""
        try:
            from urllib.parse import parse_qs, urlparse
            import re as _re
            q = parse_qs(urlparse(self.path).query)
            cam_id = (q.get("cam_id") or [""])[0].strip()
            eid_raw = (q.get("entity_id") or [""])[0].strip()
            if not _re.match(r"^[A-Za-z0-9_.-]{1,64}$", cam_id) \
                    or not eid_raw.isdigit():
                self._send_json(400, {"error": "cam_id and numeric entity_id required"})
                return
            from app.entity_gallery import list_sightings
            items = list_sightings(cam_id, int(eid_raw), SNAPSHOTS_DIR)
            self._send_json(200, {"cam_id": cam_id, "entity_id": int(eid_raw),
                                  "sightings": items})
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _review_frames_stats(self) -> None:
        try:
            from app.review_frames import usage_stats
            self._send_json(200, usage_stats(SNAPSHOTS_DIR))
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _review_frames_clear(self) -> None:
        # Same clear-then-reseed contract as the crop pool above: after wiping
        # the review_frames tree, drop back a small set of fixture frames so
        # the Review UI opens on real content on the next request.
        try:
            from app.review_frames import (clear_all,
                                            bootstrap_from_fixtures as rf_boot)
            result = clear_all(SNAPSHOTS_DIR)
            if _VISUAL_SEARCH._ready and _VISUAL_SEARCH.model is not None:
                try:
                    reseeded = rf_boot(
                        _VISUAL_SEARCH.model, DOCS_IMAGES_DIR, SNAPSHOTS_DIR)
                    result["reseeded"] = reseeded
                except Exception as e:
                    result["reseed_error"] = f"{type(e).__name__}: {e}"
            self._send_json(200, result)
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _blacklist_add(self) -> None:
        """Accept a user-drawn polygon from the Review canvas and persist it.

        Payload: {"cam_id": "...", "cls": "person"|..., "polygon": [[x,y], ...]}
        (coordinates normalized to [0, 1]). The response returns the stored
        entry so the frontend can echo confirmation.
        """
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > 8 * 1024:
            self._send_json(400, {"error": "empty or oversized body"})
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send_json(400, {"error": "body must be JSON"})
            return
        try:
            from app.auto_blacklist import add_polygon
            result = add_polygon(
                cam_id=str(body.get("cam_id") or "").strip(),
                cls=str(body.get("cls") or "").strip(),
                polygon=body.get("polygon") or [],
                reason=str(body.get("reason") or "user-marked block area"),
            )
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
            return
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})
            return
        # Reload the camera catalog so the next collector burst (locally OR
        # a hot-reload cycle on the VM) already sees the new polygon.
        try:
            from app.cameras import reload_review_overrides
            reload_review_overrides()
        except Exception:
            pass
        self._send_json(200, result)

    def _al_curve(self) -> None:
        """GET /api/al-curve -> labels-vs-mAP points from the training-run
        gate history (plan WS5).

        Local history covers runs executed ON this machine; the real
        trainer runs in GitHub Actions, so its gate records are merged in
        from the cloud (Firestore `training_events`, mirrored one doc per
        run) - without this the operator's curve stayed empty forever
        while CI trained. Best-effort with a short memory cache; the
        local file always renders even when the cloud is unreachable."""
        try:
            from app.adapters import al_curve_payload
            payload = al_curve_payload()
            try:
                cloud = _cloud_training_points()
                have = {p.get("adapter") for p in payload["points"]}
                for p in cloud:
                    if p.get("adapter") not in have:
                        payload["points"].append(p)
                payload["points"].sort(key=lambda p: (p.get("at") or "",
                                                      p.get("adapter") or ""))
                for p in payload["points"]:
                    if p.get("baseline_map50") is not None:
                        payload["baseline_map50"] = p.pop("baseline_map50")
            except Exception as ex:
                print(f"  ! al-curve cloud merge skipped: "
                      f"{type(ex).__name__}: {ex}")
            self._send_json(200, payload)
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _boost_status(self) -> None:
        """Per-(cam,cls) baseline vs current conf plus review counts.

        Powers the dashboard's "Learning proof" panel so the user can
        watch each verdict move the effective confidence for that camera.
        """
        try:
            from app.confidence_boost import details
            self._send_json(200, details())
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _model_metrics(self) -> None:
        """Scoreboard endpoint driving the header line. Cheap - it just
        walks the in-memory review store and does arithmetic. Safe to poll
        every 10s from the browser."""
        try:
            from app.model_metrics import compute, header_line, learning_curve
            metrics = compute(_review_store())
            try:
                from app.confidence_boost import summary as _cb_summary
                boost = _cb_summary()
            except Exception:
                boost = None
            metrics["header_line"] = header_line(metrics, boost)
            # Batch-by-batch mistake trend - the "is it actually getting
            # better?" chart the operator asked for.
            metrics["curve"] = learning_curve(_review_store())
            self._send_json(200, metrics)
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def do_HEAD(self) -> None:
        # Browsers use GET (not HEAD) for <video>/HLS, so this matters only to
        # dev tools like `curl -I`. Route it through the same proxy so the dev
        # check gets a real status code instead of a 404 from the static handler.
        if self.path.startswith("/tvkur/"):
            self._proxy_tvkur()
            return
        super().do_HEAD()

    # ---- YouTube-HLS relay -------------------------------------------------
    # googlevideo.com serves the live HLS manifests + segments that back the
    # picked YouTube cams, but sends no Access-Control-Allow-Origin header,
    # so hls.js in the browser is CORS-blocked from fetching them directly.
    # Same story as tvkur, same cure: relay through this server (which has
    # no CORS restriction) and rewrite every URL inside a playlist so the
    # browser's follow-up requests come back through the relay too.
    #
    #   GET /ytproxy?cam=<cam_id>   resolve the cam's live HLS URL (cached
    #                               by detect_core.resolve_stream), fetch the
    #                               playlist, rewrite, serve. On a 4xx from
    #                               googlevideo (signed URL rotated - they
    #                               expire every ~6h) invalidate + re-resolve
    #                               once, so long dashboard runs self-heal.
    #   GET /ytproxy?u=<url>        relay one absolute googlevideo URL
    #                               (nested playlist or media segment).
    #                               Host-whitelisted to *.googlevideo.com so
    #                               this can't be abused as an open proxy.

    def _yt_upstream_fetch(self, url: str):
        """GET an upstream URL; (bytes, content_type) or (None, None) on 4xx."""
        req = urllib.request.Request(url, headers={
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/126.0.0.0 Safari/537.36")})
        try:
            with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as r:
                return r.read(), r.headers.get("Content-Type")
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                return None, None
            raise

    def _proxy_yt(self) -> None:
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        cam_id = (q.get("cam") or [""])[0]
        # parse_qs already percent-decoded the value ONCE - exactly undoing
        # the quote(..., safe="") the playlist rewriter applied. A second
        # unquote here corrupted googlevideo's signed paths (%3D -> =) and
        # got every segment 403'd. Use as-is.
        raw = (q.get("u") or [""])[0]
        try:
            if cam_id:
                from app.cameras import CAMERAS
                from app.detect_core import invalidate_stream, resolve_stream
                cam = CAMERAS.get(cam_id)
                if cam is None:
                    self._send_json(404, {"error": f"unknown camera {cam_id!r}"})
                    return
                url = resolve_stream(cam)
                body, _ct = self._yt_upstream_fetch(url)
                if body is None:
                    invalidate_stream(cam_id)
                    url = resolve_stream(cam)
                    body, _ct = self._yt_upstream_fetch(url)
                if body is None:
                    self._send_json(502, {"error": "manifest fetch failed "
                                                   "after re-resolve"})
                    return
                text = body.decode("utf-8", "replace")
                # Measure how far the stream's PROGRAM-DATE-TIME live edge
                # trails wall clock (YouTube ingest+CDN latency). The live
                # analysis stamps its ticks with this offset applied, so
                # the browser can align boxes with the exact video frame
                # being displayed. Refreshes on every playlist reload.
                off = _ytproxy_pdt_offset(text)
                if off is not None:
                    from app.live_analysis import STREAM_PDT_OFFSET
                    STREAM_PDT_OFFSET[cam_id] = off
                data = _ytproxy_rewrite(text, url).encode()
                ct = "application/vnd.apple.mpegurl"
            elif raw:
                url = raw
                host = urlparse(url).hostname or ""
                if not (host == "googlevideo.com"
                        or host.endswith(".googlevideo.com")):
                    self._send_json(403, {"error": "host not allowed"})
                    return
                body, up_ct = self._yt_upstream_fetch(url)
                if body is None:
                    self._send_json(502, {"error": "upstream expired/refused"})
                    return
                if body[:16].lstrip().startswith(b"#EXTM3U"):
                    data = _ytproxy_rewrite(body.decode("utf-8", "replace"),
                                            url).encode()
                    ct = "application/vnd.apple.mpegurl"
                else:
                    data = body
                    ct = up_ct or "video/mp2t"
            else:
                self._send_json(400, {"error": "need ?cam= or ?u="})
                return
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser dropped the segment mid-flight - fine
        except Exception as e:
            self._send_json(502, {"error": f"{type(e).__name__}: {e}"})

    def _proxy_tvkur(self) -> None:
        # /tvkur/<stream_id>/<path...> -> content.tvkur.com/l/<stream_id>/<path...>
        # Strip any ?query so we mirror exactly what the browser asked for.
        path = self.path[len("/tvkur/"):]
        upstream = "https://content.tvkur.com/l/" + path
        try:
            req = urllib.request.Request(upstream, headers=_TVKUR_HEADERS)
            with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as r:
                self.send_response(r.status)
                ct = r.headers.get("Content-Type")
                if ct:
                    self.send_header("Content-Type", ct)
                # CORS open + short cache so hls.js can refresh the chunklist.
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                # stream the body in chunks - .ts segments are several MB
                while True:
                    chunk = r.read(64 * 1024)
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return  # the browser closed the segment fetch - fine
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(f"tvkur proxy error: {type(e).__name__}: {e}".encode())


def _ytproxy_pdt_offset(playlist: str):
    """Seconds between wall clock and the playlist's live edge, or None.

    Live edge = the last EXT-X-PROGRAM-DATE-TIME plus the EXTINF durations
    of every segment after it. googlevideo playlists carry one PDT tag at
    the top, so in practice this is pdt + the whole window's duration.
    """
    import datetime as _dt
    import re
    pdt = None
    dur_after = 0.0
    for line in playlist.splitlines():
        s = line.strip()
        if s.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            try:
                pdt = _dt.datetime.fromisoformat(
                    s.split(":", 1)[1].strip())
                dur_after = 0.0
            except ValueError:
                pass
        elif s.startswith("#EXTINF:") and pdt is not None:
            m = re.match(r"#EXTINF:([\d.]+)", s)
            if m:
                dur_after += float(m.group(1))
    if pdt is None:
        return None
    edge = pdt.timestamp() + dur_after
    return time.time() - edge


def _ytproxy_rewrite(playlist: str, base_url: str) -> str:
    """Route every URL inside an HLS playlist back through /ytproxy?u=.

    Handles both bare URL lines (segments / nested variant playlists,
    absolute or relative) and URI="..." attributes on tags such as
    EXT-X-MAP and EXT-X-MEDIA. Everything else passes through untouched.
    """
    import re
    from urllib.parse import quote, urljoin

    def _wrap(u: str) -> str:
        return "/ytproxy?u=" + quote(urljoin(base_url, u), safe="")

    out = []
    for line in playlist.splitlines():
        s = line.strip()
        if not s:
            out.append(line)
        elif s.startswith("#"):
            out.append(re.sub(r'URI="([^"]+)"',
                              lambda m: f'URI="{_wrap(m.group(1))}"', line))
        else:
            out.append(_wrap(s))
    return "\n".join(out) + "\n"


def make_handler_factory(directory: Path | None = None):
    """Return a handler class bound to a serving directory (defaults to web/)."""
    d = str(directory or WEB_DIR)
    return lambda *a, **k: DashboardHandler(*a, directory=d, **k)


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False


def _warm_visual_search_async() -> None:
    """Kick off YOLO load + review-pool bootstrap + anomaly-crops refresh
    in a background daemon thread. Called from bind() so the pool is
    populated by the time a first user opens the review UI - no cold-start
    "every stored crop has been reviewed" message on a fresh install.

    Safe to fire even without ultralytics installed: _VisualSearchState.get()
    catches YOLO import failures and continues in whole-image mode.
    """
    def _run() -> None:
        try:
            st = _VISUAL_SEARCH.get()
        except Exception as e:
            print(f"  ! visual-search warmup failed: {type(e).__name__}: {e}")
            return
        # Keep the embedding index warm FOREVER, not just at boot: the
        # pool-sync puller drops fresh crops every couple of minutes, and
        # deferring their embedding to the next search request meant the
        # FIRST search after hours of collecting sat behind minutes of
        # OSNet work (the operator read that as "search is broken").
        import time as _time
        while True:
            _time.sleep(120)
            try:
                # Yield to live analysis: the OSNet embedding sweep runs
                # onnxruntime's own thread pool for minutes at a time and
                # was caught (py-spy) competing with OpenVINO inference
                # for the 4 cores while an operator was actively watching
                # a layer. Crops queue up and embed when analysis stops.
                from app.live_analysis import MANAGER as _MGR
                if _MGR.any_alive():
                    continue
                # Extraction first (was the boot-time blocker, now idle-
                # only), then embedding of whatever is new.
                try:
                    from app.frame_crops import refresh as _fc_refresh
                    fc = _fc_refresh(st.embedder, SNAPSHOTS_DIR)
                    if fc.get("frames_touched"):
                        print(f"  * review-crops refresh {fc} (background)")
                except Exception as e:
                    print(f"  ! review-crops refresh failed: "
                          f"{type(e).__name__}: {e}")
                with st.refresh_lock:
                    n = st.index.refresh()
                if n:
                    print(f"  * search index: +{n} crop(s) embedded "
                          f"(background)")
            except Exception as e:
                print(f"  ! search index refresh failed: "
                      f"{type(e).__name__}: {e}")
    threading.Thread(target=_run, daemon=True,
                     name="visual-search-warmup").start()


def bind(port: int, directory: Path | None = None) -> http.server.ThreadingHTTPServer:
    """Threaded server so simultaneous video segment requests don't queue.

    Also fires an async warmup that loads YOLO in the background and
    bootstraps the review pool from fixture frames, so the first user to
    open the dashboard finds material to review already sitting there.
    Starts the pool-sync puller too: it mirrors the VM collector's
    review_frames / live_samples / reid.db down to this machine, so search
    and review operate on what the cameras actually captured instead of on
    the shipped fixtures. Without a reachable bucket it degrades silently
    to the local-only behavior.
    """
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    http.server.ThreadingHTTPServer.daemon_threads = True
    server = http.server.ThreadingHTTPServer(("", port), make_handler_factory(directory))
    _warm_visual_search_async()
    try:
        from app.pool_sync import start_pull_thread
        start_pull_thread(SNAPSHOTS_DIR)
    except Exception as e:
        print(f"  ! pool-sync puller not started: {type(e).__name__}: {e}")
    # Auto-start local ModelViewProducer + ReviewFrameProducer if a picker
    # has already written web/local_grid.json. This is what feeds the local
    # tiles' Activity Index + KPI badges when the operator picked cams that
    # are NOT the VM's active grid (typical for a Thailand pick while the VM
    # watches Turkey). The producers run inside this Python process, share
    # the same YOLO model as the visual-search warmup + live-analysis
    # sessions, and write annotated JPEGs + counts JSON that the frontend's
    # LOCAL_MODE poll reads from /snapshots/model_view/local_*.json.
    def _start_local_producers_when_ready():
        _lg = WEB_DIR / "local_grid.json"
        if not _lg.exists():
            return
        try:
            import json as _json
            grid = _json.loads(_lg.read_text(encoding="utf-8"))
            slots = grid.get("slots") or []
            if not slots:
                return
            # ONE model for the whole process: producers share the
            # visual-search model (yolov8s / its OpenVINO engine) instead
            # of loading a second yolo26m engine. On the 8GB laptop the
            # two-engine setup measurably exhausted RAM (88% used, 1GB
            # free) and pushed inference into pagefile thrash - ticks
            # ballooned to 15s. Wait for the warmup to finish loading,
            # bounded so a broken warmup doesn't hang the hook forever.
            import time as _t
            _model = None
            for _ in range(300):
                _model = _VISUAL_SEARCH.model
                if _model is not None:
                    break
                _t.sleep(1)
            if _model is None:
                print("  ! local_producers not started: model not loaded "
                      "within 5 min")
                return
            from app.local_producers import start_all as _start_all
            _start_all(slots, _model,
                       model_view_interval_s=8, review_interval_s=60)
            print(f"local_producers running: {len(slots)} slots -> "
                  f"web/snapshots/model_view/local_*.jpg (~8s per round)")
        except Exception as e:
            print(f"  ! local_producers not started: {type(e).__name__}: {e}")

    import threading as _threading
    _threading.Thread(target=_start_local_producers_when_ready,
                      daemon=True,
                      name="local-producers-autostart").start()
    return server
