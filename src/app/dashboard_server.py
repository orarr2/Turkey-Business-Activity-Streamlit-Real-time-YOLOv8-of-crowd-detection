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
        self._ready = False
        self.embedder = None
        self.model = None
        self.index = None
        self.db_path = None

    def get(self):
        with self._lock:
            if not self._ready:
                from app.visual_search import DEFAULT_DB, SnapshotIndex
                from app.reid_embed import make_embedder
                self.embedder = make_embedder(os.environ.get("REID_MODEL") or None)
                self.db_path = os.environ.get("REID_DB") or DEFAULT_DB
                weights = os.environ.get("SEARCH_YOLO", "yolov8s.pt")
                if weights.lower() not in ("off", "none", ""):
                    try:
                        from app.detect_core import load_model
                        self.model = load_model(weights)
                    except Exception as e:
                        print(f"visual-search: YOLO unavailable ({e}) - "
                              f"uploads will be embedded whole (no object "
                              f"extraction). pip install ultralytics to fix.")
                self.index = SnapshotIndex(SNAPSHOTS_DIR, embedder=self.embedder)
                self._ready = True
            return self


_VISUAL_SEARCH = _VisualSearchState()

# Review store - lazily constructed on the first labels endpoint hit. The
# store is thread-safe (single lock around its dict + rewrite), so all
# handler threads share the one instance.
_REVIEW_STORE = None
_REVIEW_STORE_LOCK = threading.Lock()


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
        sys.stdout.write("  " + (fmt % args) + "\n")

    def do_GET(self) -> None:
        if self.path.startswith("/tvkur/"):
            self._proxy_tvkur()
            return
        path = self.path.split("?")[0]
        if path == "/api/review-sample":
            self._review_sample()
            return
        if path == "/api/review-stats":
            self._review_stats()
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
        self.send_error(404, "unknown POST endpoint")

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()   # skip our no-cache re-header dance
        self.wfile.write(body)

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
        min_sim = _one("min_sim", float, 0.30)
        classes = {c.strip() for c in (q.get("classes", [""])[0]).split(",")
                   if c.strip()} or None
        time_from = _parse_time(q.get("from", [""])[0])
        time_to   = _parse_time(q.get("to", [""])[0])
        order     = q.get("order", ["time_desc"])[0]
        try:
            st = _VISUAL_SEARCH.get()
            if data:
                from app.visual_search import search_image_bytes
                result = search_image_bytes(
                    data, model=st.model, embedder=st.embedder,
                    snapshot_index=st.index, db_path=st.db_path,
                    top_n=top_n, min_sim=min_sim, classes=classes,
                    time_from=time_from, time_to=time_to)
                result["detector"] = "yolo" if st.model is not None else "whole-image"
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
    # one random un-reviewed crop with its current label and picks correct /
    # wrong-label / not-an-object. Answers persist to data/reviews.json via
    # ReviewStore.
    def _review_sample(self) -> None:
        try:
            from app.labels import sample_crop
            s = sample_crop(_review_store(), SNAPSHOTS_DIR)
            if s is None:
                self._send_json(200, {"done": True,
                                      "message": "no un-reviewed crops in the store"})
                return
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
            r = _review_store().submit(
                crop_path,
                verdict,
                original_cls=str(payload.get("original_cls", "?")),
                corrected_cls=payload.get("corrected_cls") or None,
                note=payload.get("note") or None)
            self._send_json(200, {"ok": True, "review": r.to_public(),
                                  "summary": _review_store().summary()})
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
        except Exception as e:
            print(f"  ! review-submit failed: {type(e).__name__}: {e}")
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _review_stats(self) -> None:
        try:
            self._send_json(200, _review_store().summary())
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


def bind(port: int, directory: Path | None = None) -> http.server.ThreadingHTTPServer:
    """Threaded server so simultaneous video segment requests don't queue."""
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    http.server.ThreadingHTTPServer.daemon_threads = True
    return http.server.ThreadingHTTPServer(("", port), make_handler_factory(directory))
