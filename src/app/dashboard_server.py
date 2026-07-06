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
    SEARCH_YOLO  YOLO weights for query-object extraction (default yolov8n.pt;
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
                weights = os.environ.get("SEARCH_YOLO", "yolov8n.pt")
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
        super().do_GET()

    def do_POST(self) -> None:
        if self.path.split("?")[0] == "/api/visual-search":
            self._visual_search()
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
        """POST /api/visual-search?top=12&min_sim=0.3&classes=person,car

        Body: the raw uploaded image bytes (any cv2-decodable format - the
        search page sends the File blob as-is, no multipart parsing needed).
        """
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            self._send_json(400, {"error": "empty body - send the image bytes"})
            return
        if length > MAX_UPLOAD_BYTES:
            self._send_json(413, {"error": f"image too large (>{MAX_UPLOAD_BYTES} bytes)"})
            return
        data = self.rfile.read(length)

        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)

        def _one(name, cast, default):
            try:
                return cast(q[name][0])
            except (KeyError, IndexError, ValueError):
                return default

        top_n   = max(1, min(50, _one("top", int, 12)))
        min_sim = _one("min_sim", float, 0.30)
        classes = {c.strip() for c in (q.get("classes", [""])[0]).split(",")
                   if c.strip()} or None
        try:
            from app.visual_search import search_image_bytes
            st = _VISUAL_SEARCH.get()
            result = search_image_bytes(
                data, model=st.model, embedder=st.embedder,
                snapshot_index=st.index, db_path=st.db_path,
                top_n=top_n, min_sim=min_sim, classes=classes)
            result["detector"] = "yolo" if st.model is not None else "whole-image"
            self._send_json(200, result)
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
        except Exception as e:
            print(f"  ! visual-search failed: {type(e).__name__}: {e}")
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
