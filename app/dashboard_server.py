"""Dashboard HTTP server building blocks shared by serve.py and the notebook.

Serves web/ statically AND proxies tvkur/IBB streams the browser can't reach
directly due to Referer/CORS requirements:

    GET /tvkur/<stream_id>/<path>           -> content.tvkur.com/l/<stream_id>/<path>
                                               with Referer/Origin=player.tvkur.com
    GET /snapshots/...                      -> web/snapshots/... (anomaly + returning frames)

The proxy adds Access-Control-Allow-Origin:* so hls.js in the dashboard can
fetch the master playlist and segments without browser CORS errors.
"""
from __future__ import annotations

import http.server
import socket
import socketserver
import ssl
import sys
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
