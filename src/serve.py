"""One-shot launcher for the live HTML dashboard.

    python serve.py            # serve web/ on http://localhost:8000 and open the browser
    python serve.py --port 8765
    python serve.py --no-browser

The dashboard at web/index.html is a static page that talks to Firestore directly
from the browser via onSnapshot. So all this script does is:

  1. Verify web/firebase-config.js exists (otherwise the page renders a warning).
  2. Bind a tiny static-file server to the web/ folder.
  3. Proxy tvkur HLS so the browser can play the live stream directly with
     hls.js (content.tvkur.com refuses bare requests and sends no CORS headers).
  4. Open the default browser at the root URL.

Run the collector (python -m app.collector ...) in a separate terminal to populate
Firestore; the dashboard updates the moment new writes arrive.
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

from app.dashboard_server import WEB_DIR, bind, port_is_free


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8000, help="port to serve on (default 8000)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the default browser")
    args = ap.parse_args()

    if not WEB_DIR.is_dir():
        sys.exit(f"web/ folder not found at {WEB_DIR}")

    cfg = WEB_DIR / "firebase-config.js"
    if not cfg.is_file():
        print("WARNING: web/firebase-config.js is missing. The page will load but show a")
        print("         red config-warning banner instead of live data. Copy the template:")
        print("           cp web/firebase-config.example.js web/firebase-config.js")
        print("         and paste your project's web SDK values into it.\n")

    port = args.port
    if not port_is_free(port):
        for candidate in range(port + 1, port + 21):
            if port_is_free(candidate):
                print(f"Port {port} busy; falling back to {candidate}.")
                port = candidate
                break
        else:
            sys.exit(f"Port {port} is busy and no nearby port is free. Use --port to pick one.")

    url = f"http://localhost:{port}/"
    server = bind(port)
    print(f"Serving {WEB_DIR} at {url}")
    print("Routes: /          -> web/index.html (live dashboard)")
    print("        /tvkur/... -> proxy to content.tvkur.com (autoplay for Konya tiles)")
    print("        /snapshots -> web/snapshots/ (anomalies + returning visitors)\n")
    print("Reminder: in another terminal run the collector so Firestore gets fresh data:")
    print("   python -m app.collector --interval 20 \\")
    print("       --only konya_hukumet,otogar_kavsagi,sultanahmet_1_yeni,taksim_yeni")
    print("Press Ctrl+C to stop.\n")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url, new=2)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
