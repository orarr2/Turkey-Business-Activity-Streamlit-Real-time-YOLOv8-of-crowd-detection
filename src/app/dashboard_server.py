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
                # Per-object extraction of the review-frames pool. Needs no
                # YOLO (boxes ship in the frame metadata), so it runs even
                # when the model failed to load above.
                try:
                    from app.frame_crops import refresh as _fc_refresh
                    summary = _fc_refresh(self.embedder, SNAPSHOTS_DIR)
                    if summary.get("frames_touched"):
                        print(f"visual-search: review-crops refresh {summary}")
                except Exception as e:
                    print(f"visual-search: review-crops refresh failed "
                          f"({type(e).__name__}: {e}) - continuing")
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
        if path == "/api/review-frames-stats":
            self._review_frames_stats()
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
            # A re-submission overwrites the stored review (keyed by path) but
            # must NOT nudge confidence a second time - otherwise clicking
            # through the same crop twice counts as two learning events and
            # the boost ledger drifts away from the review store.
            was_reviewed = _review_store().is_reviewed(crop_path)
            r = _review_store().submit(
                crop_path,
                verdict,
                original_cls=str(payload.get("original_cls", "?")),
                corrected_cls=payload.get("corrected_cls") or None,
                anomaly_verdict=payload.get("anomaly_verdict") or None,
                note=payload.get("note") or None)
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
            s = sample_frame(_review_store(), SNAPSHOTS_DIR)
            if s is None:
                self._send_json(200, {"done": True,
                                      "message": "no un-reviewed frames yet"})
                return
            self._send_json(200, s)
        except Exception as e:
            print(f"  ! review-frame failed: {type(e).__name__}: {e}")
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _review_frames_list(self) -> None:
        """GET /api/review-frames-list -> every stored frame + review status,
        newest first. Powers the strip that re-opens reviewed frames."""
        try:
            from app.labels import list_frames
            self._send_json(200, {"frames": list_frames(_review_store(),
                                                        SNAPSHOTS_DIR)})
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
            self._send_json(200, {"ok": True, "frame_review": r.to_public(),
                                  "summary": _review_store().summary()})
        except Exception as e:
            print(f"  ! review-frame-submit failed: {type(e).__name__}: {e}")
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
            from app.model_metrics import compute, header_line
            metrics = compute(_review_store())
            try:
                from app.confidence_boost import summary as _cb_summary
                boost = _cb_summary()
            except Exception:
                boost = None
            metrics["header_line"] = header_line(metrics, boost)
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
            _VISUAL_SEARCH.get()
        except Exception as e:
            print(f"  ! visual-search warmup failed: {type(e).__name__}: {e}")
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
    return server
